#!/usr/bin/env python3
"""Per-kernel grid/tile alignment validation for captured serving traces.

Retains the historical per-step comparison for existing experiment artifacts
and validates single-operator launch selection against profiler observations.
Runtime-name parsing and profiler access remain validation-only.

The serving side reuses the per-STEP ordered correlation from
build_gemm_tile_lut_mnk (operator-record step_id == scheduler-sidecar iter_id,
sidecar bridges nsys time -> step): within each step the synthesized family's
kernel sequence zips with the nsys family's sequence in execution order, and
each pair's GRID (+ tile, for tile-bearing families) is compared. The single-op
side has no scheduling/alignment ambiguity (one exact shape per ncu entry), so
synthesis is compared directly to the ncu GT kernel name/grid.

Families:
  serving: GEMM-s16816 (tile+grid), GEMV, FA-splitkv (num_splits+grid),
           FA-combine, fused_moe (grid), + the Tier-2 FOCUS grids (NORM/ROTARY/
           KV_CACHE/ACTIVATION) via compare_trace_with_nsys.
  single-op: GEMM (fp16 LUT built from ncu -> tile+grid consistency),
             FA (num_splits heuristic vs ncu GT), MoE (derive_moe_tiling grid vs
             ncu actual kernel grid).
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from evaluation.serving.traces.trace_alignment_io import (
    idle_iteration_ids,
    load_nsys_json,
    load_scheduler_sidecar,
    nsys_time_bounds,
    synchronization_boundaries,
    timestamp_boundaries,
)
import sqlite3
NCU_DIR = ROOT / "kernel_dataset" / "nvidia_a800"
TILE_RE = re.compile(r"(\d+)x(\d+)")


def _non_negative_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{context} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{context} must be non-negative, got {value}")
    return value


def _profiler_integer(value, context, *, positive=False):
    """Parse an integer-valued profiler/SQLite field without truncation."""

    if isinstance(value, bool):
        raise TypeError(f"{context} cannot be boolean")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        result = int(value)
    else:
        raise TypeError(
            f"{context} must be an integer or integer text, got {value!r}")
    if positive and result <= 0:
        raise ValueError(f"{context} must be positive, got {result}")
    if not positive and result < 0:
        raise ValueError(f"{context} must be non-negative, got {result}")
    return result


def parse_tile(name):
    if not isinstance(name, str) or not name:
        raise TypeError("kernel name must be a non-empty string")
    match = TILE_RE.search(name)
    if match:
        first, second = int(match.group(1)), int(match.group(2))
        if 32 <= first <= 512 and 32 <= second <= 512:
            return first, second
    return None, None


def pad64(value):
    value = _non_negative_integer(value, "pad64 value")
    return ((value + 63) // 64) * 64 if value else 0


def consistent(n, m, tile_n, tile_m, grid_x, grid_y, grid_z=1):
    """Return whether an observed 2-D or flattened GEMM grid fits its tile."""
    import math
    if not tile_n or not tile_m or not grid_x:
        return False
    if (n <= grid_x * tile_n < n + tile_n
            and m <= grid_y * tile_m < m + tile_m):
        return True
    return grid_y == 1 and grid_x == math.ceil(n / tile_n) * math.ceil(m / tile_m)


# ---------------------------------------------------------------------------
# Serving: load the FULL regenerated trace (all kernels, with grids) + a
# generic nsys family loader (any name substrings), then per-step zip+compare.
# ---------------------------------------------------------------------------

def load_regen_trace(case_dir, rank=0):
    """Load the newest regen cache (rank `rank`) -> list of (step_id, name,
    grid, op_attr) for ALL kernels (not just s16816). The regen trace is the
    recipe's synthesis, so its grids are what we validate against nsys."""
    import gzip, pickle, glob, os
    rank = _non_negative_integer(rank, "rank")
    cands = sorted(glob.glob(str(case_dir / ".regen_cache_*.pkl.gz")),
                   key=os.path.getmtime)
    if not cands:
        return None
    with gzip.open(cands[-1], "rb") as f:
        tr = pickle.load(f)
    if not isinstance(tr, dict) or not tr:
        raise TypeError("regen trace root must be a non-empty rank mapping")
    keys = sorted(tr.keys(), key=str)
    # rank key may be str ("0") or int; match rank
    key = None
    for k in keys:
        if str(k) == str(rank):
            key = k; break
    if key is None:
        raise KeyError(
            f"regen trace has no rank {rank}; available ranks: "
            f"{[str(item) for item in keys]}")
    t = tr[key]
    out = []
    if not isinstance(t, list) or not t:
        raise ValueError(f"regen trace rank {rank} must contain entries")
    for e in t:
        if not isinstance(e, dict):
            raise TypeError(f"regen trace entry must be a mapping: {e!r}")
        for field in ("step_id", "kernel_name", "grid", "block", "op_attr", "shared_mem"):
            if field not in e:
                raise ValueError(f"regen trace entry missing {field!r}: {e!r}")
        g = e["grid"]
        b = e["block"]
        if (not isinstance(g, (list, tuple))
                or not isinstance(b, (list, tuple))
                or len(g) != 3 or len(b) != 3):
            raise ValueError(f"regen launch dimensions must have length 3: {e!r}")
        if not isinstance(e["kernel_name"], str) or not e["kernel_name"]:
            raise ValueError(f"regen kernel_name must be non-empty: {e!r}")
        if not isinstance(e["op_attr"], dict):
            raise ValueError(f"regen op_attr must be an object: {e!r}")
        grid = tuple(
            _profiler_integer(value, "regen grid", positive=True)
            for value in g)
        block = tuple(
            _profiler_integer(value, "regen block", positive=True)
            for value in b)
        step_id = _profiler_integer(e["step_id"], "regen step_id")
        shared_memory = _profiler_integer(
            e["shared_mem"], "regen shared_mem")
        out.append((step_id, e["kernel_name"], grid, e["op_attr"],
                    block, shared_memory))
    return out


def _step_bounds(case_dir, device_id=None):
    """Return (iters, bounds) preferring the structural per-step
    cudaStreamSynchronize completion boundaries (no extrapolation drift) and
    falling back to ts extrapolation. _sync_boundaries returns iters trimmed to
    its sync count (dropping any capture-tail sidecar iter); the ts fallback
    returns the raw sidecar iters. Both yield bounds of shape [n_steps+1].

    device_id: multi-rank — use that rank's syncs (mapped via globalTid->dev)
    and that device's kernels."""
    iters, ts_us = load_scheduler_sidecar(case_dir)
    if not iters:
        return None, None
    sb = synchronization_boundaries(case_dir, iters, device_id=device_id)
    if sb is not None:
        return sb[1], sb[0]
    nsys0, last_end = nsys_time_bounds(case_dir, device_id)
    if ts_us and nsys0 is not None:
        # Idle engine steps (total_tokens==0) run no model forward, so the regen
        # emits NO kernels for them (no step_id) and the nsys stream has no
        # per-step block there. The sync-boundary path drops them (see
        # _idle_iter_set); the ts-extrapolation fallback MUST do the same, else
        # the retained idle iter keeps a ts window that shifts every later step's
        # nsys partition by one relative to the regen (whole-step category
        # cascade — verified on Llama-3.1-8B qps4, which lacks per-step syncs and
        # falls here). Drop idle iters from iters AND ts_us in lockstep so the
        # non-idle iter[i] <-> ts_us[i] mapping matches the regen's step_ids.
        idle = idle_iteration_ids(case_dir)
        if idle:
            _keep = [(it, t) for it, t in zip(iters, ts_us) if it not in idle]
            if _keep:
                iters = [it for it, _ in _keep]
                ts_us = [t for _, t in _keep]
        # C500 multi-warmup fix: nsys0 (first rms_norm) may anchor inside a
        # warmup forward that the sidecar EXCLUDES (sidecar starts at the first
        # REAL request forward). On 32B-TP2 this is ~4s of warmup forwards
        # before sidecar ts0, so ts-bridged boundaries are off by that gap.
        # Re-anchor nsys0 to the first nsys kernel at/after sidecar ts0_ns
        # (host->GPU clock offset is ~ms, same as 8B). This makes bounds[0]
        # = forward-1 start, matching regen (which has no warmup).
        # Gate on a LARGE offset (nsys0 vs sidecar ts0) so TP1 cases (offset
        # ~ms, no warmup) keep their verified nsys0 anchor.
        if device_id is not None and (case_dir / "nsys_kernels_full.json").exists() \
                and not (case_dir / "nsys_profile.sqlite").exists():
            ts0_ns = ts_us[0] * 1000
            if abs(nsys0 - ts0_ns) > 1_000_000_000:  # >1s offset => warmup gap
                allk = load_nsys_json(case_dir, device_id)
                if allk:
                    for st, _nm, _g, _b, _s in allk:
                        if st >= ts0_ns:
                            nsys0 = st
                            break
        return iters, timestamp_boundaries(ts_us, nsys0, last_end)
    return iters, None


def load_nsys_family(case_dir, name_subs):
    """nsys kernels (device 0, ordered by start) whose name contains ANY of
    name_subs: list of (start_ns, name, (gx, gy, gz)). Plus per-step boundaries."""
    p = case_dir / "nsys_profile.sqlite"
    if not p.exists():
        # JSON fallback (C500): no sqlite; nsys_kernels_full.json carries the
        # same grid/ts_ns. No StringId index, so filter in Python.
        allk = load_nsys_json(case_dir, 0)
        if allk is None:
            return None, None, None
        subs_l = [s.lower() for s in name_subs]
        kernels = [(st, nm, g) for st, nm, g, _b, _s in allk
                   if any(s in nm.lower() for s in subs_l)]
        iters, bounds = _step_bounds(case_dir)
        return kernels, iters, bounds
    conn = sqlite3.connect(str(p))
    # Pre-fetch matching StringIds (avoid LIKE-join full scan).
    idset = set()
    for sub in name_subs:
        for r in conn.execute(
            "SELECT id FROM StringIds WHERE value LIKE ?", (f"%{sub}%",)).fetchall():
            idset.add(r[0])
    if not idset:
        conn.close()
        return [], None, None
    ph = ",".join("?" * len(idset))
    rows = conn.execute(
        f"SELECT k.start, s.value, k.gridX, k.gridY, k.gridZ "
        f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
        f"WHERE k.demangledName IN ({ph}) AND k.deviceId=0 ORDER BY k.start",
        list(idset)).fetchall()
    conn.close()
    kernels = [(r[0], r[1], (r[2], r[3], r[4])) for r in rows]
    iters, bounds = _step_bounds(case_dir)
    return kernels, iters, bounds


def load_nsys_all(case_dir, device_id=0):
    """ALL nsys kernels (device device_id, ordered by start) with grid, block,
    shared_mem (static+dynamic) + the per-step partition + iters."""
    p = case_dir / "nsys_profile.sqlite"
    if not p.exists():
        # JSON fallback (C500): nsys_kernels_full.json has name/grid/block/smem/ts_ns.
        allk = load_nsys_json(case_dir, device_id)
        if allk is None:
            return None, None, None
        iters, bounds = _step_bounds(case_dir, device_id)
        return allk, iters, bounds
    device = _non_negative_integer(device_id, "device_id")
    with sqlite3.connect(str(p)) as connection:
        rows = connection.execute(
            "SELECT k.start, s.value, k.gridX, k.gridY, k.gridZ, "
            "k.blockX, k.blockY, k.blockZ, "
            "k.staticSharedMemory + k.dynamicSharedMemory "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName=s.id "
            "WHERE k.deviceId=? ORDER BY k.start",
            (device,),
        ).fetchall()
    kernels = []
    for row_index, row in enumerate(rows):
        start, name, *geometry = row
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"nsys row {row_index} kernel name must be non-empty")
        start = _profiler_integer(start, f"nsys row {row_index} start")
        values = tuple(
            _profiler_integer(
                value,
                f"nsys row {row_index} launch field {field_index}",
                positive=field_index < 6,
            )
            for field_index, value in enumerate(geometry)
        )
        kernels.append((
            start, name, values[:3], values[3:6], values[6]))
    iters, bounds = _step_bounds(case_dir, device_id)
    return kernels, iters, bounds


# ---------------------------------------------------------------------------
# 910B (Ascend) msprof loader — the per-invocation analog of load_nsys_all.
# msprof captures CANN aclnn op LAUNCHES (not CUDA kernels): each COMPUTE
# task carries an Op Name + Block Dim (= AIC/AIV cores used). The main
# msprof_*.db TASK ⋈ COMPUTE_TASK_INFO gives the ORDERED per-invocation stream
# (TASK.startNs is the host clock — verified aligned to the scheduler sidecar
# ts_us, so the SAME sidecar-boundary partition the A800 path uses works here).
# The aggregated ai_core_op_summary.db / op_summary CSV cannot do this: they
# collapse by Op Name and lose order + per-step timing.
# ---------------------------------------------------------------------------

def _msprof_db_path(case_dir):
    """Top-level msprof_*.db (has TASK+COMPUTE_TASK_INFO+CANN_API per-invocation).
    NOT the device_0/sqlite/ai_core_op_summary.db (that is aggregated ge_summary).
    For TP2: there are 2 PROF dirs (one per device), deviceId can be 0 or 1 in
    either — return the first by default (call _msprof_db_for_device for a
    specific deviceId)."""
    import glob
    case_dir = Path(case_dir)
    cands = glob.glob(str(case_dir / "msprof" / "PROF_*" / "msprof_*.db"))
    # exclude the device sqlite dir; the top-level db is directly under PROF_*.
    cands = [c for c in cands if "/sqlite/" not in c and "/device_" not in c]
    return cands[0] if cands else None


def _msprof_db_for_device(case_dir, device_id):
    """Find the msprof_*.db that contains tasks for the given deviceId.
    For TP2, the 2 PROF dirs have different deviceIds (0 and 1, order varies)."""
    import glob, sqlite3
    case_dir = Path(case_dir)
    cands = glob.glob(str(case_dir / "msprof" / "PROF_*" / "msprof_*.db"))
    cands = [c for c in cands if "/sqlite/" not in c and "/device_" not in c]
    device = _non_negative_integer(device_id, "device_id")
    for db in cands:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
            n = c.execute(
                "SELECT COUNT(*) FROM TASK WHERE deviceId=?",
                (device,)).fetchone()[0]
        if n > 0:
            return db
    if cands:
        raise ValueError(
            f"no msprof database contains deviceId={device}: {cands}")
    return None


def load_msprof_all(case_dir, device_id=0):
    """910B (Ascend) analog of load_nsys_all. Ordered per-invocation compute
    kernels from the main msprof_*.db TASK ⋈ COMPUTE_TASK_INFO, + sidecar step
    partition. Returns (kernels, iters, bounds) with the SAME 5-tuple schema
    load_nsys_all uses, so _compute_kernel_alignment's positional-zip loop is
    reused verbatim:

        (start_ns, op_name, (block_dim,), (mix_block_dim,), 0)

    On 910B the launch geometry is Block Dim (= AIC/AIV cores used); the
    alignment loop compares grid[0]=block_dim as launch_mm and block[0]=
    mix_block_dim as a secondary mix check (block_mm). smem=0 (CANN cube 16³
    tiling, not per-invocation). Memcpy / HCCL tasks are NOT in
    COMPUTE_TASK_INFO (only in TASK) and so are excluded — they carry no Op
    Name / Block Dim row (HCCL grid is a library constant, accepted). Step
    partition: scheduler sidecar ts_us*1000 == msprof TASK.startNs host clock
    (verified), so the A800 _ts_boundaries relative-offset logic applies,
    anchored at the first compute task's startNs.

    Returns (None, None, None) if no top-level msprof_*.db."""
    case_dir = Path(case_dir)
    db = _msprof_db_for_device(case_dir, device_id)
    if not db:
        return None, None, None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dev = _non_negative_integer(device_id, "device_id")
    # name/opType are stored as STRING_IDS integers in COMPUTE_TASK_INFO.
    sid = {i: v for i, v in conn.execute("SELECT id, value FROM STRING_IDS")}
    rows = conn.execute(
        "SELECT t.startNs, t.endNs, ci.name, ci.blockDim, ci.mixBlockDim "
        "FROM TASK t JOIN COMPUTE_TASK_INFO ci ON t.globalTaskId = ci.globalTaskId "
        f"WHERE t.deviceId={dev} ORDER BY t.startNs").fetchall()
    last_end = 0
    kernels = []
    for start_ns, end_ns, name_id, bd, mbd in rows:
        if name_id not in sid:
            raise ValueError(f"msprof task references unknown STRING_IDS id {name_id}")
        nm = sid[name_id]
        if not isinstance(nm, str) or not nm:
            raise TypeError(
                f"msprof STRING_IDS value must be non-empty: {nm!r}")
        start_ns = _profiler_integer(
            start_ns, f"msprof startNs for {nm!r}")
        end_ns = _profiler_integer(end_ns, f"msprof endNs for {nm!r}")
        if end_ns < start_ns:
            raise ValueError(
                f"msprof task end precedes start for {nm!r}: "
                f"{start_ns} -> {end_ns}")
        bdi = _profiler_integer(
            bd, f"msprof blockDim for {nm!r}", positive=True)
        mbdi = _profiler_integer(
            mbd, f"msprof mixBlockDim for {nm!r}")
        if end_ns > last_end:
            last_end = end_ns
        kernels.append((start_ns, nm, (bdi,), (mbdi,), 0))
    conn.close()
    iters, ts_us = load_scheduler_sidecar(case_dir)
    if not iters or not kernels:
        return kernels, iters, None
    # Forward+multinomial-end boundaries: aclnnEmbedding_GatherV2AiCore_GatherV2
    # is the LAST kernel of each forward's sampler (the next-input embedding
    # lookup, after the multinomial gather). Its .end = the multinomial end =
    # the boundary before the NEXT forward. bucket[i] =
    # [embedding[i-1].end, embedding[i].end) = [forward i][argmax i][multinomial
    # i (incl embedding[i])]. With the regen NOT remapping the multinomial
    # (in-step at end), this aligns (multinomial in-step for all B). Fires
    # ~1/forward (1063). This is the 910B forward-bounding signal that WORKS
    # (argmax_Cast.end failed — it's mid-sampler, not the multinomial end).
    _embed_id = None
    for _i, _v in sid.items():
        if _v == "aclnnEmbedding_GatherV2AiCore_GatherV2":
            _embed_id = _i
            break
    sync_ends = sorted(
        end_ns for start_ns, end_ns, name_id, bd, mbd in rows
        if name_id == _embed_id and end_ns)
    first_start = kernels[0][0]
    if sync_ends:
        # embedding-end boundaries (multinomial-end = next-forward boundary).
        # For each non-idle sidecar iter, its multinomial-end = the FIRST
        # embedding.end AFTER ts_us[iter]. bucket[i] = [embed[i-1].end,
        # embed[i].end) = [forward i][argmax i][multinomial i]. Requires the
        # regen to NOT remap the multinomial (in-step at end) — see
        # operator_records _remap_multinomial=False.
        import json as _json
        non_idle = []
        with open(str(case_dir / "plugin_trace.json_scheduler.jsonl")) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                _r = _json.loads(_line)
                if "total_tokens" not in _r:
                    raise ValueError("scheduler sidecar record missing total_tokens")
                total_tokens = _non_negative_integer(
                    _r["total_tokens"], "scheduler total_tokens")
                iteration = _non_negative_integer(
                    _r["iter_id"], "scheduler iter_id")
                timestamp_us = _non_negative_integer(
                    _r["ts_us"], "scheduler ts_us")
                if total_tokens > 0:
                    non_idle.append((iteration, timestamp_us * 1000))
        sync_sorted = sorted(sync_ends)
        iters = []
        sync_for = []
        _ei = 0  # embedding pointer — each embedding assigned to ONE iter (dedup)
        for _it, _ts in non_idle:
            while _ei < len(sync_sorted) and sync_sorted[_ei] <= _ts:
                _ei += 1
            if _ei < len(sync_sorted):
                iters.append(_it)
                sync_for.append(sync_sorted[_ei])
                _ei += 1
            # else: no more embeddings (uncaptured tail) — skip
        bounds = [first_start] + sync_for
        if bounds[-1] < last_end:
            bounds.append(last_end + 1)
    else:
        # FALLBACK: direct ts_us*1000 (same host-clock epoch as TASK.startNs).
        last_ts = ts_us[-1] * 1000
        bounds = [t * 1000 for t in ts_us]
        bounds.append(max(last_end + 1, last_ts + 1))
        # trim to capture window
        keep = [i for i, b in enumerate(bounds) if b < last_end]
        if keep and keep[-1] + 1 < len(bounds):
            keep.append(keep[-1] + 1)
        if len(keep) >= 2:
            bounds = [bounds[i] for i in keep]
            iters = [iters[i] for i in keep[:-1]]
    return kernels, iters, bounds


def _partition(kernels, bounds, iters):
    """Partition start-ordered kernels into per-iter lists (iters order)."""
    if bounds is None or not iters:
        return {None: list(kernels)}
    by_step = {it: [] for it in iters}
    ki = 0
    n = len(kernels)
    for idx, it in enumerate(iters):
        lo = bounds[idx]
        hi = bounds[idx + 1] if idx + 1 < len(bounds) else float("inf")
        while ki < n and kernels[ki][0] < lo:
            ki += 1
        j = ki
        while j < n and kernels[j][0] < hi:
            by_step[it].append(kernels[j]); j += 1
        ki = j
    return by_step


def _partition_910b_by_fa(kernels, iters):
    """910B per-kernel type assignment (replaces time-bucket _partition for the
    async-multinomial-overlap case). Walks the ordered kernel stream and assigns
    each kernel to a forward by FA/ArgMax anchor + Op-Name category, so the async
    multinomial of forward i (which is TIME-INTERLEAVED with forward i+1's early
    kernels on a separate stream for B>48) is correctly assigned to forward i,
    not split across a time-bucket.

    - ArgMax (aclnnArgMax_*ArgMaxV2, 1/forward) = forward-end anchor. After it,
      the multinomial i follows → assigned to the SAME forward (last_argmax_fwd).
    - multinomial-specific names (InplaceFillScalar/Subs/IndexSelect/Repeat/
      Embedding/ArgMax/IndexPutImpl) → last_argmax_fwd (the forward whose
      argmax was last seen — handles interleaving: even if a forward-named
      kernel of i+1 appears mid-multinomial, the mn kernels stay with forward i).
    - forward-named (RmsNorm/Matmul/Contiguous/rotary/KV/FA/TensorMove/
      AddRmsNormBias/SwiGlu) → current fwd_idx, INCREMENTED on the first
      forward-named kernel after an ArgMax (forward i+1 start).
    - ambiguous (InplaceCopy_Slice/Cast — appear in both multinomial + forward
      glue) → current fwd_idx (correct for the multinomial's copies/casts since
      fwd_idx is still i until the first forward-named of i+1).

    `iters` = the non-idle sidecar iter_ids in order; forward_idx k ↔ iters[k].
    Returns {iter_id: [kernels]}."""
    from collections import defaultdict
    mn_names = ("inplacefillscalar", "subs", "indexselect", "aclnnrepeat",
               "embedding")
    # NOTE: IndexPutImpl + ArgMax are the ARGMAX pipeline (in-step, → fwd_idx),
    # NOT the async multinomial — excluded from mn_names so they don't
    # mis-assign to the prev forward's multinomial.
    fwd_names = ("rmsnorm", "aclnnmatmul", "aclnncontiguous", "_triton_rope",
                "reshapeandcache", "fusedinfer", "tensormove", "addrmsnormbias",
                "swiglu")
    by_step = defaultdict(list)
    fwd_idx = 0
    last_mn_fwd = -1
    after_argmax = False

    def _assign(idx, k):
        if 0 <= idx < len(iters):
            by_step[iters[idx]].append(k)
        else:
            by_step[None].append(k)  # overflow (uncaptured tail)

    for k in kernels:
        kernel_name = k[1]
        if not isinstance(kernel_name, str) or not kernel_name:
            raise ValueError(
                f"nsys kernel name must be a non-empty string: {kernel_name!r}")
        nm = kernel_name.lower()
        if "argmax" in nm:
            _assign(fwd_idx, k)
            last_mn_fwd = fwd_idx
            after_argmax = True
        elif "tensormove" in nm and after_argmax:
            # The 2nd TensorMove of the first-layer FA output is ASYNC (runs after
            # ArgMax, in the next sync window). Assign to last_mn_fwd (the
            # producer forward, like the multinomial), NOT the current (which
            # would misplace it at the front of the next step). The 1st TM
            # (in-step, after FA, before ArgMax) → fwd_idx via the fwd branch.
            _assign(last_mn_fwd if last_mn_fwd >= 0 else fwd_idx, k)
        elif any(m in nm for m in mn_names):
            _assign(last_mn_fwd if last_mn_fwd >= 0 else fwd_idx, k)
        elif any(f in nm for f in fwd_names):
            if after_argmax:
                fwd_idx += 1
                after_argmax = False
            _assign(fwd_idx, k)
        else:
            raise ValueError(
                f"cannot assign unknown nsys kernel to a serving step: {k[1]!r}")
    return by_step




def align_serving_grids(regen_by_step, nsys_by_step, synth_subs, nsys_subs,
                        label, compare_tile=False):
    """Per-step ordered grid (+tile) alignment of one kernel family, using
    PRE-LOADED per-step regen + nsys dicts (filtered in-memory by name substr).
    Returns (n_pairs, n_grid_match, n_grid_mismatch, examples[])."""
    # filter each step's sequences to the family (in-memory). regen_by_step
    # values are (name, grid, op_attr); nsys_by_step values are (start, name, grid).
    rfil = {st: [(nm, grid, op) for (nm, grid, op) in seq
                 if any(s in nm.lower() for s in synth_subs)]
            for st, seq in regen_by_step.items()}
    nfil = {st: [(r[1], r[2]) for r in seq
                 if any(s in r[1].lower() for s in nsys_subs)]
            for st, seq in nsys_by_step.items()}
    steps = [s for s in rfil if rfil[s] and s in nfil and nfil[s]]
    n_pairs = n_match = n_mis = 0
    examples = []
    for step in steps:
        rseq = rfil[step]; nseq = nfil[step]
        for i in range(min(len(rseq), len(nseq))):
            rnm, rgrid, rop = rseq[i]
            nnm, ngrid = nseq[i]
            if compare_tile:
                ntm, ntn = parse_tile(nnm)
                if ntm is None or ntn is None:
                    raise ValueError(f"cannot parse GEMM tile from kernel name {nnm!r}")
                if "M" not in rop or "N" not in rop:
                    raise ValueError(
                        f"GEMM regen op_attr missing M/N at step {step}: {rop!r}")
                M = rop["M"]
                N = rop["N"]
                if not consistent(N, M, ntm, ntn, ngrid[0], ngrid[1], ngrid[2]):
                    raise ValueError(
                        f"observed GEMM grid is inconsistent with tile at step {step}: "
                        f"kernel={nnm!r}, grid={ngrid}, M={M}, N={N}, "
                        f"tile=({ntm}, {ntn})")
            n_pairs += 1
            if compare_tile:
                rg = (rgrid[0], rgrid[1]); ng = (ngrid[0], ngrid[1])
            else:
                rg = rgrid; ng = ngrid
            if rg == ng:
                n_match += 1
            else:
                n_mis += 1
                if len(examples) < 8:
                    examples.append({
                        "step": step, "synth": (rnm[:50], list(rgrid)),
                        "nsys": (nnm[:50], list(ngrid)),
                        "op_attr": {k: rop[k] for k in ("M", "N", "K", "B", "S",
                                                         "KV", "SP", "HQ", "HKV")
                                    if k in rop}})
    return (n_pairs, n_match, n_mis, examples)


# ---------------------------------------------------------------------------
# Single-op: synthesis dispatch vs ncu GT (one exact shape per entry).
# ---------------------------------------------------------------------------

def _build_fp16_gemm_lut():
    """fp16 K-aware GEMM LUT from ncu gemm.json (the clean fp16 source): key
    (pad64(Mtok), pad64(N), K) -> dominant tile. Used to validate the GEMM
    synthesis pipeline reproduces the ncu tile (consistency check; ncu is the
    LUT source so every shape is in it)."""
    d = json.load(open(NCU_DIR / "gemm.json"))
    dist = defaultdict(Counter)
    for v in d.values():
        a = v["op_info"]["Attr"]
        Mtok = a["B"] * a["M"]
        K, N = a["K"], a["N"]
        kn = v["kernels"][0]["Metadata"]["Kernel Name"]
        t = parse_tile(kn)
        if t[0]:
            dist[(pad64(Mtok), pad64(N), K)][t] += 1
    return {k: cc.most_common(1)[0][0] for k, cc in dist.items()}


def _parse_ncu_grid(s):
    """Parse ncu 'Grid Size' "(x,y,z)" -> (x,y,z)."""
    if not isinstance(s, str):
        raise TypeError(
            f"NCU Grid Size must be a string, got {type(s).__name__}")
    match = re.fullmatch(
        r"\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*", s)
    if match is None:
        raise ValueError(
            f"NCU Grid Size must contain exactly three dimensions: {s!r}")
    dimensions = tuple(int(group) for group in match.groups())
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            f"NCU Grid Size dimensions must be positive: {s!r}")
    return dimensions


def align_single_op_gemm():
    """For each ncu GEMM entry: does the fp16 LUT tile reproduce the GT tile
    (parse from name) and the grid formula (ceil(N/tm), ceil(M/tn)) reproduce
    the GT grid (gx,gy)? Reports tile+grid consistency."""
    import math
    lut = _build_fp16_gemm_lut()
    d = json.load(open(NCU_DIR / "gemm.json"))
    n = tile_match = tile_mis = grid_match = grid_mis = 0
    ex_tile = ex_grid = []
    for v in d.values():
        a = v["op_info"]["Attr"]
        Mtok = a["B"] * a["M"]; K, N = a["K"], a["N"]
        kn = v["kernels"][0]["Metadata"]["Kernel Name"]
        gtile = parse_tile(kn)
        ggrid = _parse_ncu_grid(v["kernels"][0]["Metadata"]["Grid Size"])
        key = (pad64(Mtok), pad64(N), K)
        if key not in lut:
            raise KeyError(f"GEMM launch LUT missing observed shape {key}")
        lut_tile = lut[key]
        n += 1
        # tile consistency
        if lut_tile and lut_tile == gtile:
            tile_match += 1
        else:
            tile_mis += 1
            if len(ex_tile) < 8:
                ex_tile.append({"key": list(key), "gtile": list(gtile),
                                "lut": list(lut_tile) if lut_tile else None})
        # grid consistency (use GT tile to derive expected grid; ignore gz=splitK)
        if gtile[0] and gtile[1]:
            exp_gx = math.ceil(N / gtile[0]); exp_gy = math.ceil(Mtok / gtile[1])
            if (ggrid[0], ggrid[1]) == (exp_gx, exp_gy):
                grid_match += 1
            else:
                grid_mis += 1
                if len(ex_grid) < 8:
                    ex_grid.append({"key": list(key), "gtile": list(gtile),
                                    "ggrid": list(ggrid),
                                    "exp": [exp_gx, exp_gy]})
    return {"n": n, "tile_match": tile_match, "tile_mismatch": tile_mis,
            "grid_match": grid_match, "grid_mismatch": grid_mis,
            "tile_examples": ex_tile, "grid_examples": ex_grid}


def align_single_op_fa():
    """For each paged decode entry, compare production launch to NCU grid.

    Static contiguous benchmarks force an explicit SP and therefore are not a
    production-heuristic validation set. Paged/paged-varlen observations use
    the serving FA2 selection path. Grid convention matches the recipe:
    num_splits>1 -> (1, num_splits, B*HKV); else (1, B, HKV)."""
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.hardware_spec import HWSpec
    from src.kernel.launch_config.select_launch_config import select_launch_config
    d = json.load(open(NCU_DIR / "flash_attn.json"))
    hardware = HWSpec(ROOT / "configs" / "hardware" / "nvidia_a800.yaml")
    n = ns_match = ns_mis = grid_match = grid_mis = 0
    ex = []
    for v in d.values():
        a = v["op_info"]["Attr"]
        B, S, HQ, HKV, D = a["B"], a["S"], a["HQ"], a["HKV"], a["D"]
        if (a["Q_LEN"] != 1
                or a["kv_layout"] not in ("paged", "paged_varlen")):
            continue
        comp = [k for k in v["kernels"]
                if "flash_fwd_splitkv_kernel" in k["Metadata"]["Kernel Name"]
                and "combine" not in k["Metadata"]["Kernel Name"]]
        if not comp:
            continue
        ggrid = _parse_ncu_grid(comp[0]["Metadata"]["Grid Size"])
        if ggrid[1] == B and ggrid[2] == HKV:
            gt_sp = 1
        elif ggrid[2] == B * HKV:
            gt_sp = ggrid[1]
        elif ggrid[1] == B * HKV:
            gt_sp = ggrid[2]
        else:
            raise ValueError(
                "FLASH_ATTN split-kv grid does not contain B*HKV: "
                f"grid={ggrid}, B*HKV={B * HKV}")
        # ncu FA = decode: seq_len=1, kv_len=S.
        launch = select_launch_config(
            "nvidia_a800", KernelKind.FLASH_ATTN,
            {"batch": B, "q_len": 1, "kv_len": S, "head_dim": D,
             "num_kv_heads": HKV},
            hardware.kernel_dtype, "decode", {"is_decode": True},
            hardware=hardware,
        )
        num_splits = launch.tiling.num_kv_splits
        exp = launch.grid
        n += 1
        if num_splits == gt_sp:
            ns_match += 1
        else:
            ns_mis += 1
            if len(ex) < 10:
                ex.append({"attr": {k: a[k] for k in ("B", "S", "HQ", "HKV", "D", "SP")},
                           "observed_splits": gt_sp,
                           "synth_splits": num_splits, "ggrid": list(ggrid),
                           "exp_grid": list(exp)})
        if ggrid == exp:
            grid_match += 1
        else:
            grid_mis += 1
            if len(ex) < 10:
                ex.append({"attr": {k: a[k] for k in ("B", "S", "HQ", "HKV", "D", "SP")},
                           "observed_splits": gt_sp,
                           "synth_splits": num_splits, "synth_grid": list(exp),
                           "ggrid": list(ggrid)})
    return {"n": n, "num_splits_match": ns_match, "num_splits_mismatch": ns_mis,
            "grid_match": grid_match, "grid_mismatch": grid_mis,
            "examples": ex}


def align_single_op_moe():
    """For each NCU MoE entry: derive the tiling grid vs the ACTUAL NCU
    fused_moe_kernel grid."""
    import math
    from src.kernel.launch_config.fused_moe_tiling import derive_fused_moe_tiling
    d = json.load(open(NCU_DIR / "fused_moe.json"))
    n = w1_match = w1_mis = w2_match = w2_mis = 0
    ex = []
    for v in d.values():
        a = v["op_info"]["Attr"]
        TOK, E, TOPK, H, INT = a["TOK"], a["E"], a["TOPK"], a["H"], a["INT"]
        mt = a["moe_tiling"]
        w1 = mt["w1"]
        w2 = mt["w2"]
        BM, BN, BK, GSM = w1["BM"], w1["BN"], w1["BK"], w1["GSM"]
        for key in ("BM", "BN", "BK", "GSM"):
            if w2[key] != w1[key]:
                raise ValueError(
                    f"FUSED_MOE w1/w2 tiling mismatch for {key}: "
                    f"{w1[key]} != {w2[key]}")
        til = derive_fused_moe_tiling(
            TOK, E, TOPK, H, INT, BM, BN, BK, GSM)
        # actual ncu fused_moe_kernel grids (in order: w1 then w2)
        fm = [k for k in v["kernels"] if "fused_moe_kernel" in k["Metadata"]["Kernel Name"]]
        if len(fm) < 2:
            raise ValueError("FUSED_MOE observation requires w1 and w2 kernels")
        n += 1
        gt_w1 = _parse_ncu_grid(fm[0]["Metadata"]["Grid Size"])[0]
        gt_w2 = _parse_ncu_grid(fm[1]["Metadata"]["Grid Size"])[0]
        if til["w1"]["grid"] == gt_w1:
            w1_match += 1
        else:
            w1_mis += 1
            if len(ex) < 10:
                ex.append({"stage": "w1", "attr": {"TOK": TOK, "E": E, "TOPK": TOPK, "H": H, "INT": INT, "BM": BM, "BN": BN},
                           "synth": til["w1"]["grid"], "gt": gt_w1})
        if til["w2"]["grid"] == gt_w2:
            w2_match += 1
        else:
            w2_mis += 1
            if len(ex) < 10:
                ex.append({"stage": "w2", "attr": {"TOK": TOK, "E": E, "TOPK": TOPK, "H": H, "INT": INT, "BM": BM, "BN": BN},
                           "synth": til["w2"]["grid"], "gt": gt_w2})
    return {"n": n, "w1_match": w1_match, "w1_mismatch": w1_mis,
            "w2_match": w2_match, "w2_mismatch": w2_mis, "examples": ex}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SERVING_FAMILIES = [
    # (label, synth_subs, nsys_subs, compare_tile). Substrings must be SPECIFIC
    # enough that LIKE %sub% does not cross-match a sibling family: the FA
    # compute filter "flash_fwd_splitkv_kernel" must NOT match the combine
    # "...splitkv_combine_kernel" (the earlier "flash_fwd_splitkv" substring did,
    # corrupting the per-step zip). Use the full leaf token.
    ("GEMM-s16816 (tile+grid)", ["s16816gemm"], ["s16816gemm"], True),
    ("GEMV",                    ["gemvx", "gemv2t"],
                                ["gemvx", "gemv2t"], False),
    ("FA-splitkv-compute",      ["flash_fwd_splitkv_kernel"],
                                ["flash_fwd_splitkv_kernel"], False),
    ("FA-combine",              ["splitkv_combine"], ["splitkv_combine"], False),
    ("fused_moe",                ["fused_moe_kernel"], ["fused_moe_kernel"], False),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment-root",
        help="directory containing captured Serving case directories",
    )
    ap.add_argument("--case", default=None, help="single serving case dir name")
    ap.add_argument("--single-op", action="store_true",
                    help="run single-op (ncu) alignment instead of serving")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.single_op:
        out = {"GEMM": align_single_op_gemm(),
               "FLASHATTN": align_single_op_fa(),
               "FUSEDMOE": align_single_op_moe()}
        print(json.dumps(out, indent=2))
        if a.out:
            json.dump(out, open(a.out, "w"), indent=2)
        return

    if not a.experiment_root:
        ap.error("--experiment-root is required unless --single-op is used")
    experiment_root = Path(a.experiment_root)
    cases = (
        [experiment_root / a.case]
        if a.case else sorted(experiment_root.iterdir())
    )
    results = {}
    for cd in cases:
        if not (cd / "meta.json").exists():
            continue
        meta = json.load(open(cd / "meta.json"))
        if "hw_config" not in meta:
            raise ValueError(f"{cd / 'meta.json'} missing hw_config")
        if Path(meta["hw_config"]).stem != "nvidia_a800":
            raise NotImplementedError(
                "historical Serving trace alignment is implemented only for "
                f"nvidia_a800, got {Path(meta['hw_config']).stem!r}")
        # Load regen trace + nsys ONCE per case, partition nsys by step ONCE.
        regen = load_regen_trace(cd)
        nsys_kernels, iters, bounds = load_nsys_all(cd)
        if regen is None or nsys_kernels is None:
            print(f"[serving] {cd.name}: SKIP (no regen/nsys)", flush=True)
            continue
        regen_by_step = defaultdict(list)
        for step_id, nm, grid, op_attr, *_ in regen:
            regen_by_step[step_id].append((nm, grid, op_attr))
        nsys_by_step = _partition(nsys_kernels, bounds, iters)
        r = {}
        for label, ss, ns, ct in SERVING_FAMILIES:
            r[label] = align_serving_grids(regen_by_step, nsys_by_step, ss, ns,
                                          label, compare_tile=ct)
        results[cd.name] = r
        print(f"[serving] {cd.name}", flush=True)
        for label, (np_, nm, nx, ex) in r.items():
            pct = 100 * nm / max(1, np_)
            print(f"    {label:28s} pairs={np_:7d} grid_match={nm:7d} "
                  f"({pct:5.1f}%) mismatch={nx}", flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2)
