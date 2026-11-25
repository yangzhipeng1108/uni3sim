#!/usr/bin/env python3
"""Unified full-eval entrypoint for kernel and Serving validation.

One command that runs both evaluation layers:
  1. single-kernel physics-model accuracy — runs the default production
     hardware matrix and prints a cross-hardware comparison table.
  2. serving end-to-end prediction — runs one canonical hardware ID and one
     Serving suite from configs/dataset/, or every configured Serving hardware
     in ``--all-hardware`` mode.

Every development host MUST run this before `git commit` (see README «full pre-commit
eval»). Data lives on different hosts, so any card/part whose dataset/trace
is absent here is SKIPPED (not a failure) — the exit code is non-zero only when
a card/part that HAS data actually fails.

Usage:
    python scripts/evaluation/run_full_evaluation.py --hardware-id metax_c500
    python scripts/evaluation/run_full_evaluation.py --part kernel
    python scripts/evaluation/run_full_evaluation.py --hardware-id metax_c500 --serving-cases ShareGPT-Qwen3-8B_TP1_n128_qps4

    # All-hardware serving mode (a host that holds all three traces):
    # runs serving for A800 + C500 + 910B in one pass, each with its own
    # hardware config + historical alignment data, and prints a cross-hardware
    # serving table.
    python scripts/evaluation/run_full_evaluation.py --all-hardware
"""
import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent

from evaluation.kernel.dataset import (
    kernel_evaluation_dataset_available,
    kernel_evaluation_dataset_location,
    require_canonical_hardware_id,
)
from src.kernel.hardware_ids import CANONICAL_HARDWARE_IDS

from evaluation.json_shape import require_positive_metric as _positive_int

# Status constants for the final summary.
PASS = "PASS"
SKIPPED = "SKIPPED"
FAIL = "FAIL"




_DATASET_CONFIG_FIELDS = frozenset({
    "schema_version", "hardware_id", "kernel", "serving",
})
_CASE_MANIFEST_FIELDS = frozenset({"schema_version", "cases"})
_CASE_FIELDS = frozenset({
    "id", "model", "qps", "tensor_parallel_size", "expert_parallel_size",
    "scheduler_pipeline_depth",
})


def _require_exact_fields(mapping, fields, context):
    if not isinstance(mapping, dict):
        raise TypeError(f"{context} must be a mapping")
    missing = set(fields) - set(mapping)
    unexpected = set(mapping) - set(fields)
    if missing or unexpected:
        raise ValueError(
            f"{context} schema mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}")
    return mapping


def _require_non_empty_string(mapping, field, context):
    value = mapping[field]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}.{field} must be a non-empty string")
    return value


def _load_yaml_mapping(path, context):
    with open(path) as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must contain a mapping: {path}")
    return payload


def _load_serving_context(hardware_id, suite_name, root):
    canonical_id = require_canonical_hardware_id(hardware_id)
    if not isinstance(suite_name, str) or not suite_name:
        raise TypeError("serving suite name must be a non-empty string")
    dataset_path = root / "configs" / "dataset" / f"{canonical_id}.yaml"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset config not found: {dataset_path}")
    config = _require_exact_fields(
        _load_yaml_mapping(dataset_path, "dataset config"),
        _DATASET_CONFIG_FIELDS,
        "dataset config",
    )
    if config["schema_version"] != 1:
        raise ValueError(
            "dataset config schema_version must be 1, got "
            f"{config['schema_version']!r}")
    configured_id = _require_non_empty_string(
        config, "hardware_id", "dataset config")
    if configured_id != canonical_id:
        raise ValueError(
            f"dataset config hardware_id {configured_id!r} does not match "
            f"requested hardware {canonical_id!r}")
    _require_exact_fields(config["kernel"], {"pipeline"}, "dataset kernel")
    serving = _require_exact_fields(
        config["serving"], {"suites"}, "dataset serving")
    suites = serving["suites"]
    if not isinstance(suites, dict):
        raise TypeError("dataset serving.suites must be a mapping")
    if suite_name not in suites:
        raise ValueError(
            f"hardware {canonical_id!r} has no Serving suite {suite_name!r}")
    suite = _require_exact_fields(
        suites[suite_name], {"historical_alignment_root"},
        f"dataset serving.suites.{suite_name}")
    historical_root = _require_non_empty_string(
        suite, "historical_alignment_root",
        f"dataset serving.suites.{suite_name}")
    hardware_config = Path("configs") / "hardware" / f"{canonical_id}.yaml"
    hardware_path = root / hardware_config
    if not hardware_path.is_file():
        raise FileNotFoundError(
            f"hardware config not found: {hardware_path}")
    case_manifest = (
        Path("configs") / "dataset" / "serving" / f"{suite_name}.yaml")
    case_path = root / case_manifest
    if not case_path.is_file():
        raise FileNotFoundError(
            f"serving suite manifest not found: {case_path}")
    return {
        "hardware_id": canonical_id,
        "hardware_config": str(hardware_config),
        "case_manifest": str(case_manifest),
        "historical_alignment_root": historical_root,
        "suite_name": suite_name,
    }


def _serving_hardware_ids(root, suite_name):
    configured = []
    # The dataset configs present are the platforms that own a dataset at all.
    # A canonical id can exist without one: a hardware projection need not
    # own a captured dataset.
    for path in sorted((root / "configs" / "dataset").glob("*.yaml")):
        hardware_id = path.stem
        if hardware_id not in CANONICAL_HARDWARE_IDS:
            raise ValueError(
                f"{path}: dataset config names a non-canonical hardware id "
                f"{hardware_id!r}")
        config = _require_exact_fields(
            _load_yaml_mapping(path, "dataset config"),
            _DATASET_CONFIG_FIELDS,
            "dataset config",
        )
        if config["schema_version"] != 1:
            raise ValueError(
                f"{path}: dataset config schema_version must be 1")
        configured_id = _require_non_empty_string(
            config, "hardware_id", "dataset config")
        if configured_id != hardware_id:
            raise ValueError(
                f"{path}: hardware_id must match filename, got "
                f"{configured_id!r}")
        _require_exact_fields(config["kernel"], {"pipeline"}, "dataset kernel")
        serving = _require_exact_fields(
            config["serving"], {"suites"}, "dataset serving")
        suites = serving["suites"]
        if not isinstance(suites, dict):
            raise TypeError("dataset serving.suites must be a mapping")
        if suite_name in suites:
            configured.append(hardware_id)
    if not configured:
        raise ValueError(f"no hardware configures Serving suite {suite_name!r}")
    return configured


def _load_serving_cases(context, root):
    path = root / context["case_manifest"]
    manifest = _require_exact_fields(
        _load_yaml_mapping(path, "serving case manifest"),
        _CASE_MANIFEST_FIELDS,
        "serving case manifest",
    )
    if manifest["schema_version"] != 1:
        raise ValueError(
            "serving case manifest schema_version must be 1, got "
            f"{manifest['schema_version']!r}")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("serving case manifest.cases must be a non-empty list")
    case_ids = []
    for index, case in enumerate(cases):
        context = f"serving case manifest.cases[{index}]"
        _require_exact_fields(case, _CASE_FIELDS, context)
        case_ids.append(_require_non_empty_string(case, "id", context))
        _require_non_empty_string(case, "model", context)
        _positive_int(case, "qps", context)
        _positive_int(case, "tensor_parallel_size", context)
        _positive_int(case, "expert_parallel_size", context)
        _positive_int(case, "scheduler_pipeline_depth", context)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("serving case manifest contains duplicate case IDs")
    return cases


def _serving_case_sort_key(cid):
    """Order serving cases by model/config group, then by NUMERIC qps — so
    qps4 < qps16 < qps64 instead of the lexicographic qps16 < qps4 < qps64."""
    head, _, tail = cid.rpartition("_qps")
    if not head:
        return (cid, 0)
    try:
        return (head, int(tail))
    except ValueError as error:
        raise ValueError(f"invalid serving case qps suffix: {cid!r}") from error


# ── Kernel alignment (category + grid) per case ──

def _kernel_cat(name):
    """Map a kernel name to a category (same as check_kernel_alignment._cat)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"kernel name must be a non-empty string: {name!r}")
    n = name
    for k, v in [("splitkv_combine", "FAcomb"), ("flash_fwd_splitkv_kernel", "FA"),
                 ("reshape_and_cache", "KV"), ("fused_add_rms", "NORM"), ("rms_norm", "NORM"),
                 ("act_and_mul", "ACT"), ("unrolled_elementwise", "UEW"),
                 ("index_elementwise", "IDXE"), ("vectorized_elementwise", "VEW"),
                 ("elementwise_kernel", "EW"), ("reduce_kernel", "RED"),
                 ("splitKreduce", "RED"), ("cutlass", "CUT"),
                 ("ampere_s16816", "GEMM"), ("ampere_bf16_s16816", "GEMM"),
                 ("topkGating", "TOPK"), ("moe_align", "MOEALGN"),
                 ("count_and_sort", "CNTSORT"), ("fused_moe_kernel", "FMOE"),
                 ("cross_device_reduce", "CDR"), ("Histogram1D", "HIST"),
                 ("triton_poi", "SAMP"), ("vectorized_gather", "GATH"),
                 ("Kernel2", "GEMM"), ("gemvx", "GEMV"), ("s1688", "GEMM"),
                 ("rotary", "ROT")]:
        if k in n:
            return v
    return "OTH"


def _kernel_cat_910b(name):
    """910B/Ascend category: maps a CANN aclnn Op Name (the msprof kernel_name)
    to the same category vocabulary the regen ascend path emits, so a positional
    zip aligns regen-vs-msprof by TYPE before name/block_dim.

    The synthetic ascend emits use these Op Names verbatim (aclnnMatmul_..,
    RmsNorm, AddRmsNormBias, FusedInferAttentionScore, _triton_rope,
    ReshapeAndCacheNdKernel, SwiGlu, aclnnGroupedMatmul_.., MoeGatingTopK, ...).
    Order matters: longer/more-specific substrings first so e.g. aclnnGroupedMatmul
    is MOE (not GEMM) and aclnnInplaceCopy is GLUE (not GEMM via the bare 'matmul'
    trap — but InplaceCopy has no matmul, so fine)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"kernel name must be a non-empty string: {name!r}")
    n = name.lower()
    for k, v in [
        # MoE (must precede GEMM — grouped matmul + routing are MoE FFN)
        ("aclnngroupedmatmul", "MOE"), ("moegatingtopk", "MOE"),
        ("moeinitrouting", "MOE"), ("moetokenunpermute", "MOE"),
        # GEMM (dense aclnnMatmul; MatMulV3 variant is also GEMM)
        ("aclnnmatmul", "GEMM"),
        # attention / FA (aclnnFusedInferAttentionScore or bare name)
        ("fusedinferattentionscore", "FA"), ("fusedinfer", "FA"),
        # rotary (vllm-ascend triton rope kernel, msprof name '_triton_rope')
        ("_triton_rope", "ROT"), ("rotary", "ROT"), ("rope", "ROT"),
        # KV cache (ReshapeAndCacheNd)
        ("reshapeandcache", "KV"), ("reshape_and_cache", "KV"),
        # norm (aclnn RmsNorm / AddRmsNormBias)
        ("addrmsnormbias", "NORM"), ("rmsnorm", "NORM"),
        # activation (SwiGlu / act_and_mul / silu_and_mul)
        ("swiglu", "ACT"), ("act_and_mul", "ACT"), ("silu_and_mul", "ACT"),
        # embedding (aclnnEmbedding_GatherV2)
        ("aclnnembedding", "EMB"),
        # HCCL comms (TP2) — library constant, accepted
        ("hcom", "COMM"), ("hccl", "COMM"), ("allreduce", "COMM"),
        ("alltoall", "COMM"),
        # sampler / reduce (aclnnArgMax, aclnnReduceSum, reduce_kernel)
        ("argmax", "RED"), ("reducesum", "RED"), ("reduce_kernel", "RED"),
        ("reducemax", "RED"), ("reducemin", "RED"),
        # gather / indexSelect (sampler + logprob)
        ("aclnnindexselect", "GATH"), ("aclnnindexputimpl", "GATH"),
        ("vectorized_gather", "GATH"), ("indexselect", "GATH"),
        # framework glue aclnn elementwise/copy/cast/slice ops (not model ops;
        # emitted as framework-glue kernels by vLLM around the model forward)
        ("inplacecopy", "GLUE"), ("inplacescatter", "GLUE"),
        ("contiguous", "GLUE"), ("aclnncast", "GLUE"),
        ("aclnnsub", "GLUE"), ("aclnnmul", "GLUE"), ("aclnndiv", "GLUE"),
        ("aclnnexp", "GLUE"), ("aclnnrepeat", "GLUE"),
        ("inplacefillscalar", "GLUE"), ("inplacezero", "GLUE"),
        ("broadcastto", "GLUE"), ("aclnnindex", "GLUE"),
    ]:
        if k in n:
            return v
    return "OTH"


def _msprof_n_ranks(case_dir):
    """Number of distinct deviceIds across ALL msprof TASK tables (TP1=1, TP2=2).
    For TP2, each rank's compute kernels are in a separate PROF dir/db."""
    import sqlite3, glob
    from pathlib import Path
    cands = glob.glob(str(Path(case_dir) / "msprof" / "PROF_*" / "msprof_*.db"))
    cands = [c for c in cands if "/sqlite/" not in c and "/device_" not in c]
    all_devs = set()
    for db in cands:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            for r in conn.execute("SELECT DISTINCT deviceId FROM TASK"):
                all_devs.add(r[0])
    if not all_devs:
        raise ValueError(f"no msprof TASK device IDs found under {case_dir}")
    return len(all_devs)


def _warmup_iter_ids(case_dir):
    """Set of sidecar iter_ids that are warmup (cudagraph-capture / shape-
    inference passes), identified from the scheduler sidecar: an iter is warmup
    iff any of its scheduled request keys starts with 'warmup'. These are NOT
    real serving execution (the E2E predictor drops them too) and the regen
    recipe does not model their setup-head cascade, so the per-kernel alignment
    metric skips them. Returns an empty set if no scheduler sidecar."""
    import json
    from pathlib import Path
    p = Path(case_dir) / "plugin_trace.json_scheduler.jsonl"
    if not p.exists():
        return set()
    warmup = set()
    with open(p) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{p}:{line_number} must contain a JSON object")
            scheduled = record["scheduled"]
            if not isinstance(scheduled, dict):
                raise TypeError(
                    f"{p}:{line_number} scheduled must be an object")
            if any(str(key).startswith("warmup") for key in scheduled):
                warmup.add(record["iter_id"])
    return warmup


def _compute_kernel_alignment(case_dir, rank=0):
    """Per-case per-rank kernel alignment. Returns a list of per-rank dicts
    (one per rank), each with:
      nsys_total, regen_total, cat_mm (positional type mismatch within min),
      extra_regen, extra_nsys, nsys_unaligned (= cat_mm + extra_nsys),
      un_pct (= nsys_unaligned / nsys_total), grid_mm, gm_pct (= grid_mm / cat_match),
      grid_mm_by_cat.
    Returns None if no regen/nsys."""
    from pathlib import Path
    from collections import defaultdict, Counter
    import sqlite3
    from evaluation.serving.traces.trace_alignment_validation import (
        load_nsys_all, load_msprof_all, _msprof_db_path, load_regen_trace,
        _partition, _partition_910b_by_fa)
    cd = Path(case_dir)
    sp = cd / "nsys_profile.sqlite"
    is_910b = False
    if sp.exists():
        with sqlite3.connect(str(sp)) as conn:
            n_ranks = conn.execute(
                "SELECT COUNT(DISTINCT deviceId) "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        if n_ranks <= 0:
            raise ValueError(f"no kernel device IDs found in {sp}")
    elif _msprof_db_path(cd):
        # 910B / Ascend: no nsys sqlite; the main msprof_*.db TASK ⋈
        # COMPUTE_TASK_INFO is the per-invocation ground truth (Op Name + Block
        # Dim). n_ranks = distinct deviceId in TASK (TP1=1, TP2=2).
        is_910b = True
        n_ranks = _msprof_n_ranks(cd)
    else:
        # JSON fallback (C500): count ranks from nsys_kernels_full.json's per_rank.
        fullp = cd / "nsys_kernels_full.json"
        n_ranks = 1
        if fullp.exists():
            with open(fullp) as f:
                d = json.load(f)
            pr = d["per_rank"]
            if not isinstance(pr, dict) or not pr:
                raise ValueError(f"{fullp} per_rank must be a non-empty object")
            n_ranks = len(pr)
        else:
            return None
    cat_fn = _kernel_cat_910b if is_910b else _kernel_cat
    results = []
    for r in range(n_ranks):
        nk, iters, bounds = (load_msprof_all(cd, device_id=r) if is_910b
                             else load_nsys_all(cd, device_id=r))
        if nk is None or not iters or bounds is None:
            raise ValueError(
                f"incomplete trace alignment inputs for {cd}, rank {r}")
        nby = (_partition_910b_by_fa(nk, iters) if is_910b
               else _partition(nk, bounds, iters))
        # NOTE: B>48 async-overlap reassignment (move forward-named kernels after
        # the multinomial start to the next bucket) was tried but didn't help
        # (step 254 still 583mm — the overlap is in the PREV bucket's multinomial
        # region, and the per-kernel name-based move is buggy for duplicates +
        # the InplaceCopy_Slice/Cast ambiguity). The FA-count type assignment
        # (_partition_910b_by_fa) replaces the time-bucket and handles the
        # interleaving correctly.
        regen = load_regen_trace(cd, rank=r)
        if regen is None:
            continue
        rby = defaultdict(list)
        for sid, nm, grid, op, blk, smem in regen:
            if is_910b:
                # 910B launch geometry = Block Dim (grid[0]); normalize regen to a
                # 1-tuple so grid_mm compares block_dim (this IS launch_mm on
                # 910B — no CUDA grid/block/smem concept). msprof grids are
                # already (block_dim,). smem set to 0 on the regen side. The mix
                # (block pos) is a CANN constant per kernel family on 910B:
                # FusedInferAttentionScore uses mix_block_dim=48 (verified: all
                # 38160 FA invocations), everything else mix=0. Set the regen
                # mix to match so block_mm reflects real mismatches (not the
                # forced-0 artifact that flagged every FA as block_mm).
                g0 = grid[0] if grid else 0
                _mix = 48 if "FusedInfer" in nm else 0
                rby[sid].append((nm, (g0,), (_mix,), 0))
            else:
                rby[sid].append((nm, tuple(grid), blk, smem))
        # Flat-list fallback (C500 / A800, NOT 910b): if all nsys kernels have
        # ts=0 (flat-list format without ts_ns, e.g. a pre-rebuild
        # Qwen3-32B-TP2 nsys_kernels_full.json), the ts-based _partition puts
        # all kernels in the first step. Use count-based partition instead.
        # Prefer mctracer_step_counts.jsonl (per-step mcLaunchKernel count from
        # mctracer sync boundaries) — proportional-scaled to nsys total.
        # A flat trace requires explicit mcTracer step counts; guessing from
        # regenerated output would make the validation self-fulfilling.
        if (not is_910b) and nk and all(k[0] == 0 for k in nk[:100]):
            nby = {}
            mc_path = cd / "mctracer_step_counts.jsonl"
            if not mc_path.exists():
                raise FileNotFoundError(
                    f"flat trace requires explicit step counts: {mc_path}")
            mc_counts = {}
            with open(mc_path) as count_file:
                for line_number, line in enumerate(count_file, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    mc_counts[record["iter_id"]] = record["n_kern"]
            missing_counts = [it for it in iters if it not in mc_counts]
            if missing_counts:
                raise ValueError(
                    f"{mc_path} is missing step counts for {missing_counts[:8]}")
            mc_total = sum(mc_counts.values())
            if mc_total <= 0:
                raise ValueError(f"{mc_path} has no positive kernel counts")
            nsys_total = len(nk)
            cursor = 0
            for it in iters:
                nsys_c = int(round(mc_counts[it] * nsys_total / mc_total))
                nby[it] = nk[cursor:cursor + nsys_c]
                cursor += nsys_c
        for it in iters:
            if it not in nby:
                nby[it] = []
        nsys_total = 0; regen_total = 0
        cat_match = 0; cat_mm = 0
        extra_regen = 0; extra_nsys = 0
        grid_mm = Counter()   # SAME kernel name, grid (=Block Dim on 910B) differs → launch_mm
        name_mm = Counter()   # same CATEGORY, DIFFERENT kernel name
        block_mm = Counter()  # same name, grid same, BLOCK (=mix_block_dim on 910B) differs
        smem_mm = Counter()   # same name, grid same, block same, SMEM differs
        # Skip warmup iters (cudagraph-capture / shape-inference passes that are
        # NOT real serving execution — the E2E predictor drops them too). On 910B
        # the single warmup iter (warmup_0) runs the forward 2× + a setup-head
        # cascade that the regen recipe does not model; counting it inflates
        # nsys_x/type_mm by ~1400 for 1 step. Identified via the scheduler
        # sidecar: an iter is warmup iff any scheduled key starts with 'warmup'.
        warmup_ids = _warmup_iter_ids(cd)
        # msprof capture-tail: the LAST forward(s) whose msprof capture ended
        # mid-execution (regen emits the full forward the scheduler ran, but
        # msprof only has the captured prefix). Two shapes:
        #  (a) TRUNCATED — msprof ends mid-body (FA/GEMM/NORM/...), not at the
        #      forward end (Embedding for Qwen3, async-batched rotary
        #      IndexSelect tail for Llama). regen is always complete (operator
        #      records); regen longer than msprof.
        #  (b) BLEED — the complete forward IMMEDIATELY BEFORE a truncated tail
        #      whose own trailing async-batch (Llama rotary IndexSelect×N) spans
        #      the capture boundary: regen emits the full N, msprof captured
        #      only the prefix (the rest lost past the capture end). regen >
        #      msprof. Skipped because the over-emit is a capture limitation,
        #      not a regen error (regen's full count is the TRUE count).
        # Comparing a full regen forward to a partial msprof is unfair, so
        # skip both — symmetric to the warmup skip at the capture head.
        def _ns_is_truncated(_ns):
            if not _ns:
                return True  # 0 nsys kernels = fully missing step (capture tail)
            _nl = _ns[-1][0] if _ns[-1] else ""
            return cat_fn(_nl) not in ("EMB", "GATH")
        # Pre-pass: cache rs/ns per iter + find truncated-tail iters.
        _rsns = {}
        _trunc = set()
        for _it in iters:
            if _it in warmup_ids:
                continue
            _rs = rby[_it]
            _ns = [(rr[1], tuple(rr[2]), rr[3], rr[4]) for rr in nby[_it]]
            _rsns[_it] = (_rs, _ns)
            if _ns_is_truncated(_ns) and len(_rs) > len(_ns):
                _trunc.add(_it)
        # Bleed: a non-truncated iter at the capture end (immediately before a
        # truncated tail, OR among the last 2 non-warmup iters) whose trailing
        # async-batch (Llama rotary IndexSelect×N) was PARTIALLY captured —
        # regen > msprof AND the extra regen tail is all GATH (the async
        # IndexSelect batch). regen's full count is the TRUE count; msprof
        # under-captured at the boundary. Skip (capture limitation, not a
        # regen error). Qwen3 has no async IndexSelect batch (rotary is
        # _triton_rope in-line), so it never trips this.
        _capture_tail = set(_trunc)
        _non_wu = [_it for _it in iters if _it not in warmup_ids]
        _last2 = set(_non_wu[-2:]) if _non_wu else set()
        for _i, _it in enumerate(iters):
            if _it in warmup_ids or _it in _capture_tail:
                continue
            if _it not in _rsns:
                raise RuntimeError(f"alignment pre-pass omitted iteration {_it!r}")
            _rs, _ns = _rsns[_it]
            if len(_rs) <= len(_ns):
                continue
            _before_trunc = (_i + 1 < len(iters)) and (iters[_i + 1] in _trunc)
            _at_end = _it in _last2
            if not (_before_trunc or _at_end):
                continue
            _extra = _rs[len(_ns):]
            if _extra and all(cat_fn(e[0]) == "GATH" for e in _extra):
                _capture_tail.add(_it)
        for it in iters:
            if it in warmup_ids or it in _capture_tail:
                continue
            if it not in _rsns:
                raise RuntimeError(f"alignment pre-pass omitted iteration {it!r}")
            rs, ns = _rsns[it]
            nsys_total += len(ns)
            regen_total += len(rs)
            m = min(len(rs), len(ns))
            for i in range(m):
                rc, nc = cat_fn(rs[i][0]), cat_fn(ns[i][0])
                if rc == nc:
                    cat_match += 1
                    if rs[i][0] != ns[i][0]:
                        name_mm[rc] += 1
                    elif tuple(sorted(rs[i][1])) != tuple(sorted(ns[i][1])):
                        # Compare the CTA-grid as a MULTISET, not an ordered
                        # tuple: the launch geometry (total CTAs, occupancy,
                        # per-tile work) is invariant under axis permutation, so
                        # a swizzle like (1187,1,1) vs (1,1187,1) is the SAME
                        # work re-labeled — not a modeling error. The MetaX
                        # (C500) mcblas autotuner picks the grid-axis assignment
                        # per-launch by a closed heuristic (M tensor-core
                        # alignment + occupancy) that is not derivable from
                        # source and varies run-to-run for degenerate/multi-token
                        # shapes (e.g. multi-token lm_head, N=vocab). A genuine
                        # tile error still changes the CTA COUNT -> the sorted
                        # multisets differ -> still flagged.
                        grid_mm[rc] += 1
                    elif rs[i][2] != ns[i][2]:
                        block_mm[rc] += 1
                    elif rs[i][3] != ns[i][3]:
                        smem_mm[rc] += 1
                else:
                    cat_mm += 1
            extra_regen += max(0, len(rs) - len(ns))
            extra_nsys += max(0, len(ns) - len(rs))
        nsys_un = cat_mm + extra_nsys
        results.append({
            "rank": r,
            "nsys_total": nsys_total,
            "regen_total": regen_total,
            "cat_mm": cat_mm,
            "extra_regen": extra_regen,
            "extra_nsys": extra_nsys,
            "nsys_unaligned": nsys_un,
            "un_pct": 100.0 * nsys_un / nsys_total if nsys_total else 0,
            "cat_match": cat_match,
            "grid_mm": sum(grid_mm.values()),
            "gm_pct": 100.0 * sum(grid_mm.values()) / cat_match if cat_match else 0,
            "grid_mm_by_cat": dict(grid_mm),
            "name_mm": sum(name_mm.values()),
            "name_mm_by_cat": dict(name_mm),
            "block_mm": sum(block_mm.values()),
            "block_mm_by_cat": dict(block_mm),
            "smem_mm": sum(smem_mm.values()),
            "smem_mm_by_cat": dict(smem_mm),
        })
    return results if results else None


def _align_one_case(cdir):
    """Module-level worker for ProcessPoolExecutor (closures aren't picklable).
    Computes one case's kernel alignment; returns (cid_placeholder, result) —
    the caller knows the cid from the submission order."""
    result = _compute_kernel_alignment(cdir, rank=0)
    if result is None:
        raise ValueError(f"no supported profiler trace found under {cdir}")
    return result


def _compute_alignment_for_cases(case_dirs, max_workers=None):
    """Compute kernel alignment for many cases in parallel. Each case is
    independent (its own nsys_profile.sqlite + .regen_cache pkl). The
    bottleneck is pkl-unpickling ~2.5M tuples/case (32B) + the per-kernel
    zip/compare — both create Python objects, which is GIL-bound, so a
    ThreadPoolExecutor gives NO speedup (threads contend on the GIL). A
    ProcessPoolExecutor (separate processes, independent GILs) gives real
    ~3× speedup (12 cases 263s → ~90s; the 3 huge 32B cases that dominate
    run concurrently). Results are byte-identical to the sequential loop
    (each case is computed in isolation; _print_alignment_table sorts).

    Args:
      case_dirs: {cid: Path} — one path per case to align.
      max_workers: cap on processes (default: min(8, len, os.cpu_count())).
    Returns: {cid: alignment_list_or_None} (None entries dropped).
    """
    from concurrent.futures import ProcessPoolExecutor
    import os as _os
    out = {}
    if not case_dirs:
        return out

    n = len(case_dirs)
    if max_workers is None:
        cpu_count = _os.cpu_count()
        if cpu_count is None:
            raise RuntimeError(
                "cannot size alignment workers: os.cpu_count() is unknown")
        cap = min(8, n, cpu_count)
    elif isinstance(max_workers, int) and not isinstance(max_workers, bool):
        if max_workers <= 0:
            raise ValueError("alignment max_workers must be positive")
        cap = max_workers
    else:
        raise TypeError("alignment max_workers must be an integer or None")
    # ProcessPool: fork inherits the parent's sys.path/imports (Linux default),
    # so the worker can import run_full_eval + validate_trace_alignment. The
    # function/arg/return are all picklable (module-level fn, Path, list[dict]).
    with ProcessPoolExecutor(max_workers=cap) as pool:
        futs = {pool.submit(_align_one_case, cd): cid for cid, cd in case_dirs.items()}
        for fut in futs:
            cid = futs[fut]
            al = fut.result()
            if al is None:
                continue
            if not isinstance(al, list) or not al:
                raise ValueError(
                    f"alignment for {cid!r} must be a non-empty list or None")
            out[cid] = al
    return out


def _print_alignment_table(rows):
    """Per-case per-rank kernel alignment table: nsys kernel count, type-seq
    mismatch count/%, grid mismatch count/%. Shows which cases are fully
    aligned (type+grid) vs partially (type only) vs not."""
    print("\n  [kernel alignment · per-case per-rank]")
    print("    nsys_kern = actual nsys kernel count | regen_kern = regenerated kernel count")
    print("    type_mm = ordered type mismatch count | regen_x = extra regenerated kernels | nsys_x = extra nsys kernels")
    print("    unaligned = unaligned nsys kernels(type_mm + nsys_x)")
    print("    grid_mm = grid differs for the same kernel name | blk_mm = block differs for the same name/grid | smem_mm = smem differs for the same name/grid/block")
    _hdr = (f"  {'case':<40s} {'rk':>2s} {'nsys_kern':>9s} {'regen_kern':>10s} {'type_mm':>7s} "
            f"{'regen_x':>7s} {'nsys_x':>7s} {'unaligned':>9s} {'grid_mm':>7s} {'blk_mm':>6s} {'smem_mm':>7s}")
    print(_hdr)
    print("  " + "-" * (len(_hdr) - 2))
    mism_rows = []
    for cid in sorted(rows, key=_serving_case_sort_key):
        c = rows[cid]
        al = c["_kernel_alignment"]
        for r in al:
            _cid = cid if r['rank'] == 0 else ""  # case name only for rank 0
            print(f"  {_cid:<40s} {r['rank']:>2d} {r['nsys_total']:>9d} {r['regen_total']:>10d} "
                  f"{r['cat_mm']:>7d} {r['extra_regen']:>7d} {r['extra_nsys']:>7d} "
                  f"{r['nsys_unaligned']:>9d} {r['grid_mm']:>7d} "
                  f"{r['block_mm']:>6d} {r['smem_mm']:>7d}")
        # collect top grid mismatch categories for the separate table
        top = Counter()
        for r in al:
            for k, v in r["grid_mm_by_cat"].items():
                top[k] += v
        if top:
            mism_rows.append((cid, top.most_common(3)))
    # Top mismatch as a SEPARATE table (not interleaved with the alignment rows)
    if mism_rows:
        print()
        print("  [top grid mismatch categories · per-case]")
        for cid, tops in mism_rows:
            print(f"  {cid:<40s} {', '.join(f'{k}:{v}' for k, v in tops)}")


# Each evaluation invocation produces one timestamped, gitignored log.  The
# directory axes describe evaluation scope and hardware; this is not a generic
# workspace for intermediate artifacts.
EXPERIMENT_LOG_ROOT = ROOT / "experiment_logs"


class _Tee:
    """Mirror everything written to stdout into a log file too, so a full-eval
    run is captured verbatim (all per-case output + the summary) while still
    streaming live to the console."""

    def __init__(self, stream, *file_handles):
        self._stream = stream
        self._file_handles = file_handles

    def write(self, data):
        self._stream.write(data)
        for fh in self._file_handles:
            fh.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        for fh in self._file_handles:
            fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# Module-level log file handle — set by main() so _print_kernel_matrix can write
# full tables directly to the log (not to terminal).
_eval_log_fh = None
_eval_log_path = None


def _wall_clock_stamp():
    """Local YYYYmmdd_HHMMSS timestamp for the log filename."""
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd, cwd, quiet=False):
    """Run a subprocess, capturing stdout. Returns (returncode, captured_text).

    quiet=False streams output live (single serial sub-eval, e.g. serving).
    quiet=True buffers silently (parallel card runs would interleave) —
    caller prints a summary afterward.
    """
    if not quiet:
        print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        if not quiet:
            sys.stdout.write(line)
            sys.stdout.flush()
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def _parse_kernel_table(text):
    """Parse the kernel-evaluation bucket table into ordered rows.

    Returns list of (label, count, forward_metrics, roofline_metrics, pipeweave_metrics)
    in table order. Each metrics tuple is (mape, med, wmae). Roofline / PipeWeave
    = None if that column is absent. count = kernel count for this bucket (int).
    """
    def parse_metrics(segment, column):
        if segment.strip() == "Err":
            return None
        number_pattern = (
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
        values = re.findall(number_pattern, segment)
        if len(values) != 3:
            raise ValueError(
                f"kernel table {column} column must contain exactly three metrics: "
                f"{segment!r}")
        try:
            return tuple(float(value) for value in values)
        except ValueError as error:
            raise ValueError(
                f"kernel table {column} contains an invalid metric: "
                f"{segment!r}") from error

    has_pipeweave = any(
        "Dataset" in line and "PipeWeave" in line
        for line in text.splitlines())
    metric_count = 4 if has_pipeweave else 3
    rows = []
    for line in text.splitlines():
        segs = line.split("│")
        # Tree glyphs may add ``│`` inside the label, so parse metric columns
        # from the right.  Standard evaluation has Forward/Oracle/Roofline;
        # explicitly requested PipeWeave adds the fourth metric column.
        if len(segs) < metric_count + 1:
            continue
        label_part = "│".join(segs[:-metric_count])
        if has_pipeweave:
            forward_seg, oracle_seg, roofline_seg, pipeweave_seg = (
                segs[-4:])
        else:
            forward_seg, oracle_seg, roofline_seg = segs[-3:]
            pipeweave_seg = None
        if "MAPE" in forward_seg:
            continue
        head = label_part.rstrip()
        if head.strip().startswith("Dataset"):
            continue
        # Extract the kernel count (trailing integer in the label, before the
        # first "│"). e.g. "├─ GEMM   220" -> count=220, label="GEMM".
        count_match = re.search(r"(\d+)\s*$", head)
        if count_match is None:
            raise ValueError(f"kernel table row has no observation count: {line!r}")
        count = int(count_match.group(1))
        head = re.sub(r"\s+\d+$", "", head)
        label = head.rstrip()
        stripped = _kernel_row_key(label)
        if not stripped or stripped.startswith("Dataset"):
            continue
        oracle_metrics = parse_metrics(oracle_seg, "Oracle")
        fwd_metrics = parse_metrics(forward_seg, "Forward")
        roof_metrics = parse_metrics(roofline_seg, "Roofline")
        pw_metrics = (
            parse_metrics(pipeweave_seg, "PipeWeave")
            if pipeweave_seg is not None else None)
        rows.append((label, count, oracle_metrics, fwd_metrics, roof_metrics, pw_metrics))
    return rows


def _kernel_row_key(label):
    """Normalize a tree label to a comparable key across cards (strip tree
    chars + whitespace), so the same bucket lines up in every column."""
    k = label.replace("├", "").replace("└", "").replace("─", "")
    k = k.replace("│", "").strip()
    return k



# The default table compares the four bundled GPU hardware targets in this
# presentation order.
DEFAULT_KERNEL_CARDS = (
    "nvidia_a800",
    "nvidia_h200",
    "metax_c500",
    "ascend_910b",
)


_FORWARD_SUBSET_HEADER = (
    "Forward rows scored over a subset of their N (unsupported cases excluded)")
_FORWARD_UNSUPPORTED_HEADER = (
    "Forward unsupported cases (explicit, excluded from scoring)")


def _parse_forward_coverage(text, rows):
    """Parse the explicit Forward exclusion sections of one card's output.

    Returns ``{"scored": [[row_index, scored, N], ...], "unsupported_cases":
    {label: count}}``; both are empty when every case was priced.  Row
    indexes address ``rows`` (the parsed bucket table) and are checked
    against its labels and counts.
    """

    lines = text.splitlines()
    scored = []
    unsupported = {}
    section = None
    for line in lines:
        stripped = line.strip()
        if stripped == _FORWARD_SUBSET_HEADER:
            section = "scored"
            continue
        if stripped == _FORWARD_UNSUPPORTED_HEADER:
            section = "unsupported"
            continue
        if not stripped or set(stripped) == {"═"}:
            continue
        if section == "scored":
            match = re.fullmatch(
                r"row\s+(\d+)\s+scored\s+(\d+)\s+/\s+(\d+)\s+(.+)", stripped)
            if match is None:
                section = None
                continue
            row_index, count, total = (int(match.group(i)) for i in (1, 2, 3))
            if (row_index >= len(rows)
                    or _kernel_row_key(rows[row_index][0])
                    != _kernel_row_key(match.group(4))
                    or rows[row_index][1] != total or count >= total):
                raise ValueError(
                    f"Forward scored-subset line does not match its row: "
                    f"{line!r}")
            scored.append([row_index, count, total])
        elif section == "unsupported":
            match = re.fullmatch(r"(\S+)\s+cases=\s*(\d+)(?:\s+/\s+\d+)?",
                                 stripped)
            if match is None:
                if not stripped.startswith("reason cases="):
                    section = None
                continue
            unsupported[match.group(1)] = int(match.group(2))
    return {"scored": scored, "unsupported_cases": unsupported}


def _run_one_kernel_card(
        card, root, case_jobs=None, case_timing_dir=None,
        semantic_output_dir=None, report_output_dir=None):
    """Run single-kernel eval for one card.

    Returns ``(status, summary_str, rows, forward_coverage)`` where rows is
    the ordered bucket list (empty unless PASS) and ``forward_coverage`` the
    explicit Forward exclusions (``_parse_forward_coverage``).
    """
    # SKIP if the card's evaluation dataset (canonical kernel_dataset/ or its
    # configured candidate root) is missing or empty.
    if not kernel_evaluation_dataset_available(card):
        _, dataset_dir = kernel_evaluation_dataset_location(card)
        return SKIPPED, f"no dataset at {dataset_dir}", [], None

    cmd = [sys.executable, "scripts/evaluation/kernel/run_kernel_evaluation.py",
           "--hardware-id", card, "--mode", "standard"]
    if case_jobs is not None:
        cmd.extend(("--jobs", str(case_jobs)))
    if case_timing_dir is not None:
        case_timing_path = Path(case_timing_dir) / f"{card}.json"
        cmd.extend(("--case-timing-output", str(case_timing_path)))
    if semantic_output_dir is not None:
        semantic_output_path = Path(semantic_output_dir) / f"{card}.json"
        cmd.extend(("--semantic-output", str(semantic_output_path)))
    from evaluation.reporting.kernel_overview import default_kernel_report_path
    if report_output_dir is not None:
        report_path = Path(report_output_dir) / (card + ".html")
        cmd.extend(("--interactive-report", "--report-output", str(report_path)))
    else:
        # Report-only default: never materialize an HTML report path (and its
        # empty report directory) when the per-card run does not write one.
        report_path = None
    rc, text = _run(cmd, root, quiet=True)
    if rc != 0:
        return FAIL, f"run_kernel_evaluation.py exited {rc}", [], None
    rows = _parse_kernel_table(text)
    if not rows:
        return (FAIL, "could not parse run_kernel_evaluation.py output", [],
                None)
    coverage = _parse_forward_coverage(text, rows)
    summary_keys = (
        "ALL", "GEMM", "FLASHATTN/EAGERATTN",
        "FUSEDMOE/UNFUSEDMOE")
    top = {_kernel_row_key(l): (f, r, p) for l, _, _, f, r, p in rows
           if _kernel_row_key(l) in summary_keys}

    def summary_metric(metrics):
        return "Err" if metrics is None else f"{metrics[0]:.1f}%"

    summary = "Forward/Roofline MAPE  " + "  ".join(
        f"{key}={summary_metric(top[key][0])}/"
        f"{summary_metric(top[key][1])}"
        for key in summary_keys if key in top)
    if coverage["unsupported_cases"]:
        summary += (
            "  Forward unsupported cases="
            f"{coverage['unsupported_cases']['ALL']}")
    return PASS, summary, rows, coverage


def run_kernel_matrix(
        root, jobs, case_jobs=None, case_timing_dir=None,
        semantic_output_dir=None, report_output_dir=None):
    """Run the default single-kernel hardware matrix in parallel; return
    {card: (status, summary, rows)}. ``jobs`` bounds card concurrency and
    ``case_jobs`` bounds each card's independent Kernel case workers. Prints
    per-card and total elapsed."""
    from concurrent.futures import ThreadPoolExecutor
    if case_jobs is not None:
        _positive_int({"case_jobs": case_jobs}, "case_jobs", "kernel matrix")
    if case_timing_dir is not None:
        Path(case_timing_dir).mkdir(parents=True, exist_ok=True)
    if semantic_output_dir is not None:
        Path(semantic_output_dir).mkdir(parents=True, exist_ok=True)
    out = {}
    max_workers = (min(len(DEFAULT_KERNEL_CARDS), jobs)
                   if jobs else len(DEFAULT_KERNEL_CARDS))
    print(f"[kernel] running {len(DEFAULT_KERNEL_CARDS)} cards in parallel "
          f"(max_workers={max_workers}) ...", flush=True)
    t_all = time.time()

    def _timed(card):
        t0 = time.time()
        res = _run_one_kernel_card(
            card, root, case_jobs=case_jobs,
            case_timing_dir=case_timing_dir,
            semantic_output_dir=semantic_output_dir, report_output_dir=report_output_dir)
        return res, time.time() - t0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_timed, card): card
            for card in DEFAULT_KERNEL_CARDS
        }
        for fut in futs:
            card = futs[fut]
            res, dt = fut.result()
            out[card] = res
            print(f"[kernel]   {card}: {res[0]}  ({dt:.3f}s)", flush=True)
    print(f"[kernel] all cards done in {time.time() - t_all:.1f}s "
          f"(wall; cards ran concurrently)", flush=True)
    return out


def _print_kernel_matrix(matrix):
    """Print Forward, Oracle, Roofline, and optional PipeWeave tables.

    Every table uses the same kernel-evaluation row structure and reports
    MAPE/WMAE per card.
    """
    cards = list(DEFAULT_KERNEL_CARDS)
    ref_rows = next((rows for (_, _, rows, _) in
                     (matrix[c] for c in cards) if rows), None)
    if ref_rows is None:
        print("  (no card produced a table — all SKIPPED/FAILED)")
        for card in cards:
            print(f"    {card}: {matrix[card][0]} — {matrix[card][1]}")
        return
    per_card = {}
    for card in cards:
        status, _, rows, _ = matrix[card]
        if status == PASS:
            d = {}
            for label, count, ora_m, fwd_m, roof_m, pw_m in rows:
                d[_kernel_row_key(label)] = (count, ora_m, fwd_m, roof_m, pw_m)
            per_card[card] = d
        else:
            per_card[card] = None

    label_w = max(len(l) for l, *_, in ref_rows) + 2
    m_w = 9   # wide enough for "1.2e+11" (7) or "12345.7" (7), right-aligned in 9
    count_w = 6
    # per-card block: │ N MAPE WMAE │
    blk_inner = count_w + m_w * 2
    blk_w = blk_inner + 3   # +3 for "│ " on each side (2) + 1
    div = "│"

    def _fmt_metric(v):
        # Compact so out-of-range values (e.g. PipeWeave on non-GPU cards, where
        # the A100 MLP extrapolates to 1e4–1e7 % MAPE) don't overflow the column
        # and merge with the next. Small values keep one decimal.
        return f"{v:.1f}" if abs(v) < 10000 else f"{v:.1e}"

    # Build the full table text once and append it to this invocation's log.
    table_lines = []
    modes = [("Forward", 1), ("Oracle", 0), ("Roofline", 2)]
    if any(
            row[5] is not None
            for card in cards
            if per_card[card] is not None
            for row in matrix[card][2]):
        modes.append(("PipeWeave", 3))
    for mode_name, metric_idx in modes:
        # per_card stores (count, oracle, forward, roofline, pipeweave), so the
        # selected metric is at metric_idx + 1.
        table_lines.append(f"\n  [single-kernel · {mode_name} MAPE/WMAE %, cross-hardware]")
        h1 = f"  {'bucket':<{label_w}}"
        for c in cards:
            h1 += f"{div} {c:^{blk_inner}}"
        table_lines.append(h1)
        h2 = f"  {'':<{label_w}}"
        for _ in cards:
            h2 += f"{div} {'N':>{count_w}}{'MAPE':>{m_w}}{'WMAE':>{m_w}}"
        table_lines.append(h2)
        # Separator line with ┼ cross at each │ column position
        sep_line = "  " + "─" * label_w
        for _ in cards:
            sep_line += "┼" + "─" * (blk_inner + 1)
        table_lines.append(sep_line)
        _prev_key = ""
        for label, _, *_ in ref_rows:
            key = _kernel_row_key(label)
            _prev_key = key
            row = f"  {label:<{label_w}}"
            for card in cards:
                m = per_card[card]
                if m is None:
                    tag = {SKIPPED: "skip", FAIL: "FAIL"}[matrix[card][0]]
                    row += f"{div} {tag:>{blk_inner}}"
                elif key in m:
                    entry = m[key]
                    count = entry[0]
                    vals = entry[metric_idx + 1]
                    if vals is not None:
                        mape, _, wmae = vals
                        row += f"{div} {count:>{count_w}}{_fmt_metric(mape):>{m_w}}{_fmt_metric(wmae):>{m_w}}"
                    else:
                        row += f"{div} {count:>{count_w}}{'-':>{m_w}}{'-':>{m_w}}"
                else:
                    row += f"{div} {'-':>{count_w}}{'-':>{m_w}}{'-':>{m_w}}"
            table_lines.append(row)

    table_text = "\n".join(table_lines)
    if _eval_log_fh is not None:
        _eval_log_fh.write(table_text + "\n")
        _eval_log_fh.flush()
    print(f"\n  full kernel table recorded in: {_eval_log_path}")

def _conformance_source_inputs(model, hardware_id, tp_size):
    """Declare an implemented source scenario; the case schema pins no runtime version."""
    from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs

    bindings = {
        "nvidia_a800": ("vllm_0_11_0", "cuda_eager_inplace", "vllm_0_11_plain_greedy_bf16"),
        "nvidia_h200": ("vllm_0_30_0", "cuda_eager_inplace", "vllm_0_30_plain_greedy_bf16"),
        "metax_c500": ("vllm_0_30_0", "cuda_eager_inplace", "vllm_0_30_plain_greedy_bf16"),
        "ascend_910b": ("vllm_0_30_0", "vllm_ascend_eager", "vllm_0_30_plain_greedy_bf16"),
    }
    if (hardware_id not in bindings or model.model_type not in ("qwen3", "llama")
            or model.num_experts or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"):
        raise NotImplementedError("conformance virtual source requires an explicit supported dense BF16 Qwen3/Llama scenario")
    if tp_size != 1 and not (hardware_id == "nvidia_a800" and tp_size == 2):
        raise NotImplementedError("unknown conformance token IDs cover dense TP1 or declared0.11 A800 TP2 masked embedding only")
    version, source, sampler = bindings[hardware_id]
    return RuntimeInputs(ExpertRoutingInput.not_applicable(), "paged_varlen", "uniform",
        model_execution_version=version, symbolic_forward_source=source, target_sampling_source=sampler)


def run_serving_part(context, root, cases_override, jobs_override):
    """Exercise the production Simulation path without a framework data plane.

    The authoritative framework integration has its own vLLM no-GPU gate. This
    full-eval layer uses the deterministic conformance backend so every host
    can exercise the common virtual forward and report exact prediction gaps.
    The manifest does not identify a runtime version/provider: selected source
    programs are declared conditional scenarios, never observed bindings.
    """
    try:
        from evaluation.serving.simulation.simulation_runner import _ConformanceBackend, _tensor_collective_configs
        from src.kernel.hardware_spec import HWSpec
        from src.serving.deployment.colocated import build_colocated_plan
        from src.serving.deployment.deployment_config import (
            DeploymentConfig, WorkerPoolConfig,
        )
        from src.serving.kv_cache.kv_capacity import (
            resolve_debug_kv_capacity_plan,
        )
        from src.serving.parallelism.parallel_config import ParallelConfig
        from src.serving.request_gen.request_stream import RequestSpec, RequestStream
        from src.serving.schedule.backend_protocol import BackendConfig
        from src.serving.simulation import Simulation, SimulationConfig
        from src.serving.workload.model_config import load_model_config
        from src.kernel.predictors.protobridge.adapter import development_profile_config_sha256

        cases = _load_serving_cases(context, root)
        if cases_override is not None:
            wanted = set(cases_override)
        else:
            wanted = {case["id"] for case in cases}
        selected = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in selected}
        if missing:
            return FAIL, f"unknown serving case(s): {sorted(missing)}", {}
        hardware = HWSpec(root / context["hardware_config"])
        if hardware.hardware_id != context["hardware_id"]:
            raise ValueError(
                "dataset config loaded the wrong hardware: "
                f"{hardware.hardware_id!r} != {context['hardware_id']!r}")
        rows = {}
        for case in selected:
            model = load_model_config(case["model"])
            if model.num_experts != 0:
                raise ValueError(
                    "full-evaluation MoE cases require an explicit expert "
                    "routing input; the legacy Serving case schema does not "
                    "provide one")
            tp_size = _positive_int(
                case, "tensor_parallel_size", f"serving case {case['id']}")
            ep_size = _positive_int(
                case, "expert_parallel_size", f"serving case {case['id']}")
            if ep_size not in (1, tp_size):
                raise ValueError(
                    "full-evaluation Serving cases use DP=1, so vLLM expert "
                    "parallel size must be 1 or equal tensor parallel size: "
                    f"EP={ep_size}, TP={tp_size}")
            if ep_size > 1:
                raise NotImplementedError(
                    "full-evaluation Serving EP awaits the validated "
                    "rank-local MoE workload slice")
            runtime_inputs = _conformance_source_inputs(model, hardware.hardware_id, tp_size)
            profile_ids = (None, None) if tp_size == 1 else tuple(
                development_profile_config_sha256(hardware.hardware_id, operation)
                for operation in ("all_reduce", "all_gather"))
            deployment = build_colocated_plan(
                DeploymentConfig("colocated", (WorkerPoolConfig(
                    "main", "colocated", hardware.hardware_id,
                    ParallelConfig(
                        tensor_parallel_size=tp_size,
                        enable_expert_parallel=False,
                    ),
                    collectives=_tensor_collective_configs("main", tp_size, hardware.hardware_id,
                        "protobridge", *profile_ids),
                ),)),
                num_layers=model.num_hidden_layers,
                num_experts=model.num_experts,
            )
            requests = RequestStream((RequestSpec(
                0.0,
                "full-eval-request",
                16,
                2,
                0,
            ),))
            conformance_capacity = resolve_debug_kv_capacity_plan(
                "main",
                kv_block_size_tokens=16,
                kv_bytes_per_token=(
                    model.kv_bytes_per_token_per_parallel_rank(tp_size, 1)),
                num_kv_blocks=40000,
            )
            backend = _ConformanceBackend(BackendConfig(
                "main", "colocated", 8192, 16,
                conformance_capacity.num_kv_blocks,
                kv_block_size_tokens=conformance_capacity.kv_block_size_tokens,
                kv_bytes_per_token=conformance_capacity.kv_bytes_per_token,
            scheduler_pipeline_depth=_positive_int(
                case,
                "scheduler_pipeline_depth",
                f"serving case {case['id']}",
            )))
            result = Simulation(
                SimulationConfig(
                    model, deployment, requests, "analytical_launch", "midlayer",
                    runtime_inputs),
                backend,
                {hardware.hardware_id: hardware},
            ).run()
            metric = result.metrics.requests["full-eval-request"]
            rows[case["id"]] = {
                "_simulation": {
                    "backend": "conformance",
                    "workload_path": "virtual_tensor",
                    "prediction_method": "midlayer",
                    "model_execution_version": runtime_inputs.model_execution_version,
                    "symbolic_forward_source": runtime_inputs.symbolic_forward_source,
                    "target_sampling_source": runtime_inputs.target_sampling_source,
                    "source_binding": "declared_conditional_source_scenario_not_observed_runtime_or_provider",
                    "collective_backend": "protobridge" if tp_size > 1 else None,
                    "collective_binding": "registered_development_profiles_not_observed_provider" if tp_size > 1 else None,
                    "token_input_scope": "unknown INT32 virtual token values; no generated token IDs",
                    "capacity_source": conformance_capacity.source_mode,
                    "normal_capacity_accepted": False,
                    "hardware_id": hardware.hardware_id,
                    "steps": len(result.steps),
                    "kernel_calls": sum(
                        step.kernel_plan.logical_call_count
                        for step in result.steps),
                    "ttft_s": metric.ttft_s,
                    "tpot_s": metric.tpot_s,
                    "latency_s": metric.latency_s,
                    "request_throughput_per_s": (
                        result.metrics.request_throughput_per_s),
                },
            }
        if not rows:
            return FAIL, "serving config contains no selected cases", {}
        return PASS, f"{len(rows)} production Simulation case(s)", rows
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"production virtual Simulation failed: {type(error).__name__}: {error}") from error


def run_serving_all_hardware(
    root, suite_name, part, cases_override, jobs_override,
):
    """Run Serving or alignment for every hardware configured for a suite.

    Each context derives one hardware config and one shared case manifest. The
    alignment mode additionally reads the configured historical alignment root.
    Returns an ordered list of
    (card_name, (status, summary, rows)).
    """
    out = []
    for card_name in _serving_hardware_ids(root, suite_name):
        context = _load_serving_context(card_name, suite_name, root)
        print(f"\n  [serving · {card_name}] "
              f"(suite {suite_name}, part={part}) ...", flush=True)
        t0 = time.time()
        if part == "alignment":
            res = _run_serving_alignment_only(
                context, root, cases_override, jobs_override)
        elif part == "serving":
            res = run_serving_part(
                context, root, cases_override, jobs_override)
        else:
            raise ValueError(
                f"all-hardware Serving run does not support part {part!r}")
        dt = time.time() - t0
        mark = {PASS: "✓", SKIPPED: "○", FAIL: "✗"}[res[0]]
        print(f"  [serving · {card_name}] {mark} {res[0]} — {res[1]} "
              f"({dt:.1f}s)", flush=True)
        out.append((card_name, res))
    return out


def _run_serving_alignment_only(context, root, cases_override, jobs_override):
    """Alignment-only serving layer for one hardware (skip predict;
    load existing comparison.json + compute kernel alignment). Mirrors the
    `--part alignment` branch in main() so the all-hardware mode supports the
    fast alignment-only part too. Returns (status, summary, rows)."""
    output_root = root / context["historical_alignment_root"]
    all_cases = [case["id"] for case in _load_serving_cases(context, root)]
    if cases_override is not None:
        want = cases_override
    else:
        want = all_cases
    srows = {}
    case_dirs = {}
    for cid in want:
        cmp_path = output_root / cid / "comparison.json"
        if cmp_path.exists():
            with open(cmp_path) as f:
                srows[cid] = json.load(f)
            case_dirs[cid] = output_root / cid
    if not srows:
        return SKIPPED, (
            "no comparison.json under "
            f"{context['historical_alignment_root']}"), {}
    align_map = _compute_alignment_for_cases(case_dirs, max_workers=jobs_override)
    for cid, al in align_map.items():
        srows[cid]["_kernel_alignment"] = al
    return PASS, f"{len(srows)} case(s) (from cache)", srows


def _print_cross_hardware_serving_summary(card_results):
    """Print the current production-simulation schema for every hardware."""
    by_case = {}
    cards = [name for name, _ in card_results]
    for card_name, (status, _summary, rows) in card_results:
        if status != PASS:
            continue
        for case_id, result in rows.items():
            if case_id not in by_case:
                by_case[case_id] = {}
            by_case[case_id][card_name] = result["_simulation"]

    if not by_case:
        print("\n  [cross-hardware serving · no successful simulation]")
        return

    case_width = max(len(case_id) for case_id in by_case) + 2
    cell_width = 32
    print("\n  [cross-hardware serving · conformance simulation]")
    print(f"  {'case':<{case_width}}" + "".join(
        f"{card:^{cell_width}}" for card in cards))
    print("  " + "-" * (case_width + cell_width * len(cards)))
    for case_id in sorted(by_case, key=_serving_case_sort_key):
        row = f"  {case_id:<{case_width}}"
        for card in cards:
            if card not in by_case[case_id]:
                row += f"{'-':>{cell_width}}"
                continue
            result = by_case[case_id][card]
            text = (
                f"steps={result['steps']} kernels={result['kernel_calls']} "
                f"lat={result['latency_s'] * 1000:.2f}ms")
            row += f"{text:>{cell_width}}"
        print(row)


def _save_serving_table(rows, name):
    """Append full serving tables to this invocation's log, not stdout."""
    import io
    buf = io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = buf
    try:
        _print_serving_table(rows)
        has_alignment = ["_kernel_alignment" in result for result in rows.values()]
        if all(has_alignment):
            _print_alignment_table(rows)
        elif any(has_alignment):
            raise ValueError("serving rows contain a partial alignment result set")
    finally:
        sys.stdout = _old_stdout
    text = buf.getvalue()
    if _eval_log_fh is not None:
        _eval_log_fh.write(text + "\n")
        _eval_log_fh.flush()
    print(f"    full serving table recorded in: {_eval_log_path}")


def _print_serving_summary(rows, name):
    """Compact one-liner per case (stdout only)."""
    print(f"  serving {name:<10} ✓ [PASS]")
    for case_id in sorted(rows, key=_serving_case_sort_key):
        result = rows[case_id]["_simulation"]
        print(
            f"    {case_id:<45} steps={result['steps']:<4d} "
            f"kernels={result['kernel_calls']:<6d} "
            f"latency={result['latency_s'] * 1000:.3f}ms")


def _print_serving_table(rows):
    """Print one row per current conformance Simulation result."""
    case_width = max(len(case_id) for case_id in rows) + 2
    print("\n  [serving · production Simulation conformance]")
    print(
        f"  {'case':<{case_width}} {'hardware':<18} {'steps':>7} "
        f"{'kernels':>9} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'latency(ms)':>12}")
    for case_id in sorted(rows, key=_serving_case_sort_key):
        result = rows[case_id]["_simulation"]
        print(
            f"  {case_id:<{case_width}} {result['hardware_id']:<18} "
            f"{result['steps']:>7d} {result['kernel_calls']:>9d} "
            f"{result['ttft_s'] * 1000:>10.3f} "
            f"{result['tpot_s'] * 1000:>10.3f} "
            f"{result['latency_s'] * 1000:>12.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--hardware-id",
        help="Canonical hardware ID for Serving evaluation. Kernel evaluation "
             "always runs the default production hardware matrix.")
    ap.add_argument(
        "--serving-suite", default="sharegpt",
        help="Serving suite under configs/dataset/serving/ (default: sharegpt).")
    ap.add_argument("--all-hardware", action="store_true",
                    help="Run the Serving layer for every hardware that configures "
                         "the selected suite "
                         "and print a cross-hardware serving table. Intended for a "
                         "host that holds all required traces. The single-kernel "
                         "matrix is unaffected and always cross-hardware.")
    ap.add_argument("--part", choices=["kernel", "serving", "both", "alignment"],
                    default="both", help="Which eval layer(s) to run. 'alignment' = "
                    "skip predict, just load existing comparison.json + compute "
                    "kernel alignment table (fast, ~1s per case).")
    ap.add_argument("--serving-cases", nargs="*", default=None,
                    help="Override serving case ids to run (subset).")
    ap.add_argument("--serving-jobs", type=int, default=None,
                    help="Parallel workers for the serving eval. Default: fully "
                         "parallel (os.cpu_count()). Set a "
                         "smaller number to limit CPU/memory or de-interleave logs.")
    ap.add_argument("--kernel-jobs", type=int, default=None,
                    help="Max cards run concurrently in the single-kernel matrix "
                         "(default: all selected cards at once).")
    ap.add_argument("--kernel-case-jobs", type=int, default=None,
                    help="Independent Kernel case workers per card; omitted uses "
                         "the per-kernel evaluator default.")
    ap.add_argument("--kernel-case-timing-dir", type=Path, default=None,
                    help="Directory for one per-case timing artifact per Kernel "
                        "hardware card.")
    ap.add_argument("--kernel-semantic-output-dir", type=Path, default=None,
                    help="Directory for one complete semantic result artifact per "
                        "Kernel hardware card.")
    ap.add_argument("--with-html", action="store_true",
                    help="Also write the offline interactive HTML kernel report "
                        "and its projections (default: Markdown-only).")
    ap.add_argument("--dump-json", default=None,
                    help="Also serialize the full structured results (single-kernel "
                         "matrix rows + per-card serving rows incl. _roofline / "
                         "_pipeweave / _kernel_alignment) to this JSON path. Used to "
                         "feed the result tables + figures without re-running eval "
                         "(run_full_eval otherwise only prints tables and restores "
                         "comparison.json to midlayer-only).")
    args = ap.parse_args()

    if args.all_hardware and args.hardware_id is not None:
        raise ValueError("--hardware-id and --all-hardware are mutually exclusive")
    if args.part == "kernel" and (
            args.hardware_id is not None or args.all_hardware):
        raise ValueError(
            "--part kernel always evaluates the default production hardware "
            "matrix; "
            "do not pass --hardware-id or --all-hardware")
    needs_serving_hardware = args.part in ("serving", "both", "alignment")
    if needs_serving_hardware and not args.all_hardware and args.hardware_id is None:
        raise ValueError(
            "--hardware-id is required for Serving evaluation unless "
            "--all-hardware is used")

    context = None
    name = "all-hardware" if args.all_hardware else None
    if args.hardware_id is not None:
        context = _load_serving_context(
            args.hardware_id, args.serving_suite, ROOT)
        name = context["hardware_id"]

    timestamp = _wall_clock_stamp()
    serving_hardware_scope = "all_hardware" if args.all_hardware else name
    if args.part == "both":
        log_paths = {
            "kernel": EXPERIMENT_LOG_ROOT / "evaluation/kernel/cross_hardware" / f"{timestamp}.log",
            "serving": EXPERIMENT_LOG_ROOT / "evaluation/logs/serving" / serving_hardware_scope / f"{timestamp}.log",
        }
        active_part = "kernel"
    elif args.part == "kernel":
        log_paths = {
            "kernel": (
                EXPERIMENT_LOG_ROOT / "evaluation/kernel/cross_hardware"
                / f"{timestamp}.log")
        }
        active_part = "kernel"
    elif args.part in ("serving", "alignment"):
        log_paths = {
            args.part: (
                EXPERIMENT_LOG_ROOT / "evaluation/logs" / args.part / serving_hardware_scope
                / f"{timestamp}.log")
        }
        active_part = args.part
    else:
        raise ValueError(f"unknown evaluation part: {args.part!r}")
    for path in log_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    log_files = {part: open(path, "w") for part, path in log_paths.items()}
    global _eval_log_fh, _eval_log_path
    _eval_log_fh = log_files[active_part]
    _eval_log_path = log_paths[active_part]
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, _eval_log_fh)

    print("=" * 70)
    if args.part == "kernel":
        print("  FULL EVAL — single-kernel cross-hardware matrix")
    elif args.part in ("serving", "both", "alignment"):
        hardware_selection = (
            "all-hardware" if context is None else context["hardware_id"])
        print(
            f"  FULL EVAL — serving hardware: {name}   "
            f"(suite: {args.serving_suite}, selection: {hardware_selection})")
    else:
        raise ValueError(f"unknown evaluation part: {args.part!r}")
    for part, path in log_paths.items():
        print(f"  {part} log: {path}")
    print("=" * 70)

    kernel_matrix = None
    kernel_report_directory = None if args.part not in ("kernel", "both") else (
        Path(__file__).resolve().parents[2]
        / "experiment_logs/evaluation/kernel/reports")
    serving_result = None
    serving_all_cards = None
    t_start = time.time()
    kernel_secs = serving_secs = None
    if args.part in ("kernel", "both"):
        if args.part == "both":
            part_label = "PART 1/2"
        elif args.part == "kernel":
            part_label = "PART"
        else:
            raise ValueError(f"unknown kernel evaluation part: {args.part!r}")
        print(
            f"\n{'#'*70}\n#  {part_label} · single-kernel accuracy "
            f"(default production matrix, cross-hw)\n{'#'*70}")
        t0 = time.time()
        kernel_report_directory = kernel_report_directory if args.with_html else None
        if kernel_report_directory is not None:
            kernel_report_directory.mkdir(parents=True, exist_ok=True)
        kernel_matrix = run_kernel_matrix(
            ROOT, args.kernel_jobs,
            case_jobs=args.kernel_case_jobs,
            case_timing_dir=args.kernel_case_timing_dir,
            semantic_output_dir=args.kernel_semantic_output_dir, report_output_dir=kernel_report_directory)
        kernel_secs = time.time() - t0
        if args.with_html:
            from evaluation.reporting.kernel_overview import write_kernel_matrix_overview
            kernel_overview_payload = {"kernel_matrix": {card: {"status": status, "summary": summary,
                "rows": [list(row) for row in rows], "forward_coverage": coverage}
                for card, (status, summary, rows, coverage) in kernel_matrix.items()},
                "kernel_reports": {card: {"status": "available" if (kernel_report_directory / (card + ".html")).is_file() else "unavailable",
                    "path": str(kernel_report_directory / (card + ".html"))} for card in kernel_matrix}}
            kernel_overview = write_kernel_matrix_overview(kernel_overview_payload, vars(args), kernel_report_directory / "matrix.html")
            print("Kernel matrix HTML report: " + kernel_overview["path"])
        # Print the cross-hardware tables IMMEDIATELY, before the (slow) Serving
        # part runs — the single-kernel results should be visible right away.
        _print_kernel_matrix(kernel_matrix)
        skipped = [
            c for c, (s, _, _, _) in kernel_matrix.items() if s == SKIPPED]
        if skipped:
            print(f"  (SKIPPED cards — no dataset on this host: {', '.join(skipped)})")
        print(f"  [timing] single-kernel part: {kernel_secs:.1f}s")
        sys.stdout.flush()
    if args.part in ("serving", "both"):
        if args.part == "both":
            _eval_log_fh = log_files["serving"]
            _eval_log_path = log_paths["serving"]
            sys.stdout = _Tee(orig_stdout, _eval_log_fh)
        if args.all_hardware:
            print(f"\n{'#'*70}\n#  PART 2/2 · serving end-to-end prediction (ALL hardware: A800/C500/910B)\n{'#'*70}")
        else:
            print(f"\n{'#'*70}\n#  PART 2/2 · serving end-to-end prediction (hardware: {name})\n{'#'*70}")
        t0 = time.time()
        if args.all_hardware:
            serving_all_cards = run_serving_all_hardware(
                ROOT, args.serving_suite, "serving",
                args.serving_cases, args.serving_jobs)
            # Flatten the all-hardware result into the single-card summary
            # slot so the existing per-case table prints once per card.
        else:
            serving_result = run_serving_part(
                context, ROOT, args.serving_cases, args.serving_jobs)
        serving_secs = time.time() - t0
        print(f"  [timing] serving part: {serving_secs:.1f}s")
    if args.part == "alignment":
        # Skip predict; just load existing comparison.json + compute alignment.
        print(f"\n{'#'*70}\n#  PART · kernel alignment only (skip predict)\n{'#'*70}")
        t0 = time.time()
        if args.all_hardware:
            serving_all_cards = run_serving_all_hardware(
                ROOT, args.serving_suite, "alignment",
                args.serving_cases, args.serving_jobs)
        else:
            output_root = ROOT / context["historical_alignment_root"]
            all_cases = [
                case["id"] for case in _load_serving_cases(context, ROOT)]
            if args.serving_cases is not None:
                want = args.serving_cases
            else:
                want = all_cases
            srows = {}
            case_dirs = {}
            for cid in want:
                cmp_path = output_root / cid / "comparison.json"
                if not cmp_path.exists():
                    raise FileNotFoundError(
                        f"alignment requires comparison data: {cmp_path}")
                with open(cmp_path) as f:
                    srows[cid] = json.load(f)
                case_dirs[cid] = output_root / cid
            align_map = _compute_alignment_for_cases(
                case_dirs, max_workers=args.serving_jobs)
            if set(align_map) != set(srows):
                raise RuntimeError("alignment did not return every requested case")
            for cid, alignment in align_map.items():
                srows[cid]["_kernel_alignment"] = alignment
            serving_result = (
                PASS, f"{len(srows)} case(s) (from cache)", srows)
        serving_secs = time.time() - t0
        print(f"  [timing] alignment-only part: {serving_secs:.1f}s")
    total_secs = time.time() - t_start

    # Summary
    if args.part == "both":
        # Detailed tables remain in their owning part's log.  The compact
        # invocation summary is useful in both records and links the pair by
        # their shared timestamp.
        sys.stdout = _Tee(orig_stdout, log_files["kernel"], log_files["serving"])
        _eval_log_fh = log_files["serving"]
        _eval_log_path = log_paths["serving"]
    print("\n" + "=" * 70)
    if args.part == "kernel":
        print("  FULL EVAL SUMMARY — single-kernel cross-hardware matrix")
    elif args.part in ("serving", "both", "alignment"):
        print(f"  FULL EVAL SUMMARY — serving hardware: {name}")
    else:
        raise ValueError(f"unknown evaluation part: {args.part!r}")
    print("=" * 70)
    any_fail = False
    verified_any = False

    if kernel_matrix is not None:
        # Compact per-card summary (the full table is in this invocation's log).
        for card, (status, summary, _, _) in kernel_matrix.items():
            mark = {PASS: "✓", SKIPPED: "○", FAIL: "✗"}[status]
            print(f"  kernel {card:<10} {mark} [{status}]  {summary}")
            if status == FAIL:
                any_fail = True
            elif status == PASS:
                verified_any = True
            elif status == SKIPPED:
                pass
            else:
                raise ValueError(f"unknown kernel evaluation status: {status!r}")
        skipped = [
            c for c, (s, _, _, _) in kernel_matrix.items() if s == SKIPPED]
        if skipped:
            print(f"  (SKIPPED cards — no dataset on this host: {', '.join(skipped)})")

    if serving_result is not None:
        status, summary, srows = serving_result
        mark = {PASS: "✓", SKIPPED: "○", FAIL: "✗"}[status]
        # Full serving table saved to file; compact summary to stdout
        if status == PASS and srows:
            _save_serving_table(srows, name)
            _print_serving_summary(srows, name)
        else:
            print(f"  serving {name:<10} {mark} [{status}]  {summary}")
        if status == FAIL:
            any_fail = True
        elif status == PASS:
            verified_any = True
        elif status == SKIPPED:
            pass
        else:
            raise ValueError(f"unknown serving evaluation status: {status!r}")
    elif serving_all_cards is not None:
        for card_name, (status, summary, srows) in serving_all_cards:
            mark = {PASS: "✓", SKIPPED: "○", FAIL: "✗"}[status]
            if status == PASS and srows:
                _save_serving_table(srows, card_name)
                _print_serving_summary(srows, card_name)
            else:
                print(f"  serving {card_name:<10} {mark} [{status}]  {summary}")
            if status == FAIL:
                any_fail = True
            elif status == PASS:
                verified_any = True
            elif status == SKIPPED:
                pass
            else:
                raise ValueError(
                    f"unknown serving evaluation status: {status!r}")
        _print_cross_hardware_serving_summary(serving_all_cards)

    print("\n" + "=" * 70)
    tparts = []
    if kernel_secs is not None:
        tparts.append(f"single-kernel {kernel_secs:.1f}s")
    if serving_secs is not None:
        tparts.append(f"serving {serving_secs:.1f}s")
    print(f"  TIMING: {'  |  '.join(tparts)}  |  total {total_secs:.1f}s")
    if any_fail:
        result = "  RESULT: FAIL — fix before committing."
        exit_code = 1
    elif not verified_any:
        result = "  RESULT: all SKIPPED (no data on this host) — nothing verified."
        exit_code = 0
    else:
        result = "  RESULT: PASS (SKIPPED parts have no data on this host)."
        exit_code = 0
    print(result)
    for part, path in log_paths.items():
        print(f"  {part} log saved: {path}")

    # Optional: serialize the structured results for the tables + figures. Both
    # the single-kernel matrix and the per-card serving rows carry all three
    # modes (Forward top-level, _roofline, _pipeweave) plus _kernel_alignment,
    # so one run feeds every downstream artifact without re-evaluating.
    if args.dump_json:
        dump = {
            "serving_hardware": name,
            "logs": {part: str(path) for part, path in log_paths.items()},
        }
        if kernel_matrix is not None:
            # rows are (label, fwd_metrics, roof_metrics, pw_metrics); metrics are
            # (mape, med, wmae) tuples or None — all JSON-serializable as lists.
            dump["kernel_matrix"] = {
                card: {"status": status, "summary": summary,
                       "rows": [list(r) for r in rows],
                       "forward_coverage": coverage}
                for card, (status, summary, rows, coverage)
                in kernel_matrix.items()}
        if kernel_matrix is not None:
            dump["kernel_reports"] = {card: {
                "status": "available" if (kernel_report_directory / (card + ".html")).is_file() else "unavailable",
                "path": str(kernel_report_directory / (card + ".html"))}
                for card in kernel_matrix} if kernel_report_directory is not None else {}
        if serving_all_cards is not None:
            dump["serving"] = {
                card_name: {"status": status, "summary": summary, "rows": srows}
                for card_name, (status, summary, srows) in serving_all_cards}
        elif serving_result is not None:
            status, summary, srows = serving_result
            dump["serving"] = {name: {"status": status, "summary": summary,
                                      "rows": srows}}
        dump_path = Path(args.dump_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump(dump, f)
        print(f"  structured results dumped: {dump_path}")

    # Restore stdout and close the log file before exiting.
    sys.stdout.flush()
    sys.stdout = orig_stdout
    for log_fh in log_files.values():
        log_fh.close()
    return exit_code
