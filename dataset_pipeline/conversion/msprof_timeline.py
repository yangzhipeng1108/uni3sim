"""Parse msprof op_summary timeline → nsys-compatible ground-truth schema.

Produces the SAME structure capture_nsys.py emits for A800/C500
({span_ns, gpu_ns, n_kernels, kernels{}, cat{}}), so the existing
build_eval_dataset comparison works unchanged. One rank = one PROF dir
(device_N), matching A800's per-deviceId partition. This is the 910B
ground-truth adapter (msprof timeline ⇄ nsys timeline).
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def _num(v):
    if v is None or (isinstance(v, str)
                     and v.strip() in ("", "NA", "N/A")):
        return None
    if isinstance(v, bool):
        raise TypeError("numeric msprof value cannot be boolean")
    if not isinstance(v, (int, float, str)):
        raise TypeError(f"invalid numeric msprof type: {type(v).__name__}")
    try:
        result = float(v.strip() if isinstance(v, str) else v)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric msprof value: {v!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"numeric msprof value must be finite: {v!r}")
    return result


def _required_num(row, field):
    value = _num(row[field])
    if value is None:
        raise ValueError(f"msprof row is missing required field {field!r}")
    return value


def _required_positive_integer(value, field):
    number = _num(value)
    if number is None or not number.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    result = int(number)
    if result <= 0:
        raise ValueError(f"{field} must be positive, got {result}")
    return result


def classify(op_type, name):
    if not isinstance(op_type, str) or not op_type:
        raise TypeError("msprof op_type must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise TypeError("msprof op name must be a non-empty string")
    lowered_type = op_type.lower()
    lowered_name = name.lower()
    if "matmul" in lowered_name or "groupedmatmul" in lowered_name:
        return "gemm"
    if any(token in lowered_name for token in ("attention", "fa", "flash")):
        return "attn"
    if "cube" in lowered_type:
        return "gemm"
    if "moe" in lowered_name or "routing" in lowered_name:
        return "moe"
    if any(token in lowered_name for token in ("swiglu", "activation", "silu")):
        return "act"
    if "norm" in lowered_name or "rmsnorm" in lowered_name:
        return "norm"
    if any(token in lowered_name for token in ("cast", "transpose", "transdata")):
        return "layout"
    return "other"


def parse_rank(prof_dir, win_start_us=None, win_end_us=None):
    """Parse one rank's op_summary. If a window [win_start_us, win_end_us]
    (Unix-epoch µs, same clock as msprof Task Start Time) is given, keep only
    kernels whose start falls inside it — the inference window (load/warmup out).
    """
    if (win_start_us is None) != (win_end_us is None):
        raise ValueError("msprof timeline window requires both start and end")
    if (win_start_us is not None and win_end_us is not None
            and win_end_us < win_start_us):
        raise ValueError("msprof timeline window end precedes its start")
    files = sorted(glob.glob(os.path.join(
        prof_dir, "mindstudio_profiler_output", "op_summary_*.csv")))
    if not files:
        raise FileNotFoundError(f"no op_summary CSV found under {prof_dir}")
    if len(files) != 1:
        raise ValueError(f"ambiguous op_summary CSVs under {prof_dir}: {files}")
    with open(files[0]) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"op_summary CSV is empty: {files[0]}")
    starts, ends, gpu_ns = [], [], 0.0
    kernels = defaultdict(
        lambda: {"count": 0, "ns": 0.0,
                 "block_dim": 0, "mix_block_dim": 0})
    cats = defaultdict(lambda: {"count": 0, "ns": 0.0})
    n_used = 0
    for r in rows:
        # Identify a real DEVICE op via non-zero aicore/aiv execution time
        # (filters out host-only/aclnn rows), but use TASK DURATION as the
        # kernel's duration — that is the device kernel wall time, the exact
        # analog of an nsys kernel dur_ns on A800/C500. (aicore_time is
        # compute-only and excludes the kernel's own MTE2/scalar time, so it is
        # NOT comparable to a Task-Duration-calibrated kernel model; the single-
        # kernel 910B dataset also uses Task Duration as duration_ns.) Within the
        # inference window Task Duration is rank-consistent (rank0≈rank1); the
        # 15× cross-rank divergence only appears on unwindowed flat sums.
        aic = _num(r["aicore_time(us)"])
        aiv = _num(r["aiv_time(us)"])
        if aic is not None and aic < 0:
            raise ValueError("msprof aicore_time(us) cannot be negative")
        if aiv is not None and aiv < 0:
            raise ValueError("msprof aiv_time(us) cannot be negative")
        if aic is not None and aic > 0:
            dev_us = aic
        elif aiv is not None and aiv > 0:
            dev_us = aiv
        else:
            dev_us = 0.0
        st_us = _required_num(r, "Task Start Time(us)")
        task_dur_us = _required_num(r, "Task Duration(us)")
        if st_us < 0:
            raise ValueError("msprof Task Start Time(us) cannot be negative")
        if task_dur_us <= 0:
            raise ValueError("msprof Task Duration(us) must be positive")
        # aclnnMatmul (cube op) often has aicore_time=0 in msprof (76% of decode
        # GEMM tasks), but Task Duration is always present. Falling through on
        # dev_us<=0 drops those decode GEMMs from the GT → GT only has prefill
        # (high per-call) → G1 GEMM ratio looks like 0.82 under-pred when the GT
        # is actually incomplete. For cube ops with a valid Task Duration, use
        # Task Duration as dev_us (it IS the device wall time). Zero-fit: Task
        # Duration is the msprof device-kernel wall time (same field used for
        # dur_ns). aclnnMatmul's OP Type is "MatMulV2"/"MatMulV3" (not "cube"),
        # so match on the op name containing "MatMul" too.
        raw_op_type = r["OP Type"]
        raw_op_name = r["Op Name"]
        if not isinstance(raw_op_type, str) or not raw_op_type.strip():
            raise TypeError("msprof device row OP Type must be a string")
        if not isinstance(raw_op_name, str) or not raw_op_name.strip():
            raise TypeError("msprof device row Op Name must be a string")
        raw_op_type = raw_op_type.strip()
        raw_op_name = raw_op_name.strip()
        op_type = raw_op_type.lower()
        op_name = raw_op_name.lower()
        is_cube_gemm = ("matmul" in op_type or "matmul" in op_name
                        or op_type in ("cube", "mix"))
        if dev_us <= 0 and task_dur_us > 0 and is_cube_gemm:
            dev_us = task_dur_us
        if dev_us <= 0:
            continue
        if win_start_us is not None and (st_us < win_start_us or st_us > win_end_us):
            continue
        n_used += 1
        dur_ns = task_dur_us * 1000.0
        starts.append(st_us * 1000.0)
        ends.append((st_us + task_dur_us) * 1000.0)
        gpu_ns += dur_ns
        nm = raw_op_name
        cat = classify(raw_op_type, nm)
        k = kernels[nm]
        k["count"] += 1
        k["ns"] += dur_ns
        # Block Dim / Mix Block Dim are constant per Op Name (the kernel's launch
        # config: AIC cores for cube=24, AIV=48, MIX_AIC FA=24+48). Captured so a
        # tile/grid-mismatch checker can compare the synthetic emit's grid to the
        # real msprof launch geometry. First non-zero wins (msprof fills it on
        # every invocation of a given op).
        bd = _required_positive_integer(r["Block Dim"], "Block Dim")
        mb = _num(r["Mix Block Dim"])
        if not k["block_dim"]:
            k["block_dim"] = bd
        if mb is not None:
            if not mb.is_integer() or mb < 0:
                raise ValueError(
                    f"msprof device op {nm!r} has invalid Mix Block Dim")
            if mb > 0 and not k["mix_block_dim"]:
                k["mix_block_dim"] = int(mb)
        c = cats[cat]
        c["count"] += 1
        c["ns"] += dur_ns
    if n_used == 0:
        raise ValueError(
            f"msprof profile {prof_dir} contains no device operations in scope")
    span_ns = max(ends) - min(starts)
    if span_ns < 0 or gpu_ns <= 0:
        raise ValueError("msprof timeline produced invalid duration totals")
    # Emit kernels as a LIST of per-name aggregates {name, dur_ns, count,
    # block_dim, mix_block_dim} — the schema run_comparison._nsys_per_rank_compute_ms
    # iterates (it sums k["dur_ns"] over non-comm k["name"]). dur_ns is the
    # per-name total device time, so summing the list reproduces the exact compute
    # total. block_dim/mix_block_dim are the real launch geometry (for the
    # mismatch checker); older consumers ignore the extra fields. (Internally we
    # aggregate by name into a dict, then flatten here.)
    kernels_list = [{"name": nm, "dur_ns": v["ns"], "count": v["count"],
                     "block_dim": v["block_dim"], "mix_block_dim": v["mix_block_dim"]}
                    for nm, v in kernels.items()]
    return {"span_ns": span_ns, "gpu_ns": gpu_ns, "n_kernels": n_used,
            "cat": cats, "kernels": kernels_list}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="msprof output dir containing PROF_* subdirs")
    ap.add_argument("--wall-ms", type=float, default=0.0,
                    help="wall time (ms) of the profiled generate window")
    ap.add_argument("--meta", default=None,
                    help="driver meta.json — reads gen_start/end_epoch_us for "
                         "inference-window filtering + wall (generate_s)")
    ap.add_argument("--emit", default=None,
                    help="write nsys_kernels.json (A800-schema GT) to this path")
    a = ap.parse_args(argv)

    if a.wall_ms < 0:
        raise ValueError("--wall-ms must be non-negative")

    win_start = win_end = None
    wall_ms = a.wall_ms
    if a.meta and not os.path.exists(a.meta):
        raise FileNotFoundError(f"metadata file does not exist: {a.meta}")
    if a.meta:
        with open(a.meta) as stream:
            m = json.load(stream)
        if not isinstance(m, dict):
            raise TypeError("metadata root must be a mapping")
        win_start = m["gen_start_epoch_us"]
        win_end = m["gen_end_epoch_us"]
        if not wall_ms and m["generate_s"]:
            wall_ms = m["generate_s"] * 1000.0
        if win_start:
            print(f"[window] inference-only: [{win_start:.0f}, {win_end:.0f}] us "
                  f"({(win_end-win_start)/1e6:.1f}s)")

    profs = sorted(glob.glob(os.path.join(a.base, "PROF_*")))
    if not profs:
        raise FileNotFoundError(f"no PROF_* directories found under {a.base}")
    per_rank = {}
    for i, p in enumerate(profs):
        r = parse_rank(p, win_start, win_end)
        per_rank[str(i)] = r
        print(f"rank{i} ({os.path.basename(p)[:24]}): "
              f"n_kernels={r['n_kernels']}, gpu_ns={r['gpu_ns']/1e6:.1f}ms, "
              f"span={r['span_ns']/1e6:.1f}ms")
        print(f"    cats: " + ", ".join(f"{c}={v['ns']/1e6:.0f}ms" for c, v in
              sorted(r['cat'].items(), key=lambda x: -x[1]['ns'])))

    if a.emit:
        # Bottleneck rank = max gpu_ns (same rule as capture_nsys/build_eval_dataset).
        bn = max(per_rank, key=lambda k: per_rank[k]["gpu_ns"])
        b = per_rank[bn]
        nsys_data = {
            "wall_ms": wall_ms,
            "span_ns": b["span_ns"], "gpu_ns": b["gpu_ns"],
            "n_kernels": b["n_kernels"], "num_devices": len(per_rank),
            "kernels": b["kernels"], "per_rank": per_rank, "bottleneck_rank": bn,
        }
        with open(a.emit, "w") as stream:
            json.dump(nsys_data, stream, indent=2)
        print(f"[emit] {a.emit}  (bottleneck rank{bn}, gpu_ns={b['gpu_ns']/1e6:.1f}ms)")
    return 0
