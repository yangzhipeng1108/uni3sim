"""Distill msprof op_summary into small JSONs (disk-cheap; delete bulky raw).

Re-capture should keep ONLY what the checker + future predictor work need, NOT
the 4-55GB raw PROF or even the 256MB-2GB sqlite. This distills the per-device
op_summary (CSV, or the ai_core_op_summary.db sqlite ge_summary+ai_core_metrics
table) into three small JSONs (~KB-MB total):

  gemm_grid_lut.json  — {(M,N,K): block_dim} for dense aclnnMatmul (MatMulV2/V3).
                        The (M,N,K)-keyed GEMM grid LUT the checker/emit consume.
                        CANN's Block Dim is deterministic per (M,N,K).
  op_grids.json       — {op_name_lower: {block_dim, mix_block_dim}} per Op Name
                        (the launch geometry for FA/Norm/MoE/Act/...).
  op_profile.json     — per-(op_name, input_shapes, block_dim) aggregated
                        {count, mean_dur_ns, mean_aic_mte2, mean_aic_cube,
                         mean_aic_scalar, mean_aic_fixpipe, mean_aiv_vec,
                         mean_aiv_scalar, mean_aiv_mte2, mean_aiv_mte3} — the
                        per-shape pipe times future predictor work (P87-class
                        per-pipe models) may want, so re-capture isn't needed
                        again for new analyses.

After distillation the raw PROF (host/, device_*/data/) + the op_summary CSV
(mindstudio_profiler_output/) + the device sqlite can all be deleted — the
JSONs carry the needed truth. nsys_kernels.json (name+dur+count GT) +
plugin_trace are kept by the capture script as before.

Sources (in order): op_summary*.csv > ai_core_op_summary.db sqlite.
"""
import argparse
import csv
import glob
import json
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path


def _parse_mnk(input_shapes):
    """Parse ``M,K;N,K`` (A=[M,K], B=[N,K]) into ``(M,N,K)``."""
    if not isinstance(input_shapes, str) or not input_shapes.strip():
        raise TypeError("MatMul profiler input_shapes must be a non-empty string")
    s = input_shapes.strip().strip('"').strip()
    parts = [p for p in s.split(";") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"invalid MatMul input_shapes: {input_shapes!r}")
    try:
        a_fields = parts[0].split(",")
        b_fields = parts[1].split(",")
        if len(a_fields) != 2 or len(b_fields) != 2:
            raise ValueError("MatMul operands must each contain two dimensions")
        a = [_required_integer(x, "MatMul A shape") for x in a_fields]
        b = [_required_integer(x, "MatMul B shape") for x in b_fields]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid MatMul input_shapes: {input_shapes!r}") from error
    return (a[0], b[0], a[1])  # (M, N, K)




def _num(v):
    if v is None or (isinstance(v, str)
                     and v.strip() in ("", "NA", "N/A")):
        return None
    if isinstance(v, bool):
        raise TypeError("numeric profiler value cannot be boolean")
    if not isinstance(v, (int, float, str)):
        raise TypeError(
            f"invalid numeric profiler type: {type(v).__name__}")
    try:
        result = float(v)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric profiler value: {v!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"numeric profiler value must be finite: {v!r}")
    return result


def _required_integer(value, field):
    number = _num(value)
    if number is None or not number.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    result = int(number)
    if result <= 0:
        raise ValueError(f"{field} must be positive, got {result}")
    return result


def _required_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"msprof field {field!r} must be a non-empty string")
    return value.strip()


def _optional_text(value, field):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"msprof field {field!r} must be a string or null")
    return value


def _iter_rows(case_dir):
    """Yield per-invocation op rows from op_summary CSV, else sqlite ge_summary."""
    csvs = glob.glob(os.path.join(str(case_dir), "msprof", "PROF_*",
                                  "mindstudio_profiler_output", "op_summary*.csv"))
    if csvs:
        for f in csvs:
            try:
                for r in csv.DictReader(open(f)):
                    yield {
                        "op_name": _required_text(r["Op Name"], "Op Name"),
                        "op_type": _required_text(r["OP Type"], "OP Type"),
                        "block_dim": r["Block Dim"],
                        "mix_block_dim": r["Mix Block Dim"],
                        "input_shapes": _optional_text(
                            r["Input Shapes"], "Input Shapes"),
                        "dur": r["Task Duration(us)"],
                        "aic_mte2": r["aic_mte2_time(us)"],
                        "aic_cube": r["aic_mac_time(us)"],
                        "aic_scalar": r["aic_scalar_time(us)"],
                        "aic_fixpipe": r["aic_fixpipe_time(us)"],
                        "aiv_vec": r["aiv_vec_time(us)"],
                        "aiv_scalar": r["aiv_scalar_time(us)"],
                        "aiv_mte2": r["aiv_mte2_time(us)"],
                        "aiv_mte3": r["aiv_mte3_time(us)"],
                    }
            except (csv.Error, OSError, KeyError, ValueError) as error:
                raise RuntimeError(
                    f"failed to parse msprof CSV {f}") from error
        return
    # sqlite fallback (CSV absent): ge_summary has the metadata; ai_core_metrics
    # has the per-pipe times. ge_summary has NO subtask_id column (one row per
    # (task_id, stream_id, batch_id)), so join on those 3 cols via ON (not USING,
    # which would require subtask_id on both sides and throw OperationalError —
    # silently swallowed per-db before this fix, producing empty JSONs).
    dbs = glob.glob(os.path.join(str(case_dir), "msprof", "PROF_*",
                                 "device_*", "sqlite", "ai_core_op_summary.db"))
    for db in dbs:
        try:
            with sqlite3.connect(db) as connection:
                q = ("SELECT g.op_name, g.op_type, g.block_dim, g.mix_block_dim, "
                     "g.input_shapes, t.duration_time, "
                     "m.aic_mte2_time, m.aic_mac_time, m.aic_scalar_time, "
                     "m.aic_fixpipe_time, m.aiv_vec_time, m.aiv_scalar_time, "
                     "m.aiv_mte2_time, m.aiv_mte3_time "
                     "FROM ge_summary g "
                     "LEFT JOIN ai_core_metrics m "
                     "ON m.task_id=g.task_id AND m.stream_id=g.stream_id "
                     "AND m.batch_id=g.batch_id "
                     "LEFT JOIN task_time t "
                     "ON t.task_id=g.task_id AND t.stream_id=g.stream_id "
                     "AND t.batch_id=g.batch_id")
                for row in connection.execute(q):
                    (op_name, op_type, bd, mb, ish, dur, amte2, acube, ascal,
                     afix, avec, ascal_v, amte2_v, amte3) = row
                    yield {
                        "op_name": _required_text(op_name, "op_name"),
                        "op_type": _required_text(op_type, "op_type"),
                        "block_dim": bd,
                        "mix_block_dim": mb,
                        "input_shapes": _optional_text(ish, "input_shapes"),
                        "dur": dur, "aic_mte2": amte2, "aic_cube": acube,
                        "aic_scalar": ascal, "aic_fixpipe": afix,
                        "aiv_vec": avec, "aiv_scalar": ascal_v,
                        "aiv_mte2": amte2_v, "aiv_mte3": amte3,
                    }
        except sqlite3.Error as error:
            raise RuntimeError(
                f"failed to parse msprof sqlite database {db}") from error


def distill(case_dir, write=True):
    """Build + (optionally write) the 3 distilled JSONs for a case dir.

    Returns (gemm_grid_lut, op_grids, op_profile). Writes to
    <case_dir>/gemm_grid_lut.json, op_grids.json, op_profile.json when write=True.
    """
    gemm_lut = {}            # (M,N,K) -> Counter(bd)  -> dominant bd
    from collections import Counter
    gemm_acc = defaultdict(Counter)
    op_grids = {}            # op_name_lower -> {block_dim, mix_block_dim}
    # op_profile: (op_name, input_shapes, block_dim) -> {count, sums...}
    prof = defaultdict(lambda: {
        "count": 0,
        "value_count": defaultdict(int),
        "dur": 0.0, "aic_mte2": 0.0, "aic_cube": 0.0,
        "aic_scalar": 0.0, "aic_fixpipe": 0.0,
        "aiv_vec": 0.0, "aiv_scalar": 0.0,
        "aiv_mte2": 0.0, "aiv_mte3": 0.0,
    })
    pipe_keys = ["dur", "aic_mte2", "aic_cube", "aic_scalar", "aic_fixpipe",
                 "aiv_vec", "aiv_scalar", "aiv_mte2", "aiv_mte3"]

    row_count = 0
    for r in _iter_rows(case_dir):
        row_count += 1
        nm = r["op_name"]
        bd_value = _num(r["block_dim"])
        mb_value = _num(r["mix_block_dim"])
        if bd_value is None or bd_value <= 0:
            raise ValueError(
                f"msprof op {nm!r} requires a positive block_dim")
        if not bd_value.is_integer():
            raise ValueError(
                f"msprof op {nm!r} block_dim must be an integer")
        bd = int(bd_value)
        # msprof omits Mix Block Dim for launches without an AIV side. Zero is
        # the provider schema's explicit "no mixed blocks" value, not a guessed
        # launch size.
        if mb_value is not None and not mb_value.is_integer():
            raise ValueError(
                f"msprof op {nm!r} mix_block_dim must be an integer")
        mb = 0 if mb_value is None else int(mb_value)
        if mb < 0:
            raise ValueError(f"msprof op {nm!r} has negative mix_block_dim")
        grid = {"block_dim": bd, "mix_block_dim": mb}
        key = nm.lower()
        if key in op_grids and op_grids[key] != grid:
            raise ValueError(
                f"inconsistent launch grid for {nm!r}: "
                f"{op_grids[key]!r} vs {grid!r}")
        op_grids[key] = grid
        # GEMM grid LUT (dense aclnnMatmul; not Batch/Grouped).
        nml = nm.lower()
        if "matmul" in nml and "batch" not in nml and "grouped" not in nml:
            mnk = _parse_mnk(r["input_shapes"])
            gemm_acc[mnk][bd] += 1
        # per-shape profile (op_name + input_shapes + block_dim)
        key = (nm, r["input_shapes"], bd)
        s = prof[key]
        s["count"] += 1
        for p in pipe_keys:
            value = _num(r[p])
            if value is not None:
                s[p] += value
                s["value_count"][p] += 1

    if row_count == 0:
        raise FileNotFoundError(
            f"no msprof op-summary rows found under {case_dir}")
    gemm_lut = {
        mnk: cnt.most_common(1)[0][0] for mnk, cnt in gemm_acc.items()
    }
    # finalize op_profile means (us -> ns for dur; keep pipe in us as msprof gives)
    op_profile = {}
    for (nm, ish, bd), s in prof.items():
        n = s["count"]
        if s["value_count"]["dur"] != n:
            raise ValueError(
                f"missing Task Duration for {nm!r}, shape={ish!r}, block={bd}")
        op_profile[f"{nm}|{ish}|{bd}"] = {
            "op_name": nm, "input_shapes": ish, "block_dim": bd,
            "count": n,
            **{
                p: (
                    None if s["value_count"][p] == 0
                    else round(s[p] / s["value_count"][p], 3)
                )
                for p in pipe_keys
            },
        }

    if write:
        gemm_path = os.path.join(str(case_dir), "gemm_grid_lut.json")
        with open(gemm_path, "w") as stream:
            json.dump({
                ",".join(map(str, key)): value
                for key, value in gemm_lut.items()
            }, stream, indent=1)
        grids_path = os.path.join(str(case_dir), "op_grids.json")
        with open(grids_path, "w") as stream:
            json.dump(op_grids, stream, indent=1)
        profile_path = os.path.join(str(case_dir), "op_profile.json")
        with open(profile_path, "w") as stream:
            json.dump(op_profile, stream, indent=1)
    return gemm_lut, op_grids, op_profile


def _delete_raw_profiler_data(case_dir):
    """Delete raw msprof payload only after an explicit CLI request."""
    import shutil

    for profile_dir in glob.glob(os.path.join(
            str(case_dir), "msprof", "PROF_*")):
        for subdirectory in ("host", "mindstudio_profiler_output"):
            path = os.path.join(profile_dir, subdirectory)
            if os.path.isdir(path):
                shutil.rmtree(path)
        for device_dir in glob.glob(os.path.join(profile_dir, "device_*")):
            for subdirectory in ("data", "sqlite"):
                path = os.path.join(device_dir, subdirectory)
                if os.path.isdir(path):
                    shutil.rmtree(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Distill an msprof operator summary.")
    parser.add_argument("case_dir", help="Serving case directory")
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete raw msprof data only after distilled files are written.",
    )
    args = parser.parse_args(argv)

    case_dir = Path(args.case_dir)
    gemm_lut, op_grids, op_profile = distill(case_dir)
    print(
        f"[distill] {case_dir}: gemm_lut={len(gemm_lut)} "
        f"op_grids={len(op_grids)} op_profile={len(op_profile)}")
    serialized_gemm_lut = {
        ",".join(map(str, key)): value
        for key, value in gemm_lut.items()
    }
    print(
        f"  sizes: gemm_grid_lut={len(json.dumps(serialized_gemm_lut))}B "
        f"op_grids={len(json.dumps(op_grids))}B "
        f"op_profile={len(json.dumps(op_profile))}B")
    if args.delete_raw:
        _delete_raw_profiler_data(case_dir)
        print("  [distill] deleted raw host/ + device_*/data + device_*/sqlite + op_summary CSV")
    return 0
