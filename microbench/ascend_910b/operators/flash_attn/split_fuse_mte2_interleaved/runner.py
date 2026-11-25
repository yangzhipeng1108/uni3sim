#!/usr/bin/env python3
"""Run the registered independent SplitFuse KV ingress cells on one idle NPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


GROUP_COUNTS = (8, 16, 24)
DESCRIPTOR_COUNTS = (1, 2, 4)
TAIL_DESCRIPTOR_COUNTS = (1, 2, 3, 4)
ISSUE_MODES = (("drained", 0), ("two_slot_overlap", 1))
PROCESS_COUNT = 3
STABILITY_LIMIT_PCT = 5.0
EXPECTED_HEADER = (
    "groups,count,issue_mode,rows,columns,stride_elements,"
    "cold_min_cycles,cold_median_cycles,cold_max_cycles,"
    "warm_min_cycles,warm_median_cycles,warm_max_cycles,"
    "tick_ns,cold_program_ns,warm_program_ns,validation")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_result(stdout: str, groups: int, count: int, mode: int) -> dict:
    lines = stdout.splitlines()
    if len(lines) != 2 or lines[0] != EXPECTED_HEADER:
        raise RuntimeError(f"unexpected probe output: {stdout!r}")
    fields = lines[1].split(",")
    if len(fields) != 16:
        raise RuntimeError(f"unexpected probe field count: {fields!r}")
    result = {
        "active_aic_groups": int(fields[0]),
        "descriptor_count": int(fields[1]),
        "issue_mode": int(fields[2]),
        "rows": int(fields[3]),
        "columns": int(fields[4]),
        "source_row_stride_elements": int(fields[5]),
        "cold_min_cycles": int(fields[6]),
        "cold_median_cycles": float(fields[7]),
        "cold_max_cycles": int(fields[8]),
        "warm_min_cycles": int(fields[9]),
        "warm_median_cycles": float(fields[10]),
        "warm_max_cycles": int(fields[11]),
        "tick_ns": float(fields[12]),
        "cold_program_ns": float(fields[13]),
        "warm_program_ns": float(fields[14]),
        "validation": fields[15],
        "stdout": stdout,
    }
    if (result["active_aic_groups"] != groups
            or result["descriptor_count"] != count
            or result["issue_mode"] != mode
            or result["rows"] != 128 or result["columns"] != 128
            or result["source_row_stride_elements"] != 1024
            or result["validation"] != "pass"
            or result["cold_program_ns"] <= 0
            or result["warm_program_ns"] <= 0):
        raise RuntimeError(f"probe reported wrong or invalid cell: {result!r}")
    return result


def run_fresh_process(binary: Path, groups: int, count: int,
                      mode: int) -> dict:
    completed = subprocess.run(
        (str(binary), str(groups), str(count), str(mode)),
        check=True, capture_output=True, text=True)
    if completed.stderr:
        raise RuntimeError(f"probe emitted stderr: {completed.stderr!r}")
    return parse_result(completed.stdout, groups, count, mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--tail-grid", action="store_true")
    args = parser.parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    visible_device = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    logical_device = os.environ["MIDLAYER_ASCEND_DEVICE"]
    if not visible_device.isdecimal() or logical_device != "0":
        raise ValueError(
            "pin one physical device via ASCEND_RT_VISIBLE_DEVICES "
            "and set MIDLAYER_ASCEND_DEVICE=0")

    descriptor_counts = (
        TAIL_DESCRIPTOR_COUNTS if args.tail_grid else DESCRIPTOR_COUNTS)
    timed_launches = 96 if args.tail_grid else 15
    if args.tail_grid and not args.binary.name.endswith("_tail"):
        raise ValueError("tail grid requires the 96-launch tail binary")
    if not args.tail_grid and args.binary.name.endswith("_tail"):
        raise ValueError("96-launch tail binary requires --tail-grid")
    cells = []
    for groups in GROUP_COUNTS:
        for count in descriptor_counts:
            for mode_name, mode in ISSUE_MODES:
                runs = [run_fresh_process(args.binary, groups, count, mode)
                        for _ in range(PROCESS_COUNT)]
                cold_values = [run["cold_program_ns"] for run in runs]
                warm_values = [run["warm_program_ns"] for run in runs]
                cold_ns = statistics.median(cold_values)
                warm_ns = statistics.median(warm_values)
                cold_span_pct = (
                    100.0 * (max(cold_values) - min(cold_values)) / cold_ns)
                warm_span_pct = (
                    100.0 * (max(warm_values) - min(warm_values)) / warm_ns)
                cells.append({
                    "active_aic_groups": groups,
                    "descriptor_count": count,
                    "issue_mode": mode_name,
                    "cold_program_ns": cold_ns,
                    "warm_program_ns": warm_ns,
                    "cold_to_warm_ratio": cold_ns / warm_ns,
                    "cold_process_median_span_pct": cold_span_pct,
                    "warm_process_median_span_pct": warm_span_pct,
                    "stable": (
                        cold_span_pct <= STABILITY_LIMIT_PCT
                        and warm_span_pct <= STABILITY_LIMIT_PCT),
                    "cold_distinguished": (
                        cold_ns >= warm_ns
                        * (1 + STABILITY_LIMIT_PCT / 100)),
                    "runs": runs,
                })
    pairs = []
    for groups in GROUP_COUNTS:
        for count in descriptor_counts:
            drained, overlap = (
                next(cell for cell in cells
                     if cell["active_aic_groups"] == groups
                     and cell["descriptor_count"] == count
                     and cell["issue_mode"] == mode_name)
                for mode_name in ("drained", "two_slot_overlap"))
            paired_pass = (
                drained["stable"] and overlap["stable"]
                and overlap["cold_program_ns"]
                <= drained["cold_program_ns"]
                * (1 + STABILITY_LIMIT_PCT / 100))
            cold_publishable = (
                paired_pass and drained["cold_distinguished"]
                and overlap["cold_distinguished"])
            drained["published"] = cold_publishable
            overlap["published"] = cold_publishable
            pairs.append({
                "active_aic_groups": groups,
                "descriptor_count": count,
                "cold_overlap_to_drained_ratio": (
                    overlap["cold_program_ns"]
                    / drained["cold_program_ns"]),
                "passed": paired_pass,
                "cold_published": cold_publishable,
            })
    source_dir = Path(__file__).resolve().parent
    record = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "mechanism": "split_fuse_bf16_interleaved_kv_cold_first_touch",
        "registration_path": str(source_dir / "REGISTRATION.md"),
        "measurement_baseline_commit": args.baseline,
        "physical_device_id": int(visible_device),
        "logical_device_id": int(logical_device),
        "binary_sha256": file_sha256(args.binary),
        "source_sha256": {
            name: file_sha256(source_dir / name)
            for name in ("kernel.cpp", "main.cpp", "CMakeLists.txt",
                         "runner.py", "REGISTRATION.md")},
        "process_count": PROCESS_COUNT,
        "descriptor_counts": descriptor_counts,
        "tail_grid": args.tail_grid,
        "timed_launches_per_process": timed_launches,
        "source_residency": "cold_first_touch_with_immediate_replay_control",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "cells": cells,
        "pairs": pairs,
        "passed": all(pair["passed"] for pair in pairs),
        "cold_published_pair_count": sum(
            pair["cold_published"] for pair in pairs),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "passed": record["passed"]}))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
