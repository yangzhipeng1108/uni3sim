#!/usr/bin/env python3
"""Run the registered mixed-S2 FIA paged MM1 task-program probe."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


PROCESS_COUNT = 5
PROFILE = 3
ACTUAL_M = 4
EXPECTED_HEADER = (
    "profile,actual_m,workers,task_counts,min_cycles,median_cycles,"
    "max_cycles,tick_ns,min_ns,median_ns,max_ns,validation")


def parse_result(stdout: str) -> dict[str, object]:
    lines = stdout.splitlines()
    if len(lines) != 2:
        raise RuntimeError(
            f"expected one CSV header and one result row, got {stdout!r}")
    if lines[0] != EXPECTED_HEADER:
        raise RuntimeError(f"unexpected CSV header: {lines[0]!r}")
    fields = lines[1].split(",")
    if len(fields) != 12:
        raise RuntimeError(f"unexpected CSV field count: {lines[1]!r}")
    result = {
        "profile": int(fields[0]),
        "actual_m": int(fields[1]),
        "workers": int(fields[2]),
        "task_counts": fields[3],
        "min_cycles": int(fields[4]),
        "median_cycles": float(fields[5]),
        "max_cycles": int(fields[6]),
        "tick_ns": float(fields[7]),
        "min_ns": float(fields[8]),
        "median_ns": float(fields[9]),
        "max_ns": float(fields[10]),
        "validation": fields[11],
    }
    if result["profile"] != PROFILE or result["actual_m"] != ACTUAL_M:
        raise RuntimeError(f"wrong probe cell: {result!r}")
    if result["workers"] != 24 or result["validation"] != "pass":
        raise RuntimeError(f"failed worker validation: {result!r}")
    return result


def run_fresh_process(binary: Path) -> dict[str, object]:
    completed = subprocess.run(
        (str(binary), str(PROFILE), str(ACTUAL_M)),
        check=True,
        text=True,
        capture_output=True,
    )
    if completed.stderr != "":
        raise RuntimeError(f"probe emitted stderr: {completed.stderr!r}")
    result = parse_result(completed.stdout)
    result["stdout"] = completed.stdout
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.output.exists():
        raise FileExistsError(args.output)
    environment = os.environ
    if environment["ASCEND_DEVICE_ID"] != "0":
        raise RuntimeError("ASCEND_DEVICE_ID must be logical device 0")
    if environment["ASCEND_RT_VISIBLE_DEVICES"] != "2":
        raise RuntimeError(
            "ASCEND_RT_VISIBLE_DEVICES must pin physical device 2")

    runs = []
    for process_index in range(PROCESS_COUNT):
        result = run_fresh_process(args.binary)
        result["fresh_process_index"] = process_index
        runs.append(result)
    medians_ns = [float(run["median_ns"]) for run in runs]
    median_ns = statistics.median(medians_ns)
    span_pct = (max(medians_ns) - min(medians_ns)) / median_ns * 100.0
    record = {
        "schema": "fia_paged_mte2_task_program_v1",
        "baseline": args.baseline,
        "hardware_id": "ascend_910b",
        "physical_device_id": environment["ASCEND_RT_VISIBLE_DEVICES"],
        "logical_device_id": environment["ASCEND_DEVICE_ID"],
        "profile": PROFILE,
        "actual_m": ACTUAL_M,
        "process_count": PROCESS_COUNT,
        "runs": runs,
        "median_of_process_medians_ns": median_ns,
        "process_median_span_pct": span_pct,
        "gates": {
            "all_workers_validate": all(
                run["validation"] == "pass" and run["workers"] == 24
                for run in runs),
            "stderr_empty": True,
            "process_median_span_at_most_5pct": span_pct <= 5.0,
        },
    }
    record["passed"] = all(record["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "passed": record["passed"],
        "span_pct": span_pct,
    }))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
