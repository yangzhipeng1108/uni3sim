#!/usr/bin/env python3
"""Run the registered FIA GQA drained ND2NZ descriptor-count-law campaign.

Every cell, gate and aggregation here is fixed by REGISTRATION.md, which was
written before any measurement.  The runner never selects between estimators.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


PROCESS_COUNT = 3
GEOMETRIES = (
    # (label, rows, columns, source_row_stride_elements)
    ("d128_gqa_kv", 256, 128, 1024),
    ("d64_gqa_kv", 256, 64, 512),
)
GROUP_COUNTS = (8, 16, 22, 24)
DESCRIPTOR_COUNTS = (8, 16, 32, 64, 128)
PROCESS_MEDIAN_SPAN_LIMIT_PCT = 5.0
COUNT_INDEPENDENCE_SPAN_LIMIT_PCT = 8.0
EXPECTED_HEADER = (
    "groups,rows,columns,stride_elements,count,min_cycles,median_cycles,"
    "max_cycles,tick_ns,program_ns,per_descriptor_ns,validation")


def parse_result(stdout: str) -> dict:
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
        "groups": int(fields[0]),
        "rows": int(fields[1]),
        "columns": int(fields[2]),
        "stride_elements": int(fields[3]),
        "count": int(fields[4]),
        "min_cycles": int(fields[5]),
        "median_cycles": float(fields[6]),
        "max_cycles": int(fields[7]),
        "tick_ns": float(fields[8]),
        "program_ns": float(fields[9]),
        "per_descriptor_ns": float(fields[10]),
        "validation": fields[11],
    }
    if result["validation"] != "pass":
        raise RuntimeError(f"failed worker validation: {result!r}")
    return result


def run_fresh_process(binary: Path, groups, rows, columns, stride, count):
    completed = subprocess.run(
        (str(binary), str(groups), str(rows), str(columns), str(stride),
         str(count)),
        check=True, text=True, capture_output=True)
    if completed.stderr != "":
        raise RuntimeError(f"probe emitted stderr: {completed.stderr!r}")
    result = parse_result(completed.stdout)
    if (result["groups"] != groups or result["rows"] != rows
            or result["columns"] != columns
            or result["stride_elements"] != stride
            or result["count"] != count):
        raise RuntimeError(f"probe reported the wrong cell: {result!r}")
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
    device = os.environ["MIDLAYER_ASCEND_DEVICE"]

    cells = []
    for label, rows, columns, stride in GEOMETRIES:
        for groups in GROUP_COUNTS:
            for count in DESCRIPTOR_COUNTS:
                runs = [
                    run_fresh_process(
                        args.binary, groups, rows, columns, stride, count)
                    for _ in range(PROCESS_COUNT)
                ]
                per_descriptor = [
                    run["per_descriptor_ns"] for run in runs]
                median = statistics.median(per_descriptor)
                span_pct = (
                    (max(per_descriptor) - min(per_descriptor)) / median
                    * 100.0)
                cells.append({
                    "geometry": label,
                    "rows": rows,
                    "columns": columns,
                    "source_row_stride_elements": stride,
                    "active_aic_groups": groups,
                    "descriptor_count": count,
                    "runs": runs,
                    "per_descriptor_ns": median,
                    "process_median_span_pct": span_pct,
                    "stable": span_pct <= PROCESS_MEDIAN_SPAN_LIMIT_PCT,
                })
                print(json.dumps({
                    "geometry": label, "groups": groups, "count": count,
                    "per_descriptor_ns": round(median, 2),
                    "span_pct": round(span_pct, 2)}), flush=True)

    count_independence = []
    for label, rows, columns, stride in GEOMETRIES:
        for groups in GROUP_COUNTS:
            values = [
                cell["per_descriptor_ns"] for cell in cells
                if cell["geometry"] == label
                and cell["active_aic_groups"] == groups]
            median = statistics.median(values)
            span_pct = (max(values) - min(values)) / median * 100.0
            count_independence.append({
                "geometry": label,
                "active_aic_groups": groups,
                "per_descriptor_ns_by_count": {
                    str(cell["descriptor_count"]): cell["per_descriptor_ns"]
                    for cell in cells
                    if cell["geometry"] == label
                    and cell["active_aic_groups"] == groups},
                "count_free_per_descriptor_ns": median,
                "count_span_pct": span_pct,
                "count_independent": (
                    span_pct <= COUNT_INDEPENDENCE_SPAN_LIMIT_PCT),
            })

    record = {
        "schema": "fia_gqa_drain_count_law_v1",
        "baseline": args.baseline,
        "hardware_id": "ascend_910b",
        "physical_device_id": device,
        "process_count": PROCESS_COUNT,
        "timing_boundary": (
            "GetSystemCycle immediately before the first descriptor's "
            "WaitFlag<MTE1_MTE2> to GetSystemCycle immediately after the "
            "last descriptor's WaitFlag<MTE2_MTE1>; divided by descriptor "
            "count; worker median per launch, process median per cell"),
        "cells": cells,
        "count_independence": count_independence,
        "gates": {
            "all_cells_stable": all(cell["stable"] for cell in cells),
            "stderr_empty": True,
            "all_cells_count_independent": all(
                entry["count_independent"] for entry in count_independence),
        },
    }
    record["passed"] = all(record["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "passed": record["passed"],
        "gates": record["gates"],
    }))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
