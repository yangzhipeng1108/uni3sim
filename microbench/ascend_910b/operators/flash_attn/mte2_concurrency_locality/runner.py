#!/usr/bin/env python3
"""Run the registered FIA drained ND2NZ concurrency/locality campaign.

Every cell, gate and aggregation here is fixed by REGISTRATION.md, which was
written before any measurement.  The runner never selects between estimators.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

PROCESS_COUNT = 3
ROWS, COLUMNS, STRIDE = 256, 128, 1024
GROUP_COUNTS = (8, 16, 22, 24)
ADDRESS_MODES = (("disjoint", 0), ("packed", 1))
DESCRIPTOR_COUNT = 64
# Amendment grid: the packed layout over the domain the disjoint law published.
AMENDMENT_GEOMETRIES = (
    ("d128_gqa_kv", 256, 128, 1024),
    ("d64_gqa_kv", 256, 64, 512),
)
AMENDMENT_DESCRIPTOR_COUNTS = (8, 16, 32, 64, 128)
PRODUCTION_DESCRIPTOR_COUNTS = (1, 2, 4)
COUNT_INDEPENDENCE_SPAN_LIMIT_PCT = 8.0
PROCESS_MEDIAN_SPAN_LIMIT_PCT = 5.0
CROSS_CHECK_LIMIT_PCT = 10.0
# aic_mte2_fia_gqa_drained_descriptor_count_law_cann_8_5_1.json, columns=128,
# descriptor count 64.  Used only as the declared instrument cross-check.
PUBLISHED_DISJOINT_NS = {8: 1240.4688, 16: 1558.9062, 22: 2109.2188,
                         24: 2293.5938}
CONFIRM_GROWTH = 1.55
REJECT_GROWTH = 1.40
EXPECTED_HEADER = (
    "groups,rows,columns,stride_elements,count,address_mode,"
    "min_cycles,median_cycles,max_cycles,tick_ns,program_ns,"
    "per_descriptor_ns,validation")


def parse_result(stdout: str) -> dict:
    lines = stdout.splitlines()
    if len(lines) != 2:
        raise RuntimeError(
            f"expected one CSV header and one result row, got {stdout!r}")
    if lines[0] != EXPECTED_HEADER:
        raise RuntimeError(f"unexpected CSV header: {lines[0]!r}")
    fields = lines[1].split(",")
    if len(fields) != 13:
        raise RuntimeError(f"unexpected CSV field count: {lines[1]!r}")
    result = {
        "groups": int(fields[0]), "rows": int(fields[1]),
        "columns": int(fields[2]), "stride_elements": int(fields[3]),
        "count": int(fields[4]), "address_mode": int(fields[5]),
        "min_cycles": int(fields[6]), "median_cycles": float(fields[7]),
        "max_cycles": int(fields[8]), "tick_ns": float(fields[9]),
        "program_ns": float(fields[10]),
        "per_descriptor_ns": float(fields[11]), "validation": fields[12],
    }
    if result["validation"] != "pass":
        raise RuntimeError(f"failed worker validation: {result!r}")
    return result


def run_fresh_process(binary: Path, groups: int, mode: int, *,
                      rows: int = ROWS, columns: int = COLUMNS,
                      stride: int = STRIDE,
                      count: int = DESCRIPTOR_COUNT) -> dict:
    completed = subprocess.run(
        (str(binary), str(groups), str(rows), str(columns), str(stride),
         str(count), str(mode)),
        check=True, text=True, capture_output=True)
    if completed.stderr != "":
        raise RuntimeError(f"probe emitted stderr: {completed.stderr!r}")
    result = parse_result(completed.stdout)
    if (result["groups"] != groups or result["address_mode"] != mode
            or result["rows"] != rows or result["columns"] != columns
            or result["stride_elements"] != stride
            or result["count"] != count):
        raise RuntimeError(f"probe reported the wrong cell: {result!r}")
    return result


def run_production_counts(args, device: str) -> int:
    """Amendment 2: both layouts at the descriptor counts production issues."""

    cells = []
    for label, rows, columns, stride in AMENDMENT_GEOMETRIES:
        for groups in GROUP_COUNTS:
            for count in PRODUCTION_DESCRIPTOR_COUNTS:
                entry = {"geometry": label, "rows": rows, "columns": columns,
                         "source_row_stride_elements": stride,
                         "active_aic_groups": groups,
                         "descriptor_count": count}
                for mode_label, mode in ADDRESS_MODES:
                    runs = [run_fresh_process(args.binary, groups, mode,
                                              rows=rows, columns=columns,
                                              stride=stride, count=count)
                            for _ in range(PROCESS_COUNT)]
                    services = [run["per_descriptor_ns"] for run in runs]
                    value = statistics.median(services)
                    span = 100.0 * (max(services) - min(services)) / value
                    entry[mode_label] = {
                        "per_descriptor_ns": value,
                        "process_median_span_pct": span,
                        "stable": span <= PROCESS_MEDIAN_SPAN_LIMIT_PCT,
                        "process_values_ns": services,
                    }
                entry["layout_factor"] = (
                    entry["packed"]["per_descriptor_ns"]
                    / entry["disjoint"]["per_descriptor_ns"])
                entry["published"] = (
                    entry["packed"]["stable"] and entry["disjoint"]["stable"])
                cells.append(entry)
                print(f"  {label:11s} groups={groups:2d} count={count:2d}  "
                      f"disjoint {entry['disjoint']['per_descriptor_ns']:8.1f}  "
                      f"packed {entry['packed']['per_descriptor_ns']:8.1f}  "
                      f"factor {entry['layout_factor']:5.3f}  "
                      f"{'ok' if entry['published'] else 'UNSTABLE'}", flush=True)
    report = {
        "schema_version": 1, "hardware_id": "ascend_910b",
        "mechanism": "aic_mte2_fia_drained_descriptor_production_count_layout",
        "registration_path": (
            "microbench/ascend_910b/operators/flash_attn/"
            "mte2_concurrency_locality/REGISTRATION.md"),
        "measurement_baseline_commit": args.baseline,
        "physical_device_id": int(device),
        "process_count": PROCESS_COUNT,
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "cells": cells,
        "published_cell_count": sum(1 for c in cells if c["published"]),
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n",
                           encoding="utf-8")
    print(json.dumps({"published_cell_count": report["published_cell_count"],
                      "total_cells": len(cells)}, indent=1))
    return 0


def run_amendment(args, device: str) -> int:
    cells = []
    for label, rows, columns, stride in AMENDMENT_GEOMETRIES:
        for groups in GROUP_COUNTS:
            by_count = {}
            for count in AMENDMENT_DESCRIPTOR_COUNTS:
                runs = [run_fresh_process(args.binary, groups, 1, rows=rows,
                                          columns=columns, stride=stride,
                                          count=count)
                        for _ in range(PROCESS_COUNT)]
                services = [run["per_descriptor_ns"] for run in runs]
                value = statistics.median(services)
                by_count[count] = {
                    "per_descriptor_ns": value,
                    "process_median_span_pct":
                        100.0 * (max(services) - min(services)) / value,
                    "process_values_ns": services,
                }
                print(f"  packed {label:11s} groups={groups:2d} count={count:3d}"
                      f"  {value:8.1f} ns  span "
                      f"{by_count[count]['process_median_span_pct']:4.1f}%",
                      flush=True)
            stable = all(
                entry["process_median_span_pct"]
                <= PROCESS_MEDIAN_SPAN_LIMIT_PCT for entry in by_count.values())
            services = [entry["per_descriptor_ns"] for entry in by_count.values()]
            median_service = statistics.median(services)
            count_span = 100.0 * (max(services) - min(services)) / median_service
            cells.append({
                "address_mode": "packed", "geometry": label, "rows": rows,
                "columns": columns, "source_row_stride_elements": stride,
                "active_aic_groups": groups,
                "measured_per_descriptor_ns_by_count": {
                    str(count): entry["per_descriptor_ns"]
                    for count, entry in by_count.items()},
                "per_count_detail": {str(k): v for k, v in by_count.items()},
                "count_span_pct": count_span,
                "count_independent":
                    count_span <= COUNT_INDEPENDENCE_SPAN_LIMIT_PCT,
                "all_counts_stable": stable,
                "published": (
                    stable
                    and count_span <= COUNT_INDEPENDENCE_SPAN_LIMIT_PCT),
                "count_free_per_descriptor_ns": median_service,
            })
    report = {
        "schema_version": 1, "hardware_id": "ascend_910b",
        "mechanism": "aic_mte2_fia_drained_descriptor_packed_count_law",
        "registration_path": (
            "microbench/ascend_910b/operators/flash_attn/"
            "mte2_concurrency_locality/REGISTRATION.md"),
        "measurement_baseline_commit": args.baseline,
        "physical_device_id": int(device),
        "process_count": PROCESS_COUNT,
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "cells": cells,
        "published_cell_count": sum(1 for c in cells if c["published"]),
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n",
                           encoding="utf-8")
    print(json.dumps({"published_cell_count": report["published_cell_count"],
                      "total_cells": len(cells)}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--cells", choices=("decision", "amendment", "production_counts"),
        default="decision")
    args = parser.parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.output.exists():
        raise FileExistsError(args.output)
    device = os.environ["MIDLAYER_ASCEND_DEVICE"]

    if args.cells == "amendment":
        return run_amendment(args, device)
    if args.cells == "production_counts":
        return run_production_counts(args, device)
    cells = []
    for label, mode in ADDRESS_MODES:
        for groups in GROUP_COUNTS:
            runs = [run_fresh_process(args.binary, groups, mode)
                    for _ in range(PROCESS_COUNT)]
            services = [run["per_descriptor_ns"] for run in runs]
            value = statistics.median(services)
            span = 100.0 * (max(services) - min(services)) / value
            cell = {
                "address_mode": label, "active_aic_groups": groups,
                "rows": ROWS, "columns": COLUMNS,
                "source_row_stride_elements": STRIDE,
                "descriptor_count": DESCRIPTOR_COUNT,
                "per_descriptor_ns": value,
                "process_median_span_pct": span,
                "stable": span <= PROCESS_MEDIAN_SPAN_LIMIT_PCT,
                "process_values_ns": services,
            }
            if label == "disjoint":
                published = PUBLISHED_DISJOINT_NS[groups]
                cell["published_ns"] = published
                cell["cross_check_deviation_pct"] = (
                    100.0 * (value - published) / published)
                cell["cross_check_passed"] = (
                    abs(cell["cross_check_deviation_pct"])
                    <= CROSS_CHECK_LIMIT_PCT)
            cells.append(cell)
            print(f"  {label:9s} groups={groups:2d}  "
                  f"{value:8.1f} ns/descriptor  span {span:4.1f}%", flush=True)

    by_mode = {(c["address_mode"], c["active_aic_groups"]): c for c in cells}
    growth = {label: by_mode[(label, 16)]["per_descriptor_ns"]
              / by_mode[(label, 8)]["per_descriptor_ns"]
              for label, _ in ADDRESS_MODES}
    packed_growth = growth["packed"]
    if packed_growth >= CONFIRM_GROWTH:
        verdict = "locality_confirmed"
    elif packed_growth < REJECT_GROWTH:
        verdict = "locality_rejected"
    else:
        verdict = "partial_no_publication"
    report = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "mechanism": "aic_mte2_fia_drained_descriptor_concurrency_locality",
        "registration_path": (
            "microbench/ascend_910b/operators/flash_attn/"
            "mte2_concurrency_locality/REGISTRATION.md"),
        "measurement_baseline_commit": args.baseline,
        "physical_device_id": int(device),
        "process_count": PROCESS_COUNT,
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "cells": cells,
        "blocks_8_to_16_growth": growth,
        "decision_thresholds": {"confirm_at_or_above": CONFIRM_GROWTH,
                                "reject_below": REJECT_GROWTH},
        "verdict": verdict,
        "all_cells_stable": all(c["stable"] for c in cells),
        "cross_check_passed": all(
            c["cross_check_passed"] for c in cells if "cross_check_passed" in c),
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "blocks_8_to_16_growth", "verdict", "all_cells_stable",
        "cross_check_passed")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
