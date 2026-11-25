#!/usr/bin/env python3
"""Audit three independent elementwise Mul compute-segment runs."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED_CELLS = tuple(
    (workers, elements, segment)
    for workers, elements in (
        (46, 1792), (46, 1280), (47, 1408), (47, 768),
        (47, 2816), (47, 1536),
        (48, 8832), (48, 8448), (48, 4352))
    for segment in (0, 1))
CONTROL_SERVICE_NS = {
    (46, 1280, 0): 80.0,
    (46, 1280, 1): 140.0,
    (46, 1792, 0): 100.0,
    (46, 1792, 1): 180.0,
    (47, 768, 0): 60.0,
    (47, 768, 1): 120.0,
    (47, 1408, 0): 80.0,
    (47, 1408, 1): 160.0,
    (47, 1536, 0): 100.0,
    (47, 1536, 1): 160.0,
    (47, 2816, 0): 140.0,
    (47, 2816, 1): 260.0,
}
CROSS_RUN_SPREAD_LIMIT_PERCENT = 20.0
CONTROL_REPRODUCTION_LIMIT_PERCENT = 20.0


def _load(path):
    cells = {cell: [] for cell in EXPECTED_CELLS}
    positions = {cell: [] for cell in EXPECTED_CELLS}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            cell = (int(row["workers"]), int(row["elements"]),
                    int(row["segment"]))
            if cell not in cells or row["validation"] != "pass":
                raise ValueError(f"invalid row in {path}: {cell}")
            value = float(row["service_ns"])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid service in {path}: {value}")
            cells[cell].append(value)
            positions[cell].append(int(row["order_position"]))
    expected_rows = len(EXPECTED_CELLS) * len(EXPECTED_CELLS) * 2
    if sum(map(len, cells.values())) != expected_rows:
        raise ValueError(f"{path} does not contain two balanced cycles")
    expected_positions = sorted(list(range(len(EXPECTED_CELLS))) * 2)
    for cell in EXPECTED_CELLS:
        if sorted(positions[cell]) != expected_positions:
            raise ValueError(f"{path} is not position-balanced for {cell}")
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [_load(path) for path in args.raw_csv]
    cells = []
    passed = True
    for cell in EXPECTED_CELLS:
        medians = [statistics.median(run[cell]) for run in runs]
        selected = statistics.median(medians)
        spread = (max(medians) - min(medians)) / max(medians) * 100.0
        control_service = None
        if cell in CONTROL_SERVICE_NS:
            control_service = CONTROL_SERVICE_NS[cell]
        control_error = (
            abs(selected - control_service) / control_service * 100.0
            if control_service is not None else None)
        cell_passed = (
            selected > 0.0
            and spread <= CROSS_RUN_SPREAD_LIMIT_PERCENT
            and (control_error is None
                 or control_error <= CONTROL_REPRODUCTION_LIMIT_PERCENT))
        passed = passed and cell_passed
        cells.append({
            "workers": cell[0], "elements": cell[1], "segment": cell[2],
            "run_medians_ns": medians, "selected_service_ns": selected,
            "cross_run_spread_percent": spread,
            "published_control_service_ns": control_service,
            "control_reproduction_error_percent": control_error,
            "passed": cell_passed,
        })
    report = {
        "schema_version": 1,
        "campaign": "elementwise_mul_compute_segments",
        "cross_run_spread_limit_percent": CROSS_RUN_SPREAD_LIMIT_PERCENT,
        "control_reproduction_limit_percent": (
            CONTROL_REPRODUCTION_LIMIT_PERCENT),
        "all_cells_passed": passed,
        "cells": cells,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
