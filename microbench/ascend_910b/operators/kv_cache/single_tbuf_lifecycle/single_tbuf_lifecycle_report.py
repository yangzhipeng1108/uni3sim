#!/usr/bin/env python3
"""Audit the exact single-TBuf lifecycle campaign."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from single_tbuf_lifecycle_runner import (
    BLOCK_DIMS, CELLS, ITERATIONS, PAYLOAD_BYTES)


FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "payload_bytes", "iterations", "short_block_cycles",
    "long_block_cycles", "tick_ns", "service_ns", "validation")


def _cycles(text, count):
    values = tuple(int(value) for value in text.split("|"))
    if len(values) != count or any(value <= 0 for value in values):
        raise ValueError("cycle vector is invalid")
    return values


def _read(paths):
    rows = []
    identities = set()
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"{path} changed the raw schema")
            for raw in reader:
                if raw["validation"] != "pass":
                    raise ValueError(f"{path} contains failed validation")
                row = {
                    "run_id": raw["run_id"],
                    "trial": int(raw["trial"]),
                    "order_position": int(raw["order_position"]),
                    "process_pid": int(raw["process_pid"]),
                    "block_dim": int(raw["block_dim"]),
                    "payload_bytes": int(raw["payload_bytes"]),
                    "iterations": int(raw["iterations"]),
                    "tick_ns": float(raw["tick_ns"]),
                    "service_ns": float(raw["service_ns"]),
                }
                identity = (
                    row["run_id"], row["trial"], row["order_position"])
                if identity in identities:
                    raise ValueError(f"duplicate raw identity {identity}")
                identities.add(identity)
                cell = (row["block_dim"], row["payload_bytes"])
                if (cell not in CELLS or row["iterations"] != ITERATIONS
                        or row["trial"] <= 0
                        or not 0 <= row["order_position"] < len(CELLS)
                        or row["process_pid"] <= 0
                        or row["tick_ns"] != 20.0
                        or not math.isfinite(row["service_ns"])
                        or row["service_ns"] <= 0.0):
                    raise ValueError(f"{path} contains an invalid row")
                short = _cycles(raw["short_block_cycles"], row["block_dim"])
                long = _cycles(raw["long_block_cycles"], row["block_dim"])
                if any(end <= start for start, end in zip(short, long)):
                    raise ValueError(f"{path} has non-increasing cycles")
                expected = (
                    (max(long) - max(short)) * row["tick_ns"]
                    / (3 * ITERATIONS))
                if not math.isclose(
                        row["service_ns"], expected,
                        rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError(f"{path} changed the derived service")
                rows.append(row)
    return rows


def audit(paths):
    rows = _read(paths)
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != 3:
        raise ValueError("campaign requires exactly three independent runs")
    for run_id in run_ids:
        run_rows = [row for row in rows if row["run_id"] == run_id]
        trials = sorted({row["trial"] for row in run_rows})
        if trials != list(range(1, 2 * len(CELLS) + 1)):
            raise ValueError("each run requires the frozen balanced trials")
        for trial in trials:
            trial_rows = [row for row in run_rows if row["trial"] == trial]
            if ({(row["block_dim"], row["payload_bytes"])
                 for row in trial_rows} != set(CELLS)
                    or {row["order_position"] for row in trial_rows}
                    != set(range(len(CELLS)))):
                raise ValueError("trial does not cover every cell and position")
    run_spreads = {}
    concurrency_spreads = {}
    services = {}
    for payload_bytes in PAYLOAD_BYTES:
        cell_medians = []
        for block_dim in BLOCK_DIMS:
            medians = tuple(statistics.median(
                row["service_ns"] for row in rows
                if row["run_id"] == run_id
                and row["block_dim"] == block_dim
                and row["payload_bytes"] == payload_bytes)
                for run_id in run_ids)
            median = statistics.median(medians)
            run_spreads[(block_dim, payload_bytes)] = (
                (max(medians) - min(medians)) / median)
            cell_medians.append(median)
        service = statistics.median(cell_medians)
        concurrency_spreads[payload_bytes] = (
            (max(cell_medians) - min(cell_medians)) / service)
        services[payload_bytes] = service
    maximum_run_spread = max(run_spreads.values())
    maximum_concurrency_spread = max(concurrency_spreads.values())
    return {
        "independent_runs": len(run_ids),
        "maximum_run_relative_median_spread": maximum_run_spread,
        "maximum_concurrency_relative_median_spread": (
            maximum_concurrency_spread),
        "service_time_ns_per_worker_by_buffer_bytes": services,
        "mechanism_publishable": (
            maximum_run_spread <= 0.10
            and maximum_concurrency_spread <= 0.10),
        "production_or_evaluation_duration_consumed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    result = audit(tuple(args.raw_csv))
    for name, value in result.items():
        print(f"{name}={value}")
    if not result["mechanism_publishable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
