#!/usr/bin/env python3
"""Audit the concurrent first-GetValue service by active AIV count."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from first_get_value_runner import BLOCK_DIMS, CELLS, REQUEST_COUNTS


SYSTEM_COUNTER_TICK_NS = 20.0
REPLICATION_RELATIVE_LIMIT = 0.20
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "request_count", "buffer_bytes", "stride_bytes",
    "node_count", "system_cycles", "elapsed_ns", "sink", "validation",
)


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each request-count cell requires at least four samples")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _load_rows(paths):
    rows = []
    identities = set()
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
                raise ValueError(f"{path} has an invalid raw schema")
            for raw in reader:
                if raw["validation"] != "pass":
                    raise ValueError(f"{path} contains failed validation")
                integers = {
                    name: int(raw[name])
                    for name in (
                        "trial", "order_position", "process_pid",
                        "block_dim", "worker", "request_count",
                        "buffer_bytes", "stride_bytes",
                        "node_count", "system_cycles", "sink",
                    )
                }
                if (integers["trial"] <= 0 or integers["process_pid"] <= 0
                        or (integers["block_dim"],
                            integers["request_count"]) not in CELLS
                        or integers["worker"] < 0
                        or integers["worker"] >= integers["block_dim"]
                        or integers["buffer_bytes"] != 4096
                        or integers["stride_bytes"] != 4
                        or integers["node_count"] != 1024
                        or integers["system_cycles"] < 0):
                    raise ValueError(f"{path} contains invalid geometry or values")
                if (integers["request_count"] > 0
                        and integers["system_cycles"] == 0):
                    raise ValueError(f"{path} contains an unmeasured request cell")
                elapsed_ns = float(raw["elapsed_ns"])
                expected_ns = integers["system_cycles"] * SYSTEM_COUNTER_TICK_NS
                if (not math.isfinite(elapsed_ns) or elapsed_ns < 0
                        or not math.isclose(
                            elapsed_ns, expected_ns, rel_tol=0.0, abs_tol=1e-9)):
                    raise ValueError(f"{path} contains inconsistent elapsed time")
                identity = (
                    raw["run_id"], integers["trial"],
                    integers["block_dim"], integers["worker"],
                    integers["request_count"])
                if identity in identities:
                    raise ValueError(f"duplicate row identity {identity}")
                identities.add(identity)
                rows.append({
                    "run_id": raw["run_id"],
                    "trial": integers["trial"],
                    "order_position": integers["order_position"],
                    "process_pid": integers["process_pid"],
                    "block_dim": integers["block_dim"],
                    "worker": integers["worker"],
                    "request_count": integers["request_count"],
                    "elapsed_ns": elapsed_ns,
                })
    if not rows:
        raise ValueError("no first-GetValue rows were provided")
    return rows


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    if scale == 0:
        return 0.0
    return abs(left - right) / scale


def _audit(rows):
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one report must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    if trials != list(range(1, len(trials) + 1)):
        raise ValueError("trial identities must be contiguous from one")
    expected_cells = set(CELLS)
    by_trial = {}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        observed_cells = {
            (row["block_dim"], row["request_count"])
            for row in trial_rows}
        if observed_cells != expected_cells:
            raise ValueError("each trial changed the frozen cell set")
        if {row["order_position"] for row in trial_rows} != set(range(len(CELLS))):
            raise ValueError("each trial must contain every order position")
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        critical = {}
        for cell in CELLS:
            block_dim, request_count = cell
            cell_rows = [
                row for row in trial_rows
                if (row["block_dim"], row["request_count"]) == cell]
            if (len(cell_rows) != block_dim
                    or {row["worker"] for row in cell_rows}
                    != set(range(block_dim))
                    or len({row["order_position"] for row in cell_rows}) != 1):
                raise ValueError(
                    "each cell must contain every AIV worker exactly once")
            critical[cell] = max(row["elapsed_ns"] for row in cell_rows)
        by_trial[trial] = critical
    expected_per_position = len(trials) // len(CELLS)
    for position in range(len(CELLS)):
        counts = {
            cell: sum(
                row["order_position"] == position
                and row["worker"] == 0
                and (row["block_dim"], row["request_count"]) == cell
                for row in rows)
            for cell in CELLS
        }
        if set(counts.values()) != {expected_per_position}:
            raise ValueError("cell/order schedule is not balanced")

    summaries = {
        cell: _quartiles([values[cell] for values in by_trial.values()])
        for cell in CELLS
    }
    per_block_dim = {}
    for block_dim in BLOCK_DIMS:
        first_samples = tuple(
            values[(block_dim, 1)] - values[(block_dim, 0)]
            for values in by_trial.values())
        first_summary = _quartiles(first_samples)
        first_odd = statistics.median(first_samples[::2])
        first_even = statistics.median(first_samples[1::2])
        reproducible = (
            _relative_difference(first_odd, first_even)
            <= REPLICATION_RELATIVE_LIMIT)
        per_block_dim[block_dim] = {
            "first_samples": first_samples,
            "first_summary": first_summary,
            "first_odd": first_odd,
            "first_even": first_even,
            "reproducible": reproducible,
            "mechanism_gate": reproducible and first_summary[0] > 0,
        }
    first_medians = tuple(
        per_block_dim[block_dim]["first_summary"][1]
        for block_dim in BLOCK_DIMS)
    concurrency_monotonic = all(
        left <= right for left, right in zip(first_medians, first_medians[1:]))
    mechanism_gate = (
        all(result["mechanism_gate"] for result in per_block_dim.values())
        and concurrency_monotonic)
    return {
        "samples_per_cell": len(trials),
        "summaries": summaries,
        "per_block_dim": per_block_dim,
        "first_medians": first_medians,
        "concurrency_monotonic": concurrency_monotonic,
        "mechanism_gate": mechanism_gate,
    }


def _print_report(audit):
    print("# Ascend 910B first-GetValue dependency-curve audit")
    print(f"samples_per_cell={audit['samples_per_cell']}")
    print("block_dim,request_count,q1_critical_ns,median_critical_ns,q3_critical_ns")
    for cell in CELLS:
        q1, median, q3 = audit["summaries"][cell]
        print(f"{cell[0]},{cell[1]},{q1:.9f},{median:.9f},{q3:.9f}")
    for block_dim in BLOCK_DIMS:
        result = audit["per_block_dim"][block_dim]
        print(
            f"block_dim={block_dim} first_increment_ns="
            f"q1:{result['first_summary'][0]:.9f},"
            f"median:{result['first_summary'][1]:.9f},"
            f"q3:{result['first_summary'][2]:.9f}")
        print(
            f"block_dim={block_dim} replication_gate="
            f"{'pass' if result['reproducible'] else 'fail'}")
    print(
        "concurrency_monotonic_gate="
        f"{'pass' if audit['concurrency_monotonic'] else 'fail'}")
    print(f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    _print_report(_audit(_load_rows(args.raw_csv)))


if __name__ == "__main__":
    main()
