#!/usr/bin/env python3
"""Audit AIV scalar-memory latency raw data with predeclared gates."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from dram_latency_runner import CELLS


REPLICATION_MAPE_LIMIT_PCT = 10.0
REPLICATION_MAX_APE_LIMIT_PCT = 20.0
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "working_set",
    "buffer_mb", "buffer_bytes", "stride_bytes", "node_count",
    "short_hops", "long_hops", "short_cycles", "short_wall_ns",
    "long_cycles", "long_wall_ns", "tick_ns", "latency_cycles",
    "latency_ns", "short_sink", "long_sink", "validation",
)


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each scalar-memory cell requires at least four samples")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _load_rows(paths):
    rows = []
    identities = set()
    expected_cells = {
        name: (buffer_mb, stride_bytes, hops)
        for name, buffer_mb, stride_bytes, hops in CELLS
    }
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
                raise ValueError(f"{path} has an invalid raw schema")
            for raw in reader:
                if raw["validation"] != "pass":
                    raise ValueError(f"{path} contains failed validation")
                working_set = raw["working_set"]
                if working_set not in expected_cells:
                    raise ValueError(f"{path} contains an unknown working set")
                integers = {
                    name: int(raw[name])
                    for name in (
                        "trial", "order_position", "process_pid", "buffer_mb",
                        "buffer_bytes", "stride_bytes", "node_count",
                        "short_hops", "long_hops", "short_cycles",
                        "short_wall_ns", "long_cycles", "long_wall_ns",
                        "short_sink", "long_sink",
                    )
                }
                if integers["trial"] <= 0 or integers["process_pid"] <= 0:
                    raise ValueError(f"{path} contains invalid process identity")
                expected = expected_cells[working_set]
                if (integers["buffer_mb"], integers["stride_bytes"],
                        integers["short_hops"]) != expected:
                    raise ValueError(f"{path} changed the frozen cell geometry")
                if integers["buffer_bytes"] != integers["buffer_mb"] * 1024**2:
                    raise ValueError(f"{path} contains invalid buffer bytes")
                if integers["node_count"] != (
                        integers["buffer_bytes"] // integers["stride_bytes"]):
                    raise ValueError(f"{path} contains invalid node count")
                if integers["long_hops"] != 4 * integers["short_hops"]:
                    raise ValueError(f"{path} contains invalid hop ratio")
                if (integers["short_cycles"] <= 0
                        or integers["long_cycles"] <= integers["short_cycles"]
                        or integers["short_wall_ns"] <= 0
                        or integers["long_wall_ns"]
                        <= integers["short_wall_ns"]):
                    raise ValueError(f"{path} contains invalid raw intervals")
                identity = (raw["run_id"], integers["trial"], working_set)
                if identity in identities:
                    raise ValueError(f"duplicate row identity {identity}")
                identities.add(identity)
                delta_cycles = (
                    integers["long_cycles"] - integers["short_cycles"])
                delta_wall_ns = (
                    integers["long_wall_ns"] - integers["short_wall_ns"])
                extra_hops = (
                    integers["long_hops"] - integers["short_hops"])
                expected_values = (
                    delta_wall_ns / delta_cycles,
                    delta_cycles / extra_hops,
                    delta_wall_ns / extra_hops,
                )
                observed_values = tuple(float(raw[name]) for name in (
                    "tick_ns", "latency_cycles", "latency_ns"))
                if any(
                    not math.isclose(observed, expected_value,
                                     rel_tol=1e-8, abs_tol=1e-8)
                    for observed, expected_value in zip(
                        observed_values, expected_values)
                ):
                    raise ValueError(f"{path} contains inconsistent derived values")
                if any(value <= 0 or not math.isfinite(value)
                       for value in observed_values):
                    raise ValueError(f"{path} contains non-positive measurement")
                rows.append({
                    "run_id": raw["run_id"],
                    "trial": integers["trial"],
                    "order_position": integers["order_position"],
                    "process_pid": integers["process_pid"],
                    "working_set": working_set,
                    "latency_ns": observed_values[2],
                    "tick_ns": observed_values[0],
                })
    if not rows:
        raise ValueError("no scalar-memory latency rows were provided")
    return rows


def _prediction_errors(training_rows, validation_rows):
    prediction = statistics.median(
        row["latency_ns"] for row in training_rows)
    errors = tuple(
        abs(prediction - row["latency_ns"]) / row["latency_ns"] * 100.0
        for row in validation_rows)
    return prediction, statistics.fmean(errors), max(errors)


def _audit(rows):
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one report must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    if trials != list(range(1, len(trials) + 1)):
        raise ValueError("trial identities must be contiguous from one")
    expected_names = {cell[0] for cell in CELLS}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        if len(trial_rows) != len(CELLS):
            raise ValueError("each trial must contain every cell exactly once")
        if {row["working_set"] for row in trial_rows} != expected_names:
            raise ValueError("each trial changed the frozen cell set")
        if {row["order_position"] for row in trial_rows} != set(range(len(CELLS))):
            raise ValueError("each trial must contain every order position")
    expected_per_position = len(trials) // len(CELLS)
    for position in range(len(CELLS)):
        counts = {
            name: sum(
                row["order_position"] == position
                and row["working_set"] == name
                for row in rows)
            for name in expected_names
        }
        if set(counts.values()) != {expected_per_position}:
            raise ValueError("cell/order schedule is not balanced")

    summaries = {}
    validations = {}
    candidates = {}
    reproducible = True
    for name in sorted(expected_names):
        cell_rows = [row for row in rows if row["working_set"] == name]
        odd = [row for row in cell_rows if row["trial"] % 2 == 1]
        even = [row for row in cell_rows if row["trial"] % 2 == 0]
        if not odd or not even:
            raise ValueError("odd/even validation requires both trial subsets")
        odd_to_even = _prediction_errors(odd, even)
        even_to_odd = _prediction_errors(even, odd)
        gate = all(
            result[1] <= REPLICATION_MAPE_LIMIT_PCT
            and result[2] <= REPLICATION_MAX_APE_LIMIT_PCT
            for result in (odd_to_even, even_to_odd))
        reproducible = reproducible and gate
        summaries[name] = _quartiles(
            [row["latency_ns"] for row in cell_rows])
        validations[name] = (odd_to_even, even_to_odd, gate)
        candidates[name] = statistics.median(
            row["latency_ns"] for row in cell_rows)
    hierarchy_separated = summaries["l2"][2] < summaries["dram"][0]
    publication_gate = reproducible and hierarchy_separated
    return {
        "samples_per_cell": len(rows) // len(CELLS),
        "summaries": summaries,
        "validations": validations,
        "candidates": candidates if publication_gate else None,
        "reproducible": reproducible,
        "hierarchy_separated": hierarchy_separated,
        "publication_gate": publication_gate,
    }


def _print_report(audit):
    print("# Ascend 910B AIV scalar-memory latency audit")
    print(f"samples_per_cell={audit['samples_per_cell']}")
    print("working_set,q1_ns,median_ns,q3_ns")
    for name in sorted(audit["summaries"]):
        q1, median, q3 = audit["summaries"][name]
        print(f"{name},{q1:.9f},{median:.9f},{q3:.9f}")
        odd_to_even, even_to_odd, gate = audit["validations"][name]
        print(
            f"{name}_odd_to_even_candidate_ns={odd_to_even[0]:.9f} "
            f"mape_pct={odd_to_even[1]:.9f} max_ape_pct={odd_to_even[2]:.9f}")
        print(
            f"{name}_even_to_odd_candidate_ns={even_to_odd[0]:.9f} "
            f"mape_pct={even_to_odd[1]:.9f} max_ape_pct={even_to_odd[2]:.9f}")
        print(f"{name}_replication_gate={'pass' if gate else 'fail'}")
    print(
        "hierarchy_iqr_separation_gate="
        f"{'pass' if audit['hierarchy_separated'] else 'fail'}")
    print(
        "publication_gate="
        f"{'pass' if audit['publication_gate'] else 'fail'}")
    if audit["candidates"] is None:
        print("latency_candidates_ns=unavailable")
    else:
        print(
            "latency_candidates_ns="
            f"l2:{audit['candidates']['l2']:.9f},"
            f"dram:{audit['candidates']['dram']:.9f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    _print_report(_audit(_load_rows(args.raw_csv)))


if __name__ == "__main__":
    main()
