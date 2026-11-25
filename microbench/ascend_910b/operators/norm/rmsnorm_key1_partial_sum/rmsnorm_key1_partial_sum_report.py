#!/usr/bin/env python3
"""Audit three RMSNorm key-1 partial-sum handoff captures."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from rmsnorm_key1_partial_sum_runner import (
    CELLS,
    PAIR_ORDERS,
    REQUIRED_INDEPENDENT_RUNS,
)


SYSTEM_COUNTER_TICK_NS = 20.0
WITHIN_RUN_RELATIVE_LIMIT = 0.10
CROSS_RUN_RELATIVE_LIMIT = 0.10
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "partial_count", "pair_order", "partial_stride_bytes",
    "group_payload_bytes", "iterations", "control_cycles",
    "target_cycles", "control_ns", "target_ns", "sink", "validation",
)


def _relative_spread(values):
    high = max(values)
    return 0.0 if high == 0 else (high - min(values)) / high


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each cell requires at least four samples")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _load_rows(path):
    rows = []
    identities = set()
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
                    "trial", "order_position", "process_pid", "block_dim",
                    "worker", "partial_count", "pair_order",
                    "partial_stride_bytes", "group_payload_bytes",
                    "iterations", "control_cycles", "target_cycles", "sink")
            }
            cell = (
                integers["block_dim"], integers["partial_count"],
                integers["pair_order"])
            control_ns = float(raw["control_ns"])
            target_ns = float(raw["target_ns"])
            identity = (
                raw["run_id"], integers["trial"], *cell,
                integers["worker"])
            if (cell not in CELLS or integers["trial"] <= 0
                    or integers["process_pid"] <= 0
                    or integers["worker"] < 0
                    or integers["worker"] >= integers["block_dim"]
                    or integers["partial_stride_bytes"] != 32
                    or integers["group_payload_bytes"]
                    != integers["partial_count"] * 32
                    or integers["iterations"] <= 0
                    or integers["control_cycles"] <= 0
                    or integers["target_cycles"] <= 0
                    or integers["sink"] != 4 * integers["partial_count"]
                    or not math.isclose(
                        control_ns,
                        integers["control_cycles"] * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)
                    or not math.isclose(
                        target_ns,
                        integers["target_cycles"] * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)
                    or identity in identities):
                raise ValueError(f"{path} contains invalid row {identity}")
            identities.add(identity)
            rows.append({
                "run_id": raw["run_id"],
                **integers,
                "incremental_ns": (
                    target_ns - control_ns) / integers["iterations"],
            })
    if not rows:
        raise ValueError(f"{path} contains no data")
    return rows


def _audit_run(path):
    rows = _load_rows(path)
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one raw file must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    if trials != list(range(1, len(trials) + 1)):
        raise ValueError("trial identities must be contiguous from one")
    if len(trials) % len(CELLS) != 0:
        raise ValueError("raw run does not contain complete schedule cycles")
    by_trial = {}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        observed_cells = {
            (row["block_dim"], row["partial_count"], row["pair_order"])
            for row in trial_rows}
        if observed_cells != set(CELLS):
            raise ValueError("each trial must contain the frozen cell set")
        if {row["order_position"] for row in trial_rows} != set(
                range(len(CELLS))):
            raise ValueError("each trial must contain every order position")
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        cells = {}
        for cell in CELLS:
            cell_rows = [
                row for row in trial_rows
                if (row["block_dim"], row["partial_count"],
                    row["pair_order"]) == cell]
            if (len(cell_rows) != cell[0]
                    or {row["worker"] for row in cell_rows}
                    != set(range(cell[0]))):
                raise ValueError("each cell must contain every worker")
            cells[cell] = cell_rows
        by_trial[trial] = cells

    services = {}
    base_cells = sorted({cell[:2] for cell in CELLS})
    if any(
            (*cell, order) not in CELLS
            for cell in base_cells for order in PAIR_ORDERS):
        raise ValueError("every mechanism cell requires both pair orders")
    for block_dim, partial_count in base_cells:
        samples_by_order = {
            pair_order: tuple(
                max(
                    row["incremental_ns"]
                    for row in by_trial[trial][
                        (block_dim, partial_count, pair_order)])
                for trial in trials)
            for pair_order in PAIR_ORDERS
        }
        samples = tuple(
            sample
            for pair_order in PAIR_ORDERS
            for sample in samples_by_order[pair_order])
        q1, median, q3 = _quartiles(samples)
        order_medians = tuple(
            statistics.median(samples_by_order[order])
            for order in PAIR_ORDERS)
        within_run_spread = (
            math.inf if median <= 0 else (q3 - q1) / median)
        services[(block_dim, partial_count)] = {
            "q1": q1,
            "median": median,
            "q3": q3,
            "pair_order_medians": order_medians,
            "within_run_relative_iqr": within_run_spread,
            "positive_signal_gate": q1 > 0,
            "pair_order_gate": (
                _relative_spread(order_medians)
                <= WITHIN_RUN_RELATIVE_LIMIT),
            "within_run_spread_gate": (
                within_run_spread <= WITHIN_RUN_RELATIVE_LIMIT),
        }
    return {
        "run_id": next(iter(run_ids)),
        "samples_per_pair_order": len(trials),
        "services": services,
        "mechanism_gate": all(
            result["positive_signal_gate"]
            and result["pair_order_gate"]
            and result["within_run_spread_gate"]
            for result in services.values()),
    }


def _audit_runs(paths):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} files")
    audits = tuple(_audit_run(path) for path in paths)
    run_ids = tuple(audit["run_id"] for audit in audits)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("independent files must use distinct run_ids")
    if not all(audit["mechanism_gate"] for audit in audits):
        raise ValueError("at least one run failed its within-run gate")
    services = {}
    for cell in sorted(audits[0]["services"]):
        medians = tuple(
            audit["services"][cell]["median"] for audit in audits)
        cross_run_spread = _relative_spread(medians)
        services[cell] = {
            "medians_by_run": dict(zip(run_ids, medians)),
            "service_time_ns": statistics.median(medians),
            "maximum_cross_run_relative_median_spread": cross_run_spread,
            "cross_run_spread_gate": (
                cross_run_spread <= CROSS_RUN_RELATIVE_LIMIT),
        }
    return {
        "run_ids": run_ids,
        "samples_per_pair_order_per_run": audits[0][
            "samples_per_pair_order"],
        "services": services,
        "mechanism_gate": all(
            result["cross_run_spread_gate"]
            for result in services.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    audit = _audit_runs(args.raw_csv)
    print("# Ascend 910B RMSNorm key-1 partial-sum handoff audit")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(
        "samples_per_pair_order_per_run="
        f"{audit['samples_per_pair_order_per_run']}")
    print(
        "block_dim,partial_count,service_time_ns,"
        "maximum_cross_run_relative_median_spread,stability")
    for cell, result in audit["services"].items():
        print(
            f"{cell[0]},{cell[1]},{result['service_time_ns']:.9f},"
            f"{result['maximum_cross_run_relative_median_spread']:.9f},"
            f"{'pass' if result['cross_run_spread_gate'] else 'fail'}")
    print(
        f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")
    if not audit["mechanism_gate"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
