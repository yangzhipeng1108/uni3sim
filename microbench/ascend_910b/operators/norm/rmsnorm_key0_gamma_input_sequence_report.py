#!/usr/bin/env python3
"""Validate RMSNorm key-0 serial gamma/input completion runs."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from rmsnorm_key0_gamma_input_sequence_runner import (
    CELLS,
    PAIR_ORDERS,
    REQUIRED_INDEPENDENT_RUNS,
)


SYSTEM_COUNTER_TICK_NS = 20.0
REPLICATION_RELATIVE_LIMIT = 0.20
INDEPENDENT_RUN_RELATIVE_LIMIT = 0.20
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "payload_bytes",
    "pair_order", "stride_bytes", "control_cycles", "request_cycles",
    "control_ns", "request_ns", "sink", "validation",
)


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0 else abs(left - right) / scale


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
                    "trial", "order_position", "process_pid",
                    "payload_bytes", "pair_order", "stride_bytes",
                    "control_cycles", "request_cycles", "sink")
            }
            cell = (integers["payload_bytes"], integers["pair_order"])
            control_ns = float(raw["control_ns"])
            request_ns = float(raw["request_ns"])
            identity = (raw["run_id"], integers["trial"], *cell)
            if (cell not in CELLS or integers["trial"] <= 0
                    or integers["process_pid"] <= 0
                    or integers["stride_bytes"] < integers["payload_bytes"]
                    or integers["control_cycles"] < 0
                    or integers["request_cycles"] <= 0
                    or integers["sink"] != 0xB4
                    or not math.isclose(
                        control_ns,
                        integers["control_cycles"] * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)
                    or not math.isclose(
                        request_ns,
                        integers["request_cycles"] * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)
                    or identity in identities):
                raise ValueError(f"{path} contains invalid row {identity}")
            identities.add(identity)
            rows.append({
                "run_id": raw["run_id"],
                **integers,
                "control_ns": control_ns,
                "request_ns": request_ns,
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
            (row["payload_bytes"], row["pair_order"])
            for row in trial_rows}
        if observed_cells != set(CELLS):
            raise ValueError("each trial must contain the frozen cell set")
        if {row["order_position"] for row in trial_rows} != set(
                range(len(CELLS))):
            raise ValueError("each trial must contain every order position")
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        by_trial[trial] = {
            (row["payload_bytes"], row["pair_order"]): row
            for row in trial_rows}

    services = {}
    payloads = sorted({cell[0] for cell in CELLS})
    if any(
            (payload_bytes, order) not in CELLS
            for payload_bytes in payloads for order in PAIR_ORDERS):
        raise ValueError("every request cell requires both pair orders")
    for payload_bytes in payloads:
        samples = tuple(
            by_trial[trial][(payload_bytes, 1)]["request_ns"]
            - by_trial[trial][(payload_bytes, 0)]["control_ns"]
            for trial in trials)
        q1, median, q3 = _quartiles(samples)
        schedule_length = len(CELLS)
        first = tuple(
            sample for index, sample in enumerate(samples)
            if (index // schedule_length) % 2 == 0)
        second = tuple(
            sample for index, sample in enumerate(samples)
            if (index // schedule_length) % 2 == 1)
        first_median = statistics.median(first)
        second_median = statistics.median(second)
        services[payload_bytes] = {
            "q1": q1,
            "median": median,
            "q3": q3,
            "first_median": first_median,
            "second_median": second_median,
            "reproducible": (
                q1 > 0
                and _relative_difference(first_median, second_median)
                <= REPLICATION_RELATIVE_LIMIT),
        }
    return {
        "run_id": next(iter(run_ids)),
        "samples_per_cell": len(trials),
        "services": services,
        "mechanism_gate": all(
            result["reproducible"] for result in services.values()),
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
        raise ValueError("at least one run failed its replication gate")
    services = {}
    for payload_bytes in sorted(audits[0]["services"]):
        medians = tuple(
            audit["services"][payload_bytes]["median"]
            for audit in audits)
        spread = _relative_difference(min(medians), max(medians))
        services[payload_bytes] = {
            "medians_by_run": dict(zip(run_ids, medians)),
            "service_time_ns": statistics.median(medians),
            "maximum_run_relative_median_spread": spread,
            "stability_gate": spread <= INDEPENDENT_RUN_RELATIVE_LIMIT,
        }
    return {
        "run_ids": run_ids,
        "samples_per_cell_per_run": audits[0]["samples_per_cell"],
        "services": services,
        "mechanism_gate": all(
            result["stability_gate"] for result in services.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    audit = _audit_runs(args.raw_csv)
    print("# Ascend 910B RMSNorm key-0 gamma/input sequence audit")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(f"samples_per_cell_per_run={audit['samples_per_cell_per_run']}")
    print(
        "payload_bytes,service_time_ns,"
        "maximum_run_relative_median_spread,stability")
    for payload_bytes, result in audit["services"].items():
        print(
            f"{payload_bytes},{result['service_time_ns']:.9f},"
            f"{result['maximum_run_relative_median_spread']:.9f},"
            f"{'pass' if result['stability_gate'] else 'fail'}")
    print(
        f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")
    if not audit["mechanism_gate"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
