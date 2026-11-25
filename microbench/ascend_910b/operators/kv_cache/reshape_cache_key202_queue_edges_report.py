#!/usr/bin/env python3
"""Audit ReshapeAndCache key-202 two-slot queue-edge measurements."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from reshape_cache_key202_queue_edges_runner import (
    BLOCK_DIM,
    CELLS,
    DEFAULT_SCHEDULE_CYCLES,
    PAYLOAD_BYTES,
    STRIDE_BYTES,
    SYSTEM_COUNTER_TICK_NS,
)


REPLICATION_RELATIVE_LIMIT = 0.20
INDEPENDENT_RUN_RELATIVE_LIMIT = 0.20
REQUIRED_INDEPENDENT_RUNS = 3
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "payload_bytes", "pair_order", "stride_bytes",
    "control_cycles", "request_cycles", "control_ns", "request_ns",
    "sink", "validation",
)


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each payload requires at least four samples")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0 else abs(left - right) / scale


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
                        "trial", "order_position", "process_pid", "block_dim",
                        "worker", "payload_bytes", "pair_order",
                        "stride_bytes", "control_cycles", "request_cycles",
                        "sink",
                    )
                }
                cell = (
                    integers["payload_bytes"], integers["pair_order"])
                if (integers["trial"] <= 0 or integers["process_pid"] <= 0
                        or integers["block_dim"] != BLOCK_DIM
                        or cell not in CELLS or integers["worker"] < 0
                        or integers["worker"] >= BLOCK_DIM
                        or integers["stride_bytes"] != STRIDE_BYTES
                        or integers["control_cycles"] < 0
                        or integers["request_cycles"] <= 0
                        or integers["sink"] != 6):
                    raise ValueError(f"{path} contains invalid values")
                control_ns = float(raw["control_ns"])
                request_ns = float(raw["request_ns"])
                if (not math.isfinite(control_ns) or control_ns < 0
                        or not math.isfinite(request_ns) or request_ns <= 0
                        or not math.isclose(
                            control_ns,
                            integers["control_cycles"]
                            * SYSTEM_COUNTER_TICK_NS,
                            rel_tol=0.0, abs_tol=1e-9)
                        or not math.isclose(
                            request_ns,
                            integers["request_cycles"]
                            * SYSTEM_COUNTER_TICK_NS,
                            rel_tol=0.0, abs_tol=1e-9)):
                    raise ValueError(f"{path} has inconsistent elapsed time")
                identity = (
                    raw["run_id"], integers["trial"], *cell,
                    integers["worker"])
                if identity in identities:
                    raise ValueError(f"duplicate row identity {identity}")
                identities.add(identity)
                rows.append({
                    "run_id": raw["run_id"],
                    "trial": integers["trial"],
                    "order_position": integers["order_position"],
                    "process_pid": integers["process_pid"],
                    "worker": integers["worker"],
                    "payload_bytes": integers["payload_bytes"],
                    "pair_order": integers["pair_order"],
                    "control_ns": control_ns,
                    "request_ns": request_ns,
                })
    if not rows:
        raise ValueError("no queue-edge rows were provided")
    return rows


def _audit(rows):
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one report must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    expected_trials = DEFAULT_SCHEDULE_CYCLES * len(CELLS)
    if trials != list(range(1, expected_trials + 1)):
        raise ValueError(
            f"run must contain the frozen {expected_trials}-trial schedule")
    by_trial = {}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        observed_cells = {
            (row["payload_bytes"], row["pair_order"])
            for row in trial_rows}
        if observed_cells != set(CELLS):
            raise ValueError("each trial changed the frozen cell set")
        if ({row["order_position"] for row in trial_rows}
                != set(range(len(CELLS)))):
            raise ValueError("each trial must contain every order position")
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        intervals = {}
        for cell in CELLS:
            cell_rows = [
                row for row in trial_rows
                if (row["payload_bytes"], row["pair_order"]) == cell]
            if (len(cell_rows) != BLOCK_DIM
                    or {row["worker"] for row in cell_rows}
                    != set(range(BLOCK_DIM))
                    or len({row["order_position"] for row in cell_rows}) != 1):
                raise ValueError("each cell must contain every worker once")
            intervals[cell] = {
                row["worker"]: {
                    "control_ns": row["control_ns"],
                    "request_ns": row["request_ns"],
                }
                for row in cell_rows
            }
        by_trial[trial] = intervals
    expected_per_position = len(trials) // len(CELLS)
    for position in range(len(CELLS)):
        counts = {
            cell: sum(
                row["order_position"] == position and row["worker"] == 0
                and (row["payload_bytes"], row["pair_order"]) == cell
                for row in rows)
            for cell in CELLS
        }
        if set(counts.values()) != {expected_per_position}:
            raise ValueError("cell/order schedule is not balanced")

    services = {}
    for payload_bytes in PAYLOAD_BYTES:
        samples = tuple(
            max(
                intervals[(payload_bytes, 1)][worker]["request_ns"]
                - intervals[(payload_bytes, 0)][worker]["control_ns"]
                for worker in range(BLOCK_DIM))
            for intervals in by_trial.values())
        q1, median, q3 = _quartiles(samples)
        first_replica = tuple(
            sample for index, sample in enumerate(samples)
            if (index // len(CELLS)) % 2 == 0)
        second_replica = tuple(
            sample for index, sample in enumerate(samples)
            if (index // len(CELLS)) % 2 == 1)
        replica_spread = _relative_difference(
            statistics.median(first_replica),
            statistics.median(second_replica))
        services[payload_bytes] = {
            "summary": (q1, median, q3),
            "replica_relative_spread": replica_spread,
            "mechanism_gate": (
                q1 > 0
                and replica_spread <= REPLICATION_RELATIVE_LIMIT),
        }
    return {
        "run_id": next(iter(run_ids)),
        "samples_per_payload": len(trials),
        "services": services,
        "mechanism_gate": all(
            service["mechanism_gate"] for service in services.values()),
    }


def _audit_independent_runs(paths):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} raw files")
    audits = tuple(_audit(_load_rows((path,))) for path in paths)
    run_ids = tuple(audit["run_id"] for audit in audits)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("raw files must have distinct run_id values")
    if not all(audit["mechanism_gate"] for audit in audits):
        raise ValueError("at least one run failed its mechanism gate")
    services = {}
    for payload_bytes in PAYLOAD_BYTES:
        medians = tuple(
            audit["services"][payload_bytes]["summary"][1]
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
        "samples_per_payload_per_run": audits[0]["samples_per_payload"],
        "services": services,
        "mechanism_gate": all(
            service["stability_gate"] for service in services.values()),
    }


def _print_report(audit):
    print("# Ascend 910B ReshapeAndCache key-202 queue-edge audit")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(
        "samples_per_payload_per_run="
        f"{audit['samples_per_payload_per_run']}")
    print(
        "block_dim,payload_bytes,service_time_ns,"
        "maximum_run_relative_median_spread,stability")
    for payload_bytes, service in audit["services"].items():
        stability = "pass" if service["stability_gate"] else "fail"
        print(
            f"{BLOCK_DIM},{payload_bytes},{service['service_time_ns']:.9f},"
            f"{service['maximum_run_relative_median_spread']:.9f},"
            f"{stability}")
    print(f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    audit = _audit_independent_runs(args.raw_csv)
    _print_report(audit)
    if not audit["mechanism_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
