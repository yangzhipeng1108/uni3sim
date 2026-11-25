#!/usr/bin/env python3
"""Audit repeated key-202 K/V queue-pair measurements."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from reshape_cache_key202_repeated_pair_runner import (
    BLOCK_DIM,
    DEFAULT_SCHEDULE_CYCLES,
    PAIR_ORDERS,
    PAYLOAD_BYTES,
    REPEAT_COUNT,
    STRIDE_BYTES,
    SYSTEM_COUNTER_TICK_NS,
)


REPLICATION_RELATIVE_LIMIT = 0.20
INDEPENDENT_RUN_RELATIVE_LIMIT = 0.20
REQUIRED_INDEPENDENT_RUNS = 3
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "payload_bytes", "repeat_count", "pair_order", "stride_bytes",
    "control_cycles", "request_cycles", "control_ns", "request_ns", "sink",
    "key_head", "value_head", "validation",
)


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0 else abs(left - right) / scale


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
                    "worker", "payload_bytes", "repeat_count", "pair_order",
                    "stride_bytes", "control_cycles", "request_cycles", "sink",
                    "key_head", "value_head",
                )
            }
            control_ns = float(raw["control_ns"])
            request_ns = float(raw["request_ns"])
            if (integers["trial"] <= 0 or integers["process_pid"] <= 0
                    or integers["block_dim"] != BLOCK_DIM
                    or integers["worker"] not in range(BLOCK_DIM)
                    or integers["payload_bytes"] != PAYLOAD_BYTES
                    or integers["repeat_count"] != REPEAT_COUNT
                    or integers["pair_order"] not in PAIR_ORDERS
                    or integers["stride_bytes"] != STRIDE_BYTES
                    or integers["control_cycles"] < 0
                    or integers["request_cycles"] <= 0
                    or integers["sink"] != 3
                    or integers["key_head"] != 0x3c00
                    or integers["value_head"] != 0x4000
                    or not math.isclose(
                        control_ns,
                        integers["control_cycles"] * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)
                    or not math.isclose(
                        request_ns,
                        integers["request_cycles"] * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)):
                raise ValueError(f"{path} contains invalid values")
            identity = (
                raw["run_id"], integers["trial"],
                integers["pair_order"], integers["worker"])
            if identity in identities:
                raise ValueError(f"duplicate row identity {identity}")
            identities.add(identity)
            rows.append({
                "run_id": raw["run_id"],
                "trial": integers["trial"],
                "order_position": integers["order_position"],
                "process_pid": integers["process_pid"],
                "worker": integers["worker"],
                "pair_order": integers["pair_order"],
                "control_ns": control_ns,
                "request_ns": request_ns,
            })
    if not rows:
        raise ValueError("no repeated queue-pair rows were provided")
    return rows


def _audit_run(rows):
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one raw file must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    expected_trials = DEFAULT_SCHEDULE_CYCLES * len(PAIR_ORDERS)
    if trials != list(range(1, expected_trials + 1)):
        raise ValueError("run does not contain the frozen trial schedule")
    samples = []
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        if ({row["pair_order"] for row in trial_rows} != set(PAIR_ORDERS)
                or {row["order_position"] for row in trial_rows}
                != set(range(len(PAIR_ORDERS)))
                or len({row["process_pid"] for row in trial_rows}) != 1):
            raise ValueError("each trial must contain both balanced cells")
        by_cell = {}
        for pair_order in PAIR_ORDERS:
            cell_rows = [
                row for row in trial_rows
                if row["pair_order"] == pair_order]
            if (len(cell_rows) != BLOCK_DIM
                    or {row["worker"] for row in cell_rows}
                    != set(range(BLOCK_DIM))):
                raise ValueError("each cell must contain every worker once")
            by_cell[pair_order] = {row["worker"]: row for row in cell_rows}
        samples.append(max(
            by_cell[1][worker]["request_ns"]
            - by_cell[0][worker]["control_ns"]
            for worker in range(BLOCK_DIM)))
    first_replica = tuple(sample for index, sample in enumerate(samples)
                          if index % 2 == 0)
    second_replica = tuple(sample for index, sample in enumerate(samples)
                           if index % 2 == 1)
    q1, median, q3 = statistics.quantiles(
        samples, n=4, method="inclusive")
    replica_spread = _relative_difference(
        statistics.median(first_replica), statistics.median(second_replica))
    return {
        "run_id": next(iter(run_ids)),
        "summary": (q1, median, q3),
        "replica_relative_spread": replica_spread,
        "mechanism_gate": q1 > 0 and replica_spread <= REPLICATION_RELATIVE_LIMIT,
    }


def _audit_campaign(paths):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} raw files")
    audits = tuple(_audit_run(_load_rows(path)) for path in paths)
    if len({audit["run_id"] for audit in audits}) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("raw files must have distinct run_id values")
    if not all(audit["mechanism_gate"] for audit in audits):
        raise ValueError("at least one run failed its mechanism gate")
    medians = tuple(audit["summary"][1] for audit in audits)
    spread = _relative_difference(min(medians), max(medians))
    return {
        "audits": audits,
        "service_time_ns": statistics.median(medians),
        "cross_run_relative_median_spread": spread,
        "mechanism_gate": spread <= INDEPENDENT_RUN_RELATIVE_LIMIT,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    audit = _audit_campaign(args.raw_csv)
    print("# Ascend 910B ReshapeAndCache key-202 repeated-pair audit")
    print(f"block_dim={BLOCK_DIM}")
    print(f"payload_bytes={PAYLOAD_BYTES}")
    print(f"repeat_count={REPEAT_COUNT}")
    print("run_id,q1_ns,median_ns,q3_ns,replica_relative_spread,stability")
    for run in audit["audits"]:
        q1, median, q3 = run["summary"]
        print(
            f"{run['run_id']},{q1:.9f},{median:.9f},{q3:.9f},"
            f"{run['replica_relative_spread']:.9f},pass")
    print(f"service_time_ns={audit['service_time_ns']:.9f}")
    print(
        "cross_run_relative_median_spread="
        f"{audit['cross_run_relative_median_spread']:.9f}")
    print(f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")
    if not audit["mechanism_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
