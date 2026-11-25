#!/usr/bin/env python3
"""Audit repeated 48-AIV ReshapeAndCache equal-chain measurements."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from reshape_cache_equal_repeated_chain_runner import (
    BLOCK_DIM,
    CELLS,
    DEFAULT_PAYLOAD_BYTES,
    DEFAULT_SCHEDULE_CYCLES,
    STRIDE_BYTES,
    SUPPORTED_PAYLOAD_BYTES,
    SYSTEM_COUNTER_TICK_NS,
)


REPLICATION_RELATIVE_LIMIT = 0.20
INDEPENDENT_RUN_RELATIVE_LIMIT = 0.20
REQUIRED_INDEPENDENT_RUNS = 3
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "payload_bytes", "repeat_count", "pair_order",
    "stride_bytes", "control_cycles", "request_cycles", "control_ns",
    "request_ns", "sink", "key_destination_head",
    "value_destination_head", "validation",
)


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0 else abs(left - right) / scale


def _load_rows(paths, payload_bytes):
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
                        "block_dim", "worker", "payload_bytes",
                        "repeat_count", "pair_order", "stride_bytes",
                        "control_cycles", "request_cycles", "sink",
                        "key_destination_head", "value_destination_head",
                    )
                }
                cell = (integers["repeat_count"], integers["pair_order"])
                control_ns = float(raw["control_ns"])
                request_ns = float(raw["request_ns"])
                if (integers["trial"] <= 0 or integers["process_pid"] <= 0
                        or cell not in CELLS
                        or integers["block_dim"] != BLOCK_DIM
                        or not 0 <= integers["worker"] < BLOCK_DIM
                        or integers["payload_bytes"] != payload_bytes
                        or integers["stride_bytes"] != STRIDE_BYTES
                        or integers["control_cycles"] < 0
                        or integers["request_cycles"] <= 0
                        or integers["sink"] != 3
                        or integers["key_destination_head"] != 0x3C00
                        or integers["value_destination_head"] != 0x4000
                        or not math.isfinite(control_ns)
                        or not math.isfinite(request_ns)
                        or control_ns != integers["control_cycles"] * (
                            SYSTEM_COUNTER_TICK_NS)
                        or request_ns != integers["request_cycles"] * (
                            SYSTEM_COUNTER_TICK_NS)):
                    raise ValueError(f"{path} contains invalid values")
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
                    "worker": integers["worker"],
                    "repeat_count": integers["repeat_count"],
                    "pair_order": integers["pair_order"],
                    "control_ns": control_ns,
                    "request_ns": request_ns,
                })
    if not rows:
        raise ValueError("no repeated-chain rows were provided")
    return rows


def _audit_one_run(rows):
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one report must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    expected_trials = DEFAULT_SCHEDULE_CYCLES * len(CELLS)
    if trials != list(range(1, expected_trials + 1)):
        raise ValueError("run changed the frozen trial schedule")
    samples = {repeat_count: [] for repeat_count, _ in CELLS}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        if ({(row["repeat_count"], row["pair_order"])
             for row in trial_rows} != set(CELLS)):
            raise ValueError("each trial changed the frozen cell set")
        for repeat_count in samples:
            intervals = {}
            for pair_order in (0, 1):
                cell_rows = [
                    row for row in trial_rows
                    if row["repeat_count"] == repeat_count
                    and row["pair_order"] == pair_order]
                if (len(cell_rows) != BLOCK_DIM
                        or {row["worker"] for row in cell_rows}
                        != set(range(BLOCK_DIM))):
                    raise ValueError("each cell must contain every worker")
                intervals[pair_order] = {
                    row["worker"]: row for row in cell_rows}
            samples[repeat_count].append(max(
                intervals[1][worker]["request_ns"]
                - intervals[0][worker]["control_ns"]
                for worker in range(BLOCK_DIM)))
    services = {}
    for repeat_count, values in samples.items():
        q1, median, q3 = statistics.quantiles(
            values, n=4, method="inclusive")
        replica_a = values[::2]
        replica_b = values[1::2]
        spread = _relative_difference(
            statistics.median(replica_a), statistics.median(replica_b))
        services[repeat_count] = {
            "q1_ns": q1,
            "median_ns": median,
            "q3_ns": q3,
            "relative_iqr": (q3 - q1) / median,
            "replica_spread": spread,
            "gate": q1 > 0 and spread <= REPLICATION_RELATIVE_LIMIT,
        }
    return next(iter(run_ids)), services


def audit(paths, payload_bytes=DEFAULT_PAYLOAD_BYTES):
    if payload_bytes not in SUPPORTED_PAYLOAD_BYTES:
        raise ValueError("unsupported repeated-chain payload size")
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} runs")
    runs = tuple(
        _audit_one_run(_load_rows((path,), payload_bytes))
        for path in paths)
    run_ids = tuple(run_id for run_id, _ in runs)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("raw files must have distinct run_id values")
    services = {}
    for repeat_count in sorted(runs[0][1]):
        medians = tuple(run[repeat_count]["median_ns"] for _, run in runs)
        cross_run_spread = _relative_difference(min(medians), max(medians))
        services[repeat_count] = {
            "run_medians_ns": medians,
            "service_time_ns": statistics.median(medians),
            "cross_run_spread": cross_run_spread,
            "gate": (
                all(run[repeat_count]["gate"] for _, run in runs)
                and cross_run_spread <= INDEPENDENT_RUN_RELATIVE_LIMIT),
        }
    return {
        "run_ids": run_ids,
        "services": services,
        "gate": all(service["gate"] for service in services.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payload-bytes", type=int, choices=SUPPORTED_PAYLOAD_BYTES,
        default=DEFAULT_PAYLOAD_BYTES)
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    result = audit(args.raw_csv, args.payload_bytes)
    print("repeat_count,service_time_ns,cross_run_spread,gate")
    for repeat_count, service in result["services"].items():
        print(
            f"{repeat_count},{service['service_time_ns']:.9f},"
            f"{service['cross_run_spread']:.9f},"
            f"{'pass' if service['gate'] else 'fail'}")
    print(f"mechanism_gate={'pass' if result['gate'] else 'fail'}")
    if not result["gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
