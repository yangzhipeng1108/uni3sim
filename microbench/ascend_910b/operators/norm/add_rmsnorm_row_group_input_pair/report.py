#!/usr/bin/env python3
"""Validate retained key-30 row input-pair completion cells."""

import argparse
import csv
import math
from pathlib import Path
import statistics


SYSTEM_COUNTER_TICK_NS = 20.0
REPLICATION_RELATIVE_LIMIT = 0.20
INDEPENDENT_RUN_RELATIVE_LIMIT = 0.20
REQUIRED_INDEPENDENT_RUNS = 3
BLOCK_DIM = 32
PAYLOAD_BYTES = (4096, 5120, 8192, 10240, 16384)
PAIR_ORDERS = (0, 1)
CELLS = tuple(
    (BLOCK_DIM, payload_bytes, pair_order)
    for payload_bytes in PAYLOAD_BYTES
    for pair_order in PAIR_ORDERS
)
PUBLISHED_PAYLOAD_BYTES = (4096, 5120)
PUBLISHED_CELLS = tuple(
    (BLOCK_DIM, payload_bytes)
    for payload_bytes in PUBLISHED_PAYLOAD_BYTES
)
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "payload_bytes", "pair_order", "stride_bytes",
    "control_cycles", "request_cycles", "control_ns", "request_ns",
    "sink", "validation",
)


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0 else abs(left - right) / scale


def _cell_index(trial_index, position):
    return (position + 7 * trial_index) % len(CELLS)


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
                    "worker", "payload_bytes", "pair_order", "stride_bytes",
                    "control_cycles", "request_cycles", "sink")
            }
            cell = (
                integers["block_dim"], integers["payload_bytes"],
                integers["pair_order"])
            control_ns = float(raw["control_ns"])
            request_ns = float(raw["request_ns"])
            identity = (
                raw["run_id"], integers["trial"], *cell,
                integers["worker"])
            if (cell not in CELLS or integers["trial"] <= 0
                    or integers["process_pid"] <= 0
                    or integers["worker"] < 0
                    or integers["worker"] >= BLOCK_DIM
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
    if len(trials) % (2 * len(CELLS)) != 0:
        raise ValueError(
            "raw run must preserve pairs of complete schedule cycles")

    samples = {cell: [] for cell in PUBLISHED_CELLS}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        intervals = {}
        for position in range(len(CELLS)):
            expected = CELLS[_cell_index(trial - 1, position)]
            cell_rows = [
                row for row in trial_rows
                if row["order_position"] == position
            ]
            observed = {
                (row["block_dim"], row["payload_bytes"], row["pair_order"])
                for row in cell_rows
            }
            if observed != {expected}:
                raise ValueError(
                    "raw run does not preserve its full position-balanced "
                    "schedule")
            if (len(cell_rows) != BLOCK_DIM
                    or {row["worker"] for row in cell_rows}
                    != set(range(BLOCK_DIM))):
                raise ValueError("each scheduled cell must contain every worker")
            intervals[expected] = {
                row["worker"]: row for row in cell_rows}
        for block_dim, payload_bytes in PUBLISHED_CELLS:
            samples[(block_dim, payload_bytes)].append(max(
                intervals[(block_dim, payload_bytes, 1)][worker][
                    "request_ns"]
                - intervals[(block_dim, payload_bytes, 0)][worker][
                    "control_ns"]
                for worker in range(block_dim)))

    services = {}
    schedule_length = len(CELLS)
    for cell, cell_samples in samples.items():
        q1, _, q3 = statistics.quantiles(
            cell_samples, n=4, method="inclusive")
        median = statistics.median(cell_samples)
        first = tuple(
            sample for index, sample in enumerate(cell_samples)
            if (index // schedule_length) % 2 == 0)
        second = tuple(
            sample for index, sample in enumerate(cell_samples)
            if (index // schedule_length) % 2 == 1)
        first_median = statistics.median(first)
        second_median = statistics.median(second)
        reproducible = (
            q1 > 0
            and _relative_difference(first_median, second_median)
            <= REPLICATION_RELATIVE_LIMIT)
        services[cell] = {
            "q1": q1,
            "median": median,
            "q3": q3,
            "first_median": first_median,
            "second_median": second_median,
            "reproducible": reproducible,
        }
    return {
        "run_id": next(iter(run_ids)),
        "samples_per_cell": len(trials),
        "services": services,
    }


def _audit_runs(paths):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} files")
    audits = tuple(_audit_run(path) for path in paths)
    run_ids = tuple(audit["run_id"] for audit in audits)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("independent files must use distinct run_ids")
    if not all(
            service["reproducible"]
            for audit in audits for service in audit["services"].values()):
        raise ValueError(
            "a published cell failed a within-run replication gate")
    services = {}
    for cell in PUBLISHED_CELLS:
        medians = tuple(
            audit["services"][cell]["median"] for audit in audits)
        spread = _relative_difference(min(medians), max(medians))
        services[cell] = {
            "medians_by_run": dict(zip(run_ids, medians)),
            "service_time_ns": statistics.median(medians),
            "maximum_run_relative_median_spread": spread,
            "stability_gate": spread <= INDEPENDENT_RUN_RELATIVE_LIMIT,
        }
    return {
        "run_ids": run_ids,
        "samples_per_cell_per_run": audits[0]["samples_per_cell"],
        "audits": audits,
        "services": services,
        "mechanism_gate": all(
            service["stability_gate"] for service in services.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    audit = _audit_runs(args.raw_csv)
    print("# Ascend 910B AddRmsNorm key-30 input-pair audit")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(f"samples_per_cell_per_run={audit['samples_per_cell_per_run']}")
    for cell, service in audit["services"].items():
        block_dim, payload_bytes = cell
        for run in audit["audits"]:
            result = run["services"][cell]
            print(
                f"{run['run_id']}: block_dim={block_dim},"
                f"payload_bytes={payload_bytes},median={result['median']:.9f},"
                f"parity={result['first_median']:.9f}/"
                f"{result['second_median']:.9f},"
                "replication="
                f"{'pass' if result['reproducible'] else 'fail'}")
        print(
            f"block_dim={block_dim},payload_bytes={payload_bytes},"
            f"service_time_ns={service['service_time_ns']:.9f},"
            "maximum_run_relative_median_spread="
            f"{service['maximum_run_relative_median_spread']:.9f},"
            f"stability={'pass' if service['stability_gate'] else 'fail'}")
    print(
        f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")
    if not audit["mechanism_gate"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
