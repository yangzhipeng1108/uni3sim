#!/usr/bin/env python3
"""Audit paired ReshapeAndCache equal-head copy-chain measurements."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from reshape_cache_equal_chain_runner import (
    CAMPAIGN_CELLS,
    CELLS,
    DEFAULT_CAMPAIGN,
    DEFAULT_SCHEDULE_CYCLES,
    PAIR_ORDERS,
)


SYSTEM_COUNTER_TICK_NS = 20.0
REPLICATION_RELATIVE_LIMIT = 0.20
INDEPENDENT_RUN_RELATIVE_LIMIT = 0.20
REQUIRED_INDEPENDENT_RUNS = 3
LEGACY_REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "payload_bytes", "pair_order", "stride_bytes",
    "control_cycles", "request_cycles", "control_ns", "request_ns",
    "sink", "key_destination_head", "value_destination_head",
    "validation",
)
REQUIRED_FIELDS = LEGACY_REQUIRED_FIELDS + ("warmup_launches",)


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each cell requires at least four samples")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _load_rows(paths, cells=CELLS, expected_warmup_launches=0):
    rows = []
    identities = set()
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            if fields not in (LEGACY_REQUIRED_FIELDS, REQUIRED_FIELDS):
                raise ValueError(f"{path} has an invalid raw schema")
            if (fields == LEGACY_REQUIRED_FIELDS
                    and expected_warmup_launches != 0):
                raise ValueError(
                    f"{path} does not record the required warmup launches")
            for raw in reader:
                if raw["validation"] != "pass":
                    raise ValueError(f"{path} contains failed validation")
                integers = {
                    name: int(raw[name])
                    for name in (
                        "trial", "order_position", "process_pid",
                        "block_dim", "worker", "payload_bytes",
                        "pair_order", "stride_bytes", "control_cycles",
                        "request_cycles", "sink", "key_destination_head",
                        "value_destination_head",
                    )
                }
                warmup_launches = (
                    0 if fields == LEGACY_REQUIRED_FIELDS
                    else int(raw["warmup_launches"]))
                cell = (
                    integers["block_dim"], integers["payload_bytes"],
                    integers["pair_order"])
                if (integers["trial"] <= 0 or integers["process_pid"] <= 0
                        or cell not in cells or integers["worker"] < 0
                        or integers["worker"] >= integers["block_dim"]
                        or warmup_launches != expected_warmup_launches
                        or integers["stride_bytes"] < integers["payload_bytes"]
                        or integers["stride_bytes"] % 32 != 0
                        or integers["control_cycles"] < 0
                        or integers["request_cycles"] <= 0
                        or integers["sink"] != 2
                        or integers["key_destination_head"] != 0x3C00
                        or integers["value_destination_head"] != 0x4000):
                    raise ValueError(f"{path} contains invalid values")
                control_ns = float(raw["control_ns"])
                request_ns = float(raw["request_ns"])
                if (not math.isfinite(control_ns) or control_ns < 0
                        or not math.isfinite(request_ns) or request_ns <= 0
                        or not math.isclose(
                            control_ns,
                            integers["control_cycles"] * SYSTEM_COUNTER_TICK_NS,
                            rel_tol=0.0, abs_tol=1e-9)
                        or not math.isclose(
                            request_ns,
                            integers["request_cycles"] * SYSTEM_COUNTER_TICK_NS,
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
                    "block_dim": integers["block_dim"],
                    "worker": integers["worker"],
                    "payload_bytes": integers["payload_bytes"],
                    "pair_order": integers["pair_order"],
                    "control_ns": control_ns,
                    "request_ns": request_ns,
                })
    if not rows:
        raise ValueError("no copy-chain rows were provided")
    return rows


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0 else abs(left - right) / scale


def _audit(rows, cells=CELLS):
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one report must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    expected_trials = DEFAULT_SCHEDULE_CYCLES * len(cells)
    if trials != list(range(1, expected_trials + 1)):
        raise ValueError(
            f"run must contain the frozen {expected_trials}-trial schedule")
    by_trial = {}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        observed_cells = {
            (row["block_dim"], row["payload_bytes"], row["pair_order"])
            for row in trial_rows}
        if observed_cells != set(cells):
            raise ValueError("each trial changed the frozen cell set")
        if ({row["order_position"] for row in trial_rows}
                != set(range(len(cells)))):
            raise ValueError("each trial must contain every order position")
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        intervals = {}
        for cell in cells:
            block_dim, payload_bytes, pair_order = cell
            cell_rows = [
                row for row in trial_rows
                if (row["block_dim"], row["payload_bytes"],
                    row["pair_order"]) == cell]
            if (len(cell_rows) != block_dim
                    or {row["worker"] for row in cell_rows}
                    != set(range(block_dim))
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
    expected_per_position = len(trials) // len(cells)
    for position in range(len(cells)):
        counts = {
            cell: sum(
                row["order_position"] == position and row["worker"] == 0
                and (row["block_dim"], row["payload_bytes"],
                     row["pair_order"]) == cell
                for row in rows)
            for cell in cells
        }
        if set(counts.values()) != {expected_per_position}:
            raise ValueError("cell/order schedule is not balanced")

    services = {}
    for block_dim, payload_bytes in sorted({cell[:2] for cell in cells}):
        samples = tuple(
            max(
                intervals[(block_dim, payload_bytes, 1)][worker]["request_ns"]
                - intervals[(block_dim, payload_bytes, 0)][worker]["control_ns"]
                for worker in range(block_dim))
            for intervals in by_trial.values())
        q1, median, q3 = _quartiles(samples)
        first_replica = tuple(
            sample for index, sample in enumerate(samples)
            if (index // len(cells)) % 2 == 0)
        second_replica = tuple(
            sample for index, sample in enumerate(samples)
            if (index // len(cells)) % 2 == 1)
        replica_spread = _relative_difference(
            statistics.median(first_replica),
            statistics.median(second_replica))
        services[(block_dim, payload_bytes)] = {
            "summary": (q1, median, q3),
            "replica_spread": replica_spread,
            "mechanism_gate": (
                q1 > 0 and replica_spread <= REPLICATION_RELATIVE_LIMIT),
        }
    return {
        "run_id": next(iter(run_ids)),
        "samples_per_cell": len(trials),
        "services": services,
        "mechanism_gate": all(
            service["mechanism_gate"] for service in services.values()),
    }


def _audit_independent_runs(
        paths, cells=CELLS, expected_warmup_launches=0):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} raw files")
    audits = tuple(
        _audit(
            _load_rows((path,), cells, expected_warmup_launches), cells)
        for path in paths)
    run_ids = tuple(audit["run_id"] for audit in audits)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("raw files must have distinct run_id values")
    if not all(audit["mechanism_gate"] for audit in audits):
        raise ValueError("at least one run failed its mechanism gate")
    services = {}
    for cell in sorted(audits[0]["services"]):
        medians = tuple(
            audit["services"][cell]["summary"][1] for audit in audits)
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
        "services": services,
        "mechanism_gate": all(
            service["stability_gate"] for service in services.values()),
    }


def _print_report(audit):
    print("# Ascend 910B ReshapeAndCache equal-chain audit")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(f"samples_per_cell_per_run={audit['samples_per_cell_per_run']}")
    print(
        "block_dim,payload_bytes,service_time_ns,"
        "maximum_run_relative_median_spread,stability")
    for cell, service in audit["services"].items():
        block_dim, payload_bytes = cell
        stability = "pass" if service["stability_gate"] else "fail"
        print(
            f"{block_dim},{payload_bytes},{service['service_time_ns']:.9f},"
            f"{service['maximum_run_relative_median_spread']:.9f},"
            f"{stability}")
    print(f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign", choices=tuple(CAMPAIGN_CELLS),
        default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--warmup-launches", type=int, choices=(0, 4, 8, 16), default=0)
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    if args.campaign == "equal_midworkers_warm":
        if args.warmup_launches == 0:
            parser.error(
                "equal_midworkers_warm requires --warmup-launches 4, 8, or 16")
    elif args.warmup_launches != 0:
        parser.error("--warmup-launches is only valid for the warm campaign")
    audit = _audit_independent_runs(
        args.raw_csv, CAMPAIGN_CELLS[args.campaign], args.warmup_launches)
    _print_report(audit)
    if not audit["mechanism_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
