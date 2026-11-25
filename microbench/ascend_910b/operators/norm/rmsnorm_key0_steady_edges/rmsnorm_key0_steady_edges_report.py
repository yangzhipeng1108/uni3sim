#!/usr/bin/env python3
"""Audit three RMSNorm key-0 warmed repeated-row edge captures."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from rmsnorm_key0_steady_edges_runner import (
    BASE_CELLS,
    BLOCK_DIMS,
    CELLS,
    DEFAULT_TRIALS,
    MODE_NAMES,
    MODE_REQUEST_COUNTS,
    PAIR_ORDERS,
    PAYLOAD_BYTES,
    REQUIRED_INDEPENDENT_RUNS,
    STRIDE_BYTES,
    SYSTEM_COUNTER_TICK_NS,
    _cell_index,
)


RELATIVE_SPREAD_LIMIT = 0.20
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "mode", "payload_bytes", "request_count", "pair_order",
    "stride_bytes", "control_cycles", "target_cycles", "control_ns",
    "target_ns", "sink", "destination_head", "validation",
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
            try:
                values = {
                    name: int(raw[name])
                    for name in REQUIRED_FIELDS[1:13]
                }
                control_ns = float(raw["control_ns"])
                target_ns = float(raw["target_ns"])
                sink = int(raw["sink"])
                destination_head = int(raw["destination_head"])
            except ValueError as exc:
                raise ValueError(f"{path} contains non-numeric data") from exc
            cell = (
                values["block_dim"], values["mode"],
                values["payload_bytes"], values["pair_order"])
            identity = (
                raw["run_id"], values["trial"], *cell, values["worker"])
            mode = values["mode"]
            expected_count = MODE_REQUEST_COUNTS[mode] if mode in range(4) else -1
            expected_sink = expected_count + 1 if mode < 2 else expected_count
            expected_destination = 0 if mode < 2 else 0x3C00
            if (cell not in CELLS or values["trial"] <= 0
                    or values["order_position"] not in range(len(CELLS))
                    or values["process_pid"] <= 0
                    or values["worker"] not in range(values["block_dim"])
                    or values["payload_bytes"] not in PAYLOAD_BYTES
                    or values["request_count"] != expected_count
                    or values["stride_bytes"] != STRIDE_BYTES
                    or values["control_cycles"] < 0
                    or values["target_cycles"] <= 0
                    or sink != expected_sink
                    or destination_head != expected_destination
                    or not math.isclose(
                        control_ns,
                        values["control_cycles"] * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)
                    or not math.isclose(
                        target_ns,
                        values["target_cycles"] * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)
                    or identity in identities):
                raise ValueError(f"{path} contains invalid row {identity}")
            identities.add(identity)
            rows.append({
                "run_id": raw["run_id"],
                **values,
                "incremental_ns_per_edge": (
                    target_ns - control_ns) / expected_count,
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
    if trials != list(range(1, DEFAULT_TRIALS + 1)):
        raise ValueError("raw run changed the frozen balanced schedule")
    by_trial = {}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        cells = {
            (row["block_dim"], row["mode"], row["payload_bytes"],
             row["pair_order"])
            for row in trial_rows}
        if cells != set(CELLS):
            raise ValueError("each trial must contain every frozen cell")
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        expected_order = tuple(
            CELLS[_cell_index(trial - 1, position)]
            for position in range(len(CELLS)))
        intervals = {}
        for position, cell in enumerate(expected_order):
            cell_rows = [
                row for row in trial_rows
                if (row["block_dim"], row["mode"], row["payload_bytes"],
                    row["pair_order"]) == cell]
            if (len(cell_rows) != cell[0]
                    or {row["worker"] for row in cell_rows}
                    != set(range(cell[0]))
                    or {row["order_position"] for row in cell_rows}
                    != {position}):
                raise ValueError("cell order or worker coverage changed")
            intervals[cell] = max(
                row["incremental_ns_per_edge"] for row in cell_rows)
        by_trial[trial] = intervals

    services = {}
    for base_cell in BASE_CELLS:
        samples_by_order = {
            pair_order: tuple(
                by_trial[trial][(*base_cell, pair_order)] for trial in trials)
            for pair_order in PAIR_ORDERS
        }
        samples = tuple(
            sample for order in PAIR_ORDERS
            for sample in samples_by_order[order])
        q1, median, q3 = _quartiles(samples)
        order_medians = tuple(
            statistics.median(samples_by_order[order])
            for order in PAIR_ORDERS)
        within_spread = math.inf if median <= 0 else (q3 - q1) / median
        services[base_cell] = {
            "median": median,
            "positive_gate": q1 > 0,
            "within_run_relative_iqr": within_spread,
            "within_run_gate": within_spread <= RELATIVE_SPREAD_LIMIT,
            "pair_order_relative_median_spread": _relative_spread(
                order_medians),
            "pair_order_gate": (
                _relative_spread(order_medians) <= RELATIVE_SPREAD_LIMIT),
        }
    return {
        "run_id": next(iter(run_ids)),
        "samples_per_pair_order": len(trials),
        "services": services,
        "mechanism_gate": all(
            value["positive_gate"] and value["within_run_gate"]
            and value["pair_order_gate"]
            for value in services.values()),
    }


def _audit_runs(paths):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} files")
    audits = tuple(_audit_run(path) for path in paths)
    run_ids = tuple(audit["run_id"] for audit in audits)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("independent files must use distinct run IDs")
    if not all(audit["mechanism_gate"] for audit in audits):
        raise ValueError("at least one run failed its within-run gate")
    services = {}
    for cell in BASE_CELLS:
        medians = tuple(audit["services"][cell]["median"] for audit in audits)
        spread = _relative_spread(medians)
        services[cell] = {
            "medians_by_run": dict(zip(run_ids, medians)),
            "service_time_ns_per_edge": statistics.median(medians),
            "cross_run_relative_median_spread": spread,
            "cross_run_gate": spread <= RELATIVE_SPREAD_LIMIT,
        }
    return {
        "run_ids": run_ids,
        "samples_per_pair_order_per_run": audits[0]["samples_per_pair_order"],
        "services": services,
        "mechanism_gate": all(
            value["cross_run_gate"] for value in services.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    audit = _audit_runs(args.raw_csv)
    print("# Ascend 910B RMSNorm key-0 warmed repeated-row edge audit")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(
        "samples_per_pair_order_per_run="
        f"{audit['samples_per_pair_order_per_run']}")
    print(
        "block_dim,mode,request_count,payload_bytes,"
        "service_time_ns_per_edge,cross_run_relative_median_spread,stability")
    for (block_dim, mode, payload_bytes), service in audit["services"].items():
        print(
            f"{block_dim},{MODE_NAMES[mode]},{MODE_REQUEST_COUNTS[mode]},"
            f"{payload_bytes},{service['service_time_ns_per_edge']:.9f},"
            f"{service['cross_run_relative_median_spread']:.9f},"
            f"{'pass' if service['cross_run_gate'] else 'fail'}")
    print(f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")
    if not audit["mechanism_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
