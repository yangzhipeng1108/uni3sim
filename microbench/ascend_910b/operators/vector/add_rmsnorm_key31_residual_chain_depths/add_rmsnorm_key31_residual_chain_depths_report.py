#!/usr/bin/env python3
"""Audit exact AddRMSNorm key-31 residual-chain depth captures."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from add_rmsnorm_key31_residual_chain_depths_runner import (
    DEFAULT_TRIALS,
    PAIR_ORDERS,
    REQUEST_BYTES,
    REQUIRED_INDEPENDENT_RUNS,
    SOURCE_SHA256,
    SYSTEM_COUNTER_TICK_NS,
    TILES_PER_ROW,
    _ordered_pair_orders,
    _sha256,
)


RELATIVE_LIMIT = 0.20
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "pair_order", "request_count", "request_bytes",
    "control_cycles", "request_cycles", "control_ns", "request_ns",
    "control_sink", "request_sink", "validation",
)
CELL_CONTRACTS = {
    2: {"block_dim": 1, "block_factor": 1, "last_block_factor": 1},
    4: {"block_dim": 32, "block_factor": 2, "last_block_factor": 2},
    12: {"block_dim": 43, "block_factor": 6, "last_block_factor": 4},
}


def _relative_spread(values):
    high = max(values)
    return math.inf if high <= 0 else (high - min(values)) / high


def _load_run(path, maximum_requests):
    contract = CELL_CONTRACTS[maximum_requests]
    rows = []
    identities = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError(f"{path} has an invalid schema")
        for raw in reader:
            try:
                values = {
                    name: int(raw[name]) for name in REQUIRED_FIELDS[1:11]}
                control_ns = float(raw["control_ns"])
                request_ns = float(raw["request_ns"])
                control_sink = int(raw["control_sink"])
                request_sink = int(raw["request_sink"])
            except ValueError as error:
                raise ValueError(
                    f"{path} contains non-numeric data") from error
            worker = values["worker"]
            expected_count = (
                contract["last_block_factor"] * TILES_PER_ROW
                if worker == contract["block_dim"] - 1
                else contract["block_factor"] * TILES_PER_ROW)
            identity = (
                raw["run_id"], values["trial"], values["pair_order"],
                worker)
            if (raw["validation"] != "pass" or identity in identities
                    or values["block_dim"] != contract["block_dim"]
                    or worker not in range(contract["block_dim"])
                    or values["pair_order"] not in PAIR_ORDERS
                    or values["request_count"] != expected_count
                    or values["request_bytes"] != REQUEST_BYTES
                    or values["control_cycles"] <= 0
                    or values["request_cycles"] <= 0
                    or control_sink != 1 or request_sink != 2
                    or not math.isclose(
                        control_ns,
                        values["control_cycles"] * SYSTEM_COUNTER_TICK_NS)
                    or not math.isclose(
                        request_ns,
                        values["request_cycles"] * SYSTEM_COUNTER_TICK_NS)):
                raise ValueError(f"{path} contains an invalid row")
            identities.add(identity)
            rows.append({
                "run_id": raw["run_id"], **values,
                "incremental_ns": request_ns - control_ns,
            })
    return rows


def _audit_run(path, maximum_requests):
    contract = CELL_CONTRACTS[maximum_requests]
    rows = _load_run(path, maximum_requests)
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one raw file must contain exactly one run ID")
    process_ids = {row["process_pid"] for row in rows}
    if len(process_ids) != 1:
        raise ValueError("one raw file must contain exactly one process PID")
    trials = sorted({row["trial"] for row in rows})
    if (not trials or trials != list(range(1, len(trials) + 1))
            or len(trials) < DEFAULT_TRIALS or len(trials) % 2):
        raise ValueError(
            "run must use a contiguous even schedule of at least 120 trials")
    indexed = {
        (row["trial"], row["pair_order"], row["worker"]): row
        for row in rows
    }
    expected_rows = len(trials) * len(PAIR_ORDERS) * contract["block_dim"]
    if len(indexed) != expected_rows:
        raise ValueError("run does not contain every trial/order/worker row")
    samples_by_order = {order: [] for order in PAIR_ORDERS}
    maximum_counts = []
    maximum_workers = []
    for trial in trials:
        expected_orders = _ordered_pair_orders(trial - 1)
        for position, order in enumerate(expected_orders):
            cell_rows = [
                indexed[(trial, order, worker)]
                for worker in range(contract["block_dim"])]
            if {row["order_position"] for row in cell_rows} != {position}:
                raise ValueError("position-balanced order gate failed")
            maximum = max(
                cell_rows, key=lambda row: row["incremental_ns"])
            samples_by_order[order].append(maximum["incremental_ns"])
            maximum_counts.append(maximum["request_count"])
            maximum_workers.append(maximum["worker"])
    samples = [
        sample for order in PAIR_ORDERS
        for sample in samples_by_order[order]]
    q1, _, q3 = statistics.quantiles(
        samples, n=4, method="inclusive")
    median = statistics.median(samples)
    order_medians = tuple(
        statistics.median(samples_by_order[order])
        for order in PAIR_ORDERS)
    within_spread = math.inf if median <= 0 else (q3 - q1) / median
    order_spread = _relative_spread(order_medians)
    maximum_count_gate = set(maximum_counts) == {maximum_requests}
    passed = (
        q1 > 0 and within_spread <= RELATIVE_LIMIT
        and order_spread <= RELATIVE_LIMIT and maximum_count_gate)
    return {
        "run_id": next(iter(run_ids)),
        "process_pid": next(iter(process_ids)),
        "trials": len(trials),
        "samples": len(samples),
        "median_ns": median,
        "q1_ns": q1,
        "q3_ns": q3,
        "within_run_relative_iqr": within_spread,
        "pair_order_medians_ns": order_medians,
        "pair_order_relative_median_spread": order_spread,
        "maximum_worker_ids": sorted(set(maximum_workers)),
        "maximum_worker_request_count_gate": maximum_count_gate,
        "positive_q1_gate": q1 > 0,
        "gate_passed": passed,
    }


def _build_report(paths_by_request_size):
    if set(paths_by_request_size) != set(CELL_CONTRACTS):
        raise ValueError("exact 2-, 4-, and 12-request cells are required")
    cells = {}
    all_process_ids = []
    raw_paths = []
    for request_size in sorted(CELL_CONTRACTS):
        paths = tuple(paths_by_request_size[request_size])
        if len(paths) != REQUIRED_INDEPENDENT_RUNS:
            raise ValueError(
                "every exact cell requires three independent run files")
        audits = tuple(
            _audit_run(path, request_size) for path in paths)
        run_ids = tuple(audit["run_id"] for audit in audits)
        process_ids = tuple(audit["process_pid"] for audit in audits)
        if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
            raise ValueError("run IDs must be distinct within every cell")
        if len(set(process_ids)) != REQUIRED_INDEPENDENT_RUNS:
            raise ValueError(
                "independent runs must use distinct process PIDs")
        medians = tuple(audit["median_ns"] for audit in audits)
        cross_spread = _relative_spread(medians)
        cell_passed = (
            all(audit["gate_passed"] for audit in audits)
            and cross_spread <= RELATIVE_LIMIT)
        cells[str(request_size)] = {
            "request_count": request_size,
            **CELL_CONTRACTS[request_size],
            "service_time_ns": (
                statistics.median(medians) if cell_passed else None),
            "cross_run_relative_median_spread": cross_spread,
            "runs": audits,
            "gate_passed": cell_passed,
        }
        all_process_ids.extend(process_ids)
        raw_paths.extend(paths)
    if len(set(all_process_ids)) != len(all_process_ids):
        raise ValueError("all exact-cell captures must use distinct processes")
    passed = all(cell["gate_passed"] for cell in cells.values())
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "CANN 8.5.1",
        "service": "add_rmsnorm_key31_residual_reload_completion_chain",
        "scope": (
            "exact BF16 key-31 maximum-worker residual reload completion "
            "chains for T1/T64/T256"),
        "request_size_unit": "source_split_d_residual_reload_chains",
        "service_time_unit": "ns_per_maximum_worker_chain",
        "lookup_policy": "exact_only",
        "service_time_ns_by_request_size": {
            request_size: cells[request_size]["service_time_ns"]
            for request_size in cells},
        "source_sha256": SOURCE_SHA256,
        "cells": cells,
        "relative_gate_limit": RELATIVE_LIMIT,
        "measurement_gate_passed": passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": passed,
        "raw_artifacts": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in raw_paths],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-2", nargs=3, required=True, type=Path)
    parser.add_argument("--cell-4", nargs=3, required=True, type=Path)
    parser.add_argument("--cell-12", nargs=3, required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = _build_report({
        2: tuple(args.cell_2),
        4: tuple(args.cell_4),
        12: tuple(args.cell_12),
    })
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    if not report["measurement_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
