#!/usr/bin/env python3
"""Audit exact AddRMSNorm key-31 residual chain captures."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from add_rmsnorm_key31_residual_chain_runner import (
    BLOCK_DIM,
    DEFAULT_TRIALS,
    LAST_BLOCK_FACTOR,
    MAXIMUM_REQUESTS,
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


def _relative_spread(values):
    high = max(values)
    return math.inf if high <= 0 else (high - min(values)) / high


def _load_run(path):
    rows = []
    identities = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError(f"{path} has an invalid schema")
        for raw in reader:
            try:
                values = {name: int(raw[name]) for name in REQUIRED_FIELDS[1:11]}
                control_ns = float(raw["control_ns"])
                request_ns = float(raw["request_ns"])
                control_sink = int(raw["control_sink"])
                request_sink = int(raw["request_sink"])
            except ValueError as exc:
                raise ValueError(f"{path} contains non-numeric data") from exc
            identity = (raw["run_id"], values["trial"], values["pair_order"], values["worker"])
            expected_count = LAST_BLOCK_FACTOR * TILES_PER_ROW if values["worker"] == 46 else MAXIMUM_REQUESTS
            if (raw["validation"] != "pass" or identity in identities
                    or values["block_dim"] != BLOCK_DIM
                    or values["worker"] not in range(BLOCK_DIM)
                    or values["pair_order"] not in PAIR_ORDERS
                    or values["request_count"] != expected_count
                    or values["request_bytes"] != REQUEST_BYTES
                    or values["control_cycles"] <= 0
                    or values["request_cycles"] <= 0
                    or control_sink != 1 or request_sink != 2
                    or not math.isclose(control_ns, values["control_cycles"] * SYSTEM_COUNTER_TICK_NS)
                    or not math.isclose(request_ns, values["request_cycles"] * SYSTEM_COUNTER_TICK_NS)):
                raise ValueError(f"{path} contains an invalid row")
            identities.add(identity)
            rows.append({
                "run_id": raw["run_id"], **values,
                "incremental_ns": request_ns - control_ns,
            })
    return rows


def _audit_run(path):
    rows = _load_run(path)
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one raw file must contain exactly one run ID")
    process_ids = {row["process_pid"] for row in rows}
    if len(process_ids) != 1:
        raise ValueError("one raw file must contain exactly one process PID")
    trials = sorted({row["trial"] for row in rows})
    if trials != list(range(1, DEFAULT_TRIALS + 1)):
        raise ValueError("run changed the frozen 120-trial schedule")
    samples_by_order = {order: [] for order in PAIR_ORDERS}
    max_workers = []
    max_worker_counts = []
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        expected_orders = _ordered_pair_orders(trial - 1)
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        for position, order in enumerate(expected_orders):
            cell_rows = [row for row in trial_rows if row["pair_order"] == order]
            if (len(cell_rows) != BLOCK_DIM
                    or {row["worker"] for row in cell_rows} != set(range(BLOCK_DIM))
                    or {row["order_position"] for row in cell_rows} != {position}):
                raise ValueError("position or all-worker identity gate failed")
            maximum = max(cell_rows, key=lambda row: row["incremental_ns"])
            samples_by_order[order].append(maximum["incremental_ns"])
            max_workers.append(maximum["worker"])
            max_worker_counts.append(maximum["request_count"])
    samples = [sample for order in PAIR_ORDERS for sample in samples_by_order[order]]
    q1, median, q3 = statistics.quantiles(samples, n=4, method="inclusive")[0], statistics.median(samples), statistics.quantiles(samples, n=4, method="inclusive")[2]
    order_medians = [statistics.median(samples_by_order[order]) for order in PAIR_ORDERS]
    within_spread = math.inf if median <= 0 else (q3 - q1) / median
    order_spread = _relative_spread(order_medians)
    passed = (q1 > 0 and within_spread <= RELATIVE_LIMIT
              and order_spread <= RELATIVE_LIMIT
              and set(max_worker_counts) == {MAXIMUM_REQUESTS})
    return {
        "run_id": next(iter(run_ids)),
        "process_pid": next(iter(process_ids)),
        "samples": len(samples),
        "median_ns": median,
        "q1_ns": q1,
        "q3_ns": q3,
        "within_run_relative_iqr": within_spread,
        "pair_order_relative_median_spread": order_spread,
        "max_worker_ids": sorted(set(max_workers)),
        "max_worker_request_count_gate": (
            set(max_worker_counts) == {MAXIMUM_REQUESTS}),
        "gate_passed": passed,
    }


def _build_report(paths):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("exactly three independent run files are required")
    audits = [_audit_run(path) for path in paths]
    run_ids = [audit["run_id"] for audit in audits]
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("run IDs must be distinct")
    process_ids = [audit["process_pid"] for audit in audits]
    if len(set(process_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("independent runs must use distinct process PIDs")
    medians = [audit["median_ns"] for audit in audits]
    cross_spread = _relative_spread(medians)
    passed = all(audit["gate_passed"] for audit in audits) and cross_spread <= RELATIVE_LIMIT
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "CANN 8.5.1",
        "service": "add_rmsnorm_key31_residual_reload_completion_chain",
        "scope": "exact BF16 T1024/H16384 key31, all-47-AIV maximum-worker 44-request completion envelope",
        "service_time_unit": "ns_per_44_request_max_worker_chain",
        "service_time_ns": statistics.median(medians) if passed else None,
        "composition": "maximum_with_same_source_stage_mte2_service",
        "source_sha256": SOURCE_SHA256,
        "source_anchors": {
            "split_d_init_and_depth": "add_rms_norm_split_d.h:33-68,329-334",
            "split_d_ordering": "add_rms_norm_split_d.h:83-103,204-235",
            "datacopypad_bytes": "rms_norm_base.h:220-232",
            "tque_event_wait": "kernel_tquebind_impl.h:203-250,289-327",
        },
        "request_contract": {
            "block_dim": BLOCK_DIM,
            "block_factor": 22,
            "last_block_factor": LAST_BLOCK_FACTOR,
            "tiles_per_row": TILES_PER_ROW,
            "request_bytes": REQUEST_BYTES,
            "normal_worker_requests": MAXIMUM_REQUESTS,
            "last_worker_requests": LAST_BLOCK_FACTOR * TILES_PER_ROW,
            "normal_worker_bytes": MAXIMUM_REQUESTS * REQUEST_BYTES,
        },
        "independent_runs": audits,
        "cross_run_relative_median_spread": cross_spread,
        "measurement_gate_passed": passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": passed,
        "raw_artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = _build_report(args.raw_csv)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    if not report["measurement_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
