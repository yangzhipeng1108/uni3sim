#!/usr/bin/env python3
"""Audit independent q2 CopyIn first/steady edge captures."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from swiglu_input_pair_loop_edges_runner import (
    BLOCK_DIM,
    CELLS,
    PAYLOAD_BYTES,
    REPETITIONS,
    _cell_index,
)


REQUIRED_RUNS = 3
REPLICATION_LIMIT = 0.20
RUN_SPREAD_LIMIT = 0.20
FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "payload_bytes", "repetitions", "pair_order",
    "control_first_cycles", "control_steady_cycles", "request_first_cycles",
    "request_steady_cycles", "control_first_ns", "control_steady_ns",
    "request_first_ns", "request_steady_ns", "sink", "validation",
)


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0.0 else abs(left - right) / scale


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path} has an invalid schema")
        rows = tuple(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError(f"{path} must contain one run ID")
    by_trial_position = {}
    identities = set()
    for raw in rows:
        if raw["validation"] != "pass":
            raise ValueError(f"{path} includes a failed device row")
        value = {
            name: int(raw[name])
            for name in FIELDS
            if name not in ("run_id", "validation") and not name.endswith("_ns")
        }
        timing = {name: float(raw[name]) for name in FIELDS if name.endswith("_ns")}
        if (
                value["block_dim"] != BLOCK_DIM
                or value["payload_bytes"] != PAYLOAD_BYTES
                or value["repetitions"] not in REPETITIONS
                or value["pair_order"] not in (0, 1)
                or value["worker"] not in range(BLOCK_DIM)
                or value["trial"] <= 0 or value["order_position"] < 0
                or value["process_pid"] <= 0
                or value["control_first_cycles"] < 0
                or value["request_first_cycles"] <= 0
                or value["sink"] != value["repetitions"] * 0xB4):
            raise ValueError(f"{path} has invalid raw values")
        if value["repetitions"] == 1:
            if (value["control_steady_cycles"] != 0
                    or value["request_steady_cycles"] != 0):
                raise ValueError(f"{path} has nonzero one-visit steady data")
        elif (value["control_steady_cycles"] < 0
              or value["request_steady_cycles"] <= 0):
            raise ValueError(f"{path} has invalid two-visit steady data")
        for stem in ("control_first", "control_steady", "request_first",
                     "request_steady"):
            if not math.isclose(
                    timing[f"{stem}_ns"], value[f"{stem}_cycles"] * 20.0,
                    rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{path} has invalid system-counter units")
        cell = (value["repetitions"], value["pair_order"])
        identity = (value["trial"], value["order_position"], value["worker"])
        if identity in identities:
            raise ValueError(f"{path} duplicates raw identity")
        identities.add(identity)
        trial_position = (value["trial"], value["order_position"])
        if trial_position not in by_trial_position:
            by_trial_position[trial_position] = []
        by_trial_position[trial_position].append(
            {**value, **timing, "cell": cell})

    trials = tuple(sorted({trial for trial, _ in by_trial_position}))
    if trials != tuple(range(1, len(trials) + 1)) or len(trials) % len(CELLS):
        raise ValueError(f"{path} does not contain balanced complete trials")
    first_samples = []
    first_with_followup_samples = []
    steady_samples = []
    for trial in trials:
        cells = {}
        for position in range(len(CELLS)):
            expected = CELLS[_cell_index(trial - 1, position)]
            trial_position = (trial, position)
            if trial_position not in by_trial_position:
                raise ValueError(f"{path} loses a balanced cell")
            rows_for_position = by_trial_position[trial_position]
            if (len(rows_for_position) != BLOCK_DIM
                    or {row["cell"] for row in rows_for_position} != {expected}
                    or {row["worker"] for row in rows_for_position}
                    != set(range(BLOCK_DIM))):
                raise ValueError(f"{path} loses a balanced cell")
            cells[expected] = {row["worker"]: row for row in rows_for_position}
        first_samples.append(max(
            cells[(1, 1)][worker]["request_first_ns"]
            - cells[(1, 0)][worker]["control_first_ns"]
            for worker in range(BLOCK_DIM)))
        first_with_followup_samples.append(max(
            cells[(2, 1)][worker]["request_first_ns"]
            - cells[(2, 0)][worker]["control_first_ns"]
            for worker in range(BLOCK_DIM)))
        steady_samples.append(max(
            cells[(2, 1)][worker]["request_steady_ns"]
            - cells[(2, 0)][worker]["control_steady_ns"]
            for worker in range(BLOCK_DIM)))
    return {
        "run_id": next(iter(run_ids)),
        "trials": len(trials),
        "first_samples_ns": first_samples,
        "first_with_followup_samples_ns": first_with_followup_samples,
        "steady_samples_ns": steady_samples,
    }


def _summary(samples):
    first_half = samples[:len(samples) // 2]
    second_half = samples[len(samples) // 2:]
    median = statistics.median(samples)
    return {
        "q1_ns": statistics.quantiles(samples, n=4, method="inclusive")[0],
        "median_ns": median,
        "q3_ns": statistics.quantiles(samples, n=4, method="inclusive")[2],
        "replication_relative_spread": _relative_difference(
            statistics.median(first_half), statistics.median(second_half)),
    }


def _audit(paths):
    if len(paths) != REQUIRED_RUNS:
        raise ValueError(f"exactly {REQUIRED_RUNS} raw files are required")
    runs = tuple(_load_run(path) for path in paths)
    run_ids = tuple(run["run_id"] for run in runs)
    if len(set(run_ids)) != REQUIRED_RUNS:
        raise ValueError("each raw file needs a distinct run ID")
    components = {}
    for name, sample_name in (
            ("first_input_pair", "first_samples_ns"),
            ("steady_input_pair", "steady_samples_ns")):
        per_run = {run["run_id"]: _summary(run[sample_name]) for run in runs}
        medians = tuple(summary["median_ns"] for summary in per_run.values())
        components[name] = {
            "medians_ns_by_run": {
                run_id: per_run[run_id]["median_ns"] for run_id in run_ids},
            "median_ns": statistics.median(medians),
            "maximum_run_relative_median_spread": _relative_difference(
                min(medians), max(medians)),
            "within_run_replication": {
                run_id: per_run[run_id]["replication_relative_spread"]
                for run_id in run_ids},
        }
        components[name]["mechanism_gate"] = (
            all(per_run[run_id]["q1_ns"] > 0.0
                and per_run[run_id]["replication_relative_spread"]
                <= REPLICATION_LIMIT for run_id in run_ids)
            and components[name]["maximum_run_relative_median_spread"]
            <= RUN_SPREAD_LIMIT)
    first_visit_invariance = {
        run["run_id"]: _relative_difference(
            statistics.median(run["first_samples_ns"]),
            statistics.median(run["first_with_followup_samples_ns"]))
        for run in runs}
    first_visit_invariance_gate = all(
        difference <= REPLICATION_LIMIT
        for difference in first_visit_invariance.values())
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "48-AIV q2 source-ordered A/B CopyIn first and second "
            "DataCopy/EnQue/DeQue/Free completion with matched PIPE_ALL "
            "controls"),
        "payload_bytes_per_input": PAYLOAD_BYTES,
        "active_aiv_workers": BLOCK_DIM,
        "queue_depth": 2,
        "components": components,
        "first_visit_with_followup_relative_difference": (
            first_visit_invariance),
        "first_visit_with_followup_gate": first_visit_invariance_gate,
        "mechanism_gate": all(
            component["mechanism_gate"] for component in components.values())
        and first_visit_invariance_gate,
        "production_or_evaluation_duration_consumed": False,
        "raw_artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = _audit(args.raw)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["mechanism_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
