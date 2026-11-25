#!/usr/bin/env python3
"""Validate the two source-ordered CANN SwiGLU tiling phases."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


BLOCK_DIMS = (24, 40, 48)
MINIMUM_TRIALS = 6
PAYLOAD_BYTES = 80
PHASE_REPETITIONS = 2
SYSTEM_COUNTER_TICK_NS = 20.0
MAXIMUM_RELATIVE_MEDIAN_SPREAD = 0.10
MAXIMUM_VALIDATION_MAPE = 0.10
MAXIMUM_VALIDATION_APE = 0.15
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "payload_bytes",
    "phase_repetitions", "block_dim", "iterations", "short_cycles",
    "long_cycles", "tick_ns", "service_time_ns", "validation",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_span(values):
    center = statistics.median(values)
    return math.inf if center <= 0 else (max(values) - min(values)) / center


def _error_metrics(points, serialized_ns_per_aiv):
    relative = []
    absolute = []
    actuals = []
    for block_dim, actual_ns in points:
        predicted_ns = serialized_ns_per_aiv * block_dim
        error_ns = abs(predicted_ns - actual_ns)
        relative.append(error_ns / actual_ns)
        absolute.append(error_ns)
        actuals.append(actual_ns)
    return {
        "mape": statistics.fmean(relative),
        "wmae": math.fsum(absolute) / math.fsum(actuals),
        "maximum_ape": max(relative),
    }


def _load_single_phase_reference(path):
    reference = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "hardware_id", "software_revision", "payload_bytes", "model",
        "parameters", "production_or_evaluation_duration_consumed",
        "mechanism_publishable",
    }
    if not required.issubset(reference):
        raise ValueError("single-phase reference is missing required fields")
    if (reference["hardware_id"] != "ascend_910b"
            or reference["software_revision"] != "cann_8.5.1"
            or reference["payload_bytes"] != PAYLOAD_BYTES
            or reference["model"]
            != "shared_aiv_tiling_materialization_affine_v1"
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["mechanism_publishable"] is not True):
        raise ValueError("single-phase reference identity is invalid")
    parameters = reference["parameters"]
    if set(parameters) != {"startup_ns", "serialized_ns_per_aiv"}:
        raise ValueError("single-phase reference parameters changed")
    return reference


def _load_rows(paths):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"{path} has an invalid header")
            file_rows = list(reader)
        if (len(file_rows) < MINIMUM_TRIALS * len(BLOCK_DIMS)
                or len(file_rows) % (MINIMUM_TRIALS * len(BLOCK_DIMS)) != 0):
            raise ValueError(f"{path} has an invalid row count")
        rows.extend(file_rows)
    return rows


def build_report(raw_paths, single_phase_reference_path, source_commit,
                 entry_source_sha256, single_source_sha256,
                 tiling_source_sha256):
    if len(raw_paths) != 3:
        raise ValueError("exactly three independent raw runs are required")
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full lowercase Git hash")
    for name, value in (
            ("entry_source_sha256", entry_source_sha256),
            ("single_source_sha256", single_source_sha256),
            ("tiling_source_sha256", tiling_source_sha256)):
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{name} must be lowercase SHA256")
    reference = _load_single_phase_reference(single_phase_reference_path)
    rows = _load_rows(raw_paths)
    trials_per_run = len(rows) // (len(raw_paths) * len(BLOCK_DIMS))
    parsed = []
    run_ids = []
    for index, row in enumerate(rows, start=1):
        run_id = row["run_id"]
        if run_id not in run_ids:
            run_ids.append(run_id)
        trial = int(row["trial"])
        order_position = int(row["order_position"])
        block_dim = int(row["block_dim"])
        service_time_ns = float(row["service_time_ns"])
        if (trial < 1 or trial > trials_per_run
                or order_position not in range(len(BLOCK_DIMS))
                or int(row["payload_bytes"]) != PAYLOAD_BYTES
                or int(row["phase_repetitions"]) != PHASE_REPETITIONS
                or block_dim not in BLOCK_DIMS
                or int(row["iterations"]) < 4096
                or float(row["tick_ns"]) != SYSTEM_COUNTER_TICK_NS
                or row["validation"] != "pass"
                or not math.isfinite(service_time_ns)
                or service_time_ns <= 0):
            raise ValueError(f"row {index} changed the frozen mechanism cell")
        parsed.append({
            "run_id": run_id,
            "trial": trial,
            "block_dim": block_dim,
            "service_time_ns": service_time_ns,
        })
    if len(run_ids) != 3:
        raise ValueError("raw inputs must contain three distinct run IDs")
    expected_cells = {
        (run_id, trial, block_dim)
        for run_id in run_ids
        for trial in range(1, trials_per_run + 1)
        for block_dim in BLOCK_DIMS
    }
    observed_cells = {
        (row["run_id"], row["trial"], row["block_dim"])
        for row in parsed
    }
    if observed_cells != expected_cells or len(parsed) != len(expected_cells):
        raise ValueError("raw rows do not cover the exact run/trial/grid matrix")
    medians = {
        run_id: {
            str(block_dim): statistics.median(
                row["service_time_ns"] for row in parsed
                if row["run_id"] == run_id
                and row["block_dim"] == block_dim)
            for block_dim in BLOCK_DIMS
        }
        for run_id in run_ids
    }
    run_spreads = {
        str(block_dim): _relative_span(tuple(
            medians[run_id][str(block_dim)] for run_id in run_ids))
        for block_dim in BLOCK_DIMS
    }
    training_run_ids = tuple(run_ids[:2])
    validation_run_ids = tuple(run_ids[2:])
    rates_by_training_run = {
        run_id: statistics.median(tuple(
            medians[run_id][str(block_dim)] / block_dim
            for block_dim in BLOCK_DIMS))
        for run_id in training_run_ids
    }
    serialized_ns_per_aiv = statistics.median(
        tuple(rates_by_training_run.values()))
    training_points = tuple(
        (block_dim, medians[run_id][str(block_dim)])
        for run_id in training_run_ids for block_dim in BLOCK_DIMS)
    validation_points = tuple(
        (block_dim, medians[run_id][str(block_dim)])
        for run_id in validation_run_ids for block_dim in BLOCK_DIMS)
    training_metrics = _error_metrics(
        training_points, serialized_ns_per_aiv)
    validation_metrics = _error_metrics(
        validation_points, serialized_ns_per_aiv)
    parameter_stability = {
        "serialized_ns_per_aiv_relative_span": _relative_span(
            tuple(rates_by_training_run.values())),
    }
    measurement_gate_passed = (
        all(value <= MAXIMUM_RELATIVE_MEDIAN_SPREAD
            for value in run_spreads.values())
        and all(value <= MAXIMUM_RELATIVE_MEDIAN_SPREAD
                for value in parameter_stability.values())
        and training_metrics["mape"] <= MAXIMUM_VALIDATION_MAPE
        and training_metrics["maximum_ape"] <= MAXIMUM_VALIDATION_APE
        and validation_metrics["mape"] <= MAXIMUM_VALIDATION_MAPE
        and validation_metrics["maximum_ape"] <= MAXIMUM_VALIDATION_APE)
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "swiglu_source_ordered_tiling_prologue_pair",
        "payload_bytes_per_phase": PAYLOAD_BYTES,
        "phase_repetitions": PHASE_REPETITIONS,
        "model": "source_ordered_aiv_tiling_materialization_pair_v1",
        "formula": "serialized_ns_per_aiv * active_aiv_workers",
        "request_size_unit": "active_aiv_workers",
        "duration_unit": "ns_per_complete_source_prologue_pair",
        "source_order": (
            "entry GET_TILING_DATA; isDoubleBuffer branch; "
            "SwigluSingle::Init GetTilingAndOffset GET_TILING_DATA"),
        "single_phase_reference": str(single_phase_reference_path),
        "single_phase_reference_sha256": _sha256(
            single_phase_reference_path),
        "parameters": {
            "serialized_ns_per_aiv": serialized_ns_per_aiv,
        },
        "raw_artifacts": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in raw_paths
        ],
        "independent_runs": len(run_ids),
        "samples_per_cell": trials_per_run,
        "block_dims": list(BLOCK_DIMS),
        "medians_ns_by_run_and_block_dim": medians,
        "run_relative_median_spread_by_block_dim": run_spreads,
        "training_run_ids": list(training_run_ids),
        "validation_run_ids": list(validation_run_ids),
        "fits_by_training_run": {
            run_id: {"serialized_ns_per_aiv": rate}
            for run_id, rate in rates_by_training_run.items()
        },
        "parameter_stability": parameter_stability,
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "maximum_relative_median_spread": MAXIMUM_RELATIVE_MEDIAN_SPREAD,
        "maximum_validation_mape": MAXIMUM_VALIDATION_MAPE,
        "maximum_validation_ape": MAXIMUM_VALIDATION_APE,
        "source_commit": source_commit,
        "operator_source_evidence": {
            "entry_source_sha256": entry_source_sha256,
            "single_source_sha256": single_source_sha256,
            "tiling_source_sha256": tiling_source_sha256,
        },
        "measurement_gate_passed": measurement_gate_passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": measurement_gate_passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path, action="append")
    parser.add_argument("--single-phase-reference", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--entry-source-sha256", required=True)
    parser.add_argument("--single-source-sha256", required=True)
    parser.add_argument("--tiling-source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        args.raw, args.single_phase_reference, args.source_commit,
        args.entry_source_sha256, args.single_source_sha256,
        args.tiling_source_sha256)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
