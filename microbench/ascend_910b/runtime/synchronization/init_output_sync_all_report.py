#!/usr/bin/env python3
"""Audit the independent FIA mixed-task outer-envelope capture."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from init_output_sync_all_runner import (
    AIV_BLOCK_DIMS,
    ELEMENTS_PER_AIV,
    TRIALS,
    _ordered_cells,
)


SYSTEM_COUNTER_TICK_NS = 20.0
TRAINING_TRIALS = TRIALS // 2
MAXIMUM_MODEL_MAPE = 0.10
MAXIMUM_MODEL_APE = 0.15
MAXIMUM_ELEMENT_MEDIAN_SPAN = 0.10
FIELDS = (
    "run_id", "trial", "order_position", "process_pid",
    "aiv_block_dim", "elements_per_aiv", "block_cycles", "tick_ns",
    "phase_time_ns", "device_elapsed_ns", "validation",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_affine(points):
    x_mean = statistics.mean(point[0] for point in points)
    y_mean = statistics.mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator <= 0:
        raise ValueError("outer-envelope fit needs distinct AIV counts")
    per_active_aiv_ns = sum(
        (x - x_mean) * (y - y_mean) for x, y in points) / denominator
    startup_ns = y_mean - per_active_aiv_ns * x_mean
    if startup_ns <= 0 or per_active_aiv_ns <= 0:
        raise ValueError("outer-envelope coefficients must be positive")
    return startup_ns, per_active_aiv_ns


def _median_points(rows, training):
    return tuple(
        (
            block_dim,
            statistics.median(
                row["outer_envelope_ns"] for row in rows
                if row["aiv_block_dim"] == block_dim
                and ((row["trial"] <= TRAINING_TRIALS) is training)),
        )
        for block_dim in AIV_BLOCK_DIMS
    )


def build_report(raw_path, source_commit, source_snapshot_sha256,
                 binary_sha256):
    if TRIALS % 2 != 0:
        raise ValueError("campaign trials must split into balanced halves")
    with raw_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("output-init raw header changed")
        source_rows = list(reader)
    expected_rows = TRIALS * len(AIV_BLOCK_DIMS) * len(ELEMENTS_PER_AIV)
    if len(source_rows) != expected_rows:
        raise ValueError("output-init raw campaign is incomplete")

    rows = []
    run_ids = set()
    process_pids = set()
    for raw in source_rows:
        run_ids.add(raw["run_id"])
        process_pids.add(int(raw["process_pid"]))
        trial = int(raw["trial"])
        order_position = int(raw["order_position"])
        block_dim = int(raw["aiv_block_dim"])
        elements = int(raw["elements_per_aiv"])
        if trial < 1 or trial > TRIALS:
            raise ValueError("output-init trial is outside the frozen campaign")
        ordered = _ordered_cells(trial - 1)
        if order_position < 0 or order_position >= len(ordered):
            raise ValueError("output-init order position is invalid")
        if ordered[order_position] != (block_dim, elements):
            raise ValueError("output-init row order changed")
        cycles = tuple(int(value) for value in raw["block_cycles"].split(";"))
        if len(cycles) != block_dim or any(value <= 0 for value in cycles):
            raise ValueError("output-init block-cycle vector is invalid")
        tick_ns = float(raw["tick_ns"])
        phase_time_ns = float(raw["phase_time_ns"])
        device_elapsed_ns = float(raw["device_elapsed_ns"])
        expected_phase_ns = max(cycles) * SYSTEM_COUNTER_TICK_NS
        if tick_ns != SYSTEM_COUNTER_TICK_NS:
            raise ValueError("output-init system-counter tick changed")
        if not math.isclose(phase_time_ns, expected_phase_ns, abs_tol=1e-9):
            raise ValueError("output-init phase time disagrees with cycles")
        if (not math.isfinite(device_elapsed_ns)
                or device_elapsed_ns <= phase_time_ns):
            raise ValueError("complete task must exceed its internal phase")
        if raw["validation"] != "pass":
            raise ValueError("output-init device validation failed")
        rows.append({
            "trial": trial,
            "aiv_block_dim": block_dim,
            "elements_per_aiv": elements,
            "outer_envelope_ns": device_elapsed_ns - phase_time_ns,
        })
    if len(run_ids) != 1 or len(process_pids) != 1 or min(process_pids) <= 0:
        raise ValueError("raw campaign must come from one healthy process")

    training_points = _median_points(rows, True)
    validation_points = _median_points(rows, False)
    startup_ns, per_active_aiv_ns = _fit_affine(training_points)

    def model_errors(points):
        return tuple(
            abs(startup_ns + per_active_aiv_ns * block_dim - duration_ns)
            / duration_ns
            for block_dim, duration_ns in points
        )

    training_errors = model_errors(training_points)
    validation_errors = model_errors(validation_points)
    element_medians = {
        block_dim: tuple(
            statistics.median(
                row["outer_envelope_ns"] for row in rows
                if row["aiv_block_dim"] == block_dim
                and row["elements_per_aiv"] == elements)
            for elements in ELEMENTS_PER_AIV
        )
        for block_dim in AIV_BLOCK_DIMS
    }
    element_spans = {
        block_dim: (max(values) - min(values)) / statistics.median(values)
        for block_dim, values in element_medians.items()
    }
    training_mape = statistics.mean(training_errors)
    validation_mape = statistics.mean(validation_errors)
    maximum_training_ape = max(training_errors)
    maximum_validation_ape = max(validation_errors)
    maximum_element_span = max(element_spans.values())
    mechanism_publishable = (
        training_mape <= MAXIMUM_MODEL_MAPE
        and validation_mape <= MAXIMUM_MODEL_MAPE
        and maximum_training_ape <= MAXIMUM_MODEL_APE
        and maximum_validation_ape <= MAXIMUM_MODEL_APE
        and maximum_element_span <= MAXIMUM_ELEMENT_MEDIAN_SPAN
    )
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "fia_mixed_task_outer_envelope",
        "model": "fia_mixed_task_outer_envelope_affine_v1",
        "request_size_unit": "active_aiv_workers",
        "duration_unit": "ns_per_complete_mixed_task_outer_envelope",
        "active_aiv_workers": list(AIV_BLOCK_DIMS),
        "elements_per_aiv": list(ELEMENTS_PER_AIV),
        "training_median_duration_ns": [
            duration for _, duration in training_points],
        "validation_median_duration_ns": [
            duration for _, duration in validation_points],
        "parameters_ns": {
            "startup": startup_ns,
            "per_active_aiv_worker": per_active_aiv_ns,
        },
        "training_model_mape": training_mape,
        "validation_model_mape": validation_mape,
        "maximum_training_model_ape": maximum_training_ape,
        "maximum_validation_model_ape": maximum_validation_ape,
        "maximum_allowed_model_mape": MAXIMUM_MODEL_MAPE,
        "maximum_allowed_model_ape": MAXIMUM_MODEL_APE,
        "element_median_span_by_active_aiv_workers": {
            str(key): value for key, value in element_spans.items()
        },
        "maximum_element_median_span": maximum_element_span,
        "maximum_allowed_element_median_span": MAXIMUM_ELEMENT_MEDIAN_SPAN,
        "training_trials": TRAINING_TRIALS,
        "validation_trials": TRIALS - TRAINING_TRIALS,
        "clock": {
            "internal_source": "GetSystemCycle",
            "internal_frequency_hz": 50_000_000,
            "internal_tick_ns": SYSTEM_COUNTER_TICK_NS,
            "complete_task_source": "aclrtEventElapsedTime",
        },
        "mechanism": (
            "complete mixed-task device interval minus the internal "
            "InitOutput-plus-SyncAll phase"),
        "composition": (
            "one ordered outer task envelope around the source-closed FIA "
            "mixed-core program"),
        "source_commit": source_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "binary_sha256": binary_sha256,
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": mechanism_publishable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        args.raw, args.source_commit, args.source_snapshot_sha256,
        args.binary_sha256)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["mechanism_publishable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
