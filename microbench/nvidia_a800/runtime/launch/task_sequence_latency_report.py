#!/usr/bin/env python3
"""Summarize the independent A800 task-sequence latency capture."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


EXPECTED_CONDITIONS = ("idle", "queued")
EXPECTED_PROGRAMS = ("empty", "dependent")
EXPECTED_COUNTS = (0, 1, 2, 4, 5, 8)
EXPECTED_TRIALS = (0, 1, 2)
TRAIN_TRIALS = (0, 1)
VALIDATION_TRIALS = (2,)
EXPECTED_SAMPLES = 50
EXPECTED_GEOMETRY_SEQUENCES = (
    "none", "fa_main", "fa_combine", "fa_main_combine",
    "fa_combine_main",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_capture(capture):
    provenance = json.loads((capture / "provenance.json").read_text())
    if provenance["schema"] != "nvidia_a800_task_sequence_latency_capture_v2":
        raise ValueError("task-sequence provenance schema changed")
    raw_path = capture / "raw.csv"
    if provenance["raw_sha256"] != _sha256(raw_path):
        raise ValueError("task-sequence raw SHA256 mismatch")
    if provenance["production_or_evaluation_duration_consumed"]:
        raise ValueError("task-sequence capture consumed production duration")
    metadata = {}
    rows = []
    geometry_rows = []
    with raw_path.open(newline="", encoding="utf-8") as stream:
        lines = stream.readlines()
    raw_lines = []
    geometry_lines = []
    for line in lines:
        if line.startswith("meta,"):
            fields = next(csv.reader([line]))
            metadata[fields[1]] = fields[2]
        elif line.startswith("raw,"):
            raw_lines.append(line.removeprefix("raw,"))
        elif line.startswith("geometry_raw,"):
            geometry_lines.append(line.removeprefix("geometry_raw,"))
        else:
            raise ValueError("task-sequence output has an unknown row type")
    reader = csv.DictReader(raw_lines)
    expected_fields = (
        "trial", "condition", "program", "kernel_count", "sample",
        "event_duration_ns", "host_submission_ns")
    if tuple(reader.fieldnames or ()) != expected_fields:
        raise ValueError("task-sequence raw fields changed")
    identities = set()
    for ordinal, raw in enumerate(reader, start=2):
        row = {
            "trial": int(raw["trial"]),
            "condition": raw["condition"],
            "program": raw["program"],
            "kernel_count": int(raw["kernel_count"]),
            "sample": int(raw["sample"]),
            "event_duration_ns": float(raw["event_duration_ns"]),
            "host_submission_ns": int(raw["host_submission_ns"]),
        }
        identity = (
            row["trial"], row["condition"], row["program"],
            row["kernel_count"], row["sample"])
        if identity in identities:
            raise ValueError(f"duplicate task-sequence row: {identity}")
        identities.add(identity)
        if (row["trial"] not in EXPECTED_TRIALS
                or row["condition"] not in EXPECTED_CONDITIONS
                or row["program"] not in EXPECTED_PROGRAMS
                or row["kernel_count"] not in EXPECTED_COUNTS
                or not 0 <= row["sample"] < EXPECTED_SAMPLES):
            raise ValueError(f"line {ordinal} is outside the frozen matrix")
        if (not math.isfinite(row["event_duration_ns"])
                or row["event_duration_ns"] < 0
                or row["host_submission_ns"] < 0):
            raise ValueError(f"line {ordinal} has invalid timing")
        rows.append(row)
    expected = {
        (trial, condition, program, count, sample)
        for trial in EXPECTED_TRIALS
        for condition in EXPECTED_CONDITIONS
        for program in EXPECTED_PROGRAMS
        for count in EXPECTED_COUNTS
        for sample in range(EXPECTED_SAMPLES)
    }
    if identities != expected:
        raise ValueError("task-sequence capture matrix is incomplete")
    geometry_reader = csv.DictReader(geometry_lines)
    expected_geometry_fields = (
        "trial", "condition", "sequence", "sample",
        "event_duration_ns", "host_submission_ns")
    if tuple(geometry_reader.fieldnames or ()) != expected_geometry_fields:
        raise ValueError("task-sequence geometry raw fields changed")
    geometry_identities = set()
    for raw in geometry_reader:
        row = {
            "trial": int(raw["trial"]),
            "condition": raw["condition"],
            "sequence": raw["sequence"],
            "sample": int(raw["sample"]),
            "event_duration_ns": float(raw["event_duration_ns"]),
            "host_submission_ns": int(raw["host_submission_ns"]),
        }
        identity = (
            row["trial"], row["condition"], row["sequence"], row["sample"])
        if identity in geometry_identities:
            raise ValueError(f"duplicate task-sequence geometry row: {identity}")
        geometry_identities.add(identity)
        if (row["trial"] not in EXPECTED_TRIALS
                or row["condition"] not in EXPECTED_CONDITIONS
                or row["sequence"] not in EXPECTED_GEOMETRY_SEQUENCES
                or not 0 <= row["sample"] < EXPECTED_SAMPLES):
            raise ValueError("task-sequence geometry row is outside the frozen matrix")
        if (not math.isfinite(row["event_duration_ns"])
                or row["event_duration_ns"] < 0
                or row["host_submission_ns"] < 0):
            raise ValueError("task-sequence geometry row has invalid timing")
        geometry_rows.append(row)
    expected_geometry = {
        (trial, condition, sequence, sample)
        for trial in EXPECTED_TRIALS
        for condition in EXPECTED_CONDITIONS
        for sequence in EXPECTED_GEOMETRY_SEQUENCES
        for sample in range(EXPECTED_SAMPLES)
    }
    if geometry_identities != expected_geometry:
        raise ValueError("task-sequence geometry capture matrix is incomplete")
    return provenance, metadata, rows, geometry_rows


def _cells(rows):
    cells = []
    for trial in EXPECTED_TRIALS:
        for condition in EXPECTED_CONDITIONS:
            for program in EXPECTED_PROGRAMS:
                for count in EXPECTED_COUNTS:
                    selected = [
                        row for row in rows
                        if row["trial"] == trial
                        and row["condition"] == condition
                        and row["program"] == program
                        and row["kernel_count"] == count
                    ]
                    cells.append({
                        "trial": trial,
                        "condition": condition,
                        "program": program,
                        "kernel_count": count,
                        "event_duration_ns": statistics.median(
                            row["event_duration_ns"] for row in selected),
                        "host_submission_ns": statistics.median(
                            row["host_submission_ns"] for row in selected),
                    })
    return cells


def _geometry_cells(rows):
    cells = []
    for trial in EXPECTED_TRIALS:
        for condition in EXPECTED_CONDITIONS:
            for sequence in EXPECTED_GEOMETRY_SEQUENCES:
                selected = [
                    row for row in rows
                    if row["trial"] == trial
                    and row["condition"] == condition
                    and row["sequence"] == sequence
                ]
                cells.append({
                    "trial": trial,
                    "condition": condition,
                    "sequence": sequence,
                    "event_duration_ns": statistics.median(
                        row["event_duration_ns"] for row in selected),
                    "host_submission_ns": statistics.median(
                        row["host_submission_ns"] for row in selected),
                })
    return cells


def _geometry_summaries(cells):
    summaries = []
    for condition in EXPECTED_CONDITIONS:
        training = {}
        validation = {}
        for sequence in EXPECTED_GEOMETRY_SEQUENCES:
            training[sequence] = statistics.median(
                cell["event_duration_ns"] for cell in cells
                if cell["condition"] == condition
                and cell["sequence"] == sequence
                and cell["trial"] in TRAIN_TRIALS)
            validation_values = [
                cell["event_duration_ns"] for cell in cells
                if cell["condition"] == condition
                and cell["sequence"] == sequence
                and cell["trial"] in VALIDATION_TRIALS]
            if len(validation_values) != 1:
                raise ValueError("geometry validation requires one frozen trial")
            validation[sequence] = validation_values[0]
        boundary = training["none"]
        event_excluded = {
            sequence: duration - boundary
            for sequence, duration in training.items()
            if sequence != "none"
        }
        validation_ape = {
            sequence: (
                abs(validation[sequence] - training[sequence])
                / validation[sequence] * 100.0)
            for sequence in EXPECTED_GEOMETRY_SEQUENCES
            if validation[sequence] > 0.0
        }
        summaries.append({
            "condition": condition,
            "training_event_duration_ns": training,
            "validation_event_duration_ns": validation,
            "event_pair_boundary_ns": boundary,
            "training_event_excluded_ns": event_excluded,
            "main_then_combine_interaction_ns": (
                training["fa_main_combine"] - training["fa_main"]
                - training["fa_combine"] + boundary),
            "combine_then_main_interaction_ns": (
                training["fa_combine_main"] - training["fa_combine"]
                - training["fa_main"] + boundary),
            "validation_mape_pct": statistics.mean(
                validation_ape.values()),
            "validation_max_ape_pct": max(validation_ape.values()),
        })
    return summaries


def _features(cell):
    count = cell["kernel_count"]
    return (1.0, float(count > 0), float(max(0, count - 1)))


def _fit(rows):
    design = np.asarray([_features(row) for row in rows], dtype=float)
    observed = np.asarray(
        [row["event_duration_ns"] for row in rows], dtype=float)
    coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
    return tuple(float(value) for value in coefficients)


def _predict(cell, coefficients):
    return sum(
        feature * coefficient
        for feature, coefficient in zip(
            _features(cell), coefficients, strict=True))


def _metrics(rows, coefficients):
    absolute = []
    percentage = []
    for row in rows:
        error = abs(_predict(row, coefficients) - row["event_duration_ns"])
        absolute.append(error)
        if row["event_duration_ns"] > 0:
            percentage.append(error / row["event_duration_ns"])
    return {
        "cell_count": len(rows),
        "mean_absolute_error_ns": statistics.mean(absolute),
        "maximum_absolute_error_ns": max(absolute),
        "mape_pct_positive_cells": 100.0 * statistics.mean(percentage),
        "maximum_ape_pct_positive_cells": 100.0 * max(percentage),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    capture = args.capture.resolve()
    provenance, metadata, rows, geometry_rows = _read_capture(capture)
    if (metadata["device_name"] != "NVIDIA A800-SXM4-80GB"
            or metadata["sm_count"] != "108"
            or metadata["threads"] != "128"
            or int(metadata["samples"]) != EXPECTED_SAMPLES
            or int(metadata["trials"]) != len(EXPECTED_TRIALS)):
        raise ValueError("task-sequence hardware or matrix metadata changed")
    cells = _cells(rows)
    geometry_cells = _geometry_cells(geometry_rows)
    models = []
    for condition in EXPECTED_CONDITIONS:
        for program in EXPECTED_PROGRAMS:
            selected = [
                cell for cell in cells
                if cell["condition"] == condition
                and cell["program"] == program]
            training = [
                cell for cell in selected if cell["trial"] in TRAIN_TRIALS]
            validation = [
                cell for cell in selected
                if cell["trial"] in VALIDATION_TRIALS]
            coefficients = _fit(training)
            models.append({
                "condition": condition,
                "program": program,
                "event_pair_boundary_ns": coefficients[0],
                "first_kernel_increment_ns": coefficients[1],
                "additional_kernel_increment_ns": coefficients[2],
                "training": _metrics(training, coefficients),
                "validation": _metrics(validation, coefficients),
            })
    report = {
        "schema": "nvidia_a800_task_sequence_latency_candidate_v2",
        "hardware_id": "nvidia_a800",
        "mechanism": "isolated_device_event_kernel_sequence",
        "duration_unit": "ns_per_task_sequence_component",
        "timer_domain": "cuda_event_gpu_timeline",
        "kernel_counts": list(EXPECTED_COUNTS),
        "train_trials": list(TRAIN_TRIALS),
        "validation_trials": list(VALIDATION_TRIALS),
        "models": models,
        "fa_launch_geometry": {
            "main_grid": [1, 4, 8],
            "main_block": [128, 1, 1],
            "main_dynamic_shared_memory_bytes": 81920,
            "combine_grid": [8, 1, 1],
            "combine_block": [128, 1, 1],
            "summaries": _geometry_summaries(geometry_cells),
        },
        "formula": (
            "event_pair_boundary_ns + I(N>0)*first_kernel_increment_ns + "
            "max(0,N-1)*additional_kernel_increment_ns"),
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "CUDA Event task boundaries, first launch service, and additional "
            "same-stream kernel transitions for operator-independent kernels, "
            "including the exact launch geometry and dynamic shared-memory "
            "reservation of the target FA main/combine sequence"),
        "excluded_conclusions": [
            "production FlashAttention duration",
            "a per-shape or per-kernel-name latency correction",
            "a kernel-internal CTA dependency or cache service",
        ],
        "source_sha256": provenance["source_sha256"],
        "runner_sha256": provenance["runner_sha256"],
        "raw_sha256": provenance["raw_sha256"],
        "capture_baseline_commit": provenance["baseline_commit"],
    }
    (capture / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (capture / "report_identity.json").write_text(
        json.dumps({
            "report_source_sha256": _sha256(Path(__file__).resolve()),
            "report_sha256": _sha256(capture / "report.json"),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
