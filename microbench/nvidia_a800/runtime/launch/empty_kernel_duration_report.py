#!/usr/bin/env python3
"""Fit the frozen A800 isolated empty-task scheduler model."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_FIELDS = (
    "trial", "grid", "block", "sample_count", "median_duration_ns")
EXPECTED_TRIALS = (1, 2, 3)
TRAINING_TRIALS = (1, 2)
VALIDATION_TRIALS = (3,)
EXPECTED_GRIDS = (
    1, 4, 16, 64, 108, 216, 432, 1024, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288, 1048576,
)
EXPECTED_BLOCKS = (1, 32, 64, 128, 256, 512, 1024)
EXPECTED_SAMPLES_PER_CELL = 200
NUM_COMPUTE_UNITS = 108
PARALLEL_WIDTH = 32


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value, name):
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(parsed) != value or parsed <= 0:
        raise ValueError(f"{name} must be a canonical positive integer")
    return parsed


def _positive_float(value, name):
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError(
                "A800 launch summary fields disagree with the frozen schema")
        rows = []
        identities = set()
        for ordinal, raw in enumerate(reader, start=2):
            row = {
                "trial": _positive_int(raw["trial"], f"line {ordinal}.trial"),
                "grid": _positive_int(raw["grid"], f"line {ordinal}.grid"),
                "block": _positive_int(raw["block"], f"line {ordinal}.block"),
                "sample_count": _positive_int(
                    raw["sample_count"], f"line {ordinal}.sample_count"),
                "median_duration_ns": _positive_float(
                    raw["median_duration_ns"],
                    f"line {ordinal}.median_duration_ns"),
            }
            identity = (row["trial"], row["grid"], row["block"])
            if identity in identities:
                raise ValueError(f"duplicate A800 launch cell: {identity}")
            identities.add(identity)
            rows.append(row)
    expected = {
        (trial, grid, block)
        for trial in EXPECTED_TRIALS
        for grid in EXPECTED_GRIDS
        for block in EXPECTED_BLOCKS
    }
    if identities != expected:
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise ValueError(
            "A800 launch summary matrix is incomplete: "
            f"missing={missing[:4]}, extra={extra[:4]}")
    if any(row["sample_count"] != EXPECTED_SAMPLES_PER_CELL for row in rows):
        raise ValueError(
            "every A800 launch cell must contain exactly 200 samples")
    return tuple(rows)


def _features(row, maximum_cta_wave_count):
    rounds = math.ceil(row["grid"] / NUM_COMPUTE_UNITS)
    waves = math.ceil(row["block"] / PARALLEL_WIDTH)
    if waves <= maximum_cta_wave_count:
        return (1.0, float(rounds), 0.0)
    return (1.0, 0.0, float(rounds * waves))


def _relative_fit(rows, maximum_cta_wave_count):
    design = np.asarray(
        [_features(row, maximum_cta_wave_count) for row in rows],
        dtype=float,
    )
    durations = np.asarray(
        [row["median_duration_ns"] for row in rows], dtype=float)
    relative_design = design / durations[:, None]
    coefficients, _, _, _ = np.linalg.lstsq(
        relative_design, np.ones(len(rows)), rcond=None)
    if np.any(coefficients <= 0) or not np.all(np.isfinite(coefficients)):
        return None
    submit_ns, cta_round_ns, wave_round_ns = coefficients
    crossover = cta_round_ns / wave_round_ns
    waves = sorted({math.ceil(row["block"] / PARALLEL_WIDTH)
                    for row in rows})
    lower = max(wave for wave in waves
                if wave <= maximum_cta_wave_count)
    upper = min(wave for wave in waves
                if wave > maximum_cta_wave_count)
    if not lower <= crossover <= upper:
        return None
    predictions = design @ coefficients
    relative_residual = (predictions - durations) / durations
    return {
        "maximum_cta_wave_count": maximum_cta_wave_count,
        "submit_ns": float(submit_ns),
        "cta_round_ns": float(cta_round_ns),
        "wave_round_ns": float(wave_round_ns),
        "crossover_wave_count": float(crossover),
        "relative_squared_error": float(
            np.dot(relative_residual, relative_residual)),
    }


def fit_model(rows):
    training = tuple(
        row for row in rows if row["trial"] in TRAINING_TRIALS)
    wave_counts = sorted({
        math.ceil(block / PARALLEL_WIDTH) for block in EXPECTED_BLOCKS})
    candidates = []
    for maximum_cta_wave_count in wave_counts[:-1]:
        candidate = _relative_fit(training, maximum_cta_wave_count)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise ValueError(
            "no non-negative self-consistent A800 launch model was fitted")
    candidates.sort(key=lambda value: value["relative_squared_error"])
    return candidates[0]


def _predict(row, model):
    rounds = math.ceil(row["grid"] / NUM_COMPUTE_UNITS)
    waves = math.ceil(row["block"] / PARALLEL_WIDTH)
    return (
        model["submit_ns"]
        + rounds * max(
            model["cta_round_ns"], model["wave_round_ns"] * waves)
    )


def _metrics(rows, model):
    observed = np.asarray(
        [row["median_duration_ns"] for row in rows], dtype=float)
    predicted = np.asarray([_predict(row, model) for row in rows])
    error = predicted - observed
    absolute_percentage = np.abs(error) / observed
    return {
        "cell_count": len(rows),
        "mape_percent": float(np.mean(absolute_percentage) * 100.0),
        "wmae_percent": float(
            np.sum(np.abs(error)) / np.sum(observed) * 100.0),
        "signed_mean_percentage_error": float(
            np.mean(error / observed) * 100.0),
        "p95_absolute_percentage_error": float(
            np.percentile(absolute_percentage, 95) * 100.0),
        "maximum_absolute_percentage_error": float(
            np.max(absolute_percentage) * 100.0),
    }


def build_report(path):
    rows = read_summary(path)
    model = fit_model(rows)
    training = tuple(
        row for row in rows if row["trial"] in TRAINING_TRIALS)
    validation = tuple(
        row for row in rows if row["trial"] in VALIDATION_TRIALS)
    return {
        "schema_version": 1,
        "hardware_id": "nvidia_a800",
        "mechanism": "isolated_empty_task_scheduler",
        "duration_source": "nsys_gpu_kernel",
        "production_or_evaluation_duration_consumed": False,
        "input": {
            "path": str(path),
            "sha256": _sha256(path),
            "trials": list(EXPECTED_TRIALS),
            "training_trials": list(TRAINING_TRIALS),
            "validation_trials": list(VALIDATION_TRIALS),
            "samples_per_cell": EXPECTED_SAMPLES_PER_CELL,
        },
        "formula": (
            "submit_ns + ceil(grid/108) * "
            "max(cta_round_ns, wave_round_ns * ceil(block/32))"
        ),
        "parameters": {
            "submit_ns": model["submit_ns"],
            "cta_round_ns": model["cta_round_ns"],
            "wave_round_ns": model["wave_round_ns"],
            "extra_grid_dimension_ns": 0.0,
            "extra_block_dimension_ns": 0.0,
        },
        "fit_diagnostics": {
            "maximum_cta_wave_count": model["maximum_cta_wave_count"],
            "crossover_wave_count": model["crossover_wave_count"],
            "training": _metrics(training, model),
            "validation": _metrics(validation, model),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
