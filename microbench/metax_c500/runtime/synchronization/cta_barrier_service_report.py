#!/usr/bin/env python3
"""Validate and reduce the C500 CTA-barrier cycle campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = 1
HARDWARE_ID = "metax_c500"
SOFTWARE_REVISION = "maca_3.7.2"
MECHANISM = "cta_barrier_service"
AP_COUNT = 104
WAVE_SIZE = 64
THREADS_PER_CTA = 256
DYNAMIC_SHARED_BYTES = 32 * 1024
RESIDENT_CTAS_PER_AP = 2
GRIDS = (1, 104, 208)
REPETITION_COUNTS = (32, 64, 128, 256)
SAMPLES_PER_CELL = 31
TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
MAX_VALIDATION_MAPE_PERCENT = 2.0
MAX_VALIDATION_MAX_APE_PERCENT = 5.0


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_int(text, label, *, positive=False):
    if not text.isascii() or not text.isdecimal():
        raise ValueError(f"{label} must be a canonical integer")
    value = int(text)
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def parse_trial(path, expected_trial, expected_device_id):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    metadata = rows[0]
    if (len(metadata) != 11
            or tuple(metadata[:3]) != ("metadata", "1", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    values = tuple(
        _canonical_int(value, f"metadata[{index}]", positive=(index != 3))
        for index, value in enumerate(metadata[3:], start=3))
    (device_id, ap_count, wave_size, clock_khz, trial, samples_per_cell,
     resident_ctas, dynamic_shared_bytes) = values
    if (device_id != expected_device_id or trial != expected_trial
            or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or samples_per_cell != SAMPLES_PER_CELL
            or resident_ctas != RESIDENT_CTAS_PER_AP
            or dynamic_shared_bytes != DYNAMIC_SHARED_BYTES):
        raise ValueError(f"{path} changed the frozen campaign contract")

    expected_count = len(GRIDS) * len(REPETITION_COUNTS) * SAMPLES_PER_CELL
    if len(rows[1:]) != expected_count:
        raise ValueError(
            f"{path} has {len(rows[1:])} samples, expected {expected_count}")
    orders = set()
    identities = set()
    samples = {}
    for row in rows[1:]:
        if len(row) != 8 or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = _canonical_int(row[1], "sample.order")
        grid = _canonical_int(row[2], "sample.grid", positive=True)
        repetitions = _canonical_int(
            row[3], "sample.repetitions", positive=True)
        sample = _canonical_int(row[4], "sample.sample")
        minimum = _canonical_int(row[5], "sample.minimum", positive=True)
        median = _canonical_int(row[6], "sample.median", positive=True)
        maximum = _canonical_int(row[7], "sample.maximum", positive=True)
        identity = (grid, repetitions, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} contains duplicate samples")
        if (grid not in GRIDS or repetitions not in REPETITION_COUNTS
                or sample >= SAMPLES_PER_CELL or not minimum <= median <= maximum):
            raise ValueError(f"{path} has a sample outside the campaign")
        orders.add(order)
        identities.add(identity)
        key = (grid, repetitions)
        if key not in samples:
            samples[key] = []
        samples[key].append(median)
    if orders != set(range(expected_count)):
        raise ValueError(f"{path} randomized order is not contiguous")
    expected_identities = {
        (grid, repetitions, sample)
        for grid in GRIDS
        for repetitions in REPETITION_COUNTS
        for sample in range(SAMPLES_PER_CELL)
    }
    if identities != expected_identities:
        raise ValueError(f"{path} sample matrix is incomplete")
    return {
        "clock_khz": clock_khz,
        "stdout_sha256": _sha256(path),
        "cell_medians_cycles": {
            key: statistics.median(values) for key, values in samples.items()
        },
    }


def _derive_line(parsed_trials, grid):
    trial_services = []
    trial_intercepts = []
    for trial in DERIVATION_TRIALS:
        cells = parsed_trials[trial]["cell_medians_cycles"]
        slopes = [
            (cells[(grid, upper)] - cells[(grid, lower)]) / (upper - lower)
            for lower, upper in zip(
                REPETITION_COUNTS, REPETITION_COUNTS[1:])
        ]
        if any(value <= 0 for value in slopes):
            raise ValueError("barrier adjacent-count slopes must be positive")
        service = statistics.median(slopes)
        intercept = statistics.median(
            cells[(grid, repetitions)] - service * repetitions
            for repetitions in REPETITION_COUNTS)
        trial_services.append(service)
        trial_intercepts.append(intercept)
    return {
        "grid": grid,
        "barrier_cycles_per_cta": statistics.median(trial_services),
        "fixed_cycles_per_cta": statistics.median(trial_intercepts),
        "derivation_trial_barrier_cycles": trial_services,
    }


def _metrics(rows):
    errors = [
        abs(row["predicted_cycles"] - row["measured_cycles"])
        / row["measured_cycles"] * 100.0
        for row in rows
    ]
    return {
        "count": len(rows),
        "mape_percent": statistics.fmean(errors),
        "max_ape_percent": max(errors),
    }


def aggregate_trials(trial_paths, expected_device_id, source_path,
                     baseline_commit):
    if (not isinstance(baseline_commit, str) or len(baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in baseline_commit)):
        raise ValueError("baseline commit must be a lowercase SHA-1")
    if len(trial_paths) != len(TRIALS):
        raise ValueError("exactly three trial paths are required")
    parsed = {
        trial: parse_trial(path, trial, expected_device_id)
        for trial, path in zip(TRIALS, trial_paths)
    }
    if len({parsed[trial]["clock_khz"] for trial in TRIALS}) != 1:
        raise ValueError("reported device clock changed across trials")
    parameters = tuple(_derive_line(parsed, grid) for grid in GRIDS)
    by_grid = {entry["grid"]: entry for entry in parameters}
    validation_rows = []
    for grid in GRIDS:
        parameter = by_grid[grid]
        for repetitions in REPETITION_COUNTS:
            measured = parsed[VALIDATION_TRIAL]["cell_medians_cycles"][
                (grid, repetitions)]
            predicted = (
                parameter["fixed_cycles_per_cta"]
                + repetitions * parameter["barrier_cycles_per_cta"])
            validation_rows.append({
                "grid": grid,
                "repetitions": repetitions,
                "measured_cycles": measured,
                "predicted_cycles": predicted,
            })
    validation = _metrics(validation_rows)
    publishable = (
        validation["mape_percent"] <= MAX_VALIDATION_MAPE_PERCENT
        and validation["max_ape_percent"] <= MAX_VALIDATION_MAX_APE_PERCENT)
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "software_revision": SOFTWARE_REVISION,
        "mechanism": MECHANISM,
        "production_or_evaluation_duration_consumed": False,
        "baseline_commit": baseline_commit,
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "trial_stdout_sha256": {
            str(trial): parsed[trial]["stdout_sha256"] for trial in TRIALS
        },
        "campaign": {
            "timer_domain": "clock64_same_cta_cycles",
            "derivation": "adjacent_runtime_repetition_count",
            "threads_per_cta": THREADS_PER_CTA,
            "dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
            "resident_ctas_per_ap": RESIDENT_CTAS_PER_AP,
            "grids": list(GRIDS),
            "repetition_counts": list(REPETITION_COUNTS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
        },
        "parameters": list(parameters),
        "selected_parameter": {
            "grid": 2 * AP_COUNT,
            "barrier_cycles_per_256_thread_cta": by_grid[2 * AP_COUNT][
                "barrier_cycles_per_cta"],
            "scope": "two resident 32-KiB CTAs per AP",
        },
        "validation_rows": validation_rows,
        "validation": validation,
        "thresholds": {
            "maximum_mape_percent": MAX_VALIDATION_MAPE_PERCENT,
            "maximum_max_ape_percent": MAX_VALIDATION_MAX_APE_PERCENT,
        },
        "publishable": publishable,
        "disposition": (
            "publishable" if publishable
            else "not_publishable_without_a_new_predeclared_campaign"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", action="append", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = aggregate_trials(
        args.trial, args.device_id, args.source, args.baseline_commit)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0 if report["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
