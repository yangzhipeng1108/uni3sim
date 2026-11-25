#!/usr/bin/env python3
"""Validate C500 CTA-local AddRMSNorm grid64 input-loop trials."""

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
MECHANISM = "bf16_add_rmsnorm_grid64_fixed_address_input_loop_cycles"
AP_COUNT = 104
WAVE_SIZE = 64
THREADS = 1024
GRIDS = (64,)
DEPTHS = (4, 5, 8, 16, 32)
ADDRESS_MODES = ("same", "progressing")
SAMPLES_PER_CELL = 31
TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
MODEL_CLOCK_HZ = 1_125_000_000
MAXIMUM_SPREAD_PERCENT = 20.0
MAXIMUM_VALIDATION_APE_PERCENT = 20.0


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


def parse_stdout(path, expected_trial, expected_device_id):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    metadata = rows[0]
    if (len(metadata) != 10
            or tuple(metadata[:3]) != ("metadata", "1", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    values = tuple(
        _canonical_int(value, f"metadata[{index}]", positive=True)
        for index, value in enumerate(metadata[3:], start=3))
    (device_id, ap_count, wave_size, reported_clock_khz, trial,
     samples_per_cell, resident_ctas) = values
    if (device_id != expected_device_id or trial != expected_trial
            or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or samples_per_cell != SAMPLES_PER_CELL
            or resident_ctas != 2):
        raise ValueError(f"{path} changed the frozen campaign contract")
    expected_identities = {
        (grid, depth, mode, sample)
        for grid in GRIDS
        for depth in DEPTHS
        for mode in ADDRESS_MODES
        for sample in range(SAMPLES_PER_CELL)
    }
    identities = set()
    orders = set()
    maximum_cycles = {
        (grid, depth, mode): []
        for grid in GRIDS
        for depth in DEPTHS
        for mode in ADDRESS_MODES
    }
    for row in rows[1:]:
        if len(row) != 9 or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = _canonical_int(row[1], "sample.order")
        grid = _canonical_int(row[2], "sample.grid", positive=True)
        depth = _canonical_int(row[3], "sample.depth", positive=True)
        mode = row[4]
        sample = _canonical_int(row[5], "sample.sample")
        minimum = _canonical_int(row[6], "sample.minimum", positive=True)
        median = _canonical_int(row[7], "sample.median", positive=True)
        maximum = _canonical_int(row[8], "sample.maximum", positive=True)
        identity = (grid, depth, mode, sample)
        if identity not in expected_identities:
            raise ValueError(f"{path} has a sample outside the campaign")
        if identity in identities or order in orders:
            raise ValueError(f"{path} contains duplicate samples")
        if not minimum <= median <= maximum:
            raise ValueError(f"{path} has unordered CTA cycle statistics")
        identities.add(identity)
        orders.add(order)
        maximum_cycles[(grid, depth, mode)].append(maximum)
    if (identities != expected_identities
            or orders != set(range(len(expected_identities)))):
        raise ValueError(f"{path} has an incomplete randomized matrix")
    return {
        "reported_clock_khz": reported_clock_khz,
        "cell_median_maximum_cycles": {
            key: statistics.median(values)
            for key, values in maximum_cycles.items()},
        "stdout_sha256": _sha256(path),
    }


def _relative_spread(values):
    return (max(values) - min(values)) / statistics.median(values) * 100.0


def _validation_ape(derived, validation):
    return abs(derived - validation) / validation * 100.0


def aggregate_trials(
        trial_paths, expected_device_id, source_path, binary_path,
        baseline_commit):
    if (len(baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in baseline_commit)):
        raise ValueError("baseline commit must be a lowercase SHA-1")
    if len(trial_paths) != len(TRIALS):
        raise ValueError("exactly three trial paths are required")
    parsed = {
        trial: parse_stdout(path, trial, expected_device_id)
        for trial, path in zip(TRIALS, trial_paths)}
    cells = []
    publishable = True
    for grid in GRIDS:
        for depth in DEPTHS:
            progressing = [
                parsed[trial]["cell_median_maximum_cycles"][
                    (grid, depth, "progressing")]
                for trial in TRIALS]
            same = [
                parsed[trial]["cell_median_maximum_cycles"][
                    (grid, depth, "same")]
                for trial in TRIALS]
            excess = [
                progressing[index] - same[index]
                for index in range(len(TRIALS))]
            derived_same = statistics.median(
                same[trial - 1] for trial in DERIVATION_TRIALS)
            derived_progressing = statistics.median(
                progressing[trial - 1] for trial in DERIVATION_TRIALS)
            derived_excess = statistics.median(
                excess[trial - 1] for trial in DERIVATION_TRIALS)
            if min(same) <= 0:
                same_spread = math.inf
                same_ape = math.inf
            else:
                same_spread = _relative_spread(same)
                same_ape = _validation_ape(
                    derived_same, same[VALIDATION_TRIAL - 1])
            if min(progressing) <= 0 or min(excess) <= 0:
                progressing_spread = math.inf
                excess_spread = math.inf
                progressing_ape = math.inf
                excess_ape = math.inf
            else:
                progressing_spread = _relative_spread(progressing)
                excess_spread = _relative_spread(excess)
                progressing_ape = _validation_ape(
                    derived_progressing, progressing[VALIDATION_TRIAL - 1])
                excess_ape = _validation_ape(
                    derived_excess, excess[VALIDATION_TRIAL - 1])
            cell_passed = (
                min(same) > 0
                and same_spread <= MAXIMUM_SPREAD_PERCENT
                and same_ape <= MAXIMUM_VALIDATION_APE_PERCENT)
            publishable = publishable and cell_passed
            cells.append({
                "grid": grid,
                "threads": THREADS,
                "depth": depth,
                "same_maximum_cycles": same,
                "derived_same_cycles": derived_same,
                "derived_same_ns": (
                    derived_same * 1.0e9 / MODEL_CLOCK_HZ),
                "same_spread_percent": same_spread,
                "same_validation_ape_percent": same_ape,
                "progressing_maximum_cycles": progressing,
                "progression_excess_cycles": excess,
                "derived_progressing_cycles": derived_progressing,
                "derived_progressing_ns": (
                    derived_progressing * 1.0e9 / MODEL_CLOCK_HZ),
                "derived_progression_excess_cycles": derived_excess,
                "derived_progression_excess_ns": (
                    derived_excess * 1.0e9 / MODEL_CLOCK_HZ),
                "progressing_spread_percent": progressing_spread,
                "progression_excess_spread_percent": excess_spread,
                "progressing_validation_ape_percent": progressing_ape,
                "progression_excess_validation_ape_percent": excess_ape,
                "publishable": cell_passed,
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "software_revision": SOFTWARE_REVISION,
        "mechanism": MECHANISM,
        "production_or_evaluation_duration_consumed": False,
        "baseline_commit": baseline_commit,
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "binary": {"path": str(binary_path), "sha256": _sha256(binary_path)},
        "trial_stdout_sha256": {
            str(trial): parsed[trial]["stdout_sha256"] for trial in TRIALS},
        "campaign": {
            "timer_domain": "clock64_cta_local_cycles",
            "model_clock_hz": MODEL_CLOCK_HZ,
            "device_id": expected_device_id,
            "threads": THREADS,
            "grids": list(GRIDS),
            "depths": list(DEPTHS),
            "address_modes": list(ADDRESS_MODES),
            "samples_per_cell": SAMPLES_PER_CELL,
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
        },
        "cells": cells,
        "thresholds": {
            "maximum_cross_trial_spread_percent": MAXIMUM_SPREAD_PERCENT,
            "maximum_validation_ape_percent": MAXIMUM_VALIDATION_APE_PERCENT,
        },
        "publishable": publishable,
        "disposition": (
            "publishable" if publishable
            else "rejected_without_selective_cell_publication"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", action="append", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = aggregate_trials(
        args.trial, args.device_id, args.source, args.binary,
        args.baseline_commit)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0 if report["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
