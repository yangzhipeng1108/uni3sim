#!/usr/bin/env python3
"""Reduce the C500 scalar-pair copy/admission mechanism campaign.

The campaign deliberately has no production workload input.  It only decides
whether one source-loop round has a grid-invariant completion increment that
could become a hardware mechanism parameter.  A non-invariant result is a
useful rejection: callers must not turn individual grid/payload cells into a
Forward lookup.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = 1
HARDWARE_ID = "metax_c500"
MECHANISM = "scalar_pair_copy_cta_admission"
AP_COUNT = 104
WAVE_SIZE = 64
BLOCK_THREADS = 512
ELEMENT_COUNTS = (512, 1024, 1536, 2048)
GRIDS = (64, 104, 208, 256, 416, 1024)
TRIALS = (1, 2, 3)
SAMPLES_PER_CELL = 17
EXPECTED_LAUNCHES_PER_BATCH = 32


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(text, label, *, positive=False):
    if not text.isascii() or not text.isdecimal():
        raise ValueError(f"{label} must be a canonical integer")
    value = int(text)
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _parse_trial(path, expected_trial):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    expected_samples = (
        len(ELEMENT_COUNTS) * len(GRIDS) * SAMPLES_PER_CELL)
    if len(rows) != expected_samples + 1:
        raise ValueError(
            f"{path} has {len(rows) - 1} samples, expected "
            f"{expected_samples}")
    metadata = rows[0]
    if (len(metadata) != 12
            or tuple(metadata[:3]) != ("metadata", "1", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    values = tuple(
        _integer(value, f"metadata[{index}]", positive=index not in (3, 11))
        for index, value in enumerate(metadata[3:], start=3))
    (device_id, ap_count, wave_size, clock_khz, trial, launches_per_batch,
     registers_per_thread, resident_ctas_per_ap, static_shared_bytes) = values
    if (device_id < 0 or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or clock_khz <= 0 or trial != expected_trial
            or launches_per_batch != EXPECTED_LAUNCHES_PER_BATCH
            or registers_per_thread <= 0 or resident_ctas_per_ap <= 0):
        raise ValueError(f"{path} changed the frozen mechanism contract")
    samples = {}
    orders = set()
    identities = set()
    for row in rows[1:]:
        if len(row) != 6 or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = _integer(row[1], "sample.order")
        elements = _integer(row[2], "sample.elements", positive=True)
        grid = _integer(row[3], "sample.grid", positive=True)
        sample = _integer(row[4], "sample.sample")
        if elements not in ELEMENT_COUNTS or grid not in GRIDS:
            raise ValueError(f"{path} has a sample outside the campaign")
        if sample >= SAMPLES_PER_CELL:
            raise ValueError(f"{path} has an invalid sample ordinal")
        duration_ms = float.fromhex(row[5])
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError(f"{path} has a non-positive duration")
        identity = (elements, grid, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} has a duplicate sample")
        orders.add(order)
        identities.add(identity)
        samples.setdefault((elements, grid), []).append(
            duration_ms * 1_000_000.0 / launches_per_batch)
    expected_identities = {
        (elements, grid, sample)
        for elements in ELEMENT_COUNTS
        for grid in GRIDS
        for sample in range(SAMPLES_PER_CELL)
    }
    if (orders != set(range(expected_samples))
            or identities != expected_identities
            or any(len(values) != SAMPLES_PER_CELL
                   for values in samples.values())):
        raise ValueError(f"{path} has an incomplete randomized campaign")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "device_id": device_id,
        "clock_khz": clock_khz,
        "registers_per_thread": registers_per_thread,
        "resident_ctas_per_ap": resident_ctas_per_ap,
        "static_shared_bytes": static_shared_bytes,
        "median_ns_by_cell": {
            key: statistics.median(values) for key, values in samples.items()},
    }


def _spread_percent(values):
    midpoint = statistics.median(values)
    if midpoint <= 0:
        raise ValueError("non-positive midpoint")
    return (max(values) - min(values)) / midpoint * 100.0


def _report(parsed_trials, baseline_commit, source):
    if len(parsed_trials) != len(TRIALS):
        raise ValueError("exactly three trial files are required")
    fields = (
        "device_id", "clock_khz", "registers_per_thread",
        "resident_ctas_per_ap", "static_shared_bytes")
    for field in fields:
        if len({parsed[field] for parsed in parsed_trials.values()}) != 1:
            raise ValueError(f"campaign {field} changed across trials")
    cells = []
    for elements in ELEMENT_COUNTS:
        for grid in GRIDS:
            values = [
                parsed_trials[trial]["median_ns_by_cell"][(elements, grid)]
                for trial in TRIALS]
            cells.append({
                "elements_per_cta": elements,
                "source_loop_rounds": elements // BLOCK_THREADS,
                "grid": grid,
                "trial_median_ns": values,
                "cross_trial_median_ns": statistics.median(values),
                "cross_trial_spread_percent": _spread_percent(values),
            })
    increments = []
    for grid in GRIDS:
        medians = {
            elements: statistics.median(
                parsed_trials[trial]["median_ns_by_cell"][(elements, grid)]
                for trial in TRIALS)
            for elements in ELEMENT_COUNTS}
        for lower, upper in zip(ELEMENT_COUNTS, ELEMENT_COUNTS[1:]):
            increments.append({
                "grid": grid,
                "from_round": lower // BLOCK_THREADS,
                "to_round": upper // BLOCK_THREADS,
                "increment_ns": medians[upper] - medians[lower],
            })
    increment_spreads = []
    for from_round in (1, 2, 3):
        values = [
            row["increment_ns"] for row in increments
            if row["from_round"] == from_round]
        increment_spreads.append({
            "from_round": from_round,
            "to_round": from_round + 1,
            "increment_ns_by_grid": values,
            "cross_grid_spread_percent": _spread_percent(values),
        })
    # A hardware-wide source-round parameter must be invariant across the
    # independent launch grids.  This campaign reports the actual spread
    # rather than choosing a production-facing tolerance or a target case.
    grid_invariant_round_increment = all(
        math.isclose(
            row["increment_ns_by_grid"][0],
            value,
            rel_tol=0.0,
            abs_tol=0.0)
        for row in increment_spreads
        for value in row["increment_ns_by_grid"][1:])
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "production_or_evaluation_duration_consumed": False,
        "baseline_commit": baseline_commit,
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
        },
        "trial_sha256": {
            str(trial): parsed_trials[trial]["sha256"] for trial in TRIALS},
        "resource_usage": {
            field: parsed_trials[TRIALS[0]][field]
            for field in fields},
        "campaign": {
            "block_threads": BLOCK_THREADS,
            "element_counts": list(ELEMENT_COUNTS),
            "grids": list(GRIDS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "launches_per_batch": EXPECTED_LAUNCHES_PER_BATCH,
            "trials": list(TRIALS),
        },
        "cells": cells,
        "round_increments": increments,
        "round_increment_spreads": increment_spreads,
        "grid_invariant_round_increment": grid_invariant_round_increment,
        "publishable_hardware_parameter": False,
        "disposition": (
            "rejected_no_grid_invariant_scalar_round_service"
            if not grid_invariant_round_increment
            else "requires_independent_contract_review"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", action="append", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.trial) != len(TRIALS):
        raise ValueError("exactly three --trial paths are required")
    parsed = {
        trial: _parse_trial(path, trial)
        for trial, path in zip(TRIALS, args.trial)}
    report = _report(parsed, args.baseline_commit, args.source)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
