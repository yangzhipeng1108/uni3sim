#!/usr/bin/env python3
"""Audit MetaX C500 statically unrolled primitive dependency captures."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = 2
HARDWARE_ID = "metax_c500"
MECHANISM = "clock64_unrolled_sfu_dependency"
SOFTWARE_REVISION = "maca_3.7.2"
PRIMITIVES = (
    "exp_f32", "rcp_f32", "rsq_f32", "rcpi_f32",
)
COUNTS = (4, 8, 16, 32, 64)
SAMPLES_PER_CELL = 31
FULL_TRIALS = (1, 2, 3)
TRAINING_TRIALS = (1, 2)
VALIDATION_TRIALS = (3,)
XCORE_HZ = 1_125_000_000.0
MAX_ADJACENT_SLOPE_DEVIATION_CYCLES = 1.0


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_stdout(path, expected_trial, expected_device_id):
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    expected_samples = len(PRIMITIVES) * len(COUNTS) * SAMPLES_PER_CELL
    if len(rows) != expected_samples + 1:
        raise ValueError("primitive dependency stdout row count is invalid")
    metadata = rows[0]
    expected_metadata = (
        "metadata", "1", "MetaX C500", str(expected_device_id), "104",
        "64", metadata[6], str(expected_trial), str(SAMPLES_PER_CELL),
    )
    if (len(metadata) != len(expected_metadata)
            or tuple(metadata) != expected_metadata
            or not metadata[6].isdigit() or int(metadata[6]) <= 0):
        raise ValueError("primitive dependency metadata is invalid")
    samples = []
    identities = set()
    for ordinal, row in enumerate(rows[1:]):
        if len(row) != 6 or row[0] != "sample":
            raise ValueError("primitive dependency sample schema is invalid")
        if not all(value.isdigit() for value in (row[1], row[3], row[4], row[5])):
            raise ValueError("primitive dependency sample integers are invalid")
        order = int(row[1])
        primitive = row[2]
        count = int(row[3])
        sample = int(row[4])
        elapsed_cycles = int(row[5])
        if order != ordinal:
            raise ValueError("primitive dependency order is not contiguous")
        if primitive not in PRIMITIVES or count not in COUNTS:
            raise ValueError("primitive dependency configuration is unknown")
        if sample < 0 or sample >= SAMPLES_PER_CELL or elapsed_cycles <= 0:
            raise ValueError("primitive dependency sample value is invalid")
        identity = (primitive, count, sample)
        if identity in identities:
            raise ValueError("primitive dependency sample is duplicated")
        identities.add(identity)
        samples.append({
            "order": order,
            "primitive": primitive,
            "count": count,
            "sample": sample,
            "elapsed_cycles": elapsed_cycles,
        })
    expected_identities = {
        (primitive, count, sample)
        for primitive in PRIMITIVES
        for count in COUNTS
        for sample in range(SAMPLES_PER_CELL)
    }
    if identities != expected_identities:
        raise ValueError("primitive dependency configuration set is incomplete")
    return tuple(samples), int(metadata[6])


def build_report(stdout_path, trial, device_id):
    if trial not in FULL_TRIALS:
        raise ValueError("primitive dependency trial must be 1, 2, or 3")
    if isinstance(device_id, bool) or not isinstance(device_id, int):
        raise TypeError("primitive dependency device ID must be an integer")
    if device_id < 0:
        raise ValueError("primitive dependency device ID cannot be negative")
    samples, reported_clock_khz = parse_stdout(
        stdout_path, trial, device_id)
    cells = []
    for primitive in PRIMITIVES:
        for count in COUNTS:
            values = [
                sample["elapsed_cycles"] for sample in samples
                if sample["primitive"] == primitive
                and sample["count"] == count
            ]
            cells.append({
                "primitive": primitive,
                "count": count,
                "sample_count": len(values),
                "median_elapsed_cycles": statistics.median(values),
                "minimum_elapsed_cycles": min(values),
                "maximum_elapsed_cycles": max(values),
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "software_revision": SOFTWARE_REVISION,
        "timer_domain": "device_clock64_xcore_cycles",
        "production_or_evaluation_duration_consumed": False,
        "trial": trial,
        "physical_device_id": device_id,
        "reported_clock_khz": reported_clock_khz,
        "xcore_hz": XCORE_HZ,
        "stdout": {"path": str(stdout_path), "sha256": _sha256(stdout_path)},
        "samples": list(samples),
        "cells": cells,
    }


def _load_report(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "hardware_id", "mechanism", "software_revision",
        "timer_domain", "production_or_evaluation_duration_consumed",
        "trial", "physical_device_id", "reported_clock_khz", "xcore_hz",
        "stdout", "samples", "cells",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("primitive dependency report schema is invalid")
    if (payload["schema_version"] != SCHEMA_VERSION
            or payload["hardware_id"] != HARDWARE_ID
            or payload["mechanism"] != MECHANISM
            or payload["software_revision"] != SOFTWARE_REVISION
            or payload["timer_domain"] != "device_clock64_xcore_cycles"
            or payload["production_or_evaluation_duration_consumed"]
            is not False or payload["xcore_hz"] != XCORE_HZ):
        raise ValueError("primitive dependency report contract is invalid")
    return payload


def _medians(reports, trials):
    values = {
        (primitive, count): []
        for primitive in PRIMITIVES for count in COUNTS
    }
    for report in reports:
        if report["trial"] not in trials:
            continue
        for cell in report["cells"]:
            identity = (cell["primitive"], cell["count"])
            if identity not in values:
                raise ValueError("primitive dependency cell is unknown")
            if cell["sample_count"] != SAMPLES_PER_CELL:
                raise ValueError("primitive dependency cell is incomplete")
            values[identity].append(cell["median_elapsed_cycles"])
    expected_per_cell = len(trials)
    if any(len(cell_values) != expected_per_cell
           for cell_values in values.values()):
        raise ValueError("primitive dependency trial set is incomplete")
    return {
        identity: statistics.median(cell_values)
        for identity, cell_values in values.items()
    }


def _primitive_diagnostic(medians, primitive):
    values = {count: medians[(primitive, count)] for count in COUNTS}
    canonical = (values[32] - values[4]) / 28.0
    adjacent = {
        "u4_u8": (values[8] - values[4]) / 4.0,
        "u8_u16": (values[16] - values[8]) / 8.0,
        "u16_u32": (values[32] - values[16]) / 16.0,
        "u32_u64": (values[64] - values[32]) / 32.0,
    }
    deviations = {
        name: abs(slope - canonical) for name, slope in adjacent.items()
    }
    return {
        "median_elapsed_cycles": {
            str(count): values[count] for count in COUNTS},
        "canonical_span": "u4_to_u32_fixed",
        "latency_xcore_cycles_per_dependent_wave_opcode": canonical,
        "latency_ns_per_dependent_wave_opcode": canonical / XCORE_HZ * 1e9,
        "adjacent_slopes": adjacent,
        "maximum_adjacent_deviation_cycles": max(deviations.values()),
        "stable": (
            math.isfinite(canonical) and canonical > 0
            and max(deviations.values())
            <= MAX_ADJACENT_SLOPE_DEVIATION_CYCLES),
    }


def aggregate_reports(paths):
    reports = [_load_report(path) for path in paths]
    actual_trials = tuple(sorted(report["trial"] for report in reports))
    if actual_trials != FULL_TRIALS or len(reports) != len(FULL_TRIALS):
        raise ValueError(
            "primitive dependency aggregate requires trials 1, 2, and 3")
    device_ids = {report["physical_device_id"] for report in reports}
    if len(device_ids) != 1:
        raise ValueError("primitive dependency aggregate mixes devices")
    training_medians = _medians(reports, TRAINING_TRIALS)
    validation_medians = _medians(reports, VALIDATION_TRIALS)
    training = {
        primitive: _primitive_diagnostic(training_medians, primitive)
        for primitive in PRIMITIVES
    }
    validation = {
        primitive: _primitive_diagnostic(validation_medians, primitive)
        for primitive in PRIMITIVES
    }
    validation_differences = {
        primitive: abs(
            validation[primitive][
                "latency_xcore_cycles_per_dependent_wave_opcode"]
            - training[primitive][
                "latency_xcore_cycles_per_dependent_wave_opcode"])
        for primitive in PRIMITIVES
    }
    publishable = (
        all(diagnostic["stable"] for diagnostic in training.values())
        and all(diagnostic["stable"] for diagnostic in validation.values())
        and all(value <= MAX_ADJACENT_SLOPE_DEVIATION_CYCLES
                for value in validation_differences.values()))
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "software_revision": SOFTWARE_REVISION,
        "timer_domain": "device_clock64_xcore_cycles",
        "production_or_evaluation_duration_consumed": False,
        "physical_device_id": next(iter(device_ids)),
        "run_reports": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "training": training,
        "frozen_validation": validation,
        "validation_latency_difference_cycles": validation_differences,
        "publication_gate": {
            "maximum_adjacent_slope_deviation_cycles": (
                MAX_ADJACENT_SLOPE_DEVIATION_CYCLES),
            "maximum_training_validation_difference_cycles": (
                MAX_ADJACENT_SLOPE_DEVIATION_CYCLES),
        },
        "publishable": publishable,
        "disposition": (
            "eligible_primitive_dependency_latency" if publishable
            else "not_publishable_without_a_new_predeclared_campaign"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="C500 primitive dependency mechanism audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--stdout", type=Path, required=True)
    analyze.add_argument("--trial", type=int, required=True)
    analyze.add_argument("--device-id", type=int, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--report", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        payload = build_report(args.stdout, args.trial, args.device_id)
    elif args.command == "aggregate":
        payload = aggregate_reports(args.report)
    else:
        raise AssertionError(f"unknown primitive dependency command: {args.command!r}")
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
