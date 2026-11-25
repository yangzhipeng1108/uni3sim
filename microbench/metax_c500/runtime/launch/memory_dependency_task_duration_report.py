#!/usr/bin/env python3
"""Audit the C500 multi-wave memory-dependency task campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPECTED_TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
EXPECTED_AP_COUNT = 104
EXPECTED_WAVE_SIZE = 64
EXPECTED_SAMPLES = 31
EXPECTED_LAUNCHES_PER_BATCH = 128
EXPECTED_MAXIMUM_GRID_SIZE = 16384
EXPECTED_MAXIMUM_LAUNCH_RANK = 2
RANK1_GRIDS = (
    1, 16, 64, 104, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384,
)
RANK1_BLOCKS = (64, 256, 512, 1024)
RANK2_GEOMETRIES = (
    (2, 2, 512),
    (4, 8, 512),
    (8, 16, 512),
    (16, 32, 512),
    (16, 128, 512),
    (16, 512, 512),
)
TRAINING_MAPE_LIMIT_PERCENT = 10.0
TRAINING_WMAE_LIMIT_PERCENT = 10.0
VALIDATION_MAPE_LIMIT_PERCENT = 10.0
VALIDATION_WMAE_LIMIT_PERCENT = 10.0
VALIDATION_MAXIMUM_APE_LIMIT_PERCENT = 20.0


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_geometries():
    rank1 = tuple(
        (grid, 1, block)
        for grid in RANK1_GRIDS
        for block in RANK1_BLOCKS)
    return rank1 + RANK2_GEOMETRIES


def _parse_trial(path, expected_trial):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 3:
        raise ValueError(f"{path} is incomplete")
    metadata = rows[0]
    if (len(metadata) != 12 or metadata[0:3] != [
            "metadata", "5", "MetaX C500"]):
        raise ValueError(f"{path} has an invalid metadata row")
    device_id = int(metadata[3])
    ap_count = int(metadata[4])
    wave_size = int(metadata[5])
    clock_khz = int(metadata[6])
    trial = int(metadata[7])
    samples_per_cell = int(metadata[8])
    launches_per_batch = int(metadata[9])
    maximum_grid_size = int(metadata[10])
    maximum_launch_rank = int(metadata[11])
    if (ap_count != EXPECTED_AP_COUNT
            or wave_size != EXPECTED_WAVE_SIZE
            or clock_khz <= 0
            or trial != expected_trial
            or samples_per_cell != EXPECTED_SAMPLES
            or launches_per_batch != EXPECTED_LAUNCHES_PER_BATCH
            or maximum_grid_size != EXPECTED_MAXIMUM_GRID_SIZE
            or maximum_launch_rank != EXPECTED_MAXIMUM_LAUNCH_RANK):
        raise ValueError(f"{path} changed the frozen campaign contract")
    expected_header = [
        "sample_header", "order", "grid_x", "grid_y", "grid_z",
        "grid_size", "block", "repeat", "pair_order",
        "host_start_ns", "host_end_ns", "empty_batch_ns",
        "dependent_batch_ns", "empty_per_launch_ns",
        "dependent_per_launch_ns", "increment_per_launch_ns",
        "empty_first_elapsed_ms_hex", "empty_second_elapsed_ms_hex",
        "dependent_first_elapsed_ms_hex",
        "dependent_second_elapsed_ms_hex", "empty_event_status",
        "dependent_event_status",
    ]
    if rows[1] != expected_header:
        raise ValueError(f"{path} has an invalid sample header")
    expected_geometries = _expected_geometries()
    expected_row_count = len(expected_geometries) * EXPECTED_SAMPLES
    if len(rows[2:]) != expected_row_count:
        raise ValueError(
            f"{path} has {len(rows[2:])} samples, expected "
            f"{expected_row_count}")
    orders = set()
    identities = set()
    values = {}
    pair_values = {}
    for row in rows[2:]:
        if len(row) != 22 or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = int(row[1])
        grid = (int(row[2]), int(row[3]), int(row[4]))
        grid_size = int(row[5])
        block = int(row[6])
        repeat = int(row[7])
        pair_order = row[8]
        host_start_ns = int(row[9])
        host_end_ns = int(row[10])
        empty_batch_ns = float(row[11])
        dependent_batch_ns = float(row[12])
        empty_per_launch_ns = float(row[13])
        dependent_per_launch_ns = float(row[14])
        increment_per_launch_ns = float(row[15])
        event_ms = tuple(float.fromhex(value) for value in row[16:20])
        expected_pair_order = (
            "dependent_first"
            if ((order + expected_trial) & 1) else "empty_first")
        geometry = (grid[0], grid[1], block)
        identity = (geometry, repeat)
        if (order in orders or identity in identities):
            raise ValueError(f"{path} has a duplicate sample")
        if (grid[2] != 1 or math.prod(grid) != grid_size
                or geometry not in expected_geometries
                or repeat < 0 or repeat >= EXPECTED_SAMPLES
                or pair_order != expected_pair_order
                or host_end_ns <= host_start_ns
                or row[20:] != ["resolved", "resolved"]):
            raise ValueError(f"{path} has an invalid sample identity")
        numeric_values = (
            empty_batch_ns, dependent_batch_ns, empty_per_launch_ns,
            dependent_per_launch_ns, increment_per_launch_ns, *event_ms)
        if (any(not math.isfinite(value) for value in numeric_values)
                or any(value <= 0 for value in numeric_values[:4])
                or any(value <= 0 for value in event_ms)):
            raise ValueError(f"{path} has an invalid timing value")
        expected_empty_batch_ns = (event_ms[0] + event_ms[1]) * 500000.0
        expected_dependent_batch_ns = (
            (event_ms[2] + event_ms[3]) * 500000.0)
        checks = (
            (empty_batch_ns, expected_empty_batch_ns),
            (dependent_batch_ns, expected_dependent_batch_ns),
            (empty_per_launch_ns,
             empty_batch_ns / EXPECTED_LAUNCHES_PER_BATCH),
            (dependent_per_launch_ns,
             dependent_batch_ns / EXPECTED_LAUNCHES_PER_BATCH),
            (increment_per_launch_ns,
             dependent_per_launch_ns - empty_per_launch_ns),
        )
        if any(not math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-3)
               for left, right in checks):
            raise ValueError(f"{path} has inconsistent event arithmetic")
        orders.add(order)
        identities.add(identity)
        # The HWSpec model is the complete dependent-task duration.  The
        # paired empty task is retained only as the independent positivity
        # gate for the dependency increment; subtracting it here would make
        # Formula subtract the empty task a second time.
        if geometry not in values:
            values[geometry] = []
        values[geometry].append(dependent_per_launch_ns)
        pair_identity = (geometry, pair_order)
        if pair_identity not in pair_values:
            pair_values[pair_identity] = []
        pair_values[pair_identity].append(increment_per_launch_ns)
    if orders != set(range(expected_row_count)):
        raise ValueError(f"{path} has a non-contiguous randomized order")
    cell_medians = {
        geometry: statistics.median(samples)
        for geometry, samples in values.items()
    }
    pair_medians = {
        identity: statistics.median(samples)
        for identity, samples in pair_values.items()
    }
    return {
        "device_id": device_id,
        "clock_khz": clock_khz,
        "cell_medians": cell_medians,
        "pair_increment_medians": pair_medians,
    }


def _solve_three_parameter_least_squares(rows):
    normal = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for features, measured in rows:
        for row in range(3):
            rhs[row] += features[row] * measured
            for column in range(3):
                normal[row][column] += features[row] * features[column]
    augmented = [normal[row] + [rhs[row]] for row in range(3)]
    for pivot in range(3):
        selected = max(range(pivot, 3), key=lambda row: abs(
            augmented[row][pivot]))
        if abs(augmented[selected][pivot]) < 1e-12:
            raise ValueError("campaign design matrix is singular")
        augmented[pivot], augmented[selected] = (
            augmented[selected], augmented[pivot])
        divisor = augmented[pivot][pivot]
        augmented[pivot] = [value / divisor for value in augmented[pivot]]
        for row in range(3):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[pivot])]
    return tuple(augmented[index][3] for index in range(3))


def _features(geometry):
    grid_x, grid_y, block = geometry
    grid_size = grid_x * grid_y
    full_chip_rounds = math.ceil(grid_size / EXPECTED_AP_COUNT)
    waves_per_cta = math.ceil(block / EXPECTED_WAVE_SIZE)
    resident_ctas = min(grid_size, EXPECTED_AP_COUNT)
    return (1.0, float(resident_ctas),
            float(full_chip_rounds * waves_per_cta))


def _metrics(rows, parameters):
    evaluated = []
    for trial, geometry, measured in rows:
        predicted = math.fsum(
            coefficient * feature
            for coefficient, feature in zip(parameters, _features(geometry)))
        if measured <= 0 or predicted <= 0:
            raise ValueError("task duration must remain positive")
        evaluated.append({
            "trial": trial,
            "grid": [geometry[0], geometry[1], 1],
            "grid_size": geometry[0] * geometry[1],
            "block_size": geometry[2],
            "measured_ns": measured,
            "predicted_ns": predicted,
            "ape_percent": abs(predicted - measured) / measured * 100.0,
        })
    measured_sum = math.fsum(row["measured_ns"] for row in evaluated)
    return {
        "case_count": len(evaluated),
        "mape_percent": statistics.fmean(
            row["ape_percent"] for row in evaluated),
        "wmae_percent": (
            math.fsum(abs(row["predicted_ns"] - row["measured_ns"])
                      for row in evaluated)
            / measured_sum * 100.0),
        "maximum_ape_percent": max(
            row["ape_percent"] for row in evaluated),
        "rows": evaluated,
    }


def build_report(raw_dir, source_snapshot, binary_path):
    trials = {
        trial: _parse_trial(raw_dir / f"trial_{trial}.csv", trial)
        for trial in EXPECTED_TRIALS
    }
    metadata = {
        (item["device_id"], item["clock_khz"])
        for item in trials.values()
    }
    if len(metadata) != 1:
        raise ValueError("trial hardware metadata is inconsistent")
    device_id, clock_khz = metadata.pop()
    rank1_geometries = tuple(
        geometry for geometry in _expected_geometries()
        if geometry[1] == 1)
    derivation_medians = {
        geometry: statistics.median(
            trials[trial]["cell_medians"][geometry]
            for trial in DERIVATION_TRIALS)
        for geometry in rank1_geometries
    }
    training_source = [
        (_features(geometry), derivation_medians[geometry])
        for geometry in rank1_geometries
    ]
    parameters = _solve_three_parameter_least_squares(training_source)
    if any(not math.isfinite(value) or value <= 0 for value in parameters):
        raise ValueError(
            f"fitted physical coefficients must be positive: {parameters}")
    training = _metrics([
        (0, geometry, derivation_medians[geometry])
        for geometry in rank1_geometries
    ], parameters)
    validation = _metrics([
        (VALIDATION_TRIAL, geometry,
         trials[VALIDATION_TRIAL]["cell_medians"][geometry])
        for geometry in _expected_geometries()
    ], parameters)
    rank2_validation = _metrics([
        (VALIDATION_TRIAL, geometry,
         trials[VALIDATION_TRIAL]["cell_medians"][geometry])
        for geometry in RANK2_GEOMETRIES
    ], parameters)
    gates = {
        "coefficients_strictly_positive": all(
            value > 0 for value in parameters),
        "training_mape": (
            training["mape_percent"] < TRAINING_MAPE_LIMIT_PERCENT),
        "training_wmae": (
            training["wmae_percent"] < TRAINING_WMAE_LIMIT_PERCENT),
        "validation_mape": (
            validation["mape_percent"] < VALIDATION_MAPE_LIMIT_PERCENT),
        "validation_wmae": (
            validation["wmae_percent"] < VALIDATION_WMAE_LIMIT_PERCENT),
        "validation_maximum_ape": (
            validation["maximum_ape_percent"]
            < VALIDATION_MAXIMUM_APE_LIMIT_PERCENT),
        "rank2_validation_mape": (
            rank2_validation["mape_percent"]
            < VALIDATION_MAPE_LIMIT_PERCENT),
        "rank2_validation_wmae": (
            rank2_validation["wmae_percent"]
            < VALIDATION_WMAE_LIMIT_PERCENT),
        "rank2_validation_maximum_ape": (
            rank2_validation["maximum_ape_percent"]
            < VALIDATION_MAXIMUM_APE_LIMIT_PERCENT),
    }
    return {
        "schema_version": 2,
        "mechanism": (
            "paired_batched_multirank_warm_default_cache_"
            "memory_dependency_task"),
        "hardware_id": "metax_c500",
        "software_revision": "maca_3.7.2",
        "device_id": device_id,
        "observed_clock_khz": clock_khz,
        "timer_domain": "mcevent_paired_default_stream",
        "pair_pattern": "abba",
        "formula": (
            "base_ns + min(grid_size, 104) * resident_cta_ns + "
            "ceil(grid_size / 104) * ceil(block_size / 64) * "
            "wave_round_ns"),
        "derivation_scope": "rank1_trial_1_2_cell_medians",
        "validation_scope": "all_rank1_and_rank2_trial_3",
        "production_or_evaluation_duration_consumed": False,
        "maximum_grid_size": EXPECTED_MAXIMUM_GRID_SIZE,
        "maximum_launch_rank": EXPECTED_MAXIMUM_LAUNCH_RANK,
        "minimum_block_size": min(RANK1_BLOCKS),
        "maximum_block_size": max(RANK1_BLOCKS),
        "launches_per_batch": EXPECTED_LAUNCHES_PER_BATCH,
        "samples_per_cell": EXPECTED_SAMPLES,
        "parameters": {
            "base_ns": parameters[0],
            "resident_cta_ns": parameters[1],
            "wave_round_ns": parameters[2],
        },
        "publication_gate_limits_percent": {
            "training_mape": TRAINING_MAPE_LIMIT_PERCENT,
            "training_wmae": TRAINING_WMAE_LIMIT_PERCENT,
            "validation_mape": VALIDATION_MAPE_LIMIT_PERCENT,
            "validation_wmae": VALIDATION_WMAE_LIMIT_PERCENT,
            "validation_maximum_ape": (
                VALIDATION_MAXIMUM_APE_LIMIT_PERCENT),
            "rank2_validation_mape": VALIDATION_MAPE_LIMIT_PERCENT,
            "rank2_validation_wmae": VALIDATION_WMAE_LIMIT_PERCENT,
            "rank2_validation_maximum_ape": (
                VALIDATION_MAXIMUM_APE_LIMIT_PERCENT),
        },
        "publication_gates": gates,
        "publishable": all(gates.values()),
        "training": training,
        "validation": validation,
        "rank2_validation": rank2_validation,
        "nonpositive_pair_increment_medians": [
            {
                "trial": trial,
                "grid": [geometry[0], geometry[1], 1],
                "block_size": geometry[2],
                "pair_order": pair_order,
                "increment_ns": increment,
            }
            for trial in EXPECTED_TRIALS
            for (geometry, pair_order), increment in trials[trial][
                "pair_increment_medians"].items()
            if increment <= 0
        ],
        "source_sha256": _sha256(source_snapshot),
        "report_script_sha256": _sha256(Path(__file__)),
        "binary_sha256": _sha256(binary_path),
        "trial_sha256": {
            str(trial): _sha256(raw_dir / f"trial_{trial}.csv")
            for trial in EXPECTED_TRIALS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.raw_dir, args.source_snapshot, args.binary)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        "parameters": report["parameters"],
        "training": {
            key: value for key, value in report["training"].items()
            if key != "rows"
        },
        "validation": {
            key: value for key, value in report["validation"].items()
            if key != "rows"
        },
        "rank2_validation": {
            key: value for key, value in report["rank2_validation"].items()
            if key != "rows"
        },
        "publication_gates": report["publication_gates"],
        "publishable": report["publishable"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not report["publishable"]:
        raise SystemExit("memory-dependency task publication gates failed")


if __name__ == "__main__":
    main()
