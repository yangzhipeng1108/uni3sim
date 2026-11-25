#!/usr/bin/env python3
"""Validate and reduce the C500 runtime-counted reduction campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = 6
HARDWARE_ID = "metax_c500"
SOFTWARE_REVISION = "maca_3.7.2"
MECHANISM = "block_reduction_dependency"
AP_COUNT = 104
WAVE_SIZE = 64
BLOCK_THREADS = (128, 256, 1024)
ACTIVE_UNIT_GRIDS = (1, 2, 4, 8, 16, 32, 64, 104)
GRIDS = ACTIVE_UNIT_GRIDS + (208, 416, 832, 1664, 3328)
REPETITION_COUNTS = (4, 8, 16, 32)
SAMPLES_PER_CELL = 31
LAUNCHES_PER_BATCH = 16
FULL_TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
RESIDENT_CTAS_BY_THREADS = {128: 16, 256: 8, 1024: 2}
MAX_VALIDATION_MAPE_PERCENT = 15.0
MAX_VALIDATION_WMAE_PERCENT = 15.0
MAX_VALIDATION_MAX_APE_PERCENT = 30.0


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


def _positive_hex_float(text, label):
    try:
        value = float.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a hexadecimal float") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def parse_stdout(path, expected_trial, expected_device_id):
    """Parse one frozen trial and return median ns/launch by count cell."""

    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    metadata = rows[0]
    if (len(metadata) != 15
            or tuple(metadata[:3]) != ("metadata", "6", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    integers = tuple(
        _canonical_int(
            value, f"metadata[{index}]", positive=(index != 3))
        for index, value in enumerate(metadata[3:], start=3)
    )
    (device_id, ap_count, wave_size, clock_khz, trial, samples_per_cell,
     launches_per_batch, registers_per_thread, shared_bytes_per_cta,
     resident_128, resident_256, resident_1024) = integers
    if (device_id != expected_device_id or trial != expected_trial
            or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or samples_per_cell != SAMPLES_PER_CELL
            or launches_per_batch != LAUNCHES_PER_BATCH):
        raise ValueError(f"{path} changed the frozen campaign contract")
    resident = {
        128: resident_128,
        256: resident_256,
        1024: resident_1024,
    }
    if resident != RESIDENT_CTAS_BY_THREADS:
        raise ValueError(
            f"{path} block-reduction occupancy changed: {resident}")

    expected_rows = (
        len(BLOCK_THREADS) * len(GRIDS) * len(REPETITION_COUNTS)
        * SAMPLES_PER_CELL)
    if len(rows[1:]) != expected_rows:
        raise ValueError(
            f"{path} has {len(rows[1:])} samples, expected {expected_rows}")
    orders = set()
    identities = set()
    samples = {}
    for row in rows[1:]:
        if len(row) != 7 or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = _canonical_int(row[1], "sample.order")
        threads = _canonical_int(row[2], "sample.threads", positive=True)
        grid = _canonical_int(row[3], "sample.grid", positive=True)
        repetitions = _canonical_int(
            row[4], "sample.repetitions", positive=True)
        sample = _canonical_int(row[5], "sample.sample")
        duration_ms = _positive_hex_float(row[6], "sample.duration_ms")
        if (threads not in BLOCK_THREADS or grid not in GRIDS
                or repetitions not in REPETITION_COUNTS
                or sample >= SAMPLES_PER_CELL):
            raise ValueError(f"{path} has a sample outside the campaign")
        identity = (threads, grid, repetitions, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} contains duplicate samples")
        orders.add(order)
        identities.add(identity)
        key = (threads, grid, repetitions)
        if key not in samples:
            samples[key] = []
        samples[key].append(
            duration_ms * 1_000_000.0 / LAUNCHES_PER_BATCH)
    if orders != set(range(expected_rows)):
        raise ValueError(f"{path} randomized order is not contiguous")
    expected_identities = {
        (threads, grid, repetitions, sample)
        for threads in BLOCK_THREADS
        for grid in GRIDS
        for repetitions in REPETITION_COUNTS
        for sample in range(SAMPLES_PER_CELL)
    }
    if identities != expected_identities:
        raise ValueError(f"{path} sample matrix is incomplete")
    if any(len(values) != SAMPLES_PER_CELL for values in samples.values()):
        raise ValueError(f"{path} sample cells are incomplete")
    return {
        "trial": trial,
        "device_id": device_id,
        "clock_khz": clock_khz,
        "registers_per_thread": registers_per_thread,
        "shared_bytes_per_cta": shared_bytes_per_cta,
        "resident_ctas_by_threads": resident,
        "cell_medians_ns": {
            key: statistics.median(values)
            for key, values in samples.items()
        },
        "stdout_sha256": _sha256(path),
    }


def _depth(grid):
    if grid <= AP_COUNT:
        return 1
    if grid % AP_COUNT:
        raise ValueError("campaign grid must be one or an AP-count multiple")
    return grid // AP_COUNT


def _service_by_grid(parsed, threads):
    services = {}
    for grid in GRIDS:
        slopes = []
        for lower, upper in zip(REPETITION_COUNTS, REPETITION_COUNTS[1:]):
            lower_ns = parsed["cell_medians_ns"][(threads, grid, lower)]
            upper_ns = parsed["cell_medians_ns"][(threads, grid, upper)]
            slope = (upper_ns - lower_ns) / (upper - lower)
            if slope <= 0:
                raise ValueError(
                    "block-reduction adjacent-count service must be positive: "
                    f"threads={threads}, grid={grid}, interval={lower}:{upper}")
            slopes.append(slope)
        services[grid] = statistics.median(slopes)
    return services


def _derive_thread_parameters(parsed_trials, threads):
    trial_services = {
        trial: _service_by_grid(parsed_trials[trial], threads)
        for trial in DERIVATION_TRIALS
    }
    training_cells = {
        grid: statistics.median(
            trial_services[trial][grid] for trial in DERIVATION_TRIALS)
        for grid in GRIDS
    }
    first_ns_by_active_units = {
        grid: training_cells[grid] for grid in ACTIVE_UNIT_GRIDS}
    depth_cells = {
        _depth(grid): training_cells[grid]
        for grid in GRIDS if grid >= AP_COUNT
    }
    resident_ctas = RESIDENT_CTAS_BY_THREADS[threads]
    resident_slopes = []
    queued_slopes = []
    ordered_depths = sorted(depth_cells)
    for lower, upper in zip(ordered_depths, ordered_depths[1:]):
        slope = (
            (depth_cells[upper] - depth_cells[lower]) / (upper - lower))
        if upper <= resident_ctas:
            resident_slopes.append(slope)
        elif lower >= resident_ctas:
            queued_slopes.append(slope)
        else:
            raise ValueError("campaign interval crosses occupancy boundary")
    if not resident_slopes or not queued_slopes:
        raise ValueError("campaign does not identify both service regimes")
    resident_ns = statistics.median(resident_slopes)
    queued_ns = statistics.median(queued_slopes)
    if min((*first_ns_by_active_units.values(), resident_ns, queued_ns)) <= 0:
        raise ValueError("derived block-reduction services must be positive")
    return {
        "threads": threads,
        "resident_ctas_per_ap": resident_ctas,
        "first_block_active_units": list(ACTIVE_UNIT_GRIDS),
        "first_block_service_ns": [
            first_ns_by_active_units[grid] for grid in ACTIVE_UNIT_GRIDS],
        "first_block_interpolation": "piecewise_linear",
        "resident_block_service_ns_per_ap": resident_ns,
        "queued_block_service_ns_per_ap": queued_ns,
        "resident_block_rate_per_ap_s": 1e9 / resident_ns,
        "queued_block_rate_per_ap_s": 1e9 / queued_ns,
    }


def _first_block_service_ns(parameter, active_units):
    points = parameter["first_block_active_units"]
    services = parameter["first_block_service_ns"]
    if active_units < points[0] or active_units > points[-1]:
        raise ValueError("active units are outside the measured curve")
    if active_units == points[0]:
        return services[0]
    upper = 1
    while upper < len(points) and active_units > points[upper]:
        upper += 1
    lower = upper - 1
    span = points[upper] - points[lower]
    fraction = (active_units - points[lower]) / span
    return services[lower] + fraction * (
        services[upper] - services[lower])


def _predict_ns(parameter, grid):
    depth = _depth(grid)
    resident_ctas = parameter["resident_ctas_per_ap"]
    return (
        _first_block_service_ns(parameter, min(grid, AP_COUNT))
        + min(max(depth - 1, 0), resident_ctas - 1)
        * parameter["resident_block_service_ns_per_ap"]
        + max(depth - resident_ctas, 0)
        * parameter["queued_block_service_ns_per_ap"]
    )


def _metrics(rows):
    absolute = [abs(row["predicted_ns"] - row["measured_ns"])
                for row in rows]
    ape = [error / row["measured_ns"] * 100.0
           for error, row in zip(absolute, rows)]
    return {
        "count": len(rows),
        "mape_percent": statistics.fmean(ape),
        "wmae_percent": math.fsum(absolute) / math.fsum(
            row["measured_ns"] for row in rows) * 100.0,
        "max_ape_percent": max(ape),
    }


def aggregate_trials(trial_paths, expected_device_id, source_path,
                     baseline_commit):
    if (not isinstance(baseline_commit, str)
            or len(baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in baseline_commit)):
        raise ValueError("baseline commit must be a lowercase SHA-1")
    if len(trial_paths) != len(FULL_TRIALS):
        raise ValueError("exactly three trial paths are required")
    parsed = {
        trial: parse_stdout(path, trial, expected_device_id)
        for trial, path in zip(FULL_TRIALS, trial_paths)
    }
    if len({parsed[trial]["clock_khz"] for trial in FULL_TRIALS}) != 1:
        raise ValueError("reported device clock changed across trials")
    resource_identities = {
        (
            parsed[trial]["registers_per_thread"],
            parsed[trial]["shared_bytes_per_cta"],
        )
        for trial in FULL_TRIALS
    }
    if len(resource_identities) != 1:
        raise ValueError("block-reduction resources changed across trials")
    parameters = tuple(
        _derive_thread_parameters(parsed, threads)
        for threads in BLOCK_THREADS
    )
    validation_rows = []
    validation_services = {
        threads: _service_by_grid(parsed[VALIDATION_TRIAL], threads)
        for threads in BLOCK_THREADS
    }
    for parameter in parameters:
        threads = parameter["threads"]
        for grid in GRIDS:
            validation_rows.append({
                "threads": threads,
                "grid": grid,
                "measured_ns": validation_services[threads][grid],
                "predicted_ns": _predict_ns(parameter, grid),
            })
    metrics = _metrics(validation_rows)
    publishable = (
        metrics["mape_percent"] <= MAX_VALIDATION_MAPE_PERCENT
        and metrics["wmae_percent"] <= MAX_VALIDATION_WMAE_PERCENT
        and metrics["max_ape_percent"] <= MAX_VALIDATION_MAX_APE_PERCENT)
    registers_per_thread, shared_bytes_per_cta = next(
        iter(resource_identities))
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
            str(trial): parsed[trial]["stdout_sha256"]
            for trial in FULL_TRIALS
        },
        "campaign": {
            "timer_domain": "mcevent_paired_default_stream",
            "derivation": "adjacent_runtime_repetition_count",
            "trials": list(FULL_TRIALS),
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
            "block_threads": list(BLOCK_THREADS),
            "first_block_active_unit_grids": list(ACTIVE_UNIT_GRIDS),
            "grids": list(GRIDS),
            "repetition_counts": list(REPETITION_COUNTS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "launches_per_batch": LAUNCHES_PER_BATCH,
        },
        "resource_usage": {
            "registers_per_thread": registers_per_thread,
            "shared_bytes_per_cta": shared_bytes_per_cta,
            "resident_ctas_by_threads": {
                str(key): value
                for key, value in RESIDENT_CTAS_BY_THREADS.items()
            },
        },
        "parameters": list(parameters),
        "validation_rows": validation_rows,
        "validation": metrics,
        "thresholds": {
            "maximum_mape_percent": MAX_VALIDATION_MAPE_PERCENT,
            "maximum_wmae_percent": MAX_VALIDATION_WMAE_PERCENT,
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
