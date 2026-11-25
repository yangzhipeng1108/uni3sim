#!/usr/bin/env python3
"""Validate and reduce the C500 BF16 vector-memory round campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = 5
HARDWARE_ID = "metax_c500"
SUPPORTED_SOFTWARE_REVISIONS = ("maca_3.7.2", "maca_3.8.2.2")
MECHANISM = "bf16_vector_memory_dependency_round"
AP_COUNT = 104
WAVE_SIZE = 64
BLOCK_THREADS = (256, 512, 1024)
FUNCTIONS = ("add_square", "scale_multiply")
ACTIVE_UNIT_GRIDS = (1, 2, 4, 8, 16, 32, 64, 104)
GRIDS = ACTIVE_UNIT_GRIDS + (208, 416, 832, 1664)
REPETITION_COUNTS = (4, 8, 16, 32)
SAMPLES_PER_CELL = 21
LAUNCHES_PER_BATCH = 8
TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
RESIDENT_CTAS_BY_THREADS = {256: 2, 512: 2, 1024: 2}
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


def _normalize_software_revision(value):
    if not isinstance(value, str):
        raise TypeError("software revision must be a string")
    if value not in SUPPORTED_SOFTWARE_REVISIONS:
        raise ValueError(
            "software revision must be one of "
            f"{SUPPORTED_SOFTWARE_REVISIONS}, got {value!r}")
    return value


def parse_stdout(path, expected_trial, expected_device_id):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    metadata = rows[0]
    if (len(metadata) != 20
            or tuple(metadata[:3]) != ("metadata", "5", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    integers = tuple(
        _canonical_int(
            value, f"metadata[{index}]",
            positive=(index not in (3, 11, 16)))
        for index, value in enumerate(metadata[3:], start=3)
    )
    (device_id, ap_count, wave_size, clock_khz, trial, samples_per_cell,
     launches_per_batch, add_registers_per_thread, add_shared_bytes_per_cta,
     add_resident_256, add_resident_512, add_resident_1024,
     scale_registers_per_thread, scale_shared_bytes_per_cta,
     scale_resident_256, scale_resident_512,
     scale_resident_1024) = integers
    if (device_id != expected_device_id or trial != expected_trial
            or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or samples_per_cell != SAMPLES_PER_CELL
            or launches_per_batch != LAUNCHES_PER_BATCH):
        raise ValueError(f"{path} changed the frozen campaign contract")
    resources = {
        "add_square": {
            "registers_per_thread": add_registers_per_thread,
            "shared_bytes_per_cta": add_shared_bytes_per_cta,
            "resident_ctas_by_threads": {
                256: add_resident_256, 512: add_resident_512,
                1024: add_resident_1024},
        },
        "scale_multiply": {
            "registers_per_thread": scale_registers_per_thread,
            "shared_bytes_per_cta": scale_shared_bytes_per_cta,
            "resident_ctas_by_threads": {
                256: scale_resident_256, 512: scale_resident_512,
                1024: scale_resident_1024},
        },
    }
    for function, resource in resources.items():
        if resource["resident_ctas_by_threads"] != RESIDENT_CTAS_BY_THREADS:
            raise ValueError(
                f"{path} {function} vector-memory occupancy changed: "
                f"{resource['resident_ctas_by_threads']}")
    expected_rows = (
        len(FUNCTIONS) * len(BLOCK_THREADS) * len(GRIDS)
        * len(REPETITION_COUNTS)
        * SAMPLES_PER_CELL)
    if len(rows[1:]) != expected_rows:
        raise ValueError(
            f"{path} has {len(rows[1:])} samples, expected {expected_rows}")
    orders = set()
    identities = set()
    samples = {}
    for row in rows[1:]:
        if len(row) != 8 or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = _canonical_int(row[1], "sample.order")
        function = row[2]
        threads = _canonical_int(row[3], "sample.threads", positive=True)
        grid = _canonical_int(row[4], "sample.grid", positive=True)
        repetitions = _canonical_int(
            row[5], "sample.repetitions", positive=True)
        sample = _canonical_int(row[6], "sample.sample")
        duration_ms = _positive_hex_float(row[7], "sample.duration_ms")
        if (function not in FUNCTIONS or threads not in BLOCK_THREADS
                or grid not in GRIDS
                or repetitions not in REPETITION_COUNTS
                or sample >= SAMPLES_PER_CELL):
            raise ValueError(f"{path} has a sample outside the campaign")
        identity = (function, threads, grid, repetitions, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} contains duplicate samples")
        orders.add(order)
        identities.add(identity)
        key = (function, threads, grid, repetitions)
        if key not in samples:
            samples[key] = []
        samples[key].append(
            duration_ms * 1_000_000.0 / LAUNCHES_PER_BATCH)
    if orders != set(range(expected_rows)):
        raise ValueError(f"{path} randomized order is not contiguous")
    expected_identities = {
        (function, threads, grid, repetitions, sample)
        for function in FUNCTIONS
        for threads in BLOCK_THREADS
        for grid in GRIDS
        for repetitions in REPETITION_COUNTS
        for sample in range(SAMPLES_PER_CELL)
    }
    if identities != expected_identities:
        raise ValueError(f"{path} sample matrix is incomplete")
    return {
        "trial": trial,
        "device_id": device_id,
        "clock_khz": clock_khz,
        "resource_usage_by_function": resources,
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


def _service_by_grid(parsed, function, threads):
    services = {}
    for grid in GRIDS:
        slopes = []
        for lower, upper in zip(REPETITION_COUNTS, REPETITION_COUNTS[1:]):
            lower_ns = parsed["cell_medians_ns"][
                (function, threads, grid, lower)]
            upper_ns = parsed["cell_medians_ns"][
                (function, threads, grid, upper)]
            slope = (upper_ns - lower_ns) / (upper - lower)
            if slope <= 0:
                raise ValueError(
                    "BF16 vector-memory adjacent-count service must be "
                    f"positive: function={function}, threads={threads}, "
                    f"grid={grid}, "
                    f"interval={lower}:{upper}")
            slopes.append(slope)
        services[grid] = statistics.median(slopes)
    return services


def _derive_parameters(parsed_trials, function, threads):
    trial_services = {
        trial: _service_by_grid(parsed_trials[trial], function, threads)
        for trial in DERIVATION_TRIALS
    }
    training_cells = {
        grid: statistics.median(
            trial_services[trial][grid] for trial in DERIVATION_TRIALS)
        for grid in GRIDS
    }
    active_unit_ns = {
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
    if min((*active_unit_ns.values(), resident_ns, queued_ns)) <= 0:
        raise ValueError("derived vector-memory services must be positive")
    return {
        "function": function,
        "threads": threads,
        "resident_ctas_per_ap": resident_ctas,
        "first_cta_round_active_units": list(ACTIVE_UNIT_GRIDS),
        "first_cta_round_service_ns": [
            active_unit_ns[grid] for grid in ACTIVE_UNIT_GRIDS],
        "first_cta_round_interpolation": "piecewise_linear",
        "resident_cta_round_service_ns_per_ap": resident_ns,
        "queued_cta_round_service_ns_per_ap": queued_ns,
        "resident_cta_round_rate_per_ap_s": 1e9 / resident_ns,
        "queued_cta_round_rate_per_ap_s": 1e9 / queued_ns,
    }


def _first_cta_round_service_ns(parameter, active_units):
    points = parameter["first_cta_round_active_units"]
    services = parameter["first_cta_round_service_ns"]
    if active_units < points[0] or active_units > points[-1]:
        raise ValueError("active units are outside the measured curve")
    upper = 1
    while upper < len(points) and active_units > points[upper]:
        upper += 1
    lower = upper - 1
    span = points[upper] - points[lower]
    fraction = (active_units - points[lower]) / span
    return services[lower] + fraction * (services[upper] - services[lower])


def _predict_ns(parameter, grid):
    depth = _depth(grid)
    resident_ctas = parameter["resident_ctas_per_ap"]
    first_ns = _first_cta_round_service_ns(
        parameter, min(grid, AP_COUNT))
    return (
        first_ns
        + min(max(depth - 1, 0), resident_ctas - 1)
        * parameter["resident_cta_round_service_ns_per_ap"]
        + max(depth - resident_ctas, 0)
        * parameter["queued_cta_round_service_ns_per_ap"]
    )


def _metrics(rows):
    absolute = [
        abs(row["predicted_ns"] - row["measured_ns"])
        for row in rows]
    ape = [
        error / row["measured_ns"] * 100.0
        for error, row in zip(absolute, rows)]
    return {
        "count": len(rows),
        "mape_percent": statistics.fmean(ape),
        "wmae_percent": math.fsum(absolute) / math.fsum(
            row["measured_ns"] for row in rows) * 100.0,
        "max_ape_percent": max(ape),
    }


def aggregate_trials(
        trial_paths, expected_device_id, source_path, baseline_commit,
        software_revision):
    normalized_software_revision = _normalize_software_revision(
        software_revision)
    if (not isinstance(baseline_commit, str)
            or len(baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in baseline_commit)):
        raise ValueError("baseline commit must be a lowercase SHA-1")
    if len(trial_paths) != len(TRIALS):
        raise ValueError("exactly three trial paths are required")
    parsed = {
        trial: parse_stdout(path, trial, expected_device_id)
        for trial, path in zip(TRIALS, trial_paths)
    }
    if len({parsed[trial]["clock_khz"] for trial in TRIALS}) != 1:
        raise ValueError("reported device clock changed across trials")
    resources = {
        json.dumps(
            parsed[trial]["resource_usage_by_function"], sort_keys=True)
        for trial in TRIALS
    }
    if len(resources) != 1:
        raise ValueError("vector-memory resources changed across trials")
    parameters = tuple(
        _derive_parameters(parsed, function, threads)
        for function in FUNCTIONS
        for threads in BLOCK_THREADS)
    validation_services = {
        (function, threads): _service_by_grid(
            parsed[VALIDATION_TRIAL], function, threads)
        for function in FUNCTIONS
        for threads in BLOCK_THREADS
    }
    validation_rows = []
    for parameter in parameters:
        threads = parameter["threads"]
        function = parameter["function"]
        for grid in GRIDS:
            validation_rows.append({
                "function": function,
                "threads": threads,
                "grid": grid,
                "measured_ns": validation_services[
                    (function, threads)][grid],
                "predicted_ns": _predict_ns(parameter, grid),
            })
    metrics = _metrics(validation_rows)
    publishable = (
        metrics["mape_percent"] <= MAX_VALIDATION_MAPE_PERCENT
        and metrics["wmae_percent"] <= MAX_VALIDATION_WMAE_PERCENT
        and metrics["max_ape_percent"] <= MAX_VALIDATION_MAX_APE_PERCENT)
    resource_usage_by_function = parsed[TRIALS[0]][
        "resource_usage_by_function"]
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "software_revision": normalized_software_revision,
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
            "timer_domain": "mcevent_default_stream",
            "derivation": "adjacent_runtime_repetition_count",
            "trials": list(TRIALS),
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
            "block_threads": list(BLOCK_THREADS),
            "physical_block_threads": 1024,
            "functions": list(FUNCTIONS),
            "grids": list(GRIDS),
            "repetition_counts": list(REPETITION_COUNTS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "launches_per_batch": LAUNCHES_PER_BATCH,
            "warm_working_set_bytes_per_operand": (
                AP_COUNT * max(BLOCK_THREADS) * 16),
        },
        "resource_usage_by_function": {
            function: {
                "registers_per_thread": resource[
                    "registers_per_thread"],
                "shared_bytes_per_cta": resource[
                    "shared_bytes_per_cta"],
                "resident_ctas_by_threads": {
                    str(key): value
                    for key, value in resource[
                        "resident_ctas_by_threads"].items()
                },
            }
            for function, resource in resource_usage_by_function.items()
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
    parser.add_argument(
        "--software-revision", required=True,
        choices=SUPPORTED_SOFTWARE_REVISIONS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = aggregate_trials(
        args.trial, args.device_id, args.source, args.baseline_commit,
        args.software_revision)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0 if report["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
