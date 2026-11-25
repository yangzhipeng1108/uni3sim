#!/usr/bin/env python3
"""Validate and reduce the C500 plain RMSNorm scalar-round campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = 4
HARDWARE_ID = "metax_c500"
SUPPORTED_SOFTWARE_REVISIONS = ("maca_3.7.2", "maca_3.8.2.2")
MECHANISM = "bf16_plain_rmsnorm_scalar_source_round"
REFERENCE_SOURCE_PATH = (
    "csrc/layernorm_kernels.cu")
REFERENCE_SOURCE_SHA256 = (
    "056986926c196168a2edd654ef79e07aac704404ef092fd7019a496d0ea0188a")
AP_COUNT = 104
WAVE_SIZE = 64
BLOCK_THREADS = (128, 1024)
FUNCTIONS = ("input_square", "output_scale")
ACTIVE_UNIT_GRIDS = (1, 2, 4, 8, 16, 32, 64, 104)
GRIDS_BY_THREADS = {
    128: ACTIVE_UNIT_GRIDS + (208, 416, 832, 1664, 3328),
    1024: ACTIVE_UNIT_GRIDS + (208, 312, 416, 832),
}
ROUND_COUNTS = (2, 3, 4, 8, 16, 32)
SAMPLES_PER_CELL = 15
LAUNCHES_PER_BATCH = 64
TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
RESOURCE_SIGNATURES = {
    "input_square": {
        "registers_per_thread": 10,
        "shared_bytes_per_cta": 0,
        "resident_ctas_by_threads": {128: 16, 1024: 2},
    },
    "output_scale": {
        "registers_per_thread": 16,
        "shared_bytes_per_cta": 0,
        "resident_ctas_by_threads": {128: 16, 1024: 2},
    },
}
MAX_VALIDATION_MAPE_PERCENT = 10.0
MAX_VALIDATION_WAPE_PERCENT = 10.0
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


def _resource_signature(metadata):
    return {
        "input_square": {
            "registers_per_thread": metadata[7],
            "shared_bytes_per_cta": metadata[8],
            "resident_ctas_by_threads": {
                128: metadata[9],
                1024: metadata[10],
            },
        },
        "output_scale": {
            "registers_per_thread": metadata[11],
            "shared_bytes_per_cta": metadata[12],
            "resident_ctas_by_threads": {
                128: metadata[13],
                1024: metadata[14],
            },
        },
    }


def parse_stdout(path, expected_trial, expected_device_id):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    if (len(header) != 18
            or tuple(header[:3]) != ("metadata", "2", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    metadata = tuple(
        _canonical_int(
            value, f"metadata[{index}]",
            positive=(index not in (3, 11, 15)))
        for index, value in enumerate(header[3:], start=3)
    )
    (device_id, ap_count, wave_size, clock_khz, trial, samples_per_cell,
     launches_per_batch, _input_registers, _input_shared,
     _input_resident_128, _input_resident_1024,
     _output_registers, _output_shared,
     _output_resident_128, _output_resident_1024) = metadata
    if (device_id != expected_device_id or trial != expected_trial
            or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or samples_per_cell != SAMPLES_PER_CELL
            or launches_per_batch != LAUNCHES_PER_BATCH):
        raise ValueError(f"{path} changed the frozen campaign contract")
    resources = _resource_signature(metadata)
    if resources != RESOURCE_SIGNATURES:
        raise ValueError(
            f"{path} changed the compiled resource signature: {resources}")
    expected_count = (
        len(FUNCTIONS)
        * sum(len(grids) for grids in GRIDS_BY_THREADS.values())
        * len(ROUND_COUNTS)
        * SAMPLES_PER_CELL)
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
        function = row[2]
        threads = _canonical_int(row[3], "sample.threads", positive=True)
        grid = _canonical_int(row[4], "sample.grid", positive=True)
        rounds = _canonical_int(row[5], "sample.rounds", positive=True)
        sample = _canonical_int(row[6], "sample.sample")
        duration_ms = _positive_hex_float(row[7], "sample.duration_ms")
        if (function not in FUNCTIONS or threads not in BLOCK_THREADS
                or grid not in GRIDS_BY_THREADS[threads]
                or rounds not in ROUND_COUNTS
                or sample >= SAMPLES_PER_CELL):
            raise ValueError(f"{path} has a sample outside the campaign")
        identity = (function, threads, grid, rounds, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} contains duplicate samples")
        orders.add(order)
        identities.add(identity)
        key = (function, threads, grid, rounds)
        if key not in samples:
            samples[key] = []
        samples[key].append(
            duration_ms * 1_000_000.0 / LAUNCHES_PER_BATCH)
    if orders != set(range(expected_count)):
        raise ValueError(f"{path} randomized order is not contiguous")
    expected_identities = {
        (function, threads, grid, rounds, sample)
        for function in FUNCTIONS
        for threads in BLOCK_THREADS
        for grid in GRIDS_BY_THREADS[threads]
        for rounds in ROUND_COUNTS
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
            key: statistics.median(values) for key, values in samples.items()},
        "stdout_sha256": _sha256(path),
    }


def _service_by_grid(parsed, function, threads):
    round_mean = statistics.fmean(ROUND_COUNTS)
    denominator = math.fsum(
        (rounds - round_mean) ** 2 for rounds in ROUND_COUNTS)
    if denominator <= 0:
        raise ValueError("source-round regression requires distinct counts")
    services = {}
    for grid in GRIDS_BY_THREADS[threads]:
        durations = tuple(
            parsed["cell_medians_ns"][(function, threads, grid, rounds)]
            for rounds in ROUND_COUNTS)
        duration_mean = statistics.fmean(durations)
        services[grid] = math.fsum(
            (rounds - round_mean) * (duration - duration_mean)
            for rounds, duration in zip(ROUND_COUNTS, durations)
        ) / denominator
    if any(not math.isfinite(value) or value <= 0
           for value in services.values()):
        raise ValueError(
            "steady source-round differences must be finite and positive")
    return {
        grid: services[grid] for grid in GRIDS_BY_THREADS[threads]
    }


def _derive_parameters(parsed_trials, function, threads):
    trial_services = {
        trial: _service_by_grid(parsed_trials[trial], function, threads)
        for trial in DERIVATION_TRIALS
    }
    services = {
        grid: statistics.median(
            trial_services[trial][grid] for trial in DERIVATION_TRIALS)
        for grid in GRIDS_BY_THREADS[threads]
    }
    first_ns = {grid: services[grid] for grid in ACTIVE_UNIT_GRIDS}
    depth_ns = {
        grid // AP_COUNT: services[grid]
        for grid in GRIDS_BY_THREADS[threads] if grid >= AP_COUNT
    }
    resident_ctas = RESOURCE_SIGNATURES[function][
        "resident_ctas_by_threads"][threads]
    resident_slopes = []
    queued_slopes = []
    depths = sorted(depth_ns)
    for lower, upper in zip(depths, depths[1:]):
        slope = (depth_ns[upper] - depth_ns[lower]) / (upper - lower)
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
    if min((*first_ns.values(), resident_ns, queued_ns)) <= 0:
        raise ValueError("derived scalar-round services must be positive")
    return {
        "function": function,
        "block_threads": threads,
        "minimum_round_depth": min(ROUND_COUNTS),
        "maximum_round_depth": max(ROUND_COUNTS),
        "resident_ctas_per_ap": resident_ctas,
        "first_cta_round_active_units": list(ACTIVE_UNIT_GRIDS),
        "first_cta_round_service_ns": [
            first_ns[grid] for grid in ACTIVE_UNIT_GRIDS],
        "first_cta_round_interpolation": "piecewise_linear",
        "resident_cta_round_service_ns_per_ap": resident_ns,
        "queued_cta_round_service_ns_per_ap": queued_ns,
        "resident_cta_round_rate_per_ap_s": 1e9 / resident_ns,
        "queued_cta_round_rate_per_ap_s": 1e9 / queued_ns,
    }


def _first_service_ns(parameter, active_units):
    points = parameter["first_cta_round_active_units"]
    services = parameter["first_cta_round_service_ns"]
    if active_units in points:
        return services[points.index(active_units)]
    if active_units < points[0] or active_units > points[-1]:
        raise ValueError("active units are outside the measured curve")
    upper = next(index for index, point in enumerate(points)
                 if active_units < point)
    lower = upper - 1
    fraction = (
        (active_units - points[lower]) / (points[upper] - points[lower]))
    return services[lower] + fraction * (
        services[upper] - services[lower])


def _predict_ns(parameter, grid):
    if grid <= AP_COUNT:
        depth = 1
        active_units = grid
    elif grid % AP_COUNT == 0:
        depth = grid // AP_COUNT
        active_units = AP_COUNT
    else:
        raise ValueError("multi-wave campaign grid must be an AP multiple")
    resident = parameter["resident_ctas_per_ap"]
    return (
        _first_service_ns(parameter, active_units)
        + min(max(depth - 1, 0), resident - 1)
        * parameter["resident_cta_round_service_ns_per_ap"]
        + max(depth - resident, 0)
        * parameter["queued_cta_round_service_ns_per_ap"]
    )


def _metrics(rows):
    absolute = [
        abs(row["predicted_ns"] - row["measured_ns"]) for row in rows]
    ape = [
        error / row["measured_ns"] * 100.0
        for error, row in zip(absolute, rows)]
    return {
        "count": len(rows),
        "mape_percent": statistics.fmean(ape),
        "wape_percent": math.fsum(absolute) / math.fsum(
            row["measured_ns"] for row in rows) * 100.0,
        "max_ape_percent": max(ape),
    }


def _disassembly_evidence(path):
    text = Path(path).read_text(encoding="utf-8")
    input_marker = "rmsnorm_scalar_round<0>"
    output_marker = "rmsnorm_scalar_round<1>"
    input_start = text.find(input_marker)
    output_start = text.find(output_marker)
    if input_start < 0 or output_start <= input_start:
        raise ValueError("disassembly lacks both RMSNorm scalar functions")
    input_section = text[input_start:output_start]
    output_section = text[output_start:]
    evidence = {
        "input_ldg_u16": input_section.count("ldg_u16"),
        "input_fmac_f32": input_section.count("fmac_f32"),
        "output_ldg_u16": output_section.count("ldg_u16"),
        "output_mul_f32": output_section.count("mul_f32"),
        "output_bf16_round_bias": sum(
            "add_u32" in line and "0x7fff" in line
            for line in output_section.splitlines()),
        "output_stg_b16": output_section.count("stg_b16"),
    }
    expected = {
        "input_ldg_u16": 1,
        "input_fmac_f32": 1,
        "output_ldg_u16": 2,
        "output_mul_f32": 2,
        "output_bf16_round_bias": 2,
        "output_stg_b16": 1,
    }
    if evidence != expected:
        raise ValueError(
            f"compiled scalar-round instruction signature changed: {evidence}")
    return evidence


def aggregate_trials(
        trial_paths, expected_device_id, source_path, binary_path,
        disassembly_path, baseline_commit, software_revision):
    software_revision = _normalize_software_revision(software_revision)
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
    parameters = tuple(
        _derive_parameters(parsed, function, threads)
        for function in FUNCTIONS
        for threads in BLOCK_THREADS)
    holdout_services = {
        (function, threads): _service_by_grid(
            parsed[VALIDATION_TRIAL], function, threads)
        for function in FUNCTIONS
        for threads in BLOCK_THREADS
    }
    validation_rows = []
    for parameter in parameters:
        function = parameter["function"]
        threads = parameter["block_threads"]
        for grid in GRIDS_BY_THREADS[threads]:
            validation_rows.append({
                "function": function,
                "block_threads": threads,
                "grid": grid,
                "measured_ns": holdout_services[(function, threads)][grid],
                "predicted_ns": _predict_ns(parameter, grid),
            })
    validation = _metrics(validation_rows)
    by_function = {
        f"{function}_block{threads}": _metrics([
            row for row in validation_rows
            if (row["function"] == function
                and row["block_threads"] == threads)])
        for function in FUNCTIONS
        for threads in BLOCK_THREADS
    }
    publishable = (
        validation["mape_percent"] <= MAX_VALIDATION_MAPE_PERCENT
        and validation["wape_percent"] <= MAX_VALIDATION_WAPE_PERCENT
        and validation["max_ape_percent"] <= MAX_VALIDATION_MAX_APE_PERCENT)
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "software_revision": software_revision,
        "mechanism": MECHANISM,
        "production_or_evaluation_duration_consumed": False,
        "baseline_commit": baseline_commit,
        "production_source_reference": {
            "path": REFERENCE_SOURCE_PATH,
            "sha256": REFERENCE_SOURCE_SHA256,
        },
        "probe_source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "binary": {
            "path": str(binary_path),
            "sha256": _sha256(binary_path),
        },
        "disassembly": {
            "path": str(disassembly_path),
            "sha256": _sha256(disassembly_path),
            "instruction_signature": _disassembly_evidence(
                disassembly_path),
        },
        "trial_stdout_sha256": {
            str(trial): parsed[trial]["stdout_sha256"] for trial in TRIALS},
        "campaign": {
            "timer_domain": "mcevent_default_stream",
            "device_id": expected_device_id,
            "cell_reduction": "median",
            "round_service_derivation": (
                "least_squares_all_runtime_round_counts"),
            "round_service_fit_counts": list(ROUND_COUNTS),
            "trials": list(TRIALS),
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
            "block_threads": list(BLOCK_THREADS),
            "functions": list(FUNCTIONS),
            "grids_by_threads": {
                str(threads): list(GRIDS_BY_THREADS[threads])
                for threads in BLOCK_THREADS
            },
            "round_counts": list(ROUND_COUNTS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "launches_per_batch": LAUNCHES_PER_BATCH,
        },
        "resource_usage_by_function": RESOURCE_SIGNATURES,
        "parameters": list(parameters),
        "validation_rows": validation_rows,
        "validation_by_function": by_function,
        "validation": validation,
        "thresholds": {
            "maximum_mape_percent": MAX_VALIDATION_MAPE_PERCENT,
            "maximum_wape_percent": MAX_VALIDATION_WAPE_PERCENT,
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
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--disassembly", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument(
        "--software-revision", required=True,
        choices=SUPPORTED_SOFTWARE_REVISIONS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = aggregate_trials(
        args.trial, args.device_id, args.source, args.binary,
        args.disassembly, args.baseline_commit, args.software_revision)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0 if report["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
