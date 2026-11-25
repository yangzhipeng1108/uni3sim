#!/usr/bin/env python3
"""Validate the C500 scalar AddRMSNorm two-phase chain campaign."""

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
MECHANISM = "bf16_generic_add_rmsnorm_scalar_phase_chain"
REFERENCE_SOURCE_PATH = (
    "csrc/layernorm_kernels.cu")
REFERENCE_SOURCE_SHA256 = (
    "056986926c196168a2edd654ef79e07aac704404ef092fd7019a496d0ea0188a")
AP_COUNT = 104
WAVE_SIZE = 64
BLOCK_THREADS = 1024
FUNCTIONS = ("input_add_square", "output_scale", "phase_chain")
GRIDS = (1, 2, 4, 8, 16, 32, 64, 104)
PUBLISHED_ACTIVE_UNITS = (2, 4, 8, 16, 32)
ROUND_COUNTS = (8, 16, 32)
SAMPLES_PER_CELL = 15
LAUNCHES_PER_BATCH = 8
TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
RESOURCE_SIGNATURES = {
    "input_add_square": {
        "registers_per_thread": 16,
        "shared_bytes_per_cta": 0,
        "resident_ctas_per_ap": 2,
    },
    "output_scale": {
        "registers_per_thread": 16,
        "shared_bytes_per_cta": 0,
        "resident_ctas_per_ap": 2,
    },
    "phase_chain": {
        "registers_per_thread": 16,
        "shared_bytes_per_cta": 4,
        "resident_ctas_per_ap": 2,
    },
}
MAXIMUM_VALIDATION_MAPE_PERCENT = 5.0
MAXIMUM_VALIDATION_MAX_APE_PERCENT = 10.0


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


def _resource_signatures(metadata):
    return {
        function: {
            "registers_per_thread": metadata[7 + 3 * index],
            "shared_bytes_per_cta": metadata[8 + 3 * index],
            "resident_ctas_per_ap": metadata[9 + 3 * index],
        }
        for index, function in enumerate(FUNCTIONS)
    }


def parse_stdout(path, expected_trial, expected_device_id):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    if (len(header) != 19
            or tuple(header[:3]) != ("metadata", "1", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    metadata = tuple(
        _canonical_int(
            value, f"metadata[{index}]",
            positive=(index not in (11, 14, 17)))
        for index, value in enumerate(header[3:], start=3)
    )
    (device_id, ap_count, wave_size, clock_khz, trial, samples_per_cell,
     launches_per_batch, *_resources) = metadata
    if (device_id != expected_device_id or trial != expected_trial
            or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or samples_per_cell != SAMPLES_PER_CELL
            or launches_per_batch != LAUNCHES_PER_BATCH):
        raise ValueError(f"{path} changed the frozen campaign contract")
    resources = _resource_signatures(metadata)
    if resources != RESOURCE_SIGNATURES:
        raise ValueError(
            f"{path} changed the compiled resource signature: {resources}")
    expected_count = (
        len(FUNCTIONS) * len(GRIDS) * len(ROUND_COUNTS)
        * SAMPLES_PER_CELL)
    if len(rows[1:]) != expected_count:
        raise ValueError(
            f"{path} has {len(rows[1:])} samples, expected {expected_count}")
    orders = set()
    identities = set()
    samples = {}
    for row in rows[1:]:
        if len(row) != 7 or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = _canonical_int(row[1], "sample.order")
        function = row[2]
        grid = _canonical_int(row[3], "sample.grid", positive=True)
        rounds = _canonical_int(row[4], "sample.rounds", positive=True)
        sample = _canonical_int(row[5], "sample.sample")
        duration_ms = _positive_hex_float(row[6], "sample.duration_ms")
        if (function not in FUNCTIONS or grid not in GRIDS
                or rounds not in ROUND_COUNTS
                or sample >= SAMPLES_PER_CELL):
            raise ValueError(f"{path} has a sample outside the campaign")
        identity = (function, grid, rounds, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} contains duplicate samples")
        orders.add(order)
        identities.add(identity)
        key = (function, grid, rounds)
        if key not in samples:
            samples[key] = []
        samples[key].append(
            duration_ms * 1_000_000.0 / LAUNCHES_PER_BATCH)
    if orders != set(range(expected_count)):
        raise ValueError(f"{path} randomized order is not contiguous")
    return {
        "trial": trial,
        "device_id": device_id,
        "clock_khz": clock_khz,
        "resource_usage_by_function": resources,
        "cell_medians_ns": {
            key: statistics.median(values)
            for key, values in samples.items()},
        "stdout_sha256": _sha256(path),
    }


def _round_services(parsed, grid):
    lower_rounds, upper_rounds = ROUND_COUNTS[-2:]
    delta = upper_rounds - lower_rounds
    return {
        function: (
            parsed["cell_medians_ns"][(function, grid, upper_rounds)]
            - parsed["cell_medians_ns"][(function, grid, lower_rounds)]
        ) / delta
        for function in FUNCTIONS
    }


def _chain_excess(services):
    return (
        services["phase_chain"]
        - services["input_add_square"]
        - services["output_scale"])


def _metrics(rows):
    errors = [
        abs(row["predicted_ns"] - row["measured_ns"])
        for row in rows]
    ape = [
        error / row["measured_ns"] * 100.0
        for error, row in zip(errors, rows)]
    return {
        "count": len(rows),
        "mape_percent": statistics.fmean(ape),
        "wape_percent": (
            math.fsum(errors)
            / math.fsum(row["measured_ns"] for row in rows) * 100.0),
        "max_ape_percent": max(ape),
    }


def aggregate_trials(
        trial_paths, expected_device_id, source_path, binary_path,
        baseline_commit):
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

    services = {
        trial: {grid: _round_services(parsed[trial], grid) for grid in GRIDS}
        for trial in TRIALS}
    excess_ns = {
        grid: statistics.median(
            _chain_excess(services[trial][grid])
            for trial in DERIVATION_TRIALS)
        for grid in PUBLISHED_ACTIVE_UNITS}
    if any(value <= 0 for value in excess_ns.values()):
        raise ValueError("published phase-chain excess must remain positive")
    boundary_excess = {
        str(grid): [
            _chain_excess(services[trial][grid])
            for trial in DERIVATION_TRIALS]
        for grid in (1, 64)}
    if (all(value > 0 for value in boundary_excess["1"])
            or any(value > 0 for value in boundary_excess["64"])):
        raise ValueError("phase-chain support boundaries changed")

    validation_rows = []
    for grid in PUBLISHED_ACTIVE_UNITS:
        holdout = services[VALIDATION_TRIAL][grid]
        measured_ns = holdout["phase_chain"]
        predicted_ns = (
            holdout["input_add_square"] + holdout["output_scale"]
            + excess_ns[grid])
        validation_rows.append({
            "active_units": grid,
            "measured_ns": measured_ns,
            "predicted_ns": predicted_ns,
            "ape_percent": (
                abs(predicted_ns - measured_ns) / measured_ns * 100.0),
        })
    validation = _metrics(validation_rows)
    publishable = (
        validation["mape_percent"] <= MAXIMUM_VALIDATION_MAPE_PERCENT
        and validation["max_ape_percent"]
        <= MAXIMUM_VALIDATION_MAX_APE_PERCENT)
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "software_revision": SOFTWARE_REVISION,
        "mechanism": MECHANISM,
        "production_or_evaluation_duration_consumed": False,
        "baseline_commit": baseline_commit,
        "production_source_reference": {
            "path": REFERENCE_SOURCE_PATH,
            "sha256": REFERENCE_SOURCE_SHA256,
            "specialization": (
                "width-ignored generic path selected by "
                "_typeConvert<c10::BFloat16>::exists == false"),
        },
        "probe_source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "binary": {"path": str(binary_path), "sha256": _sha256(binary_path)},
        "trial_stdout_sha256": {
            str(trial): parsed[trial]["stdout_sha256"] for trial in TRIALS},
        "campaign": {
            "timer_domain": "mcevent_default_stream",
            "device_id": expected_device_id,
            "derivation": "deepest_adjacent_scalar_round_difference",
            "trials": list(TRIALS),
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
            "block_threads": BLOCK_THREADS,
            "functions": list(FUNCTIONS),
            "grids": list(GRIDS),
            "round_counts": list(ROUND_COUNTS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "launches_per_batch": LAUNCHES_PER_BATCH,
        },
        "resource_usage_by_function": RESOURCE_SIGNATURES,
        "parameter": {
            "minimum_round_depth": min(ROUND_COUNTS),
            "maximum_round_depth": max(ROUND_COUNTS),
            "active_units": list(PUBLISHED_ACTIVE_UNITS),
            "services": {
                str(grid): {
                    "service": (
                        "add_rmsnorm_phase_chain_excess_first_1024_"
                        f"active{grid}"),
                    "service_time_ns_per_scalar_round": excess_ns[grid],
                    "service_rate_per_ap_s": 1.0e9 / excess_ns[grid],
                }
                for grid in PUBLISHED_ACTIVE_UNITS
            },
            "interpolation": "none_exact_active_unit_points",
            "support_boundary": (
                "measured positive active-unit interval bracketed by an "
                "unstable active1 point and negative active64 point"),
        },
        "excluded_boundary_points": boundary_excess,
        "validation_rows": validation_rows,
        "validation": validation,
        "thresholds": {
            "maximum_mape_percent": MAXIMUM_VALIDATION_MAPE_PERCENT,
            "maximum_max_ape_percent": MAXIMUM_VALIDATION_MAX_APE_PERCENT,
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
