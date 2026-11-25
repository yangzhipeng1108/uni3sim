#!/usr/bin/env python3
"""Validate the C500 scalar RMSNorm two-phase chain campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = 2
HARDWARE_ID = "metax_c500"
SOFTWARE_REVISION = "maca_3.7.2"
MECHANISM = "bf16_plain_rmsnorm_published_scalar_phase_chain"
REFERENCE_SOURCE_PATH = (
    "csrc/layernorm_kernels.cu")
REFERENCE_SOURCE_SHA256 = (
    "056986926c196168a2edd654ef79e07aac704404ef092fd7019a496d0ea0188a")
AP_COUNT = 104
WAVE_SIZE = 64
BLOCK_THREADS = 1024
FUNCTIONS = ("input_square", "output_scale", "phase_chain")
GRIDS = (1, 2, 4, 8, 16, 32, 64, 104, 208, 312, 416, 832)
PUBLISHED_ACTIVE_UNITS = (1, 2, 4, 8)
PUBLISHED_PHASE_ENTRY_ACTIVE_UNITS = (1, 2, 4, 8, 16)
ROUND_COUNTS = (4, 8, 16, 32)
PHASE_ENTRY_ROUND_COUNTS = (8, 16, 32)
SAMPLES_PER_CELL = 15
LAUNCHES_PER_BATCH = 8
TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
RESOURCE_SIGNATURES = {
    "input_square": {
        "registers_per_thread": 10,
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
# The independent non-empty-task scheduler separates the complete launch
# envelope from the fixed device-side body of the chained source phases.  A
# block1024 CTA contains sixteen wave64s.
LAUNCH_SUBMIT_NS = 2191.6317887313203
LAUNCH_CTA_DESCRIPTOR_NS = 1.0164408046126985
LAUNCH_MULTI_WAVE_ROUND_NS = 59.92870334645187


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
    expected_identities = {
        (function, grid, rounds, sample)
        for function in FUNCTIONS
        for grid in GRIDS
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


def _round_services(parsed, grid):
    lower_rounds, upper_rounds = ROUND_COUNTS[-2:]
    round_delta = upper_rounds - lower_rounds
    return {
        function: (
            parsed["cell_medians_ns"][(function, grid, upper_rounds)]
            - parsed["cell_medians_ns"][(function, grid, lower_rounds)]
        ) / round_delta
        for function in FUNCTIONS
    }


def _chain_excess(services):
    return (
        services["phase_chain"]
        - services["input_square"]
        - services["output_scale"])


def _phase_chain_affine(parsed, grid):
    """Return the deep-round chain slope and fixed intercept."""

    lower_rounds, upper_rounds = ROUND_COUNTS[-2:]
    lower_ns = parsed["cell_medians_ns"][
        ("phase_chain", grid, lower_rounds)]
    upper_ns = parsed["cell_medians_ns"][
        ("phase_chain", grid, upper_rounds)]
    slope_ns = (upper_ns - lower_ns) / (upper_rounds - lower_rounds)
    intercept_ns = lower_ns - slope_ns * lower_rounds
    return slope_ns, intercept_ns


def _block1024_scheduler_floor_ns(grid):
    """Return the independent complete launch envelope for this probe."""

    scheduler_rounds = math.ceil(grid / AP_COUNT)
    waves_per_cta = BLOCK_THREADS // WAVE_SIZE
    return (
        LAUNCH_SUBMIT_NS
        + LAUNCH_CTA_DESCRIPTOR_NS * grid
        + LAUNCH_MULTI_WAVE_ROUND_NS * scheduler_rounds * waves_per_cta)


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


def _disassembly_evidence(path):
    text = Path(path).read_text(encoding="utf-8")
    symbols = tuple(
        f"rmsnorm_phase_chain<{index}>" in text for index in range(3))
    barrier_count = sum(
        line.strip().endswith("barrier") for line in text.splitlines())
    if symbols != (True, True, True) or barrier_count != 1:
        raise ValueError(
            "phase-chain disassembly lacks three functions or one barrier")
    return {
        "input_square_symbol": symbols[0],
        "output_scale_symbol": symbols[1],
        "phase_chain_symbol": symbols[2],
        "phase_chain_barrier_count": barrier_count,
    }


def aggregate_trials(
        trial_paths, expected_device_id, source_path, binary_path,
        disassembly_path, baseline_commit):
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
        trial: {
            grid: _round_services(parsed[trial], grid) for grid in GRIDS}
        for trial in TRIALS
    }
    excess_ns = {
        grid: statistics.median(
            _chain_excess(services[trial][grid])
            for trial in DERIVATION_TRIALS)
        for grid in PUBLISHED_ACTIVE_UNITS
    }
    if any(value <= 0 for value in excess_ns.values()):
        raise ValueError("published phase-chain excess must remain positive")
    next_grid_excess = tuple(
        _chain_excess(services[trial][16])
        for trial in DERIVATION_TRIALS)
    if any(value > 0 for value in next_grid_excess):
        raise ValueError(
            "the first excluded active-unit point must remain nonpositive")

    phase_entry_parameters = {}
    phase_entry_slopes = {}
    for grid in PUBLISHED_PHASE_ENTRY_ACTIVE_UNITS:
        trial_affine = {
            trial: _phase_chain_affine(parsed[trial], grid)
            for trial in DERIVATION_TRIALS}
        phase_entry_slopes[grid] = statistics.median(
            value[0] for value in trial_affine.values())
        phase_entry_parameters[grid] = statistics.median(
            value[1] - _block1024_scheduler_floor_ns(grid)
            for value in trial_affine.values())
    if any(value <= 0 for value in phase_entry_parameters.values()):
        raise ValueError("published fixed phase entries must remain positive")
    first_excluded_phase_entries = tuple(
        _phase_chain_affine(parsed[trial], 32)[1]
        - _block1024_scheduler_floor_ns(32)
        for trial in DERIVATION_TRIALS)
    if any(value > 0 for value in first_excluded_phase_entries):
        raise ValueError(
            "the first excluded fixed phase entry must remain nonpositive")

    validation_rows = []
    for grid in PUBLISHED_ACTIVE_UNITS:
        holdout = services[VALIDATION_TRIAL][grid]
        measured = holdout["phase_chain"]
        predicted = (
            holdout["input_square"] + holdout["output_scale"]
            + excess_ns[grid])
        validation_rows.append({
            "active_units": grid,
            "measured_ns_per_round": measured,
            "predicted_ns_per_round": predicted,
            "ape_percent": abs(predicted - measured) / measured * 100.0,
        })
    validation = _metrics([
        {
            "measured_ns": row["measured_ns_per_round"],
            "predicted_ns": row["predicted_ns_per_round"],
        }
        for row in validation_rows])
    phase_entry_validation_rows = []
    for grid in PUBLISHED_PHASE_ENTRY_ACTIVE_UNITS:
        for rounds in PHASE_ENTRY_ROUND_COUNTS:
            measured_ns = parsed[VALIDATION_TRIAL]["cell_medians_ns"][
                ("phase_chain", grid, rounds)]
            predicted_ns = (
                _block1024_scheduler_floor_ns(grid)
                + phase_entry_parameters[grid]
                + phase_entry_slopes[grid] * rounds)
            phase_entry_validation_rows.append({
                "active_units": grid,
                "rounds": rounds,
                "measured_ns": measured_ns,
                "predicted_ns": predicted_ns,
                "ape_percent": (
                    abs(predicted_ns - measured_ns) / measured_ns * 100.0),
            })
    phase_entry_validation = _metrics(phase_entry_validation_rows)
    publishable = (
        validation["mape_percent"] <= MAXIMUM_VALIDATION_MAPE_PERCENT
        and validation["max_ape_percent"]
        <= MAXIMUM_VALIDATION_MAX_APE_PERCENT
        and phase_entry_validation["mape_percent"]
        <= MAXIMUM_VALIDATION_MAPE_PERCENT
        and phase_entry_validation["max_ape_percent"]
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
                "deepest_adjacent_round_difference"),
            "steady_round_depth_interval": list(ROUND_COUNTS[-2:]),
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
                        "rmsnorm_phase_chain_excess_first_1024"
                        if grid == 1 else
                        f"rmsnorm_phase_chain_excess_first_1024_active{grid}"),
                    "service_time_ns_per_round": excess_ns[grid],
                    "service_rate_per_ap_s": 1.0e9 / excess_ns[grid],
                }
                for grid in PUBLISHED_ACTIVE_UNITS
            },
            "interpolation": "none_exact_active_unit_points",
            "maximum_supported_active_units": max(PUBLISHED_ACTIVE_UNITS),
            "support_boundary": (
                "maximal contiguous measured prefix before both derivation "
                "trials report nonpositive chain excess at active_units=16"),
        },
        "phase_entry_parameter": {
            "minimum_round_depth": min(PHASE_ENTRY_ROUND_COUNTS),
            "maximum_round_depth": max(PHASE_ENTRY_ROUND_COUNTS),
            "active_units": list(PUBLISHED_PHASE_ENTRY_ACTIVE_UNITS),
            "services": {
                str(grid): {
                    "service": (
                        "rmsnorm_phase_entry_first_1024"
                        if grid == 1 else
                        f"rmsnorm_phase_entry_first_1024_active{grid}"),
                    "service_time_ns_per_cta": phase_entry_parameters[grid],
                    "service_rate_per_ap_s": (
                        1.0e9 / phase_entry_parameters[grid]),
                    "deep_round_service_ns": phase_entry_slopes[grid],
                }
                for grid in PUBLISHED_PHASE_ENTRY_ACTIVE_UNITS
            },
            "scheduler_subtraction": {
                "model": "isolated_nonempty_task_scheduler_v1",
                "submit_ns": LAUNCH_SUBMIT_NS,
                "cta_descriptor_ns": LAUNCH_CTA_DESCRIPTOR_NS,
                "multi_wave_round_ns": LAUNCH_MULTI_WAVE_ROUND_NS,
                "waves_per_cta": BLOCK_THREADS // WAVE_SIZE,
            },
            "interpolation": "none_exact_active_unit_points",
            "support_boundary": (
                "maximal contiguous measured prefix with a positive fixed "
                "device-side phase entry after scheduler subtraction"),
        },
        "first_excluded_point": {
            "active_units": 16,
            "derivation_excess_ns_per_round": list(next_grid_excess),
        },
        "first_excluded_phase_entry": {
            "active_units": 32,
            "derivation_service_ns_per_cta": list(
                first_excluded_phase_entries),
        },
        "validation_rows": validation_rows,
        "validation": validation,
        "phase_entry_validation_rows": phase_entry_validation_rows,
        "phase_entry_validation": phase_entry_validation,
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
    parser.add_argument("--disassembly", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = aggregate_trials(
        args.trial, args.device_id, args.source, args.binary,
        args.disassembly, args.baseline_commit)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0 if report["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
