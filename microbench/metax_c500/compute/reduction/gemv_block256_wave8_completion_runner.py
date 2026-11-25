#!/usr/bin/env python3
"""Reduce the independent C500 block-256 GEMV completion campaign."""

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
MECHANISM = "four_wave_eight_output_reduction_completion"
AP_COUNT = 104
WAVE_SIZE = 64
PEUS_PER_AP = 4
BLOCK_THREADS = 256
WAVES_PER_CTA = BLOCK_THREADS // WAVE_SIZE
OUTPUTS_PER_CTA = 8
STATIC_SHARED_BYTES = 144
WAVE_COMPLETION_SLOTS_PER_AP = PEUS_PER_AP // WAVES_PER_CTA
GRIDS = (1, 104, 192, 208, 416, 640, 832, 1664)
FIT_GRIDS = (1, 104, 208, 416, 832, 1664)
HELD_OUT_GRIDS = (192, 640)
REPETITION_COUNTS = (4, 8, 16, 32)
MODES = ("control", "reduce8")
SAMPLES_PER_CELL = 31
LAUNCHES_PER_BATCH = 16
FULL_TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
MAX_VALIDATION_MAPE_PERCENT = 15.0
MAX_VALIDATION_MAX_APE_PERCENT = 25.0


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
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def parse_stdout(path, expected_trial, expected_device_id):
    """Parse one immutable trial into median nanoseconds per launch."""

    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    metadata = rows[0]
    if (len(metadata) != 16
            or tuple(metadata[:3]) != ("metadata", "1", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    integers = tuple(
        _canonical_int(
            value, f"metadata[{index}]",
            positive=index not in (3, 12, 13))
        for index, value in enumerate(metadata[3:], start=3)
    )
    (device_id, ap_count, wave_size, clock_khz, trial, samples_per_cell,
     launches_per_batch, control_registers, reduce8_registers,
     control_shared_bytes, reduce8_shared_bytes, control_resident_ctas,
     reduce8_resident_ctas) = integers
    if (device_id != expected_device_id or trial != expected_trial
            or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or samples_per_cell != SAMPLES_PER_CELL
            or launches_per_batch != LAUNCHES_PER_BATCH):
        raise ValueError(f"{path} changed the frozen campaign contract")
    if control_shared_bytes != 0 or reduce8_shared_bytes != STATIC_SHARED_BYTES:
        raise ValueError(f"{path} changed the shared-memory envelope")
    if min(control_resident_ctas, reduce8_resident_ctas) <= 0:
        raise ValueError(f"{path} reports non-positive occupancy")
    if control_resident_ctas != reduce8_resident_ctas:
        raise ValueError(
            f"{path} control/reduce8 occupancy differs: "
            f"{control_resident_ctas} != {reduce8_resident_ctas}")

    expected_count = (
        len(MODES) * len(GRIDS) * len(REPETITION_COUNTS)
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
        mode = row[2]
        grid = _canonical_int(row[3], "sample.grid", positive=True)
        repetitions = _canonical_int(
            row[4], "sample.repetitions", positive=True)
        sample = _canonical_int(row[5], "sample.sample")
        duration_ms = _positive_hex_float(row[6], "sample.duration_ms")
        if (mode not in MODES or grid not in GRIDS
                or repetitions not in REPETITION_COUNTS
                or sample >= SAMPLES_PER_CELL):
            raise ValueError(f"{path} has a sample outside the campaign")
        identity = (mode, grid, repetitions, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} contains duplicate samples")
        orders.add(order)
        identities.add(identity)
        cell = (mode, grid, repetitions)
        if cell not in samples:
            samples[cell] = []
        samples[cell].append(
            duration_ms * 1_000_000.0 / LAUNCHES_PER_BATCH)
    if orders != set(range(expected_count)):
        raise ValueError(f"{path} randomized order is not contiguous")
    expected_identities = {
        (mode, grid, repetitions, sample)
        for mode in MODES
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
        "resources": {
            "control": {
                "registers_per_thread": control_registers,
                "shared_bytes_per_cta": control_shared_bytes,
                "resident_ctas_per_ap": control_resident_ctas,
            },
            "reduce8": {
                "registers_per_thread": reduce8_registers,
                "shared_bytes_per_cta": reduce8_shared_bytes,
                "resident_ctas_per_ap": reduce8_resident_ctas,
            },
        },
        "cell_medians_ns": {
            key: statistics.median(values)
            for key, values in samples.items()
        },
        "stdout_sha256": _sha256(path),
    }


def _mode_slopes(parsed, mode, grid):
    slopes = []
    for lower, upper in zip(REPETITION_COUNTS, REPETITION_COUNTS[1:]):
        lower_ns = parsed["cell_medians_ns"][(mode, grid, lower)]
        upper_ns = parsed["cell_medians_ns"][(mode, grid, upper)]
        slope = (upper_ns - lower_ns) / (upper - lower)
        if not math.isfinite(slope):
            raise ValueError(
                "adjacent repetition service must be finite: "
                f"mode={mode}, grid={grid}, interval={lower}:{upper}")
        slopes.append(slope)
    return slopes


def _completion_service_by_grid(parsed):
    services = {}
    for grid in GRIDS:
        control = _mode_slopes(parsed, "control", grid)
        reduce8 = _mode_slopes(parsed, "reduce8", grid)
        deltas = tuple(
            reduce_ns - control_ns
            for control_ns, reduce_ns in zip(control, reduce8)
        )
        if any(delta <= 0.0 for delta in deltas):
            raise ValueError(
                "eight-output reduction service must exceed its control: "
                f"grid={grid}, deltas={deltas}")
        services[grid] = statistics.median(deltas)
    return services


def _cohorts(grid):
    depth = math.ceil(grid / AP_COUNT)
    return math.ceil(depth / WAVE_COMPLETION_SLOTS_PER_AP)


def _derive_parameters(parsed_trials):
    trial_services = {
        trial: _completion_service_by_grid(parsed_trials[trial])
        for trial in DERIVATION_TRIALS
    }
    training = {
        grid: statistics.median(
            trial_services[trial][grid] for trial in DERIVATION_TRIALS)
        for grid in FIT_GRIDS
    }
    resource = parsed_trials[DERIVATION_TRIALS[0]]["resources"]["reduce8"]
    if any(
            parsed_trials[trial]["resources"]["reduce8"] != resource
            for trial in FULL_TRIALS):
        raise ValueError("reduce8 resource identity changed across trials")
    first_ns = statistics.median(training[grid] for grid in (1, AP_COUNT))
    boundary_grids = (AP_COUNT, 2 * AP_COUNT, 4 * AP_COUNT,
                      8 * AP_COUNT, 16 * AP_COUNT)
    slopes = []
    for lower_grid, upper_grid in zip(
            boundary_grids, boundary_grids[1:]):
        slopes.append(
            (training[upper_grid] - training[lower_grid])
            / (_cohorts(upper_grid) - _cohorts(lower_grid)))
    subsequent_ns = statistics.median(slopes)
    if min(first_ns, subsequent_ns) <= 0.0:
        raise ValueError("derived completion services must be positive")
    return {
        "block_threads": BLOCK_THREADS,
        "waves_per_cta": WAVES_PER_CTA,
        "outputs_per_cta": OUTPUTS_PER_CTA,
        "static_shared_bytes_per_cta": STATIC_SHARED_BYTES,
        "resident_ctas_per_ap": resource["resident_ctas_per_ap"],
        "wave_completion_slots_per_ap": WAVE_COMPLETION_SLOTS_PER_AP,
        "first_wave_cohort_completion_ns_per_ap": first_ns,
        "subsequent_wave_cohort_completion_ns_per_ap": subsequent_ns,
        "first_wave_cohort_completion_rate_per_ap_s": 1.0e9 / first_ns,
        "subsequent_wave_cohort_completion_rate_per_ap_s": (
            1.0e9 / subsequent_ns),
    }


def _predict_ns(parameters, grid):
    return (
        parameters["first_wave_cohort_completion_ns_per_ap"]
        + (_cohorts(grid) - 1)
        * parameters["subsequent_wave_cohort_completion_ns_per_ap"])


def _metrics(rows):
    apes = [
        abs(row["predicted_ns"] - row["measured_ns"])
        / row["measured_ns"] * 100.0
        for row in rows
    ]
    return {
        "count": len(rows),
        "mape_percent": statistics.fmean(apes),
        "max_ape_percent": max(apes),
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
    parameters = _derive_parameters(parsed)
    validation_rows = []
    for trial in FULL_TRIALS:
        services = _completion_service_by_grid(parsed[trial])
        validation_grids = (
            GRIDS if trial == VALIDATION_TRIAL else HELD_OUT_GRIDS)
        for grid in validation_grids:
            validation_rows.append({
                "trial": trial,
                "grid": grid,
                "measured_ns": services[grid],
                "predicted_ns": _predict_ns(parameters, grid),
                "held_out_geometry": grid in HELD_OUT_GRIDS,
                "held_out_trial": trial == VALIDATION_TRIAL,
            })
    metrics = _metrics(validation_rows)
    publishable = (
        metrics["mape_percent"] <= MAX_VALIDATION_MAPE_PERCENT
        and metrics["max_ape_percent"] <= MAX_VALIDATION_MAX_APE_PERCENT)
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "software_revision": SOFTWARE_REVISION,
        "mechanism": MECHANISM,
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "baseline_commit": baseline_commit,
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "analysis": {
            "path": (
                "microbench/metax_c500/compute/reduction/"
                "gemv_block256_wave8_completion_runner.py"),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "forward_scope": {
            "dtype": "bf16",
            "block_threads": BLOCK_THREADS,
            "waves_per_cta": WAVES_PER_CTA,
            "outputs_per_cta": OUTPUTS_PER_CTA,
            "static_shared_bytes_per_cta": STATIC_SHARED_BYTES,
            "minimum_grid_size": 1,
            "maximum_resident_grid_size": (
                AP_COUNT * parameters["resident_ctas_per_ap"]),
            "requires_four_wave_cta": True,
            "requires_fully_resident_grid": True,
        },
        "trial_stdout_sha256": {
            str(trial): parsed[trial]["stdout_sha256"]
            for trial in FULL_TRIALS
        },
        "campaign": {
            "timer_domain": "mcevent_paired_default_stream",
            "derivation": (
                "adjacent_runtime_repetition_count_reduce8_minus_control"),
            "trials": list(FULL_TRIALS),
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
            "block_threads": BLOCK_THREADS,
            "waves_per_cta": WAVES_PER_CTA,
            "outputs_per_cta": OUTPUTS_PER_CTA,
            "peus_per_ap": PEUS_PER_AP,
            "grids": list(GRIDS),
            "fit_grids": list(FIT_GRIDS),
            "held_out_grids": list(HELD_OUT_GRIDS),
            "repetition_counts": list(REPETITION_COUNTS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "launches_per_batch": LAUNCHES_PER_BATCH,
        },
        "resources": {
            str(trial): parsed[trial]["resources"]
            for trial in FULL_TRIALS
        },
        "parameters": parameters,
        "validation_rows": validation_rows,
        "validation": metrics,
        "thresholds": {
            "maximum_mape_percent": MAX_VALIDATION_MAPE_PERCENT,
            "maximum_max_ape_percent": MAX_VALIDATION_MAX_APE_PERCENT,
        },
        "publishable": publishable,
        "disposition": (
            "candidate_for_forward_evaluation" if publishable
            else "rejected_before_forward_evaluation"),
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
