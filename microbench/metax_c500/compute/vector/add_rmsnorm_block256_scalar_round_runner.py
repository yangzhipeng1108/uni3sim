#!/usr/bin/env python3
"""Reduce the C500 physical-block256 scalar AddRMSNorm round campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


HARDWARE_ID = "metax_c500"
SOFTWARE_REVISION = "maca_3.7.2"
MECHANISM = "bf16_generic_add_rmsnorm_physical_block256_scalar_round"
AP_COUNT = 104
PHYSICAL_BLOCK_THREADS = 256
FUNCTIONS = ("add_square", "scale_multiply")
ACTIVE_THREAD_COUNTS = (64, 128, 256)
ACTIVE_UNIT_GRIDS = (1, 2, 4, 8, 16, 32, 64, 104)
GRIDS = ACTIVE_UNIT_GRIDS + (208, 416, 832, 1664)
ROUND_COUNTS = (4, 8, 16, 32, 64)
SAMPLES_PER_CELL = 15
LAUNCHES_PER_BATCH = 8
TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
EXPECTED_RESIDENT_CTAS = 8
MAXIMUM_MAPE_PERCENT = 10.0
MAXIMUM_MAX_APE_PERCENT = 25.0


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


def _parse(path, expected_trial, expected_device):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    if (len(header) != 16
            or tuple(header[:3]) != ("metadata", "1", "MetaX C500")):
        raise ValueError(f"{path} has an invalid metadata row")
    values = tuple(
        _canonical_int(
            value, f"metadata[{index}]",
            positive=(index not in (11, 14)))
        for index, value in enumerate(header[3:], start=3))
    (device, ap_count, wave_size, clock_khz, trial, samples_per_cell,
     launches_per_batch, input_registers, input_shared, input_resident,
     output_registers, output_shared, output_resident) = values
    if (device != expected_device or trial != expected_trial
            or ap_count != AP_COUNT or wave_size != 64
            or samples_per_cell != SAMPLES_PER_CELL
            or launches_per_batch != LAUNCHES_PER_BATCH):
        raise ValueError(f"{path} changed the frozen campaign contract")
    resources = {
        "add_square": {
            "registers_per_thread": input_registers,
            "shared_bytes_per_cta": input_shared,
            "resident_ctas_per_ap": input_resident,
        },
        "scale_multiply": {
            "registers_per_thread": output_registers,
            "shared_bytes_per_cta": output_shared,
            "resident_ctas_per_ap": output_resident,
        },
    }
    if any(
            item["resident_ctas_per_ap"] != EXPECTED_RESIDENT_CTAS
            for item in resources.values()):
        raise ValueError(f"{path} changed physical block256 residency")
    expected_rows = (
        len(FUNCTIONS) * len(ACTIVE_THREAD_COUNTS) * len(GRIDS)
        * len(ROUND_COUNTS) * SAMPLES_PER_CELL)
    if len(rows[1:]) != expected_rows:
        raise ValueError(
            f"{path} has {len(rows[1:])} samples, expected {expected_rows}")
    orders = set()
    identities = set()
    samples = {
        (function, active_threads, grid, rounds): []
        for function in FUNCTIONS
        for active_threads in ACTIVE_THREAD_COUNTS
        for grid in GRIDS
        for rounds in ROUND_COUNTS
    }
    for row in rows[1:]:
        if len(row) != 8 or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = _canonical_int(row[1], "sample.order")
        function = row[2]
        active_threads = _canonical_int(
            row[3], "sample.active_threads", positive=True)
        grid = _canonical_int(row[4], "sample.grid", positive=True)
        rounds = _canonical_int(row[5], "sample.rounds", positive=True)
        sample = _canonical_int(row[6], "sample.sample")
        duration_ms = _positive_hex_float(row[7], "sample.duration_ms")
        if (function not in FUNCTIONS
                or active_threads not in ACTIVE_THREAD_COUNTS
                or grid not in GRIDS or rounds not in ROUND_COUNTS
                or sample >= SAMPLES_PER_CELL):
            raise ValueError(f"{path} has a sample outside the campaign")
        identity = (function, active_threads, grid, rounds, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} contains duplicate samples")
        orders.add(order)
        identities.add(identity)
        samples[(function, active_threads, grid, rounds)].append(
            duration_ms * 1_000_000.0 / LAUNCHES_PER_BATCH)
    if orders != set(range(expected_rows)):
        raise ValueError(f"{path} randomized order is not contiguous")
    return {
        "clock_khz": clock_khz,
        "resources": resources,
        "stdout_sha256": _sha256(path),
        "cell_medians_ns": {
            key: statistics.median(values)
            for key, values in samples.items()},
    }


def _round_service(parsed, function, active_threads, grid):
    slopes = []
    for lower, upper in zip(ROUND_COUNTS, ROUND_COUNTS[1:]):
        lower_ns = parsed["cell_medians_ns"][
            (function, active_threads, grid, lower)]
        upper_ns = parsed["cell_medians_ns"][
            (function, active_threads, grid, upper)]
        slope = (upper_ns - lower_ns) / (upper - lower)
        if slope <= 0.0:
            raise ValueError(
                "scalar-round adjacent-depth service must be positive: "
                f"{function}/{active_threads}/{grid}/{lower}:{upper}")
        slopes.append(slope)
    return statistics.median(slopes)


def _derive(parsed, function, active_threads):
    trial_services = {
        trial: {
            grid: _round_service(
                parsed[trial], function, active_threads, grid)
            for grid in GRIDS}
        for trial in DERIVATION_TRIALS}
    cells = {
        grid: statistics.median(
            trial_services[trial][grid] for trial in DERIVATION_TRIALS)
        for grid in GRIDS}
    first = {grid: cells[grid] for grid in ACTIVE_UNIT_GRIDS}
    depth = {grid // AP_COUNT: cells[grid]
             for grid in GRIDS if grid >= AP_COUNT}
    resident = []
    queued = []
    ordered = sorted(depth)
    for lower, upper in zip(ordered, ordered[1:]):
        slope = (depth[upper] - depth[lower]) / (upper - lower)
        if upper <= EXPECTED_RESIDENT_CTAS:
            resident.append(slope)
        elif lower >= EXPECTED_RESIDENT_CTAS:
            queued.append(slope)
        else:
            raise ValueError("grid interval crosses the residency boundary")
    resident_ns = statistics.median(resident)
    queued_ns = statistics.median(queued)
    if min(*first.values(), resident_ns, queued_ns) <= 0.0:
        raise ValueError("derived scalar-round services must be positive")
    return {
        "function": function,
        "active_threads": active_threads,
        "first_active_units": list(ACTIVE_UNIT_GRIDS),
        "first_service_ns": [first[grid] for grid in ACTIVE_UNIT_GRIDS],
        "resident_service_ns": resident_ns,
        "queued_service_ns": queued_ns,
        "resident_ctas_per_ap": EXPECTED_RESIDENT_CTAS,
    }


def _interpolate(points, values, target):
    if target in points:
        return values[points.index(target)]
    upper = next(index for index, point in enumerate(points)
                 if target < point)
    lower = upper - 1
    fraction = (target - points[lower]) / (points[upper] - points[lower])
    return values[lower] + fraction * (values[upper] - values[lower])


def _predict(parameter, grid):
    active = min(grid, AP_COUNT)
    depth = math.ceil(grid / AP_COUNT)
    first = _interpolate(
        parameter["first_active_units"], parameter["first_service_ns"],
        active)
    return (
        first
        + min(max(depth - 1, 0), EXPECTED_RESIDENT_CTAS - 1)
        * parameter["resident_service_ns"]
        + max(depth - EXPECTED_RESIDENT_CTAS, 0)
        * parameter["queued_service_ns"])


def _metrics(rows):
    errors = [abs(row["predicted_ns"] - row["measured_ns"]) for row in rows]
    apes = [error / row["measured_ns"] * 100.0
            for error, row in zip(errors, rows)]
    return {
        "count": len(rows),
        "mape_percent": statistics.fmean(apes),
        "wape_percent": math.fsum(errors) / math.fsum(
            row["measured_ns"] for row in rows) * 100.0,
        "max_ape_percent": max(apes),
    }


def aggregate(trial_paths, device_id, source, binary, baseline_commit):
    parsed = {
        trial: _parse(path, trial, device_id)
        for trial, path in zip(TRIALS, trial_paths)}
    if len({parsed[trial]["clock_khz"] for trial in TRIALS}) != 1:
        raise ValueError("device clock changed across trials")
    if len({json.dumps(parsed[trial]["resources"], sort_keys=True)
            for trial in TRIALS}) != 1:
        raise ValueError("compiled resource signature changed across trials")
    parameters = [
        _derive(parsed, function, active_threads)
        for function in FUNCTIONS
        for active_threads in ACTIVE_THREAD_COUNTS]
    validation_rows = []
    for parameter in parameters:
        for grid in GRIDS:
            measured = _round_service(
                parsed[VALIDATION_TRIAL], parameter["function"],
                parameter["active_threads"], grid)
            predicted = _predict(parameter, grid)
            validation_rows.append({
                "function": parameter["function"],
                "active_threads": parameter["active_threads"],
                "grid": grid,
                "measured_ns": measured,
                "predicted_ns": predicted,
                "ape_percent": abs(predicted - measured) / measured * 100.0,
            })
    metrics = _metrics(validation_rows)
    publishable = (
        metrics["mape_percent"] <= MAXIMUM_MAPE_PERCENT
        and metrics["max_ape_percent"] <= MAXIMUM_MAX_APE_PERCENT)
    return {
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "software_revision": SOFTWARE_REVISION,
        "mechanism": MECHANISM,
        "production_or_evaluation_duration_consumed": False,
        "baseline_commit": baseline_commit,
        "source": {"path": str(source), "sha256": _sha256(source)},
        "binary": {"path": str(binary), "sha256": _sha256(binary)},
        "trial_stdout_sha256": {
            str(trial): parsed[trial]["stdout_sha256"] for trial in TRIALS},
        "campaign": {
            "timer_domain": "mcevent_default_stream",
            "device_id": device_id,
            "physical_block_threads": PHYSICAL_BLOCK_THREADS,
            "active_thread_counts": list(ACTIVE_THREAD_COUNTS),
            "grids": list(GRIDS),
            "round_counts": list(ROUND_COUNTS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "launches_per_batch": LAUNCHES_PER_BATCH,
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
        },
        "resource_usage_by_function": parsed[TRIALS[0]]["resources"],
        "parameters": parameters,
        "validation_rows": validation_rows,
        "validation": metrics,
        "thresholds": {
            "maximum_mape_percent": MAXIMUM_MAPE_PERCENT,
            "maximum_max_ape_percent": MAXIMUM_MAX_APE_PERCENT,
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
    if len(args.trial) != len(TRIALS):
        raise ValueError("exactly three trial paths are required")
    report = aggregate(
        args.trial, args.device_id, args.source, args.binary,
        args.baseline_commit)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0 if report["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
