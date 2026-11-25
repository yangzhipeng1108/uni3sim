#!/usr/bin/env python3
"""Audit C500 source-level scalar-math dependency-chain trials."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


CAMPAIGNS = {
    "floating_math": {
        "functions": (
            "exp_neg_f32",
            "reciprocal_one_plus_f32",
        ),
        "required_publishable": ("exp_neg_f32",),
        "validation_kind": "zero_intercept_total",
        "block_threads": 64,
        "counts": (4, 8, 16, 32, 64),
    },
    "floating_math_runtime": {
        "functions": (
            "exp_neg_f32",
            "reciprocal_one_plus_f32",
        ),
        "required_publishable": (
            "exp_neg_f32",
            "reciprocal_one_plus_f32",
        ),
        "validation_kind": "incremental_rate",
        "block_threads": 512,
        "counts": (4, 8, 16, 32, 64),
    },
    "floating_math_throughput": {
        "functions": (
            "exp_neg_f32",
            "reciprocal_one_plus_f32",
        ),
        "required_publishable": (
            "exp_neg_f32",
            "reciprocal_one_plus_f32",
        ),
        "validation_kind": "event_latency_and_throughput",
        "block_threads": 512,
        "grids": (1, 416),
        "launches_per_batch": 16,
        "counts": (64, 128, 256, 512, 1024),
    },
    "integer_divrem": {
        "functions": (
            "div_u32",
            "rem_u32",
            "div_u64",
            "rem_u64",
        ),
        "required_publishable": (
            "div_u32",
            "rem_u32",
            "div_u64",
            "rem_u64",
        ),
        "validation_kind": "incremental_rate",
        "block_threads": 64,
        "counts": (4, 8, 16, 32, 64),
    },
    "conversion_address": {
        "functions": (
            "bf16_roundtrip_pair",
            "address_multiply_add_u64",
        ),
        "required_publishable": (
            "bf16_roundtrip_pair",
            "address_multiply_add_u64",
        ),
        "validation_kind": "incremental_rate",
        "block_threads": 64,
        "counts": (4, 8, 16, 32, 64),
    },
}
EXPECTED_COUNTS = (4, 8, 16, 32, 64)
EXPECTED_TRIALS = (1, 2, 3)
EXPECTED_SAMPLES = 31
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
MAX_VALIDATION_APE_PERCENT = 15.0


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_trial(
        path, expected_trial, expected_functions, expected_block_threads,
        expected_counts, expected_grids=None,
        expected_launches_per_batch=None):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0][0] != "metadata":
        raise ValueError(f"{path} has an invalid metadata row")
    metadata = rows[0]
    grids = None
    launches_per_batch = None
    if metadata[1] == "1" and len(metadata) == 9:
        block_threads = 64
        trial_index = 7
        samples_index = 8
    elif metadata[1] == "2" and len(metadata) == 10:
        block_threads = int(metadata[7])
        trial_index = 8
        samples_index = 9
    elif metadata[1] == "3" and len(metadata) == 13:
        block_threads = int(metadata[7])
        trial_index = 8
        samples_index = 9
        launches_per_batch = int(metadata[10])
        grids = (int(metadata[11]), int(metadata[12]))
    else:
        raise ValueError(f"{path} has an unsupported metadata schema")
    if metadata[2] != "MetaX C500":
        raise ValueError(f"{path} has an unexpected schema or device")
    device_id = int(metadata[3])
    ap_count = int(metadata[4])
    wave_size = int(metadata[5])
    clock_khz = int(metadata[6])
    trial = int(metadata[trial_index])
    samples = int(metadata[samples_index])
    if (trial != expected_trial or ap_count != 104 or wave_size != 64
            or clock_khz <= 0 or block_threads != expected_block_threads
            or block_threads % wave_size
            or samples != EXPECTED_SAMPLES):
        raise ValueError(f"{path} changed the frozen campaign contract")
    event_campaign = expected_grids is not None
    if event_campaign:
        if (metadata[1] != "3" or grids != expected_grids
                or launches_per_batch != expected_launches_per_batch):
            raise ValueError(f"{path} changed the event campaign contract")
    elif metadata[1] == "3":
        raise ValueError(f"{path} uses event data for a clock campaign")
    expected_rows = (
        len(expected_functions) * len(expected_counts) * EXPECTED_SAMPLES
        * (len(expected_grids) if event_campaign else 1))
    if len(rows[1:]) != expected_rows:
        raise ValueError(f"{path} is incomplete")
    values = {}
    orders = set()
    identities = set()
    for row in rows[1:]:
        expected_columns = 7 if event_campaign else 6
        if len(row) != expected_columns or row[0] != "sample":
            raise ValueError(f"{path} has an invalid sample row")
        order = int(row[1])
        function = row[2]
        if event_campaign:
            grid = int(row[3])
            count = int(row[4])
            sample = int(row[5])
            elapsed_ms = float.fromhex(row[6])
            value = elapsed_ms * 1_000_000.0 / launches_per_batch
        else:
            grid = None
            count = int(row[3])
            sample = int(row[4])
            value = int(row[5])
        if (function not in expected_functions
                or (event_campaign and grid not in expected_grids)
                or count not in expected_counts
                or sample < 0 or sample >= EXPECTED_SAMPLES
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{path} has an invalid sample value")
        identity = (function, grid, count, sample)
        if order in orders or identity in identities:
            raise ValueError(f"{path} has a duplicate sample")
        orders.add(order)
        identities.add(identity)
        key = ((function, grid, count) if event_campaign
               else (function, count))
        if key not in values:
            values[key] = []
        values[key].append(value)
    if orders != set(range(expected_rows)):
        raise ValueError(f"{path} has a non-contiguous randomized order")
    return {
        "device_id": device_id,
        "ap_count": ap_count,
        "wave_size": wave_size,
        "clock_khz": clock_khz,
        "block_threads": block_threads,
        "grids": expected_grids,
        "launches_per_batch": expected_launches_per_batch,
        "trial": trial,
        "values": values,
    }


def _derive_function(trials, function, validation_kind, counts):
    medians = {
        trial: {
            count: statistics.median(
                trials[trial]["values"][(function, count)])
            for count in counts
        }
        for trial in EXPECTED_TRIALS
    }
    slopes = []
    for trial in DERIVATION_TRIALS:
        for low, high in zip(counts, counts[1:]):
            slopes.append(
                (medians[trial][high] - medians[trial][low])
                / (high - low))
    cycles_per_function = statistics.median(slopes)
    base_cycles = statistics.median([
        medians[trial][count] - cycles_per_function * count
        for trial in DERIVATION_TRIALS
        for count in counts
    ])
    validation = []
    if validation_kind == "zero_intercept_total":
        for count in counts:
            measured = medians[VALIDATION_TRIAL][count]
            predicted = cycles_per_function * count
            ape_percent = abs(predicted - measured) / measured * 100.0
            validation.append({
                "count": count,
                "measured_cycles": measured,
                "predicted_cycles": predicted,
                "ape_percent": ape_percent,
            })
    elif validation_kind == "incremental_rate":
        # Runtime-loop campaigns use one binary for every count.  Differencing
        # adjacent counts removes clock64/loop-entry overhead and validates the
        # exact incremental service rate consumed by HWSpec.
        for low, high in zip(counts, counts[1:]):
            measured = (
                (medians[VALIDATION_TRIAL][high]
                 - medians[VALIDATION_TRIAL][low])
                / (high - low))
            predicted = cycles_per_function
            ape_percent = abs(predicted - measured) / measured * 100.0
            validation.append({
                "count_interval": [low, high],
                "measured_cycles_per_function": measured,
                "predicted_cycles_per_function": predicted,
                "ape_percent": ape_percent,
            })
    else:
        raise ValueError(f"unknown validation kind: {validation_kind!r}")
    mape = statistics.fmean(row["ape_percent"] for row in validation)
    if validation_kind == "zero_intercept_total":
        measured_values = [
            row["measured_cycles"] for row in validation]
        predicted_values = [
            row["predicted_cycles"] for row in validation]
    elif validation_kind == "incremental_rate":
        measured_values = [
            row["measured_cycles_per_function"] for row in validation]
        predicted_values = [
            row["predicted_cycles_per_function"] for row in validation]
    else:
        raise AssertionError(
            f"unhandled validated kind: {validation_kind!r}")
    measured_sum = math.fsum(measured_values)
    wmae = (
        math.fsum(abs(predicted - measured) for predicted, measured in zip(
            predicted_values, measured_values))
        / measured_sum * 100.0)
    maximum_ape = max(row["ape_percent"] for row in validation)
    return {
        "cycles_per_function": cycles_per_function,
        "base_cycles": base_cycles,
        "validation_kind": validation_kind,
        "derivation_trials": list(DERIVATION_TRIALS),
        "validation_trial": VALIDATION_TRIAL,
        "validation": validation,
        "validation_mape_percent": mape,
        "validation_wmae_percent": wmae,
        "validation_max_ape_percent": maximum_ape,
        "maximum_allowed_validation_ape_percent": (
            MAX_VALIDATION_APE_PERCENT),
        "publishable": maximum_ape < MAX_VALIDATION_APE_PERCENT,
    }


def _derive_event_function(trials, function, ap_count, grids, counts):
    medians = {
        trial: {
            (grid, count): statistics.median(
                trials[trial]["values"][(function, grid, count)])
            for grid in grids
            for count in counts
        }
        for trial in EXPECTED_TRIALS
    }
    slopes_by_grid = {grid: [] for grid in grids}
    for trial in DERIVATION_TRIALS:
        for grid in grids:
            for low, high in zip(counts, counts[1:]):
                slopes_by_grid[grid].append(
                    (medians[trial][(grid, high)]
                     - medians[trial][(grid, low)])
                    / (high - low))
    slope_ns_by_grid = {
        grid: statistics.median(slopes)
        for grid, slopes in slopes_by_grid.items()
    }
    if any(value <= 0 for value in slope_ns_by_grid.values()):
        raise ValueError(
            f"{function} event campaign has a non-positive increment")
    single_grid, saturated_grid = grids
    latency_ns = slope_ns_by_grid[single_grid]
    saturated_rate_per_ap_s = (
        saturated_grid / ap_count
        / (slope_ns_by_grid[saturated_grid] * 1e-9))
    validation = []
    for grid in grids:
        predicted = slope_ns_by_grid[grid]
        for low, high in zip(counts, counts[1:]):
            measured = (
                (medians[VALIDATION_TRIAL][(grid, high)]
                 - medians[VALIDATION_TRIAL][(grid, low)])
                / (high - low))
            if measured <= 0:
                raise ValueError(
                    f"{function} validation increment is non-positive")
            validation.append({
                "grid": grid,
                "count_interval": [low, high],
                "measured_ns_per_block_layer": measured,
                "predicted_ns_per_block_layer": predicted,
                "ape_percent": abs(predicted - measured) / measured * 100.0,
            })
    mape = statistics.fmean(row["ape_percent"] for row in validation)
    measured_sum = math.fsum(
        row["measured_ns_per_block_layer"] for row in validation)
    wmae = (
        math.fsum(abs(
            row["predicted_ns_per_block_layer"]
            - row["measured_ns_per_block_layer"])
            for row in validation)
        / measured_sum * 100.0)
    maximum_ape = max(row["ape_percent"] for row in validation)
    validation_by_grid = {}
    for grid in grids:
        rows = [row for row in validation if row["grid"] == grid]
        grid_measured_sum = math.fsum(
            row["measured_ns_per_block_layer"] for row in rows)
        validation_by_grid[str(grid)] = {
            "mape_percent": statistics.fmean(
                row["ape_percent"] for row in rows),
            "wmae_percent": (
                math.fsum(abs(
                    row["predicted_ns_per_block_layer"]
                    - row["measured_ns_per_block_layer"])
                    for row in rows)
                / grid_measured_sum * 100.0),
            "maximum_ape_percent": max(
                row["ape_percent"] for row in rows),
        }
    latency_publishable = (
        validation_by_grid[str(single_grid)]["maximum_ape_percent"]
        < MAX_VALIDATION_APE_PERCENT)
    throughput_publishable = (
        validation_by_grid[str(saturated_grid)]["maximum_ape_percent"]
        < MAX_VALIDATION_APE_PERCENT)
    return {
        "latency_ns_per_block_layer": latency_ns,
        "saturated_block_layers_per_ap_s": saturated_rate_per_ap_s,
        "single_grid": single_grid,
        "saturated_grid": saturated_grid,
        "derivation_trials": list(DERIVATION_TRIALS),
        "validation_trial": VALIDATION_TRIAL,
        "validation": validation,
        "validation_by_grid": validation_by_grid,
        "validation_mape_percent": mape,
        "validation_wmae_percent": wmae,
        "validation_max_ape_percent": maximum_ape,
        "maximum_allowed_validation_ape_percent": (
            MAX_VALIDATION_APE_PERCENT),
        "latency_publishable": latency_publishable,
        "throughput_publishable": throughput_publishable,
        "publishable": latency_publishable and throughput_publishable,
    }


def build_report(raw_dir, source_snapshot, campaign):
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign: {campaign!r}")
    campaign_spec = CAMPAIGNS[campaign]
    expected_functions = campaign_spec["functions"]
    expected_counts = campaign_spec["counts"]
    expected_grids = None
    if "grids" in campaign_spec:
        expected_grids = campaign_spec["grids"]
    expected_launches_per_batch = None
    if "launches_per_batch" in campaign_spec:
        expected_launches_per_batch = campaign_spec["launches_per_batch"]
    trials = {
        trial: _parse_trial(
            raw_dir / f"trial_{trial}.csv", trial, expected_functions,
            campaign_spec["block_threads"], expected_counts, expected_grids,
            expected_launches_per_batch)
        for trial in EXPECTED_TRIALS
    }
    metadata = {
        (value["device_id"], value["ap_count"], value["wave_size"],
         value["clock_khz"], value["block_threads"], value["grids"],
         value["launches_per_batch"])
        for value in trials.values()
    }
    if len(metadata) != 1:
        raise ValueError("trial hardware metadata is inconsistent")
    (device_id, ap_count, wave_size, clock_khz, block_threads,
     grids, launches_per_batch) = metadata.pop()
    validation_kind = campaign_spec["validation_kind"]
    if validation_kind == "event_latency_and_throughput":
        functions = {
            function: _derive_event_function(
                trials, function, ap_count, grids, expected_counts)
            for function in expected_functions
        }
    elif validation_kind in ("zero_intercept_total", "incremental_rate"):
        functions = {
            function: _derive_function(
                trials, function, validation_kind, expected_counts)
            for function in expected_functions
        }
    else:
        raise ValueError(
            f"unknown semantic dependency validation kind: {validation_kind!r}")
    return {
        "schema_version": 1,
        "mechanism": "source_level_scalar_dependency_chain",
        "campaign": campaign,
        "hardware_id": "metax_c500",
        "software_revision": "maca_3.7.2",
        "device_id": device_id,
        "ap_count": ap_count,
        "wave_size": wave_size,
        "block_threads": block_threads,
        "waves_per_block": block_threads // wave_size,
        "grids": list(grids) if grids is not None else None,
        "launches_per_batch": launches_per_batch,
        "observed_clock_khz": clock_khz,
        "report_script_sha256": _sha256(Path(__file__)),
        "source_sha256": _sha256(source_snapshot),
        "raw_sha256": {
            f"trial_{trial}.csv": _sha256(
                raw_dir / f"trial_{trial}.csv")
            for trial in EXPECTED_TRIALS
        },
        "functions": functions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--campaign", required=True, choices=CAMPAIGNS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.raw_dir, args.source_snapshot, args.campaign)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["functions"], indent=2, sort_keys=True))
    validation_kind = CAMPAIGNS[args.campaign]["validation_kind"]
    publish_field = {
        "event_latency_and_throughput": "throughput_publishable",
        "zero_intercept_total": "publishable",
        "incremental_rate": "publishable",
    }[validation_kind]
    failed = tuple(
        function
        for function in CAMPAIGNS[args.campaign]["required_publishable"]
        if not report["functions"][function][publish_field])
    if failed:
        raise SystemExit(
            f"required semantic dependency gates failed: {failed}")


if __name__ == "__main__":
    main()
