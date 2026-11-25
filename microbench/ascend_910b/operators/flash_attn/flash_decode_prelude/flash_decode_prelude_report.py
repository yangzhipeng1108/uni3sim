#!/usr/bin/env python3
"""Validate and summarize a CANN FIA FlashDecode-prelude capture."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from flash_decode_prelude_runner import (
    ACTIVE_AIC_GROUPS,
    MODES,
    TRIALS,
    _ordered_cells,
)


SYSTEM_COUNTER_TICK_NS = 20.0
MAXIMUM_HALF_MEDIAN_DELTA = 0.20
MAXIMUM_VALIDATION_MAPE = 0.10
MAXIMUM_VALIDATION_APE = 0.20
FIELDS = (
    "run_id",
    "trial",
    "order_position",
    "process_pid",
    "mode",
    "active_aic_groups",
    "actor_cycles",
    "validation",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_affine(points):
    x_mean = statistics.mean(x for x, _ in points)
    y_mean = statistics.mean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator <= 0:
        raise ValueError("FlashDecode prelude fit needs distinct group counts")
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in points
    ) / denominator
    intercept = y_mean - slope * x_mean
    if intercept <= 0 or slope <= 0:
        raise ValueError("FlashDecode prelude coefficients must be positive")
    return intercept, slope


def build_report(
        raw_path, *, source_commit, source_snapshot_sha256,
        binary_sha256, production_source_commit,
        fia_kernel_sha256, flash_decode_source_sha256):
    with raw_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("FlashDecode prelude raw header changed")
        source_rows = list(reader)
    expected_rows = TRIALS * len(ACTIVE_AIC_GROUPS) * len(MODES)
    if len(source_rows) != expected_rows:
        raise ValueError("FlashDecode prelude raw campaign is incomplete")

    rows = []
    run_ids = set()
    process_ids = set()
    for raw in source_rows:
        run_ids.add(raw["run_id"])
        process_ids.add(int(raw["process_pid"]))
        trial = int(raw["trial"])
        position = int(raw["order_position"])
        mode = int(raw["mode"])
        groups = int(raw["active_aic_groups"])
        if trial < 1 or trial > TRIALS:
            raise ValueError("FlashDecode prelude trial is outside campaign")
        order = _ordered_cells(trial - 1)
        if position < 0 or position >= len(order):
            raise ValueError("FlashDecode prelude order position is invalid")
        if order[position] != (mode, groups):
            raise ValueError("FlashDecode prelude row order changed")
        cycles = tuple(int(value) for value in raw["actor_cycles"].split(";"))
        if len(cycles) != 3 * groups or any(value <= 0 for value in cycles):
            raise ValueError("FlashDecode prelude actor cycles are invalid")
        if raw["validation"] != "pass":
            raise ValueError("FlashDecode prelude device validation failed")
        rows.append({
            "trial": trial,
            "mode": mode,
            "active_aic_groups": groups,
            "completion_ns": max(cycles) * SYSTEM_COUNTER_TICK_NS,
        })
    if len(run_ids) != 1 or len(process_ids) != 1:
        raise ValueError("FlashDecode prelude raw must come from one process")

    def cell_values(mode, groups, first_trial, last_trial):
        values = [
            row["completion_ns"] for row in rows
            if row["mode"] == mode
            and row["active_aic_groups"] == groups
            and first_trial <= row["trial"] <= last_trial
        ]
        if len(values) != last_trial - first_trial + 1:
            raise ValueError("FlashDecode prelude cell is incomplete")
        return values

    half = TRIALS // 2
    if TRIALS % 2 != 0:
        raise ValueError("FlashDecode prelude trials must split evenly")
    medians = {}
    spans = {}
    half_median_deltas = {}
    for mode in MODES:
        medians[mode] = {}
        spans[mode] = {}
        half_median_deltas[mode] = {}
        for groups in ACTIVE_AIC_GROUPS:
            values = cell_values(mode, groups, 1, TRIALS)
            median = statistics.median(values)
            first = statistics.median(cell_values(mode, groups, 1, half))
            second = statistics.median(
                cell_values(mode, groups, half + 1, TRIALS))
            if not math.isfinite(median) or median <= 0:
                raise ValueError("FlashDecode prelude median is invalid")
            medians[mode][groups] = median
            spans[mode][groups] = (max(values) - min(values)) / median
            half_median_deltas[mode][groups] = abs(first - second) / median

    training_points = tuple(
        (groups, statistics.median(cell_values(2, groups, 1, half)))
        for groups in ACTIVE_AIC_GROUPS)
    validation_points = tuple(
        (groups, statistics.median(
            cell_values(2, groups, half + 1, TRIALS)))
        for groups in ACTIVE_AIC_GROUPS)
    startup_ns, per_group_ns = _fit_affine(training_points)
    validation_apes = {
        groups: abs(startup_ns + per_group_ns * groups - duration_ns)
        / duration_ns
        for groups, duration_ns in validation_points
    }
    validation_mape = statistics.mean(validation_apes.values())
    validation_maximum_ape = max(validation_apes.values())
    maximum_half_median_delta = max(half_median_deltas[2].values())
    publishable = (
        maximum_half_median_delta <= MAXIMUM_HALF_MEDIAN_DELTA
        and validation_mape <= MAXIMUM_VALIDATION_MAPE
        and validation_maximum_ape <= MAXIMUM_VALIDATION_APE
    )
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "fia_flash_decode_terminal_prelude",
        "model": "fia_flash_decode_terminal_prelude_affine_v1",
        "request_size_unit": "active_aic_groups",
        "duration_unit": "ns_per_complete_terminal_prelude",
        "measured_modes": {
            "0": "all_actor_sync_all",
            "1": "all_actor_icache_preload_2_then_sync_all",
            "2": "aiv_buffer_reset_init_then_all_actor_icache_preload_2_then_sync_all",
        },
        "active_aic_groups": list(ACTIVE_AIC_GROUPS),
        "observed_median_duration_ns_by_mode": {
            str(mode): [medians[mode][groups] for groups in ACTIVE_AIC_GROUPS]
            for mode in MODES
        },
        "parameters_ns": {
            "startup": startup_ns,
            "per_active_aic_group": per_group_ns,
        },
        "validation_mape": validation_mape,
        "validation_maximum_ape": validation_maximum_ape,
        "maximum_allowed_validation_mape": MAXIMUM_VALIDATION_MAPE,
        "maximum_allowed_validation_ape": MAXIMUM_VALIDATION_APE,
        "validation_ape_by_active_aic_groups": {
            str(key): value for key, value in validation_apes.items()
        },
        "relative_span_by_mode_and_active_aic_groups": {
            str(mode): {
                str(key): value for key, value in spans[mode].items()
            }
            for mode in MODES
        },
        "relative_half_median_delta_by_mode_and_active_aic_groups": {
            str(mode): {
                str(key): value
                for key, value in half_median_deltas[mode].items()
            }
            for mode in MODES
        },
        "maximum_complete_prelude_half_median_delta": (
            maximum_half_median_delta),
        "maximum_allowed_half_median_delta": MAXIMUM_HALF_MEDIAN_DELTA,
        "trials": TRIALS,
        "clock": {
            "source": "GetSystemCycle",
            "frequency_hz": 50_000_000,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
        },
        "mechanism": (
            "source-counted AIV FlashDecode buffer Reset/InitBuffer followed "
            "by ICachePreLoad(2) and an all-AIC/AIV SyncAll"),
        "composition": (
            "one terminal dependency stage between the source main FIA DAG "
            "and FlashDecode Vector arithmetic"),
        "source_commit": source_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "binary_sha256": binary_sha256,
        "production_source_commit": production_source_commit,
        "fia_kernel_sha256": fia_kernel_sha256,
        "flash_decode_source_sha256": flash_decode_source_sha256,
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": publishable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--production-source-commit", required=True)
    parser.add_argument("--fia-kernel-sha256", required=True)
    parser.add_argument("--flash-decode-source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        args.raw,
        source_commit=args.source_commit,
        source_snapshot_sha256=args.source_snapshot_sha256,
        binary_sha256=args.binary_sha256,
        production_source_commit=args.production_source_commit,
        fia_kernel_sha256=args.fia_kernel_sha256,
        flash_decode_source_sha256=args.flash_decode_source_sha256,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["mechanism_publishable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
