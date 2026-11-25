#!/usr/bin/env python3
"""Audit three independent elementwise source-control lifecycle captures."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from elementwise_control_runner import (
    AIV_WORKERS,
    CALIBRATION_TILE_COUNTS,
    CELLS,
    EXPECTED_CANN,
    HOLDOUT_TILE_COUNTS,
    RAW_FIELDS,
    SOURCE_FILENAMES,
    STATIC_REFERENCE_FILENAME,
    STATIC_REFERENCE_SHA256,
    SYSTEM_COUNTER_HZ,
    SYSTEM_COUNTER_TICK_NS,
    TILE_COUNTS,
    TRIALS,
    VARIANTS,
    _ordered_cells,
    _sha256,
    _snapshot_sha256,
)


REQUIRED_RUN_COUNT = 3
CROSS_RUN_LIMIT = 0.20
POSITION_BALANCE_LIMIT = 0.20
HOLDOUT_MAPE_LIMIT = 0.15
HOLDOUT_MAX_APE_LIMIT = 0.25


def _relative_span(values):
    center = statistics.median(values)
    if center <= 0:
        raise ValueError("relative-span center must be positive")
    return (max(values) - min(values)) / center


def _worker_vector(text, *, positive):
    values = tuple(int(value) for value in text.split(";"))
    if len(values) != AIV_WORKERS:
        raise ValueError("raw row does not contain exactly 48 AIV values")
    if positive and any(value <= 0 for value in values):
        raise ValueError("raw cycle vector contains a non-positive value")
    if not positive and any(value == 0 for value in values):
        raise ValueError("raw checksum vector contains zero")
    return values


def _expected_campaign():
    return {
        "variants": list(VARIANTS),
        "tile_counts": list(TILE_COUNTS),
        "calibration_tile_counts": list(CALIBRATION_TILE_COUNTS),
        "holdout_tile_counts": list(HOLDOUT_TILE_COUNTS),
        "aiv_workers": AIV_WORKERS,
        "trials": TRIALS,
        "order": "balanced_forward_reverse",
    }


def _load_run(run_dir, benchmark_root):
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["schema"] != (
            "ascend_910b_elementwise_control_lifecycle_provenance_v1"):
        raise ValueError("unsupported lifecycle provenance schema")
    if provenance["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("capture consumed forbidden production duration")
    if provenance["source_snapshot_sha256"] != _snapshot_sha256(benchmark_root):
        raise ValueError("capture source snapshot differs from report source")
    if set(provenance["source_files"]) != set(SOURCE_FILENAMES):
        raise ValueError("capture source file set changed")
    for filename in SOURCE_FILENAMES:
        if provenance["source_files"][filename] != _sha256(
                benchmark_root / filename):
            raise ValueError(f"capture source file changed: {filename}")
    reference = benchmark_root.parents[2] / "references" / STATIC_REFERENCE_FILENAME
    if (_sha256(reference) != STATIC_REFERENCE_SHA256
            or provenance["static_reference"]["sha256"]
            != STATIC_REFERENCE_SHA256):
        raise ValueError("static DAG accounting reference changed")
    if provenance["cann"]["identity"] != EXPECTED_CANN:
        raise ValueError("capture CANN identity changed")
    if provenance["device"]["model"] != "910B2":
        raise ValueError("capture device is not 910B2")
    if provenance["clock"] != {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS}:
        raise ValueError("capture clock changed")
    if provenance["campaign"] != _expected_campaign():
        raise ValueError("capture campaign changed")
    raw_path = run_dir / provenance["capture"]["raw"]
    if _sha256(raw_path) != provenance["capture"]["raw_sha256"]:
        raise ValueError("capture raw SHA256 mismatch")

    rows = []
    identities = set()
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("capture raw schema changed")
        for raw in reader:
            trial = int(raw["trial"])
            if trial < 1 or trial > TRIALS:
                raise ValueError("capture trial is outside the campaign")
            direction, cells = _ordered_cells(trial - 1)
            position = int(raw["order_position"])
            if position < 0 or position >= len(cells):
                raise ValueError("capture order position is invalid")
            variant, tile_count = cells[position]
            cycles = _worker_vector(raw["worker_cycles"], positive=True)
            _worker_vector(raw["worker_checksums"], positive=False)
            completion_cycles = int(raw["completion_cycles"])
            completion_ns = float(raw["completion_ns"])
            identity = (trial, variant, tile_count)
            if (identity in identities or raw["run_id"] != provenance["run_id"]
                    or raw["order_direction"] != direction
                    or raw["variant"] != variant
                    or int(raw["tile_count"]) != tile_count
                    or int(raw["process_pid"]) <= 0
                    or raw["validation"] != "ok"
                    or completion_cycles != max(cycles)
                    or not math.isfinite(completion_ns)
                    or not math.isclose(
                        completion_ns,
                        completion_cycles * SYSTEM_COUNTER_TICK_NS,
                        rel_tol=0.0, abs_tol=1e-9)):
                raise ValueError("capture row changed the frozen campaign")
            identities.add(identity)
            rows.append({
                "run_id": provenance["run_id"],
                "trial": trial,
                "direction": direction,
                "variant": variant,
                "tile_count": tile_count,
                "completion_ns": completion_ns,
            })
    if len(rows) != TRIALS * len(CELLS):
        raise ValueError("capture is incomplete")
    return provenance, rows


def _fit_affine(samples):
    if tuple(sorted(samples)) != tuple(CALIBRATION_TILE_COUNTS):
        raise ValueError("affine calibration domain changed")
    x_mean = statistics.mean(samples)
    y_mean = statistics.mean(samples.values())
    denominator = sum((x - x_mean) ** 2 for x in samples)
    slope = sum(
        (x - x_mean) * (samples[x] - y_mean) for x in samples
    ) / denominator
    return y_mean - slope * x_mean, slope


def _build_report(run_dirs):
    if len(run_dirs) != REQUIRED_RUN_COUNT:
        raise ValueError("report requires exactly three capture directories")
    benchmark_root = Path(__file__).resolve().parent
    loaded = tuple(_load_run(path, benchmark_root) for path in run_dirs)
    provenances = tuple(value[0] for value in loaded)
    rows = tuple(row for value in loaded for row in value[1])
    run_ids = tuple(value["run_id"] for value in provenances)
    if len(set(run_ids)) != REQUIRED_RUN_COUNT:
        raise ValueError("report requires three independent run IDs")
    frozen_fields = (
        "source_commit", "source_snapshot_sha256", "source_files", "binary",
        "static_reference", "cann", "device", "clock", "campaign",
    )
    for field in frozen_fields:
        if any(
                value[field] != provenances[0][field]
                for value in provenances[1:]):
            raise ValueError(f"independent runs disagree on {field}")

    run_cell_medians = {}
    position_errors = {}
    for run_id in run_ids:
        for variant, tile_count in CELLS:
            cell_rows = tuple(
                row for row in rows
                if row["run_id"] == run_id
                and row["variant"] == variant
                and row["tile_count"] == tile_count)
            if len(cell_rows) != TRIALS:
                raise ValueError("combined runs lost a campaign cell")
            values = tuple(row["completion_ns"] for row in cell_rows)
            run_cell_medians[(run_id, variant, tile_count)] = (
                statistics.median(values))
            forward = tuple(
                row["completion_ns"] for row in cell_rows
                if row["direction"] == "forward")
            reverse = tuple(
                row["completion_ns"] for row in cell_rows
                if row["direction"] == "reverse")
            if len(forward) != TRIALS // 2 or len(reverse) != TRIALS // 2:
                raise ValueError("capture is not forward/reverse balanced")
            position_errors[f"{run_id}:{variant}:{tile_count}"] = (
                abs(statistics.median(forward) - statistics.median(reverse))
                / statistics.median(values))

    cell_medians = {}
    cross_run_spans = {}
    for variant, tile_count in CELLS:
        medians = tuple(
            run_cell_medians[(run_id, variant, tile_count)]
            for run_id in run_ids)
        cell_medians[(variant, tile_count)] = statistics.median(medians)
        cross_run_spans[f"{variant}:{tile_count}"] = _relative_span(medians)

    deltas = {}
    for tile_count in TILE_COUNTS:
        control = cell_medians[("control_baseline", tile_count)]
        dag = cell_medians[("dag_tbuf", tile_count)]
        full = cell_medians[("full_lifecycle", tile_count)]
        deltas[tile_count] = {
            "dag_tbuf_minus_control_ns": dag - control,
            "full_lifecycle_minus_dag_tbuf_ns": full - dag,
            "full_lifecycle_minus_control_ns": full - control,
        }
    candidate_samples = {
        tile_count: deltas[tile_count]["full_lifecycle_minus_control_ns"]
        for tile_count in CALIBRATION_TILE_COUNTS
    }
    intercept, slope = _fit_affine(candidate_samples)
    holdout = []
    for tile_count in HOLDOUT_TILE_COUNTS:
        observed = deltas[tile_count]["full_lifecycle_minus_control_ns"]
        predicted = intercept + slope * tile_count
        if observed <= 0:
            error = math.inf
        else:
            error = abs(predicted - observed) / observed
        holdout.append({
            "tile_count": tile_count,
            "observed_ns": observed,
            "predicted_ns": predicted,
            "absolute_percentage_error": error,
        })
    holdout_mape = statistics.mean(
        value["absolute_percentage_error"] for value in holdout)
    holdout_max_ape = max(
        value["absolute_percentage_error"] for value in holdout)
    positive_deltas = all(
        math.isfinite(value)
        and value > 0
        for per_tile in deltas.values()
        for value in per_tile.values())
    positive_parameters = (
        math.isfinite(intercept) and intercept > 0
        and math.isfinite(slope) and slope > 0)
    gates = {
        "source_binary_reference_raw_and_device_identity_complete": True,
        "exactly_three_independent_runs": True,
        "all_cells_cross_run_median_spread_at_most_20pct": (
            max(cross_run_spans.values()) <= CROSS_RUN_LIMIT),
        "all_cells_position_balance_error_at_most_20pct": (
            max(position_errors.values()) <= POSITION_BALANCE_LIMIT),
        "all_paired_deltas_positive": positive_deltas,
        "positive_affine_intercept_and_slope": positive_parameters,
        "holdout_mape_at_most_15pct": holdout_mape <= HOLDOUT_MAPE_LIMIT,
        "holdout_max_ape_at_most_25pct": (
            holdout_max_ape <= HOLDOUT_MAX_APE_LIMIT),
    }
    publishable = all(gates.values())
    parameters = None
    if publishable:
        parameters = {
            "startup_ns": intercept,
            "per_tile_ns": slope,
        }
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": "elementwise_source_control_lifecycle_48_aiv_v1",
        "mechanism_publishable": publishable,
        "production_or_evaluation_duration_consumed": False,
        "scope": (
            "source-control-only seven-node traversal, twelve buffer-ID "
            "lookups, one VECCALC TBuf and exact per-tile MTE2/V/MTE3 mutex "
            "lifecycle; excludes BF16 Add/Mul tensor payloads"),
        "duration_unit": "ns_per_48_aiv_source_control_program_completion",
        "model": {
            "formula": "startup_ns + tile_count * per_tile_ns",
            "parameters_ns": parameters,
            "calibration_tile_counts": list(CALIBRATION_TILE_COUNTS),
            "holdout_tile_counts": list(HOLDOUT_TILE_COUNTS),
            "composition": (
                "max(existing_48_aiv_tile_pipeline_span, "
                "source_control_candidate); never add the two spans"),
        },
        "gates": gates,
        "thresholds": {
            "cross_run_relative_span": CROSS_RUN_LIMIT,
            "position_balance_relative_error": POSITION_BALANCE_LIMIT,
            "holdout_mean_absolute_percentage_error": HOLDOUT_MAPE_LIMIT,
            "holdout_maximum_absolute_percentage_error": HOLDOUT_MAX_APE_LIMIT,
        },
        "diagnostics": {
            "cell_medians_ns": {
                f"{variant}:{tile_count}": value
                for (variant, tile_count), value in cell_medians.items()
            },
            "paired_deltas_ns": {
                str(tile_count): value for tile_count, value in deltas.items()
            },
            "cell_cross_run_relative_spans": cross_run_spans,
            "position_balance_relative_errors": position_errors,
            "holdout_mean_absolute_percentage_error": holdout_mape,
            "holdout_maximum_absolute_percentage_error": holdout_max_ape,
            "holdout_cells": holdout,
        },
        "provenance": {
            "run_ids": list(run_ids),
            "source_commit": provenances[0]["source_commit"],
            "source_snapshot_sha256": provenances[0][
                "source_snapshot_sha256"],
            "binary_sha256": provenances[0]["binary"]["sha256"],
            "static_reference_sha256": STATIC_REFERENCE_SHA256,
            "raw_sha256": [
                value["capture"]["raw_sha256"] for value in provenances
            ],
            "cann_source_identities": provenances[0]["cann"][
                "source_identities"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=REQUIRED_RUN_COUNT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"report output already exists: {args.output}")
    report = _build_report(args.run_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
