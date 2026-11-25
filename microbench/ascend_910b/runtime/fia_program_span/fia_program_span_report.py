#!/usr/bin/env python3
"""Audit three independent FIA nonempty-controller span captures."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from fia_program_span_runner import (
    CELLS,
    EXPECTED_CANN,
    HOLDOUT_CELLS,
    RAW_FIELDS,
    PROGRAM_REPETITIONS,
    SOURCE_FILENAMES,
    SYSTEM_COUNTER_HZ,
    SYSTEM_COUNTER_TICK_NS,
    TASK_COUNTS,
    TRAINING_GROUP_COUNTS,
    TRIALS,
    _ordered_cells,
    _sha256,
    _snapshot_sha256,
)


REPORT_RUN_COUNT = 3
CROSS_RUN_LIMIT = 0.20
POSITION_BALANCE_LIMIT = 0.20
VALIDATION_MAPE_LIMIT = 0.15
VALIDATION_MAX_APE_LIMIT = 0.35


def _relative_error(left, right):
    if left <= 0 or right <= 0:
        raise ValueError("relative-error operands must be positive")
    return abs(left - right) / right


def _relative_span(values):
    center = statistics.median(values)
    return (max(values) - min(values)) / center


def _cycles(text, expected_length, label):
    values = tuple(int(value) for value in text.split(";"))
    if len(values) != expected_length or any(value <= 0 for value in values):
        raise ValueError(f"{label} has an invalid actor vector")
    return values


def _load_run(run_dir, benchmark_root):
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["schema"] != "ascend_910b_fia_program_span_provenance_v1":
        raise ValueError("unsupported FIA program-span provenance schema")
    if provenance["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("capture consumed forbidden production duration")
    if provenance["source_snapshot_sha256"] != _snapshot_sha256(benchmark_root):
        raise ValueError("capture source snapshot differs from report source")
    if set(provenance["source_files"]) != set(SOURCE_FILENAMES):
        raise ValueError("capture source file set changed")
    for filename in SOURCE_FILENAMES:
        if provenance["source_files"][filename] != _sha256(benchmark_root / filename):
            raise ValueError(f"capture source file changed: {filename}")
    if provenance["cann"]["identity"] != EXPECTED_CANN:
        raise ValueError("capture CANN identity changed")
    if provenance["device"]["model"] != "910B2":
        raise ValueError("capture device is not 910B2")
    if provenance["clock"] != {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS}:
        raise ValueError("capture clock changed")
    expected_campaign = {
        "training_group_counts": list(TRAINING_GROUP_COUNTS),
        "task_counts": list(TASK_COUNTS),
        "holdout_profiles": {
            name: list(counts) for name, counts in HOLDOUT_CELLS
        },
        "trials": TRIALS,
        "program_repetitions": PROGRAM_REPETITIONS,
        "order": "balanced_forward_reverse",
    }
    if provenance["campaign"] != expected_campaign:
        raise ValueError("capture campaign changed")
    raw_path = run_dir / provenance["capture"]["raw"]
    if _sha256(raw_path) != provenance["capture"]["raw_sha256"]:
        raise ValueError("capture raw SHA256 mismatch")
    rows = []
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("capture raw schema changed")
        for raw in reader:
            trial = int(raw["trial"])
            direction, cells = _ordered_cells(trial - 1)
            position = int(raw["order_position"])
            profile_id, task_counts = cells[position]
            if (raw["run_id"] != provenance["run_id"]
                    or raw["order_direction"] != direction
                    or raw["profile_id"] != profile_id
                    or tuple(int(value) for value in raw["task_counts"].split(";"))
                    != task_counts or raw["validation"] != "ok"):
                raise ValueError("capture row changed the frozen campaign")
            block_dim = len(task_counts)
            completion_cycles = max(
                _cycles(raw["aic_cycles"], block_dim, "aic_cycles")
                + _cycles(raw["aiv0_cycles"], block_dim, "aiv0_cycles")
                + _cycles(raw["aiv1_cycles"], block_dim, "aiv1_cycles"))
            rows.append({
                "run_id": provenance["run_id"],
                "trial": trial,
                "direction": direction,
                "profile_id": profile_id,
                "task_counts": task_counts,
                "completion_ns": completion_cycles * SYSTEM_COUNTER_TICK_NS,
            })
    if len(rows) != TRIALS * len(CELLS):
        raise ValueError("capture is incomplete")
    return provenance, rows


def _predict(parameters, single_task_lut, task_counts):
    active_groups = len(task_counts)
    if active_groups not in single_task_lut:
        raise ValueError("task profile lacks an exact active-group anchor")
    multi_counts = tuple(value for value in task_counts if value >= 2)
    if not multi_counts:
        return single_task_lut[active_groups]
    duration = max(
        parameters["startup_local_ns"],
        len(multi_counts) * parameters["startup_shared_ns_per_active_group"])
    for depth in range(3, max(multi_counts) + 1):
        surviving_groups = sum(value >= depth for value in multi_counts)
        duration += max(
            parameters["steady_local_ns"],
            surviving_groups
            * parameters["steady_shared_ns_per_active_group"])
    return max(single_task_lut[active_groups], duration)


def _build_report(run_dirs):
    root = Path(__file__).resolve().parent
    loaded = [_load_run(path, root) for path in run_dirs]
    provenances = [value[0] for value in loaded]
    rows = [row for value in loaded for row in value[1]]
    run_ids = [value["run_id"] for value in provenances]
    if len(set(run_ids)) != REPORT_RUN_COUNT:
        raise ValueError("report requires three independent run IDs")
    frozen_fields = (
        "source_commit", "source_snapshot_sha256", "source_files",
        "binary", "cann", "device", "clock", "campaign",
    )
    for field in frozen_fields:
        if any(value[field] != provenances[0][field] for value in provenances[1:]):
            raise ValueError(f"independent runs disagree on {field}")
    medians = {}
    for run_id in run_ids:
        for profile_id, _ in CELLS:
            values = [
                row["completion_ns"] for row in rows
                if row["run_id"] == run_id and row["profile_id"] == profile_id
            ]
            if len(values) != TRIALS:
                raise ValueError("combined runs lost a campaign cell")
            medians[(run_id, profile_id)] = statistics.median(values)
    frozen_cell_medians = {
        profile_id: statistics.median([
            medians[(run_id, profile_id)] for run_id in run_ids
        ])
        for profile_id, _ in CELLS
    }
    cross_run_spans = {
        profile_id: _relative_span([
            medians[(run_id, profile_id)] for run_id in run_ids
        ])
        for profile_id, _ in CELLS
    }
    position_errors = {}
    for run_id in run_ids:
        for profile_id, _ in CELLS:
            forward = statistics.median([
                row["completion_ns"] for row in rows
                if row["run_id"] == run_id
                and row["profile_id"] == profile_id
                and row["direction"] == "forward"
            ])
            reverse = statistics.median([
                row["completion_ns"] for row in rows
                if row["run_id"] == run_id
                and row["profile_id"] == profile_id
                and row["direction"] == "reverse"
            ])
            position_errors[f"{run_id}:{profile_id}"] = (
                abs(forward - reverse) / statistics.median((forward, reverse)))
    b1_n2 = frozen_cell_medians["uniform_b1_n2"]
    b1_n12 = frozen_cell_medians["uniform_b1_n12"]
    b24_n2 = frozen_cell_medians["uniform_b24_n2"]
    b24_n12 = frozen_cell_medians["uniform_b24_n12"]
    parameters = {
        "startup_local_ns": b1_n2,
        "startup_shared_ns_per_active_group": b24_n2 / 24,
        "steady_local_ns": (b1_n12 - b1_n2) / 10,
        "steady_shared_ns_per_active_group": (
            (b24_n12 - b24_n2) / (10 * 24)),
    }
    single_task_lut = {
        group_count: frozen_cell_medians[f"uniform_b{group_count}_n1"]
        for group_count in TRAINING_GROUP_COUNTS
    }
    single_task_lut[22] = frozen_cell_medians["holdout_b22_n1"]
    training_profiles = {
        "uniform_b1_n1", "uniform_b1_n2", "uniform_b1_n12",
        "uniform_b24_n1", "uniform_b24_n2", "uniform_b24_n12",
        "uniform_b8_n1", "uniform_b16_n1", "holdout_b22_n1",
    }
    validation = []
    for profile_id, task_counts in CELLS:
        if profile_id in training_profiles:
            continue
        observed = frozen_cell_medians[profile_id]
        predicted = _predict(parameters, single_task_lut, task_counts)
        validation.append({
            "profile_id": profile_id,
            "task_counts": list(task_counts),
            "observed_ns": observed,
            "predicted_ns": predicted,
            "absolute_percentage_error": _relative_error(predicted, observed),
        })
    validation_mape = statistics.mean(
        value["absolute_percentage_error"] for value in validation)
    validation_max_ape = max(
        value["absolute_percentage_error"] for value in validation)
    positive_parameters = all(value > 0 for value in parameters.values())
    gates = {
        "source_binary_and_raw_identity_complete": True,
        "positive_mechanism_parameters": positive_parameters,
        "all_cells_cross_run_at_most_20pct": (
            max(cross_run_spans.values()) <= CROSS_RUN_LIMIT),
        "all_cells_position_balance_at_most_20pct": (
            max(position_errors.values()) <= POSITION_BALANCE_LIMIT),
        "held_out_topology_and_intermediate_counts_mape": (
            validation_mape <= VALIDATION_MAPE_LIMIT),
        "held_out_topology_and_intermediate_counts_max_ape": (
            validation_max_ape <= VALIDATION_MAX_APE_LIMIT),
    }
    publishable = all(gates.values())
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": "cann_fia_nonempty_controller_span_v1",
        "mechanism_publishable": publishable,
        "production_or_evaluation_duration_consumed": False,
        "scope": (
            "source-bound nonempty three-slot FIA controller and three-handoff "
            "program span; excludes production attention data bodies"),
        "duration_unit": "ns_per_complete_aic_group_program_span",
        "model": {
            "formula": (
                "max(single_task_duration_lut_ns[B], max(startup_local_ns, "
                "count(N_i>=2)*startup_shared_ns_per_active_group) + "
                "sum(d=3..max(N_i), max(steady_local_ns, "
                "count(N_i>=d)*steady_shared_ns_per_active_group)))"),
            "parameters_ns": parameters if publishable else None,
            "single_task_duration_lut_ns": (
                {str(key): value for key, value in single_task_lut.items()}
                if publishable else None),
            "exact_active_group_domain": [1, 8, 16, 22, 24],
            "exact_task_count_domain": [1, 12],
            "measured_task_counts": list(TASK_COUNTS),
            "holdout_profiles": {
                name: list(counts) for name, counts in HOLDOUT_CELLS
            },
        },
        "gates": gates,
        "thresholds": {
            "cross_run_relative_span": CROSS_RUN_LIMIT,
            "position_balance_relative_error": POSITION_BALANCE_LIMIT,
            "validation_mean_absolute_percentage_error": VALIDATION_MAPE_LIMIT,
            "validation_maximum_absolute_percentage_error": VALIDATION_MAX_APE_LIMIT,
        },
        "diagnostics": {
            "cell_medians_ns": frozen_cell_medians,
            "cell_cross_run_relative_spans": cross_run_spans,
            "position_balance_relative_errors": position_errors,
            "validation_mean_absolute_percentage_error": validation_mape,
            "validation_maximum_absolute_percentage_error": validation_max_ape,
            "validation_cells": validation,
        },
        "provenance": {
            "run_ids": run_ids,
            "source_commit": provenances[0]["source_commit"],
            "source_snapshot_sha256": provenances[0]["source_snapshot_sha256"],
            "binary_sha256": provenances[0]["binary"]["sha256"],
            "raw_sha256": [
                value["capture"]["raw_sha256"] for value in provenances
            ],
            "cann_source_identities": provenances[0]["cann"]["source_identities"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=REPORT_RUN_COUNT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"report output already exists: {args.output}")
    report = _build_report(args.run_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
