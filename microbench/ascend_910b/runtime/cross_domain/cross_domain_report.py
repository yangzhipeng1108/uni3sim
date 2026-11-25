#!/usr/bin/env python3
"""Audit the preregistered Ascend 910B cross-domain scheduler campaign."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from cross_domain_runner import (
    BLOCK_DIMS,
    CELLS,
    EXPECTED_CANN,
    HETEROGENEOUS_CELLS,
    RAW_FIELDS,
    SOURCE_FILENAMES,
    SYSTEM_COUNTER_HZ,
    SYSTEM_COUNTER_TICK_NS,
    TASK_COUNTS,
    TRAINING_BLOCK_DIMS,
    TRAINING_TASK_COUNTS,
    TRIALS,
    VALIDATION_BLOCK_DIMS,
    _ordered_cells,
    _sha256,
    _snapshot_sha256,
)


EXPECTED_PROVENANCE_FIELDS = {
    "schema",
    "run_id",
    "captured_at_utc",
    "source_commit",
    "source_identity_mode",
    "source_snapshot_sha256",
    "source_files",
    "binary",
    "cann",
    "device",
    "idle_telemetry",
    "clock",
    "measurement_scope",
    "production_or_evaluation_duration_consumed",
    "campaign",
    "capture",
}
PARAMETER_RUN_COUNT = 2
REPORT_RUN_COUNT = 3
PARAMETER_NAMES = (
    "startup_local_ns",
    "startup_shared_ns_per_active_aic_group",
    "steady_local_ns_per_task",
    "steady_shared_ns_per_task_active_aic_group",
)
PARAMETER_SPAN_LIMIT = 0.15
HOLDOUT_MAPE_LIMIT = 0.15
HOLDOUT_WEIGHTED_MAE_LIMIT = 0.10
HOLDOUT_MAX_APE_LIMIT = 0.35
ORDER_MEDIAN_LIMIT = 0.10
ORDER_P95_LIMIT = 0.25


def _relative_error(observed, expected):
    if observed <= 0 or expected <= 0:
        raise ValueError("relative-error operands must be positive")
    return abs(observed - expected) / expected


def _symmetric_relative_error(left, right):
    if left <= 0 or right <= 0:
        raise ValueError("symmetric-error operands must be positive")
    return abs(left - right) / statistics.median((left, right))


def _relative_span(values):
    if not values or any(value <= 0 for value in values):
        raise ValueError("relative-span values must be positive")
    center = statistics.median(values)
    return (max(values) - min(values)) / center


def _nearest_rank_percentile(values, probability):
    if not values or probability <= 0 or probability > 1:
        raise ValueError("percentile inputs are invalid")
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[index]


def _cycles(text, expected_length, label):
    values = tuple(int(value) for value in text.split(";"))
    if len(values) != expected_length or any(value <= 0 for value in values):
        raise ValueError(
            f"{label} must contain {expected_length} positive cycles")
    return values


def _require_mapping(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} has an invalid schema")


def _load_provenance(run_dir, benchmark_root):
    path = run_dir / "provenance.json"
    with path.open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    _require_mapping(
        provenance, EXPECTED_PROVENANCE_FIELDS, "provenance")
    if provenance["schema"] != "ascend_910b_cross_domain_provenance_v5":
        raise ValueError("unsupported cross-domain provenance schema")
    if provenance["source_identity_mode"] != "audited_source_snapshot":
        raise ValueError("cross-domain capture lacks an audited snapshot")
    if provenance["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("capture consumed forbidden production duration")
    if provenance["source_snapshot_sha256"] != _snapshot_sha256(
            benchmark_root):
        raise ValueError("capture source snapshot differs from report source")
    _require_mapping(
        provenance["source_files"], SOURCE_FILENAMES,
        "provenance.source_files")
    for filename in SOURCE_FILENAMES:
        if (provenance["source_files"][filename]
                != _sha256(benchmark_root / filename)):
            raise ValueError(f"capture source file changed: {filename}")
    binary = provenance["binary"]
    _require_mapping(
        binary, ("path", "sha256", "build_command"),
        "provenance.binary")
    cann = provenance["cann"]
    _require_mapping(
        cann,
        ("identity", "version_file", "version_file_sha256",
         "source_identities"),
        "provenance.cann")
    if cann["identity"] != EXPECTED_CANN:
        raise ValueError("capture does not identify frozen CANN 8.5.1")
    device = provenance["device"]
    _require_mapping(
        device,
        ("physical_device_id", "model", "npu_smi_version",
         "logical_device_id", "selection_environment"),
        "provenance.device")
    if device["model"] != "910B2" or device["logical_device_id"] != 0:
        raise ValueError("capture device is not logical-device-0 910B2")
    idle = provenance["idle_telemetry"]
    _require_mapping(idle, ("pre", "post"), "provenance.idle_telemetry")
    for endpoint in ("pre", "post"):
        _require_mapping(
            idle[endpoint],
            ("aic_percent", "aiv_percent", "hbm_bandwidth_percent"),
            f"provenance.idle_telemetry.{endpoint}")
        if any(value != 0 for value in idle[endpoint].values()):
            raise ValueError("capture endpoint telemetry is not idle")
    if provenance["clock"] != {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS}:
        raise ValueError("capture clock contract changed")
    if provenance["campaign"] != {
            "block_dims": list(BLOCK_DIMS),
            "training_block_dims": list(TRAINING_BLOCK_DIMS),
            "validation_block_dims": list(VALIDATION_BLOCK_DIMS),
            "task_counts": list(TASK_COUNTS),
            "training_task_counts": list(TRAINING_TASK_COUNTS),
            "heterogeneous_profiles": {
                profile_id: list(task_counts)
                for profile_id, task_counts in HETEROGENEOUS_CELLS
            },
            "trials": TRIALS,
            "order": "balanced_forward_reverse"}:
        raise ValueError("capture campaign changed")
    capture = provenance["capture"]
    _require_mapping(
        capture,
        ("raw", "raw_sha256", "raw_rows", "runner_pid", "argv",
         "working_directory", "trials"),
        "provenance.capture")
    raw_path = run_dir / capture["raw"]
    if _sha256(raw_path) != capture["raw_sha256"]:
        raise ValueError("capture raw SHA256 mismatch")
    if capture["raw_rows"] != TRIALS * len(CELLS):
        raise ValueError("capture raw row count changed")
    if len(capture["trials"]) != TRIALS:
        raise ValueError("capture trial artifacts are incomplete")
    for trial_index, artifact in enumerate(capture["trials"], start=1):
        _require_mapping(
            artifact,
            ("trial", "direction", "pid", "command", "stdout",
             "stdout_sha256", "stderr", "stderr_sha256"),
            "provenance.capture.trial")
        direction, _ = _ordered_cells(trial_index - 1)
        if artifact["trial"] != trial_index or artifact["direction"] != direction:
            raise ValueError("capture trial order changed")
        for kind in ("stdout", "stderr"):
            artifact_path = run_dir / artifact[kind]
            if _sha256(artifact_path) != artifact[f"{kind}_sha256"]:
                raise ValueError(f"capture {kind} SHA256 mismatch")
        if (run_dir / artifact["stderr"]).read_text(encoding="utf-8"):
            raise ValueError("capture child stderr is not empty")
    return provenance, raw_path


def _load_rows(raw_path, run_id):
    rows = []
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("cross-domain raw schema changed")
        for raw in reader:
            if raw["run_id"] != run_id:
                raise ValueError("raw run ID disagrees with provenance")
            trial = int(raw["trial"])
            order_position = int(raw["order_position"])
            profile_id = raw["profile_id"]
            block_dim = int(raw["block_dim"])
            task_counts = tuple(
                int(value) for value in raw["task_counts"].split(";"))
            if trial < 1 or trial > TRIALS:
                raise ValueError("raw trial is outside the frozen campaign")
            direction, cells = _ordered_cells(trial - 1)
            if raw["order_direction"] != direction:
                raise ValueError("raw order direction changed")
            if order_position < 0 or order_position >= len(cells):
                raise ValueError("raw order position is invalid")
            if cells[order_position] != (profile_id, task_counts):
                raise ValueError("raw cell order changed")
            if len(task_counts) != block_dim:
                raise ValueError("raw task-count vector changed blockDim")
            aic = _cycles(raw["aic_cycles"], block_dim, "aic_cycles")
            aiv0 = _cycles(raw["aiv0_cycles"], block_dim, "aiv0_cycles")
            aiv1 = _cycles(raw["aiv1_cycles"], block_dim, "aiv1_cycles")
            if int(raw["process_pid"]) <= 0 or int(raw["host_wall_ns"]) <= 0:
                raise ValueError("raw process or wall-health field is invalid")
            if raw["validation"] != "ok":
                raise ValueError("raw device validation failed")
            completion_cycles = max(aic + aiv0 + aiv1)
            actor_pair_error = max(
                _relative_error(left, right)
                for left, right in zip(aiv0, aiv1))
            rows.append({
                "run_id": run_id,
                "trial": trial,
                "direction": direction,
                "profile_id": profile_id,
                "block_dim": block_dim,
                "task_counts": task_counts,
                "completion_ns": completion_cycles * SYSTEM_COUNTER_TICK_NS,
                "actor_pair_error": actor_pair_error,
            })
    if len(rows) != TRIALS * len(CELLS):
        raise ValueError("raw campaign is incomplete")
    identities = [
        (row["trial"], row["profile_id"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("raw campaign contains duplicate cells")
    return rows


def _median_by_cell(rows):
    result = {}
    for run_id in sorted({row["run_id"] for row in rows}):
        for profile_id, _ in CELLS:
            values = [
                row["completion_ns"] for row in rows
                if row["run_id"] == run_id
                and row["profile_id"] == profile_id
            ]
            if len(values) != TRIALS:
                raise ValueError("combined campaign lost a cell")
            result[(run_id, profile_id)] = statistics.median(values)
    return result


def _derive_endpoint_parameters(cell_medians, run_id):
    single_task_count, first_multi_task_count, last_task_count = (
        TRAINING_TASK_COUNTS)
    if (single_task_count, first_multi_task_count) != (1, 2):
        raise ValueError("training endpoints lost the ring termination split")
    effective_startup = {}
    effective_steady = {}
    for block_dim in TRAINING_BLOCK_DIMS:
        startup = cell_medians[
            (run_id, f"uniform_b{block_dim}_n{first_multi_task_count}")]
        endpoint = cell_medians[
            (run_id, f"uniform_b{block_dim}_n{last_task_count}")]
        effective_startup[block_dim] = startup
        effective_steady[block_dim] = (
            (endpoint - startup)
            / (last_task_count - first_multi_task_count))
    parameters = {
        "startup_local_ns": min(effective_startup.values()),
        "startup_shared_ns_per_active_aic_group": min(
            value / block_dim
            for block_dim, value in effective_startup.items()),
        "steady_local_ns_per_task": min(effective_steady.values()),
        "steady_shared_ns_per_task_active_aic_group": min(
            value / block_dim
            for block_dim, value in effective_steady.items()),
    }
    return parameters, {
        "run_id": run_id,
        "effective_startup_ns": {
            str(block_dim): effective_startup[block_dim]
            for block_dim in TRAINING_BLOCK_DIMS
        },
        "effective_steady_ns_per_task": {
            str(block_dim): effective_steady[block_dim]
            for block_dim in TRAINING_BLOCK_DIMS
        },
        "decomposed_parameters": parameters,
    }


def _predict(parameters, single_task_duration_lut, task_counts):
    if (not isinstance(task_counts, tuple) or not task_counts
            or len(task_counts) > 24
            or any(value < 1 or value > 128 for value in task_counts)):
        raise ValueError("prediction geometry is outside the frozen domain")
    block_dim = len(task_counts)
    if block_dim not in single_task_duration_lut:
        raise ValueError("single-task LUT lacks the exact active group count")
    single_task_span = single_task_duration_lut[block_dim]
    multi_task_counts = tuple(
        value for value in task_counts if value >= 2)
    if not multi_task_counts:
        return single_task_span
    active_multi_groups = len(multi_task_counts)
    multi_task_span = max(
        parameters["startup_local_ns"],
        active_multi_groups
        * parameters["startup_shared_ns_per_active_aic_group"],
    )
    for depth in range(3, max(multi_task_counts) + 1):
        active_groups = sum(
            value >= depth for value in multi_task_counts)
        multi_task_span += max(
            parameters["steady_local_ns_per_task"],
            active_groups
            * parameters["steady_shared_ns_per_task_active_aic_group"],
        )
    return max(single_task_span, multi_task_span)


def _metric_summary(
        cell_medians, cells, parameters, single_task_duration_lut):
    errors = []
    absolute_errors = []
    observed_values = []
    worst = None
    for run_id, profile_id, task_counts in cells:
        observed = cell_medians[(run_id, profile_id)]
        predicted = _predict(
            parameters, single_task_duration_lut, task_counts)
        error = _relative_error(predicted, observed)
        absolute_error = abs(predicted - observed)
        errors.append(error)
        absolute_errors.append(absolute_error)
        observed_values.append(observed)
        if worst is None or error > worst["absolute_percentage_error"]:
            worst = {
                "run_id": run_id,
                "profile_id": profile_id,
                "block_dim": len(task_counts),
                "task_counts": list(task_counts),
                "observed_ns": observed,
                "predicted_ns": predicted,
                "absolute_percentage_error": error,
            }
    if not errors or worst is None:
        raise ValueError("metric cell set must be non-empty")
    return {
        "cell_count": len(errors),
        "mean_absolute_percentage_error": statistics.mean(errors),
        "weighted_mean_absolute_error": (
            sum(absolute_errors) / sum(observed_values)),
        "maximum_absolute_percentage_error": max(errors),
        "worst_cell": worst,
    }


def _order_summary(rows, run_ids):
    errors = []
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
            errors.append(_symmetric_relative_error(forward, reverse))
    return {
        "cell_count": len(errors),
        "median_relative_error": statistics.median(errors),
        "p95_relative_error": _nearest_rank_percentile(errors, 0.95),
        "maximum_relative_error": max(errors),
    }


def _build_report(run_dirs):
    benchmark_root = Path(__file__).resolve().parent
    provenances = []
    rows = []
    for run_dir in run_dirs:
        provenance, raw_path = _load_provenance(run_dir, benchmark_root)
        provenances.append(provenance)
        rows.extend(_load_rows(raw_path, provenance["run_id"]))
    run_ids = [provenance["run_id"] for provenance in provenances]
    if len(run_ids) != REPORT_RUN_COUNT or len(set(run_ids)) != REPORT_RUN_COUNT:
        raise ValueError("report requires three distinct ordered run IDs")
    frozen_fields = (
        "source_commit", "source_snapshot_sha256", "source_files",
        "binary", "cann", "device", "clock", "measurement_scope",
        "campaign",
    )
    for field in frozen_fields:
        if any(
                provenance[field] != provenances[0][field]
                for provenance in provenances[1:]):
            raise ValueError(f"independent runs disagree on {field}")

    cell_medians = _median_by_cell(rows)
    parameter_run_ids = run_ids[:PARAMETER_RUN_COUNT]
    validation_run_id = run_ids[PARAMETER_RUN_COUNT]
    run_parameters = []
    run_single_task_luts = []
    endpoint_diagnostics = []
    for run_id in parameter_run_ids:
        parameters, diagnostics = _derive_endpoint_parameters(
            cell_medians, run_id)
        run_parameters.append(parameters)
        run_single_task_luts.append({
            block_dim: cell_medians[
                (run_id, f"uniform_b{block_dim}_n1")]
            for block_dim in BLOCK_DIMS
        })
        endpoint_diagnostics.append(diagnostics)
    endpoint_positive = all(
        value > 0
        for parameters in run_parameters
        for value in parameters.values()
    ) and all(
        value > 0
        for lut in run_single_task_luts
        for value in lut.values())
    if endpoint_positive:
        parameter_spans = {
            name: _relative_span([
                parameters[name] for parameters in run_parameters
            ])
            for name in PARAMETER_NAMES
        }
        frozen_parameters = {
            name: statistics.median([
                parameters[name] for parameters in run_parameters
            ])
            for name in PARAMETER_NAMES
        }
        single_task_lut_spans = {
            block_dim: _relative_span([
                lut[block_dim] for lut in run_single_task_luts])
            for block_dim in BLOCK_DIMS
        }
        frozen_single_task_lut = {
            block_dim: statistics.median([
                lut[block_dim] for lut in run_single_task_luts])
            for block_dim in BLOCK_DIMS
        }
        parameter_profiles = {
            f"uniform_b{block_dim}_n1"
            for block_dim in BLOCK_DIMS
        } | {
            f"uniform_b{block_dim}_n{task_count}"
            for block_dim in TRAINING_BLOCK_DIMS
            for task_count in TRAINING_TASK_COUNTS[1:]
        }
        holdout_cells = [
            (run_id, profile_id, task_counts)
            for run_id in parameter_run_ids
            for profile_id, task_counts in CELLS
            if profile_id not in parameter_profiles
        ]
        validation_cells = [
            (validation_run_id, profile_id, task_counts)
            for profile_id, task_counts in CELLS
        ]
        heterogeneous_validation_cells = [
            (validation_run_id, profile_id, task_counts)
            for profile_id, task_counts in HETEROGENEOUS_CELLS
        ]
        holdout_metrics = _metric_summary(
            cell_medians, holdout_cells, frozen_parameters,
            frozen_single_task_lut)
        validation_metrics = _metric_summary(
            cell_medians, validation_cells, frozen_parameters,
            frozen_single_task_lut)
        heterogeneous_validation_metrics = _metric_summary(
            cell_medians, heterogeneous_validation_cells,
            frozen_parameters, frozen_single_task_lut)
    else:
        parameter_spans = {name: None for name in PARAMETER_NAMES}
        frozen_parameters = {name: None for name in PARAMETER_NAMES}
        single_task_lut_spans = {
            block_dim: None for block_dim in BLOCK_DIMS}
        frozen_single_task_lut = {
            block_dim: None for block_dim in BLOCK_DIMS}
        holdout_metrics = None
        validation_metrics = None
        heterogeneous_validation_metrics = None

    order_metrics = _order_summary(rows, run_ids)
    actor_pair_errors = [row["actor_pair_error"] for row in rows]
    gates = {
        "provenance_and_raw_complete": True,
        "positive_endpoint_decomposition": endpoint_positive,
    }
    for name in PARAMETER_NAMES:
        span = parameter_spans[name]
        gates[f"{name}_cross_run_span"] = (
            span is not None and span <= PARAMETER_SPAN_LIMIT)
    gates["single_task_lut_cross_run_span"] = all(
        span is not None and span <= PARAMETER_SPAN_LIMIT
        for span in single_task_lut_spans.values())
    gates.update({
        "parameter_run_holdout_mape": (
            holdout_metrics is not None
            and holdout_metrics["mean_absolute_percentage_error"]
            <= HOLDOUT_MAPE_LIMIT),
        "parameter_run_holdout_weighted_mae": (
            holdout_metrics is not None
            and holdout_metrics["weighted_mean_absolute_error"]
            <= HOLDOUT_WEIGHTED_MAE_LIMIT),
        "parameter_run_holdout_max_ape": (
            holdout_metrics is not None
            and holdout_metrics["maximum_absolute_percentage_error"]
            <= HOLDOUT_MAX_APE_LIMIT),
        "frozen_validation_mape": (
            validation_metrics is not None
            and validation_metrics["mean_absolute_percentage_error"]
            <= HOLDOUT_MAPE_LIMIT),
        "frozen_validation_weighted_mae": (
            validation_metrics is not None
            and validation_metrics["weighted_mean_absolute_error"]
            <= HOLDOUT_WEIGHTED_MAE_LIMIT),
        "frozen_validation_max_ape": (
            validation_metrics is not None
            and validation_metrics["maximum_absolute_percentage_error"]
            <= HOLDOUT_MAX_APE_LIMIT),
        "frozen_heterogeneous_validation_mape": (
            heterogeneous_validation_metrics is not None
            and heterogeneous_validation_metrics[
                "mean_absolute_percentage_error"]
            <= HOLDOUT_MAPE_LIMIT),
        "frozen_heterogeneous_validation_max_ape": (
            heterogeneous_validation_metrics is not None
            and heterogeneous_validation_metrics[
                "maximum_absolute_percentage_error"]
            <= HOLDOUT_MAX_APE_LIMIT),
        "balanced_order_median": (
            order_metrics["median_relative_error"] <= ORDER_MEDIAN_LIMIT),
        "balanced_order_p95": (
            order_metrics["p95_relative_error"] <= ORDER_P95_LIMIT),
    })
    publishable = all(gates.values())
    return {
        "schema": "ascend_910b_cross_domain_report_v5",
        "mechanism_publishable": publishable,
        "production_or_evaluation_duration_consumed": False,
        "model": {
            "formula": (
                "max(single_task_duration_lut_ns[B], "
                "max(startup_local_ns, count(N_i>=2) * "
                "startup_shared_ns_per_active_aic_group) + sum over "
                "d=3..max(N_i) of max(steady_local_ns_per_task, "
                "count(N_i>=d) * "
                "steady_shared_ns_per_task_active_aic_group))"
            ),
            "physical_interpretation": (
                "the N=1 terminal ring is an independently measured exact "
                "active-group branch; N>=2 groups use the frozen anchored "
                "startup, then leave the shared steady scheduler stream at "
                "their source task-count boundary"
            ),
            "task_unit": (
                "one empty FIA MM1_to_Vec1_to_MM2_to_Vec2 handoff-ring "
                "step per active AIC group"
            ),
            "timer": "GetSystemCycle at 50 MHz",
            "parameters": (
                frozen_parameters if publishable
                else {name: None for name in PARAMETER_NAMES}
            ),
            "single_task_duration_lut_ns": {
                str(block_dim): (
                    frozen_single_task_lut[block_dim]
                    if publishable else None)
                for block_dim in BLOCK_DIMS
            },
            "validated_block_dims": list(BLOCK_DIMS),
            "validated_task_counts": list(TASK_COUNTS),
            "validated_heterogeneous_profiles": {
                profile_id: list(task_counts)
                for profile_id, task_counts in HETEROGENEOUS_CELLS
            },
            "applicability": (
                "CANN 8.5.1 KERNEL_TYPE_MIX_AIC_1_2 source-bound empty "
                "three-handoff ring only; not a complete production kernel "
                "duration"
            ),
        },
        "gates": gates,
        "thresholds": {
            "parameter_cross_run_relative_span": PARAMETER_SPAN_LIMIT,
            "holdout_mean_absolute_percentage_error": HOLDOUT_MAPE_LIMIT,
            "holdout_weighted_mean_absolute_error": (
                HOLDOUT_WEIGHTED_MAE_LIMIT),
            "holdout_maximum_absolute_percentage_error": (
                HOLDOUT_MAX_APE_LIMIT),
            "balanced_order_median_relative_error": ORDER_MEDIAN_LIMIT,
            "balanced_order_p95_relative_error": ORDER_P95_LIMIT,
        },
        "diagnostics": {
            "parameter_run_endpoints": endpoint_diagnostics,
            "candidate_parameters": frozen_parameters,
            "parameter_cross_run_relative_spans": parameter_spans,
            "single_task_lut_cross_run_relative_spans": {
                str(block_dim): single_task_lut_spans[block_dim]
                for block_dim in BLOCK_DIMS
            },
            "parameter_run_geometry_and_task_holdout": holdout_metrics,
            "frozen_run_validation": validation_metrics,
            "frozen_heterogeneous_validation": (
                heterogeneous_validation_metrics),
            "balanced_order": order_metrics,
            "aiv_pair_relative_error": {
                "median": statistics.median(actor_pair_errors),
                "p95": _nearest_rank_percentile(actor_pair_errors, 0.95),
                "maximum": max(actor_pair_errors),
            },
        },
        "provenance": {
            "parameter_run_ids": parameter_run_ids,
            "frozen_validation_run_id": validation_run_id,
            "source_commit": provenances[0]["source_commit"],
            "source_snapshot_sha256": provenances[0][
                "source_snapshot_sha256"],
            "binary_sha256": provenances[0]["binary"]["sha256"],
            "raw_sha256": [
                provenance["capture"]["raw_sha256"]
                for provenance in provenances
            ],
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
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
