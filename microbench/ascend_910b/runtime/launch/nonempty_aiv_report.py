#!/usr/bin/env python3
"""Audit and freeze the predeclared 910B minimum non-empty AIV task model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


MODEL_TAG = "isolated_nonempty_aiv_task_scheduler_v1"
MIX_AIV_MODEL_TAG = "isolated_nonempty_mix_aiv_task_scheduler_v1"
AIV_MEMORY_TASK_MODEL_TAG = "isolated_aiv_memory_task_floor_v1"
AIV_NORM_CONTROL_TASK_MODEL_TAG = "isolated_aiv_norm_control_task_floor_v1"
EXPECTED_RUNS = 4
EXPECTED_BLOCK_DIMS = tuple(range(1, 49))
MINIMUM_LAUNCHES_PER_POINT = 200
TRAINING_RUNS = (1, 2)
VALIDATION_RUNS = (3, 4)
MAXIMUM_MAPE_PERCENT = 10.0
MAXIMUM_VALIDATION_APE_PERCENT = 25.0
MAXIMUM_TRAINING_PARAMETER_RELATIVE_SPAN = 0.10


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value, context):
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    return value


def _require_positive_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _validate_manifest_checksum(capture_dir, manifest_path):
    checksum_path = capture_dir / "manifest.sha256"
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    if fields != [_sha256(manifest_path), manifest_path.name]:
        raise ValueError("manifest.sha256 does not authenticate manifest.json")


def _validate_inventory(capture_dir, inventory):
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("manifest checksum_inventory must be a non-empty list")
    paths = set()
    for index, raw_entry in enumerate(inventory):
        entry = _require_mapping(raw_entry, f"checksum_inventory[{index}]")
        if set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("checksum inventory entry schema changed")
        path = capture_dir / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"inventoried file is missing: {path}")
        if entry["path"] in paths:
            raise ValueError(f"duplicate checksum inventory path {entry['path']!r}")
        paths.add(entry["path"])
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"inventoried size changed for {path}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"inventoried SHA-256 changed for {path}")


def _load_capture(capture_dir, expected_campaign):
    capture_dir = capture_dir.resolve()
    manifest_path = capture_dir / "manifest.json"
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    _validate_manifest_checksum(capture_dir, manifest_path)
    required = {
        "schema_version", "hardware_id", "capture_state", "capture_scope",
        "evidence_status", "candidate_eligibility", "hw_spec_eligibility",
        "duration_policy", "production_or_evaluation_duration_consumed",
        "campaign", "runs", "launches_per_point", "point_count_per_run",
        "schedule", "raw_task_time", "checksum_inventory", "baseline_commit",
        "source_archive", "deployed_source_archive", "tools", "device",
        "preflight", "postflight",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != 13:
        raise ValueError("non-empty AIV capture requires manifest schema 13")
    if manifest["hardware_id"] != "ascend_910b":
        raise ValueError("capture hardware must be ascend_910b")
    if manifest["capture_state"] != "complete":
        raise ValueError("capture is not complete")
    if manifest["capture_scope"] != "candidate_capture":
        raise ValueError("ABI smoke cannot enter the non-empty AIV report")
    if manifest["campaign"] != expected_campaign:
        raise ValueError(
            f"capture campaign must be {expected_campaign}")
    if manifest["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("capture consumed production/evaluation duration")
    if manifest["runs"] != EXPECTED_RUNS:
        raise ValueError(f"capture must contain exactly {EXPECTED_RUNS} runs")
    launches = _require_positive_integer(
        manifest["launches_per_point"], "launches_per_point")
    if launches < MINIMUM_LAUNCHES_PER_POINT:
        raise ValueError(
            f"capture requires at least {MINIMUM_LAUNCHES_PER_POINT} launches")
    if manifest["point_count_per_run"] != len(EXPECTED_BLOCK_DIMS):
        raise ValueError("capture point count must cover AIV block dims 1..48")
    if manifest["schedule"] != "paired_forward_reverse_rotations":
        raise ValueError("capture does not use the frozen balanced schedule")
    _validate_inventory(capture_dir, manifest["checksum_inventory"])
    raw_path = Path(manifest["raw_task_time"])
    if raw_path.is_absolute() and not raw_path.is_relative_to(capture_dir):
        relocated_raw_path = capture_dir / raw_path.name
        inventoried_paths = {
            entry["path"] for entry in manifest["checksum_inventory"]}
        if (raw_path.name not in inventoried_paths
                or not relocated_raw_path.is_file()):
            raise ValueError(
                "relocated raw_task_time is not authenticated by the "
                "capture inventory")
        raw_path = relocated_raw_path
    elif not raw_path.is_absolute():
        raw_path = capture_dir / raw_path
    raw_path = raw_path.resolve()
    try:
        raw_path.relative_to(capture_dir)
    except ValueError as error:
        raise ValueError("raw_task_time escapes the capture directory") from error
    rows = []
    with raw_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"raw task file has blank line {line_number}")
            rows.append(_require_mapping(
                json.loads(line), f"raw task row {line_number}"))
    return manifest, manifest_path, raw_path, rows


def _cell_medians(rows, launches_per_point, contract):
    if not isinstance(contract, dict):
        raise TypeError("campaign contract must be a mapping")
    required_contract = {
        "task_class", "task_time_role", "task_time_task_type",
        "ge_summary_task_type", "task_time_components",
    }
    if set(contract) != required_contract:
        raise ValueError("campaign task contract schema changed")
    grouped = {}
    for index, row in enumerate(rows):
        context = f"raw task row {index + 1}"
        required = {
            "run", "trial", "task_class", "requested_block_dim",
            "requested_mix_block_dim", "duration_time_ns",
            "task_time_role", "observed_block_dim",
            "observed_mix_block_dim", "task_time_task_type",
            "ge_summary_task_type", "task_time_components",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"{context} is missing fields: {sorted(missing)}")
        run = _require_positive_integer(row["run"], f"{context}.run")
        trial = _require_positive_integer(row["trial"], f"{context}.trial")
        block_dim = _require_positive_integer(
            row["requested_block_dim"], f"{context}.requested_block_dim")
        duration = _require_positive_integer(
            row["duration_time_ns"], f"{context}.duration_time_ns")
        if run > EXPECTED_RUNS:
            raise ValueError(f"{context}.run exceeds the frozen run count")
        if block_dim not in EXPECTED_BLOCK_DIMS:
            raise ValueError(f"{context} has an unsupported block dimension")
        if trial > launches_per_point:
            raise ValueError(f"{context}.trial exceeds launches_per_point")
        expected_values = {
            "task_class": contract["task_class"],
            "requested_mix_block_dim": 0,
            "task_time_role": contract["task_time_role"],
            "observed_block_dim": block_dim,
            "task_time_task_type": contract["task_time_task_type"],
            "ge_summary_task_type": contract["ge_summary_task_type"],
        }
        for name, expected in expected_values.items():
            if row[name] != expected:
                raise ValueError(
                    f"{context}.{name} changed: {row[name]!r} != {expected!r}")
        if row["observed_mix_block_dim"] not in (None, 0):
            raise ValueError(f"{context}.observed_mix_block_dim must be null/zero")
        components = row["task_time_components"]
        if not isinstance(components, list):
            raise TypeError(f"{context}.task_time_components must be a list")
        expected_components = contract["task_time_components"]
        if len(components) != len(expected_components):
            raise ValueError(
                f"{context}.task_time_components length changed: "
                f"{len(components)} != {len(expected_components)}")
        for component_index, (component, expected_component) in enumerate(
                zip(components, expected_components)):
            component_context = (
                f"{context}.task_time_components[{component_index}]")
            if not isinstance(component, dict):
                raise TypeError(f"{component_context} must be a mapping")
            for name, expected in expected_component.items():
                if name not in component:
                    raise ValueError(
                        f"{component_context} is missing {name!r}")
                if component[name] != expected:
                    raise ValueError(
                        f"{component_context}.{name} changed: "
                        f"{component[name]!r} != {expected!r}")
        cell = (run, block_dim)
        if cell not in grouped:
            grouped[cell] = {}
        if trial in grouped[cell]:
            raise ValueError(
                f"duplicate trial {trial} for run {run}, block_dim {block_dim}")
        grouped[cell][trial] = duration
    expected_cells = {
        (run, block_dim)
        for run in range(1, EXPECTED_RUNS + 1)
        for block_dim in EXPECTED_BLOCK_DIMS
    }
    if set(grouped) != expected_cells:
        raise ValueError("raw task rows do not cover the frozen run/grid matrix")
    expected_trials = set(range(1, launches_per_point + 1))
    medians = {}
    for cell, trials in grouped.items():
        if set(trials) != expected_trials:
            raise ValueError(f"cell {cell!r} does not contain every launch trial")
        medians[cell] = float(statistics.median(trials.values()))
    return medians


def _fit_affine(points):
    if len(points) < 2:
        raise ValueError("affine fit requires at least two points")
    count = float(len(points))
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_xx = sum(x * x for x, _ in points)
    sum_xy = sum(x * y for x, y in points)
    denominator = count * sum_xx - sum_x * sum_x
    if denominator <= 0:
        raise ValueError("affine fit has a degenerate block-dimension design")
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return intercept, slope


def _metrics(points, intercept, slope):
    apes = []
    absolute = 0.0
    actual_total = 0.0
    for block_dim, actual in points:
        predicted = intercept + slope * block_dim
        if predicted <= 0:
            raise ValueError("frozen affine model predicts a non-positive floor")
        error = abs(predicted - actual)
        apes.append(100.0 * error / actual)
        absolute += error
        actual_total += actual
    return {
        "mape_percent": sum(apes) / len(apes),
        "wmae_percent": 100.0 * absolute / actual_total,
        "max_ape_percent": max(apes),
    }


def _relative_span(values):
    center = statistics.median(values)
    if center <= 0:
        return math.inf
    return (max(values) - min(values)) / center


def build_report(capture_dir, campaign="nonempty_aiv"):
    campaign_contracts = {
        "nonempty_aiv": {
            "model": MODEL_TAG,
            "application_scope": (
                "hardware_wide_minimum_nonempty_aiv_task_floor"),
            "duration_unit": "ns_per_complete_nonempty_aiv_task",
            "task_contract": {
                "task_class": "aiv_nonempty",
                "task_time_role": "complete_task",
                "task_time_task_type": "AI_CORE",
                "ge_summary_task_type": "AI_VECTOR_CORE",
                "task_time_components": (
                    {"task_time_role": "complete_task",
                     "task_time_task_type": "AI_CORE"},
                ),
            },
        },
        "nonempty_mix_aiv": {
            "model": MIX_AIV_MODEL_TAG,
            "application_scope": (
                "hardware_wide_minimum_nonempty_cann_mix_aiv_1_0_task_floor"),
            "duration_unit": "ns_per_complete_nonempty_mix_aiv_task",
            "task_contract": {
                "task_class": "mix_aiv_nonempty",
                "task_time_role": "complete_mixed_envelope",
                "task_time_task_type": "AI_CORE",
                "ge_summary_task_type": "MIX_AIV",
                "task_time_components": (
                    {"task_time_role": "complete_mixed_envelope",
                     "task_time_task_type": "AI_CORE"},
                    {"task_time_role": "contained_mixed_component",
                     "task_time_task_type": "MIX_AIV"},
                ),
            },
        },
        "aiv_memory_task": {
            "model": AIV_MEMORY_TASK_MODEL_TAG,
            "application_scope": (
                "hardware_wide_minimum_mte2_vector_mte3_aiv_task_floor"),
            "duration_unit": "ns_per_complete_nonempty_aiv_task",
            "task_contract": {
                "task_class": "aiv_memory_task",
                "task_time_role": "complete_task",
                "task_time_task_type": "AI_CORE",
                "ge_summary_task_type": "AI_VECTOR_CORE",
                "task_time_components": (
                    {"task_time_role": "complete_task",
                     "task_time_task_type": "AI_CORE"},
                ),
            },
        },
        "aiv_norm_control_task": {
            "model": AIV_NORM_CONTROL_TASK_MODEL_TAG,
            "application_scope": (
                "source_bound_normalization_control_task_floor"),
            "duration_unit": "ns_per_complete_nonempty_aiv_task",
            "task_contract": {
                "task_class": "aiv_norm_control_task",
                "task_time_role": "complete_task",
                "task_time_task_type": "AI_CORE",
                "ge_summary_task_type": "AI_VECTOR_CORE",
                "task_time_components": (
                    {"task_time_role": "complete_task",
                     "task_time_task_type": "AI_CORE"},
                ),
            },
        },
    }
    if campaign not in campaign_contracts:
        raise ValueError(f"unsupported AIV task-floor campaign {campaign!r}")
    contract = campaign_contracts[campaign]
    manifest, manifest_path, raw_path, rows = _load_capture(
        capture_dir, campaign)
    launches = manifest["launches_per_point"]
    medians = _cell_medians(rows, launches, contract["task_contract"])
    training_points = [
        (block_dim, medians[(run, block_dim)])
        for run in TRAINING_RUNS for block_dim in EXPECTED_BLOCK_DIMS
    ]
    validation_points = [
        (block_dim, medians[(run, block_dim)])
        for run in VALIDATION_RUNS for block_dim in EXPECTED_BLOCK_DIMS
    ]
    intercept, slope = _fit_affine(training_points)
    per_training_run = {
        run: _fit_affine([
            (block_dim, medians[(run, block_dim)])
            for block_dim in EXPECTED_BLOCK_DIMS])
        for run in TRAINING_RUNS
    }
    intercept_span = _relative_span(
        [parameters[0] for parameters in per_training_run.values()])
    slope_span = _relative_span(
        [parameters[1] for parameters in per_training_run.values()])
    training_metrics = _metrics(training_points, intercept, slope)
    validation_metrics = _metrics(validation_points, intercept, slope)
    gates = {
        "positive_submit_ns": intercept > 0,
        "positive_descriptor_ns_per_aiv": slope > 0,
        "training_mape": (
            training_metrics["mape_percent"] <= MAXIMUM_MAPE_PERCENT),
        "validation_mape": (
            validation_metrics["mape_percent"] <= MAXIMUM_MAPE_PERCENT),
        "validation_max_ape": (
            validation_metrics["max_ape_percent"]
            <= MAXIMUM_VALIDATION_APE_PERCENT),
        "training_submit_relative_span": (
            intercept_span <= MAXIMUM_TRAINING_PARAMETER_RELATIVE_SPAN),
        "training_descriptor_relative_span": (
            slope_span <= MAXIMUM_TRAINING_PARAMETER_RELATIVE_SPAN),
    }
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "model": contract["model"],
        "formula": "submit_ns + descriptor_ns_per_aiv * aiv_block_dim",
        "duration_unit": contract["duration_unit"],
        "timer_domain": "msprof_task_time_complete_task",
        "application_scope": contract["application_scope"],
        "composition": "max_floor_not_additive",
        "training_runs": list(TRAINING_RUNS),
        "validation_runs": list(VALIDATION_RUNS),
        "block_dims": list(EXPECTED_BLOCK_DIMS),
        "launches_per_point": launches,
        "parameters": {
            "submit_ns": intercept,
            "descriptor_ns_per_aiv": slope,
        },
        "per_training_run_parameters": {
            str(run): {
                "submit_ns": parameters[0],
                "descriptor_ns_per_aiv": parameters[1],
            }
            for run, parameters in per_training_run.items()
        },
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "parameter_stability": {
            "submit_relative_span": intercept_span,
            "descriptor_relative_span": slope_span,
        },
        "gates": gates,
        "mechanism_publishable": all(gates.values()),
        "gate_thresholds": {
            "maximum_mape_percent": MAXIMUM_MAPE_PERCENT,
            "maximum_validation_ape_percent": (
                MAXIMUM_VALIDATION_APE_PERCENT),
            "maximum_training_parameter_relative_span": (
                MAXIMUM_TRAINING_PARAMETER_RELATIVE_SPAN),
        },
        "evidence": {
            "capture_directory": str(Path(capture_dir).resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "raw_task_time_sha256": _sha256(raw_path),
            "baseline_commit": manifest["baseline_commit"],
            "source_archive": manifest["source_archive"],
            "deployed_source_archive": manifest["deployed_source_archive"],
            "tools": manifest["tools"],
            "device": manifest["device"],
            "preflight": manifest["preflight"],
            "postflight": manifest["postflight"],
        },
        "microbenchmark_task_duration_consumed": True,
        "production_or_evaluation_duration_consumed": False,
    }


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--campaign", choices=(
            "nonempty_aiv", "nonempty_mix_aiv", "aiv_memory_task",
            "aiv_norm_control_task"),
        default="nonempty_aiv")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    report = build_report(args.capture_dir, args.campaign)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
