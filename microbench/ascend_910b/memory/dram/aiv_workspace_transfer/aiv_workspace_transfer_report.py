#!/usr/bin/env python3
"""Validate and summarize AIV workspace-transfer capture artifacts."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from aiv_workspace_transfer_runner import (
    AIC_GROUPS,
    CELLS,
    CHILD_FIELDS,
    COMMIT_PATTERN,
    EXPECTED_FIA_VECTOR_SOURCE_SHA256,
    ITERATIONS,
    LOGICAL_DEVICE_ID,
    MODES,
    RAW_FIELDS,
    RUN_ID_PATTERN,
    SHA256_PATTERN,
    SOURCE_FILENAMES,
    SYSTEM_COUNTER_HZ,
    SYSTEM_COUNTER_TICK_NS,
    TRIALS,
    _active_counts,
    _ordered_cells,
    _parse_child_row,
    _parse_device_identity,
    _sha256,
    _snapshot_sha256,
)


TRAINING_TRIALS = tuple(range(1, TRIALS // 2 + 1))
VALIDATION_TRIALS = tuple(range(TRIALS // 2 + 1, TRIALS + 1))
MAXIMUM_RELATIVE_TRIAL_SPREAD = 0.15
MAXIMUM_VALIDATION_MAPE = 0.15
MAXIMUM_VALIDATION_APE = 0.40
MANIFEST_FIELDS = {
    "schema", "run_id", "captured_at_utc", "source_commit",
    "source_identity_mode", "source_snapshot_sha256", "source_files",
    "build", "cann", "source_semantics", "device", "idle_telemetry",
    "clock", "measurement_scope",
    "production_or_evaluation_duration_consumed", "campaign", "capture",
}


def _load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _require_exact_mapping(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} has an invalid schema")


def _validate_sha256(value, label):
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")


def _relative_span(values):
    center = statistics.median(values)
    if center <= 0:
        return math.inf
    return (max(values) - min(values)) / center


def _domains(mode):
    if mode.startswith("vec1_"):
        return ("vec1",)
    if mode.startswith("vec2_"):
        return ("vec2",)
    raise ValueError(f"unsupported mode {mode}")


def _completion(row, domain):
    if domain == "vec1":
        return row["vec1_completion_ns"]
    if domain == "vec2":
        return row["vec2_completion_ns"]
    if domain == "critical":
        return row["critical_completion_ns"]
    raise ValueError(f"unsupported completion domain {domain}")


def _expected_build_commands(benchmark_root, build_dir):
    """Return the exact source-bound commands accepted by the report."""

    return {
        "configure": [
            "cmake", "-S", str(benchmark_root), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1",
            "-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.1",
            "-DSOC_VERSION=Ascend910B2"],
        "compile": [
            "cmake", "--build", str(build_dir), "--target",
            "aiv_workspace_transfer_bench", "--parallel", "4"],
    }


def _validate_manifest(capture_dir):
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"capture manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    _require_exact_mapping(manifest, MANIFEST_FIELDS, "manifest")
    if manifest["schema"] != "ascend_910b_aiv_workspace_transfer_manifest_v4":
        raise ValueError("unsupported AIV workspace-transfer manifest schema")
    if RUN_ID_PATTERN.fullmatch(manifest["run_id"]) is None:
        raise ValueError("capture run ID is invalid")
    if COMMIT_PATTERN.fullmatch(manifest["source_commit"]) is None:
        raise ValueError("capture source commit is invalid")
    _validate_sha256(
        manifest["source_snapshot_sha256"], "source snapshot identity")
    benchmark_root = Path(__file__).resolve().parent
    if _snapshot_sha256(benchmark_root) != manifest["source_snapshot_sha256"]:
        raise ValueError("report source snapshot differs from capture")
    captured_at = datetime.fromisoformat(manifest["captured_at_utc"])
    if captured_at.tzinfo is None or captured_at.utcoffset().total_seconds() != 0:
        raise ValueError("capture timestamp must be timezone-aware UTC")
    if manifest["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("capture consumed forbidden production duration")
    if set(manifest["source_files"]) != set(SOURCE_FILENAMES):
        raise ValueError("capture source-file set changed")
    for name, value in manifest["source_files"].items():
        _validate_sha256(value, f"source identity for {name}")
        if _sha256(benchmark_root / name) != value:
            raise ValueError(f"report source identity differs for {name}")
    build = manifest["build"]
    _require_exact_mapping(
        build,
        {"source_snapshot_sha256", "build_directory", "configure",
         "compile", "binary_path", "binary_sha256"},
        "source-bound build")
    if build["source_snapshot_sha256"] != manifest["source_snapshot_sha256"]:
        raise ValueError("build source snapshot differs from manifest")
    expected_build_dir = capture_dir / "build"
    if Path(build["build_directory"]).resolve() != expected_build_dir.resolve():
        raise ValueError("build directory is outside the capture")
    binary_path = Path(build["binary_path"])
    if (binary_path.resolve()
            != (expected_build_dir / "aiv_workspace_transfer_bench").resolve()
            or not binary_path.is_file()
            or _sha256(binary_path) != build["binary_sha256"]):
        raise ValueError("source-bound benchmark binary identity changed")
    _validate_sha256(build["binary_sha256"], "benchmark binary identity")
    expected_build_commands = _expected_build_commands(
        Path(__file__).resolve().parent, expected_build_dir)
    for phase in ("configure", "compile"):
        artifact = build[phase]
        _require_exact_mapping(
            artifact,
            {"command", "return_code", "stdout", "stdout_sha256",
             "stderr", "stderr_sha256"},
            f"build {phase}")
        if (artifact["command"] != expected_build_commands[phase]
                or artifact["return_code"] != 0):
            raise ValueError(f"build {phase} command changed")
        for stream in ("stdout", "stderr"):
            path = capture_dir / artifact[stream]
            if (not path.is_file()
                    or _sha256(path) != artifact[f"{stream}_sha256"]):
                raise ValueError(f"build {phase} {stream} identity changed")
    _require_exact_mapping(
        manifest["cann"],
        {"root", "identity", "version_file", "version_file_sha256",
         "fia_vector_source"},
        "CANN identity")
    if manifest["cann"]["identity"] != {
            "package": "Ascend-cann-toolkit",
            "version": "8.5.1",
            "innerversion": "V100R001C25SPC002B220"}:
        raise ValueError("capture CANN identity changed")
    _validate_sha256(
        manifest["cann"]["version_file_sha256"], "CANN version identity")
    vector_source = manifest["cann"]["fia_vector_source"]
    _require_exact_mapping(
        vector_source, {"path", "sha256", "symbols", "required_markers"},
        "CANN FIA vector source")
    if vector_source["sha256"] != EXPECTED_FIA_VECTOR_SOURCE_SHA256:
        raise ValueError("capture used a different CANN FIA vector source")
    _require_exact_mapping(
        manifest["source_semantics"],
        {"vec1", "vec2", "topology", "actor_mapping"},
        "source semantics")
    if manifest["source_semantics"] != {
            "vec1": {
                "symbol": "FiaBlockVecNonQuant<FIAT>::DealBmm1ResBaseBlock",
                "actual_m": 4,
                "s512_gm_to_ub_bytes_per_task": 8192,
                "s512_ub_to_gm_bytes_per_task": 4096,
                "s1024_gm_to_ub_bytes_per_task": 16384,
                "s1024_ub_to_gm_bytes_per_task": 8192,
            },
            "vec2": {
                "symbol": "FiaBlockVecNonQuant<FIAT>::DealBmm2ResBaseBlock",
                "actual_m": 4,
                "final_state_read_bytes_per_task": 1024,
                "final_state_write_bytes_per_task": 512,
                "dependency": "MTE3_MTE2 event/wait before state reuse",
            },
            "topology": "KERNEL_TYPE_MIX_AIC_1_2",
            "actor_mapping": (
                "D64 mDealSize=4 runs both Vec1 and Vec2 on primary AIV; "
                "secondary AIV has vecDealM=0")}:
        raise ValueError("capture source semantics changed")
    device = manifest["device"]
    _require_exact_mapping(
        device,
        {"physical_device_id", "logical_device_id", "model", "board_id",
         "pci_identity", "selection_environment", "identity_telemetry"},
        "device identity")
    if (not isinstance(device["physical_device_id"], int)
            or isinstance(device["physical_device_id"], bool)
            or device["physical_device_id"] < 0
            or device["logical_device_id"] != LOGICAL_DEVICE_ID):
        raise ValueError("capture device IDs are invalid")
    if device["selection_environment"] != {
            "ASCEND_DEVICE_ID": str(LOGICAL_DEVICE_ID),
            "ASCEND_RT_VISIBLE_DEVICES": str(device["physical_device_id"])}:
        raise ValueError("capture device masking contract changed")
    identity_text = {}
    expected_identity_commands = {
        "global": ["/usr/local/bin/npu-smi", "info"],
        "board": [
            "/usr/local/bin/npu-smi", "info", "-t", "board", "-i",
            str(device["physical_device_id"])],
    }
    _require_exact_mapping(
        device["identity_telemetry"], {"global", "board"},
        "device identity telemetry")
    for label in ("global", "board"):
        artifact = device["identity_telemetry"][label]
        _require_exact_mapping(
            artifact, {"command", "path", "sha256"},
            f"device {label} telemetry")
        if artifact["command"] != expected_identity_commands[label]:
            raise ValueError(f"device {label} telemetry command changed")
        path = capture_dir / artifact["path"]
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"device {label} telemetry identity changed")
        identity_text[label] = path.read_text(encoding="utf-8")
    parsed_device = _parse_device_identity(
        identity_text["global"], identity_text["board"],
        device["physical_device_id"])
    for field in (
            "physical_device_id", "logical_device_id", "model", "board_id",
            "pci_identity", "selection_environment"):
        if parsed_device[field] != device[field]:
            raise ValueError(f"retained telemetry disagrees with device.{field}")
    _require_exact_mapping(
        manifest["idle_telemetry"], {"pre", "post"}, "idle telemetry")
    for moment in ("pre", "post"):
        telemetry = manifest["idle_telemetry"][moment]
        _require_exact_mapping(
            telemetry, {"values", "path", "sha256", "command"},
            f"{moment} telemetry")
        if telemetry["values"] != {
                "aic_percent": 0,
                "aiv_percent": 0,
                "hbm_bandwidth_percent": 0}:
            raise ValueError(f"{moment} telemetry is not idle")
        telemetry_path = capture_dir / telemetry["path"]
        if not telemetry_path.is_file() or _sha256(telemetry_path) != telemetry["sha256"]:
            raise ValueError(f"{moment} telemetry artifact identity changed")
    if manifest["clock"] != {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
            "derivation": (
                "max_i((long_i-short_i)*tick/(3*iterations)) over "
                "same-invocation physical-actor pairs")}:
        raise ValueError("capture clock contract changed")
    if manifest["campaign"] != {
            "aic_groups": list(AIC_GROUPS),
            "aiv_per_aic": 2,
            "modes": list(MODES),
            "iterations": ITERATIONS,
            "trials": TRIALS,
            "cell_count": len(CELLS),
            "order": "repeated_position_balanced_forward_reverse_phases"}:
        raise ValueError("capture campaign changed")
    capture = manifest["capture"]
    _require_exact_mapping(
        capture,
        {"raw", "raw_sha256", "raw_rows", "runner_pid", "argv",
         "working_directory", "trials"},
        "capture artifacts")
    raw_path = capture_dir / capture["raw"]
    if not raw_path.is_file() or _sha256(raw_path) != capture["raw_sha256"]:
        raise ValueError("raw CSV identity changed")
    if capture["raw_rows"] != TRIALS * len(CELLS):
        raise ValueError("manifest raw-row count changed")
    if len(capture["trials"]) != TRIALS:
        raise ValueError("manifest trial-artifact count changed")
    return manifest, raw_path


def _decode_row(raw, expected_cell, expected_direction, expected_position):
    if raw["order_direction"] != expected_direction:
        raise ValueError("raw row changed the expected direction")
    if int(raw["order_position"]) != expected_position:
        raise ValueError("raw row changed the expected position")
    if int(raw["process_pid"]) <= 0:
        raise ValueError("raw row has an invalid child PID")
    child_line = ",".join(raw[field] for field in CHILD_FIELDS)
    _parse_child_row(child_line, expected_cell)
    return {
        "run_id": raw["run_id"],
        "trial": int(raw["trial"]),
        "order_position": expected_position,
        "mode": expected_cell[0],
        "aic_groups": expected_cell[1],
        "vec1_completion_ns": float(raw["vec1_critical_completion_ns"]),
        "vec2_completion_ns": float(raw["vec2_critical_completion_ns"]),
        "critical_completion_ns": float(raw["critical_completion_ns"]),
    }


def _load_capture(capture_dir):
    manifest, raw_path = _validate_manifest(capture_dir)
    rows = []
    identities = set()
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("raw CSV has an invalid schema")
        for raw in reader:
            trial = int(raw["trial"])
            if trial not in range(1, TRIALS + 1):
                raise ValueError("raw row has an invalid trial")
            position = int(raw["order_position"])
            direction, ordered = _ordered_cells(trial - 1)
            if position not in range(len(ordered)):
                raise ValueError("raw row has an invalid order position")
            identity = (trial, position)
            if identity in identities:
                raise ValueError(f"raw CSV duplicates row {identity}")
            identities.add(identity)
            row = _decode_row(raw, ordered[position], direction, position)
            if row["run_id"] != manifest["run_id"]:
                raise ValueError("raw run ID differs from manifest")
            rows.append(row)
    expected = {
        (trial, position)
        for trial in range(1, TRIALS + 1)
        for position in range(len(CELLS))
    }
    if identities != expected:
        raise ValueError("raw CSV does not cover the complete campaign")
    return manifest, rows


def _cell_summary(rows, mode, groups, domain):
    selected = [
        row for row in rows
        if row["mode"] == mode and row["aic_groups"] == groups
    ]
    training = [
        _completion(row, domain)
        for row in selected if row["trial"] in TRAINING_TRIALS
    ]
    validation = [
        _completion(row, domain)
        for row in selected if row["trial"] in VALIDATION_TRIALS
    ]
    expected_count = len(TRAINING_TRIALS)
    if len(training) != expected_count or len(validation) != expected_count:
        raise ValueError("cell does not have balanced training/validation trials")
    if any(not math.isfinite(value) or value <= 0 for value in training + validation):
        raise ValueError("cell contains non-positive completion time")
    return {
        "training_median_ns": statistics.median(training),
        "validation_median_ns": statistics.median(validation),
        "training_relative_span": _relative_span(training),
        "validation_relative_span": _relative_span(validation),
        "samples_per_phase": expected_count,
    }


def build_result(manifest, rows):
    """Build active-group order-statistic completion parameters."""

    summaries = {}
    for mode, groups in CELLS:
        for domain in _domains(mode):
            key = f"{mode}|aic_groups={groups}|domain={domain}"
            summaries[key] = _cell_summary(rows, mode, groups, domain)
    parameters = {}
    validation = {}
    for mode in MODES:
        parameters[mode] = {}
        validation[mode] = {}
        domain, = _domains(mode)
        for groups in AIC_GROUPS:
            summary = summaries[
                f"{mode}|aic_groups={groups}|domain={domain}"]
            parameter = summary["training_median_ns"]
            observed = summary["validation_median_ns"]
            error = abs(parameter - observed) / observed
            parameters[mode][str(groups)] = parameter
            validation[mode][str(groups)] = {
                "validation_ape": error,
                "training_relative_span": summary[
                    "training_relative_span"],
                "validation_relative_span": summary[
                    "validation_relative_span"],
            }
    group_gates = {}
    for groups in AIC_GROUPS:
        selected = [
            validation[mode][str(groups)] for mode in MODES]
        maximum_trial_spread = max(
            max(metrics["training_relative_span"],
                metrics["validation_relative_span"])
            for metrics in selected)
        maximum_validation_ape = max(
            metrics["validation_ape"] for metrics in selected)
        validation_mape = statistics.fmean(
            metrics["validation_ape"] for metrics in selected)
        group_gates[str(groups)] = {
            "observed_maximum_relative_trial_spread": maximum_trial_spread,
            "observed_validation_mape": validation_mape,
            "observed_maximum_validation_ape": maximum_validation_ape,
            "passed": (
                maximum_trial_spread <= MAXIMUM_RELATIVE_TRIAL_SPREAD
                and validation_mape <= MAXIMUM_VALIDATION_MAPE
                and maximum_validation_ape <= MAXIMUM_VALIDATION_APE),
        }
    eligible_groups = tuple(
        groups for groups in AIC_GROUPS
        if group_gates[str(groups)]["passed"])
    gates = {
        "maximum_relative_trial_spread": MAXIMUM_RELATIVE_TRIAL_SPREAD,
        "maximum_validation_mape": MAXIMUM_VALIDATION_MAPE,
        "maximum_validation_ape": MAXIMUM_VALIDATION_APE,
        "by_active_aic_group": group_gates,
        "passed": bool(eligible_groups),
    }
    return {
        "schema": "ascend_910b_aiv_workspace_transfer_result_v4",
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": "fia_aiv_workspace_transfer_completion_v4",
        "run_id": manifest["run_id"],
        "manifest_sha256": None,
        "parameter_unit": "ns_per_source_task_critical_completion",
        "parameter_interpretation": (
            "maximum same-invocation completion across the active primary "
            "AIV actor set; compose with that source stage's aggregate "
            "read-plus-write bandwidth time using max"),
        "training_trials": list(TRAINING_TRIALS),
        "validation_trials": list(VALIDATION_TRIALS),
        "cell_summaries": summaries,
        "completion_parameters_ns": parameters,
        "eligible_active_aic_groups": list(eligible_groups),
        "validation": validation,
        "gates": gates,
        "production_or_evaluation_duration_consumed": False,
        "candidate_eligible": gates["passed"],
    }


def build_candidate(result, result_path):
    """Create a detached candidate that cannot silently enter Forward."""

    eligible = result["candidate_eligible"]
    rejection_reasons = []
    if not result["gates"]["passed"]:
        rejection_reasons.append("predeclared measurement gates failed")
    return {
        "schema": "ascend_910b_aiv_workspace_transfer_candidate_v4",
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": result["mechanism_id"],
        "source_result": {
            "path": result_path.name,
            "sha256": _sha256(result_path),
        },
        "bounds": {
            "task_kind": "mixed",
            "aic_groups_measured": list(AIC_GROUPS),
            "aic_groups_eligible": result["eligible_active_aic_groups"],
            "aiv_per_aic": 2,
            "active_aiv_per_aic_for_d64_m4": 1,
            "vec1_s512_gm_to_ub_bytes_per_task": 8192,
            "vec1_s512_ub_to_gm_bytes_per_task": 4096,
            "vec1_s1024_gm_to_ub_bytes_per_task": 16384,
            "vec1_s1024_ub_to_gm_bytes_per_task": 8192,
            "vec2_final_state_read_bytes_per_task": 1024,
            "vec2_final_state_write_bytes_per_task": 512,
        },
        "parameter_unit": result["parameter_unit"],
        "completion_parameters_ns": ({
            mode: {
                str(groups): result["completion_parameters_ns"][mode][
                    str(groups)]
                for groups in result["eligible_active_aic_groups"]
            }
            for mode in MODES
        } if eligible else None),
        "composition": result["parameter_interpretation"],
        "candidate_eligible": eligible,
        "rejection_reasons": rejection_reasons,
        "release_status": (
            "candidate_only_not_canonical" if eligible
            else "rejected_not_publishable"),
        "production_or_evaluation_duration_consumed": False,
    }


def _write_json_exclusive(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    args = parser.parse_args()
    capture_dir = args.capture_dir.resolve()
    manifest, rows = _load_capture(capture_dir)
    result = build_result(manifest, rows)
    result["manifest_sha256"] = _sha256(capture_dir / "manifest.json")
    result_path = capture_dir / "result.json"
    candidate_path = capture_dir / "candidate.json"
    if result_path.exists() or candidate_path.exists():
        raise FileExistsError("result.json or candidate.json already exists")
    _write_json_exclusive(result_path, result)
    candidate = build_candidate(result, result_path)
    _write_json_exclusive(candidate_path, candidate)
    print(json.dumps({
        "result": str(result_path),
        "candidate": str(candidate_path),
        "candidate_eligible": candidate["candidate_eligible"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
