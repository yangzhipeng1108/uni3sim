#!/usr/bin/env python3
"""Audit mixed-core CANN FIA tiling-data materialization trials."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


AIC_GROUPS = (1, 8, 16, 24)
AIV_PER_GROUP = 2
EXPECTED_TRIALS = 8
PAYLOAD_BYTES = 1560
AIV_TRANSFER_BYTES = 1568
SYSTEM_COUNTER_TICK_NS = 20.0
MAXIMUM_RELATIVE_MEDIAN_SPREAD = 0.01
MAXIMUM_VALIDATION_MAPE = 0.06
MAXIMUM_VALIDATION_APE = 0.15
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "payload_bytes",
    "aic_block_dim", "aiv_block_dim", "iterations", "short_aic_cycles",
    "long_aic_cycles", "short_aiv_cycles", "long_aiv_cycles", "tick_ns",
    "aic_service_ns", "aiv_service_ns", "critical_service_ns",
    "validation",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_span(values):
    center = statistics.median(values)
    if center <= 0:
        return math.inf
    return (max(values) - min(values)) / center


def _fit_affine(points):
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = math.fsum(
        (point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0:
        raise ValueError("FIA tiling fit requires distinct AIV counts")
    slope = math.fsum(
        (aiv_count - mean_x) * (duration_ns - mean_y)
        for aiv_count, duration_ns in points) / denominator
    intercept = mean_y - slope * mean_x
    if (not math.isfinite(intercept) or intercept <= 0
            or not math.isfinite(slope) or slope <= 0):
        raise ValueError("FIA tiling affine coefficients must be positive")
    return intercept, slope


def _error_metrics(points, intercept, slope):
    errors = [
        abs(intercept + slope * count - actual) / actual
        for count, actual in points
    ]
    return {
        "mape": statistics.fmean(errors),
        "maximum_ape": max(errors),
    }


def _load_rows(paths):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"{path} has an invalid header")
            file_rows = list(reader)
        if len(file_rows) != EXPECTED_TRIALS * len(AIC_GROUPS):
            raise ValueError(f"{path} has an invalid row count")
        rows.extend(file_rows)
    return rows


def build_report(raw_paths, source_commit, source_snapshot_sha256,
                 binary_sha256, macro_generator_sha256,
                 fia_entry_sha256, fia_kernel_sha256):
    """Build the publishability report for one frozen mechanism."""

    if len(raw_paths) != 3:
        raise ValueError("exactly three independent raw runs are required")
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full lowercase Git hash")
    for value, name in (
            (source_snapshot_sha256, "source_snapshot_sha256"),
            (binary_sha256, "binary_sha256"),
            (macro_generator_sha256, "macro_generator_sha256"),
            (fia_entry_sha256, "fia_entry_sha256"),
            (fia_kernel_sha256, "fia_kernel_sha256")):
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA256")
    rows = _load_rows(raw_paths)
    parsed = []
    run_ids = []
    for index, row in enumerate(rows, start=1):
        context = f"row {index}"
        run_id = row["run_id"]
        trial = int(row["trial"])
        order_position = int(row["order_position"])
        payload_bytes = int(row["payload_bytes"])
        aic_groups = int(row["aic_block_dim"])
        aiv_workers = int(row["aiv_block_dim"])
        iterations = int(row["iterations"])
        tick_ns = float(row["tick_ns"])
        aic_ns = float(row["aic_service_ns"])
        aiv_ns = float(row["aiv_service_ns"])
        critical_ns = float(row["critical_service_ns"])
        if run_id not in run_ids:
            run_ids.append(run_id)
        if (trial not in range(1, EXPECTED_TRIALS + 1)
                or order_position not in range(len(AIC_GROUPS))):
            raise ValueError(f"{context} has an invalid schedule position")
        if (payload_bytes != PAYLOAD_BYTES
                or aic_groups not in AIC_GROUPS
                or aiv_workers != AIV_PER_GROUP * aic_groups):
            raise ValueError(f"{context} changed the frozen mechanism cell")
        if iterations < 4096:
            raise ValueError(f"{context} has too few iterations")
        if tick_ns != SYSTEM_COUNTER_TICK_NS:
            raise ValueError(f"{context} changed the SYS_CNT tick")
        if any(not math.isfinite(value) or value <= 0
               for value in (aic_ns, aiv_ns, critical_ns)):
            raise ValueError(f"{context} has an invalid service time")
        if not math.isclose(
                critical_ns, max(aic_ns, aiv_ns),
                rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{context} has an invalid critical service")
        if row["validation"] != "pass":
            raise ValueError(f"{context} failed device validation")
        if (len(row["short_aic_cycles"].split(":")) != aic_groups
                or len(row["long_aic_cycles"].split(":")) != aic_groups
                or len(row["short_aiv_cycles"].split(":")) != aiv_workers
                or len(row["long_aiv_cycles"].split(":")) != aiv_workers):
            raise ValueError(f"{context} changed the actor topology")
        parsed.append({
            "run_id": run_id,
            "trial": trial,
            "aic_groups": aic_groups,
            "aiv_workers": aiv_workers,
            "aic_service_ns": aic_ns,
            "aiv_service_ns": aiv_ns,
        })
    if len(run_ids) != 3:
        raise ValueError("raw inputs must contain three distinct run IDs")
    expected_cells = {
        (run_id, trial, group)
        for run_id in run_ids
        for trial in range(1, EXPECTED_TRIALS + 1)
        for group in AIC_GROUPS
    }
    observed_cells = {
        (row["run_id"], row["trial"], row["aic_groups"])
        for row in parsed
    }
    if observed_cells != expected_cells or len(parsed) != len(expected_cells):
        raise ValueError("raw rows do not cover the exact run/trial/grid matrix")

    medians = {
        run_id: {
            str(group): {
                "aic_service_ns": statistics.median(
                    row["aic_service_ns"] for row in parsed
                    if row["run_id"] == run_id
                    and row["aic_groups"] == group),
                "aiv_service_ns": statistics.median(
                    row["aiv_service_ns"] for row in parsed
                    if row["run_id"] == run_id
                    and row["aic_groups"] == group),
            }
            for group in AIC_GROUPS
        }
        for run_id in run_ids
    }
    spread_by_group = {
        str(group): {
            domain: _relative_span(tuple(
                medians[run_id][str(group)][domain]
                for run_id in run_ids))
            for domain in ("aic_service_ns", "aiv_service_ns")
        }
        for group in AIC_GROUPS
    }
    training_run_ids = tuple(run_ids[:2])
    validation_run_ids = tuple(run_ids[2:])
    training_aic = tuple(
        medians[run_id][str(group)]["aic_service_ns"]
        for run_id in training_run_ids for group in AIC_GROUPS)
    aic_completion_ns = statistics.median(training_aic)
    training_aiv = tuple(
        (AIV_PER_GROUP * group,
         medians[run_id][str(group)]["aiv_service_ns"])
        for run_id in training_run_ids for group in AIC_GROUPS)
    validation_aiv = tuple(
        (AIV_PER_GROUP * group,
         medians[run_id][str(group)]["aiv_service_ns"])
        for run_id in validation_run_ids for group in AIC_GROUPS)
    aiv_startup_ns, aiv_per_active_worker_ns = _fit_affine(training_aiv)
    validation_metrics = _error_metrics(
        validation_aiv, aiv_startup_ns, aiv_per_active_worker_ns)
    validation_aic_relative_errors = tuple(
        abs(aic_completion_ns - medians[run_id][str(group)][
            "aic_service_ns"]) / medians[run_id][str(group)][
                "aic_service_ns"]
        for run_id in validation_run_ids for group in AIC_GROUPS)
    measurement_gate_passed = (
        all(value <= MAXIMUM_RELATIVE_MEDIAN_SPREAD
            for domain_spreads in spread_by_group.values()
            for value in domain_spreads.values())
        and validation_metrics["mape"] <= MAXIMUM_VALIDATION_MAPE
        and validation_metrics["maximum_ape"] <= MAXIMUM_VALIDATION_APE
        and max(validation_aic_relative_errors)
        <= MAXIMUM_RELATIVE_MEDIAN_SPREAD
    )
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": "fia_mixed_tiling_data_materialization_v1",
        "scope": "one source-mandatory actor-local FIA tiling prelude",
        "composition": (
            "AIC and AIV preludes begin concurrently; each domain gates only "
            "its own first source stage"),
        "formula": {
            "aic": "aic_completion_ns",
            "aiv": (
                "aiv_startup_ns + active_aiv_workers * "
                "aiv_per_active_worker_ns"),
        },
        "bounds": {
            "task_kind": "mixed",
            "payload_bytes": PAYLOAD_BYTES,
            "aiv_transfer_bytes": AIV_TRANSFER_BYTES,
            "aiv_per_group": AIV_PER_GROUP,
            "minimum_active_aic_groups": min(AIC_GROUPS),
            "maximum_active_aic_groups": max(AIC_GROUPS),
        },
        "parameters_ns": {
            "aic_completion_ns": aic_completion_ns,
            "aiv_startup_ns": aiv_startup_ns,
            "aiv_per_active_worker_ns": aiv_per_active_worker_ns,
        },
        "validation": {
            "training_run_ids": list(training_run_ids),
            "validation_run_ids": list(validation_run_ids),
            "medians_ns_by_run_and_aic_groups": medians,
            "relative_median_spread_by_aic_groups": spread_by_group,
            "aiv_model": validation_metrics,
            "aic_maximum_relative_error": max(
                validation_aic_relative_errors),
            "maximum_relative_median_spread": (
                MAXIMUM_RELATIVE_MEDIAN_SPREAD),
            "maximum_validation_mape": MAXIMUM_VALIDATION_MAPE,
            "maximum_validation_ape": MAXIMUM_VALIDATION_APE,
            "measurement_gate_passed": measurement_gate_passed,
        },
        "source_semantics": {
            "aic": "copy_data_align64(local, GM, 1560)",
            "aiv": (
                "1568-byte GM-to-UB copy; MTE2-to-Scalar event; 1560-byte "
                "UB-to-local copy; PIPE_ALL"),
            "topology": "KERNEL_TYPE_MIX_AIC_1_2",
            "macro_generator_sha256": macro_generator_sha256,
            "fia_entry_sha256": fia_entry_sha256,
            "fia_kernel_sha256": fia_kernel_sha256,
        },
        "provenance": {
            "source_commit": source_commit,
            "source_snapshot_sha256": source_snapshot_sha256,
            "binary_sha256": binary_sha256,
            "raw_artifacts": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in raw_paths
            ],
        },
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": measurement_gate_passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path, action="append")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--macro-generator-sha256", required=True)
    parser.add_argument("--fia-entry-sha256", required=True)
    parser.add_argument("--fia-kernel-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        args.raw, args.source_commit, args.source_snapshot_sha256,
        args.binary_sha256, args.macro_generator_sha256,
        args.fia_entry_sha256, args.fia_kernel_sha256)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
