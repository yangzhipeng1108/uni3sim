#!/usr/bin/env python3
"""Audit CANN AIV tiling-data materialization trials."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


BLOCK_DIMS = (1, 24, 48)
MINIMUM_TRIALS = 6
SUPPORTED_PAYLOAD_BYTES = (28, 48, 56, 80, 192)
DEFAULT_PAYLOAD_BYTES = 28
SYSTEM_COUNTER_TICK_NS = 20.0
MAXIMUM_RELATIVE_MEDIAN_SPREAD = 0.10
MAXIMUM_VALIDATION_MAPE = 0.10
MAXIMUM_VALIDATION_APE = 0.15
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "payload_bytes",
    "block_dim", "iterations", "short_cycles", "long_cycles", "tick_ns",
    "service_time_ns", "validation",
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
        raise ValueError("affine fit requires distinct block dimensions")
    slope = math.fsum(
        (block_dim - mean_x) * (duration_ns - mean_y)
        for block_dim, duration_ns in points) / denominator
    intercept = mean_y - slope * mean_x
    if (not math.isfinite(intercept) or intercept <= 0
            or not math.isfinite(slope) or slope <= 0):
        raise ValueError("tiling materialization affine fit must be positive")
    return intercept, slope


def _error_metrics(points, startup_ns, serialized_ns_per_aiv):
    errors = []
    absolute_errors = []
    actuals = []
    for block_dim, actual_ns in points:
        predicted_ns = startup_ns + serialized_ns_per_aiv * block_dim
        absolute_error = abs(predicted_ns - actual_ns)
        errors.append(absolute_error / actual_ns)
        absolute_errors.append(absolute_error)
        actuals.append(actual_ns)
    return {
        "mape": statistics.fmean(errors),
        "wmae": math.fsum(absolute_errors) / math.fsum(actuals),
        "maximum_ape": max(errors),
    }


def _complete_max_error_metrics(
        points, single_worker_completion_ns, serialized_ns_per_aiv):
    errors = []
    absolute_errors = []
    actuals = []
    for block_dim, actual_ns in points:
        predicted_ns = max(
            single_worker_completion_ns,
            serialized_ns_per_aiv * block_dim)
        absolute_error = abs(predicted_ns - actual_ns)
        errors.append(absolute_error / actual_ns)
        absolute_errors.append(absolute_error)
        actuals.append(actual_ns)
    return {
        "mape": statistics.fmean(errors),
        "wmae": math.fsum(absolute_errors) / math.fsum(actuals),
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
        if (len(file_rows) < MINIMUM_TRIALS * len(BLOCK_DIMS)
                or len(file_rows) % (MINIMUM_TRIALS * len(BLOCK_DIMS)) != 0):
            raise ValueError(f"{path} has an invalid row count")
        rows.extend(file_rows)
    return rows


def build_report(raw_paths, source_commit, source_sha256,
                 template_path, template_sha256, host_tiler_sha256,
                 payload_bytes=DEFAULT_PAYLOAD_BYTES):
    if len(raw_paths) != 3:
        raise ValueError("exactly three independent raw runs are required")
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full lowercase Git hash")
    if payload_bytes not in SUPPORTED_PAYLOAD_BYTES:
        raise ValueError(
            f"payload_bytes must be one of {SUPPORTED_PAYLOAD_BYTES}")
    for value, name in (
            (source_sha256, "source_sha256"),
            (template_sha256, "template_sha256"),
            (host_tiler_sha256, "host_tiler_sha256")):
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA256")
    rows = _load_rows(raw_paths)
    trials_per_run = len(rows) // (len(raw_paths) * len(BLOCK_DIMS))
    parsed = []
    run_ids = []
    for index, row in enumerate(rows, start=1):
        context = f"row {index}"
        run_id = row["run_id"]
        trial = int(row["trial"])
        order_position = int(row["order_position"])
        observed_payload_bytes = int(row["payload_bytes"])
        block_dim = int(row["block_dim"])
        iterations = int(row["iterations"])
        tick_ns = float(row["tick_ns"])
        service_time_ns = float(row["service_time_ns"])
        if run_id not in run_ids:
            run_ids.append(run_id)
        if (trial < 1 or trial > trials_per_run
                or order_position not in range(len(BLOCK_DIMS))):
            raise ValueError(f"{context} has an invalid schedule position")
        if observed_payload_bytes not in SUPPORTED_PAYLOAD_BYTES:
            raise ValueError(f"{context} changed the frozen payload size")
        if (observed_payload_bytes != payload_bytes
                or block_dim not in BLOCK_DIMS):
            raise ValueError(f"{context} changed the frozen mechanism cell")
        if iterations < 4096:
            raise ValueError(f"{context} has too few iterations")
        if tick_ns != SYSTEM_COUNTER_TICK_NS:
            raise ValueError(f"{context} changed the SYS_CNT tick")
        if not math.isfinite(service_time_ns) or service_time_ns <= 0:
            raise ValueError(f"{context} has an invalid service time")
        if row["validation"] != "pass":
            raise ValueError(f"{context} failed device validation")
        parsed.append({
            "run_id": run_id,
            "trial": trial,
            "order_position": order_position,
            "block_dim": block_dim,
            "service_time_ns": service_time_ns,
        })
    if len(run_ids) != 3:
        raise ValueError("raw inputs must contain three distinct run IDs")

    expected_cells = {
        (run_id, trial, block_dim)
        for run_id in run_ids
        for trial in range(1, trials_per_run + 1)
        for block_dim in BLOCK_DIMS
    }
    observed_cells = {
        (row["run_id"], row["trial"], row["block_dim"])
        for row in parsed
    }
    if observed_cells != expected_cells or len(parsed) != len(expected_cells):
        raise ValueError("raw rows do not cover the exact run/trial/grid matrix")

    medians_by_run_and_block = {
        run_id: {
            str(block_dim): statistics.median([
                row["service_time_ns"] for row in parsed
                if row["run_id"] == run_id
                and row["block_dim"] == block_dim
            ])
            for block_dim in BLOCK_DIMS
        }
        for run_id in run_ids
    }
    run_spreads = {
        str(block_dim): _relative_span([
            medians_by_run_and_block[run_id][str(block_dim)]
            for run_id in run_ids
        ])
        for block_dim in BLOCK_DIMS
    }
    training_run_ids = tuple(run_ids[:2])
    validation_run_ids = tuple(run_ids[2:])
    training_points = tuple(
        (block_dim, medians_by_run_and_block[run_id][str(block_dim)])
        for run_id in training_run_ids
        for block_dim in BLOCK_DIMS
    )
    validation_points = tuple(
        (block_dim, medians_by_run_and_block[run_id][str(block_dim)])
        for run_id in validation_run_ids
        for block_dim in BLOCK_DIMS
    )
    if payload_bytes in (48, 56, 192):
        fits_by_training_run = {
            run_id: (
                medians_by_run_and_block[run_id]["1"],
                statistics.median((
                    medians_by_run_and_block[run_id]["24"] / 24,
                    medians_by_run_and_block[run_id]["48"] / 48,
                )),
            )
            for run_id in training_run_ids
        }
        single_worker_completion_ns = statistics.median(
            fit[0] for fit in fits_by_training_run.values())
        serialized_ns_per_aiv = statistics.median(
            fit[1] for fit in fits_by_training_run.values())
        training_metrics = _complete_max_error_metrics(
            training_points, single_worker_completion_ns,
            serialized_ns_per_aiv)
        validation_metrics = _complete_max_error_metrics(
            validation_points, single_worker_completion_ns,
            serialized_ns_per_aiv)
        parameter_stability = {
            "single_worker_completion_relative_span": _relative_span(tuple(
                fit[0] for fit in fits_by_training_run.values())),
            "serialized_ns_per_aiv_relative_span": _relative_span(tuple(
                fit[1] for fit in fits_by_training_run.values())),
        }
        model = {
            48: "rope_aiv_tiling_materialization_complete_max_v1",
            56: "broadcast_one_dim_tiling_materialization_complete_max_v1",
            192: "gather_v3_aiv_tiling_materialization_complete_max_v1",
        }[payload_bytes]
        formula = (
            "max(single_worker_completion_ns, serialized_ns_per_aiv * "
            "active_aiv_workers)")
        parameters = {
            "single_worker_completion_ns": single_worker_completion_ns,
            "serialized_ns_per_aiv": serialized_ns_per_aiv,
        }
        fits_report = {
            run_id: {
                "single_worker_completion_ns": fit[0],
                "serialized_ns_per_aiv": fit[1],
            }
            for run_id, fit in fits_by_training_run.items()
        }
    else:
        fits_by_training_run = {
            run_id: _fit_affine(tuple(
                (block_dim,
                 medians_by_run_and_block[run_id][str(block_dim)])
                for block_dim in BLOCK_DIMS))
            for run_id in training_run_ids
        }
        startup_ns = statistics.median(
            fit[0] for fit in fits_by_training_run.values())
        serialized_ns_per_aiv = statistics.median(
            fit[1] for fit in fits_by_training_run.values())
        training_metrics = _error_metrics(
            training_points, startup_ns, serialized_ns_per_aiv)
        validation_metrics = _error_metrics(
            validation_points, startup_ns, serialized_ns_per_aiv)
        parameter_stability = {
            "startup_relative_span": _relative_span(tuple(
                fit[0] for fit in fits_by_training_run.values())),
            "serialized_ns_per_aiv_relative_span": _relative_span(tuple(
                fit[1] for fit in fits_by_training_run.values())),
        }
        model = "shared_aiv_tiling_materialization_affine_v1"
        formula = "startup_ns + serialized_ns_per_aiv * active_aiv_workers"
        parameters = {
            "startup_ns": startup_ns,
            "serialized_ns_per_aiv": serialized_ns_per_aiv,
        }
        fits_report = {
            run_id: {
                "startup_ns": fit[0],
                "serialized_ns_per_aiv": fit[1],
            }
            for run_id, fit in fits_by_training_run.items()
        }
    measurement_gate_passed = (
        all(value <= MAXIMUM_RELATIVE_MEDIAN_SPREAD
            for value in run_spreads.values())
        and all(value <= MAXIMUM_RELATIVE_MEDIAN_SPREAD
                for value in parameter_stability.values())
        and validation_metrics["mape"] <= MAXIMUM_VALIDATION_MAPE
        and validation_metrics["maximum_ape"] <= MAXIMUM_VALIDATION_APE
    )
    if payload_bytes == 48:
        mechanism = (
            "one AIV Rope 48-byte InitTilingData source sequence: "
            "64-byte rounded GM-to-UB DataCopy, PIPE_ALL barrier, twelve "
            "uint32 scalar reads, and PIPE_ALL barrier")
        scope = "CANN AIV Rope 48-byte InitTilingData source sequence"
    elif payload_bytes == 56:
        mechanism = (
            "one AIV 56-byte BroadcastOneDim InitTilingData GM-to-UB copy, "
            "MTE2-to-Scalar handoff, UB-to-local copy, and PIPE_ALL barrier")
        scope = "CANN AIV BroadcastOneDim typed tiling-struct materialization"
    elif payload_bytes == 192:
        mechanism = (
            "one AIV 192-byte GatherV3 InitTilingData GM-to-UB copy, "
            "MTE2-to-Scalar handoff, UB-to-local copy, and PIPE_ALL barrier")
        scope = "CANN AIV GatherV3 typed tiling-struct materialization"
    else:
        mechanism = (
            f"one AIV {payload_bytes}-byte InitTilingData GM-to-UB copy, "
            "MTE2-to-Scalar handoff, UB-to-local copy, and PIPE_ALL barrier")
        scope = "CANN AIV typed tiling-struct materialization"
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "tiling_struct_materialization",
        "payload_bytes": payload_bytes,
        "model": model,
        "formula": formula,
        "request_size_unit": "active_aiv_workers",
        "duration_unit": "ns_per_complete_tiling_materialization_phase",
        "minimum_active_aiv_workers": min(BLOCK_DIMS),
        "maximum_active_aiv_workers": max(BLOCK_DIMS),
        "parameters": parameters,
        "mechanism": mechanism,
        "scope": scope,
        "composition": "one shared phase before every AIV operator body",
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "template_path": template_path,
        "template_sha256": template_sha256,
        "host_tiler_sha256": host_tiler_sha256,
        "raw_artifacts": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in raw_paths
        ],
        "independent_runs": len(run_ids),
        "samples_per_cell": trials_per_run,
        "block_dims": list(BLOCK_DIMS),
        "medians_ns_by_run_and_block_dim": medians_by_run_and_block,
        "run_relative_median_spread_by_block_dim": run_spreads,
        "training_run_ids": list(training_run_ids),
        "validation_run_ids": list(validation_run_ids),
        "fits_by_training_run": fits_report,
        "parameter_stability": parameter_stability,
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "maximum_relative_median_spread": MAXIMUM_RELATIVE_MEDIAN_SPREAD,
        "maximum_validation_mape": MAXIMUM_VALIDATION_MAPE,
        "maximum_validation_ape": MAXIMUM_VALIDATION_APE,
        "measurement_gate_passed": measurement_gate_passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": measurement_gate_passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path, action="append")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--template-path", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--host-tiler-sha256", required=True)
    parser.add_argument(
        "--payload-bytes", type=int, choices=SUPPORTED_PAYLOAD_BYTES,
        default=DEFAULT_PAYLOAD_BYTES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        args.raw, args.source_commit, args.source_sha256,
        args.template_path, args.template_sha256, args.host_tiler_sha256,
        args.payload_bytes)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
