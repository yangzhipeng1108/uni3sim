#!/usr/bin/env python3
"""Measure generic Python attention-metadata layer binding service on Ascend.

The Ascend vLLM runner attaches one already-built metadata object to every
attention-layer key in a Python dictionary.  This runner isolates only that
source statement with synthetic layer names and an opaque object: it creates
no NPU tensor, runs no model, and consumes no Serving workload or duration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)
CHAIN_LENGTHS = (1, 8, 64, 512)
FIT_CHAIN_LENGTHS = frozenset((1, 8, 64))
TRIALS = 100
WARMUP_REPEATS = 20
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _append_json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile needs samples")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("linear fit needs at least two points")
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("attention-metadata binding fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        raise RuntimeError("attention-metadata binding fit is invalid")
    return float(intercept), float(slope)


def _validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    owner_size = len(OUTPUT_OWNER_PARTS)
    offsets = tuple(
        index
        for index in range(len(resolved.parts) - owner_size + 1)
        if resolved.parts[index:index + owner_size] == OUTPUT_OWNER_PARTS
    )
    if len(offsets) != 1 or len(resolved.parts) <= offsets[0] + owner_size:
        raise ValueError("--output-dir must be a run below the Ascend owner")
    if resolved.exists():
        raise FileExistsError("--output-dir must not already exist")
    return resolved


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    arguments = parser.parse_args()
    arguments.output_dir = _validate_output_dir(arguments.output_dir)
    if (len(arguments.baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.baseline_commit)):
        parser.error("--baseline-commit must be a lowercase full Git SHA")
    if len(arguments.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    return arguments


def _bind_metadata(layer_names: tuple[str, ...], metadata: object) -> dict[str, object]:
    """Execute exactly the source-level ``attn_metadata_dict`` assignment."""

    result = {}
    for layer_name in layer_names:
        result[layer_name] = metadata
    if len(result) != len(layer_names) or any(
            value is not metadata for value in result.values()):
        raise RuntimeError("metadata binding did not preserve source identity")
    return result


def main() -> None:
    arguments = _arguments()
    runner_path = Path(__file__).resolve()
    runner_sha256 = _sha256(runner_path)
    if runner_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from the audited snapshot")
    arguments.output_dir.mkdir(parents=True)
    trials_path = arguments.output_dir / "trials.jsonl"
    manifest_path = arguments.output_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "hardware_id": HARDWARE_ID,
        "baseline_commit": arguments.baseline_commit,
        "container_name": arguments.container_name,
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "contract": {
            "source_file": (
                "vllm_ascend/worker/model_runner_v1.py::"
                "NPUModelRunner._build_attention_metadata"),
            "source_statement": (
                "for layer_name in attn_group.layer_names: "
                "attn_metadata_dict[layer_name] = attn_metadata_i"),
            "parameter": "python_attention_metadata_dictionary_binding",
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": sorted(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": 512,
            "trials_per_cell": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "host_clock": "time.perf_counter_ns",
            "host_thread_cpu_clock": "time.thread_time_ns",
            "npu_access": "none",
            "production_or_evaluation_duration_consumed": False,
            "production_model_or_shape_consumed": False,
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
    }
    _write_json(manifest_path, manifest)
    try:
        metadata = object()
        samples = {count: [] for count in CHAIN_LENGTHS}
        for count in CHAIN_LENGTHS:
            layer_names = tuple(f"synthetic-attention-layer-{index}"
                                for index in range(count))
            for _ in range(WARMUP_REPEATS):
                _bind_metadata(layer_names, metadata)
        sequence = 0
        for trial in range(TRIALS):
            ordered_counts = CHAIN_LENGTHS[trial % len(CHAIN_LENGTHS):] + (
                CHAIN_LENGTHS[:trial % len(CHAIN_LENGTHS)])
            if trial % 2:
                ordered_counts = tuple(reversed(ordered_counts))
            for count in ordered_counts:
                layer_names = tuple(f"synthetic-attention-layer-{index}"
                                    for index in range(count))
                cpu_start_ns = time.thread_time_ns()
                wall_start_ns = time.perf_counter_ns()
                _bind_metadata(layer_names, metadata)
                wall_end_ns = time.perf_counter_ns()
                cpu_end_ns = time.thread_time_ns()
                wall_ns = wall_end_ns - wall_start_ns
                cpu_ns = cpu_end_ns - cpu_start_ns
                if wall_ns <= 0 or cpu_ns <= 0:
                    raise RuntimeError("attention-metadata timing must be positive")
                sequence += 1
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "sequence": sequence,
                    "trial": trial + 1,
                    "layer_binding_count": count,
                    "host_call_wall_ns": wall_ns,
                    "host_call_thread_cpu_ns": cpu_ns,
                    "host_call_non_cpu_wall_ns": wall_ns - cpu_ns,
                }
                _append_json_line(trials_path, row)
                samples[count].append(row)
        cells = []
        medians = {}
        for count in CHAIN_LENGTHS:
            wall_values = [row["host_call_wall_ns"] for row in samples[count]]
            cpu_values = [row["host_call_thread_cpu_ns"] for row in samples[count]]
            medians[count] = {
                "wall_ns": float(statistics.median(wall_values)),
                "thread_cpu_ns": float(statistics.median(cpu_values)),
            }
            cells.append({
                "layer_binding_count": count,
                "sample_count": len(samples[count]),
                "host_call_wall_ns": {
                    "p10": _percentile(wall_values, 0.1),
                    "median": medians[count]["wall_ns"],
                    "p90": _percentile(wall_values, 0.9),
                },
                "host_call_thread_cpu_ns": {
                    "p10": _percentile(cpu_values, 0.1),
                    "median": medians[count]["thread_cpu_ns"],
                    "p90": _percentile(cpu_values, 0.9),
                },
            })
        wall_intercept_ns, wall_slope_ns = _linear_fit([
            (count, medians[count]["wall_ns"])
            for count in sorted(FIT_CHAIN_LENGTHS)
        ])
        cpu_intercept_ns, cpu_slope_ns = _linear_fit([
            (count, medians[count]["thread_cpu_ns"])
            for count in sorted(FIT_CHAIN_LENGTHS)
        ])
        holdout_wall_ns = medians[512]["wall_ns"]
        holdout_cpu_ns = medians[512]["thread_cpu_ns"]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "validation": "pass",
            "hardware_id": HARDWARE_ID,
            "baseline_commit": arguments.baseline_commit,
            "runner": {"path": str(runner_path), "sha256": runner_sha256},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "cells": cells,
            "attention_metadata_layer_binding_candidate": {
                "status": "diagnostic_only_requires_source_occurrence_metadata",
                "mechanism": "attention_metadata_layer_binding_prepare",
                "fit_layer_binding_counts": sorted(FIT_CHAIN_LENGTHS),
                "holdout_layer_binding_count": 512,
                "wall_fixed_ns_per_metadata_group": wall_intercept_ns,
                "wall_ns_per_layer_binding": wall_slope_ns,
                "thread_cpu_fixed_ns_per_metadata_group": cpu_intercept_ns,
                "thread_cpu_ns_per_layer_binding": cpu_slope_ns,
                "holdout_wall_measured_ns": holdout_wall_ns,
                "holdout_wall_predicted_ns": (
                    wall_intercept_ns + wall_slope_ns * 512),
                "holdout_wall_signed_error_percent": 100.0 * (
                    (wall_intercept_ns + wall_slope_ns * 512)
                    / holdout_wall_ns - 1.0),
                "holdout_thread_cpu_measured_ns": holdout_cpu_ns,
                "holdout_thread_cpu_predicted_ns": (
                    cpu_intercept_ns + cpu_slope_ns * 512),
                "holdout_thread_cpu_signed_error_percent": 100.0 * (
                    (cpu_intercept_ns + cpu_slope_ns * 512)
                    / holdout_cpu_ns - 1.0),
            },
            "ended_at_utc": _utc_now(),
        }
        _write_json(arguments.output_dir / "summary.json", summary)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["completed_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
