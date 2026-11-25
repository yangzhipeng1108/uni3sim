#!/usr/bin/env python3
"""Measure source-local ModelRunner outer preparation on the Ascend host.

The probe isolates the three host-only regions around ``NPUModelRunner``'s
forward: the input-index preparation, the attention-metadata dictionary
binding, and the post-forward logits-index preparation.  It uses synthetic
CPU arrays and opaque metadata objects; it does not instantiate a model,
execute an operator, consume Serving traces, or read a Serving duration.

The source occurrence contract is intentionally separate from this runner:
production places one event at the token-embedding boundary for input and
metadata preparation and one event at the LM-head boundary for logits
preparation.  The source snapshot identity is recorded in the output so a
profile cannot be silently reused after the vLLM implementation changes.
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

import numpy as np
import torch


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)
SCHEMA_VERSION = 1
REQUEST_COUNTS = (1, 8, 32, 64)
TOKEN_COUNTS = (1, 8, 64, 512)
LAYER_COUNTS = (1, 8, 32, 48)
FIT_LAYER_COUNTS = frozenset((1, 8, 32))
TRIALS = 40
WARMUP_REPEATS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _median(values: list[int]) -> float:
    if not values:
        raise ValueError("median requires samples")
    return float(statistics.median(values))


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires samples")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("linear fit requires at least two points")
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise ValueError("linear fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        raise ValueError("linear fit is invalid")
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
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    arguments = parser.parse_args()
    if (len(arguments.baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.baseline_commit)):
        parser.error("--baseline-commit must be a lowercase full Git SHA")
    if (len(arguments.expected_source_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in arguments.expected_source_sha256)):
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    arguments.output_dir = _validate_output_dir(arguments.output_dir)
    arguments.source_snapshot = arguments.source_snapshot.resolve()
    if not arguments.source_snapshot.is_file():
        parser.error("--source-snapshot must name a source file")
    return arguments


def _input_prepare(request_count: int, token_count: int) -> tuple[int, int]:
    """Mirror the source's CPU index arithmetic without device/model work."""

    scheduled = np.full(
        request_count, token_count // request_count, dtype=np.int32)
    scheduled[: token_count % request_count] += 1
    req_indices = np.repeat(np.arange(request_count, dtype=np.int32), scheduled)
    arange = np.arange(token_count, dtype=np.int32)
    computed = np.arange(request_count, dtype=np.int32) * 17
    positions = computed[req_indices] + arange
    token_indices = positions + req_indices * max(token_count, 1)
    source_length = (
        token_count * max(request_count, 1)
        + max(request_count, 1) * 17
        + token_count
        + 1
    )
    source = torch.arange(source_length, dtype=torch.int32)
    index = torch.from_numpy(token_indices)
    selected = torch.index_select(source, 0, index)
    if selected.numel() != token_count:
        raise RuntimeError("input preparation changed token count")
    return int(selected[0]), int(selected[-1])


def _forward_metadata(layer_count: int, request_count: int) -> int:
    """Mirror ``_build_attention_metadata``'s source binding loop."""

    metadata = {
        "num_reqs": request_count,
        "num_actual_tokens": request_count,
        "causal": True,
    }
    layer_names = tuple(f"model.layers.{index}.self_attn" for index in range(layer_count))
    attn_metadata_dict: dict[str, object] = {}
    for layer_name in layer_names:
        attn_metadata_dict[layer_name] = metadata
    if len(attn_metadata_dict) != layer_count:
        raise RuntimeError("metadata binding lost source layers")
    return len(attn_metadata_dict)


def _logits_prepare(request_count: int, token_count: int) -> int:
    """Mirror the source's post-forward sample-index selection."""

    hidden = torch.arange(max(token_count, request_count), dtype=torch.float32)
    logits_indices = torch.arange(request_count, dtype=torch.int64)
    logits_indices = torch.remainder(logits_indices, hidden.numel())
    sample_hidden = hidden[logits_indices]
    if sample_hidden.numel() != request_count:
        raise RuntimeError("logits preparation changed request count")
    return int(sample_hidden.numel())


def _measure(
    mechanism: str,
    dimensions: tuple[int, int],
    trials_path: Path,
    sequence_start: int,
) -> tuple[dict[str, object], int]:
    first, second = dimensions
    if mechanism == "model_runner_input_prepare":
        operation = lambda: _input_prepare(first, second)
        dimension_names = ("request_count", "token_count")
    elif mechanism == "model_runner_forward_metadata_prepare":
        operation = lambda: _forward_metadata(first, second)
        dimension_names = ("layer_count", "request_count")
    elif mechanism == "model_runner_logits_prepare":
        operation = lambda: _logits_prepare(first, second)
        dimension_names = ("request_count", "token_count")
    else:
        raise ValueError(f"unknown mechanism: {mechanism!r}")
    for _ in range(WARMUP_REPEATS):
        operation()
    wall_values: list[int] = []
    cpu_values: list[int] = []
    sequence = sequence_start
    for trial in range(TRIALS):
        cpu_start = time.thread_time_ns()
        wall_start = time.perf_counter_ns()
        operation()
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        wall_ns = wall_end - wall_start
        cpu_ns = cpu_end - cpu_start
        if wall_ns <= 0 or cpu_ns <= 0:
            raise RuntimeError("ModelRunner timing must be positive")
        sequence += 1
        row = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "trial": trial + 1,
            "mechanism": mechanism,
            dimension_names[0]: first,
            dimension_names[1]: second,
            "host_call_wall_ns": wall_ns,
            "host_call_thread_cpu_ns": cpu_ns,
            "host_call_non_cpu_wall_ns": wall_ns - cpu_ns,
        }
        _append_json_line(trials_path, row)
        wall_values.append(wall_ns)
        cpu_values.append(cpu_ns)
    return ({
        "mechanism": mechanism,
        dimension_names[0]: first,
        dimension_names[1]: second,
        "sample_count": TRIALS,
        "host_call_wall_ns": {
            "p10": _percentile(wall_values, 0.10),
            "median": _median(wall_values),
            "p90": _percentile(wall_values, 0.90),
        },
        "host_call_thread_cpu_ns": {
            "p10": _percentile(cpu_values, 0.10),
            "median": _median(cpu_values),
            "p90": _percentile(cpu_values, 0.90),
        },
    }, sequence)


def main() -> None:
    arguments = _arguments()
    runner_path = Path(__file__).resolve()
    source_sha256 = _sha256(arguments.source_snapshot)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("source snapshot SHA256 differs from audited identity")
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True)
    trials_path = output_dir / "trials.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, {
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
        "source_snapshot": {
            "path": str(arguments.source_snapshot),
            "sha256": source_sha256,
        },
        "runner": {
            "path": str(runner_path),
            "sha256": _sha256(runner_path),
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    cells: list[dict[str, object]] = []
    sequence = 0
    for mechanism, dimensions in (
        ("model_runner_input_prepare", tuple((r, t) for r in REQUEST_COUNTS for t in TOKEN_COUNTS)),
        ("model_runner_forward_metadata_prepare", tuple((l, r) for l in LAYER_COUNTS for r in REQUEST_COUNTS)),
        ("model_runner_logits_prepare", tuple((r, t) for r in REQUEST_COUNTS for t in TOKEN_COUNTS)),
    ):
        for dimension in dimensions:
            cell, sequence = _measure(mechanism, dimension, trials_path, sequence)
            cells.append(cell)
    metadata_cells = [
        cell for cell in cells
        if cell["mechanism"] == "model_runner_forward_metadata_prepare"
    ]
    fit_points = [
        (int(cell["layer_count"]), float(cell["host_call_wall_ns"]["median"]))
        for cell in metadata_cells
        if int(cell["layer_count"]) in FIT_LAYER_COUNTS
        and int(cell["request_count"]) == 32
    ]
    metadata_intercept_ns, metadata_slope_ns = _linear_fit(fit_points)
    # Production uses one source outer event at the fixed Qwen/Llama decoder
    # boundary.  The 32-layer point is the registered reference; the 48-layer
    # cell remains a holdout for detecting accidental architecture fitting.
    reference_metadata = next(
        cell for cell in metadata_cells
        if cell["layer_count"] == 32 and cell["request_count"] == 32)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "kind": "ascend_model_runner_outer_prepare_microbenchmark",
        "hardware_id": HARDWARE_ID,
        "baseline_commit": arguments.baseline_commit,
        "source_snapshot": {
            "path": str(arguments.source_snapshot),
            "sha256": source_sha256,
            "symbols": [
                "NPUModelRunner._prepare_inputs",
                "NPUModelRunner._build_attention_metadata",
                "NPUModelRunner.execute_model",
            ],
        },
        "synthetic_contract": {
            "mechanisms": [
                "model_runner_input_prepare",
                "model_runner_forward_metadata_prepare",
                "model_runner_logits_prepare",
            ],
            "request_counts": list(REQUEST_COUNTS),
            "token_counts": list(TOKEN_COUNTS),
            "layer_counts": list(LAYER_COUNTS),
            "metadata_fit_layer_counts": sorted(FIT_LAYER_COUNTS),
            "metadata_holdout_layer_count": 48,
            "production_or_evaluation_duration_consumed": False,
            "model_or_operator_executed": False,
            "npu_access": "none",
            "timers": ["time.perf_counter_ns", "time.thread_time_ns"],
        },
        "cells": cells,
        "production_reference": {
            "model_runner_forward_metadata_prepare": {
                "layer_count": 32,
                "request_count": 32,
                "wall_ns_per_source_event": reference_metadata[
                    "host_call_wall_ns"]["median"],
                "fit_intercept_ns": metadata_intercept_ns,
                "fit_slope_ns_per_layer": metadata_slope_ns,
            },
            "model_runner_input_prepare": {
                "wall_ns_per_source_event": next(
                    cell["host_call_wall_ns"]["median"] for cell in cells
                    if cell["mechanism"] == "model_runner_input_prepare"
                    and cell["request_count"] == 32
                    and cell["token_count"] == 512),
            },
            "model_runner_logits_prepare": {
                "wall_ns_per_source_event": next(
                    cell["host_call_wall_ns"]["median"] for cell in cells
                    if cell["mechanism"] == "model_runner_logits_prepare"
                    and cell["request_count"] == 32
                    and cell["token_count"] == 512),
            },
        },
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(manifest_path, {
        **json.loads(manifest_path.read_text()),
        "status": "pass",
        "ended_at_utc": summary["ended_at_utc"],
        "summary_path": str(output_dir / "summary.json"),
    })


if __name__ == "__main__":
    main()
