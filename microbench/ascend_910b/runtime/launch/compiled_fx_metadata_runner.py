#!/usr/bin/env python3
"""Measure generic FX metadata-node and Python-container Host service on Ascend.

The probes use only NPU tensor metadata operations and a Python tuple.  They
do not reproduce a Serving operator, model graph, Serving shape, or duration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)
CHAIN_LENGTHS = (1, 8, 64, 512)
FIT_CHAIN_LENGTHS = frozenset((1, 8, 64))
TRIALS = 16
WARMUP_REPEATS = 4
SCHEMA_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--baseline-commit", type=str, required=True)
    return parser.parse_args()


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


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires samples")
    return float(statistics.median(values))


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("metadata-node fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if (not math.isfinite(intercept) or not math.isfinite(slope)
            or slope <= 0):
        raise RuntimeError("metadata-node fit is invalid")
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


def _view_metadata_chain(source: torch.Tensor, count: int) -> torch.Tensor:
    value = source
    for index in range(count):
        if index % 2:
            value = torch.ops.aten.view.default(value, [256, 256])
        else:
            value = torch.ops.aten.view.default(value, [65536])
    return value


def _reshape_metadata_chain(source: torch.Tensor, count: int) -> torch.Tensor:
    """Exercise the exact ``aten.reshape`` metadata op in the exported FX code."""

    value = source
    for index in range(count):
        if index % 2:
            value = torch.ops.aten.reshape.default(value, [256, 256])
        else:
            value = torch.ops.aten.reshape.default(value, [65536])
    return value


def _empty_metadata_chain(source: torch.Tensor, count: int) -> torch.Tensor:
    """Exercise the exported FX graph's ``aten.empty.memory_format`` node."""

    value = source
    for _ in range(count):
        value = torch.ops.aten.empty.memory_format(
            [256, 256],
            dtype=source.dtype,
            device=source.device,
            pin_memory=False,
        )
    return value


def _tuple_getitem_chain(source: torch.Tensor, count: int) -> torch.Tensor:
    values = (source, source)
    value = source
    for index in range(count):
        value = values[index % 2]
    return value


def _measure(
    mechanism: str,
    source: torch.Tensor,
    count: int,
    trials_path: Path,
) -> dict[str, object]:
    if mechanism == "fx_aten_view_metadata":
        operation = _view_metadata_chain
    elif mechanism == "fx_aten_reshape_metadata":
        operation = _reshape_metadata_chain
    elif mechanism == "fx_aten_empty_metadata":
        operation = _empty_metadata_chain
    elif mechanism == "fx_python_tuple_getitem":
        operation = _tuple_getitem_chain
    else:
        raise ValueError(f"unsupported metadata mechanism: {mechanism!r}")
    for _ in range(WARMUP_REPEATS):
        result = operation(source, count)
        if result.device != source.device:
            raise RuntimeError("metadata operation changed device")
    torch.npu.synchronize()
    wall_values = []
    cpu_values = []
    for trial_index in range(TRIALS):
        cpu_start = time.thread_time_ns()
        wall_start = time.perf_counter_ns()
        result = operation(source, count)
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        if result.device != source.device:
            raise RuntimeError("metadata operation changed device")
        wall_ns = wall_end - wall_start
        cpu_ns = cpu_end - cpu_start
        if wall_ns <= 0 or cpu_ns <= 0:
            raise RuntimeError("metadata-node timing must be positive")
        _append_json_line(trials_path, {
            "mechanism": mechanism,
            "chain_length": count,
            "trial_index": trial_index,
            "host_call_wall_ns": wall_ns,
            "host_call_thread_cpu_ns": cpu_ns,
        })
        wall_values.append(float(wall_ns))
        cpu_values.append(float(cpu_ns))
    return {
        "mechanism": mechanism,
        "chain_length": count,
        "sample_count": TRIALS,
        "host_call_wall_ns": {"median": _median(wall_values)},
        "host_call_thread_cpu_ns": {"median": _median(cpu_values)},
    }


def _fit(cells: list[dict[str, object]], mechanism: str) -> dict[str, float]:
    measured = {
        cell["chain_length"]: cell
        for cell in cells
        if cell["mechanism"] == mechanism
    }
    if set(measured) != set(CHAIN_LENGTHS):
        raise RuntimeError("metadata-node cells do not cover the declared sweep")
    fit_points = [
        (count, float(measured[count]["host_call_wall_ns"]["median"]))
        for count in CHAIN_LENGTHS
        if count in FIT_CHAIN_LENGTHS
    ]
    intercept, slope = _linear_fit(fit_points)
    holdout = measured[512]["host_call_wall_ns"]["median"]
    predicted = intercept + slope * 512
    return {
        "wall_intercept_ns": intercept,
        "wall_slope_ns_per_node": slope,
        "holdout_wall_measured_ns": float(holdout),
        "holdout_wall_predicted_ns": predicted,
        "holdout_wall_signed_error_percent": (
            (predicted - float(holdout)) / float(holdout) * 100.0
        ),
    }


def main() -> None:
    arguments = _arguments()
    if isinstance(arguments.device_id, bool) or arguments.device_id < 0:
        raise ValueError("--device-id must be a non-negative integer")
    if (len(arguments.baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.baseline_commit)):
        raise ValueError("--baseline-commit must be a lowercase full Git SHA")
    output_dir = _validate_output_dir(arguments.output_dir)
    output_dir.mkdir(parents=True)
    trials_path = output_dir / "trials.jsonl"
    source = torch.empty((256, 256), dtype=torch.float32,
                         device=f"npu:{arguments.device_id}")
    torch.npu.synchronize()
    cells = []
    for mechanism in (
            "fx_aten_view_metadata",
            "fx_aten_reshape_metadata",
            "fx_aten_empty_metadata",
            "fx_python_tuple_getitem"):
        for count in CHAIN_LENGTHS:
            cells.append(_measure(mechanism, source, count, trials_path))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ascend_compiled_fx_metadata_microbenchmark",
        "hardware_id": HARDWARE_ID,
        "baseline_commit": arguments.baseline_commit,
        "environment": {
            "device_id": arguments.device_id,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "started_at_utc": _utc_now(),
        "synthetic_contract": {
            "source_shape": [256, 256],
            "mechanisms": [
                "fx_aten_view_metadata",
                "fx_aten_reshape_metadata",
                "fx_aten_empty_metadata",
                "fx_python_tuple_getitem",
            ],
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": sorted(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": 512,
            "production_or_evaluation_duration_consumed": False,
        },
        "cells": cells,
        "fit_by_mechanism": {
            mechanism: _fit(cells, mechanism)
            for mechanism in (
                "fx_aten_view_metadata",
                "fx_aten_reshape_metadata",
                "fx_aten_empty_metadata",
                "fx_python_tuple_getitem")
        },
        "ended_at_utc": _utc_now(),
    }
    _write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
