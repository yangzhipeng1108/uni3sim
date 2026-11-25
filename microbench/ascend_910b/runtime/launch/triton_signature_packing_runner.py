#!/usr/bin/env python3
"""Measure generic Triton launcher cost for a source-static signature.

This runner deliberately uses a tiny copy kernel and generic tensors.  It does
not reproduce a Serving operator or consume Serving durations.  It measures
the generic Triton source-static signature and a generic three-output Torch
custom-op composite with the same mechanism descriptor.
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
import re
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
CHAIN_LENGTHS = (1, 8, 64, 512)
HOLDOUT_CHAIN_LENGTH = 2048
TRIALS = 20
WARMUP_REPEATS = 8
SCHEMA_VERSION = 2


@triton.jit
def _dynamic_11_kernel(
    input_ptr,
    output_ptr,
    unused_ptr_0,
    unused_ptr_1,
    unused_ptr_2,
    unused_ptr_3,
    unused_ptr_4,
    unused_ptr_5,
    element_count,
    positions_ptr,
    cache_ptr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def _qkv_static_signature_kernel(
    input_ptr,
    output_ptr,
    unused_ptr_0,
    unused_ptr_1,
    unused_ptr_2,
    unused_ptr_3,
    unused_ptr_4,
    unused_ptr_5,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    bias: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    is_partial_rope: tl.constexpr,
    num_vectorcore: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    qk_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    positions_ptr,
    cache_ptr,
    block_size: tl.constexpr,
):
    # All constexpr values remain explicit launcher parameters.  They also
    # appear in this compile-time expression so Triton retains the complete
    # specialization contract rather than a reduced unused-argument wrapper.
    static_zero = (
        q_hidden_size * 0 + kv_hidden_size * 0 + total_hidden_size * 0
        + eps * 0 + bias * 0 + head_dim * 0 + rope_dim * 0
        + half_rope_dim * 0 + is_partial_rope * 0 + num_vectorcore * 0
        + batch_size_per_iter_per_vec * 0 + qk_head_nums_per_iter_per_vec * 0
        + q_head_num * 0 + kv_head_num * 0 + qk_head_num_sum * 0
        + v_batch_size_per_iter_per_vec * 0
    )
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < (batch_size + static_zero)
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


@torch.library.custom_op(
    "midlayer_runtime_probe::three_output_triton_static_signature",
    mutates_args=(),
)
def _three_output_triton_static_signature(
    input_tensor: torch.Tensor,
    cache_tensor: torch.Tensor,
    positions_tensor: torch.Tensor,
    q_weight_tensor: torch.Tensor,
    k_weight_tensor: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias_tensor: torch.Tensor | None = None,
    k_bias_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generic eager composite: dispatch, three allocations, then one launch."""

    q_output = torch.empty_like(input_tensor)
    k_output = torch.empty_like(input_tensor)
    v_output = torch.empty_like(input_tensor)
    q_bias = q_weight_tensor if q_bias_tensor is None else q_bias_tensor
    k_bias = k_weight_tensor if k_bias_tensor is None else k_bias_tensor
    _qkv_static_signature_kernel[(32,)](
        input_tensor, q_output, k_output, v_output, q_weight_tensor, q_bias,
        k_weight_tensor, k_bias, input_tensor.numel(),
        q_hidden_size, kv_hidden_size, q_hidden_size + 2 * kv_hidden_size,
        eps, False, head_dim, head_dim // 2, head_dim // 4, False, 32,
        1, 1, 1, 1, 2, 1, positions_tensor, cache_tensor, block_size=128,
    )
    return q_output, k_output, v_output


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _append_json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("Triton signature fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if (not math.isfinite(intercept) or not math.isfinite(slope)
            or slope <= 0):
        raise RuntimeError("Triton signature fit is invalid")
    return intercept, slope


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    arguments = parser.parse_args()
    arguments.output_dir = _validate_output_dir(arguments.output_dir)
    if arguments.device_id < 0:
        parser.error("--device-id must be non-negative")
    if BASELINE_PATTERN.fullmatch(arguments.baseline_commit) is None:
        parser.error("--baseline-commit must be a full lowercase Git commit")
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    if (len(arguments.expected_source_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in arguments.expected_source_sha256)):
        parser.error("--expected-source-sha256 must be a lowercase SHA256")
    return arguments


def _state(device_id: int) -> dict[str, object]:
    device = f"npu:{device_id}"
    source = torch.arange(4096, dtype=torch.float32, device=device)
    return {
        "source": source,
        "destination": torch.empty_like(source),
        "auxiliary": torch.empty_like(source),
        "element_count": int(source.numel()),
    }


def _plain_three_output_triton_composite(state: dict[str, object]) -> None:
    source = state["source"]
    auxiliary = state["auxiliary"]
    element_count = state["element_count"]
    if not isinstance(element_count, int):
        raise AssertionError("element count must be an integer")
    q_output = torch.empty_like(source)
    k_output = torch.empty_like(source)
    v_output = torch.empty_like(source)
    _qkv_static_signature_kernel[(32,)](
        source, q_output, k_output, v_output, auxiliary, auxiliary,
        auxiliary, auxiliary, element_count,
        64, 32, 128, 1e-6, False, 64, 32, 16, False, 32, 1, 1, 1,
        1, 2, 1, auxiliary, auxiliary, block_size=128,
    )


def _launch(kind: str, state: dict[str, object]) -> None:
    source = state["source"]
    destination = state["destination"]
    auxiliary = state["auxiliary"]
    element_count = state["element_count"]
    if not isinstance(element_count, int):
        raise AssertionError("element count must be an integer")
    grid = (32,)
    if kind == "dynamic_11":
        _dynamic_11_kernel[grid](
            source, destination, auxiliary, auxiliary, auxiliary, auxiliary,
            auxiliary, auxiliary, element_count, auxiliary, auxiliary,
            block_size=128,
        )
        return
    if kind == "qkv_static_signature":
        _qkv_static_signature_kernel[grid](
            source, destination, auxiliary, auxiliary, auxiliary, auxiliary,
            auxiliary, auxiliary, element_count,
            64, 32, 128, 1e-6, False, 64, 32, 16, False, 32, 1, 1, 1,
            1, 2, 1, auxiliary, auxiliary, block_size=128,
        )
        return
    if kind == "plain_three_output_triton_composite":
        _plain_three_output_triton_composite(state)
        return
    if kind == "custom_op_three_output_triton_composite":
        torch.ops.midlayer_runtime_probe.three_output_triton_static_signature.default(
            source, auxiliary, auxiliary, auxiliary, auxiliary,
            64, 32, 64, 1e-6, None, None,
        )
        return
    raise ValueError(f"unknown signature kind: {kind!r}")


def _measure(kind: str, length: int, state: dict[str, object]) -> dict[str, int]:
    stream = torch.npu.current_stream()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record(stream)
    wall_start_ns = time.perf_counter_ns()
    thread_cpu_start_ns = time.thread_time_ns()
    for _ in range(length):
        _launch(kind, state)
    thread_cpu_end_ns = time.thread_time_ns()
    wall_end_ns = time.perf_counter_ns()
    end_event.record(stream)
    end_event.synchronize()
    return {
        "host_call_wall_ns": wall_end_ns - wall_start_ns,
        "host_call_thread_cpu_ns": thread_cpu_end_ns - thread_cpu_start_ns,
        "device_elapsed_ns": int(start_event.elapsed_time(end_event) * 1_000_000),
    }


def _fit_from_cells(cells: list[dict[str, object]], kind: str) -> dict[str, float]:
    selected = [cell for cell in cells if cell["signature_kind"] == kind]
    wall_intercept, wall_slope = _linear_fit([
        (int(cell["chain_length"]), float(cell["host_call_wall_ns"]["median"]))
        for cell in selected
        if int(cell["chain_length"]) in CHAIN_LENGTHS
    ])
    cpu_intercept, cpu_slope = _linear_fit([
        (int(cell["chain_length"]), float(cell["host_call_thread_cpu_ns"]["median"]))
        for cell in selected
        if int(cell["chain_length"]) in CHAIN_LENGTHS
    ])
    holdout = next(
        cell for cell in selected
        if int(cell["chain_length"]) == HOLDOUT_CHAIN_LENGTH)
    return {
        "wall_fixed_intercept_ns": wall_intercept,
        "wall_slope_ns_per_launch": wall_slope,
        "thread_cpu_fixed_intercept_ns": cpu_intercept,
        "thread_cpu_slope_ns_per_launch": cpu_slope,
        "holdout_wall_measured_ns": float(
            holdout["host_call_wall_ns"]["median"]),
        "holdout_wall_predicted_ns": wall_intercept + wall_slope * HOLDOUT_CHAIN_LENGTH,
        "holdout_thread_cpu_measured_ns": float(
            holdout["host_call_thread_cpu_ns"]["median"]),
        "holdout_thread_cpu_predicted_ns": (
            cpu_intercept + cpu_slope * HOLDOUT_CHAIN_LENGTH),
    }


def main() -> None:
    arguments = _parse_args()
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from audited snapshot")
    arguments.output_dir.mkdir(parents=True)
    manifest_path = arguments.output_dir / "manifest.json"
    trials_path = arguments.output_dir / "trials.jsonl"
    summary_path = arguments.output_dir / "summary.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "baseline_commit": arguments.baseline_commit,
        "source": {"path": str(source_path), "sha256": source_sha256},
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "container_name": arguments.container_name,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "physical_device": arguments.device_id,
        },
        "contract": {
            "parameter_hypothesis": (
                "generic_triton_constexpr_signature_and_three_output_"
                "custom_op_composite"),
            "measurement_boundary": (
                "Python eager entry through asynchronous return; generic "
                "tensors and no Serving operation"),
            "dynamic_argument_count": 11,
            "constexpr_argument_count": 16,
            "signature_order_matches": (
                "split_qkv_rmsnorm_rope_kernel dynamic/constexpr layout"),
        },
        "started_at": _utc_now(),
    }
    _write_json(manifest_path, manifest)
    state = _state(arguments.device_id)
    cells: list[dict[str, object]] = []
    for signature_kind in (
            "dynamic_11",
            "qkv_static_signature",
            "plain_three_output_triton_composite",
            "custom_op_three_output_triton_composite",
    ):
        for _ in range(WARMUP_REPEATS):
            _launch(signature_kind, state)
        torch.npu.synchronize()
        for chain_length in (*CHAIN_LENGTHS, HOLDOUT_CHAIN_LENGTH):
            samples = [_measure(signature_kind, chain_length, state)
                       for _ in range(TRIALS)]
            for trial_index, sample in enumerate(samples):
                _append_json_line(trials_path, {
                    "signature_kind": signature_kind,
                    "chain_length": chain_length,
                    "trial_index": trial_index,
                    **sample,
                })
            cells.append({
                "signature_kind": signature_kind,
                "chain_length": chain_length,
                "sample_count": len(samples),
                **{
                    metric: {
                        "median": statistics.median([
                            float(sample[metric]) for sample in samples]),
                        "p10": _percentile([
                            float(sample[metric]) for sample in samples], 0.10),
                        "p90": _percentile([
                            float(sample[metric]) for sample in samples], 0.90),
                    }
                    for metric in (
                        "host_call_wall_ns", "host_call_thread_cpu_ns",
                        "device_elapsed_ns",
                    )
                },
            })
    dynamic_fit = _fit_from_cells(cells, "dynamic_11")
    static_fit = _fit_from_cells(cells, "qkv_static_signature")
    plain_composite_fit = _fit_from_cells(
        cells, "plain_three_output_triton_composite")
    custom_op_composite_fit = _fit_from_cells(
        cells, "custom_op_three_output_triton_composite")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "validation": "pass",
        "mechanism": (
            "generic Triton constexpr signature specialization and packing"),
        "status": "diagnostic_only_requires_source_static_descriptor",
        "source_static_descriptor": {
            "source_kernel": "split_qkv_rmsnorm_rope_kernel",
            "dynamic_argument_count": 11,
            "constexpr_argument_count": 16,
        },
        "dynamic_11_fit": dynamic_fit,
        "qkv_static_signature_fit": static_fit,
        "plain_three_output_triton_composite_fit": plain_composite_fit,
        "custom_op_three_output_triton_composite_fit": custom_op_composite_fit,
        "constexpr_signature_increment": {
            "wall_ns_per_launch": (
                static_fit["wall_slope_ns_per_launch"]
                - dynamic_fit["wall_slope_ns_per_launch"]),
            "thread_cpu_ns_per_launch": (
                static_fit["thread_cpu_slope_ns_per_launch"]
                - dynamic_fit["thread_cpu_slope_ns_per_launch"]),
        },
        "custom_op_composite_increment": {
            "wall_ns_per_launch": (
                custom_op_composite_fit["wall_slope_ns_per_launch"]
                - plain_composite_fit["wall_slope_ns_per_launch"]),
            "thread_cpu_ns_per_launch": (
                custom_op_composite_fit["thread_cpu_slope_ns_per_launch"]
                - plain_composite_fit["thread_cpu_slope_ns_per_launch"]),
        },
        "cells": cells,
    }
    _write_json(summary_path, summary)
    manifest["status"] = "completed"
    manifest["completed_at"] = _utc_now()
    _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
