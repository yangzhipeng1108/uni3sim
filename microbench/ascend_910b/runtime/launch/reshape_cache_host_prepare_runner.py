#!/usr/bin/env python3
"""Measure queued-device impact on the generic Ascend reshape-cache API.

This isolates the Host API entry through ``DeviceOperator.reshape_and_cache``
with deliberately small, non-Serving tensors.  The optional predecessor is a
generic preallocated BF16 matmul chain.  Its enqueue and execution are outside
the timed interval: only the reshape-cache API call to asynchronous return is
measured.
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
import traceback

import torch
import torch_npu
import vllm_ascend.ops  # noqa: F401  # Register the actual custom operation.
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import enable_custom_op


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
CHAIN_LENGTHS = (1, 8, 64, 256, 1024)
FIT_CHAIN_LENGTHS = frozenset((1, 8, 64, 256))
HOLDOUT_CHAIN_LENGTH = 1024
TRIALS = 10
WARMUP_REPEATS = 4
PREDECESSOR_MM_DIMENSION = 1024
SCHEMA_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _append_json_line(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _linear_fit(points):
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("reshape-cache Host-prepare fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        raise RuntimeError("reshape-cache Host-prepare fit is invalid")
    return intercept, slope


def _validate_output_dir(path):
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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--predecessor-mm-count", type=int, required=True)
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
    if arguments.predecessor_mm_count < 0:
        parser.error("--predecessor-mm-count must be non-negative")
    return arguments


def _make_state(device_id):
    device = f"npu:{device_id}"
    key = torch.randn((16, 2, 16), dtype=torch.bfloat16, device=device)
    value = torch.randn((16, 2, 16), dtype=torch.bfloat16, device=device)
    key_cache = torch.zeros((1, 128, 2, 16), dtype=torch.bfloat16, device=device)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(16, dtype=torch.int32, device=device)
    predecessor_left = torch.full(
        (PREDECESSOR_MM_DIMENSION, PREDECESSOR_MM_DIMENSION),
        0.25,
        dtype=torch.bfloat16,
        device=device,
    )
    predecessor_right = torch.full_like(predecessor_left, 0.5)
    predecessor_output = torch.empty_like(predecessor_left)
    return {
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "slot_mapping": slot_mapping,
        "predecessor_left": predecessor_left,
        "predecessor_right": predecessor_right,
        "predecessor_output": predecessor_output,
    }


def _enqueue_predecessor(state, count):
    for _ in range(count):
        torch.mm(
            state["predecessor_left"],
            state["predecessor_right"],
            out=state["predecessor_output"],
        )


def _reshape_cache(state):
    DeviceOperator.reshape_and_cache(
        state["key"],
        state["value"],
        state["key_cache"],
        state["value_cache"],
        state["slot_mapping"],
    )


def _measure_chain(state, length, predecessor_mm_count):
    host_wall_ns = 0
    host_thread_cpu_ns = 0
    for _ in range(length):
        _enqueue_predecessor(state, predecessor_mm_count)
        wall_start_ns = time.perf_counter_ns()
        thread_cpu_start_ns = time.thread_time_ns()
        _reshape_cache(state)
        thread_cpu_end_ns = time.thread_time_ns()
        wall_end_ns = time.perf_counter_ns()
        host_wall_ns += wall_end_ns - wall_start_ns
        host_thread_cpu_ns += thread_cpu_end_ns - thread_cpu_start_ns
    torch.npu.synchronize()
    return {
        "host_api_wall_ns": host_wall_ns,
        "host_api_thread_cpu_ns": host_thread_cpu_ns,
        "host_api_non_cpu_wall_ns": host_wall_ns - host_thread_cpu_ns,
    }


def main():
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
        "status": "initializing",
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
                "generic_cann_custom_op_prepare_when_device_queue_is_busy"
            ),
            "operation": "DeviceOperator.reshape_and_cache",
            "synthetic_tensor_shapes": {
                "key_and_value": [16, 2, 16],
                "cache": [1, 128, 2, 16],
                "slot_mapping": [16],
            },
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": sorted(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": HOLDOUT_CHAIN_LENGTH,
            "trials_per_cell": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "predecessor": {
                "kind": "generic_bf16_torch_mm",
                "dimension": PREDECESSOR_MM_DIMENSION,
                "count": arguments.predecessor_mm_count,
                "timing": "queued_before_each_api_and_excluded_from_window",
            },
            "timed_boundary": "reshape_cache_api_to_asynchronous_return",
            "device_synchronization": "outside_timed_window",
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
    }
    _write_json(manifest_path, manifest)
    try:
        NPUPlatform.import_kernels()
        if not enable_custom_op():
            raise RuntimeError("vLLM-Ascend custom-op registration failed")
        torch.npu.set_device(arguments.device_id)
        state = _make_state(arguments.device_id)
        for _ in range(WARMUP_REPEATS):
            _enqueue_predecessor(state, arguments.predecessor_mm_count)
            _reshape_cache(state)
        torch.npu.synchronize()
        samples = {length: [] for length in CHAIN_LENGTHS}
        sequence = 0
        for trial in range(TRIALS):
            ordered = CHAIN_LENGTHS[trial % len(CHAIN_LENGTHS):] + CHAIN_LENGTHS[:trial % len(CHAIN_LENGTHS)]
            if trial % 2:
                ordered = tuple(reversed(ordered))
            for order, length in enumerate(ordered, start=1):
                sequence += 1
                timing = _measure_chain(
                    state, length, arguments.predecessor_mm_count)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "sequence": sequence,
                    "trial": trial + 1,
                    "order": order,
                    "chain_length": length,
                    **timing,
                }
                samples[length].append(row)
                _append_json_line(trials_path, row)
        if not torch.isfinite(state["key_cache"]).all().item():
            raise RuntimeError("reshape-cache output validation failed")
        medians = {
            length: {
                "wall_ns": statistics.median(
                    sample["host_api_wall_ns"] for sample in samples[length]),
                "thread_cpu_ns": statistics.median(
                    sample["host_api_thread_cpu_ns"] for sample in samples[length]),
            }
            for length in CHAIN_LENGTHS
        }
        wall_intercept, wall_slope = _linear_fit(tuple(
            (length, medians[length]["wall_ns"])
            for length in sorted(FIT_CHAIN_LENGTHS)
        ))
        cpu_intercept, cpu_slope = _linear_fit(tuple(
            (length, medians[length]["thread_cpu_ns"])
            for length in sorted(FIT_CHAIN_LENGTHS)
        ))
        holdout = medians[HOLDOUT_CHAIN_LENGTH]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "predecessor": manifest["contract"]["predecessor"],
            "measurement_boundary": manifest["contract"]["timed_boundary"],
            "fit": {
                "wall_intercept_ns": wall_intercept,
                "wall_slope_ns_per_call": wall_slope,
                "thread_cpu_intercept_ns": cpu_intercept,
                "thread_cpu_slope_ns_per_call": cpu_slope,
                "holdout_wall_measured_ns": holdout["wall_ns"],
                "holdout_wall_predicted_ns": (
                    wall_intercept + wall_slope * HOLDOUT_CHAIN_LENGTH),
                "holdout_wall_signed_error_percent": 100.0 * (
                    (wall_intercept + wall_slope * HOLDOUT_CHAIN_LENGTH)
                    / holdout["wall_ns"] - 1.0),
                "holdout_thread_cpu_measured_ns": holdout["thread_cpu_ns"],
                "holdout_thread_cpu_predicted_ns": (
                    cpu_intercept + cpu_slope * HOLDOUT_CHAIN_LENGTH),
                "holdout_thread_cpu_signed_error_percent": 100.0 * (
                    (cpu_intercept + cpu_slope * HOLDOUT_CHAIN_LENGTH)
                    / holdout["thread_cpu_ns"] - 1.0),
            },
            "cells": [
                {
                    "chain_length": length,
                    "sample_count": len(samples[length]),
                    "host_api_wall_ns": {
                        "p10": _percentile(
                            [sample["host_api_wall_ns"]
                             for sample in samples[length]], 0.10),
                        "median": medians[length]["wall_ns"],
                        "p90": _percentile(
                            [sample["host_api_wall_ns"]
                             for sample in samples[length]], 0.90),
                    },
                    "host_api_thread_cpu_ns": {
                        "p10": _percentile(
                            [sample["host_api_thread_cpu_ns"]
                             for sample in samples[length]], 0.10),
                        "median": medians[length]["thread_cpu_ns"],
                        "p90": _percentile(
                            [sample["host_api_thread_cpu_ns"]
                             for sample in samples[length]], 0.90),
                    },
                }
                for length in CHAIN_LENGTHS
            ],
        }
        _write_json(summary_path, summary)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
