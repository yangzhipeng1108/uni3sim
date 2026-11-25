#!/usr/bin/env python3
"""Measure generic HCCL Host submission without charging Device completion."""

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
import torch.distributed as dist
import torch_npu


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
OPERATIONS = ("all_reduce", "all_gather")
CALL_STACKS = ("torch_distributed", "vllm_tensor_parallel")
MESSAGE_BYTES = (4096, 65536, 1048576, 4194304)
CALL_COUNTS = (1, 8, 32, 128, 256)
FIT_CALL_COUNTS = frozenset((1, 8, 32, 128))
HOLDOUT_CALL_COUNT = 256
TRIALS = 10
WARMUP_REPEATS = 3
PREDECESSOR_MM_DIMENSION = 1024
SCHEMA_VERSION = 3


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
        raise RuntimeError("collective Host-submit fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
        raise RuntimeError("collective Host-submit fit is invalid")
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
    return resolved


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--call-stack", required=True, choices=CALL_STACKS)
    parser.add_argument("--predecessor-mm-count", type=int, required=True)
    arguments = parser.parse_args()
    arguments.output_dir = _validate_output_dir(arguments.output_dir)
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


def _buffers(operation, message_bytes, call_count, rank, device):
    element_count = message_bytes // 2
    inputs = tuple(
        torch.full(
            (element_count,), rank + 1, dtype=torch.bfloat16, device=device)
        for _ in range(call_count)
    )
    if operation == "all_gather":
        outputs = tuple(
            torch.empty(
                (dist.get_world_size() * element_count,),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(call_count)
        )
    elif operation == "all_reduce":
        outputs = ()
    else:
        raise ValueError(f"unknown collective operation: {operation!r}")
    return inputs, outputs


def _predecessor_buffers(device):
    """Allocate a generic queued-compute control independently of collectives."""

    left = torch.full(
        (PREDECESSOR_MM_DIMENSION, PREDECESSOR_MM_DIMENSION),
        0.25,
        dtype=torch.bfloat16,
        device=device,
    )
    right = torch.full(
        (PREDECESSOR_MM_DIMENSION, PREDECESSOR_MM_DIMENSION),
        0.5,
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.empty_like(left)
    return left, right, output


def _enqueue_predecessor_mm(count, buffers):
    """Queue generic BF16 GEMMs without waiting or charging their duration."""

    left, right, output = buffers
    for _ in range(count):
        torch.mm(left, right, out=output)


def _reset_and_synchronize(operation, inputs, rank):
    if operation == "all_reduce":
        for tensor in inputs:
            tensor.fill_(rank + 1)
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.synchronize()


def _submit_chain(operation, inputs, outputs, call_stack, vllm_tp_group):
    handles = []
    if call_stack == "torch_distributed":
        if operation == "all_reduce":
            for tensor in inputs:
                handles.append(dist.all_reduce(
                    tensor, op=dist.ReduceOp.SUM, async_op=True))
        elif operation == "all_gather":
            for output, tensor in zip(outputs, inputs, strict=True):
                handles.append(dist.all_gather_into_tensor(
                    output, tensor, async_op=True))
        else:
            raise ValueError(f"unknown collective operation: {operation!r}")
        if any(handle is None for handle in handles):
            raise RuntimeError(
                "asynchronous collective did not return a Work handle")
    elif call_stack == "vllm_tensor_parallel":
        if vllm_tp_group is None:
            raise RuntimeError("vLLM tensor-parallel group is unavailable")
        if operation == "all_reduce":
            handles.extend(vllm_tp_group.all_reduce(tensor) for tensor in inputs)
        elif operation == "all_gather":
            handles.extend(
                vllm_tp_group.all_gather(tensor, dim=0) for tensor in inputs)
        else:
            raise ValueError(f"unknown collective operation: {operation!r}")
        if any(not isinstance(handle, torch.Tensor) for handle in handles):
            raise RuntimeError("vLLM collective did not return a Tensor")
    else:
        raise ValueError(f"unknown collective call stack: {call_stack!r}")
    return handles


def _drain(handles, call_stack):
    drain_wall_start_ns = time.perf_counter_ns()
    drain_cpu_start_ns = time.thread_time_ns()
    if call_stack == "torch_distributed":
        for handle in handles:
            handle.wait()
    elif call_stack != "vllm_tensor_parallel":
        raise ValueError(f"unknown collective call stack: {call_stack!r}")
    torch.npu.synchronize()
    drain_cpu_end_ns = time.thread_time_ns()
    drain_wall_end_ns = time.perf_counter_ns()
    return {
        "completion_wall_ns": drain_wall_end_ns - drain_wall_start_ns,
        "completion_thread_cpu_ns": drain_cpu_end_ns - drain_cpu_start_ns,
    }


def _validate_result(
        operation, inputs, outputs, handles, world_size, call_stack):
    result_tensors = (
        inputs if call_stack == "torch_distributed" and operation == "all_reduce"
        else outputs if call_stack == "torch_distributed"
        else handles
    )
    if operation == "all_reduce":
        expected = world_size * (world_size + 1) // 2
        if result_tensors[0][0].item() != expected:
            raise RuntimeError("all_reduce validation failed")
    elif operation == "all_gather":
        element_count = inputs[0].numel()
        for source_rank in range(world_size):
            if (result_tensors[0][source_rank * element_count].item()
                    != source_rank + 1):
                raise RuntimeError("all_gather validation failed")
    else:
        raise ValueError(f"unknown collective operation: {operation!r}")


def _measure(operation, inputs, outputs, call_stack, vllm_tp_group):
    wall_start_ns = time.perf_counter_ns()
    thread_cpu_start_ns = time.thread_time_ns()
    handles = _submit_chain(
        operation, inputs, outputs, call_stack, vllm_tp_group)
    thread_cpu_end_ns = time.thread_time_ns()
    wall_end_ns = time.perf_counter_ns()
    timing = {
        "host_submit_wall_ns": wall_end_ns - wall_start_ns,
        "host_submit_thread_cpu_ns": thread_cpu_end_ns - thread_cpu_start_ns,
        "host_submit_non_cpu_wall_ns": (
            (wall_end_ns - wall_start_ns)
            - (thread_cpu_end_ns - thread_cpu_start_ns)
        ),
    }
    timing.update(_drain(handles, call_stack))
    return timing, handles


def _cell_summary(samples):
    result = {}
    for field in (
            "host_submit_wall_ns", "host_submit_thread_cpu_ns",
            "host_submit_non_cpu_wall_ns", "completion_wall_ns",
            "completion_thread_cpu_ns"):
        values = [float(sample[field]) for sample in samples]
        result[field] = {
            "min": min(values),
            "p10": _percentile(values, 0.10),
            "median": statistics.median(values),
            "p90": _percentile(values, 0.90),
            "max": max(values),
        }
    return result


def main():
    arguments = _parse_args()
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from audited snapshot")
    if not dist.is_hccl_available():
        raise RuntimeError("HCCL backend is unavailable")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("Host-submit experiment requires exactly two ranks")
    visible_devices = tuple(
        int(value) for value in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    )
    if len(visible_devices) != world_size or len(set(visible_devices)) != world_size:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain two unique devices")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.output_dir / f"manifest-rank-{rank}.json"
    trials_path = arguments.output_dir / f"trials-rank-{rank}.jsonl"
    summary_path = arguments.output_dir / f"summary-rank-{rank}.json"
    if any(path.exists() for path in (manifest_path, trials_path, summary_path)):
        raise FileExistsError("rank output already exists")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "baseline_commit": arguments.baseline_commit,
        "source": {"path": str(source_path), "sha256": source_sha256},
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "container_name": arguments.container_name,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "physical_device": visible_devices[local_rank],
        },
        "contract": {
            "parameter_hypothesis": (
                "generic_collective_submit_when_device_queue_is_busy_per_call"
            ),
            "call_stack": arguments.call_stack,
            "operations": list(OPERATIONS),
            "message_bytes": list(MESSAGE_BYTES),
            "call_counts": list(CALL_COUNTS),
            "fit_call_counts": sorted(FIT_CALL_COUNTS),
            "holdout_call_count": HOLDOUT_CALL_COUNT,
            "trials_per_cell": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "predecessor": {
                "kind": "generic_bf16_torch_mm",
                "dimension": PREDECESSOR_MM_DIMENSION,
                "count": arguments.predecessor_mm_count,
                "timing": "queued_before_collective_and_excluded_from_window",
            },
            "host_wall_clock": "time.perf_counter_ns",
            "host_thread_cpu_clock": "time.thread_time_ns",
            "timed_boundary": "async_collective_api_calls_only",
            "completion_wait_and_device_sync_excluded": True,
            "aggregation": "per-operation-message-size median-count linear fit",
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
    }
    _write_json(manifest_path, manifest)
    vllm_tp_group = None
    try:
        torch.npu.set_device(local_rank)
        vllm_tp_group = None
        if arguments.call_stack == "torch_distributed":
            dist.init_process_group(backend="hccl")
        else:
            from vllm.distributed.parallel_state import (
                init_model_parallel_group,
            )

            dist.init_process_group(backend="hccl")
            vllm_tp_group = init_model_parallel_group(
                [list(range(world_size))],
                local_rank,
                backend="hccl",
                use_message_queue_broadcaster=False,
                group_name="tp_host_submit_microbench",
            )
        device = f"npu:{local_rank}"
        cells = tuple(
            (operation, message_bytes, call_count)
            for operation in OPERATIONS
            for message_bytes in MESSAGE_BYTES
            for call_count in CALL_COUNTS
        )
        buffers = {
            cell: _buffers(cell[0], cell[1], cell[2], rank, device)
            for cell in cells
        }
        predecessor_buffers = _predecessor_buffers(device)
        for cell in cells:
            operation, _message_bytes, _call_count = cell
            inputs, outputs = buffers[cell]
            for _ in range(WARMUP_REPEATS):
                _reset_and_synchronize(operation, inputs, rank)
                _enqueue_predecessor_mm(
                    arguments.predecessor_mm_count, predecessor_buffers)
                handles = _submit_chain(
                    operation, inputs, outputs,
                    arguments.call_stack, vllm_tp_group)
                _drain(handles, arguments.call_stack)
            _validate_result(
                operation, inputs, outputs, handles, world_size,
                arguments.call_stack)

        samples = {cell: [] for cell in cells}
        sequence = 0
        for trial in range(TRIALS):
            rotation = trial % len(cells)
            ordered = cells[rotation:] + cells[:rotation]
            if trial % 2:
                ordered = tuple(reversed(ordered))
            for order, cell in enumerate(ordered, start=1):
                operation, message_bytes, call_count = cell
                inputs, outputs = buffers[cell]
                _reset_and_synchronize(operation, inputs, rank)
                _enqueue_predecessor_mm(
                    arguments.predecessor_mm_count, predecessor_buffers)
                sequence += 1
                timing, handles = _measure(
                    operation, inputs, outputs,
                    arguments.call_stack, vllm_tp_group)
                _validate_result(
                    operation, inputs, outputs, handles, world_size,
                    arguments.call_stack)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "sequence": sequence,
                    "trial": trial + 1,
                    "order": order,
                    "rank": rank,
                    "operation": operation,
                    "message_bytes": message_bytes,
                    "call_count": call_count,
                    **timing,
                }
                samples[cell].append(row)
                _append_json_line(trials_path, row)

        fits = []
        cells_summary = []
        for operation in OPERATIONS:
            for message_bytes in MESSAGE_BYTES:
                medians = {}
                for call_count in CALL_COUNTS:
                    cell_samples = samples[(operation, message_bytes, call_count)]
                    wall_values = [
                        sample["host_submit_wall_ns"] for sample in cell_samples]
                    cpu_values = [
                        sample["host_submit_thread_cpu_ns"]
                        for sample in cell_samples]
                    medians[call_count] = {
                        "wall_ns": statistics.median(wall_values),
                        "thread_cpu_ns": statistics.median(cpu_values),
                    }
                    cells_summary.append({
                        "operation": operation,
                        "message_bytes": message_bytes,
                        "call_count": call_count,
                        "timing": _cell_summary(cell_samples),
                    })
                wall_intercept, wall_slope = _linear_fit(tuple(
                    (call_count, medians[call_count]["wall_ns"])
                    for call_count in sorted(FIT_CALL_COUNTS)
                ))
                cpu_intercept, cpu_slope = _linear_fit(tuple(
                    (call_count, medians[call_count]["thread_cpu_ns"])
                    for call_count in sorted(FIT_CALL_COUNTS)
                ))
                holdout_wall = medians[HOLDOUT_CALL_COUNT]["wall_ns"]
                holdout_cpu = medians[HOLDOUT_CALL_COUNT]["thread_cpu_ns"]
                fits.append({
                    "operation": operation,
                    "message_bytes": message_bytes,
                    "wall_intercept_ns": wall_intercept,
                    "wall_slope_ns_per_call": wall_slope,
                    "thread_cpu_intercept_ns": cpu_intercept,
                    "thread_cpu_slope_ns_per_call": cpu_slope,
                    "holdout_wall_relative_error": (
                        (wall_intercept + wall_slope * HOLDOUT_CALL_COUNT)
                        / holdout_wall - 1.0
                    ),
                    "holdout_thread_cpu_relative_error": (
                        (cpu_intercept + cpu_slope * HOLDOUT_CALL_COUNT)
                        / holdout_cpu - 1.0
                    ),
                })
        summary = {
            "schema_version": SCHEMA_VERSION,
            "rank": rank,
            "call_stack": arguments.call_stack,
            "predecessor": manifest["contract"]["predecessor"],
            "measurement_boundary": "async_collective_api_calls_only",
            "fits": fits,
            "system_median_wall_slope_ns_per_call": statistics.median(
                fit["wall_slope_ns_per_call"] for fit in fits),
            "system_median_thread_cpu_slope_ns_per_call": statistics.median(
                fit["thread_cpu_slope_ns_per_call"] for fit in fits),
            "cells": cells_summary,
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
    finally:
        if vllm_tp_group is not None:
            vllm_tp_group.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
