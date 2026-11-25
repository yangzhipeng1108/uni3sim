#!/usr/bin/env python3
"""Test whether Ascend eager enqueue service depends on Device queue pressure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import torch
import torch_npu


OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
MATRIX_SIZES = (64, 256, 1024)
CHAIN_LENGTHS = (1, 8, 64, 512, 2048)
TRIALS = 12
WARMUP_REPEATS = 4


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_json_line(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


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


def _linear_slope(points):
    count = len(points)
    x_sum = sum(x for x, _ in points)
    y_sum = sum(y for _, y in points)
    xx_sum = sum(x * x for x, _ in points)
    xy_sum = sum(x * y for x, y in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("queue-pressure fit is singular")
    return (count * xy_sum - x_sum * y_sum) / denominator


def _run_chain(length, left, right, output):
    source = left
    destination = output
    for _ in range(length):
        torch.mm(source, right, out=destination)
        source, destination = destination, source
    return source


def _measure(stream, length, state):
    left, right, output = state
    stream.synchronize()
    wall_start_ns = time.perf_counter_ns()
    thread_start_ns = time.thread_time_ns()
    result = _run_chain(length, left, right, output)
    thread_end_ns = time.thread_time_ns()
    wall_end_ns = time.perf_counter_ns()
    stream.synchronize()
    completion_ns = time.perf_counter_ns()
    return result, {
        "host_enqueue_wall_ns": wall_end_ns - wall_start_ns,
        "host_enqueue_thread_cpu_ns": thread_end_ns - thread_start_ns,
        "host_enqueue_non_cpu_wall_ns": (
            wall_end_ns - wall_start_ns - (thread_end_ns - thread_start_ns)
        ),
        "device_tail_after_enqueue_ns": completion_ns - wall_end_ns,
        "end_to_end_ns": completion_ns - wall_start_ns,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    args = parser.parse_args()
    args.output_dir = _validate_output_dir(args.output_dir)
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if len(args.baseline_commit) != 40:
        parser.error("--baseline-commit must be a full Git commit")
    if len(args.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    return args


def main():
    args = _parse_args()
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path)
    if source_sha256 != args.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from the audited snapshot")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.json"
    trials_path = args.output_dir / "trials.jsonl"
    summary_path = args.output_dir / "summary.json"
    manifest = {
        "schema_version": 1,
        "status": "initializing",
        "baseline_commit": args.baseline_commit,
        "source": {"path": str(source_path), "sha256": source_sha256},
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "container_name": args.container_name,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "contract": {
            "hypothesis": (
                "eager enqueue service increases when the same API produces "
                "more Device work and fills the runtime queue"
            ),
            "operation": "torch.mm(left, right, out=output)",
            "matrix_sizes": list(MATRIX_SIZES),
            "chain_lengths": list(CHAIN_LENGTHS),
            "trials_per_cell": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "allocation_excluded": True,
            "jit_warmup_excluded": True,
            "completion_sync_excluded_from_host_enqueue": True,
            "host_clocks": ["time.perf_counter_ns", "time.thread_time_ns"],
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
    }
    _write_json(manifest_path, manifest)
    try:
        torch.npu.set_device(args.device_id)
        stream = torch.npu.Stream(device=args.device_id)
        samples = {
            (size, length): []
            for size in MATRIX_SIZES for length in CHAIN_LENGTHS
        }
        with torch.npu.stream(stream):
            states = {
                size: (
                    torch.ones((size, size), dtype=torch.float16, device="npu"),
                    torch.eye(size, dtype=torch.float16, device="npu"),
                    torch.empty((size, size), dtype=torch.float16, device="npu"),
                )
                for size in MATRIX_SIZES
            }
            for size in MATRIX_SIZES:
                for _ in range(WARMUP_REPEATS):
                    _run_chain(8, *states[size])
            stream.synchronize()
            cells = tuple(
                (size, length)
                for size in MATRIX_SIZES for length in CHAIN_LENGTHS
            )
            sequence = 0
            for trial in range(TRIALS):
                offset = trial % len(cells)
                ordered = cells[offset:] + cells[:offset]
                if trial % 2:
                    ordered = tuple(reversed(ordered))
                for order, (size, length) in enumerate(ordered, start=1):
                    sequence += 1
                    result, timing = _measure(
                        stream, length, states[size])
                    row = {
                        "schema_version": 1,
                        "sequence": sequence,
                        "trial": trial + 1,
                        "order": order,
                        "matrix_size": size,
                        "chain_length": length,
                        **timing,
                    }
                    samples[(size, length)].append(row)
                    _append_json_line(trials_path, row)
            stream.synchronize()
            if not torch.isfinite(result).all().item():
                raise RuntimeError("tensor validation failed")
        cells_summary = []
        slopes = []
        for size in MATRIX_SIZES:
            wall_points = []
            thread_points = []
            for length in CHAIN_LENGTHS:
                rows = samples[(size, length)]
                wall_median = statistics.median(
                    row["host_enqueue_wall_ns"] for row in rows)
                thread_median = statistics.median(
                    row["host_enqueue_thread_cpu_ns"] for row in rows)
                tail_median = statistics.median(
                    row["device_tail_after_enqueue_ns"] for row in rows)
                cells_summary.append({
                    "matrix_size": size,
                    "chain_length": length,
                    "sample_count": len(rows),
                    "host_enqueue_wall_median_ns": wall_median,
                    "host_enqueue_thread_cpu_median_ns": thread_median,
                    "device_tail_after_enqueue_median_ns": tail_median,
                })
                if length <= 512:
                    wall_points.append((length, wall_median))
                    thread_points.append((length, thread_median))
            slopes.append({
                "matrix_size": size,
                "fit_chain_lengths": [1, 8, 64, 512],
                "host_enqueue_wall_slope_ns_per_call": _linear_slope(wall_points),
                "host_enqueue_thread_cpu_slope_ns_per_call": _linear_slope(
                    thread_points),
            })
        summary = {
            "schema_version": 1,
            "parameter": "device_queue_pressure_effect_on_eager_enqueue",
            "cells": cells_summary,
            "slopes": slopes,
        }
        _write_json(summary_path, summary)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _utc_now()
        manifest["artifacts"] = {
            "trials": {"path": str(trials_path), "sha256": _sha256(trials_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        }
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
