#!/usr/bin/env python3
"""Measure generic pinned CPU-to-NPU buffer-commit Host service on Ascend."""

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
from vllm.v1.utils import CpuGpuBuffer


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
CHAIN_LENGTHS = (1, 8, 64, 512, 2048)
FIT_CHAIN_LENGTHS = frozenset((1, 8, 64, 512))
# These deliberately do not include the selected Serving batch's 747 tokens.
# They cover sub-cache-line, small-control, and page-scale H2D submissions.
PAYLOAD_ELEMENTS = (17, 257, 4099)
TRIALS = 20
WARMUP_REPEATS = 5
SCHEMA_VERSION = 1


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


def _fit_line(points: tuple[tuple[int, float], ...]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("linear fit requires at least two points")
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0:
        raise ValueError("linear fit has no count variance")
    slope = sum(
        (point[0] - mean_x) * (point[1] - mean_y)
        for point in points
    ) / denominator
    intercept = mean_y - slope * mean_x
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise RuntimeError("pinned H2D fit is non-finite")
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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    if not re.fullmatch(r"[0-9a-f]{64}", arguments.expected_source_sha256):
        parser.error("--expected-source-sha256 must be a lowercase SHA256")
    return arguments


def _measure_chain(
    stream: torch.npu.Stream,
    buffer: CpuGpuBuffer,
    element_count: int,
    chain_length: int,
) -> dict[str, int | float]:
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record(stream)
    wall_start_ns = time.perf_counter_ns()
    thread_cpu_start_ns = time.thread_time_ns()
    for _ in range(chain_length):
        buffer.copy_to_gpu(element_count)
    thread_cpu_end_ns = time.thread_time_ns()
    wall_end_ns = time.perf_counter_ns()
    end_event.record(stream)
    end_event.synchronize()
    wall_ns = wall_end_ns - wall_start_ns
    thread_cpu_ns = thread_cpu_end_ns - thread_cpu_start_ns
    if wall_ns < thread_cpu_ns:
        raise RuntimeError("pinned H2D Host wall time is below thread CPU time")
    return {
        "host_call_wall_ns": wall_ns,
        "host_call_thread_cpu_ns": thread_cpu_ns,
        "host_call_non_cpu_wall_ns": wall_ns - thread_cpu_ns,
        "device_elapsed_ns": float(start_event.elapsed_time(end_event)) * 1_000_000.0,
    }


def _fit_payload(
    payload_elements: int,
    samples: dict[tuple[int, int], list[dict[str, int | float]]],
) -> dict[str, object]:
    medians = {}
    for chain_length in CHAIN_LENGTHS:
        cell = samples[(payload_elements, chain_length)]
        medians[chain_length] = {
            field: statistics.median(float(sample[field]) for sample in cell)
            for field in (
                "host_call_wall_ns",
                "host_call_thread_cpu_ns",
                "host_call_non_cpu_wall_ns",
                "device_elapsed_ns",
            )
        }
    fit_points_wall = tuple(
        (chain_length, medians[chain_length]["host_call_wall_ns"])
        for chain_length in sorted(FIT_CHAIN_LENGTHS)
    )
    fit_points_cpu = tuple(
        (chain_length, medians[chain_length]["host_call_thread_cpu_ns"])
        for chain_length in sorted(FIT_CHAIN_LENGTHS)
    )
    wall_intercept, wall_slope = _fit_line(fit_points_wall)
    cpu_intercept, cpu_slope = _fit_line(fit_points_cpu)
    holdout_wall = medians[2048]["host_call_wall_ns"]
    holdout_cpu = medians[2048]["host_call_thread_cpu_ns"]
    return {
        "payload_elements": payload_elements,
        "payload_bytes": payload_elements * 4,
        "medians_by_chain_length": {
            str(chain_length): medians[chain_length]
            for chain_length in CHAIN_LENGTHS
        },
        "wall_intercept_ns": wall_intercept,
        "wall_slope_ns_per_commit": wall_slope,
        "thread_cpu_intercept_ns": cpu_intercept,
        "thread_cpu_slope_ns_per_commit": cpu_slope,
        "holdout_chain_length": 2048,
        "holdout_wall_measured_ns": holdout_wall,
        "holdout_wall_predicted_ns": wall_intercept + wall_slope * 2048,
        "holdout_wall_signed_error_percent": (
            100.0 * ((wall_intercept + wall_slope * 2048) / holdout_wall - 1.0)
        ),
        "holdout_thread_cpu_measured_ns": holdout_cpu,
        "holdout_thread_cpu_predicted_ns": cpu_intercept + cpu_slope * 2048,
        "holdout_thread_cpu_signed_error_percent": (
            100.0 * ((cpu_intercept + cpu_slope * 2048) / holdout_cpu - 1.0)
        ),
    }


def main() -> None:
    arguments = _arguments()
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from the audited snapshot")
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
        },
        "contract": {
            "mechanism": "vllm_v1_pinned_cpu_gpu_buffer_commit",
            "source_api": "vllm.v1.utils.CpuGpuBuffer.copy_to_gpu",
            "dtype": "int32",
            "payload_elements": list(PAYLOAD_ELEMENTS),
            "payload_exclusion": "selected Serving payload is not benchmarked",
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": sorted(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": 2048,
            "trials_per_cell": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "host_clock": "time.perf_counter_ns",
            "host_thread_cpu_clock": "time.thread_time_ns",
            "device_clock": "torch.npu.Event(enable_timing=True)",
            "host_boundary": "CpuGpuBuffer.copy_to_gpu entry to return",
            "synchronization_excluded_from_host_boundary": True,
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
    }
    _write_json(manifest_path, manifest)
    try:
        torch.npu.set_device(arguments.device_id)
        stream = torch.npu.Stream(device=arguments.device_id)
        max_elements = max(PAYLOAD_ELEMENTS)
        buffers = {
            payload_elements: CpuGpuBuffer(
                max_elements,
                dtype=torch.int32,
                device=torch.device(f"npu:{arguments.device_id}"),
                pin_memory=True,
            )
            for payload_elements in PAYLOAD_ELEMENTS
        }
        samples = {
            (payload_elements, chain_length): []
            for payload_elements in PAYLOAD_ELEMENTS
            for chain_length in CHAIN_LENGTHS
        }
        cells = tuple(samples)
        with torch.npu.stream(stream):
            for payload_elements, chain_length in cells:
                for _ in range(WARMUP_REPEATS):
                    _measure_chain(
                        stream,
                        buffers[payload_elements],
                        payload_elements,
                        chain_length,
                    )
                stream.synchronize()
            sequence = 0
            for trial in range(TRIALS):
                offset = trial % len(cells)
                ordered = cells[offset:] + cells[:offset]
                if trial % 2:
                    ordered = tuple(reversed(ordered))
                for order, (payload_elements, chain_length) in enumerate(
                    ordered, start=1
                ):
                    sequence += 1
                    timing = _measure_chain(
                        stream,
                        buffers[payload_elements],
                        payload_elements,
                        chain_length,
                    )
                    samples[(payload_elements, chain_length)].append(timing)
                    _append_json_line(trials_path, {
                        "schema_version": SCHEMA_VERSION,
                        "sequence": sequence,
                        "trial": trial + 1,
                        "order": order,
                        "payload_elements": payload_elements,
                        "payload_bytes": payload_elements * 4,
                        "chain_length": chain_length,
                        **timing,
                    })
        fits = tuple(
            _fit_payload(payload_elements, samples)
            for payload_elements in PAYLOAD_ELEMENTS
        )
        wall_slopes = tuple(
            float(fit["wall_slope_ns_per_commit"]) for fit in fits
        )
        cpu_slopes = tuple(
            float(fit["thread_cpu_slope_ns_per_commit"]) for fit in fits
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "source": {"path": str(source_path), "sha256": source_sha256},
            "mechanism": "vllm_v1_pinned_cpu_gpu_buffer_commit",
            "measurement_boundary": (
                "CpuGpuBuffer.copy_to_gpu entry to return; post-chain "
                "synchronization is excluded from Host service"
            ),
            "fits_by_payload": fits,
            "cross_payload_summary": {
                "wall_slope_median_ns_per_commit": statistics.median(wall_slopes),
                "wall_slope_min_ns_per_commit": min(wall_slopes),
                "wall_slope_max_ns_per_commit": max(wall_slopes),
                "thread_cpu_slope_median_ns_per_commit": statistics.median(cpu_slopes),
                "thread_cpu_slope_min_ns_per_commit": min(cpu_slopes),
                "thread_cpu_slope_max_ns_per_commit": max(cpu_slopes),
                "payload_independence_check": (
                    "compare independently fit slopes before any scalar "
                    "runtime_prepare parameter is enabled"
                ),
            },
            "production_eligibility": (
                "pending source occurrence descriptor and independent "
                "payload-independence review"
            ),
        }
        _write_json(summary_path, summary)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
    except Exception as error:
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
    main()
