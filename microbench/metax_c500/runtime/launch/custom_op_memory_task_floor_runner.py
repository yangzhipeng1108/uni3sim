#!/usr/bin/env python3
"""Measure a geometry-aware PyTorch custom-op memory-task lower bound."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics

import torch
from torch.utils.cpp_extension import load


SCHEMA_VERSION = 1
HARDWARE_ID = "metax_c500"
EXPECTED_TORCH_VERSION = "2.10.0+metax3.8.2.2"
EXPECTED_AP_COUNT = 104
EXPECTED_WAVE_SIZE = 64
GRID_SIZES = (1, 16, 64, 104, 256, 1024, 4096, 16384)
BLOCK_SIZES = (64, 256, 512, 1024)
SAMPLES_PER_CELL = 15
WARMUPS_PER_CELL = 8
EVENT_BOOTSTRAP_ROUNDS = 128


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure_once(call):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    duration_ns = float(start.elapsed_time(end)) * 1.0e6
    if not math.isfinite(duration_ns) or duration_ns <= 0:
        raise RuntimeError("device-event duration must be finite and positive")
    return duration_ns


def _load_library(source, build_directory):
    build_directory.mkdir(parents=True, exist_ok=True)
    load(
        name="midlayer_c500_custom_op_memory_task_floor",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--trial", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--software-revision", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build-directory", required=True, type=Path)
    args = parser.parse_args()
    if args.device_id < 0:
        raise ValueError("device-id must be non-negative")
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"custom-op source is unavailable: {source}")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"expected torch {EXPECTED_TORCH_VERSION}, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("MetaX CUDA-compatible device is unavailable")
    _load_library(source, args.build_directory.resolve())
    torch.cuda.set_device(args.device_id)
    properties = torch.cuda.get_device_properties(args.device_id)
    if properties.multi_processor_count != EXPECTED_AP_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_AP_COUNT} APs, got "
            f"{properties.multi_processor_count}")
    maximum_elements = max(GRID_SIZES) * (max(BLOCK_SIZES) // EXPECTED_WAVE_SIZE)
    input_tensor = torch.ones(
        maximum_elements, dtype=torch.uint32, device="cuda")
    output_tensor = torch.zeros_like(input_tensor)

    def launch(kind, grid_size, block_size):
        if kind == "empty":
            torch.ops.midlayer_microbench.custom_op_empty_task(
                input_tensor, grid_size, 1, block_size)
        elif kind == "memory":
            torch.ops.midlayer_microbench.custom_op_memory_dependency_task(
                input_tensor, output_tensor, grid_size, 1, block_size)
        else:
            raise AssertionError(f"unknown custom-op task kind: {kind!r}")

    cells = tuple(
        (kind, grid_size, block_size)
        for kind in ("empty", "memory")
        for grid_size in GRID_SIZES
        for block_size in BLOCK_SIZES)
    for kind, grid_size, block_size in cells:
        for _ in range(WARMUPS_PER_CELL):
            launch(kind, grid_size, block_size)
    torch.cuda.synchronize()
    for bootstrap in range(EVENT_BOOTSTRAP_ROUNDS):
        kind, grid_size, block_size = cells[bootstrap % len(cells)]
        _measure_once(
            lambda kind=kind, grid_size=grid_size,
            block_size=block_size: launch(kind, grid_size, block_size))

    schedule = [
        cell
        for cell in cells
        for _ in range(SAMPLES_PER_CELL)
    ]
    random.Random(500_911 + 7_919 * args.trial).shuffle(schedule)
    samples = {
        f"{kind}:grid{grid_size}:block{block_size}": []
        for kind, grid_size, block_size in cells
    }
    for kind, grid_size, block_size in schedule:
        duration_ns = _measure_once(
            lambda kind=kind, grid_size=grid_size,
            block_size=block_size: launch(kind, grid_size, block_size))
        samples[
            f"{kind}:grid{grid_size}:block{block_size}"
        ].append(duration_ns)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "software_revision": args.software_revision,
        "torch_version": torch.__version__,
        "device_id": args.device_id,
        "device_name": properties.name,
        "multi_processor_count": properties.multi_processor_count,
        "trial": args.trial,
        "pid": os.getpid(),
        "timer": "torch_cuda_event",
        "profiler_attached": False,
        "mechanism": "warm_pytorch_custom_op_memory_task_floor",
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "probe_operators": [
            "midlayer_microbench.custom_op_empty_task",
            "midlayer_microbench.custom_op_memory_dependency_task",
        ],
        "grid_sizes": list(GRID_SIZES),
        "block_sizes": list(BLOCK_SIZES),
        "wave_size": EXPECTED_WAVE_SIZE,
        "samples_per_cell": SAMPLES_PER_CELL,
        "event_bootstrap_rounds": EVENT_BOOTSTRAP_ROUNDS,
        "warmups_per_cell": WARMUPS_PER_CELL,
        "source_sha256": _sha256(source),
        "samples_ns": samples,
        "medians_ns": {
            key: statistics.median(values)
            for key, values in samples.items()
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
