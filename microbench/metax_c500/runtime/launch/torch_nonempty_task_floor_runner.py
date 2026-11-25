#!/usr/bin/env python3
"""Measure a warm BF16 binary TensorIterator task floor with graph replay."""

import argparse
import json
import math
import os
import random
import statistics

import torch


SCHEMA_VERSION = 2
HARDWARE_ID = "metax_c500"
EXPECTED_TORCH_VERSION = "2.4.0+metax3.7.2.0"
PROBE_ELEMENTS = (1, 1024, 16384, 1048576)
REPETITION_COUNTS = (100, 500)
SAMPLES_PER_CELL = 31
WARMUPS_PER_CELL = 32


def _measure_replay(graph):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    duration_ns = float(start.elapsed_time(end)) * 1.0e6
    if not math.isfinite(duration_ns) or duration_ns <= 0:
        raise RuntimeError("graph replay duration must be finite and positive")
    return duration_ns


def _measure_cell(call, trial):
    for _ in range(WARMUPS_PER_CELL):
        call()
    torch.cuda.synchronize()
    graphs = []
    for repetitions in REPETITION_COUNTS:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repetitions):
                call()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        graphs.append(graph)

    schedule = [sample % 2 for sample in range(SAMPLES_PER_CELL)]
    random.Random(500_731 + 7_919 * trial).shuffle(schedule)
    samples = []
    for sample, first_index in enumerate(schedule):
        order = (first_index, 1 - first_index)
        durations = {}
        for index in order:
            durations[index] = _measure_replay(graphs[index])
        differential_ns = (
            (durations[1] - durations[0])
            / (REPETITION_COUNTS[1] - REPETITION_COUNTS[0]))
        if not math.isfinite(differential_ns) or differential_ns <= 0:
            raise RuntimeError(
                f"graph replay differential must be positive: {durations!r}")
        samples.append({
            "sample": sample,
            "order": list(order),
            "lower_replay_ns": durations[0],
            "higher_replay_ns": durations[1],
            "differential_ns": differential_ns,
        })
    values = [sample["differential_ns"] for sample in samples]
    return {
        "samples": samples,
        "minimum_ns": min(values),
        "median_ns": statistics.median(values),
        "maximum_ns": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--trial", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--software-revision", required=True)
    args = parser.parse_args()
    if args.device_id < 0:
        raise ValueError("device-id must be non-negative")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"expected torch {EXPECTED_TORCH_VERSION}, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("MetaX CUDA-compatible device is unavailable")
    torch.cuda.set_device(args.device_id)
    properties = torch.cuda.get_device_properties(args.device_id)
    left = torch.ones(max(PROBE_ELEMENTS), dtype=torch.bfloat16, device="cuda")
    right = torch.ones_like(left)
    output = torch.empty_like(left)
    cells = list(PROBE_ELEMENTS)
    random.Random(500_731 + 7_919 * args.trial).shuffle(cells)
    results = {}
    for elements in cells:
        results[str(elements)] = _measure_cell(
            lambda elements=elements: torch.maximum(
                left[:elements], right[:elements], out=output[:elements]),
            args.trial,
        )
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "software_revision": args.software_revision,
        "torch_version": torch.__version__,
        "device_id": args.device_id,
        "device_name": properties.name,
        "multi_processor_count": properties.multi_processor_count,
        "trial": args.trial,
        "pid": os.getpid(),
        "timer": "paired_differential_events_around_cuda_graph_replay",
        "profiler_attached": False,
        "mechanism": "warm_bf16_binary_tensor_iterator_task_floor",
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "probe_operator": "torch.maximum",
        "probe_elements": list(PROBE_ELEMENTS),
        "repetition_counts": list(REPETITION_COUNTS),
        "samples_per_cell": SAMPLES_PER_CELL,
        "warmups_per_cell": WARMUPS_PER_CELL,
        "results": results,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
