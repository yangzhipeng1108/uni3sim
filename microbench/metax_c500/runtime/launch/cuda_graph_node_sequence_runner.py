#!/usr/bin/env python3
"""Measure C500 CUDA-Graph completion for a fixed synthetic node sequence.

The probe captures only preallocated ``torch.maximum`` nodes.  It contains no
production kernel, routing, profiler observation, or evaluation workload.
Its paired-replay differential removes the graph-event setup shared by the
two repetition counts, leaving one ordered graph sequence completion time.
"""

import argparse
import json
import math
import os
import random
import statistics

import torch


HARDWARE_ID = "metax_c500"
NODE_COUNTS = (1, 2, 4, 8)
ELEMENT_COUNT = 16_384
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
    if not math.isfinite(duration_ns) or duration_ns <= 0.0:
        raise RuntimeError("graph replay duration must be finite and positive")
    return duration_ns


def _measure_node_sequence(left, right, outputs, node_count, trial):
    def call_sequence():
        for output in outputs[:node_count]:
            torch.maximum(left, right, out=output)

    for _ in range(WARMUPS_PER_CELL):
        call_sequence()
    torch.cuda.synchronize()
    graphs = []
    for repetitions in REPETITION_COUNTS:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repetitions):
                call_sequence()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        graphs.append(graph)

    schedule = [sample % 2 for sample in range(SAMPLES_PER_CELL)]
    random.Random(410_729 + 5_039 * trial + node_count).shuffle(schedule)
    samples = []
    for sample, first_index in enumerate(schedule):
        order = (first_index, 1 - first_index)
        durations = {}
        for index in order:
            durations[index] = _measure_replay(graphs[index])
        sequence_ns = (
            (durations[1] - durations[0])
            / (REPETITION_COUNTS[1] - REPETITION_COUNTS[0]))
        if not math.isfinite(sequence_ns) or sequence_ns <= 0.0:
            raise RuntimeError(
                "graph replay differential must be positive: "
                f"{durations!r}")
        samples.append({
            "sample": sample,
            "order": list(order),
            "lower_replay_ns": durations[0],
            "higher_replay_ns": durations[1],
            "sequence_ns": sequence_ns,
            "per_node_ns": sequence_ns / node_count,
        })
    sequence_values = [sample["sequence_ns"] for sample in samples]
    per_node_values = [sample["per_node_ns"] for sample in samples]
    return {
        "node_count": node_count,
        "samples": samples,
        "sequence_minimum_ns": min(sequence_values),
        "sequence_median_ns": statistics.median(sequence_values),
        "sequence_maximum_ns": max(sequence_values),
        "per_node_minimum_ns": min(per_node_values),
        "per_node_median_ns": statistics.median(per_node_values),
        "per_node_maximum_ns": max(per_node_values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--trial", required=True, type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.device_id < 0:
        raise ValueError("device-id must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("MetaX CUDA-compatible device is unavailable")
    torch.cuda.set_device(args.device_id)
    properties = torch.cuda.get_device_properties(args.device_id)
    left = torch.ones(ELEMENT_COUNT, dtype=torch.bfloat16, device="cuda")
    right = torch.ones_like(left)
    outputs = tuple(torch.empty_like(left) for _ in range(max(NODE_COUNTS)))
    results = {
        str(node_count): _measure_node_sequence(
            left, right, outputs, node_count, args.trial)
        for node_count in NODE_COUNTS
    }
    print(json.dumps({
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "torch_version": torch.__version__,
        "device_id": args.device_id,
        "device_name": properties.name,
        "multi_processor_count": properties.multi_processor_count,
        "trial": args.trial,
        "pid": os.getpid(),
        "timer": "paired_differential_events_around_cuda_graph_replay",
        "profiler_attached": False,
        "mechanism": "ordered_bf16_cuda_graph_node_sequence_completion",
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "probe_operator": "torch.maximum",
        "element_count": ELEMENT_COUNT,
        "node_counts": list(NODE_COUNTS),
        "repetition_counts": list(REPETITION_COUNTS),
        "samples_per_cell": SAMPLES_PER_CELL,
        "warmups_per_cell": WARMUPS_PER_CELL,
        "results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
