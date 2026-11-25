#!/usr/bin/env python3
"""Measure C500 eager D2D copy service and the launches around it.

Synthetic mechanism probe for the runtime pattern a vLLM PIECEWISE step makes
at each eager attention region: a producer kernel writes a buffer, a
contiguous BF16 ``copy_`` moves it (the runtime issues a D2D memcpy task),
a second producer and a second smaller copy follow, and the next captured
CUDA Graph is replayed.  Every launch is a preallocated ``torch.add`` or
``Tensor.copy_``; the copy byte counts are a fixed sweep, not taken from any
workload.  A long spin kernel is queued first so the Host has submitted the
whole sequence before the device reaches it; every recorded interval is
therefore device-side.  A control arm replaces both copies by nothing, so the
same producer -> Graph boundary is measured without a preceding copy task.
No production kernel, model, Serving workload or evaluation duration
participates.

Output: per (first copy bytes, second copy bytes) cell and per boundary
(producer -> copy, copy duration, copy -> producer, copy -> Graph entry,
and producer -> Graph entry for the control arm)
the device-timestamp samples in ns.  Events are ordered by their end
timestamp: a copy task's start stamp can precede the end of the kernel it
follows.

Usage (inside the C500 vLLM container, one idle device):
  python dtod_copy_boundary_runner.py OUT.json
"""

import json
import statistics
import sys

import torch


GRAPH_NODES = 8
REPETITIONS = 48
TRIALS = 3
SMALL_ELEMENTS = 8192
FIRST_COPY_BYTES = (8192, 65536, 262144, 1048576, 2097152, 4194304)
SECOND_COPY_BYTES = (8192, 65536)
SPIN_CYCLES = 200_000_000
MEMCPY_MARK = "Memcpy"


def _device_events(prof):
    events = sorted(
        (event for event in prof.events()
         if event.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda event: event.time_range.end)
    return [event for event in events if "spin_kernel" not in event.name]


def _gap_ns(previous, current):
    return 1000.0 * (current.time_range.start - previous.time_range.end)


def _duration_ns(event):
    return 1000.0 * (event.time_range.end - event.time_range.start)


def _run_trial(graph, first_bytes, second_bytes, buffers, with_copies):
    left, right = buffers["left"], buffers["right"]
    src1, dst1 = buffers["src1"], buffers["dst1"]
    src2, dst2 = buffers["src2"], buffers["dst2"]
    n1, n2 = first_bytes // 2, second_bytes // 2
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.cuda._sleep(SPIN_CYCLES)
        for _ in range(REPETITIONS):
            graph.replay()
            torch.add(src1[:n1], 1.0, out=src1[:n1])
            if with_copies:
                dst1[:n1].copy_(src1[:n1])
            torch.add(src2[:n2], 1.0, out=src2[:n2])
            if with_copies:
                dst2[:n2].copy_(src2[:n2])
        torch.cuda.synchronize()
    events = _device_events(prof)
    period = GRAPH_NODES + (4 if with_copies else 2)
    if len(events) != REPETITIONS * period:
        raise RuntimeError(
            f"expected {REPETITIONS * period} device events, observed "
            f"{len(events)}")
    samples = {name: [] for name in (
        "producer_to_copy1", "copy1_duration", "copy1_to_producer",
        "producer_to_copy2", "copy2_duration",
        "copy2_to_graph_entry", "producer_to_graph_entry")}
    for start in range(0, len(events), period):
        rep = events[start:start + period]
        eager = rep[GRAPH_NODES:]
        names = [event.name for event in eager]
        if with_copies:
            if not (MEMCPY_MARK in names[1] and MEMCPY_MARK in names[3]):
                raise RuntimeError(f"unexpected eager order: {names}")
            samples["producer_to_copy1"].append(_gap_ns(eager[0], eager[1]))
            samples["copy1_duration"].append(_duration_ns(eager[1]))
            samples["copy1_to_producer"].append(_gap_ns(eager[1], eager[2]))
            samples["producer_to_copy2"].append(_gap_ns(eager[2], eager[3]))
            samples["copy2_duration"].append(_duration_ns(eager[3]))
        elif any(MEMCPY_MARK in name for name in names):
            raise RuntimeError(f"control arm issued a copy: {names}")
        if start + period < len(events):
            entry = ("copy2_to_graph_entry" if with_copies
                     else "producer_to_graph_entry")
            samples[entry].append(_gap_ns(eager[-1], events[start + period]))
    return {name: values for name, values in samples.items() if values}


def main():
    out_path = sys.argv[1]
    torch.cuda.set_device(0)
    largest = max(FIRST_COPY_BYTES) // 2
    buffers = {
        "left": torch.randn(SMALL_ELEMENTS, dtype=torch.bfloat16, device="cuda"),
        "right": torch.randn(SMALL_ELEMENTS, dtype=torch.bfloat16, device="cuda"),
        "src1": torch.randn(largest, dtype=torch.bfloat16, device="cuda"),
        "dst1": torch.empty(largest, dtype=torch.bfloat16, device="cuda"),
        "src2": torch.randn(max(SECOND_COPY_BYTES) // 2, dtype=torch.bfloat16,
                            device="cuda"),
        "dst2": torch.empty(max(SECOND_COPY_BYTES) // 2, dtype=torch.bfloat16,
                            device="cuda"),
    }
    out_graph = [torch.empty_like(buffers["left"]) for _ in range(GRAPH_NODES)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for output in out_graph:
            torch.add(buffers["left"], buffers["right"], out=output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for output in out_graph:
            torch.add(buffers["left"], buffers["right"], out=output)
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()

    cells = []
    arms = [(first, second, True) for first in FIRST_COPY_BYTES
            for second in SECOND_COPY_BYTES]
    arms.append((FIRST_COPY_BYTES[0], SECOND_COPY_BYTES[0], False))
    for first, second, with_copies in arms:
        _run_trial(graph, first, second, buffers, with_copies)  # warm-up
        trials = [_run_trial(graph, first, second, buffers, with_copies)
                  for _ in range(TRIALS)]
        summary = {
            name: {
                "trial_median_ns": [statistics.median(trial[name])
                                    for trial in trials],
                "trial_mean_ns": [statistics.fmean(trial[name])
                                  for trial in trials],
            }
            for name in trials[0]
        }
        cells.append({
            "with_copies": with_copies,
            "first_copy_bytes": first,
            "second_copy_bytes": second,
            "summary": summary,
            "trials": trials,
        })
        print(json.dumps({
            "with_copies": with_copies, "first": first, "second": second,
            **{name: [round(v / 1000.0, 2) for v in value["trial_mean_ns"]]
               for name, value in summary.items()}}), flush=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": "midlayer.metax_c500.dtod_copy_boundary.v1",
            "hardware_id": "metax_c500",
            "torch_version": torch.__version__,
            "graph_nodes": GRAPH_NODES,
            "repetitions": REPETITIONS,
            "timer_domain": "kineto_device_timestamps_host_ahead",
            "production_or_evaluation_duration_consumed": False,
            "cells": cells,
        }, handle, indent=1)


if __name__ == "__main__":
    main()
