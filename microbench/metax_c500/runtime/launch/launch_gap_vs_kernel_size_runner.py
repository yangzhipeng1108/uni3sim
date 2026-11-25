#!/usr/bin/env python3
"""Measure how C500 device-side launch gaps depend on the preceding kernel.

Extends ``piecewise_graph_boundary_gap_runner.py``: the same host-ahead
alternation (one captured graph of ``GRAPH_NODES`` nodes, then ``EAGER_OPS``
eager launches) is repeated with synthetic ``torch.add`` launches of several
element counts, so the preceding launch runs for a few microseconds up to
hundreds of microseconds.  If the runtime/command processor stages the next
launch while the current one executes, the boundary gap shrinks with the
preceding kernel's duration.  No production kernel, model, Serving workload
or evaluation duration participates.

Usage (inside the C500 vLLM container, one idle device):
  python launch_gap_vs_kernel_size_runner.py OUT.json
"""

import json
import statistics
import sys

import torch


GRAPH_NODES = 8
EAGER_OPS = 6
REPETITIONS = 32
TRIALS = 3
ELEMENT_COUNTS = (8192, 1 << 20, 1 << 23, 1 << 25)
SPIN_CYCLES = 400_000_000


def _trial(graph, left, right, out_eager):
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.cuda._sleep(SPIN_CYCLES)
        for _ in range(REPETITIONS):
            graph.replay()
            for output in out_eager:
                torch.add(left, right, out=output)
        torch.cuda.synchronize()
    kernels = sorted(
        (event for event in prof.events()
         if event.device_type == torch.autograd.DeviceType.CUDA
         and "spin_kernel" not in event.name),
        key=lambda event: event.time_range.start)
    expected = REPETITIONS * (GRAPH_NODES + EAGER_OPS)
    if len(kernels) != expected:
        raise RuntimeError(
            f"expected {expected} device launches, observed {len(kernels)}")
    gaps = {"graph_entry": [], "graph_exit": [], "eager_to_eager": [],
            "graph_node_to_node": []}
    durations = []
    period = GRAPH_NODES + EAGER_OPS
    for index in range(1, len(kernels)):
        previous, current = kernels[index - 1], kernels[index]
        durations.append(previous.time_range.elapsed_us())
        gap_ns = 1000.0 * (current.time_range.start - previous.time_range.end)
        position = index % period
        if position == 0:
            gaps["graph_entry"].append(gap_ns)
        elif position < GRAPH_NODES:
            gaps["graph_node_to_node"].append(gap_ns)
        elif position == GRAPH_NODES:
            gaps["graph_exit"].append(gap_ns)
        else:
            gaps["eager_to_eager"].append(gap_ns)
    return gaps, statistics.median(durations)


def _cell(elements):
    left = torch.randn(elements, dtype=torch.bfloat16, device="cuda")
    right = torch.randn(elements, dtype=torch.bfloat16, device="cuda")
    out_graph = [torch.empty_like(left) for _ in range(GRAPH_NODES)]
    out_eager = [torch.empty_like(left) for _ in range(EAGER_OPS)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for output in out_graph:
            torch.add(left, right, out=output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for output in out_graph:
            torch.add(left, right, out=output)
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    trials = [_trial(graph, left, right, out_eager) for _ in range(TRIALS)]
    return {
        "elements": elements,
        "median_kernel_us": statistics.median(t[1] for t in trials),
        "trial_median_gap_ns": {
            name: [statistics.median(t[0][name]) for t in trials]
            for name in trials[0][0]
        },
    }


def main():
    torch.cuda.set_device(0)
    cells = [_cell(elements) for elements in ELEMENT_COUNTS]
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": "midlayer.metax_c500.launch_gap_vs_kernel_size.v1",
            "hardware_id": "metax_c500",
            "torch_version": torch.__version__,
            "timer_domain": "kineto_device_timestamps_host_ahead",
            "production_or_evaluation_duration_consumed": False,
            "cells": cells,
        }, handle, indent=1)
    for cell in cells:
        print(cell["elements"], round(cell["median_kernel_us"], 2),
              {k: [round(x) for x in v]
               for k, v in cell["trial_median_gap_ns"].items()})


if __name__ == "__main__":
    main()
