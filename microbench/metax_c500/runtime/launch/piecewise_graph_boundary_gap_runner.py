#!/usr/bin/env python3
"""Measure C500 device-side gaps at CUDA-Graph / eager launch boundaries.

Synthetic mechanism probe: every launch is a preallocated ``torch.add`` on a
16-KiB BF16 tensor.  One repetition replays a captured graph of
``GRAPH_NODES`` nodes and then issues ``EAGER_OPS`` eager launches, the same
alternation a vLLM PIECEWISE step makes around each eager attention region.
A long spin kernel is queued first so the Host has submitted the whole
sequence before the device reaches it; every measured gap is therefore
device-side.  No production kernel, model, Serving workload or evaluation
duration participates.

Output: per boundary class (graph entry, graph exit, eager-to-eager,
graph node-to-node) the gap samples in ns from kineto device timestamps.

Usage (inside the C500 vLLM container, one idle device):
  python piecewise_graph_boundary_gap_runner.py OUT.json
"""

import json
import statistics
import sys

import torch


GRAPH_NODES = 8
EAGER_OPS = 6
REPETITIONS = 64
TRIALS = 3
ELEMENTS = 8192
SPIN_CYCLES = 200_000_000


def _run_trial(graph, left, right, out_eager):
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
         if event.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda event: event.time_range.start)
    kernels = [event for event in kernels if "spin_kernel" not in event.name]
    expected = REPETITIONS * (GRAPH_NODES + EAGER_OPS)
    if len(kernels) != expected:
        raise RuntimeError(
            f"expected {expected} device launches, observed {len(kernels)}")
    gaps = {"graph_entry": [], "graph_exit": [], "eager_to_eager": [],
            "graph_node_to_node": []}
    period = GRAPH_NODES + EAGER_OPS
    for index in range(1, len(kernels)):
        previous, current = kernels[index - 1], kernels[index]
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
    return gaps


def main():
    out_path = sys.argv[1]
    torch.cuda.set_device(0)
    left = torch.randn(ELEMENTS, dtype=torch.bfloat16, device="cuda")
    right = torch.randn(ELEMENTS, dtype=torch.bfloat16, device="cuda")
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
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    trials = [_run_trial(graph, left, right, out_eager)
              for _ in range(TRIALS)]
    summary = {
        name: {
            "trial_median_ns": [statistics.median(trial[name])
                                for trial in trials],
            "samples_per_trial": len(trials[0][name]),
        }
        for name in trials[0]
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": "midlayer.metax_c500.piecewise_graph_boundary_gap.v1",
            "hardware_id": "metax_c500",
            "torch_version": torch.__version__,
            "graph_nodes": GRAPH_NODES,
            "eager_ops": EAGER_OPS,
            "repetitions": REPETITIONS,
            "elements": ELEMENTS,
            "timer_domain": "kineto_device_timestamps_host_ahead",
            "production_or_evaluation_duration_consumed": False,
            "summary": summary,
            "trials": trials,
        }, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
