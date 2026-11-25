#!/usr/bin/env python3
"""Measure the A800 warm floor of one non-empty BF16 binary task.

A800's HWSpec carries `isolated_empty_task_scheduler_v1`: its launch floor is
calibrated on a task that never issues a load.  C500's own span-floor runner
states why that is not a production bound -- "the empty body is reported
alongside but never published, because it describes a launch that touches
nothing and every production kernel touches memory" -- and C500 instead
publishes a non-empty task floor that the HWSpec consumes through
`runtime_task_latency_models`.  This probe produces the A800 counterpart on
the same terms.

Method mirrors `bf16_binary_tensor_iterator_task_floor_maca_3_7_2.json`: one
BF16 binary TensorIterator task (torch.maximum) is captured in a CUDA graph and
replayed; the paired differential between a graph holding `repeat` tasks and
one holding `repeat // 2` removes graph-replay setup and event overhead, so
what remains is per-task.  The published value is the minimum differential over
the element sweep, because composing by maximum needs a bound no task beats,
not a typical duration.

No production operator, kernel, or Serving duration participates: the task is
torch.maximum on synthetic tensors.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def _graph_for(elements: int, repeat: int, device: torch.device):
    a = torch.rand(elements, dtype=torch.bfloat16, device=device)
    b = torch.rand(elements, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(a)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.maximum(a, b, out=out)
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeat):
            torch.maximum(a, b, out=out)
    return graph, (a, b, out)


def _replay_ns(graph, iters: int, device: torch.device) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--elements", type=int, nargs="+",
                        default=[1, 1024, 16384, 1048576])
    parser.add_argument("--repeats", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    samples = []
    for elements in args.elements:
        for repeat in args.repeats:
            high, _keep_high = _graph_for(elements, repeat, device)
            low, _keep_low = _graph_for(elements, repeat // 2, device)
            high_ns = _replay_ns(high, args.iters, device)
            low_ns = _replay_ns(low, args.iters, device)
            per_task_ns = (high_ns - low_ns) / (repeat - repeat // 2)
            samples.append({
                "elements": elements,
                "repeat_high": repeat,
                "repeat_low": repeat // 2,
                "high_replay_ns": high_ns,
                "low_replay_ns": low_ns,
                "per_task_ns": per_task_ns,
            })
            del high, low, _keep_high, _keep_low
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    document = {
        "schema": "nvidia_a800_bf16_binary_task_floor_raw_v1",
        "hardware_id": "nvidia_a800",
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physical_device_index": args.device_index,
        "timer_domain": "paired_differential_events_around_cuda_graph_replay",
        "task_class": "pytorch_bf16_binary_tensor_iterator",
        "target_operator_used": False,
        "production_or_evaluation_duration_consumed": False,
        "iters_per_measurement": args.iters,
        "samples": samples,
        "minimum_per_task_ns": min(s["per_task_ns"] for s in samples),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
    print(json.dumps({
        "minimum_per_task_ns": document["minimum_per_task_ns"],
        "by_elements": {
            str(e): round(min(s["per_task_ns"] for s in samples
                              if s["elements"] == e), 2)
            for e in args.elements},
    }, indent=1))


if __name__ == "__main__":
    main()
