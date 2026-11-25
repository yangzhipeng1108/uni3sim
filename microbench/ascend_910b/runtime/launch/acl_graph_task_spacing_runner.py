"""Measure independent ACL Graph task-chain device service on one Ascend NPU.

The chain is alternating out-of-place 4096-element FP32 adds. It has no model
operator or Serving workload. A separate msprof wrapper may inspect the task
intervals; this runner records event timing without using profiler durations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa: F401  # Register the NPU device and graph API.


CHAIN_LENGTHS = (1, 64, 512)
TRIALS = 12


def _chain(left: torch.Tensor, right: torch.Tensor, increment: torch.Tensor,
           count: int) -> None:
    source, destination = left, right
    for _ in range(count):
        torch.add(source, increment, out=destination)
        source, destination = destination, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.device_id != 0:
        raise ValueError("this campaign is assigned to NPU 0")
    if (len(args.source_commit) != 40
            or any(ch not in "0123456789abcdef" for ch in args.source_commit)):
        raise ValueError("--source-commit must be a full lowercase Git SHA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if "graph_task_spacing_20261029" not in output.parts:
        raise ValueError("output must be inside the registered task campaign")

    torch.npu.set_device(args.device_id)
    stream = torch.npu.Stream(device=args.device_id)
    pool = torch.npu.graph_pool_handle()
    with torch.npu.stream(stream):
        left = torch.zeros(4096, device="npu:0", dtype=torch.float32)
        right = torch.empty_like(left)
        increment = torch.ones_like(left)
        graphs = {}
        for count in CHAIN_LENGTHS:
            _chain(left, right, increment, count)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=pool, stream=stream):
                _chain(left, right, increment, count)
            for _ in range(3):
                graph.replay()
            stream.synchronize()
            graphs[count] = graph

        samples = []
        for trial in range(TRIALS):
            lengths = CHAIN_LENGTHS if trial % 2 == 0 else CHAIN_LENGTHS[::-1]
            for count in lengths:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record(stream)
                host_start_ns = time.perf_counter_ns()
                graphs[count].replay()
                host_submit_ns = time.perf_counter_ns() - host_start_ns
                end.record(stream)
                end.synchronize()
                samples.append({
                    "trial": trial,
                    "chain_length": count,
                    "event_elapsed_ns": float(start.elapsed_time(end)) * 1_000_000,
                    "host_submit_ns": host_submit_ns,
                })
    if any(row["event_elapsed_ns"] <= 0 for row in samples):
        raise RuntimeError("device event timing must be positive")
    summary = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "device_id": args.device_id,
        "source_commit": args.source_commit,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mechanism": "single-stream ACL Graph FP32 add task chain",
        "tensor_numel": 4096,
        "chain_lengths": list(CHAIN_LENGTHS),
        "trials_per_length": TRIALS,
        "median_event_elapsed_ns": {
            str(count): statistics.median(row["event_elapsed_ns"] for row in samples
                                          if row["chain_length"] == count)
            for count in CHAIN_LENGTHS
        },
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "median_event_elapsed_ns":
                      summary["median_event_elapsed_ns"]}))


if __name__ == "__main__":
    main()
