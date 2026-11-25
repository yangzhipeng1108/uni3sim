"""Independent preallocated eager Add chain for 910B device gap screening."""

import argparse
import json
import os
import time

import torch
import torch_npu  # noqa: F401 - registers NPU backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--chain", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    args = parser.parse_args()
    if args.chain not in (64, 256) or args.repeats not in (3, 5):
        raise ValueError("probe requires 64/256 calls and 3/5 repeats")

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    left = torch.ones(4096, dtype=torch.float32, device=device)
    right = torch.zeros_like(left)
    output = torch.empty_like(left)
    stream = torch.npu.Stream(device=args.device)
    torch.npu.set_stream(stream)

    def chain() -> None:
        for _ in range(args.chain):
            torch.add(left, right, out=output)

    for _ in range(3):
        chain()
        stream.synchronize()

    graph = None
    if args.mode == "graph":
        graph = torch.npu.NPUGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.npu.graph(graph, pool=pool, stream=stream):
            chain()
        graph.replay()
        stream.synchronize()

    rows = []
    for repeat in range(args.repeats):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        stream.synchronize()
        start_event.record(stream)
        host_start = time.perf_counter_ns()
        if args.mode == "eager":
            chain()
        else:
            graph.replay()
        host_end = time.perf_counter_ns()
        end_event.record(stream)
        stream.synchronize()
        rows.append({
            "repeat": repeat,
            "host_submit_ns": host_end - host_start,
            "device_elapsed_ns": start_event.elapsed_time(end_event) * 1e6,
        })
    print(json.dumps({
        "schema_version": 1,
        "kind": "ascend_910b_eager_device_gap_independent_probe",
        "visible_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "chain": args.chain,
        "repeats": args.repeats,
        "mode": args.mode,
        "operation": "preallocated torch.add float32 4096 elements",
        "rows": rows,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
