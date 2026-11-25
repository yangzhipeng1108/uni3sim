#!/usr/bin/env python3
"""Measure vLLM collective Host service with completion outside the window."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import init_model_parallel_group


OPERATIONS = ("all_reduce", "all_gather")
MESSAGE_BYTES = (65536, 1048576)
CALL_COUNTS = (1, 8, 32, 128)
TRIALS = 6


def _slope(points):
    x_mean = statistics.mean(point[0] for point in points)
    y_mean = statistics.mean(point[1] for point in points)
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in points
    ) / sum((x - x_mean) ** 2 for x, _ in points)


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware-id", choices=("nvidia_a800", "nvidia_h200", "metax_c500"),
        required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eager-enqueue-us", type=float, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("benchmark requires exactly two ranks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    group = init_model_parallel_group(
        [list(range(world_size))], local_rank, backend="nccl",
        use_message_queue_broadcaster=False,
        group_name="host_prepare_microbench",
    )
    try:
        samples = []
        for operation in OPERATIONS:
            for message_bytes in MESSAGE_BYTES:
                element_count = message_bytes // 2
                buffers = {
                    count: tuple(
                        torch.full(
                            (element_count,), rank + 1,
                            dtype=torch.bfloat16, device="cuda",
                        )
                        for _ in range(count)
                    )
                    for count in CALL_COUNTS
                }
                for count in CALL_COUNTS:
                    for _ in range(3):
                        for tensor in buffers[count]:
                            if operation == "all_reduce":
                                group.all_reduce(tensor)
                            else:
                                group.all_gather(tensor, dim=0)
                        torch.cuda.synchronize()
                        dist.barrier()
                    for trial in range(TRIALS):
                        torch.cuda.synchronize()
                        dist.barrier()
                        started = time.perf_counter_ns()
                        outputs = []
                        for tensor in buffers[count]:
                            if operation == "all_reduce":
                                outputs.append(group.all_reduce(tensor))
                            else:
                                outputs.append(group.all_gather(tensor, dim=0))
                        ended = time.perf_counter_ns()
                        torch.cuda.synchronize()
                        samples.append({
                            "operation": operation,
                            "message_bytes": message_bytes,
                            "call_count": count,
                            "trial": trial + 1,
                            "host_interval_ns": ended - started,
                        })
        fits = []
        for operation in OPERATIONS:
            for message_bytes in MESSAGE_BYTES:
                points = []
                for count in CALL_COUNTS:
                    values = [
                        row["host_interval_ns"] for row in samples
                        if row["operation"] == operation
                        and row["message_bytes"] == message_bytes
                        and row["call_count"] == count
                    ]
                    points.append((count, statistics.median(values)))
                service_ns = _slope(points)
                fits.append({
                    "operation": operation,
                    "message_bytes": message_bytes,
                    "host_service_ns_per_call": service_ns,
                    "runtime_prepare_ns_per_call": max(
                        0.0, service_ns - args.eager_enqueue_us * 1000.0,
                    ),
                })
        result = {
            "schema_version": 1,
            "hardware_id": args.hardware_id,
            "rank": rank,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(local_rank),
            },
            "contract": {
                "boundary": "vllm_collective_api_async_host_interval",
                "completion_sync_excluded": True,
                "message_bytes": list(MESSAGE_BYTES),
                "call_counts": list(CALL_COUNTS),
                "trials": TRIALS,
                "eager_enqueue_us": args.eager_enqueue_us,
            },
            "fits": fits,
            "system_median_runtime_prepare_ns": {
                operation: statistics.median(
                    fit["runtime_prepare_ns_per_call"] for fit in fits
                    if fit["operation"] == operation
                )
                for operation in OPERATIONS
            },
            "samples": samples,
        }
        _write_json(args.output_dir / f"summary-rank-{rank}.json", result)
        if rank == 0:
            print(json.dumps(result["system_median_runtime_prepare_ns"], sort_keys=True))
    finally:
        group.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
