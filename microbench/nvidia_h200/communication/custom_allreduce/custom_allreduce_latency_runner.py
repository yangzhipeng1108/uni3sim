#!/usr/bin/env python3
"""Measure vLLM custom all-reduce service time as CUDA-graph nodes.

The vLLM tensor-parallel all-reduce selects the IPC custom
all-reduce kernel for inputs below its ``max_size`` (8 MiB) on a fully
NVLink-connected node, and it captures that kernel into the decode CUDA
graph.  This runner isolates that mechanism: each rank captures one graph
that chains ``CHAIN_LENGTH`` dependent custom all-reduce calls on a BF16
``tokens x hidden`` tensor, replays it after a cross-rank barrier, and
reports the per-node service time from CUDA events.  NCCL all-reduce is
measured the same way for reference.  No Serving workload, production
kernel duration or evaluation data is consumed.

The probe constructs ``CustomAllreduce`` standalone on a gloo group with its
default 8 MiB ``max_size`` (``symm_mem_enabled=False``), so the same
mechanism is measured on vLLM 0.24 and 0.30 and at 2/4/8 ranks.  Which
backend vLLM 0.30 actually dispatches a TP all-reduce to is measured
separately by ``../tp_collectives/vllm_tp_collectives_runner.py``.

Usage (one process tree owns every listed device; hold every GPU's
``gpu<N>.proc.lock``):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
        custom_allreduce_latency_runner.py --world-sizes 2 4 8 \
        --output-dir <run_dir> --source-commit <sha>
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


HARDWARE_ID = "nvidia_h200"
EXPECTED_DEVICE_NAME = "NVIDIA H200"
HIDDEN = 7168
TOKEN_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
                256, 384, 512)
CHAIN_LENGTH = 64
WARMUP_REPLAYS = 5
TRIALS = 30
CUSTOM_AR_MAX_BYTES = 8192 * 1024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph_chain(op, source: torch.Tensor):
    """Capture CHAIN_LENGTH dependent all-reduce calls into one graph."""

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            current = source
            for _ in range(CHAIN_LENGTH):
                current = op(current, capturing=False)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        current = source
        for _ in range(CHAIN_LENGTH):
            current = op(current, capturing=True)
    return graph


def _time_graph(graph, group) -> list:
    for _ in range(WARMUP_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(TRIALS):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e6 / CHAIN_LENGTH)
    return samples


def _worker(rank: int, world_size: int, port: int, result_path: str):
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    torch.cuda.set_device(rank)
    device_name = torch.cuda.get_device_name(rank)
    if device_name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected device {device_name!r}")
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world_size)
    cpu_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    custom = CustomAllreduce(cpu_group, torch.device(f"cuda:{rank}"))
    if custom.disabled:
        raise RuntimeError("vLLM custom all-reduce is disabled")

    rows = []
    for tokens in TOKEN_COUNTS:
        payload_bytes = tokens * HIDDEN * 2
        if payload_bytes >= CUSTOM_AR_MAX_BYTES:
            continue
        t_start = time.time()
        source = torch.ones(
            tokens, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")

        def custom_op(tensor, capturing):
            output = custom.custom_all_reduce(tensor)
            if output is None:
                raise RuntimeError("input rejected by custom all-reduce")
            return output

        with custom.capture():
            custom_graph = _graph_chain(custom_op, source)
        custom_samples = _time_graph(custom_graph, cpu_group)

        def nccl_op(tensor, capturing):
            output = tensor.clone()
            dist.all_reduce(output, group=nccl_group)
            return output

        nccl_graph = _graph_chain(nccl_op, source)
        nccl_samples = _time_graph(nccl_graph, cpu_group)
        rows.append({
            "tokens": tokens,
            "hidden": HIDDEN,
            "payload_bytes": payload_bytes,
            "custom_all_reduce_ns_per_node": custom_samples,
            "nccl_all_reduce_plus_clone_ns_per_node": nccl_samples,
            "t_start_unix": t_start,
            "t_end_unix": time.time(),
        })
        del custom_graph, nccl_graph
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

    Path(result_path).write_text(json.dumps({
        "rank": rank, "device_name": device_name, "rows": rows}) + "\n")
    custom.close()
    dist.destroy_process_group()


def _environment() -> dict:
    import vllm

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout.split()[0]
    return {
        "hardware_id": HARDWARE_ID,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "driver_version": driver,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "vllm_version": vllm.__version__,
        "python": platform.python_version(),
        "host_cpu": platform.processor(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29611)
    parser.add_argument("--source-commit", required=True,
                        help="repository commit the runner was copied from")
    args = parser.parse_args()
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if max(args.world_sizes) > len(visible):
        raise ValueError("world size exceeds CUDA_VISIBLE_DEVICES")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve()
    commit = args.source_commit
    report = {
        "schema_version": 1,
        "mechanism": "vllm_custom_all_reduce_cuda_graph_node_service",
        "timer_domain": "cuda_event_around_graph_replay_divided_by_chain",
        "chain_length": CHAIN_LENGTH,
        "trials": TRIALS,
        "warmup_replays": WARMUP_REPLAYS,
        "dtype": "bfloat16",
        "custom_all_reduce_max_bytes": CUSTOM_AR_MAX_BYTES,
        "production_or_evaluation_duration_consumed": False,
        "runner": {"path": str(runner), "sha256": _sha256(runner),
                   "git_commit": commit},
        "command": " ".join([sys.executable] + sys.argv),
        "environment": _environment(),
        "world_sizes": {},
    }
    for index, world_size in enumerate(args.world_sizes):
        paths = [str(args.output_dir / f"ws{world_size}_rank{rank}.json")
                 for rank in range(world_size)]
        context = mp.get_context("spawn")
        processes = [
            context.Process(target=_worker,
                            args=(rank, world_size, args.port + index,
                                  paths[rank]))
            for rank in range(world_size)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"world size {world_size} worker failed: "
                    f"exit {process.exitcode}")
        per_rank = [json.loads(Path(path).read_text()) for path in paths]
        summary = []
        for row_index, row in enumerate(per_rank[0]["rows"]):
            # A collective completes when its slowest rank completes.
            custom_median = max(
                statistics.median(
                    rank_rows["rows"][row_index][
                        "custom_all_reduce_ns_per_node"])
                for rank_rows in per_rank)
            nccl_median = max(
                statistics.median(
                    rank_rows["rows"][row_index][
                        "nccl_all_reduce_plus_clone_ns_per_node"])
                for rank_rows in per_rank)
            summary.append({
                "tokens": row["tokens"],
                "payload_bytes": row["payload_bytes"],
                "custom_all_reduce_ns": custom_median,
                "nccl_all_reduce_plus_clone_ns": nccl_median,
                "t_start_unix": row["t_start_unix"],
                "t_end_unix": row["t_end_unix"],
            })
        report["world_sizes"][str(world_size)] = {
            "per_rank_raw": paths, "summary": summary}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for world_size, block in report["world_sizes"].items():
        print(f"world_size={world_size}")
        for row in block["summary"]:
            print(f"  tokens={row['tokens']:4d} bytes={row['payload_bytes']:8d} "
                  f"custom={row['custom_all_reduce_ns']/1e3:7.2f}us "
                  f"nccl+clone={row['nccl_all_reduce_plus_clone_ns']/1e3:7.2f}us")


if __name__ == "__main__":
    main()
