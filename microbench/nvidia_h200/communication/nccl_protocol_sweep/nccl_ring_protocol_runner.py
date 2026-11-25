#!/usr/bin/env python3
"""Measure two-rank NCCL Ring collectives per protocol as CUDA-graph nodes.

The ProtoBridge NCCL record prices one collective as ``protocol startup +
ring_phase_count * protocol phase_fixed`` plus a transfer term, with an
optional Simple all-reduce ring bandwidth per participant count.  This
runner produces the independent mechanism evidence for those inputs on a
two-GPU H200 pair: for each NCCL protocol it starts a fresh process pair with
``NCCL_ALGO=Ring`` and ``NCCL_PROTO=<protocol>`` (NCCL reads both once per
process), captures ``chain`` back-to-back collectives on a BF16 buffer into
one CUDA graph, replays it after a cross-rank barrier and reports the
per-node service from CUDA events (median of ``TRIALS`` per rank, slowest
rank).  The ``auto`` pass leaves NCCL's own selection in place and keeps the
NCCL ``TUNING``/``COLL`` log so the chosen algorithm and protocol per size are
recorded.  ``all_reduce`` sizes are the per-rank buffer; ``all_gather``
sizes are one rank's input shard, the payload Midlayer records.  No Serving
workload, production kernel duration or evaluation data is consumed.

Usage (one process tree owns both devices; hold both GPUs' proc locks):
    CUDA_VISIBLE_DEVICES=4,5 python nccl_ring_protocol_runner.py \
        --output-dir <run_dir> --source-commit <sha>
"""

import argparse
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
WORLD_SIZE = 2
PROTOCOLS = ("auto", "LL", "LL128", "Simple")
OPERATIONS = ("all_reduce", "all_gather")
MESSAGE_BYTES = tuple(2 ** exponent for exponent in range(8, 29))
LONG_CHAIN = 32
SHORT_CHAIN = 8
SHORT_CHAIN_FROM_BYTES = 8 * 1024 * 1024
WARMUP_REPLAYS = 3
TRIALS = 20
NCCL_ENVIRONMENT = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_DEBUG",
                    "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chain_length(message_bytes: int) -> int:
    return SHORT_CHAIN if message_bytes >= SHORT_CHAIN_FROM_BYTES else LONG_CHAIN


def _collective(operation: str, source: torch.Tensor, gathered: torch.Tensor,
                group) -> None:
    if operation == "all_reduce":
        dist.all_reduce(source, group=group)
    elif operation == "all_gather":
        dist.all_gather_into_tensor(gathered, source, group=group)
    else:
        raise ValueError(f"unsupported operation: {operation!r}")


def _graph(operation: str, chain: int, source, gathered, group):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _collective(operation, source, gathered, group)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(chain):
            _collective(operation, source, gathered, group)
    return graph


def _time_graph(graph, chain: int, barrier_group) -> list:
    for _ in range(WARMUP_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(TRIALS):
        dist.barrier(group=barrier_group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e6 / chain)
    return samples


def _worker(rank: int, port: int, result_path: str):
    torch.cuda.set_device(rank)
    device_name = torch.cuda.get_device_name(rank)
    if device_name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected device {device_name!r}")
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=WORLD_SIZE)
    cpu_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    rows = []
    for operation in OPERATIONS:
        for message_bytes in MESSAGE_BYTES:
            chain = _chain_length(message_bytes)
            # Zeros keep every all-reduce result finite across the chain.
            source = torch.zeros(
                message_bytes // 2, dtype=torch.bfloat16, device=f"cuda:{rank}")
            gathered = torch.empty(
                WORLD_SIZE * source.numel(), dtype=torch.bfloat16,
                device=f"cuda:{rank}")
            t_start = time.time()
            graph = _graph(operation, chain, source, gathered, nccl_group)
            samples = _time_graph(graph, chain, cpu_group)
            rows.append({
                "operation": operation,
                "message_bytes": message_bytes,
                "chain_length": chain,
                "ns_per_node": samples,
                "t_start_unix": t_start,
                "t_end_unix": time.time(),
            })
            del graph, source, gathered
            torch.cuda.synchronize()
            dist.barrier(group=cpu_group)
    Path(result_path).write_text(json.dumps({
        "rank": rank, "device_name": device_name,
        "nccl_environment": {name: os.environ[name]
                             for name in NCCL_ENVIRONMENT if name in os.environ},
        "rows": rows}) + "\n")
    dist.destroy_process_group()


def _environment() -> dict:
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
        "python": platform.python_version(),
    }


def _set_protocol_environment(protocol: str, output_dir: Path) -> None:
    for name in NCCL_ENVIRONMENT:
        if name in os.environ:
            del os.environ[name]
    if protocol != "auto":
        os.environ["NCCL_ALGO"] = "Ring"
        os.environ["NCCL_PROTO"] = protocol
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,TUNING,COLL"
    os.environ["NCCL_DEBUG_FILE"] = str(
        output_dir / f"nccl_{protocol}.%h.%p.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29711)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS,
                        default=list(PROTOCOLS))
    parser.add_argument("--source-commit", required=True,
                        help="repository commit the runner was copied from")
    args = parser.parse_args()
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) != WORLD_SIZE:
        raise ValueError("CUDA_VISIBLE_DEVICES must list exactly two GPUs")
    for name in NCCL_ENVIRONMENT:
        if name in os.environ:
            raise ValueError(f"{name} is set by this runner per protocol")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve()
    report = {
        "schema_version": 1,
        "mechanism": "nccl_ring_protocol_cuda_graph_node_service",
        "timer_domain": "cuda_event_around_graph_replay_divided_by_chain",
        "world_size": WORLD_SIZE,
        "dtype": "bfloat16",
        "message_bytes": list(MESSAGE_BYTES),
        "message_bytes_semantics": {
            "all_reduce": "per_rank_buffer_bytes",
            "all_gather": "per_rank_input_shard_bytes",
        },
        "chain_length": {"below_bytes": SHORT_CHAIN_FROM_BYTES,
                         "long": LONG_CHAIN, "short": SHORT_CHAIN},
        "trials": TRIALS,
        "warmup_replays": WARMUP_REPLAYS,
        "statistic": "max_over_ranks_of_per_rank_median",
        "production_or_evaluation_duration_consumed": False,
        "runner": {"path": str(runner), "sha256": _sha256(runner),
                   "git_commit": args.source_commit},
        "command": " ".join([sys.executable] + sys.argv),
        "environment": _environment(),
        "protocols": {},
    }
    for index, protocol in enumerate(args.protocols):
        _set_protocol_environment(protocol, args.output_dir)
        paths = [str(args.output_dir / f"{protocol}_rank{rank}.json")
                 for rank in range(WORLD_SIZE)]
        context = mp.get_context("spawn")
        processes = [
            context.Process(target=_worker,
                            args=(rank, args.port + index, paths[rank]))
            for rank in range(WORLD_SIZE)]
        started = time.time()
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"protocol {protocol} worker failed: exit {process.exitcode}")
        per_rank = [json.loads(Path(path).read_text()) for path in paths]
        summary = []
        for row_index, row in enumerate(per_rank[0]["rows"]):
            for rank_rows in per_rank:
                other = rank_rows["rows"][row_index]
                if (other["operation"], other["message_bytes"]) != (
                        row["operation"], row["message_bytes"]):
                    raise RuntimeError("rank rows are not aligned")
            summary.append({
                "operation": row["operation"],
                "message_bytes": row["message_bytes"],
                "ns_per_call": max(
                    statistics.median(rank_rows["rows"][row_index]["ns_per_node"])
                    for rank_rows in per_rank),
            })
        report["protocols"][protocol] = {
            "nccl_environment": per_rank[0]["nccl_environment"],
            "per_rank_raw": paths,
            "wall_s": time.time() - started,
            "summary": summary,
        }
        for row in summary:
            print(f"{protocol:6s} {row['operation']:10s} "
                  f"bytes={row['message_bytes']:10d} "
                  f"{row['ns_per_call'] / 1e3:10.2f} us", flush=True)
    for name in NCCL_ENVIRONMENT:
        if name in os.environ:
            del os.environ[name]
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
