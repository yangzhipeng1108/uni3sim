#!/usr/bin/env python3
"""Measure FlashInfer trtllm all-reduce + residual RMSNorm fusion service.

vLLM 0.24 on a single NVSwitch H200 node replaces the decoder's TP
all-reduce and the following residual-add RMSNorm with one FlashInfer
``allreduce_fusion`` call (trtllm backend, pattern kARResidualRMSNorm,
BF16, one-shot Lamport while the payload is below its one-shot limit).  This
runner isolates that collective mechanism: each rank captures one CUDA graph
chaining ``CHAIN_LENGTH`` dependent fused calls on a ``tokens x hidden``
tensor, replays it after a cross-rank barrier, and reports the per-node
service time from CUDA events.  No Serving workload, production kernel
duration or evaluation data is consumed.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python flashinfer_allreduce_fusion_runner.py \
        --world-sizes 2 4 --output-dir <run_dir> --source-commit <sha>
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


HARDWARE_ID = "nvidia_h200"
EXPECTED_DEVICE_NAME = "NVIDIA H200"
HIDDEN = 7168
TOKEN_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
CHAIN_LENGTH = 64
WARMUP_REPLAYS = 5
TRIALS = 30
RMS_EPS = 1e-6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker(rank, world_size, port, result_path, token_counts):
    import flashinfer.comm as flashinfer_comm
    from flashinfer.comm.mnnvl import TorchDistBackend

    torch.cuda.set_device(rank)
    device_name = torch.cuda.get_device_name(rank)
    if device_name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected device {device_name!r}")
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))
    group = dist.group.WORLD
    cpu_group = dist.new_group(backend="gloo")
    workspace = flashinfer_comm.create_allreduce_fusion_workspace(
        backend="trtllm", world_size=world_size, rank=rank,
        max_token_num=max(token_counts), hidden_dim=HIDDEN, dtype=torch.bfloat16,
        comm_backend=TorchDistBackend(group=group), group=group)
    pattern = flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm
    layout = flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4
    gamma = torch.ones(HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")

    rows = []
    for tokens in token_counts:
        hidden_states = torch.ones(
            tokens, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")
        residual = torch.zeros_like(hidden_states)
        norm_out = torch.empty_like(hidden_states)
        residual_out = torch.empty_like(hidden_states)

        def fused():
            flashinfer_comm.allreduce_fusion(
                input=hidden_states, workspace=workspace, pattern=pattern,
                launch_with_pdl=True, output=None, residual_out=residual_out,
                norm_out=norm_out, quant_out=None, scale_out=None,
                residual_in=residual, rms_gamma=gamma, rms_eps=RMS_EPS,
                scale_factor=None, layout_code=layout, use_oneshot=None,
                fp32_acc=False, trigger_completion_at_end=True)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3 * CHAIN_LENGTH):
                fused()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(CHAIN_LENGTH):
                fused()
        for _ in range(WARMUP_REPLAYS):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(TRIALS):
            dist.barrier(group=cpu_group)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e6 / CHAIN_LENGTH)
        rows.append({"tokens": tokens, "hidden": HIDDEN,
                     "payload_bytes": tokens * HIDDEN * 2,
                     "fused_ns_per_node": samples})
        del graph
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
    Path(result_path).write_text(json.dumps(
        {"rank": rank, "device_name": device_name, "rows": rows}) + "\n")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29711)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--token-counts", type=int, nargs="+",
                        default=list(TOKEN_COUNTS),
                        help="token rows per call (workspace sized to the max)")
    args = parser.parse_args()
    token_counts = tuple(args.token_counts)
    if min(token_counts) < 1:
        raise ValueError("token counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import importlib.util
    import flashinfer
    # The framework only labels provenance (vLLM 0.24 or SGLang 0.5.12 env);
    # exactly one of them is installed in each measurement environment.
    frameworks = [name for name in ("vllm", "sglang")
                  if importlib.util.find_spec(name) is not None]
    if len(frameworks) != 1:
        raise RuntimeError(f"expected exactly one serving framework, got {frameworks}")
    framework = __import__(frameworks[0])
    runner = Path(__file__).resolve()
    report = {
        "schema_version": 1,
        "mechanism": "flashinfer_trtllm_allreduce_residual_rmsnorm_graph_node_service",
        "timer_domain": "cuda_event_around_graph_replay_divided_by_chain",
        "chain_length": CHAIN_LENGTH, "trials": TRIALS,
        "warmup_replays": WARMUP_REPLAYS, "dtype": "bfloat16",
        "pattern": "kARResidualRMSNorm", "backend": "trtllm",
        "max_token_num": max(token_counts), "token_counts": list(token_counts),
        "production_or_evaluation_duration_consumed": False,
        "runner": {"path": str(runner), "sha256": _sha256(runner),
                   "git_commit": args.source_commit},
        "command": " ".join([sys.executable] + sys.argv),
        "environment": {
            "hardware_id": HARDWARE_ID,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "flashinfer_version": flashinfer.__version__,
            f"{frameworks[0]}_version": framework.__version__,
            "python": platform.python_version()},
        "world_sizes": {},
    }
    for index, world_size in enumerate(args.world_sizes):
        paths = [str(args.output_dir / f"ws{world_size}_rank{r}.json")
                 for r in range(world_size)]
        context = mp.get_context("spawn")
        processes = [context.Process(target=_worker,
                                     args=(r, world_size, args.port + index, paths[r],
                                           token_counts))
                     for r in range(world_size)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"world size {world_size} worker failed")
        per_rank = [json.loads(Path(p).read_text()) for p in paths]
        summary = [{
            "tokens": row["tokens"], "payload_bytes": row["payload_bytes"],
            # A collective completes when its slowest rank completes.
            "fused_ns": max(statistics.median(r["rows"][i]["fused_ns_per_node"])
                            for r in per_rank)}
            for i, row in enumerate(per_rank[0]["rows"])]
        report["world_sizes"][str(world_size)] = {"per_rank_raw": paths,
                                                   "summary": summary}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for world_size, block in report["world_sizes"].items():
        print(f"world_size={world_size}")
        for row in block["summary"]:
            print(f"  tokens={row['tokens']:4d} fused={row['fused_ns']/1e3:7.2f}us")


if __name__ == "__main__":
    main()
