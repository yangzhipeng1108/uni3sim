#!/usr/bin/env python3
"""Measure the TP collectives vLLM 0.30 actually dispatches, as graph nodes.

vLLM 0.30 no longer routes a TP all-reduce straight to its IPC custom
all-reduce.  ``CudaCommunicator.all_reduce`` tries, in order, FlashInfer
all-reduce (``VLLM_ALLREDUCE_USE_FLASHINFER`` defaults on; SM90 TP8 accepts
payloads up to 0.5 MiB), NCCL symmetric memory (off unless
``VLLM_USE_NCCL_SYMM_MEM``), the custom all-reduce (its ``max_size`` shrinks
to 256 KiB on SM90 TP8 once torch symmetric memory is enabled), torch
symmetric-memory multimem all-reduce (up to 64 MiB) and finally PyNccl.  At
compile time the ``fuse_allreduce_rms`` pass (default on for SM90, TP > 1)
replaces all-reduce + residual RMSNorm with one FlashInfer
``allreduce_fusion`` call for token counts up to
``0.5 MiB / (hidden * 2)``.

This runner initialises vLLM's own distributed state exactly as a TP worker
does (``init_distributed_environment`` + ``ensure_model_parallel_initialized``)
so every selection decision is vLLM's, records which backend each payload is
dispatched to (by evaluating the same predicates in the same order as
``CudaCommunicator.all_reduce``), and times per-node service with the same
probe as the four-rank custom all-reduce reference: one CUDA graph chaining
dependent collectives, replayed after a cross-rank barrier, CUDA events,
median of ``TRIALS`` per rank, slowest rank.  Series:

* ``tp_all_reduce_dispatch``: ``get_tp_group().all_reduce`` (vLLM's own
  dispatch) on a BF16 ``tokens x hidden`` tensor.
* ``flashinfer_fused_ar_residual_rmsnorm``: vLLM's
  ``call_trtllm_fused_allreduce_norm`` with the pass's kwargs
  (kARResidualRMSNorm, PDL, fp32 accumulation, pass max_token_num).
* ``flashinfer_fused_ar_residual_rmsnorm_model_op``: the same kernel as
  called by ``vllm/models/common/ops/fused_allreduce_rms_norm.py`` (separate
  ``norm_out``), the path DeepseekV32 / GlmMoeDsa take because they run under
  breakable CUDA graphs with compilation mode NONE.
* ``pynccl_all_reduce``: PyNccl all-reduce forced for every payload.
* ``tp_all_gather_dim0`` / ``tp_reduce_scatter_dim0``: vLLM's dispatch.
* ``tp_all_gather_logits``: ``get_tp_group().all_gather(dim=-1)`` on the
  vocab-parallel logits shape ``tokens x vocab/TP`` (LogitsProcessor path).

No Serving workload, production kernel duration or evaluation data is
consumed.

Usage (one process tree owns every listed device; hold every GPU lock):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python vllm_tp_collectives_runner.py \
        --world-sizes 8 --hidden 7168 --output-dir <run_dir> \
        --source-commit <sha>
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
EXPECTED_VLLM_VERSION_PREFIX = "0.30.0"
# Decode grid identical to the four-rank custom all-reduce reference, then
# prefill-chunk sizes up to the default max_num_batched_tokens scale.
DECODE_TOKEN_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
                       192, 256, 384, 512)
PREFILL_TOKEN_COUNTS = (1024, 2048, 4096, 8192)
QUICK_TOKEN_COUNTS = (1, 512)
DECODE_CHAIN_LENGTH = 64
PREFILL_CHAIN_LENGTH = 16
WARMUP_REPLAYS = 5
TRIALS = 30
QUICK_TRIALS = 3
RMS_EPS = 1e-5
LOGITS_VOCAB = 154880  # GLM-5.2 vocab_size (configs/model/glm-5.2-fp8.json)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chain_length(tokens: int) -> int:
    return DECODE_CHAIN_LENGTH if tokens <= 512 else PREFILL_CHAIN_LENGTH


def _dispatched_all_reduce_backend(comm, tensor) -> str:
    """Mirror CudaCommunicator.all_reduce's predicate order (vLLM 0.30.0)."""
    from vllm.distributed.device_communicators.all_reduce_utils import (
        should_nccl_symm_mem_allreduce,
    )

    fi = comm.fi_ar_comm
    use_fi = fi is not None and not fi.disabled and fi.should_use_fi_ar(tensor)
    if (comm.pynccl_comm is not None and not use_fi
            and should_nccl_symm_mem_allreduce(comm.pynccl_comm.world_size,
                                               tensor)):
        return "NCCL_SYMM_MEM"
    if comm.qr_comm is not None and not comm.qr_comm.disabled \
            and comm.qr_comm.should_quick_allreduce(tensor):
        return "QUICK_REDUCE"
    if comm.fi_pcie_ipc_ar_comm is not None \
            and comm.fi_pcie_ipc_ar_comm.should_use(tensor):
        return "FLASHINFER_PCIE_IPC"
    if use_fi:
        return "FLASHINFER"
    if comm.use_aiter_allreduce:
        raise RuntimeError("AITER all-reduce enabled on CUDA; unexpected")
    if comm.ca_comm is not None and not comm.ca_comm.disabled \
            and comm.ca_comm.should_custom_ar(tensor):
        return "CUSTOM"
    if comm.symm_mem_comm is not None \
            and comm.symm_mem_comm.should_use_symm_mem(tensor):
        return "SYMM_MEM_MULTIMEM" if comm.symm_mem_comm.world_size in \
            comm.symm_mem_comm._WORLD_SIZES_MULTIMEM[
                comm.symm_mem_comm.device_capability] else "SYMM_MEM_TWO_SHOT"
    if comm.pynccl_comm is None or comm.pynccl_comm.disabled:
        raise RuntimeError("no PyNccl communicator; vLLM would use torch.dist")
    return "PYNCCL"


def _capture_chain(step, chain_length: int, device):
    """Warm up then capture ``chain_length`` calls of ``step`` in one graph,
    inside vLLM's own graph_capture context (custom AR buffer registration)."""
    from vllm.distributed.parallel_state import graph_capture

    with graph_capture(device) as context:
        for _ in range(3):
            step.reset()
            for _ in range(chain_length):
                step()
        torch.cuda.synchronize()
        step.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=context.stream):
            for _ in range(chain_length):
                step()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph, chain_length: int, trials: int, cpu_group) -> list:
    for _ in range(WARMUP_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        dist.barrier(group=cpu_group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e6 / chain_length)
    return samples


class _Chain:
    """A dependent chain: each call consumes the previous call's output."""

    def __init__(self, fn, source):
        self.fn = fn
        self.source = source
        self.current = source

    def reset(self):
        self.current = self.source

    def __call__(self):
        output = self.fn(self.current)
        if output is None:
            raise RuntimeError("collective rejected its input")
        self.current = output


class _Repeat:
    """Repeated calls on one fixed input; outputs are discarded."""

    def __init__(self, fn, source):
        self.fn = fn
        self.source = source

    def reset(self):
        return None

    def __call__(self):
        if self.fn(self.source) is None:
            raise RuntimeError("collective rejected its input")


class _InPlace:
    """Repeated in-place call on fixed buffers (fused AR+RMSNorm mutates its
    input and residual, so successive calls are data dependent)."""

    def __init__(self, fn):
        self.fn = fn

    def reset(self):
        return None

    def __call__(self):
        self.fn()


def _worker(rank: int, world_size: int, port: int, hidden: int,
            token_counts: tuple, trials: int, result_path: str):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    device_name = torch.cuda.get_device_name(rank)
    if device_name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected device {device_name!r}")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size, rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank, backend="nccl")
        ensure_model_parallel_initialized(world_size, 1)
    tp = get_tp_group()
    comm = tp.device_communicator
    cpu_group = tp.cpu_group

    from vllm.compilation.passes.fusion import allreduce_rms_fusion as arf
    from vllm.config.compilation import PassConfig
    from vllm.distributed.device_communicators import flashinfer_all_reduce
    import flashinfer.comm as flashinfer_comm

    fi_backend_resolved, fi_allow_trtllm_fallback = (
        flashinfer_all_reduce._resolve_fi_ar_backend())
    fusion_max_bytes = PassConfig().flashinfer_max_size(world_size)
    if fusion_max_bytes is None:
        raise RuntimeError(f"no FlashInfer fusion size for world {world_size}")
    fusion_max_tokens = fusion_max_bytes // (hidden * 2)
    # Create the process-global FlashInfer workspace the way the fusion pass
    # does at compile time (it runs before any standalone all-reduce).
    workspace = flashinfer_all_reduce.get_fi_ar_workspace(
        world_size=world_size, rank=rank, max_token_num=fusion_max_tokens,
        hidden_dim=hidden, dtype=torch.bfloat16, group=cpu_group)
    if workspace is None:
        raise RuntimeError("FlashInfer all-reduce workspace creation failed")
    selection = {
        "use_custom_allreduce": comm.use_custom_allreduce,
        "use_torch_symm_mem": comm.use_torch_symm_mem,
        "use_flashinfer_allreduce": comm.use_flashinfer_allreduce,
        "use_flashinfer_pcie_ipc_allreduce":
            comm.use_flashinfer_pcie_ipc_allreduce,
        "flashinfer_ar_disabled": comm.fi_ar_comm.disabled,
        "flashinfer_ar_max_workspace_bytes":
            comm.fi_ar_comm.max_workspace_size,
        "flashinfer_backend_resolved": fi_backend_resolved,
        "flashinfer_allow_trtllm_fallback": fi_allow_trtllm_fallback,
        "flashinfer_workspace_backend": workspace.backend,
        "fusion_max_bytes": fusion_max_bytes,
        "fusion_max_token_num": fusion_max_tokens,
        "custom_ar_disabled": comm.ca_comm.disabled,
        "custom_ar_max_bytes": comm.ca_comm.max_size,
        "custom_ar_fully_connected": comm.ca_comm.fully_connected,
        "symm_mem_disabled": comm.symm_mem_comm.disabled,
        "symm_mem_max_bytes": (None if comm.symm_mem_comm.disabled
                               else comm.symm_mem_comm.max_size),
        "pynccl_disabled": comm.pynccl_comm.disabled,
    }

    rows = []

    def record(series, tokens, shape, payload_bytes, backend, step,
               extra):
        chain = _chain_length(tokens)
        t_start = time.time()
        graph = _capture_chain(step, chain, device)
        samples = _time_graph(graph, chain, trials, cpu_group)
        t_end = time.time()
        row = {"series": series, "tokens": tokens, "shape": list(shape),
               "payload_bytes": payload_bytes, "backend": backend,
               "chain_length": chain, "ns_per_node": samples,
               "t_start_unix": t_start, "t_end_unix": t_end}
        row.update(extra)
        rows.append(row)
        del graph
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

    for tokens in token_counts:
        source = torch.ones(tokens, hidden, dtype=torch.bfloat16,
                            device=device)
        payload = tokens * hidden * 2

        # 1. vLLM's own all-reduce dispatch.
        backend = _dispatched_all_reduce_backend(comm, source)
        record("tp_all_reduce_dispatch", tokens, source.shape, payload,
               backend, _Chain(tp.all_reduce, source), {})

        # 2. Fused all-reduce + residual RMSNorm, only where the pass applies.
        if tokens <= fusion_max_tokens:
            hidden_states = torch.ones(tokens, hidden, dtype=torch.bfloat16,
                                       device=device)
            residual = torch.zeros_like(hidden_states)
            gamma = torch.ones(hidden, dtype=torch.bfloat16, device=device)
            params = arf.FlashInferFusedAllReduceParams(
                world_size=world_size, max_token_num=fusion_max_tokens)
            kwargs = params.get_trtllm_fused_allreduce_kwargs()
            capability = torch.cuda.get_device_capability(rank)
            use_oneshot = arf._select_flashinfer_allreduce_use_oneshot(
                workspace.backend, capability[0] * 10 + capability[1],
                world_size, payload)

            def fused(hs=hidden_states, res=residual, g=gamma, kw=kwargs):
                arf.call_trtllm_fused_allreduce_norm(
                    allreduce_in=hs, residual=res, rms_gamma=g,
                    rms_eps=RMS_EPS, norm_out=None, quant_out=None,
                    scale_out=None,
                    pattern_code=flashinfer_comm.AllReduceFusionPattern
                    .kARResidualRMSNorm, **kw)

            record("flashinfer_fused_ar_residual_rmsnorm", tokens,
                   hidden_states.shape, payload,
                   f"FLASHINFER_{workspace.backend.upper()}",
                   _InPlace(fused),
                   {"use_oneshot": use_oneshot, "fusion_kwargs": kwargs})

            # Same kernel as called by the model-level op
            # vllm/models/common/ops/fused_allreduce_rms_norm.py, which
            # DeepseekV32/GlmMoeDsa use because they run under breakable
            # CUDA graphs with compilation mode NONE (no fusion pass): it
            # passes a fresh norm_out, so the new residual lands in the
            # all-reduce input buffer and the norm in norm_out.
            def fused_model_op(hs=hidden_states, res=residual, g=gamma,
                               kw=kwargs):
                arf.call_trtllm_fused_allreduce_norm(
                    allreduce_in=hs, residual=res, rms_gamma=g,
                    rms_eps=RMS_EPS, norm_out=torch.empty_like(hs),
                    quant_out=None, scale_out=None, weight_bias=0.0,
                    pattern_code=flashinfer_comm.AllReduceFusionPattern
                    .kARResidualRMSNorm, **kw)

            record("flashinfer_fused_ar_residual_rmsnorm_model_op", tokens,
                   hidden_states.shape, payload,
                   f"FLASHINFER_{workspace.backend.upper()}",
                   _InPlace(fused_model_op),
                   {"use_oneshot": use_oneshot, "fusion_kwargs": kwargs})

        # 3. PyNccl all-reduce forced (NCCL reference for every payload).
        record("pynccl_all_reduce", tokens, source.shape, payload, "PYNCCL",
               _Chain(comm.pynccl_comm.all_reduce, source), {})

        # 4. All-gather / reduce-scatter along dim 0 via vLLM's dispatch.
        # Repeated calls on one input: same-stream graph nodes serialise, and
        # no extra kernel is introduced to feed an output back.
        record("tp_all_gather_dim0", tokens, source.shape,
               payload * world_size, "PYNCCL",
               _Repeat(lambda x: tp.all_gather(x, dim=0), source),
               {"payload_bytes_meaning": "gathered_output_bytes"})
        if tokens % world_size == 0:
            record("tp_reduce_scatter_dim0", tokens, source.shape, payload,
                   "PYNCCL",
                   _Repeat(lambda x: tp.reduce_scatter(x, dim=0), source),
                   {"payload_bytes_meaning": "full_input_bytes"})

        # 5. Logits all-gather (LogitsProcessor, dim=-1, vocab-parallel).
        # Logits exist only for sampled positions (<= max_num_seqs), so the
        # prefill-chunk token counts are skipped.  vLLM's dim=-1 path includes
        # its movedim/reshape copy kernel, which is part of what it executes.
        if tokens <= 512:
            vocab_per_rank = LOGITS_VOCAB // world_size
            logits = torch.ones(tokens, vocab_per_rank, dtype=torch.bfloat16,
                                device=device)
            record("tp_all_gather_logits", tokens, logits.shape,
                   tokens * vocab_per_rank * world_size * 2, "PYNCCL",
                   _Repeat(lambda x: tp.all_gather(x, dim=-1), logits),
                   {"payload_bytes_meaning": "gathered_output_bytes",
                    "includes_vllm_movedim_copy": True})

    Path(result_path).write_text(json.dumps({
        "rank": rank, "device_name": device_name, "selection": selection,
        "rows": rows}) + "\n")
    dist.barrier(group=cpu_group)
    destroy_model_parallel()
    destroy_distributed_environment()


def _environment() -> dict:
    import flashinfer
    import vllm

    if not vllm.__version__.startswith(EXPECTED_VLLM_VERSION_PREFIX):
        raise RuntimeError(f"expected vLLM {EXPECTED_VLLM_VERSION_PREFIX}, "
                           f"got {vllm.__version__}")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout.split()[0]
    env_keys = sorted(k for k in os.environ
                      if k.startswith(("NCCL_", "VLLM_", "CUDA_", "TORCH_")))
    return {
        "hardware_id": HARDWARE_ID,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "driver_version": driver,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "torch_nccl_version": ".".join(
            str(v) for v in torch.cuda.nccl.version()),
        "vllm_version": vllm.__version__,
        "flashinfer_version": flashinfer.__version__,
        "python": platform.python_version(),
        "env": {k: os.environ[k] for k in env_keys},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--hidden", type=int, required=True,
                        help="7168 = DSV3/table grid; 6144 = GLM-5.2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29811)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--quick", action="store_true",
                        help=f"smoke: tokens {QUICK_TOKEN_COUNTS}, "
                             f"{QUICK_TRIALS} trials")
    args = parser.parse_args()
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if max(args.world_sizes) > len(visible):
        raise ValueError("world size exceeds CUDA_VISIBLE_DEVICES")
    if args.quick:
        token_counts, trials = QUICK_TOKEN_COUNTS, QUICK_TRIALS
    else:
        token_counts = DECODE_TOKEN_COUNTS + PREFILL_TOKEN_COUNTS
        trials = TRIALS
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve()
    report = {
        "schema_version": 1,
        "mechanism": "vllm_0_30_tp_collective_graph_node_service",
        "timer_domain": "cuda_event_around_graph_replay_divided_by_chain",
        "decode_chain_length": DECODE_CHAIN_LENGTH,
        "prefill_chain_length": PREFILL_CHAIN_LENGTH,
        "trials": trials, "warmup_replays": WARMUP_REPLAYS,
        "dtype": "bfloat16", "hidden": args.hidden, "quick": args.quick,
        "token_counts": list(token_counts),
        "production_or_evaluation_duration_consumed": False,
        "runner": {"path": str(runner), "sha256": _sha256(runner),
                   "git_commit": args.source_commit},
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
                                  args.hidden, token_counts, trials,
                                  paths[rank]))
            for rank in range(world_size)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"world size {world_size} worker failed: "
                                   f"exit {process.exitcode}")
        per_rank = [json.loads(Path(path).read_text()) for path in paths]
        selections = [r["selection"] for r in per_rank]
        if any(s != selections[0] for s in selections):
            raise RuntimeError(f"ranks disagree on backend selection: "
                               f"{selections}")
        summary = []
        for row_index, row in enumerate(per_rank[0]["rows"]):
            for rank_rows in per_rank:
                other = rank_rows["rows"][row_index]
                if (other["series"], other["tokens"], other["backend"]) != \
                        (row["series"], row["tokens"], row["backend"]):
                    raise RuntimeError("rank rows are misaligned")
            summary.append({
                "series": row["series"], "tokens": row["tokens"],
                "shape": row["shape"], "payload_bytes": row["payload_bytes"],
                "backend": row["backend"],
                # A collective completes when its slowest rank completes.
                "ns": max(statistics.median(r["rows"][row_index]
                                            ["ns_per_node"])
                          for r in per_rank),
                "t_start_unix": row["t_start_unix"],
                "t_end_unix": row["t_end_unix"]})
        report["world_sizes"][str(world_size)] = {
            "selection": selections[0], "per_rank_raw": paths,
            "summary": summary}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for world_size, block in report["world_sizes"].items():
        print(f"world_size={world_size} selection={block['selection']}")
        for row in block["summary"]:
            print(f"  {row['series']:38s} tokens={row['tokens']:5d} "
                  f"bytes={row['payload_bytes']:10d} "
                  f"backend={row['backend']:18s} {row['ns']/1e3:9.2f}us")


if __name__ == "__main__":
    main()
