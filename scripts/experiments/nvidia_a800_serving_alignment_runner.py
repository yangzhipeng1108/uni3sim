#!/usr/bin/env python3
"""Capture one real A800 vLLM eager-prefill window for kernel alignment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--vocab-size", type=int, default=128256)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if args.physical_gpu not in range(7):
        raise ValueError("--physical-gpu must be in 0..6; GPU 7 is reserved")
    if args.prompt_tokens <= 0 or args.warmup_tokens <= 0:
        raise ValueError("prompt and warmup tokens must be positive")
    if args.max_model_len < args.prompt_tokens + 1:
        raise ValueError("--max-model-len must accommodate prompt plus output")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.vocab_size <= 2:
        raise ValueError("--vocab-size must exceed two")


def _prompt(token_count: int, token_id: int, vocab_size: int) -> dict[str, list[int]]:
    return {
        "prompt_token_ids": [token_id] + [
            2 + index % (vocab_size - 2) for index in range(token_count - 1)]
    }


def _clock_state(physical_gpu: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            "nvidia-smi", "-i", str(physical_gpu),
            "--query-gpu=index,pstate,clocks.current.sm,clocks.current.memory,"
            "persistence_mode,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ], check=True, text=True, capture_output=True)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"unexpected nvidia-smi record: {fields!r}")
    index, pstate, sm, memory, persistence, utilization, memory_used = fields
    state = {
        "physical_gpu_index": int(index),
        "pstate": pstate,
        "sm_clock_mhz": int(sm),
        "memory_clock_mhz": int(memory),
        "persistence_mode": persistence,
        "utilization_percent": int(utilization),
        "memory_used_mib": int(memory_used),
    }
    if (state["physical_gpu_index"] != physical_gpu
            or state["pstate"] != "P0"
            or state["sm_clock_mhz"] != 1140
            or state["memory_clock_mhz"] != 1593
            or state["persistence_mode"] != "Enabled"):
        raise RuntimeError(f"A800 clock lock is not literal: {state}")
    return state


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=False)

    # vLLM starts its engine worker with spawn.  Nsight captures the explicitly
    # bounded eager generate interval; no CUDA Graph is enabled by this runner.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams
    import torch

    clock_before = _clock_state(args.physical_gpu)
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.prompt_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=42,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=1, temperature=0.0, ignore_eos=True)
    llm.generate(
        _prompt(args.warmup_tokens, 2, args.vocab_size), sampling_params,
        use_tqdm=False)
    torch.cuda.synchronize()

    started = time.perf_counter_ns()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        outputs = llm.generate(
            _prompt(args.prompt_tokens, 1, args.vocab_size), sampling_params,
            use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    ended = time.perf_counter_ns()
    clock_after = _clock_state(args.physical_gpu)
    properties = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": 1,
        "hardware_id": "nvidia_a800",
        "execution_kind": "vllm_llm_generate_eager_prefill",
        "model": str(args.model),
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": 1,
        "warmup_tokens": args.warmup_tokens,
        "max_num_batched_tokens": args.prompt_tokens,
        "clock_before": clock_before,
        "clock_after": clock_after,
        "device": {
            "logical_index": 0,
            "physical_index": args.physical_gpu,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        },
        "profiler_range": "CUDA Runtime cudaProfilerStart/Stop",
        "profiled_generate_wall_ns": ended - started,
        "output_token_count": len(outputs[0].outputs[0].token_ids),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
