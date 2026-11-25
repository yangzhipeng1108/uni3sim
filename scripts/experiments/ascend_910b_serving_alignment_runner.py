#!/usr/bin/env python3
"""Capture one real Ascend 910B vLLM eager-prefill window for kernel alignment."""

import argparse
import json
from pathlib import Path
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--pre-target-sleep-seconds", type=int, default=0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.prompt_tokens <= 0:
        raise ValueError("--prompt-tokens must be positive")
    if args.warmup_tokens <= 0:
        raise ValueError("--warmup-tokens must be positive")
    if args.max_model_len < args.prompt_tokens + 1:
        raise ValueError("--max-model-len must accommodate prompt plus output")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.vocab_size <= 2:
        raise ValueError("--vocab-size must exceed two")
    if args.pre_target_sleep_seconds < 0:
        raise ValueError("--pre-target-sleep-seconds must be non-negative")


def _prompt(token_count: int, vocab_size: int) -> list[int]:
    return [1] + [2 + index % (vocab_size - 2) for index in range(token_count - 1)]


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.prompt_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        ignore_eos=True,
    )
    llm.generate(
        {"prompt_token_ids": _prompt(args.warmup_tokens, args.vocab_size)},
        sampling_params,
        use_tqdm=False,
    )
    # msprof's task-time collection must begin after model initialization and
    # the warmup Generate.  This explicit staging interval is outside the
    # measured request and lets the caller choose a bounded profiler delay
    # that covers the real prefill, rather than filling the task buffer during
    # engine startup.
    if args.pre_target_sleep_seconds:
        time.sleep(args.pre_target_sleep_seconds)
    start_ns = time.time_ns()
    outputs = llm.generate(
        {"prompt_token_ids": _prompt(args.prompt_tokens, args.vocab_size)},
        sampling_params,
        use_tqdm=False,
    )
    end_ns = time.time_ns()
    result = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "execution_kind": "vllm_llm_generate_eager_prefill",
        "model": args.model,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": 1,
        "warmup_tokens": args.warmup_tokens,
        "pre_target_sleep_seconds": args.pre_target_sleep_seconds,
        "max_num_batched_tokens": args.prompt_tokens,
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "elapsed_us": (end_ns - start_ns) / 1000.0,
        "output_token_count": len(outputs[0].outputs[0].token_ids),
    }
    (output_dir / "capture_metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
