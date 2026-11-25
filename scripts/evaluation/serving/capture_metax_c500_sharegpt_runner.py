#!/usr/bin/env python3
"""Capture one C500 ShareGPT cell with a frozen request stream and optional hooks."""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw artifact: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _requests(path: Path) -> tuple[list[float], list[int], list[int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 128:
        raise ValueError("C500 ShareGPT capture requires exactly 128 canonical requests")
    expected_fields = {
        "Query Time Stamp [ms]",
        "In Token [tokens]",
        "Out Token [tokens]",
        "Cached hit [tokens]",
    }
    if not rows or set(rows[0]) != expected_fields:
        raise ValueError("canonical requests.csv fields are invalid")
    arrivals = []
    input_lengths = []
    output_lengths = []
    for index, row in enumerate(rows):
        if set(row) != expected_fields:
            raise ValueError(f"request row fields differ at index {index}")
        if row["Cached hit [tokens]"] != "0":
            raise ValueError("C500 recapture does not support prefix-hit request rows")
        arrivals.append(float(row["Query Time Stamp [ms]"]) / 1000.0)
        input_lengths.append(int(row["In Token [tokens]"]))
        output_lengths.append(int(row["Out Token [tokens]"]))
    if arrivals[0] != 0.0 or any(
        right < left for left, right in zip(arrivals, arrivals[1:])
    ):
        raise ValueError("canonical arrival sequence is invalid")
    return arrivals, input_lengths, output_lengths


def _sample_prompts(model: str, dataset: str, seed: int, expected_inputs: list[int], expected_outputs: list[int]) -> list[str]:
    from transformers import AutoTokenizer
    from vllm.benchmarks.datasets import ShareGPTDataset

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    samples = ShareGPTDataset(dataset_path=dataset, random_seed=seed).sample(
        tokenizer, num_requests=len(expected_inputs), output_len=None
    )
    inputs = [sample.prompt_len for sample in samples]
    outputs = [sample.expected_output_len for sample in samples]
    if inputs != expected_inputs:
        raise ValueError("sampled ShareGPT input lengths differ from canonical requests.csv")
    if outputs != expected_outputs:
        raise ValueError("sampled ShareGPT output lengths differ from canonical requests.csv")
    return [sample.prompt for sample in samples]


def _engine(model: str, tp: int, ep: int, *, graph_execution: bool, seed: int):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    arguments: dict[str, Any] = {
        "model": model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "enforce_eager": not graph_execution,
        "max_model_len": 8192,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "trust_remote_code": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 8192,
    }
    if graph_execution:
        arguments.update({
            "async_scheduling": True,
            "cudagraph_metrics": True,
            "enable_prefix_caching": True,
            "block_size": 16,
            "max_num_seqs": 256,
            "scheduling_policy": "fcfs",
            "safetensors_load_strategy": "prefetch",
            "safetensors_prefetch_num_threads": 1,
            "seed": seed,
            "compilation_config": {"mode": 3},
        })
    if ep == 2:
        arguments["enable_expert_parallel"] = True
    elif ep != 1:
        raise ValueError("expert parallel size must be one or two")
    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**arguments)), arguments


async def _run_requests(
    engine: object,
    prompts: list[str],
    arrivals: list[float],
    output_lengths: list[int],
    *,
    warmup: bool,
) -> tuple[list[dict[str, Any]], float]:
    from vllm import SamplingParams

    if warmup:
        warmup_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        async for _ in engine.generate("hello " * 64, warmup_params, request_id="warmup_0"):
            pass
    started = time.perf_counter()
    results: list[dict[str, Any] | None] = [None] * len(prompts)

    async def submit(index: int) -> None:
        delay = arrivals[index] - (time.perf_counter() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        sent = time.perf_counter()
        first_token: float | None = None
        token_count = 0
        params = SamplingParams(
            temperature=0.0,
            max_tokens=output_lengths[index],
            min_tokens=output_lengths[index],
            ignore_eos=True,
        )
        async for output in engine.generate(prompts[index], params, request_id=f"req_{index}"):
            if not output.outputs:
                continue
            token_count = len(output.outputs[0].token_ids)
            if first_token is None and token_count > 0:
                first_token = time.perf_counter()
        ended = time.perf_counter()
        if first_token is None:
            raise RuntimeError(f"request {index} produced no output token")
        if token_count != output_lengths[index]:
            raise RuntimeError(
                f"request {index} output length differs: {token_count} != {output_lengths[index]}"
            )
        results[index] = {
            "request_id": f"req_{index}",
            "arrival_s": arrivals[index],
            "send_time_s": sent - started,
            "ttft_ms": (first_token - sent) * 1000.0,
            "e2e_ms": (ended - sent) * 1000.0,
            "n_output_tokens": token_count,
            "tpot_ms": (ended - first_token) * 1000.0 / (token_count - 1)
            if token_count > 1 else None,
        }

    await asyncio.gather(*(submit(index) for index in range(len(prompts))))
    wall_seconds = time.perf_counter() - started
    if any(result is None for result in results):
        raise RuntimeError("capture completed without every request result")
    return [result for result in results if result is not None], wall_seconds


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("clean", "profile"), required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--target-qps", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--expert-parallel-size", type=int, required=True)
    parser.add_argument("--graph-execution", action="store_true")
    parser.add_argument("--request-count", type=int, default=128)
    parser.add_argument("--output-token-cap", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.target_qps <= 0:
        raise ValueError("target QPS must be positive")
    if args.tensor_parallel_size not in (1, 2):
        raise ValueError("tensor parallel size must be one or two")
    if not 1 <= args.request_count <= 128:
        raise ValueError("request count must be between one and 128")
    if args.output_token_cap is not None and args.output_token_cap < 2:
        raise ValueError("output token cap must be at least two")
    if not args.case_dir.is_dir():
        raise FileNotFoundError(f"case directory is absent: {args.case_dir}")
    if not args.dataset.is_file() or not args.requests.is_file():
        raise FileNotFoundError("dataset and canonical requests.csv must both exist")
    arrivals, input_lengths, output_lengths = _requests(args.requests)
    prompts = _sample_prompts(
        args.model, str(args.dataset), args.seed, input_lengths, output_lengths
    )
    arrivals = arrivals[:args.request_count]
    prompts = prompts[:args.request_count]
    input_lengths = input_lengths[:args.request_count]
    output_lengths = output_lengths[:args.request_count]
    if args.output_token_cap is not None:
        output_lengths = [
            min(length, args.output_token_cap) for length in output_lengths
        ]
    engine, engine_arguments = _engine(
        args.model, args.tensor_parallel_size, args.expert_parallel_size,
        graph_execution=args.graph_execution,
        seed=args.seed,
    )
    try:
        results, wall_seconds = asyncio.run(
            _run_requests(
                engine, prompts, arrivals, output_lengths, warmup=True
            )
        )
    finally:
        engine.shutdown()
    metric_payload = {
        "wall_s": wall_seconds,
        "throughput_rps": len(results) / wall_seconds,
        "qps": args.target_qps,
        "num_requests": len(results),
        "per_request": results,
    }
    if args.mode == "clean":
        _write_json(args.case_dir / "serving_metrics.json", metric_payload)
        _write_json(args.case_dir / "arrival_schedule.json", {
            "qps": args.target_qps,
            "seed": args.seed,
            "num_requests": len(arrivals),
            "arrival_times_s": arrivals,
        })
        _write_json(args.case_dir / "meta.json", {
            "case_id": args.case_dir.name,
            "model": args.model,
            "num_requests": len(results),
            "tensor_parallel_size": args.tensor_parallel_size,
            "expert_parallel_size": args.expert_parallel_size,
            "gpu_ids": [int(value) for value in os.environ["MACA_VISIBLE_DEVICES"].split(",")],
            "workload": "sharegpt",
            "dataset_path": str(args.dataset),
            "sample_seed": args.seed,
            "qps": args.target_qps,
            "poisson_seed": args.seed,
        })
    else:
        _write_json(args.case_dir / "profile_metrics.json", metric_payload)
        _write_json(args.case_dir / "plugin_trace.json_provenance.json", {
            "mode": "serving",
            "producer": "dataset_pipeline.serving_dataset.vllm_capture_plugin",
            "scheduler": "real-vLLM",
            "operators": "real-GPUModelRunner",
            "qps": args.target_qps,
            "num_requests": len(results),
            "model": args.model,
            "prompt_lengths": input_lengths,
            "output_lengths": output_lengths,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": args.expert_parallel_size,
            "runtime_version": importlib.metadata.version("vllm"),
            "engine_kwargs": engine_arguments,
            "enforce_eager": not args.graph_execution,
            "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE": os.environ[
                "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE"
            ],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
