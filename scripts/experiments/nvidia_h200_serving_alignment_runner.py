#!/usr/bin/env python3
"""Capture one H200 ShareGPT alignment raw case (operators + Scheduler + Nsight).

This is the H200 counterpart of the capture that produced every A800
``serve_exp`` ShareGPT raw case: one eager in-process vLLM run replays the
canonical A800 request file and arrival schedule while
``dataset_pipeline.serving_dataset.vllm_operator_trace`` records the Scheduler
sidecar and per-rank operator traces, and the caller wraps this process in
``nsys profile`` so the device kernels of every formal forward can be
attributed through the NVTX step ranges.  The files written are exactly the
A800 raw layout: ``meta.json``, ``arrival_schedule.json``,
``serving_metrics.json``, ``plugin_trace.json_provenance.json``,
``plugin_trace.json_scheduler.jsonl`` and
``plugin_trace.json_operators.jsonl.rank*``.  Metrics from this run are
profiler-attached raw evidence; official metrics come from the separate
no-profiler ground-truth runs.

The process must start with the instrumentation bootstrap on ``PYTHONPATH``
(``dataset_pipeline/serving_dataset/instrumentation_bootstrap`` and the
repository root) so every spawned vLLM process installs the capture.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_RUNTIME_VERSIONS = ("0.11.0", "0.30.0")
EXPECTED_FIELDS = (
    "Query Time Stamp [ms]",
    "In Token [tokens]",
    "Out Token [tokens]",
    "Cached hit [tokens]",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw artifact: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _requests(path: Path) -> tuple[list[float], list[int], list[int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 128:
        raise ValueError("ShareGPT capture requires exactly 128 canonical requests")
    arrivals, inputs, outputs = [], [], []
    for index, row in enumerate(rows):
        if tuple(row) != EXPECTED_FIELDS:
            raise ValueError(f"canonical request fields differ at index {index}")
        if row["Cached hit [tokens]"] != "0":
            raise ValueError("ShareGPT capture does not support prefix-hit request rows")
        arrivals.append(float(row["Query Time Stamp [ms]"]) / 1000.0)
        inputs.append(int(row["In Token [tokens]"]))
        outputs.append(int(row["Out Token [tokens]"]))
    if arrivals[0] != 0.0 or any(b < a for a, b in zip(arrivals, arrivals[1:])):
        raise ValueError("canonical arrival sequence is invalid")
    return arrivals, inputs, outputs


def _sample_prompts(
    model: str, dataset: str, seed: int, inputs: list[int], outputs: list[int]
) -> list[str]:
    from transformers import AutoTokenizer
    from vllm.benchmarks.datasets import ShareGPTDataset

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    samples = ShareGPTDataset(dataset_path=dataset, random_seed=seed).sample(
        tokenizer, num_requests=len(inputs), output_len=None)
    if [sample.prompt_len for sample in samples] != inputs:
        raise ValueError("sampled ShareGPT input lengths differ from canonical requests")
    if [sample.expected_output_len for sample in samples] != outputs:
        raise ValueError("sampled ShareGPT output lengths differ from canonical requests")
    return [sample.prompt for sample in samples]


def _engine_arguments(model: str, tp: int, ep: int) -> dict[str, Any]:
    # The A800 raw engine arguments, verbatim, plus an explicit synchronous
    # Scheduler: vLLM 0.11.0 defaulted to it, vLLM 0.30.0 defaults to async.
    arguments: dict[str, Any] = {
        "model": model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "enforce_eager": True,
        "max_model_len": 8192,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "trust_remote_code": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 8192,
        "async_scheduling": False,
    }
    if ep == 2:
        arguments["enable_expert_parallel"] = True
    elif ep != 1:
        raise ValueError("expert parallel size must be one or two")
    return arguments


async def _replay(
    engine: Any,
    prompts: list[str],
    arrivals: list[float],
    outputs: list[int],
) -> tuple[list[dict[str, Any]], float, int, int]:
    from vllm import SamplingParams

    warmup = SamplingParams(temperature=0.0, max_tokens=1)
    async for _ in engine.generate(" ".join(["hello"] * 64), warmup, request_id="warmup_0"):
        pass
    results: list[dict[str, Any] | None] = [None] * len(prompts)
    start_epoch_us = time.time_ns() // 1000
    started = time.perf_counter()

    async def submit(index: int) -> None:
        delay = arrivals[index] - (time.perf_counter() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        sent = time.perf_counter()
        first_token = None
        token_count = 0
        params = SamplingParams(
            temperature=0.0, max_tokens=outputs[index], min_tokens=outputs[index],
            ignore_eos=True)
        async for output in engine.generate(prompts[index], params, request_id=f"req_{index}"):
            if not output.outputs:
                continue
            token_count = len(output.outputs[0].token_ids)
            if first_token is None and token_count > 0:
                first_token = time.perf_counter()
        ended = time.perf_counter()
        if first_token is None:
            raise RuntimeError(f"request {index} produced no output token")
        if token_count != outputs[index]:
            raise RuntimeError(
                f"request {index} output length differs: {token_count} != {outputs[index]}")
        results[index] = {
            "request_id": f"req_{index}",
            "arrival_s": arrivals[index],
            "send_time_s": sent - started,
            "ttft_ms": (first_token - sent) * 1000.0,
            "e2e_ms": (ended - sent) * 1000.0,
            "n_output_tokens": token_count,
            "tpot_ms": (
                (ended - first_token) * 1000.0 / (token_count - 1)
                if token_count > 1 else None),
        }

    await asyncio.gather(*(submit(index) for index in range(len(prompts))))
    wall_seconds = time.perf_counter() - started
    end_epoch_us = time.time_ns() // 1000
    if any(result is None for result in results):
        raise RuntimeError("capture completed without every request result")
    return [r for r in results if r is not None], wall_seconds, start_epoch_us, end_epoch_us


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--target-qps", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--expert-parallel-size", type=int, required=True)
    parser.add_argument("--runtime-version", choices=SUPPORTED_RUNTIME_VERSIONS, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    installed = version("vllm").split("+", 1)[0]
    if installed != args.runtime_version:
        raise RuntimeError(f"installed vLLM {installed} != requested {args.runtime_version}")
    if args.target_qps <= 0:
        raise ValueError("target QPS must be positive")
    if args.tensor_parallel_size not in (1, 2):
        raise ValueError("tensor parallel size must be one or two")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name the owned GPUs")
    devices = [int(value) for value in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    if len(devices) != args.tensor_parallel_size:
        raise ValueError("CUDA_VISIBLE_DEVICES count must equal TP")
    if not args.case_dir.is_dir() or any(args.case_dir.iterdir()):
        raise ValueError(f"case directory must exist and be empty: {args.case_dir}")
    case_dir = args.case_dir.resolve()
    sidecar = case_dir / "plugin_trace.json_scheduler.jsonl"
    operators = case_dir / "plugin_trace.json_operators.jsonl"
    for name, expected in (
        ("MIDLAYER_SCHEDULER_SIDECAR_PATH", sidecar),
        ("MIDLAYER_OPERATOR_TRACE_PATH", operators),
    ):
        if name not in os.environ or Path(os.environ[name]) != expected:
            raise RuntimeError(f"{name} must be exported as {expected}")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Request IDs must stay the literal req_N the raw adapter maps; vLLM 0.30.0
    # otherwise appends a random suffix to every in-process request ID.
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    from dataset_pipeline.serving_dataset.vllm_operator_trace import install_from_environment

    if not install_from_environment():
        raise RuntimeError("vLLM operator capture did not install")

    arrivals, inputs, outputs = _requests(args.requests)
    prompts = _sample_prompts(args.model, str(args.dataset), args.seed, inputs, outputs)
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_arguments = _engine_arguments(
        args.model, args.tensor_parallel_size, args.expert_parallel_size)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_arguments))
    try:
        results, wall_seconds, start_epoch_us, end_epoch_us = asyncio.run(
            _replay(engine, prompts, arrivals, outputs))
    finally:
        engine.shutdown()
    operator_files = sorted(case_dir.glob("plugin_trace.json_operators.jsonl.rank*"))
    if [path.name for path in operator_files] != [
        f"plugin_trace.json_operators.jsonl.rank{rank}"
        for rank in range(args.tensor_parallel_size)
    ]:
        raise RuntimeError(f"operator traces do not cover every rank: {operator_files}")
    if not sidecar.is_file():
        raise RuntimeError("Scheduler sidecar was not written")
    _write_json(case_dir / "serving_metrics.json", {
        "wall_s": wall_seconds,
        "throughput_rps": len(results) / wall_seconds,
        "qps": args.target_qps,
        "num_requests": len(results),
        "per_request": results,
    })
    _write_json(case_dir / "arrival_schedule.json", {
        "qps": args.target_qps,
        "seed": args.seed,
        "num_requests": len(arrivals),
        "arrival_times_s": arrivals,
        "source_requests_csv": str(args.requests.resolve()),
        "source_requests_sha256": _sha256(args.requests),
    })
    _write_json(case_dir / "meta.json", {
        "case_id": case_dir.name,
        "model": args.model,
        "num_requests": len(results),
        "tensor_parallel_size": args.tensor_parallel_size,
        "expert_parallel_size": args.expert_parallel_size,
        "gpu_ids": devices,
        "hardware_id": "nvidia_h200",
        "workload": "sharegpt",
        "dataset_path": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "sample_seed": args.seed,
        "qps": args.target_qps,
        "poisson_seed": args.seed,
        "runtime_version": installed,
        "gen_start_epoch_us": start_epoch_us,
        "gen_end_epoch_us": end_epoch_us,
    })
    _write_json(case_dir / "plugin_trace.json_provenance.json", {
        "scheduler_sidecar": str(sidecar),
        "operator_trace": str(operators),
        "mode": "serving",
        "producer": "dataset_pipeline.serving_dataset.vllm_operator_trace",
        "scheduler": "real-vLLM",
        "operators": "real-ModelRunner",
        "qps": args.target_qps,
        "num_requests": len(results),
        "model": args.model,
        "prompt_lengths": inputs,
        "output_lengths": [result["n_output_tokens"] for result in results],
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": args.expert_parallel_size,
        "engine_kwargs": {k: v for k, v in engine_arguments.items() if k != "model"},
        "runtime_version": installed,
        "nvtx_step_ranges": ["midlayer-op-step:<iter_id>", "midlayer-op-sample:<iter_id>"],
        "python_executable": sys.executable,
    })
    print(json.dumps({"status": "success", "case_dir": str(case_dir),
                      "wall_s": wall_seconds}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
