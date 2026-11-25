#!/usr/bin/env python3
"""Measure independent vLLM 0.11 CPU Scheduler state service.

The synthetic prefill/mixed/decode grid contains no Serving request, model
execution, or production duration.  Scheduler construction, request creation,
and running-request priming are outside the timed schedule/update boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


COUNTS = (1, 8, 32)
PROMPT_LENGTHS = (64, 256)
PHASES = ("prefill", "mixed", "decode")
OUTCOMES = ("finish", "continue")
BLOCK_SIZE = 16
NUM_BLOCKS = 4096


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scheduler(model: ModelConfig) -> AsyncScheduler:
    cache = CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=True)
    native = VllmConfig(
        model_config=model,
        scheduler_config=SchedulerConfig(
            max_model_len=4096, max_num_batched_tokens=8192,
            max_num_seqs=128, enable_chunked_prefill=True,
            async_scheduling=True),
        cache_config=cache,
        device_config=DeviceConfig(device="cpu"),
    )
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128,
        dtype=torch.bfloat16)
    kv = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[KVCacheTensor(
            size=NUM_BLOCKS * spec.page_size_bytes * 36,
            shared_by=["layer"])],
        kv_cache_groups=[KVCacheGroupSpec(
            layer_names=["layer"], kv_cache_spec=spec)],
    )
    cache.num_gpu_blocks = NUM_BLOCKS
    return AsyncScheduler(native, kv, StructuredOutputManager(native))


def _update(scheduler: AsyncScheduler, output) -> int:
    ids = list(output.num_scheduled_tokens)
    model_output = ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={request_id: index
                         for index, request_id in enumerate(ids)},
        sampled_token_ids=[[1] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[None for _ in ids],
    )
    start = time.perf_counter_ns()
    scheduler.update_from_output(output, model_output)
    return time.perf_counter_ns() - start


def _check_output(output, count: int, tokens: int) -> None:
    if (len(output.num_scheduled_tokens) != count
            or output.total_num_scheduled_tokens != tokens):
        raise RuntimeError("synthetic Scheduler batch differs from declared state")


def _sample(model: ModelConfig, count: int, length: int, ordinal: int,
            phase: str, outcome: str) -> dict[str, int]:
    scheduler = _scheduler(model)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    if phase == "prefill":
        running_count = 0
    elif phase == "mixed":
        running_count = count // 2
    elif phase == "decode":
        running_count = count
    else:
        raise ValueError("unknown synthetic Scheduler phase")
    if outcome not in OUTCOMES:
        raise ValueError("unknown synthetic Scheduler outcome")
    for index in range(running_count):
        token = index + 1
        scheduler.add_request(Request(
            request_id=f"synthetic-{ordinal}-running-{index}",
            prompt_token_ids=[token] * length,
            sampling_params=SamplingParams(
                max_tokens=2 if outcome == "finish" else 3),
            pooling_params=None,
            eos_token_id=None,
            block_hasher=hasher,
        ))
    if running_count:
        priming = scheduler.schedule()
        _check_output(priming, running_count, running_count * length)
        _update(scheduler, priming)
        if scheduler.get_num_unfinished_requests() != running_count:
            raise RuntimeError("synthetic running-request priming lost requests")
    for index in range(running_count, count):
        token = index + 1
        scheduler.add_request(Request(
            request_id=f"synthetic-{ordinal}-fresh-{index}",
            prompt_token_ids=[token] * length,
            sampling_params=SamplingParams(
                max_tokens=1 if outcome == "finish" else 2),
            pooling_params=None,
            eos_token_id=None,
            block_hasher=hasher,
        ))
    start = time.perf_counter_ns()
    output = scheduler.schedule()
    plan_ns = time.perf_counter_ns() - start
    _check_output(output, count, running_count + (count - running_count) * length)
    update_ns = _update(scheduler, output)
    if outcome == "continue":
        if scheduler.get_num_unfinished_requests() != count:
            raise RuntimeError("synthetic continuing update lost requests")
        final = scheduler.schedule()
        _check_output(final, count, count)
        _update(scheduler, final)
    if scheduler.get_num_unfinished_requests() != 0:
        raise RuntimeError("synthetic completed state retained requests")
    return {"plan_ns": plan_ns, "update_ns": update_ns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 5 or args.samples > 50:
        raise ValueError("samples must be between 5 and 50")
    if importlib.metadata.version("vllm") != "0.11.0":
        raise RuntimeError("this probe requires the installed vLLM 0.11.0")
    model_source = args.model_config.resolve()
    if not model_source.is_file():
        raise FileNotFoundError(model_source)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = output_path.parent / "synthetic_model"
    model_dir.mkdir(exist_ok=True)
    local_config = model_dir / "config.json"
    local_config.write_bytes(model_source.read_bytes())
    model = ModelConfig(
        model=str(model_dir), dtype="bfloat16", max_model_len=4096,
        skip_tokenizer_init=True, enforce_eager=True)
    init_none_hash(sha256)
    rows = []
    for phase_index, phase in enumerate(PHASES):
        for outcome_index, outcome in enumerate(OUTCOMES):
            for length in PROMPT_LENGTHS:
                for count in COUNTS:
                    if phase == "mixed" and count == 1:
                        continue
                    for sample in range(args.samples):
                        ordinal = (phase_index * 100000000
                                   + outcome_index * 10000000
                                   + length * 10000 + count * 100 + sample)
                        result = _sample(
                            model, count, length, ordinal, phase, outcome)
                        rows.append({"phase": phase, "update_outcome": outcome,
                                     "prompt_tokens": length,
                                     "request_count": count,
                                     "sample": sample, **result})
    summaries = []
    for phase in PHASES:
        for outcome in OUTCOMES:
            for length in PROMPT_LENGTHS:
                for count in COUNTS:
                    if phase == "mixed" and count == 1:
                        continue
                    cell = [row for row in rows if row["phase"] == phase
                            and row["update_outcome"] == outcome
                            and row["prompt_tokens"] == length
                            and row["request_count"] == count]
                    summaries.append({
                        "phase": phase, "update_outcome": outcome,
                        "prompt_tokens": length, "request_count": count,
                        "median_plan_ns": statistics.median(
                            row["plan_ns"] for row in cell),
                        "median_update_ns": statistics.median(
                            row["update_ns"] for row in cell),
                    })
    source_files = {
        "scheduler": Path(inspect.getfile(AsyncScheduler)).resolve(),
        "request": Path(inspect.getfile(Request)).resolve(),
        "kv_cache": Path(inspect.getfile(FullAttentionSpec)).resolve(),
    }
    payload = {
        "schema_version": 1,
        "kind": "independent_vllm011_cpu_scheduler_state_grid",
        "vllm_version": "0.11.0",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "model_config_sha256": _digest(local_config),
        "source_sha256": {name: _digest(path)
                          for name, path in source_files.items()},
        "probe_source_sha256": _digest(Path(__file__).resolve()),
        "prefix_caching": True,
        "prefix_relation": "unique_synthetic_token_sequence_per_request",
        "update_outcomes": list(OUTCOMES),
        "phases": list(PHASES),
        "running_request_priming": "one_prefill_step_outside_timed_boundary",
        "measurement_boundary": "schedule_and_update_from_output_cpu_perf_counter_ns",
        "production_or_evaluation_duration_consumed": False,
        "sample_count_per_cell": args.samples,
        "rows": rows,
        "summaries": summaries,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output_path),
                      "summaries": summaries}))


if __name__ == "__main__":
    main()
