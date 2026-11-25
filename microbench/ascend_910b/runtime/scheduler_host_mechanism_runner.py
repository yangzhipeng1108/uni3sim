#!/usr/bin/env python3
"""Measure isolated vLLM 0.18 Scheduler plan and update CPU service.

Synthetic requests exercise prefill, decode, and mixed control paths. No
model worker, device execution, ShareGPT request, or Serving duration enters
this measurement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


COUNTS = (1, 8, 16, 32, 64, 120)
PHASES = ("prefill", "decode", "mixed")
PROMPT_TOKENS = 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(samples: list[int]) -> dict[str, int]:
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "min_ns": ordered[0],
        "median_ns": int(statistics.median(ordered)),
        "p90_ns": ordered[(len(ordered) - 1) * 9 // 10],
        "max_ns": ordered[-1],
    }


def _scheduler(model_path: str, enable_prefix_caching: bool) -> Scheduler:
    model = ModelConfig(
        model=model_path, trust_remote_code=False, dtype="float16",
        seed=42, skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=128, max_num_batched_tokens=8192,
        max_model_len=4096, enable_chunked_prefill=True,
        is_encoder_decoder=model.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16, gpu_memory_utilization=0.9,
        cache_dtype="auto", enable_prefix_caching=enable_prefix_caching,
    )
    config = VllmConfig(
        model_config=model, scheduler_config=scheduler_config,
        cache_config=cache_config, parallel_config=ParallelConfig(),
    )
    kv_cache = KVCacheConfig(
        num_blocks=4096, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(
            ["layer"], FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=1,
                dtype=torch.float32))],
    )
    cache_config.num_gpu_blocks = 4096
    return Scheduler(
        vllm_config=config, kv_cache_config=kv_cache, block_size=16,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(config),
    )


def _update(scheduler: Scheduler, output, expected: int) -> int:
    ids = list(output.num_scheduled_tokens)
    if len(ids) != expected or any(
            count < 1 for count in output.num_scheduled_tokens.values()):
        raise ValueError("synthetic Scheduler batch differs from declared count")
    model_output = ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={request_id: index for index, request_id in enumerate(ids)},
        sampled_token_ids=[[1] for _ in ids],
    )
    start = time.perf_counter_ns()
    scheduler.update_from_output(output, model_output)
    return time.perf_counter_ns() - start


def _sample(scheduler: Scheduler, phase: str, count: int, sample: int,
            prompt_tokens: int, update_outcome: str) -> tuple[int, int]:
    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)
    if phase == "decode":
        running_count = count
    elif phase == "mixed":
        running_count = count // 2
    elif phase == "prefill":
        running_count = 0
    else:
        raise ValueError("unknown Scheduler phase")
    for index in range(running_count):
        scheduler.add_request(Request(
            request_id=f"{sample}-running-{index}",
            prompt_token_ids=[sample * max(COUNTS) + index + 1] * prompt_tokens,
            sampling_params=SamplingParams(
                max_tokens=2 if update_outcome == "finish" else 3),
            pooling_params=None, block_hasher=hasher,
        ))
    if running_count:
        warmup = scheduler.schedule()
        _update(scheduler, warmup, running_count)
    for index in range(count - running_count):
        scheduler.add_request(Request(
            request_id=f"{sample}-fresh-{index}",
            prompt_token_ids=[sample * max(COUNTS) + running_count + index + 1]
            * prompt_tokens,
            sampling_params=SamplingParams(
                max_tokens=1 if update_outcome == "finish" else 2),
            pooling_params=None, block_hasher=hasher,
        ))
    start = time.perf_counter_ns()
    output = scheduler.schedule()
    plan_ns = time.perf_counter_ns() - start
    update_ns = _update(scheduler, output, count)
    if update_outcome == "continue":
        if scheduler.get_num_unfinished_requests() != count:
            raise RuntimeError("synthetic continuing update lost requests")
        final_output = scheduler.schedule()
        _update(scheduler, final_output, count)
    if scheduler.has_unfinished_requests():
        remaining = tuple(
            (request_id, request.status.name, request.num_computed_tokens,
             request.num_output_tokens)
            for request_id, request in scheduler.requests.items())
        raise RuntimeError(
            f"synthetic Scheduler did not drain after {phase}/{count}: "
            f"{remaining[:3]!r}")
    # The source retains finished IDs until the next zero-work schedule so
    # the model runner can clear them.  Consume that control message outside
    # the timed occurrence before starting another independent sample.
    if scheduler.has_finished_requests():
        cleanup = scheduler.schedule()
        if cleanup.total_num_scheduled_tokens != 0 or scheduler.has_requests():
            raise RuntimeError("synthetic Scheduler cleanup did not drain")
    return plan_ns, update_ns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--warmup-count", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--update-outcome", choices=("finish", "continue"),
                        default="finish")
    parser.add_argument("--counts", type=int, nargs="+", default=COUNTS)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--container-name", required=True)
    args = parser.parse_args()
    if args.sample_count < 1 or args.warmup_count < 0:
        raise ValueError("sample and warmup counts are invalid")
    if (args.prompt_tokens < 1 or args.prompt_tokens > 4096
            or not args.counts or tuple(sorted(set(args.counts))) != tuple(args.counts)
            or any(count not in COUNTS for count in args.counts)
            or max(args.counts) * args.prompt_tokens > 8192):
        raise ValueError("synthetic prompt/count grid exceeds its declared domain")
    if not args.container_name.strip():
        raise ValueError("execution container must be nonempty")
    if (len(args.baseline_commit) != 40
            or any(c not in "0123456789abcdef" for c in args.baseline_commit)):
        raise ValueError("baseline commit must be a lowercase full SHA")
    version = importlib.metadata.version("vllm")
    if version not in ("0.18.0", "0.18.0+empty"):
        raise ValueError("Scheduler probe requires the installed vLLM 0.18.0 build")
    scheduler_source = Path(__import__("vllm.v1.core.sched.scheduler", fromlist=["Scheduler"]).__file__).resolve()
    cells = []
    for phase in PHASES:
        for count in args.counts:
            if phase == "mixed" and count == 1:
                continue
            scheduler = _scheduler(args.model_path, args.enable_prefix_caching)
            plan_samples = []
            update_samples = []
            for sample in range(args.warmup_count + args.sample_count):
                plan_ns, update_ns = _sample(
                    scheduler, phase, count, sample, args.prompt_tokens,
                    args.update_outcome)
                if sample >= args.warmup_count:
                    plan_samples.append(plan_ns)
                    update_samples.append(update_ns)
            cells.append({
                "phase": phase, "scheduled_request_count": count,
                "plan_ns": _summary(plan_samples),
                "update_ns": _summary(update_samples),
            })
    source = Path(__file__).resolve()
    result = {
        "schema_version": 1,
        "mechanism": "vllm_0_18_0_cpu_scheduler_plan_update",
        "boundary": "Synthetic Scheduler.schedule and update_from_output only; no worker, device, ShareGPT request, or production duration",
        "host": {"machine": platform.machine(), "platform": platform.platform()},
        "software": {"vllm_version": version,
                     "scheduler_source": str(scheduler_source),
                     "scheduler_sha256": _sha256(scheduler_source)},
        "runner": {"path": str(source), "sha256": _sha256(source)},
        "execution": {"container_name": args.container_name,
                      "baseline_commit": args.baseline_commit,
                      "model_path": args.model_path,
                      "model_config_sha256": _sha256(Path(args.model_path) / "config.json"),
                      "sample_count": args.sample_count,
                      "warmup_count": args.warmup_count,
                      "counts": args.counts},
        "synthetic_request": {"prompt_tokens": args.prompt_tokens,
                              "prefill_max_output_tokens": (
                                  1 if args.update_outcome == "finish" else 2),
                              "running_max_output_tokens": (
                                  2 if args.update_outcome == "finish" else 3),
                              "timed_update_outcome": args.update_outcome,
                              "prefix_caching": args.enable_prefix_caching,
                              "unique_prompt_per_request_and_sample": True},
        "cells": cells,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
