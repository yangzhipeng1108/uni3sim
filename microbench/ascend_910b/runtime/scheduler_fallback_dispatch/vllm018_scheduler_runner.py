#!/usr/bin/env python3
"""Direct vLLM 0.18 Scheduler fallback-dispatch microbenchmark.

The probe is intentionally independent of Midlayer Serving code.  It creates
the vLLM Scheduler with a synthetic one-token request batch, measures one
``schedule`` and one ``update_from_output`` occurrence, and repeats this for
request counts 121--128.  No ShareGPT trace or production duration is read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.outputs import ModelRunnerOutput


COUNTS = tuple(range(121, 129))


def _summary(samples: list[int]) -> dict[str, int]:
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "min_ns": min(ordered),
        "median_ns": int(statistics.median(ordered)),
        "p95_ns": ordered[(len(ordered) - 1) * 95 // 100],
        "max_ns": max(ordered),
    }


def _scheduler(count: int, model_path: str) -> Scheduler:
    model = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=128,
        max_num_batched_tokens=8192,
        max_model_len=4096,
        enable_chunked_prefill=True,
        is_encoder_decoder=model.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    config = VllmConfig(
        model_config=model,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    kv_cache = KVCacheConfig(
        num_blocks=4096,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(
            ["layer"], FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=1,
                dtype=torch.float32))],
    )
    cache_config.num_gpu_blocks = 4096
    return Scheduler(
        vllm_config=config,
        kv_cache_config=kv_cache,
        block_size=16,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(config),
    )


def _one_sample(scheduler: Scheduler, count: int, sample_index: int) -> tuple[int, int]:
    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)
    sampling = SamplingParams(max_tokens=2)
    requests = [Request(
        request_id=f"{sample_index}-{index}", prompt_token_ids=[index],
        sampling_params=sampling, pooling_params=None,
        block_hasher=hasher) for index in range(count)]
    for request in requests:
        scheduler.add_request(request)
    # Advance the synthetic requests once so the measured occurrence is the
    # decode/fallback dispatch state rather than the initial pure-prefill
    # admission.  This warmup is outside the timed interval.
    warmup_output = scheduler.schedule()
    warmup_ids = list(warmup_output.num_scheduled_tokens)
    scheduler.update_from_output(
        warmup_output,
        ModelRunnerOutput(
            req_ids=warmup_ids,
            req_id_to_index={request_id: index for index, request_id in enumerate(warmup_ids)},
            sampled_token_ids=[[1] for _ in warmup_ids],
        ),
    )
    start = time.perf_counter_ns()
    output = scheduler.schedule()
    schedule_ns = time.perf_counter_ns() - start
    req_ids = list(output.num_scheduled_tokens)
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={request_id: index for index, request_id in enumerate(req_ids)},
        sampled_token_ids=[[1] for _ in req_ids],
    )
    start = time.perf_counter_ns()
    scheduler.update_from_output(output, model_output)
    update_ns = time.perf_counter_ns() - start
    return schedule_ns, update_ns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--model-path", default="facebook/opt-125m")
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--device-id", type=int, required=True)
    args = parser.parse_args()
    if args.sample_count < 1 or args.device_id not in (0, 1):
        raise ValueError("sample count must be positive")
    if not args.container_name.strip():
        raise ValueError("execution container must be nonempty")
    if len(args.baseline_commit) != 40 or any(
            character not in "0123456789abcdef" for character in args.baseline_commit):
        raise ValueError("baseline commit must be a full lowercase SHA")
    cells = []
    for count in COUNTS:
        plan = []
        update = []
        scheduler = _scheduler(count, args.model_path)
        for sample_index in range(args.sample_count):
            plan_ns, update_ns = _one_sample(scheduler, count, sample_index)
            plan.append(plan_ns)
            update.append(update_ns)
        cells.append({
            "scheduled_request_count": count,
            "plan_ns": _summary(plan),
            "update_ns": _summary(update),
        })
    source = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "kind": "ascend_scheduler_fallback_dispatch_microbenchmark",
        "hardware_id": "ascend_910b",
        "host": {"machine": platform.machine(), "platform": platform.platform()},
        "execution": {
            "container_name": args.container_name,
            "device_id": args.device_id,
            "baseline_commit": args.baseline_commit,
        },
        "runner": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "measurement": {
            "boundary": "direct_vllm_scheduler_schedule_and_update_only",
            "synthetic_request": {"input_tokens": 1, "output_tokens": 1},
            "occurrence_contract": {
                "admission": "one synthetic batch per sample",
                "plan": "one schedule occurrence per sample",
                "update": "one update_from_output occurrence per sample",
                "production_or_evaluation_duration_consumed": False,
            },
            "sample_count": args.sample_count,
        },
        "cells": cells,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
