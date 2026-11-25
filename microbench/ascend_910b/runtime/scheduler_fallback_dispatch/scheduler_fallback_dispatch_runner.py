#!/usr/bin/env python3
"""Measure Scheduler dispatch outside the Serving workload.

This probe creates synthetic requests in an isolated native-vLLM Scheduler
process and measures only ``plan_step``/output-update service for request
counts that are outside the published Ascend profile (121--128).  It does not
consume ShareGPT traces or production durations.  The source occurrence
contract is explicit: one synthetic admission batch, one scheduler plan, one
model-output update, repeated until the requested sample count is reached.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.measure_vllm_scheduler_service import (  # noqa: E402
    _quantile,
)
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig  # noqa: E402
from src.serving.backend.vllm_scheduler_process import (  # noqa: E402
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
)
from src.serving.schedule.backend_protocol import BackendConfig  # noqa: E402
from src.serving.workload.model_config import load_model_config  # noqa: E402


COUNTS = tuple(range(121, 129))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(samples: tuple[int, ...]) -> dict[str, int]:
    return {
        "count": len(samples),
        "min_ns": min(samples),
        "median_ns": int(median(samples)),
        "p95_ns": _quantile(samples, 95, 100),
        "max_ns": max(samples),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-id", default="ascend_910b")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--max-batched-tokens", type=int, default=8192)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.hardware_id != "ascend_910b":
        raise ValueError("this runner is owned by ascend_910b")
    if args.sample_count < 1 or args.max_batched_tokens < 128:
        raise ValueError("sample count and token capacity must be positive")
    model_path = args.model_config.resolve()
    model = load_model_config(model_path)
    def new_backend() -> VLLMBackend:
        return VLLMBackend.from_model(
            BackendConfig(
                "ascend-fallback-dispatch-microbench",
                "colocated",
                args.max_batched_tokens,
                max(COUNTS),
                4096,
                kv_block_size_tokens=16,
                kv_bytes_per_token=model.kv_bytes_per_token_per_parallel_rank(1, 1),
                scheduler_pipeline_depth=1,
            ),
            VLLMSchedulerConfig(
                str(model_path),
                max_model_len=4096,
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
            ),
        )

    cells = []
    runtime = None
    for request_count in COUNTS:
        backend = new_backend()
        try:
            samples = backend.measure_scheduler_service(
                sample_count=args.sample_count,
                requests_per_batch=request_count,
                prompt_tokens=1,
                output_tokens=1,
            )
            cells.append({
                "scheduled_request_count": request_count,
                "plan_ns": _summary(samples["schedule_ns"]),
                "update_ns": _summary(samples["update_ns"]),
            })
            runtime = backend.runtime_info
        finally:
            backend.close()
    if runtime is None:
        raise RuntimeError("scheduler microbenchmark produced no runtime manifest")
    runner = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "kind": "ascend_scheduler_fallback_dispatch_microbenchmark",
        "hardware_id": args.hardware_id,
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "runner": {"path": str(runner), "sha256": _sha256(runner)},
        "scheduler": {
            "vllm_version": VLLM_SCHEDULER_VERSION,
            "source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
            "runtime_manifest_sha256": hashlib.sha256(
                json.dumps(runtime.__dict__, sort_keys=True, default=list).encode()
            ).hexdigest(),
        },
        "measurement": {
            "boundary": "isolated_scheduler_plan_and_output_update_only",
            "sample_count": args.sample_count,
            "synthetic_request": {"input_tokens": 1, "output_tokens": 1},
            "occurrence_contract": {
                "admission": "one_synthetic_batch_per_sample",
                "plan": "one_plan_step_per_sample",
                "update": "one_model_output_update_per_sample",
                "production_duration_consumed": False,
                "serving_trace_consumed": False,
            },
        },
        "cells": cells,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
