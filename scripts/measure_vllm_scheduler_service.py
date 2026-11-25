"""Capture an independently reproducible native-vLLM Scheduler CPU profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from statistics import median
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.backend.vllm_scheduler_process import (
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
)
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.workload.model_config import load_model_config


def _quantile(samples: tuple[int, ...], numerator: int, denominator: int) -> int:
    if not samples:
        raise ValueError("Scheduler service samples cannot be empty")
    ordered = tuple(sorted(samples))
    index = (len(ordered) - 1) * numerator // denominator
    return ordered[index]


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
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--vllm-version", choices=("0.11.0", "0.30.0"), default="0.11.0")
    parser.add_argument("--vllm030-python", type=Path)
    parser.add_argument("--vllm030-source", type=Path)
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--include-step-records", action="store_true",
        help="0.30 only: export actual source-bound per-step phase/count and Scheduler timestamps")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-host-cpu-model")
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--max-batched-tokens", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--num-kv-blocks", type=int, default=4096)
    parser.add_argument("--kv-block-size-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in (
            "sample_count", "max_batched_tokens", "max_requests",
            "num_kv_blocks", "kv_block_size_tokens", "max_model_len",
            "prompt_tokens", "output_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.prompt_tokens > args.max_batched_tokens:
        raise ValueError("--prompt-tokens cannot exceed --max-batched-tokens")
    if args.vllm_version == "0.30.0":
        if args.vllm030_python is None or args.vllm030_source is None:
            raise ValueError("0.30 measurement requires --vllm030-python and --vllm030-source")
    elif (args.vllm030_python is not None or args.vllm030_source is not None
            or args.async_scheduling):
        raise ValueError("0.30 SDK paths and async scheduling require --vllm-version 0.30.0")
    if args.include_step_records and args.vllm_version != "0.30.0":
        raise ValueError("source-bound step records require --vllm-version 0.30.0")
    cpu_names = {
        line.partition(":")[2].strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("model name")
    }
    if len(cpu_names) != 1 or not next(iter(cpu_names)):
        raise RuntimeError("Scheduler service capture needs one CPU model identity")
    cpu_model = next(iter(cpu_names))
    if args.expected_host_cpu_model is not None:
        expected = args.expected_host_cpu_model.casefold().split(" cpu @", 1)[0].strip()
        actual = cpu_model.casefold().split(" cpu @", 1)[0].strip()
        if not expected or actual != expected:
            raise RuntimeError(
                f"Host CPU {cpu_model!r} differs from expected "
                f"{args.expected_host_cpu_model!r}")
    model_path = args.model_config.resolve()
    raw_model_config = model_path.read_bytes()
    model = load_model_config(model_path)
    kv_bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(1, 1)
    settings = VLLMSchedulerConfig(
        str(model_path), max_model_len=args.max_model_len,
        enable_chunked_prefill=True, enable_prefix_caching=args.enable_prefix_caching,
        prefix_cache_mode="trace_no_hit" if args.enable_prefix_caching else "trace",
        async_scheduling=args.async_scheduling,
        scheduler_version=args.vllm_version,
    )
    if args.vllm_version == "0.30.0":
        if not isinstance(kv_bytes_per_token, int):
            raise ValueError("0.30 Scheduler microbenchmark requires integral KV bytes per token")
        backend = VLLMBackend.from_vllm030_byte_budget(
            settings, python_executable=args.vllm030_python,
            source_root=args.vllm030_source, pool_name="scheduler-microbench",
            max_batched_tokens=args.max_batched_tokens,
            max_requests=args.max_requests,
            kv_block_size_tokens=args.kv_block_size_tokens,
            available_memory_by_rank=(
                args.num_kv_blocks * args.kv_block_size_tokens * kv_bytes_per_token,),
            supported_layouts_by_rank=(("BLHNC", "BLNHC"),),
        )
    else:
        source_identity = {
            "version": VLLM_SCHEDULER_VERSION,
            "source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
        }
        backend = VLLMBackend.from_model(
            BackendConfig(
                "scheduler-microbench", "colocated", args.max_batched_tokens,
                args.max_requests, args.num_kv_blocks,
                kv_block_size_tokens=args.kv_block_size_tokens,
                kv_bytes_per_token=kv_bytes_per_token,
                scheduler_pipeline_depth=1,
            ), settings,
        )
    try:
        measurement_arguments = dict(sample_count=args.sample_count,
            requests_per_batch=args.max_requests, prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens)
        if args.include_step_records:
            measurement_arguments["include_step_records"] = True
        samples = backend.measure_scheduler_service(**measurement_arguments)
        runtime = backend.runtime_info
    finally:
        backend.close()
    runtime_manifest = runtime if isinstance(runtime, dict) else runtime.__dict__
    if args.vllm_version == "0.30.0":
        source_identity = runtime_manifest["source_identity"]
    payload = {
        "schema_version": 1,
        "hardware_id": args.hardware_id,
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "scheduler": {
            "vllm_version": source_identity["version"],
            "source_commit": source_identity["source_commit"],
            "source_identity": source_identity,
            "runtime_manifest_sha256": hashlib.sha256(
                json.dumps(runtime_manifest, sort_keys=True, default=list).encode(
                    "utf-8")).hexdigest(),
        },
        "measurement": {
            "boundary": "in_scheduler_process_schedule_and_output_update_only",
            "sample_count": args.sample_count,
            "control_workload": {
                "max_batched_tokens": args.max_batched_tokens,
                "max_requests": args.max_requests,
                "num_kv_blocks": args.num_kv_blocks,
                "kv_block_size_tokens": args.kv_block_size_tokens,
                "max_model_len": args.max_model_len,
                "requests_per_batch": args.max_requests,
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
                "async_scheduling": args.async_scheduling,
                "enable_prefix_caching": args.enable_prefix_caching,
                "model_config_sha256": hashlib.sha256(raw_model_config).hexdigest(),
            },
        },
        "service_ns": {
            "schedule": _summary(samples["schedule_ns"]),
            "update_from_output": _summary(samples["update_ns"]),
        },
        "samples_ns": {
            "schedule": samples["schedule_ns"],
            "update_from_output": samples["update_ns"],
        },
    }
    if args.include_step_records:
        source = runtime_manifest["scheduler_measurement_source"]
        if source["source_identity"] != source_identity:
            raise ValueError("measurement source disagrees with its Scheduler runtime")
        payload["scheduler"]["measurement_source"] = source
        payload["scheduler"]["scheduler_constructor"] = source["scheduler_constructor"]
        payload["software"] = {"vllm": source_identity["version"]}
        payload["measurement"]["control_workload"].update(source["control_workload"])
        payload["measurement"]["producer_file_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        payload["steps"] = []
        names = ("schedule_start_ns", "schedule_end_ns", "scheduler_update_start_ns",
            "scheduler_update_end_ns", "scheduled_request_count", "prefill_request_count", "decode_request_count")
        for index in range(args.sample_count):
            record = {name: samples[name][index] for name in names}
            prefill, decode = record["prefill_request_count"], record["decode_request_count"]
            record["step_index"] = index
            record["phase"] = "mixed" if prefill and decode else "prefill" if prefill else "decode"
            payload["steps"].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
