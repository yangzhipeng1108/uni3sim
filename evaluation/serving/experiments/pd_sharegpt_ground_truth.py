"""Replay a frozen ShareGPT case through an already-running real 1P1D router."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from evaluation.serving.experiments import sharegpt_ground_truth


def run_pd_ground_truth(
    *,
    model: str,
    dataset: str,
    case_manifest: str | Path,
    suite_manifest: str | Path,
    router_base_url: str,
    seed: int,
    max_model_len: int,
    kv_block_size_tokens: int,
    run_contract_path: str | Path,
    trace_enabled: bool,
) -> dict[str, object]:
    """Measure one frozen request stream without starting a replacement server."""

    case = sharegpt_ground_truth._load_case_contract(
        case_manifest,
        model_path=model,
        suite_manifest_path=suite_manifest,
    )
    contract = _load_contract(run_contract_path)
    _validate_contract_case(contract, case)
    request_stream = sharegpt_ground_truth.load_canonical_requests(case["requests_path"])
    requests = request_stream.requests
    if requests[0].arrival_time_s != 0:
        raise ValueError("canonical PD request schedule must start at zero")
    if any(request.output_tokens < 1 for request in requests):
        raise ValueError("canonical PD requests must generate at least one token")
    if any(request.prefix_hit_tokens != 0 for request in requests):
        raise ValueError("PD benchmark does not accept pre-injected prefix hits")
    if not isinstance(trace_enabled, bool):
        raise TypeError("PD trace_enabled must be boolean")
    tokenizer, samples = sharegpt_ground_truth._sample_sharegpt(
        model=model, dataset=dataset, seed=seed, requests=requests
    )
    prefix_evidence = sharegpt_ground_truth._prefix_cache_reuse_evidence(
        tokenizer, samples, kv_block_size_tokens=kv_block_size_tokens
    )
    import vllm

    benchmark_boundary_count = 0

    def mark_formal_window() -> None:
        nonlocal benchmark_boundary_count
        benchmark_boundary_count += 1

    metrics = asyncio.run(sharegpt_ground_truth._run_http_benchmark(
        base_url=router_base_url,
        served_model_name=case["identity"]["model_id"],
        samples=samples,
        requests=requests,
        require_request_id=True,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        benchmark_boundary_callback=mark_formal_window,
    ))
    if benchmark_boundary_count != 1:
        raise RuntimeError("PD benchmark formal-window boundary was not recorded")
    metrics["max_dispatch_lag_ms"] = max(
        request["dispatch_lag_ms"] for request in metrics["per_request"]
    )
    return {
        "schema_version": 1,
        "status": "success",
        "workload": {
            "kind": "sharegpt_canonical_schedule",
            "request_count": len(requests),
            "expected_qps": case["runtime"]["target_qps"],
            "scheduled_duration_s": requests[-1].arrival_time_s,
            "seed": seed,
            "case_manifest_path": case["manifest_path"],
            "case_manifest_sha256": case["manifest_sha256"],
            "suite_manifest_path": case["suite_manifest_path"],
            "suite_manifest_sha256": case["suite_manifest_sha256"],
            "identity": case["identity"],
            "model_config_path": case["model_config_path"],
            "model_config_sha256": case["model_config_sha256"],
            "requests_path": case["requests_path"],
            "requests_sha256": case["requests_sha256"],
            "request_stream_sha256": sharegpt_ground_truth.request_stream_sha256(
                request_stream
            ),
            "dataset_path": str(Path(dataset).resolve(strict=True)),
            "dataset_sha256": sharegpt_ground_truth.sha256_file(dataset),
            "prompt_and_output_lengths_validated": True,
            "prefix_cache_reuse_evidence": prefix_evidence,
        },
        "server": {
            "router_base_url": router_base_url.rstrip("/"),
            "client_vllm_version": vllm.__version__,
            "pd_run_id": _mapping(contract["capture_identity"], "capture identity")["run_id"],
            "profiler_attached": False,
            "scheduler_trace": {"enabled": trace_enabled},
        },
        "provenance": {
            **sharegpt_ground_truth.git_code_identity(
                sharegpt_ground_truth.ROOT
            ),
            "argv": list(sys.argv),
        },
        "metrics": metrics,
    }


def _load_contract(path: str | Path) -> Mapping[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"PD run contract is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("PD run contract is invalid JSON") from error
    contract = _mapping(value, "PD run contract")
    required = {
        "schema_version", "kind", "identity", "runtime", "capture_identity",
        "observer_effect_validation", "provenance_gaps",
    }
    if set(contract) != required:
        raise ValueError("PD run contract has an invalid field set")
    if contract["schema_version"] != 1 or contract["kind"] != "sharegpt_pd_run_contract":
        raise ValueError("PD run contract identity is invalid")
    return contract


def _validate_contract_case(
    contract: Mapping[str, object], case: Mapping[str, object]
) -> None:
    identity = _mapping(contract["identity"], "PD run identity")
    source_identity = _mapping(case["identity"], "ShareGPT case identity")
    for field in ("case_id", "hardware_id", "model_id"):
        if identity[field] != source_identity[field]:
            raise ValueError(f"PD run contract {field} differs from requested case")
    runtime = _mapping(contract["runtime"], "PD run runtime")
    if runtime["deployment"] != "pd_disaggregated":
        raise ValueError("PD run contract deployment is invalid")
    if runtime["capture_enable_prefix_caching"] is not True:
        raise ValueError("PD run contract must enable prefix caching")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay frozen ShareGPT requests through an existing 1P1D router"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--suite-manifest", required=True)
    parser.add_argument("--router-base-url", required=True)
    parser.add_argument("--run-contract", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-block-size-tokens", type=int, required=True)
    parser.add_argument(
        "--trace-enabled",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"refusing to overwrite PD ground truth: {args.output}")
    if args.seed < 0 or args.max_model_len < 1 or args.kv_block_size_tokens < 1:
        raise ValueError("PD benchmark numeric arguments are invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_pd_ground_truth(
        model=args.model,
        dataset=args.dataset,
        case_manifest=args.case_manifest,
        suite_manifest=args.suite_manifest,
        router_base_url=args.router_base_url,
        seed=args.seed,
        max_model_len=args.max_model_len,
        kv_block_size_tokens=args.kv_block_size_tokens,
        run_contract_path=args.run_contract,
        trace_enabled=args.trace_enabled,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
