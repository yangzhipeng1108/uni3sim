#!/usr/bin/env python3
"""Compare source-derived activation storage with existing startup records."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.serving.metrics.activation_memory import (
    compare_symbolic_startup_logs, write_activation_comparison, write_symbolic_startup_storage_report,
)
from src.serving.workload.runtime_inputs import ExpertRoutingInput
import json


def _unique_compilation_options(pairs):
    values = {}
    for name, value in pairs:
        if name in values:
            raise ValueError("duplicate source compilation option: " + name)
        values[name] = value
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runtime-memory-policy", nargs="+")
    source.add_argument("--startup-log", nargs="+")
    source.add_argument("--source-only", action="store_true")
    parser.add_argument("--max-batched-tokens", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--source-profile")
    parser.add_argument("--model-execution-version", choices=("vllm_0_11_0", "vllm_0_18_0", "vllm_0_30_0"), default="vllm_0_30_0",
        help="version-bound dense startup;0.11 supports explicit PyTorch-sort/FlashInfer0.4.1, Ascend0.18 is head-only")
    parser.add_argument("--source-lifetime-policy", choices=("vllm_0_18_npugraph_ex_piecewise",),
        help="explicit source-only partition lifetime policy; does not select a runtime backend")
    parser.add_argument("--source-compilation-options-json",
        help="complete declared upstream compile options as a JSON object; only with --source-only")
    parser.add_argument("--upstream-vllm-root", type=Path, help="independent AST check of known worker device-buffer layouts")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--mtp-num-speculative-tokens", type=int, choices=(0, 2), default=0,
        help="GLM startup target-only (0) or two sequential MTP dummy passes (2); source storage only")
    parser.add_argument("--v4-source-use-deep-gemm", choices=("enabled", "disabled"))
    parser.add_argument("--source-workspace-initial-capacity-bytes", nargs="+", type=int)
    parser.add_argument("--source-indexer-max-logits-mb", type=int,
        help="explicit VLLM_SPARSE_INDEXER_MAX_LOGITS_MB source condition, in MiB; not read from this process environment")
    parser.add_argument("--expert-routing-source-id")
    parser.add_argument("--expert-routing-seed", type=int)
    parser.add_argument("--hardware-config", type=Path,
                        help="source-only CUDA sampler SM count from an existing HWSpec")
    parser.add_argument("--base-runtime-memory-policy", type=Path)
    parser.add_argument("--framework-source-epoch", choices=("before_profile", "after_profile"),
        help="explicit epoch covered by the base framework reserve; required for V4/GLM policy projection")
    parser.add_argument("--derived-runtime-memory-policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", choices=("symbolic_forward", "dense_mlp", "symbolic_startup"), default="symbolic_forward")
    args = parser.parse_args()
    compilation_options = None
    if args.source_lifetime_policy is not None or args.source_compilation_options_json is not None:
        if not args.source_only:
            parser.error("source lifetime and compilation options require --source-only")
    if args.source_compilation_options_json is not None:
        try:
            values = json.loads(args.source_compilation_options_json, object_pairs_hook=_unique_compilation_options)
        except ValueError as error:
            parser.error(str(error))
        if not isinstance(values, dict):
            parser.error("source compilation options must be a JSON object")
        compilation_options = tuple((name, tuple(value) if name == "splitting_ops" and isinstance(value, list)
            else value) for name, value in values.items())
    if args.source_only:
        if (args.program != "symbolic_startup" or args.max_batched_tokens is None
                or args.max_requests is None or args.source_profile is None):
            parser.error("source-only mode requires symbolic_startup, token/request capacities and an explicit source profile")
        routing = None
        if (args.expert_routing_source_id is None) != (args.expert_routing_seed is None):
            parser.error("routing source identity and seed must be supplied together")
        if args.expert_routing_source_id is not None:
            routing = ExpertRoutingInput.seeded_assumption(args.expert_routing_source_id, args.expert_routing_seed)
        result = write_symbolic_startup_storage_report(args.model, args.output, tokens=args.max_batched_tokens,
            max_requests=args.max_requests, source_profile=args.source_profile, hardware_config=args.hardware_config,
            base_runtime_policy=args.base_runtime_memory_policy, derived_runtime_policy=args.derived_runtime_memory_policy,
            tensor_parallel_size=args.tensor_parallel_size, source_max_model_len=args.max_model_len,
            source_use_deep_gemm=None if args.v4_source_use_deep_gemm is None else args.v4_source_use_deep_gemm == "enabled",
            workspace_initial_capacity_bytes=None if args.source_workspace_initial_capacity_bytes is None else tuple(args.source_workspace_initial_capacity_bytes),
            indexer_max_logits_mb=args.source_indexer_max_logits_mb, expert_routing=routing,
            framework_source_epoch=args.framework_source_epoch, upstream_root=args.upstream_vllm_root,
            mtp_num_speculative_tokens=args.mtp_num_speculative_tokens, model_execution_version=args.model_execution_version,
            source_lifetime_policy=args.source_lifetime_policy, source_compilation_options=compilation_options)
        print(result["summary"])
        return (int(result["worker_storage_source_comparison"]["failed"] != 0)
            if "worker_storage_source_comparison" in result else 0)
    if args.model_execution_version != "vllm_0_30_0":
        parser.error("versioned0.11 startup storage requires --source-only; log/runtime binding is not implemented")
    if args.startup_log is not None:
        if (args.max_batched_tokens is None or args.max_requests is None
                or args.source_profile is None or args.program != "symbolic_startup"):
            parser.error("startup logs require symbolic_startup, token/request capacities and an explicit source profile")
        if (args.base_runtime_memory_policy is not None or args.derived_runtime_memory_policy is not None
                or args.tensor_parallel_size != 1 or args.mtp_num_speculative_tokens or any(value is not None for value in (
                    args.max_model_len, args.v4_source_use_deep_gemm, args.source_workspace_initial_capacity_bytes,
                    args.source_indexer_max_logits_mb, args.expert_routing_source_id, args.expert_routing_seed,
                    args.framework_source_epoch, args.upstream_vllm_root))):
            parser.error("startup log comparison currently binds declared Qwen TP1 storage only; no policy promotion")
        result = compare_symbolic_startup_logs(args.model, tuple(args.startup_log), tokens=args.max_batched_tokens,
            max_requests=args.max_requests, source_profile=args.source_profile, hardware_config=args.hardware_config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print("numeric:", result["summary"], "acceptance:", result["acceptance_summary"])
        return 0 if result["acceptance_summary"]["pass"] == result["acceptance_summary"]["total"] else 1
    if (args.max_requests is not None or args.source_profile is not None or args.program == "symbolic_startup"
            or args.hardware_config is not None or args.base_runtime_memory_policy is not None
            or args.derived_runtime_memory_policy is not None or args.tensor_parallel_size != 1 or args.mtp_num_speculative_tokens
            or any(value is not None for value in (args.max_model_len, args.v4_source_use_deep_gemm,
                args.source_workspace_initial_capacity_bytes, args.source_indexer_max_logits_mb,
                args.expert_routing_source_id, args.expert_routing_seed, args.framework_source_epoch, args.upstream_vllm_root))):
        parser.error("startup source choices require --source-only; observed runtime binding is not implemented")
    if args.max_batched_tokens is not None:
        parser.error("policy mode takes token capacity from its binding")
    result = write_activation_comparison(args.model, tuple(args.runtime_memory_policy), args.output, program=args.program)
    print(result["summary"])
    return 0 if all(value == 0 for name, value in result["summary"].items() if name not in ("total", "pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
