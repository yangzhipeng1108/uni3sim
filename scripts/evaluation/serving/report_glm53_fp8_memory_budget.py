#!/usr/bin/env python3
"""Emit GLM-5.3 FP8 per-rank HBM and native KV allocation budgets."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm import VLLMSchedulerConfig
from src.serving.kv_cache.kv_capacity import (
    load_runtime_memory_policy, resolve_glm_block_fp8_kv_capacity_plan)
from src.serving.kv_cache.vllm_allocation import allocate_vllm030_kv_cache
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/model/glm-5.3.json")
    parser.add_argument("--hardware-config", default="configs/hardware/nvidia_h200.yaml")
    parser.add_argument("--runtime-memory-policy",
        default="configs/runtime_memory_policy/nvidia_h200/glm-5.3-fp8-tp8-pd.json")
    parser.add_argument("--vllm-sdk-python", type=Path, required=True)
    parser.add_argument("--vllm-source-root", type=Path, required=True)
    parser.add_argument("--mtp-num-speculative-tokens", type=int, default=0,
        choices=(0, 5), help="decode-only GLM MTP proposer depth; prefill pool always resolves at depth 0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_model_config(args.model)
    hardware = HWSpec(args.hardware_config)
    policies = load_runtime_memory_policy(args.runtime_memory_policy)
    parallel = ParallelConfig(tensor_parallel_size=8)
    hardware_sha256 = hashlib.sha256(Path(args.hardware_config).read_bytes()).hexdigest()
    if args.mtp_num_speculative_tokens:
        # The capacity path binds the decode-stage K5 scope through the
        # process gate; the assumption supplies the declared A60 source
        # identity. It is a configuration-only binding: the budget run never
        # submits these synthetic requests to a backend, and the hash is a
        # real content address of that declared non-executing stream.
        from src.serving.backend.mtp_acceptance_assumption import (
            MTPAcceptanceAssumption)
        from src.serving.request_gen.request_stream import (
            RequestSpec, RequestStream, request_stream_sha256)
        synthetic_binding_stream = RequestStream((
            RequestSpec(arrival_time_s=0.0,
                request_id="glm53-mtp5-budget-binding-row-0",
                input_tokens=1, output_tokens=1, prefix_hit_tokens=0),
        ))
        binding_stream_sha256 = request_stream_sha256(synthetic_binding_stream)
        budget_assumption = MTPAcceptanceAssumption(
            source_id="glm53-support-memory-budget-v1",
            proposal_depth=args.mtp_num_speculative_tokens,
            mean_acceptance_rate="0.6",
            root_seed=0,
            request_stream_sha256=binding_stream_sha256)

    def build(pool_name):
        policy = policies.policy_for_pool(pool_name)
        binding = policy.binding
        depth = (args.mtp_num_speculative_tokens
            if binding.worker_role == "decode" else 0)
        settings = VLLMSchedulerConfig(
            args.model, tensor_parallel_size=8, pipeline_parallel_size=1,
            max_model_len=binding.max_model_len,
            gpu_memory_utilization=(policy.gpu_memory_utilization_numerator
                / policy.gpu_memory_utilization_denominator),
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            prefix_cache_mode="trace", scheduler_version="0.30.0",
            mtp_num_speculative_tokens=depth,
            mtp_acceptance_policy=("maximum_entropy" if depth else "disabled"),
            mtp_acceptance_source_id=("glm53-support-memory-budget-v1" if depth else "disabled"),
            mtp_acceptance_assumption=(budget_assumption if depth else None),
            request_stream_sha256=(binding_stream_sha256 if depth else None))
        allocator = partial(
            allocate_vllm030_kv_cache, scheduler_config=settings,
            python_executable=args.vllm_sdk_python, source_root=args.vllm_source_root,
            max_batched_tokens=binding.max_batched_tokens, max_requests=binding.max_requests,
            supported_layouts_by_rank=(("LBHNC",),) * 8, pool_name=pool_name,
            pipeline_layer_partition=None,
            worker_role=binding.worker_role,
            process_consumer=lambda process: process.abort_observation())
        projected, plan, report = resolve_glm_block_fp8_kv_capacity_plan(
            policy, model=model, parallelism=parallel, hardware=hardware,
            hardware_config_sha256=hardware_sha256, allocator=allocator,
            expert_routing=ExpertRoutingInput.seeded_assumption(
                "glm53-support-memory-budget-v1", 0),
            mtp_num_speculative_tokens=depth)
        return {
            "pool_name": pool_name,
            "mtp_num_speculative_tokens": depth,
            "projected_policy": asdict(projected),
            "capacity_plan": asdict(plan),
            "source_report": report,
        }

    pool_names = tuple(sorted(policies.policies))
    with ThreadPoolExecutor(max_workers=len(pool_names)) as executor:
        pools = tuple(executor.map(build, pool_names))
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "CPU analytical HBM budget with real vLLM0.30 native allocation; not measured H200 runtime acceptance",
        "mtp_acceptance_binding": (None if not args.mtp_num_speculative_tokens else {
            "kind": "configuration_only_non_executing",
            "synthetic_request_stream_sha256": binding_stream_sha256,
            "synthetic_request_stream_rows": [
                {
                    "arrival_time_s": request.arrival_time_s,
                    "request_id": request.request_id,
                    "input_tokens": request.input_tokens,
                    "output_tokens": request.output_tokens,
                    "prefix_hit_tokens": request.prefix_hit_tokens,
                }
                for request in synthetic_binding_stream.requests
            ],
            "note": "the budget allocator never submits these rows to a backend; "
                "the hash only binds the declared K5 maximum-entropy acceptance "
                "source to this non-executing configuration",
        }),
        "model": args.model,
        "model_config_sha256": model.config_sha256,
        "hardware_config": args.hardware_config,
        "hardware_config_sha256": hardware_sha256,
        "runtime_memory_policy": args.runtime_memory_policy,
        "mtp_num_speculative_tokens": args.mtp_num_speculative_tokens,
        "parallelism": asdict(parallel),
        "pools": pools,
        "normal_capacity_accepted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for pool in pools:
        plan = pool["capacity_plan"]
        available = [rank["available_kv_bytes"] for rank in plan["rank_budgets"]]
        print(f"{pool['pool_name']}: blocks={plan['num_kv_blocks']} "
              f"available_kv_bytes={min(available)}..{max(available)}")


if __name__ == "__main__":
    main()
