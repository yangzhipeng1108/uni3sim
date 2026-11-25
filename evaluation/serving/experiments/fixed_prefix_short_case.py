"""Synthetic full-request check for 0.30 trace hits and fixed KV tiers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from evaluation.serving.simulation.execution_artifact import (
    StreamingExecutionArtifactAccumulator, execution_artifact_manifest,
)
from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.deployment.deployment_config import DeploymentConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.kv_cache.kv_cache_config import KVCacheConfig, KVCacheTierConfig
from src.serving.kv_cache.kv_cache_events import KVCacheAction
from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.simulation import Simulation, SimulationConfig
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs


def run_short_case(
    *, sdk_python: Path, sdk_source: Path,
) -> dict[str, object]:
    model_path = "configs/model/qwen3-tp2-example"
    model = load_model_config(model_path)
    block_tokens = 16
    block_bytes = block_tokens * model.kv_bytes_per_token_per_parallel_rank(1, 1)
    cache = KVCacheConfig((
        KVCacheTierConfig("hbm", 16 * block_bytes, block_tokens),
        KVCacheTierConfig("host", 16 * block_bytes, block_tokens, "pcie"),
    ), block_bytes // block_tokens, "lru")
    pool = WorkerPoolConfig("main", "colocated", "nvidia_a800",
        ParallelConfig(), kv_cache=cache)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (pool,)), num_layers=model.num_hidden_layers)
    requests = RequestStream((
        RequestSpec(0, "a", 96, 4, prefix_hit_tokens=80),
        RequestSpec(0.01, "b", 96, 4, prefix_hit_tokens=80),
    ))
    runtime = RuntimeInputs(ExpertRoutingInput.not_applicable(),
        "paged_varlen", "uniform", model_execution_version="vllm_0_30_0",
        target_sampling_source="vllm_0_30_plain_greedy_bf16",
        symbolic_forward_source="cuda_eager_inplace")
    config = SimulationConfig(model, deployment, requests, "analytical_launch",
        "roofline", runtime)
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    scenarios = []
    def make_backend():
        return VLLMBackend.from_vllm030_byte_budget(
            VLLMSchedulerConfig(model=model_path, max_model_len=128,
                tensor_parallel_size=1, enable_chunked_prefill=False,
                scheduler_version="0.30.0"),
            python_executable=sdk_python, source_root=sdk_source,
            pool_name="main", max_batched_tokens=128, max_requests=1,
            kv_block_size_tokens=block_tokens,
            available_memory_by_rank=(16 * block_bytes,),
            supported_layouts_by_rank=(("BLHNC", "BLNHC"),))

    for percent in (100, 80, 0):
        backend = make_backend()
        manager = KVCacheTierManager(cache, fixed_prefix_hbm_percent=percent)
        started = time.perf_counter()
        try:
            result = Simulation(config, backend,
                {"nvidia_a800": hardware}, kv_cache_tier_manager=manager).run()
        finally:
            backend.close()
        wall_s = time.perf_counter() - started
        if set(result.metrics.requests) != {"a", "b"}:
            raise RuntimeError("synthetic short case did not complete both requests")
        first_schedules = [scheduled for step in result.steps
            for scheduled in step.backend_step.schedule.requests
            if scheduled.trace_prefix_hit_tokens]
        if len(first_schedules) != 2:
            raise RuntimeError("synthetic short case lost first-schedule hit facts")
        transfers = result.kv_offload.transfers
        hit_events = [event for step in result.steps
            for event in step.backend_step.kv_cache_events
            if event.action is KVCacheAction.HIT]
        hbm_hit_blocks = sum((event.token_end - event.token_start)
            // block_tokens for event in hit_events if event.tier == "hbm")
        host_hit_blocks = sum((event.token_end - event.token_start)
            // block_tokens for event in hit_events if event.tier == "host")
        if any(record.event.action.value != "load" for record in transfers):
            raise RuntimeError("fixed tier case unexpectedly saved KV to Host")
        if result.kv_offload.final_tier_state["residency"]:
            raise RuntimeError("synthetic short case retained request-private KV")
        artifact = execution_artifact_manifest(result,
            performance_timing_status="unsupported_a800_analytical_launch")
        scenarios.append({
            "hbm_percent": percent,
            "host_percent": 100 - percent,
            "completed_requests": len(result.metrics.requests),
            "observed_hit_tokens_per_request": 80,
            "effective_hit_tokens_per_request": [
                item.scheduler_prefix_hit_tokens for item in first_schedules],
            "hbm_hit_blocks_total": hbm_hit_blocks,
            "host_hit_blocks_total": host_hit_blocks,
            "h2d_bytes_per_rank": sum(record.event.bytes for record in transfers),
            "d2h_bytes_per_rank": 0,
            "transfer_intervals_s": [
                [record.completion.start_time_s, record.completion.end_time_s]
                for record in transfers],
            "metrics": asdict(result.metrics),
            "simulation_wall_s": wall_s,
            "execution_artifact_schema_version": artifact["schema_version"],
            "execution_artifact_sha256": artifact["artifact_sha256"],
            "final_tier_state": result.kv_offload.final_tier_state,
        })
    streaming_artifact = StreamingExecutionArtifactAccumulator()
    streaming_backend = make_backend()
    streaming_started = time.perf_counter()
    try:
        streamed = Simulation(replace(config,
            step_retention="streaming_aggregate"), streaming_backend,
            {"nvidia_a800": hardware},
            kv_cache_tier_manager=KVCacheTierManager(
                cache, fixed_prefix_hbm_percent=80),
            streaming_step_consumer=streaming_artifact.consume).run()
    finally:
        streaming_backend.close()
    streaming_wall_s = time.perf_counter() - streaming_started
    if asdict(streamed.metrics) != scenarios[1]["metrics"]:
        raise RuntimeError("streaming KV scenario changed request metrics")
    compact_artifact = streaming_artifact.manifest(streamed,
        performance_timing_status="unsupported_a800_analytical_launch")
    control_requests = RequestStream((
        RequestSpec(0, "a", 96, 4),
        RequestSpec(0.01, "b", 96, 4),
    ))
    control_config = replace(config, requests=control_requests)
    control_results = []
    for use_tier_manager in (False, True):
        control_backend = make_backend()
        try:
            control = Simulation(control_config, control_backend,
                {"nvidia_a800": hardware},
                kv_cache_tier_manager=(KVCacheTierManager(
                    cache, fixed_prefix_hbm_percent=80)
                    if use_tier_manager else None)).run()
        finally:
            control_backend.close()
        if set(control.metrics.requests) != {"a", "b"}:
            raise RuntimeError("no-hit control did not complete both requests")
        control_results.append(control)
    baseline, tier_enabled = control_results
    baseline_calls = sum(len(step.kernel_plan.calls) for step in baseline.steps)
    tier_calls = sum(len(step.kernel_plan.calls) for step in tier_enabled.steps)
    if asdict(baseline.metrics) != asdict(tier_enabled.metrics):
        raise RuntimeError("no-hit tier manager changed offload-off request metrics")
    if baseline_calls != tier_calls or len(baseline.steps) != len(tier_enabled.steps):
        raise RuntimeError("no-hit tier manager changed offload-off execution work")
    if baseline.kv_offload is not None:
        raise RuntimeError("offload-off control unexpectedly reported KV transfers")
    if tier_enabled.kv_offload is None or tier_enabled.kv_offload.transfers:
        raise RuntimeError("no-hit tier manager unexpectedly reported KV transfers")
    if tier_enabled.kv_offload.final_tier_state["residency"]:
        raise RuntimeError("no-hit tier manager retained request-private KV")
    return {
        "source_kind": "synthetic_fixture",
        "model_path": model_path,
        "model_config_sha256": model.config_sha256,
        "requests": [asdict(request) for request in requests],
        "block_tokens": block_tokens,
        "block_bytes_per_rank": block_bytes,
        "hbm_capacity_bytes_per_rank": 16 * block_bytes,
        "host_capacity_bytes_per_rank": 16 * block_bytes,
        "placement_assumption": "request-private hit blocks prewarmed at arrival; Host load retains both valid copies",
        "transfer_cost_assumption": "A800 HWSpec PCIe 32 GB/s roofline byte lower bound; startup latency unavailable and omitted",
        "scenarios": scenarios,
        "streaming_80": {
            "completed_requests": len(streamed.metrics.requests),
            "num_steps": streamed.num_steps,
            "num_kernel_calls": streamed.num_kernel_calls,
            "h2d_bytes_per_rank": compact_artifact["streaming_aggregate"]["modeled_kv_offload"]["h2d_bytes_per_rank"],
            "execution_artifact_schema_version": compact_artifact["schema_version"],
            "execution_artifact_sha256": compact_artifact["artifact_sha256"],
            "completion_records_sha256": compact_artifact["streaming_aggregate"]["modeled_kv_offload"]["completion_records_sha256"],
            "wall_s": streaming_wall_s,
        },
        "offload_off_no_hit_control": {
            "requests": [asdict(request) for request in control_requests],
            "completed_requests": len(baseline.metrics.requests),
            "steps": len(baseline.steps),
            "kernel_calls": baseline_calls,
            "metrics": asdict(baseline.metrics),
            "tier_enabled_metrics_equal": True,
            "tier_enabled_work_equal": True,
            "tier_enabled_transfer_count": 0,
            "tier_enabled_final_tier_state": tier_enabled.kv_offload.final_tier_state,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-python", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_short_case(
        sdk_python=args.sdk_python, sdk_source=args.sdk_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
