#!/usr/bin/env python3
"""Run ordinary Kimi-K3 text Serving or its native H200 capacity plan."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.deployment.deployment_config import DeploymentConfig, PipelineP2PConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.kv_cache.kv_capacity import (
    kv_capacity_manifest, load_runtime_memory_policy, resolve_analytical_kv_capacity_plan,
    validate_source_activation_capacity,
)
from src.serving.kv_cache.activation_memory import SourceActivationMemoryAccumulator, symbolic_forward_storage_report
from src.serving.kv_cache.model_residency import kimi_k3_source_workspace_bytes
from src.serving.kv_cache.vllm_allocation import allocate_vllm030_kv_cache
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.simulation import Simulation, SimulationConfig
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs
from src.serving.workload.traced_models import build_symbolic_workload


def _step_report(step):
    coverage = {}
    prediction_time_sum_s = 0.0
    for call in step.kernel_plan.calls:
        spec = call.kernel_spec
        key = f"{spec.kind.value}/{spec.variant}"
        if key not in coverage:
            coverage[key] = {
                "calls": 0,
                "analytical_source_calls": 0,
                "analytical_flops": 0,
                "typed_logical_read_bytes": 0.0,
                "typed_logical_write_bytes": 0.0,
                "prediction_time_sum_s": 0.0,
            }
        prediction = step.predictions[call.call_id]
        flops = 0
        if spec.kind.value == "GEMM":
            flops = 2 * spec.shape["b"] * spec.shape["m"] * spec.shape["n"] * spec.shape["k"]
        elif spec.kind.value == "FUSED_MOE":
            flops = (6 * spec.shape["tokens"] * spec.shape["topk"]
                * spec.shape["hidden"] * spec.shape["intermediate"])
        read_bytes = 0.0
        write_bytes = 0.0
        if prediction.diagnostics is not None:
            for access in prediction.diagnostics.memory_access_classes:
                if access.access_kind == "read":
                    read_bytes += access.logical_bytes
                elif access.access_kind == "write":
                    write_bytes += access.logical_bytes
                else:
                    raise ValueError(f"unknown typed memory access kind: {access.access_kind}")
        coverage[key]["calls"] += 1
        coverage[key]["analytical_source_calls"] += int(spec.execution_support == "unsupported")
        coverage[key]["analytical_flops"] += flops
        coverage[key]["typed_logical_read_bytes"] += read_bytes
        coverage[key]["typed_logical_write_bytes"] += write_bytes
        coverage[key]["prediction_time_sum_s"] += prediction.t_total_s
        prediction_time_sum_s += prediction.t_total_s
    return {
        "step_id": step.backend_step.schedule.step_id,
        "scheduled_requests": [asdict(request) for request in step.backend_step.schedule.requests],
        "graph_nodes": len(step.workload.nodes),
        "kernel_calls": len(step.kernel_plan.calls),
        "priced_predictions": len(step.predictions),
        "unpriced_calls": len(step.kernel_plan.calls) - len(step.predictions),
        "predicted_duration_s": step.execution_timeline.duration_s,
        "kernel_prediction_time_sum_s": prediction_time_sum_s,
        "analytical_flops": sum(variant["analytical_flops"] for variant in coverage.values()),
        "typed_logical_read_bytes": sum(variant["typed_logical_read_bytes"] for variant in coverage.values()),
        "typed_logical_write_bytes": sum(variant["typed_logical_write_bytes"] for variant in coverage.values()),
        "call_coverage": coverage,
        "critical_path_contributions": [
            {"call_id": call_id, "duration_s": duration_s}
            for call_id, duration_s in step.execution_timeline.critical_path_contributions
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="configs/model/kimi-k3.json")
    parser.add_argument("--hardware", default="configs/hardware/nvidia_h200.yaml")
    parser.add_argument("--layout", choices=("pp16", "tp16", "tp1"), default="pp16")
    parser.add_argument("--prediction-method", choices=("midlayer", "roofline"), default="roofline")
    parser.add_argument("--prompt-tokens", type=int, action="append",
                        help="request prompt length; repeat for a small concurrent stream")
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=640)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-requests", type=int, default=2)
    parser.add_argument("--vllm-sdk-python", type=Path,
                        default=Path(os.environ["MIDLAYER_VLLM030_PYTHON"])
                        if "MIDLAYER_VLLM030_PYTHON" in os.environ else None)
    parser.add_argument("--vllm-source-root", type=Path,
                        default=Path(os.environ["MIDLAYER_VLLM030_SOURCE"])
                        if "MIDLAYER_VLLM030_SOURCE" in os.environ else None)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--source-memory-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prompts = tuple(args.prompt_tokens or (512,))
    if args.capacity_only and args.source_memory_only:
        parser.error("capacity-only and source-memory-only are separate report modes")
    if args.layout == "tp16" and not args.capacity_only:
        parser.error("TP16 currently supports capacity-only reports; use TP1 or PP16 for requests/source memory")
    if (args.vllm_sdk_python is None or args.vllm_source_root is None
            or not args.vllm_sdk_python.is_file() or not args.vllm_source_root.is_dir()):
        parser.error("explicit existing vLLM0.30 SDK Python and source checkout are required")
    if any(type(value) is not int or value < 512 for value in prompts):
        parser.error("K3 ordinary source requires prompt lengths >= 512")
    if args.output_tokens < 1 or args.output_tokens > 8:
        parser.error("initial K3 validation uses 1--8 output tokens per request")
    if max(prompts) + args.output_tokens > args.max_model_len:
        parser.error("prompt plus output exceeds max_model_len")
    if args.max_requests < len(prompts):
        parser.error("max_requests cannot be smaller than the request stream")
    model = load_model_config(args.model)
    hardware = HWSpec(args.hardware)
    tp, pp = {"pp16": (1, 16), "tp16": (16, 1), "tp1": (1, 1)}[args.layout]
    parallelism = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp)
    source_workspace = kimi_k3_source_workspace_bytes(
        model, parallelism, max_batched_tokens=args.max_batched_tokens,
        max_requests=args.max_requests, source_multiprocessors=hardware.num_compute_units)
    scheduler = VLLMSchedulerConfig(args.model, max_model_len=args.max_model_len,
        tensor_parallel_size=tp, pipeline_parallel_size=pp, enable_chunked_prefill=False,
        scheduler_version="0.30.0")
    policy_path = Path(f"configs/runtime_memory_policy/nvidia_h200/kimi-k3-{args.layout}.json")
    normal = args.layout != "tp1"
    capacity = None
    policies = None
    if normal:
        policies = load_runtime_memory_policy(policy_path)
        policy = policies.policy_for_pool("main")
        allocator = partial(allocate_vllm030_kv_cache, scheduler_config=scheduler,
            python_executable=args.vllm_sdk_python, source_root=args.vllm_source_root,
            max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests,
            supported_layouts_by_rank=(("BLHNC",),) * parallelism.worker_count)
        capacity = resolve_analytical_kv_capacity_plan("main", hardware=hardware,
            hardware_config_sha256=policy.binding.hardware_config_sha256,
            model=model, parallelism=parallelism, deployment="colocated", worker_role="colocated",
            kv_block_size_tokens=args.max_model_len, max_model_len=args.max_model_len,
            max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests,
            policy=policy, allocator=allocator, scheduler_version="0.30.0")
        if args.capacity_only:
            payload = kv_capacity_manifest((capacity,), policies, models_by_pool={"main": model})
            payload["source_workspace_budget"] = source_workspace
            payload["scope"] = "native CPU0.30 hybrid allocation, source weights and bounded source reserve ledger; actual H200 runtime not measured"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(args.output)
            return
        backend = VLLMBackend.from_vllm030_capacity_plan(capacity, scheduler,
            python_executable=args.vllm_sdk_python, source_root=args.vllm_source_root,
            max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests,
            supported_layouts_by_rank=(("BLHNC",),) * parallelism.worker_count)
    else:
        if args.capacity_only:
            parser.error("TP1 is a mathematical main chain, not a deployable H200 capacity plan")
        backend = VLLMBackend.from_vllm030_byte_budget(scheduler,
            python_executable=args.vllm_sdk_python, source_root=args.vllm_source_root,
            pool_name="main", max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests,
            kv_block_size_tokens=args.max_model_len, available_memory_by_rank=(2 * 1024**3,),
            supported_layouts_by_rank=(("BLHNC",),))
    pipeline = tuple(PipelineP2PConfig(stage, stage + 1, "send_recv", "nvlink", "point_to_point",
        "k3-source-pipeline", "analytical_roofline_v1") for stage in range(pp - 1))
    deployment = build_deployment_plan(DeploymentConfig("colocated", (
        WorkerPoolConfig("main", "colocated", "nvidia_h200", parallelism, pipeline_p2p=pipeline),)),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    requests = RequestStream(tuple(
        RequestSpec(index * 1e-6, f"r{index}", prompt, args.output_tokens)
        for index, prompt in enumerate(prompts)))
    runtime = RuntimeInputs(ExpertRoutingInput.seeded_assumption("k3-declared-serving-routing", 0),
        "mla", "uniform", model_execution_version="vllm_0_30_0",
        symbolic_forward_source="vllm_0_30_k3_mxfp4_sync", target_sampling_source="vllm_0_30_plain_greedy_bf16",
        symbolic_source_max_model_len=args.max_model_len,
        symbolic_source_max_batched_tokens=args.max_batched_tokens)
    if args.source_memory_only:
        backend.add_requests(tuple(requests))
        step = backend.plan_step(0.0)
        if step is None:
            raise RuntimeError("source-memory scenario did not schedule a request")
        step_runtime = replace(runtime, embedding_token_ids=step.embedding_token_ids,
            model_source_inputs=step.model_source_inputs)
        captured = []
        def observe(source, graph):
            captured.append((source, graph))
        build_symbolic_workload(step.schedule, model, deployment, launch_policy="analytical_launch",
            runtime_inputs=step_runtime, kv_cache_events=step.kv_cache_events,
            hardware_by_id={"nvidia_h200": hardware}, trace_observer=observe)
        source, graph = captured[0]
        report = symbolic_forward_storage_report(source, graph,
            source_profile=runtime.symbolic_forward_source)
        if normal:
            accumulator = SourceActivationMemoryAccumulator()
            accumulator.consume("main", step.schedule.step_id, report)
            report["capacity_validation"] = validate_source_activation_capacity(
                accumulator.manifest(total_steps=1), plans=(capacity,), policy_config=policies,
                pool_ranks={"main": tuple(range(parallelism.worker_count))},
                source_workspace_budget=source_workspace)
        report.update(layout=args.layout, capacity=None if capacity is None else asdict(capacity),
            source_workspace_budget=source_workspace, runtime_info=backend.runtime_info)
        backend.complete_step(step, 0.0)
        while backend.has_work():
            step = backend.plan_step(0.0)
            if step is None:
                raise RuntimeError("source-memory cleanup could not advance the Scheduler")
            backend.complete_step(step, 0.0)
        backend.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(args.output)
        return
    config = SimulationConfig(model, deployment, requests, "analytical_launch", args.prediction_method, runtime,
        execution_workers=1)
    started = time.perf_counter()
    result = Simulation(config, backend, {"nvidia_h200": hardware}).run()
    wall_s = time.perf_counter() - started
    emitted = {request.request_id: 0 for request in requests}
    for step in result.steps:
        for request_id, count in step.backend_step.generated_tokens.items():
            emitted[request_id] += count
    completed_request_count = sum(count == args.output_tokens for count in emitted.values())
    steps = [_step_report(step) for step in result.steps]
    metrics = asdict(result.metrics)
    if completed_request_count != len(requests):
        raise RuntimeError(f"incomplete K3 request output: {emitted}")
    if any(step["unpriced_calls"] != 0 for step in steps):
        raise RuntimeError("K3 complete-request report requires every emitted kernel call to be priced")
    if any(step["predicted_duration_s"] <= 0 for step in steps):
        raise RuntimeError("K3 complete-request report requires positive step durations")
    if any(request["ttft_s"] is None or request["ttft_s"] <= 0
            or request["tpot_s"] is None or request["tpot_s"] < 0
            or (args.output_tokens > 1 and request["tpot_s"] <= 0)
            for request in metrics["requests"].values()):
        raise RuntimeError(f"invalid K3 complete-request latency metrics: {metrics}")
    if (metrics["request_throughput_per_s"] <= 0
            or metrics["output_token_throughput_per_s"] <= 0):
        raise RuntimeError(f"invalid K3 complete-request throughput metrics: {metrics}")
    source_memory = result.source_activation_memory
    capacity_validation = (None if not normal else validate_source_activation_capacity(
        source_memory, plans=(capacity,), policy_config=policies,
        pool_ranks={"main": tuple(range(parallelism.worker_count))},
        source_workspace_budget=source_workspace))
    payload = {
        "schema_version": 1,
        "layout": args.layout,
        "prediction_method": args.prediction_method,
        "normal_capacity_plan": normal,
        "normal_capacity_accepted": bool(capacity_validation is not None
            and capacity_validation["normal_capacity_accepted"]),
        "capacity": None if capacity is None else asdict(capacity),
        "source_workspace_budget": source_workspace,
        "source_activation_memory": source_memory,
        "capacity_validation": capacity_validation,
        "runtime_info": backend.runtime_info,
        "requests": [asdict(request) for request in requests],
        "emitted_tokens": emitted,
        "completed_request_count": completed_request_count,
        "metrics": metrics,
        "simulation_wall_time_s": wall_s,
        "steps": steps,
        "execution_cache_stats": result.execution_cache_stats,
        "scope": "ordinary Simulation.run with real guarded CPU0.30 Scheduler, declared virtual source, length-only feedback and seeded routing; not numerical model generation, provider execution or measured accuracy",
        "timing_aggregation": "step duration is the execution critical path; per-variant prediction_time_sum_s is a non-overlap-adjusted diagnostic sum across calls and ranks, not a stage latency",
        "accuracy_validation": f"not measured on H200; analytical source/{args.prediction_method} projection only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
