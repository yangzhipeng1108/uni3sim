"""Small reproducible complete-model and four-pool interactive report examples."""
from dataclasses import asdict
from pathlib import Path
import json
import time

from evaluation.reporting.serving_interactive import write_interactive_report
from evaluation.serving.simulation.execution_artifact import execution_artifact_manifest
from evaluation.serving.simulation.interactive_recording import InteractiveRecording
from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.deployment.deployment_config import (
    CollectiveConfig, DeploymentConfig, RouteConfig, WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.host_timing import load_host_launch_profile
from src.serving.kv_cache.kv_cache_config import KVCacheConfig, KVCacheTierConfig
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.simulation import MultiSimulation, Simulation, SimulationConfig
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs


def run_example(mode, output, sdk_python, source_root, *, record=True):
    """Run real CPU Scheduler with virtual forward; never needs GPU or weights."""
    if mode not in ("single", "tp2", "pd", "four"):
        raise ValueError("unknown interactive example mode")
    four = mode == "four"
    pd = mode in ("pd", "four")
    tp = 2 if mode == "tp2" else 1
    hardware_id = "nvidia_a800" if four else "nvidia_h200"
    model_path = "configs/model/qwen3-example" if four else "configs/model/qwen3-4b.json"
    model = load_model_config(model_path)
    hardware = {hardware_id: HWSpec(f"configs/hardware/{hardware_id}.yaml")}
    collectives = tuple(CollectiveConfig(operation, "ring", "nvlink", "ring", "example-fabric",
        "analytical_roofline_v1") for operation in ("all_reduce", "all_gather")) if tp == 2 else ()
    if four:
        cache = KVCacheConfig((KVCacheTierConfig("hbm", 32768, 16),), 256)
        pools = tuple(WorkerPoolConfig(f"{role}-{i}", role, hardware_id,
            ranks=(j * 2 + i,), kv_cache=cache)
            for j, role in enumerate(("prefill", "decode")) for i in range(2))
    else:
        names = ("prefill", "decode") if pd else ("main",)
        pools = tuple(WorkerPoolConfig(name, name if pd else "colocated", hardware_id,
            ParallelConfig(tensor_parallel_size=tp), collectives=collectives) for name in names)
    routes = tuple(RouteConfig(p.name, d.name, "kv", "send_recv", "nvlink", "point_to_point",
        resource_id="example-pd-link", backend="analytical_roofline_v1")
        for p in pools if p.role == "prefill" for d in pools if d.role == "decode")
    deployment = build_deployment_plan(DeploymentConfig("pd_disagg" if pd else "colocated", pools, routes),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    version = "0.11.0" if four else "0.30.0"
    runtime = RuntimeInputs(ExpertRoutingInput.not_applicable(), "paged_varlen", "uniform",
        model_execution_version="vllm_0_11_0" if four else "vllm_0_30_0",
        target_sampling_source="vllm_0_11_plain_greedy_bf16" if four else "vllm_0_30_plain_greedy_bf16",
        symbolic_forward_source="cuda_eager_inplace")
    requests = RequestStream(tuple(RequestSpec(float(batch * 10), f"{batch}-{i}", 32, 3,
        prompt_token_ids=(1 if batch < 2 else 2,) * 32, output_token_ids=(3, 4, 5))
        for batch in range(3) for i in range(2))) if four else RequestStream(
            (RequestSpec(0., "a", 8, 3), RequestSpec(1e-6, "b", 4, 2)) if pd else (RequestSpec(0., "a", 8, 3),))
    backends = {}
    try:
        for pool in pools:
            settings = VLLMSchedulerConfig(model=model_path, max_model_len=128 if four else 512,
                tensor_parallel_size=tp, enable_chunked_prefill=four, scheduler_version=version,
                enable_prefix_caching=four, prefix_cache_mode="token_ids" if four else "trace")
            if four:
                backend = VLLMBackend.from_model(BackendConfig(pool.name, pool.role, 16, 2, 8,
                    kv_bytes_per_token=256, scheduler_pipeline_depth=1), settings)
            else:
                backend = VLLMBackend.from_vllm030_byte_budget(settings,
                    python_executable=Path(sdk_python), source_root=Path(source_root),
                    supported_layouts_by_rank=(("BLHNC", "BLNHC"),) * tp,
                    pool_name=pool.name, role=pool.role, max_batched_tokens=2048,
                    max_requests=2, kv_block_size_tokens=16,
                    available_memory_by_rank=(512 * 1024 * 1024,) * tp)
            backends[pool.name] = backend
        config = SimulationConfig(model, deployment, requests, "analytical_launch", "roofline", runtime,
            timing_mode="device_only" if four else "device_plus_launch",
            host_launch_profiles=None if four else {hardware_id: load_host_launch_profile(
                "configs/host_timing/host_launch_profiles/nvidia_h200_vllm030.json")})
        started = time.perf_counter()
        simulator = MultiSimulation(config, backends, hardware) if pd else Simulation(config, backends["main"], hardware)
        result = simulator.run()
        wall = time.perf_counter() - started
        payload = {"execution_artifact": execution_artifact_manifest(result, performance_timing_status="projected_by_midlayer"),
            "metrics": asdict(result.metrics), "simulation_wall_time_s": wall,
            "prediction_method": config.prediction_method, "launch_policy": config.launch_policy,
            "schedule_end_time_s": result.schedule_trace.end_time_s,
            "source_activation_memory": result.source_activation_memory,
            "kv_capacity": {"scope": "explicit test budget; not normal capacity acceptance",
                "backend_budgets": [asdict(backend.config) for backend in backends.values()]}}
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        state = {"metrics": asdict(result.metrics), "completed_requests": len(result.metrics.requests),
            "emitted_tokens": {r.request_id: sum(step.backend_step.generated_tokens[r.request_id]
                if r.request_id in step.backend_step.generated_tokens else 0 for step in result.steps) for r in requests},
            "pools": sorted({s.backend_step.schedule.pool_name for s in result.steps}),
            "source_activation_memory": result.source_activation_memory,
            "transfers": [asdict(t) for t in result.schedule_trace.transfers],
            "routing": [asdict(r) for r in result.schedule_trace.route_decisions],
            "simulation_wall_time_s": wall, "capacity_scope": "explicit test KV budget; incomplete activation; not normal capacity acceptance"}
        (output / "state.json").write_text(json.dumps(state, separators=(",", ":")))
        if record:
            recorder = InteractiveRecording(hardware)
            entries = {entry.backend_step.schedule.step_id: entry for entry in result.schedule_trace.entries}
            for step in result.steps:
                recorder.record(step, entries[step.backend_step.schedule.step_id])
            report = write_interactive_report(recorder.visualization_payload(payload, result), output / "report.html")
            (output / "generation.json").write_text(json.dumps({**report,
                "resource_recording_s": recorder.recording_wall_s, "simulation_wall_time_s": wall}, indent=2))
            return report
        return {"simulation_wall_time_s": wall, "record": False}
    finally:
        for backend in backends.values():
            backend.close()
