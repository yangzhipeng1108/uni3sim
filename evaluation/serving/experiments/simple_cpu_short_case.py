"""Run a controlled native vLLM 0.30 SimpleCPU store/hit/load request sequence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.deployment.deployment_config import (
    CollectiveConfig, DeploymentConfig, WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.kv_cache.kv_cache_config import KVCacheConfig, KVCacheTierConfig
from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.simulation import Simulation, SimulationConfig
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs


def run_short_case(
    *, sdk_python: Path, sdk_source: Path, tensor_parallel_size: int,
    gpu_blocks: int, cpu_capacity_bytes_per_rank: int,
    pcie_bandwidth_bytes_s: int | None = None,
    terminal_zero_output: bool = False,
    replacement_cycle: bool = False,
) -> dict[str, object]:
    if tensor_parallel_size not in (1, 2):
        raise ValueError("controlled SimpleCPU case requires TP1 or TP2")
    if (type(gpu_blocks) is not int or gpu_blocks < 8
            or type(cpu_capacity_bytes_per_rank) is not int
            or cpu_capacity_bytes_per_rank <= 0):
        raise ValueError("controlled SimpleCPU capacities are invalid")
    if (pcie_bandwidth_bytes_s is not None
            and (type(pcie_bandwidth_bytes_s) is not int
                 or pcie_bandwidth_bytes_s <= 0)):
        raise ValueError("PCIe bandwidth scenario must be positive bytes/s")
    if terminal_zero_output and replacement_cycle:
        raise ValueError("terminal zero-output and replacement cycle are separate cases")

    model_path = "configs/model/qwen3-tp2-example"
    model = load_model_config(model_path)
    block_tokens = 16
    block_bytes = (block_tokens
        * model.kv_bytes_per_token_per_parallel_rank(tensor_parallel_size, 1))
    if cpu_capacity_bytes_per_rank < block_bytes:
        raise ValueError("CPU capacity must hold at least one rank-local KV page")
    if terminal_zero_output:
        request_specs = (RequestSpec(0.0, "first", 96, 0,
            prompt_token_ids=(7,) * 96, output_token_ids=()),)
    else:
        request_specs = (
            RequestSpec(0.0, "first", 96, 4,
                prompt_token_ids=(7,) * 96, output_token_ids=(8, 9, 10, 11)),
            RequestSpec(0.01, "different", 96, 4,
                prompt_token_ids=(6,) * 96, output_token_ids=(12, 13, 14, 15)),
            RequestSpec(0.02, "same", 96, 4,
                prompt_token_ids=(7,) * 96, output_token_ids=(16, 17, 18, 19)),
        )
        if replacement_cycle:
            request_specs += (RequestSpec(0.03, "different_again", 96, 4,
                prompt_token_ids=(6,) * 96,
                output_token_ids=(20, 21, 22, 23)),)
    requests = RequestStream(request_specs)
    sdk_started = time.perf_counter()
    backend = VLLMBackend.from_vllm030_byte_budget(
        VLLMSchedulerConfig(
            model=model_path, tensor_parallel_size=tensor_parallel_size,
            max_model_len=128, enable_chunked_prefill=False,
            enable_prefix_caching=True, prefix_cache_mode="token_ids",
            scheduler_version="0.30.0",
            simple_cpu_offload_bytes_per_rank=cpu_capacity_bytes_per_rank,
        ),
        python_executable=sdk_python, source_root=sdk_source,
        pool_name="main", max_batched_tokens=128, max_requests=1,
        kv_block_size_tokens=block_tokens,
        available_memory_by_rank=(gpu_blocks * block_bytes,)
            * tensor_parallel_size,
        supported_layouts_by_rank=(("BLHNC", "BLNHC"),)
            * tensor_parallel_size,
    )
    try:
        actual_gpu_blocks = backend.config.num_kv_blocks
        cache = KVCacheConfig((
            KVCacheTierConfig("hbm", actual_gpu_blocks * block_bytes,
                              block_tokens),
            KVCacheTierConfig("host",
                cpu_capacity_bytes_per_rank // block_bytes * block_bytes,
                block_tokens, "pcie"),
        ), block_bytes // block_tokens, "lru")
        collectives = (tuple(CollectiveConfig(
            operation, "ring", "nvlink", "ring", "source-fabric",
            "analytical_roofline_v1")
            for operation in ("all_reduce", "all_gather"))
            if tensor_parallel_size == 2 else ())
        pool = WorkerPoolConfig(
            "main", "colocated", "nvidia_a800",
            ParallelConfig(tensor_parallel_size=tensor_parallel_size),
            ranks=tuple(range(tensor_parallel_size)), kv_cache=cache,
            collectives=collectives,
        )
        deployment = build_deployment_plan(
            DeploymentConfig("colocated", (pool,)),
            num_layers=model.num_hidden_layers)
        runtime = RuntimeInputs(
            ExpertRoutingInput.not_applicable(), "paged_varlen", "uniform",
            model_execution_version="vllm_0_30_0",
            target_sampling_source="vllm_0_30_plain_greedy_bf16",
            symbolic_forward_source="cuda_eager_inplace",
        )
        hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
        if pcie_bandwidth_bytes_s is not None:
            hardware.interconnect["pcie_bw_bytes_s"] = pcie_bandwidth_bytes_s
        started = time.perf_counter()
        result = Simulation(
            SimulationConfig(model, deployment, requests,
                "analytical_launch", "roofline", runtime),
            backend, {"nvidia_a800": hardware},
            kv_cache_tier_manager=KVCacheTierManager(cache),
        ).run()
        simulation_wall_s = time.perf_counter() - started
        expected_requests = {request.request_id for request in requests}
        if set(result.metrics.requests) != expected_requests:
            raise RuntimeError("controlled SimpleCPU case did not finish all requests")
        if result.kv_offload is None:
            raise RuntimeError("controlled SimpleCPU case lost KV offload state")
        transfers = result.kv_offload.transfers
        physical_bytes = {rank: {"h2d": 0, "d2h": 0}
                          for rank in range(tensor_parallel_size)}
        transfer_rows = []
        for record in transfers:
            action = record.event.action.value
            direction = "h2d" if action == "load" else "d2h"
            for physical in record.completion.physical_transfers:
                physical_bytes[physical.rank][direction] += physical.bytes
            transfer_rows.append({
                "action": action,
                "request_id": record.event.request_id,
                "source_request_id": record.event.source_request_id,
                "token_start": record.event.token_start,
                "token_end": record.event.token_end,
                "bytes_per_rank": record.event.bytes,
                "physical_transfers": tuple(asdict(physical)
                    for physical in record.completion.physical_transfers),
            })
        load_dependencies = []
        for step in result.steps:
            step_loads = tuple(record for record in transfers
                if record.step_id == step.backend_step.schedule.step_id
                and record.event.action.value == "load")
            if not step_loads:
                continue
            load_call_ids = {
                physical.call_id for record in step_loads
                for physical in record.completion.physical_transfers
            }
            first_compute = min((interval.start_time_s
                for interval in step.execution_timeline.intervals
                if interval.call_id not in load_call_ids
                and interval.resource != "pcie"), default=None)
            if first_compute is None:
                raise RuntimeError("SimpleCPU load step has no model computation")
            latest_load = max(physical.end_time_s
                for record in step_loads
                for physical in record.completion.physical_transfers)
            first_load = min(physical.start_time_s
                for record in step_loads
                for physical in record.completion.physical_transfers)
            if first_compute + 1e-15 < latest_load:
                raise RuntimeError("SimpleCPU model computation started before Host load")
            load_dependencies.append({
                "step_id": step.backend_step.schedule.step_id,
                "last_h2d_completion_s": latest_load,
                "first_compute_start_s": first_compute,
                "h2d_gate_span_s": latest_load - first_load,
            })
        first_schedules = {}
        for step in result.steps:
            for scheduled in step.backend_step.schedule.requests:
                if scheduled.request_id not in first_schedules:
                    first_schedules[scheduled.request_id] = {
                        "content_prefix_hit_tokens": (
                            scheduled.content_prefix_hit_tokens),
                        "computed_tokens": scheduled.computed_tokens,
                        "scheduled_tokens": scheduled.scheduled_tokens,
                    }
        return {
            "source_kind": "synthetic_explicit_token_ids",
            "connector": "vllm_0_30_0_SimpleCPUOffloadConnector_eager_cpu",
            "model_path": model_path,
            "model_config_sha256": model.config_sha256,
            "requests": tuple(asdict(request) for request in requests),
            "terminal_zero_output": terminal_zero_output,
            "replacement_cycle": replacement_cycle,
            "tensor_parallel_size": tensor_parallel_size,
            "requested_gpu_blocks": gpu_blocks,
            "effective_gpu_blocks": actual_gpu_blocks,
            "block_tokens": block_tokens,
            "block_bytes_per_rank": block_bytes,
            "cpu_capacity_bytes_per_rank": cpu_capacity_bytes_per_rank,
            "effective_cpu_capacity_bytes_per_rank": (
                cache.tiers[1].capacity_bytes),
            "pcie_bandwidth_bytes_s": hardware.interconnect["pcie_bw_bytes_s"],
            "transfer_cost_status": "roofline_byte_lower_bound_no_startup_latency",
            "load_only_step_representation": (
                "native_empty_step_completed_by_control_plane; H2D is a "
                "predecessor of the resumed model step"),
            "first_schedules": first_schedules,
            "completed_requests": len(result.metrics.requests),
            "transfer_only_step_count": sum(
                step.backend_step.schedule.is_transfer_only for step in result.steps),
            "metrics": asdict(result.metrics),
            "physical_bytes_by_rank": physical_bytes,
            "transfers": transfer_rows,
            "load_dependencies": load_dependencies,
            "final_tier_state": result.kv_offload.final_tier_state,
            "simulation_wall_s": simulation_wall_s,
            "total_wall_s_including_sdk_startup": time.perf_counter() - sdk_started,
        }
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-python", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2),
                        required=True)
    parser.add_argument("--gpu-blocks", type=int, required=True)
    parser.add_argument("--cpu-capacity-bytes-per-rank", type=int,
                        required=True)
    parser.add_argument("--pcie-bandwidth-bytes-s", type=int)
    parser.add_argument("--terminal-zero-output", action="store_true")
    parser.add_argument("--replacement-cycle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_short_case(
        sdk_python=args.sdk_python, sdk_source=args.sdk_source,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_blocks=args.gpu_blocks,
        cpu_capacity_bytes_per_rank=args.cpu_capacity_bytes_per_rank,
        pcie_bandwidth_bytes_s=args.pcie_bandwidth_bytes_s,
        terminal_zero_output=args.terminal_zero_output,
        replacement_cycle=args.replacement_cycle,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
