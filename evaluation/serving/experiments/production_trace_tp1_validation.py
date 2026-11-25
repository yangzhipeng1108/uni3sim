"""TP1 colocated Production Trace point validation."""

from __future__ import annotations

import math
from typing import Any
from evaluation.serving.simulation.execution_artifact import execution_artifact_manifest, execution_step_components_manifest, prefix_alignment_manifest
from src.serving.backend.mtp_acceptance_replay import mtp_acceptance_step_from_backend_step, mtp_acceptance_steps_manifest
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import DeploymentConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import deployment_plan_manifest
from src.serving.kv_cache.kv_lifecycle_replay import load_kv_lifecycle_replay
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestStream
from src.serving.schedule.schedule_trace import ScheduleTrace, ScheduleTraceEntry
from src.serving.metrics.serving_metrics import compute_serving_metrics
from src.serving.execution import ServingExecutionContext
from src.serving.simulation import SimulationResult, SimulationStep
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs

from evaluation.serving.experiments.production_trace_contracts import (
    backend_step_from_output_manifest,
    canonical_production_trace_metrics_manifest,
    expected_performance_timing_status,
    explicit_expert_routing_input,
    interval_union_duration_from_output,
    json_round_trip,
    load_expected_hardware,
    replay_pinned_parallel_scheduler,
    validate_canonical_production_trace_step_duration,
    validate_production_trace_capacity,
    validate_production_trace_scheduler_binding,
)

from evaluation.serving.experiments.production_trace_colocated_kv_replay import (
    validate_colocated_kv_lifecycle_replay,
)


def validate_colocated_tp1_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    mtp = expected_simulation["mtp"]
    mtp_enabled = mtp["acceptance_policy"] != "disabled"
    replay_identity = expected_simulation["kv_lifecycle_replay"]
    if replay_identity is None:
        kv_lifecycle_replay = None
    else:
        try:
            kv_lifecycle_replay = load_kv_lifecycle_replay(
                replay_identity["path"])
        except (OSError, TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                "TP=PP=1 Host-KV Production Trace replay cannot be "
                "independently reloaded") from error
        if (kv_lifecycle_replay.source_sha256
                != replay_identity["source_sha256"]
                or kv_lifecycle_replay.artifact_id
                != replay_identity["artifact_id"]):
            raise ValueError(
                "TP=PP=1 Host-KV Production Trace replay identity changed")
    if kv_lifecycle_replay is not None:
        context = "TP=PP=1 Host-KV Production Trace"
    elif mtp_enabled:
        context = "TP=PP=1 GLM4-MoE MTP Production Trace"
    else:
        context = "TP=PP=1 Production Trace"
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(),
                kv_cache=(
                    None
                    if kv_lifecycle_replay is None
                    else kv_lifecycle_replay.cache_config
                ),
            ),),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} worker-pool topology disagrees with command manifest")

    scheduler = expected_simulation["scheduler"]
    runtime = result["scheduler_runtime"]["main"]
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(expected_model.num_hidden_layers)
    ]
    if runtime["scheduler_kv_cache_tensor_owners"] != expected_owners:
        raise ValueError(
            f"{context} Scheduler KV owners disagree with the pinned model")
    validate_production_trace_capacity(
        result,
        scheduler,
        ("main",),
        context=context,
    )
    expected_kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 1))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(f"{context} KV capacity disagrees with the model")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(f"{context} routing or transfer state is invalid")
    if kv_lifecycle_replay is not None:
        validate_colocated_kv_lifecycle_replay(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
            deployment,
            runtime,
            kv_lifecycle_replay,
            tensor_parallel_size=1,
            context=context,
        )
        return
    if (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(f"{context} unexpectedly enabled KV offload")
    if (result["mtp_scenario"] != mtp
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(
            f"{context} MTP scenario disagrees with command manifest")

    routing_config = expected_simulation["expert_routing"]
    if mtp_enabled:
        expert_routing = explicit_expert_routing_input(
            routing_config, context)
    else:
        if routing_config is not None:
            raise ValueError(
                f"{context} unexpectedly enabled expert routing")
        expert_routing = ExpertRoutingInput.not_applicable()
    runtime_inputs = RuntimeInputs(
        expert_routing,
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    expected_runtime_inputs = {
        "expert_routing": json_round_trip(
            expert_routing.manifest()),
        "attention_kv_layout": expected_simulation["attention_kv_layout"],
        "attention_kv_distribution": expected_simulation[
            "attention_kv_distribution"],
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }
    if result["execution_provenance"]["runtime_inputs"] != (
            expected_runtime_inputs):
        raise ValueError(
            f"{context} runtime inputs disagree with the command manifest")

    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"] != []
            or artifact_trace["transfers"] != []
            or artifact["transfers"] != []):
        raise ValueError(
            f"{context} execution artifact contains a route or transfer")
    steps = artifact["steps"]
    counts = artifact["counts"]
    if (not steps
            or result["num_steps"] != len(steps)
            or counts["schedule_steps"] != len(steps)
            or counts["simulation_steps"] != len(steps)
            or counts["simulation_transfers"] != 0
            or counts["trace_transfers"] != 0
            or counts["modeled_kv_offload_transfers"] != 0):
        raise ValueError(f"{context} execution artifact counts disagree")
    trace_entries = artifact_trace["entries"]
    entries_by_id = {
        entry["step_id"]: entry for entry in trace_entries
    }
    if (len(entries_by_id) != len(trace_entries)
            or len(entries_by_id) != len(steps)):
        raise ValueError(f"{context} schedule trace coverage disagrees")

    hardware = load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    total_call_count = 0
    worker_pool_intervals = []
    all_step_ids = set()
    all_call_ids = set()
    backend_steps_by_id = {}
    simulation_steps = []
    execution = ServingExecutionContext(
        expected_model,
        deployment,
        hardware_by_id,
        expected_simulation["launch_policy"],
        expected_simulation["prediction_method"],
        runtime_inputs,
        execution_mode=result["execution_mode"],
    )
    for step in steps:
        step_id = step["step_id"]
        if (step_id in all_step_ids
                or step_id not in entries_by_id
                or step["pool_name"] != "main"):
            raise ValueError(f"{context} execution step identity is invalid")
        all_step_ids.add(step_id)
        entry = entries_by_id[step_id]
        if (entry["pool_name"] != "main"
                or step["schedule_interval"] != {
                    "start_time_s": entry["start_time_s"],
                    "end_time_s": entry["end_time_s"],
                }):
            raise ValueError(f"{context} schedule interval identity is invalid")
        try:
            backend_step = backend_step_from_output_manifest(
                step["backend_step"])
            components = execution.execute_step(
                backend_step.schedule,
                kv_cache_events=backend_step.kv_cache_events,
            )
            workload = components.workload
            kernel_plan = components.kernel_plan
            predictions = components.predictions
            timeline = components.execution_timeline
        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"{context} step cannot be canonically rematerialized") \
                from error
        backend_steps_by_id[step_id] = backend_step
        expected_components = json_round_trip(
            execution_step_components_manifest(
                workload,
                kernel_plan,
                predictions,
                timeline,
                origin_time_s=step["schedule_interval"]["start_time_s"],
            ))
        actual_components = {
            field: step[field]
            for field in expected_components
        }
        if actual_components != expected_components:
            raise ValueError(
                f"{context} canonical Scheduler-to-KernelSpec "
                "rematerialization disagrees with the execution artifact")
        validate_canonical_production_trace_step_duration(
            step,
            timeline.duration_s,
            context=context,
        )
        simulation_steps.append(SimulationStep(
            backend_step,
            workload,
            kernel_plan,
            predictions,
            timeline,
        ))
        calls = step["kernel_plan"]["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        allowed_kernel_kinds = {"VECTOR", "GEMM", "FLASH_ATTN"}
        if mtp_enabled:
            allowed_kernel_kinds.add("FUSED_MOE")
        if (not calls
                or all_call_ids.intersection(call_ids)
                or {call["rank"] for call in calls} != {0}
                or any(
                    call["pool_name"] != "main"
                    or call["kernel_spec"]["hardware_id"] != hardware_id
                    or call["kernel_spec"]["kind"]
                    not in allowed_kernel_kinds
                    for call in calls
                )):
            raise ValueError(
                f"{context} rank-local compute coverage is invalid")
        if mtp_enabled:
            node_ids = tuple(
                node["node_id"] for node in step["workload_graph"]["nodes"])
            if any(
                    not any(f":mtp-pass-{pass_index}:" in node_id
                            for node_id in node_ids)
                    for pass_index in range(mtp["num_speculative_tokens"])):
                raise ValueError(
                    f"{context} MTP proposer workload coverage is invalid")
        all_call_ids.update(call_ids)
        intervals = step["execution_timeline"]["intervals"]
        intervals_by_id = {
            interval["call_id"]: interval for interval in intervals
        }
        if set(intervals_by_id) != set(call_ids):
            raise ValueError(f"{context} timeline coverage is invalid")
        worker_pool_intervals.extend(
            (
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            )
            for interval in intervals
        )
        total_call_count += len(calls)
    if (set(entries_by_id) != all_step_ids
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count):
        raise ValueError(f"{context} kernel or step counts disagree")

    canonical_schedule_trace = ScheduleTrace(
        ScheduleTraceEntry(
            backend_steps_by_id[entry["step_id"]],
            entry["start_time_s"],
            entry["end_time_s"],
        )
        for entry in trace_entries
    )
    validate_production_trace_scheduler_binding(
        expected_requests,
        canonical_schedule_trace,
        scheduler["kv_block_size_tokens"],
        result,
        max_requests=scheduler["max_requests"],
        context=context,
    )
    replay_pinned_parallel_scheduler(
        expected_requests,
        canonical_schedule_trace,
        scheduler,
        expected_model,
        expected_model_path,
        runtime,
        num_kv_blocks=runtime["num_kv_blocks"],
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        expected_mtp=mtp,
        context=context,
    )

    if mtp_enabled:
        expected_mtp_outcomes = {
            "schema_version": 1,
            "source_kind": "explicit_assumption",
            "source_id": mtp["acceptance_source_id"],
            "steps": mtp_acceptance_steps_manifest(
                mtp_acceptance_step_from_backend_step(
                    step_index,
                    backend_steps_by_id[entry["step_id"]],
                )
                for step_index, entry in enumerate(trace_entries)
            ),
        }
    else:
        expected_mtp_outcomes = None
    if result["mtp_acceptance_outcomes"] != expected_mtp_outcomes:
        raise ValueError(
            f"{context} MTP acceptance outcomes disagree with canonical "
            "Scheduler steps")

    canonical_metrics = compute_serving_metrics(
        expected_requests,
        canonical_schedule_trace,
    )
    canonical_result = SimulationResult(
        canonical_schedule_trace,
        canonical_metrics,
        tuple(simulation_steps),
    )
    expected_prefix_alignment = json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Scheduler")
    expected_artifact = json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                expected_performance_timing_status(expected_simulation)),
        ))
    if artifact != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    expected_metrics = canonical_production_trace_metrics_manifest(
        deployment,
        expected_requests,
        canonical_result,
        expected_slo_contract,
    )
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")

    system_resources = result["metrics"]["system_resources"]
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    worker_pool_busy_time_s = interval_union_duration_from_output(
        worker_pool_intervals,
        observation_window_s,
        context=f"{context} worker pool",
    )
    worker_pool_idle_time_s = observation_window_s - worker_pool_busy_time_s
    worker_pool_resources = system_resources["worker_pool_utilization"]
    if set(worker_pool_resources) != {"main"}:
        raise ValueError(
            f"{context} worker-pool utilization coverage is invalid")
    worker_pool = worker_pool_resources["main"]
    if (observation_window["basis"]
            != "simulation_origin_to_last_completion"
            or not math.isclose(
                observation_window_s,
                result["schedule_end_time_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or system_resources["communication_resource_utilization"] != {}
            or system_resources["link_utilization"] != {}
            or set(worker_pool) != {
                "kernel_interval_count", "busy_time_s", "idle_time_s",
                "utilization"}
            or worker_pool["kernel_interval_count"] != total_call_count
            or not math.isclose(
                worker_pool["busy_time_s"],
                worker_pool_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["idle_time_s"],
                worker_pool_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["utilization"],
                worker_pool_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or set(system_resources["scheduler_queue"]["by_pool"])
            != {"main"}):
        raise ValueError(
            f"{context} system resource metrics disagree with the pool")
