"""EP2, PP2, and TP2-PP2 colocated Production Trace point
validation."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any
from evaluation.serving.metrics.latency_distribution import latency_distribution_manifest
from evaluation.serving.metrics.slo_goodput import slo_goodput_manifest
from evaluation.serving.metrics.system_resource_analysis import critical_path_manifest, system_resource_manifest
from evaluation.serving.simulation.execution_artifact import execution_artifact_manifest, execution_step_components_manifest, prefix_alignment_manifest
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, PipelineP2PConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import deployment_plan_manifest
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


def validate_colocated_ep2_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    context = "TP=EP=2 Production Trace"
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    resource_id = "nvlink:main:tensor-fabric"
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            resource_id,
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                ),
                collectives=collectives,
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
        expected_model.kv_bytes_per_token_per_parallel_rank(2, 1))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(
            f"{context} KV capacity disagrees with model sharding")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(f"{context} routing or transfer state is invalid")
    if (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(f"{context} unexpectedly enabled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} unexpectedly enabled MTP")

    routing_config = expected_simulation["expert_routing"]
    expert_routing = explicit_expert_routing_input(routing_config, context)
    runtime_inputs = RuntimeInputs(
        expert_routing,
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    expected_runtime_inputs = {
        "expert_routing": json_round_trip(expert_routing.manifest()),
        "attention_kv_layout": expected_simulation["attention_kv_layout"],
        "attention_kv_distribution": expected_simulation[
            "attention_kv_distribution"],
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }
    if result["execution_provenance"]["runtime_inputs"] != (
            expected_runtime_inputs):
        raise ValueError(
            f"{context} seeded expert-routing identity disagrees with the "
            "command manifest")

    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"] != []
            or artifact_trace["transfers"] != []
            or artifact["transfers"] != []):
        raise ValueError(
            f"{context} execution artifact contains a route or transfer")
    artifact_body = {
        key: artifact[key]
        for key in artifact
        if key != "artifact_sha256"
    }
    expected_artifact_sha256 = hashlib.sha256(json.dumps(
        artifact_body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if artifact["artifact_sha256"] != expected_artifact_sha256:
        raise ValueError(f"{context} execution artifact SHA256 is invalid")
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
    fused_moe_call_count = 0
    communication_call_count = 0
    communication_bytes = 0
    communication_operation_counts: dict[str, int] = {}
    communication_topology_counts: dict[str, int] = {}
    communication_bytes_by_semantics: dict[str, int] = {}
    communication_intervals = []
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
        simulation_steps.append(SimulationStep(
            backend_step,
            workload,
            kernel_plan,
            predictions,
            timeline,
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

        calls = step["kernel_plan"]["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        if all_call_ids.intersection(call_ids):
            raise ValueError(
                f"{context} contains duplicate cross-step call IDs")
        all_call_ids.update(call_ids)
        fused_nodes = tuple(
            node for node in step["workload_graph"]["nodes"]
            if node["kind"] == "fused_moe"
        )
        invocations: dict[str, list[dict[str, Any]]] = {}
        for node in fused_nodes:
            attributes = node["attributes"]
            invocation_id = attributes["moe_invocation_id"]
            if invocation_id not in invocations:
                invocations[invocation_id] = []
            invocations[invocation_id].append(node)
            rank = node["rank"]
            global_counts = attributes["tokens_per_expert"]
            local_counts = attributes["local_tokens_per_expert"]
            routing_provenance = attributes["routing_provenance"]
            if (rank not in (0, 1)
                    or attributes["expert_parallel_size"] != 2
                    or attributes["expert_rank"] != rank
                    or attributes["expert_placement_strategy"] != "linear"
                    or attributes["owned_experts"] != [rank]
                    or len(global_counts) != 2
                    or len(local_counts) != 2
                    or sum(global_counts)
                    != node["shape"]["tokens"] * node["shape"]["topk"]
                    or local_counts != [
                        global_counts[0] if rank == 0 else 0,
                        global_counts[1] if rank == 1 else 0,
                    ]
                    or routing_provenance["source_kind"]
                    != "explicit_assumption"
                    or routing_provenance["source_id"]
                    != expert_routing.source_id
                    or routing_provenance["algorithm"]
                    != expert_routing.algorithm
                    or routing_provenance["distribution"]
                    != expert_routing.distribution
                    or routing_provenance["hot_expert_count"]
                    != expert_routing.hot_expert_count
                    or routing_provenance["hot_expert_weight"]
                    != expert_routing.hot_expert_weight
                    or routing_provenance["root_seed"]
                    != expert_routing.root_seed
                    or routing_provenance["policy_descriptor_sha256"]
                    != expert_routing.policy_descriptor_sha256
                    or routing_provenance["mechanism_source"] != "Modeled"
                    or routing_provenance["validation_status"] != "Projected"):
                raise ValueError(
                    f"{context} FUSED_MOE routing or ownership is invalid")
        if (len(invocations) != expected_model.num_hidden_layers
                or any(
                    {node["rank"] for node in nodes} != {0, 1}
                    or len(nodes) != 2
                    for nodes in invocations.values()
                )):
            raise ValueError(
                f"{context} FUSED_MOE invocation coverage is invalid")

        intervals_by_id = {
            interval["call_id"]: interval
            for interval in step["execution_timeline"]["intervals"]
        }
        worker_pool_intervals.extend(
            (
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            )
            for interval in intervals_by_id.values()
        )
        for call in calls:
            kind = call["kernel_spec"]["kind"]
            if kind == "FUSED_MOE":
                fused_moe_call_count += 1
            elif kind == "COMMUNICATION":
                transfer = call["kernel_spec"]["transfer"]
                if (transfer["operation"] == "all_to_all"
                        or transfer["resource_id"] != resource_id
                        or transfer["backend"] != "analytical_roofline_v1"):
                    raise ValueError(
                        f"{context} emitted an unsupported collective")
                operation = transfer["operation"]
                topology = transfer["topology"]
                semantics = transfer["bytes_semantics"]
                if operation not in communication_operation_counts:
                    communication_operation_counts[operation] = 0
                if topology not in communication_topology_counts:
                    communication_topology_counts[topology] = 0
                if semantics not in communication_bytes_by_semantics:
                    communication_bytes_by_semantics[semantics] = 0
                communication_operation_counts[operation] += 1
                communication_topology_counts[topology] += 1
                communication_bytes_by_semantics[semantics] += transfer["bytes"]
                communication_bytes += transfer["bytes"]
                communication_call_count += 1
                interval = intervals_by_id[call["call_id"]]
                communication_intervals.append((
                    interval["absolute_start_time_s"],
                    interval["absolute_end_time_s"],
                ))
            elif kind in {"VECTOR", "GEMM", "FLASH_ATTN"}:
                pass
            else:
                raise ValueError(
                    f"{context} emitted unsupported kernel kind {kind!r}")
        if not fused_nodes:
            raise ValueError(f"{context} step emitted no FUSED_MOE workload")
        total_call_count += len(calls)
    if (set(entries_by_id) != all_step_ids
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count
            or fused_moe_call_count <= 0
            or communication_call_count <= 0
            or communication_bytes <= 0
            or set(communication_operation_counts)
            != {"all_gather", "all_reduce"}):
        raise ValueError(
            f"{context} kernel, MoE, or collective counts disagree")

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
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        expected_mtp=expected_simulation["mtp"],
        context=context,
    )

    canonical_metrics = compute_serving_metrics(
        expected_requests,
        canonical_schedule_trace,
    )
    canonical_result = SimulationResult(
        canonical_schedule_trace,
        canonical_metrics,
        tuple(simulation_steps),
    )
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
    expected_metrics = json_round_trip({
        "request_throughput_per_s": (
            canonical_metrics.request_throughput_per_s),
        "output_token_throughput_per_s": (
            canonical_metrics.output_token_throughput_per_s),
        "token_throughput_per_s": canonical_metrics.token_throughput_per_s,
        "latency_distributions": latency_distribution_manifest(
            canonical_metrics.requests),
        "slo_goodput": slo_goodput_manifest(
            expected_requests,
            canonical_metrics,
            canonical_schedule_trace.end_time_s,
            expected_slo_contract,
        ),
        "system_resources": system_resource_manifest(
            deployment,
            expected_requests,
            canonical_result,
        ),
        "critical_path": critical_path_manifest(
            deployment,
            expected_requests,
            canonical_result,
        ),
        "requests": {
            request_id: {
                "ttft_s": metric.ttft_s,
                "tpot_s": metric.tpot_s,
                "latency_s": metric.latency_s,
            }
            for request_id, metric in canonical_metrics.requests.items()
        },
    })
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")

    system_resources = result["metrics"]["system_resources"]
    communication_resources = system_resources[
        "communication_resource_utilization"]
    if set(communication_resources) != {resource_id}:
        raise ValueError(
            f"{context} communication resources disagree with the pool")
    communication = communication_resources[resource_id]
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    communication_busy_time_s = interval_union_duration_from_output(
        communication_intervals,
        observation_window_s,
        context=f"{context} communication fabric",
    )
    communication_idle_time_s = (
        observation_window_s - communication_busy_time_s)
    if (communication["backend"] != "analytical_roofline_v1"
            or communication["backend_config_sha256"] is not None
            or communication["hardware_id"] != hardware_id
            or communication["link"] != "nvlink"
            or communication["timing_status"]
            != "projected_analytical_roofline"
            or communication["call_count"] != communication_call_count
            or communication["operation_counts"]
            != communication_operation_counts
            or communication["topology_counts"]
            != communication_topology_counts
            or communication["bytes_by_semantics"]
            != communication_bytes_by_semantics
            or not math.isclose(
                communication["busy_time_s"],
                communication_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["idle_time_s"],
                communication_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["utilization"],
                communication_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )):
        raise ValueError(
            f"{context} communication metrics disagree with collective calls")
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


def validate_colocated_pp2_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
) -> None:
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    resource_id = "nvlink:main:pipeline-fabric"
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(pipeline_parallel_size=2),
                pipeline_p2p=(PipelineP2PConfig(
                    0,
                    1,
                    "send_recv",
                    "nvlink",
                    "point_to_point",
                    resource_id,
                    "analytical_roofline_v1",
                ),),
            ),),
        ),
        num_layers=expected_model.num_hidden_layers,
    )
    expected_plan = json_round_trip(
        deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            "PP=2 Production Trace worker-pool topology disagrees with "
            "command manifest")

    scheduler = expected_simulation["scheduler"]
    runtime = result["scheduler_runtime"]["main"]
    local_layer_count = expected_model.num_hidden_layers // 2
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(local_layer_count)
    ]
    if runtime["scheduler_kv_cache_tensor_owners"] != expected_owners:
        raise ValueError(
            "PP=2 Production Trace Scheduler KV owners disagree with the "
            "pinned Qwen3 stage")
    validate_production_trace_capacity(
        result,
        scheduler,
        ("main",),
        context="PP=2 Production Trace",
    )
    expected_kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 2))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(
            "PP=2 Production Trace KV capacity disagrees with model "
            "stage sharding")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(
            "PP=2 Production Trace routing or transfer state is invalid")
    if (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(
            "PP=2 Production Trace unexpectedly enabled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(
            "PP=2 Production Trace unexpectedly enabled MTP")

    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"] != []
            or artifact_trace["transfers"] != []
            or artifact["transfers"] != []):
        raise ValueError(
            "PP=2 Production Trace execution artifact contains a route or "
            "transfer")
    steps = artifact["steps"]
    counts = artifact["counts"]
    if (not steps
            or result["num_steps"] != len(steps)
            or counts["schedule_steps"] != len(steps)
            or counts["simulation_steps"] != len(steps)
            or counts["simulation_transfers"] != 0
            or counts["trace_transfers"] != 0
            or counts["modeled_kv_offload_transfers"] != 0):
        raise ValueError(
            "PP=2 Production Trace execution artifact counts disagree")
    trace_entries = artifact_trace["entries"]
    entries_by_id = {
        entry["step_id"]: entry for entry in trace_entries
    }
    if (len(entries_by_id) != len(trace_entries)
            or len(entries_by_id) != len(steps)):
        raise ValueError(
            "PP=2 Production Trace schedule trace coverage disagrees")

    hardware = load_expected_hardware(
        expected_simulation, hardware_id, "PP=2 Production Trace")
    hardware_by_id = {hardware_id: hardware}
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    total_call_count = 0
    communication_call_count = 0
    communication_bytes = 0
    communication_intervals = []
    worker_pool_intervals = []
    all_step_ids = set()
    all_call_ids = set()
    backend_steps_by_id = {}
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
            raise ValueError(
                "PP=2 Production Trace execution step identity is invalid")
        all_step_ids.add(step_id)
        entry = entries_by_id[step_id]
        if (entry["pool_name"] != "main"
                or step["schedule_interval"] != {
                    "start_time_s": entry["start_time_s"],
                    "end_time_s": entry["end_time_s"],
                }):
            raise ValueError(
                "PP=2 Production Trace schedule interval identity is invalid")
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
                "PP=2 Production Trace step cannot be canonically "
                "rematerialized") from error
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
                "PP=2 Production Trace canonical Scheduler-to-KernelSpec "
                "rematerialization disagrees with the execution artifact")
        validate_canonical_production_trace_step_duration(
            step,
            timeline.duration_s,
            context="PP=2 Production Trace",
        )
        calls = step["kernel_plan"]["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        if all_call_ids.intersection(call_ids):
            raise ValueError(
                "PP=2 Production Trace contains duplicate cross-step call IDs")
        all_call_ids.update(call_ids)
        p2p_calls = tuple(
            call for call in calls
            if call["kernel_spec"]["kind"] == "COMMUNICATION"
        )
        if (len(p2p_calls) != 2
                or tuple(
                    call["kernel_spec"]["transfer"]["operation"]
                    for call in p2p_calls
                ) != ("send_recv", "send_recv")):
            raise ValueError(
                "PP=2 Production Trace step must contain the two pipeline "
                "payload transfers")
        total_call_count += len(calls)
        communication_call_count += len(p2p_calls)
        communication_bytes += sum(
            call["kernel_spec"]["transfer"]["bytes"]
            for call in p2p_calls
        )
        intervals_by_id = {
            interval["call_id"]: interval
            for interval in step["execution_timeline"]["intervals"]
        }
        worker_pool_intervals.extend(
            (
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            )
            for interval in intervals_by_id.values()
        )
        communication_intervals.extend(
            (
                intervals_by_id[call["call_id"]][
                    "absolute_start_time_s"],
                intervals_by_id[call["call_id"]][
                    "absolute_end_time_s"],
            )
            for call in p2p_calls
        )
    if (set(entries_by_id) != all_step_ids
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count):
        raise ValueError(
            "PP=2 Production Trace kernel or schedule counts disagree")
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
        context="PP=2 Production Trace",
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
        pipeline_parallel_size=2,
        enable_expert_parallel=False,
        expected_mtp=expected_simulation["mtp"],
        context="PP=2 Production Trace",
    )

    system_resources = result["metrics"]["system_resources"]
    communication_resources = system_resources[
        "communication_resource_utilization"]
    if set(communication_resources) != {resource_id}:
        raise ValueError(
            "PP=2 Production Trace communication resources disagree with "
            "the pipeline plan")
    communication = communication_resources[resource_id]
    if (communication["backend"] != "analytical_roofline_v1"
            or communication["backend_config_sha256"] is not None
            or communication["hardware_id"] != hardware_id
            or communication["link"] != "nvlink"
            or communication["timing_status"]
            != "projected_analytical_roofline"
            or communication["call_count"] != communication_call_count
            or communication["operation_counts"] != {
                "send_recv": communication_call_count}
            or communication["topology_counts"] != {
                "point_to_point": communication_call_count}
            or communication["bytes_by_semantics"] != {
                "payload": communication_bytes}):
        raise ValueError(
            "PP=2 Production Trace communication metrics disagree with "
            "pipeline P2P calls")
    ordered_intervals = sorted(communication_intervals)
    communication_busy_time_s = 0.0
    previous_end_time_s = None
    for start_time_s, end_time_s in ordered_intervals:
        if (end_time_s <= start_time_s
                or (previous_end_time_s is not None
                    and start_time_s < previous_end_time_s
                    and not math.isclose(
                        start_time_s,
                        previous_end_time_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ))):
            raise ValueError(
                "PP=2 Production Trace communication intervals overlap or "
                "are non-positive")
        communication_busy_time_s += end_time_s - start_time_s
        previous_end_time_s = end_time_s
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    expected_idle_time_s = observation_window_s - communication_busy_time_s
    if (observation_window["basis"]
            != "simulation_origin_to_last_completion"
            or not math.isclose(
                observation_window_s,
                result["schedule_end_time_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or expected_idle_time_s < 0.0
            or not math.isclose(
                communication["busy_time_s"],
                communication_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["idle_time_s"],
                expected_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["utilization"],
                communication_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )):
        raise ValueError(
            "PP=2 Production Trace communication utilization disagrees "
            "with execution intervals")
    worker_pool_busy_time_s = interval_union_duration_from_output(
        worker_pool_intervals,
        observation_window_s,
        context="PP=2 Production Trace worker pool",
    )
    worker_pool_idle_time_s = (
        observation_window_s - worker_pool_busy_time_s)
    worker_pool_resources = system_resources["worker_pool_utilization"]
    if set(worker_pool_resources) != {"main"}:
        raise ValueError(
            "PP=2 Production Trace worker-pool utilization coverage "
            "disagrees with the deployment")
    worker_pool = worker_pool_resources["main"]
    if (system_resources["link_utilization"] != {}
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
            "PP=2 Production Trace system resource metrics disagree with "
            "the worker pool")


def validate_colocated_tp2_pp2_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    context = "TP=2, PP=2 Production Trace"
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    tensor_resource_id = "nvlink:main:tensor-fabric"
    pipeline_resource_id = "nvlink:main:pipeline-fabric"
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            tensor_resource_id,
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(
                    tensor_parallel_size=2,
                    pipeline_parallel_size=2,
                ),
                collectives=collectives,
                pipeline_p2p=(PipelineP2PConfig(
                    0,
                    1,
                    "send_recv",
                    "nvlink",
                    "point_to_point",
                    pipeline_resource_id,
                    "analytical_roofline_v1",
                ),),
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
    local_layer_count = expected_model.num_hidden_layers // 2
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(local_layer_count)
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
        expected_model.kv_bytes_per_token_per_parallel_rank(2, 2))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(
            f"{context} KV capacity disagrees with TP/PP model sharding")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(f"{context} routing or transfer state is invalid")
    if (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(f"{context} unexpectedly enabled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} unexpectedly enabled MTP")

    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    expected_runtime_inputs = {
        "expert_routing": json_round_trip(
            runtime_inputs.expert_routing.manifest()),
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
        raise ValueError(f"{context} execution artifact contains a transfer")
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
    resource_stats = {
        tensor_resource_id: {
            "call_count": 0,
            "operation_counts": {"all_gather": 0, "all_reduce": 0},
            "topology_counts": {"ring": 0},
            "bytes_by_semantics": {"per_rank_input": 0},
            "intervals": [],
        },
        pipeline_resource_id: {
            "call_count": 0,
            "operation_counts": {"send_recv": 0},
            "topology_counts": {"point_to_point": 0},
            "bytes_by_semantics": {"payload": 0},
            "intervals": [],
        },
    }
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
        if (not calls
                or all_call_ids.intersection(call_ids)
                or {call["rank"] for call in calls} != {0, 1, 2, 3}
                or any(
                    call["pool_name"] != "main"
                    or call["kernel_spec"]["hardware_id"] != hardware_id
                    for call in calls
                )):
            raise ValueError(f"{context} four-rank coverage is invalid")
        all_call_ids.update(call_ids)
        calls_by_id = {call["call_id"]: call for call in calls}
        communication_calls = tuple(
            call for call in calls
            if call["kernel_spec"]["kind"] == "COMMUNICATION"
        )
        step_operation_counts = {
            "all_gather": 0,
            "all_reduce": 0,
            "send_recv": 0,
        }
        for call in communication_calls:
            operation = call["kernel_spec"]["transfer"]["operation"]
            if operation not in step_operation_counts:
                raise ValueError(
                    f"{context} emitted unsupported communication {operation!r}")
            step_operation_counts[operation] += 1
        expected_all_reduce_count = (
            1 + 2 * expected_model.num_hidden_layers)
        expected_communication_count = (
            expected_all_reduce_count + 3 + 4)
        if (len(communication_calls) != expected_communication_count
                or step_operation_counts != {
                    "all_gather": 3,
                    "all_reduce": expected_all_reduce_count,
                    "send_recv": 4,
                }):
            raise ValueError(
                f"{context} communication operation counts disagree")

        source_frontier_id = (
            f"{step_id}:layer-{local_layer_count - 1}:down:tp-all-reduce")
        expected_lane_pairs = ((0, 2), (1, 3))
        p2p_ids_by_tensor = {}
        for tensor_name in ("hidden_states", "residual"):
            tensor_slug = tensor_name.replace("_", "-")
            p2p_ids = tuple(
                f"{step_id}:pp-0-to-1:{tensor_slug}:tp-rank-{lane_index}"
                for lane_index in range(2)
            )
            p2p_ids_by_tensor[tensor_name] = p2p_ids
            boundary_id = (
                f"{step_id}:pp-0-to-1:{tensor_slug}:tp-all-gather")
            if (any(call_id not in calls_by_id for call_id in p2p_ids)
                    or boundary_id not in calls_by_id):
                raise ValueError(
                    f"{context} pipeline boundary call coverage is invalid")
            for call_id, lane_pair in zip(
                    p2p_ids, expected_lane_pairs, strict=True):
                call = calls_by_id[call_id]
                transfer = call["kernel_spec"]["transfer"]
                if (call["dependencies"] != [source_frontier_id]
                        or transfer["participants"] != list(lane_pair)
                        or transfer["source_rank"] != lane_pair[0]
                        or transfer["destination_rank"] != lane_pair[1]):
                    raise ValueError(
                        f"{context} source-to-P2P lane dependency is invalid")
            boundary_call = calls_by_id[boundary_id]
            boundary_transfer = boundary_call["kernel_spec"]["transfer"]
            if (boundary_call["dependencies"] != list(p2p_ids)
                    or boundary_transfer["participants"] != [2, 3]
                    or boundary_transfer["ring_order"] != [2, 3]):
                raise ValueError(
                    f"{context} P2P-to-all-gather dependency is invalid")
        boundary_ids = tuple(
            f"{step_id}:pp-0-to-1:{tensor_name.replace('_', '-')}:"
            "tp-all-gather"
            for tensor_name in ("hidden_states", "residual")
        )
        destination_layer_index = local_layer_count
        for lane_index in range(2):
            destination_norm_id = (
                f"{step_id}:layer-{destination_layer_index}:input-norm:"
                f"tp-rank-{lane_index}")
            if (destination_norm_id not in calls_by_id
                    or calls_by_id[destination_norm_id]["dependencies"]
                    != list(boundary_ids)):
                raise ValueError(
                    f"{context} boundary-to-destination dependency is invalid")

        token_count = sum(
            request["input_tokens"] + request["output_tokens"]
            for request in step["backend_step"]["schedule"]["requests"]
        )
        if token_count <= 0:
            raise ValueError(f"{context} Scheduler step contains no token work")
        full_tensor_bytes = (
            token_count * expected_model.hidden_size * expected_model.dtype_bytes)
        if full_tensor_bytes % 2:
            raise ValueError(f"{context} boundary bytes do not divide TP size")
        expected_wire_bytes = full_tensor_bytes // 2
        source_transfer = calls_by_id[source_frontier_id]["kernel_spec"][
            "transfer"]
        if (source_transfer["operation"] != "all_reduce"
                or source_transfer["participants"] != [0, 1]
                or source_transfer["bytes"] != full_tensor_bytes):
            raise ValueError(f"{context} source all-reduce bytes are invalid")
        for tensor_name in ("hidden_states", "residual"):
            for call_id in p2p_ids_by_tensor[tensor_name]:
                if calls_by_id[call_id]["kernel_spec"]["transfer"][
                        "bytes"] != expected_wire_bytes:
                    raise ValueError(f"{context} P2P wire bytes are invalid")
            boundary_id = (
                f"{step_id}:pp-0-to-1:"
                f"{tensor_name.replace('_', '-')}:tp-all-gather")
            if calls_by_id[boundary_id]["kernel_spec"]["transfer"][
                    "bytes"] != expected_wire_bytes:
                raise ValueError(
                    f"{context} boundary all-gather bytes are invalid")

        intervals_by_id = {
            interval["call_id"]: interval
            for interval in step["execution_timeline"]["intervals"]
        }
        worker_pool_intervals.extend(
            (
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            )
            for interval in intervals_by_id.values()
        )
        for call in communication_calls:
            transfer = call["kernel_spec"]["transfer"]
            operation = transfer["operation"]
            resource_id = transfer["resource_id"]
            if resource_id not in resource_stats:
                raise ValueError(f"{context} communication fabric is invalid")
            if (transfer["backend"] != "analytical_roofline_v1"
                    or transfer["backend_config_sha256"] is not None
                    or transfer["link"] != "nvlink"):
                raise ValueError(
                    f"{context} communication backend contract is invalid")
            if resource_id == tensor_resource_id:
                if operation == "all_reduce":
                    expected_reduction = "sum"
                elif operation == "all_gather":
                    expected_reduction = None
                else:
                    raise ValueError(
                        f"{context} tensor collective operation is invalid")
                if (operation not in ("all_reduce", "all_gather")
                        or transfer["algorithm"] != "ring"
                        or transfer["direction"] != "collective"
                        or transfer["topology"] != "ring"
                        or transfer["participants"] not in ([0, 1], [2, 3])
                        or transfer["ring_order"] != transfer["participants"]
                        or transfer["bytes_semantics"] != "per_rank_input"
                        or transfer["source_rank"] is not None
                        or transfer["destination_rank"] is not None
                        or transfer["reduction"] != expected_reduction):
                    raise ValueError(
                        f"{context} tensor collective contract is invalid")
            elif resource_id == pipeline_resource_id:
                if (operation != "send_recv"
                        or transfer["algorithm"] != "send_recv"
                        or transfer["direction"] != "point_to_point"
                        or transfer["topology"] != "point_to_point"
                        or tuple(transfer["participants"])
                        not in expected_lane_pairs
                        or transfer["ring_order"] != []
                        or transfer["bytes_semantics"] != "payload"
                        or transfer["reduction"] is not None):
                    raise ValueError(
                        f"{context} pipeline P2P contract is invalid")
            else:
                raise AssertionError("validated communication fabric vanished")
            stats = resource_stats[resource_id]
            topology = transfer["topology"]
            semantics = transfer["bytes_semantics"]
            stats["call_count"] += 1
            stats["operation_counts"][operation] += 1
            stats["topology_counts"][topology] += 1
            stats["bytes_by_semantics"][semantics] += transfer["bytes"]
            interval = intervals_by_id[call["call_id"]]
            stats["intervals"].append((
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            ))
        total_call_count += len(calls)

    if (set(entries_by_id) != all_step_ids
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count):
        raise ValueError(f"{context} kernel or schedule counts disagree")
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
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        enable_expert_parallel=False,
        expected_mtp=expected_simulation["mtp"],
        context=context,
    )
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

    system_resources = result["metrics"]["system_resources"]
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    communication_resources = system_resources[
        "communication_resource_utilization"]
    if set(communication_resources) != set(resource_stats):
        raise ValueError(
            f"{context} communication resource coverage is invalid")
    for resource_id, stats in resource_stats.items():
        resource = communication_resources[resource_id]
        busy_time_s = interval_union_duration_from_output(
            stats["intervals"],
            observation_window_s,
            context=f"{context} {resource_id}",
        )
        idle_time_s = observation_window_s - busy_time_s
        if (resource["backend"] != "analytical_roofline_v1"
                or resource["backend_config_sha256"] is not None
                or resource["hardware_id"] != hardware_id
                or resource["link"] != "nvlink"
                or resource["timing_status"]
                != "projected_analytical_roofline"
                or resource["call_count"] != stats["call_count"]
                or resource["operation_counts"]
                != stats["operation_counts"]
                or resource["topology_counts"]
                != stats["topology_counts"]
                or resource["bytes_by_semantics"]
                != stats["bytes_by_semantics"]
                or not math.isclose(
                    resource["busy_time_s"],
                    busy_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    resource["idle_time_s"],
                    idle_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    resource["utilization"],
                    busy_time_s / observation_window_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{context} communication metrics disagree for {resource_id}")
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
            or system_resources["link_utilization"] != {}
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
        raise ValueError(f"{context} system resource metrics disagree")
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
