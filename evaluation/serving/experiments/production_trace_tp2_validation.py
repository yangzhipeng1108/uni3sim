"""TP2 colocated Production Trace point validation."""

from __future__ import annotations

import math
from typing import Any
from evaluation.serving.metrics.latency_distribution import latency_distribution_manifest
from evaluation.serving.metrics.slo_goodput import slo_goodput_manifest
from evaluation.serving.metrics.system_resource_analysis import critical_path_manifest, system_resource_manifest
from evaluation.serving.simulation.execution_artifact import execution_artifact_manifest, execution_step_components_manifest, prefix_alignment_manifest
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec, PhysicalTransferConfig
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    WorkerPoolConfig,
)
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
    expected_performance_timing_status,
    json_round_trip,
    load_expected_hardware,
    replay_pinned_parallel_scheduler,
    validate_canonical_production_trace_step_duration,
    validate_production_trace_capacity,
    validate_production_trace_scheduler_binding,
)

from evaluation.serving.experiments.production_trace_colocated_kv_replay import (
    validate_colocated_kv_lifecycle_replay,
    tp2_collective_bytes_from_schedule_and_model,
)


def validate_colocated_tp2_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    replay_identity = expected_simulation["kv_lifecycle_replay"]
    if replay_identity is None:
        kv_lifecycle_replay = None
        context = "TP=2 Production Trace"
    else:
        try:
            kv_lifecycle_replay = load_kv_lifecycle_replay(
                replay_identity["path"])
        except (OSError, TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                "TP=2 Host-KV Production Trace replay cannot be "
                "independently reloaded") from error
        if (kv_lifecycle_replay.source_sha256
                != replay_identity["source_sha256"]
                or kv_lifecycle_replay.artifact_id
                != replay_identity["artifact_id"]):
            raise ValueError(
                "TP=2 Host-KV Production Trace replay identity changed")
        context = "TP=2 Host-KV Production Trace"
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
                ParallelConfig(tensor_parallel_size=2),
                kv_cache=(
                    None
                    if kv_lifecycle_replay is None
                    else kv_lifecycle_replay.cache_config
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
            "TP=2 Production Trace worker-pool topology disagrees with "
            "command manifest")

    scheduler = expected_simulation["scheduler"]
    runtime = result["scheduler_runtime"]["main"]
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(expected_model.num_hidden_layers)
    ]
    if runtime["scheduler_kv_cache_tensor_owners"] != expected_owners:
        raise ValueError(
            "TP=2 Production Trace Scheduler KV owners disagree with the "
            "pinned Qwen3 model")
    validate_production_trace_capacity(
        result,
        scheduler,
        ("main",),
        context="TP=2 Production Trace",
    )
    expected_kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(2, 1))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(
            "TP=2 Production Trace KV capacity disagrees with model "
            "sharding")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(
            "TP=2 Production Trace routing or transfer state is invalid")
    if (kv_lifecycle_replay is None
            and (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            })):
        raise ValueError(
            "TP=2 Production Trace unexpectedly enabled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(
            "TP=2 Production Trace unexpectedly enabled MTP")
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
            tensor_parallel_size=2,
            context=context,
        )
        return

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
        raise ValueError(
            "TP=2 Production Trace execution artifact contains a route or "
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
            "TP=2 Production Trace execution artifact counts disagree")
    trace_entries = artifact_trace["entries"]
    entries_by_id = {
        entry["step_id"]: entry for entry in trace_entries
    }
    if (len(entries_by_id) != len(trace_entries)
            or len(entries_by_id) != len(steps)):
        raise ValueError(f"{context} schedule trace coverage disagrees")

    operation_counts = {"all_gather": 0, "all_reduce": 0}
    communication_bytes = 0
    communication_intervals = []
    total_call_count = 0
    all_call_ids = set()
    compute_kinds = {"GEMM", "FLASH_ATTN", "VECTOR"}
    hardware = load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    all_step_ids = set()
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
            raise ValueError(
                "TP=2 Production Trace execution step used an unknown pool")
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
            workload_object = components.workload
            kernel_plan_object = components.kernel_plan
            prediction_objects = components.predictions
            timeline_object = components.execution_timeline
        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"{context} step cannot be canonically rematerialized") \
                from error
        backend_steps_by_id[step_id] = backend_step
        expected_components = json_round_trip(
            execution_step_components_manifest(
                workload_object,
                kernel_plan_object,
                prediction_objects,
                timeline_object,
                origin_time_s=step["schedule_interval"]["start_time_s"],
            ))
        actual_components = {
            field: step[field]
            for field in expected_components
        }
        schedule = step["backend_step"]["schedule"]
        kernel_plan = step["kernel_plan"]
        calls = kernel_plan["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        if (not calls
                or len(call_ids) != len(set(call_ids))
                or kernel_plan["call_count"] != len(calls)
                or kernel_plan["topological_order"] != list(call_ids)):
            raise ValueError(
                "TP=2 Production Trace KernelPlan identity is invalid")
        workload = step["workload_graph"]
        nodes = workload["nodes"]
        nodes_by_id = {node["node_id"]: node for node in nodes}
        if (len(nodes_by_id) != len(nodes)
                or workload["node_count"] != len(nodes)
                or set(nodes_by_id) != set(call_ids)
                or workload["topological_order"] != list(call_ids)
                or workload["edge_count"] != len(workload["edges"])):
            raise ValueError(
                "TP=2 Production Trace WorkloadGraph identity is invalid")
        expected_lowering = [{
            "node_id": node["node_id"],
            "call_ids": [node["node_id"]],
        } for node in nodes]
        lowering = step["lowering"]
        if (lowering["relation"]
                != "workload_nodes_to_kernel_calls_many_to_many_v1"
                or lowering["node_to_call_ids"] != expected_lowering):
            raise ValueError(
                "TP=2 Production Trace lowering identity is invalid")
        if all_call_ids.intersection(call_ids):
            raise ValueError(
                "TP=2 Production Trace contains duplicate cross-step call IDs")
        all_call_ids.update(call_ids)
        predictions = step["predictions"]
        predictions_by_id = {
            prediction["call_id"]: prediction
            for prediction in predictions
        }
        if (len(predictions_by_id) != len(predictions)
                or set(predictions_by_id) != set(call_ids)):
            raise ValueError(
                "TP=2 Production Trace predictions do not cover KernelPlan")
        if ({call["rank"] for call in calls} != {0, 1}
                or any(call["pool_name"] != "main"
                       or call["kernel_spec"]["hardware_id"] != hardware_id
                       for call in calls)):
            raise ValueError(
                "TP=2 Production Trace rank coverage disagrees with the "
                "worker pool")
        call_position = {
            call_id: index for index, call_id in enumerate(call_ids)
        }
        graph_predecessors = {call_id: [] for call_id in call_ids}
        for edge in workload["edges"]:
            if (edge["source"] not in call_position
                    or edge["destination"] not in call_position):
                raise ValueError(
                    "TP=2 Production Trace WorkloadGraph edge is invalid")
            graph_predecessors[edge["destination"]].append(edge["source"])
        calls_by_id = {call["call_id"]: call for call in calls}
        for call in calls:
            call_id = call["call_id"]
            if call["source_node_ids"] != [call_id]:
                raise ValueError(
                    "TP=2 Production Trace call/source-node identity is "
                    "invalid")
            dependencies = call["dependencies"]
            if (len(dependencies) != len(set(dependencies))
                    or set(dependencies) != set(graph_predecessors[call_id])
                    or len(dependencies)
                    != len(graph_predecessors[call_id])
                    or any(dependency not in call_position
                           or call_position[dependency] >= call_position[call_id]
                           for dependency in dependencies)):
                raise ValueError(
                    "TP=2 Production Trace KernelPlan dependency is invalid")
            kernel_kind = call["kernel_spec"]["kind"]
            if kernel_kind == "COMMUNICATION":
                if call_id.endswith(":tp-all-reduce"):
                    base_id = call_id[:-len(":tp-all-reduce")]
                elif call_id.endswith(":tp-all-gather"):
                    base_id = call_id[:-len(":tp-all-gather")]
                else:
                    raise ValueError(
                        "TP=2 Production Trace collective call ID is invalid")
                expected_dependencies = [
                    f"{base_id}:tp-rank-0",
                    f"{base_id}:tp-rank-1",
                ]
                if dependencies != expected_dependencies:
                    raise ValueError(
                        "TP=2 Production Trace collective dependency is "
                        "invalid")
            elif kernel_kind in compute_kinds:
                if call["rank"] == 0:
                    rank_suffix = ":tp-rank-0"
                    other_rank_suffix = ":tp-rank-1"
                elif call["rank"] == 1:
                    rank_suffix = ":tp-rank-1"
                    other_rank_suffix = ":tp-rank-0"
                else:
                    raise ValueError(
                        "TP=2 Production Trace compute call rank is invalid")
                if (not call_id.endswith(rank_suffix)
                        or any(dependency.endswith(other_rank_suffix)
                               for dependency in dependencies)):
                    raise ValueError(
                        "TP=2 Production Trace rank-local DAG is invalid")
            else:
                raise ValueError(
                    "TP=2 Production Trace emitted an unsupported kernel "
                    "kind")
        intervals = step["execution_timeline"]["intervals"]
        intervals_by_id = {
            interval["call_id"]: interval for interval in intervals
        }
        if (len(intervals_by_id) != len(intervals)
                or set(intervals_by_id) != set(call_ids)):
            raise ValueError(
                "TP=2 Production Trace timeline does not cover KernelPlan")
        for call in calls:
            call_id = call["call_id"]
            interval = intervals_by_id[call_id]
            prediction = predictions_by_id[call_id]
            duration_s = interval["end_time_s"] - interval["start_time_s"]
            absolute_duration_s = (
                interval["absolute_end_time_s"]
                - interval["absolute_start_time_s"])
            if (interval["pool_name"] != call["pool_name"]
                    or interval["rank"] != call["rank"]
                    or interval["resource"] != call["resource"]
                    or not math.isclose(
                        duration_s,
                        absolute_duration_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        duration_s,
                        prediction["t_total_s"],
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )):
                raise ValueError(
                    "TP=2 Production Trace prediction/timeline identity is "
                    "invalid")
            kernel_kind = call["kernel_spec"]["kind"]
            if kernel_kind == "COMMUNICATION":
                expected_reserved_resources = [
                    f"communication:endpoint:{hardware_id}:0:nvlink",
                    f"communication:endpoint:{hardware_id}:1:nvlink",
                    f"communication:fabric:{resource_id}",
                ]
                communication_intervals.append((
                    interval["absolute_start_time_s"],
                    interval["absolute_end_time_s"],
                ))
            elif kernel_kind in compute_kinds:
                expected_reserved_resources = [
                    f"pool:main:rank:{call['rank']}:compute",
                ]
            else:
                raise ValueError(
                    "TP=2 Production Trace emitted an unsupported kernel "
                    "kind")
            if interval["reserved_resource_ids"] != expected_reserved_resources:
                raise ValueError(
                    "TP=2 Production Trace reserved resources disagree with "
                    "KernelPlan")
        step_operations = set()
        for call in calls:
            kernel_spec = call["kernel_spec"]
            if kernel_spec["kind"] != "COMMUNICATION":
                continue
            transfer = kernel_spec["transfer"]
            operation = transfer["operation"]
            if operation not in operation_counts:
                raise ValueError(
                    "TP=2 Production Trace emitted an unsupported collective")
            if operation == "all_reduce":
                expected_reduction = "sum"
            elif operation == "all_gather":
                expected_reduction = None
            else:
                raise AssertionError(
                    "validated TP=2 collective operation vanished")
            if (call["rank"] != 0
                    or call["resource"] != resource_id
                    or kernel_spec["dtype"] is not None
                    or kernel_spec["shape"] != {}
                    or kernel_spec["attributes"] != {}
                    or kernel_spec["variant"] is not None
                    or kernel_spec["launch"] is not None
                    or transfer["algorithm"] != "ring"
                    or transfer["backend"] != "analytical_roofline_v1"
                    or transfer["backend_config_sha256"] is not None
                    or transfer["bytes_semantics"] != "per_rank_input"
                    or not isinstance(transfer["bytes"], int)
                    or isinstance(transfer["bytes"], bool)
                    or transfer["bytes"] <= 0
                    or transfer["direction"] != "collective"
                    or transfer["link"] != "nvlink"
                    or transfer["topology"] != "ring"
                    or transfer["resource_id"] != resource_id
                    or transfer["participants"] != [0, 1]
                    or transfer["ring_order"] != [0, 1]
                    or transfer["reduction"] != expected_reduction
                    or transfer["source_rank"] is not None
                    or transfer["destination_rank"] is not None
                    or transfer["tier"] is not None):
                raise ValueError(
                    "TP=2 Production Trace collective KernelSpec is invalid")
            expected_bytes = tp2_collective_bytes_from_schedule_and_model(
                expected_model,
                schedule,
                call,
                calls_by_id,
                nodes_by_id,
            )
            if transfer["bytes"] != expected_bytes:
                raise ValueError(
                    "TP=2 Production Trace collective bytes disagree with "
                    "the Scheduler/model shape")
            prediction = predictions_by_id[call["call_id"]]
            if (prediction["bottleneck_group"] != "communication"
                    or prediction["t_launch_s"] != 0.0
                    or prediction["t_total_s"] <= 0.0
                    or prediction["t_per_group_s"] != {
                        "communication": prediction["t_total_s"]}
                    or prediction["t_per_resource_s"] != {
                        "communication": prediction["t_total_s"]}):
                raise ValueError(
                    "TP=2 Production Trace collective prediction is invalid")
            try:
                canonical_kernel = KernelSpec(
                    kind=KernelKind.COMMUNICATION,
                    hardware_id=hardware_id,
                    transfer=PhysicalTransferConfig(**transfer),
                )
                canonical_prediction = predict(
                    canonical_kernel,
                    hardware,
                    expected_simulation["prediction_method"],
                )
            except (TypeError, ValueError, NotImplementedError) as error:
                raise ValueError(
                    "TP=2 Production Trace collective cannot be repriced") \
                    from error
            if (prediction["t_total_s"] != canonical_prediction.t_total_s
                    or prediction["t_launch_s"]
                    != canonical_prediction.t_launch_s
                    or prediction["bottleneck_group"]
                    != canonical_prediction.bottleneck_group
                    or prediction["t_per_group_s"]
                    != dict(canonical_prediction.t_per_group)
                    or prediction["t_per_resource_s"]
                    != dict(canonical_prediction.t_per_resource)):
                raise ValueError(
                    "TP=2 Production Trace collective prediction disagrees "
                    "with kernel_base.predict")
            operation_counts[operation] += 1
            communication_bytes += transfer["bytes"]
            step_operations.add(operation)
        if step_operations != set(operation_counts):
            raise ValueError(
                "TP=2 Production Trace step is missing a required collective")
        if actual_components != expected_components:
            raise ValueError(
                f"{context} canonical Scheduler-to-KernelSpec "
                "rematerialization disagrees with the execution artifact")
        validate_canonical_production_trace_step_duration(
            step,
            timeline_object.duration_s,
            context=context,
        )
        simulation_steps.append(SimulationStep(
            backend_step,
            workload_object,
            kernel_plan_object,
            prediction_objects,
            timeline_object,
        ))
        total_call_count += len(calls)
    if (set(entries_by_id) != all_step_ids
            or operation_counts["all_gather"] <= 0
            or operation_counts["all_reduce"] <= 0
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count):
        raise ValueError(
            "TP=2 Production Trace kernel or collective counts disagree")

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
    communication_resources = system_resources[
        "communication_resource_utilization"]
    if set(communication_resources) != {resource_id}:
        raise ValueError(
            "TP=2 Production Trace communication resources disagree with "
            "the worker pool")
    communication = communication_resources[resource_id]
    if (communication["backend"] != "analytical_roofline_v1"
            or communication["backend_config_sha256"] is not None
            or communication["hardware_id"] != hardware_id
            or communication["link"] != "nvlink"
            or communication["timing_status"]
            != "projected_analytical_roofline"
            or communication["call_count"] != sum(operation_counts.values())
            or communication["operation_counts"] != operation_counts
            or communication["topology_counts"] != {
                "ring": sum(operation_counts.values())}
            or communication["bytes_by_semantics"] != {
                "per_rank_input": communication_bytes}):
        raise ValueError(
            "TP=2 Production Trace communication metrics disagree with "
            "collective calls")
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
                "TP=2 Production Trace communication intervals overlap or "
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
            "TP=2 Production Trace communication utilization disagrees "
            "with execution intervals")
    if (system_resources["link_utilization"] != {}
            or set(system_resources["worker_pool_utilization"]) != {"main"}
            or system_resources["worker_pool_utilization"]["main"][
                "kernel_interval_count"] != total_call_count
            or set(system_resources["scheduler_queue"]["by_pool"])
            != {"main"}):
        raise ValueError(
            "TP=2 Production Trace system resource metrics disagree with "
            "the worker pool")

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
