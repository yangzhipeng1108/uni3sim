"""Production trace contracts; extracted without changing the established contracts."""

from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
from pathlib import Path
from typing import Any
from evaluation.serving.metrics.latency_distribution import latency_distribution_manifest
from evaluation.serving.metrics.slo_goodput import slo_goodput_manifest
from evaluation.serving.metrics.system_resource_analysis import critical_path_manifest, system_resource_manifest
from evaluation.serving.simulation.execution_artifact import performance_timing_status
from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm_scheduler_process import VLLMSchedulerRuntimeInfo, vllm_scheduler_source_identity
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.backend.mtp_acceptance_assumption import MTPAcceptanceAssumption
from src.serving.deployment.deployment_plan import DeploymentPlan
from src.serving.kv_cache.kv_cache_events import KVCacheEvent
from src.serving.request_gen.production_trace import ProductionTraceReaderConfig, SuccessfulRequestFilter, build_production_trace_replay, load_production_trace, production_trace_manifest, production_trace_replay_manifest
from src.serving.request_gen.request_stream import RequestStream, request_stream_sha256
from src.serving.schedule.backend_protocol import BackendConfig, scheduler_effective_prefix_hit_tokens
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep, ScheduledRequest
from src.serving.schedule.schedule_trace import RouteDecision, ScheduleTrace
from src.serving.schedule.single_pool_engine import run_single_pool_schedule
from src.serving.metrics.serving_metrics import compute_serving_metrics
from src.serving.simulation import SimulationResult
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.runtime_inputs import ExpertRoutingInput


ROOT = Path(__file__).resolve().parents[3]


def require_exact_output_object(
    value: object,
    expected_fields: set[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    supplied_fields = set(value)
    if supplied_fields != expected_fields:
        raise ValueError(
            f"{context} fields disagree with schema: "
            f"missing={sorted(expected_fields - supplied_fields)}, "
            f"unexpected={sorted(supplied_fields - expected_fields)}")
    return value


def validate_production_trace_v2_output_schema(payload: object) -> None:
    sweep = require_exact_output_object(
        payload,
        {
            "schema_version",
            "kind",
            "interpretation",
            "interactivity_tgs",
            "peak_selection",
            "saturation_analysis",
            "scale_order",
            "arrival_transform",
            "point_count",
            "arrival_rate_scales",
            "source_trace",
            "execution_provenance",
            "points",
        },
        context="Production Trace sweep",
    )
    points = sweep["points"]
    if not isinstance(points, list):
        raise ValueError("Production Trace sweep points must be a list")
    for point_index, point_value in enumerate(points):
        point_context = f"Production Trace sweep point {point_index}"
        point = require_exact_output_object(
            point_value,
            {
                "arrival_rate_scale",
                "request_stream_sha256",
                "validation_status",
                "metrics",
                "result",
            },
            context=point_context,
        )
        require_exact_output_object(
            point["validation_status"],
            {
                "trace_replay",
                "scheduler_control_plane",
                "performance_timing",
            },
            context=f"{point_context} validation status",
        )
        require_exact_output_object(
            point["metrics"],
            {
                "request_throughput_per_s",
                "output_token_throughput_per_s",
                "token_throughput_per_s",
                "latency_distributions",
                "slo_goodput",
            },
            context=f"{point_context} summary metrics",
        )
        result = require_exact_output_object(
            point["result"],
            {
                "version",
                "backend",
                "deployment_id",
                "hardware_ids",
                "model_compute_dtype",
                "model_config_provenance",
                "deployment",
                "deployment_plan",
                "routing_policy",
                "route_decision_version",
                "transfer_trace_version",
                "kv_handoff_mode",
                "kv_capacity",
                "kv_lifecycle_replay",
                "launch_policy",
                "prediction_method",
                "execution_mode",
                "execution_cache_stats",
                "performance_timing_status",
                "execution_provenance",
                "execution_artifact",
                "scheduler_runtime",
                "mtp_scenario",
                "mtp_acceptance_outcomes",
                "mtp_pd_handoff_contract",
                "arrival_process",
                "request_arrivals_s",
                "workload_hash",
                "num_steps",
                "num_transfers",
                "num_kernel_calls",
                "schedule_end_time_s",
                "route_decisions",
                "transfers",
                "metrics",
                "request_source",
            },
            context=f"{point_context} result",
        )
        require_exact_output_object(
            result["request_source"],
            {
                "schema_version",
                "kind",
                "trace",
                "replay",
                "zero_output_contract",
                "prefix_alignment",
            },
            context=f"{point_context} request source",
        )
        require_exact_output_object(
            result["metrics"],
            {
                "request_throughput_per_s",
                "output_token_throughput_per_s",
                "token_throughput_per_s",
                "latency_distributions",
                "slo_goodput",
                "system_resources",
                "critical_path",
                "requests",
            },
            context=f"{point_context} result metrics",
        )
        artifact = require_exact_output_object(
            result["execution_artifact"],
            {
                "schema_version",
                "hash_method",
                "performance_timing_status",
                "counts",
                "schedule_trace",
                "steps",
                "transfers",
                "modeled_kv_offload",
                "artifact_sha256",
            },
            context=f"{point_context} execution artifact",
        )
        require_exact_output_object(
            artifact["counts"],
            {
                "schedule_steps",
                "simulation_steps",
                "simulation_transfers",
                "trace_transfers",
                "modeled_kv_offload_transfers",
                "kernel_calls",
            },
            context=f"{point_context} execution artifact counts",
        )
        schedule_trace = require_exact_output_object(
            artifact["schedule_trace"],
            {
                "route_decision_version",
                "transfer_trace_version",
                "entries",
                "route_decisions",
                "transfers",
            },
            context=f"{point_context} execution artifact schedule trace",
        )
        entries = schedule_trace["entries"]
        steps = artifact["steps"]
        transfers = artifact["transfers"]
        if (not isinstance(entries, list)
                or not isinstance(steps, list)
                or not isinstance(transfers, list)):
            raise ValueError(
                f"{point_context} execution artifact collections must be lists")
        for entry_index, entry in enumerate(entries):
            require_exact_output_object(
                entry,
                {"step_id", "pool_name", "start_time_s", "end_time_s"},
                context=(f"{point_context} execution artifact schedule "
                         f"entry {entry_index}"),
            )
        for step_index, step in enumerate(steps):
            validated_step = require_exact_output_object(
                step,
                {
                    "step_id",
                    "pool_name",
                    "schedule_interval",
                    "backend_step",
                    "workload_graph",
                    "lowering",
                    "kernel_plan",
                    "predictions",
                    "execution_timeline",
                },
                context=(f"{point_context} execution artifact step "
                         f"{step_index}"),
            )
            try:
                backend_step_from_output_manifest(
                    validated_step["backend_step"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{point_context} execution artifact step {step_index} "
                    "has an invalid BackendStep") from error
            validate_execution_components_output_schema(
                validated_step,
                context=(f"{point_context} execution artifact step "
                         f"{step_index}"),
            )
        for transfer_index, transfer in enumerate(transfers):
            validated_transfer = require_exact_output_object(
                transfer,
                {
                    "request_id",
                    "source_engine_id",
                    "destination_engine_id",
                    "trace_transfer",
                    "workload_graph",
                    "lowering",
                    "kernel_plan",
                    "predictions",
                    "execution_timeline",
                },
                context=(f"{point_context} execution artifact transfer "
                         f"{transfer_index}"),
            )
            validate_execution_components_output_schema(
                validated_transfer,
                context=(f"{point_context} execution artifact transfer "
                         f"{transfer_index}"),
            )


def require_output_list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def validate_execution_components_output_schema(
    owner: dict[str, Any],
    *,
    context: str,
) -> None:
    workload = owner["workload_graph"]
    if not isinstance(workload, dict):
        raise ValueError(f"{context} workload graph must be an object")
    workload_fields = {
        "node_count", "edge_count", "topological_order", "nodes", "edges",
    }
    compact_workload = "representation" in workload
    if compact_workload:
        workload_fields.update({
            "representation", "logical_node_count", "logical_edge_count",
            "block_repeats",
        })
    workload = require_exact_output_object(
        workload,
        workload_fields,
        context=f"{context} workload graph",
    )
    for node_index, node in enumerate(require_output_list(
            workload["nodes"], context=f"{context} workload nodes")):
        require_exact_output_object(
            node,
            {
                "node_id", "kind", "pool_name", "shape", "dtype",
                "variant", "rank", "attributes",
            },
            context=f"{context} workload node {node_index}",
        )
    for edge_index, edge in enumerate(require_output_list(
            workload["edges"], context=f"{context} workload edges")):
        require_exact_output_object(
            edge,
            {"source", "destination", "kind"},
            context=f"{context} workload edge {edge_index}",
        )
    if compact_workload:
        if workload["representation"] != (
                "materialized_templates_with_logical_repeats_v1"):
            raise ValueError(f"{context} workload representation is invalid")
        for repeat_index, repeat in enumerate(require_output_list(
                workload["block_repeats"],
                context=f"{context} workload block repeats")):
            repeat = require_exact_output_object(
                repeat,
                {
                    "block_id", "node_ids", "repeat_count",
                    "feedback_edges", "trace_node_ids",
                },
                context=f"{context} workload block repeat {repeat_index}",
            )
            for edge_index, edge in enumerate(require_output_list(
                    repeat["feedback_edges"],
                    context=(f"{context} workload block repeat "
                             f"{repeat_index} feedback edges"))):
                require_exact_output_object(
                    edge,
                    {"source", "destination", "kind"},
                    context=(f"{context} workload block repeat "
                             f"{repeat_index} feedback edge {edge_index}"),
                )
            for trace_index, trace in enumerate(require_output_list(
                    repeat["trace_node_ids"],
                    context=(f"{context} workload block repeat "
                             f"{repeat_index} trace nodes"))):
                require_exact_output_object(
                    trace,
                    {"template_node_id", "logical_node_ids"},
                    context=(f"{context} workload block repeat "
                             f"{repeat_index} trace node {trace_index}"),
                )

    lowering = require_exact_output_object(
        owner["lowering"],
        {"relation", "node_to_call_ids"},
        context=f"{context} lowering",
    )
    for mapping_index, mapping in enumerate(require_output_list(
            lowering["node_to_call_ids"],
            context=f"{context} lowering mappings")):
        require_exact_output_object(
            mapping,
            {"node_id", "call_ids"},
            context=f"{context} lowering mapping {mapping_index}",
        )

    kernel_plan = owner["kernel_plan"]
    if not isinstance(kernel_plan, dict):
        raise ValueError(f"{context} kernel plan must be an object")
    kernel_plan_fields = {"call_count", "topological_order", "calls"}
    compact_kernel_plan = "representation" in kernel_plan
    if compact_kernel_plan:
        kernel_plan_fields.update({
            "representation", "logical_call_count", "block_repeats",
        })
    if compact_kernel_plan != compact_workload:
        raise ValueError(
            f"{context} workload and KernelPlan compact forms disagree")
    kernel_plan = require_exact_output_object(
        kernel_plan,
        kernel_plan_fields,
        context=f"{context} kernel plan",
    )
    for call_index, call_value in enumerate(require_output_list(
            kernel_plan["calls"], context=f"{context} kernel calls")):
        call = require_exact_output_object(
            call_value,
            {
                "call_id", "pool_name", "rank", "resource",
                "source_node_ids", "dependencies", "kernel_spec",
            },
            context=f"{context} kernel call {call_index}",
        )
        spec = require_exact_output_object(
            call["kernel_spec"],
            {
                "kind", "hardware_id", "shape", "dtype", "variant",
                "launch", "transfer", "attributes",
            },
            context=f"{context} kernel call {call_index} spec",
        )
        launch = spec["launch"]
        if launch is not None:
            launch = require_exact_output_object(
                launch,
                {
                    "algorithm", "grid", "block", "tiling", "cluster",
                    "parameters", "execution_variant",
                },
                context=f"{context} kernel call {call_index} launch",
            )
            if launch["tiling"] is not None:
                require_exact_output_object(
                    launch["tiling"],
                    {
                        "tile_shape", "pipeline_depth", "split_k",
                        "num_kv_splits",
                    },
                    context=(f"{context} kernel call {call_index} launch "
                             "tiling"),
                )
        transfer = spec["transfer"]
        if transfer is not None:
            require_exact_output_object(
                transfer,
                {
                    "algorithm", "bytes", "direction", "link", "topology",
                    "tier", "participants", "operation", "bytes_semantics",
                    "source_rank", "destination_rank", "ring_order",
                    "reduction", "backend", "resource_id",
                    "backend_config_sha256",
                },
                context=f"{context} kernel call {call_index} transfer",
            )
    if compact_kernel_plan:
        if kernel_plan["representation"] != (
                "materialized_templates_with_logical_repeats_v1"):
            raise ValueError(
                f"{context} KernelPlan representation is invalid")
        for repeat_index, repeat in enumerate(require_output_list(
                kernel_plan["block_repeats"],
                context=f"{context} kernel block repeats")):
            repeat = require_exact_output_object(
                repeat,
                {
                    "block_id", "call_ids", "repeat_count",
                    "feedback_dependencies",
                },
                context=f"{context} kernel block repeat {repeat_index}",
            )
            for dependency_index, dependency in enumerate(
                    require_output_list(
                        repeat["feedback_dependencies"],
                        context=(f"{context} kernel block repeat "
                                 f"{repeat_index} feedback dependencies"))):
                require_exact_output_object(
                    dependency,
                    {"destination_call_id", "source_call_ids"},
                    context=(f"{context} kernel block repeat {repeat_index} "
                             f"feedback dependency {dependency_index}"),
                )

    for prediction_index, prediction in enumerate(require_output_list(
            owner["predictions"], context=f"{context} predictions")):
        require_exact_output_object(
            prediction,
            {
                "call_id", "t_total_s", "t_launch_s", "bottleneck_group",
                "t_per_group_s", "t_per_resource_s",
            },
            context=f"{context} prediction {prediction_index}",
        )
    timeline = owner["execution_timeline"]
    if not isinstance(timeline, dict):
        raise ValueError(f"{context} execution timeline must be an object")
    timeline_fields = {
        "time_basis", "origin_time_s", "duration_s", "intervals",
    }
    compact_timeline = "representation" in timeline
    if compact_timeline:
        timeline_fields.update({
            "representation", "logical_interval_count",
            "call_occurrence_counts", "pool_activity",
            "reserved_resource_activity", "block_repeats",
        })
    if compact_timeline != compact_kernel_plan:
        raise ValueError(
            f"{context} KernelPlan and timeline compact forms disagree")
    timeline = require_exact_output_object(
        timeline,
        timeline_fields,
        context=f"{context} execution timeline",
    )
    for interval_index, interval in enumerate(require_output_list(
            timeline["intervals"], context=f"{context} timeline intervals")):
        require_exact_output_object(
            interval,
            {
                "call_id", "pool_name", "rank", "resource",
                "reserved_resource_ids", "start_time_s", "end_time_s",
                "absolute_start_time_s", "absolute_end_time_s",
            },
            context=f"{context} timeline interval {interval_index}",
        )
    if compact_timeline:
        if timeline["representation"] != (
                "materialized_templates_with_logical_repeats_v1"):
            raise ValueError(f"{context} timeline representation is invalid")
        for count_index, count in enumerate(require_output_list(
                timeline["call_occurrence_counts"],
                context=f"{context} timeline occurrence counts")):
            require_exact_output_object(
                count,
                {"call_id", "occurrence_count"},
                context=(f"{context} timeline occurrence count "
                         f"{count_index}"),
            )
        for activity_field, identity_field in (
                ("pool_activity", "pool_name"),
                ("reserved_resource_activity", "resource_id")):
            for activity_index, activity in enumerate(require_output_list(
                    timeline[activity_field],
                    context=f"{context} timeline {activity_field}")):
                activity = require_exact_output_object(
                    activity,
                    {identity_field, "interval_count", "intervals"},
                    context=(f"{context} timeline {activity_field} "
                             f"{activity_index}"),
                )
                for activity_interval_index, interval in enumerate(
                        require_output_list(
                            activity["intervals"],
                            context=(f"{context} timeline {activity_field} "
                                     f"{activity_index} intervals"))):
                    require_exact_output_object(
                        interval,
                        {
                            "start_time_s", "end_time_s",
                            "absolute_start_time_s", "absolute_end_time_s",
                        },
                        context=(f"{context} timeline {activity_field} "
                                 f"{activity_index} interval "
                                 f"{activity_interval_index}"),
                    )
        for repeat_index, repeat in enumerate(require_output_list(
                timeline["block_repeats"],
                context=f"{context} timeline block repeats")):
            repeat = require_exact_output_object(
                repeat,
                {
                    "block_id", "repeat_count", "first_iteration_start_time_s",
                    "last_iteration_end_time_s", "call_intervals",
                },
                context=f"{context} timeline block repeat {repeat_index}",
            )
            for call_index, call in enumerate(require_output_list(
                    repeat["call_intervals"],
                    context=(f"{context} timeline block repeat "
                             f"{repeat_index} calls"))):
                call = require_exact_output_object(
                    call,
                    {"call_id", "intervals"},
                    context=(f"{context} timeline block repeat {repeat_index} "
                             f"call {call_index}"),
                )
                for call_interval_index, interval in enumerate(
                        require_output_list(
                            call["intervals"],
                            context=(f"{context} timeline block repeat "
                                     f"{repeat_index} call {call_index} "
                                     "intervals"))):
                    require_exact_output_object(
                        interval,
                        {
                            "start_time_s", "end_time_s",
                            "absolute_start_time_s", "absolute_end_time_s",
                        },
                        context=(f"{context} timeline block repeat "
                                 f"{repeat_index} call {call_index} interval "
                                 f"{call_interval_index}"),
                    )


def expected_performance_timing_status(
    simulation: dict[str, Any],
) -> str:
    hardware = simulation["hardware"]
    return performance_timing_status(
        simulation["launch_policy"],
        (hardware["hardware_id"],),
    )


def load_expected_hardware(expected_simulation, hardware_id, context) -> HWSpec:
    """Load the hardware the simulation was recorded against, and check it.

    The recorded config path and the expected hardware id must agree; a
    mismatch means the artifact being validated was produced for another card.
    """

    hardware = HWSpec(str(expected_simulation["hardware"]["config_path"]))
    if hardware.hardware_id != hardware_id:
        raise ValueError(f"{context} hardware config identity changed")
    return hardware


def production_trace_replays_from_command_input(
    expected_input: dict[str, Any],
) -> tuple[Any, ...]:
    expected_trace = expected_input["trace"]
    field_mapping = expected_trace["field_mapping"]
    success_filter = expected_trace["success_filter"]
    try:
        trace = load_production_trace(
            expected_trace["source_path"],
            ProductionTraceReaderConfig(
                source_format=expected_trace["source_format"],
                arrival_time_field=field_mapping["arrival_time"],
                input_tokens_field=field_mapping["input_tokens"],
                output_tokens_field=field_mapping["output_tokens"],
                prefix_hit_tokens_field=field_mapping[
                    "prefix_hit_tokens"],
                request_id_field=field_mapping["request_id"],
                input_time_unit=expected_trace["input_time_unit"],
                successful_request_filter=SuccessfulRequestFilter(
                    mode=success_filter["mode"],
                    field_name=success_filter["field_name"],
                    accepted_value=success_filter["accepted_value"],
                    provenance=success_filter["provenance"],
                ),
            ),
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "Production Trace command input cannot be independently "
            "reloaded") from error
    if production_trace_manifest(trace) != expected_trace:
        raise ValueError(
            "Production Trace command input changed after materialization")
    return tuple(
        build_production_trace_replay(
            trace,
            arrival_rate_scale=arrival_rate_scale,
        )
        for arrival_rate_scale in expected_input["arrival_rate_scales"]
    )


def validate_production_trace_replay_binding(
    expected_trace: dict[str, Any],
    expected_replay: Any,
    point: dict[str, Any],
) -> None:
    expected_replay_manifest = production_trace_replay_manifest(
        expected_replay)
    if (point["request_stream_sha256"]
            != expected_replay.request_stream_sha256):
        raise ValueError(
            "Production Trace sweep request stream disagrees with the "
            "command Trace replay")
    result = point["result"]
    request_source = result["request_source"]
    if set(request_source) != {
            "schema_version", "kind", "trace", "replay",
            "zero_output_contract", "prefix_alignment"}:
        raise ValueError(
            "Production Trace sweep request source fields disagree with "
            "the canonical contract")
    if (request_source["schema_version"] != 1
            or request_source["kind"] != "production_trace"
            or request_source["trace"] != expected_trace
            or request_source["replay"] != expected_replay_manifest
            or result["arrival_process"] != expected_replay_manifest):
        raise ValueError(
            "Production Trace sweep Scheduler input disagrees with the "
            "command Trace replay")
    expected_arrivals = [
        {
            "request_id": request.request_id,
            "arrival_time_s": request.arrival_time_s,
        }
        for request in expected_replay.requests
    ]
    if result["request_arrivals_s"] != expected_arrivals:
        raise ValueError(
            "Production Trace sweep request arrivals disagree with the "
            "command Trace replay")


def validate_production_trace_scheduler_runtime(
    runtime: dict[str, Any],
    expected_scheduler: dict[str, Any],
    expected_model: ModelConfig,
    expected_mtp: dict[str, Any],
    *,
    role: str,
    tensor_parallel_field: str,
) -> None:
    expected_runtime_fields = {
        field.name for field in fields(VLLMSchedulerRuntimeInfo)
    }
    if set(runtime) != expected_runtime_fields:
        raise ValueError(
            "Production Trace sweep Scheduler runtime fields disagree with "
            "the pinned contract")
    scheduler_fields = {
        "max_num_seqs": "max_requests",
        "max_num_batched_tokens": "max_batched_tokens",
        "kv_block_size_tokens": "kv_block_size_tokens",
        "max_model_len": "max_model_len",
        "requested_tensor_parallel_size": tensor_parallel_field,
        "effective_tensor_parallel_size": tensor_parallel_field,
        "requested_pipeline_parallel_size": "prefill_pp_size",
        "effective_pipeline_parallel_size": "prefill_pp_size",
        "requested_expert_parallel": "enable_expert_parallel",
        "effective_expert_parallel": "enable_expert_parallel",
    }
    for runtime_field, manifest_field in scheduler_fields.items():
        if runtime[runtime_field] != expected_scheduler[manifest_field]:
            raise ValueError(
                "Production Trace sweep Scheduler identity disagrees with "
                f"command manifest: {runtime_field}")
    if (expected_scheduler["capacity_mode"]
            == "debug_explicit_block_count"
            and runtime["num_kv_blocks"]
            != expected_scheduler["num_kv_blocks"]):
        raise ValueError(
            "Production Trace sweep Scheduler debug capacity disagrees with "
            "command manifest")
    if (isinstance(runtime["num_kv_blocks"], bool)
            or not isinstance(runtime["num_kv_blocks"], int)
            or runtime["num_kv_blocks"] < 1):
        raise ValueError(
            "Production Trace sweep Scheduler capacity must be positive")
    if runtime["model_config_sha256"] != expected_model.config_sha256:
        raise ValueError(
            "Production Trace sweep Scheduler model identity disagrees with "
            "command manifest")
    source_identity = vllm_scheduler_source_identity()
    actual_source_hashes = tuple(
        tuple(record) for record in runtime["source_hashes"])
    if (runtime["version"] != source_identity["version"]
            or runtime["source_commit"] != source_identity["source_commit"]
            or actual_source_hashes != source_identity["source_hashes"]
            or runtime["package_python_file_count"]
            != source_identity["package_python_file_count"]
            or runtime["package_python_manifest_sha256"]
            != source_identity["package_python_manifest_sha256"]):
        raise ValueError(
            "Production Trace sweep Scheduler source identity disagrees "
            "with the pinned runtime")
    if runtime["backend_role"] != role:
        raise ValueError(
            "Production Trace sweep Scheduler role disagrees with command "
            "manifest")
    if expected_mtp["acceptance_policy"] == "disabled":
        expected_speculative_method = "disabled"
        expected_bootstrap = "version_pinned_scheduler_only_cpu"
    elif expected_mtp["acceptance_policy"] in (
            "fixed_count", "maximum_entropy"):
        expected_speculative_method = "mtp"
        expected_bootstrap = "version_pinned_engine_args_mtp_cpu"
    else:
        raise ValueError(
            "Production Trace command manifest has an unsupported MTP "
            "acceptance policy")
    if runtime["speculative_method"] != expected_speculative_method:
        raise ValueError(
            "Production Trace sweep Scheduler speculative method disagrees "
            "with command manifest")
    expected_tp_size = expected_scheduler[tensor_parallel_field]
    expected_pp_size = expected_scheduler["prefill_pp_size"]
    try:
        expected_kv_heads = expected_model.kv_heads_per_tensor_parallel_rank(
            expected_tp_size)
        expected_kv_bytes = (
            expected_model.kv_bytes_per_token_per_parallel_rank(
                expected_tp_size,
                expected_pp_size,
            ))
    except ValueError as error:
        raise ValueError(
            "Production Trace Scheduler model parallelism is invalid") \
            from error
    expected_local_layers = (
        expected_model.num_hidden_layers // expected_pp_size)
    expected_page_size_bytes = (
        expected_scheduler["kv_block_size_tokens"]
        * expected_kv_bytes // expected_local_layers)
    owners = runtime["scheduler_kv_cache_tensor_owners"]
    if (runtime["platform"] != "cpu"
            or runtime["requested_dtype"] != "bfloat16"
            or runtime["effective_dtype"] != "bfloat16"
            or runtime["device_guard"]
            != "python_tripwire+cuda_state+procfs"
            or runtime["bootstrap"]
            != expected_bootstrap
            or runtime["requested_chunked_prefill"] is not True
            or runtime["effective_chunked_prefill"] is not True
            or runtime["requested_prefix_caching"] is not False
            or runtime["effective_prefix_caching"] is not False
            or runtime["prefix_oracle_mode"]
            != "trace_length_private_nonshared_v1"
            or runtime["scheduler_pipeline_stage_rank"] != 0
            or runtime["scheduler_pipeline_stage_layer_start"] != 0
            or runtime["scheduler_pipeline_stage_layer_end"]
            != expected_local_layers
            or runtime["scheduler_kv_cache_num_heads_per_rank"]
            != expected_kv_heads
            or runtime["scheduler_kv_cache_page_size_bytes"]
            != expected_page_size_bytes
            or len(owners) != expected_local_layers
            or any(not isinstance(owner, list)
                   or len(owner) != 1
                   or not isinstance(owner[0], str)
                   or not owner[0]
                   for owner in owners)
            or runtime["max_num_partial_prefills"] != 1
            or runtime["max_long_partial_prefills"] != 1
            or runtime["long_prefill_token_threshold"] != 0
            or runtime["scheduler_policy"] != "fcfs"
            or runtime["num_lookahead_slots"]
            != expected_mtp["num_speculative_tokens"]
            or runtime["attention_kv_layout"] != "paged_varlen"):
        raise ValueError(
            "Production Trace sweep Scheduler runtime or KV geometry "
            "disagrees with the pinned contract")
    expected_mtp_runtime = {
        "mtp_num_speculative_tokens": expected_mtp[
            "num_speculative_tokens"],
        "mtp_acceptance_policy": expected_mtp["acceptance_policy"],
        "mtp_fixed_accepted_tokens": expected_mtp[
            "fixed_accepted_tokens"],
        "mtp_acceptance_source_id": expected_mtp[
            "acceptance_source_id"],
        "mtp_acceptance_replay_path": None,
        "mtp_acceptance_replay_sha256": None,
        "mtp_acceptance_replay_artifact_id": None,
        "mtp_acceptance_replay_step_count": None,
        "mtp_acceptance_assumption_sha256": None,
        "mtp_request_stream_sha256": None,
    }
    if expected_mtp["acceptance_policy"] == "disabled":
        expected_mtp_runtime.update({
            "mtp_padded_drafter_batch": False,
            "mtp_draft_model_type": "disabled",
            "mtp_num_mtp_layers": 0,
            "mtp_draft_max_model_len": 0,
        })
    elif expected_mtp["acceptance_policy"] in (
            "fixed_count", "maximum_entropy"):
        expected_mtp_runtime.update({
            "mtp_padded_drafter_batch": True,
            "mtp_draft_model_type": "glm4_moe_mtp",
            "mtp_num_mtp_layers": expected_model.num_mtp_layers,
            "mtp_draft_max_model_len": expected_scheduler["max_model_len"],
        })
        if expected_mtp["acceptance_policy"] == "maximum_entropy":
            assumption = expected_mtp["acceptance_assumption"]
            expected_mtp_runtime.update({
                "mtp_acceptance_assumption_sha256": assumption[
                    "policy_descriptor_sha256"],
                "mtp_request_stream_sha256": assumption[
                    "request_stream_sha256"],
            })
    else:
        raise AssertionError("validated MTP acceptance policy vanished")
    if any(
            runtime[field] != value
            for field, value in expected_mtp_runtime.items()):
        raise ValueError(
            "Production Trace sweep Scheduler MTP runtime disagrees with "
            "command manifest")
    if role in ("colocated", "attention"):
        expected_handoff_mode = "not_applicable"
    elif role in (
            "prefill", "decode", "prefill-attention", "decode-attention"):
        expected_handoff_mode = "metadata_only_private_destination_kv"
    else:
        raise ValueError(
            f"unknown Production Trace Scheduler role: {role!r}")
    if runtime["kv_handoff_mode"] != expected_handoff_mode:
        raise ValueError(
            "Production Trace sweep Scheduler KV handoff mode disagrees "
            "with command manifest")


def explicit_expert_routing_input(
    routing_config: object,
    context: str,
) -> ExpertRoutingInput:
    if not isinstance(routing_config, dict):
        raise ValueError(
            f"{context} command manifest lost explicit expert routing")
    uniform_fields = {"source_kind", "source_id", "root_seed"}
    skew_fields = uniform_fields | {
        "hot_expert_count", "hot_expert_weight"}
    if (set(routing_config) not in (uniform_fields, skew_fields)
            or routing_config["source_kind"] != "explicit_assumption"):
        raise ValueError(
            f"{context} command manifest lost explicit expert routing")
    if set(routing_config) == uniform_fields:
        return ExpertRoutingInput.seeded_assumption(
            routing_config["source_id"],
            routing_config["root_seed"],
        )
    return ExpertRoutingInput.skewed_assumption(
        routing_config["source_id"],
        routing_config["root_seed"],
        hot_expert_count=routing_config["hot_expert_count"],
        hot_expert_weight=routing_config["hot_expert_weight"],
    )


def maximum_entropy_mtp_assumption(
    scenario: dict[str, Any],
) -> MTPAcceptanceAssumption | None:
    """Reconstruct a canonical assumption for independent replay."""

    if scenario["acceptance_policy"] != "maximum_entropy":
        return None
    manifest = scenario["acceptance_assumption"]
    assumption = MTPAcceptanceAssumption(
        manifest["source_id"],
        manifest["proposal_depth"],
        manifest["mean_acceptance_rate"],
        manifest["root_seed"],
        manifest["request_stream_sha256"],
        manifest["sampling_key"] if "sampling_key" in manifest else "step_stream_v1",
    )
    if assumption.manifest() != manifest:
        raise ValueError(
            "Production Trace maximum-entropy MTP assumption is not "
            "canonical")
    return assumption


def validate_tp1_kv_offload_invariants(
    artifact: dict[str, Any],
    model: ModelConfig,
    *,
    expected_hbm_blocks: int,
    context: str,
) -> None:
    block_size_tokens = 16
    block_bytes = (
        block_size_tokens
        * model.kv_bytes_per_token_per_parallel_rank(1, 1)
    )
    modeled = artifact["modeled_kv_offload"]
    final_state = modeled["final_tier_state"]
    tiers = final_state["tiers"]
    if expected_hbm_blocks == 2:
        expected_hbm_description = "2-HBM"
    elif expected_hbm_blocks == 3:
        expected_hbm_description = "3-HBM"
    else:
        raise AssertionError("unsupported formal Host-KV HBM block count")
    if (artifact["schema_version"] not in (7, 8)
            or modeled["schema_version"] != 4
            or modeled["availability"] != "available_modeled_serial"
            or modeled["mode"] != "modeled_host_kv_serial"
            or modeled["ledger_semantics"]
            != "logical_request_blocks_with_single_rank_capacity"
            or modeled["physical_transfer_count"] != 2
            or modeled["scope"]["host_link_utilization"]
            != "available_in_system_resource_manifest_v5_"
            "host_kv_link_utilization_per_rank"
            or final_state["block_size_tokens"] != block_size_tokens
            or final_state["block_bytes"] != block_bytes
            or final_state["configured_policy"] != "lru"
            or final_state["inflight_ready_events"] != []
            or final_state["residency"] != []
            or set(tiers) != {"hbm", "host"}
            or tiers["hbm"]["capacity_bytes"]
            != expected_hbm_blocks * block_bytes
            or tiers["host"]["capacity_bytes"] != block_bytes
            or any(
                tier[field] != 0
                for tier in tiers.values()
                for field in (
                    "used_bytes", "reserved_bytes", "resident_blocks")
            )):
        raise ValueError(
            f"{context} final tier ledger violates the fixed "
            f"{expected_hbm_description}/1-Host block contract")

    expected_routes = (
        ("evict", "hbm", "host"),
        ("load", "host", "hbm"),
    )
    transfers = modeled["transfers"]
    if len(transfers) != len(expected_routes):
        raise ValueError(f"{context} requires exactly two KV transfers")
    for transfer, expected_route in zip(
            transfers, expected_routes, strict=True):
        action, source_tier, destination_tier = expected_route
        intent = transfer["intent"]
        logical = transfer["logical_completion"]
        physical_transfers = transfer["physical_transfers"]
        if len(physical_transfers) != 1:
            raise ValueError(
                f"{context} TP1 transfer requires one physical owner")
        physical = physical_transfers[0]
        completion = physical["completion"]
        interval = physical["relative_interval"]
        expected_resource_id = (
            f"pool:{physical['pool_name']}:rank:{physical['rank']}:pcie")
        if (intent["action"] != action
                or intent["source_tier"] != source_tier
                or intent["destination_tier"] != destination_tier
                or intent["tier"] != "host"
                or intent["token_start"] != 0
                or intent["token_end"] != block_size_tokens
                or intent["bytes"] != block_bytes
                or logical["action"] != action
                or logical["source_tier"] != source_tier
                or logical["destination_tier"] != destination_tier
                or logical["bytes_per_rank"] != block_bytes
                or logical["physical_transfer_count"] != 1
                or logical["link"] != "pcie"
                or completion["link"] != "pcie"
                or completion["bytes"] != block_bytes
                or physical["resource_id"] != expected_resource_id
                or physical["prediction_total_s"] <= 0
                or not math.isclose(
                    interval["end_time_s"] - interval["start_time_s"],
                    physical["prediction_total_s"],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{context} KV transfer geometry, route, or canonical "
                "PCIe prediction is invalid")

    kv_offload_calls = []
    for step in artifact["steps"]:
        for call in step["kernel_plan"]["calls"]:
            if call["kernel_spec"]["kind"] == "KV_OFFLOAD":
                kv_offload_calls.append(call)
    if len(kv_offload_calls) != len(expected_routes):
        raise ValueError(
            f"{context} requires exactly two KV_OFFLOAD KernelSpec calls")
    for call, expected_route in zip(
            kv_offload_calls, expected_routes, strict=True):
        action, _source_tier, _destination_tier = expected_route
        spec = call["kernel_spec"]
        physical = spec["transfer"]
        if (spec["hardware_id"] != "nvidia_a800"
                or call["resource"] != "pcie"
                or physical["algorithm"] != "dma"
                or physical["bytes"] != block_bytes
                or physical["direction"] != action
                or physical["link"] != "pcie"
                or physical["tier"] != "host"):
            raise ValueError(
                f"{context} KV_OFFLOAD KernelSpec physical transfer is "
                "invalid")


def validate_tp2_kv_offload_invariants(
    artifact: dict[str, Any],
    model: ModelConfig,
    *,
    context: str,
) -> None:
    block_size_tokens = 16
    block_bytes = (
        block_size_tokens
        * model.kv_bytes_per_token_per_parallel_rank(2, 1)
    )
    modeled = artifact["modeled_kv_offload"]
    final_state = modeled["final_tier_state"]
    tiers = final_state["tiers"]
    if (artifact["schema_version"] not in (7, 8)
            or modeled["schema_version"] != 4
            or modeled["availability"] != "available_modeled_rank_local"
            or modeled["mode"] != "modeled_host_kv_rank_local_lockstep"
            or modeled["ledger_semantics"]
            != "logical_request_blocks_with_lockstep_per_rank_capacity"
            or modeled["physical_transfer_count"] != 4
            or modeled["scope"]["multi_rank"]
            != "validated_colocated_qwen3_tp2_lockstep"
            or modeled["scope"]["host_link_utilization"]
            != "available_in_system_resource_manifest_v5_"
            "host_kv_link_utilization_per_rank"
            or final_state["block_size_tokens"] != block_size_tokens
            or final_state["block_bytes"] != block_bytes
            or final_state["configured_policy"] != "lru"
            or final_state["inflight_ready_events"] != []
            or final_state["residency"] != []
            or set(tiers) != {"hbm", "host"}
            or tiers["hbm"]["capacity_bytes"] != 2 * block_bytes
            or tiers["host"]["capacity_bytes"] != block_bytes
            or any(
                tier[field] != 0
                for tier in tiers.values()
                for field in (
                    "used_bytes", "reserved_bytes", "resident_blocks")
            )):
        raise ValueError(
            f"{context} rank-local artifact or final tier ledger is invalid")

    expected_routes = (
        ("evict", "hbm", "host"),
        ("load", "host", "hbm"),
    )
    transfers = modeled["transfers"]
    if len(transfers) != len(expected_routes):
        raise ValueError(f"{context} requires exactly two logical transfers")
    steps_by_id = {
        step["step_id"]: step for step in artifact["steps"]
    }
    for transfer, expected_route in zip(
            transfers, expected_routes, strict=True):
        action, source_tier, destination_tier = expected_route
        intent = transfer["intent"]
        logical = transfer["logical_completion"]
        physical_transfers = transfer["physical_transfers"]
        if (intent["action"] != action
                or intent["source_tier"] != source_tier
                or intent["destination_tier"] != destination_tier
                or intent["tier"] != "host"
                or intent["token_start"] != 0
                or intent["token_end"] != block_size_tokens
                or intent["bytes"] != block_bytes
                or logical["action"] != action
                or logical["source_tier"] != source_tier
                or logical["destination_tier"] != destination_tier
                or logical["link"] != "pcie"
                or logical["bytes_per_rank"] != block_bytes
                or logical["physical_transfer_count"] != 2
                or tuple(item["rank"] for item in physical_transfers)
                != (0, 1)):
            raise ValueError(
                f"{context} logical rank-local transfer is invalid")
        step_id = transfer["step_id"]
        if step_id not in steps_by_id:
            raise ValueError(f"{context} transfer references an unknown step")
        step = steps_by_id[step_id]
        calls = {
            call["call_id"]: call for call in step["kernel_plan"]["calls"]
        }
        intervals = {
            interval["call_id"]: interval
            for interval in step["execution_timeline"]["intervals"]
        }
        for physical in physical_transfers:
            rank = physical["rank"]
            call_id = physical["call_id"]
            completion = physical["completion"]
            if call_id not in calls or call_id not in intervals:
                raise ValueError(
                    f"{context} physical transfer lacks a call or interval")
            call = calls[call_id]
            interval = intervals[call_id]
            transfer_config = call["kernel_spec"]["transfer"]
            duration_s = (
                physical["relative_interval"]["end_time_s"]
                - physical["relative_interval"]["start_time_s"]
            )
            if (physical["pool_name"] != "main"
                    or completion["rank"] != rank
                    or completion["pool_name"] != "main"
                    or completion["node_id"] != physical["node_id"]
                    or completion["call_id"] != call_id
                    or completion["bytes"] != block_bytes
                    or completion["link"] != "pcie"
                    or physical["resource_id"]
                    != f"pool:main:rank:{rank}:pcie"
                    or call["rank"] != rank
                    or call["pool_name"] != "main"
                    or call["resource"] != "pcie"
                    or call["kernel_spec"]["kind"] != "KV_OFFLOAD"
                    or transfer_config["algorithm"] != "dma"
                    or transfer_config["bytes"] != block_bytes
                    or transfer_config["direction"] != action
                    or transfer_config["link"] != "pcie"
                    or transfer_config["tier"] != "host"
                    or interval["rank"] != rank
                    or interval["pool_name"] != "main"
                    or interval["resource"] != "pcie"
                    or duration_s <= 0
                    or not math.isclose(
                        duration_s,
                        physical["prediction_total_s"],
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )):
                raise ValueError(
                    f"{context} physical rank transfer is invalid")


def validate_host_kv_system_resources(
    resources: dict[str, Any],
    *,
    pool_name: str,
    ranks: tuple[int, ...],
    hardware_id: str,
    block_bytes: int,
    context: str,
) -> None:
    expected_resource_ids = {
        f"pool:{pool_name}:rank:{rank}:pcie" for rank in ranks
    }
    host_resources = resources["host_kv_link_utilization"]
    if (resources["schema_version"] != 5
            or resources["host_kv_link_activity_basis"]
            != "per_rank_resource_union_of_canonical_kv_offload_physical_"
            "completion_intervals"
            or set(host_resources) != expected_resource_ids):
        raise ValueError(
            f"{context} Host-KV resource coverage is invalid")
    observation_window_s = resources["observation_window"]["duration_s"]
    for rank in ranks:
        resource_id = f"pool:{pool_name}:rank:{rank}:pcie"
        record = host_resources[resource_id]
        if (record["pool_name"] != pool_name
                or record["rank"] != rank
                or record["hardware_id"] != hardware_id
                or record["link"] != "pcie"
                or record["physical_transfer_count"] != 2
                or record["action_counts"] != {"evict": 1, "load": 1}
                or record["bytes_by_action"]
                != {"evict": block_bytes, "load": block_bytes}
                or record["transferred_bytes"] != 2 * block_bytes
                or record["busy_time_s"] <= 0
                or record["idle_time_s"] < 0
                or not math.isclose(
                    record["busy_time_s"] + record["idle_time_s"],
                    observation_window_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    record["utilization"],
                    record["busy_time_s"] / observation_window_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{context} Host-KV resource {resource_id!r} is invalid")


def validate_canonical_production_trace_step_duration(
    step: dict[str, Any],
    canonical_duration_s: float,
    *,
    context: str,
) -> None:
    interval = step["schedule_interval"]
    recorded_duration_s = (
        interval["end_time_s"] - interval["start_time_s"])
    if not math.isclose(
            recorded_duration_s,
            canonical_duration_s,
            rel_tol=1e-12,
            abs_tol=1e-15,
            ):
        raise ValueError(
            f"{context} Scheduler step duration disagrees with the "
            "canonical execution timeline")


def canonical_production_trace_metrics_manifest(
    deployment: DeploymentPlan,
    requests: RequestStream,
    result: SimulationResult,
    slo_contract: Any,
) -> dict[str, Any]:
    metrics = result.metrics
    return json_round_trip({
        "request_throughput_per_s": metrics.request_throughput_per_s,
        "output_token_throughput_per_s": (
            metrics.output_token_throughput_per_s),
        "token_throughput_per_s": metrics.token_throughput_per_s,
        "latency_distributions": latency_distribution_manifest(
            metrics.requests),
        "slo_goodput": slo_goodput_manifest(
            requests,
            metrics,
            result.schedule_trace.end_time_s,
            slo_contract,
        ),
        "system_resources": system_resource_manifest(
            deployment,
            requests,
            result,
        ),
        "critical_path": critical_path_manifest(
            deployment,
            requests,
            result,
        ),
        "requests": {
            request_id: {
                "ttft_s": metric.ttft_s,
                "tpot_s": metric.tpot_s,
                "latency_s": metric.latency_s,
            }
            for request_id, metric in metrics.requests.items()
        },
    })


def backend_step_from_output_manifest(
    raw: dict[str, Any],
) -> BackendStep:
    if not isinstance(raw, dict):
        raise TypeError("serialized BackendStep must be an object")
    if set(raw) != {"schedule", "kv_cache_events", "generated_tokens"}:
        raise ValueError(
            "serialized BackendStep fields disagree with the canonical "
            "contract")
    raw_events = raw["kv_cache_events"]
    if not isinstance(raw_events, list):
        raise TypeError(
            "serialized BackendStep kv_cache_events must be a list")
    generated_tokens = raw["generated_tokens"]
    if not isinstance(generated_tokens, dict):
        raise TypeError(
            "serialized BackendStep generated_tokens must be an object")
    step = BackendStep(
        schedule=schedule_step_from_output_manifest(raw["schedule"]),
        kv_cache_events=tuple(
            kv_cache_event_from_output_manifest(event)
            for event in raw_events
        ),
        generated_tokens=generated_tokens,
    )
    expected = {
        "schedule": json_round_trip(asdict(step.schedule)),
        "kv_cache_events": [
            json_round_trip(asdict(event))
            for event in step.kv_cache_events
        ],
        "generated_tokens": {
            request_id: step.generated_tokens[request_id]
            for request_id in sorted(step.generated_tokens)
        },
    }
    normalized = {**raw, "schedule": json_round_trip(asdict(step.schedule))}
    if normalized != expected:
        raise ValueError("serialized BackendStep derived fields disagree")
    return step


def validate_production_trace_scheduler_binding(
    expected_requests: RequestStream,
    schedule_trace: ScheduleTrace,
    kv_block_size_tokens: int,
    result: dict[str, Any],
    *,
    max_requests: int,
    context: str,
) -> None:
    if (isinstance(max_requests, bool)
            or not isinstance(max_requests, int)
            or max_requests <= 0):
        raise ValueError(
            f"{context} max_requests must be a positive integer")
    requests_by_id = {
        request.request_id: request for request in expected_requests
    }
    first_schedule_ids = set()
    unique_prompt_tokens = {
        request_id: 0 for request_id in requests_by_id
    }
    for entry in schedule_trace.entries:
        schedule = entry.backend_step.schedule
        if len(schedule.requests) > max_requests:
            raise ValueError(
                f"{context} exceeded the configured Scheduler "
                "max_requests")
        if set(schedule.preempted_request_ids) - set(requests_by_id):
            raise ValueError(
                f"{context} preempted an unknown request")
        for scheduled in schedule.requests:
            request_id = scheduled.request_id
            if request_id not in requests_by_id:
                raise ValueError(
                    f"{context} scheduled an unknown request")
            request = requests_by_id[request_id]
            if entry.start_time_s < request.arrival_time_s:
                raise ValueError(
                    f"{context} scheduled a request before its "
                    "Trace arrival")
            if request_id not in first_schedule_ids:
                expected_scheduler_hit = (
                    scheduler_effective_prefix_hit_tokens(
                        request, kv_block_size_tokens))
                if (scheduled.trace_prefix_hit_tokens
                        != request.prefix_hit_tokens
                        or scheduled.scheduler_prefix_hit_tokens
                        != expected_scheduler_hit
                        or scheduled.computed_tokens
                        != expected_scheduler_hit):
                    raise ValueError(
                        f"{context} first Scheduler step "
                        "disagrees with the command Trace request")
                unique_prompt_tokens[request_id] += expected_scheduler_hit
                first_schedule_ids.add(request_id)
            elif (scheduled.trace_prefix_hit_tokens != 0
                  or scheduled.scheduler_prefix_hit_tokens != 0):
                raise ValueError(
                    f"{context} repeated first-step prefix "
                    "metadata")
            recomputed_prompt_tokens = min(
                scheduled.input_tokens,
                scheduled.recomputed_tokens,
            )
            unique_prompt_tokens[request_id] += (
                scheduled.input_tokens - recomputed_prompt_tokens)
    if first_schedule_ids != set(requests_by_id):
        raise ValueError(
            f"{context} Scheduler did not cover every command "
            "Trace request")
    for request_id, request in requests_by_id.items():
        if unique_prompt_tokens[request_id] != request.input_tokens:
            raise ValueError(
                f"{context} Scheduler prompt work disagrees "
                "with the command Trace request")
    try:
        metrics = compute_serving_metrics(
            expected_requests,
            schedule_trace,
        )
    except ValueError as error:
        raise ValueError(
            f"{context} Scheduler output-token ledger "
            "disagrees with the command Trace requests") from error
    actual_metrics = result["metrics"]
    if (not math.isclose(
            schedule_trace.end_time_s,
            result["schedule_end_time_s"],
            rel_tol=1e-12,
            abs_tol=1e-15,
            )
            or not math.isclose(
                metrics.request_throughput_per_s,
                actual_metrics["request_throughput_per_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                metrics.output_token_throughput_per_s,
                actual_metrics["output_token_throughput_per_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                metrics.token_throughput_per_s,
                actual_metrics["token_throughput_per_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )):
        raise ValueError(
            f"{context} metrics disagree with the canonical "
            "Scheduler trace")
    expected_request_metrics = {
        request_id: {
            "ttft_s": metric.ttft_s,
            "tpot_s": metric.tpot_s,
            "latency_s": metric.latency_s,
        }
        for request_id, metric in metrics.requests.items()
    }
    if actual_metrics["requests"] != expected_request_metrics:
        raise ValueError(
            f"{context} request metrics disagree with the "
            "canonical Scheduler trace")


def replay_pinned_parallel_scheduler(
    expected_requests: RequestStream,
    expected_trace: ScheduleTrace,
    scheduler: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    expected_runtime: dict[str, Any],
    *,
    num_kv_blocks: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    enable_expert_parallel: bool,
    expected_mtp: dict[str, Any],
    context: str,
) -> None:
    backend = None
    replay_error = None
    replay_trace = None
    try:
        mtp_assumption = maximum_entropy_mtp_assumption(expected_mtp)
        if (mtp_assumption is not None
                and mtp_assumption.request_stream_sha256
                != request_stream_sha256(expected_requests)):
            raise ValueError(
                f"{context} maximum-entropy MTP assumption disagrees with "
                "the replayed request stream")
        backend = VLLMBackend.from_model(
            BackendConfig(
                pool_name="main",
                role="colocated",
                max_batched_tokens=scheduler["max_batched_tokens"],
                max_requests=scheduler["max_requests"],
                num_kv_blocks=num_kv_blocks,
                kv_block_size_tokens=scheduler[
                    "kv_block_size_tokens"],
                kv_bytes_per_token=(
                    expected_model.
                    kv_bytes_per_token_per_parallel_rank(
                        tensor_parallel_size,
                        pipeline_parallel_size,
                    )),
            scheduler_pipeline_depth=scheduler["scheduler_pipeline_depth"]),
            VLLMSchedulerConfig(
                model=expected_model_path,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                dtype="bfloat16",
                max_model_len=scheduler["max_model_len"],
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                mtp_num_speculative_tokens=expected_mtp[
                    "num_speculative_tokens"],
                mtp_acceptance_policy=expected_mtp[
                    "acceptance_policy"],
                mtp_fixed_accepted_tokens=expected_mtp[
                    "fixed_accepted_tokens"],
                mtp_acceptance_source_id=expected_mtp[
                    "acceptance_source_id"],
                mtp_acceptance_assumption=mtp_assumption,
                request_stream_sha256=(
                    None
                    if mtp_assumption is None
                    else mtp_assumption.request_stream_sha256
                ),
            ),
        )
        if json_round_trip(asdict(backend.runtime_info)) != expected_runtime:
            raise ValueError(
                f"{context} pinned Scheduler replay runtime "
                "disagrees with the recorded runtime")
        expected_entries = expected_trace.entries
        replay_step_index = 0

        def validated_step_duration(actual_step: BackendStep) -> float:
            nonlocal replay_step_index
            if replay_step_index >= len(expected_entries):
                raise ValueError(
                    f"{context} pinned Scheduler replay emitted "
                    "an extra step")
            expected_entry = expected_entries[replay_step_index]
            if actual_step != expected_entry.backend_step:
                raise ValueError(
                    f"{context} BackendStep disagrees with an "
                    "independent pinned Scheduler replay")
            replay_step_index += 1
            return expected_entry.end_time_s - expected_entry.start_time_s

        replay_trace = run_single_pool_schedule(
            backend,
            expected_requests,
            validated_step_duration,
        )
        if replay_step_index != len(expected_entries):
            raise ValueError(
                f"{context} pinned Scheduler replay emitted "
                "fewer steps than the execution artifact")
    except (TypeError, ValueError, RuntimeError, TimeoutError) as error:
        replay_error = error
    finally:
        if backend is not None:
            try:
                backend.close()
            except (TypeError, ValueError, RuntimeError, TimeoutError) as error:
                if replay_error is None:
                    replay_error = error
    if replay_error is not None:
        raise ValueError(
            f"{context} cannot be reproduced by the pinned "
            "Scheduler") from replay_error
    if replay_trace is None:
        raise AssertionError("successful pinned Scheduler replay vanished")
    if len(replay_trace.entries) != len(expected_trace.entries):
        raise AssertionError("validated pinned Scheduler step count changed")
    for actual_entry, expected_entry in zip(
            replay_trace.entries, expected_trace.entries, strict=True):
        if (actual_entry.backend_step != expected_entry.backend_step
                or not math.isclose(
                    actual_entry.start_time_s,
                    expected_entry.start_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    actual_entry.end_time_s,
                    expected_entry.end_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{context} pinned Scheduler replay interval "
                "disagrees with the execution artifact")


def interval_union_duration_from_output(
    intervals: list[tuple[float, float]],
    observation_window_s: float,
    *,
    context: str,
) -> float:
    if not intervals:
        raise ValueError(f"{context} contains no execution intervals")
    ordered = sorted(intervals)
    union_start = None
    union_end = None
    total_s = 0.0
    for start_time_s, end_time_s in ordered:
        if (isinstance(start_time_s, bool)
                or isinstance(end_time_s, bool)
                or not isinstance(start_time_s, (int, float))
                or not isinstance(end_time_s, (int, float))
                or not math.isfinite(start_time_s)
                or not math.isfinite(end_time_s)
                or start_time_s < 0.0
                or end_time_s < start_time_s
                or end_time_s > observation_window_s):
            raise ValueError(f"{context} contains an invalid interval")
        if union_start is None:
            union_start = start_time_s
            union_end = end_time_s
        elif start_time_s <= union_end:
            union_end = max(union_end, end_time_s)
        else:
            total_s += union_end - union_start
            union_start = start_time_s
            union_end = end_time_s
    if union_start is None or union_end is None:
        raise AssertionError("validated non-empty interval set vanished")
    return total_s + union_end - union_start


_LEGACY_SCHEDULED_REQUEST_OUTPUT_FIELDS = frozenset((
    "request_id", "phase", "input_tokens", "output_tokens", "computed_tokens",
    "recomputed_tokens", "prefix_replay_tokens", "resumed_from_preemption",
    "trace_prefix_hit_tokens", "scheduler_prefix_hit_tokens", "prefix_scheduler_delta_tokens",
    "raw_model_output_tokens", "mtp_proposed_tokens", "mtp_verified_tokens", "mtp_accepted_tokens",
    "mtp_rejected_tokens", "mtp_proposer_tokens", "mtp_next_proposed_tokens", "mtp_proposer_mode",
    "mtp_acceptance_source_kind", "mtp_acceptance_source_id", "content_prefix_hit_tokens",
))
_OPTIONAL_SCHEDULED_REQUEST_OUTPUT_FIELDS = frozenset(("data_parallel_rank", "mtp_padding_tokens"))
_SCHEDULED_REQUEST_OUTPUT_FIELDS = _LEGACY_SCHEDULED_REQUEST_OUTPUT_FIELDS | _OPTIONAL_SCHEDULED_REQUEST_OUTPUT_FIELDS


def schedule_step_from_output_manifest(
    raw: dict[str, Any],
) -> ScheduleStep:
    """Read the existing output contract, normalizing legacy DP ownership.

    Prior output manifests have the complete fixed legacy request field set
    and represent rank0. Current manifests additionally serialize the explicit
    data_parallel_rank and native invalid MTP padding (default0). All other missing/extra fields remain invalid; this
    does not relax the separate canonical dataset schema or mutate input.
    """
    if not isinstance(raw, dict):
        raise TypeError("serialized ScheduleStep must be an object")
    schedule_fields = fields(ScheduleStep)
    expected_fields = {field.name for field in schedule_fields}
    if set(raw) != expected_fields:
        raise ValueError(
            "serialized ScheduleStep fields disagree with the canonical "
            "contract")
    raw_requests = raw["requests"]
    if not isinstance(raw_requests, list) or not raw_requests:
        raise TypeError(
            "serialized ScheduleStep requests must be a non-empty list")
    request_fields = fields(ScheduledRequest)
    if {field.name for field in request_fields} != _SCHEDULED_REQUEST_OUTPUT_FIELDS:
        raise RuntimeError("ScheduledRequest output reader requires an explicit field-contract update")
    requests = []
    normalized_requests = []
    for raw_request in raw_requests:
        if not isinstance(raw_request, dict):
            raise ValueError(
                "serialized ScheduledRequest fields disagree with the "
                "canonical contract")
        supplied_fields = set(raw_request)
        if supplied_fields - _OPTIONAL_SCHEDULED_REQUEST_OUTPUT_FIELDS != _LEGACY_SCHEDULED_REQUEST_OUTPUT_FIELDS:
            raise ValueError("serialized ScheduledRequest fields disagree with the canonical contract")
        normalized_request = dict(raw_request)
        # Explicit compatibility normalization at the output schema boundary;
        # supplied invalid values are never replaced by the optional default.
        for name in _OPTIONAL_SCHEDULED_REQUEST_OUTPUT_FIELDS:
            if name not in normalized_request:
                normalized_request[name] = 0
        request = ScheduledRequest(**{
            field.name: normalized_request[field.name]
            for field in request_fields
            if field.init
        })
        if json_round_trip(asdict(request)) != normalized_request:
            raise ValueError(
                "serialized ScheduledRequest derived fields disagree")
        requests.append(request)
        normalized_requests.append(normalized_request)
    schedule_kwargs = {}
    for field in schedule_fields:
        if not field.init:
            continue
        if field.name == "requests":
            schedule_kwargs[field.name] = tuple(requests)
        elif field.name in expected_fields - {"requests"}:
            schedule_kwargs[field.name] = raw[field.name]
        else:
            raise AssertionError(
                f"unknown ScheduleStep constructor field: {field.name!r}")
    schedule = ScheduleStep(**schedule_kwargs)
    normalized = {**raw, "requests": normalized_requests}
    if json_round_trip(asdict(schedule)) != normalized:
        raise ValueError("serialized ScheduleStep derived fields disagree")
    return schedule


def kv_cache_event_from_output_manifest(
    raw: dict[str, Any],
) -> KVCacheEvent:
    if not isinstance(raw, dict):
        raise TypeError("serialized KVCacheEvent must be an object")
    event_fields = fields(KVCacheEvent)
    expected_fields = {field.name for field in event_fields}
    if set(raw) != expected_fields:
        raise ValueError(
            "serialized KVCacheEvent fields disagree with the canonical "
            "contract")
    event = KVCacheEvent(**{
        field.name: raw[field.name]
        for field in event_fields
        if field.init
    })
    if json_round_trip(asdict(event)) != raw:
        raise ValueError("serialized KVCacheEvent derived fields disagree")
    return event


def route_decision_from_output_manifest(
    raw: dict[str, Any],
) -> RouteDecision:
    if not isinstance(raw, dict):
        raise TypeError("serialized RouteDecision must be an object")
    decision_fields = fields(RouteDecision)
    expected_fields = {field.name for field in decision_fields}
    if set(raw) != expected_fields:
        raise ValueError(
            "serialized RouteDecision fields disagree with the canonical "
            "contract")
    decision = RouteDecision(**{
        field.name: raw[field.name]
        for field in decision_fields
        if field.init
    })
    if json_round_trip(asdict(decision)) != raw:
        raise ValueError("serialized RouteDecision derived fields disagree")
    return decision


def json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ))


def validate_production_trace_capacity(
    result: dict[str, Any],
    scheduler: dict[str, Any],
    expected_pools: tuple[str, ...],
    *,
    context: str,
) -> None:
    capacity = result["kv_capacity"]
    if capacity["schema_version"] != 2:
        raise ValueError(
            f"{context} KV capacity source disagrees with command manifest")
    source = capacity["source"]
    if scheduler["capacity_mode"] == "analytical":
        expected_source_mode = "analytical"
        expected_policy_path = str(scheduler["runtime_memory_policy"])
        if (source["mode"] != expected_source_mode
                or source["runtime_memory_policy_path"] != expected_policy_path
                or not isinstance(
                    source["runtime_memory_policy_sha256"], str)):
            raise ValueError(
                f"{context} analytical KV capacity source disagrees with "
                "command manifest")
    elif scheduler["capacity_mode"] == "debug_explicit_block_count":
        expected_source_mode = "debug_explicit_block_count"
        if source != {
                "mode": expected_source_mode,
                "runtime_memory_policy_path": None,
                "runtime_memory_policy_sha256": None,
        }:
            raise ValueError(
                f"{context} debug KV capacity source disagrees with command "
                "manifest")
    else:
        raise ValueError(f"{context} has an unknown KV capacity mode")
    capacity_plans = capacity["scheduler_pools"]
    plans_by_pool = {
        plan["pool_name"]: plan for plan in capacity_plans
    }
    if (len(plans_by_pool) != len(capacity_plans)
            or set(plans_by_pool) != set(expected_pools)):
        raise ValueError(
            f"{context} KV capacity pools disagree with command manifest")
    for pool_name, plan in plans_by_pool.items():
        runtime = result["scheduler_runtime"][pool_name]
        owner_count = len(runtime["scheduler_kv_cache_tensor_owners"])
        page_size_bytes = runtime["scheduler_kv_cache_page_size_bytes"]
        bytes_numerator = owner_count * page_size_bytes
        if (owner_count <= 0
                or isinstance(page_size_bytes, bool)
                or not isinstance(page_size_bytes, int)
                or page_size_bytes <= 0
                or bytes_numerator % scheduler["kv_block_size_tokens"] != 0):
            raise ValueError(f"{context} Scheduler KV geometry is invalid")
        expected_kv_bytes_per_token = (
            bytes_numerator // scheduler["kv_block_size_tokens"])
        expected_block_bytes = (
            scheduler["kv_block_size_tokens"]
            * expected_kv_bytes_per_token)
        if (plan["source_mode"] != expected_source_mode
                or plan["num_kv_blocks"] != runtime["num_kv_blocks"]
                or plan["kv_block_size_tokens"]
                != scheduler["kv_block_size_tokens"]
                or plan["kv_bytes_per_token"]
                != expected_kv_bytes_per_token
                or plan["kv_block_bytes"] != expected_block_bytes):
            raise ValueError(
                f"{context} KV capacity geometry disagrees with command "
                "manifest")
        if expected_source_mode == "analytical":
            if (plan["runtime_memory_policy"] is None
                    or not plan["rank_budgets"]
                    or plan["native_vllm_allocation"] is None):
                raise ValueError(
                    f"{context} analytical KV capacity evidence is incomplete")
            utilization = plan["runtime_memory_policy"][
                "gpu_memory_utilization"]
            expected_utilization = (
                utilization["numerator"] / utilization["denominator"])
        elif (plan["runtime_memory_policy"] is not None
              or plan["rank_budgets"]
              or plan["native_vllm_allocation"] is not None
              or plan["num_kv_blocks"] != scheduler["num_kv_blocks"]):
            raise ValueError(
                f"{context} debug KV capacity carries analytical evidence")
        else:
            expected_utilization = 0.9
        if (runtime["requested_gpu_memory_utilization"]
                != expected_utilization
                or runtime["effective_gpu_memory_utilization"]
                != expected_utilization):
            raise ValueError(
                f"{context} Scheduler memory utilization disagrees with KV "
                "capacity policy")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(
                f"duplicate Production Trace sweep JSON field: {key!r}")
        payload[key] = value
    return payload


def reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON numeric constant: {value!r}")
