"""Production trace disaggregated validation; extracted without changing the established contracts."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from typing import Any
from evaluation.serving.simulation.execution_artifact import execution_artifact_manifest, prefix_alignment_manifest
from src.kernel.kernel_spec import KernelKind
from src.serving.backend.vllm_scheduler_process import vllm_scheduler_runtime_sha256
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.deployment.af_disagg import build_af_disagg_plan
from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, RouteConfig, RouterConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import deployment_plan_manifest
from src.serving.deployment.pd_disagg import build_pd_disagg_plan
from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager
from src.serving.kv_cache.kv_lifecycle_replay import load_kv_lifecycle_replay
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestStream, request_stream_sha256
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.simulation import MultiSimulation, SimulationConfig
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs
from src.serving.workload.workload_graph import WorkloadKind

from evaluation.serving.experiments.production_trace_contracts import (
    canonical_production_trace_metrics_manifest,
    expected_performance_timing_status,
    json_round_trip,
    load_expected_hardware,
    route_decision_from_output_manifest,
    validate_host_kv_system_resources,
    validate_production_trace_capacity,
    validate_production_trace_scheduler_runtime,
    validate_tp1_kv_offload_invariants,
)


def artifact_af_payload_names_by_transfer_id(
    result: dict[str, Any],
    *,
    context: str,
) -> dict[str, str]:
    """Validate ordered AF tensor pairs already carried by artifact v5."""

    payload_names = {}
    for step in result["execution_artifact"]["steps"]:
        trace_ids_by_template = {}
        workload_graph = step["workload_graph"]
        if "block_repeats" in workload_graph:
            for repeat in workload_graph["block_repeats"]:
                for trace_node in repeat["trace_node_ids"]:
                    trace_ids_by_template[
                        trace_node["template_node_id"]] = tuple(
                            trace_node["logical_node_ids"])
        nodes = tuple(
            node for node in workload_graph["nodes"]
            if node["kind"] == WorkloadKind.AF_ACTIVATION_TRANSFER.value
        )
        if len(nodes) % 2 != 0:
            raise ValueError(
                f"{context} AF artifact must contain complete tensor pairs")
        for pair_index in range(0, len(nodes), 2):
            pair = nodes[pair_index:pair_index + 2]
            tensors = []
            logical_ids_by_node = []
            for node in pair:
                node_id = node["node_id"]
                attributes = node["attributes"]
                if "payload_tensor" not in attributes:
                    raise ValueError(
                        f"{context} AF artifact lacks payload_tensor")
                payload = attributes["payload_tensor"]
                if set(payload) != {"name", "shape", "dtype", "bytes"}:
                    raise ValueError(
                        f"{context} AF artifact payload_tensor fields are "
                        "invalid")
                tensors.append(payload)
                logical_ids_by_node.append(
                    trace_ids_by_template[node_id]
                    if node_id in trace_ids_by_template else (node_id,))
            if tuple(tensor["name"] for tensor in tensors) != (
                    "hidden_states", "residual"):
                raise ValueError(
                    f"{context} AF artifact tensor pair must be ordered "
                    "hidden_states then residual")
            if any(
                    tensors[0][field] != tensors[1][field]
                    for field in ("shape", "dtype", "bytes")):
                raise ValueError(
                    f"{context} AF artifact tensor pair geometry disagrees")
            if len(logical_ids_by_node[0]) != len(logical_ids_by_node[1]):
                raise ValueError(
                    f"{context} AF artifact tensor pair repeat counts disagree")
            for logical_ids, tensor in zip(
                    logical_ids_by_node, tensors, strict=True):
                for logical_id in logical_ids:
                    if logical_id in payload_names:
                        raise ValueError(
                            f"{context} AF artifact repeats transfer node_id")
                    payload_names[logical_id] = tensor["name"]
    activation_transfer_ids = {
        transfer["transfer_id"]
        for transfer in result["transfers"]
        if transfer["payload"] == "activation"
    }
    if set(payload_names) != activation_transfer_ids:
        raise ValueError(
            f"{context} AF artifact nodes and activation Trace disagree")
    return payload_names


def validate_af_production_trace_point(
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
        context = "AF Production Trace"
    else:
        context = "AF+Host-KV Production Trace"
        try:
            kv_lifecycle_replay = load_kv_lifecycle_replay(
                replay_identity["path"])
        except (OSError, TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"{context} replay cannot be independently reloaded") \
                from error
        if (kv_lifecycle_replay.source_sha256
                != replay_identity["source_sha256"]
                or kv_lifecycle_replay.artifact_id
                != replay_identity["artifact_id"]):
            raise ValueError(f"{context} replay identity changed")
        expected_replay_manifest = json_round_trip(
            kv_lifecycle_replay.manifest())
        if (result["kv_lifecycle_replay"] != expected_replay_manifest
                or kv_lifecycle_replay.context.request_stream_sha256
                != request_stream_sha256(expected_requests)):
            raise ValueError(
                f"{context} replay disagrees with the command manifest or "
                "Trace")
    af_disagg = expected_simulation["af_disagg"]
    if (af_disagg is None
            or expected_simulation["pd_disagg"] is not None):
        raise ValueError(f"{context} command deployment config is invalid")
    if (expected_simulation["expert_routing"] is not None
            or expected_simulation["mtp"]["acceptance_policy"]
            != "disabled"):
        raise ValueError(f"{context} command enabled an unsupported feature")
    scheduler_runtime = result["scheduler_runtime"]
    if set(scheduler_runtime) != {"attention"}:
        raise ValueError(
            f"{context} must contain only the Attention Scheduler runtime")
    scheduler = expected_simulation["scheduler"]
    validate_production_trace_scheduler_runtime(
        scheduler_runtime["attention"],
        scheduler,
        expected_model,
        expected_simulation["mtp"],
        role="attention",
        tensor_parallel_field="prefill_tp_size",
    )
    validate_production_trace_capacity(
        result,
        scheduler,
        ("attention",),
        context=context,
    )
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["kv_lifecycle_replay"] != (
                None
                if kv_lifecycle_replay is None
                else json_round_trip(kv_lifecycle_replay.manifest())
            )
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} routing or lifecycle state is invalid")
    if (kv_lifecycle_replay is None
            and result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(f"{context} unexpectedly enabled KV offload")

    transfer_config = af_disagg["activation_transfer"]
    activation_bytes = expected_model.hidden_size * expected_model.dtype_bytes
    request_id = expected_requests.requests[0].request_id
    route_decisions = result["route_decisions"]
    if len(route_decisions) != 1:
        raise ValueError(
            f"{context} must contain one ingress decision for Attention")
    decision = route_decisions[0]
    if (decision["deployment_id"] != result["deployment_id"]
            or decision["request_id"] != request_id
            or decision["stage"] != "ingress_to_colocated"
            or decision["policy"] != "least_queued_tokens"
            or decision["source_engine_id"] is not None
            or decision["selected_engine_id"] != "attention"
            or decision["candidate_queued_tokens"] != [["attention", 0]]):
        raise ValueError(
            f"{context} ingress route must select the sole Attention "
            "Scheduler from zero visible queued tokens")
    tensor_parallel_size = scheduler["prefill_tp_size"]
    if (tensor_parallel_size not in (1, 2)
            or af_disagg["ffn_tp_size"] != tensor_parallel_size):
        raise ValueError(f"{context} Attention and FFN TP sizes are invalid")
    attention_ranks = tuple(range(tensor_parallel_size))
    ffn_ranks = tuple(range(
        tensor_parallel_size, 2 * tensor_parallel_size))
    steps = result["execution_artifact"]["steps"]
    if tensor_parallel_size == 1:
        request = expected_requests.requests[0]
        expected_scheduled_tokens = (
            request.input_tokens - request.prefix_hit_tokens,
            1,
        )
    elif tensor_parallel_size == 2:
        expected_scheduled_tokens = (2,)
    else:
        raise AssertionError("validated AF TP size vanished")
    actual_scheduled_tokens = tuple(
        sum(
            request["input_tokens"] + request["output_tokens"]
            for request in step["backend_step"]["schedule"]["requests"]
        )
        for step in steps
    )
    if (tuple(step["pool_name"] for step in steps)
            != tuple("attention" for _ in expected_scheduled_tokens)
            or actual_scheduled_tokens != expected_scheduled_tokens):
        raise ValueError(
            f"{context} Scheduler steps disagree with the validated Trace")

    payload_names_by_transfer_id = artifact_af_payload_names_by_transfer_id(
        result, context=context)
    expected_transfer_identities = []
    for token_count in expected_scheduled_tokens:
        byte_count = token_count * activation_bytes
        for _layer_index in range(expected_model.num_hidden_layers):
            for attention_rank, ffn_rank in zip(
                    attention_ranks, ffn_ranks, strict=True):
                for tensor_name in ("hidden_states", "residual"):
                    expected_transfer_identities.append((
                        (request_id,), "attention", "ffn", byte_count,
                        attention_rank, ffn_rank, tensor_name,
                    ))
            for attention_rank, ffn_rank in zip(
                    attention_ranks, ffn_ranks, strict=True):
                for tensor_name in ("hidden_states", "residual"):
                    expected_transfer_identities.append((
                        (request_id,), "ffn", "attention", byte_count,
                        ffn_rank, attention_rank, tensor_name,
                    ))
    transfers = result["transfers"]
    if (result["num_steps"] != len(expected_scheduled_tokens)
            or result["num_transfers"] != len(expected_transfer_identities)
            or len(transfers) != len(expected_transfer_identities)):
        raise ValueError(
            f"{context} Scheduler step or activation transfer count is "
            "invalid")
    expected_transfer_records = []
    for expected in expected_transfer_identities:
        (request_ids, source_pool, destination_pool, byte_count, source_rank,
         destination_rank, tensor_name) = expected
        expected_reserved_resources = (
            "communication:endpoint:nvidia_a800:"
            f"{source_rank}:{transfer_config['link']}",
            "communication:endpoint:nvidia_a800:"
            f"{destination_rank}:{transfer_config['link']}",
            "communication:fabric:"
            f"{transfer_config['resource_id']}",
        )
        expected_transfer_records.append((
            request_ids,
            source_pool,
            destination_pool,
            byte_count,
            expected_reserved_resources,
            tensor_name,
        ))
    actual_transfer_records = [
        (
            tuple(transfer["request_ids"]),
            transfer["source_pool_id"],
            transfer["destination_pool_id"],
            transfer["byte_count"],
            tuple(transfer["reserved_resource_ids"]),
            payload_names_by_transfer_id[transfer["transfer_id"]],
        )
        for transfer in transfers
    ]
    if sorted(actual_transfer_records) != sorted(expected_transfer_records):
        raise ValueError(
            f"{context} activation transfer disagrees with the canonical "
            "Attention/FFN lane, bytes, or resource reservation")
    for transfer in transfers:
        if (transfer["payload"] != transfer_config["payload"]
                or transfer["resource_id"] != transfer_config["resource_id"]
                or transfer["link"] != transfer_config["link"]
                or transfer["end_time_s"] <= transfer["start_time_s"]):
            raise ValueError(
                f"{context} activation transfer disagrees with the canonical "
                "Attention/FFN route, bytes, or resource reservation")

    communication_calls = tuple(
        call
        for step in result["execution_artifact"]["steps"]
        for call in step["kernel_plan"]["calls"]
        if call["kernel_spec"]["kind"] == KernelKind.COMMUNICATION.value
    )
    lane_calls = tuple(
        call for call in communication_calls
        if call["kernel_spec"]["transfer"]["operation"] == "send_recv"
    )
    collective_calls = tuple(
        call for call in communication_calls
        if call["kernel_spec"]["transfer"]["operation"]
        in ("all_reduce", "all_gather")
    )
    expected_collective_count = (
        0 if tensor_parallel_size == 1
        else 6 * len(expected_scheduled_tokens)
    )
    if (len(lane_calls) != len(transfers)
            or len(collective_calls) != expected_collective_count
            or len(communication_calls)
            != len(transfers) + expected_collective_count):
        raise ValueError(
            f"{context} COMMUNICATION KernelSpec count is invalid")
    lane_calls_by_id = {}
    for call in lane_calls:
        call_id = call["call_id"]
        if call_id in lane_calls_by_id:
            raise ValueError(f"{context} repeated lane transfer call_id")
        lane_calls_by_id[call_id] = call
    if set(lane_calls_by_id) != {
            transfer["transfer_id"] for transfer in transfers}:
        raise ValueError(
            f"{context} lane COMMUNICATION calls disagree with transfers")
    ranks_by_reserved_resources = {
        record[4]: (expected[4], expected[5])
        for record, expected in zip(
            expected_transfer_records,
            expected_transfer_identities,
            strict=True,
        )
    }
    for transfer in transfers:
        call = lane_calls_by_id[transfer["transfer_id"]]
        source_pool = transfer["source_pool_id"]
        byte_count = transfer["byte_count"]
        source_rank, destination_rank = ranks_by_reserved_resources[
            tuple(transfer["reserved_resource_ids"])]
        physical = call["kernel_spec"]["transfer"]
        if (call["pool_name"] != source_pool
                or call["rank"] != source_rank
                or call["resource"] != transfer_config["resource_id"]
                or physical["algorithm"] != transfer_config["algorithm"]
                or physical["bytes"] != byte_count
                or physical["bytes_semantics"] != "payload"
                or physical["operation"] != transfer_config["algorithm"]
                or physical["link"] != transfer_config["link"]
                or physical["topology"] != transfer_config["topology"]
                or physical["backend"] != transfer_config["backend"]
                or physical["resource_id"] != transfer_config["resource_id"]
                or physical["source_rank"] != source_rank
                or physical["destination_rank"] != destination_rank
                or tuple(physical["participants"])
                != (source_rank, destination_rank)):
            raise ValueError(
                f"{context} COMMUNICATION KernelSpec disagrees with the "
                "canonical physical activation transfer")

    resources = result["metrics"]["system_resources"]
    if set(resources["worker_pool_utilization"]) != {"attention", "ffn"}:
        raise ValueError(f"{context} worker utilization pools are invalid")
    if set(resources["scheduler_queue"]["by_pool"]) != {"attention"}:
        raise ValueError(
            f"{context} must expose only the Attention Scheduler queue")
    router = af_disagg["router"]
    router_queue = resources["router_queue"]
    if (router_queue["resource_id"] != router["resource_id"]
            or router_queue["configured_dispatch_service_time_s"]
            != router["dispatch_service_time_s"]):
        raise ValueError(f"{context} Router metrics disagree with the manifest")
    link = resources["link_utilization"]
    if set(link) != {transfer_config["resource_id"]}:
        raise ValueError(f"{context} link utilization resources are invalid")
    link_metrics = link[transfer_config["resource_id"]]
    expected_transferred_bytes = sum(
        transfer["byte_count"] for transfer in transfers)
    if (link_metrics["transfer_count"] != len(transfers)
            or link_metrics["transferred_bytes"]
            != expected_transferred_bytes):
        raise ValueError(f"{context} link utilization totals are invalid")
    communication = resources["communication_resource_utilization"]
    expected_communication_resources = {transfer_config["resource_id"]}
    if tensor_parallel_size == 2:
        expected_communication_resources.update({
            "nvlink:attention:tensor-fabric",
            "nvlink:ffn:tensor-fabric",
        })
    elif tensor_parallel_size == 1:
        pass
    else:
        raise AssertionError("validated AF TP size vanished")
    if set(communication) != expected_communication_resources:
        raise ValueError(
            f"{context} communication utilization resources are invalid")
    communication_metrics = communication[transfer_config["resource_id"]]
    if (communication_metrics["timing_status"]
            != "projected_analytical_roofline"
            or communication_metrics["operation_counts"]
            != {"send_recv": len(transfers)}):
        raise ValueError(
            f"{context} communication timing status or operation counts are "
            "invalid")
    if tensor_parallel_size == 2:
        for resource_id in (
                "nvlink:attention:tensor-fabric",
                "nvlink:ffn:tensor-fabric"):
            fabric = communication[resource_id]
            expected_operation_counts = {}
            for call in collective_calls:
                physical = call["kernel_spec"]["transfer"]
                if physical["resource_id"] != resource_id:
                    continue
                operation = physical["operation"]
                if operation in expected_operation_counts:
                    expected_operation_counts[operation] += 1
                else:
                    expected_operation_counts[operation] = 1
            if (fabric["timing_status"]
                    != "projected_analytical_roofline"
                    or fabric["operation_counts"]
                    != expected_operation_counts):
                raise ValueError(
                    f"{context} tensor fabric metrics are invalid")

    validate_af_canonical_rematerialization(
        expected_simulation,
        expected_model,
        expected_model_path,
        result,
        expected_requests,
        expected_slo_contract,
        kv_lifecycle_replay,
        context=context,
    )


def validate_af_canonical_rematerialization(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
    kv_lifecycle_replay: Any,
    *,
    context: str,
) -> None:
    af_disagg = expected_simulation["af_disagg"]
    if af_disagg is None:
        raise AssertionError("validated AF configuration vanished")
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    transfer = af_disagg["activation_transfer"]
    router = af_disagg["router"]
    scheduler = expected_simulation["scheduler"]
    tensor_parallel_size = scheduler["prefill_tp_size"]
    if (tensor_parallel_size not in (1, 2)
            or af_disagg["ffn_tp_size"] != tensor_parallel_size):
        raise AssertionError("validated AF TP configuration vanished")

    def collectives(pool_name: str) -> tuple[CollectiveConfig, ...]:
        if tensor_parallel_size == 1:
            return ()
        if tensor_parallel_size == 2:
            resource_id = f"nvlink:{pool_name}:tensor-fabric"
            return (
                CollectiveConfig(
                    "all_reduce", "ring", "nvlink", "ring", resource_id,
                    "analytical_roofline_v1",
                ),
                CollectiveConfig(
                    "all_gather", "ring", "nvlink", "ring", resource_id,
                    "analytical_roofline_v1",
                ),
            )
        raise AssertionError("validated AF TP size vanished")

    attention_ranks = tuple(range(tensor_parallel_size))
    ffn_ranks = tuple(range(
        tensor_parallel_size, 2 * tensor_parallel_size))
    deployment = build_af_disagg_plan(
        DeploymentConfig(
            "af_disagg",
            (
                WorkerPoolConfig(
                    "attention",
                    "attention",
                    hardware_id,
                    ParallelConfig(
                        tensor_parallel_size=tensor_parallel_size),
                    attention_ranks,
                    kv_cache=(
                        None
                        if kv_lifecycle_replay is None
                        else kv_lifecycle_replay.cache_config
                    ),
                    collectives=collectives("attention"),
                ),
                WorkerPoolConfig(
                    "ffn",
                    "ffn",
                    hardware_id,
                    ParallelConfig(
                        tensor_parallel_size=af_disagg["ffn_tp_size"]),
                    ffn_ranks,
                    collectives=collectives("ffn"),
                ),
            ),
            (
                RouteConfig(
                    "attention",
                    "ffn",
                    transfer["payload"],
                    transfer["algorithm"],
                    transfer["link"],
                    transfer["topology"],
                    resource_id=transfer["resource_id"],
                    backend=transfer["backend"],
                ),
                RouteConfig(
                    "ffn",
                    "attention",
                    transfer["payload"],
                    transfer["algorithm"],
                    transfer["link"],
                    transfer["topology"],
                    resource_id=transfer["resource_id"],
                    backend=transfer["backend"],
                ),
            ),
            RouterConfig(
                router["resource_id"],
                router["dispatch_service_time_s"],
            ),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} deployment disagrees with canonical rematerialization")

    hardware = load_expected_hardware(
        expected_simulation, hardware_id, context)
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(
            tensor_parallel_size, 1))
    try:
        with ExitStack() as construction_lifecycle:
            backend = VLLMBackend.from_model(
                BackendConfig(
                    "attention",
                    "attention",
                    scheduler["max_batched_tokens"],
                    scheduler["max_requests"],
                    scheduler_runtime["attention"]["num_kv_blocks"],
                    kv_block_size_tokens=scheduler["kv_block_size_tokens"],
                    kv_bytes_per_token=kv_bytes_per_token,
                scheduler_pipeline_depth=scheduler["scheduler_pipeline_depth"]),
                VLLMSchedulerConfig(
                    expected_model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=1,
                    enable_expert_parallel=False,
                    dtype="bfloat16",
                    max_model_len=scheduler["max_model_len"],
                    enable_chunked_prefill=True,
                    enable_prefix_caching=False,
                    mtp_num_speculative_tokens=0,
                    mtp_acceptance_policy="disabled",
                    mtp_fixed_accepted_tokens=0,
                    mtp_acceptance_source_id="disabled",
                ),
            )
            construction_lifecycle.callback(backend.close)
            expected_runtime = result["scheduler_runtime"]["attention"]
            if json_round_trip(asdict(backend.runtime_info)) != expected_runtime:
                raise ValueError(
                    f"{context} pinned Attention Scheduler runtime disagrees "
                    "with the recorded runtime")
            runtime_sha256 = (
                None
                if kv_lifecycle_replay is None
                else vllm_scheduler_runtime_sha256(backend.runtime_info)
            )
            canonical_result = MultiSimulation(
                SimulationConfig(
                    expected_model,
                    deployment,
                    expected_requests,
                    expected_simulation["launch_policy"],
                    expected_simulation["prediction_method"],
                    runtime_inputs,
                ),
                {"attention": backend},
                {hardware_id: hardware},
                kv_cache_tier_manager=(
                    None
                    if kv_lifecycle_replay is None
                    else KVCacheTierManager(
                        kv_lifecycle_replay.cache_config)
                ),
                kv_lifecycle_replay=kv_lifecycle_replay,
                scheduler_runtime_sha256=runtime_sha256,
            ).run()
            construction_lifecycle.pop_all()
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError,
            TimeoutError) as error:
        raise ValueError(
            f"{context} cannot be canonically reproduced from the command "
            "Trace, lifecycle replay, and pinned Attention Scheduler") \
            from error

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
    if result["execution_artifact"] != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    if kv_lifecycle_replay is not None:
        modeled_offload = expected_artifact["modeled_kv_offload"]
        if (modeled_offload["availability"] != "available_modeled_serial"
                or modeled_offload["intent_source"]["kind"]
                != "scheduler_event_replay"
                or modeled_offload["intent_source"]["replay"][
                    "source_sha256"] != kv_lifecycle_replay.source_sha256):
            raise ValueError(
                f"{context} modeled lifecycle or final tier state is invalid")
        validate_tp1_kv_offload_invariants(
            expected_artifact,
            expected_model,
            expected_hbm_blocks=scheduler["num_kv_blocks"],
            context=context,
        )
    expected_route_decisions = expected_artifact["schedule_trace"][
        "route_decisions"]
    expected_transfers = expected_artifact["schedule_trace"]["transfers"]
    expected_kernel_calls = sum(
        step.kernel_plan.logical_call_count
        for step in canonical_result.steps)
    if (result["route_decisions"] != expected_route_decisions
            or result["transfers"] != expected_transfers
            or result["num_steps"] != len(canonical_result.steps)
            or result["num_transfers"]
            != len(canonical_result.schedule_trace.transfers)
            or result["num_kernel_calls"] != expected_kernel_calls
            or result["schedule_end_time_s"]
            != canonical_result.schedule_trace.end_time_s):
        raise ValueError(
            f"{context} counts or ordered activation trace disagree with "
            "canonical rematerialization")
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
    if kv_lifecycle_replay is not None:
        validate_host_kv_system_resources(
            expected_metrics["system_resources"],
            pool_name="attention",
            ranks=(0,),
            hardware_id=hardware_id,
            block_bytes=(
                16
                * expected_model.kv_bytes_per_token_per_parallel_rank(
                    tensor_parallel_size, 1)
            ),
            context=context,
        )
        request_id = expected_requests.requests[0].request_id
        critical_path_contributions = expected_metrics["critical_path"][
            "requests"][request_id]["e2e"]["contribution_s_by_category"]
        if ("kernel:kv_offload" not in critical_path_contributions
                or critical_path_contributions["kernel:kv_offload"] <= 0):
            raise ValueError(
                f"{context} critical path lacks KV offload contribution")


def validate_pd_af_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    context = "PD+AF Production Trace"
    pd_disagg = expected_simulation["pd_disagg"]
    af_disagg = expected_simulation["af_disagg"]
    if pd_disagg is None or af_disagg is None:
        raise ValueError(f"{context} command deployment config is invalid")
    instance_pair = (
        pd_disagg["num_prefill_instances"],
        pd_disagg["num_decode_instances"],
    )
    if (instance_pair not in ((1, 1), (1, 2), (2, 1), (2, 2))
            or expected_simulation["kv_lifecycle_replay"] is not None
            or expected_simulation["expert_routing"] is not None
            or expected_simulation["mtp"]["acceptance_policy"]
            != "disabled"):
        raise ValueError(f"{context} command enabled an unsupported feature")

    scheduler_runtime = result["scheduler_runtime"]
    if pd_disagg["num_prefill_instances"] == 1:
        prefill_pools = ("prefill-attention",)
    elif pd_disagg["num_prefill_instances"] == 2:
        prefill_pools = (
            "prefill-attention-0", "prefill-attention-1")
    else:
        raise AssertionError("validated PD+AF prefill instance count vanished")
    if pd_disagg["num_decode_instances"] == 1:
        decode_pools = ("decode-attention",)
    elif pd_disagg["num_decode_instances"] == 2:
        decode_pools = ("decode-attention-0", "decode-attention-1")
    else:
        raise AssertionError("validated PD+AF decode instance count vanished")
    scheduler_pools = (*prefill_pools, *decode_pools)
    if set(scheduler_runtime) != set(scheduler_pools):
        raise ValueError(
            f"{context} must contain exactly the P and D Attention "
            "Scheduler runtimes")
    scheduler = expected_simulation["scheduler"]
    for prefill_pool in prefill_pools:
        validate_production_trace_scheduler_runtime(
            scheduler_runtime[prefill_pool],
            scheduler,
            expected_model,
            expected_simulation["mtp"],
            role="prefill-attention",
            tensor_parallel_field="prefill_tp_size",
        )
    for decode_pool in decode_pools:
        validate_production_trace_scheduler_runtime(
            scheduler_runtime[decode_pool],
            scheduler,
            expected_model,
            expected_simulation["mtp"],
            role="decode-attention",
            tensor_parallel_field="decode_tp_size",
        )
    validate_production_trace_capacity(
        result,
        scheduler,
        scheduler_pools,
        context=context,
    )
    if (result["routing_policy"] != pd_disagg["routing_policy"]
            or result["kv_handoff_mode"] != pd_disagg["kv_handoff_mode"]
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["kv_lifecycle_replay"] is not None
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} routing or lifecycle state is invalid")
    if result["execution_artifact"]["modeled_kv_offload"] != {
            "schema_version": 2,
            "availability": "not_enabled",
            "intent_source": None,
            "transfers": [],
            "final_tier_state": None,
    }:
        raise ValueError(f"{context} unexpectedly enabled KV offload")

    request_ids = tuple(
        request.request_id for request in expected_requests.requests)
    route_decisions = result["route_decisions"]
    if instance_pair == (1, 1):
        request_prefill_pools = (prefill_pools[0],)
        request_decode_pools = (decode_pools[0],)
        expected_decisions = (
            (
                request_ids[0], "ingress_to_prefill", None,
                prefill_pools[0], [[prefill_pools[0], 0]],
            ),
            (
                request_ids[0], "prefill_to_decode", prefill_pools[0],
                decode_pools[0], [[decode_pools[0], 0]],
            ),
        )
    elif instance_pair == (1, 2):
        request_prefill_pools = tuple(
            prefill_pools[0] for _request_id in request_ids)
        request_decode_pools = (
            decode_pools[0], decode_pools[1], decode_pools[0])
        expected_decisions = tuple(
            (
                request_id, "ingress_to_prefill", None, prefill_pools[0],
                [[prefill_pools[0], index]],
            )
            for index, request_id in enumerate(request_ids)
        ) + tuple(
            (
                request_id,
                "prefill_to_decode",
                prefill_pools[0],
                destination,
                [
                    [decode_pools[0], queued_tokens[0]],
                    [decode_pools[1], queued_tokens[1]],
                ],
            )
            for request_id, destination, queued_tokens in zip(
                request_ids,
                request_decode_pools,
                ((0, 0), (1, 0), (1, 1)),
                strict=True,
            )
        )
    elif instance_pair == (2, 2):
        request_prefill_pools = (
            prefill_pools[0], prefill_pools[1])
        request_decode_pools = (decode_pools[0], decode_pools[1])
        expected_decisions = (
            (
                request_ids[0], "ingress_to_prefill", None,
                prefill_pools[0],
                [[prefill_pools[0], 0], [prefill_pools[1], 0]],
            ),
            (
                request_ids[1], "ingress_to_prefill", None,
                prefill_pools[1],
                [[prefill_pools[0], 4], [prefill_pools[1], 0]],
            ),
            (
                request_ids[0], "prefill_to_decode", prefill_pools[0],
                decode_pools[0],
                [[decode_pools[0], 0], [decode_pools[1], 0]],
            ),
            (
                request_ids[1], "prefill_to_decode", prefill_pools[1],
                decode_pools[1],
                [[decode_pools[0], 1], [decode_pools[1], 0]],
            ),
        )
    elif instance_pair == (2, 1):
        request_prefill_pools = (
            prefill_pools[0], prefill_pools[1])
        request_decode_pools = (decode_pools[0], decode_pools[0])
        expected_decisions = (
            (
                request_ids[0], "ingress_to_prefill", None,
                prefill_pools[0],
                [[prefill_pools[0], 0], [prefill_pools[1], 0]],
            ),
            (
                request_ids[1], "ingress_to_prefill", None,
                prefill_pools[1],
                [[prefill_pools[0], 4], [prefill_pools[1], 0]],
            ),
            (
                request_ids[0], "prefill_to_decode", prefill_pools[0],
                decode_pools[0], [[decode_pools[0], 0]],
            ),
            (
                request_ids[1], "prefill_to_decode", prefill_pools[1],
                decode_pools[0], [[decode_pools[0], 1]],
            ),
        )
    else:
        raise AssertionError("validated PD+AF instance pair vanished")
    if len(route_decisions) != len(expected_decisions):
        raise ValueError(
            f"{context} route decision count disagrees with the validated "
            "instance and request slice")
    for decision, expected in zip(
            route_decisions, expected_decisions, strict=True):
        request_id, stage, source, destination, candidates = expected
        if (decision["deployment_id"] != result["deployment_id"]
                or decision["request_id"] != request_id
                or decision["stage"] != stage
                or decision["policy"] != pd_disagg["routing_policy"]
                or decision["source_engine_id"] != source
                or decision["selected_engine_id"] != destination
                or decision["candidate_queued_tokens"] != candidates):
            raise ValueError(
                f"{context} route decision disagrees with the canonical "
                "P/D topology and least-queued-tokens snapshot")

    activation = af_disagg["activation_transfer"]
    kv_transfer = pd_disagg["kv_transfer"]
    activation_bytes = expected_model.hidden_size * expected_model.dtype_bytes
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 1))
    if len(prefill_pools) == 1:
        prefill_pairs = ((prefill_pools[0], "prefill-ffn"),)
    elif len(prefill_pools) == 2:
        prefill_pairs = (
            (prefill_pools[0], "prefill-ffn-0"),
            (prefill_pools[1], "prefill-ffn-1"),
        )
    else:
        raise AssertionError("validated PD+AF prefill pool set vanished")
    if len(decode_pools) == 1:
        decode_pairs = ((decode_pools[0], "decode-ffn"),)
    elif len(decode_pools) == 2:
        decode_pairs = (
            (decode_pools[0], "decode-ffn-0"),
            (decode_pools[1], "decode-ffn-1"),
        )
    else:
        raise AssertionError("validated PD+AF decode pool set vanished")
    pool_pairs = (*prefill_pairs, *decode_pairs)
    ffn_by_attention = {
        attention_pool: ffn_pool
        for attention_pool, ffn_pool in pool_pairs
    }
    pool_ranks = {
        pool_name: 2 * pair_index + rank_offset
        for pair_index, pair in enumerate(pool_pairs)
        for rank_offset, pool_name in enumerate(pair)
    }
    payload_names_by_transfer_id = artifact_af_payload_names_by_transfer_id(
        result, context=context)
    expected_transfer_identities = []
    for request, prefill_pool, decode_pool in zip(
            expected_requests.requests,
            request_prefill_pools,
            request_decode_pools,
            strict=True):
        prefill_ffn_pool = ffn_by_attention[prefill_pool]
        decode_ffn_pool = ffn_by_attention[decode_pool]
        prefill_bytes = request.input_tokens * activation_bytes
        for tensor_name in ("hidden_states", "residual"):
            expected_transfer_identities.append((
                (request.request_id,), "activation", prefill_pool,
                prefill_ffn_pool, prefill_bytes, tensor_name,
            ))
        for tensor_name in ("hidden_states", "residual"):
            expected_transfer_identities.append((
                (request.request_id,), "activation", prefill_ffn_pool,
                prefill_pool, prefill_bytes, tensor_name,
            ))
        expected_transfer_identities.append((
            (request.request_id,), "kv", prefill_pool, decode_pool,
            request.input_tokens * kv_bytes_per_token, None,
        ))
        for _decode_step in range(request.output_tokens - 1):
            for tensor_name in ("hidden_states", "residual"):
                expected_transfer_identities.append((
                    (request.request_id,), "activation", decode_pool,
                    decode_ffn_pool, activation_bytes, tensor_name,
                ))
            for tensor_name in ("hidden_states", "residual"):
                expected_transfer_identities.append((
                    (request.request_id,), "activation", decode_ffn_pool,
                    decode_pool, activation_bytes, tensor_name,
                ))
    transfers = result["transfers"]
    expected_step_count = sum(
        request.output_tokens for request in expected_requests.requests)
    if (result["num_steps"] != expected_step_count
            or result["num_transfers"] != len(expected_transfer_identities)
            or len(transfers) != len(expected_transfer_identities)):
        raise ValueError(
            f"{context} step or transfer count disagrees with the validated "
            "request slice")
    actual_transfer_identities = []
    for transfer in transfers:
        if transfer["payload"] == "activation":
            tensor_name = payload_names_by_transfer_id[
                transfer["transfer_id"]]
        elif transfer["payload"] == "kv":
            tensor_name = None
        else:
            raise ValueError(f"{context} transfer payload is invalid")
        actual_transfer_identities.append((
            tuple(transfer["request_ids"]),
            transfer["payload"],
            transfer["source_pool_id"],
            transfer["destination_pool_id"],
            transfer["byte_count"],
            tensor_name,
        ))
    if sorted(actual_transfer_identities) != sorted(
            expected_transfer_identities):
        raise ValueError(
            f"{context} transfer disagrees with the canonical payload, "
            "endpoint, request, or byte count")
    transfer_resources = {}
    for transfer in transfers:
        payload = transfer["payload"]
        source_pool = transfer["source_pool_id"]
        destination_pool = transfer["destination_pool_id"]
        byte_count = transfer["byte_count"]
        source_rank = pool_ranks[source_pool]
        destination_rank = pool_ranks[destination_pool]
        if payload == "activation":
            config = activation
        elif payload == "kv":
            config = kv_transfer
        else:
            raise ValueError(f"{context} transfer payload is invalid")
        endpoints = tuple(sorted((source_pool, destination_pool)))
        resource_id = f"{config['link']}:{endpoints[0]}--{endpoints[1]}"
        expected_reserved_resources = (
            "communication:endpoint:nvidia_a800:"
            f"{source_rank}:{config['link']}",
            "communication:endpoint:nvidia_a800:"
            f"{destination_rank}:{config['link']}",
            f"communication:fabric:{resource_id}",
        )
        if (transfer["resource_id"] != resource_id
                or transfer["link"] != config["link"]
                or tuple(transfer["reserved_resource_ids"])
                != expected_reserved_resources
                or transfer["end_time_s"] <= transfer["start_time_s"]):
            raise ValueError(
                f"{context} transfer disagrees with the canonical payload, "
                "endpoint, bytes, or resource reservation")
        if resource_id not in transfer_resources:
            transfer_resources[resource_id] = [0, 0]
        transfer_resources[resource_id][0] += 1
        transfer_resources[resource_id][1] += byte_count

    artifact = result["execution_artifact"]
    steps = artifact["steps"]
    kv_artifacts = artifact["transfers"]
    if len(kv_artifacts) != len(expected_requests.requests):
        raise ValueError(f"{context} KV transfer artifact count is invalid")
    expected_prefill_pool_by_request = dict(zip(
        request_ids, request_prefill_pools, strict=True))
    expected_decode_pool_by_request = dict(zip(
        request_ids, request_decode_pools, strict=True))
    kv_artifact_by_request = {}
    for kv_artifact in kv_artifacts:
        request_id = kv_artifact["request_id"]
        if request_id in kv_artifact_by_request:
            raise ValueError(f"{context} repeated one request KV artifact")
        kv_artifact_by_request[request_id] = kv_artifact
    if set(kv_artifact_by_request) != set(request_ids):
        raise ValueError(f"{context} KV transfer requests are invalid")
    for request_id in request_ids:
        kv_artifact = kv_artifact_by_request[request_id]
        kv_trace = kv_artifact["trace_transfer"]
        prefill_steps = tuple(
            step for step in steps
            if step["pool_name"]
            == expected_prefill_pool_by_request[request_id]
            and any(
                request["request_id"] == request_id
                for request in step["backend_step"]["schedule"]["requests"])
        )
        decode_steps = tuple(
            step for step in steps
            if step["pool_name"]
            == expected_decode_pool_by_request[request_id]
            and any(
                request["request_id"] == request_id
                for request in step["backend_step"]["schedule"]["requests"])
        )
        if len(prefill_steps) != 1 or not decode_steps:
            raise ValueError(
                f"{context} request Scheduler step ownership is invalid")
        first_decode_step = min(
            decode_steps,
            key=lambda step: (
                step["schedule_interval"]["start_time_s"], step["step_id"]),
        )
        if (kv_trace["payload"] != "kv"
                or kv_trace["source_pool_id"]
                != expected_prefill_pool_by_request[request_id]
                or kv_trace["destination_pool_id"]
                != expected_decode_pool_by_request[request_id]
                or tuple(kv_trace["request_ids"]) != (request_id,)
                or kv_trace["start_time_s"]
                < prefill_steps[0]["schedule_interval"]["end_time_s"]
                or first_decode_step["schedule_interval"]["start_time_s"]
                < kv_trace["end_time_s"]):
            raise ValueError(
                f"{context} must complete each prefill before KV handoff "
                "and start decode only after that handoff completes")

    step_communication_calls = tuple(
        call
        for step in steps
        for call in step["kernel_plan"]["calls"]
        if call["kernel_spec"]["kind"] == KernelKind.COMMUNICATION.value
    )
    kv_communication_calls = tuple(
        call
        for transfer_artifact in kv_artifacts
        for call in transfer_artifact["kernel_plan"]["calls"]
        if call["kernel_spec"]["kind"] == KernelKind.COMMUNICATION.value
    )
    communication_calls = (*step_communication_calls, *kv_communication_calls)
    if len(communication_calls) != len(transfers):
        raise ValueError(
            f"{context} must lower every activation and KV transfer to one "
            "COMMUNICATION KernelSpec")
    calls_by_id = {}
    for call in communication_calls:
        call_id = call["call_id"]
        if call_id in calls_by_id:
            raise ValueError(f"{context} repeated COMMUNICATION call_id")
        calls_by_id[call_id] = call
    transfer_ids = {transfer["transfer_id"] for transfer in transfers}
    if set(calls_by_id) != transfer_ids:
        raise ValueError(
            f"{context} COMMUNICATION calls disagree with transfer IDs")
    for transfer in transfers:
        call = calls_by_id[transfer["transfer_id"]]
        payload = transfer["payload"]
        source_pool = transfer["source_pool_id"]
        destination_pool = transfer["destination_pool_id"]
        byte_count = transfer["byte_count"]
        source_rank = pool_ranks[source_pool]
        destination_rank = pool_ranks[destination_pool]
        if payload == "activation":
            config = activation
        elif payload == "kv":
            config = kv_transfer
        else:
            raise AssertionError("validated PD+AF transfer payload vanished")
        endpoints = tuple(sorted((source_pool, destination_pool)))
        resource_id = f"{config['link']}:{endpoints[0]}--{endpoints[1]}"
        expected_physical_transfer = {
            "algorithm": config["algorithm"],
            "bytes": byte_count,
            "direction": "point_to_point",
            "link": config["link"],
            "topology": config["topology"],
            "tier": None,
            "participants": [source_rank, destination_rank],
            "operation": config["algorithm"],
            "bytes_semantics": "payload",
            "source_rank": source_rank,
            "destination_rank": destination_rank,
            "ring_order": [],
            "reduction": None,
            "backend": config["backend"],
            "resource_id": resource_id,
            "backend_config_sha256": None,
        }
        expected_kernel_spec = {
            "kind": KernelKind.COMMUNICATION.value,
            "hardware_id": expected_simulation["hardware"]["hardware_id"],
            "shape": {},
            "dtype": None,
            "variant": None,
            "launch": None,
            "transfer": expected_physical_transfer,
            "attributes": {},
        }
        if (call["pool_name"] != source_pool
                or call["rank"] != source_rank
                or call["resource"] != resource_id
                or call["kernel_spec"] != expected_kernel_spec):
            raise ValueError(
                f"{context} COMMUNICATION KernelSpec disagrees with the "
                "canonical physical transfer contract")

    resources = result["metrics"]["system_resources"]
    worker_pools = set(pool_ranks)
    if set(resources["worker_pool_utilization"]) != worker_pools:
        raise ValueError(f"{context} worker utilization pools are invalid")
    if set(resources["scheduler_queue"]["by_pool"]) != set(scheduler_pools):
        raise ValueError(f"{context} Scheduler queue pools are invalid")
    router = pd_disagg["router"]
    router_queue = resources["router_queue"]
    if (router_queue["resource_id"] != router["resource_id"]
            or router_queue["configured_dispatch_service_time_s"]
            != router["dispatch_service_time_s"]):
        raise ValueError(f"{context} Router metrics disagree with the manifest")
    link_metrics = resources["link_utilization"]
    communication_metrics = resources["communication_resource_utilization"]
    expected_link_resources = {
        f"{activation['link']}:"
        f"{min(attention_pool, ffn_pool)}--"
        f"{max(attention_pool, ffn_pool)}"
        for attention_pool, ffn_pool in pool_pairs
    }
    expected_link_resources.update(
        f"{kv_transfer['link']}:"
        f"{min(prefill_pool, decode_pool)}--"
        f"{max(prefill_pool, decode_pool)}"
        for prefill_pool in prefill_pools
        for decode_pool in decode_pools
    )
    if (set(link_metrics) != expected_link_resources
            or set(communication_metrics) != set(transfer_resources)):
        raise ValueError(f"{context} communication resources are invalid")
    for resource_id in expected_link_resources:
        if resource_id in transfer_resources:
            transfer_count, transferred_bytes = transfer_resources[resource_id]
        else:
            transfer_count, transferred_bytes = 0, 0
        if (link_metrics[resource_id]["transfer_count"] != transfer_count
                or link_metrics[resource_id]["transferred_bytes"]
                != transferred_bytes):
            raise ValueError(
                f"{context} link metrics are inconsistent")
    for resource_id, (transfer_count, transferred_bytes) in (
            transfer_resources.items()):
        if (communication_metrics[resource_id]["timing_status"]
                != "projected_analytical_roofline"
                or communication_metrics[resource_id]["operation_counts"]
                != {"send_recv": transfer_count}):
            raise ValueError(
                f"{context} communication metrics are inconsistent")

    validate_pd_af_canonical_rematerialization(
        expected_simulation,
        expected_model,
        expected_model_path,
        result,
        expected_requests,
        expected_slo_contract,
    )


def validate_pd_af_canonical_rematerialization(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    context = "PD+AF Production Trace"
    pd_disagg = expected_simulation["pd_disagg"]
    af_disagg = expected_simulation["af_disagg"]
    if pd_disagg is None or af_disagg is None:
        raise AssertionError("validated PD+AF configuration vanished")
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    instance_pair = (
        pd_disagg["num_prefill_instances"],
        pd_disagg["num_decode_instances"],
    )
    if instance_pair not in ((1, 1), (1, 2), (2, 1), (2, 2)):
        raise AssertionError("validated PD+AF instance pair vanished")
    if pd_disagg["num_prefill_instances"] == 1:
        prefill_pairs = (("prefill-attention", "prefill-ffn"),)
    elif pd_disagg["num_prefill_instances"] == 2:
        prefill_pairs = (
            ("prefill-attention-0", "prefill-ffn-0"),
            ("prefill-attention-1", "prefill-ffn-1"),
        )
    else:
        raise AssertionError("validated PD+AF prefill instance count vanished")
    if pd_disagg["num_decode_instances"] == 1:
        decode_pairs = (("decode-attention", "decode-ffn"),)
    elif pd_disagg["num_decode_instances"] == 2:
        decode_pairs = (
            ("decode-attention-0", "decode-ffn-0"),
            ("decode-attention-1", "decode-ffn-1"),
        )
    else:
        raise AssertionError("validated PD+AF decode instance count vanished")
    pool_pairs = (*prefill_pairs, *decode_pairs)
    role_pairs = (
        *((pair, "prefill-attention", "prefill-ffn")
          for pair in prefill_pairs),
        *((pair, "decode-attention", "decode-ffn")
          for pair in decode_pairs),
    )
    pools = tuple(
        (pool_name, role, 2 * pair_index + rank_offset)
        for pair_index, (pair, attention_role, ffn_role) in enumerate(
            role_pairs)
        for rank_offset, (pool_name, role) in enumerate((
            (pair[0], attention_role),
            (pair[1], ffn_role),
        ))
    )
    worker_pools = tuple(WorkerPoolConfig(
        name,
        role,
        hardware_id,
        ParallelConfig(tensor_parallel_size=1),
        (rank,),
    ) for name, role, rank in pools)
    activation = af_disagg["activation_transfer"]
    kv_transfer = pd_disagg["kv_transfer"]

    def route(
        source: str,
        destination: str,
        payload: str,
        config: dict[str, Any],
    ) -> RouteConfig:
        endpoints = tuple(sorted((source, destination)))
        return RouteConfig(
            source,
            destination,
            payload,
            config["algorithm"],
            config["link"],
            config["topology"],
            resource_id=(
                f"{config['link']}:{endpoints[0]}--{endpoints[1]}"),
            backend=config["backend"],
        )

    routes = tuple(
        route(source_pool, destination_pool, "activation", activation)
        for attention_pool, ffn_pool in pool_pairs
        for source_pool, destination_pool in (
            (attention_pool, ffn_pool),
            (ffn_pool, attention_pool),
        )
    ) + tuple(
        route(prefill_pair[0], decode_pair[0], "kv", kv_transfer)
        for prefill_pair in prefill_pairs
        for decode_pair in decode_pairs
    )
    router = pd_disagg["router"]
    deployment = build_pd_disagg_plan(
        DeploymentConfig(
            "pd_af_disagg",
            worker_pools,
            routes,
            RouterConfig(
                router["resource_id"],
                router["dispatch_service_time_s"],
            ),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} deployment disagrees with canonical rematerialization")

    hardware = load_expected_hardware(
        expected_simulation, hardware_id, context)
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    scheduler = expected_simulation["scheduler"]
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 1))
    backends = {}
    try:
        with ExitStack() as construction_lifecycle:
            scheduler_roles = (
                *tuple(
                    (prefill_pair[0], "prefill-attention")
                    for prefill_pair in prefill_pairs
                ),
                *tuple(
                    (decode_pair[0], "decode-attention")
                    for decode_pair in decode_pairs
                ),
            )
            for pool_name, role in scheduler_roles:
                backend = VLLMBackend.from_model(
                    BackendConfig(
                        pool_name,
                        role,
                        scheduler["max_batched_tokens"],
                        scheduler["max_requests"],
                        scheduler_runtime[pool_name]["num_kv_blocks"],
                        kv_block_size_tokens=scheduler[
                            "kv_block_size_tokens"],
                        kv_bytes_per_token=kv_bytes_per_token,
                    scheduler_pipeline_depth=scheduler[
                        "scheduler_pipeline_depth"]),
                    VLLMSchedulerConfig(
                        expected_model_path,
                        tensor_parallel_size=1,
                        pipeline_parallel_size=1,
                        enable_expert_parallel=False,
                        dtype="bfloat16",
                        max_model_len=scheduler["max_model_len"],
                        enable_chunked_prefill=True,
                        enable_prefix_caching=False,
                        mtp_num_speculative_tokens=0,
                        mtp_acceptance_policy="disabled",
                        mtp_fixed_accepted_tokens=0,
                        mtp_acceptance_source_id="disabled",
                    ),
                )
                construction_lifecycle.callback(backend.close)
                expected_runtime = result["scheduler_runtime"][pool_name]
                if json_round_trip(asdict(backend.runtime_info)) != (
                        expected_runtime):
                    raise ValueError(
                        f"{context} pinned Scheduler runtime for "
                        f"{pool_name!r} disagrees with the recorded runtime")
                backends[pool_name] = backend
            canonical_result = MultiSimulation(
                SimulationConfig(
                    expected_model,
                    deployment,
                    expected_requests,
                    expected_simulation["launch_policy"],
                    expected_simulation["prediction_method"],
                    runtime_inputs,
                ),
                backends,
                {hardware_id: hardware},
            ).run()
            construction_lifecycle.pop_all()
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError,
            TimeoutError) as error:
        raise ValueError(
            f"{context} cannot be canonically reproduced from the command "
            "Trace and pinned P/D Attention Schedulers") from error

    expected_prefix_alignment = json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Schedulers")
    expected_artifact = json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                expected_performance_timing_status(expected_simulation)),
        ))
    if result["execution_artifact"] != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    expected_route_decisions = expected_artifact["schedule_trace"][
        "route_decisions"]
    expected_transfers = expected_artifact["schedule_trace"]["transfers"]
    expected_kernel_calls = (
        sum(step.kernel_plan.logical_call_count
            for step in canonical_result.steps)
        + sum(
            transfer.kernel_plan.logical_call_count
            for transfer in canonical_result.transfers
        )
    )
    if (result["route_decisions"] != expected_route_decisions
            or result["transfers"] != expected_transfers
            or result["num_steps"] != len(canonical_result.steps)
            or result["num_transfers"]
            != len(canonical_result.schedule_trace.transfers)
            or result["num_kernel_calls"] != expected_kernel_calls
            or result["schedule_end_time_s"]
            != canonical_result.schedule_trace.end_time_s):
        raise ValueError(
            f"{context} counts or ordered transfer trace disagree with "
            "canonical rematerialization")
    expected_metrics = canonical_production_trace_metrics_manifest(
        deployment,
        expected_requests,
        canonical_result,
        expected_slo_contract,
    )
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Schedulers and "
            "execution trace")


def validate_pd_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    if expected_simulation["af_disagg"] is not None:
        raise ValueError("PD Production Trace command retained AF config")
    pd_disagg = expected_simulation["pd_disagg"]
    if pd_disagg is None:
        raise ValueError("PD Production Trace command lost PD configuration")
    expected_scheduler = expected_simulation["scheduler"]
    prefill_pools = tuple(
        f"prefill-{index}"
        for index in range(pd_disagg["num_prefill_instances"])
    )
    decode_pools = tuple(
        f"decode-{index}"
        for index in range(pd_disagg["num_decode_instances"])
    )
    expected_pools = prefill_pools + decode_pools
    scheduler_runtime = result["scheduler_runtime"]
    if set(scheduler_runtime) != set(expected_pools):
        raise ValueError(
            "PD Production Trace sweep Scheduler pools disagree with "
            "command manifest")
    for pool_name in prefill_pools:
        validate_production_trace_scheduler_runtime(
            scheduler_runtime[pool_name],
            expected_scheduler,
            expected_model,
            expected_simulation["mtp"],
            role="prefill",
            tensor_parallel_field="prefill_tp_size",
        )
    for pool_name in decode_pools:
        validate_production_trace_scheduler_runtime(
            scheduler_runtime[pool_name],
            expected_scheduler,
            expected_model,
            expected_simulation["mtp"],
            role="decode",
            tensor_parallel_field="decode_tp_size",
        )
    validate_pd_deployment_plan(
        expected_simulation,
        result,
        prefill_pools,
        decode_pools,
    )
    validate_pd_capacity_and_routes(
        expected_simulation,
        result,
        prefill_pools,
        decode_pools,
    )
    validate_pd_canonical_rematerialization(
        expected_simulation,
        expected_model,
        expected_model_path,
        result,
        expected_requests,
        expected_slo_contract,
        prefill_pools,
        decode_pools,
    )


def validate_pd_deployment_plan(
    expected_simulation: dict[str, Any],
    result: dict[str, Any],
    prefill_pools: tuple[str, ...],
    decode_pools: tuple[str, ...],
) -> None:
    pd_disagg = expected_simulation["pd_disagg"]
    if pd_disagg is None:
        raise AssertionError("validated PD configuration vanished")
    plan = result["deployment_plan"]
    if (plan["schema_version"] != 6
            or plan["mode"] != "pd_disagg"
            or plan["deployment_id"] != result["deployment_id"]):
        raise ValueError(
            "PD Production Trace deployment identity disagrees with "
            "command manifest")
    if plan["router"] != pd_disagg["router"]:
        raise ValueError(
            "PD Production Trace router contract disagrees with command "
            "manifest")
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    default_parallelism = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "enable_expert_parallel": False,
        "expert_communication_backend": "allgather_reducescatter",
        "enable_eplb": False,
        "sequence_parallel_size": 1,
        "context_parallel_size": 1,
        "data_parallel_size": 1,
    }
    expected_pool_contracts = {}
    for rank, pool_name in enumerate(prefill_pools):
        expected_pool_contracts[pool_name] = {
            "name": pool_name,
            "role": "prefill",
            "hardware_id": hardware_id,
            "ranks": [rank],
            "parallelism": default_parallelism,
            "kv_cache": None,
            "collectives": [],
            "pipeline_p2p": [],
        }
    decode_rank_offset = len(prefill_pools)
    for index, pool_name in enumerate(decode_pools):
        expected_pool_contracts[pool_name] = {
            "name": pool_name,
            "role": "decode",
            "hardware_id": hardware_id,
            "ranks": [decode_rank_offset + index],
            "parallelism": default_parallelism,
            "kv_cache": None,
            "collectives": [],
            "pipeline_p2p": [],
        }
    pools = plan["worker_pools"]
    actual_pool_contracts = {pool["name"]: pool for pool in pools}
    if (len(actual_pool_contracts) != len(pools)
            or actual_pool_contracts != expected_pool_contracts):
        raise ValueError(
            "PD Production Trace worker-pool topology disagrees with "
            "command manifest")
    transfer = pd_disagg["kv_transfer"]
    expected_routes = {}
    for source_pool in prefill_pools:
        for destination_pool in decode_pools:
            endpoints = tuple(sorted((source_pool, destination_pool)))
            resource_id = (
                f"{transfer['link']}:{endpoints[0]}--{endpoints[1]}")
            expected_routes[(source_pool, destination_pool)] = {
                "source_pool": source_pool,
                "destination_pool": destination_pool,
                "payload": transfer["payload"],
                "algorithm": transfer["algorithm"],
                "link": transfer["link"],
                "topology": transfer["topology"],
                "resource_id": resource_id,
                "backend": transfer["backend"],
                "backend_config_sha256": None,
            }
    routes = plan["routes"]
    actual_routes = {
        (route["source_pool"], route["destination_pool"]): route
        for route in routes
    }
    if (len(actual_routes) != len(routes)
            or actual_routes != expected_routes):
        raise ValueError(
            "PD Production Trace transfer routes disagree with command "
            "manifest")


def validate_pd_capacity_and_routes(
    expected_simulation: dict[str, Any],
    result: dict[str, Any],
    prefill_pools: tuple[str, ...],
    decode_pools: tuple[str, ...],
) -> None:
    pd_disagg = expected_simulation["pd_disagg"]
    if pd_disagg is None:
        raise AssertionError("validated PD configuration vanished")
    if (result["routing_policy"] != pd_disagg["routing_policy"]
            or result["kv_handoff_mode"] != pd_disagg["kv_handoff_mode"]
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2):
        raise ValueError(
            "PD Production Trace routing contract disagrees with command "
            "manifest")
    if result["kv_lifecycle_replay"] is not None:
        raise ValueError(
            "PD Production Trace sweep unexpectedly enabled KV offload")
    if result["execution_artifact"]["modeled_kv_offload"][
            "availability"] != "not_enabled":
        raise ValueError(
            "PD Production Trace execution unexpectedly modeled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None):
        raise ValueError(
            "PD Production Trace sweep unexpectedly enabled MTP")
    expected_pools = prefill_pools + decode_pools
    expected_pool_set = set(expected_pools)
    validate_production_trace_capacity(
        result,
        expected_simulation["scheduler"],
        expected_pools,
        context="PD Production Trace",
    )
    capacity_plans = result["kv_capacity"]["scheduler_pools"]
    plans_by_pool = {
        plan["pool_name"]: plan for plan in capacity_plans
    }
    if (len(plans_by_pool) != len(capacity_plans)
            or set(plans_by_pool) != expected_pool_set):
        raise ValueError(
            "PD Production Trace KV capacity pools disagree with command "
            "manifest")
    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"]
            != result["route_decisions"]
            or artifact_trace["transfers"] != result["transfers"]):
        raise ValueError(
            "PD Production Trace routing records disagree with the execution "
            "artifact")
    artifact_counts = artifact["counts"]
    if (result["num_transfers"] != len(result["transfers"])
            or artifact_counts["trace_transfers"]
            != len(result["transfers"])
            or artifact_counts["simulation_transfers"]
            != len(artifact["transfers"])
            or len(artifact["transfers"]) != len(result["transfers"])):
        raise ValueError(
            "PD Production Trace transfer counts disagree with the execution "
            "artifact")
    trace_transfers_by_id = {
        transfer["transfer_id"]: transfer
        for transfer in result["transfers"]
    }
    if len(trace_transfers_by_id) != len(result["transfers"]):
        raise ValueError(
            "PD Production Trace contains duplicate transfer IDs")
    artifact_transfers_by_id = {
        transfer["trace_transfer"]["transfer_id"]:
        transfer["trace_transfer"]
        for transfer in artifact["transfers"]
    }
    if (len(artifact_transfers_by_id) != len(artifact["transfers"])
            or artifact_transfers_by_id != trace_transfers_by_id):
        raise ValueError(
            "PD Production Trace transfers disagree with the execution "
            "artifact")
    plan_routes = {
        (route["source_pool"], route["destination_pool"]): route
        for route in result["deployment_plan"]["routes"]
    }
    prefill_set = set(prefill_pools)
    decode_set = set(decode_pools)
    prefill_assignment = {}
    decode_assignment = {}
    for step in artifact["steps"]:
        pool_name = step["pool_name"]
        schedule = step["backend_step"]["schedule"]
        if (pool_name not in expected_pool_set
                or schedule["pool_name"] != pool_name):
            raise ValueError(
                "PD Production Trace execution step uses an unknown pool")
        if pool_name in prefill_set:
            assignment = prefill_assignment
        elif pool_name in decode_set:
            assignment = decode_assignment
        else:
            raise AssertionError("validated PD execution pool vanished")
        for request in schedule["requests"]:
            request_id = request["request_id"]
            if (request_id in assignment
                    and assignment[request_id] != pool_name):
                raise ValueError(
                    "PD Production Trace request executed in multiple pools "
                    "for one role")
            assignment[request_id] = pool_name
    ingress_by_request = {}
    handoff_by_request = {}
    for raw_decision in result["route_decisions"]:
        try:
            route_decision_from_output_manifest(raw_decision)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "PD Production Trace route decision violates the canonical "
                f"RouteDecision contract: {error}") from error
        decision = raw_decision
        if (decision["deployment_id"] != result["deployment_id"]
                or decision["policy"] != pd_disagg["routing_policy"]):
            raise ValueError(
                "PD Production Trace route decision identity disagrees "
                "with command manifest")
        candidate_ids = tuple(
            candidate[0]
            for candidate in decision["candidate_queued_tokens"])
        if decision["stage"] == "ingress_to_prefill":
            if (candidate_ids != tuple(sorted(prefill_pools))
                    or decision["selected_engine_id"] not in prefill_set
                    or decision["source_engine_id"] is not None):
                raise ValueError(
                    "PD Production Trace ingress route disagrees with "
                    "command manifest")
            destination = ingress_by_request
        elif decision["stage"] == "prefill_to_decode":
            if (candidate_ids != tuple(sorted(decode_pools))
                    or decision["selected_engine_id"] not in decode_set
                    or decision["source_engine_id"] not in prefill_set):
                raise ValueError(
                    "PD Production Trace handoff route disagrees with "
                    "command manifest")
            destination = handoff_by_request
        else:
            raise ValueError(
                "PD Production Trace contains an unsupported route stage")
        request_id = decision["request_id"]
        if request_id in destination:
            raise ValueError(
                "PD Production Trace contains duplicate per-request route "
                "decisions")
        destination[request_id] = decision
    arrival_ids = tuple(
        arrival["request_id"] for arrival in result["request_arrivals_s"])
    if (len(arrival_ids) != len(set(arrival_ids))
            or set(arrival_ids) != set(result["metrics"]["requests"])
            or set(arrival_ids) != set(ingress_by_request)
            or set(arrival_ids) != set(prefill_assignment)):
        raise ValueError(
            "PD Production Trace ingress coverage disagrees with requests")
    for request_id, decision in ingress_by_request.items():
        if decision["selected_engine_id"] != prefill_assignment[request_id]:
            raise ValueError(
                "PD Production Trace ingress route disagrees with execution "
                "pool assignment")
    if set(handoff_by_request) != set(decode_assignment):
        raise ValueError(
            "PD Production Trace handoff coverage disagrees with decode "
            "execution")
    for request_id, decision in handoff_by_request.items():
        if (decision["source_engine_id"] != prefill_assignment[request_id]
                or decision["selected_engine_id"]
                != decode_assignment[request_id]):
            raise ValueError(
                "PD Production Trace handoff route disagrees with execution "
                "pool assignment")
    transfers_by_request = {}
    for transfer in result["transfers"]:
        endpoints = (
            transfer["source_pool_id"], transfer["destination_pool_id"])
        if endpoints not in plan_routes:
            raise ValueError(
                "PD Production Trace transfer references an unknown route")
        route = plan_routes[endpoints]
        if (transfer["payload"] != "kv"
                or transfer["resource_id"] != route["resource_id"]
                or transfer["link"] != route["link"]):
            raise ValueError(
                "PD Production Trace transfer disagrees with command "
                "manifest")
        request_ids = transfer["request_ids"]
        if len(request_ids) != 1:
            raise ValueError(
                "PD Production Trace KV transfer must contain one request")
        request_id = request_ids[0]
        if request_id in transfers_by_request:
            raise ValueError(
                "PD Production Trace contains duplicate per-request KV "
                "transfers")
        transfers_by_request[request_id] = transfer
    if set(transfers_by_request) != set(handoff_by_request):
        raise ValueError(
            "PD Production Trace transfer coverage disagrees with handoff "
            "routing")
    for request_id, transfer in transfers_by_request.items():
        decision = handoff_by_request[request_id]
        if (transfer["source_pool_id"] != decision["source_engine_id"]
                or transfer["destination_pool_id"]
                != decision["selected_engine_id"]):
            raise ValueError(
                "PD Production Trace transfer endpoints disagree with "
                "handoff routing")
    system_resources = result["metrics"]["system_resources"]
    if set(system_resources["worker_pool_utilization"]) != expected_pool_set:
        raise ValueError(
            "PD Production Trace worker utilization pools disagree with "
            "command manifest")
    scheduler_queue = system_resources["scheduler_queue"]
    if set(scheduler_queue["by_pool"]) != expected_pool_set:
        raise ValueError(
            "PD Production Trace Scheduler queue pools disagree with command "
            "manifest")
    router_queue = system_resources["router_queue"]
    if (router_queue["resource_id"] != pd_disagg["router"]["resource_id"]
            or router_queue["configured_dispatch_service_time_s"]
            != pd_disagg["router"]["dispatch_service_time_s"]):
        raise ValueError(
            "PD Production Trace Router metrics disagree with command "
            "manifest")


def validate_pd_canonical_rematerialization(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
    prefill_pools: tuple[str, ...],
    decode_pools: tuple[str, ...],
) -> None:
    context = "PD Production Trace"
    pd_disagg = expected_simulation["pd_disagg"]
    if pd_disagg is None:
        raise AssertionError("validated PD configuration vanished")
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    worker_pools = []
    for rank, pool_name in enumerate(prefill_pools):
        worker_pools.append(WorkerPoolConfig(
            pool_name,
            "prefill",
            hardware_id,
            ParallelConfig(),
            (rank,),
        ))
    decode_rank_offset = len(prefill_pools)
    for index, pool_name in enumerate(decode_pools):
        worker_pools.append(WorkerPoolConfig(
            pool_name,
            "decode",
            hardware_id,
            ParallelConfig(),
            (decode_rank_offset + index,),
        ))
    transfer = pd_disagg["kv_transfer"]
    routes = []
    for source_pool in prefill_pools:
        for destination_pool in decode_pools:
            endpoints = tuple(sorted((source_pool, destination_pool)))
            routes.append(RouteConfig(
                source_pool,
                destination_pool,
                transfer["payload"],
                transfer["algorithm"],
                transfer["link"],
                transfer["topology"],
                resource_id=(
                    f"{transfer['link']}:{endpoints[0]}--{endpoints[1]}"),
                backend=transfer["backend"],
            ))
    router = pd_disagg["router"]
    deployment = build_pd_disagg_plan(
        DeploymentConfig(
            "pd_disagg",
            tuple(worker_pools),
            tuple(routes),
            RouterConfig(
                router["resource_id"],
                router["dispatch_service_time_s"],
            ),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} deployment disagrees with canonical "
            "rematerialization")
    hardware = load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    scheduler = expected_simulation["scheduler"]
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 1))
    backends = {}
    try:
        with ExitStack() as construction_lifecycle:
            for pool_name in prefill_pools + decode_pools:
                role = (
                    "prefill" if pool_name in prefill_pools else "decode")
                backend = VLLMBackend.from_model(
                    BackendConfig(
                        pool_name,
                        role,
                        scheduler["max_batched_tokens"],
                        scheduler["max_requests"],
                        result["scheduler_runtime"][pool_name][
                            "num_kv_blocks"],
                        kv_block_size_tokens=scheduler[
                            "kv_block_size_tokens"],
                        kv_bytes_per_token=kv_bytes_per_token,
                    scheduler_pipeline_depth=scheduler[
                        "scheduler_pipeline_depth"]),
                    VLLMSchedulerConfig(
                        expected_model_path,
                        tensor_parallel_size=1,
                        pipeline_parallel_size=1,
                        enable_expert_parallel=False,
                        dtype="bfloat16",
                        max_model_len=scheduler["max_model_len"],
                        enable_chunked_prefill=True,
                        enable_prefix_caching=False,
                        mtp_num_speculative_tokens=0,
                        mtp_acceptance_policy="disabled",
                        mtp_fixed_accepted_tokens=0,
                        mtp_acceptance_source_id="disabled",
                    ),
                )
                construction_lifecycle.callback(backend.close)
                expected_runtime = result["scheduler_runtime"][pool_name]
                if json_round_trip(asdict(backend.runtime_info)) != (
                        expected_runtime):
                    raise ValueError(
                        f"{context} pinned Scheduler runtime for "
                        f"{pool_name!r} disagrees with the recorded runtime")
                backends[pool_name] = backend
            canonical_result = MultiSimulation(
                SimulationConfig(
                    expected_model,
                    deployment,
                    expected_requests,
                    expected_simulation["launch_policy"],
                    expected_simulation["prediction_method"],
                    runtime_inputs,
                ),
                backends,
                hardware_by_id,
            ).run()
            construction_lifecycle.pop_all()
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError,
            TimeoutError) as error:
        raise ValueError(
            f"{context} cannot be canonically reproduced from the command "
            "Trace and pinned Scheduler") from error

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
    if result["execution_artifact"] != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    expected_route_decisions = expected_artifact["schedule_trace"][
        "route_decisions"]
    expected_transfers = expected_artifact["schedule_trace"]["transfers"]
    expected_kernel_calls = (
        sum(step.kernel_plan.logical_call_count
            for step in canonical_result.steps)
        + sum(
            transfer.kernel_plan.logical_call_count
            for transfer in canonical_result.transfers
        )
    )
    if (result["route_decisions"] != expected_route_decisions
            or result["transfers"] != expected_transfers
            or result["num_steps"] != len(canonical_result.steps)
            or result["num_transfers"]
            != len(canonical_result.schedule_trace.transfers)
            or result["num_kernel_calls"] != expected_kernel_calls
            or result["schedule_end_time_s"]
            != canonical_result.schedule_trace.end_time_s):
        raise ValueError(
            f"{context} counts or ordered routing trace disagree with "
            "canonical rematerialization")
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
