"""Shared colocated Production Trace validation helpers: KV
lifecycle replay comparison and TP collective payload bytes."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from typing import Any
from evaluation.serving.simulation.execution_artifact import (
    execution_artifact_manifest,
    prefix_alignment_manifest,
)
from src.serving.backend.vllm_scheduler_process import vllm_scheduler_runtime_sha256
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.deployment.deployment_plan import DeploymentPlan
from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager
from src.serving.request_gen.request_stream import RequestStream, request_stream_sha256
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.simulation import Simulation, SimulationConfig
from src.serving.workload.model_config import tp_collective_payload_bytes
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs
from src.serving.workload.shape_inference import infer_gemm_dimensions

from evaluation.serving.experiments.production_trace_contracts import (
    canonical_production_trace_metrics_manifest,
    expected_performance_timing_status,
    json_round_trip,
    load_expected_hardware,
    validate_host_kv_system_resources,
    validate_tp1_kv_offload_invariants,
    validate_tp2_kv_offload_invariants,
)


def validate_colocated_kv_lifecycle_replay(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
    deployment: DeploymentPlan,
    recorded_runtime: dict[str, Any],
    kv_lifecycle_replay: Any,
    *,
    tensor_parallel_size: int,
    context: str,
) -> None:
    scheduler = expected_simulation["scheduler"]
    if (expected_simulation["mtp"]["acceptance_policy"] != "disabled"
            or expected_simulation["expert_routing"] is not None
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} unexpectedly enabled MTP or MoE routing")
    expected_replay_manifest = json_round_trip(
        kv_lifecycle_replay.manifest())
    if (result["kv_lifecycle_replay"] != expected_replay_manifest
            or kv_lifecycle_replay.context.request_stream_sha256
            != request_stream_sha256(expected_requests)):
        raise ValueError(
            f"{context} replay disagrees with the command manifest or Trace")

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
        raise ValueError(f"{context} runtime inputs disagree")

    hardware_id = expected_simulation["hardware"]["hardware_id"]
    hardware = load_expected_hardware(
        expected_simulation, hardware_id, context)
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(
            tensor_parallel_size, 1))
    backend_config = BackendConfig(
        "main",
        "colocated",
        scheduler["max_batched_tokens"],
        scheduler["max_requests"],
        scheduler["num_kv_blocks"],
        kv_block_size_tokens=scheduler["kv_block_size_tokens"],
        kv_bytes_per_token=kv_bytes_per_token,
    scheduler_pipeline_depth=scheduler["scheduler_pipeline_depth"])
    try:
        with ExitStack() as construction_lifecycle:
            backend = VLLMBackend.from_model(
                backend_config,
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
            canonical_runtime = json_round_trip(asdict(backend.runtime_info))
            if canonical_runtime != recorded_runtime:
                raise ValueError(
                    f"{context} pinned Scheduler runtime disagrees with the "
                    "recorded runtime")
            runtime_sha256 = vllm_scheduler_runtime_sha256(
                backend.runtime_info)
            canonical_result = Simulation(
                SimulationConfig(
                    expected_model,
                    deployment,
                    expected_requests,
                    expected_simulation["launch_policy"],
                    expected_simulation["prediction_method"],
                    runtime_inputs,
                ),
                backend,
                {hardware_id: hardware},
                kv_cache_tier_manager=KVCacheTierManager(
                    kv_lifecycle_replay.cache_config),
                kv_lifecycle_replay=kv_lifecycle_replay,
                scheduler_runtime_sha256=runtime_sha256,
            ).run()
            # Simulation.run owns the backend after construction and closes it
            # on both success and failure.  Keep the ExitStack callback only
            # for failures before run() assumes that ownership.
            construction_lifecycle.pop_all()
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError,
            TimeoutError) as error:
        raise ValueError(
            f"{context} cannot be canonically reproduced from the command "
            "Trace, replay, and pinned Scheduler") from error

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
    modeled_offload = expected_artifact["modeled_kv_offload"]
    expected_availability = (
        "available_modeled_serial"
        if tensor_parallel_size == 1
        else "available_modeled_rank_local"
    )
    if (modeled_offload["availability"] != expected_availability
            or modeled_offload["intent_source"]["kind"]
            != "scheduler_event_replay"
            or modeled_offload["intent_source"]["replay"]["source_sha256"]
            != kv_lifecycle_replay.source_sha256):
        raise ValueError(
            f"{context} modeled lifecycle or final tier state is invalid")
    if tensor_parallel_size == 1:
        validate_tp1_kv_offload_invariants(
            expected_artifact,
            expected_model,
            expected_hbm_blocks=scheduler["num_kv_blocks"],
            context=context,
        )
    elif tensor_parallel_size == 2:
        validate_tp2_kv_offload_invariants(
            expected_artifact,
            expected_model,
            context=context,
        )
    else:
        raise AssertionError(
            "formal Host-KV verification supports only TP=1 or TP=2")
    counts = expected_artifact["counts"]
    expected_kernel_calls = sum(
        step.kernel_plan.logical_call_count
        for step in canonical_result.steps)
    if (result["num_steps"] != len(canonical_result.steps)
            or result["num_kernel_calls"] != expected_kernel_calls
            or result["schedule_end_time_s"]
            != canonical_result.schedule_trace.end_time_s
            or counts["modeled_kv_offload_transfers"] != 2):
        raise ValueError(f"{context} counts or completion time disagree")
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
    validate_host_kv_system_resources(
        expected_metrics["system_resources"],
        pool_name="main",
        ranks=tuple(range(tensor_parallel_size)),
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


def tp2_collective_bytes_from_schedule_and_model(
    model: ModelConfig,
    schedule: dict[str, Any],
    collective_call: dict[str, Any],
    calls_by_id: dict[str, dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> int:
    requests = schedule["requests"]
    token_count = sum(
        request["input_tokens"] + request["output_tokens"]
        for request in requests
    )
    if token_count <= 0:
        raise ValueError(
            "TP=2 Production Trace Scheduler step has no token work")
    call_id = collective_call["call_id"]
    transfer = collective_call["kernel_spec"]["transfer"]
    if call_id.endswith(":tp-all-reduce"):
        base_id = call_id[:-len(":tp-all-reduce")]
    elif call_id.endswith(":tp-all-gather"):
        base_id = call_id[:-len(":tp-all-gather")]
    else:
        raise ValueError(
            "TP=2 Production Trace collective call ID is invalid")
    producer_ids = [
        f"{base_id}:tp-rank-0",
        f"{base_id}:tp-rank-1",
    ]
    if collective_call["dependencies"] != producer_ids:
        raise ValueError(
            "TP=2 Production Trace collective producers are invalid")

    if base_id.endswith(":token-embedding"):
        expected_operation = "all_reduce"
        expected_kernel_kind = "VECTOR"
        expected_kernel_variant = "gather"
        element_count = token_count * model.hidden_size
        expected_kernel_shape = {
            "elements": element_count,
            "rows": token_count,
            "hidden": model.hidden_size,
            "dtype_bytes": model.dtype_bytes,
            "element_count": element_count,
        }
        expected_workload_kind = "gather"
        expected_workload_variant = "embedding"
        expected_workload_shape = {
            "elements": element_count,
            "rows": token_count,
            "hidden": model.hidden_size,
        }
        expected_bytes = tp_collective_payload_bytes(
            model,
            expected_operation,
            token_count,
        )
    elif base_id.endswith(":o-proj"):
        expected_operation = "all_reduce"
        expected_kernel_kind = "GEMM"
        expected_kernel_variant = "gemm"
        m, n, k = infer_gemm_dimensions(
            model, "o_proj", token_count, tp_size=2)
        expected_kernel_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
            "dtype_bytes": model.dtype_bytes,
        }
        expected_workload_kind = "linear"
        expected_workload_variant = "gemm"
        expected_workload_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
        }
        expected_bytes = tp_collective_payload_bytes(
            model,
            expected_operation,
            token_count,
        )
    elif base_id.endswith(":down"):
        expected_operation = "all_reduce"
        expected_kernel_kind = "GEMM"
        expected_kernel_variant = "gemm"
        m, n, k = infer_gemm_dimensions(
            model, "down_proj", token_count, tp_size=2)
        expected_kernel_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
            "dtype_bytes": model.dtype_bytes,
        }
        expected_workload_kind = "linear"
        expected_workload_variant = "gemm"
        expected_workload_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
        }
        expected_bytes = tp_collective_payload_bytes(
            model,
            expected_operation,
            token_count,
        )
    elif base_id.endswith(":lm-head"):
        expected_operation = "all_gather"
        expected_kernel_kind = "GEMM"
        target_logit_rows = len(requests)
        m, n, k = infer_gemm_dimensions(
            model, "lm_head", target_logit_rows, tp_size=2)
        if m == 1:
            expected_kernel_variant = "gemv"
        elif m > 1:
            expected_kernel_variant = "gemm"
        else:
            raise ValueError(
                "TP=2 Production Trace LM-head rows must be positive")
        expected_kernel_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
            "dtype_bytes": model.dtype_bytes,
        }
        expected_workload_kind = "lm_head"
        expected_workload_variant = expected_kernel_variant
        expected_workload_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
        }
        expected_bytes = tp_collective_payload_bytes(
            model,
            expected_operation,
            target_logit_rows,
            local_output_width=n,
        )
    else:
        raise ValueError(
            "TP=2 Production Trace emitted a collective for an unsupported "
            "operator")
    if transfer["operation"] != expected_operation:
        raise ValueError(
            "TP=2 Production Trace collective operation disagrees with its "
            "producer")

    for producer_id in producer_ids:
        producer = calls_by_id[producer_id]
        kernel_spec = producer["kernel_spec"]
        node = nodes_by_id[producer_id]
        if (producer["resource"] != "compute"
                or kernel_spec["kind"] != expected_kernel_kind
                or kernel_spec["dtype"] != "bf16"
                or kernel_spec["variant"] != expected_kernel_variant
                or kernel_spec["shape"] != expected_kernel_shape
                or kernel_spec["transfer"] is not None
                or node["kind"] != expected_workload_kind
                or node["pool_name"] != producer["pool_name"]
                or node["rank"] != producer["rank"]
                or node["shape"] != expected_workload_shape
                or node["dtype"] != "bf16"
                or node["variant"] != expected_workload_variant):
            raise ValueError(
                "TP=2 Production Trace collective producer shape disagrees "
                "with the Scheduler/model")

    collective_node = nodes_by_id[call_id]
    expected_collective_attributes = {
        "algorithm": transfer["algorithm"],
        "bytes": transfer["bytes"],
        "bytes_semantics": transfer["bytes_semantics"],
        "backend": transfer["backend"],
        "backend_config_sha256": transfer["backend_config_sha256"],
        "direction": transfer["direction"],
        "link": transfer["link"],
        "topology": transfer["topology"],
        "participants": transfer["participants"],
        "operation": transfer["operation"],
        "source_rank": transfer["source_rank"],
        "destination_rank": transfer["destination_rank"],
        "ring_order": transfer["ring_order"],
        "reduction": transfer["reduction"],
        "resource_id": transfer["resource_id"],
    }
    if (collective_node["kind"] != "collective"
            or collective_node["pool_name"] != collective_call["pool_name"]
            or collective_node["rank"] != collective_call["rank"]
            or collective_node["shape"] != {}
            or collective_node["dtype"] is not None
            or collective_node["variant"] is not None
            or collective_node["attributes"]
            != expected_collective_attributes):
        raise ValueError(
            "TP=2 Production Trace collective lowering identity is invalid")
    return expected_bytes
