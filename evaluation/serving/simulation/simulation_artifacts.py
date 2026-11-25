"""Simulation artifacts; extracted without changing the established contracts."""

from __future__ import annotations

from src.serving.kv_cache.kv_capacity import validate_ffn_source_capacity, validate_source_activation_capacity

import hashlib
import json
import os
import stat
from pathlib import Path
from evaluation.serving.experiments.experiment_plan import git_code_identity, load_workload_artifact
from evaluation.serving.simulation.execution_artifact import performance_timing_status
from src.kernel.hardware_spec import HWSpec
from src.kernel.predictors.protobridge.adapter import development_profile_config_sha256, resolve_registered_profile
from src.serving.backend.mtp_acceptance_replay import mtp_acceptance_replay_manifest, mtp_acceptance_step_from_backend_step, mtp_acceptance_steps_manifest, pd_decode_admission_contract_sha256
from src.serving.deployment.deployment_config import worker_role_operator_group, worker_role_phase
from src.serving.host_timing import schedule_phase
from src.serving.request_gen.production_trace import build_production_trace_replay, load_production_trace, production_trace_manifest, production_trace_replay_manifest
from src.serving.request_gen import CANONICAL_EQUAL_ARRIVAL_ORDER, CANONICAL_REQUEST_COLUMNS, CANONICAL_REQUEST_ID_POLICY
from src.serving.request_gen.canonical_requests import parse_canonical_requests_bytes
from src.serving.request_gen.request_stream import RequestSpec, RequestStream, request_stream_sha256
from src.serving.schedule.schedule_step import BackendStep
from src.serving.simulation import StreamingSimulationResult
from src.serving.workload.runtime_inputs import RuntimeInputs

from evaluation.serving.simulation.simulation_configuration import (
    ROOT,
    arrival_schedule,
    production_trace_reader_config,
    require_expected_sha256,
    uses_development_protobridge,
)
from dataclasses import asdict


def host_timing_manifest(
    result,
    host_launch_profiles,
    host_runtime_prepare_profiles,
) -> dict[str, object]:
    """Return additive and overlap-aware Host/Device timing evidence."""

    entries = result.schedule_trace.entries
    if not entries:
        raise ValueError("Host timing requires a non-empty schedule trace")
    mode = entries[0].timing_mode
    if any(entry.timing_mode != mode for entry in entries):
        raise ValueError("schedule trace mixes Serving timing modes")
    if isinstance(result, StreamingSimulationResult):
        steps_by_id = None
        timings_by_id = {
            timing.step_id: timing for timing in result.step_timings
        }
    else:
        timings_by_id = None
        steps_by_id = {
            step.backend_step.schedule.step_id: step for step in result.steps
        }
        if set(steps_by_id) != {
                entry.backend_step.schedule.step_id for entry in entries}:
            raise RuntimeError(
                "Host timing steps disagree with schedule trace entries")
    rows = []
    for entry in entries:
        schedule = entry.backend_step.schedule
        if entry.execution_wall_s is None:
            raise RuntimeError("simulation trace lacks execution wall time")
        if steps_by_id is None:
            if mode == "device_only":
                predicted_device_s = entry.execution_wall_s
                launch_cpu_s = 0.0
                launch_critical_path_s = 0.0
                overlap_s = 0.0
                eager_dispatch_count = 0
                graph_replay_count = 0
                launch_kinds = ()
            elif mode in (
                    "device_plus_launch",
                    "device_plus_launch_plus_scheduler"):
                timing = timings_by_id[schedule.step_id]
                predicted_device_s = timing.predicted_device_time_s
                launch_cpu_s = timing.launch_cpu_time_s
                launch_critical_path_s = timing.launch_critical_path_s
                overlap_s = timing.host_device_overlap_s
                eager_dispatch_count = timing.eager_dispatch_count
                graph_replay_count = timing.graph_replay_count
                launch_kinds = timing.launch_kinds
            else:
                raise ValueError(
                    f"unsupported Serving timing mode: {mode!r}")
        else:
            timeline = steps_by_id[schedule.step_id].execution_timeline
            predicted_device_s = timeline.predicted_device_time_s
            launch_cpu_s = timeline.launch_cpu_time_s
            launch_critical_path_s = timeline.launch_critical_path_s
            overlap_s = timeline.host_device_overlap_s
            eager_dispatch_count = timeline.eager_dispatch_count
            graph_replay_count = timeline.graph_replay_count
            launch_kinds = timeline.launch_kinds
        rows.append({
            "step_id": schedule.step_id,
            "pool_name": schedule.pool_name,
            "phase": schedule_phase(schedule),
            "scheduled_request_count": len(schedule.requests),
            "prefill_token_count": sum(
                request.input_tokens for request in schedule.requests),
            "decode_token_count": sum(
                request.output_tokens for request in schedule.requests),
            "predicted_device_s": predicted_device_s,
            "launch_cpu_s": launch_cpu_s,
            "launch_critical_path_s": launch_critical_path_s,
            "host_device_overlap_s": overlap_s,
            "scheduler_plan_cpu_s": entry.scheduler_plan_cpu_s,
            "scheduler_update_cpu_s": entry.scheduler_update_cpu_s,
            "execution_wall_s": entry.execution_wall_s,
            "step_wall_s": entry.end_time_s - entry.start_time_s,
            "eager_dispatch_count": eager_dispatch_count,
            "graph_replay_count": graph_replay_count,
            "launch_kinds": launch_kinds,
        })
    aggregate_fields = (
        "predicted_device_s", "launch_cpu_s", "launch_critical_path_s",
        "host_device_overlap_s", "scheduler_plan_cpu_s",
        "scheduler_update_cpu_s", "execution_wall_s", "step_wall_s",
        "eager_dispatch_count", "graph_replay_count",
    )
    aggregate = {
        field: sum(row[field] for row in rows) for field in aggregate_fields
    }
    aggregate["scheduler_cpu_s"] = (
        aggregate["scheduler_plan_cpu_s"]
        + aggregate["scheduler_update_cpu_s"])
    aggregate["schedule_makespan_s"] = result.schedule_trace.end_time_s
    aggregate["step_count"] = len(rows)
    return {
        "schema_version": 1,
        "timing_mode": mode,
        "scheduler_measurement_boundary": (
            "in_scheduler_process_schedule_and_output_update_only"),
        "launch_measurement_boundary": "host_submit_only_no_device_sync",
        "profiles": (
            [] if host_launch_profiles is None else [
                host_launch_profiles[hardware_id].manifest()
                for hardware_id in sorted(host_launch_profiles)
            ]),
        "runtime_prepare_profiles": (
            [] if host_runtime_prepare_profiles is None else [
                host_runtime_prepare_profiles[hardware_id].manifest()
                for hardware_id in sorted(host_runtime_prepare_profiles)
            ]),
        "aggregate": aggregate,
        "steps": rows,
    }


def persistence_projection(payload: dict[str, object]) -> dict[str, object]:
    """Project every external request source before persistence."""

    if "request_source" not in payload:
        return payload
    request_source = payload["request_source"]
    if not isinstance(request_source, dict):
        raise TypeError("simulation request_source must be a dict")
    if request_source["kind"] == "production_trace":
        return production_trace_aggregate_result(payload)
    if request_source["kind"] == "canonical_requests":
        return canonical_requests_aggregate_result(payload)
    raise ValueError(
        f"unknown simulation request source kind: "
        f"{request_source['kind']!r}")


def production_trace_aggregate_result(
    payload: dict[str, object],
) -> dict[str, object]:
    """Project an in-memory Trace result to a request-safe persisted form."""

    if not isinstance(payload, dict):
        raise TypeError("production Trace aggregate projection requires a dict")
    if "request_source" not in payload:
        raise ValueError(
            "production Trace aggregate projection requires request_source")
    request_source = payload["request_source"]
    if not isinstance(request_source, dict):
        raise TypeError("production Trace request_source must be a dict")
    if request_source["kind"] != "production_trace":
        raise ValueError(
            "aggregate projection is reserved for production Trace results")
    prefix_alignment = request_source["prefix_alignment"]
    if not isinstance(prefix_alignment, dict):
        raise TypeError("production Trace prefix alignment must be a dict")
    alignment_entries = prefix_alignment["entries"]
    if not isinstance(alignment_entries, (list, tuple)):
        raise TypeError("production Trace prefix entries must be a sequence")
    alignment_totals = {
        "request_count": len(alignment_entries),
        "trace_prefix_hit_tokens": sum(
            entry["trace_prefix_hit_tokens"] for entry in alignment_entries),
        "scheduler_prefix_hit_tokens": sum(
            entry["scheduler_prefix_hit_tokens"]
            for entry in alignment_entries),
        "prefix_scheduler_delta_tokens": sum(
            entry["prefix_scheduler_delta_tokens"]
            for entry in alignment_entries),
    }
    if prefix_alignment["rule"] == "native_content_lookup_complete_blocks_last_prompt_token_recomputed":
        alignment_totals["content_prefix_hit_tokens"] = sum(
            entry["content_prefix_hit_tokens"] for entry in alignment_entries)
    aggregate_request_source = {
        "schema_version": request_source["schema_version"],
        "kind": request_source["kind"],
        "trace": request_source["trace"],
        "replay": request_source["replay"],
        "zero_output_contract": request_source["zero_output_contract"],
        "prefix_alignment": {
            "schema_version": prefix_alignment["schema_version"],
            "owner": prefix_alignment["owner"],
            "vllm_version": prefix_alignment["vllm_version"],
            "rule": prefix_alignment["rule"],
            "kv_block_size_tokens": (
                prefix_alignment["kv_block_size_tokens"]),
            "aggregate": alignment_totals,
            "request_level_entries_sha256": canonical_json_sha256(
                alignment_entries),
            "request_level_entries": "omitted",
        },
    }
    return request_safe_aggregate_result(
        payload,
        result_kind="production_trace_simulation_aggregate",
        projection="aggregate_only_production_trace_v1",
        aggregate_request_source=aggregate_request_source,
    )


def canonical_requests_aggregate_result(
    payload: dict[str, object],
) -> dict[str, object]:
    """Project canonical requests to a request-safe persisted result."""

    if not isinstance(payload, dict):
        raise TypeError(
            "canonical requests aggregate projection requires a dict")
    if "request_source" not in payload:
        raise ValueError(
            "canonical requests aggregate projection requires request_source")
    request_source = payload["request_source"]
    if not isinstance(request_source, dict):
        raise TypeError("canonical requests request_source must be a dict")
    if request_source["kind"] != "canonical_requests":
        raise ValueError(
            "canonical aggregate projection requires canonical requests")
    aggregate_request_source = {
        "schema_version": request_source["schema_version"],
        "kind": request_source["kind"],
        "input_path": request_source["input_path"],
        "source_path": request_source["source_path"],
        "source_sha256": request_source["source_sha256"],
        "source_size_bytes": request_source["source_size_bytes"],
        "input_schema": request_source["input_schema"],
        "row_count": request_source["row_count"],
        "request_id_policy": request_source["request_id_policy"],
        "equal_arrival_order": request_source["equal_arrival_order"],
        "normalized_request_stream_sha256": (
            request_source["normalized_request_stream_sha256"]),
    }
    return request_safe_aggregate_result(
        payload,
        result_kind="canonical_requests_simulation_aggregate",
        projection="aggregate_only_canonical_requests_v1",
        aggregate_request_source=aggregate_request_source,
    )


def request_safe_aggregate_result(
    payload: dict[str, object],
    *,
    result_kind: str,
    projection: str,
    aggregate_request_source: dict[str, object],
) -> dict[str, object]:
    """Omit request-level content from one externally sourced simulation."""

    execution_artifact = payload["execution_artifact"]
    if not isinstance(execution_artifact, dict):
        raise TypeError("simulation execution artifact must be a dict")
    aggregate_artifact = {
        "schema_version": execution_artifact["schema_version"],
        "hash_method": execution_artifact["hash_method"],
        "performance_timing_status": (
            execution_artifact["performance_timing_status"]),
        "counts": execution_artifact["counts"],
        "artifact_sha256": execution_artifact["artifact_sha256"],
        "request_level_artifact": "omitted",
    }
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("simulation metrics must be a dict")
    aggregate_metrics = {
        "request_throughput_per_s": metrics["request_throughput_per_s"],
        "output_token_throughput_per_s": (
            metrics["output_token_throughput_per_s"]),
        "token_throughput_per_s": metrics["token_throughput_per_s"],
        "latency_distributions": metrics["latency_distributions"],
        "slo_goodput": metrics["slo_goodput"],
        "system_resources_sha256": canonical_json_sha256(
            metrics["system_resources"]),
        "critical_path_sha256": canonical_json_sha256(
            metrics["critical_path"]),
        "request_level_metrics": "omitted",
    }
    kv_lifecycle = payload["kv_lifecycle_replay"]
    if kv_lifecycle is None:
        aggregate_kv_lifecycle = None
    else:
        aggregate_kv_lifecycle = {
            "manifest_sha256": canonical_json_sha256(kv_lifecycle),
            "request_level_manifest": "omitted",
        }
    mtp_outcomes = aggregate_mtp_acceptance_outcomes(
        payload["mtp_acceptance_outcomes"])
    mtp_pd_handoff = aggregate_mtp_pd_handoff(
        payload["mtp_pd_handoff_contract"])
    full_result_sha256 = canonical_json_sha256(payload)
    return {
        **{key: payload[key] for key in ("interactive_report", "report_output_costs") if key in payload},
        "schema_version": 1,
        "kind": result_kind,
        "persistence": {
            "projection": projection,
            "request_level_content": "omitted",
            "full_in_memory_result_sha256": full_result_sha256,
        },
        "version": payload["version"],
        "backend": payload["backend"],
        "deployment_id": payload["deployment_id"],
        "hardware_ids": payload["hardware_ids"],
        "model_compute_dtype": payload["model_compute_dtype"],
        "model_config_provenance": payload["model_config_provenance"],
        "deployment": payload["deployment"],
        "deployment_plan": payload["deployment_plan"],
        "routing_policy": payload["routing_policy"],
        "kv_handoff_mode": payload["kv_handoff_mode"],
        **({"pd_kv_lifecycle": {
            "event_count": len(payload["pd_kv_lifecycle"]),
            "events_sha256": canonical_json_sha256(payload["pd_kv_lifecycle"]),
            "request_level_manifest": "omitted",
        }} if "pd_kv_lifecycle" in payload else {}),
        "kv_capacity": payload["kv_capacity"],
        "kv_lifecycle_replay": aggregate_kv_lifecycle,
        "launch_policy": payload["launch_policy"],
        "prediction_method": payload["prediction_method"],
        "execution_mode": payload["execution_mode"],
        "execution_cache_stats": payload["execution_cache_stats"],
        "execution_parallelism": payload["execution_parallelism"],
        "serving_timing_mode": payload["serving_timing_mode"],
        "host_timing": payload["host_timing"],
        "performance_timing_status": payload["performance_timing_status"],
        "execution_provenance": payload["execution_provenance"],
        "workload_assumption_binding": (
            payload["workload_assumption_binding"]),
        "main_chain_status": payload["main_chain_status"],
        "execution_artifact": aggregate_artifact,
        "scheduler_progress": payload["scheduler_progress"],
        "scheduler_runtime": payload["scheduler_runtime"],
        "mtp_scenario": payload["mtp_scenario"],
        "mtp_acceptance_outcomes": mtp_outcomes,
        "mtp_pd_handoff_contract": mtp_pd_handoff,
        "arrival_process": payload["arrival_process"],
        "num_steps": payload["num_steps"],
        "num_transfers": payload["num_transfers"],
        "num_kernel_calls": payload["num_kernel_calls"],
        **({"source_activation_memory": aggregate_source_activation_memory(
            payload["source_activation_memory"])}
           if "source_activation_memory" in payload else {}),
        "schedule_end_time_s": payload["schedule_end_time_s"],
        "simulation_wall_time_s": payload["simulation_wall_time_s"],
        "simulation_started_monotonic_s": payload["simulation_started_monotonic_s"],
        "simulation_finished_monotonic_s": payload["simulation_finished_monotonic_s"],
        "evaluation_wall_time_breakdown_s": (
            payload["evaluation_wall_time_breakdown_s"]),
        "process_peak_rss_bytes": payload["process_peak_rss_bytes"],
        "metrics": aggregate_metrics,
        "request_source": aggregate_request_source,
        **({"adaptive_tail_worker_grants": payload["adaptive_tail_worker_grants"]}
           if "adaptive_tail_worker_grants" in payload else {}),
    }


def aggregate_mtp_acceptance_outcomes(
    value: object,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("MTP acceptance outcomes must be a dict or null")
    steps = value["steps"]
    if not isinstance(steps, (list, tuple)):
        raise TypeError("MTP acceptance outcome steps must be a sequence")
    totals = {
        "scheduled_request_events": 0,
        "target_model_workload_tokens": 0,
        "target_verification_tokens": 0,
        "proposer_tokens": 0,
        "proposer_workload_tokens": 0,
        "proposed_tokens": 0,
        "verified_tokens": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "emitted_tokens": 0,
    }
    for step in steps:
        requests = step["requests"]
        totals["scheduled_request_events"] += len(requests)
        for request in requests:
            scheduled_tokens = request["input_tokens"] + request["output_tokens"]
            totals["target_model_workload_tokens"] += scheduled_tokens
            totals["target_verification_tokens"] += (
                request["mtp_verified_tokens"])
            totals["proposer_tokens"] += request["mtp_proposer_tokens"]
            if request["mtp_proposer_tokens"]:
                totals["proposer_workload_tokens"] += (
                    scheduled_tokens + request["mtp_proposer_tokens"] - 1)
            totals["proposed_tokens"] += request["mtp_proposed_tokens"]
            totals["verified_tokens"] += request["mtp_verified_tokens"]
            totals["accepted_tokens"] += request["mtp_accepted_tokens"]
            totals["rejected_tokens"] += request["mtp_rejected_tokens"]
            totals["emitted_tokens"] += request["emitted_tokens"]
    if totals["verified_tokens"] != (
            totals["accepted_tokens"] + totals["rejected_tokens"]):
        raise RuntimeError(
            "aggregate MTP accepted/rejected token conservation failed")
    aggregate = {
        "schema_version": value["schema_version"],
        "source_kind": value["source_kind"],
        "source_id": value["source_id"],
        "step_count": len(steps),
        "totals": totals,
        "request_level_outcomes_sha256": canonical_json_sha256(value),
        "request_level_outcomes": "omitted",
    }
    if value["schema_version"] == 2:
        aggregate["scope"] = value["scope"]
        aggregate["scheduler_pools"] = value["scheduler_pools"]
    elif value["schema_version"] != 1:
        raise ValueError("unknown MTP acceptance outcomes schema version")
    return aggregate


def aggregate_mtp_pd_handoff(
    value: object,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("MTP PD handoff contract must be a dict or null")
    common_fields = {
        "schema_version",
        "contract_id",
        "scheduler_pools",
        "prefill_mtp",
        "decode_admission",
        "kv_handoff_mode",
        "kv_handoff_lanes",
        "decode_preemption_resume",
        "scheduler_reference",
        "decode_mtp",
        "validation_status",
        "requests",
    }
    schema_version = value["schema_version"]
    if schema_version == 1:
        expected_fields = common_fields
    elif schema_version == 2:
        expected_fields = common_fields | {
            "acceptance_replay",
            "ingress_request_stream_sha256",
            "decode_admission_contract_sha256",
            "decode_arrival_time_binding",
        }
    elif schema_version == 3:
        expected_fields = common_fields | {
            "acceptance_assumption",
            "ingress_request_stream_sha256",
            "decode_admission_contract_sha256",
            "decode_arrival_time_binding",
        }
    else:
        raise ValueError("unknown MTP PD handoff schema version")
    if set(value) != expected_fields:
        raise ValueError(
            "MTP PD handoff fields disagree with its schema")
    requests = value["requests"]
    if not isinstance(requests, (list, tuple)):
        raise TypeError("MTP PD handoff requests must be a sequence")
    numeric_fields = (
        "target_output_tokens",
        "prefill_emitted_tokens",
        "decode_emitted_tokens",
        "decode_raw_model_output_tokens",
        "decode_raw_output_steps",
        "decode_mtp_proposed_tokens",
        "decode_mtp_verified_tokens",
        "decode_mtp_accepted_tokens",
        "decode_mtp_rejected_tokens",
        "total_emitted_tokens",
    )
    totals = {
        field: sum(request[field] for request in requests)
        for field in numeric_fields
    }
    if totals["decode_mtp_verified_tokens"] != (
            totals["decode_mtp_accepted_tokens"]
            + totals["decode_mtp_rejected_tokens"]):
        raise RuntimeError("aggregate PD MTP token conservation failed")
    contract = {
        name: value[name]
        for name in expected_fields - {"requests"}
    }
    contract["aggregate_requests"] = {
        "request_count": len(requests),
        "totals": totals,
        "request_level_handoff_sha256": canonical_json_sha256(value),
        "request_level_handoff": "omitted",
    }
    return contract


def workload_assumption_binding_manifest(
    request_stream_identity: str,
    runtime_inputs: RuntimeInputs,
    args,
) -> dict[str, object]:
    """Bind independent request, MoE-routing, and MTP policy identities."""

    if (not isinstance(request_stream_identity, str)
            or len(request_stream_identity) != 64
            or any(character not in "0123456789abcdef"
                   for character in request_stream_identity)):
        raise ValueError("workload assumption binding requires request SHA256")
    routing = runtime_inputs.expert_routing.manifest()
    if routing["source_kind"] == "not_applicable":
        routing_identity = None
    elif routing["source_kind"] == "explicit_assumption":
        routing_identity = routing["policy_descriptor_sha256"]
    elif routing["source_kind"] == "profile_observation":
        routing_identity = routing["policy_descriptor_sha256"]
    elif routing["source_kind"] == "replay_observation":
        replay = routing["replay"]
        if not isinstance(replay, dict):
            raise RuntimeError(
                "expert routing replay manifest lost its replay payload")
        routing_identity = replay["canonical_payload_sha256"]
    else:
        raise ValueError(
            "unknown expert routing source kind in workload binding")
    if args.mtp_acceptance_policy == "disabled":
        mtp_identity = None
    elif args.mtp_acceptance_policy == "fixed_count":
        mtp_identity = canonical_json_sha256({
            "schema_version": 1,
            "acceptance_policy": "fixed_count",
            "num_speculative_tokens": args.mtp_num_speculative_tokens,
            "fixed_accepted_tokens": args.mtp_fixed_accepted_tokens,
            "acceptance_source_id": args.mtp_acceptance_source_id,
        })
    elif args.mtp_acceptance_policy == "maximum_entropy":
        assumption = args.mtp_acceptance_assumption
        if assumption is None:
            raise ValueError(
                "maximum-entropy workload binding requires an assumption")
        mtp_identity = assumption.policy_descriptor_sha256
    elif args.mtp_acceptance_policy == "replay":
        replay = args.mtp_acceptance_replay
        if replay is None:
            raise ValueError("replayed workload binding requires an artifact")
        mtp_identity = replay.source_sha256
    else:
        raise ValueError("unknown MTP policy in workload binding")
    components = {
        "request_stream_sha256": request_stream_identity,
        "expert_routing_policy_sha256": routing_identity,
        "mtp_acceptance_policy_sha256": mtp_identity,
    }
    return {
        "schema_version": 1,
        **components,
        "composite_sha256": canonical_json_sha256(components),
        "independence_contract": (
            "separate_request_routing_and_acceptance_components_v1"),
    }


def canonical_json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_execution_provenance(
    args,
    runtime_inputs: RuntimeInputs,
    hardware_by_id: dict[str, HWSpec],
    timing_status: str,
    *,
    repository_at_start: dict[str, object],
) -> dict[str, object]:
    if not isinstance(runtime_inputs, RuntimeInputs):
        raise TypeError("execution provenance requires RuntimeInputs")
    if not isinstance(hardware_by_id, dict) or not hardware_by_id:
        raise TypeError(
            "execution provenance requires a non-empty hardware mapping")
    hardware_ids = tuple(sorted(hardware_by_id))
    expected_timing_status = performance_timing_status(
        args.launch_policy,
        hardware_ids,
        uses_development_protobridge=uses_development_protobridge(args),
    )
    if timing_status != expected_timing_status:
        raise ValueError(
            "execution provenance performance timing status disagrees with "
            "launch policy and loaded hardware")
    for hardware_id in hardware_ids:
        hardware = hardware_by_id[hardware_id]
        if not isinstance(hardware, HWSpec):
            raise TypeError(
                "execution provenance hardware values must be HWSpec")
        if hardware.hardware_id != hardware_id:
            raise ValueError(
                "execution provenance hardware key and HWSpec ID disagree")
    if args.decode_hardware_config is None:
        decode_config = None
    else:
        decode_config = file_provenance(args.decode_hardware_config)
    hardware_configs = {
        "prefill": file_provenance(args.hardware_config),
        "decode": decode_config,
        "additional_colocated": tuple(
            file_provenance(path)
            for path in args.additional_colocated_hardware_config
        ),
    }
    launch_data = launch_data_provenance(hardware_ids, repository_root=ROOT)
    if args.prediction_method in (
            "midlayer", "roofline", "roofline_span_floor"):
        protobridge_profiles = {}
        uses_p2p = (
            (args.deployment == "pd_disagg" and args.kv_transfer_backend == "protobridge")
            or (args.collective_backend == "protobridge" and (
                args.prefill_pp_size > 1 or args.deployment in ("af_disagg", "pd_af_disagg"))))
        if uses_p2p:
            if args.deployment in ("pd_disagg", "pd_af_disagg"):
                p2p_link = args.transfer_link
            elif args.deployment == "af_disagg":
                p2p_link = args.activation_transfer_link
            else:
                # Pipeline P2P only: intra-node transfers share the
                # tensor-collective scale-up fabric.
                p2p_link = resolve_registered_profile(
                    development_profile_config_sha256(
                        hardware_ids[0], "all_reduce")).link
            protobridge_profiles["send_recv"] = (
                development_profile_config_sha256(
                    hardware_ids[0], "send_recv", link=p2p_link))
        if args.collective_backend == "analytical_roofline_v1":
            pass
        elif args.collective_backend == "protobridge":
            tensor_parallel_sizes = [args.prefill_tp_size]
            if args.decode_tp_size is not None:
                tensor_parallel_sizes.append(args.decode_tp_size)
            if args.ffn_tp_size is not None:
                tensor_parallel_sizes.append(args.ffn_tp_size)
            if any(size > 1 for size in tensor_parallel_sizes):
                protobridge_profiles.update({
                    "all_reduce": args.tensor_all_reduce_profile_sha256,
                    "all_gather": args.tensor_all_gather_profile_sha256,
                })
        else:
            raise ValueError(
                "unknown collective backend for provenance: "
                f"{args.collective_backend!r}")
        if protobridge_profiles:
            predictor_inputs = {
                "protobridge": {
                    "backend": "protobridge",
                    "source_kind": "development_only",
                    "profile_sha256_by_operation": protobridge_profiles,
                },
            }
        else:
            predictor_inputs = None
    elif args.prediction_method == "pipeweave":
        predictor_inputs = {
            "implementation": directory_provenance(
                ROOT / "pipeweave",
                allowed_suffixes=(".py",),
            ),
            "configs": directory_provenance(
                ROOT / "configs" / "baselines" / "pipeweave",
                allowed_suffixes=(".yaml", ".yml", ".json"),
            ),
        }
    else:
        raise ValueError(
            f"unknown prediction method for provenance: "
            f"{args.prediction_method!r}")
    runtime_manifest = {
        "expert_routing": runtime_inputs.expert_routing.manifest(),
        "attention_kv_layout": runtime_inputs.attention_kv_layout,
        "attention_kv_distribution": runtime_inputs.attention_kv_distribution,
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }
    if args.mtp_draft_token_ids_by_step is not None:
        runtime_manifest["mtp_draft_token_inputs"] = {
            "source": "explicit_step_request_drafts",
            "step_count": len(args.mtp_draft_token_ids_by_step),
            "validation": "native Scheduler checks depth, vocabulary, scheduled request IDs and accepted output prefix",
        }
    if args.model_source_normalization is not None:
        runtime_manifest["model_source_normalization"] = args.model_source_normalization
    if args.vllm_async_terminal_guard_source is not None:
        runtime_manifest["async_terminal_guard_source"] = (
            args.vllm_async_terminal_guard_source)
    if runtime_inputs.model_execution_version != "legacy":
        runtime_manifest["model_execution_version"] = runtime_inputs.model_execution_version
        if runtime_inputs.dense_source_input_policy is not None:
            runtime_manifest["dense_source_input_policy"] = {
                "value": runtime_inputs.dense_source_input_policy,
                "capture_limits": runtime_inputs.dense_source_capture_limits,
                "source_id": "declared-synthetic-canonical-lengths-zero-ids-and-tail-v2",
                "scope": "known backend IDs retained; unknown scheduled IDs and graph tails explicitly zero; embedding locality approximate, not observed content or native capture certification"}
        runtime_manifest["conditional_model_sources"] = {
            "v4_eager_metadata_source": runtime_inputs.v4_eager_metadata_source,
            "v4_source_max_model_len": runtime_inputs.v4_source_max_model_len,
            "v4_routing_source": runtime_inputs.v4_routing_source,
            "v4_source_use_deep_gemm": runtime_inputs.v4_source_use_deep_gemm,
            "target_sampling_source": runtime_inputs.target_sampling_source,
            "mtp_draft_sampling_source": runtime_inputs.mtp_draft_sampling_source,
        }
    if runtime_inputs.symbolic_forward_source is not None:
        runtime_manifest["symbolic_forward_source"] = runtime_inputs.symbolic_forward_source
        if runtime_inputs.bf16_expert_model is not None:
            runtime_manifest["bf16_expert_model"] = {
                "value": runtime_inputs.bf16_expert_model,
                "scope": "declared single-chunk eager BF16 EP2 Triton/default-config reference; captured mathematical aggregate retained; provider and workspace unbound",
                "source_reference": "existing versioned prepared expert source recipe and public Kernel predictors; external router retained once; no internal captured Host submissions",
            }
        if runtime_inputs.bf16_gemm_model is not None:
            runtime_manifest["bf16_gemm_model"] = {
                "value": runtime_inputs.bf16_gemm_model,
                "scope": "ordinary homogeneous unbiased BF16 Linear/head; fixed analytical tile program, native cuBLAS provider unobserved",
                "source_reference": "existing analytical_cublas_tiling and GPU pipeline resource model; no production duration inputs",
            }
        if runtime_inputs.norm_source_batch_invariant is not None:
            runtime_manifest["norm_source_batch_invariant"] = {
                "value": runtime_inputs.norm_source_batch_invariant,
                "scope": "explicit upstream default-source assumption; capture flag and native provider unobserved",
                "source_reference": "vLLM0.30 envs.py VLLM_BATCH_INVARIANT default 0; model_executor/layers/layernorm.py",
            }
        if runtime_inputs.maca_copy_runtime_source is not None:
            runtime_manifest["maca_copy_runtime_source"] = {
                "source": runtime_inputs.maca_copy_runtime_source,
                "scope": "declared native allocation and initially unset target-runtime flags; not historical provider attestation",
                "source_reference": "microbench/metax_c500/references/native_allocator_torch_2_4_maca_3_7_2.json",
            }
        runtime_manifest["symbolic_source_max_model_len"] = runtime_inputs.symbolic_source_max_model_len
        runtime_manifest["symbolic_source_max_batched_tokens"] = runtime_inputs.symbolic_source_max_batched_tokens
        runtime_manifest["symbolic_source_max_requests"] = runtime_inputs.symbolic_source_max_requests
        runtime_manifest["glm_indexer_selection_source"] = runtime_inputs.glm_indexer_selection_source
        runtime_manifest["glm_prefill_attention_source"] = runtime_inputs.glm_prefill_attention_source
    if runtime_inputs.symbolic_workspace_initial_capacity_bytes is not None:
        runtime_manifest["symbolic_workspace_initial_capacity_bytes"] = runtime_inputs.symbolic_workspace_initial_capacity_bytes
    if args.vllm_step_inputs is not None:
        runtime_manifest["vllm_step_source_inputs"] = {
            step_id: asdict(state) for step_id, state in args.vllm_step_inputs.items()}
        runtime_manifest["vllm_step_token_scope"] = (
            "explicit caller per-step scheduled/padding input scenarios; source_kind/source_id preserved; "
            "not inferred numerical model output or GPU observation")
    if args.glm_mtp_native_source_inputs:
        runtime_manifest["glm_mtp_native_source_inputs"] = {
            "placement": "D_only_P_target_prefill", "proposal_depth": args.mtp_num_speculative_tokens,
            "scope": "native length-only synthetic feedback; padding kept separate from real proposals; not numerical GLM output"}
    if args.glm_mtp_step_inputs is not None:
        runtime_manifest["glm_mtp_step_source_inputs"] = {
            step_id: asdict(state) for step_id, state in args.glm_mtp_step_inputs.items()}
        runtime_manifest["glm_mtp_initial_target_token_ids"] = tuple(args.glm_mtp_initial_target_token_ids)
        runtime_manifest["glm_mtp_token_scope"] = (
            "explicit caller draft scenarios; executed target IDs come from BackendStep's CPU length-only protocol; not numerical model output")
    if runtime_inputs.v4_source_max_batched_tokens is not None:
        runtime_manifest["conditional_model_sources"]["v4_source_max_batched_tokens"] = runtime_inputs.v4_source_max_batched_tokens
    if runtime_inputs.v4_indexer_selection_source is not None:
        runtime_manifest["conditional_model_sources"]["v4_indexer_selection_source"] = runtime_inputs.v4_indexer_selection_source
    return {
        "schema_version": 4,
        "repository": execution_repository_identity(repository_at_start),
        "hardware_configs": hardware_configs,
        "loaded_hardware_ids": hardware_ids,
        "performance_timing_status": timing_status,
        "runtime_inputs": runtime_manifest,
        "launch_inputs": {
            "launch_policy": args.launch_policy,
            "launch_data": launch_data,
        },
        "microbenchmark_inputs": {
            "runtime_artifact_loaded": False,
            "consumption": "materialized_parameters_in_hardware_config",
        },
        "prediction_method": args.prediction_method,
        "external_predictor_inputs": predictor_inputs,
    }


def file_provenance(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    try:
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            stream.seek(0)
            verification = hashlib.sha256()
            verified_size = 0
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                verification.update(chunk)
                verified_size += len(chunk)
            after = os.fstat(stream.fileno())
        named = resolved.stat()
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")

        def identity(stat_result):
            # Windows derives the mode permission bits from the read-only
            # attribute, so os.fstat and os.stat can disagree on st_mode for
            # the same unchanged file.  Compare the file-type bits only.
            values = tuple(getattr(stat_result, field) for field in fields)
            return values[:2] + (values[2] & stat.S_IFMT(values[2]),) + values[3:]

        if (identity(before) != identity(after)
                or identity(after) != identity(named)
                or size != after.st_size or verified_size != size
                or digest.digest() != verification.digest() or Path(path).resolve() != resolved):
            raise RuntimeError(f"provenance input changed while hashing: {resolved}")
    except OSError as error:
        raise RuntimeError(
            f"cannot read provenance input file: {resolved}") from error
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def launch_data_provenance(hardware_ids: tuple[str, ...], *, repository_root: Path = ROOT) -> dict:
    launch_data = {}
    for hardware_id in hardware_ids:
        # Each hardware owns its observed LUT. H200 may share analytical
        # launch code with A800, but its injected choices use H200 data.
        launch_data[hardware_id] = directory_provenance(
            repository_root / "launch_data" / hardware_id,
            allowed_suffixes=(".json", ".zst"),
        )
    return launch_data


def directory_provenance(
    path: str | Path,
    *,
    allowed_suffixes: tuple[str, ...],
) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise RuntimeError(
            f"provenance input directory does not exist: {resolved}")
    if (not isinstance(allowed_suffixes, tuple)
            or not allowed_suffixes
            or any(not isinstance(suffix, str) or not suffix
                   for suffix in allowed_suffixes)
            or ("*" in allowed_suffixes and allowed_suffixes != ("*",))):
        raise ValueError(
            "provenance allowed_suffixes must be non-empty strings")
    files = tuple(
        candidate
        for candidate in sorted(resolved.rglob("*"))
        if candidate.is_file() and (allowed_suffixes == ("*",) or candidate.suffix in allowed_suffixes)
    )
    if not files:
        raise RuntimeError(
            f"provenance input directory has no selected files: {resolved}")
    entries = []
    for file_path in files:
        file_manifest = file_provenance(file_path)
        entries.append({
            "relative_path": file_path.relative_to(resolved).as_posix(),
            "sha256": file_manifest["sha256"],
            "size_bytes": file_manifest["size_bytes"],
        })
    serialized = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "path": str(resolved),
        "tree_sha256": hashlib.sha256(serialized).hexdigest(),
        "file_count": len(entries),
        "files": tuple(entries),
    }


def repository_provenance() -> dict[str, object]:
    return git_code_identity(ROOT)


def vllm_step_index(step: BackendStep) -> int:
    if not isinstance(step, BackendStep):
        raise TypeError("vLLM step index requires BackendStep")
    parts = step.schedule.step_id.rsplit(":", 2)
    if len(parts) != 3 or not parts[0] or parts[1] != "vllm":
        raise ValueError(
            "MTP outcome requires canonical <pool>:vllm:<index> step ID")
    try:
        index = int(parts[2])
    except ValueError as error:
        raise ValueError(
            "MTP outcome step index must be an integer") from error
    if str(index) != parts[2] or index < 0:
        raise ValueError(
            "MTP outcome step index must be canonical and non-negative")
    return index


def mtp_acceptance_outcomes_manifest(args, result, deployment) -> dict | None:
    if args.mtp_acceptance_policy == "disabled":
        return None
    if args.mtp_acceptance_policy in ("fixed_count", "maximum_entropy"):
        source_kind = "explicit_assumption"
    elif args.mtp_acceptance_policy == "replay":
        source_kind = "replay_observation"
    else:
        raise ValueError(
            "unknown MTP acceptance policy for outcomes: "
            f"{args.mtp_acceptance_policy!r}")
    if args.deployment == "colocated":
        scheduler_pools = tuple(sorted(deployment.worker_pools))
        if isinstance(result, StreamingSimulationResult):
            selected_backend_steps = tuple(
                entry.backend_step for entry in result.schedule_trace.entries)
        else:
            selected_backend_steps = tuple(
                simulation_step.backend_step
                for simulation_step in result.steps
            )
        schema_version = 1
        scope = None
    elif args.deployment == "af_disagg":
        scheduler_pools = tuple(sorted(
            pool_name
            for pool_name, pool in deployment.worker_pools.items()
            if pool.role == "attention"
        ))
        if len(scheduler_pools) != 1:
            raise RuntimeError(
                "the validated AF MTP outcome scope requires one Attention "
                "pool")
        selected_backend_steps = tuple(
            simulation_step.backend_step for simulation_step in result.steps)
        if any(
                backend_step.schedule.pool_name
                != scheduler_pools[0]
                for backend_step in selected_backend_steps):
            raise RuntimeError(
                "AF MTP outcomes contain a non-Attention Scheduler step")
        schema_version = 1
        scope = None
    elif args.deployment in ("pd_disagg", "pd_af_disagg"):
        scheduler_pools = tuple(sorted(
            pool_name
            for pool_name, pool in deployment.worker_pools.items()
            if (worker_role_phase(pool.role) == "decode"
                and worker_role_operator_group(pool.role) != "ffn")
        ))
        if len(scheduler_pools) != 1:
            raise RuntimeError(
                "the validated PD MTP outcome scope requires one decode pool")
        if isinstance(result, StreamingSimulationResult):
            selected_backend_steps = tuple(
                entry.backend_step
                for entry in result.schedule_trace.entries
                if entry.backend_step.schedule.pool_name in scheduler_pools
            )
        else:
            selected_backend_steps = tuple(
                simulation_step.backend_step
                for simulation_step in result.steps
                if simulation_step.backend_step.schedule.pool_name
                in scheduler_pools
            )
        schema_version = 2
        scope = "decode_only"
    else:
        raise RuntimeError(
            "enabled MTP outcome reporting does not support deployment "
            f"{args.deployment!r}")
    manifest = {
        "schema_version": schema_version,
        "source_kind": source_kind,
        "source_id": args.mtp_acceptance_source_id,
        "steps": mtp_acceptance_steps_manifest(
            mtp_acceptance_step_from_backend_step(
                vllm_step_index(backend_step),
                backend_step,
            )
            for backend_step in selected_backend_steps
        ),
    }
    if scope is not None:
        manifest["scope"] = scope
        manifest["scheduler_pools"] = list(scheduler_pools)
    return manifest


def mtp_pd_handoff_contract(
    args,
    requests: RequestStream,
    result,
    deployment,
    scheduler_runtime: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    if args.mtp_num_speculative_tokens == 0:
        return None
    if (args.deployment not in ("pd_disagg", "pd_af_disagg")
            or args.backend != "vllm"):
        return None
    prefill_pools = tuple(sorted(
        pool_name
        for pool_name, pool in deployment.worker_pools.items()
        if (worker_role_phase(pool.role) == "prefill"
            and worker_role_operator_group(pool.role) != "ffn")
    ))
    decode_pools = tuple(sorted(
        pool_name
        for pool_name, pool in deployment.worker_pools.items()
        if (worker_role_phase(pool.role) == "decode"
            and worker_role_operator_group(pool.role) != "ffn")
    ))
    if len(prefill_pools) != 1 or len(decode_pools) != 1:
        raise RuntimeError(
            "the validated PD MTP handoff contract requires one P and one D")
    prefill_pool = prefill_pools[0]
    decode_pool = decode_pools[0]
    prefill_runtime = scheduler_runtime[prefill_pool]
    decode_runtime = scheduler_runtime[decode_pool]
    if (prefill_runtime["speculative_method"] != "disabled"
            or prefill_runtime["mtp_num_speculative_tokens"] != 0
            or prefill_runtime["mtp_acceptance_policy"] != "disabled"):
        raise RuntimeError(
            "PD MTP requires the prefill Scheduler to disable MTP")
    if (decode_runtime["speculative_method"] != "mtp"
            or decode_runtime["mtp_num_speculative_tokens"]
            != args.mtp_num_speculative_tokens
            or decode_runtime["mtp_acceptance_source_id"]
            != args.mtp_acceptance_source_id):
        raise RuntimeError(
            "PD MTP decode Scheduler runtime disagrees with the requested "
            "scenario")
    if args.mtp_acceptance_policy == "fixed_count":
        if (decode_runtime["mtp_acceptance_policy"] != "fixed_count"
                or decode_runtime["mtp_fixed_accepted_tokens"]
                != args.mtp_fixed_accepted_tokens
                or decode_runtime["mtp_acceptance_replay_path"] is not None
                or args.mtp_acceptance_replay is not None):
            raise RuntimeError(
                "PD MTP decode Scheduler runtime disagrees with the requested "
                "fixed-count scenario")
        replay_manifest = None
        assumption_manifest = None
    elif args.mtp_acceptance_policy == "maximum_entropy":
        assumption = args.mtp_acceptance_assumption
        if (assumption is None
                or decode_runtime["mtp_acceptance_policy"]
                != "maximum_entropy"
                or decode_runtime["mtp_fixed_accepted_tokens"] != 0
                or decode_runtime["mtp_acceptance_replay_path"] is not None
                or decode_runtime["mtp_acceptance_assumption_sha256"]
                != assumption.policy_descriptor_sha256
                or decode_runtime["mtp_request_stream_sha256"]
                != assumption.request_stream_sha256):
            raise RuntimeError(
                "PD MTP decode Scheduler runtime disagrees with the requested "
                "maximum-entropy scenario")
        replay_manifest = None
        assumption_manifest = assumption.manifest()
    elif args.mtp_acceptance_policy == "replay":
        replay = args.mtp_acceptance_replay
        if (replay is None
                or decode_runtime["mtp_acceptance_policy"] != "replay"
                or decode_runtime["mtp_fixed_accepted_tokens"] != 0
                or decode_runtime["mtp_acceptance_replay_path"]
                != replay.source_path
                or decode_runtime["mtp_acceptance_replay_sha256"]
                != replay.source_sha256
                or decode_runtime["mtp_acceptance_replay_artifact_id"]
                != replay.artifact_id
                or decode_runtime["mtp_acceptance_replay_step_count"]
                != len(replay.steps)
                or decode_runtime["mtp_request_stream_sha256"]
                != replay.scheduler_context.ingress_identity_sha256):
            raise RuntimeError(
                "PD MTP decode Scheduler runtime disagrees with the requested "
                "replay scenario")
        replay_manifest = mtp_acceptance_replay_manifest(replay)
        assumption_manifest = None
    else:
        raise RuntimeError(
            "PD MTP handoff received an unsupported acceptance policy")

    requests_by_id = {
        request.request_id: request for request in requests
    }
    accounting = {
        request.request_id: {
            "prefill_emitted_tokens": 0,
            "decode_emitted_tokens": 0,
            "decode_raw_model_output_tokens": 0,
            "decode_raw_output_steps": 0,
            "decode_mtp_proposed_tokens": 0,
            "decode_mtp_verified_tokens": 0,
            "decode_mtp_accepted_tokens": 0,
            "decode_mtp_rejected_tokens": 0,
        }
        for request in requests
    }
    transfers_by_request = {
        request.request_id: [] for request in requests
    }
    for transfer in result.schedule_trace.transfers:
        if (transfer.payload != "kv"
                or transfer.source_pool_id != prefill_pool
                or transfer.destination_pool_id != decode_pool):
            continue
        for request_id in transfer.request_ids:
            if request_id not in transfers_by_request:
                raise RuntimeError(
                    f"PD MTP transferred unknown request {request_id!r}")
            transfers_by_request[request_id].append(transfer)
    decode_entry_starts_by_request = {
        request.request_id: [] for request in requests
    }
    for entry in result.schedule_trace.entries:
        if entry.backend_step.schedule.pool_name != decode_pool:
            continue
        for scheduled in entry.backend_step.schedule.requests:
            request_id = scheduled.request_id
            if request_id not in decode_entry_starts_by_request:
                raise RuntimeError(
                    f"PD MTP decode trace contains unknown request "
                    f"{request_id!r}")
            decode_entry_starts_by_request[request_id].append(
                (entry.start_time_s, entry.end_time_s))
    first_decode_seen: set[str] = set()
    # Decode-pool preemption frees a resident request's KV; the request later
    # resumes by recomputing its prefix locally in the decode pool (the
    # prefill-side KV was released after the handoff, so no re-transfer is
    # possible).  Those resume rows are prefill-phase work in the decode pool
    # and are exempt from the decode-only row contract below; first-admission
    # semantics still apply to every request's initial decode row.
    decode_preempted_ids: set[str] = set()
    if isinstance(result, StreamingSimulationResult):
        # Streaming execution consumes and releases the heavy SimulationStep
        # IR, but the canonical schedule trace retains each BackendStep.  The
        # PD MTP handoff contract only needs scheduler accounting from that
        # BackendStep, so validate the same contract against the trace rather
        # than requiring materialized per-step execution artifacts.
        backend_steps = tuple(
            entry.backend_step for entry in result.schedule_trace.entries)
    else:
        backend_steps = tuple(
            simulation_step.backend_step for simulation_step in result.steps)
    for backend_step in backend_steps:
        pool_name = backend_step.schedule.pool_name
        if pool_name == prefill_pool:
            for scheduled in backend_step.schedule.requests:
                request_id = scheduled.request_id
                if request_id not in accounting:
                    raise RuntimeError(
                        f"PD MTP scheduled unknown request {request_id!r}")
                if scheduled.mtp_acceptance_source_kind != "disabled":
                    raise RuntimeError(
                        "PD MTP prefill Scheduler produced speculative state "
                        f"for {request_id!r}")
                accounting[request_id]["prefill_emitted_tokens"] += (
                    backend_step.generated_tokens[request_id])
        elif pool_name == decode_pool:
            decode_preempted_ids.update(
                backend_step.schedule.preempted_request_ids)
            for scheduled in backend_step.schedule.requests:
                request_id = scheduled.request_id
                if request_id not in accounting:
                    raise RuntimeError(
                        f"PD MTP scheduled unknown request {request_id!r}")
                emitted = backend_step.generated_tokens[request_id]
                if ((scheduled.phase != "decode" or scheduled.input_tokens != 0)
                        and request_id not in decode_preempted_ids):
                    raise RuntimeError(
                        "PD MTP decode pool scheduled non-decode work for "
                        f"{request_id!r}")
                if request_id not in first_decode_seen:
                    request = requests_by_id[request_id]
                    if (scheduled.computed_tokens != request.input_tokens
                            or scheduled.output_tokens != 1
                            or scheduled.raw_model_output_tokens != 1
                            or emitted != 1
                            or scheduled.trace_prefix_hit_tokens != 0
                            or scheduled.scheduler_prefix_hit_tokens != 0
                            or scheduled.resumed_from_preemption
                            or any((
                                scheduled.mtp_proposed_tokens,
                                scheduled.mtp_verified_tokens,
                                scheduled.mtp_accepted_tokens,
                                scheduled.mtp_rejected_tokens,
                            ))):
                        raise RuntimeError(
                            "PD MTP decode admission must start from full prompt "
                            "KV, one terminal seed, and empty draft state for "
                            f"{request_id!r}")
                    first_decode_seen.add(request_id)
                accounting[request_id]["decode_emitted_tokens"] += emitted
                accounting[request_id][
                    "decode_raw_model_output_tokens"] += (
                        scheduled.raw_model_output_tokens)
                if scheduled.raw_model_output_tokens:
                    accounting[request_id]["decode_raw_output_steps"] += 1
                accounting[request_id]["decode_mtp_proposed_tokens"] += (
                    scheduled.mtp_proposed_tokens)
                accounting[request_id]["decode_mtp_verified_tokens"] += (
                    scheduled.mtp_verified_tokens)
                accounting[request_id]["decode_mtp_accepted_tokens"] += (
                    scheduled.mtp_accepted_tokens)
                accounting[request_id]["decode_mtp_rejected_tokens"] += (
                    scheduled.mtp_rejected_tokens)
        else:
            raise RuntimeError(
                f"PD MTP produced a step for unexpected pool {pool_name!r}")

    request_manifests = []
    for request in requests:
        counts = accounting[request.request_id]
        emitted_tokens = (
            counts["prefill_emitted_tokens"]
            + counts["decode_emitted_tokens"])
        if emitted_tokens != request.output_tokens:
            raise RuntimeError(
                "PD MTP emitted-token conservation failed for "
                f"{request.request_id!r}: expected={request.output_tokens}, "
                f"actual={emitted_tokens}")
        if (counts["decode_mtp_verified_tokens"]
                != counts["decode_mtp_accepted_tokens"]
                + counts["decode_mtp_rejected_tokens"]):
            raise RuntimeError(
                "PD MTP accepted/rejected token conservation failed for "
                f"{request.request_id!r}")
        if (counts["decode_raw_model_output_tokens"]
                != counts["decode_mtp_accepted_tokens"]
                + counts["decode_raw_output_steps"]):
            raise RuntimeError(
                "PD MTP raw output does not equal accepted draft prefixes plus "
                f"terminal tokens for {request.request_id!r}")
        if request.request_id in first_decode_seen:
            if counts["prefill_emitted_tokens"] != 1:
                raise RuntimeError(
                    "PD MTP decode admission requires one prefill terminal "
                    f"seed for {request.request_id!r}")
            transfers = transfers_by_request[request.request_id]
            expected_transfer_lanes = len(
                deployment.worker_pools[prefill_pool].ranks)
            if len(transfers) != expected_transfer_lanes:
                raise RuntimeError(
                    "PD MTP requires one P-to-D KV transfer per source rank "
                    f"for {request.request_id!r}: expected="
                    f"{expected_transfer_lanes}, actual={len(transfers)}")
            if len({transfer.transfer_id for transfer in transfers}) != len(
                    transfers):
                raise RuntimeError(
                    "PD MTP P-to-D KV transfer IDs must be unique for "
                    f"{request.request_id!r}")
            decode_entry_intervals = decode_entry_starts_by_request[
                request.request_id]
            latest_transfer_end = max(
                transfer.end_time_s for transfer in transfers)
            if (not decode_entry_intervals
                    or min(
                        start_time for start_time, _ in decode_entry_intervals)
                    < latest_transfer_end):
                raise RuntimeError(
                    "PD MTP decode scheduling started before KV transfer "
                    f"completion for {request.request_id!r}: "
                    f"decode_intervals={decode_entry_intervals!r}, "
                    f"latest_transfer_end={latest_transfer_end!r}")
        elif counts["decode_emitted_tokens"] != 0:
            raise RuntimeError(
                "PD MTP decode accounting lacks an admission step for "
                f"{request.request_id!r}")
        elif transfers_by_request[request.request_id]:
            raise RuntimeError(
                "PD MTP transferred a request that never entered decode for "
                f"{request.request_id!r}")
        request_manifests.append({
            "request_id": request.request_id,
            "target_output_tokens": request.output_tokens,
            **counts,
            "total_emitted_tokens": emitted_tokens,
        })
    common_contract = {
        "scheduler_pools": {
            "prefill": prefill_pool,
            "decode": decode_pool,
        },
        "prefill_mtp": "disabled",
        "decode_admission": {
            "computed_tokens": "full_prompt_length",
            "terminal_seed_tokens": 1,
            "initial_draft_tokens": 0,
            "prefix_hit_reinjection": False,
            "requires_completed_kv_transfer": True,
        },
        "kv_handoff_mode": "metadata_only_private_destination_kv",
        "kv_handoff_lanes": "one_request_scoped_transfer_per_prefill_rank",
        "decode_preemption_resume": (
            "local_prefix_recompute_after_handoff_kv_release"),
        "scheduler_reference": {
            "vllm_version": decode_runtime["version"],
            "vllm_source_commit": decode_runtime["source_commit"],
            "vllm_package_manifest_sha256": (
                decode_runtime["package_python_manifest_sha256"]),
        },
    }
    if assumption_manifest is not None:
        common_contract["scheduler_reference"]["alignment_gate"] = (
            "deterministic_maximum_entropy_0_to_L")
        return {
            "schema_version": 3,
            "contract_id": (
                "vllm_0_11_maximum_entropy_decode_fresh_mtp_pd_v1"),
            **common_contract,
            "decode_mtp": "maximum_entropy_fresh_state",
            "acceptance_assumption": assumption_manifest,
            "ingress_request_stream_sha256": (
                assumption_manifest["request_stream_sha256"]),
            "decode_admission_contract_sha256": (
                pd_decode_admission_contract_sha256(requests)),
            "decode_arrival_time_binding": (
                "projected_transfer_completion_not_assumption_identity"),
            "validation_status": {
                "scheduler_control_plane": (
                    "validated_pinned_vllm_reference"),
                "acceptance": "projected_explicit_assumption",
                "kernel_and_transfer_timing": "projected",
            },
            "requests": request_manifests,
        }
    if replay_manifest is None:
        common_contract["scheduler_reference"]["alignment_gate"] = (
            "seeded_fixed_count_0_1_L")
        return {
            "schema_version": 1,
            "contract_id": "vllm_0_11_seeded_decode_fresh_mtp_pd_v1",
            **common_contract,
            "decode_mtp": "fixed_count_fresh_state",
            "validation_status": {
                "scheduler_control_plane": (
                    "validated_pinned_vllm_reference"),
                "acceptance": "projected_explicit_assumption",
                "kernel_and_transfer_timing": "projected",
            },
            "requests": request_manifests,
        }
    common_contract["scheduler_reference"]["alignment_gate"] = (
        "seeded_decode_replay_contract_0_1_L")
    return {
        "schema_version": 2,
        "contract_id": "vllm_0_11_seeded_decode_replay_mtp_pd_v2",
        **common_contract,
        "decode_mtp": "replay_fresh_state",
        "acceptance_replay": replay_manifest,
        "ingress_request_stream_sha256": (
            args.mtp_acceptance_replay.scheduler_context.
            ingress_request_stream_sha256),
        "decode_admission_contract_sha256": (
            args.mtp_acceptance_replay.scheduler_context.
            decode_admission_contract_sha256),
        "decode_arrival_time_binding": (
            "projected_transfer_completion_not_replay_identity"),
        "validation_status": {
            "scheduler_control_plane": "validated_pinned_vllm_reference",
            "acceptance": "replayed_external_observation",
            "kernel_and_transfer_timing": "projected",
        },
        "requests": request_manifests,
    }


def load_request_source(args) -> tuple[
    RequestStream,
    dict[str, object],
    str | None,
    dict[str, object] | None,
    int,
]:
    if args.workload is not None:
        artifact = load_workload_artifact(args.workload)
        arrival_times, arrival_process = arrival_schedule(args, artifact)
        requests = RequestStream(
            RequestSpec(
                arrival_time,
                request_id,
                prompt_length,
                artifact.token_controls["max_tokens"],
                0,
            )
            for arrival_time, request_id, prompt_length in zip(
                arrival_times,
                artifact.request_ids,
                artifact.prompt_lengths,
            )
        )
        return requests, arrival_process, artifact.hash, None, 2
    elif args.production_trace is not None:
        trace = load_production_trace(
            args.production_trace,
            production_trace_reader_config(args),
        )
        require_expected_sha256(
            "production Trace",
            args.trace_source_sha256,
            trace.manifest.source_sha256,
        )
        if args.trace_replay_mode == "original":
            replay = build_production_trace_replay(
                trace, arrival_rate_scale=None)
        elif args.trace_replay_mode == "arrival_rate_scale":
            replay = build_production_trace_replay(
                trace,
                arrival_rate_scale=args.trace_arrival_rate_scale,
            )
        else:
            raise ValueError(
                f"unknown production Trace replay mode: "
                f"{args.trace_replay_mode!r}")
        replay_manifest = production_trace_replay_manifest(replay)
        request_source = {
            "schema_version": 1,
            "kind": "production_trace",
            "trace": production_trace_manifest(trace),
            "replay": replay_manifest,
            "zero_output_contract": {
                "mode": "pinned_vllm_raw_terminal_tail_truncation_v1",
                "raw_terminal_tokens": 1,
                "emitted_tokens": 0,
                "prompt_workload": "scheduled",
                "ttft_tpot": "not_applicable_null",
            },
        }
        return replay.requests, replay_manifest, None, request_source, 3
    elif args.canonical_requests is not None:
        source_path = Path(args.canonical_requests)
        try:
            source_bytes = source_path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"cannot read canonical requests source: {source_path}") from error
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        require_expected_sha256(
            "canonical requests",
            args.canonical_requests_sha256,
            source_sha256,
        )
        requests = parse_canonical_requests_bytes(
            source_bytes,
            source_path=source_path,
        )
        normalized_sha256 = request_stream_sha256(requests)
        arrival_process = {
            "schema_version": 1,
            "kind": "recorded_canonical_arrivals",
            "request_stream_sha256": normalized_sha256,
            "equal_arrival_order": CANONICAL_EQUAL_ARRIVAL_ORDER,
        }
        if args.closed_loop_client_count is not None:
            if (len(requests) % args.closed_loop_client_count) != 0:
                raise ValueError(
                    "closed-loop requires the canonical request count to be "
                    f"a multiple of the client count: {len(requests)} "
                    f"requests, {args.closed_loop_client_count} clients")
            arrival_process = {
                "schema_version": 1,
                "kind": "closed_loop_zero_think_time",
                "request_stream_sha256": normalized_sha256,
                "client_count": args.closed_loop_client_count,
                "requests_per_client": (
                    len(requests) // args.closed_loop_client_count),
                "chain_assignment": "interleaved",
                "think_time_s": 0.0,
                "equal_arrival_order": CANONICAL_EQUAL_ARRIVAL_ORDER,
            }
        request_source = {
            "schema_version": 1,
            "kind": "canonical_requests",
            "input_path": str(source_path),
            "source_path": str(source_path.resolve()),
            "source_sha256": source_sha256,
            "source_size_bytes": len(source_bytes),
            "input_schema": {
                "schema_version": 1,
                "format": "csv",
                "columns": CANONICAL_REQUEST_COLUMNS,
            },
            "row_count": len(requests),
            "request_id_policy": CANONICAL_REQUEST_ID_POLICY,
            "equal_arrival_order": CANONICAL_EQUAL_ARRIVAL_ORDER,
            "normalized_request_stream_sha256": normalized_sha256,
        }
        return requests, arrival_process, None, request_source, 3
    else:
        raise AssertionError("request source arguments were not normalized")


def aggregate_source_activation_memory(memory: dict[str, object]) -> dict[str, object]:
    """Keep rank peaks without step identities or source allocation names.

    The in-memory result retains its scalar series and live-storage witnesses.
    External request persistence deliberately omits those per-step details,
    just as it omits the request-level timeline and execution artifact.
    """
    if not isinstance(memory, dict):
        raise TypeError("source activation memory aggregate requires a dict")
    return {name: memory[name] for name in (
        "reported_steps", "total_simulation_steps", "all_steps_reported",
        "complete", "normal_capacity_accepted", "scope")} | {
        "retention": "pool_rank_peaks_only_request_level_detail_omitted",
        "pools": {pool_name: {"source_profile": pool["source_profile"],
            "ranks": {rank: {"known_activation_peak_bytes": record["known_activation_peak_bytes"]}
                for rank, record in pool["ranks"].items()}}
            for pool_name, pool in memory["pools"].items()}}



def execution_repository_identity(repository_at_start: dict[str, object]) -> dict[str, object]:
    """Observe the worktree at both boundaries, without claiming loaded code.

    Python imports can precede the entry observation or occur during a run.
    Equal endpoint identities do not prove the absence of transient changes,
    nor that every imported module came from either observed worktree.
    """
    repository_at_end = repository_provenance()
    return {
        **repository_at_end,
        "identity_recorded_at": "run_end",
        "run_start_identity": dict(repository_at_start),
        "identity_changed_between_boundaries": repository_at_start != repository_at_end,
        "identity_observation_scope": "entry_and_exit_only_not_continuous",
        "exact_imported_code_snapshot_binding": "unbound",
    }



def source_activation_capacity_manifest(memory, plans, policy, deployment, model, hardware_by_id,
        source_workspace_budget=None):
    """Join executed source lifetimes to existing budgets without filling gaps."""
    if memory is None:
        return {"status": ("not_applicable_debug_capacity" if policy is None else "unavailable_source_activation_report"),
            "normal_capacity_accepted": False}
    ffn = validate_ffn_source_capacity(memory, model=model, deployment=deployment, hardware_by_id=hardware_by_id)
    known_ffn = {"ffn_known_source_capacity": ffn} if ffn["pools"] else {}
    if policy is None:
        return {"status": "not_applicable_debug_capacity", "normal_capacity_accepted": False, **known_ffn}
    source_pools = set(memory["pools"])
    planned_pools = set(plans)
    if source_pools != planned_pools:
        # AF FFN workers have source lifetimes but no Scheduler KV plan. Keep
        # this missing ledger explicit without blocking analytical timing.
        return {"status": "unsupported_source_pool_capacity_coverage",
            "source_pools_without_capacity_plan": sorted(source_pools - planned_pools),
            "capacity_pools_without_source_report": sorted(planned_pools - source_pools),
            "normal_capacity_accepted": False, **known_ffn}
    return {"status": "executed_source_peaks_fit_existing_budget",
        **validate_source_activation_capacity(memory, plans=plans.values(), policy_config=policy,
            pool_ranks={name: deployment.worker_pools[name].ranks for name in plans},
            source_workspace_budget=source_workspace_budget), **known_ffn}
