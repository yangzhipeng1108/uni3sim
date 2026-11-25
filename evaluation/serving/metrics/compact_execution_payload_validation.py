"""Execution payload validation for compact ShareGPT case results:
execution artifacts, capacity checks, cache statistics, wall time, and
Scheduler runtime identity."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import types
from typing import get_args, get_origin, get_type_hints, Mapping

from evaluation.serving.experiments.experiment_manifest import (
    resolve_runtime_memory_policy_path,
)
from src.serving.backend.vllm_scheduler_process import (
    VLLMSchedulerRuntimeInfo,
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
    vllm_scheduler_runtime_sha256,
    vllm_scheduler_source_identity,
)
from src.serving.workload.model_config import load_model_config

from evaluation.reporting.standard_evaluation_report import (
    MIDLAYER_WALL_TIME_PHASES,
)

from evaluation.json_shape import require_object_value as _mapping

from evaluation.serving.metrics.compact_shape_validators import (
    _exact_fields,
    _sequence,
    _string,
    _sha256,
    _integer,
    _finite_number,
)
from evaluation.serving.metrics.compact_case_contract import (
    Phase1Config,
    Phase1Cell,
    _model_config_path,
)
from evaluation.serving.metrics.compact_case_contract import _GIT_COMMIT


EXECUTION_ARTIFACT_FIELDS = {
    "schema_version",
    "hash_method",
    "performance_timing_status",
    "counts",
    "schedule_trace",
    "steps",
    "transfers",
    "modeled_kv_offload",
    "artifact_sha256",
}
_STREAMING_EXECUTION_ARTIFACT_FIELDS = {
    "schema_version",
    "hash_method",
    "performance_timing_status",
    "counts",
    "streaming_aggregate",
    "artifact_sha256",
}
_EXECUTION_ARTIFACT_COUNT_FIELDS = {
    "schedule_steps",
    "simulation_steps",
    "simulation_transfers",
    "trace_transfers",
    "modeled_kv_offload_transfers",
    "kernel_calls",
}
_EXECUTION_ARTIFACT_HASH_METHOD = "canonical_json_sort_keys_compact_utf8_v1"
STREAMING_EXECUTION_ARTIFACT_HASH_METHODS = {
    9: "ordered_step_manifest_sha256_chain_v1",
    10: "ordered_exact_execution_contract_sha256_chain_v2",
    11: "ordered_exact_execution_contract_sha256_chain_v3",
    12: "ordered_exact_execution_contract_sha256_chain_v4",
    13: "ordered_exact_execution_contract_sha256_chain_v5",
    14: "ordered_exact_execution_contract_sha256_chain_v6",
    16: "ordered_exact_execution_contract_sha256_chain_v6",
}
_MAIN_CHAIN = (
    "RequestStream -> SchedulerBackend/BackendStep/ScheduleStep "
    "-> build_workload -> WorkloadGraph -> lower_workload -> "
    "KernelSpec -> predict"
)


def _validate_midlayer_wall_time(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    simulation_wall_time_s = _finite_number(
        payload["simulation_wall_time_s"], "simulation wall time"
    )
    if simulation_wall_time_s < 0.0:
        raise ValueError("simulation wall time must be nonnegative")
    raw_phases = _mapping(
        payload["evaluation_wall_time_breakdown_s"],
        "evaluation wall time breakdown",
    )
    if set(raw_phases) != set(MIDLAYER_WALL_TIME_PHASES):
        raise ValueError("evaluation wall time breakdown fields are invalid")
    phases = {
        name: _finite_number(raw_phases[name], f"{name} wall time")
        for name in MIDLAYER_WALL_TIME_PHASES
    }
    if any(value < 0.0 for value in phases.values()):
        raise ValueError("evaluation wall time breakdown must be nonnegative")
    tolerance = max(1e-9, simulation_wall_time_s * 1e-6)
    if abs(sum(phases.values()) - simulation_wall_time_s) > tolerance:
        raise ValueError(
            "evaluation wall time breakdown does not sum to simulation wall time"
        )
    return {
        "schema_version": 1,
        "scope": "midlayer_simulation_cpu_wall",
        "simulation_wall_time_s": simulation_wall_time_s,
        "phase_wall_time_s": phases,
    }


def _validate_debug_capacity(
    repository_root: str | Path,
    capacity_value: object,
    config: Phase1Config,
    cell: Phase1Cell,
    expected_num_kv_blocks: int,
) -> dict[str, object]:
    """Validate the explicit fixture capacity without claiming HBM evidence."""

    capacity = _mapping(capacity_value, "KV capacity manifest")
    _exact_fields(
        capacity,
        {"schema_version", "source", "scheduler_pools"},
        "KV capacity manifest",
    )
    if capacity["schema_version"] != 2:
        raise ValueError("KV capacity manifest schema_version must equal 2")
    source = _mapping(capacity["source"], "KV capacity source")
    _exact_fields(
        source,
        {
            "mode",
            "runtime_memory_policy_path",
            "runtime_memory_policy_sha256",
        },
        "KV capacity source",
    )
    if source != {
        "mode": "debug_explicit_block_count",
        "runtime_memory_policy_path": None,
        "runtime_memory_policy_sha256": None,
    }:
        raise ValueError("debug evaluation requires explicit debug KV capacity")
    pools = _sequence(capacity["scheduler_pools"], "KV capacity pools")
    if len(pools) != 1:
        raise ValueError("debug evaluation requires exactly one KV capacity pool")
    pool = _mapping(pools[0], "main KV capacity pool")
    _exact_fields(
        pool,
        {
            "pool_name", "source_mode", "hardware_id",
            "model_config_sha256", "worker_role", "tensor_parallel_size",
            "pipeline_parallel_size", "expert_parallel_size",
            "kv_block_size_tokens", "kv_bytes_per_token", "kv_block_bytes",
            "num_kv_blocks", "runtime_memory_policy", "rank_budgets",
            "native_vllm_allocation",
        },
        "main KV capacity pool",
    )
    expected_none_fields = (
        "hardware_id",
        "model_config_sha256",
        "worker_role",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "runtime_memory_policy",
        "native_vllm_allocation",
    )
    if (pool["pool_name"] != "main"
            or pool["source_mode"] != "debug_explicit_block_count"
            or any(pool[field] is not None for field in expected_none_fields)
            or tuple(_sequence(
                pool["rank_budgets"], "debug KV capacity rank budgets"))):
        raise ValueError("debug KV capacity carries analytical identity")
    model = load_model_config(str(_model_config_path(
        repository_root, cell.model.simulation_model_path)))
    prediction = config.prediction.values
    expected_kv_bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(
        prediction["tensor_parallel_size"],
        prediction["pipeline_parallel_size"],
    )
    expected_values = {
        "kv_block_size_tokens": prediction["kv_block_size_tokens"],
        "kv_bytes_per_token": expected_kv_bytes_per_token,
        "kv_block_bytes": (
            prediction["kv_block_size_tokens"] * expected_kv_bytes_per_token
        ),
        "num_kv_blocks": expected_num_kv_blocks,
    }
    for field, expected in expected_values.items():
        if pool[field] != expected:
            raise ValueError(f"debug KV capacity {field} mismatch")
    return {
        "source_mode": "debug_explicit_block_count",
        "runtime_memory_policy_path": None,
        "runtime_memory_policy_sha256": None,
        "num_kv_blocks": expected_num_kv_blocks,
        "available_kv_bytes_by_rank": (),
        "allocator_symbol": None,
        "allocator_version": None,
        "allocator_source_commit": None,
    }


def _validate_analytical_capacity(
    repository_root: str | Path,
    capacity_value: object,
    config: Phase1Config,
    cell: Phase1Cell,
) -> dict[str, object]:
    prediction = config.prediction.values
    tensor_parallel_size = prediction["tensor_parallel_size"]
    pipeline_parallel_size = prediction["pipeline_parallel_size"]
    expert_parallel_size = prediction["expert_parallel_size"]
    worker_count = tensor_parallel_size * pipeline_parallel_size
    capacity = _mapping(capacity_value, "KV capacity manifest")
    _exact_fields(
        capacity,
        {"schema_version", "source", "scheduler_pools"},
        "KV capacity manifest",
    )
    if capacity["schema_version"] != 2:
        raise ValueError("KV capacity manifest schema_version must equal 2")
    source = _mapping(capacity["source"], "KV capacity source")
    _exact_fields(
        source,
        {
            "mode",
            "runtime_memory_policy_path",
            "runtime_memory_policy_sha256",
        },
        "KV capacity source",
    )
    if source["mode"] != "analytical":
        raise ValueError("phase-1 Forward requires analytical KV capacity")
    expected_policy_path = Path(_expand_runtime_memory_policy_path(
        repository_root, config, cell, cell.model.dataset_model_id)).resolve()
    if Path(source["runtime_memory_policy_path"]).resolve() != expected_policy_path:
        raise ValueError("KV capacity runtime memory policy path mismatch")
    policy_sha256 = _sha256(
        source["runtime_memory_policy_sha256"],
        "runtime memory policy SHA256",
    )
    pools = _sequence(capacity["scheduler_pools"], "KV capacity pools")
    if len(pools) != 1:
        raise ValueError("phase-1 Forward requires exactly one KV capacity pool")
    pool = _mapping(pools[0], "main KV capacity pool")
    _exact_fields(
        pool,
        {
            "pool_name", "source_mode", "hardware_id",
            "model_config_sha256", "worker_role", "tensor_parallel_size",
            "pipeline_parallel_size", "expert_parallel_size",
            "kv_block_size_tokens", "kv_bytes_per_token", "kv_block_bytes",
            "num_kv_blocks", "runtime_memory_policy", "rank_budgets",
            "native_vllm_allocation",
        },
        "main KV capacity pool",
    )
    expected_identity = {
        "pool_name": "main",
        "source_mode": "analytical",
        "hardware_id": cell.hardware.hardware_id,
        "model_config_sha256": cell.model.model_config_sha256,
        "worker_role": "colocated",
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "kv_block_size_tokens": prediction["kv_block_size_tokens"],
    }
    for field, expected in expected_identity.items():
        if pool[field] != expected:
            raise ValueError(f"KV capacity pool {field} mismatch")
    for field in ("kv_bytes_per_token", "kv_block_bytes", "num_kv_blocks"):
        if _integer(pool[field], f"KV capacity {field}") < 1:
            raise ValueError(f"KV capacity {field} must be positive")
    if pool["kv_block_bytes"] != (
            pool["kv_block_size_tokens"] * pool["kv_bytes_per_token"]):
        raise ValueError("KV capacity block-byte accounting mismatch")
    rank_budgets = _sequence(pool["rank_budgets"], "KV capacity rank budgets")
    if len(rank_budgets) != worker_count:
        raise ValueError("KV capacity rank count must equal TP x PP")
    model = load_model_config(str(_model_config_path(
        repository_root, cell.model.simulation_model_path)))
    if model.num_hidden_layers % pipeline_parallel_size:
        raise ValueError("KV capacity model layers must divide pipeline parallelism")
    local_layers = model.num_hidden_layers // pipeline_parallel_size
    expected_kv_bytes_per_token = (
        model.kv_bytes_per_token_per_parallel_rank(
            tensor_parallel_size,
            pipeline_parallel_size,
        )
    )
    if pool["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError("KV capacity bytes per token disagree with model geometry")
    normalized_rank_budgets = []
    for expected_rank, rank_value in enumerate(rank_budgets):
        rank = _mapping(rank_value, "KV capacity rank budget")
        _exact_fields(
            rank,
            {
                "rank", "tensor_parallel_rank", "pipeline_parallel_rank",
                "expert_parallel_rank", "layer_start", "layer_end",
                "physical_hbm_bytes", "utilized_hbm_bytes",
                "model_weight_resident_bytes", "non_kv_reserve_bytes",
                "available_kv_bytes",
            },
            "KV capacity rank budget",
        )
        for field in rank:
            minimum = 1 if field.endswith("_bytes") else 0
            if _integer(rank[field], f"KV capacity rank {field}") < minimum:
                raise ValueError(
                    f"KV capacity rank {field} must be at least {minimum}")
        expected_tp_rank = expected_rank % tensor_parallel_size
        expected_pp_rank = expected_rank // tensor_parallel_size
        expected_ep_rank = expected_tp_rank if expert_parallel_size > 1 else 0
        expected_coordinates = {
            "rank": expected_rank,
            "tensor_parallel_rank": expected_tp_rank,
            "pipeline_parallel_rank": expected_pp_rank,
            "expert_parallel_rank": expected_ep_rank,
            "layer_start": expected_pp_rank * local_layers,
            "layer_end": (expected_pp_rank + 1) * local_layers,
        }
        if any(
            rank[field] != expected
            for field, expected in expected_coordinates.items()
        ):
            raise ValueError("KV capacity rank coordinates disagree with TP/PP/EP")
        if rank["available_kv_bytes"] != (
                rank["utilized_hbm_bytes"]
                - rank["model_weight_resident_bytes"]
                - rank["non_kv_reserve_bytes"]):
            raise ValueError("KV capacity rank byte accounting mismatch")
        normalized_rank_budgets.append(rank)
    policy = _mapping(
        pool["runtime_memory_policy"], "KV capacity runtime memory policy")
    _exact_fields(
        policy,
        {
            "binding", "gpu_memory_utilization", "non_kv_reserves",
            "non_kv_reserve_bytes", "reserve_scope",
            "completeness_assertion", "evidence_status",
        },
        "KV capacity runtime memory policy",
    )
    if policy["completeness_assertion"] != (
            "all_non_kv_reserves_for_bound_runtime"):
        raise ValueError("KV capacity policy completeness assertion mismatch")
    if policy["reserve_scope"] != "verified_uniform_across_pool_ranks":
        raise ValueError("KV capacity policy reserve scope mismatch")
    if policy["evidence_status"] != "verified_runtime_evidence":
        raise ValueError(
            "KV capacity policy evidence status must be "
            "'verified_runtime_evidence' for this verified-suite consumer")
    binding = _mapping(policy["binding"], "KV capacity policy binding")
    expected_binding = {
        "hardware_id": cell.hardware.hardware_id,
        "hardware_config_sha256": cell.hardware.hardware_config_sha256,
        "model_config_sha256": cell.model.model_config_sha256,
        "deployment": prediction["deployment"],
        "worker_role": "colocated",
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "max_model_len": prediction["max_model_len"],
        "max_batched_tokens": prediction["max_batched_tokens"],
        "max_requests": prediction["max_requests"],
        "kv_block_size_tokens": prediction["kv_block_size_tokens"],
    }
    _exact_fields(binding, set(expected_binding), "KV capacity policy binding")
    if any(
        binding[field] != expected
        for field, expected in expected_binding.items()
    ):
        raise ValueError("KV capacity policy binding disagrees with suite contract")
    allocation = _mapping(
        pool["native_vllm_allocation"], "native vLLM KV allocation")
    required_allocation_fields = {
        "version", "source_commit", "package_python_manifest_sha256",
        "allocator_symbol", "scheduler_config_symbol", "kv_spec_type",
        "kv_block_size_tokens", "max_model_len",
        "kv_page_size_bytes_per_layer",
        "num_kv_blocks", "worker_num_kv_blocks",
        "worker_allocated_tensor_bytes", "scheduler_tensor_owners",
    }
    _exact_fields(
        allocation, required_allocation_fields, "native vLLM KV allocation")
    if (allocation["version"] != VLLM_SCHEDULER_VERSION
            or allocation["source_commit"] != VLLM_SCHEDULER_SOURCE_COMMIT):
        raise ValueError("native KV allocator identity mismatch")
    if allocation["allocator_symbol"] != (
            "vllm.v1.core.kv_cache_utils.get_kv_cache_configs"):
        raise ValueError("native KV allocator symbol mismatch")
    if allocation["scheduler_config_symbol"] != (
            "vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config"):
        raise ValueError("native Scheduler KV config symbol mismatch")
    if allocation["num_kv_blocks"] != pool["num_kv_blocks"]:
        raise ValueError("native and analytical KV block counts disagree")
    if allocation["max_model_len"] != config.prediction.values["max_model_len"]:
        raise ValueError("native KV allocator max_model_len mismatch")
    if allocation["kv_block_size_tokens"] != prediction["kv_block_size_tokens"]:
        raise ValueError("native KV allocator block size mismatch")
    expected_page_size = (
        prediction["kv_block_size_tokens"]
        * 2
        * model.kv_heads_per_tensor_parallel_rank(tensor_parallel_size)
        * model.head_dim
        * model.dtype_bytes
    )
    if allocation["kv_page_size_bytes_per_layer"] != expected_page_size:
        raise ValueError("native KV allocator page geometry mismatch")
    worker_blocks = _sequence(
        allocation["worker_num_kv_blocks"],
        "native KV allocation worker block counts",
    )
    worker_tensor_bytes = _sequence(
        allocation["worker_allocated_tensor_bytes"],
        "native KV allocation worker tensor bytes",
    )
    if (
        len(worker_blocks) != worker_count
        or len(worker_tensor_bytes) != worker_count
    ):
        raise ValueError("native KV allocator worker count must equal TP x PP")
    if any(value != pool["num_kv_blocks"] for value in worker_blocks):
        raise ValueError("native KV allocator worker block counts disagree")
    expected_allocated_bytes = pool["num_kv_blocks"] * pool["kv_block_bytes"]
    if any(value != expected_allocated_bytes for value in worker_tensor_bytes):
        raise ValueError("native KV allocator worker tensor bytes disagree")
    scheduler_owners = tuple(
        tuple(_sequence(value, "native KV tensor owners"))
        for value in _sequence(
            allocation["scheduler_tensor_owners"],
            "native KV Scheduler tensor owners",
        )
    )
    expected_owners = tuple(
        (f"model.layers.{index}.self_attn.attn",)
        for index in range(local_layers)
    )
    if scheduler_owners != expected_owners:
        raise ValueError("native KV Scheduler tensor ownership mismatch")
    return {
        "source_mode": "analytical",
        "runtime_memory_policy_path": str(expected_policy_path),
        "runtime_memory_policy_sha256": policy_sha256,
        "num_kv_blocks": pool["num_kv_blocks"],
        "available_kv_bytes_by_rank": tuple(
            item["available_kv_bytes"] for item in normalized_rank_budgets
        ),
        "allocator_symbol": allocation["allocator_symbol"],
        "allocator_version": allocation["version"],
        "allocator_source_commit": allocation["source_commit"],
    }


def _validate_scheduler_runtime(
    repository_root: str | Path,
    scheduler_value: object,
    config: Phase1Config,
    cell: Phase1Cell,
    expected_num_kv_blocks: int,
) -> tuple[dict[str, object], str]:
    scheduler = _mapping(scheduler_value, "main scheduler runtime")
    runtime_fields = {field.name for field in fields(VLLMSchedulerRuntimeInfo)}
    if set(scheduler) != runtime_fields:
        raise ValueError(
            "phase-1 Scheduler runtime fields disagree with the pinned contract"
        )
    annotations = get_type_hints(VLLMSchedulerRuntimeInfo)
    for field_name, annotation in annotations.items():
        if not _runtime_value_matches_type(scheduler[field_name], annotation):
            raise TypeError(
                f"phase-1 Scheduler runtime {field_name} has the wrong type"
            )
    runtime = VLLMSchedulerRuntimeInfo(**scheduler)

    source_identity = vllm_scheduler_source_identity()
    for field_name, expected in source_identity.items():
        if scheduler[field_name] != expected:
            raise ValueError(
                f"effective Scheduler source identity {field_name} mismatch"
            )
    prediction = config.prediction.values
    tensor_parallel_size = prediction["tensor_parallel_size"]
    pipeline_parallel_size = prediction["pipeline_parallel_size"]
    expert_parallel_enabled = prediction["expert_parallel_size"] > 1
    fixed_values = {
        "platform": "cpu",
        "model_config_sha256": cell.model.model_config_sha256,
        "requested_dtype": "bfloat16",
        "effective_dtype": "bfloat16",
        "requested_tensor_parallel_size": tensor_parallel_size,
        "effective_tensor_parallel_size": tensor_parallel_size,
        "requested_pipeline_parallel_size": pipeline_parallel_size,
        "effective_pipeline_parallel_size": pipeline_parallel_size,
        "requested_expert_parallel": expert_parallel_enabled,
        "effective_expert_parallel": expert_parallel_enabled,
        "scheduler_pipeline_stage_rank": 0,
        "scheduler_pipeline_stage_layer_start": 0,
        "device_guard": "python_tripwire+cuda_state+procfs",
        "bootstrap": "version_pinned_scheduler_only_cpu",
        "requested_chunked_prefill": prediction["enable_chunked_prefill"],
        "effective_chunked_prefill": prediction["enable_chunked_prefill"],
        "requested_prefix_caching": prediction["enable_prefix_caching"],
        # The pinned CPU Scheduler deliberately disables vLLM's content-hash
        # cache. Logical reuse comes from the trace-length prefix oracle.
        "effective_prefix_caching": False,
        "prefix_oracle_mode": "trace_length_private_nonshared_v1",
        "speculative_method": "disabled",
        "mtp_num_speculative_tokens": prediction["mtp_num_speculative_tokens"],
        "mtp_acceptance_policy": prediction["mtp_acceptance_policy"],
        "mtp_fixed_accepted_tokens": prediction["mtp_fixed_accepted_tokens"],
        "mtp_acceptance_source_id": prediction["mtp_acceptance_source_id"],
        "mtp_acceptance_replay_path": None,
        "mtp_acceptance_replay_sha256": None,
        "mtp_acceptance_replay_artifact_id": None,
        "mtp_acceptance_replay_step_count": None,
        "mtp_acceptance_assumption_sha256": None,
        "mtp_request_stream_sha256": None,
        "mtp_padded_drafter_batch": False,
        "mtp_draft_model_type": "disabled",
        "mtp_num_mtp_layers": 0,
        "mtp_draft_max_model_len": 0,
        "max_num_batched_tokens": prediction["max_batched_tokens"],
        "max_num_seqs": prediction["max_requests"],
        "max_model_len": prediction["max_model_len"],
        "max_num_partial_prefills": 1,
        "max_long_partial_prefills": 1,
        "long_prefill_token_threshold": 0,
        "kv_block_size_tokens": prediction["kv_block_size_tokens"],
        "num_kv_blocks": expected_num_kv_blocks,
        "scheduler_policy": "fcfs",
        "num_lookahead_slots": 0,
        "attention_kv_layout": prediction["attention_kv_layout"],
        "backend_role": prediction["deployment"],
        "kv_handoff_mode": "not_applicable",
    }
    for field_name, expected in fixed_values.items():
        if scheduler[field_name] != expected:
            raise ValueError(f"effective Scheduler runtime {field_name} mismatch")
    if scheduler["requested_async_scheduling"] != scheduler[
            "effective_async_scheduling"]:
        raise ValueError("effective Scheduler async mode mismatch")

    model = load_model_config(str(_model_config_path(
        repository_root, cell.model.simulation_model_path)))
    if model.attention_layout != "standard_gqa":
        raise ValueError("phase-1 Scheduler KV geometry requires standard GQA")
    if model.num_hidden_layers % pipeline_parallel_size:
        raise ValueError("Scheduler model layers must divide pipeline parallelism")
    local_layer_count = model.num_hidden_layers // pipeline_parallel_size
    kv_head_count = model.kv_heads_per_tensor_parallel_rank(
        tensor_parallel_size)
    expected_page_size = (
        scheduler["kv_block_size_tokens"]
        * 2
        * kv_head_count
        * model.head_dim
        * model.dtype_bytes
    )
    expected_owners = tuple(
        (f"model.layers.{index}.self_attn.attn",)
        for index in range(local_layer_count)
    )
    if (scheduler["scheduler_pipeline_stage_layer_end"]
            != local_layer_count
            or scheduler["scheduler_kv_cache_num_heads_per_rank"]
            != kv_head_count
            or scheduler["scheduler_kv_cache_page_size_bytes"]
            != expected_page_size
            or scheduler["scheduler_kv_cache_tensor_owners"]
            != expected_owners):
        raise ValueError(
            "effective Scheduler KV geometry disagrees with the frozen model"
        )
    return dict(scheduler), vllm_scheduler_runtime_sha256(runtime)


def _runtime_value_matches_type(value: object, annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is types.UnionType:
        return any(
            _runtime_value_matches_type(value, member)
            for member in get_args(annotation)
        )
    if origin is tuple:
        if type(value) is not tuple:
            return False
        members = get_args(annotation)
        if len(members) == 2 and members[1] is Ellipsis:
            return all(
                _runtime_value_matches_type(item, members[0]) for item in value
            )
        return len(value) == len(members) and all(
            _runtime_value_matches_type(item, member)
            for item, member in zip(value, members)
        )
    if annotation is type(None):
        return value is None
    if annotation is float:
        return type(value) is float and math.isfinite(value)
    if annotation in (bool, int, str):
        return type(value) is annotation
    raise TypeError(f"unsupported Scheduler runtime annotation: {annotation!r}")


def _validate_execution_artifact(
    payload: Mapping[str, object],
    expected_timing_status: str,
) -> dict[str, object]:
    artifact = _mapping(payload["execution_artifact"], "execution artifact")
    if artifact["schema_version"] in STREAMING_EXECUTION_ARTIFACT_HASH_METHODS:
        return _validate_streaming_execution_artifact(
            payload, artifact, expected_timing_status)
    _exact_fields(artifact, EXECUTION_ARTIFACT_FIELDS, "execution artifact")
    schema_version = artifact["schema_version"]
    if type(schema_version) is not int or schema_version not in (5, 7, 8, 15):
        raise ValueError("execution artifact schema_version must be 5, 7, 8, or 15")
    if artifact["hash_method"] != _EXECUTION_ARTIFACT_HASH_METHOD:
        raise ValueError("execution artifact hash_method mismatch")
    timing_status = artifact["performance_timing_status"]
    if (timing_status != expected_timing_status
            or payload["performance_timing_status"] != timing_status):
        raise ValueError("execution artifact performance timing status mismatch")
    counts = _mapping(artifact["counts"], "execution artifact counts")
    _exact_fields(
        counts,
        _EXECUTION_ARTIFACT_COUNT_FIELDS,
        "execution artifact counts",
    )
    if any(type(counts[field]) is not int for field in counts):
        raise TypeError("execution artifact counts must be exact integers")
    if any(counts[field] < 0 for field in counts):
        raise ValueError("execution artifact counts must be non-negative")

    schedule_trace = _mapping(
        artifact["schedule_trace"], "execution artifact schedule trace"
    )
    _exact_fields(
        schedule_trace,
        {
            "route_decision_version",
            "transfer_trace_version",
            "entries",
            "route_decisions",
            "transfers",
        },
        "execution artifact schedule trace",
    )
    if (type(schedule_trace["route_decision_version"]) is not int
            or schedule_trace["route_decision_version"] != 2
            or type(schedule_trace["transfer_trace_version"]) is not int
            or schedule_trace["transfer_trace_version"] != 2):
        raise ValueError("execution artifact trace versions mismatch")
    schedule_entries = _artifact_tuple(
        schedule_trace["entries"], "execution artifact schedule entries"
    )
    _artifact_tuple(
        schedule_trace["route_decisions"],
        "execution artifact route decisions",
    )
    trace_transfers = _artifact_tuple(
        schedule_trace["transfers"], "execution artifact trace transfers"
    )
    steps = _artifact_tuple(artifact["steps"], "execution artifact steps")
    transfers = _artifact_tuple(
        artifact["transfers"], "execution artifact transfers"
    )
    modeled_kv_offload = _mapping(
        artifact["modeled_kv_offload"], "execution artifact modeled KV offload"
    )
    modeled_transfers = _artifact_tuple(
        modeled_kv_offload["transfers"],
        "execution artifact modeled KV transfers",
    )
    availability = _string(
        modeled_kv_offload["availability"],
        "execution artifact modeled KV availability",
    )
    if availability == "not_enabled":
        _exact_fields(
            modeled_kv_offload,
            {
                "schema_version",
                "availability",
                "intent_source",
                "transfers",
                "final_tier_state",
            },
            "disabled modeled KV artifact",
        )
        if (type(modeled_kv_offload["schema_version"]) is not int
                or modeled_kv_offload["schema_version"] != 2
                or schema_version not in (5, 8, 15)
                or modeled_transfers != ()):
            raise ValueError("disabled modeled KV artifact is inconsistent")
    elif availability in (
            "available_modeled_rank_local",
            "available_modeled_serial",
    ):
        _exact_fields(
            modeled_kv_offload,
            {
                "schema_version",
                "availability",
                "mode",
                "validation_status",
                "intent_source",
                "ledger_semantics",
                "physical_transfer_count",
                "scope",
                "transfers",
                "final_tier_state",
            },
            "available modeled KV artifact",
        )
        physical_transfer_count = _integer(
            modeled_kv_offload["physical_transfer_count"],
            "modeled KV physical transfer count",
        )
        if (type(modeled_kv_offload["schema_version"]) is not int
                or modeled_kv_offload["schema_version"] != 4
                or schema_version not in (7, 8, 15)
                or (not modeled_transfers
                    and modeled_kv_offload["mode"] != "fixed_prefix_tier_replay")
                or physical_transfer_count < len(modeled_transfers)):
            raise ValueError("available modeled KV artifact schema mismatch")
    else:
        raise ValueError(
            f"unknown modeled KV artifact availability: {availability!r}"
        )

    kernel_calls = sum(
        _artifact_call_count(step, "execution artifact step") for step in steps
    ) + sum(
        _artifact_call_count(transfer, "execution artifact transfer")
        for transfer in transfers
    )
    actual_counts = {
        "schedule_steps": len(schedule_entries),
        "simulation_steps": len(steps),
        "simulation_transfers": len(transfers),
        "trace_transfers": len(trace_transfers),
        "modeled_kv_offload_transfers": len(modeled_transfers),
        "kernel_calls": kernel_calls,
    }
    if dict(counts) != actual_counts:
        raise ValueError("execution artifact counts disagree with artifact body")
    top_level_counts = {
        "num_steps": _integer(payload["num_steps"], "simulation num_steps"),
        "num_transfers": _integer(
            payload["num_transfers"], "simulation num_transfers"
        ),
        "num_kernel_calls": _integer(
            payload["num_kernel_calls"], "simulation num_kernel_calls"
        ),
    }
    if (top_level_counts["num_steps"] != actual_counts["schedule_steps"]
            or top_level_counts["num_steps"]
            != actual_counts["simulation_steps"]
            or top_level_counts["num_transfers"]
            != actual_counts["trace_transfers"]
            or top_level_counts["num_transfers"]
            != actual_counts["simulation_transfers"]
            or top_level_counts["num_kernel_calls"] != kernel_calls):
        raise ValueError(
            "simulation top-level counts disagree with execution artifact"
        )

    artifact_sha = _sha256(
        artifact["artifact_sha256"], "execution artifact SHA256"
    )
    body = {
        field: artifact[field]
        for field in EXECUTION_ARTIFACT_FIELDS
        if field != "artifact_sha256"
    }
    if _streaming_canonical_json_sha256(body) != artifact_sha:
        raise ValueError("execution artifact body SHA256 mismatch")

    main_chain = _mapping(payload["main_chain_status"], "main chain status")
    _exact_fields(
        main_chain,
        {"status", "chain", "execution_artifact_sha256"},
        "main chain status",
    )
    if (main_chain["status"] != "executed"
            or main_chain["chain"] != (_SYMBOLIC_MAIN_CHAIN if schema_version == 15 else _MAIN_CHAIN)
            or main_chain["execution_artifact_sha256"] != artifact_sha):
        raise ValueError("main chain did not execute the canonical artifact")
    return {
        "schema_version": schema_version,
        "performance_timing_status": timing_status,
        "counts": dict(counts),
        "artifact_sha256": artifact_sha,
    }


def _validate_streaming_execution_artifact(
    payload: Mapping[str, object],
    artifact: Mapping[str, object],
    expected_timing_status: str,
) -> dict[str, object]:
    _exact_fields(
        artifact,
        _STREAMING_EXECUTION_ARTIFACT_FIELDS,
        "streaming execution artifact",
    )
    schema_version = artifact["schema_version"]
    if schema_version not in STREAMING_EXECUTION_ARTIFACT_HASH_METHODS:
        raise ValueError(
            "streaming execution artifact schema_version must be 9, 10, 11, 12, 13, 14, or 16")
    if (artifact["hash_method"]
            != STREAMING_EXECUTION_ARTIFACT_HASH_METHODS[schema_version]):
        raise ValueError("streaming execution artifact hash_method mismatch")
    timing_status = artifact["performance_timing_status"]
    if (timing_status != expected_timing_status
            or payload["performance_timing_status"] != timing_status):
        raise ValueError(
            "streaming execution artifact performance timing status mismatch")

    counts = _mapping(
        artifact["counts"], "streaming execution artifact counts")
    _exact_fields(
        counts,
        _EXECUTION_ARTIFACT_COUNT_FIELDS,
        "streaming execution artifact counts",
    )
    if any(type(counts[field]) is not int for field in counts):
        raise TypeError(
            "streaming execution artifact counts must be exact integers")
    if any(counts[field] < 0 for field in counts):
        raise ValueError(
            "streaming execution artifact counts must be non-negative")
    if (counts["schedule_steps"] != counts["simulation_steps"]
            or (schema_version != 16
                and counts["modeled_kv_offload_transfers"] != 0)):
        raise ValueError(
            "streaming execution artifact counts violate the retention scope")

    streaming = _mapping(
        artifact["streaming_aggregate"],
        "streaming execution artifact aggregate",
    )
    chain_field = (
        "ordered_step_manifest_chain_sha256"
        if schema_version == 9
        else "ordered_step_execution_contract_chain_sha256"
    )
    streaming_fields = {
        "retention",
        "representation",
        chain_field,
        "request_level_artifact",
    }
    if schema_version in (13, 14, 16):
        streaming_fields.add("transfer_summary")
    if schema_version == 16:
        streaming_fields.add("modeled_kv_offload")
    _exact_fields(streaming, streaming_fields,
                  "streaming execution artifact aggregate")
    if streaming["retention"] != "step_ir_consumed_then_released":
        raise ValueError("streaming execution artifact retention mismatch")
    if streaming["representation"] not in (
            "full_step_ir_consumed_without_retention",
            "compact_step_ir_consumed_without_retention"):
        raise ValueError("streaming execution artifact representation mismatch")
    _sha256(
        streaming[chain_field],
        "streaming execution artifact step chain SHA256",
    )
    if streaming["request_level_artifact"] != "not_retained":
        raise ValueError(
            "streaming execution artifact request-level retention mismatch")
    if schema_version in (13, 14, 16):
        transfer_summary = _mapping(
            streaming["transfer_summary"],
            "streaming execution artifact transfer summary",
        )
        _exact_fields(
            transfer_summary,
            {
                "retention",
                "trace_transfer_count",
                "simulation_transfer_count",
                "trace_transfer_manifest_sha256",
            },
            "streaming execution artifact transfer summary",
        )
        if transfer_summary["retention"] != (
                "trace_manifest_digest_and_counts"):
            raise ValueError(
                "streaming execution artifact transfer retention mismatch")
        for field in ("trace_transfer_count", "simulation_transfer_count"):
            if (type(transfer_summary[field]) is not int
                    or transfer_summary[field] < 0):
                raise ValueError(
                    "streaming execution artifact transfer counts must be "
                    "non-negative integers")
        if (transfer_summary["trace_transfer_count"]
                != counts["trace_transfers"]
                or transfer_summary["simulation_transfer_count"]
                != counts["simulation_transfers"]):
            raise ValueError(
                "streaming execution artifact transfer summary disagrees "
                "with counts")
        _sha256(
            transfer_summary["trace_transfer_manifest_sha256"],
            "streaming transfer manifest SHA256",
        )
        if _streaming_canonical_json_sha256(payload["transfers"]) != (
                transfer_summary["trace_transfer_manifest_sha256"]):
            raise ValueError(
                "streaming transfer manifest digest disagrees with payload")
    elif (counts["simulation_transfers"] != 0
          or counts["trace_transfers"] != 0):
        raise ValueError(
            "legacy streaming execution artifact cannot retain transfers")

    if schema_version == 16:
        offload = _mapping(streaming["modeled_kv_offload"],
            "streaming modeled KV offload")
        _exact_fields(offload, {
            "mode", "validation_status", "transfer_count",
            "h2d_bytes_per_rank", "d2h_bytes_per_rank",
            "completion_records_sha256", "final_tier_state",
        }, "streaming modeled KV offload")
        if (offload["mode"] not in ("fixed_prefix_tier_replay", "modeled_host_kv_serial")
                or offload["validation_status"] not in (
                    "modeled_request_private_full_blocks",
                    "modeled_shared_prefix_full_blocks")
                or type(offload["transfer_count"]) is not int
                or offload["transfer_count"] != counts["modeled_kv_offload_transfers"]
                or any(type(offload[field]) is not int or offload[field] < 0
                    for field in ("h2d_bytes_per_rank", "d2h_bytes_per_rank"))):
            raise ValueError("streaming modeled KV offload counts disagree")
        _sha256(offload["completion_records_sha256"],
            "streaming KV completion records SHA256")
        final_tier_state = _mapping(offload["final_tier_state"],
            "streaming KV final tier state")
        if ("validation_status" not in final_tier_state
                or final_tier_state["validation_status"] != offload["validation_status"]):
            raise ValueError("streaming KV final tier validation status disagrees")
        if tuple(_sequence(final_tier_state["inflight_ready_events"],
                "streaming KV final tier inflight events")):
            raise ValueError("streaming KV final tier state retains transfers")
        tiers = _mapping(final_tier_state["tiers"],
            "streaming KV final tier capacities")
        if set(tiers) != {"hbm", "host"}:
            raise ValueError("streaming KV final tier state has invalid tiers")
        for tier in ("hbm", "host"):
            state = _mapping(tiers[tier], f"streaming KV {tier} tier")
            if type(state["reserved_bytes"]) is not int or state["reserved_bytes"] != 0:
                raise ValueError("streaming KV final tier state retains reserved capacity")

    top_level_counts = {
        "num_steps": _integer(payload["num_steps"], "simulation num_steps"),
        "num_transfers": _integer(
            payload["num_transfers"], "simulation num_transfers"),
        "num_kernel_calls": _integer(
            payload["num_kernel_calls"], "simulation num_kernel_calls"),
    }
    if (top_level_counts["num_steps"] != counts["schedule_steps"]
            or top_level_counts["num_steps"] != counts["simulation_steps"]
            or (schema_version not in (13, 14, 16)
                and top_level_counts["num_transfers"] != 0)
            or (schema_version in (13, 14, 16)
                and top_level_counts["num_transfers"]
                != counts["trace_transfers"])
            or top_level_counts["num_kernel_calls"] != counts["kernel_calls"]):
        raise ValueError(
            "simulation top-level counts disagree with streaming artifact")

    artifact_sha = _sha256(
        artifact["artifact_sha256"],
        "streaming execution artifact SHA256",
    )
    body = {
        field: artifact[field]
        for field in _STREAMING_EXECUTION_ARTIFACT_FIELDS
        if field != "artifact_sha256"
    }
    if _streaming_canonical_json_sha256(body) != artifact_sha:
        raise ValueError("streaming execution artifact body SHA256 mismatch")

    main_chain = _mapping(payload["main_chain_status"], "main chain status")
    _exact_fields(
        main_chain,
        {"status", "chain", "execution_artifact_sha256"},
        "main chain status",
    )
    if (main_chain["status"] != "executed"
            or main_chain["chain"] != (_SYMBOLIC_MAIN_CHAIN if schema_version in (14, 16) else _MAIN_CHAIN)
            or main_chain["execution_artifact_sha256"] != artifact_sha):
        raise ValueError("main chain did not execute the streaming artifact")
    return {
        "schema_version": schema_version,
        "performance_timing_status": timing_status,
        "counts": dict(counts),
        "artifact_sha256": artifact_sha,
    }


def _artifact_tuple(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple from the in-memory producer")
    return value


def _artifact_call_count(value: object, label: str) -> int:
    record = _mapping(value, label)
    kernel_plan = _mapping(record["kernel_plan"], f"{label} kernel plan")
    count_field = (
        "logical_call_count"
        if "logical_call_count" in kernel_plan else "call_count"
    )
    count = _integer(kernel_plan[count_field], f"{label} kernel call count")
    if count < 0:
        raise ValueError(f"{label} kernel call count must be non-negative")
    return count


def _streaming_canonical_json_sha256(payload: object) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for chunk in encoder.iterencode(payload):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _validate_execution_cache_stats(
    value: object,
    execution_mode: str,
) -> Mapping[str, object]:
    stats = _mapping(value, "execution cache stats")
    cache_names = ("kernel_lowering", "kernel_prediction")
    _exact_fields(
        stats,
        {"execution_mode", *cache_names},
        "execution cache stats",
    )
    if stats["execution_mode"] != execution_mode:
        raise ValueError("execution cache stats mode mismatch")
    normalized: dict[str, object] = {"execution_mode": execution_mode}
    for cache_name in cache_names:
        cache = stats[cache_name]
        if execution_mode == "reference":
            if cache is not None:
                raise ValueError(
                    f"reference execution cache {cache_name} must be null")
            normalized[cache_name] = None
            continue
        cache_stats = _mapping(cache, f"{cache_name} cache stats")
        _exact_fields(
            cache_stats,
            {"entries", "hits", "misses"},
            f"{cache_name} cache stats",
        )
        normalized_cache = {}
        for field in ("entries", "hits", "misses"):
            normalized_cache[field] = _integer(
                cache_stats[field], f"{cache_name} cache {field}")
            if normalized_cache[field] < 0:
                raise ValueError(
                    f"{cache_name} cache {field} must be nonnegative")
        if normalized_cache["entries"] != normalized_cache["misses"]:
            raise ValueError(
                f"{cache_name} cache entries must equal misses")
        normalized[cache_name] = normalized_cache
    return normalized


def _expand_runtime_memory_policy_path(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    model_id: str,
) -> str:
    template = config.prediction.values["runtime_memory_policy_path"]
    return str(resolve_runtime_memory_policy_path(
        template,
        repository_root=repository_root,
        hardware_id=cell.hardware.hardware_id,
        model_id=model_id,
    ))


_SYMBOLIC_MAIN_CHAIN = _MAIN_CHAIN.replace("build_workload", "build_symbolic_workload")


def validate_execution_repository_provenance(
    execution: Mapping[str, object], *, require_unchanged: bool,
) -> Mapping[str, object]:
    """Validate observed repository identities, never a frozen import claim."""
    if execution["schema_version"] not in (3, 4):
        raise ValueError("execution provenance schema_version must be 3 or 4")
    repository = _mapping(execution["repository"], "execution repository")
    if execution["schema_version"] == 4:
        identity_fields = {"repository_root", "git_commit", "worktree_state", "changed_path_count",
            "git_status_short", "status_porcelain_sha256", "git_diff_hash"}
        observation_fields = {"identity_recorded_at", "run_start_identity", "identity_changed_between_boundaries",
            "identity_observation_scope", "exact_imported_code_snapshot_binding"}
        _exact_fields(repository, identity_fields | observation_fields, "execution repository")
        start = _mapping(repository["run_start_identity"], "execution start repository")
        _exact_fields(start, identity_fields, "execution start repository")
        end = {key: repository[key] for key in identity_fields}
        for identity in (start, end):
            commit = _string(identity["git_commit"], "repository git_commit")
            if _GIT_COMMIT.fullmatch(commit) is None:
                raise ValueError("repository git_commit must be lowercase hex")
            _string(identity["repository_root"], "repository root")
            _string(identity["worktree_state"], "repository worktree state")
            count = _integer(identity["changed_path_count"], "repository changed path count")
            paths = _sequence(identity["git_status_short"], "repository changed paths")
            if count < 0 or count != len(paths) or any(not isinstance(path, str) or not path for path in paths):
                raise ValueError("repository changed paths disagree with their count")
            _sha256(identity["git_diff_hash"], "repository diff SHA256")
            _sha256(identity["status_porcelain_sha256"], "repository status SHA256")
        if (repository["identity_recorded_at"] != "run_end"
                or repository["identity_observation_scope"] != "entry_and_exit_only_not_continuous"
                or repository["exact_imported_code_snapshot_binding"] != "unbound"):
            raise ValueError("repository observations cannot assert an exact imported-code snapshot")
        changed = repository["identity_changed_between_boundaries"]
        if type(changed) is not bool or changed != (dict(start) != end):
            raise ValueError("repository boundary-change declaration disagrees with its observed identities")
        if require_unchanged and changed:
            raise ValueError("repository changed during execution; same-commit formal acceptance is unavailable")
    return repository
