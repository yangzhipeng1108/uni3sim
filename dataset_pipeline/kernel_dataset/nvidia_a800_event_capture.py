"""No-profiler device-Event capture plans for NVIDIA A800 runners."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from dataset_pipeline.kernel_dataset.config import (
    canonical_json_sha256,
    load_kernel_dataset_mapping,
    get_kernel_dataset_path,
    load_pipeline_config,
)
from dataset_pipeline.kernel_dataset.nvidia_a800 import (
    _apply_canonical_gemm_timing_dtypes,
    _build_jobs_with_executables,
    _resolve_kernel_cfg,
)


A800_L2_BYTES = 40 * 1024 * 1024
A800_VECTOR_MINIMUM_INPUT_RING_MIB = 320
A800_VECTOR_EVENT_BATCH_PUBLIC_CALL_COUNTS = frozenset({32, 128, 512, 2048})
A800_NONVECTOR_EVENT_STEADY_STATE_PUBLIC_CALL_COUNTS = frozenset({1, 8, 32})
A800_VECTOR_EVENT_RING_PROTOCOL_VERSION = 2
A800_VECTOR_GRAPH_PROTOCOL_VERSION = 3
A800_VECTOR_DIFFERENTIAL_GRAPH_PROTOCOL_VERSION = 5
A800_VECTOR_POWER_WARMUP_GEMM_DIMENSION = 4096
A800_VECTOR_POWER_WARMUP_GEMM_ITERATIONS = 1000
A800_VECTOR_EVENT_TIMING_MODES = frozenset({
    "direct_batched_event",
    "cuda_graph_full_ring_replay",
    "cuda_graph_differential",
})


def build_nvidia_a800_vector_power_warmup_provenance() -> Mapping[str, Any]:
    """Describe the fixed power-state conditioning used by Graph timing."""

    return {
        "kind": "bf16_gemm_power_warmup_and_per_graph_replay_maintenance",
        "matrix_dimension": A800_VECTOR_POWER_WARMUP_GEMM_DIMENSION,
        "initial_gemm_iterations": A800_VECTOR_POWER_WARMUP_GEMM_ITERATIONS,
        "per_replay_gemm_iterations": 1,
        "per_replay_placement": (
            "same_current_stream_immediately_before_start_event"),
        "synchronized_before_graph_measurement": True,
        "included_in_event_interval": False,
    }


def _positive_workload_integer(
        attributes: Mapping[str, Any], name: str, context: str) -> int:
    value = attributes[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} {name} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} {name} must be positive")
    return value


def build_nvidia_a800_vector_event_batch_contract(
        attributes: Mapping[str, Any], *, input_ring_mib: int,
        public_call_count: int) -> Mapping[str, Any]:
    """Derive the exact A800 Vector rotating-buffer allocation contract."""

    if not isinstance(attributes, Mapping) or not attributes:
        raise TypeError("A800 Vector attributes must be a non-empty mapping")
    if isinstance(input_ring_mib, bool) or not isinstance(input_ring_mib, int):
        raise TypeError("A800 Vector input_ring_mib must be an integer")
    if input_ring_mib < A800_VECTOR_MINIMUM_INPUT_RING_MIB:
        raise ValueError(
            "A800 Vector input ring must be at least 320 MiB (8x L2)")
    if input_ring_mib * 1024 * 1024 % A800_L2_BYTES != 0:
        raise ValueError(
            "A800 Vector input ring must be an integer multiple of L2")
    if (isinstance(public_call_count, bool)
            or not isinstance(public_call_count, int)):
        raise TypeError("A800 Vector public_call_count must be an integer")
    if public_call_count not in A800_VECTOR_EVENT_BATCH_PUBLIC_CALL_COUNTS:
        raise ValueError(
            "A800 Vector Event batches require an explicitly selected public "
            "call count in {32, 128, 512, 2048}")

    context = "A800 Vector workload"
    op = attributes["op"]
    if not isinstance(op, str) or not op:
        raise TypeError(f"{context} op must be a non-empty string")
    dtype_bytes = _positive_workload_integer(
        attributes, "dtype_bytes", context)
    tok = _positive_workload_integer(attributes, "tok", context)

    if op in {"rmsnorm", "add_rmsnorm", "elementwise"}:
        hidden = _positive_workload_integer(attributes, "hidden", context)
        tensor_bytes = tok * hidden * dtype_bytes
        if op == "rmsnorm":
            dynamic_inputs = {"x": tensor_bytes}
            preallocated_outputs = {"output": tensor_bytes}
            in_place_outputs = []
            padded_cache_output = False
        elif op == "add_rmsnorm":
            dynamic_inputs = {"x": tensor_bytes, "residual": tensor_bytes}
            preallocated_outputs = {}
            in_place_outputs = ["x", "residual"]
            padded_cache_output = False
        elif op == "elementwise":
            dynamic_inputs = {"left": tensor_bytes, "right": tensor_bytes}
            preallocated_outputs = {"output": tensor_bytes}
            in_place_outputs = []
            padded_cache_output = False
        else:
            raise AssertionError(f"validated Vector op was not dispatched: {op}")
    elif op == "silu":
        intermediate = _positive_workload_integer(
            attributes, "intermediate", context)
        dynamic_inputs = {
            "input_tensor": tok * 2 * intermediate * dtype_bytes}
        preallocated_outputs = {
            "output": tok * intermediate * dtype_bytes}
        in_place_outputs = []
        padded_cache_output = False
    elif op == "rope":
        num_heads = _positive_workload_integer(
            attributes, "num_heads", context)
        num_kv_heads = _positive_workload_integer(
            attributes, "num_kv_heads", context)
        head_dim = _positive_workload_integer(
            attributes, "head_dim", context)
        dynamic_inputs = {
            "q": tok * num_heads * head_dim * dtype_bytes,
            "k": tok * num_kv_heads * head_dim * dtype_bytes,
        }
        preallocated_outputs = {}
        in_place_outputs = ["q", "k"]
        padded_cache_output = False
    elif op == "kvcache":
        num_kv_heads = _positive_workload_integer(
            attributes, "num_kv_heads", context)
        head_dim = _positive_workload_integer(
            attributes, "head_dim", context)
        tensor_bytes = tok * num_kv_heads * head_dim * dtype_bytes
        dynamic_inputs = {"key": tensor_bytes, "value": tensor_bytes}
        preallocated_outputs = {
            "key_cache_destination": tensor_bytes,
            "value_cache_destination": tensor_bytes,
        }
        in_place_outputs = []
        padded_cache_output = True
    else:
        raise ValueError(f"unsupported A800 Vector op: {op!r}")

    input_bytes_per_call = sum(dynamic_inputs.values())
    requested_input_bytes = input_ring_mib * 1024 * 1024
    slot_count = math.ceil(requested_input_bytes / input_bytes_per_call)
    slot_stride = 1 if slot_count == 1 else slot_count // 2 + 1
    if slot_stride >= slot_count and slot_count != 1:
        slot_stride = 1
    while math.gcd(slot_stride, slot_count) != 1:
        slot_stride += 1
        if slot_stride >= slot_count:
            slot_stride = 1
    allocated_input_bytes = slot_count * input_bytes_per_call
    output_bytes_per_call = sum(preallocated_outputs.values())
    logical_output_bytes = slot_count * output_bytes_per_call
    if padded_cache_output:
        cache_slot_count = math.ceil(slot_count * tok / 16) * 16
        allocated_output_bytes = (
            cache_slot_count * num_kv_heads * head_dim * dtype_bytes * 2)
    else:
        allocated_output_bytes = logical_output_bytes
    return {
        "protocol_version": A800_VECTOR_EVENT_RING_PROTOCOL_VERSION,
        "mode": "rotating_preallocated_vector_buffers_full_ring_cycle_reuse",
        "hardware_l2_bytes": A800_L2_BYTES,
        "minimum_input_ring_bytes": (
            A800_VECTOR_MINIMUM_INPUT_RING_MIB * 1024 * 1024),
        "requested_input_ring_bytes": requested_input_bytes,
        "allocated_input_ring_bytes": allocated_input_bytes,
        "input_bytes_per_public_call": input_bytes_per_call,
        "dynamic_input_bytes_per_operand": dynamic_inputs,
        "preallocated_output_bytes_per_operand": preallocated_outputs,
        "allocated_output_ring_bytes": allocated_output_bytes,
        "output_padding_bytes": allocated_output_bytes - logical_output_bytes,
        "in_place_output_operands": in_place_outputs,
        "public_call_count": public_call_count,
        "slot_count": slot_count,
        "slot_stride": slot_stride,
        "selection": (
            "coprime_stride_reuse_only_after_full_input_ring_cycle"),
        "reuse_contract": "reuse_only_after_full_input_ring_cycle",
        "initialization": "torch.randn_or_zeros_before_timing",
        "pre_target_sweep": "none",
    }


def build_nvidia_a800_vector_event_batch_provenance(
        public_call_count: int) -> Mapping[str, Any]:
    """Describe the explicitly selected v2 Event batch and ring traversal."""

    if (isinstance(public_call_count, bool)
            or not isinstance(public_call_count, int)):
        raise TypeError("A800 Vector public_call_count must be an integer")
    if public_call_count not in A800_VECTOR_EVENT_BATCH_PUBLIC_CALL_COUNTS:
        raise ValueError(
            "A800 Vector Event batches require an explicitly selected public "
            "call count in {32, 128, 512, 2048}")
    return {
        "protocol_version": A800_VECTOR_EVENT_RING_PROTOCOL_VERSION,
        "public_call_count": public_call_count,
        "public_call_count_selection": "explicit_cli_value_not_duration_based",
        "input_selection": (
            "advance_coprime_ring_once_per_public_call_reuse_only_after_"
            "full_input_ring_cycle"),
        "duration_reporting": (
            "aggregate_current_stream_event_duration_divided_by_"
            "public_call_count"),
    }


def build_nvidia_a800_vector_graph_contract(
        attributes: Mapping[str, Any], *, input_ring_mib: int
        ) -> Mapping[str, Any]:
    """Derive the A800 Vector CUDA-Graph full-input-ring contract.

    One replay traverses every preallocated input slot exactly once in
    coprime-stride order.  The ring is at least eight A800 L2 capacities, so
    the next replay reaches a slot only after a full clearly-larger-than-L2
    cycle.  Capture and replay counts are structural facts of the allocation,
    never selected from measured duration.
    """

    direct_contract = build_nvidia_a800_vector_event_batch_contract(
        attributes, input_ring_mib=input_ring_mib, public_call_count=32)
    contract = dict(direct_contract)
    slot_count = contract["slot_count"]
    del contract["public_call_count"]
    contract["protocol_version"] = A800_VECTOR_GRAPH_PROTOCOL_VERSION
    contract["mode"] = (
        "rotating_preallocated_vector_buffers_full_ring_cuda_graph_replay")
    contract["selection"] = (
        "fixed_coprime_full_ring_cycle_inside_each_cuda_graph_replay")
    contract["reuse_contract"] = (
        "reuse_only_after_full_input_ring_cycle_and_between_graph_replays")
    contract["graph_cycle_public_call_count"] = slot_count
    contract["graph_repetition_counts"] = [slot_count, 2 * slot_count]
    return contract


def build_nvidia_a800_vector_graph_provenance(
        contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the immutable CUDA-Graph replay provenance for one ring."""

    if not isinstance(contract, Mapping):
        raise TypeError("A800 Vector CUDA-Graph contract must be a mapping")
    slot_count = contract["slot_count"]
    if (isinstance(slot_count, bool) or not isinstance(slot_count, int)
            or slot_count <= 0):
        raise ValueError("A800 Vector CUDA-Graph slot count must be positive")
    if contract["protocol_version"] != A800_VECTOR_GRAPH_PROTOCOL_VERSION:
        raise ValueError("A800 Vector CUDA-Graph protocol mismatch")
    if contract["graph_cycle_public_call_count"] != slot_count:
        raise ValueError("A800 Vector CUDA-Graph cycle count mismatch")
    if contract["graph_repetition_counts"] != [slot_count, 2 * slot_count]:
        raise ValueError("A800 Vector CUDA-Graph repetition counts mismatch")
    return {
        "protocol_version": A800_VECTOR_GRAPH_PROTOCOL_VERSION,
        "graph_backend": "torch.cuda.CUDAGraph",
        "input_selection": (
            "fixed_coprime_full_ring_cycle_inside_each_cuda_graph_replay"),
        "graph_cycle_public_call_count": slot_count,
        "graph_repetition_counts": [slot_count, 2 * slot_count],
        "duration_reporting": (
            "external_cuda_event_around_full_ring_graph_replay_divided_by_"
            "graph_public_call_count"),
    }


def build_nvidia_a800_vector_differential_graph_contract(
        attributes: Mapping[str, Any], *, minimum_distinct_input_mib: int
        ) -> Mapping[str, Any]:
    """Derive a paired N/2N Graph contract with unique input slots.

    The lower graph covers at least eight A800 L2 capacities.  The higher
    graph shares the exact lower-graph prefix and adds an equally large suffix
    whose input slots are all distinct from the prefix.  This permits the
    marginal public-OP estimator ``(T_high - T_low) / N`` without either graph
    reusing a dynamic input address.
    """

    lower_contract = build_nvidia_a800_vector_event_batch_contract(
        attributes, input_ring_mib=minimum_distinct_input_mib,
        public_call_count=32)
    lower_count = lower_contract["slot_count"]
    higher_count = 2 * lower_count
    contract = dict(lower_contract)
    del contract["public_call_count"]
    contract["protocol_version"] = (
        A800_VECTOR_DIFFERENTIAL_GRAPH_PROTOCOL_VERSION)
    contract["mode"] = (
        "unique_preallocated_vector_buffers_paired_differential_cuda_graph")
    contract["selection"] = (
        "coprime_unique_lower_prefix_and_equal_unique_higher_suffix")
    contract["reuse_contract"] = (
        "no_dynamic_input_slot_reuse_within_either_graph")
    minimum_distinct_input_bytes = minimum_distinct_input_mib * 1024 * 1024
    contract["minimum_distinct_input_bytes"] = minimum_distinct_input_bytes
    contract["requested_input_ring_bytes"] = 2 * minimum_distinct_input_bytes
    contract["slot_count"] = higher_count
    slot_stride = 1 if higher_count == 1 else higher_count // 2 + 1
    if slot_stride >= higher_count and higher_count != 1:
        slot_stride = 1
    while math.gcd(slot_stride, higher_count) != 1:
        slot_stride += 1
        if slot_stride >= higher_count:
            slot_stride = 1
    contract["slot_stride"] = slot_stride
    contract["allocated_input_ring_bytes"] = (
        higher_count * contract["input_bytes_per_public_call"])
    logical_output_bytes = higher_count * sum(
        contract["preallocated_output_bytes_per_operand"].values())
    is_cache_workload = attributes["op"] == "kvcache"
    if is_cache_workload:
        tok = _positive_workload_integer(
            attributes, "tok", "A800 Vector workload")
        num_kv_heads = _positive_workload_integer(
            attributes, "num_kv_heads", "A800 Vector workload")
        head_dim = _positive_workload_integer(
            attributes, "head_dim", "A800 Vector workload")
        dtype_bytes = _positive_workload_integer(
            attributes, "dtype_bytes", "A800 Vector workload")
        cache_slot_count = math.ceil(higher_count * tok / 16) * 16
        allocated_output_bytes = (
            cache_slot_count * num_kv_heads * head_dim * dtype_bytes * 2)
    else:
        allocated_output_bytes = logical_output_bytes
    contract["allocated_output_ring_bytes"] = allocated_output_bytes
    contract["output_padding_bytes"] = (
        allocated_output_bytes - logical_output_bytes)
    contract["lower_graph_public_call_count"] = lower_count
    contract["higher_graph_public_call_count"] = higher_count
    contract["graph_repetition_counts"] = [lower_count, higher_count]
    return contract


def build_nvidia_a800_vector_differential_graph_provenance(
        contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return immutable provenance for paired differential CUDA Graphs."""

    if not isinstance(contract, Mapping):
        raise TypeError(
            "A800 Vector differential CUDA-Graph contract must be a mapping")
    lower_count = contract["lower_graph_public_call_count"]
    higher_count = contract["higher_graph_public_call_count"]
    slot_count = contract["slot_count"]
    if contract["protocol_version"] != (
            A800_VECTOR_DIFFERENTIAL_GRAPH_PROTOCOL_VERSION):
        raise ValueError("A800 Vector differential CUDA-Graph protocol mismatch")
    if (isinstance(lower_count, bool) or not isinstance(lower_count, int)
            or lower_count <= 0 or higher_count != 2 * lower_count
            or slot_count != higher_count
            or contract["graph_repetition_counts"] != [
                lower_count, higher_count]):
        raise ValueError(
            "A800 Vector differential CUDA-Graph repetition mismatch")
    return {
        "protocol_version": A800_VECTOR_DIFFERENTIAL_GRAPH_PROTOCOL_VERSION,
        "graph_backend": "torch.cuda.CUDAGraph",
        "input_selection": (
            "shared_unique_lower_prefix_plus_equal_unique_higher_suffix"),
        "lower_graph_public_call_count": lower_count,
        "higher_graph_public_call_count": higher_count,
        "graph_repetition_counts": [lower_count, higher_count],
        "sample_pairing": (
            "alternating_lower_then_higher_and_higher_then_lower"),
        "duration_reporting": "(T_high-T_low)/(N_high-N_low)",
    }






def build_nvidia_a800_moe_topk_ids(
        *, token_counts: object, num_tokens: int, num_experts: int,
        top_k: int, context: str) -> tuple[tuple[int, ...], ...]:
    """Build deterministic top-k IDs with the exact canonical histogram."""

    for name, value in (
            ("num_tokens", num_tokens), ("num_experts", num_experts),
            ("top_k", top_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{context} {name} must be a positive integer")
    if top_k > num_experts:
        raise ValueError(f"{context} top_k cannot exceed num_experts")
    if not isinstance(token_counts, list):
        raise TypeError(f"{context} tok_per_expert must be a list")
    if len(token_counts) != num_experts:
        raise ValueError(
            f"{context} tok_per_expert length must equal num_experts")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in token_counts):
        raise TypeError(
            f"{context} tok_per_expert must contain only integers")
    if any(value < 0 or value > num_tokens for value in token_counts):
        raise ValueError(
            f"{context} tok_per_expert values must be in [0, num_tokens]")
    if sum(token_counts) != num_tokens * top_k:
        raise ValueError(
            f"{context} tok_per_expert must sum to num_tokens * top_k")

    remaining = [(-count, expert) for expert, count in enumerate(token_counts)
                 if count]
    heapq.heapify(remaining)
    rows = []
    for token_index in range(num_tokens):
        if len(remaining) < top_k:
            raise ValueError(
                f"{context} tok_per_expert cannot form {top_k} distinct "
                f"experts for token {token_index}")
        selected = [heapq.heappop(remaining) for _ in range(top_k)]
        rows.append(tuple(expert for _, expert in selected))
        for negative_count, expert in selected:
            if negative_count < -1:
                heapq.heappush(remaining, (negative_count + 1, expert))
    if remaining:
        raise ValueError(f"{context} tok_per_expert was not fully consumed")
    return tuple(rows)


def _remove_required_option(command: Sequence[str], option: str) -> list[str]:
    indexes = [index for index, value in enumerate(command) if value == option]
    if len(indexes) != 1:
        raise ValueError(
            f"runner command must contain exactly one {option}: {command!r}")
    index = indexes[0]
    if index + 1 >= len(command):
        raise ValueError(f"runner command has no value for {option}")
    return list(command[:index]) + list(command[index + 2:])


def _remove_optional_option(command: Sequence[str], option: str) -> list[str]:
    indexes = [index for index, value in enumerate(command) if value == option]
    if not indexes:
        return list(command)
    if len(indexes) != 1:
        raise ValueError(
            f"runner command repeats optional {option}: {command!r}")
    index = indexes[0]
    if index + 1 >= len(command):
        raise ValueError(f"runner command has no value for {option}")
    return list(command[:index]) + list(command[index + 2:])


def _support_decision(family: str, command: Sequence[str]) -> tuple[bool, str | None]:
    if family not in {"gemm", "flash_attn", "fused_moe", "vector"}:
        raise ValueError(f"unsupported NVIDIA A800 family: {family!r}")
    if not command:
        raise ValueError("A800 runner command must not be empty")
    return True, None


def build_nvidia_a800_event_plans(
        *, config_path: Path,
        selected_case_ids: Mapping[str, Sequence[str]]) -> list[Mapping[str, Any]]:
    """Build exact runner commands for the fixed A800 sample."""

    config = load_pipeline_config(str(config_path))
    if config["hardware_id"] != "nvidia_a800":
        raise ValueError("A800 Event capture requires nvidia_a800 config")
    selected_families = set(selected_case_ids)
    allowed_families = {"gemm", "flash_attn", "fused_moe", "vector"}
    if not selected_families or not selected_families <= allowed_families:
        raise ValueError(
            "A800 Event selection must contain known non-empty families")
    plans = []
    for family in ("gemm", "flash_attn", "fused_moe", "vector"):
        if family not in selected_case_ids:
            continue
        requested = list(selected_case_ids[family])
        if (not requested or len(requested) != len(set(requested))
                or any(not isinstance(case_id, str) or not case_id
                       for case_id in requested)):
            raise ValueError(
                f"A800 selected {family} case IDs must be unique and non-empty")
        kernel_config = _resolve_kernel_cfg(config, family)
        jobs = _build_jobs_with_executables(
            config, family, kernel_config, "ncu-not-used", sys.executable)
        dataset_path = Path(get_kernel_dataset_path(config, family))
        dataset = load_kernel_dataset_mapping(dataset_path)
        jobs = _apply_canonical_gemm_timing_dtypes(family, jobs, dataset)
        job_index = {job["filename"]: job for job in jobs}
        if len(job_index) != len(jobs):
            raise ValueError(f"A800 {family} runner jobs contain duplicate IDs")
        for case_id in requested:
            if case_id not in job_index:
                raise KeyError(
                    f"selected case is absent from A800 runner jobs: {case_id}")
            if case_id not in dataset:
                raise KeyError(
                    f"selected case is absent from A800 dataset: {case_id}")
            command = _remove_required_option(
                job_index[case_id]["runner_cmd"], "--nvtx_range_name")
            command = _remove_optional_option(command, "--tiling_output")
            if command[0] != sys.executable:
                raise ValueError(f"A800 runner uses unexpected Python: {case_id}")
            runner_path = Path(command[1]).resolve()
            if not runner_path.is_file():
                raise FileNotFoundError(
                    f"A800 runner is absent for {case_id}: {runner_path}")
            supported, unsupported_reason = _support_decision(family, command)
            attributes = dataset[case_id]["op_info"]["Attr"]
            if family == "fused_moe":
                topk_ids = build_nvidia_a800_moe_topk_ids(
                    token_counts=attributes["tok_per_expert"],
                    num_tokens=attributes["TOK"],
                    num_experts=attributes["E"],
                    top_k=attributes["TOPK"],
                    context=f"A800 canonical case {case_id}",
                )
                if len(topk_ids) != attributes["TOK"]:
                    raise AssertionError(
                        f"A800 canonical routing row mismatch: {case_id}")
                command += [
                    "--expected-tok-per-expert-json",
                    json.dumps(
                        attributes["tok_per_expert"], separators=(",", ":")),
                ]
            plans.append({
                "hardware_id": "nvidia_a800",
                "family": family,
                "case_id": case_id,
                "runner_path": runner_path,
                "command": command,
                "canonical_attributes": attributes,
                "canonical_attributes_sha256": canonical_json_sha256(
                    attributes),
                "supported": supported,
                "unsupported_reason": unsupported_reason,
            })
    return plans


def capture_nvidia_a800_supported_case(
        *, plan: Mapping[str, Any], output: Path, baseline_commit: str,
        source_snapshot_sha256: str, timer_sanity_sha256: str,
        warmup_iterations: int, sample_count: int,
        cache_eviction_mib: int = 0, vector_input_ring_mib: int = 0,
        vector_operation_repetitions_per_sample: int = 1,
        vector_event_timing_mode: str = "direct_batched_event",
        telemetry_physical_gpu_index: int | None = None,
        nonvector_operation_repetitions_per_sample: int = 1,
        ) -> Mapping[str, Any]:
    """Execute one supported plan and require a structured Event artifact."""

    if plan["supported"] is not True:
        raise ValueError(f"cannot execute unsupported plan: {plan['case_id']}")
    if (telemetry_physical_gpu_index is not None
            and (isinstance(telemetry_physical_gpu_index, bool)
                 or not isinstance(telemetry_physical_gpu_index, int)
                 or telemetry_physical_gpu_index < 0)):
        raise ValueError(
            "telemetry_physical_gpu_index must be a nonnegative integer or None")
    if (isinstance(nonvector_operation_repetitions_per_sample, bool)
            or not isinstance(nonvector_operation_repetitions_per_sample, int)
            or nonvector_operation_repetitions_per_sample not in
            A800_NONVECTOR_EVENT_STEADY_STATE_PUBLIC_CALL_COUNTS):
        raise ValueError(
            "A800 non-Vector Event repetitions must be one of 1, 8, 32")
    if (plan["family"] == "vector"
            and nonvector_operation_repetitions_per_sample != 1):
        raise ValueError(
            "A800 non-Vector Event repetitions cannot be used for Vector")
    command = [
        *plan["command"],
        "--event-output", str(output),
        "--event-case-id", plan["case_id"],
        "--event-baseline-commit", baseline_commit,
        "--event-source-snapshot-sha256", source_snapshot_sha256,
        "--event-timer-sanity-sha256", timer_sanity_sha256,
        "--event-canonical-attributes-sha256",
        plan["canonical_attributes_sha256"],
        "--event-warmup", str(warmup_iterations),
        "--event-samples", str(sample_count),
    ]
    if cache_eviction_mib:
        command += ["--event-cache-eviction-mib", str(cache_eviction_mib)]
    if telemetry_physical_gpu_index is not None:
        command += [
            "--event-telemetry-gpu", str(telemetry_physical_gpu_index)]
    if (plan["family"] != "vector"
            and nonvector_operation_repetitions_per_sample != 1):
        command += [
            "--event-steady-state-public-op-repetitions",
            str(nonvector_operation_repetitions_per_sample)]
    vector_batch_contract = None
    if plan["family"] == "vector":
        if vector_event_timing_mode not in A800_VECTOR_EVENT_TIMING_MODES:
            raise ValueError(
                "unsupported A800 Vector Event timing mode: "
                f"{vector_event_timing_mode}")
        uses_ring = vector_input_ring_mib != 0
        uses_batch = vector_operation_repetitions_per_sample != 1
        uses_graph = vector_event_timing_mode in {
            "cuda_graph_full_ring_replay", "cuda_graph_differential"}
        uses_differential_graph = (
            vector_event_timing_mode == "cuda_graph_differential")
        if uses_graph:
            if not uses_ring or uses_batch:
                raise ValueError(
                    "A800 Vector CUDA-Graph replay requires an input ring "
                    "and exactly one direct public call")
        elif uses_ring != uses_batch:
            raise ValueError(
                "A800 Vector capture requires ring and batching together")
        if uses_batch or uses_graph:
            if cache_eviction_mib != 0:
                raise ValueError(
                    "A800 Vector rotating ring capture cannot use a cache "
                    "eviction sweep")
            if uses_graph:
                if uses_differential_graph:
                    vector_batch_contract = (
                        build_nvidia_a800_vector_differential_graph_contract(
                            plan["canonical_attributes"],
                            minimum_distinct_input_mib=vector_input_ring_mib))
                else:
                    vector_batch_contract = (
                        build_nvidia_a800_vector_graph_contract(
                            plan["canonical_attributes"],
                            input_ring_mib=vector_input_ring_mib))
            else:
                vector_batch_contract = (
                    build_nvidia_a800_vector_event_batch_contract(
                        plan["canonical_attributes"],
                        input_ring_mib=vector_input_ring_mib,
                        public_call_count=(
                            vector_operation_repetitions_per_sample)))
            command += [
                "--event-input-ring-mib", str(vector_input_ring_mib),
                "--event-vector-timing-mode", vector_event_timing_mode,
            ]
            if not uses_graph:
                command += [
                    "--event-operation-repetitions-per-sample",
                    str(vector_operation_repetitions_per_sample),
                ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"A800 Event runner failed for {plan['case_id']}: "
            f"stderr={completed.stderr!r}")
    if not output.is_file():
        raise RuntimeError(
            f"A800 Event runner returned success without output: "
            f"{plan['case_id']}")
    if vector_batch_contract is not None:
        with output.open(encoding="utf-8") as stream:
            artifact = json.load(stream)
        if artifact["workload"]["input_ring"] != vector_batch_contract:
            raise ValueError(
                "A800 measured Vector input ring disagrees with plan: "
                f"{plan['case_id']}")
        if uses_differential_graph:
            expected_batch = (
                build_nvidia_a800_vector_differential_graph_provenance(
                    vector_batch_contract))
        elif uses_graph:
            expected_batch = build_nvidia_a800_vector_graph_provenance(
                vector_batch_contract)
        else:
            expected_batch = build_nvidia_a800_vector_event_batch_provenance(
                vector_operation_repetitions_per_sample)
        if artifact["workload"]["event_operation_batch"] != expected_batch:
            raise ValueError(
                "A800 measured Vector operation batch disagrees with plan: "
                f"{plan['case_id']}")
        measurement = artifact["measurement"]
        if uses_differential_graph:
            if (measurement["schema_version"] != 3
                    or measurement["mode"]
                    != "paired_differential_events_around_cuda_graph_replay"
                    or measurement["operation_cycle_length"]
                    != vector_batch_contract["slot_count"]
                    or measurement["repetition_counts"]
                    != vector_batch_contract["graph_repetition_counts"]
                    or measurement["distinct_input_contract"][
                        "configured_minimum_bytes"]
                    != vector_batch_contract[
                        "minimum_distinct_input_bytes"]):
                raise ValueError(
                    "A800 measured Vector differential CUDA-Graph replay "
                    f"disagrees with plan: {plan['case_id']}")
            if (artifact["workload"]["power_warmup"]
                    != build_nvidia_a800_vector_power_warmup_provenance()
                    or artifact["timer"]["power_state_condition"]
                    != "sustained_bf16_gemm_before_each_graph_replay"
                    or measurement[
                        "pre_event_operation_executions_per_replay"] != 1):
                raise ValueError(
                    "A800 measured Vector differential CUDA-Graph power "
                    f"conditioning disagrees with plan: {plan['case_id']}")
        elif uses_graph:
            if (measurement["schema_version"] != 3
                    or measurement["mode"]
                    != "external_events_around_cuda_graph_replay"
                    or measurement["operation_cycle_length"]
                    != vector_batch_contract["slot_count"]
                    or measurement["repetition_counts"]
                    != vector_batch_contract["graph_repetition_counts"]):
                raise ValueError(
                    "A800 measured Vector CUDA-Graph replay disagrees with "
                    f"plan: {plan['case_id']}")
        elif (measurement["schema_version"] != 4
              or measurement["operation_repetitions_per_sample"]
              != vector_operation_repetitions_per_sample):
            raise ValueError(
                "A800 measured Vector Event batch disagrees with plan: "
                f"{plan['case_id']}")
    if plan["family"] == "fused_moe":
        with output.open(encoding="utf-8") as stream:
            artifact = json.load(stream)
        for field, expected in (
                ("case_id", plan["case_id"]),
                ("family", "fused_moe"),
                ("canonical_attributes_sha256",
                 plan["canonical_attributes_sha256"])):
            if artifact[field] != expected:
                raise ValueError(
                    f"A800 measured FusedMoE {field} disagrees with plan: "
                    f"{plan['case_id']}")
        actual_counts = artifact["workload"]["tok_per_expert"]
        expected_counts = plan["canonical_attributes"]["tok_per_expert"]
        build_nvidia_a800_moe_topk_ids(
            token_counts=actual_counts,
            num_tokens=plan["canonical_attributes"]["TOK"],
            num_experts=plan["canonical_attributes"]["E"],
            top_k=plan["canonical_attributes"]["TOPK"],
            context=f"A800 measured case {plan['case_id']}",
        )
        if actual_counts != expected_counts:
            raise ValueError(
                "A800 measured FusedMoE routing disagrees with canonical Attr: "
                f"{plan['case_id']}")
    return {
        "case_id": plan["case_id"],
        "status": "measured",
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output": str(output),
    }
