"""CUDA Event execution for preallocated NVIDIA A800 Vector workloads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from dataset_pipeline.kernel_dataset.nvidia_a800_event_capture import (
    A800_VECTOR_EVENT_TIMING_MODES,
    build_nvidia_a800_vector_differential_graph_provenance,
    build_nvidia_a800_vector_event_batch_provenance,
    build_nvidia_a800_vector_graph_provenance,
    build_nvidia_a800_vector_power_warmup_provenance,
)


class NvidiaA800EventOptions(Protocol):
    cache_eviction_mib: int
    warmup_iterations: int
    sample_count: int


class NvidiaA800VectorEventBackend(Protocol):
    measure_current_stream_events: Callable[..., Mapping[str, Any]]
    measure_differential_rotating_graph_replay_events: Callable[
        ..., Mapping[str, Any]]
    measure_graph_replay_events: Callable[..., Mapping[str, Any]]
    measure_rotating_graph_replay_events: Callable[..., Mapping[str, Any]]
    prepare_bf16_cache_eviction_sweep: Callable[..., tuple[Any, int]]
    validate_cache_eviction_mib: Callable[..., None]
    write_event_case_record: Callable[..., None]


def make_rotating_operation(
        slot_count: int, slot_stride: int,
        operation: Callable[[int], Any]) -> Callable[[], None]:
    if (isinstance(slot_count, bool) or not isinstance(slot_count, int)
            or slot_count <= 0):
        raise ValueError("slot_count must be a positive integer")
    if (isinstance(slot_stride, bool) or not isinstance(slot_stride, int)
            or slot_stride <= 0
            or (slot_count > 1 and slot_stride >= slot_count)
            or (slot_count == 1 and slot_stride != 1)):
        raise ValueError(
            "slot_stride must be 1 for a one-slot ring or in "
            "(0, slot_count) otherwise")
    if math.gcd(slot_count, slot_stride) != 1:
        raise ValueError("slot_stride must be coprime with slot_count")
    if not callable(operation):
        raise TypeError("rotating operation must be callable")
    slot_index = 0
    latest_result = None

    def rotating_operation() -> None:
        nonlocal latest_result, slot_index
        index = slot_index
        slot_index = (slot_index + slot_stride) % slot_count
        latest_result = operation(index)
        return None

    def fixed_operation(index: int) -> Callable[[], Any]:
        def invoke() -> Any:
            return operation(index)
        return invoke

    # CUDA Graph capture must bind every input pointer in its replay plan.
    # Keeping the fixed tuple on the same callable avoids a second workload
    # implementation: direct timing advances it dynamically, while graph timing
    # captures one complete coprime traversal of those identical public calls.
    rotating_operation.full_ring_operations = tuple(  # type: ignore[attr-defined]
        fixed_operation((offset * slot_stride) % slot_count)
        for offset in range(slot_count))

    return rotating_operation


def allocate_ring_tensor(
        *, torch_module: Any, shape: tuple[int, ...], dtype: Any,
        contract: Mapping[str, Any], contract_section: str, operand: str,
        empty: bool = False) -> Any:
    full_shape = (contract["slot_count"], *shape)
    if empty:
        tensor = torch_module.empty(*full_shape, device="cuda", dtype=dtype)
    else:
        tensor = torch_module.randn(*full_shape, device="cuda", dtype=dtype)
    actual_bytes = tensor.numel() * tensor.element_size()
    expected_bytes = (
        contract["slot_count"] * contract[contract_section][operand])
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{operand} ring byte mismatch: expected={expected_bytes}, "
            f"actual={actual_bytes}")
    return tensor


def measure_nvidia_a800_vector_event(
        operation: Callable[[], Any], *, torch_module: Any,
        event_options: NvidiaA800EventOptions,
        event_backend: NvidiaA800VectorEventBackend,
        workload: Mapping[str, Any],
        allocation_evidence: str,
        input_ring_contract: Mapping[str, Any] | None,
        operation_repetitions_per_sample: int, timing_mode: str,
        runner_path: Path,
        environment: Mapping[str, Any],
        power_maintenance_operation: Callable[[], Any] | None = None,
        power_warmup: Mapping[str, Any] | None = None,
        ) -> None:
    measure_current_stream_events = (
        event_backend.measure_current_stream_events)
    measure_differential_rotating_graph_replay_events = (
        event_backend.measure_differential_rotating_graph_replay_events)
    measure_graph_replay_events = event_backend.measure_graph_replay_events
    measure_rotating_graph_replay_events = (
        event_backend.measure_rotating_graph_replay_events)
    prepare_bf16_cache_eviction_sweep = (
        event_backend.prepare_bf16_cache_eviction_sweep)
    write_event_case_record = event_backend.write_event_case_record
    if timing_mode not in A800_VECTOR_EVENT_TIMING_MODES:
        raise ValueError(f"unsupported A800 Vector timing mode: {timing_mode}")
    event_backend.validate_cache_eviction_mib(
        event_options.cache_eviction_mib, minimum_nonzero_mib=320,
        hardware_id="nvidia_a800")
    uses_differential_graph = timing_mode == "cuda_graph_differential"
    uses_graph = timing_mode in {
        "cuda_graph_full_ring_replay", "cuda_graph_differential"}
    if ((power_maintenance_operation is None) != (power_warmup is None)):
        raise ValueError(
            "A800 Vector power operation and provenance must be supplied together")
    if power_maintenance_operation is not None:
        if timing_mode != "cuda_graph_differential":
            raise ValueError(
                "A800 Vector power maintenance is only valid for differential "
                "CUDA-Graph timing")
        if not callable(power_maintenance_operation):
            raise TypeError("A800 Vector power maintenance must be callable")
        if power_warmup != build_nvidia_a800_vector_power_warmup_provenance():
            raise ValueError("invalid A800 Vector power-warmup provenance")
    elif timing_mode == "cuda_graph_differential":
        raise ValueError(
            "A800 Vector differential CUDA-Graph timing requires power "
            "maintenance")
    if input_ring_contract is None:
        if timing_mode != "direct_batched_event":
            raise ValueError("A800 Vector CUDA-Graph replay requires an input ring")
        if operation_repetitions_per_sample != 1:
            raise ValueError(
                "batched A800 Vector Event timing requires an input ring")
        cache_eviction_operation, cache_eviction_bytes = (
            prepare_bf16_cache_eviction_sweep(
                torch_module=torch_module, device="cuda",
                cache_eviction_mib=event_options.cache_eviction_mib))
        timer_mode = "one_exact_public_op_per_event_interval"
        if cache_eviction_bytes:
            cache_condition = "queued_pre_sample_device_sweep"
        else:
            cache_condition = "ordinary_warm_state"
        output_behavior = "exact_runner_call"
    elif timing_mode == "direct_batched_event":
        if event_options.cache_eviction_mib != 0:
            raise ValueError(
                "A800 Vector input ring cannot be mixed with a cache sweep")
        if input_ring_contract["public_call_count"] != (
                operation_repetitions_per_sample):
            raise ValueError(
                "A800 Vector input-ring and Event batch counts disagree")
        cache_eviction_operation, cache_eviction_bytes = None, 0
        timer_mode = (
            "fixed_count_public_ops_over_full_ring_cycle_per_event_interval_"
            "reported_per_call")
        cache_condition = input_ring_contract["mode"]
        output_behavior = "preallocated_rotating_output_ring_or_in_place"

    elif uses_differential_graph:
        if event_options.cache_eviction_mib != 0:
            raise ValueError(
                "A800 Vector differential CUDA-Graph input ring cannot mix "
                "a cache sweep")
        if operation_repetitions_per_sample != 1:
            raise ValueError(
                "A800 Vector differential CUDA-Graph replay does not use "
                "direct batches")
        if not hasattr(operation, "full_ring_operations"):
            raise ValueError(
                "A800 Vector differential CUDA-Graph replay requires fixed "
                "ring operations")
        operations = operation.full_ring_operations  # type: ignore[attr-defined]
        if not isinstance(operations, tuple) or len(operations) < 2:
            raise ValueError(
                "A800 Vector differential CUDA-Graph replay needs at least "
                "two fixed operations")
        expected_counts = input_ring_contract["graph_repetition_counts"]
        if (expected_counts != [
                input_ring_contract["lower_graph_public_call_count"],
                input_ring_contract["higher_graph_public_call_count"]]
                or len(operations) != expected_counts[1]):
            raise ValueError(
                "A800 Vector differential CUDA-Graph replay count disagrees "
                "with input ring")
        cache_eviction_operation, cache_eviction_bytes = None, 0
        timer_mode = "paired_differential_events_around_cuda_graph_replay"
        cache_condition = input_ring_contract["mode"]
        output_behavior = "preallocated_unique_output_ring_or_in_place"

    elif timing_mode == "cuda_graph_full_ring_replay":
        if event_options.cache_eviction_mib != 0:
            raise ValueError(
                "A800 Vector CUDA-Graph input ring cannot mix a cache sweep")
        if operation_repetitions_per_sample != 1:
            raise ValueError(
                "A800 Vector CUDA-Graph replay does not use direct batches")
        if not hasattr(operation, "full_ring_operations"):
            raise ValueError(
                "A800 Vector CUDA-Graph replay requires a full-ring operation")
        operations = operation.full_ring_operations  # type: ignore[attr-defined]
        if not isinstance(operations, tuple) or not operations:
            raise ValueError(
                "A800 Vector CUDA-Graph replay needs non-empty fixed operations")
        expected_counts = input_ring_contract["graph_repetition_counts"]
        if expected_counts != [len(operations), 2 * len(operations)]:
            raise ValueError(
                "A800 Vector CUDA-Graph replay count disagrees with input ring")
        cache_eviction_operation, cache_eviction_bytes = None, 0
        timer_mode = (
            "external_cuda_events_around_full_ring_cuda_graph_replay_"
            "reported_per_public_call")
        cache_condition = input_ring_contract["mode"]
        output_behavior = "preallocated_rotating_output_ring_or_in_place"
    else:
        raise AssertionError(
            f"validated A800 Vector timing mode was not dispatched: {timing_mode}")

    if uses_differential_graph:
        measurement = measure_differential_rotating_graph_replay_events(
            operations, device_module=torch_module.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            repetition_counts=tuple(expected_counts),
            per_call_distinct_input_bytes=input_ring_contract[
                "input_bytes_per_public_call"],
            minimum_distinct_input_bytes=input_ring_contract[
                "minimum_distinct_input_bytes"],
            hardware_l2_bytes=input_ring_contract["hardware_l2_bytes"],
            pre_event_operation=power_maintenance_operation)
    elif uses_graph:
        if len(operations) == 1:
            measurement = measure_graph_replay_events(
                operations[0], device_module=torch_module.cuda,
                warmup_iterations=event_options.warmup_iterations,
                sample_count=event_options.sample_count,
                repetition_counts=tuple(expected_counts))
        else:
            measurement = measure_rotating_graph_replay_events(
                operations, device_module=torch_module.cuda,
                warmup_iterations=event_options.warmup_iterations,
                sample_count=event_options.sample_count,
                repetition_counts=tuple(expected_counts))
    elif timing_mode == "direct_batched_event":
        measurement = measure_current_stream_events(
            operation, device_module=torch_module.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes,
            operation_repetitions_per_sample=operation_repetitions_per_sample)
    else:
        raise AssertionError(
            f"validated A800 Vector timing mode was not measured: {timing_mode}")
    normalized_workload = dict(workload)
    if uses_graph:
        normalized_workload["cache_eviction_bytes_before_replay"] = (
            cache_eviction_bytes)
    else:
        normalized_workload["cache_eviction_bytes_before_sample"] = (
            cache_eviction_bytes)
    if input_ring_contract is not None:
        normalized_workload["input_ring"] = dict(input_ring_contract)
        if uses_graph:
            normalized_workload["event_measurement_mode"] = timing_mode
            if uses_differential_graph:
                normalized_workload["distinct_input_contract"] = measurement[
                    "distinct_input_contract"]
                normalized_workload["event_operation_batch"] = dict(
                    build_nvidia_a800_vector_differential_graph_provenance(
                        input_ring_contract))
            else:
                normalized_workload["event_operation_batch"] = dict(
                    build_nvidia_a800_vector_graph_provenance(
                        input_ring_contract))
        else:
            normalized_workload["event_operation_batch"] = dict(
                build_nvidia_a800_vector_event_batch_provenance(
                    operation_repetitions_per_sample))
    if power_warmup is not None:
        normalized_workload["power_warmup"] = dict(power_warmup)
    write_event_case_record(
        options=event_options, hardware_id="nvidia_a800", family="vector",
        runner_path=runner_path,
        timer={
            "api": "torch.cuda.Event",
            "implementation": "CUDA_Runtime_Event",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_used",
            "mode": timer_mode,
            "cache_condition": cache_condition,
            "power_state_condition": (
                "sustained_bf16_gemm_before_each_graph_replay"
                if power_warmup is not None else "ordinary_warm_state"),
        },
        stream_contract={
            "event_stream": "torch.cuda.current_stream",
            "all_op_streams_joined": True,
            "evidence": (
                "the exact selected vector operator contains one canonical "
                "kernel and the call path has no explicit secondary stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": output_behavior,
            "workspace_behavior": "exact_runner_call",
            "evidence": allocation_evidence,
        },
        workload=normalized_workload, environment=environment,
        measurement=measurement)
