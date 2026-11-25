"""Vector/elementwise workload implementation for 910B dataset capture.

Profiles the LLM glue ops (AIV/vector-domain on 910B) that the single-op
dataset previously lacked entirely (only GEMM/MoE/FA existed). Each op mirrors
how vllm-ascend serves it, so msprof captures the real per-pipe cost.

Ops (--op):
  rmsnorm      : torch_npu.npu_rms_norm(x, weight, eps)          (standalone, no residual)
  add_rmsnorm  : torch_npu.npu_add_rms_norm(x, residual, w, eps) (residual+norm fused — serving main path)
  rope         : torch_npu._npu_rotary_embedding(...)            (Llama standalone, ATB in-place)
  silu         : torch_npu.npu_swiglu(concat[gate|up])           (Llama SwiGLU = AscendSiluAndMul)
  elementwise  : x + y  (aclnnAdd; --op2 mul for aclnnMul)
  kvcache      : vllm reshape_and_cache (→ aclnnReshapeAndCacheNd on NPU)

Each op mirrors how vllm-ascend serves it (verified against
vllm_ascend/ops/{layernorm,rotary_embedding,activation}.py + upstream
vllm/.../layers/{layernorm,rotary_embedding/base,activation}.py). Small token
counts (tok∈{1..16}) are the decode error hotspot, so the grid emphasizes those.
Real model dims — see configs/dataset/kernel/vector.yaml
`layers:` (Qwen3/Llama/DeepSeek).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/kernel_dataset/ascend_910b/vector_runner.py"

from dataset_pipeline.kernel_dataset.ascend_910b_event_capture import (
    ASCEND_910B_L2_BYTES,
    ASCEND_VECTOR_DIRECT_MODE,
    ASCEND_VECTOR_GRAPH_REPETITION_SELECTION_RULE,
    ASCEND_VECTOR_NPU_GRAPH_MODE,
)
DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
TIMELINE_POWER_WARMUP_PUBLIC_CALL_COUNT = 20


def _missing_event_backend(*_args, **_kwargs):
    raise RuntimeError(
        "Ascend Vector runtime requires the script-owned Event backend")


EventRunOptions = _missing_event_backend
add_event_cli_arguments = _missing_event_backend
file_sha256 = _missing_event_backend
measure_differential_rotating_npu_graph_replay_events = _missing_event_backend
measure_current_stream_events = _missing_event_backend
parse_event_run_options = _missing_event_backend
prepare_bf16_cache_eviction_sweep = _missing_event_backend
require_lower_hex = _missing_event_backend
validate_cache_eviction_mib = _missing_event_backend
write_event_case_record = _missing_event_backend
write_kernel_golden_json = _missing_event_backend


def configure_event_backend(backend):
    """Inject the cross-owner Event API from the executable script boundary."""

    global EventRunOptions
    global add_event_cli_arguments
    global file_sha256
    global measure_differential_rotating_npu_graph_replay_events
    global measure_current_stream_events
    global parse_event_run_options
    global prepare_bf16_cache_eviction_sweep
    global require_lower_hex
    global validate_cache_eviction_mib
    global write_event_case_record
    global write_kernel_golden_json
    EventRunOptions = backend.EventRunOptions
    add_event_cli_arguments = backend.add_event_cli_arguments
    file_sha256 = backend.file_sha256
    measure_differential_rotating_npu_graph_replay_events = (
        backend.measure_differential_rotating_npu_graph_replay_events)
    measure_current_stream_events = backend.measure_current_stream_events
    parse_event_run_options = backend.parse_event_run_options
    prepare_bf16_cache_eviction_sweep = (
        backend.prepare_bf16_cache_eviction_sweep)
    require_lower_hex = backend.require_lower_hex
    validate_cache_eviction_mib = backend.validate_cache_eviction_mib
    write_event_case_record = backend.write_event_case_record
    write_kernel_golden_json = backend.write_kernel_golden_json


def _warm_run(fn, warmup=5, profile_sweep=None):
    torch.npu.synchronize()
    for _ in range(warmup):
        if profile_sweep is not None:
            profile_sweep()
        fn()
    torch.npu.synchronize()
    if profile_sweep is not None:
        profile_sweep()
    out = fn()
    torch.npu.synchronize()
    return out
def _timeline_run(fn, profile_sweep, public_call_count,
                  warmup_public_call_count=0):
    """Submit exactly one cold public call per retained Timeline sample."""
    if not isinstance(public_call_count, int):
        raise TypeError("timeline_public_call_count must be an integer")
    if public_call_count <= 0:
        raise ValueError("timeline_public_call_count must be positive")
    if (isinstance(warmup_public_call_count, bool)
            or not isinstance(warmup_public_call_count, int)
            or warmup_public_call_count < 0):
        raise ValueError("timeline warmup public-call count must be nonnegative")
    torch.npu.synchronize()
    out = None
    for _ in range(warmup_public_call_count):
        out = fn()
    torch.npu.synchronize()
    for _ in range(public_call_count):
        profile_sweep()
        out = fn()
    torch.npu.synchronize()
    return out
def _environment():
    device_index = torch.npu.current_device()
    properties = torch.npu.get_device_properties(device_index)
    return {
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device_index": device_index,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "input_seed": 20260826,
    }
def _validate_event_input_ring_mib(event_input_ring_mib):
    if (isinstance(event_input_ring_mib, bool)
            or not isinstance(event_input_ring_mib, int)):
        raise TypeError("event_input_ring_mib must be an integer")
    if event_input_ring_mib < 0:
        raise ValueError("event_input_ring_mib must be nonnegative")
    if 0 < event_input_ring_mib < 1536:
        raise ValueError(
            "Ascend Vector input ring must be 0 or at least 1536 MiB "
            "(8x L2)")


def _validate_event_operation_repetitions(operation_repetitions):
    if (isinstance(operation_repetitions, bool)
            or not isinstance(operation_repetitions, int)):
        raise TypeError(
            "event_operation_repetitions_per_sample must be an integer")
    if operation_repetitions not in {1, 32, 512, 2048, 8192}:
        raise ValueError(
            "Ascend Vector Event operation repetitions must be 1, 32, 512, "
            "2048, or 8192")


def _validate_graph_arguments(
        event_vector_mode, event_graph_repetitions,
        event_graph_buffer_count):
    if event_vector_mode == ASCEND_VECTOR_DIRECT_MODE:
        if event_graph_repetitions is not None or event_graph_buffer_count != 0:
            raise ValueError(
                "Ascend Vector graph arguments require NPUGraph mode")
        return
    if event_vector_mode != ASCEND_VECTOR_NPU_GRAPH_MODE:
        raise ValueError(
            f"unsupported Ascend Vector Event mode: {event_vector_mode!r}")
    if (not isinstance(event_graph_repetitions, tuple)
            or len(event_graph_repetitions) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in event_graph_repetitions)):
        raise ValueError(
            "Ascend Vector NPUGraph repetitions must be two positive integers")
    lower_count, higher_count = event_graph_repetitions
    if lower_count >= higher_count:
        raise ValueError(
            "Ascend Vector NPUGraph repetitions must be strictly increasing")
    if (isinstance(event_graph_buffer_count, bool)
            or not isinstance(event_graph_buffer_count, int)
            or event_graph_buffer_count != higher_count):
        raise ValueError(
            "Ascend Vector NPUGraph buffer count must equal the higher "
            "repetition count")


def _build_graph_call_set(buffer_count, allocate_call):
    """Materialize one independent storage group per captured public call."""

    if (isinstance(buffer_count, bool) or not isinstance(buffer_count, int)
            or buffer_count <= 0):
        raise ValueError("graph buffer count must be a positive integer")
    if not callable(allocate_call):
        raise TypeError("graph call allocator must be callable")
    operations = []
    storage_groups = []
    for index in range(buffer_count):
        operation, storage_group = allocate_call(index)
        if not callable(operation):
            raise TypeError(f"graph call {index} operation is not callable")
        if not isinstance(storage_group, tuple) or not storage_group:
            raise TypeError(
                f"graph call {index} storage group must be a non-empty tuple")
        operations.append(operation)
        storage_groups.append(storage_group)
    return tuple(operations), tuple(storage_groups)


def _normal_dense(shape, *, dtype, device):
    """Allocate one backing tensor, then initialize it in place."""

    backing = torch.empty(shape, dtype=dtype, device=device)
    backing.normal_()
    return backing


def _require_distinct_slots(slot_count, minimum_distinct_slots):
    if (isinstance(minimum_distinct_slots, bool)
            or not isinstance(minimum_distinct_slots, int)
            or minimum_distinct_slots <= 0):
        raise ValueError("minimum distinct Vector ring slots must be positive")
    if slot_count < minimum_distinct_slots:
        raise ValueError(
            "Ascend Vector input ring has fewer slots than public calls in "
            "one Event interval")


def _rotating_input_operation(
        shape, dtype, event_input_ring_mib, operation, *,
        minimum_distinct_slots=1):
    """Return an exact public-call operation over disjoint dense input slots."""
    _validate_event_input_ring_mib(event_input_ring_mib)
    if event_input_ring_mib == 0:
        return None, None
    if (not isinstance(shape, tuple) or not shape
            or any(not isinstance(value, int) or value <= 0
                   for value in shape)):
        raise ValueError(f"input ring shape must contain positive integers: {shape!r}")
    input_element_count = math.prod(shape)
    input_bytes = input_element_count * dtype.itemsize
    requested_bytes = event_input_ring_mib * 1024 * 1024
    slot_count = math.ceil(requested_bytes / input_bytes)
    allocated_bytes = slot_count * input_bytes
    _require_distinct_slots(slot_count, minimum_distinct_slots)
    backing = _normal_dense(
        (slot_count, input_element_count), dtype=dtype, device="npu")
    if not backing.is_contiguous() or backing.storage_offset() != 0:
        raise RuntimeError(
            "rotating input ring requires a zero-offset contiguous backing")
    slot_stride = max(1, slot_count // 2 + 1)
    while math.gcd(slot_stride, slot_count) != 1:
        slot_stride += 1
    slot_index = 0
    latest_result = None
    def rotating_operation():
        nonlocal latest_result, slot_index
        input_tensor = backing[slot_index].view(shape)
        if not input_tensor.is_contiguous():
            raise RuntimeError("rotating input slot must stay contiguous")
        slot_index = (slot_index + slot_stride) % slot_count
        latest_result = operation(input_tensor)
        return None
    contract = {
        "mode": "rotating_dense_input_backing_ring",
        "requested_bytes": requested_bytes,
        "allocated_bytes": allocated_bytes,
        "input_bytes_per_call": input_bytes,
        "slot_count": slot_count,
        "slot_stride": slot_stride,
        "selection": (
            "coprime_stride_permutation_of_disjoint_contiguous_backing_slots"),
        "storage_offset": "slot_index_times_input_element_count",
        "initialization": "torch.empty_then_normal_before_timing",
        "pre_target_sweep": "none",
    }
    return rotating_operation, contract


def _rotating_paired_input_operation(
        shape, dtype, event_input_ring_mib, operation, *,
        dynamic_input_operands, static_input_operands,
        minimum_distinct_slots=1):
    """Return an exact two-input public call over paired dense ring slots."""
    _validate_event_input_ring_mib(event_input_ring_mib)
    if event_input_ring_mib == 0:
        return None, None
    if event_input_ring_mib < 2048:
        raise ValueError(
            "Ascend paired Vector input ring must be at least 2048 MiB")
    if (not isinstance(shape, tuple) or not shape
            or any(not isinstance(value, int) or value <= 0
                   for value in shape)):
        raise ValueError(
            f"paired input ring shape must contain positive integers: {shape!r}")
    if (not isinstance(dynamic_input_operands, tuple)
            or len(dynamic_input_operands) != 2
            or len(set(dynamic_input_operands)) != 2
            or any(not isinstance(name, str) or not name
                   for name in dynamic_input_operands)):
        raise ValueError(
            "paired input ring requires two distinct dynamic input names")
    if (not isinstance(static_input_operands, tuple)
            or len(set(static_input_operands)) != len(static_input_operands)
            or any(not isinstance(name, str) or not name
                   for name in static_input_operands)
            or set(dynamic_input_operands) & set(static_input_operands)):
        raise ValueError("paired input ring static input names are invalid")
    input_element_count = math.prod(shape)
    input_bytes_per_operand = input_element_count * dtype.itemsize
    input_bytes_per_call = 2 * input_bytes_per_operand
    requested_bytes = event_input_ring_mib * 1024 * 1024
    slot_count = math.ceil(requested_bytes / input_bytes_per_call)
    allocated_bytes = slot_count * input_bytes_per_call
    _require_distinct_slots(slot_count, minimum_distinct_slots)
    backing = _normal_dense(
        (slot_count, 2, input_element_count), dtype=dtype, device="npu")
    if not backing.is_contiguous() or backing.storage_offset() != 0:
        raise RuntimeError(
            "rotating paired input ring requires a zero-offset contiguous backing")
    slot_stride = max(1, slot_count // 2 + 1)
    while math.gcd(slot_stride, slot_count) != 1:
        slot_stride += 1
    slot_index = 0
    latest_result = None
    def rotating_operation():
        nonlocal latest_result, slot_index
        input_pair = backing[slot_index]
        first = input_pair[0].view(shape)
        second = input_pair[1].view(shape)
        if not first.is_contiguous() or not second.is_contiguous():
            raise RuntimeError("rotating paired input slots must stay contiguous")
        slot_index = (slot_index + slot_stride) % slot_count
        latest_result = operation(first, second)
        return None
    contract = {
        "mode": "rotating_paired_dense_input_backing_ring",
        "requested_bytes": requested_bytes,
        "allocated_bytes": allocated_bytes,
        "input_bytes_per_call": input_bytes_per_call,
        "input_bytes_per_operand": input_bytes_per_operand,
        "dynamic_input_operands": list(dynamic_input_operands),
        "static_input_operands": list(static_input_operands),
        "slot_count": slot_count,
        "slot_stride": slot_stride,
        "selection": (
            "coprime_stride_permutation_of_disjoint_contiguous_paired_backing_slots"),
        "storage_offset": "slot_index_times_paired_input_element_count",
        "initialization": "torch.empty_then_normal_before_timing",
        "pre_target_sweep": "none",
    }
    return rotating_operation, contract


def _rotating_asymmetric_paired_input_operation(
        first_shape, second_shape, dtype, event_input_ring_mib, operation, *,
        dynamic_input_operands, static_input_operands,
        minimum_distinct_slots=1):
    """Rotate two differently shaped dense inputs through a shared cold ring."""
    _validate_event_input_ring_mib(event_input_ring_mib)
    if event_input_ring_mib == 0:
        return None, None
    for name, shape in (("first", first_shape), ("second", second_shape)):
        if (not isinstance(shape, tuple) or not shape
                or any(not isinstance(value, int) or value <= 0
                       for value in shape)):
            raise ValueError(
                f"asymmetric paired input {name} shape is invalid: {shape!r}")
    if (not isinstance(dynamic_input_operands, tuple)
            or len(dynamic_input_operands) != 2
            or len(set(dynamic_input_operands)) != 2
            or any(not isinstance(name, str) or not name
                   for name in dynamic_input_operands)):
        raise ValueError(
            "asymmetric paired input ring requires two distinct dynamic input names")
    if (not isinstance(static_input_operands, tuple)
            or len(set(static_input_operands)) != len(static_input_operands)
            or any(not isinstance(name, str) or not name
                   for name in static_input_operands)
            or set(dynamic_input_operands) & set(static_input_operands)):
        raise ValueError("asymmetric paired input ring static input names are invalid")
    first_elements = math.prod(first_shape)
    second_elements = math.prod(second_shape)
    first_bytes = first_elements * dtype.itemsize
    second_bytes = second_elements * dtype.itemsize
    input_bytes_per_call = first_bytes + second_bytes
    requested_bytes = event_input_ring_mib * 1024 * 1024
    slot_count = math.ceil(requested_bytes / input_bytes_per_call)
    allocated_bytes = slot_count * input_bytes_per_call
    _require_distinct_slots(slot_count, minimum_distinct_slots)
    first_backing = _normal_dense(
        (slot_count, first_elements), dtype=dtype, device="npu")
    second_backing = _normal_dense(
        (slot_count, second_elements), dtype=dtype, device="npu")
    slot_stride = max(1, slot_count // 2 + 1)
    while math.gcd(slot_stride, slot_count) != 1:
        slot_stride += 1
    slot_index = 0
    latest_result = None

    def rotating_operation():
        nonlocal latest_result, slot_index
        first = first_backing[slot_index].view(first_shape)
        second = second_backing[slot_index].view(second_shape)
        if not first.is_contiguous() or not second.is_contiguous():
            raise RuntimeError("asymmetric paired input slots must stay contiguous")
        slot_index = (slot_index + slot_stride) % slot_count
        latest_result = operation(first, second)
        return None

    return rotating_operation, {
        "mode": "rotating_asymmetric_paired_dense_input_backing_ring",
        "requested_bytes": requested_bytes,
        "allocated_bytes": allocated_bytes,
        "input_bytes_per_call": input_bytes_per_call,
        "input_bytes_per_operand": {
            dynamic_input_operands[0]: first_bytes,
            dynamic_input_operands[1]: second_bytes,
        },
        "dynamic_input_operands": list(dynamic_input_operands),
        "static_input_operands": list(static_input_operands),
        "slot_count": slot_count,
        "slot_stride": slot_stride,
        "selection": (
            "coprime_stride_permutation_of_disjoint_asymmetric_dense_input_slots"),
        "storage_offset": (
            "slot_index_times_per_operand_element_count_in_separate_backings"),
        "initialization": "torch.empty_then_normal_before_timing",
        "pre_target_sweep": "none",
    }


def _rotating_kvcache_operation(
        key_shape, cache_shape, dtype, event_input_ring_mib, operation, *,
        minimum_distinct_slots=1):
    """Rotate KV inputs and their non-overlapping cache destinations together."""
    _validate_event_input_ring_mib(event_input_ring_mib)
    if event_input_ring_mib == 0:
        return None, None
    for name, shape in (("key", key_shape), ("cache", cache_shape)):
        if (not isinstance(shape, tuple) or not shape
                or any(not isinstance(value, int) or value <= 0
                       for value in shape)):
            raise ValueError(f"KV-cache rotating {name} shape is invalid: {shape!r}")
    key_elements = math.prod(key_shape)
    cache_elements = math.prod(cache_shape)
    operand_bytes = {
        "key": key_elements * dtype.itemsize,
        "value": key_elements * dtype.itemsize,
        "key_cache": cache_elements * dtype.itemsize,
        "value_cache": cache_elements * dtype.itemsize,
    }
    input_bytes_per_call = sum(operand_bytes.values())
    requested_bytes = event_input_ring_mib * 1024 * 1024
    slot_count = math.ceil(requested_bytes / input_bytes_per_call)
    allocated_bytes = slot_count * input_bytes_per_call
    _require_distinct_slots(slot_count, minimum_distinct_slots)
    key_backing = _normal_dense(
        (slot_count, *key_shape), dtype=dtype, device="npu")
    value_backing = _normal_dense(
        (slot_count, *key_shape), dtype=dtype, device="npu")
    key_cache_backing = torch.zeros(
        (slot_count, *cache_shape), dtype=dtype, device="npu")
    value_cache_backing = torch.zeros(
        (slot_count, *cache_shape), dtype=dtype, device="npu")
    slot_stride = max(1, slot_count // 2 + 1)
    while math.gcd(slot_stride, slot_count) != 1:
        slot_stride += 1
    slot_index = 0

    def rotating_operation():
        nonlocal slot_index
        index = slot_index
        slot_index = (slot_index + slot_stride) % slot_count
        return operation(
            key_backing[index], value_backing[index],
            key_cache_backing[index], value_cache_backing[index])

    return rotating_operation, {
        "mode": "rotating_kvcache_input_and_destination_ring",
        "requested_bytes": requested_bytes,
        "allocated_bytes": allocated_bytes,
        "input_bytes_per_call": input_bytes_per_call,
        "input_bytes_per_operand": operand_bytes,
        "dynamic_input_operands": [
            "key", "value", "key_cache", "value_cache"],
        "static_input_operands": ["slot_indices"],
        "slot_count": slot_count,
        "slot_stride": slot_stride,
        "selection": (
            "coprime_stride_permutation_of_disjoint_kv_inputs_and_cache_destinations"),
        "storage_offset": (
            "slot_index_times_per_operand_element_count_in_separate_backings"),
        "initialization": "random_inputs_zeroed_cache_before_timing",
        "pre_target_sweep": "none",
    }


def _prepare_device_power_warmup(gemm_iterations):
    if gemm_iterations == 0:
        return None, None
    if gemm_iterations != 1000:
        raise ValueError(
            "Ascend Vector device power warmup must use exactly 1000 GEMMs")
    left = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu")
    right = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu")
    output = torch.empty(4096, 4096, dtype=torch.bfloat16, device="npu")
    def operation():
        return torch.mm(left, right, out=output)

    for _ in range(gemm_iterations):
        operation()
    torch.npu.synchronize()
    return operation, {
        "kind": "bf16_gemm_power_warmup_and_per_event_maintenance",
        "matrix_dimension": 4096,
        "initial_gemm_iterations": gemm_iterations,
        "per_event_gemm_iterations": 1,
        "per_event_placement": (
            "same_current_stream_immediately_before_start_event"),
        "synchronized_before_event_samples": True,
        "included_in_event_interval": False,
    }


def _validate_timeline_power_warmup(gemm_iterations):
    if gemm_iterations not in {0, 1000}:
        raise ValueError(
            "Ascend Vector Timeline power warmup must be 0 or 1000 GEMMs")


def _run(fn, *, event_options: EventRunOptions | None, workload,
         allocation_evidence, profile_cache_eviction_mib=0,
         timeline_public_call_count=None, event_input_ring_mib=0,
         input_ring_contract=None, event_power_warmup_gemm_iterations=0,
         timeline_power_warmup_gemm_iterations=0,
         event_operation_repetitions_per_sample=1,
         persistent_batch_contract=None,
         event_vector_mode=ASCEND_VECTOR_DIRECT_MODE,
         event_graph_repetitions=None, event_graph_buffer_count=0,
         graph_operations=None, graph_storage_groups=None):
    _validate_graph_arguments(
        event_vector_mode, event_graph_repetitions,
        event_graph_buffer_count)
    validate_cache_eviction_mib(
        profile_cache_eviction_mib, minimum_nonzero_mib=1536,
        hardware_id="ascend_910b")
    if event_options is None:
        if event_vector_mode != ASCEND_VECTOR_DIRECT_MODE:
            raise ValueError("Ascend Vector NPUGraph mode requires Event mode")
        _validate_timeline_power_warmup(
            timeline_power_warmup_gemm_iterations)
        if event_operation_repetitions_per_sample != 1:
            raise ValueError(
                "batched Ascend Vector Event timing requires Event mode")
        if persistent_batch_contract is not None:
            raise ValueError(
                "persistent batch contract requires Event mode")
        if event_input_ring_mib != 0:
            raise ValueError("event_input_ring_mib requires Event mode")
        profile_sweep, _ = prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="npu",
            cache_eviction_mib=profile_cache_eviction_mib,
            # Timeline captures include every submitted kernel. A normal Add
            # sweep is indistinguishable from an elementwise-Add target in an
            # Op Summary and previously created millisecond-scale false Vector
            # durations. Fill still writes every byte of the retained 2 GiB
            # BF16 buffer, but has a different runtime symbol.
            sweep_operation="fill")
        if timeline_public_call_count is None:
            if timeline_power_warmup_gemm_iterations:
                raise ValueError(
                    "Timeline power warmup requires Timeline public calls")
            return _warm_run(fn, profile_sweep=profile_sweep)
        power_warmup_operation, _ = _prepare_device_power_warmup(
            timeline_power_warmup_gemm_iterations)
        timeline_warmup_public_call_count = (
            TIMELINE_POWER_WARMUP_PUBLIC_CALL_COUNT
            if power_warmup_operation is not None else 0)
        return _timeline_run(
            fn, profile_sweep, timeline_public_call_count,
            warmup_public_call_count=timeline_warmup_public_call_count)
    if profile_cache_eviction_mib != 0:
        raise ValueError(
            "profile_cache_eviction_mib cannot be mixed with Event mode")
    if timeline_public_call_count is not None:
        raise ValueError(
            "timeline_public_call_count requires Timeline cache eviction")
    if timeline_power_warmup_gemm_iterations:
        raise ValueError("Timeline power warmup requires Timeline mode")
    validate_cache_eviction_mib(
        event_options.cache_eviction_mib, minimum_nonzero_mib=1536,
        hardware_id="ascend_910b")
    _validate_event_input_ring_mib(event_input_ring_mib)
    _validate_event_operation_repetitions(
        event_operation_repetitions_per_sample)
    if event_power_warmup_gemm_iterations not in {0, 1000}:
        raise ValueError(
            "Ascend Vector device power warmup must be 0 or 1000 GEMMs")
    if event_input_ring_mib:
        if event_options.cache_eviction_mib != 0:
            raise ValueError(
                "event_input_ring_mib cannot be mixed with a pre-target "
                "cache eviction sweep")
        if input_ring_contract is None:
            raise ValueError(
                "event_input_ring_mib requires an exact rotating input "
                "operation")
    elif event_power_warmup_gemm_iterations:
        raise ValueError(
            "Ascend Vector device power warmup requires an input ring")
    if (event_operation_repetitions_per_sample != 1
            and event_input_ring_mib == 0):
        raise ValueError(
            "batched Ascend Vector Event timing requires the dense input ring")
    if (event_operation_repetitions_per_sample != 1
            and persistent_batch_contract is None):
        raise ValueError(
            "batched Ascend Vector Event timing requires persistent batch "
            "provenance")
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        if event_input_ring_mib:
            raise ValueError(
                "Ascend Vector NPUGraph mode cannot use the eager input ring")
        if event_power_warmup_gemm_iterations:
            raise ValueError(
                "Ascend Vector NPUGraph mode cannot use eager power warmup")
        if event_operation_repetitions_per_sample != 1:
            raise ValueError(
                "Ascend Vector NPUGraph mode cannot use eager operation batching")
        if persistent_batch_contract is not None:
            raise ValueError(
                "Ascend Vector NPUGraph mode cannot use persistent eager capture")
        if event_options.cache_eviction_mib != 2048:
            raise ValueError(
                "Ascend Vector NPUGraph measurement requires a 2 GiB sweep")
        if (not isinstance(graph_operations, tuple)
                or len(graph_operations) != event_graph_buffer_count):
            raise ValueError(
                "Ascend Vector NPUGraph operations do not match buffer count")
        if (not isinstance(graph_storage_groups, tuple)
                or len(graph_storage_groups) != event_graph_buffer_count):
            raise ValueError(
                "Ascend Vector NPUGraph storage groups do not match buffer count")
    elif graph_operations is not None or graph_storage_groups is not None:
        raise ValueError("graph call sets require Ascend Vector NPUGraph mode")

    def operation():
        return fn()
    cache_eviction_operation = None
    cache_eviction_bytes = 0
    if event_options.cache_eviction_mib:
        cache_eviction_operation, cache_eviction_bytes = (
            prepare_bf16_cache_eviction_sweep(
                torch_module=torch,
                device="npu",
                cache_eviction_mib=event_options.cache_eviction_mib,
            ))
    power_maintenance_operation, power_warmup = _prepare_device_power_warmup(
        event_power_warmup_gemm_iterations)
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        measurement = measure_differential_rotating_npu_graph_replay_events(
            graph_operations,
            operation_storage_groups=graph_storage_groups,
            device_module=torch.npu,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            hardware_l2_bytes=ASCEND_910B_L2_BYTES,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes,
            repetition_counts=event_graph_repetitions,
            repetition_selection_rule=(
                ASCEND_VECTOR_GRAPH_REPETITION_SELECTION_RULE),
        )
    else:
        measurement = measure_current_stream_events(
            operation,
            device_module=torch.npu,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes,
            pre_event_operation=power_maintenance_operation,
            operation_repetitions_per_sample=(
                event_operation_repetitions_per_sample),
        )
    normalized_workload = dict(workload)
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        normalized_workload["cache_eviction_bytes_before_replay"] = (
            cache_eviction_bytes)
        normalized_workload["event_graph"] = {
            "backend": "torch.npu.NPUGraph",
            "lower_repetition_count": event_graph_repetitions[0],
            "higher_repetition_count": event_graph_repetitions[1],
            "preallocated_operation_count": event_graph_buffer_count,
            "input_allocation": (
                "independent_storage_per_captured_public_call"),
            "estimator": "(T_high-T_low)/(N_high-N_low)",
        }
    else:
        normalized_workload["cache_eviction_bytes_before_sample"] = (
            cache_eviction_bytes)
    if input_ring_contract is not None:
        normalized_workload["input_ring"] = input_ring_contract
    if power_warmup is not None:
        normalized_workload["power_warmup"] = power_warmup
    if event_operation_repetitions_per_sample != 1:
        if (input_ring_contract["mode"]
                == "rotating_paired_dense_input_backing_ring"):
            input_selection = (
                "advance_coprime_paired_dense_input_ring_once_per_public_call")
        elif input_ring_contract["mode"] == "rotating_dense_input_backing_ring":
            input_selection = (
                "advance_coprime_dense_input_ring_once_per_public_call")
        elif (input_ring_contract["mode"]
              == "rotating_asymmetric_paired_dense_input_backing_ring"):
            input_selection = (
                "advance_coprime_asymmetric_paired_dense_input_ring_once_per_public_call")
        elif (input_ring_contract["mode"]
              == "rotating_kvcache_input_and_destination_ring"):
            input_selection = (
                "advance_coprime_kvcache_input_and_destination_ring_once_per_public_call")
        else:
            raise AssertionError(
                "validated input-ring mode was not dispatched")
        normalized_workload["event_operation_batch"] = {
            "operation_repetitions_per_sample": (
                event_operation_repetitions_per_sample),
            "input_selection": input_selection,
            "duration_reporting": (
                "aggregate_current_stream_event_duration_divided_by_public_call_count"),
        }
    if persistent_batch_contract is not None:
        normalized_workload["persistent_batch"] = persistent_batch_contract
    normalized_workload["event_measurement_mode"] = event_vector_mode
    write_event_case_record(
        options=event_options,
        hardware_id="ascend_910b",
        family="vector",
        runner_path=RUNNER_PATH,
        timer={
            "api": "torch.npu.Event",
            "implementation": "ACL_Runtime_Event_via_torch_npu",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_applicable",
            "mode": (
                "paired_differential_events_around_npu_graph_replay"
                if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE else (
                    "one_exact_public_op_per_event_interval"
                    if event_operation_repetitions_per_sample == 1 else
                    "fixed_count_distinct_public_ops_per_event_interval_reported_per_call")),
            "cache_condition": (
                "same_stream_2_gib_bf16_sweep_before_every_graph_replay"
                if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE else (
                    input_ring_contract["mode"]
                    if input_ring_contract is not None
                    else (
                        "queued_pre_sample_device_sweep"
                        if cache_eviction_bytes else "ordinary_warm_state"))),
            "power_state_condition": (
                "sustained_bf16_gemm_before_each_event_interval"
                if power_warmup is not None else "ordinary_warm_state"),
        },
        stream_contract={
            "event_stream": "torch.npu.current_stream",
            "all_op_streams_joined": True,
            "evidence": (
                "the exact torch_npu call is submitted through the current "
                "NPU stream and the runner creates no secondary stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": (
                "all_graph_capture_results_retained_per_call"
                if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE
                else "exact_runner_call"),
            "workspace_behavior": "exact_runner_call",
            "evidence": allocation_evidence,
        },
        workload=normalized_workload,
        environment=_environment(),
        measurement=measurement,
    )
    return None
def run_rmsnorm(tok, hidden, dtype="bf16", event_options=None,
                profile_cache_eviction_mib=0, timeline_public_call_count=None,
                event_input_ring_mib=0,
                event_power_warmup_gemm_iterations=0,
                timeline_power_warmup_gemm_iterations=0,
                event_operation_repetitions_per_sample=1,
                persistent_batch_contract=None,
                event_vector_mode=ASCEND_VECTOR_DIRECT_MODE,
                event_graph_repetitions=None, event_graph_buffer_count=0):
    torch.npu.set_device(0)
    dt = DTYPE_MAP[dtype.lower()]
    weight = torch.randn(hidden, dtype=dt, device="npu")
    eps = 1e-6
    x = torch.randn(tok, hidden, dtype=dt, device="npu")
    operation = lambda: torch_npu.npu_rms_norm(x, weight, eps)
    rotating_operation, input_ring_contract = _rotating_input_operation(
        (tok, hidden), dt, event_input_ring_mib,
        lambda ring_x: torch_npu.npu_rms_norm(ring_x, weight, eps),
        minimum_distinct_slots=event_operation_repetitions_per_sample)
    if rotating_operation is not None:
        operation = rotating_operation
    graph_operations = None
    graph_storage_groups = None
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        def allocate_graph_call(_index):
            graph_x = torch.randn(tok, hidden, dtype=dt, device="npu")
            graph_weight = torch.randn(hidden, dtype=dt, device="npu")
            return (
                lambda graph_x=graph_x, graph_weight=graph_weight:
                torch_npu.npu_rms_norm(graph_x, graph_weight, eps),
                (graph_x, graph_weight),
            )
        graph_operations, graph_storage_groups = _build_graph_call_set(
            event_graph_buffer_count, allocate_graph_call)
    return _run(
        operation,
        event_options=event_options,
        workload={"op": "rmsnorm", "tok": tok, "hidden": hidden,
                  "dtype": dtype, "dtype_bytes": dt.itemsize},
        profile_cache_eviction_mib=profile_cache_eviction_mib,
        timeline_public_call_count=timeline_public_call_count,
        event_input_ring_mib=event_input_ring_mib,
        input_ring_contract=input_ring_contract,
        event_power_warmup_gemm_iterations=(
            event_power_warmup_gemm_iterations),
        timeline_power_warmup_gemm_iterations=(
            timeline_power_warmup_gemm_iterations),
        event_operation_repetitions_per_sample=(
            event_operation_repetitions_per_sample),
        persistent_batch_contract=persistent_batch_contract,
        event_vector_mode=event_vector_mode,
        event_graph_repetitions=event_graph_repetitions,
        event_graph_buffer_count=event_graph_buffer_count,
        graph_operations=graph_operations,
        graph_storage_groups=graph_storage_groups,
        allocation_evidence=(
            "the Event bracket preserves the exact npu_rms_norm call; the "
            "activation input is either preallocated or a disjoint dense "
            "backing-ring slot, and the returned tuple is retained through the end "
            "Event"),
    )
def run_add_rmsnorm(tok, hidden, dtype="bf16", event_options=None,
                    profile_cache_eviction_mib=0,
                    timeline_public_call_count=None, event_input_ring_mib=0,
                    event_power_warmup_gemm_iterations=0,
                    timeline_power_warmup_gemm_iterations=0,
                    event_operation_repetitions_per_sample=1,
                    persistent_batch_contract=None,
                    event_vector_mode=ASCEND_VECTOR_DIRECT_MODE,
                    event_graph_repetitions=None,
                    event_graph_buffer_count=0):
    # Serving main path: AscendRMSNorm.forward_oot with residual → npu_add_rms_norm
    # (vllm_ascend/ops/layernorm.py:77). Upstream RMSNorm.forward forks on
    # residual: None→rms_norm (first layer only, 1/(2N)), else→fused_add_rms_norm
    # (the (2N-1)/(2N) main path). This op fuses residual+x then rms_norm.
    torch.npu.set_device(0)
    dt = DTYPE_MAP[dtype.lower()]
    x = torch.randn(tok, hidden, dtype=dt, device="npu")
    residual = torch.randn(tok, hidden, dtype=dt, device="npu")
    weight = torch.randn(hidden, dtype=dt, device="npu")
    eps = 1e-6
    # npu_add_rms_norm returns (out, _, residual); we profile the fused op itself.
    operation = lambda: torch_npu.npu_add_rms_norm(
        x, residual, weight, eps)
    rotating_operation, input_ring_contract = (
        _rotating_paired_input_operation(
            (tok, hidden), dt, event_input_ring_mib,
            lambda ring_x, ring_residual: torch_npu.npu_add_rms_norm(
                ring_x, ring_residual, weight, eps),
            dynamic_input_operands=("x", "residual"),
            static_input_operands=("weight",),
            minimum_distinct_slots=event_operation_repetitions_per_sample))
    if rotating_operation is not None:
        operation = rotating_operation
    graph_operations = None
    graph_storage_groups = None
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        def allocate_graph_call(_index):
            graph_x = torch.randn(tok, hidden, dtype=dt, device="npu")
            graph_residual = torch.randn(
                tok, hidden, dtype=dt, device="npu")
            graph_weight = torch.randn(hidden, dtype=dt, device="npu")
            return (
                lambda graph_x=graph_x, graph_residual=graph_residual,
                graph_weight=graph_weight: torch_npu.npu_add_rms_norm(
                    graph_x, graph_residual, graph_weight, eps),
                (graph_x, graph_residual, graph_weight),
            )
        graph_operations, graph_storage_groups = _build_graph_call_set(
            event_graph_buffer_count, allocate_graph_call)
    return _run(
        operation,
        event_options=event_options,
        workload={"op": "add_rmsnorm", "tok": tok, "hidden": hidden,
                  "dtype": dtype, "dtype_bytes": dt.itemsize},
        profile_cache_eviction_mib=profile_cache_eviction_mib,
        timeline_public_call_count=timeline_public_call_count,
        event_input_ring_mib=event_input_ring_mib,
        input_ring_contract=input_ring_contract,
        event_power_warmup_gemm_iterations=(
            event_power_warmup_gemm_iterations),
        timeline_power_warmup_gemm_iterations=(
            timeline_power_warmup_gemm_iterations),
        event_operation_repetitions_per_sample=(
            event_operation_repetitions_per_sample),
        persistent_batch_contract=persistent_batch_contract,
        event_vector_mode=event_vector_mode,
        event_graph_repetitions=event_graph_repetitions,
        event_graph_buffer_count=event_graph_buffer_count,
        graph_operations=graph_operations,
        graph_storage_groups=graph_storage_groups,
        allocation_evidence=(
            "the Event bracket preserves the exact npu_add_rms_norm call; "
            "x and residual either use preallocated tensors or paired dense "
            "backing-ring slots, weight stays static, and only the latest "
            "returned tuple is retained instead of materializing 512 results"),
    )
def run_rope(tok, num_heads, num_kv_heads, head_dim, dtype="bf16",
             event_options=None, profile_cache_eviction_mib=0,
             timeline_public_call_count=None, event_input_ring_mib=0,
             event_power_warmup_gemm_iterations=0,
             timeline_power_warmup_gemm_iterations=0,
             event_operation_repetitions_per_sample=1,
             persistent_batch_contract=None,
             event_vector_mode=ASCEND_VECTOR_DIRECT_MODE,
             event_graph_repetitions=None, event_graph_buffer_count=0):
    torch.npu.set_device(0)
    dt = DTYPE_MAP[dtype.lower()]
    # Llama uses this standalone in-place ATB RoPE path; Qwen3/DeepSeek fuse it.
    # The cache is [cos_half|sin_half] with shape (tok, head_dim), matching vLLM.
    rotary_dim = head_dim
    q_flat = torch.randn(
        tok, num_heads, head_dim, dtype=dt, device="npu").reshape(tok, -1)
    k_flat = torch.randn(
        tok, num_kv_heads, head_dim, dtype=dt, device="npu").reshape(tok, -1)
    positions = torch.arange(tok, dtype=torch.int32, device="npu")
    theta = 1.0 / (10000 ** (
        torch.arange(0, head_dim, 2, device="npu", dtype=torch.float32)
        / head_dim))
    freqs = (torch.arange(tok, device="npu", dtype=torch.float32)
             .unsqueeze(1) * theta).float()
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dt)
    operation = lambda: torch_npu._npu_rotary_embedding(
        positions, q_flat, k_flat, rotary_dim, cos_sin_cache, True)
    rotating_operation, input_ring_contract = (
        _rotating_asymmetric_paired_input_operation(
            tuple(q_flat.shape), tuple(k_flat.shape), dt,
            event_input_ring_mib,
            lambda ring_q, ring_k: torch_npu._npu_rotary_embedding(
                positions, ring_q, ring_k, rotary_dim, cos_sin_cache, True),
            dynamic_input_operands=("q_flat", "k_flat"),
            static_input_operands=("positions", "cos_sin_cache"),
            minimum_distinct_slots=event_operation_repetitions_per_sample))
    if rotating_operation is not None:
        operation = rotating_operation
    graph_operations = None
    graph_storage_groups = None
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        def allocate_graph_call(_index):
            graph_q = torch.randn(
                tok, num_heads * head_dim, dtype=dt, device="npu")
            graph_k = torch.randn(
                tok, num_kv_heads * head_dim, dtype=dt, device="npu")
            graph_positions = positions.clone()
            graph_cos_sin_cache = cos_sin_cache.clone()
            return (
                lambda graph_positions=graph_positions, graph_q=graph_q,
                graph_k=graph_k,
                graph_cos_sin_cache=graph_cos_sin_cache:
                torch_npu._npu_rotary_embedding(
                    graph_positions, graph_q, graph_k, rotary_dim,
                    graph_cos_sin_cache, True),
                (graph_positions, graph_q, graph_k, graph_cos_sin_cache),
            )
        graph_operations, graph_storage_groups = _build_graph_call_set(
            event_graph_buffer_count, allocate_graph_call)
    return _run(
        operation,
        event_options=event_options,
        workload={
            "op": "rope", "tok": tok, "num_heads": num_heads,
            "num_kv_heads": num_kv_heads, "head_dim": head_dim,
            "dtype": dtype, "dtype_bytes": dt.itemsize,
        },
        profile_cache_eviction_mib=profile_cache_eviction_mib,
        timeline_public_call_count=timeline_public_call_count,
        event_input_ring_mib=event_input_ring_mib,
        input_ring_contract=input_ring_contract,
        event_power_warmup_gemm_iterations=(
            event_power_warmup_gemm_iterations),
        timeline_power_warmup_gemm_iterations=(
            timeline_power_warmup_gemm_iterations),
        event_operation_repetitions_per_sample=(
            event_operation_repetitions_per_sample),
        persistent_batch_contract=persistent_batch_contract,
        event_vector_mode=event_vector_mode,
        event_graph_repetitions=event_graph_repetitions,
        event_graph_buffer_count=event_graph_buffer_count,
        graph_operations=graph_operations,
        graph_storage_groups=graph_storage_groups,
        allocation_evidence=(
            "_npu_rotary_embedding updates preallocated flattened q and k "
            "views in place; the dynamic q/k inputs are either fixed "
            "preallocated tensors or disjoint ring slots, while positions and "
            "cos/sin cache are prepared before timing"),
    )
def run_silu(tok, intermediate, dtype="bf16", event_options=None,
             profile_cache_eviction_mib=0, timeline_public_call_count=None,
             event_input_ring_mib=0,
             event_power_warmup_gemm_iterations=0,
             timeline_power_warmup_gemm_iterations=0,
             event_operation_repetitions_per_sample=1,
             persistent_batch_contract=None,
             event_vector_mode=ASCEND_VECTOR_DIRECT_MODE,
             event_graph_repetitions=None, event_graph_buffer_count=0):
    # vLLM Ascend routes concat[gate|up] to npu_swiglu for SwiGLU.
    torch.npu.set_device(0)
    dt = DTYPE_MAP[dtype.lower()]
    input_tensor = torch.randn(
        tok, 2 * intermediate, dtype=dt, device="npu")
    operation = lambda: torch_npu.npu_swiglu(input_tensor)
    rotating_operation, input_ring_contract = _rotating_input_operation(
        (tok, 2 * intermediate), dt, event_input_ring_mib,
        lambda ring_input: torch_npu.npu_swiglu(ring_input),
        minimum_distinct_slots=event_operation_repetitions_per_sample)
    if rotating_operation is not None:
        operation = rotating_operation
    graph_operations = None
    graph_storage_groups = None
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        def allocate_graph_call(_index):
            graph_input = torch.randn(
                tok, 2 * intermediate, dtype=dt, device="npu")
            return (
                lambda graph_input=graph_input:
                torch_npu.npu_swiglu(graph_input),
                (graph_input,),
            )
        graph_operations, graph_storage_groups = _build_graph_call_set(
            event_graph_buffer_count, allocate_graph_call)
    return _run(
        operation,
        event_options=event_options,
        workload={"op": "silu", "tok": tok,
                  "intermediate": intermediate, "dtype": dtype,
                  "dtype_bytes": dt.itemsize},
        profile_cache_eviction_mib=profile_cache_eviction_mib,
        timeline_public_call_count=timeline_public_call_count,
        event_input_ring_mib=event_input_ring_mib,
        input_ring_contract=input_ring_contract,
        event_power_warmup_gemm_iterations=(
            event_power_warmup_gemm_iterations),
        timeline_power_warmup_gemm_iterations=(
            timeline_power_warmup_gemm_iterations),
        event_operation_repetitions_per_sample=(
            event_operation_repetitions_per_sample),
        persistent_batch_contract=persistent_batch_contract,
        event_vector_mode=event_vector_mode,
        event_graph_repetitions=event_graph_repetitions,
        event_graph_buffer_count=event_graph_buffer_count,
        graph_operations=graph_operations,
        graph_storage_groups=graph_storage_groups,
        allocation_evidence=(
            "the Event bracket preserves the exact npu_swiglu call; its "
            "input is either preallocated or a disjoint dense backing-ring slot, "
            "and its returned tensor is retained through the end Event"),
    )
def run_elementwise(tok, hidden, dtype="bf16", op2="add",
                    event_options=None, profile_cache_eviction_mib=0,
                    timeline_public_call_count=None, event_input_ring_mib=0,
                    event_power_warmup_gemm_iterations=0,
                    timeline_power_warmup_gemm_iterations=0,
                    event_operation_repetitions_per_sample=1,
                    persistent_batch_contract=None,
                    event_vector_mode=ASCEND_VECTOR_DIRECT_MODE,
                    event_graph_repetitions=None,
                    event_graph_buffer_count=0):
    torch.npu.set_device(0)
    dt = DTYPE_MAP[dtype.lower()]
    left = torch.randn(tok, hidden, dtype=dt, device="npu")
    right = torch.randn(tok, hidden, dtype=dt, device="npu")
    if op2 == "add":
        operation = lambda: left + right
    elif op2 == "mul":
        operation = lambda: left * right
    else:
        raise ValueError(f"unknown elementwise operation: {op2!r}")
    if op2 == "add":
        paired_operation = lambda ring_left, ring_right: ring_left + ring_right
    elif op2 == "mul":
        paired_operation = lambda ring_left, ring_right: ring_left * ring_right
    else:
        raise AssertionError(f"validated unknown elementwise operation: {op2!r}")
    rotating_operation, input_ring_contract = (
        _rotating_paired_input_operation(
            (tok, hidden), dt, event_input_ring_mib, paired_operation,
            dynamic_input_operands=("left", "right"),
            static_input_operands=(),
            minimum_distinct_slots=event_operation_repetitions_per_sample))
    if rotating_operation is not None:
        operation = rotating_operation
    graph_operations = None
    graph_storage_groups = None
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        def allocate_graph_call(_index):
            graph_left = torch.randn(tok, hidden, dtype=dt, device="npu")
            graph_right = torch.randn(tok, hidden, dtype=dt, device="npu")
            if op2 == "add":
                graph_operation = (
                    lambda graph_left=graph_left, graph_right=graph_right:
                    graph_left + graph_right)
            elif op2 == "mul":
                graph_operation = (
                    lambda graph_left=graph_left, graph_right=graph_right:
                    graph_left * graph_right)
            else:
                raise AssertionError(
                    f"validated unknown elementwise operation: {op2!r}")
            return graph_operation, (graph_left, graph_right)
        graph_operations, graph_storage_groups = _build_graph_call_set(
            event_graph_buffer_count, allocate_graph_call)
    return _run(
        operation,
        event_options=event_options,
        workload={"op": "elementwise", "op2": op2, "tok": tok,
                  "hidden": hidden, "dtype": dtype,
                  "dtype_bytes": dt.itemsize},
        profile_cache_eviction_mib=profile_cache_eviction_mib,
        timeline_public_call_count=timeline_public_call_count,
        event_input_ring_mib=event_input_ring_mib,
        input_ring_contract=input_ring_contract,
        event_power_warmup_gemm_iterations=(
            event_power_warmup_gemm_iterations),
        timeline_power_warmup_gemm_iterations=(
            timeline_power_warmup_gemm_iterations),
        event_operation_repetitions_per_sample=(
            event_operation_repetitions_per_sample),
        persistent_batch_contract=persistent_batch_contract,
        event_vector_mode=event_vector_mode,
        event_graph_repetitions=event_graph_repetitions,
        event_graph_buffer_count=event_graph_buffer_count,
        graph_operations=graph_operations,
        graph_storage_groups=graph_storage_groups,
        allocation_evidence=(
            "the Event bracket preserves the exact tensor add/multiply call; "
            "both operands either use preallocated tensors or paired dense "
            "backing-ring slots, and only the latest returned tensor is "
            "retained instead of materializing 512 results"),
    )
def run_kvcache(tok, num_kv_heads, head_dim, batch=1, dtype="bf16",
                event_options=None, profile_cache_eviction_mib=0,
                timeline_public_call_count=None, event_input_ring_mib=0,
                event_power_warmup_gemm_iterations=0,
                timeline_power_warmup_gemm_iterations=0,
                event_operation_repetitions_per_sample=1,
                persistent_batch_contract=None,
                event_vector_mode=ASCEND_VECTOR_DIRECT_MODE,
                event_graph_repetitions=None, event_graph_buffer_count=0):
    """ReshapeAndCache: write (tok, num_kv_heads, head_dim) of new KV into a paged
    KV-cache pool via slot_indices (mirrors vllm-ascend serving)."""
    if min(tok, num_kv_heads, head_dim, batch) <= 0:
        raise ValueError("KV-cache workload dimensions must be positive")
    torch.npu.set_device(0)
    dt = DTYPE_MAP[dtype.lower()]
    # new KV: (num_tokens, num_kv_heads, head_dim) — 3D, as vllm-ascend passes it.
    key = torch.randn(
        tok, num_kv_heads, head_dim, dtype=dt, device="npu")
    value = torch.randn(
        tok, num_kv_heads, head_dim, dtype=dt, device="npu")
    # paged pool: (num_blocks, block_size, num_kv_heads, head_dim).
    block_size = 128
    num_blocks = (tok + block_size - 1) // block_size
    key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                            dtype=dt, device="npu")
    value_cache = torch.zeros_like(key_cache)
    # slot_indices: one int32 per token → flat slot in the pool.
    slot_indices = torch.arange(tok, dtype=torch.int32, device="npu")
    operation = lambda: torch_npu._npu_reshape_and_cache(
        key=key, value=value, key_cache=key_cache,
        value_cache=value_cache, slot_indices=slot_indices)
    rotating_operation, input_ring_contract = _rotating_kvcache_operation(
        tuple(key.shape), tuple(key_cache.shape), dt, event_input_ring_mib,
        lambda ring_key, ring_value, ring_key_cache, ring_value_cache:
        torch_npu._npu_reshape_and_cache(
            key=ring_key, value=ring_value, key_cache=ring_key_cache,
            value_cache=ring_value_cache, slot_indices=slot_indices),
        minimum_distinct_slots=event_operation_repetitions_per_sample)
    if rotating_operation is not None:
        operation = rotating_operation
    graph_operations = None
    graph_storage_groups = None
    if event_vector_mode == ASCEND_VECTOR_NPU_GRAPH_MODE:
        def allocate_graph_call(_index):
            graph_key = torch.randn(
                tok, num_kv_heads, head_dim, dtype=dt, device="npu")
            graph_value = torch.randn(
                tok, num_kv_heads, head_dim, dtype=dt, device="npu")
            graph_key_cache = torch.zeros(
                num_blocks, block_size, num_kv_heads, head_dim,
                dtype=dt, device="npu")
            graph_value_cache = torch.zeros(
                num_blocks, block_size, num_kv_heads, head_dim,
                dtype=dt, device="npu")
            graph_slot_indices = torch.arange(
                tok, dtype=torch.int32, device="npu")
            return (
                lambda graph_key=graph_key, graph_value=graph_value,
                graph_key_cache=graph_key_cache,
                graph_value_cache=graph_value_cache,
                graph_slot_indices=graph_slot_indices:
                torch_npu._npu_reshape_and_cache(
                    key=graph_key, value=graph_value,
                    key_cache=graph_key_cache,
                    value_cache=graph_value_cache,
                    slot_indices=graph_slot_indices),
                (graph_key, graph_value, graph_key_cache,
                 graph_value_cache, graph_slot_indices),
            )
        graph_operations, graph_storage_groups = _build_graph_call_set(
            event_graph_buffer_count, allocate_graph_call)
    return _run(
        operation,
        event_options=event_options,
        workload={
            "op": "kvcache", "tok": tok,
            "num_kv_heads": num_kv_heads, "head_dim": head_dim,
            "batch": batch, "dtype": dtype, "dtype_bytes": dt.itemsize,
            "scale_contract": (
                "not_applicable_bf16_api_has_no_scale_operands"),
            "shared_cache_write_contract": (
                "contiguous_nonoverlapping_ranges_in_shared_cache_backing"),
        },
        profile_cache_eviction_mib=profile_cache_eviction_mib,
        timeline_public_call_count=timeline_public_call_count,
        event_input_ring_mib=event_input_ring_mib,
        input_ring_contract=input_ring_contract,
        event_power_warmup_gemm_iterations=(
            event_power_warmup_gemm_iterations),
        timeline_power_warmup_gemm_iterations=(
            timeline_power_warmup_gemm_iterations),
        event_operation_repetitions_per_sample=(
            event_operation_repetitions_per_sample),
        persistent_batch_contract=persistent_batch_contract,
        event_vector_mode=event_vector_mode,
        event_graph_repetitions=event_graph_repetitions,
        event_graph_buffer_count=event_graph_buffer_count,
        graph_operations=graph_operations,
        graph_storage_groups=graph_storage_groups,
        allocation_evidence=(
            "_npu_reshape_and_cache writes into preallocated key/value cache "
            "tensors; persistent Event mode rotates both new-KV inputs and "
            "non-overlapping cache destinations without allocating outputs "
            "inside the timed call"),
    )


_EVENT_BATCH_MANIFEST_KEYS = frozenset((
    "schema_version", "hardware_id", "family", "baseline_commit",
    "source_snapshot_sha256", "timer_sanity_sha256", "warmup_iterations",
    "sample_count", "event_input_ring_mib",
    "event_power_warmup_gemm_iterations",
    "event_operation_repetitions_per_sample", "cases",
))
_EVENT_BATCH_CASE_KEYS = frozenset((
    "case_id", "canonical_attributes_sha256", "runner_arguments",
))
_EVENT_BATCH_RUNNER_ARGUMENTS = frozenset((
    "--op", "--tok", "--hidden", "--num_heads", "--head_dim",
    "--intermediate", "--num_kv_heads", "--batch", "--dtype", "--op2",
    "--event-vector-mode",
))


def _positive_runner_argument(arguments, name):
    value = arguments[name]
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"persistent Vector runner argument {name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(
            f"persistent Vector runner argument {name} must be positive")
    return parsed


def _load_event_batch_manifest(path):
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or set(manifest) != _EVENT_BATCH_MANIFEST_KEYS:
        raise ValueError("invalid Ascend persistent Vector Event manifest")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported Ascend persistent Vector Event manifest schema")
    if manifest["hardware_id"] != "ascend_910b":
        raise ValueError("persistent Vector Event manifest hardware must be ascend_910b")
    if manifest["family"] != "vector":
        raise ValueError("persistent Vector Event manifest family must be vector")
    for field, length in (
            ("baseline_commit", 40), ("source_snapshot_sha256", 64),
            ("timer_sanity_sha256", 64)):
        require_lower_hex(manifest[field], length, field)
    for field in ("warmup_iterations", "sample_count"):
        if (isinstance(manifest[field], bool)
                or not isinstance(manifest[field], int)
                or manifest[field] <= 0):
            raise ValueError(
                f"persistent Vector Event manifest {field} must be positive")
    _validate_event_input_ring_mib(manifest["event_input_ring_mib"])
    if manifest["event_input_ring_mib"] == 0:
        raise ValueError("persistent Vector Event manifest requires input ring")
    if manifest["event_power_warmup_gemm_iterations"] != 1000:
        raise ValueError(
            "persistent Vector Event manifest requires 1000 GEMM power warmup")
    _validate_event_operation_repetitions(
        manifest["event_operation_repetitions_per_sample"])
    if manifest["event_operation_repetitions_per_sample"] not in {
            32, 512, 2048, 8192}:
        raise ValueError(
            "persistent Vector Event manifest requires 32, 512, 2048, or "
            "8192 calls per sample")
    cases = manifest["cases"]
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, dict)
                   or set(case) != _EVENT_BATCH_CASE_KEYS for case in cases)):
        raise ValueError("persistent Vector Event manifest cases are invalid")
    case_ids = []
    for case in cases:
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("persistent Vector Event case ID is invalid")
        case_ids.append(case_id)
        require_lower_hex(
            case["canonical_attributes_sha256"], 64,
            "canonical_attributes_sha256")
        arguments = case["runner_arguments"]
        if (not isinstance(arguments, dict)
                or set(arguments) != _EVENT_BATCH_RUNNER_ARGUMENTS):
            raise ValueError(
                f"persistent Vector Event runner arguments are invalid: {case_id}")
        if arguments["--op"] not in {
                "rmsnorm", "add_rmsnorm", "rope", "silu", "elementwise",
                "kvcache"}:
            raise ValueError(
                f"persistent Vector Event op is unsupported: {case_id}")
        if arguments["--dtype"] not in DTYPE_MAP:
            raise ValueError(
                f"persistent Vector Event dtype is unsupported: {case_id}")
        if arguments["--op2"] not in {"add", "mul"}:
            raise ValueError(
                f"persistent Vector Event op2 is unsupported: {case_id}")
        if arguments["--event-vector-mode"] != ASCEND_VECTOR_DIRECT_MODE:
            raise ValueError(
                f"persistent Vector Event mode is invalid: {case_id}")
        for name in (
                "--tok", "--hidden", "--num_heads", "--head_dim",
                "--intermediate", "--num_kv_heads", "--batch"):
            _positive_runner_argument(arguments, name)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("persistent Vector Event manifest repeats case IDs")
    if (any(case["runner_arguments"]["--op"] in {
            "add_rmsnorm", "elementwise", "kvcache"} for case in cases)
            and manifest["event_input_ring_mib"] < 2048):
        raise ValueError(
            "persistent paired-input Vector Event manifest requires at least "
            "a 2048 MiB input ring")
    return manifest


def _run_event_batch(manifest_path, output_dir):
    manifest = _load_event_batch_manifest(manifest_path)
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(ROOT / "experiment_logs/kernel_golden")
    except ValueError as error:
        raise ValueError(
            "persistent Vector Event output must stay under "
            f"experiment_logs/kernel_golden: {output_dir}") from error
    manifest_sha256 = file_sha256(manifest_path)
    torch.npu.set_device(0)
    torch.manual_seed(20260826)
    torch.npu.manual_seed(20260826)
    results = []
    cases = manifest["cases"]
    for ordinal, case in enumerate(cases):
        case_id = case["case_id"]
        arguments = case["runner_arguments"]
        event_options = EventRunOptions(
            output=output_dir / "cases" / f"{case_id}.json",
            case_id=case_id,
            baseline_commit=manifest["baseline_commit"],
            source_snapshot_sha256=manifest["source_snapshot_sha256"],
            timer_sanity_sha256=manifest["timer_sanity_sha256"],
            canonical_attributes_sha256=case["canonical_attributes_sha256"],
            warmup_iterations=manifest["warmup_iterations"],
            sample_count=manifest["sample_count"],
            cache_eviction_mib=0,
        )
        batch_contract = {
            "mode": "one_process_serial_vector_event_capture",
            "manifest_sha256": manifest_sha256,
            "case_ordinal": ordinal,
            "case_count": len(cases),
        }
        op = arguments["--op"]
        if op == "rmsnorm":
            run_rmsnorm(
                _positive_runner_argument(arguments, "--tok"),
                _positive_runner_argument(arguments, "--hidden"),
                arguments["--dtype"], event_options=event_options,
                event_input_ring_mib=manifest["event_input_ring_mib"],
                event_power_warmup_gemm_iterations=(
                    manifest["event_power_warmup_gemm_iterations"]),
                event_operation_repetitions_per_sample=(
                    manifest["event_operation_repetitions_per_sample"]),
                persistent_batch_contract=batch_contract)
        elif op == "silu":
            run_silu(
                _positive_runner_argument(arguments, "--tok"),
                _positive_runner_argument(arguments, "--intermediate"),
                arguments["--dtype"], event_options=event_options,
                event_input_ring_mib=manifest["event_input_ring_mib"],
                event_power_warmup_gemm_iterations=(
                    manifest["event_power_warmup_gemm_iterations"]),
                event_operation_repetitions_per_sample=(
                    manifest["event_operation_repetitions_per_sample"]),
                persistent_batch_contract=batch_contract)
        elif op == "add_rmsnorm":
            run_add_rmsnorm(
                _positive_runner_argument(arguments, "--tok"),
                _positive_runner_argument(arguments, "--hidden"),
                arguments["--dtype"], event_options=event_options,
                event_input_ring_mib=manifest["event_input_ring_mib"],
                event_power_warmup_gemm_iterations=(
                    manifest["event_power_warmup_gemm_iterations"]),
                event_operation_repetitions_per_sample=(
                    manifest["event_operation_repetitions_per_sample"]),
                persistent_batch_contract=batch_contract)
        elif op == "rope":
            run_rope(
                _positive_runner_argument(arguments, "--tok"),
                _positive_runner_argument(arguments, "--num_heads"),
                _positive_runner_argument(arguments, "--num_kv_heads"),
                _positive_runner_argument(arguments, "--head_dim"),
                arguments["--dtype"], event_options=event_options,
                event_input_ring_mib=manifest["event_input_ring_mib"],
                event_power_warmup_gemm_iterations=(
                    manifest["event_power_warmup_gemm_iterations"]),
                event_operation_repetitions_per_sample=(
                    manifest["event_operation_repetitions_per_sample"]),
                persistent_batch_contract=batch_contract)
        elif op == "elementwise":
            run_elementwise(
                _positive_runner_argument(arguments, "--tok"),
                _positive_runner_argument(arguments, "--hidden"),
                arguments["--dtype"], arguments["--op2"],
                event_options=event_options,
                event_input_ring_mib=manifest["event_input_ring_mib"],
                event_power_warmup_gemm_iterations=(
                    manifest["event_power_warmup_gemm_iterations"]),
                event_operation_repetitions_per_sample=(
                    manifest["event_operation_repetitions_per_sample"]),
                persistent_batch_contract=batch_contract)
        elif op == "kvcache":
            run_kvcache(
                _positive_runner_argument(arguments, "--tok"),
                _positive_runner_argument(arguments, "--num_kv_heads"),
                _positive_runner_argument(arguments, "--head_dim"),
                _positive_runner_argument(arguments, "--batch"),
                arguments["--dtype"], event_options=event_options,
                event_input_ring_mib=manifest["event_input_ring_mib"],
                event_power_warmup_gemm_iterations=(
                    manifest["event_power_warmup_gemm_iterations"]),
                event_operation_repetitions_per_sample=(
                    manifest["event_operation_repetitions_per_sample"]),
                persistent_batch_contract=batch_contract)
        else:
            raise AssertionError(f"validated unsupported persistent op: {op}")
        torch.npu.synchronize()
        torch.npu.empty_cache()
        results.append({"case_id": case_id, "status": "measured"})
        print(json.dumps({
            "case_id": case_id, "completed": len(results),
            "expected": len(cases), "status": "measured",
        }, sort_keys=True), flush=True)
    summary = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "family": "vector",
        "baseline_commit": manifest["baseline_commit"],
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "timer_sanity_sha256": manifest["timer_sanity_sha256"],
        "event_input_ring_mib": manifest["event_input_ring_mib"],
        "event_power_warmup_gemm_iterations": (
            manifest["event_power_warmup_gemm_iterations"]),
        "event_operation_repetitions_per_sample": (
            manifest["event_operation_repetitions_per_sample"]),
        "manifest_sha256": manifest_sha256,
        "selected_case_count": len(cases),
        "measured_case_count": len(results),
        "results": results,
    }
    write_kernel_golden_json(output_dir / "capture_summary.json", summary)
    print(json.dumps({
        "measured": len(results),
        "output": str((output_dir / "capture_summary.json").resolve()),
    }, sort_keys=True))


def _try_run_event_batch():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--event-batch-manifest", type=Path)
    parser.add_argument("--event-batch-output-dir", type=Path)
    args, remaining = parser.parse_known_args()
    if args.event_batch_manifest is None:
        if args.event_batch_output_dir is not None:
            parser.error("event-batch-output-dir requires event-batch-manifest")
        return False
    if args.event_batch_output_dir is None or remaining:
        parser.error(
            "event batch requires exactly event-batch-manifest and "
            "event-batch-output-dir")
    _run_event_batch(args.event_batch_manifest, args.event_batch_output_dir)
    return True


def main():
    if _try_run_event_batch():
        return 0
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True,
                   choices=["rmsnorm", "add_rmsnorm", "rope", "silu", "elementwise", "kvcache"])
    p.add_argument("--tok", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--num_heads", type=int, required=True)
    p.add_argument("--head_dim", type=int, required=True)
    p.add_argument("--intermediate", type=int, required=True)
    p.add_argument("--num_kv_heads", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--dtype", required=True, choices=sorted(DTYPE_MAP))
    p.add_argument("--op2", required=True, choices=["add", "mul"])
    p.add_argument("--profile-cache-eviction-mib", type=int, default=0)
    p.add_argument("--event-input-ring-mib", type=int, default=0)
    p.add_argument("--event-power-warmup-gemm-iterations", type=int, default=0)
    p.add_argument("--timeline-power-warmup-gemm-iterations", type=int, default=0)
    p.add_argument(
        "--event-operation-repetitions-per-sample", type=int, default=1)
    p.add_argument("--timeline-public-call-count", type=int)
    p.add_argument(
        "--event-vector-mode",
        choices=(ASCEND_VECTOR_DIRECT_MODE, ASCEND_VECTOR_NPU_GRAPH_MODE),
        default=ASCEND_VECTOR_DIRECT_MODE,
    )
    p.add_argument("--event-graph-repetitions", type=int, nargs=2)
    p.add_argument("--event-graph-buffer-count", type=int, default=0)
    add_event_cli_arguments(p)
    a = p.parse_args()
    event_options = parse_event_run_options(a)
    torch.npu.set_device(0)
    if event_options is not None:
        torch.manual_seed(20260826)
        torch.npu.manual_seed(20260826)
    if a.event_input_ring_mib and a.op not in (
            "rmsnorm", "add_rmsnorm", "rope", "silu", "elementwise",
            "kvcache"):
        raise ValueError(
            "event_input_ring_mib is currently implemented only for "
            "rmsnorm, add_rmsnorm, rope, silu, elementwise, and kvcache")
    if a.event_power_warmup_gemm_iterations and not a.event_input_ring_mib:
        raise ValueError("event power warmup requires event input ring")
    _validate_timeline_power_warmup(a.timeline_power_warmup_gemm_iterations)
    _validate_event_operation_repetitions(
        a.event_operation_repetitions_per_sample)
    event_graph_repetitions = (
        tuple(a.event_graph_repetitions)
        if a.event_graph_repetitions is not None else None)
    _validate_graph_arguments(
        a.event_vector_mode, event_graph_repetitions,
        a.event_graph_buffer_count)
    if (a.event_operation_repetitions_per_sample != 1
            and not a.event_input_ring_mib):
        raise ValueError(
            "batched Event timing requires event input ring")
    if a.op == "rmsnorm":
        run_rmsnorm(
            a.tok, a.hidden, a.dtype, event_options=event_options,
            profile_cache_eviction_mib=a.profile_cache_eviction_mib,
            timeline_public_call_count=a.timeline_public_call_count,
            event_input_ring_mib=a.event_input_ring_mib,
            event_power_warmup_gemm_iterations=(
                a.event_power_warmup_gemm_iterations),
            timeline_power_warmup_gemm_iterations=(
                a.timeline_power_warmup_gemm_iterations),
            event_operation_repetitions_per_sample=(
                a.event_operation_repetitions_per_sample),
            event_vector_mode=a.event_vector_mode,
            event_graph_repetitions=event_graph_repetitions,
            event_graph_buffer_count=a.event_graph_buffer_count)
    elif a.op == "add_rmsnorm":
        run_add_rmsnorm(
            a.tok, a.hidden, a.dtype, event_options=event_options,
            profile_cache_eviction_mib=a.profile_cache_eviction_mib,
            timeline_public_call_count=a.timeline_public_call_count,
            event_input_ring_mib=a.event_input_ring_mib,
            event_power_warmup_gemm_iterations=(
                a.event_power_warmup_gemm_iterations),
            timeline_power_warmup_gemm_iterations=(
                a.timeline_power_warmup_gemm_iterations),
            event_operation_repetitions_per_sample=(
                a.event_operation_repetitions_per_sample),
            event_vector_mode=a.event_vector_mode,
            event_graph_repetitions=event_graph_repetitions,
            event_graph_buffer_count=a.event_graph_buffer_count)
    elif a.op == "rope":
        run_rope(
            a.tok, a.num_heads, a.num_kv_heads, a.head_dim, a.dtype,
            event_options=event_options,
            profile_cache_eviction_mib=a.profile_cache_eviction_mib,
            timeline_public_call_count=a.timeline_public_call_count,
            event_input_ring_mib=a.event_input_ring_mib,
            event_power_warmup_gemm_iterations=(
                a.event_power_warmup_gemm_iterations),
            timeline_power_warmup_gemm_iterations=(
                a.timeline_power_warmup_gemm_iterations),
            event_operation_repetitions_per_sample=(
                a.event_operation_repetitions_per_sample),
            event_vector_mode=a.event_vector_mode,
            event_graph_repetitions=event_graph_repetitions,
            event_graph_buffer_count=a.event_graph_buffer_count)
    elif a.op == "silu":
        run_silu(
            a.tok, a.intermediate, a.dtype, event_options=event_options,
            profile_cache_eviction_mib=a.profile_cache_eviction_mib,
            timeline_public_call_count=a.timeline_public_call_count,
            event_input_ring_mib=a.event_input_ring_mib,
            event_power_warmup_gemm_iterations=(
                a.event_power_warmup_gemm_iterations),
            timeline_power_warmup_gemm_iterations=(
                a.timeline_power_warmup_gemm_iterations),
            event_operation_repetitions_per_sample=(
                a.event_operation_repetitions_per_sample),
            event_vector_mode=a.event_vector_mode,
            event_graph_repetitions=event_graph_repetitions,
            event_graph_buffer_count=a.event_graph_buffer_count)
    elif a.op == "elementwise":
        run_elementwise(
            a.tok, a.hidden, a.dtype, a.op2,
            event_options=event_options,
            profile_cache_eviction_mib=a.profile_cache_eviction_mib,
            timeline_public_call_count=a.timeline_public_call_count,
            event_input_ring_mib=a.event_input_ring_mib,
            event_power_warmup_gemm_iterations=(
                a.event_power_warmup_gemm_iterations),
            timeline_power_warmup_gemm_iterations=(
                a.timeline_power_warmup_gemm_iterations),
            event_operation_repetitions_per_sample=(
                a.event_operation_repetitions_per_sample),
            event_vector_mode=a.event_vector_mode,
            event_graph_repetitions=event_graph_repetitions,
            event_graph_buffer_count=a.event_graph_buffer_count)
    elif a.op == "kvcache":
        run_kvcache(
            a.tok, a.num_kv_heads, a.head_dim, a.batch, a.dtype,
            event_options=event_options,
            profile_cache_eviction_mib=a.profile_cache_eviction_mib,
            timeline_public_call_count=a.timeline_public_call_count,
            event_input_ring_mib=a.event_input_ring_mib,
            event_power_warmup_gemm_iterations=(
                a.event_power_warmup_gemm_iterations),
            timeline_power_warmup_gemm_iterations=(
                a.timeline_power_warmup_gemm_iterations),
            event_operation_repetitions_per_sample=(
                a.event_operation_repetitions_per_sample),
            event_vector_mode=a.event_vector_mode,
            event_graph_repetitions=event_graph_repetitions,
            event_graph_buffer_count=a.event_graph_buffer_count)
    else:
        raise AssertionError(f"argparse accepted an unknown vector op: {a.op!r}")
    print("VEC done")
    return 0
