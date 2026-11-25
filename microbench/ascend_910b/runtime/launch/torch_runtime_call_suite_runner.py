#!/usr/bin/env python3
"""Measure one generic eager runtime-call service parameter on Ascend."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import traceback

import torch
import torch_npu
from vllm.triton_utils import tl, triton
import vllm_ascend.ops  # noqa: F401  # Register vLLM-Ascend custom ops.
import vllm_ascend.patch.worker.patch_unquantized_gemm  # noqa: F401
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.ops import rotary_embedding


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
GENERIC_OPERATION_KINDS = (
    "add_out",
    "matmul_out",
    "index_select_out",
    "reduce_sum_out",
)
ALLOCATING_OPERATION_KINDS = (
    "add_alloc",
    "matmul_alloc",
    "index_select_alloc",
    "reduce_sum_alloc",
)
RUNTIME_PREPARE_OPERATION_KINDS = ("current_stream",)
TRITON_DYNAMIC_ARGUMENT_OPERATION_KINDS = (
    "triton_copy",
    "triton_copy_dynamic_5",
    "triton_copy_dynamic_8",
    "triton_copy_dynamic_11",
)
TRITON_DYNAMIC_ARGUMENT_COUNTS = {
    "triton_copy": 2,
    "triton_copy_dynamic_5": 5,
    "triton_copy_dynamic_8": 8,
    "triton_copy_dynamic_11": 11,
}
CUSTOM_DISPATCH_ARGUMENT_OPERATION_KINDS = (
    "custom_dispatch_arguments_1",
    "custom_dispatch_arguments_4",
    "custom_dispatch_arguments_8",
    "custom_dispatch_arguments_12",
)
CUSTOM_DISPATCH_ARGUMENT_COUNTS = {
    "custom_dispatch_arguments_1": 1,
    "custom_dispatch_arguments_4": 4,
    "custom_dispatch_arguments_8": 8,
    "custom_dispatch_arguments_12": 12,
}
CUSTOM_OP_PREPARE_OPERATION_KINDS = (
    "ascend_moe_gating_topk",
    "ascend_moe_init_routing",
    "ascend_reshape_and_cache",
)
MOE_MLP_PREPARE_OPERATION_KINDS = (
    "ascend_moe_grouped_matmul",
    "ascend_moe_swiglu",
    "ascend_moe_token_unpermute",
)
MODEL_FORWARD_PREPARE_OPERATION_KINDS = (
    "ascend_unquantized_gemm",
)
MOE_GMM_DESCRIPTOR_GROUP_COUNTS = (4, 16, 64, 128)
MOE_GMM_DESCRIPTOR_OPERATION_KINDS = tuple(
    f"ascend_moe_grouped_matmul_groups_{group_count}"
    for group_count in MOE_GMM_DESCRIPTOR_GROUP_COUNTS
)
MOE_GMM_DESCRIPTOR_OPERATION_GROUP_COUNTS = {
    operation_kind: group_count
    for operation_kind, group_count in zip(
        MOE_GMM_DESCRIPTOR_OPERATION_KINDS,
        MOE_GMM_DESCRIPTOR_GROUP_COUNTS,
        strict=True,
    )
}
STREAM_EVENT_OPERATION_KINDS = ("ascend_stream_record_event",)
ROTARY_CACHE_UPDATE_OPERATION_KINDS = ("ascend_rotary_cache_update",)
OPERATION_KINDS = (
    GENERIC_OPERATION_KINDS
    + ALLOCATING_OPERATION_KINDS
    + RUNTIME_PREPARE_OPERATION_KINDS
    + CUSTOM_OP_PREPARE_OPERATION_KINDS
    + MOE_MLP_PREPARE_OPERATION_KINDS
    + MODEL_FORWARD_PREPARE_OPERATION_KINDS
    + MOE_GMM_DESCRIPTOR_OPERATION_KINDS
    + STREAM_EVENT_OPERATION_KINDS
    + ROTARY_CACHE_UPDATE_OPERATION_KINDS
    + TRITON_DYNAMIC_ARGUMENT_OPERATION_KINDS
    + CUSTOM_DISPATCH_ARGUMENT_OPERATION_KINDS
    + ("routing_index_mask_mul", "routing_topk")
)
CHAIN_LENGTHS = (1, 8, 64, 512, 2048)
FIT_CHAIN_LENGTHS = frozenset((1, 8, 64, 512))
TRIALS = 20
WARMUP_REPEATS = 5
SCHEMA_VERSION = 12


@torch.library.custom_op(
    "midlayer_runtime_probe::dispatch_arguments_1", mutates_args=())
def _dispatch_arguments_1(value_0: torch.Tensor) -> None:
    return None


@torch.library.custom_op(
    "midlayer_runtime_probe::dispatch_arguments_4", mutates_args=())
def _dispatch_arguments_4(
    value_0: torch.Tensor,
    value_1: torch.Tensor,
    value_2: torch.Tensor,
    value_3: torch.Tensor,
) -> None:
    return None


@torch.library.custom_op(
    "midlayer_runtime_probe::dispatch_arguments_8", mutates_args=())
def _dispatch_arguments_8(
    value_0: torch.Tensor,
    value_1: torch.Tensor,
    value_2: torch.Tensor,
    value_3: torch.Tensor,
    value_4: torch.Tensor,
    value_5: torch.Tensor,
    value_6: torch.Tensor,
    value_7: torch.Tensor,
) -> None:
    return None


@torch.library.custom_op(
    "midlayer_runtime_probe::dispatch_arguments_12", mutates_args=())
def _dispatch_arguments_12(
    value_0: torch.Tensor,
    value_1: torch.Tensor,
    value_2: torch.Tensor,
    value_3: torch.Tensor,
    value_4: torch.Tensor,
    value_5: torch.Tensor,
    value_6: torch.Tensor,
    value_7: torch.Tensor,
    value_8: torch.Tensor,
    value_9: torch.Tensor,
    value_10: torch.Tensor,
    value_11: torch.Tensor,
) -> None:
    return None


@triton.jit
def _copy_kernel(source, destination, element_count: tl.constexpr,
                 block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(source + offsets, mask=mask)
    tl.store(destination + offsets, values, mask=mask)


@triton.jit
def _copy_kernel_dynamic_5(
    source,
    destination,
    unused_pointer_0,
    unused_pointer_1,
    element_count,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(source + offsets, mask=mask)
    tl.store(destination + offsets, values, mask=mask)


@triton.jit
def _copy_kernel_dynamic_8(
    source,
    destination,
    unused_pointer_0,
    unused_pointer_1,
    unused_pointer_2,
    unused_pointer_3,
    unused_pointer_4,
    element_count,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(source + offsets, mask=mask)
    tl.store(destination + offsets, values, mask=mask)


@triton.jit
def _copy_kernel_dynamic_11(
    source,
    destination,
    unused_pointer_0,
    unused_pointer_1,
    unused_pointer_2,
    unused_pointer_3,
    unused_pointer_4,
    unused_pointer_5,
    unused_pointer_6,
    unused_pointer_7,
    element_count,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    values = tl.load(source + offsets, mask=mask)
    tl.store(destination + offsets, values, mask=mask)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _append_json_line(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _linear_fit(points):
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("runtime-call fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
        raise RuntimeError("runtime-call fit is invalid")
    return intercept, slope


def _unconstrained_linear_fit(points):
    """Return a diagnostic fit without treating a nonpositive slope as valid."""

    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("runtime-call diagnostic fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise RuntimeError("runtime-call diagnostic fit is non-finite")
    return intercept, slope


def _validate_output_dir(path):
    resolved = path.resolve()
    owner_size = len(OUTPUT_OWNER_PARTS)
    offsets = tuple(
        index
        for index in range(len(resolved.parts) - owner_size + 1)
        if resolved.parts[index:index + owner_size] == OUTPUT_OWNER_PARTS
    )
    if len(offsets) != 1 or len(resolved.parts) <= offsets[0] + owner_size:
        raise ValueError("--output-dir must be a run below the Ascend owner")
    if resolved.exists():
        raise FileExistsError("--output-dir must not already exist")
    return resolved


def _make_state(device_id):
    device = f"npu:{device_id}"
    state = {
        "add_left": torch.zeros(4096, dtype=torch.float32, device=device),
        "add_right": torch.empty(4096, dtype=torch.float32, device=device),
        "add_increment": torch.ones(4096, dtype=torch.float32, device=device),
        "matrix_left": torch.ones((64, 64), dtype=torch.float16, device=device),
        "matrix_right": torch.eye(64, dtype=torch.float16, device=device),
        "matrix_output": torch.empty((64, 64), dtype=torch.float16, device=device),
        "index_source": torch.ones((4096, 8), dtype=torch.float32, device=device),
        "index": torch.arange(0, 4096, 2, dtype=torch.int64, device=device),
        "index_output": torch.empty((2048, 8), dtype=torch.float32, device=device),
        "reduce_source": torch.ones((256, 64), dtype=torch.float32, device=device),
        "reduce_output": torch.empty(256, dtype=torch.float32, device=device),
        "routing_topk_ids": torch.remainder(
            torch.arange(4096 * 8, dtype=torch.int32, device=device), 128
        ).reshape(4096, 8),
        "routing_expert_map": torch.arange(
            128, dtype=torch.int32, device=device
        ),
        "routing_weights": torch.ones(
            (4096, 8), dtype=torch.float32, device=device
        ),
        "routing_mapped": torch.empty(
            (4096, 8), dtype=torch.int32, device=device
        ),
        "routing_mask": torch.empty(
            (4096, 8), dtype=torch.bool, device=device
        ),
        "routing_scaled_weights": torch.empty(
            (4096, 8), dtype=torch.float32, device=device
        ),
        "routing_topk_source": torch.randn(
            (4096, 128), dtype=torch.float32, device=device
        ),
        "routing_topk_values": torch.empty(
            (4096, 8), dtype=torch.float32, device=device
        ),
        "routing_topk_indices": torch.empty(
            (4096, 8), dtype=torch.int64, device=device
        ),
        # Deliberately small synthetic shapes isolate the steady-state Host
        # preparation/submit path without reproducing a production operator
        # workload or retaining a production-shape duration.
        "custom_router_logits": torch.randn(
            (16, 16), dtype=torch.float32, device=device
        ),
        "custom_hidden": torch.randn(
            (16, 64), dtype=torch.bfloat16, device=device
        ),
        "custom_topk_ids": torch.remainder(
            torch.arange(16 * 2, dtype=torch.int32, device=device), 16
        ).reshape(16, 2),
        "custom_key": torch.randn(
            (16, 2, 16), dtype=torch.bfloat16, device=device
        ),
        "custom_value": torch.randn(
            (16, 2, 16), dtype=torch.bfloat16, device=device
        ),
        "custom_key_cache": torch.zeros(
            (1, 128, 2, 16), dtype=torch.bfloat16, device=device
        ),
        "custom_value_cache": torch.zeros(
            (1, 128, 2, 16), dtype=torch.bfloat16, device=device
        ),
        "custom_slot_mapping": torch.arange(
            16, dtype=torch.int32, device=device
        ),
        # These inputs deliberately use tiny, non-serving dimensions.  The
        # benchmark measures the Host API path of the actual three MoE MLP
        # calls, not a production MoE duration.
        "custom_moe_w1": torch.randn(
            (16, 64, 128), dtype=torch.bfloat16, device=device
        ),
        "custom_moe_w2": torch.randn(
            (16, 64, 64), dtype=torch.bfloat16, device=device
        ),
        "custom_moe_gate_up": torch.randn(
            (16, 128), dtype=torch.bfloat16, device=device
        ),
        "custom_moe_group_list": torch.ones(
            16, dtype=torch.int64, device=device
        ),
        "custom_moe_permuted": torch.randn(
            (32, 64), dtype=torch.bfloat16, device=device
        ),
        "custom_moe_sorted_indices": torch.arange(
            32, dtype=torch.int32, device=device
        ),
        "custom_moe_probs": torch.ones(
            (16, 2), dtype=torch.bfloat16, device=device
        ),
        # This is the actual Ascend backend entry used by ordinary vLLM
        # Q/K/V/O linears, with deliberately non-serving dimensions.  It
        # measures one generic custom-op-to-aclnnMatmul Host path only.
        "custom_unquantized_gemm_input": torch.randn(
            (8, 64), dtype=torch.bfloat16, device=device
        ),
        "custom_unquantized_gemm_weight": torch.randn(
            (64, 64), dtype=torch.bfloat16, device=device
        ),
        "triton_left": torch.ones(4096, dtype=torch.float32, device=device),
        "triton_right": torch.empty(4096, dtype=torch.float32, device=device),
    } | {
        f"custom_moe_descriptor_hidden_{group_count}": torch.randn(
            (group_count, 64), dtype=torch.bfloat16, device=device
        )
        for group_count in MOE_GMM_DESCRIPTOR_GROUP_COUNTS
    } | {
        f"custom_moe_descriptor_weight_{group_count}": torch.randn(
            (group_count, 64, 128), dtype=torch.bfloat16, device=device
        )
        for group_count in MOE_GMM_DESCRIPTOR_GROUP_COUNTS
    } | {
        f"custom_moe_descriptor_group_list_{group_count}": torch.ones(
            group_count, dtype=torch.int64, device=device
        )
        for group_count in MOE_GMM_DESCRIPTOR_GROUP_COUNTS
    }
    # The production helper has a small global cache interface.  This setup
    # supplies arbitrary, deliberately non-serving cache dimensions and is
    # not a model, a Serving request, or a production tensor shape.  The
    # measured body below is exactly the runtime cache-update mechanism that
    # runs before an async decoder forward.
    rotary_embedding._cos_sin_cache = torch.randn(
        (257, 128), dtype=torch.float32, device=device)
    rotary_embedding._cos = torch.empty(
        (1, 17, 1, 128), dtype=torch.float32, device=device)
    rotary_embedding._sin = torch.empty_like(rotary_embedding._cos)
    rotary_embedding._cos_slice = None
    rotary_embedding._sin_slice = None
    state["rotary_positions"] = torch.remainder(
        torch.arange(17, dtype=torch.int64, device=device), 257)
    return state


def _run_chain(kind, length, state):
    if kind == "add_out":
        source = state["add_left"]
        destination = state["add_right"]
        for _ in range(length):
            torch.add(source, state["add_increment"], out=destination)
            source, destination = destination, source
        return source
    if kind == "matmul_out":
        source = state["matrix_left"]
        destination = state["matrix_output"]
        for _ in range(length):
            torch.mm(source, state["matrix_right"], out=destination)
            source, destination = destination, source
        return source
    if kind == "index_select_out":
        for _ in range(length):
            torch.index_select(
                state["index_source"], 0, state["index"],
                out=state["index_output"])
        return state["index_output"]
    if kind == "reduce_sum_out":
        for _ in range(length):
            torch.sum(
                state["reduce_source"], dim=1,
                out=state["reduce_output"])
        return state["reduce_output"]
    if kind == "add_alloc":
        source = state["add_left"]
        for _ in range(length):
            source = torch.add(source, state["add_increment"])
        return source
    if kind == "matmul_alloc":
        source = state["matrix_left"]
        for _ in range(length):
            source = torch.mm(source, state["matrix_right"])
        return source
    if kind == "index_select_alloc":
        result = None
        for _ in range(length):
            result = torch.index_select(
                state["index_source"], 0, state["index"])
        return result
    if kind == "reduce_sum_alloc":
        result = None
        for _ in range(length):
            result = torch.sum(state["reduce_source"], dim=1)
        return result
    if kind == "current_stream":
        stream = None
        for _ in range(length):
            stream = torch.npu.current_stream()
        return stream
    if kind == "ascend_rotary_cache_update":
        for _ in range(length):
            rotary_embedding.update_cos_sin(state["rotary_positions"])
        cos, sin = rotary_embedding.get_cos_and_sin_slice()
        if cos is None or sin is None:
            raise RuntimeError("rotary cache update did not publish slices")
        return cos
    if kind in CUSTOM_DISPATCH_ARGUMENT_OPERATION_KINDS:
        argument_count = CUSTOM_DISPATCH_ARGUMENT_COUNTS[kind]
        function_by_count = {
            1: _dispatch_arguments_1,
            4: _dispatch_arguments_4,
            8: _dispatch_arguments_8,
            12: _dispatch_arguments_12,
        }
        function = function_by_count[argument_count]
        arguments = (state["custom_hidden"],) * argument_count
        for _ in range(length):
            function(*arguments)
        return state["custom_hidden"]
    if kind == "routing_index_mask_mul":
        topk_ids = state["routing_topk_ids"]
        for _ in range(length):
            state["routing_mapped"].copy_(
                state["routing_expert_map"][topk_ids]
            )
            state["routing_mask"].copy_(topk_ids != -1)
            torch.mul(
                state["routing_weights"],
                state["routing_mask"],
                out=state["routing_scaled_weights"],
            )
        return state["routing_scaled_weights"]
    if kind == "routing_topk":
        for _ in range(length):
            torch.topk(
                state["routing_topk_source"],
                8,
                dim=1,
                largest=True,
                sorted=True,
                out=(
                    state["routing_topk_values"],
                    state["routing_topk_indices"],
                ),
            )
        return state["routing_topk_values"]
    if kind == "ascend_moe_gating_topk":
        result = None
        for _ in range(length):
            result = DeviceOperator.moe_gating_top_k(
                state["custom_router_logits"],
                k=2,
                k_group=1,
                group_count=1,
                group_select_mode=1,
                renorm=0,
                norm_type=0,
                out_flag=False,
            )
        return result
    if kind == "ascend_moe_init_routing":
        result = None
        for _ in range(length):
            result = DeviceOperator.npu_moe_init_routing(
                state["custom_hidden"],
                state["custom_topk_ids"],
                active_num=32,
                expert_num=16,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, 16],
                quant_mode=-1,
            )
        return result
    if kind == "ascend_reshape_and_cache":
        for _ in range(length):
            DeviceOperator.reshape_and_cache(
                state["custom_key"],
                state["custom_value"],
                state["custom_key_cache"],
                state["custom_value_cache"],
                state["custom_slot_mapping"],
            )
        return state["custom_key_cache"]
    if kind == "ascend_moe_grouped_matmul":
        result = None
        for iteration in range(length):
            weight = (
                state["custom_moe_w1"]
                if iteration % 2 == 0 else state["custom_moe_w2"]
            )
            result = torch_npu.npu_grouped_matmul(
                x=[state["custom_hidden"]],
                weight=[weight],
                split_item=2,
                group_list_type=1,
                group_type=0,
                group_list=state["custom_moe_group_list"],
            )[0]
        return result
    if kind in MOE_GMM_DESCRIPTOR_OPERATION_KINDS:
        group_count = MOE_GMM_DESCRIPTOR_OPERATION_GROUP_COUNTS[kind]
        hidden_states = state[
            f"custom_moe_descriptor_hidden_{group_count}"]
        weights = state[f"custom_moe_descriptor_weight_{group_count}"]
        group_list = state[
            f"custom_moe_descriptor_group_list_{group_count}"]
        result = None
        for _ in range(length):
            result = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[weights],
                split_item=2,
                group_list_type=1,
                group_type=0,
                group_list=group_list,
            )[0]
        return result
    if kind == "ascend_moe_swiglu":
        result = None
        for _ in range(length):
            result = torch_npu.npu_swiglu(state["custom_moe_gate_up"])
        return result
    if kind == "ascend_moe_token_unpermute":
        result = None
        for _ in range(length):
            result = torch_npu.npu_moe_token_unpermute(
                permuted_tokens=state["custom_moe_permuted"],
                sorted_indices=state["custom_moe_sorted_indices"],
                probs=state["custom_moe_probs"],
            )
        return result
    if kind == "ascend_unquantized_gemm":
        result = None
        for _ in range(length):
            result = torch.ops.vllm.unquantized_gemm(
                state["custom_unquantized_gemm_input"],
                state["custom_unquantized_gemm_weight"],
                None,
            )
        return result
    if kind == "ascend_stream_record_event":
        events = []
        for _ in range(length):
            events.append(torch.npu.current_stream().record_event())
        return events
    if kind == "triton_copy":
        source = state["triton_left"]
        destination = state["triton_right"]
        for _ in range(length):
            _copy_kernel[(32,)](
                source,
                destination,
                4096,
                block_size=128,
            )
            source, destination = destination, source
        return source
    if kind == "triton_copy_dynamic_5":
        source = state["triton_left"]
        destination = state["triton_right"]
        for _ in range(length):
            _copy_kernel_dynamic_5[(32,)](
                source,
                destination,
                source,
                source,
                4096,
                block_size=128,
            )
            source, destination = destination, source
        return source
    if kind == "triton_copy_dynamic_8":
        source = state["triton_left"]
        destination = state["triton_right"]
        for _ in range(length):
            _copy_kernel_dynamic_8[(32,)](
                source,
                destination,
                source,
                source,
                source,
                source,
                source,
                4096,
                block_size=128,
            )
            source, destination = destination, source
        return source
    if kind == "triton_copy_dynamic_11":
        source = state["triton_left"]
        destination = state["triton_right"]
        for _ in range(length):
            _copy_kernel_dynamic_11[(32,)](
                source,
                destination,
                source,
                source,
                source,
                source,
                source,
                source,
                source,
                source,
                4096,
                block_size=128,
            )
            source, destination = destination, source
        return source
    raise ValueError(f"unknown operation kind: {kind}")


def _measure(stream, kind, length, state):
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record(stream)
    wall_start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    result = _run_chain(kind, length, state)
    cpu_end = time.thread_time_ns()
    wall_end = time.perf_counter_ns()
    end_event.record(stream)
    end_event.synchronize()
    device_ns = float(start_event.elapsed_time(end_event)) * 1_000_000.0
    return result, {
        "host_call_wall_ns": wall_end - wall_start,
        "host_call_thread_cpu_ns": cpu_end - cpu_start,
        "host_call_non_cpu_wall_ns": (wall_end - wall_start) - (cpu_end - cpu_start),
        "device_elapsed_ns": device_ns,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    args = parser.parse_args()
    args.output_dir = _validate_output_dir(args.output_dir)
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if BASELINE_PATTERN.fullmatch(args.baseline_commit) is None:
        parser.error("--baseline-commit must be a full lowercase Git commit")
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if len(args.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    return args


def main():
    args = _parse_args()
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path)
    if source_sha256 != args.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from the audited snapshot")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.json"
    trials_path = args.output_dir / "trials.jsonl"
    summary_path = args.output_dir / "summary.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "baseline_commit": args.baseline_commit,
        "source": {"path": str(source_path), "sha256": source_sha256},
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "container_name": args.container_name,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "contract": {
            "parameter": "one_system_wide_eager_runtime_call_service",
            "operation_kinds": list(OPERATION_KINDS),
            "moe_mlp_prepare_operation_kinds": list(
                MOE_MLP_PREPARE_OPERATION_KINDS),
            "moe_gmm_descriptor_group_counts": list(
                MOE_GMM_DESCRIPTOR_GROUP_COUNTS),
            "stream_event_operation_kinds": list(
                STREAM_EVENT_OPERATION_KINDS),
            "rotary_cache_update_operation_kinds": list(
                ROTARY_CACHE_UPDATE_OPERATION_KINDS),
            "aggregation": (
                "generic_median_excludes_runtime_prepare_and_triton"),
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": sorted(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": 2048,
            "trials_per_cell": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "host_clock": "time.perf_counter_ns",
            "host_thread_cpu_clock": "time.thread_time_ns",
            "device_clock": "torch.npu.Event(enable_timing=True)",
            "synchronization_excluded_from_host_call": True,
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
    }
    _write_json(manifest_path, manifest)
    try:
        NPUPlatform.import_kernels()
        if not enable_custom_op():
            raise RuntimeError("vLLM-Ascend custom-op registration failed")
        torch.npu.set_device(args.device_id)
        stream = torch.npu.Stream(device=args.device_id)
        samples = {(kind, length): [] for kind in OPERATION_KINDS for length in CHAIN_LENGTHS}
        with torch.npu.stream(stream):
            state = _make_state(args.device_id)
            for kind in OPERATION_KINDS:
                for length in CHAIN_LENGTHS:
                    for _ in range(WARMUP_REPEATS):
                        _run_chain(kind, length, state)
                    stream.synchronize()
            sequence = 0
            cells = tuple((kind, length) for kind in OPERATION_KINDS for length in CHAIN_LENGTHS)
            for trial in range(TRIALS):
                offset = trial % len(cells)
                ordered = cells[offset:] + cells[:offset]
                if trial % 2:
                    ordered = tuple(reversed(ordered))
                for order, (kind, length) in enumerate(ordered, start=1):
                    sequence += 1
                    result, timing = _measure(stream, kind, length, state)
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "sequence": sequence,
                        "trial": trial + 1,
                        "order": order,
                        "operation_kind": kind,
                        "chain_length": length,
                        **timing,
                    }
                    samples[(kind, length)].append(row)
                    _append_json_line(trials_path, row)
            stream.synchronize()
            if not all(torch.isfinite(tensor).all().item() for tensor in state.values()):
                raise RuntimeError("tensor validation failed")
            del result
        fits = []
        cells_summary = []
        for kind in OPERATION_KINDS:
            medians = {}
            for length in CHAIN_LENGTHS:
                cell_samples = samples[(kind, length)]
                wall = [sample["host_call_wall_ns"] for sample in cell_samples]
                cpu = [sample["host_call_thread_cpu_ns"] for sample in cell_samples]
                medians[length] = {
                    "wall_ns": statistics.median(wall),
                    "thread_cpu_ns": statistics.median(cpu),
                }
                cells_summary.append({
                    "operation_kind": kind,
                    "chain_length": length,
                    "sample_count": len(cell_samples),
                    "host_call_wall_ns": {
                        "p10": _percentile(wall, 0.1),
                        "median": medians[length]["wall_ns"],
                        "p90": _percentile(wall, 0.9),
                    },
                    "host_call_thread_cpu_ns": {
                        "p10": _percentile(cpu, 0.1),
                        "median": statistics.median(cpu),
                        "p90": _percentile(cpu, 0.9),
                    },
                })
            wall_intercept, wall_slope = _linear_fit([
                (length, medians[length]["wall_ns"])
                for length in sorted(FIT_CHAIN_LENGTHS)
            ])
            cpu_intercept, cpu_slope = _linear_fit([
                (length, medians[length]["thread_cpu_ns"])
                for length in sorted(FIT_CHAIN_LENGTHS)
            ])
            holdout_wall = medians[2048]["wall_ns"]
            holdout_cpu = medians[2048]["thread_cpu_ns"]
            fits.append({
                "operation_kind": kind,
                "wall_fixed_intercept_ns": wall_intercept,
                "wall_slope_ns_per_call": wall_slope,
                "thread_cpu_fixed_intercept_ns": cpu_intercept,
                "thread_cpu_slope_ns_per_call": cpu_slope,
                "holdout_wall_measured_ns": holdout_wall,
                "holdout_wall_predicted_ns": (
                    wall_intercept + wall_slope * 2048),
                "holdout_wall_signed_error_percent": 100.0 * (
                    (wall_intercept + wall_slope * 2048) / holdout_wall - 1.0),
                "holdout_thread_cpu_measured_ns": holdout_cpu,
                "holdout_thread_cpu_predicted_ns": (
                    cpu_intercept + cpu_slope * 2048),
                "holdout_thread_cpu_signed_error_percent": 100.0 * (
                    (cpu_intercept + cpu_slope * 2048) / holdout_cpu - 1.0),
            })
        generic_fits = tuple(
            fit for fit in fits
            if fit["operation_kind"] in GENERIC_OPERATION_KINDS)
        parameter_ns = statistics.median(
            fit["wall_slope_ns_per_call"] for fit in generic_fits)
        parameter_thread_cpu_ns = statistics.median(
            fit["thread_cpu_slope_ns_per_call"] for fit in generic_fits)
        fit_by_kind = {
            fit["operation_kind"]: fit for fit in fits
        }
        allocation_pairs = (
            ("add_out", "add_alloc"),
            ("matmul_out", "matmul_alloc"),
            ("index_select_out", "index_select_alloc"),
            ("reduce_sum_out", "reduce_sum_alloc"),
        )
        allocation_wall_increments = tuple(
            fit_by_kind[allocating]["wall_slope_ns_per_call"]
            - fit_by_kind[preallocated]["wall_slope_ns_per_call"]
            for preallocated, allocating in allocation_pairs
        )
        allocation_cpu_increments = tuple(
            fit_by_kind[allocating]["thread_cpu_slope_ns_per_call"]
            - fit_by_kind[preallocated]["thread_cpu_slope_ns_per_call"]
            for preallocated, allocating in allocation_pairs
        )
        triton_fit, = tuple(
            fit for fit in fits if fit["operation_kind"] == "triton_copy")
        triton_argument_fits = tuple(
            fit for fit in fits
            if fit["operation_kind"]
            in TRITON_DYNAMIC_ARGUMENT_OPERATION_KINDS
        )
        triton_argument_fit_by_count = {
            TRITON_DYNAMIC_ARGUMENT_COUNTS[fit["operation_kind"]]: fit
            for fit in triton_argument_fits
        }
        triton_argument_wall_intercept_ns, triton_argument_wall_slope_ns = (
            _linear_fit([
                (
                    dynamic_argument_count,
                    triton_argument_fit_by_count[dynamic_argument_count][
                        "wall_slope_ns_per_call"
                    ],
                )
                for dynamic_argument_count in (2, 5, 8)
            ])
        )
        (
            triton_argument_thread_cpu_intercept_ns,
            triton_argument_thread_cpu_slope_ns,
        ) = _linear_fit([
            (
                dynamic_argument_count,
                triton_argument_fit_by_count[dynamic_argument_count][
                    "thread_cpu_slope_ns_per_call"
                ],
            )
            for dynamic_argument_count in (2, 5, 8)
        ])
        triton_argument_holdout = triton_argument_fit_by_count[11]
        triton_argument_holdout_wall_ns = triton_argument_holdout[
            "wall_slope_ns_per_call"
        ]
        triton_argument_holdout_thread_cpu_ns = triton_argument_holdout[
            "thread_cpu_slope_ns_per_call"
        ]
        custom_dispatch_argument_fits = tuple(
            fit for fit in fits
            if fit["operation_kind"]
            in CUSTOM_DISPATCH_ARGUMENT_OPERATION_KINDS
        )
        custom_dispatch_argument_fit_by_count = {
            CUSTOM_DISPATCH_ARGUMENT_COUNTS[fit["operation_kind"]]: fit
            for fit in custom_dispatch_argument_fits
        }
        (
            custom_dispatch_argument_wall_intercept_ns,
            custom_dispatch_argument_wall_slope_ns,
        ) = _linear_fit([
            (
                argument_count,
                custom_dispatch_argument_fit_by_count[argument_count][
                    "wall_slope_ns_per_call"
                ],
            )
            for argument_count in (1, 4, 8)
        ])
        (
            custom_dispatch_argument_thread_cpu_intercept_ns,
            custom_dispatch_argument_thread_cpu_slope_ns,
        ) = _linear_fit([
            (
                argument_count,
                custom_dispatch_argument_fit_by_count[argument_count][
                    "thread_cpu_slope_ns_per_call"
                ],
            )
            for argument_count in (1, 4, 8)
        ])
        custom_dispatch_argument_holdout = (
            custom_dispatch_argument_fit_by_count[12])
        current_stream_fit, = tuple(
            fit for fit in fits
            if fit["operation_kind"] == "current_stream")
        routing_fits = {
            fit["operation_kind"]: fit
            for fit in fits
            if fit["operation_kind"] in (
                "routing_index_mask_mul", "routing_topk")
        }
        custom_op_fits = tuple(
            fit for fit in fits
            if fit["operation_kind"] in CUSTOM_OP_PREPARE_OPERATION_KINDS)
        custom_op_total_ns = statistics.median(
            fit["wall_slope_ns_per_call"] for fit in custom_op_fits)
        custom_op_thread_cpu_ns = statistics.median(
            fit["thread_cpu_slope_ns_per_call"] for fit in custom_op_fits)
        descriptor_fit_by_group_count = {
            MOE_GMM_DESCRIPTOR_OPERATION_GROUP_COUNTS[
                fit["operation_kind"]
            ]: fit
            for fit in fits
            if fit["operation_kind"] in MOE_GMM_DESCRIPTOR_OPERATION_KINDS
        }
        descriptor_fit_group_counts = (4, 16, 64)
        (
            descriptor_wall_fixed_ns,
            descriptor_wall_ns_per_group,
        ) = _unconstrained_linear_fit([
            (
                group_count,
                descriptor_fit_by_group_count[group_count][
                    "wall_slope_ns_per_call"],
            )
            for group_count in descriptor_fit_group_counts
        ])
        (
            descriptor_thread_cpu_fixed_ns,
            descriptor_thread_cpu_ns_per_group,
        ) = _unconstrained_linear_fit([
            (
                group_count,
                descriptor_fit_by_group_count[group_count][
                    "thread_cpu_slope_ns_per_call"],
            )
            for group_count in descriptor_fit_group_counts
        ])
        descriptor_holdout = descriptor_fit_by_group_count[128]
        descriptor_status = (
            "diagnostic_only_requires_source_descriptor_metadata"
            if descriptor_wall_ns_per_group > 0
            else "unsupported_nonpositive_group_scaling"
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "validation": "pass",
            "parameter": {
                "name": "eager_runtime_call_service_ns",
                "wall_value": parameter_ns,
                "thread_cpu_value": parameter_thread_cpu_ns,
                "aggregation": (
                    "median_of_generic_non_triton_per-kind_fit_slopes"),
            },
            "triton_runtime_prepare_candidate": {
                "status": "diagnostic_only_requires_dispatch_metadata",
                "wall_total_slope_ns_per_call": triton_fit[
                    "wall_slope_ns_per_call"],
                "wall_increment_over_generic_ns_per_call": (
                    triton_fit["wall_slope_ns_per_call"] - parameter_ns),
                "thread_cpu_total_slope_ns_per_call": triton_fit[
                    "thread_cpu_slope_ns_per_call"],
                "thread_cpu_increment_over_generic_ns_per_call": (
                    triton_fit["thread_cpu_slope_ns_per_call"]
                    - parameter_thread_cpu_ns),
            },
            "triton_dynamic_argument_prepare_candidate": {
                "status": (
                    "diagnostic_only_requires_source_launch_descriptor"),
                "mechanism": (
                    "warm-cache Triton launcher argument specialization and "
                    "packing before asynchronous submission"),
                "fit_dynamic_argument_counts": [2, 5, 8],
                "holdout_dynamic_argument_count": 11,
                "wall_fixed_ns_per_launch": (
                    triton_argument_wall_intercept_ns),
                "wall_ns_per_dynamic_argument": (
                    triton_argument_wall_slope_ns),
                "thread_cpu_fixed_ns_per_launch": (
                    triton_argument_thread_cpu_intercept_ns),
                "thread_cpu_ns_per_dynamic_argument": (
                    triton_argument_thread_cpu_slope_ns),
                "holdout_wall_measured_ns_per_launch": (
                    triton_argument_holdout_wall_ns),
                "holdout_wall_predicted_ns_per_launch": (
                    triton_argument_wall_intercept_ns
                    + triton_argument_wall_slope_ns * 11),
                "holdout_wall_signed_error_percent": 100.0 * (
                    (
                        triton_argument_wall_intercept_ns
                        + triton_argument_wall_slope_ns * 11
                    )
                    / triton_argument_holdout_wall_ns
                    - 1.0
                ),
                "holdout_thread_cpu_measured_ns_per_launch": (
                    triton_argument_holdout_thread_cpu_ns),
                "holdout_thread_cpu_predicted_ns_per_launch": (
                    triton_argument_thread_cpu_intercept_ns
                    + triton_argument_thread_cpu_slope_ns * 11),
                "holdout_thread_cpu_signed_error_percent": 100.0 * (
                    (
                        triton_argument_thread_cpu_intercept_ns
                        + triton_argument_thread_cpu_slope_ns * 11
                    )
                    / triton_argument_holdout_thread_cpu_ns
                    - 1.0
                ),
                "observations": [
                    {
                        "operation_kind": fit["operation_kind"],
                        "dynamic_argument_count": (
                            TRITON_DYNAMIC_ARGUMENT_COUNTS[
                                fit["operation_kind"]
                        ]),
                        "wall_ns_per_launch": fit[
                            "wall_slope_ns_per_call"],
                        "thread_cpu_ns_per_launch": fit[
                            "thread_cpu_slope_ns_per_call"],
                    }
                    for fit in triton_argument_fits
                ],
            },
            "custom_dispatch_argument_prepare_candidate": {
                "status": (
                    "diagnostic_only_requires_source_dispatch_descriptor"),
                "mechanism": (
                    "backend-neutral Torch custom-op dispatch and tensor "
                    "argument packing without a child Device operation"),
                "fit_tensor_argument_counts": [1, 4, 8],
                "holdout_tensor_argument_count": 12,
                "wall_fixed_ns_per_call": (
                    custom_dispatch_argument_wall_intercept_ns),
                "wall_ns_per_tensor_argument": (
                    custom_dispatch_argument_wall_slope_ns),
                "thread_cpu_fixed_ns_per_call": (
                    custom_dispatch_argument_thread_cpu_intercept_ns),
                "thread_cpu_ns_per_tensor_argument": (
                    custom_dispatch_argument_thread_cpu_slope_ns),
                "holdout_wall_measured_ns_per_call": (
                    custom_dispatch_argument_holdout[
                        "wall_slope_ns_per_call"]),
                "holdout_wall_predicted_ns_per_call": (
                    custom_dispatch_argument_wall_intercept_ns
                    + custom_dispatch_argument_wall_slope_ns * 12),
                "holdout_wall_signed_error_percent": 100.0 * (
                    (
                        custom_dispatch_argument_wall_intercept_ns
                        + custom_dispatch_argument_wall_slope_ns * 12
                    )
                    / custom_dispatch_argument_holdout[
                        "wall_slope_ns_per_call"]
                    - 1.0
                ),
                "holdout_thread_cpu_measured_ns_per_call": (
                    custom_dispatch_argument_holdout[
                        "thread_cpu_slope_ns_per_call"]),
                "holdout_thread_cpu_predicted_ns_per_call": (
                    custom_dispatch_argument_thread_cpu_intercept_ns
                    + custom_dispatch_argument_thread_cpu_slope_ns * 12),
                "holdout_thread_cpu_signed_error_percent": 100.0 * (
                    (
                        custom_dispatch_argument_thread_cpu_intercept_ns
                        + custom_dispatch_argument_thread_cpu_slope_ns * 12
                    )
                    / custom_dispatch_argument_holdout[
                        "thread_cpu_slope_ns_per_call"]
                    - 1.0
                ),
                "observations": [
                    {
                        "operation_kind": fit["operation_kind"],
                        "tensor_argument_count": (
                            CUSTOM_DISPATCH_ARGUMENT_COUNTS[
                                fit["operation_kind"]
                        ]),
                        "wall_ns_per_call": fit["wall_slope_ns_per_call"],
                        "thread_cpu_ns_per_call": fit[
                            "thread_cpu_slope_ns_per_call"],
                    }
                    for fit in custom_dispatch_argument_fits
                ],
            },
            "output_allocation_runtime_prepare_candidate": {
                "status": (
                    "diagnostic_only_requires_source_occurrence_metadata"),
                "aggregation": (
                    "median_of_matched_allocating_minus_preallocated_slopes"),
                "operation_pairs": [
                    {
                        "preallocated": preallocated,
                        "allocating": allocating,
                        "wall_increment_ns_per_call": wall_increment,
                        "thread_cpu_increment_ns_per_call": cpu_increment,
                    }
                    for (
                        (preallocated, allocating),
                        wall_increment,
                        cpu_increment,
                    ) in zip(
                        allocation_pairs,
                        allocation_wall_increments,
                        allocation_cpu_increments,
                        strict=True,
                    )
                ],
                "wall_value_ns_per_call": statistics.median(
                    allocation_wall_increments),
                "thread_cpu_value_ns_per_call": statistics.median(
                    allocation_cpu_increments),
                "measurement_boundary": (
                    "incremental normal-output tensor creation over matched "
                    "preallocated out call, no device synchronization"),
            },
            "current_stream_runtime_prepare_candidate": {
                "status": "diagnostic_only_requires_source_occurrence_metadata",
                "wall_slope_ns_per_call": current_stream_fit[
                    "wall_slope_ns_per_call"],
                "thread_cpu_slope_ns_per_call": current_stream_fit[
                    "thread_cpu_slope_ns_per_call"],
                "holdout_wall_signed_error_percent": current_stream_fit[
                    "holdout_wall_signed_error_percent"],
                "holdout_thread_cpu_signed_error_percent": current_stream_fit[
                    "holdout_thread_cpu_signed_error_percent"],
            },
            "rotary_cache_update_runtime_prepare_candidate": {
                "status": "diagnostic_only_requires_source_occurrence_metadata",
                "operation_kind": "ascend_rotary_cache_update",
                "mechanism": (
                    "NPU rotary cosine/sine cache slice update before the "
                    "decoder forward, measured on synthetic cache tensors"),
                "wall_slope_ns_per_call": fit_by_kind[
                    "ascend_rotary_cache_update"]["wall_slope_ns_per_call"],
                "thread_cpu_slope_ns_per_call": fit_by_kind[
                    "ascend_rotary_cache_update"]["thread_cpu_slope_ns_per_call"],
                "holdout_wall_signed_error_percent": fit_by_kind[
                    "ascend_rotary_cache_update"]["holdout_wall_signed_error_percent"],
                "holdout_thread_cpu_signed_error_percent": fit_by_kind[
                    "ascend_rotary_cache_update"]["holdout_thread_cpu_signed_error_percent"],
                "measurement_boundary": (
                    "rotary_embedding.update_cos_sin entry to async return; "
                    "synchronization is outside the timed interval"),
            },
            "routing_runtime_prepare_candidates": {
                kind: {
                    "status": "diagnostic_only_requires_source_occurrence_metadata",
                    "wall_slope_ns_per_call": fit[
                        "wall_slope_ns_per_call"],
                    "thread_cpu_slope_ns_per_call": fit[
                        "thread_cpu_slope_ns_per_call"],
                    "holdout_wall_signed_error_percent": fit[
                        "holdout_wall_signed_error_percent"],
                    "holdout_thread_cpu_signed_error_percent": fit[
                        "holdout_thread_cpu_signed_error_percent"],
                }
                for kind, fit in routing_fits.items()
            },
            "dynamic_custom_op_runtime_prepare_candidate": {
                "status": (
                    "diagnostic_only_requires_source_occurrence_metadata"),
                "aggregation": (
                    "median_of_predeclared_custom_op_host_call_slopes"),
                "operation_kinds": list(
                    CUSTOM_OP_PREPARE_OPERATION_KINDS),
                "wall_total_ns_per_call": custom_op_total_ns,
                "wall_increment_over_generic_ns_per_call": (
                    custom_op_total_ns - parameter_ns),
                "thread_cpu_total_ns_per_call": custom_op_thread_cpu_ns,
                "thread_cpu_increment_over_generic_ns_per_call": (
                    custom_op_thread_cpu_ns - parameter_thread_cpu_ns),
                "measurement_boundary": (
                    "python_custom_op_entry_to_async_return_no_device_sync"),
            },
            "moe_mlp_runtime_prepare_candidates": {
                kind: {
                    "status": (
                        "diagnostic_only_requires_source_occurrence_metadata"),
                    "wall_total_ns_per_call": fit_by_kind[kind][
                        "wall_slope_ns_per_call"],
                    "wall_increment_over_generic_ns_per_call": (
                        fit_by_kind[kind]["wall_slope_ns_per_call"]
                        - parameter_ns),
                    "thread_cpu_total_ns_per_call": fit_by_kind[kind][
                        "thread_cpu_slope_ns_per_call"],
                    "thread_cpu_increment_over_generic_ns_per_call": (
                        fit_by_kind[kind]["thread_cpu_slope_ns_per_call"]
                        - parameter_thread_cpu_ns),
                    "holdout_wall_signed_error_percent": fit_by_kind[kind][
                        "holdout_wall_signed_error_percent"],
                    "measurement_boundary": (
                        "actual torch_npu MoE API call to asynchronous return; "
                        "tiny synthetic non-serving tensors; post-measure "
                        "synchronization excluded"),
                }
                for kind in MOE_MLP_PREPARE_OPERATION_KINDS
            },
            "unquantized_gemm_runtime_prepare_candidate": {
                "status": (
                    "diagnostic_only_requires_source_occurrence_metadata"),
                "operation_kind": "ascend_unquantized_gemm",
                "wall_total_ns_per_call": fit_by_kind[
                    "ascend_unquantized_gemm"]["wall_slope_ns_per_call"],
                "wall_increment_over_generic_ns_per_call": (
                    fit_by_kind["ascend_unquantized_gemm"]
                    ["wall_slope_ns_per_call"] - parameter_ns),
                "thread_cpu_total_ns_per_call": fit_by_kind[
                    "ascend_unquantized_gemm"]["thread_cpu_slope_ns_per_call"],
                "thread_cpu_increment_over_generic_ns_per_call": (
                    fit_by_kind["ascend_unquantized_gemm"]
                    ["thread_cpu_slope_ns_per_call"]
                    - parameter_thread_cpu_ns),
                "holdout_wall_signed_error_percent": fit_by_kind[
                    "ascend_unquantized_gemm"]
                    ["holdout_wall_signed_error_percent"],
                "measurement_boundary": (
                    "vllm unquantized_gemm custom-op entry to asynchronous "
                    "return; tiny synthetic tensors and post-measure "
                    "synchronization excluded"),
            },
            "moe_grouped_gemm_descriptor_prepare_candidate": {
                "status": descriptor_status,
                "mechanism": (
                    "CANN grouped-GEMM group-list descriptor traversal and "
                    "argument preparation before asynchronous submit"),
                "synthetic_hidden_width": 64,
                "synthetic_output_width": 128,
                "fit_group_counts": list(descriptor_fit_group_counts),
                "holdout_group_count": 128,
                "wall_fixed_ns_per_call": descriptor_wall_fixed_ns,
                "wall_ns_per_group": descriptor_wall_ns_per_group,
                "thread_cpu_fixed_ns_per_call": (
                    descriptor_thread_cpu_fixed_ns),
                "thread_cpu_ns_per_group": (
                    descriptor_thread_cpu_ns_per_group),
                "holdout_wall_measured_ns_per_call": descriptor_holdout[
                    "wall_slope_ns_per_call"],
                "holdout_wall_predicted_ns_per_call": (
                    descriptor_wall_fixed_ns
                    + descriptor_wall_ns_per_group * 128),
                "holdout_wall_signed_error_percent": 100.0 * (
                    (
                        descriptor_wall_fixed_ns
                        + descriptor_wall_ns_per_group * 128
                    )
                    / descriptor_holdout["wall_slope_ns_per_call"]
                    - 1.0
                ),
                "measurement_boundary": (
                    "npu_grouped_matmul Host call to asynchronous return; "
                    "synthetic fixed-width tensors and varying group-list "
                    "length; post-measure synchronization excluded"),
            },
            "stream_event_runtime_prepare_candidate": {
                "status": (
                    "diagnostic_only_requires_source_occurrence_metadata"),
                "operation_kind": "ascend_stream_record_event",
                "wall_total_ns_per_event": fit_by_kind[
                    "ascend_stream_record_event"][
                    "wall_slope_ns_per_call"],
                "thread_cpu_total_ns_per_event": fit_by_kind[
                    "ascend_stream_record_event"][
                    "thread_cpu_slope_ns_per_call"],
                "holdout_wall_signed_error_percent": fit_by_kind[
                    "ascend_stream_record_event"][
                    "holdout_wall_signed_error_percent"],
                "measurement_boundary": (
                    "torch.npu.current_stream().record_event() to return; "
                    "event synchronization excluded"),
            },
            "fits": fits,
            "cells": cells_summary,
        }
        _write_json(summary_path, summary)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
