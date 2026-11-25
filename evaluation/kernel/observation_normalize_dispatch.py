"""Kernel observation normalization dispatch across kernel kinds."""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional

from src.kernel.kernel_spec import KernelSpec, L2ExecutionContext
from evaluation.kernel.profiler_observation import extract_ncu_metrics

from evaluation.kernel.observation_readers import (
    _non_empty_string,
)
from evaluation.kernel.observation_normalize_gemm_vector import (
    _normalize_gemm,
    _normalize_eager_attn,
    _normalize_vector,
)
from evaluation.kernel.observation_normalize_flash_attention import (
    _normalize_flash_attention,
)
from evaluation.kernel.observation_normalize_moe import (
    _normalize_unfused_moe,
    _normalize_fused_moe,
)


def normalize_kernel_spec(op_info: Mapping, kdata: Mapping,
                          hardware_id: str, hardware,
                          *, tensor_ordinal: Optional[int] = None,
                          kernel_ordinal: Optional[int] = None,
                          moe_reduce_ordinal: Optional[int] = None,
                          gemm_split_k: Optional[int] = None,
                          gemm_split_k_partial_bytes: Optional[int] = None,
                          gemm_split_k_partial_dram_bytes: Optional[int] = None,
                          profiler_metrics: Optional[Mapping] = None,
                          l2_execution_context: Optional[
                              L2ExecutionContext] = None,
                          ) -> KernelSpec:
    """Return a complete production spec for one profiler kernel observation."""

    if kernel_ordinal is not None:
        if isinstance(kernel_ordinal, bool) or not isinstance(
                kernel_ordinal, int):
            raise TypeError("kernel_ordinal must be an integer or None")
        if kernel_ordinal < 0:
            raise ValueError("kernel_ordinal cannot be negative")
    if moe_reduce_ordinal is not None:
        if (isinstance(moe_reduce_ordinal, bool)
                or not isinstance(moe_reduce_ordinal, int)):
            raise TypeError("moe_reduce_ordinal must be an integer or None")
        if moe_reduce_ordinal < 0:
            raise ValueError("moe_reduce_ordinal cannot be negative")
    profiler_metrics = (
        extract_ncu_metrics(kdata)
        if profiler_metrics is None else profiler_metrics)
    if not isinstance(profiler_metrics, Mapping):
        raise TypeError("normalize_kernel_spec profiler_metrics must be a mapping")
    if l2_execution_context is None:
        l2_execution_context = L2ExecutionContext()
    if not isinstance(l2_execution_context, L2ExecutionContext):
        raise TypeError(
            "normalize_kernel_spec l2_execution_context must be an "
            "L2ExecutionContext or None")
    if (hardware.kernel_memory_model == "npu_tile"
            and profiler_metrics["op_type"] is None):
        raise ValueError(
            "npu_tile observation requires an explicit profiler Op Type")
    op_type = _non_empty_string(op_info, "Type", "op_info").upper()
    attributes = op_info["Attr"]
    if op_type == "GEMM":
        spec = _normalize_gemm(
            attributes, kdata, hardware_id, hardware, gemm_split_k,
            gemm_split_k_partial_bytes,
            gemm_split_k_partial_dram_bytes)
        return dataclasses.replace(
            spec, l2_execution_context=l2_execution_context)
    elif op_type == "FLASHATTN":
        return _normalize_flash_attention(
            attributes, kdata, hardware_id, hardware)
    elif op_type == "EAGERATTN":
        return _normalize_eager_attn(
            attributes, kdata, hardware_id, hardware)
    elif op_type == "FUSEDMOE":
        return _normalize_fused_moe(
            attributes, kdata, hardware_id, hardware, tensor_ordinal,
            kernel_ordinal, moe_reduce_ordinal,
            profiler_metrics=profiler_metrics)
    elif op_type == "UNFUSEDMOE":
        return _normalize_unfused_moe(
            attributes, kdata, hardware_id, hardware)
    elif op_type in (
            "NORM", "ROTARY", "ACTIVATION", "KV_CACHE", "ELEMENTWISE",
            "REDUCE", "GATHER"):
        return _normalize_vector(
            op_type, attributes, kdata, hardware_id, hardware)
    else:
        raise NotImplementedError(
            f"observation normalization is not implemented for {op_type}")
