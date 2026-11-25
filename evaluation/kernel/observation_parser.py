"""Normalize runtime launch records for single-kernel validation only.

Runtime names and profiler schemas are deliberately confined to this module.
Production launch selection starts from structured workload semantics and does
not call any of these helpers.
"""

from __future__ import annotations

from typing import Mapping, Optional

from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.tiling_spec import (
    parse_complete_gemm_tile,
    parse_flash_attention_trait,
    parse_flash_attention_tile,
    parse_gemm_tile,
    parse_pipeline_stages,
)


_GLOBAL_NON_TENSOR_PRIMITIVES = (
    "elementwise", "vectorized_elementwise", "unrolled_elementwise",
    "index_elementwise", "cast", "copy", "direct_copy", "fill",
    "reduce", "splitkreduce", "devdstelementwiseadd",
    "rms_norm", "rmsnorm", "fused_add_rms_norm", "rotary", "rope",
    "swiglu", "act_and_mul", "softmax",
    "gather", "indexselect", "sort", "histogram", "topkgating",
    "moe_align", "moe_sum", "count_and_sort", "enable_if",
    "moeinitrouting", "moegatingtopk", "moetokenunpermute",
    "moefinalizerouting", "moecomputeexperttokens",
    "combine", "transpose", "range", "reshape_and_cache",
    "reshapeandcache", "internal_gemvx", "gemv2t", "b16gemvt", "gemv",
    "triton_poi", "indexselectsmallindex", "get_mla_metadata",
    "fwd_kernel_stage2",
)

_TENSOR_MAIN_PRIMITIVES = {
    "GEMM": (
        "s16816gemm", "s1688gemm", "gemm", "matmul", "cutlass",
        "gemmini", "xmma", "b16gemm", "nvjet_",
    ),
    "FLASHATTN": (
        "flash", "attention", "fusedinfer", "fused_infer",
        "fwd_grouped_kernel", "fwd_kernel",
    ),
    "FUSEDMOE": (
        "groupedmatmul", "grouped_matmul", "fused_moe",
    ),
    "UNFUSEDMOE": (
        "moe_decomposed", "gemmini_moe_decomposed", "pim_moe_decomposed",
    ),
    "EAGERATTN": (
        "gemmini_eagerattn", "pim_eagerattn", "eager_attn", "eagerattn",
    ),
}


class KernelClassificationError(ValueError):
    """An observed runtime kernel has no registered primitive classification."""


def is_non_tensor(op_type, runtime_name, ascend_op_type=None):
    """Classify an observed kernel without a silent tensor fallback."""

    if not isinstance(op_type, str) or not op_type:
        raise TypeError("operator type must be a non-empty string")
    if not isinstance(runtime_name, str) or not runtime_name:
        raise TypeError("runtime kernel name must be a non-empty string")
    if ascend_op_type is not None:
        if not isinstance(ascend_op_type, str) or not ascend_op_type:
            raise TypeError(
                "Ascend profiler op_type must be a non-empty string or null")
        normalized_type = ascend_op_type.lower()
        if normalized_type == "vector":
            return True
        if normalized_type in ("cube", "mix"):
            return False
        raise KernelClassificationError(
            f"unknown Ascend profiler op_type: {ascend_op_type!r}")
    lowered = runtime_name.lower()
    if any(token in lowered for token in _GLOBAL_NON_TENSOR_PRIMITIVES):
        return True
    if op_type not in _TENSOR_MAIN_PRIMITIVES:
        raise KernelClassificationError(
            f"unknown operator type for runtime classification: {op_type!r}")
    if any(token in lowered for token in _TENSOR_MAIN_PRIMITIVES[op_type]):
        return False
    raise KernelClassificationError(
        f"cannot classify runtime kernel {runtime_name!r} for {op_type!r}; "
        "register it as a non-tensor primitive or as the operator's main compute")


def observed_launch_config(
    metrics: Mapping,
    kernel_kind: Optional[KernelKind | str] = None,
    shape: Optional[Mapping[str, int]] = None,
    *,
    algorithm: str = "observed_runtime_launch",
    variant: Optional[str] = None,
) -> LaunchConfig:
    """Build structured geometry from extracted profiler measurements.

    ``algorithm`` and ``variant`` are supplied by the validation classifier;
    the raw runtime name is never copied into the production launch object.
    """
    kind = KernelKind.parse(kernel_kind) if kernel_kind is not None else None
    tiling = None
    if kind is KernelKind.GEMM:
        parsed = parse_complete_gemm_tile(metrics["kernel_name"])
        if parsed is not None:
            tile_m, tile_n, tile_k, depth = parsed
            if tile_k is None:
                raise ValueError(
                    "cannot infer GEMM K tile from observed runtime kernel: "
                    f"{metrics['kernel_name']!r}")
            tiling = TilingSpec(
                tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
                pipeline_depth=depth,
                split_k=metrics["grid_z"] if metrics["grid_z"] > 1 else 1,
            )
    elif kind is KernelKind.FLASH_ATTN:
        parsed = parse_flash_attention_tile(metrics["kernel_name"])
        if parsed is not None:
            query_tile, kv_tile = parsed
            if shape is None or "head_dim" not in shape:
                raise ValueError(
                    "FLASH_ATTN observed launch requires shape['head_dim']")
            head = shape["head_dim"]
            if isinstance(head, bool) or not isinstance(head, int):
                raise TypeError(
                    "FLASH_ATTN observed shape.head_dim must be an integer")
            if head <= 0:
                raise ValueError(
                    "FLASH_ATTN observed shape.head_dim must be positive")
            if metrics["grid_z"] <= 0:
                raise ValueError("observed FLASH_ATTN grid_z must be positive")
            tiling = TilingSpec(
                tile_shape={"q": query_tile, "kv": kv_tile, "head": head},
                num_kv_splits=metrics["grid_z"],
            )
    elif kind in (
            KernelKind.EAGER_ATTN, KernelKind.FUSED_MOE,
            KernelKind.UNFUSED_MOE, KernelKind.VECTOR,
            KernelKind.KV_CACHE):
        pass
    elif kind is None:
        pass
    else:
        raise ValueError(
            f"observed launch geometry is not valid for {kind.value}")
    return LaunchConfig(
        algorithm=algorithm if variant is None else f"{algorithm}:{variant}",
        grid=(metrics["grid_x"], metrics["grid_y"], metrics["grid_z"]),
        block=(metrics["block_x"], metrics["block_y"], metrics["block_z"]),
        tiling=tiling,
        execution_variant=variant,
    )
