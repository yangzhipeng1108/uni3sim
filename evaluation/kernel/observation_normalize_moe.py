"""MoE observation normalization for kernel evaluation: fused and
unfused MoE plus per-hardware MoE auxiliary launch attributes."""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional

from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    MOE_CAST_ROLES,
    MOE_FILL_ROLES,
)
from src.kernel.launch_config import TilingSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.launch_config.select_launch_config import (
    select_fused_moe_reduce_work_fractions,
)
from evaluation.kernel.observation_parser import (
    is_non_tensor,
    observed_launch_config,
)
from evaluation.kernel.profiler_observation import extract_ncu_metrics

from evaluation.json_shape import require_positive_metric as _positive_int

from evaluation.kernel.observation_readers import (
    _A800_PYTORCH_2_8_BF16_REDUCE_SYMBOL,
    _a800_moe_sum_symbol,
)
from evaluation.kernel.observation_normalize_gemm_vector import (
    _require_source_bound_vector_observation,
)


# The one hardware each source-derived MoE auxiliary handler normalizes.
# A variant whose handler is hardware-specific names its hardware here and the
# message its mismatch raises; the handler body is unchanged.  ``{variant}``
# and ``{hardware_id}`` are filled at the raise site.
_MOE_AUX_HARDWARE = {
    "moe_reduce": (
        "nvidia_a800",
        "source-derived moe_reduce observation normalization is "
        "defined only for nvidia_a800, got {hardware_id!r}"),
    "moe_swiglu": (
        "ascend_910b",
        "source-derived moe_swiglu normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_compute_expert_tokens": (
        "ascend_910b",
        "source-derived moe_compute_expert_tokens normalization is "
        "defined only for ascend_910b, got {hardware_id!r}"),
    "moe_cast": (
        "ascend_910b",
        "typed moe_cast normalization is defined only for "
        "ascend_910b, got {hardware_id!r}"),
    "moe_init_routing": (
        "ascend_910b",
        "source-derived {variant} normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_finalize_routing": (
        "ascend_910b",
        "source-derived {variant} normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_transpose": (
        "ascend_910b",
        "source-derived {variant} normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_range": (
        "ascend_910b",
        "source-derived {variant} normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_silu": (
        "metax_c500",
        "source-derived {variant} work partition is defined only "
        "for metax_c500, got {hardware_id!r}"),
    "moe_mul": (
        "metax_c500",
        "source-derived {variant} work partition is defined only "
        "for metax_c500, got {hardware_id!r}"),
}


def require_moe_aux_hardware(variant: str, hardware_id: str) -> None:
    """Fail loud when a hardware-specific MoE auxiliary sees another hardware."""

    if variant not in _MOE_AUX_HARDWARE:
        return
    expected, message = _MOE_AUX_HARDWARE[variant]
    if hardware_id != expected:
        raise ValueError(message.format(variant=variant, hardware_id=hardware_id))


_ASCEND_910B_MOE_COMPUTE_EXPERT_TOKENS_CODE_HASH = (
    "d1c627adc45619675efeff94a5330a80")
_ASCEND_910B_MOE_CAST_RUNTIME_BY_ROLE = {
    "topk_ids": (
        "Cast_9f288b80370c6545ffe8cef142d37f5c_"
        "high_performance_210000000"),
    "expert_tokens": (
        "Cast_e87590d11ccda8b259ab6b1ea7212319_"
        "high_performance_210000000"),
}


def _normalize_unfused_moe(op_attributes: Mapping, kdata: Mapping,
                           hardware_id: str, hardware) -> KernelSpec:
    """Normalize one aggregate router/expert/reduction decomposition."""

    metrics = extract_ncu_metrics(kdata)
    if is_non_tensor(
            "UNFUSEDMOE", metrics["kernel_name"],
            ascend_op_type=metrics["op_type"]):
        raise ValueError(
            "UNFUSEDMOE dataset entry must describe its aggregate tensor path")
    if hardware.kernel_memory_model not in (
            "pim_bank", "systolic_scratchpad"):
        raise ValueError(
            "UNFUSEDMOE observations require a cacheless execution model")
    shape = {
        "tokens": _positive_int(op_attributes, "TOK", "UNFUSED_MOE Attr"),
        "experts": _positive_int(op_attributes, "E", "UNFUSED_MOE Attr"),
        "topk": _positive_int(op_attributes, "TOPK", "UNFUSED_MOE Attr"),
        "hidden": _positive_int(op_attributes, "H", "UNFUSED_MOE Attr"),
        "intermediate": _positive_int(
            op_attributes, "INT", "UNFUSED_MOE Attr"),
    }
    raw_counts = op_attributes["tok_per_expert"]
    if not isinstance(raw_counts, list):
        raise TypeError("UNFUSED_MOE Attr.tok_per_expert must be a list")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in raw_counts):
        raise TypeError(
            "UNFUSED_MOE Attr.tok_per_expert must contain integers")
    launch = observed_launch_config(
        metrics, KernelKind.UNFUSED_MOE, shape,
        algorithm="observed_unfused_moe", variant="decomposed")
    return KernelSpec(
        kind=KernelKind.UNFUSED_MOE,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant="decomposed",
        launch=launch,
        attributes={
            "tokens_per_expert": tuple(raw_counts),
            "expert_weight_tile_rows": None,
        },
    )


def _fused_moe_projection(tiling, runtime_name: str, grid_size: int,
                          tensor_ordinal: Optional[int]) -> str:
    lowered = runtime_name.lower()
    if lowered.endswith("_w1") or "_w1" in lowered:
        return "w1"
    if lowered.endswith("_w2") or "_w2" in lowered:
        return "w2"
    if tiling is not None:
        matches = [
            name for name in ("w1", "w2")
            if _positive_int(
                tiling[name], "grid", f"FUSED_MOE {name} tiling")
            == grid_size]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 2 and tensor_ordinal in (0, 1):
            return ("w1", "w2")[tensor_ordinal]
    raise ValueError(
        "cannot determine fused-MoE projection from structured observation; "
        "pass tensor_ordinal when the two observed grids collide")


def _normalize_fused_moe(op_attributes: Mapping, kdata: Mapping,
                         hardware_id: str, hardware,
                         tensor_ordinal: Optional[int],
                         kernel_ordinal: Optional[int],
                         moe_reduce_ordinal: Optional[int], *,
                         profiler_metrics: Optional[Mapping] = None
                         ) -> KernelSpec:
    metrics = (
        extract_ncu_metrics(kdata)
        if profiler_metrics is None else profiler_metrics)
    if not isinstance(metrics, Mapping):
        raise TypeError("FUSEDMOE profiler_metrics must be a mapping")
    runtime_name = metrics["kernel_name"]
    if is_non_tensor(
            "FUSEDMOE", runtime_name,
            ascend_op_type=metrics["op_type"]):
        return _normalize_moe_aux(
            op_attributes, kdata, hardware_id, hardware, kernel_ordinal,
            moe_reduce_ordinal, profiler_metrics=metrics)
    memory_model = hardware.kernel_memory_model
    if memory_model == "npu_tile":
        projection = _fused_moe_projection(
            None, runtime_name, metrics["grid_size"], tensor_ordinal)
    elif memory_model == "gpu_mshr":
        projection = _fused_moe_projection(
            op_attributes["moe_tiling"], runtime_name,
            metrics["grid_size"], tensor_ordinal)
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")
    shape = {
        "tokens": _positive_int(op_attributes, "TOK", "FUSED_MOE Attr"),
        "experts": _positive_int(op_attributes, "E", "FUSED_MOE Attr"),
        "topk": _positive_int(op_attributes, "TOPK", "FUSED_MOE Attr"),
        "hidden": _positive_int(op_attributes, "H", "FUSED_MOE Attr"),
        "intermediate": _positive_int(
            op_attributes, "INT", "FUSED_MOE Attr"),
    }
    op_type = metrics["op_type"]
    if op_type == "cube":
        algorithm = "cann_grouped_matmul"
    elif op_type is None:
        algorithm = "triton_grouped_gemm"
    else:
        raise ValueError(
            f"unsupported FUSED_MOE tensor profiler op_type: {op_type!r}")
    launch = observed_launch_config(
        metrics, KernelKind.FUSED_MOE, shape,
        algorithm=algorithm,
        variant=projection)
    parameters = {
        "group_size_m": 0.0,
        "padded_rows": 0.0,
        "mte2_tile_bytes": 0.0,
        "weight_tile_rows": 0.0,
        "operand_replay_a": 1.0,
        "operand_replay_b": 1.0,
    }
    raw_counts = op_attributes["tok_per_expert"]
    if not isinstance(raw_counts, list):
        raise TypeError("FUSED_MOE Attr.tok_per_expert must be a list")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in raw_counts):
        raise TypeError(
            "FUSED_MOE Attr.tok_per_expert must contain integers")
    attributes = {
        "tokens_per_expert": tuple(raw_counts),
        "local_tokens_per_expert": tuple(raw_counts),
        "expert_parallel_size": 1,
        "expert_rank": 0,
        "owned_experts": tuple(range(shape["experts"])),
        "expert_weight_tile_rows": None,
    }
    if op_type == "cube":
        active_counts = [
            count for count in attributes["tokens_per_expert"] if count > 0]
        if not active_counts:
            raise ValueError("FUSED_MOE routing must contain an active expert")
        canonical_launch = select_launch_config(
            hardware_id,
            KernelKind.FUSED_MOE,
            shape,
            hardware.kernel_dtype,
            projection,
            {
                "launch_policy": "analytical_launch",
                "expert_parallel_size": 1,
            },
            hardware,
        )
        tiling = canonical_launch.tiling
        parameters = dict(canonical_launch.parameters)
    elif op_type is None:
        if projection in ("w1", "w2"):
            tiling_projection = projection
        else:
            raise ValueError(
                f"unknown FUSED_MOE projection: {projection!r}")
        tiling_data = op_attributes["moe_tiling"][tiling_projection]
        tiling = TilingSpec(tile_shape={
            "m": _positive_int(tiling_data, "BM", "FUSED_MOE tiling"),
            "n": _positive_int(tiling_data, "BN", "FUSED_MOE tiling"),
            "k": _positive_int(tiling_data, "BK", "FUSED_MOE tiling"),
        })
        parameters.update({
            "group_size_m": _positive_int(
                tiling_data, "GSM", "FUSED_MOE tiling"),
            "padded_rows": _positive_int(
                tiling_data, "EM", "FUSED_MOE tiling"),
        })
        if hardware_id == "nvidia_h200":
            # NSYS records include the actual dynamic shared allocation. Keep
            # this launch fact attached to the observed program so a
            # source-scoped mechanism cannot match on tile shape alone.
            parameters["dynamic_shared_memory_bytes"] = (
                metrics["dynamic_smem_per_block"])
            parameters["registers_per_thread"] = metrics["regs_per_thread"]
            parameters["static_shared_bytes"] = metrics["static_smem_per_block"]
            from src.kernel.launch_config.nvidia_h200.moe_source_launch import (
                observed_operand_pipeline_depth,
            )
            tiling = dataclasses.replace(tiling, pipeline_depth=observed_operand_pipeline_depth(
                shape, hardware.kernel_dtype, tiling, parameters["group_size_m"],
                metrics["block_size"], parameters["dynamic_shared_memory_bytes"],
                registers_per_thread=parameters["registers_per_thread"],
                static_shared_bytes=parameters["static_shared_bytes"]))
        if hardware_id == "metax_c500":
            from src.kernel.launch_config.metax_c500.fused_moe_launch import (
                select_fused_moe_operand_replay,
                select_fused_moe_w2_routed_weight_program,
            )
            from src.kernel.model_contract import (
                SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
            )
            operand_replay = select_fused_moe_operand_replay(
                int(tiling.tile_shape["m"]))
            parameters["operand_replay_a"] = operand_replay
            parameters["operand_replay_b"] = operand_replay
            parameters[
                SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER
            ] = select_fused_moe_w2_routed_weight_program(projection)
    else:
        raise ValueError(
            f"unhandled FUSED_MOE tensor profiler op_type: {op_type!r}")
    launch = dataclasses.replace(
        launch, tiling=tiling, parameters=parameters)
    return KernelSpec(
        kind=KernelKind.FUSED_MOE,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant=projection,
        launch=launch,
        attributes=attributes,
    )


def _moe_aux_variant(runtime_name: str, ascend_op_type: Optional[str]) -> str:
    lowered = runtime_name.lower()
    if ascend_op_type:
        if "swiglu" in lowered:
            return "moe_swiglu"
        if "initrouting" in lowered:
            return "moe_init_routing"
        if "finalizerouting" in lowered:
            return "moe_finalize_routing"
        if "computeexperttokens" in lowered:
            return "moe_compute_expert_tokens"
        if "transpose" in lowered:
            return "moe_transpose"
        if "cast" in lowered:
            return "moe_cast"
        if "range" in lowered or "arange" in lowered:
            return "moe_range"
        if "softmax" in lowered:
            return "moe_softmax"
        if "topk" in lowered:
            return "moe_topk"
        if "gating" in lowered:
            return "moe_gating"
        raise ValueError(f"unknown Huawei MoE auxiliary implementation: {runtime_name}")
    if "act_and_mul" in lowered:
        return "moe_act_and_mul"
    if "silu_kernel" in lowered:
        return "moe_silu"
    if "mulfunctor" in lowered:
        return "moe_mul"
    # PyTorch's CUDA reduction symbol also embeds ``sum_functor``.  Match the
    # implementation entry point first so default top-k reductions do not get
    # confused with vLLM's dedicated ``moe_sum`` kernel or MetaX's distinct
    # TensorIterator implementation.
    if "reduce_kernel" in lowered:
        return "moe_reduce"
    if "moe_sum" in lowered or "sum_functor" in lowered:
        return "moe_sum"
    if "moe_align" in lowered:
        if "opt_128" in lowered:
            return "moe_align_opt_128"
        if "small_batch" in lowered:
            return "moe_align_small_batch"
        return "moe_align"
    if "count_and_sort" in lowered:
        return "moe_count_and_sort"
    if "vectorized_elementwise" in lowered:
        return "moe_fill"
    if "enable_if" in lowered:
        return "moe_enable_if"
    if "softmax" in lowered:
        return "moe_softmax"
    if "topk" in lowered:
        return "moe_topk"
    if "gating" in lowered:
        return "moe_gating"
    raise ValueError(f"unknown MoE auxiliary implementation: {runtime_name}")


def _normalize_a800_moe_reduce_attributes(
        shape: Mapping[str, int], runtime_name: str, launch, hardware,
        moe_reduce_ordinal: Optional[int]) -> Mapping[str, float]:
    """Bind one observed launch to a PyTorch 2.8 output-slice fraction."""

    if (hardware.kernel_dtype != "bf16"
            or runtime_name != _A800_PYTORCH_2_8_BF16_REDUCE_SYMBOL):
        raise ValueError(
            "A800 PyTorch moe_reduce observation does not match the retained "
            "BF16 PyTorch 2.8 reduction symbol")
    fractions = select_fused_moe_reduce_work_fractions(
        "nvidia_a800", shape, hardware)
    if moe_reduce_ordinal is None:
        raise ValueError(
            "A800 PyTorch moe_reduce requires its physical reduce ordinal")
    if not 0 <= moe_reduce_ordinal < len(fractions):
        raise ValueError(
            "A800 PyTorch moe_reduce ordinal is outside the source-derived "
            f"partition: ordinal={moe_reduce_ordinal}, count={len(fractions)}")
    fraction = fractions[moe_reduce_ordinal]
    expected = select_launch_config(
        "nvidia_a800", KernelKind.VECTOR, shape,
        hardware.kernel_dtype, "moe_reduce",
        {"work_fraction": fraction}, hardware)
    if launch.grid != expected.grid or launch.block != expected.block:
        raise ValueError(
            "A800 PyTorch moe_reduce launch disagrees with its ordered "
            "source-derived output slice: "
            f"ordinal={moe_reduce_ordinal}, "
            f"grid={launch.grid}, block={launch.block}, "
            f"expected_grid={expected.grid}, "
            f"expected_block={expected.block}")
    return {"work_fraction": fraction}


def _normalize_ascend_moe_compute_expert_tokens_launch(
        shape: Mapping[str, int], metrics: Mapping, observed_launch,
        hardware):
    """Bind the observed helper to CANN's source-selected program."""

    canonical_launch = select_launch_config(
        "ascend_910b", KernelKind.VECTOR, shape,
        hardware.kernel_dtype, "moe_compute_expert_tokens",
        {"launch_policy": "injected_launch"}, hardware)
    tiling_key = int(canonical_launch.parameters["cann_tiling_key"])
    expected_runtime_name = (
        "MoeComputeExpertTokens_"
        f"{_ASCEND_910B_MOE_COMPUTE_EXPERT_TOKENS_CODE_HASH}_"
        f"{tiling_key}_mix_aiv")
    _require_source_bound_vector_observation(
        "ascend_910b", "moe_compute_expert_tokens",
        metrics["kernel_name"], observed_launch, canonical_launch,
        expected_runtime_name)
    return dataclasses.replace(
        observed_launch,
        algorithm=canonical_launch.algorithm,
        parameters=canonical_launch.parameters,
        execution_variant=canonical_launch.execution_variant,
        source_program=canonical_launch.source_program,
    )


def _normalize_ascend_moe_source_launch(
        shape: Mapping[str, int], metrics: Mapping, observed_launch,
        variant: str, attributes: Mapping[str, object], hardware):
    """Require one retained CANN MoE helper to match its source tiler."""

    selection_attributes = {
        "launch_policy": "analytical_launch",
        **attributes,
    }
    canonical_launch = select_launch_config(
        "ascend_910b", KernelKind.VECTOR, shape,
        hardware.kernel_dtype, variant, selection_attributes, hardware)
    if (observed_launch.grid != canonical_launch.grid
            or observed_launch.block != canonical_launch.block):
        raise ValueError(
            f"Ascend {variant} profiler geometry disagrees with the CANN "
            f"source selector: expected grid={canonical_launch.grid}, "
            f"block={canonical_launch.block}, actual "
            f"grid={observed_launch.grid}, block={observed_launch.block}")
    return dataclasses.replace(
        observed_launch,
        algorithm=canonical_launch.algorithm,
        parameters=canonical_launch.parameters,
        execution_variant=canonical_launch.execution_variant,
        source_program=canonical_launch.source_program,
    )


def _normalize_metax_moe_fill_attributes(
        shape: Mapping[str, int], metrics: Mapping, launch,
        hardware_id: str, kernel_ordinal: Optional[int]) -> Mapping[str, str]:
    """Bind each native int32 Fill launch to its source-level output tensor."""

    if hardware_id != "metax_c500":
        raise ValueError(
            "VECTOR/moe_fill observation semantics are version-bound to "
            f"metax_c500, got {hardware_id!r}")
    if kernel_ordinal is None:
        raise ValueError(
            "metax_c500 VECTOR/moe_fill requires the physical kernel ordinal")
    if kernel_ordinal >= len(MOE_FILL_ROLES):
        raise ValueError(
            "metax_c500 VECTOR/moe_fill must be one of the first three "
            f"physical kernels, got ordinal {kernel_ordinal}")
    fill_role = MOE_FILL_ROLES[kernel_ordinal]

    block_m = 16 if shape["tokens"] <= shape["experts"] else 64
    max_padded = (
        shape["tokens"] * shape["topk"]
        + shape["experts"] * (block_m - 1)
    )
    elements_by_role = {
        "sorted_token_ids": max_padded,
        "expert_ids": (max_padded + block_m - 1) // block_m,
        "num_tokens_post_padded": 1,
    }
    elements = elements_by_role[fill_role]

    runtime_name = metrics["kernel_name"]
    standard_template = (
        "vectorized_elementwise_kernel<4, at::native::FillFunctor<int>")
    optimized_template = (
        "vectorized_elementwise_kernel_nullary_opt<1024, 4, int, "
        "at::native::FillFunctor<int>")
    if standard_template in runtime_name:
        expected_block_x = 256
    elif optimized_template in runtime_name:
        expected_block_x = 1024
    else:
        raise ValueError(
            "metax_c500 VECTOR/moe_fill runtime symbol does not name a "
            "source-supported int32 vec4 Fill template")
    if launch.block != (expected_block_x, 1, 1):
        raise ValueError(
            "metax_c500 VECTOR/moe_fill block disagrees with its runtime "
            f"template: expected={(expected_block_x, 1, 1)}, "
            f"actual={launch.block}")
    expected_grid = (
        (elements + expected_block_x * 4 - 1)
        // (expected_block_x * 4),
        1,
        1,
    )
    if launch.grid != expected_grid:
        raise ValueError(
            "metax_c500 VECTOR/moe_fill grid disagrees with the source "
            f"tensor length for {fill_role}: expected={expected_grid}, "
            f"actual={launch.grid}")
    return {"fill_role": fill_role}


# vLLM 0.30.0 ``_prepare_expert_assignment`` (vllm/model_executor/layers/
# fused_moe/fused_moe.py): when ``num_tokens * top_k * 4 <= num_experts`` and
# no expert map is set, ``moe_align_block_size`` is skipped and the only
# helper is ``torch.full((1,), topk_ids.numel() * BLOCK_SIZE_M, int32)`` --
# the ``num_tokens_post_padded`` scalar.  PyTorch's vectorized FillFunctor
# launch uses ``num_threads() = C10_WARP_SIZE * 4`` = 128 threads and one CTA
# per 512 elements.
_H200_VLLM030_NAIVE_ASSIGNMENT_SPARSITY = 4
_H200_TORCH_FILL_SYMBOL_PREFIX = (
    "void at::native::vectorized_elementwise_kernel<(int)4, "
    "at::native::FillFunctor<int>, ")
_H200_TORCH_VECTORIZED_THREADS = 128
_H200_TORCH_VECTORIZED_ELEMENTS_PER_THREAD = 4


def _normalize_h200_moe_fill_attributes(
        shape: Mapping[str, int], metrics: Mapping, launch,
        kernel_ordinal: Optional[int]) -> Mapping[str, str]:
    """Bind the H200 vLLM 0.30 naive-assignment Fill to its source tensor."""

    naive = (
        shape["tokens"] * shape["topk"]
        * _H200_VLLM030_NAIVE_ASSIGNMENT_SPARSITY <= shape["experts"])
    if not naive:
        raise ValueError(
            "nvidia_h200 VECTOR/moe_fill appears only on the vLLM 0.30 "
            "naive block-assignment path (TOK*TOPK*4 <= E)")
    if kernel_ordinal != 0:
        raise ValueError(
            "nvidia_h200 VECTOR/moe_fill must be the first physical kernel, "
            f"got ordinal {kernel_ordinal}")
    if not metrics["kernel_name"].startswith(_H200_TORCH_FILL_SYMBOL_PREFIX):
        raise ValueError(
            "nvidia_h200 VECTOR/moe_fill runtime symbol is not the int32 "
            "vec4 FillFunctor template")
    elements = 1
    per_cta = (_H200_TORCH_VECTORIZED_THREADS
               * _H200_TORCH_VECTORIZED_ELEMENTS_PER_THREAD)
    expected_grid = ((elements + per_cta - 1) // per_cta, 1, 1)
    expected_block = (_H200_TORCH_VECTORIZED_THREADS, 1, 1)
    if launch.grid != expected_grid or launch.block != expected_block:
        raise ValueError(
            "nvidia_h200 VECTOR/moe_fill geometry disagrees with the "
            f"one-element torch.full launch: expected {expected_grid}/"
            f"{expected_block}, actual {launch.grid}/{launch.block}")
    return {"fill_role": "num_tokens_post_padded"}


# vLLM 0.30.0 csrc/libtorch_stable/moe/moe_align_sum_kernels.cu ``moe_sum``
# (non-pad-aware path, BF16, contiguous hidden): 16-byte vectors
# (VEC = 8 BF16), block 256, grid min(ceil(TOK * H / 8 / 256), 65535);
# ``moe_sum_vec_kernel<..., TOPK, false>`` for TOPK in {1, 2, 4, 6, 8, 9},
# otherwise ``moe_sum_vec_dynamic_kernel``.
_H200_MOE_SUM_STATIC_TOPK = (1, 2, 4, 6, 8, 9)
_H200_MOE_SUM_BLOCK = 256
_H200_MOE_SUM_VEC_BF16 = 8
_H200_MOE_SUM_MAX_GRID = 65535


def _validate_h200_moe_sum_launch(
        shape: Mapping[str, int], metrics: Mapping, launch, hardware) -> None:
    if hardware.kernel_dtype != "bf16":
        raise ValueError("nvidia_h200 moe_sum source binding is BF16 only")
    hidden = shape["hidden"]
    if hidden % _H200_MOE_SUM_VEC_BF16:
        raise NotImplementedError(
            "nvidia_h200 moe_sum scalar (non-vectorized) path is not bound")
    topk = shape["topk"]
    if topk in _H200_MOE_SUM_STATIC_TOPK:
        expected_prefix = (
            "void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, int, "
            f"(int){topk}, (bool)0>(")
    else:
        expected_prefix = (
            "void vllm::moe::moe_sum_vec_dynamic_kernel<c10::BFloat16, int, "
            "(bool)0>(")
    if not metrics["kernel_name"].startswith(expected_prefix):
        raise ValueError(
            "nvidia_h200 moe_sum runtime symbol disagrees with the vLLM 0.30 "
            f"TOPK={topk} dispatch")
    total = shape["tokens"] * (hidden // _H200_MOE_SUM_VEC_BF16)
    expected_grid = (
        min((total + _H200_MOE_SUM_BLOCK - 1) // _H200_MOE_SUM_BLOCK,
            _H200_MOE_SUM_MAX_GRID), 1, 1)
    if (launch.grid != expected_grid
            or launch.block != (_H200_MOE_SUM_BLOCK, 1, 1)):
        raise ValueError(
            "nvidia_h200 moe_sum geometry disagrees with the vLLM 0.30 "
            f"source: expected {expected_grid}/({_H200_MOE_SUM_BLOCK}, 1, 1),"
            f" actual {launch.grid}/{launch.block}")


def _normalize_moe_aux(op_attributes: Mapping, kdata: Mapping,
                       hardware_id: str, hardware,
                       kernel_ordinal: Optional[int],
                       moe_reduce_ordinal: Optional[int], *,
                       profiler_metrics: Optional[Mapping] = None
                       ) -> KernelSpec:
    metrics = (
        extract_ncu_metrics(kdata)
        if profiler_metrics is None else profiler_metrics)
    if not isinstance(metrics, Mapping):
        raise TypeError("MoE auxiliary profiler_metrics must be a mapping")
    variant = _moe_aux_variant(metrics["kernel_name"], metrics["op_type"])
    shape = {
        "tokens": _positive_int(op_attributes, "TOK", "FUSED_MOE Attr"),
        "experts": _positive_int(op_attributes, "E", "FUSED_MOE Attr"),
        "topk": _positive_int(op_attributes, "TOPK", "FUSED_MOE Attr"),
        "hidden": _positive_int(op_attributes, "H", "FUSED_MOE Attr"),
        "intermediate": _positive_int(
            op_attributes, "INT", "FUSED_MOE Attr"),
        "dtype_bytes": 2,
    }
    launch = observed_launch_config(
        metrics, KernelKind.VECTOR, shape,
        algorithm="observed_moe_aux", variant=variant)
    attributes = {}
    if variant == "moe_fill" and hardware_id == "nvidia_h200":
        attributes = _normalize_h200_moe_fill_attributes(
            shape, metrics, launch, kernel_ordinal)
    elif variant == "moe_fill":
        attributes = _normalize_metax_moe_fill_attributes(
            shape, metrics, launch, hardware_id, kernel_ordinal)
    elif variant == "moe_sum":
        if hardware_id == "nvidia_h200":
            _validate_h200_moe_sum_launch(shape, metrics, launch, hardware)
        elif hardware_id == "nvidia_a800":
            if (hardware.kernel_dtype != "bf16"
                    or metrics["kernel_name"]
                    != _a800_moe_sum_symbol(shape["topk"])):
                raise ValueError(
                    "A800 moe_sum observation does not match the exact "
                    "retained vLLM 0.11.0 BF16 symbol")
        elif hardware_id == "metax_c500":
            pass
        else:
            raise ValueError(
                "moe_sum observation normalization is defined only for "
                "nvidia_a800 and metax_c500, got "
                f"{hardware_id!r}")
        # MetaX TensorIterator splits reductions once the logical input crosses
        # its signed 30-bit indexing window.  Both observed invocations retain
        # the same launch grid, but the independent per-invocation counters
        # show that each reads and writes exactly half of the logical tensor.
        logical_input_elements = (
            shape["tokens"] * shape["topk"] * shape["hidden"])
        if hardware_id == "metax_c500" and logical_input_elements > 2 ** 30:
            work_fraction = 0.5
        else:
            work_fraction = 1.0
        attributes = {"work_fraction": work_fraction}
    elif variant == "moe_reduce":
        require_moe_aux_hardware(variant, hardware_id)
        attributes = _normalize_a800_moe_reduce_attributes(
            shape, metrics["kernel_name"], launch, hardware,
            moe_reduce_ordinal)
    elif variant == "moe_swiglu":
        require_moe_aux_hardware(variant, hardware_id)
        if metrics["kernel_name"] != "SwiGlu_3_high_performance_27":
            raise ValueError(
                "Ascend moe_swiglu observation does not match the exact "
                "SwiGlu_3_high_performance_27 source symbol")
        routed_rows = shape["tokens"] * shape["topk"]
        attributes = {"routed_rows": routed_rows}
        source_launch = select_launch_config(
            hardware_id,
            KernelKind.VECTOR,
            shape,
            hardware.kernel_dtype,
            variant,
            {
                "launch_policy": "analytical_launch",
                "routed_rows": routed_rows,
            },
            hardware,
        )
        if (launch.grid != source_launch.grid
                or launch.block != source_launch.block):
            raise ValueError(
                "Ascend moe_swiglu profiler geometry disagrees with the "
                "CANN source selector: "
                f"expected grid={source_launch.grid}, "
                f"block={source_launch.block}, actual grid={launch.grid}, "
                f"block={launch.block}")
        launch = source_launch
    elif variant == "moe_compute_expert_tokens":
        require_moe_aux_hardware(variant, hardware_id)
        raw_counts = op_attributes["tok_per_expert"]
        if (not isinstance(raw_counts, list)
                or any(isinstance(value, bool)
                       or not isinstance(value, int)
                       for value in raw_counts)):
            raise TypeError(
                "FUSED_MOE Attr.tok_per_expert must contain integers")
        attributes = {"tokens_per_expert": tuple(raw_counts)}
        launch = _normalize_ascend_moe_compute_expert_tokens_launch(
            shape, metrics, launch, hardware)
    elif variant == "moe_cast":
        require_moe_aux_hardware(variant, hardware_id)
        roles = [
            role for role in MOE_CAST_ROLES
            if metrics["kernel_name"]
            == _ASCEND_910B_MOE_CAST_RUNTIME_BY_ROLE[role]
        ]
        if len(roles) != 1:
            raise ValueError(
                "Ascend moe_cast runtime symbol does not identify exactly "
                f"one source role: {metrics['kernel_name']!r}")
        attributes = {"cast_role": roles[0]}
        launch = _normalize_ascend_moe_source_launch(
            shape, metrics, launch, variant, attributes, hardware)
    elif variant in {
            "moe_init_routing", "moe_finalize_routing",
            "moe_transpose", "moe_range"}:
        require_moe_aux_hardware(variant, hardware_id)
        launch = _normalize_ascend_moe_source_launch(
            shape, metrics, launch, variant, attributes, hardware)
    elif variant in {"moe_silu", "moe_mul"}:
        require_moe_aux_hardware(variant, hardware_id)
        logical_elements = (
            shape["tokens"] * shape["topk"] * shape["intermediate"])
        work_fraction = 0.5 if logical_elements > 2 ** 29 else 1.0
        attributes = {"work_fraction": work_fraction}
        launch_attributes = {
            "launch_policy": "injected_launch",
            **attributes,
        }
        source_launch = select_launch_config(
            hardware_id, KernelKind.VECTOR, shape,
            hardware.kernel_dtype, variant, launch_attributes, hardware)
        if (launch.grid != source_launch.grid
                or launch.block != source_launch.block):
            raise ValueError(
                f"MetaX {variant} observation disagrees with its "
                "source-derived TensorIterator partition")
        launch = dataclasses.replace(
            launch,
            algorithm=source_launch.algorithm,
            execution_variant=source_launch.execution_variant,
        )
    elif variant == "moe_align" and hardware_id == "nvidia_h200":
        from src.kernel.launch_config.select_launch_config import moe_alignment_capacity
        raw_counts = op_attributes["tok_per_expert"]
        if (not isinstance(raw_counts, list)
                or any(isinstance(value, bool)
                       or not isinstance(value, int) or value < 0
                       for value in raw_counts)):
            raise TypeError("FUSED_MOE Attr.tok_per_expert must contain non-negative integers")
        if len(raw_counts) != shape["experts"] or sum(raw_counts) != shape["tokens"] * shape["topk"]:
            raise ValueError("ordinary alignment routing histogram disagrees with its FUSED_MOE shape")
        block_size = _positive_int(
            op_attributes["moe_tiling"]["w1"], "BM", "FUSED_MOE W1 tiling")
        attributes = {
            "alignment_block_size": block_size,
            "alignment_capacity": moe_alignment_capacity(
                hardware_id, shape["tokens"], shape["topk"], shape["experts"], block_size),
            "tokens_per_expert": tuple(raw_counts),
        }
    elif variant in {
            "moe_softmax",
            "moe_topk", "moe_gating", "moe_act_and_mul",
            "moe_align", "moe_align_opt_128",
            "moe_align_small_batch", "moe_count_and_sort",
            "moe_enable_if"}:
        pass
    else:
        raise ValueError(
            f"unhandled normalized MoE auxiliary variant: {variant!r}")
    return KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant=variant,
        launch=launch,
        attributes=attributes,
    )
