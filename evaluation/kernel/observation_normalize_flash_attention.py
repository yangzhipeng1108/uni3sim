"""FlashAttention observation normalization for kernel evaluation."""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Mapping

from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config import TilingSpec
from src.kernel.launch_config.tiling_spec import (
    METAX_MLA_SCHEDULER_PARTITIONS,
    derive_mla_scheduler_work,
)
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
    FA3_COMBINE_ALGORITHM,
    FA3_DECODE_ALGORITHM,
    FA3_GQA_OBSERVED_030_ALGORITHM,
    FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM,
    FA3_GQA_COMBINE_030_ALGORITHM,
    select_v030_gqa_fresh_prefill_launch,
    select_fa3_prepare_varlen_launch,
    validate_v030_gqa_observed_launch,
    v030_gqa_dynamic_splits,
)
from src.kernel.predictors.flash_attn import (
    FA3_UNMODELED_OBSERVATION_ALGORITHM,
    fa3_decode_dynamic_splits,
)
from src.kernel.launch_config.tiling_spec import parse_flash_attention_trait
from evaluation.kernel.observation_parser import (
    is_non_tensor,
    observed_launch_config,
)
from evaluation.kernel.profiler_observation import extract_ncu_metrics
from src.kernel.model_contract import SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER

from evaluation.json_shape import require_positive_metric as _positive_int

from evaluation.kernel.observation_readers import (
    _optional_positive_int,
    _non_empty_string,
    _finite_number,
    _positive_int_sequence,
)


_METAX_372_STATIC_FLASH_TRAITS = frozenset({
    (64, 16, 16, 1),
    (64, 64, 64, 4),
    (128, 16, 16, 1),
    (128, 64, 64, 4),
    (128, 128, 64, 4),
})


_METAX_372_Q64_PREFILL_RUNTIME = (
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, "
    "4, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, "
    "64, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, "
    "false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, "
    "int, int)")
_METAX_372_D64_Q64_PREFILL_RUNTIME = (
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<64, 64, 64, "
    "4, true, true, mctlass::bfloat16_t, 64, Flash_kernel_traits<64, "
    "64, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, "
    "false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, "
    "int, int)")
_METAX_372_Q64_PREFILL_ALGORITHM = (
    "observed_metax_flash_attention_trait_direct:prefill_q64_full_tile")
_METAX_372_Q64_PACKED_PREFILL_ALGORITHM = (
    "observed_metax_flash_attention_trait_packed:prefill_q64_full_tile")
_METAX_372_Q64_DECODE_RUNTIME = (
    "void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, "
    "true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, "
    "64, 64, 4, mctlass::bfloat16_t> >, false, false, false, false, "
    "false, false, true, false, false, 1, false, (Arch)1000>"
    "(mcFlashAttn::Flash_fwd_params, int, int)")
_METAX_372_Q64_DECODE_ALGORITHM = (
    "observed_metax_flash_attention_trait_direct:decode_q64_full_tile")
_METAX_372_Q128_PREFILL_RUNTIME = (
    "void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, "
    "true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, "
    "128, 64, 4, mctlass::bfloat16_t> >, false, true, false, false, "
    "false, false, true, false, false, 1, false, (Arch)1000>"
    "(mcFlashAttn::Flash_fwd_params, int, int)")
_METAX_372_Q128_SPLIT_PREFILL_RUNTIME = (
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 128, 64, "
    "4, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, "
    "128, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, "
    "false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, "
    "int, int)")
_METAX_372_Q128_PREFILL_ALGORITHM = (
    "observed_metax_flash_attention_trait_direct:prefill_q128_full_tile")
_METAX_372_Q128_SPLIT_PREFILL_ALGORITHM = (
    "observed_metax_flash_attention_trait_direct:"
    "prefill_q128_split_full_tile")


def _metax_372_static_flash_trait(runtime_name: str, metrics: Mapping,
                                  shape: Mapping[str, int], hardware):
    """Return a version-bound C500 ``(D, BlockM, BlockN, warps)`` trait.

    The MACA 3.7.2 flash-attn 2.6.3 candidate registry names these exact five
    BF16 tuples as ``hdim_qk/hdim_v/block_m/block_n/k_nwarps``.  Runtime
    dispatch, tail arithmetic and CTA phase order are separate facts and are
    deliberately not inferred here.
    """

    trait = parse_flash_attention_trait(runtime_name)
    if trait is None or "flash_fwd_splitkv_mla_kernel" in runtime_name.lower():
        return None
    if hardware.hardware_id != "metax_c500":
        raise ValueError(
            "MetaX FlashAttention trait metadata requires metax_c500")
    if trait not in _METAX_372_STATIC_FLASH_TRAITS:
        raise ValueError(
            "MetaX MACA 3.7.2 FlashAttention runtime exposed an unregistered "
            f"static trait: {trait}")
    head_dim, block_m, block_n, warps = trait
    if head_dim != shape["head_dim"]:
        raise ValueError(
            "MetaX FlashAttention trait head dimension disagrees with the "
            f"workload: trait={head_dim}, shape={shape['head_dim']}")
    expected_block_size = warps * 64
    if metrics["block_size"] != expected_block_size:
        raise ValueError(
            "MetaX FlashAttention trait warp count disagrees with the "
            f"observed block: trait={warps}, block={metrics['block_size']}")
    return head_dim, block_m, block_n, warps


def _normalize_flash_attention(op_attributes: Mapping, kdata: Mapping,
                               hardware_id: str, hardware) -> KernelSpec:
    metrics = extract_ncu_metrics(kdata)
    runtime_name = metrics["kernel_name"]
    query_length = _positive_int(
        op_attributes, "Q_LEN", "FLASH_ATTN Attr")
    kv_length = _positive_int(
        op_attributes, "KV_LEN", "FLASH_ATTN Attr")
    causal = op_attributes["causal"]
    kv_layout = _non_empty_string(
        op_attributes, "kv_layout", "FLASH_ATTN Attr")
    if hardware_id == "metax_c500" and kv_layout == "mla":
        # The pinned runner passes causal=True to flash_mla_with_kvcache even
        # though all 16 historical case labels recorded C0/False.  Decode Q=1
        # has the same attended-element count under bottom-right masking, but
        # the normalized execution contract must carry the fact actually
        # supplied to the host ABI.  Reject a future incompatible recording
        # instead of silently accepting two meanings.
        if causal is not False:
            raise ValueError(
                "canonical MetaX MLA observation must retain its historical "
                "causal=False label before runner-fact normalization")
        causal = True
    kv_distribution = _non_empty_string(
        op_attributes, "kv_dist", "FLASH_ATTN Attr")
    batch = _positive_int(op_attributes, "B", "FLASH_ATTN Attr")
    query_lengths = _positive_int_sequence(
        op_attributes, "Q_LENS", "FLASH_ATTN Attr", batch,
        query_length)
    kv_lengths = _positive_int_sequence(
        op_attributes, "KV_LENS", "FLASH_ATTN Attr", batch,
        kv_length)
    if any(request_query > request_kv
           for request_query, request_kv in zip(query_lengths, kv_lengths)):
        raise ValueError(
            "FLASH_ATTN each request requires Q_LENS <= KV_LENS: "
            f"Q_LENS={query_lengths}, KV_LENS={kv_lengths}")
    requested_query_length = _positive_int(
        op_attributes, "Q_REQUESTED", "FLASH_ATTN Attr")
    requested_kv_length = _positive_int(
        op_attributes, "KV_REQUESTED", "FLASH_ATTN Attr")
    if requested_kv_length != _positive_int(
            op_attributes, "S", "FLASH_ATTN Attr"):
        raise ValueError(
            "FLASH_ATTN KV_REQUESTED must equal the case S bucket")
    if requested_kv_length < kv_length:
        raise ValueError(
            "FLASH_ATTN KV_REQUESTED cannot be smaller than KV_LEN")
    if kv_layout == "chunked_prefill":
        if sum(query_lengths) > requested_query_length:
            raise ValueError(
                "FLASH_ATTN chunked query lengths exceed Q_REQUESTED")
    elif requested_query_length != query_length:
        raise ValueError(
            "non-chunked FLASH_ATTN Q_REQUESTED must equal Q_LEN")
    split_policy = _non_empty_string(
        op_attributes, "split_policy", "FLASH_ATTN Attr")
    if split_policy not in ("forced", "serving"):
        raise ValueError(
            f"unsupported FLASH_ATTN split_policy: {split_policy!r}")
    shape = {
        "batch": batch,
        "q_len": query_length,
        "kv_len": kv_length,
        "head_dim": _positive_int(op_attributes, "D", "FLASH_ATTN Attr"),
        "num_query_heads": _positive_int(
            op_attributes, "HQ", "FLASH_ATTN Attr"),
        "num_kv_heads": _positive_int(
            op_attributes, "HKV", "FLASH_ATTN Attr"),
    }
    copy_dtype = None
    copy_vector_width = None
    if "direct_copy_kernel_cuda" in runtime_name:
        upstream_bf16 = (
            runtime_name.startswith("void at::native::elementwise_kernel<")
            and "lambda(c10::BFloat16)" in runtime_name
            and "elementwise_kernel<(int)128, (int)4" in runtime_name
        )
        upstream_float32 = (
            runtime_name.startswith("void at::native::elementwise_kernel<")
            and "lambda(float)" in runtime_name
            and "elementwise_kernel<(int)128, (int)2" in runtime_name
        )
        metax_bf16 = (
            hardware_id == "metax_c500"
            and runtime_name.startswith((
                "void at::native::elementwise_kernel_3_1_"
                "transpose12_half_copy_64<",
                "void at::native::elementwise_kernel_4_1_"
                "transpose12_copy_64<",
            ))
            and "c10::BFloat16" in runtime_name
            and "lambda(c10::BFloat16)" in runtime_name
        )
        if upstream_bf16 or metax_bf16:
            copy_dtype = "bf16"
            copy_vector_width = 4
            copy_elements = (
                sum(query_lengths) * shape["num_query_heads"]
                * shape["head_dim"])
        elif upstream_float32:
            copy_dtype = "float32"
            copy_vector_width = 2
            copy_elements = sum(query_lengths) * shape["num_query_heads"]
        else:
            raise ValueError(
                "FLASH_ATTN direct-copy kernel has an unknown or unsupported "
                "dtype/vector "
                f"template: {runtime_name!r}")
        if hardware_id == "metax_c500":
            allowed_copy_layouts = (
                "contiguous", "paged", "paged_varlen")
        elif hardware_id == "nvidia_a800":
            allowed_copy_layouts = ("paged", "paged_varlen")
        else:
            raise ValueError(
                "FLASH_ATTN direct-copy observation is unsupported on "
                f"hardware {hardware_id!r}")
        if query_length != 1 or kv_layout not in allowed_copy_layouts:
            raise ValueError(
                "FLASH_ATTN direct-copy observation has an unsupported "
                f"decode layout: hardware={hardware_id!r}, "
                f"kv_layout={kv_layout!r}")
        if metax_bf16:
            # MetaX's transpose12 copy assigns one workgroup to each 64-value
            # slice of the outer (Q,D) plane and loops over heads inside the
            # workgroup.  It is therefore not the upstream flat elementwise
            # launch even though both kernels copy the same semantic tensor.
            expected_grid_x = math.ceil(
                sum(query_lengths) * shape["head_dim"] / 64)
        else:
            expected_grid_x = math.ceil(
                copy_elements / (metrics["block_size"] * copy_vector_width))
        if ((metrics["grid_x"], metrics["grid_y"], metrics["grid_z"])
                != (expected_grid_x, 1, 1)):
            raise ValueError(
                "FLASH_ATTN direct-copy grid disagrees with semantic element "
                f"count: grid={(metrics['grid_x'], metrics['grid_y'], metrics['grid_z'])}, "
                f"expected={(expected_grid_x, 1, 1)}")
        copy_shape = {"element_count": copy_elements}
        copy_launch = observed_launch_config(
            metrics, KernelKind.VECTOR, copy_shape,
            algorithm="observed_flash_attention_copy", variant="elementwise")
        if hardware_id == "nvidia_a800":
            if copy_dtype == "bf16":
                tensor_iterator_dimensions = 4
            elif copy_dtype == "float32":
                tensor_iterator_dimensions = 3
            else:
                raise ValueError(
                    "A800 FLASH_ATTN direct-copy dtype is unsupported: "
                    f"{copy_dtype!r}")
            copy_launch = dataclasses.replace(
                copy_launch,
                parameters={
                    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER: (
                        float(tensor_iterator_dimensions)),
                },
            )
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape=copy_shape,
            dtype=copy_dtype,
            variant="elementwise",
            launch=copy_launch,
            attributes={
                "reads": 1, "writes": 1, "operation": "copy"},
        )

    if hardware_id == "metax_c500" and kv_layout == "mla":
        lowered_runtime_name = runtime_name.lower()
        if "get_mla_metadata_kernel" in lowered_runtime_name:
            if query_length != 1:
                raise ValueError(
                    "MetaX FlashMLA metadata setup supports decode Q=1 only")
            scheduler_partitions = METAX_MLA_SCHEDULER_PARTITIONS
            scheduler_metadata_ints = scheduler_partitions * 8
            split_metadata_ints = batch + 1
            scheduler_work = derive_mla_scheduler_work(kv_lengths)
            expected_dynamic_smem = 4 * (5 * batch + 1)
            launch_header = kdata["Sections"]["Launch Statistics"]["Header"]
            dynamic_smem = launch_header[
                "Dynamic Shared Memory Per Block [byte/block]"]
            if isinstance(dynamic_smem, bool) or not isinstance(
                    dynamic_smem, int):
                raise TypeError(
                    "MetaX FlashMLA metadata dynamic shared memory must be "
                    "an integer")
            actual_grid = (
                metrics["grid_x"], metrics["grid_y"], metrics["grid_z"])
            if actual_grid != (1, 1, 1):
                raise ValueError(
                    "MetaX FlashMLA metadata grid must be (1, 1, 1), got "
                    f"{actual_grid}")
            if metrics["block_size"] != 32:
                raise ValueError(
                    "MetaX FlashMLA metadata block size must be 32, got "
                    f"{metrics['block_size']}")
            if dynamic_smem != expected_dynamic_smem:
                raise ValueError(
                    "MetaX FlashMLA metadata dynamic shared memory disagrees "
                    f"with 4 * (5 * B + 1): actual={dynamic_smem}, "
                    f"expected={expected_dynamic_smem}")
            metadata_shape = {
                "batch": batch,
                "kv_block_count": scheduler_work.kv_block_count,
                "scheduler_partitions": scheduler_partitions,
                "scheduler_metadata_ints": scheduler_metadata_ints,
                "split_metadata_ints": split_metadata_ints,
                "scheduler_segment_count": scheduler_work.segment_count,
                "scheduler_worker_run_count": (
                    scheduler_work.segment_count
                    - scheduler_work.same_worker_transitions),
            }
            metadata_launch = observed_launch_config(
                metrics, KernelKind.VECTOR, metadata_shape,
                algorithm="observed_flash_mla_metadata",
                variant="flash_mla_metadata")
            return KernelSpec(
                kind=KernelKind.VECTOR,
                hardware_id=hardware_id,
                shape=metadata_shape,
                dtype="int32",
                variant="flash_mla_metadata",
                launch=metadata_launch,
            )
        if ("fillfunctor<bool>" in lowered_runtime_name
                or "fill_functor<bool>" in lowered_runtime_name):
            if query_length != 1:
                raise ValueError(
                    "MetaX FlashMLA bool fill supports decode Q=1 only")
            fill_shape = {"element_count": sum(query_lengths)}
            fill_launch = observed_launch_config(
                metrics, KernelKind.VECTOR, fill_shape,
                algorithm="observed_flash_mla_bool_fill",
                variant="elementwise")
            return KernelSpec(
                kind=KernelKind.VECTOR,
                hardware_id=hardware_id,
                shape=fill_shape,
                dtype="uint8",
                variant="elementwise",
                launch=fill_launch,
                attributes={
                    "reads": 0, "writes": 1, "operation": "fill"},
            )

    if hardware_id == "nvidia_h200":
        return _normalize_h200_flash_attention(
            op_attributes, metrics, shape, query_lengths, kv_lengths,
            causal, kv_layout, kv_distribution, split_policy, hardware)
    non_tensor = is_non_tensor(
        "FLASHATTN", runtime_name, ascend_op_type=metrics["op_type"])
    if non_tensor:
        variant = "combine"
    elif query_length == 1:
        variant = "decode"
    elif query_length > 1:
        variant = "prefill"
    else:
        raise ValueError("FLASH_ATTN query length must be positive")
    launch = observed_launch_config(
        metrics, KernelKind.FLASH_ATTN, shape,
        algorithm="observed_flash_attention", variant=variant)
    metax_static_trait = None
    metax_padded_tensor_work = False
    metax_tile16_decode_counter_program = False
    if (hardware_id == "metax_c500" and not non_tensor
            and kv_layout != "mla"):
        metax_static_trait = _metax_372_static_flash_trait(
            runtime_name, metrics, shape, hardware)
    if hardware_id == "metax_c500" and kv_layout == "mla" and non_tensor:
        # The pinned FlashMLA invocation emits a dedicated combine launch.
        # Give that observed implementation fact its own canonical identity so
        # the predictor can preserve the unresolved MACA partial layout
        # without branching on hardware ID or changing ordinary/A800 combine
        # semantics.
        launch = dataclasses.replace(
            launch, algorithm="observed_metax_flash_mla_combine")
    if hardware_id == "metax_c500" and not non_tensor:
        lowered_runtime_name = runtime_name.lower()
        runtime_core_name = (
            lowered_runtime_name[len("void "):]
            if lowered_runtime_name.startswith("void ")
            else lowered_runtime_name
        )
        # Three MACA 3.7.2 layer-1 observations bind this non-causal direct
        # specialization across GQA ratios 5/8 and non-GQA.  Every fully
        # covered workgroup executes 4,096 MMA instructions; the package
        # registry's BlockM/BlockN meaning therefore closes the complete
        # 128x64 QK/PV program for each full KV tile, including the Q=1 row
        # tail.  The 32-workgroup non-GQA capture covered 31 workgroups, so
        # its raw total is not treated as a full-grid counter: grid scaling
        # follows the identical static CTA program.  A partial final KV tile
        # remains unproved and keeps the semantic lower bound.
        metax_padded_tensor_work = (
            metax_static_trait == (128, 128, 64, 4)
            and variant == "decode"
            and causal is False
            and all(length % 64 == 0 for length in kv_lengths)
            and runtime_core_name.startswith("flash_fwd_kernel<")
        )
        observed_tile = (
            None if launch.tiling is None
            else launch.tiling.tile_shape)
        tile16_worker = (
            runtime_name.startswith((
                "void flash_fwd_kernel<",
                "void flash_fwd_splitkv_kernel<",
            ))
            and metrics["block_size"] == 64
            and observed_tile is not None
            and {"q", "kv"}.issubset(observed_tile)
            and (observed_tile["q"], observed_tile["kv"]) == (16, 16)
        )
        if tile16_worker:
            logical_q_tile = (
                32 if variant == "prefill" and batch == 1 else 16)
            tile_shape = dict(observed_tile)
            tile_shape.update({
                "q": logical_q_tile,
                "query_heads_per_workgroup": 4,
            })
            launch = dataclasses.replace(
                launch,
                tiling=dataclasses.replace(
                    launch.tiling, tile_shape=tile_shape),
            )
            metax_tile16_decode_counter_program = (
                metax_static_trait in {
                    (64, 16, 16, 1),
                    (128, 16, 16, 1),
                }
                and variant == "decode"
                and causal is False)
            if metax_tile16_decode_counter_program:
                # Exact layer-1 counters close eight (D64) or sixteen (D128)
                # MMA instructions for every scheduled BlockN step, including
                # ragged final steps.
                metax_padded_tensor_work = True
            if variant == "prefill" and kv_layout == "paged":
                algorithm = (
                    "observed_metax_flash_attention_grouped_direct"
                    if batch == 1 else
                    "observed_metax_flash_attention_persistent_queue")
                launch = dataclasses.replace(launch, algorithm=algorithm)
            elif (variant == "prefill"
                  and kv_layout == "chunked_prefill"):
                launch = dataclasses.replace(
                    launch,
                    algorithm=(
                        "observed_metax_flash_attention_packed_varlen_queue"))
            elif (variant == "decode"
                  and kv_layout == "paged_varlen"):
                recorded_splits = _positive_int(
                    op_attributes, "SP", "FLASH_ATTN Attr")
                request_major = (
                    metrics["grid_x"] == shape["num_kv_heads"]
                    and metrics["grid_y"] >= batch
                    and metrics["grid_z"] == recorded_splits)
                if not request_major:
                    launch = dataclasses.replace(
                        launch,
                        algorithm=(
                            "observed_metax_flash_attention_"
                            "packed_varlen_queue"))
        elif "flash_fwd_splitkv_mla_kernel" in lowered_runtime_name:
            launch = dataclasses.replace(
                launch,
                algorithm=(
                    "observed_metax_flash_attention_persistent_queue"))
        elif (variant == "prefill"
              and kv_layout == "chunked_prefill"
              and runtime_core_name.startswith(
                  "flash_fwd_splitkv_kernel<")
              and observed_tile is not None
              and {"q", "kv"}.issubset(observed_tile)
              and (observed_tile["q"], observed_tile["kv"]) == (64, 64)
              and metrics["block_size"] == 256):
            tile_shape = dict(observed_tile)
            tile_shape["query_heads_per_workgroup"] = 1
            launch = dataclasses.replace(
                launch,
                algorithm=(
                    "observed_metax_flash_attention_packed_varlen_queue"),
                tiling=dataclasses.replace(
                    launch.tiling, tile_shape=tile_shape),
            )
    if hardware_id == "ascend_910b":
        if op_attributes["SP"] is not None:
            raise ValueError(
                "Ascend 910B FLASH_ATTN observation must record SP=None")
        if op_attributes["SP_REQUESTED"] is not None:
            raise ValueError(
                "Ascend 910B FLASH_ATTN observation must record "
                "SP_REQUESTED=None")
        # CANN exposes one fused attention operator rather than a public
        # split-KV + combine launch contract. The internal TilingSpec uses one
        # logical partition so the shared predictor contract remains complete;
        # this is not a dataset SP field or a user-controlled CANN input.
        splits = 1
        production_launch = select_launch_config(
            hardware_id,
            KernelKind.FLASH_ATTN,
            shape,
            hardware.kernel_dtype,
            variant,
            {
                "is_decode": variant == "decode",
                "causal": causal,
                "kv_layout": kv_layout,
                "kv_dist": kv_distribution,
                "q_lens": query_lengths,
                "kv_lens": kv_lengths,
                "launch_policy": "injected_launch",
            },
            hardware=hardware,
        )
        if (launch.grid != production_launch.grid
                or launch.block != production_launch.block):
            raise ValueError(
                "Ascend 910B FLASH_ATTN observed launch disagrees with "
                "the exact production launch fact: "
                f"observed={launch.grid}/{launch.block}, "
                f"production={production_launch.grid}/"
                f"{production_launch.block}")
        algorithm_prefix = "cann_flash_attention_score:"
        if not production_launch.algorithm.startswith(algorithm_prefix):
            raise ValueError(
                "Ascend 910B FLASH_ATTN production algorithm is invalid")
        production_variant = production_launch.algorithm[
            len(algorithm_prefix):]
        expected_symbol_suffix = f"_{production_variant}_mix_aic"
        if not runtime_name.endswith(expected_symbol_suffix):
            raise ValueError(
                "Ascend 910B FLASH_ATTN observed runtime variant disagrees "
                "with the exact production launch fact")
        ascend_tiling = production_launch.tiling
    elif hardware_id in ("metax_c500", "nvidia_a800"):
        splits = _positive_int(op_attributes, "SP", "FLASH_ATTN Attr")
        requested_splits = _optional_positive_int(
            op_attributes, "SP_REQUESTED", "FLASH_ATTN Attr")
        metadata_selected_mla = (
            hardware_id == "metax_c500"
            and kv_layout == "mla"
            and split_policy == "serving")
        if requested_splits is None and not metadata_selected_mla:
            raise ValueError(
                f"{hardware_id} FLASH_ATTN requires SP_REQUESTED")
        if (split_policy == "forced"
                and requested_splits is not None
                and requested_splits != splits):
            raise ValueError(
                "forced FLASH_ATTN SP must equal SP_REQUESTED")
        if (hardware_id == "nvidia_a800" and not non_tensor
                and kv_layout != "mla"):
            # The executed split count is a versioned discrete launch fact,
            # not a duration-derived parameter.  Re-run the canonical selector
            # in forced-SP mode to bind the observation to the pinned FA2
            # source family and its exact semantic tile.  Keep the profiler's
            # physical grid and resource fields below: packed/varlen launch
            # geometry is not reconstructed by the generic selector.
            production_launch = select_launch_config(
                hardware_id,
                KernelKind.FLASH_ATTN,
                shape,
                hardware.kernel_dtype,
                variant,
                {
                    "is_decode": variant == "decode",
                    "causal": causal,
                    "kv_layout": kv_layout,
                    "kv_dist": kv_distribution,
                    "q_lens": query_lengths,
                    "kv_lens": kv_lengths,
                    "launch_policy": "injected_launch",
                    "split_policy": "forced",
                    "num_kv_splits": splits,
                },
                hardware=hardware,
            )
            production_tiling = production_launch.tiling
            observed_tiling = launch.tiling
            if production_tiling is None or observed_tiling is None:
                raise ValueError(
                    "A800 FA2 main launch requires source and observed tiles")
            semantic_axes = ("q", "kv", "head")
            production_tile = tuple(
                production_tiling.tile_shape[name]
                for name in semantic_axes)
            observed_tile = tuple(
                observed_tiling.tile_shape[name]
                for name in semantic_axes)
            if observed_tile != production_tile:
                raise ValueError(
                    "A800 FA2 observed tile disagrees with the pinned source "
                    f"launch: observed={observed_tile}, "
                    f"production={production_tile}")
            if launch.block != production_launch.block:
                raise ValueError(
                    "A800 FA2 observed block disagrees with the pinned source "
                    f"launch: observed={launch.block}, "
                    f"production={production_launch.block}")
            lowered_runtime_name = runtime_name.lower()
            if production_launch.algorithm == "flash_attn_dense":
                expected_symbol = (
                    "flash_fwd_kernel<" in lowered_runtime_name
                    and "flash_fwd_splitkv_kernel<" not in lowered_runtime_name)
            elif production_launch.algorithm in (
                    "flash_attn_splitkv", "vllm_flash_attn_splitkv"):
                expected_symbol = (
                    "flash_fwd_splitkv_kernel<" in lowered_runtime_name)
            else:
                raise ValueError(
                    "A800 FA2 selector returned an unsupported source family: "
                    f"{production_launch.algorithm!r}")
            if not expected_symbol:
                raise ValueError(
                    "A800 FA2 runtime symbol disagrees with the pinned source "
                    f"family {production_launch.algorithm!r}: "
                    f"{runtime_name!r}")
            launch = dataclasses.replace(
                launch,
                algorithm=production_launch.algorithm,
                tiling=production_tiling,
                parameters=production_launch.parameters,
            )
        elif (hardware_id == "nvidia_a800" and non_tensor
              and kv_layout != "mla"):
            production_launch = select_launch_config(
                hardware_id,
                KernelKind.FLASH_ATTN,
                shape,
                hardware.kernel_dtype,
                "combine",
                {
                    "launch_policy": "injected_launch",
                    "num_kv_splits": splits,
                    "kv_layout": kv_layout,
                },
                hardware=hardware,
            )
            if (launch.grid != production_launch.grid
                    or launch.block != production_launch.block):
                raise ValueError(
                    "A800 FA2 combine observation disagrees with the pinned "
                    "source launch: "
                    f"observed={launch.grid}/{launch.block}, "
                    f"production={production_launch.grid}/"
                    f"{production_launch.block}")
            if "flash_fwd_splitkv_combine_kernel<" not in runtime_name:
                raise ValueError(
                    "A800 FA2 combine runtime symbol disagrees with the "
                    f"pinned source family: {runtime_name!r}")
            template = re.search(
                r">>,\s*\(int\)(\d+),\s*\(int\)(\d+),\s*\(bool\)",
                runtime_name,
            )
            if template is None:
                raise ValueError(
                    "A800 FA2 combine runtime symbol does not expose "
                    "kBlockM/Log_max_splits")
            observed_block_m = int(template.group(1))
            observed_log_max_splits = int(template.group(2))
            expected_block_m = (
                production_launch.block_size * 4 // shape["head_dim"])
            expected_log_max_splits = (splits - 1).bit_length()
            if (observed_block_m, observed_log_max_splits) != (
                    expected_block_m, expected_log_max_splits):
                raise ValueError(
                    "A800 FA2 combine template disagrees with the pinned "
                    "source dispatch: "
                    f"observed={(observed_block_m, observed_log_max_splits)}, "
                    f"expected={(expected_block_m, expected_log_max_splits)}")
            launch = dataclasses.replace(
                launch,
                algorithm=production_launch.algorithm,
                tiling=production_launch.tiling,
                parameters=production_launch.parameters,
            )
    else:
        raise ValueError(
            f"unsupported FLASH_ATTN observation hardware: {hardware_id!r}")
    grouped_mla_stage1 = runtime_name == "_fwd_grouped_kernel_stage1"
    grouped_mla_stage2 = (
        runtime_name == "_fwd_kernel_stage2" and kv_layout == "mla")
    if grouped_mla_stage1:
        if (kv_layout != "mla" or query_length != 1
                or shape["head_dim"] != 512):
            raise ValueError(
                "_fwd_grouped_kernel_stage1 requires MLA decode with D=512")
        head_tile = 16
        kv_tile = 32
        expected_grid = (
            batch, math.ceil(shape["num_query_heads"] / head_tile), splits)
        actual_grid = (
            metrics["grid_x"], metrics["grid_y"], metrics["grid_z"])
        if actual_grid != expected_grid:
            raise ValueError(
                "grouped MLA launch grid disagrees with B/head-groups/SP: "
                f"actual={actual_grid}, expected={expected_grid}")
        # vLLM v0.11's Ampere fallback fixes BLOCK_H=16, BLOCK_N=32 and
        # qk_rope_dim=64.  These are structured launch facts; BLOCK_H is a
        # query-head tile, never a Q-row tile.
        tiling = TilingSpec(
            tile_shape={
                "query_heads": head_tile,
                "kv": kv_tile,
                "head": shape["head_dim"],
                "qk_head": shape["head_dim"] + 64,
                "value_head": shape["head_dim"],
            },
            num_kv_splits=splits,
        )
        launch = dataclasses.replace(
            launch,
            algorithm="triton_grouped_mla_stage1",
            tiling=tiling,
            parameters={
                "unpadded_flops": 0.0,
                # Grouped MLA is a Triton kernel, not FA2's SM80
                # m16n8k16 + gemm_rs implementation.
                "mma_smem_bytes_per_flop": 0.0,
                "split_scratch_dram_bytes_per_element": 0.0,
                "page_table_bytes_per_kv_panel": 0.0,
            },
        )
    elif grouped_mla_stage2:
        if (variant != "combine" or query_length != 1
                or shape["head_dim"] != 512
                or shape["num_query_heads"] != 64
                or shape["num_kv_heads"] != 1
                or splits <= 1):
            raise ValueError(
                "_fwd_kernel_stage2 requires split grouped-MLA decode")
        expected_grid = (batch, shape["num_query_heads"], 1)
        if launch.grid != expected_grid or launch.block != (128, 1, 1):
            raise ValueError(
                "grouped MLA stage2 launch disagrees with its pinned source: "
                f"actual={launch.grid}/{launch.block}, "
                f"expected={expected_grid}/(128, 1, 1)")
        launch = dataclasses.replace(
            launch,
            algorithm="triton_grouped_mla_stage2",
            tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
            parameters={
                "unpadded_flops": 1.0,
                "mma_smem_bytes_per_flop": 0.0,
                # The pinned stage-2 program reads FP32 Mid_O scratch from a
                # preceding launch.  Its NCU L2/DRAM counters close the full
                # four-byte scratch read for every canonical batch size,
                # including footprints smaller than L2, so this is an
                # inter-kernel DRAM handoff rather than an L2-resident input.
                "split_scratch_dram_bytes_per_element": 4.0,
                "page_table_bytes_per_kv_panel": 0.0,
            },
        )
    # MACA 3.7.2's versioned candidate registry binds the five standard trait
    # tuples to BlockM/BlockN.  Only direct launches may publish that static
    # tile here: queue classifiers still lack a semantic-job-to-worker map and
    # therefore retain an empty production tile.  A tile alone does not prove
    # tail arithmetic, traffic padding or a timed CTA worker schedule.
    metax_direct_trait_tile = (
        hardware_id == "metax_c500"
        and metax_static_trait is not None
        and launch.algorithm.startswith("observed_flash_attention:")
    )
    if metax_direct_trait_tile:
        launch = dataclasses.replace(
            launch,
            algorithm=(
                "observed_metax_flash_attention_trait_direct:"
                f"{variant}"),
        )
    metax_q64_prefill_runtime = (
        (metax_static_trait == (128, 64, 64, 4)
         and runtime_name == _METAX_372_Q64_PREFILL_RUNTIME)
        or (metax_static_trait == (64, 64, 64, 4)
            and runtime_name == _METAX_372_D64_Q64_PREFILL_RUNTIME)
    )
    metax_q64_prefill_source_program = (
        metax_direct_trait_tile
        and metax_q64_prefill_runtime
        and variant == "prefill"
        and causal is True
        and kv_layout in ("paged", "paged_varlen")
        and splits == 1
        and query_length % 64 == 0
        and all(
            query == key_value and query % 64 == 0
            for query, key_value in zip(query_lengths, kv_lengths))
        and launch.grid == (
            shape["num_query_heads"], query_length // 64, batch)
        and launch.block == (256, 1, 1)
    )
    if metax_q64_prefill_source_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q64_PREFILL_ALGORITHM)
        metax_padded_tensor_work = True
    metax_q64_packed_counter_program = (
        metax_q64_prefill_runtime
        and variant == "prefill"
        and causal is True
        and kv_layout == "chunked_prefill"
        and splits == 1
        and launch.block == (256, 1, 1)
        and launch.algorithm
        == "observed_metax_flash_attention_packed_varlen_queue")
    if metax_q64_packed_counter_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q64_PACKED_PREFILL_ALGORITHM)
        metax_padded_tensor_work = True
    metax_q64_decode_source_program = (
        metax_direct_trait_tile
        and runtime_name == _METAX_372_Q64_DECODE_RUNTIME
        and metax_static_trait == (128, 64, 64, 4)
        and variant == "decode"
        and causal is False
        and kv_layout == "contiguous"
        and splits == 1
        and all(query == 1 for query in query_lengths)
        and all(length % 64 == 0 for length in kv_lengths)
        and launch.grid == (shape["num_kv_heads"], batch, 1)
        and launch.block == (256, 1, 1)
    )
    if metax_q64_decode_source_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q64_DECODE_ALGORITHM)
        # The exact-symbol layer-1 counter reports 2,048 MMA instructions per
        # CTA for the eight-panel S512 program, independent of GQA ratio.  The
        # compiler therefore executes the complete BlockM64 Tensor program;
        # only global Q/O accesses are predicated to live GQA rows.
        metax_padded_tensor_work = True
    metax_q128_prefill_source_program = (
        metax_direct_trait_tile
        and runtime_name == _METAX_372_Q128_PREFILL_RUNTIME
        and metax_static_trait == (128, 128, 64, 4)
        and variant == "prefill"
        and causal is True
        and kv_layout == "contiguous"
        and splits == 1
        and query_length % 128 == 0
        and all(
            query == key_value and query % 128 == 0
            for query, key_value in zip(query_lengths, kv_lengths))
        and launch.grid == (
            shape["num_query_heads"], batch, query_length // 128)
        and launch.block == (256, 1, 1)
    )
    if metax_q128_prefill_source_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q128_PREFILL_ALGORITHM)
        metax_padded_tensor_work = True
    metax_q128_split_prefill_source_program = (
        metax_direct_trait_tile
        and runtime_name == _METAX_372_Q128_SPLIT_PREFILL_RUNTIME
        and metax_static_trait == (128, 128, 64, 4)
        and variant == "prefill"
        and causal is True
        and kv_layout in ("paged", "paged_varlen", "chunked_prefill")
        and splits == 1
        and query_length % 128 == 0
        and all(
            query == query_length
            and query % 128 == 0
            and key_value >= query
            and key_value % 64 == 0
            for query, key_value in zip(query_lengths, kv_lengths))
        and launch.grid == (
            shape["num_query_heads"], query_length // 128, batch)
        and launch.block == (256, 1, 1)
    )
    if metax_q128_split_prefill_source_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q128_SPLIT_PREFILL_ALGORITHM)
        metax_padded_tensor_work = True
    has_attention_tile = (
        launch.tiling is not None
        and {"q", "kv"}.issubset(launch.tiling.tile_shape)
        and (hardware_id != "metax_c500"
             or metax_direct_trait_tile
             or metax_tile16_decode_counter_program
             or metax_q64_packed_counter_program)
    )
    if metax_direct_trait_tile:
        expected_tile = metax_static_trait[1:3]
        observed_tile = tuple(
            int(launch.tiling.tile_shape[name]) for name in ("q", "kv"))
        if observed_tile != expected_tile:
            raise ValueError(
                "MetaX MACA 3.7.2 static trait disagrees with the normalized "
                f"tile: trait={expected_tile}, tile={observed_tile}")
    if grouped_mla_stage1 or grouped_mla_stage2:
        pass
    elif hardware_id == "ascend_910b":
        if ascend_tiling is None:
            raise ValueError(
                "Ascend 910B FLASH_ATTN production launch requires tiling")
        tiling = dataclasses.replace(
            ascend_tiling, num_kv_splits=splits)
        launch = dataclasses.replace(
            launch, algorithm=production_launch.algorithm)
    elif has_attention_tile:
        tiling = dataclasses.replace(
            launch.tiling, num_kv_splits=splits)
    else:
        tiling = TilingSpec(
            tile_shape={}, num_kv_splits=splits)
    if not grouped_mla_stage1 and hardware_id == "ascend_910b":
        # The exact CANN launch package owns both the discrete variant and any
        # source-reflected worker schedule.  The profiler launch above is used
        # only to prove that its physical grid/block and runtime symbol agree
        # with that package.  Replacing these parameters with the generic
        # three-field observation schema would silently discard the canonical
        # source schedule from single-kernel evaluation.
        launch = dataclasses.replace(
            launch,
            tiling=tiling,
            parameters=production_launch.parameters,
            source_program=production_launch.source_program,
        )
    elif not (grouped_mla_stage1 or grouped_mla_stage2):
        if hardware_id == "nvidia_a800" and has_attention_tile:
            mma_smem_bytes_per_flop = 0.125
        elif hardware_id in ("nvidia_a800", "metax_c500"):
            mma_smem_bytes_per_flop = 0.0
        else:
            raise AssertionError(
                "FLASH_ATTN observation hardware was not normalized")
        launch = dataclasses.replace(
            launch,
            tiling=tiling,
            parameters={
                "unpadded_flops": (
                    0.0
                    if (hardware_id == "nvidia_a800" and has_attention_tile)
                    or metax_padded_tensor_work
                    else 1.0),
                "mma_smem_bytes_per_flop": mma_smem_bytes_per_flop,
                # Standard paged split main/combine families materialize an
                # FP32 scratch round trip.  A800's source-bound paged path and
                # C500's retained counters both expose that global-memory
                # traffic; MLA does not close uniformly.
                "split_scratch_dram_bytes_per_element": (
                    4.0
                    if (splits > 1 and (
                        (hardware_id == "metax_c500" and kv_layout != "mla")
                        or hardware_id == "nvidia_a800"))
                    else 0.0),
                "page_table_bytes_per_kv_panel": (
                    4.0 if (hardware_id == "nvidia_a800" and splits > 1
                            and kv_layout in ("paged", "paged_varlen")
                            and variant not in ("combine", "split_combine"))
                    else 0.0),
            },
        )
    if not isinstance(causal, bool):
        raise ValueError("FLASH_ATTN causal must be boolean")
    attributes = {
        "causal": causal,
        "kv_layout": kv_layout,
        "kv_dist": kv_distribution,
        "q_lens": query_lengths,
        "kv_lens": kv_lengths,
        "fa_ksteps": None,
        "fa_eff_mfrac": 1.0,
    }
    if "FA_KSTEPS" in op_attributes:
        attributes["fa_ksteps"] = _positive_int(
            op_attributes, "FA_KSTEPS", "FLASH_ATTN Attr")
    if "FA_EFF_MFRAC" in op_attributes:
        attributes["fa_eff_mfrac"] = _finite_number(
            op_attributes, "FA_EFF_MFRAC", "FLASH_ATTN Attr")
    return KernelSpec(
        kind=KernelKind.FLASH_ATTN,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant=variant,
        launch=launch,
        attributes=attributes,
    )


# H200 kernel-dataset runner (scripts/kernel_dataset/nvidia_h200/
# flash_attn_runner.py, MLA_ROPE_DIM): the absorbed MLA call passes q_pe /
# k_pe with the DeepSeek rope head 64 and q_nope / kv_c with the latent D.
_H200_MLA_ROPE_DIM = 64
_H200_FA3_MAIN_SYMBOL = "FlashAttnFwdSm90"
_H200_FA3_COMBINE_SYMBOL = "FlashAttnFwdCombine"


def _normalize_v030_gqa_observed(
        op_attributes, metrics, shape, attributes, observed, hardware):
    """Decode only versioned template/launch facts, never profiler duration."""

    name = metrics["kernel_name"]
    head_dim = shape["head_dim"]
    splits = _positive_int(op_attributes, "SP", "FLASH_ATTN Attr")
    if name.startswith("void flash::prepare_varlen_num_blocks_kernel<"):
        prepare = re.match(r"void flash::prepare_varlen_num_blocks_kernel<"
                           r"\(int\)(\d+), \(bool\)0>", name)
        launch = select_fa3_prepare_varlen_launch(shape["batch"])
        if (prepare is None or int(prepare[1]) * 32 != launch.block_size
                or observed.grid != launch.grid or observed.block != launch.block
                or metrics["dynamic_smem_per_block"] != 0):
            raise ValueError("H200 FA3 prepare launch differs from source")
        return KernelSpec(
            kind=KernelKind.VECTOR, hardware_id="nvidia_h200",
            shape={"batch": shape["batch"]}, dtype="int32",
            variant="fa3_prepare_varlen", launch=launch,
            attributes={"num_splits_static": splits})
    if name.startswith("void at::native::vectorized_elementwise_kernel<(int)4, "
                       "at::native::FillFunctor<int>, "):
        fill_shape, fill_attributes = {"element_count": 1}, {
            "reads": 0, "writes": 1, "operation": "fill"}
        launch = select_launch_config(
            "nvidia_h200", KernelKind.VECTOR, fill_shape, "int32",
            "elementwise", fill_attributes, hardware)
        if (observed.grid != launch.grid or observed.block != launch.block
                or metrics["dynamic_smem_per_block"] != 0):
            raise ValueError("H200 FA3 semaphore Fill differs from source")
        return KernelSpec(
            kind=KernelKind.VECTOR, hardware_id="nvidia_h200", shape=fill_shape,
            dtype="int32", variant="elementwise", launch=launch,
            attributes=fill_attributes)
    if "FlashAttnFwdCombine<" in name:
        combine = re.search(
            r"FlashAttnFwdCombine<cute::tuple<cute::C<\(int\)(\d+)>, "
            r"cute::C<\(int\)(\d+)>>, \(int\)(\d+), \(int\)256, "
            r"\(int\)1, \(bool\)0, \(bool\)1, cutlass::bfloat16_t, "
            r"float, cutlass::arch::Sm90>", name)
        if combine is None:
            return None
        if op_attributes["FA3_SPLIT_COMBINE"] is not True:
            raise ValueError("H200 FA3 GQA combine source template differs")
        rows, columns, log_splits = (int(v) for v in combine.groups())
        if splits > 2 ** log_splits:
            raise ValueError("H200 FA3 GQA combine split capacity is too small")
        launch = dataclasses.replace(
            observed, algorithm=FA3_GQA_COMBINE_030_ALGORITHM,
            tiling=TilingSpec(tile_shape={"rows": rows, "value_head": columns},
                              num_kv_splits=splits),
            parameters={"unpadded_flops": 1.0, "mma_smem_bytes_per_flop": 0.0,
                        "split_scratch_dram_bytes_per_element": 4.0 if splits > 1 else 0.0,
                        "page_table_bytes_per_kv_panel": 0.0},
            execution_variant="combine")
        kernel = KernelSpec(
            kind=KernelKind.FLASH_ATTN, hardware_id="nvidia_h200", shape=shape,
            dtype=hardware.kernel_dtype, variant="combine", launch=launch,
            attributes=attributes)
        validate_v030_gqa_observed_launch(kernel, hardware)
        return kernel
    if _H200_FA3_MAIN_SYMBOL not in name:
        return None
    scheduler = re.search(
        r"flash::VarlenDynamicPersistentTileScheduler<"
        r"\(int\)(\d+), \(int\)(\d+), \(int\)(\d+), \(int\)(\d+), "
        r"\(bool\)([01]), \(bool\)1, \(bool\)1, \(bool\)([01]), "
        r"\(bool\)0, \(bool\)1>", name)
    if scheduler is None:
        return None
    block_m, block_n, mma_threads, producer_threads, split, lpt = (
        int(v) for v in scheduler.groups())
    causal = attributes["causal"] and shape["q_len"] > 1
    mainloop = re.search(
        r"cutlass::arch::Sm90, ((?:\(bool\)[01], ){12})"
        r"cutlass::bfloat16_t, \(int\)(\d+)>", name)
    expected_flags = (
        int(causal), 0, 0, 1,
        int(attributes["kv_layout"] in ("paged", "paged_varlen", "chunked_prefill")),
        0, 0, 1, 1, 1, split, 0)
    if (mainloop is None or tuple(int(value) for value in re.findall(
            r"\(bool\)([01])", mainloop[1])) != expected_flags):
        raise ValueError("H200 FA3 GQA mainloop flags differ from source")
    if (lpt != int(causal) or split != int(op_attributes["FA3_SPLIT_COMBINE"])
            or observed.block_size != mma_threads + math.ceil(producer_threads / 128) * 128
            or "CollectiveMainloopFwdSm90<(int)2," not in name
            or f"cute::C<(int){block_m}>, cute::C<(int){block_n}>, "
               f"cute::C<(int){head_dim}>>" not in name):
        raise ValueError("H200 FA3 GQA mainloop and scheduler disagree")
    ratio = shape["num_query_heads"] // shape["num_kv_heads"]
    dynamic = v030_gqa_dynamic_splits(
        attributes["q_lens"], attributes["kv_lens"], ratio=ratio,
        kv_heads=shape["num_kv_heads"], block_m=block_m, block_n=block_n,
        split_cap=splits, num_sms=int(hardware.num_compute_units))
    if dynamic != tuple(op_attributes["SP_PER_REQUEST"]):
        raise ValueError("H200 FA3 GQA per-request splits differ from source")
    variant = "decode" if shape["q_len"] == 1 else "prefill"
    launch = dataclasses.replace(
        observed, algorithm=(FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM if split
                             else FA3_GQA_OBSERVED_030_ALGORITHM),
        tiling=TilingSpec(tile_shape={"q": block_m, "kv": block_n,
                                     "head_dim": head_dim},
                          pipeline_depth=2, num_kv_splits=splits),
        parameters={"unpadded_flops": 0.0, "mma_smem_bytes_per_flop": 0.0,
                    "split_scratch_dram_bytes_per_element": 4.0 if splits > 1 else 0.0,
                    "page_table_bytes_per_kv_panel": (
                        4.0 if splits > 1 and attributes["kv_layout"] in ("paged", "paged_varlen") else 0.0)},
        execution_variant=variant)
    kernel = KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id="nvidia_h200", shape=shape,
        dtype=hardware.kernel_dtype, variant=variant, launch=launch,
        attributes=attributes)
    validate_v030_gqa_observed_launch(kernel, hardware)
    return kernel


def _normalize_h200_flash_attention(
        op_attributes: Mapping, metrics: Mapping, shape: Mapping,
        query_lengths, kv_lengths, causal, kv_layout: str,
        kv_distribution: str, split_policy: str, hardware) -> KernelSpec:
    """Bind one H200 FA3 launch to the source-derived FA3 package.

    Absorbed MLA decode and uniform fresh BF16 D128 GQA prefill have derived
    H200 templates.  Each launch is re-selected from
    the pinned source (the runner calls FA3 with ``num_splits=0``, i.e. the
    heuristic, eager: no full-CUDA-graph split bound) and must agree with the
    observed grid/block, the observed combine launch and the recorded
    per-request dynamic splits; no duration is read.  Other FA3 launches
    are normalized under an explicit unmodeled identity that every pricing
    method rejects.
    """

    runtime_name = metrics["kernel_name"]
    batch = shape["batch"]
    splits = _positive_int(op_attributes, "SP", "FLASH_ATTN Attr")
    observed = observed_launch_config(
        metrics, KernelKind.FLASH_ATTN, shape,
        algorithm="observed_flash_attention")
    is_main = _H200_FA3_MAIN_SYMBOL in runtime_name
    is_combine = _H200_FA3_COMBINE_SYMBOL in runtime_name
    common_attributes = {
        "causal": causal,
        "kv_layout": kv_layout,
        "kv_dist": kv_distribution,
        "q_lens": query_lengths,
        "kv_lens": kv_lengths,
    }
    if kv_layout != "mla":
        if is_combine:
            variant = "combine"
        elif shape["q_len"] == 1:
            variant = "decode"
        else:
            variant = "prefill"
        if (variant == "prefill" and split_policy == "serving"
                and splits == 1 and op_attributes["FA3_SPLIT_COMBINE"] is False
                and tuple(op_attributes["SP_PER_REQUEST"]) == (1,) * batch
                and shape["head_dim"] == 128 and causal is True
                and kv_layout in ("paged", "paged_varlen")
                and query_lengths == kv_lengths
                and query_lengths == (shape["q_len"],) * batch):
            source_attributes = {
                **common_attributes, "fa_ksteps": None,
                "fa_eff_mfrac": 1.0}
            try:
                selected = select_v030_gqa_fresh_prefill_launch(
                    shape, hardware.kernel_dtype,
                    {**source_attributes, "num_kv_splits": 1}, hardware)
            except NotImplementedError:
                selected = None
            if selected is not None:
                if is_main:
                    ratio = shape["num_query_heads"] // shape["num_kv_heads"]
                    if ("CollectiveMainloopFwdSm90<(int)2" not in runtime_name
                            or "VarlenDynamicPersistentTileScheduler<"
                            "(int)128, (int)128, (int)256, (int)128" not in runtime_name
                            or runtime_name.count(f"(int){ratio}>") < 2
                            or metrics["regs_per_thread"] != 168
                            or metrics["dynamic_smem_per_block"] != 166912
                            or observed.grid != selected.grid
                            or observed.block != selected.block):
                        raise ValueError("H200 FA3 GQA observed main differs from pinned source")
                    return KernelSpec(
                        kind=KernelKind.FLASH_ATTN,
                        hardware_id="nvidia_h200", shape=shape,
                        dtype=hardware.kernel_dtype, variant="prefill",
                        launch=selected, attributes=source_attributes)
                if runtime_name.startswith(
                        "void at::native::vectorized_elementwise_kernel<(int)4, "
                        "at::native::FillFunctor<int>, "):
                    fill_shape = {"element_count": 1}
                    fill_attributes = {"reads": 0, "writes": 1, "operation": "fill"}
                    fill_launch = select_launch_config(
                        "nvidia_h200", KernelKind.VECTOR, fill_shape, "int32",
                        "elementwise", fill_attributes, hardware)
                    if (observed.grid != fill_launch.grid
                            or observed.block != fill_launch.block
                            or metrics["regs_per_thread"] != 16
                            or metrics["dynamic_smem_per_block"] != 0):
                        raise ValueError("H200 FA3 GQA observed Fill differs from pinned source")
                    return KernelSpec(
                        kind=KernelKind.VECTOR,
                        hardware_id="nvidia_h200", shape=fill_shape,
                        dtype="int32", variant="elementwise",
                        launch=fill_launch, attributes=fill_attributes)
        source_attributes = {
            **common_attributes, "fa_ksteps": None, "fa_eff_mfrac": 1.0}
        source_kernel = _normalize_v030_gqa_observed(
            op_attributes, metrics, shape, source_attributes, observed, hardware)
        if source_kernel is not None:
            return source_kernel
        return KernelSpec(
            kind=KernelKind.FLASH_ATTN,
            hardware_id="nvidia_h200",
            shape=shape,
            dtype=hardware.kernel_dtype,
            variant=variant,
            launch=dataclasses.replace(
                observed,
                algorithm=FA3_UNMODELED_OBSERVATION_ALGORITHM,
                tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
                parameters={
                    "unpadded_flops": 1.0,
                    "mma_smem_bytes_per_flop": 0.0,
                    "split_scratch_dram_bytes_per_element": 0.0,
                    "page_table_bytes_per_kv_panel": 0.0,
                },
                execution_variant=variant,
            ),
            attributes={
                **common_attributes, "fa_ksteps": None, "fa_eff_mfrac": 1.0},
        )
    if not (is_main or is_combine):
        raise ValueError(
            "H200 FA3 MLA case contains a launch outside the FA3 main/"
            f"combine pair: {runtime_name!r}")
    # The runner calls FA3 MLA with causal=True regardless of the case label
    # (A800 canonical MLA cases are labelled C0); a one-token query executes
    # non-causally in FA3 either way, so only Q_LEN > 1 must carry True.
    if causal is not True and not (causal is False and shape["q_len"] == 1):
        raise ValueError(
            "H200 FA3 MLA multi-token observation must be causal")
    causal = True
    common_attributes["causal"] = True
    # The runner rejects non-uniform MLA lengths, so the executed KV
    # distribution is uniform whatever label the A800 case list carried
    # (its MLA cases are labelled ``skew``); bind the executed fact.
    if len(set(kv_lengths)) != 1 or len(set(query_lengths)) != 1:
        raise ValueError("H200 FA3 MLA observation requires uniform lengths")
    common_attributes["kv_dist"] = "uniform"
    if split_policy != "serving" or shape["num_kv_heads"] != 1:
        raise ValueError(
            "H200 FA3 MLA observation must be the one-latent-head serving "
            "call with FA3's own split heuristic")
    latent = shape["head_dim"]
    mla_shape = {
        "batch": batch,
        "q_len": shape["q_len"],
        "kv_len": shape["kv_len"],
        "num_query_heads": shape["num_query_heads"],
        "qk_head_dim": latent + _H200_MLA_ROPE_DIM,
        "v_head_dim": latent,
        "kv_lora_rank": latent,
    }
    mla_attributes = {
        **common_attributes,
        "full_cuda_graphs": False,
        "max_cudagraph_capture_size": 0,
    }
    main_launch = select_launch_config(
        "nvidia_h200", KernelKind.FLASH_ATTN, mla_shape,
        hardware.kernel_dtype, "mla_decode", mla_attributes, hardware)
    if main_launch.algorithm != FA3_DECODE_ALGORITHM:
        raise NotImplementedError(
            "H200 MLA observation is outside the derived FA3 decode domain: "
            f"selector returned {main_launch.algorithm!r}")
    static_splits = main_launch.tiling.num_kv_splits
    dynamic_splits = fa3_decode_dynamic_splits(
        query_lengths, kv_lengths, shape["num_query_heads"], static_splits,
        int(hardware.num_compute_units))
    if (list(dynamic_splits) != list(op_attributes["SP_PER_REQUEST"])
            or (static_splits > 1) != op_attributes["FA3_SPLIT_COMBINE"]):
        raise ValueError(
            "H200 FA3 MLA scheduler metadata disagrees with the source "
            f"split derivation: static={static_splits}, derived="
            f"{list(dynamic_splits)}, recorded="
            f"{op_attributes['SP_PER_REQUEST']}")
    if is_main:
        selected = main_launch
        spec_shape = mla_shape
        variant = "mla_decode"
        attributes = mla_attributes
    else:
        spec_shape = {
            "batch": batch,
            "q_len": shape["q_len"],
            "kv_len": shape["kv_len"],
            "head_dim": latent,
            "num_query_heads": shape["num_query_heads"],
            "num_kv_heads": 1,
        }
        attributes = {
            **common_attributes, "fa_ksteps": None, "fa_eff_mfrac": 1.0}
        selected = select_launch_config(
            "nvidia_h200", KernelKind.FLASH_ATTN, spec_shape,
            hardware.kernel_dtype, "combine",
            {**attributes, "num_kv_splits": static_splits}, hardware)
        if selected.algorithm != FA3_COMBINE_ALGORITHM:
            raise ValueError(
                "H200 FA3 MLA combine was not selected by the FA3 package")
        variant = "combine"
    if (observed.grid != selected.grid
            or observed.block != selected.block):
        raise ValueError(
            "H200 FA3 MLA observed launch disagrees with the pinned source: "
            f"observed={observed.grid}/{observed.block}, "
            f"source={selected.grid}/{selected.block}")
    return KernelSpec(
        kind=KernelKind.FLASH_ATTN,
        hardware_id="nvidia_h200",
        shape=spec_shape,
        dtype=hardware.kernel_dtype,
        variant=variant,
        launch=selected,
        attributes=attributes,
    )
