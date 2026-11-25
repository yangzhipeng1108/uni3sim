"""NVIDIA H200 source-bound vector launch selection.

vLLM 0.30 runs DeepSeek-V4's two-tier sparse attention through the SM90
FlashMLA sparse kernels (``vllm/model_executor/models/deepseek_v4/nvidia/
flashmla.py``).  The non-duration launch facts below come from the
``kc_v4attn_models__v030`` campaign ``window_01.launch.json`` records
(grid/block/registers only; durations are never consumed):

* decode main kernel
  ``sm90::decode::sparse::flash_fwd_splitkv_mla_fp8_sparse_kernel`` launches
  grid ``(HQ // 64, 1, sms // (HQ // 64))`` with block ``(384, 1, 1)``
  (HQ64 -> ``(1, 1, 132)``, HQ128 -> ``(2, 1, 66)`` on the 132-SM H200).
* prefill main kernel launches grid ``(q_len * HQ // 64, 1, 1)`` with block
  ``(384, 1, 1)``.

DSA sparse variants require their own source binding; they cannot project
another hardware package's analytical geometry.
"""

from typing import Mapping, Optional


def moe_count_sort_source_program():
    """Name the independently bound CUDA auxiliary argument/dataflow ABI.

    vLLM0.30 moe_align_sum_kernels.cu's public non-LoRA wrapper passes a
    null token mask to _count_and_sort_expert_tokens: an INT32 atomic rank
    in cumsum selects each routed item's sorted-token output. The existing
    memory owner separately checks the exact observed algorithms, geometry
    and routing context before using this ABI. This is no compiler-resource
    or cross-runtime launch declaration.
    """
    return "cuda_moe_count_sort_int32_unmasked"


from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import (
    require_dtype, require_shape, require_variant)
from src.kernel.launch_config.nvidia_h200.source_vector_launch import (
    SOURCE_VECTOR_VARIANTS, select_kimi_k3_situ_launch, select_mla_rope_launch,
    select_source_vector_launch, select_torch_vector_launch)
from src.kernel.launch_config.nvidia_h200.cache_launch import CACHE_VARIANTS, select_cache_launch
from src.kernel.launch_config.nvidia_h200.moe_aux_launch import (
    SOURCE_MOE_AUX_VARIANTS, select_source_moe_aux_launch)
from src.kernel.launch_config.nvidia_h200.kda_launch import select_kda_decode_launch, select_kda_prefill_launch


_DSV4_SPARSE_VARIANTS = ("dsv4_sparse_prefill", "dsv4_sparse_decode")
_MHC_TILELANG_VARIANTS = (
    "mhc_post_prenorm", "mhc_pre_norm", "mhc_pre_norm_broadcast",
    "mhc_post", "hc_head",
)


def select_sparse_attn_indexer_launch(shape, dtype, variant, attributes, hardware):
    """Preserve the public zero-work alias descriptor, with no CUDA work.

    The singleton LaunchConfig is the existing Core representation of an
    alias, not an observed or predicted one-thread GPU launch. The indexer
    predictor validates alias storage and returns zero work for it. A real
    indexer instead requires its complete prepared multi-stage recipe.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("H200 index reuse requires H200 hardware")
    if variant != "dsa_index_reuse":
        raise NotImplementedError("H200 indexer requires its complete prepared source recipe")
    require_dtype(dtype, ("int32",))
    require_shape(shape, ("element_count",))
    if attributes["reuse_semantics"] != "alias":
        raise ValueError("index reuse must alias its current producer without copying")
    return LaunchConfig("dsa_index_alias", (1, 1, 1), (1, 1, 1),
        execution_variant="dsa_index_reuse")


def pytorch_int32_int64_cast_launch(elements, variant):
    """Torch2.8 SM90 Copy entry, bound independently of the sm80 binary."""
    from src.kernel.launch_config.selection_primitives import pytorch_2_8_integer_copy_launch
    return pytorch_2_8_integer_copy_launch(elements, variant, "sm90")


def pytorch_embedding_index_select_launch_domain(rows, hidden, vocab, elements):
    from src.kernel.launch_config.selection_primitives import pytorch_2_8_embedding_gather_domain
    return pytorch_2_8_embedding_gather_domain(rows, hidden, vocab, elements, 132)


def pytorch_logits_cast_launch(shape, dtype, attributes):
    """SM90 Float-output Copy: BF16 bits expanded by one IMAD.U32 per value."""
    require_dtype(dtype, ("bf16",))
    elements, = require_shape(shape, ("element_count",))
    if attributes["sampling_source"] != "vllm_0_11_plain_sampler_bf16" or "copy_dispatch" in attributes:
        raise ValueError("H200 logits Copy requires its0.11 BF16 source")
    if "copy_input_contiguous" not in attributes or attributes["copy_input_contiguous"] is not True:
        raise NotImplementedError("H200 logits Copy requires proven contiguous input")
    if elements <= 0 or 4 * (elements - 1) > 2**31 - 1:
        raise NotImplementedError("H200 logits Copy requires an unsplit32-bit iterator")
    return LaunchConfig(algorithm="pytorch_2_8_sm90_bfloat_to_float",
        grid=((-(-elements // 512)), 1, 1), block=(128, 1, 1),
        parameters={"registers_per_thread": 30.0, "values_per_thread": 4.0},
        execution_variant="vllm_logits_fp32_cast")


def pytorch_id_narrow_launch(shape, dtype, attributes):
    """SM90 Int-output Copy reads the low32 word at each Long input address."""
    require_dtype(dtype, ("int64",))
    elements, rows = require_shape(shape, ("element_count", "rows"))
    if (attributes["sampling_source"] != "vllm_0_11_plain_greedy_bf16"
            or attributes["stage"] != "output_int32" or elements != rows):
        raise ValueError("H200 ID narrowing requires its0.11 greedy INT64 output source")
    if elements <= 0 or 8 * (elements - 1) > 2**31 - 1:
        raise NotImplementedError("H200 ID narrowing requires one unsplit32-bit iterator")
    return LaunchConfig(algorithm="pytorch_2_8_sm90_int64_to_int32",
        grid=((-(-elements // 512)), 1, 1), block=(128, 1, 1),
        parameters={"registers_per_thread": 30.0, "values_per_thread": 4.0},
        execution_variant="vllm_plain_greedy_sample")


def pytorch_argmax_source_launch(shape, dtype, attributes):
    """Torch2.8 Reduce.cuh geometry/scratch, independent of typed service rates.

    The fresh FP32 logits are contiguous, the reduction is the last dimension,
    output_vec_size is1, and input_vec_size is4. The recorded SM90 Float entry
    is used for every block geometry below; its template512 is a maximum,
    not a claim that block.x is always512.
    """
    require_dtype(dtype, ("fp32",))
    elements, rows, vocab = require_shape(shape, ("element_count", "rows", "vocab"))
    if ("sampling_source" not in attributes or "stage" not in attributes
            or attributes["sampling_source"] != "vllm_0_11_plain_greedy_bf16"
            or attributes["stage"] != "argmax"):
        raise ValueError("H200 argmax requires its0.11 contiguous FP32 greedy source")
    source_properties = ("source_num_compute_units", "source_max_threads_per_unit")
    if any(key not in attributes for key in source_properties):
        raise ValueError("argmax requires its complete positive hardware source properties")
    sms, threads_per_sm = (attributes[key] for key in source_properties)
    if any(type(value) is not int or value <= 0 for value in (sms, threads_per_sm)):
        raise ValueError("argmax requires explicit positive hardware source properties")
    if threads_per_sm < 512:
        raise NotImplementedError("argmax source requires admission of its maximum512-thread block")
    if (rows <= 0 or vocab <= 128 or elements != rows * vocab
            or max(4 * elements, 8 * rows) >= 2**31):
        raise NotImplementedError("argmax source requires vectorized contiguous unsplit32-bit rows")
    dim0, dim1 = vocab // 4, rows
    x = min(1 << (dim0.bit_length() - 1), 512, 32)
    y = min(1 << (dim1.bit_length() - 1), 512 // x)
    x = min(1 << (dim0.bit_length() - 1), 512 // y)
    step_input = x
    values_per_thread = -(-vocab // step_input)
    reduce_y = values_per_thread >= min(y * 16, 256)
    step_output = 1
    if reduce_y:
        step_input *= y
    else:
        step_output *= y
    grid_x = -(-rows // step_output)
    values_per_thread = -(-vocab // step_input)
    target_grid = sms * (threads_per_sm // (x * y))
    ctas = 1
    if reduce_y and values_per_thread >= 256 and grid_x <= target_grid:
        ctas = max(min(-(-target_grid // grid_x), -(-values_per_thread // 16)),
            -(-values_per_thread // 256))
    shared = 16 * x * y if reduce_y or x > 32 else 0
    return LaunchConfig(algorithm="pytorch_2_8_sm90_argmax_float_long",
        execution_variant="vllm_plain_greedy_sample", grid=(grid_x, ctas, 1), block=(x, y, 1),
        parameters={"registers_per_thread": 32.0, "values_per_thread": 4.0,
            "input_vector_size": 4.0, "dynamic_shared_bytes": float(shared),
            "static_shared_bytes": 1040.0,
            "global_partial_bytes": float(16 * rows * ctas if ctas > 1 else 0),
            "global_semaphore_bytes": float(4 * grid_x if ctas > 1 else 0)})


def owns_select_vector_launch(shape, dtype, variant, attributes, hardware):
    del shape, dtype, hardware
    if "source_moe_router_stage" in attributes:
        return True
    if variant in ("kimi_k3_kda_decode", "kimi_k3_situ", "kimi_k3_kda_prefill",
                   "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill"):
        return True
    if variant in (SOURCE_VECTOR_VARIANTS | SOURCE_MOE_AUX_VARIANTS) and "model_execution_version" in attributes:
        return True
    if variant == "elementwise" and "tensor_iterator_source" in attributes:
        return True
    return (variant in ("vllm_embedding_id_widen", "vllm_logits_row_select", "vllm_logits_fp32_cast")
        or variant == "vllm_plain_greedy_sample" and (attributes["stage"] == "output_int32"
            or "source_num_compute_units" in attributes)
        or variant == "gather" and "index_select_source" in attributes
        or variant == "elementwise" and "operation" in attributes
        and attributes["operation"] == "cast_int32_int64")


def select_vector_launch(shape, dtype, variant, attributes, hardware):
    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 vector source requires H200 hardware")
    if "source_moe_router_stage" in attributes:
        from src.kernel.launch_config.nvidia_h200.moe_router_launch import select_legacy_router_stage
        return select_legacy_router_stage(shape, dtype, variant, attributes, hardware)
    if variant == "kimi_k3_kda_decode":
        return select_kda_decode_launch(shape, dtype, variant, attributes, hardware)
    if variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill"):
        return select_kda_prefill_launch(shape, dtype, variant, attributes, hardware)
    if variant == "kimi_k3_situ":
        return select_kimi_k3_situ_launch(shape, dtype, variant, attributes, hardware)
    if variant in SOURCE_MOE_AUX_VARIANTS and "model_execution_version" in attributes:
        return select_source_moe_aux_launch(shape, dtype, variant, attributes, hardware)
    if variant in SOURCE_VECTOR_VARIANTS and "model_execution_version" in attributes:
        return select_source_vector_launch(shape, dtype, variant, attributes, hardware)
    if variant in SOURCE_VECTOR_VARIANTS:
        raise NotImplementedError(
            "H200 vector launch requires an explicit bound model_execution_version")
    if (variant == "elementwise" and "tensor_iterator_source" in attributes
            and attributes["operation"] != "cast_int32_int64"):
        if attributes["tensor_iterator_source"] == "pytorch_2_13_sm90":
            if attributes["operation"] != "fill":
                raise NotImplementedError("Torch2.13 public vector dispatch binds only the explicit fill program")
            if (type(attributes["reads"]) is not int or attributes["reads"] != 0
                    or type(attributes["writes"]) is not int or attributes["writes"] != 1):
                raise ValueError("Torch2.13 fill requires zero reads and one write")
            from dataclasses import replace
            from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_fill_launch
            return replace(select_torch213_fill_launch(shape, dtype, attributes, hardware),
                execution_variant=variant)
        return select_torch_vector_launch(shape, dtype, variant, attributes, hardware)
    if variant == "vllm_logits_fp32_cast":
        return pytorch_logits_cast_launch(shape, dtype, attributes)
    if variant == "vllm_plain_greedy_sample":
        if attributes["stage"] == "argmax":
            if hardware is not None and (
                    attributes["source_num_compute_units"] != int(hardware.num_compute_units)
                    or attributes["source_max_threads_per_unit"] != int(hardware.max_threads_per_unit)):
                raise ValueError("argmax source hardware properties disagree with HWSpec")
            return pytorch_argmax_source_launch(shape, dtype, attributes)
        return pytorch_id_narrow_launch(shape, dtype, attributes)
    if variant in ("gather", "vllm_logits_row_select"):
        require_dtype(dtype, ("bf16",))
        rows, hidden, source_rows, elements = require_shape(
            shape, ("rows", "hidden", "source_rows", "element_count"))
        if variant == "gather":
            if attributes["index_select_source"] != "vllm_0_11_embedding":
                raise ValueError("H200 embedding gather requires its0.11 source")
            if not pytorch_embedding_index_select_launch_domain(rows, hidden, source_rows, elements):
                raise NotImplementedError("H200 embedding gather requires its bounded aligned32-bit source domain")
        elif (attributes["model_execution_version"] != "vllm_0_11_0"
                or attributes["index_source"] != "query_start_loc_int32_to_int64"):
            raise ValueError("H200 row selection requires its0.11 materialized Long indices")
        elif (rows <= 0 or source_rows <= 0 or hidden <= 0 or hidden % 128
                or elements != rows * hidden
                or max(2 * source_rows * hidden, 2 * elements, 8 * rows) >= 2**31):
            raise NotImplementedError("H200 row selection requires contiguous aligned unsplit rows")
        from src.kernel.launch_config.selection_primitives import pytorch_2_8_row_gather_launch
        return pytorch_2_8_row_gather_launch(rows, hidden, variant, "sm90",
            small_index=variant == "gather" and rows <= 16)
    if variant == "elementwise" and attributes["operation"] == "fill":
        # The accepted FA3 source emits exactly one INT32 semaphore Fill.
        # Other fills need their own explicit TensorIterator source contract.
        require_dtype(dtype, ("int32",))
        if (set(attributes) != {"operation", "reads", "writes"}
                or type(attributes["reads"]) is not int or attributes["reads"] != 0
                or type(attributes["writes"]) is not int or attributes["writes"] != 1):
            raise ValueError("FA3 semaphore Fill requires zero reads and one write")
        elements, = require_shape(shape, ("element_count",))
        if elements != 1:
            raise NotImplementedError("unqualified H200 Fill requires the single-INT32 FA3 semaphore source")
        return LaunchConfig("elementwise", (1, 1, 1), (128, 1, 1),
            execution_variant="elementwise")
    require_dtype(dtype, ("int32",))
    elements, = require_shape(shape, ("element_count",))
    if variant == "vllm_embedding_id_widen":
        rows, = require_shape(shape, ("rows",))
        if rows != elements or attributes["embedding_source"] != "vllm_0_11_unquantized_tp1":
            raise ValueError("H200 integer Copy requires its0.11 embedding source")
    elif variant == "elementwise":
        if (attributes["operation"] != "cast_int32_int64"
                or attributes["tensor_iterator_source"] != "pytorch_2_8_sm90"):
            raise ValueError("H200 integer Copy requires its exact SM90 source")
    else:
        raise ValueError("H200 integer Copy variant is unsupported")
    return pytorch_int32_int64_cast_launch(elements, variant)


def mhc_post_prenorm_config(tokens: int, hidden: int, hc_mult: int):
    """Translate vLLM 0.30 mhc_fused_post_pre_split_config, without timing."""
    from src.kernel.launch_config.nvidia_h200.mhc_launch import mhc_fused_source_config
    return mhc_fused_source_config(tokens, hidden, hc_mult)


def mhc_prenorm_splits(tokens: int, input_size: int, hardware) -> int:
    """DeepGEMM output split count; does not establish its launch geometry.

    tilelang.py _hc_prenorm_gemm_outputs and tilelang_kernels.py
    compute_num_split(64, K, ceil(T/64)) on the H200 supported path.
    """
    from src.kernel.launch_config.nvidia_h200.mhc_launch import compute_mhc_prenorm_splits
    return compute_mhc_prenorm_splits(tokens, input_size, int(hardware.num_compute_units))


def select_mhc_stage_launches(shape, dtype, variant, attributes, hardware):
    """Select the source stages, including unbound DeepGEMM calls as None."""
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("mHC stages require H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    require_variant(variant, ("mhc_pre", "mhc_post_pre", "mhc_post", "hc_head"), "mHC")
    elements, hidden, hc_mult = require_shape(shape, ("element_count", "hidden", "hc_mult"))
    if elements % (hidden * hc_mult):
        raise ValueError("mHC elements must contain complete stream rows")
    tokens = elements // (hidden * hc_mult)
    if "source_use_deep_gemm" in attributes and type(attributes["source_use_deep_gemm"]) is not bool:
        raise TypeError("mHC source_use_deep_gemm must be an explicit boolean")

    def selected(stage, extra):
        stage_shape = {**shape, **extra}
        launch = select_target_vector_launch(stage_shape, dtype, stage, {}, hardware)
        return stage, stage_shape, launch

    def prenorm(broadcast):
        # _hc_prenorm_gemm_outputs ORs supported() with !fallback. The
        # broadcast call forbids fallback even with the env flag disabled.
        # True/absent flag cannot establish installed-library support.
        env_declared_disabled = ("source_use_deep_gemm" in attributes
                                 and attributes["source_use_deep_gemm"] is False)
        splits = (1 if not broadcast and env_declared_disabled else
                  mhc_prenorm_splits(tokens, hidden * (1 if broadcast else hc_mult), hardware))
        stage = "mhc_prenorm_broadcast" if broadcast else "mhc_prenorm"
        # Both external implementations retain unbound native launch facts.
        return (stage, {**shape, "n_splits": splits}, None), splits

    if variant == "mhc_pre":
        broadcast = attributes["source_mhc_broadcast"] if "source_mhc_broadcast" in attributes else True
        if type(broadcast) is not bool:
            raise TypeError("mHC pre broadcast branch must be a boolean")
        projection, splits = prenorm(broadcast)
        return (projection, selected("mhc_pre_norm_broadcast" if broadcast else "mhc_pre_norm", {
            "n_splits": splits, "sinkhorn_iters": attributes["sinkhorn_iters"]}))
    if variant == "mhc_post_pre":
        config = mhc_post_prenorm_config(tokens, hidden, hc_mult)
        if config is None:
            projection, splits = prenorm(False)
            prefix = (selected("mhc_post", {}), projection)
        else:
            tile_n, splits = config
            prefix = (selected("mhc_post_prenorm", {"tile_n": tile_n, "n_splits": splits}),)
        return (*prefix, selected("mhc_pre_norm", {
            "n_splits": splits, "sinkhorn_iters": attributes["sinkhorn_iters"]}))
    if variant in ("mhc_post", "hc_head"):
        return (selected(variant, {}),)
    raise AssertionError(f"unhandled logical mHC variant {variant!r}")


def owns_select_target_vector_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    del shape, dtype, attributes, hardware
    return variant in ("mla_rope", "layernorm", "dsv4_moe_route", "dsv4_inv_rope_fp8_quant", "dsv4_short_context_topk", "dsv4_q_kv_rmsnorm", "dsv32_fused_norm_rope", "dsv32_fused_q", "dsv32_eh_norm", "dsv32_embed_eh_norm", "dsv32_embed_norm") or variant in _MHC_TILELANG_VARIANTS


def select_target_vector_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """vLLM FusedQKVRMSNormKernel: token x two independent reductions."""
    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("H200 target vector source requires H200 hardware")
    if variant == "mla_rope":
        return select_mla_rope_launch(shape, dtype, attributes, hardware)
    if variant == "layernorm":
        raise NotImplementedError("H200 LayerNorm requires its complete FP32 cast/native-layernorm/cast source recipe")
    if variant == "dsv4_moe_route":
        require_dtype(dtype, ("fp32",))
        rows, experts, topk = require_shape(shape, ("rows", "experts", "topk"))
        if attributes["routing_source"] == "vllm_0_30_cuda_hash_top6_pure_text":
            if (attributes["expert_id_selection"] != "hash_table"
                    or experts not in (256, 384) or topk != 6):
                raise ValueError("V4 hash launch requires normalized top6 hash selection")
            # topk_softplus_sqrt_kernels.cu launchDsv4HashTopk: four
            # token warps/CTA, with warp>=num_rows returning in the tail.
            return LaunchConfig("vllm_dsv4_hash_top6_cuda_source_projection",
                (-(-rows // 4), 1, 1), (128, 1, 1), execution_variant=variant)
        if (attributes["routing_source"] != "vllm_0_30_triton_top6_pure_text"
                or attributes["expert_id_selection"] != "score_topk"
                or experts not in (256, 384) or topk != 6):
            raise ValueError("V4 route launch requires explicit pure-text Triton top6 source")
        # dsv4_topk.py: one CTA/token, num_warps=1. FP32 contiguous logits
        # and bias, renormalize=True, INT32 output, no VL are source assumptions.
        return LaunchConfig("vllm_dsv4_top6_triton_source_projection",
            (rows, 1, 1), (32, 1, 1), execution_variant=variant)
    if variant == "dsv4_inv_rope_fp8_quant":
        require_dtype(dtype, ("fp8",))
        rows, heads = require_shape(shape, ("seq_len", "num_query_heads"))
        # SM90 FP32 scale rows have 16-byte TMA alignment too: _impl
        # launches padding rows to zero-fill them. __call__ sets one warp.
        aligned_rows = -(-rows // 4) * 4
        return LaunchConfig("dsv4_fused_inv_rope_fp8_quant_analytical",
            (aligned_rows, heads, 1), (32, 1, 1), execution_variant=variant)
    if variant == "dsv4_short_context_topk":
        require_dtype(dtype, ("int32",))
        rows, capacity, ratio, elements = require_shape(shape, (
            "seq_len", "topk_capacity", "compress_ratio", "element_count"))
        if ratio != 4 or elements != rows * capacity:
            raise ValueError("V4 short-context fill requires full C4 topk rows")
        # attention.py:1082-1089 explicitly num_warps=8, grid=(num_tokens,).
        return LaunchConfig("vllm_dsv4_short_context_topk_triton",
            (rows, 1, 1), (256, 1, 1), execution_variant=variant)
    require_dtype(dtype, ("bf16",))
    if variant == "dsv32_fused_norm_rope":
        rows, = require_shape(shape, ("rows",))
        if type(attributes["has_indexer"]) is not bool:
            raise TypeError("DSA norm/RoPE requires its explicit indexer branch")
        if (attributes["mla_cache_dtype"] != "bf16"
                or attributes["cache_slot_scope"] not in (
                    "all_rows_valid_slots_projection", "outputs_only_no_cache_slots")
                or attributes["index_rope_interleave"] is not True):
            raise NotImplementedError("DSA norm/RoPE source requires its BF16 cache and interleaved index RoPE")
        # common/kernels.py::fused_norm_rope fixes grid=(num_tokens,4),
        # including idle pid0/3 in shared-indexer layers. The compiler's
        # non-autotuned four-warp default is independently bound to H200
        # by all68 kc_dsa_fused_norm_rope__v030 window_01 observations
        # (IDX0/IDX1,34 each), rather than the local Triton3.4 install.
        # Block equality with those same records is consistency only;
        # registers/shared resources remain separately unbound for0.30.
        return LaunchConfig(
            algorithm="vllm_0_30_sm90_dsa_fused_norm_rope",
            grid=(rows, 4, 1), block=(128, 1, 1), execution_variant=variant)
    if variant == "dsv32_fused_q":
        rows, heads, index_heads = require_shape(shape, ("rows", "q_heads", "index_heads"))
        grid_heads = max(-(-heads // 2), index_heads if attributes["has_indexer"] else 1)
        # common/kernels.py explicitly num_warps=1; the CuTeDSL alternative
        # requires SM100 and FP8 MQA, so cannot run on this SM90 BF16 path.
        return LaunchConfig(
            algorithm="vllm_v32_fused_q_triton_bf16_sm90",
            grid=(rows, 3, grid_heads), block=(32, 1, 1), execution_variant=variant)
    if variant == "dsv32_embed_norm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        block = 1 << (hidden - 1).bit_length()
        # fused_embed_norm.py explicitly selects this many warps.
        warps = min(32, max(4, block // 512))
        return LaunchConfig("vllm_v32_fused_embed_norm_triton",
            (rows, 1, 1), (32 * warps, 1, 1), execution_variant=variant)
    if variant in ("dsv32_eh_norm", "dsv32_embed_eh_norm"):
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        # vLLM 0.30 fused_eh_norm / fused_embed_eh_norm use grid=(T,)
        # without an explicit num_warps. Local Triton 3.4.0 NVIDIA
        # CUDAOptions defaults to four warps. This analytical projection
        # is explicitly NOT a binding of the unavailable 0.30 runtime's
        # compiler. Its version/default must be rechecked for W2.
        return LaunchConfig(
            algorithm="vllm_v32_eh_norm_local_triton_3_4_default_projection",
            grid=(rows, 1, 1), block=(128, 1, 1), execution_variant=variant)
    if variant in _MHC_TILELANG_VARIANTS:
        elements, hidden, hc_mult = require_shape(
            shape, ("element_count", "hidden", "hc_mult"))
        if elements % (hidden * hc_mult):
            raise ValueError("mHC elements must contain complete stream rows")
        tokens = elements // (hidden * hc_mult)
        if variant == "mhc_post_prenorm":
            config = mhc_post_prenorm_config(tokens, hidden, hc_mult)
            if config is None:
                raise ValueError("mHC fused post/prenorm shape is not supported")
            tile_n, splits = config
            if (shape["tile_n"], shape["n_splits"]) != config:
                raise ValueError("mHC fused post/prenorm split geometry disagrees")
            grid = (tokens, hc_mult * (hc_mult + 2) // tile_n, splits)
            threads = 128
        elif variant in ("mhc_pre_norm", "mhc_pre_norm_broadcast"):
            grid, threads = (tokens, 1, 1), 96
        elif variant in ("mhc_post", "hc_head"):
            grid, threads = (tokens, 1, 1), 128
        else:
            raise AssertionError(f"unhandled mHC variant {variant!r}")
        return LaunchConfig(
            algorithm=f"vllm_{variant}_tilelang", grid=grid,
            block=(threads, 1, 1), execution_variant=variant)
    selected = require_variant(variant, ("dsv4_q_kv_rmsnorm",), "H200 VECTOR")
    rows, q_hidden, kv_hidden = require_shape(
        shape, ("rows", "q_hidden", "kv_hidden"))
    block_size = 1 << (max(q_hidden, kv_hidden) - 1).bit_length()
    warps = 8 if block_size >= 2048 else 4
    return LaunchConfig(
        algorithm="vllm_fused_q_kv_rmsnorm_triton",
        grid=(rows, 2, 1), block=(32 * warps, 1, 1),
        execution_variant=selected,
    )


def owns_select_kv_cache_update_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    del shape, dtype, attributes, hardware
    return variant == "dsv4_qnorm_rope_kv_insert" or variant in CACHE_VARIANTS


def select_kv_cache_update_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Use the vLLM 0.30 CUDA wrapper's discrete launch geometry.

    csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu,
    launchFusedDeepseekV4Templated: eight warps per CTA; below
    NUM_TOKEN_CUTOFF=1024 each warp owns one padded Q/KV head slot.
    At/above the cutoff a CTA loops over the slots of one token.
    """
    if variant in CACHE_VARIANTS:
        return select_cache_launch(shape, dtype, variant, attributes, hardware)
    require_dtype(dtype, ("bf16",))
    execution_variant = require_variant(
        variant, ("dsv4_qnorm_rope_kv_insert",), "H200 KV_CACHE")
    tokens, heads = require_shape(shape, ("seq_len", "padded_query_heads"))
    if heads not in (8, 16, 32, 64, 128):
        raise ValueError("DSV4 QNorm/RoPE requires a compiled padded head count")
    blocks = -(-(tokens * (heads + 1)) // 8) if tokens < 1024 else tokens
    return LaunchConfig(
        algorithm="dsv4_fused_qnorm_rope_kv_insert",
        grid=(blocks, 1, 1),
        block=(256, 1, 1),
        execution_variant=execution_variant,
    )


def owns_select_sparse_attn_launch(shape: Mapping[str, int],
                                   dtype: Optional[str],
                                   variant: Optional[str],
                                   attributes: Mapping[str, object],
                                   hardware) -> bool:
    """Whether the request is a DeepSeek-V4 FlashMLA sparse launch."""

    del shape, dtype, attributes, hardware
    return (variant is not None
            and variant.strip().lower() in _DSV4_SPARSE_VARIANTS)


def select_sparse_attn_launch(shape: Mapping[str, int],
                              dtype: Optional[str],
                              variant: Optional[str],
                              attributes: Mapping[str, object],
                              hardware) -> LaunchConfig:
    """Select the source-evidenced SM90 FlashMLA sparse-attention grid."""

    require_dtype(dtype, ("bf16",))
    execution_variant = require_variant(
        variant, _DSV4_SPARSE_VARIANTS, "H200 SPARSE_ATTN")
    batch, q_len, heads = require_shape(
        shape, ("batch", "q_len", "num_query_heads"))
    if heads % 64:
        raise ValueError(
            "DSV4 FlashMLA sparse attention requires num_query_heads "
            "divisible by the 64-head kernel tile")
    sms = int(hardware.num_compute_units)
    if execution_variant == "dsv4_sparse_decode":
        head_ctas = heads // 64
        if sms % head_ctas:
            raise ValueError(
                "DSV4 FlashMLA split-KV decode requires the SM count "
                "divisible by the head-CTA count")
        grid = (head_ctas, 1, sms // head_ctas)
        algorithm = "flashmla_dsv4_splitkv_sparse_decode"
    elif execution_variant == "dsv4_sparse_prefill":
        q_lens = tuple(attributes["q_lens"])
        if (len(q_lens) != batch
                or any(isinstance(q, bool) or not isinstance(q, int) or q <= 0 for q in q_lens)
                or max(q_lens) != q_len):
            raise ValueError("V4 sparse prefill query lengths disagree with batch shape")
        # flashmla.py passes concatenated q[query_start:query_end] to the
        # sparse kernel. Varlen requests do not create batch*max_q padding.
        grid = (sum(q_lens) * (heads // 64), 1, 1)
        algorithm = "flashmla_dsv4_sparse_prefill"
    else:
        raise AssertionError(
            f"unhandled H200 DSV4 sparse variant: {execution_variant!r}")
    return LaunchConfig(
        algorithm=algorithm,
        grid=grid,
        block=(384, 1, 1),
        execution_variant=execution_variant,
    )


def owns_select_dsv4_decode_split_counts(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Whether the request is a DeepSeek-V4 split-KV decode plan request."""

    del shape, dtype, attributes, hardware
    return variant is not None and variant.strip().lower() == (
        "dsv4_sparse_decode")


def select_dsv4_decode_split_counts(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> tuple[int, ...]:
    """Return the analytical per-request split-KV budget for DSV4 decode.

    This is NOT the FlashMLA tile scheduler.  The real plan is produced by
    the FlashMLA in-kernel planner: ``flash_mla_with_kvcache`` enters
    ``fwd_kvcache_mla_fp8``, which allocates and fills
    ``tile_scheduler_metadata`` / ``num_splits`` on the first call of each
    layer type (vLLM 0.30.0 ``deepseek_v4/nvidia/flashmla.py:225-231``;
    allocation contract ``v1/attention/backends/mla/sparse_swa.py:824-858``
    ``build_tile_scheduler``).  That planner balances chunk work across the
    SM slots with per-request split granularity; its output is a discrete
    launch observation the kernel-evaluation lane carries as
    ``flash_mla_sched_meta.num_splits``.  Without the recorded plan (the
    Serving lane has no observation), this selector derives an analytical
    even-share budget (``analytical_even_share_split_budget_v1``): each
    request may consume at most an even share of the SM slots, capped by its
    own chunk count::

        splits_r = min(chunks_r, max(1, slots // batch))

    Discrete constants and their non-duration sources:

    * 64-head pack — the FP8 decode kernel supports only h_q in {64, 128}
      (``DeepseekV4FlashMLAAttention.get_padded_num_q_heads``,
      ``flashmla.py:78-85``); the observed grid x is ``HQ // 64``.
    * SM slots ``sms // (HQ // 64)`` — observed decode grid z (132 at HQ64,
      66 at HQ128 on the 132-SM H200) and the recorded
      ``flash_mla_sched_meta.tile_scheduler_metadata_shape`` of
      ``[slots, 8]``: one tile-scheduler row per SM slot.
    * ``page_block_size = 64`` chunk granularity — recorded
      ``flash_mla_sched_meta.page_block_size``; a request's chunk count is
      ``ceil(selected_states / 64) + ceil(swa_window / 64)`` over both cache
      tiers.

    Approximation scope (measured, not proven): against the
    ``kc_v4attn_models__v030`` campaign's 5080 decode requests the rule
    reproduces the recorded per-request splits for 3841 (75.6%), overcounts
    967 (19.0%) and undercounts 272 (5.4%), worst undercount ratio 2.0x at
    batch 128 / HQ128 / ctx 786432.  The real plan is NOT slot-conserving
    (recorded totals reach 190 splits against 66 slots at batch 128 — splits
    time-multiplex a slot), so no slot-budget upper bound exists; the
    campaign agreement above is non-duration validation on this campaign's
    batch x context domain only, not a general proof.  Error direction when
    wrong: undercounting splits underestimates the split-KV combine partial
    traffic, overcounting overestimates it.
    """

    del dtype
    require_variant(variant, ("dsv4_sparse_decode",),
                    "H200 DSV4 decode split counts")
    batch, heads = require_shape(shape, ("batch", "num_query_heads"))
    if heads % 64:
        raise ValueError(
            "DSV4 split-KV decode requires num_query_heads divisible by "
            "the 64-head kernel tile")
    head_ctas = heads // 64
    sms = int(hardware.num_compute_units)
    if sms % head_ctas:
        raise ValueError(
            "DSV4 FlashMLA split-KV decode requires the SM count "
            "divisible by the head-CTA count")
    slots = sms // head_ctas
    budget = max(1, slots // batch)
    selected = tuple(int(v) for v in attributes["selected_kv_lens"])
    swa = tuple(int(v) for v in attributes["swa_kv_lens"])
    if len(selected) != batch or len(swa) != batch:
        raise ValueError(
            "DSV4 decode split counts require per-request selected/SWA "
            "lengths covering every batch entry")
    return tuple(
        min(-(-states // 64) + -(-window // 64), budget)
        for states, window in zip(selected, swa, strict=True))
