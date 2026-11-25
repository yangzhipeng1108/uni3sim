"""Normalize profiler observations into complete production ``KernelSpec`` objects.

This is the only single-kernel path allowed to inspect runtime implementation
names.  The resulting spec contains semantic variants and structured launch
facts, never the name itself or measured counters/duration.

This module is the compatibility re-export layer for the historical
import surface; the implementation lives in the sibling modules of
this package.
"""

from evaluation.kernel.observation_readers import (
    _A800_PYTORCH_2_8_BF16_REDUCE_SYMBOL,
    _optional_positive_int,
    _non_empty_string,
    _finite_number,
    _positive_int_sequence,
    validate_kernel_observation_sequence,
    _validate_nvidia_a800_moe_reduce_sequence,
    _a800_moe_sum_symbol,
    observed_flash_attention_splits,
    _observed_h200_fa3_splits,
    observed_nvjet_gemm_split_k,
    _gemm_tensor_split,
    observed_gemm_split_k,
    observed_gemm_split_k_partial_bytes,
    observed_gemm_split_k_partial_dram_bytes,
)

from evaluation.kernel.observation_normalize_gemm_vector import (
    _ASCEND_SOURCE_BOUND_VECTOR_VARIANTS,
    _is_source_bound_vector_observation,
    _SILU_RUNTIME_SYMBOLS,
    _ASCEND_ELEMENTWISE_RUNTIME_SYMBOLS,
    _METAX_VECTOR_RUNTIME_SYMBOLS,
    _A800_ELEMENTWISE_RUNTIME_SYMBOLS,
    _A800_VECTOR_RUNTIME_SYMBOLS,
    _A800_CUBLASLT_OUTPUT_DTYPE_REDUCER,
    _GEMM_ATTR_DTYPES,
    _normalize_gemm,
    _decoded_grid_cta_raster,
    _normalize_eager_attn,
    _validate_observed_vector_element_count,
    _validate_canonical_vector_element_count,
    _require_source_bound_vector_observation,
    _normalize_vector,
)

from evaluation.kernel.observation_normalize_flash_attention import (
    _METAX_372_STATIC_FLASH_TRAITS,
    _METAX_372_Q64_PREFILL_RUNTIME,
    _METAX_372_D64_Q64_PREFILL_RUNTIME,
    _METAX_372_Q64_PREFILL_ALGORITHM,
    _METAX_372_Q64_PACKED_PREFILL_ALGORITHM,
    _METAX_372_Q64_DECODE_RUNTIME,
    _METAX_372_Q64_DECODE_ALGORITHM,
    _METAX_372_Q128_PREFILL_RUNTIME,
    _METAX_372_Q128_SPLIT_PREFILL_RUNTIME,
    _METAX_372_Q128_PREFILL_ALGORITHM,
    _METAX_372_Q128_SPLIT_PREFILL_ALGORITHM,
    _metax_372_static_flash_trait,
    _normalize_flash_attention,
    _H200_MLA_ROPE_DIM,
    _H200_FA3_MAIN_SYMBOL,
    _H200_FA3_COMBINE_SYMBOL,
    _normalize_h200_flash_attention,
)

from evaluation.kernel.observation_normalize_moe import (
    _MOE_AUX_HARDWARE,
    require_moe_aux_hardware,
    _ASCEND_910B_MOE_COMPUTE_EXPERT_TOKENS_CODE_HASH,
    _ASCEND_910B_MOE_CAST_RUNTIME_BY_ROLE,
    _normalize_unfused_moe,
    _fused_moe_projection,
    _normalize_fused_moe,
    _moe_aux_variant,
    _normalize_a800_moe_reduce_attributes,
    _normalize_ascend_moe_compute_expert_tokens_launch,
    _normalize_ascend_moe_source_launch,
    _normalize_metax_moe_fill_attributes,
    _H200_VLLM030_NAIVE_ASSIGNMENT_SPARSITY,
    _H200_TORCH_FILL_SYMBOL_PREFIX,
    _H200_TORCH_VECTORIZED_THREADS,
    _H200_TORCH_VECTORIZED_ELEMENTS_PER_THREAD,
    _normalize_h200_moe_fill_attributes,
    _H200_MOE_SUM_STATIC_TOPK,
    _H200_MOE_SUM_BLOCK,
    _H200_MOE_SUM_VEC_BF16,
    _H200_MOE_SUM_MAX_GRID,
    _validate_h200_moe_sum_launch,
    _normalize_moe_aux,
)

from evaluation.kernel.observation_normalize_dispatch import (
    normalize_kernel_spec,
)

from evaluation.kernel.dsv4_case_specs import (
    _DSV4_CASE_HARDWARE_ID,
    _DSV4_LAYER_COMPRESS_RATIOS,
    _DSV4_COMPRESSED_STATE_BYTES,
    _DSV4_SWA_TOKEN_BYTES,
    _DSV4_HEAD_DIM,
    _DSV4_QK_HEAD_DIM,
    _DSV4_KV_BLOCK_SIZE,
    _DSV4_MODELS,
    _require_dsv4_case_hardware,
    _dsv4_case_record,
    _require_non_negative_int,
    _require_int_sequence,
    _require_evidence_int,
    _require_c128a_topk_width,
    _observed_kernel_names,
    _require_observed_kernel,
    _selected_spec,
    dsv4_case_report_axes,
    _v4attn_attention_spec,
    _v4attn_compressor_spec,
    _v4attn_mhc_spec,
    build_v4attn_case_specs,
    build_wq_fused_moe_case_specs,
    build_dsv4_case_specs,
)
