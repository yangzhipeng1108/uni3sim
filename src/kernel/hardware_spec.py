"""Hardware spec loader: resource model.

Every hardware component is a Resource with a throughput capacity.
Resources that share hardware are grouped (demands add within a group).
Formula: t = max over groups (sum of demand/capacity within group) + C_launch.

Pinned reference digests own historical capture locations and process identities;
source validation checks their structure without repeating machine-specific values.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import yaml


from src.kernel.ascend_910b_mechanism_loaders import (
    AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE as _AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE,
    AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE_SHA256 as _AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE_SHA256,
    AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE as _AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE,
    AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE_SHA256 as _AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE_SHA256,
    AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE as _AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE,
    AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE_SHA256 as _AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE_SHA256,
    AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE as _AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE,
    AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE_SHA256 as _AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE_SHA256,
    AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE as _AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE,
    AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE_SHA256 as _AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE_SHA256,
    AIC_GEMM_FINITE_TRANSITION_SIGNATURE_FIELDS as _AIC_GEMM_FINITE_TRANSITION_SIGNATURE_FIELDS,
    AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE as _AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE,
    AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE_SHA256 as _AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE as _AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE,
    AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE as _AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE,
    AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE as _AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE,
    AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_LONG_STREAM_COMPLETION_REFERENCE as _AIC_GROUPED_MATMUL_LONG_STREAM_COMPLETION_REFERENCE,
    AIC_GROUPED_MATMUL_LONG_STREAM_COMPLETION_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_LONG_STREAM_COMPLETION_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE as _AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE,
    AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE as _AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE,
    AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE as _AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE,
    AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_SHORT_STREAM_COMPLETION_REFERENCE as _AIC_GROUPED_MATMUL_SHORT_STREAM_COMPLETION_REFERENCE,
    AIC_GROUPED_MATMUL_SHORT_STREAM_COMPLETION_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_SHORT_STREAM_COMPLETION_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE as _AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE,
    AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE as _AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE,
    AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE_SHA256,
    AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE as _AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE,
    AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE_SHA256 as _AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE_SHA256,
    AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE as _AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE,
    AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE_SHA256 as _AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE_SHA256,
    AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE as _AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE,
    AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE_SHA256 as _AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE_SHA256,
    AIC_MATMUL_V2_FINITE_INGRESS_COMPLETION_REFERENCE as _AIC_MATMUL_V2_FINITE_INGRESS_COMPLETION_REFERENCE,
    AIC_MATMUL_V2_FINITE_INGRESS_COMPLETION_REFERENCE_SHA256 as _AIC_MATMUL_V2_FINITE_INGRESS_COMPLETION_REFERENCE_SHA256,
    AIC_MATMUL_V2_K13824_FINITE_TRANSITION_REFERENCE as _AIC_MATMUL_V2_K13824_FINITE_TRANSITION_REFERENCE,
    AIC_MATMUL_V2_K13824_FINITE_TRANSITION_REFERENCE_SHA256 as _AIC_MATMUL_V2_K13824_FINITE_TRANSITION_REFERENCE_SHA256,
    AIC_MATMUL_V2_K14336_M128_N128_FINITE_INGRESS_COMPLETION_REFERENCE as _AIC_MATMUL_V2_K14336_M128_N128_FINITE_INGRESS_COMPLETION_REFERENCE,
    AIC_MATMUL_V2_K14336_M128_N128_FINITE_INGRESS_COMPLETION_REFERENCE_SHA256 as _AIC_MATMUL_V2_K14336_M128_N128_FINITE_INGRESS_COMPLETION_REFERENCE_SHA256,
    AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE as _AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE,
    AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE_SHA256 as _AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE_SHA256,
    AIC_MATMUL_V2_K8192_FINITE_TRANSITION_REFERENCE as _AIC_MATMUL_V2_K8192_FINITE_TRANSITION_REFERENCE,
    AIC_MATMUL_V2_K8192_FINITE_TRANSITION_REFERENCE_SHA256 as _AIC_MATMUL_V2_K8192_FINITE_TRANSITION_REFERENCE_SHA256,
    AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE as _AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE,
    AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE_SHA256 as _AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE_SHA256,
    AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE as _AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE,
    AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE_SHA256 as _AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE_SHA256,
    AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE as _AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE,
    AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE_SHA256 as _AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE_SHA256,
    AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE as _AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE,
    AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE_SHA256 as _AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE_SHA256,
    AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE as _AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE,
    AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE_SHA256 as _AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE_SHA256,
    AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE as _AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE,
    AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE_SHA256 as _AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE_SHA256,
    AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE as _AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE,
    AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE_SHA256 as _AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE_SHA256,
    AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE as _AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE,
    AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE_SHA256 as _AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE_SHA256,
    AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE as _AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE,
    AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE_SHA256 as _AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE_SHA256,
    AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE as _AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE,
    AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE_SHA256 as _AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE_SHA256,
    AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE as _AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE,
    AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE_SHA256 as _AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE_SHA256,
    AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE as _AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE,
    AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE_SHA256 as _AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE_SHA256,
    AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE as _AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE,
    AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE_SHA256 as _AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE_SHA256,
    AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE as _AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE,
    AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256 as _AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256,
    AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE as _AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE,
    AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE_SHA256 as _AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE_SHA256,
    AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE as _AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE,
    AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE_SHA256 as _AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE_SHA256,
    AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE as _AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE,
    AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256 as _AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256,
    AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE as _AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE,
    AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE_SHA256 as _AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE_SHA256,
    AIC_MTE2_FINITE_STREAM_REFERENCE as _AIC_MTE2_FINITE_STREAM_REFERENCE,
    AIC_MTE2_FINITE_STREAM_REFERENCE_SHA256 as _AIC_MTE2_FINITE_STREAM_REFERENCE_SHA256,
    AIC_MTE2_GEMM_K13824_SOURCE_ORDER_COMPLETION_REFERENCE as _AIC_MTE2_GEMM_K13824_SOURCE_ORDER_COMPLETION_REFERENCE,
    AIC_MTE2_GEMM_K13824_SOURCE_ORDER_COMPLETION_REFERENCE_SHA256 as _AIC_MTE2_GEMM_K13824_SOURCE_ORDER_COMPLETION_REFERENCE_SHA256,
    AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE as _AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE,
    AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE_SHA256 as _AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE_SHA256,
    AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE as _AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE,
    AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE_SHA256 as _AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE_SHA256,
    AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE as _AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE,
    AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE_SHA256 as _AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE_SHA256,
    AIC_MTE2_GEMM_REGION_COMPLETION_16X16_REFERENCE as _AIC_MTE2_GEMM_REGION_COMPLETION_16X16_REFERENCE,
    AIC_MTE2_GEMM_REGION_COMPLETION_16X16_REFERENCE_SHA256 as _AIC_MTE2_GEMM_REGION_COMPLETION_16X16_REFERENCE_SHA256,
    AIC_MTE2_GEMM_REGION_COMPLETION_16X32_REFERENCE as _AIC_MTE2_GEMM_REGION_COMPLETION_16X32_REFERENCE,
    AIC_MTE2_GEMM_REGION_COMPLETION_16X32_REFERENCE_SHA256 as _AIC_MTE2_GEMM_REGION_COMPLETION_16X32_REFERENCE_SHA256,
    AIC_MTE2_GEMM_REGION_COMPLETION_256X112_REFERENCE as _AIC_MTE2_GEMM_REGION_COMPLETION_256X112_REFERENCE,
    AIC_MTE2_GEMM_REGION_COMPLETION_256X112_REFERENCE_SHA256 as _AIC_MTE2_GEMM_REGION_COMPLETION_256X112_REFERENCE_SHA256,
    AIC_MTE2_GEMM_REGION_COMPLETION_8X27_REFERENCE as _AIC_MTE2_GEMM_REGION_COMPLETION_8X27_REFERENCE,
    AIC_MTE2_GEMM_REGION_COMPLETION_8X27_REFERENCE_SHA256 as _AIC_MTE2_GEMM_REGION_COMPLETION_8X27_REFERENCE_SHA256,
    AIC_MTE2_GEMM_REGION_COMPLETION_REFERENCE as _AIC_MTE2_GEMM_REGION_COMPLETION_REFERENCE,
    AIC_MTE2_GEMM_REGION_COMPLETION_REFERENCE_SHA256 as _AIC_MTE2_GEMM_REGION_COMPLETION_REFERENCE_SHA256,
    AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE as _AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE,
    AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE_SHA256 as _AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE_SHA256,
    AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE as _AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE,
    AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE_SHA256 as _AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE_SHA256,
    AIC_MTE2_GEMM_SOURCE_ORDER_COMPLETION_REFERENCE as _AIC_MTE2_GEMM_SOURCE_ORDER_COMPLETION_REFERENCE,
    AIC_MTE2_GEMM_SOURCE_ORDER_COMPLETION_REFERENCE_SHA256 as _AIC_MTE2_GEMM_SOURCE_ORDER_COMPLETION_REFERENCE_SHA256,
    AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS as _AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS,
    AIC_MTE2_LEGACY_L1_PAIR_REFERENCE as _AIC_MTE2_LEGACY_L1_PAIR_REFERENCE,
    AIC_MTE2_LEGACY_L1_PAIR_REFERENCE_SHA256 as _AIC_MTE2_LEGACY_L1_PAIR_REFERENCE_SHA256,
    AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE as _AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE,
    AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE_SHA256 as _AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE_SHA256,
    AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE as _AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE,
    AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE_SHA256 as _AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE_SHA256,
    AIC_MTE2_ND2NZ_33COPY_GROUP_SERVICE as _AIC_MTE2_ND2NZ_33COPY_GROUP_SERVICE,
    AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE as _AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE,
    AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE_SHA256 as _AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE_SHA256,
    AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE as _AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE,
    AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE_SHA256 as _AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE_SHA256,
    AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE as _AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE,
    AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE_SHA256 as _AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE_SHA256,
    AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE as _AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE,
    AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE_SHA256 as _AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE_SHA256,
    load_aic_deterministic_split_k_collective_barrier_service as _load_aic_deterministic_split_k_collective_barrier_service,
    load_aic_deterministic_split_k_source_queue_service as _load_aic_deterministic_split_k_source_queue_service,
    load_aic_fixpipe_bf16_finite_stream_service as _load_aic_fixpipe_bf16_finite_stream_service,
    load_aic_fixpipe_bf16_long_stream_service as _load_aic_fixpipe_bf16_long_stream_service,
    load_aic_fixpipe_bf16_terminal_tile_service as _load_aic_fixpipe_bf16_terminal_tile_service,
    load_aic_gemm_exact_transition_reference as _load_aic_gemm_exact_transition_reference,
    load_aic_gemm_finite_ingress_completion_service as _load_aic_gemm_finite_ingress_completion_service,
    load_aic_gemm_short_pipeline_completion_service as _load_aic_gemm_short_pipeline_completion_service,
    load_aic_grouped_matmul_entry_metadata_pair_service as _load_aic_grouped_matmul_entry_metadata_pair_service,
    load_aic_grouped_matmul_finite_stream_completion_service as _load_aic_grouped_matmul_finite_stream_completion_service,
    load_aic_grouped_matmul_k2880_output_tile_service as _load_aic_grouped_matmul_k2880_output_tile_service,
    load_aic_grouped_matmul_k2880_wide_base_tail_service as _load_aic_grouped_matmul_k2880_wide_base_tail_service,
    load_aic_grouped_matmul_m32_short_source_prelude_service as _load_aic_grouped_matmul_m32_short_source_prelude_service,
    load_aic_grouped_matmul_short_source_prelude_service as _load_aic_grouped_matmul_short_source_prelude_service,
    load_aic_grouped_matmul_short_source_tail_prelude_service as _load_aic_grouped_matmul_short_source_tail_prelude_service,
    load_aic_grouped_matmul_step4_w2_bounded_b_ring_extension_service as _load_aic_grouped_matmul_step4_w2_bounded_b_ring_extension_service,
    load_aic_grouped_matmul_step4_worker_sequence_base_service as _load_aic_grouped_matmul_step4_worker_sequence_base_service,
    load_aic_grouped_matmul_step4_worker_sequence_service as _load_aic_grouped_matmul_step4_worker_sequence_service,
    load_aic_grouped_matmul_stream_completion_reference as _load_aic_grouped_matmul_stream_completion_reference,
    load_aic_matmul_v2_base_k_lifecycle_service as _load_aic_matmul_v2_base_k_lifecycle_service,
    load_aic_matmul_v2_dynamic_m_tail_control_service as _load_aic_matmul_v2_dynamic_m_tail_control_service,
    load_aic_matmul_v2_small_m_base_k_control_service as _load_aic_matmul_v2_small_m_base_k_control_service,
    load_aic_matmul_v3_asw_mte2_pair_service as _load_aic_matmul_v3_asw_mte2_pair_service,
    load_aic_matmul_v3_split_k_resident_tile_service as _load_aic_matmul_v3_split_k_resident_tile_service,
    load_aic_mte2_cold_legacy_k14336_stride_service as _load_aic_mte2_cold_legacy_k14336_stride_service,
    load_aic_mte2_cold_n_major_b_stride_service as _load_aic_mte2_cold_n_major_b_stride_service,
    load_aic_mte2_cold_operand_pair_stream_service as _load_aic_mte2_cold_operand_pair_stream_service,
    load_aic_mte2_cold_small_m_stride_service as _load_aic_mte2_cold_small_m_stride_service,
    load_aic_mte2_cold_v3_k1536_n5120_stride_service as _load_aic_mte2_cold_v3_k1536_n5120_stride_service,
    load_aic_mte2_fia_gqa_drained_count_law as _load_aic_mte2_fia_gqa_drained_count_law,
    load_aic_mte2_fia_gqa_finite_drain_service as _load_aic_mte2_fia_gqa_finite_drain_service,
    load_aic_mte2_fia_mha_finite_drain_service as _load_aic_mte2_fia_mha_finite_drain_service,
    load_aic_mte2_fia_mla_strided_service as _load_aic_mte2_fia_mla_strided_service,
    load_aic_mte2_finite_stream_completion_service as _load_aic_mte2_finite_stream_completion_service,
    load_aic_mte2_gemm_large_resident_service as _load_aic_mte2_gemm_large_resident_service,
    load_aic_mte2_gemm_persistent_issue_service as _load_aic_mte2_gemm_persistent_issue_service,
    load_aic_mte2_gemm_region_completion_reference as _load_aic_mte2_gemm_region_completion_reference,
    load_aic_mte2_gemm_region_completion_services as _load_aic_mte2_gemm_region_completion_services,
    load_aic_mte2_gemm_region_pipeline_completion_service as _load_aic_mte2_gemm_region_pipeline_completion_service,
    load_aic_mte2_gemm_region_tail_completion_services as _load_aic_mte2_gemm_region_tail_completion_services,
    load_aic_mte2_gemm_shared_source_queue_service as _load_aic_mte2_gemm_shared_source_queue_service,
    load_aic_mte2_gemm_source_order_completion_service as _load_aic_mte2_gemm_source_order_completion_service,
    load_aic_mte2_legacy_l1_pair_service as _load_aic_mte2_legacy_l1_pair_service,
    load_aic_mte2_mixed_ring_long_overlap_domain as _load_aic_mte2_mixed_ring_long_overlap_domain,
    load_aic_mte2_nd2nz_33copy_group_service as _load_aic_mte2_nd2nz_33copy_group_service,
    load_aic_mte2_nd2nz_large_contiguous_service as _load_aic_mte2_nd2nz_large_contiguous_service,
    load_aic_mte2_operand_pair_finite_overlap_domain as _load_aic_mte2_operand_pair_finite_overlap_domain,
    load_aic_mte2_operand_pair_overlap_domain as _load_aic_mte2_operand_pair_overlap_domain,
    load_aic_mte2_paged_gqa128_stride_service as _load_aic_mte2_paged_gqa128_stride_service,
    load_split_fuse_mte2_short_tail_service as _load_split_fuse_mte2_short_tail_service,
)
from src.kernel.hardware_reference_validation import (
    linear_interpolate as _linear_interpolate,
    require_exact_fields as _require_exact_fields,
)

def _mechanism_reference_hardware_id(hardware_id: str) -> str:
    # Every card reads only its own measured mechanism references.
    return hardware_id


# Only the hardware whose spec must gate service ownership on the pairing
# of hardware id and its config's ``features.kernel_memory_model`` is
# registered here: a spec owns the services of a registered hardware only
# when both the id and this declared model match.  Hardware ids not
# registered here fail closed -- looking one up raises KeyError, which is
# the intended behavior.
_MEMORY_MODEL_BY_HARDWARE = {
    "ascend_910b": "npu_tile",
    "metax_c500": "gpu_mshr",
}

from src.kernel.model_contract import (
    AIV_COMPLETE_TASK_FLOOR_ONLY,
    AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_PROLOGUE,
    GPU_SOURCE_PIPELINE_OUTER_ENVELOPE,
    METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS,
    MemoryAccessClassService,
    MemoryAccessClassSignature,
    PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
    PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS,
    TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
    TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR,
    WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
)


_AIC_FIXPIPE_BF16_FINITE_STREAM_SIGNATURE_BY_SERVICE = {
    "aic_fixpipe_bf16_finite_stream_16x176_22_workers": (16, 176, 22),
    "aic_fixpipe_bf16_finite_stream_32x256_16_workers": (32, 256, 16),
    "aic_fixpipe_bf16_finite_stream_32x256_20_workers": (32, 256, 20),
    "aic_fixpipe_bf16_finite_stream_32x256_22_workers": (32, 256, 22),
    "aic_fixpipe_bf16_finite_stream_48x256_22_workers": (48, 256, 22),
    "aic_fixpipe_bf16_finite_stream_64x256_20_workers": (64, 256, 20),
    "aic_fixpipe_bf16_finite_stream_64x256_22_workers": (64, 256, 22),
    "aic_fixpipe_bf16_finite_stream_128x256_24_workers": (128, 256, 24),
}
_FIA_AIV_INSTRUCTION_COLD_FILL_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "fia_aiv_instruction_cold_fill_cann_8_5_1.json")
_FIA_AIV_INSTRUCTION_COLD_FILL_REFERENCE_SHA256 = (
    "3ea065b636d464b454af430fa3bfbaf7c4c59ccd84273cf6834e6477e1e37b3d")
_FIA_AIV_INSTRUCTION_COLD_FILL_SUPPLEMENT_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "fia_aiv_instruction_cold_fill_g16_s1024_d64_cann_8_5_1.json")
_FIA_AIV_INSTRUCTION_COLD_FILL_SUPPLEMENT_REFERENCE_SHA256 = (
    "251b775fb0ddc1c8f0086d1e3e94023badc52cde717b7cacaf8f7fa98f749f55")
_FIA_AIV_INSTRUCTION_COLD_FILL_G21_SUPPLEMENT_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "fia_aiv_instruction_cold_fill_g21_s2048_d128_cann_8_5_1.json")
_FIA_AIV_INSTRUCTION_COLD_FILL_G21_SUPPLEMENT_REFERENCE_SHA256 = (
    "398bf61f5c16c9f8e0fc5f2c48266960fc60af63273b9da5b9059d9d208ac22f")
_FIA_AIV_INSTRUCTION_COLD_FILL_G11_S512_SUPPLEMENT_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "fia_aiv_instruction_cold_fill_g11_s512_d128_cann_8_5_1.json")
_FIA_AIV_INSTRUCTION_COLD_FILL_G11_S512_SUPPLEMENT_REFERENCE_SHA256 = (
    "157e49860b27aa532fe2f9d6197c2cd6a57742eae12eda5a25c9aec1d6a83735")
_FIA_AIV_INSTRUCTION_COLD_FILL_G22_S512_SUPPLEMENT_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "fia_aiv_instruction_cold_fill_g22_s512_d128_cann_8_5_1.json")
_FIA_AIV_INSTRUCTION_COLD_FILL_G22_S512_SUPPLEMENT_REFERENCE_SHA256 = (
    "a59bd93cbbdc9c9a9f89ed28c35b89213355c88a8755822150ef3cf189df2f2f")
_AIV_ROPE_TILING_MATERIALIZATION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "rope_tiling_data_materialization_cann_8_5_1.json")
_AIV_ROPE_TILING_MATERIALIZATION_REFERENCE_SHA256 = (
    "fb323b7aa62f903de9d81dec492cb169d9c444c6fc017dcf491c608be3f6a711")
_AIV_BROADCAST_ONE_DIM_TILING_MATERIALIZATION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "broadcast_one_dim_tiling_materialization_cann_8_5_1.json")
_AIV_BROADCAST_ONE_DIM_TILING_MATERIALIZATION_REFERENCE_SHA256 = (
    "2ebf225ae5cd9285a3064a795113d99f4a3c2a5270e41b55fda3425cd35a4ffe")
_METAX_GEMV_WAVE8_COMPLETION_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "gemv_wave8_completion_maca_3_7_2.json")
_METAX_GEMV_WAVE8_COMPLETION_REFERENCE_SHA256 = (
    "8b28b91f1bcd0556c4def54d9cd3cab75fe8c848f086b7dc4a8746985e0947ff")
_METAX_GEMV_BLOCK256_WAVE8_COMPLETION_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "gemv_block256_wave8_completion_maca_3_7_2.json")
_METAX_GEMV_BLOCK256_WAVE8_COMPLETION_REFERENCE_SHA256 = (
    "b2684092e0794067ff08c6c793ca3d756ad23a2937a490974f1b4a0a26341074")
_METAX_GEMM_BF16_MMA_ACCUMULATOR_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "gemm_bf16_mma_accumulator_sweep_maca_3_7_2.json")
_METAX_GEMM_BF16_MMA_ACCUMULATOR_REFERENCE_SHA256 = (
    "bd6a4f51e2bf6bfa07a29fb8d6b4295b3cdc706158be7bee541dd1df6b142cb6")
_METAX_FUSED_MOE_BM32_BF16_MMA_OCCUPANCY_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fused_moe_bm32_bf16_mma_occupancy_maca_3_7_2.json")
_METAX_FUSED_MOE_BM32_BF16_MMA_OCCUPANCY_REFERENCE_SHA256 = (
    "0a36e74aba1900d6c640752ea22c40d518fef639a44d7a96cdba0832025cd144")
_METAX_FUSED_MOE_BM32_OPERAND_PIPELINE_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fused_moe_bm32_grouped_operand_pipeline_maca_3_7_2.json")
_METAX_FUSED_MOE_BM32_OPERAND_PIPELINE_REFERENCE_SHA256 = (
    "1a7686c1cd901edc8d88118630791e4751ce9a194186bd6671d2c794a81433d7")
_METAX_FUSED_MOE_BM16_OPERAND_PIPELINE_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fused_moe_bm16_grouped_operand_pipeline_maca_3_7_2.json")
_METAX_FUSED_MOE_BM16_OPERAND_PIPELINE_REFERENCE_SHA256 = (
    "b49864a7030c30da027b3f147cbb20f9fd917e780345f624363cc596976e99c5")
_METAX_FUSED_MOE_BM128_BN128_OPERAND_PIPELINE_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fused_moe_bm128_bn128_grouped_operand_pipeline_maca_3_7_2.json")
_METAX_FUSED_MOE_BM128_BN128_OPERAND_PIPELINE_REFERENCE_SHA256 = (
    "6ca337cacd987e29c9c6736fb17cc885b5b1653c92ee3dd30921e478a914a0e1")
_METAX_FUSED_MOE_BM64_BN128_WARPS8_OPERAND_PIPELINE_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fused_moe_bm64_bn128_warps8_grouped_operand_pipeline_maca_3_7_2.json")
_METAX_FUSED_MOE_BM64_BN128_WARPS8_OPERAND_PIPELINE_REFERENCE_SHA256 = (
    "b169b3d561cfdbd966cc4943210ce9f87a0d79ab165db0f09b33eae2214c9fd5")
_METAX_FUSED_MOE_BM64_BN128_WARPS4_OPERAND_PIPELINE_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fused_moe_bm64_bn128_warps4_grouped_operand_pipeline_maca_3_7_2.json")
_METAX_FUSED_MOE_BM64_BN128_WARPS4_OPERAND_PIPELINE_REFERENCE_SHA256 = (
    "27fc374e9b6589cd4105bc3278283153cb0d075aaf39694b5586bd7eede49cc8")
_METAX_FUSED_MOE_BM32_BN128_OPERAND_PIPELINE_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fused_moe_bm32_bn128_grouped_operand_pipeline_maca_3_7_2.json")
_METAX_FUSED_MOE_BM32_BN128_OPERAND_PIPELINE_REFERENCE_SHA256 = (
    "189ed490fb8b5970e0e7658299034e79cee2692156900f4e18c255c678c4a6a8")
_METAX_FA_TILE16_OPERAND_COMPLETION_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fa_tile16_operand_completion_maca_3_7_2.json")
_METAX_FA_TILE16_OPERAND_COMPLETION_REFERENCE_SHA256 = (
    "de00b47a94504a7b92f3a5b68670e27fe40e5a66e2709c3106b3d2df3bac86d2")
_METAX_FA_TILE16_OPERAND_COMPLETION_D64_G16_P64_REFERENCE = Path(
    "microbench/metax_c500/references/"
    "fa_tile16_operand_completion_d64_g16_p64_maca_3_7_2.json")
_METAX_FA_TILE16_OPERAND_COMPLETION_D64_G16_P64_REFERENCE_SHA256 = (
    "841375bd89a38ada48bbcbcc73baaa120f8baa0b11519670ac031ee03cb091a7")
_AIV_GATHER_V3_TILING_MATERIALIZATION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "gather_v3_tiling_materialization_cann_8_5_1.json")
_AIV_GATHER_V3_TILING_MATERIALIZATION_REFERENCE_SHA256 = (
    "c588302c19b40923e1c9b4cbe6ee0cf5046442158be5d084d9a66f409a611cb7")
_TOP_LEVEL_FIELDS = {
    "hardware_id", "clock_domains", "components", "resource_groups",
    "memory_subsystem", "scheduling", "interconnect", "features",
}
_SCHEDULING_REQUIRED_FIELDS = {
    "parallel_unit", "parallel_width", "num_compute_units",
    "max_warps_per_unit", "max_blocks_per_unit", "max_regs_per_unit",
    "max_smem_per_unit_bytes", "max_smem_per_block_bytes",
    "max_threads_per_unit", "dispatch_overhead_ns",
    "model_clock_domain", "dram_clock_domain",
}
_SCHEDULING_OPTIONAL_FIELDS = {
    "prepared_moe_operand_resources",
    "register_allocation_unit_per_warp",
    "empty_kernel_duration_ns", "empty_kernel_duration_lut",
    "launch_latency_model", "cta_dispatch_cost",
    "memory_dependency_task_latency_model",
    "runtime_task_latency_models", "tensor_iterator_copy_task_latency_model",
    "aic_empty_kernel_duration_lut",
    "aiv_empty_kernel_duration_lut",
    "aiv_nonempty_task_latency_model",
    "mix_aiv_nonempty_task_latency_model",
    "aiv_worker_start_fill_model",
    "aiv_memory_task_latency_model",
    "aiv_tiling_materialization_latency_model",
    "aiv_tiling_materialization_80b_latency_model",
    "mixed_empty_kernel_duration_lut",
    "vector_reduce_sum_latency_lut",
    "register_reduction_dependency_cycles",
    "fa2_local_spill_dependency_cycles",
    "fa2_source_task_outer_envelope",
    "fa2_dense_d128_causal_local_spill_dependency_cycles",
    "cta_barrier_service_cycles",
    "sm80_depth3_cta_barrier_service_cycles",
    "flash_mla_combine_barrier_service_cycles",
    "imcontinuous_vec4_reduction_source_service",
    "fa2_combine_dependency_cycles",
    "metax_fa2_combine_dependency_cycles",
    "fa2_oaccum_load_use_cycles",
    "rmsnorm_shared_gamma_l1_residency",
    "add_rmsnorm_source_round_pipeline",
    "add_rmsnorm_block256_source_round_pipeline",
    "grouped_gemm_tile_pipeline",
    "grouped_gemm_fp8_block_tile_pipelines",
    "sglang_fp8_block_tile_pipelines",
    "finite_async_operand_stream_completion",
    "finite_mixed_operand_stream_startup",
    "l2_operand_residency_curve",
    "dram_finite_read_stream_ramp",
    "dram_finite_tma_read_stream_ramp",
    "small_kernel_span_floor",
    "cuda_graph_small_node_span",
    "cooperative_grid_sync_service",
    "split_merge_chunk_service",
    "stream_memcpy_service",
    "cuda_graph_side_stream_fork_node_boundary",
    # wp10 (H200 FA3 MLA): measured FlashAttention-3 mechanism services.
    "fa3_mla_mechanisms",
    "bf16_wgmma_tensor_service_clock",
    "fa3_gqa_wgmma_batches",
    "ss_ex2_busy_sharing",
    "warp_integer_dependencies",
    "packed_bf16_conversion",
    "moe_align_shared_atomic_e128",
    "moe_align_fill_bytes_per_s",
    "moe_activation_mixed_stream",
    "cp_async_mbarrier_notification",
    "fa3_parameter_reads",
    "fa3_gqa_compiled_dependencies",
    "async_copy_cold_ready",
    "early_exit_cta_retire",
    "gemm_bf16_mma_saturation",
    "gemm_tensor_activity_clock",
    "gemm_tensor_instruction_service",
    "ordinary_u16_store_completion",
    "metax_fused_moe_bm32_bf16_mma_occupancy",
    "h200_fused_moe_w2_active_sm_clock",
    "fa_tile16_operand_completion",
    "fia_cross_domain_three_handoff",
    "fia_aiv_sync_all_service",
    "fia_flash_decode_terminal_prelude_service",
    "fia_tiling_data_materialization_service",
    "fia_mixed_task_outer_envelope",
    "fia_aiv_workspace_transfer_service",
    "aic_mte2_nd2nz_service",
    "aic_mte2_operand_pair_stream_service",
    "aic_mte1_load_pair_completion_service",
    "aic_grouped_matmul_mte1_pair_completion_service",
    "aic_fixpipe_full_tile_service",
    "fia_block_reduce_sum_service",
    "fia_block_reduce_max_service",
    "fia_brcb_service",
    "fia_kv_line_fill_span",
    "block_fp8_gemm_tma_ring_service",
}
_FEATURE_REQUIRED_FIELDS = {
    "kernel_memory_model", "cluster_supported", "cluster_size",
    "kernel_dtype", "l1_cache_enabled", "multicast_supported",
    "hw_decompression", "compression_ratio",
}
_FEATURE_OPTIONAL_FIELDS = {
    "analytical_vector_dependency_paths", "gpu_store_completion_scope",
    "async_output_release",
    "full_graph_physical_extent_dispatch",
    "full_graph_source_program_physical_extent",
    "piecewise_graph_physical_extent_dispatch"}
_INTERCONNECT_FIELDS = {
    "nvlink_unidir_bw_GB_s", "nvlink_signal_latency_ns", "nccl_alpha_ns",
    "pcie_bw_GB_s", "pcie_latency_ns", "infiniband_unidir_bw_GB_s",
}
_INTERCONNECT_OPTIONAL_FIELDS = {
    "custom_all_reduce_service", "sglang_custom_all_reduce_service",
    "sglang_flashinfer_allreduce_fusion_service"}
_CUSTOM_ALL_REDUCE_MODELS = {
    "custom_all_reduce_service": "vllm_custom_all_reduce_graph_node_v1",
    "sglang_custom_all_reduce_service":
        "sglang_custom_all_reduce_v2_graph_node_v1",
    # flashinfer trtllm all-reduce + residual RMSNorm (kARResidualRMSNorm),
    # SGLang's auto-enabled fusion; max_payload_bytes bounds the measured
    # token range, not a runtime fallback size.
    "sglang_flashinfer_allreduce_fusion_service":
        "sglang_flashinfer_allreduce_residual_rmsnorm_graph_node_v1",
}


def _normalize_custom_all_reduce_service(
        value, field="custom_all_reduce_service"):
    """Normalise the measured vLLM custom all-reduce graph-node service.

    Pinned vLLM selects its IPC custom all-reduce for a fully NVLink-connected
    tensor-parallel group whenever the payload is below ``max_size``.  An
    independent microbenchmark chains that kernel inside one CUDA graph and
    publishes its per-collective service time by participant count and
    payload.  A request steps down to the largest sampled payload at or below
    it, which chooses no interpolation shape; a request below the smallest
    sample takes the smallest sample, the measured latency floor.
    """

    if not isinstance(value, dict):
        raise TypeError(f"interconnect.{field} must be a mapping")
    expected = {
        "model": _CUSTOM_ALL_REDUCE_MODELS[field],
        "duration_unit": "ns_per_collective",
        "timer_domain": "cuda_event_around_graph_replay_divided_by_chain",
        "lookup_rule": (
            "step_down_largest_sampled_payload_not_exceeding_request_"
            "floor_at_smallest_sample"),
    }
    _require_exact_fields(
        f"interconnect.{field}", value,
        set(expected) | {
            "software_revision", "max_payload_bytes", "reference_path",
            "reference_sha256", "production_or_evaluation_duration_consumed",
            "service_ns_by_participants"})
    for name, want in expected.items():
        if value[name] != want:
            raise ValueError(
                f"interconnect.{field}.{name} must be {want!r}")
    if value["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(
            "custom all-reduce service must not consume production or "
            "evaluation durations")
    max_payload = value["max_payload_bytes"]
    if isinstance(max_payload, bool) or not isinstance(max_payload, int) or max_payload <= 0:
        raise ValueError(
            "custom_all_reduce_service.max_payload_bytes must be a positive "
            "integer")
    by_participants = value["service_ns_by_participants"]
    if not isinstance(by_participants, dict) or not by_participants:
        raise TypeError(
            "custom_all_reduce_service.service_ns_by_participants must be a "
            "non-empty mapping")
    tables = {}
    for raw_participants, raw_table in by_participants.items():
        participants = int(raw_participants)
        if participants < 2:
            raise ValueError(
                "custom all-reduce participant count must be at least two")
        if not isinstance(raw_table, dict) or not raw_table:
            raise TypeError(
                "custom all-reduce service table must be a non-empty mapping")
        rows = []
        for raw_bytes in sorted(raw_table, key=lambda item: int(item)):
            payload = int(raw_bytes)
            service = float(raw_table[raw_bytes])
            if payload <= 0 or payload >= max_payload:
                raise ValueError(
                    "custom all-reduce sampled payload must be positive and "
                    "below max_payload_bytes")
            if not math.isfinite(service) or service <= 0:
                raise ValueError(
                    "custom all-reduce service must be finite and positive")
            rows.append((payload, service))
        tables[participants] = tuple(rows)
    return {"max_payload_bytes": max_payload, "tables": tables}


def _require_no_duration_consumption(value, owner) -> None:
    """Require an evidence record to declare that it consumes no duration.

    The forward model may read only duration-free evidence; a record that
    admits consuming a production or evaluation duration is rejected at the
    boundary rather than silently shaping a prediction.
    """

    if value["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(
            f"{owner} cannot consume production or evaluation duration")


def _read_owned_reference(value, owner, hardware_id, repository_root) -> bytes:
    """Read a measured reference after checking its contract and its owner.

    Every measured mechanism records a duration-free evidence file inside
    ``microbench/<hardware_id>/references/``; this reads it only after the
    record declares no duration consumption, the path stays inside its owner,
    and the content matches the recorded SHA256.
    """

    _require_no_duration_consumption(value, owner)
    reference_path = Path(value["reference_path"])
    _require_owned_reference_path(reference_path, owner, hardware_id)
    payload = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != value["reference_sha256"]:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    return payload


def _normalize_async_copy_cold_ready(value, repository_root, hardware_id):
    """Independent fixed-latency / finite-bandwidth copy-only services."""
    owner = "scheduling.async_copy_cold_ready"
    _require_exact_fields(owner, value, {
        "model", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed"})
    if value["model"] != "isolated_cold_async_copy_ready_v1":
        raise ValueError(f"{owner} model is unsupported")
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    if (ref["hardware_id"] != hardware_id or ref["mechanism"] != "cold_async_copy_ready"
            or ref["production_or_evaluation_duration_consumed"] is not False
            or type(ref["producer_threads"]) is not int or ref["producer_threads"] != 128
            or type(ref["resident_ctas_per_sm"]) is not int or ref["resident_ctas_per_sm"] != 1):
        raise ValueError(f"{owner} reference domain differs")
    for name in ("minimum_payload_bytes", "maximum_payload_bytes"):
        if type(ref[name]) is not int or ref[name] <= 0:
            raise ValueError(f"{owner} payload domain is invalid")
    if ref["minimum_payload_bytes"] > ref["maximum_payload_bytes"]:
        raise ValueError(f"{owner} payload interval is reversed")
    _require_exact_fields(owner + ".profiles", ref["profiles"], {"cp_async", "tma"})
    profiles = {}
    for path, rows in ref["profiles"].items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{owner} requires nonempty path services")
        previous = 0
        previous_bandwidth = 0.0
        normalized = []
        for row in rows:
            _require_exact_fields(owner + ".profile", row, {
                "producers", "latency_ns", "aggregate_bandwidth_bytes_s"})
            producers = row["producers"]
            if type(producers) is not int or producers <= previous:
                raise ValueError(f"{owner} producer counts must increase")
            previous = producers
            for name in ("latency_ns", "aggregate_bandwidth_bytes_s"):
                number = row[name]
                if (isinstance(number, bool) or not isinstance(number, (int, float))
                        or not math.isfinite(number) or number <= 0):
                    raise ValueError(f"{owner} service must be finite and positive")
            if row["aggregate_bandwidth_bytes_s"] < previous_bandwidth:
                raise ValueError(f"{owner} aggregate capacity must not decrease")
            previous_bandwidth = row["aggregate_bandwidth_bytes_s"]
            normalized.append(dict(row))
        profiles[path] = tuple(normalized)
    return {"minimum_payload_bytes": ref["minimum_payload_bytes"],
            "maximum_payload_bytes": ref["maximum_payload_bytes"],
            "profiles": profiles}
def _normalize_prepared_moe_operand_resources(value, repository_root, hardware_id):
    owner = "scheduling.prepared_moe_operand_resources"
    _require_exact_fields(owner, value, {"model", "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    if value["model"] != "prepared_bf16_ep_operand_resources_v1" or hardware_id != "nvidia_h200":
        raise ValueError("prepared operand resource model requires H200")
    reference = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    if (set(reference) != {"hardware_id", "production_or_evaluation_duration_consumed", "domain", "resources", "evidence", "transfer_limitations"}
            or reference["hardware_id"] != "nvidia_h200"
            or reference["production_or_evaluation_duration_consumed"] is not False
            or set(reference["resources"]) != {"w1", "w2"}):
        raise ValueError("prepared operand resource reference has an invalid domain")
    expected_domain = {"minimum_tokens", "maximum_tokens", "experts", "local_experts", "topk", "hidden", "intermediate",
        "tile_m", "tile_n", "tile_k", "group_size_m", "threads_per_cta", "dtype", "expert_parallel_size"}
    domain = reference["domain"]
    if set(domain) != expected_domain or domain["dtype"] != "bf16":
        raise ValueError("prepared operand resource domain fields differ")
    if any(type(value) is not int or value <= 0 for name, value in domain.items() if name != "dtype"):
        raise ValueError("prepared operand resource extents must be positive integers")
    if domain["minimum_tokens"] > domain["maximum_tokens"]:
        raise ValueError("prepared operand resource token interval is reversed")
    evidence = reference["evidence"]
    if set(evidence) != {"summary_path", "summary_sha256", "captured_tokens", "source_ring_reference"}:
        raise ValueError("prepared operand resource evidence fields differ")
    summary_path = Path(evidence["summary_path"])
    if (summary_path.is_absolute() or ".." in summary_path.parts
            or summary_path.parts[:3] != ("profiler_output", "kernel_dataset", hardware_id)
            or not isinstance(evidence["summary_sha256"], str) or len(evidence["summary_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in evidence["summary_sha256"])):
        raise ValueError("prepared operand resource evidence is outside its owner or lacks a SHA256")
    samples = evidence["captured_tokens"]
    if (not isinstance(samples, list) or not samples
            or any(type(value) is not int or not domain["minimum_tokens"] <= value <= domain["maximum_tokens"] for value in samples)
            or len(set(samples)) != len(samples)):
        raise ValueError("prepared operand resource samples differ from its domain")
    ring_path = Path(evidence["source_ring_reference"])
    if ring_path.name != evidence["source_ring_reference"] or ring_path.suffix != ".json":
        raise ValueError("prepared operand ring evidence must name its reference owner")
    if (not isinstance(reference["transfer_limitations"], list) or not reference["transfer_limitations"]
            or any(not isinstance(value, str) or not value.strip() for value in reference["transfer_limitations"])):
        raise ValueError("prepared operand resource transfer limitations are required")
    for resource in reference["resources"].values():
        if (set(resource) != {"registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes", "operand_ring_slots"}
                or any(type(value) is not int or value < 0 for value in resource.values())
                or resource["registers_per_thread"] <= 0 or resource["operand_ring_slots"] <= 0
                or resource["dynamic_shared_bytes"] != resource["operand_ring_slots"]
                    * (domain["tile_m"] + domain["tile_n"]) * domain["tile_k"] * 2):
            raise ValueError("prepared operand resource footprint is inconsistent")
    return reference


def _require_owned_reference_path(reference_path, owner, hardware_id) -> None:
    """Require a measured reference to live under its own hardware directory.

    Every measured-mechanism reference is stored at
    ``microbench/<hardware_id>/references/...``; a path that escapes that
    directory is a contract violation rather than a look-up failure.
    """

    if (reference_path.is_absolute() or ".." in reference_path.parts
            or reference_path.parts[:3] != (
                "microbench", hardware_id, "references")):
        raise ValueError(f"{owner}.reference_path is outside its owner")


# Each measured-mechanism label names the toolchain it was captured with.
# A card re-measured with the same probe records its own label; the pinned
# A800 label admits exactly the H200 re-measurement's label and nothing else.
_REMEASURED_SOFTWARE_REVISIONS = {
    "cuda_12_2_sm80": ("cuda_12_2_sm80", "cuda_12_2_sm90"),
    "cuda_12.2_sm80": ("cuda_12.2_sm80", "cuda_12.2_sm90"),
    "flash_attn_2_8_3_cuda_12_2_sm80": (
        "flash_attn_2_8_3_cuda_12_2_sm80",
        "flash_attn_2_8_3_source_cuda_12_2_sm90_probe"),
    "pytorch_2.8.0_cuda_12.8_sm80_source_semantics": (
        "pytorch_2.8.0_cuda_12.8_sm80_source_semantics",
        "pytorch_2.8.0_cuda_12.8_sm90_source_semantics"),
    "torch_2.8.0+cu128_cuda_sm80": (
        "torch_2.8.0+cu128_cuda_sm80", "torch_2.11.0+cu129_cuda_sm90"),
    "vllm_0.11.0_cuda_12.2_sm80": (
        "vllm_0.11.0_cuda_12.2_sm80", "vllm_0.11.0_source_cuda_12.9_sm90"),
}


def _accepted_software_revisions(pinned):
    """Return the labels a pinned A800 software revision admits."""
    if pinned in _REMEASURED_SOFTWARE_REVISIONS:
        return _REMEASURED_SOFTWARE_REVISIONS[pinned]
    return (pinned,)


def _text_field_matches(name, actual, expected):
    """Exact text match, except a software revision admits its re-measurement."""
    if name == "software_revision" and isinstance(expected, str):
        return actual in _accepted_software_revisions(expected)
    return actual == expected


# Measurement provenance a reference may carry beside its contract fields.
_REFERENCE_OPTIONAL_FIELDS = frozenset({"provenance"})


def _require_fields_with_optional(section, values, required, optional):
    missing = required - set(values)
    unknown = set(values) - required - optional
    if missing or unknown:
        raise ValueError(
            f"{section} schema mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")


def _validate_top_level_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("hardware config root must be a mapping")
    _require_exact_fields("hardware config", cfg, _TOP_LEVEL_FIELDS)
    if (isinstance(cfg["features"], dict) and "kernel_memory_model" in cfg["features"]
            and cfg["features"]["kernel_memory_model"] == "roofline"):
        _require_fields_with_optional("roofline features", cfg["features"],
            {"kernel_memory_model", "kernel_dtype", "parameter_source"}, {"fpga_execution", "gpu_vector_memory_only", "async_output_release"})
        if "gpu_vector_memory_only" in cfg["features"]:
            if (cfg["hardware_id"] not in ("hygon_bw1100", "nvidia_rtx_pro_5500")
                    or type(cfg["features"]["gpu_vector_memory_only"]) is not bool):
                raise ValueError("gpu_vector_memory_only requires BW1100 or RTX PRO 5500 and an explicit boolean")
        if cfg["features"]["kernel_dtype"] not in ("bf16", "fp16", "fp8", "int8"):
            raise ValueError("roofline profile requires an explicit supported primary dtype")
        source = cfg["features"]["parameter_source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("roofline profile requires its parameter source")
        if "fpga_execution" in cfg["features"]:
            if cfg["hardware_id"] != "intel_agmf039r47":
                raise ValueError("FPGA execution parameters require their declared Intel hardware identity")
            execution = cfg["features"]["fpga_execution"]
            if not isinstance(execution, dict):
                raise ValueError("fpga_execution must be a mapping")
            _require_exact_fields("fpga_execution", execution, {
                "device_scope", "chips_per_card", "clock_hz", "mm_m_parallel",
                "mm_k_parallel", "mm_n_parallel", "vector_elements_per_cycle",
                "vector_latency_cycles", "hbm_latency_ns", "receive_compute_send_overlap",
                "ddr_usage", "ddr_reference"})
            for name in ("chips_per_card", "clock_hz", "mm_m_parallel", "mm_k_parallel",
                         "mm_n_parallel", "vector_elements_per_cycle", "vector_latency_cycles"):
                if type(execution[name]) is not int or execution[name] <= 0:
                    raise ValueError(f"fpga_execution.{name} must be a positive integer")
            if (execution["device_scope"] != "fpga_chip"
                    or execution["ddr_usage"] != "communication_staging_only"
                    or type(execution["receive_compute_send_overlap"]) is not bool):
                raise ValueError("FPGA requires explicit chip scope, communication-only DDR and boolean overlap capability")
            latency = execution["hbm_latency_ns"]
            if (isinstance(latency, bool) or not isinstance(latency, (float, int))
                    or not math.isfinite(latency) or latency <= 0):
                raise ValueError("FPGA HBM latency must be finite and positive")
            reference = execution["ddr_reference"]
            if not isinstance(reference, dict):
                raise ValueError("DDR reference must be a mapping")
            _require_exact_fields("DDR reference", reference, {"source_url", "reference_kind",
                "clock_hz", "cas_cycles", "rcd_cycles", "rp_cycles", "controller_latency_ns"})
            if (not isinstance(reference["source_url"], str) or not reference["source_url"].startswith("https://")
                    or reference["reference_kind"] != "ddr4_3200_cl22_22_22_device_only"):
                raise ValueError("DDR reference requires a source and an explicit device-only timing scope")
            for name in ("clock_hz", "cas_cycles", "rcd_cycles", "rp_cycles"):
                if type(reference[name]) is not int or reference[name] <= 0:
                    raise ValueError(f"DDR reference {name} must be a positive integer")
            if (reference["clock_hz"], reference["cas_cycles"], reference["rcd_cycles"], reference["rp_cycles"]) != (1600000000, 22, 22, 22):
                raise ValueError("DDR4-3200 CL22-22-22 reference identity disagrees with its timing fields")
            if reference["controller_latency_ns"] is not None:
                raise ValueError("the reference DDR4 device does not establish FPGA controller latency")
        for section in ("clock_domains", "memory_subsystem", "scheduling"):
            if not isinstance(cfg[section], dict) or cfg[section]:
                raise ValueError(f"aggregate roofline profile cannot declare native {section}")
        ic = cfg["interconnect"]
        if not isinstance(ic, dict):
            raise ValueError("aggregate interconnect must be a mapping")
        if ic:
            _require_fields_with_optional("aggregate interconnect", ic, {
                "roce_unidir_bw_GB_s", "roce_devices_per_nic",
                "roce_switch_topology", "roce_signal_latency_ns",
                "roce_collective_startup_ns", "parameter_source"}, {"pcie_bw_bytes_s"})
            if "pcie_bw_bytes_s" in ic:
                value = ic["pcie_bw_bytes_s"]
                if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                    raise ValueError("aggregate PCIe bandwidth must be finite positive bytes/s")
            for name in ("roce_unidir_bw_GB_s", "roce_signal_latency_ns",
                         "roce_collective_startup_ns"):
                value = ic[name]
                if value is None and name != "roce_unidir_bw_GB_s":
                    continue
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"interconnect.{name} must be finite and positive or an optional null latency")
            if type(ic["roce_devices_per_nic"]) is not int or ic["roce_devices_per_nic"] <= 0:
                raise ValueError("roce_devices_per_nic must be a positive integer")
            if ic["roce_switch_topology"] != "fully_connected":
                raise ValueError("aggregate RoCE requires an explicitly nonblocking fully_connected switch")
            if not isinstance(ic["parameter_source"], str) or not ic["parameter_source"].strip():
                raise ValueError("aggregate interconnect requires its parameter source")
        if (not isinstance(cfg["components"], list) or not cfg["components"]
                or cfg["resource_groups"] != []):
            raise ValueError("aggregate roofline profile requires resources and no native resource groups")
        return
    if not isinstance(cfg["clock_domains"], dict) or not cfg["clock_domains"]:
        raise ValueError("clock_domains must be a non-empty mapping")
    for name, frequency in cfg["clock_domains"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("clock domain names must be non-empty strings")
        if (isinstance(frequency, bool)
                or not isinstance(frequency, (int, float))):
            raise TypeError(
                f"clock domain {name!r} frequency must be numeric")
        value = float(frequency)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"clock domain {name!r} must be finite and positive")
    if not isinstance(cfg["components"], list) or not cfg["components"]:
        raise ValueError("components must be a non-empty list")
    if not isinstance(cfg["resource_groups"], list) or not cfg["resource_groups"]:
        raise ValueError("resource_groups must be a non-empty list")
    for section in ("memory_subsystem", "scheduling", "interconnect", "features"):
        if not isinstance(cfg[section], dict):
            raise ValueError(f"{section} must be a mapping")
    _require_fields_with_optional(
        "scheduling", cfg["scheduling"],
        _SCHEDULING_REQUIRED_FIELDS, _SCHEDULING_OPTIONAL_FIELDS)
    _require_fields_with_optional(
        "features", cfg["features"],
        _FEATURE_REQUIRED_FIELDS, _FEATURE_OPTIONAL_FIELDS)
    _require_fields_with_optional(
        "interconnect", cfg["interconnect"],
        _INTERCONNECT_FIELDS, _INTERCONNECT_OPTIONAL_FIELDS)


def _normalize_empty_kernel_duration_lut(value):
    if not isinstance(value, dict):
        raise TypeError("scheduling.empty_kernel_duration_lut must be a mapping")
    interpolated_fields = {"grid_points", "duration_ns_by_block"}
    exact_fields = {"geometry_policy", "duration_unit", "entries"}
    if set(value) == exact_fields:
        if value["geometry_policy"] != "exact_grid_and_block_only":
            raise ValueError(
                "exact empty-kernel LUT requires exact_grid_and_block_only"
            )
        if value["duration_unit"] != "ns_per_launch":
            raise ValueError(
                "exact empty-kernel LUT duration_unit must be ns_per_launch"
            )
        entries = value["entries"]
        if not isinstance(entries, list) or not entries:
            raise TypeError(
                "exact empty-kernel LUT entries must be a non-empty list"
            )
        normalized = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("exact empty-kernel LUT entries must be mappings")
            _require_exact_fields(
                "exact empty-kernel LUT entry", entry,
                {"grid", "block", "duration_ns"},
            )
            dimensions = []
            for name in ("grid", "block"):
                dim3 = entry[name]
                if (not isinstance(dim3, (list, tuple)) or len(dim3) != 3
                        or any(isinstance(axis, bool)
                               or not isinstance(axis, int) or axis <= 0
                               for axis in dim3)):
                    raise ValueError(
                        f"exact empty-kernel LUT {name} must be a positive dim3"
                    )
                dimensions.append(tuple(dim3))
            grid, block = dimensions
            if math.prod(block) > 1024:
                raise ValueError(
                    "exact empty-kernel LUT block cannot exceed 1024 threads"
                )
            duration = entry["duration_ns"]
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))):
                raise TypeError(
                    "exact empty-kernel LUT durations must be numeric"
                )
            normalized_duration = float(duration)
            if (not math.isfinite(normalized_duration)
                    or normalized_duration <= 0):
                raise ValueError(
                    "exact empty-kernel LUT durations must be finite and positive"
                )
            identity = (grid, block)
            if identity in normalized:
                raise ValueError(
                    "exact empty-kernel LUT contains a duplicate geometry"
                )
            normalized[identity] = normalized_duration
        return "exact_dim3", normalized
    if set(value) != interpolated_fields:
        _require_exact_fields(
            "scheduling.empty_kernel_duration_lut", value,
            interpolated_fields,
        )
    grid_points = value["grid_points"]
    if not isinstance(grid_points, list) or len(grid_points) < 2:
        raise TypeError(
            "empty_kernel_duration_lut.grid_points must contain at least two values")
    if any(isinstance(point, bool) or not isinstance(point, int)
           for point in grid_points):
        raise TypeError("dispatch overhead grid points must be integers")
    if grid_points[0] != 1 or any(
            left >= right for left, right in zip(grid_points, grid_points[1:])):
        raise ValueError(
            "dispatch overhead grid points must start at one and increase")

    duration_by_block = value["duration_ns_by_block"]
    if not isinstance(duration_by_block, dict) or not duration_by_block:
        raise TypeError(
            "empty_kernel_duration_lut.duration_ns_by_block must be a non-empty mapping")
    normalized = {}
    for block_size, durations in duration_by_block.items():
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError("dispatch overhead block sizes must be integers")
        if block_size <= 0:
            raise ValueError("dispatch overhead block sizes must be positive")
        if not isinstance(durations, list) or len(durations) != len(grid_points):
            raise ValueError(
                "each dispatch overhead duration curve must match grid_points")
        curve = []
        for duration in durations:
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))):
                raise TypeError("dispatch overhead durations must be numeric")
            normalized_duration = float(duration)
            if not math.isfinite(normalized_duration) or normalized_duration <= 0:
                raise ValueError(
                    "dispatch overhead durations must be finite and positive")
            curve.append(normalized_duration)
        normalized[block_size] = tuple(curve)
    block_points = tuple(sorted(normalized))
    if block_points[-1] > 1024:
        raise ValueError("CUDA dispatch LUT block sizes cannot exceed 1024")
    return "interpolated_flat", tuple(grid_points), block_points, normalized


def _normalize_cta_dispatch_cost(value):
    """Normalise the measured cost of dispatching a launch's CTAs.

    The launch model bills a full-chip round, so CTA count inside one wave is
    free there.  A geometry-controlled probe measures the real cost and finds
    it independent of block size and of dynamic shared memory, and additive
    rather than hidden by execution, so a card that measures it publishes one
    table over grid size and it applies to any launch.  Beyond a wave the cost
    becomes residency-dependent, which cta_round_ns already models, so the
    table stops at the card's full-chip width and larger grids reuse its last
    entry.  Cards publishing nothing keep their previous behaviour.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("scheduling.cta_dispatch_cost must be a mapping")
    expected = {
        "model": "geometry_independent_cta_dispatch_cost_v1",
        "duration_unit": "ns_per_launch",
        "timer_domain": "paired_differential_cuda_graph_replay",
        "derivation_rule": "median_over_block_sizes_minus_single_cta",
        "lookup_rule": "step_down_largest_sampled_grid_not_exceeding_request",
    }
    _require_exact_fields(
        "scheduling.cta_dispatch_cost", value,
        set(expected) | {"software_revision", "cost_ns_by_grid"})
    for name, want in expected.items():
        if value[name] != want:
            raise ValueError(
                f"scheduling.cta_dispatch_cost.{name} must be {want!r}")
    table = value["cost_ns_by_grid"]
    if not isinstance(table, dict) or not table:
        raise TypeError(
            "scheduling.cta_dispatch_cost.cost_ns_by_grid must be a "
            "non-empty mapping")
    normalized = {}
    previous = None
    for key in sorted(table, key=lambda item: int(item)):
        grid = int(key)
        cost = table[key]
        if grid <= 0:
            raise ValueError(
                "scheduling.cta_dispatch_cost grid must be positive")
        if (isinstance(cost, bool) or not isinstance(cost, (int, float))
                or not math.isfinite(cost) or cost < 0.0):
            raise ValueError(
                "scheduling.cta_dispatch_cost cost must be non-negative")
        if previous is not None and cost < previous:
            raise ValueError(
                "scheduling.cta_dispatch_cost must be non-decreasing in "
                "grid: dispatching more CTAs cannot cost less")
        normalized[grid] = float(cost)
        previous = float(cost)
    if min(normalized) != 1 or normalized[1] != 0.0:
        raise ValueError(
            "scheduling.cta_dispatch_cost must start at grid 1 with zero "
            "cost, since it is measured against a single CTA")
    return {
        **{name: value[name] for name in
           ("model", "duration_unit", "timer_domain", "software_revision",
            "derivation_rule", "lookup_rule")},
        "cost_ns_by_grid": normalized,
    }


def _normalize_launch_latency_model(value):
    """Normalize one isolated hardware-wide launch scheduling model.

    The tagged variants describe different independently measured scheduler
    mechanisms.  Neither schema can carry an operator, kernel-name, dtype,
    tile, or shape discriminator; those would turn launch timing into a
    production residual correction rather than a mechanism parameter.
    """

    if not isinstance(value, dict):
        raise TypeError("scheduling.launch_latency_model must be a mapping")
    if "model" not in value:
        raise ValueError(
            "scheduling.launch_latency_model requires a model tag")
    model = value["model"]
    if not isinstance(model, str):
        raise TypeError(
            "scheduling.launch_latency_model.model must be a string")
    common_fields = {"model", "duration_unit", "timer_domain"}
    if model == "isolated_empty_task_scheduler_v1":
        metadata_fields = set()
        coefficient_fields = {
            "submit_ns", "cta_round_ns", "wave_round_ns",
            "extra_grid_dimension_ns", "extra_block_dimension_ns",
        }
        timer_domains = {"nsys_gpu_kernel"}
        positive_fields = {"submit_ns", "cta_round_ns", "wave_round_ns"}
    elif model == "isolated_nonempty_task_scheduler_v1":
        metadata_fields = {"software_revision"}
        coefficient_fields = {
            "submit_ns", "cta_descriptor_ns", "single_wave_round_ns",
            "multi_wave_round_ns",
        }
        timer_domains = {
            "mctracer_gpu_kernel_event", "mcevent_gpu_kernel"}
        positive_fields = coefficient_fields
    else:
        raise ValueError(
            "scheduling.launch_latency_model.model is unsupported: "
            f"{model!r}")
    fields = common_fields | metadata_fields | coefficient_fields
    _require_exact_fields("scheduling.launch_latency_model", value, fields)
    normalized_metadata = {}
    for name in metadata_fields:
        metadata = value[name]
        if not isinstance(metadata, str):
            raise TypeError(
                f"scheduling.launch_latency_model.{name} must be a string")
        if not metadata.strip():
            raise ValueError(
                f"scheduling.launch_latency_model.{name} must be non-empty")
        normalized_metadata[name] = metadata
    if value["duration_unit"] != "ns_per_launch":
        raise ValueError(
            "scheduling.launch_latency_model.duration_unit must be "
            "'ns_per_launch'")
    timer_domain = value["timer_domain"]
    if not isinstance(timer_domain, str):
        raise TypeError(
            "scheduling.launch_latency_model.timer_domain must be a string")
    if timer_domain not in timer_domains:
        raise ValueError(
            "scheduling.launch_latency_model.timer_domain is unsupported: "
            f"{timer_domain!r}")
    coefficients = {}
    for name in coefficient_fields:
        coefficient = value[name]
        if (isinstance(coefficient, bool)
                or not isinstance(coefficient, (int, float))):
            raise TypeError(
                f"scheduling.launch_latency_model.{name} must be numeric")
        normalized = float(coefficient)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError(
                f"scheduling.launch_latency_model.{name} must be finite "
                "and non-negative")
        coefficients[name] = normalized
    if any(coefficients[name] <= 0 for name in positive_fields):
        raise ValueError(
            "scheduling.launch_latency_model required services must be "
            "positive")
    return {
        "model": model,
        "duration_unit": value["duration_unit"],
        "timer_domain": timer_domain,
        **normalized_metadata,
        **coefficients,
    }


def _normalize_fa2_source_task_outer_envelope(value):
    """Normalize the bounded SM80 source-pipeline task outer envelope."""

    owner = "scheduling.fa2_source_task_outer_envelope"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    expected = {
        "model", "duration_unit", "timer_domain", "software_revision",
        "threads_per_cta", "minimum_grid_size", "maximum_grid_size",
        "maximum_outer_envelope_ns", "source_sha256", "binary_sha256",
    }
    _require_exact_fields(owner, value, expected)
    text = {
        "model": "nsys_source_task_outer_envelope_v1",
        "duration_unit": "ns_per_launch",
        "timer_domain": "nsys_gpu_kernel_and_globaltimer_cta_envelope",
        "software_revision": "cuda_12_2_sm80",
    }
    for name, expected_value in text.items():
        if name == "software_revision":
            if value[name] not in _accepted_software_revisions(expected_value):
                raise ValueError(f"{owner}.{name} must be {expected_value!r}")
            text[name] = value[name]
        elif value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    if value["threads_per_cta"] != 128:
        raise ValueError(f"{owner}.threads_per_cta must be 128")
    for name in ("minimum_grid_size", "maximum_grid_size"):
        if (isinstance(value[name], bool) or not isinstance(value[name], int)
                or value[name] <= 0):
            raise ValueError(f"{owner}.{name} must be a positive integer")
    if value["minimum_grid_size"] > value["maximum_grid_size"]:
        raise ValueError(f"{owner} grid range is inverted")
    outer = value["maximum_outer_envelope_ns"]
    if (isinstance(outer, bool) or not isinstance(outer, (int, float))
            or not math.isfinite(float(outer)) or float(outer) <= 0.0):
        raise ValueError(f"{owner}.maximum_outer_envelope_ns must be positive")
    digests = {}
    for name in ("source_sha256", "binary_sha256"):
        digest = value[name]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
        digests[name] = digest
    return {
        **text,
        "threads_per_cta": value["threads_per_cta"],
        "minimum_grid_size": value["minimum_grid_size"],
        "maximum_grid_size": value["maximum_grid_size"],
        "maximum_outer_envelope_ns": float(outer),
        **digests,
    }


def _normalize_rmsnorm_shared_gamma_l1_residency(value):
    """Normalize the source-bound A800 RMSNorm shared-gamma contract."""

    owner = "scheduling.rmsnorm_shared_gamma_l1_residency"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "source_shared_gamma_residency_tier_v2",
        "byte_unit": "bf16_gamma_bytes_per_cta",
        "software_revision": "vllm_0.11.0_cuda_12.2_sm80",
        "dtype": "bf16",
    }
    integer_fields = {
        "threads_per_cta": 1024,
        "registers_per_thread": 24,
        "static_shared_memory_bytes_per_cta": 168,
        "resident_ctas_per_sm": 2,
        "launch_lifetime_maximum_gamma_bytes": 4096,
        "maximum_evidenced_gamma_bytes": 32768,
    }
    fields = set(text_fields) | set(integer_fields) | {
        "upstream_source_sha256", "reference_sha256"}
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    for name, expected in integer_fields.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    for name in ("upstream_source_sha256", "reference_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
    return {**text_fields, **integer_fields,
            "upstream_source_sha256": value["upstream_source_sha256"],
            "reference_sha256": value["reference_sha256"]}


def _normalize_add_rmsnorm_source_round_pipeline(value):
    """Normalize one source-bound SM80 Add-RMSNorm phase composition."""

    owner = "scheduling.add_rmsnorm_source_round_pipeline"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "sm80_bf16_add_rmsnorm_block1024_one_wave_v8",
        "software_revision": "vllm_0.11.0_cuda_12.2_sm80",
        "dtype": "bf16",
        "timer_domain": "clock64_same_cta",
    }
    integer_fields = {
        "hidden_size": 2048,
        "additional_hidden_size": 2560,
        "second_additional_hidden_size": 4096,
        "third_additional_hidden_size": 5120,
        "fourth_additional_hidden_size": 8192,
        "vector_width_elements": 8,
        "active_threads_per_source_round": 256,
        "additional_active_threads_per_source_round": 320,
        "second_additional_active_threads_per_source_round": 512,
        "third_additional_active_threads_per_source_round": 640,
        "fourth_additional_active_threads_per_source_round": 1024,
        "threads_per_cta": 1024,
        "registers_per_thread": 32,
        "static_shared_memory_bytes_per_cta": 168,
        "resident_ctas_per_sm": 2,
    }
    curve_fields = (
        "active_unit_points",
        "additional_active_unit_points",
        "second_additional_active_unit_points",
        "third_additional_active_unit_points",
        "fourth_additional_active_unit_points",
    )
    fields = (set(text_fields) | set(integer_fields) | set(curve_fields)
              | {"maximum_one_wave_grid_blocks"} | {
        "production_function_sass_sha256", "reference_sha256",
        "fourth_additional_reference_sha256",
        "h4096_grid64_cold_entry_reference_sha256"})
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    for name, expected in integer_fields.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    maximum_one_wave_grid_blocks = value["maximum_one_wave_grid_blocks"]
    if (isinstance(maximum_one_wave_grid_blocks, bool)
            or not isinstance(maximum_one_wave_grid_blocks, int)
            or maximum_one_wave_grid_blocks not in (108, 132)):
        raise ValueError(
            f"{owner}.maximum_one_wave_grid_blocks must be 108 or 132")
    for name in (
            "production_function_sass_sha256", "reference_sha256",
            "fourth_additional_reference_sha256",
            "h4096_grid64_cold_entry_reference_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
    # Source-evidence coordinates: the last point is one full wave of the
    # card that measured the curve (108 A800 SMs, 132 H200 SMs).
    one_wave = maximum_one_wave_grid_blocks
    expected_curves = {
        "active_unit_points": [1, 4, 16, one_wave],
        "additional_active_unit_points": [1, 4, 16, one_wave],
        "second_additional_active_unit_points": [1, 4, 16, one_wave],
        "third_additional_active_unit_points": [1, 4, 8, 16, one_wave],
        "fourth_additional_active_unit_points": [1, 4, 8, 16, one_wave],
    }
    curves = {}
    for curve_field in curve_fields:
        curve = value[curve_field]
        expected_curve = expected_curves[curve_field]
        if (not isinstance(curve, list)
                or any(isinstance(point, bool) or not isinstance(point, int)
                       for point in curve)
                or curve != expected_curve):
            raise ValueError(
                f"{owner}.{curve_field} must equal {expected_curve}")
        curves[curve_field] = tuple(curve)
    return {
        **text_fields,
        **integer_fields,
        "maximum_one_wave_grid_blocks": maximum_one_wave_grid_blocks,
        **curves,
        "production_function_sass_sha256": value[
            "production_function_sass_sha256"],
        "reference_sha256": value["reference_sha256"],
        "fourth_additional_reference_sha256": value[
            "fourth_additional_reference_sha256"],
        "h4096_grid64_cold_entry_reference_sha256": value[
            "h4096_grid64_cold_entry_reference_sha256"],
    }


def _normalize_add_rmsnorm_block256_source_round_pipeline(value):
    """Normalize the resident block-256 SM80 Add-RMSNorm pipeline."""

    owner = "scheduling.add_rmsnorm_block256_source_round_pipeline"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "sm80_bf16_add_rmsnorm_block256_resident_v1",
        "software_revision": "vllm_0.11.0_cuda_12.2_sm80",
        "dtype": "bf16",
        "timer_domain": "globaltimer_same_cta",
    }
    integer_fields = {
        "vector_width_elements": 8,
        "source_round_elements": 2048,
        "threads_per_cta": 256,
        "registers_per_thread": 32,
        "static_shared_memory_bytes_per_cta": 168,
        "resident_ctas_per_sm": 8,
        "minimum_grid_blocks": 1,
    }
    width_field = "full_tail_pair_active_threads"
    digest_fields = {
        "service_reference_sha256", "full_tail_pair_reference_sha256",
        "production_function_sass_sha256", "reference_sha256"}
    _require_exact_fields(
        owner, value,
        (set(text_fields) | set(integer_fields) | digest_fields
         | {width_field, "maximum_grid_blocks"}))
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    for name, expected in integer_fields.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    maximum_grid_blocks = value["maximum_grid_blocks"]
    if (isinstance(maximum_grid_blocks, bool)
            or not isinstance(maximum_grid_blocks, int)
            or maximum_grid_blocks not in (864, 1056)):
        raise ValueError(f"{owner}.maximum_grid_blocks must be 864 or 1056")
    for name in digest_fields:
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
    widths = value[width_field]
    if (not isinstance(widths, list)
            or any(isinstance(width, bool) or not isinstance(width, int)
                   for width in widths)
            or widths != [64, 128]):
        raise ValueError(f"{owner}.{width_field} must equal [64, 128]")
    return {
        **text_fields,
        **integer_fields,
        "maximum_grid_blocks": maximum_grid_blocks,
        width_field: tuple(widths),
        **{name: value[name] for name in digest_fields},
    }


def _normalize_grouped_gemm_tile_pipeline(value):
    """Normalize one source-shaped GPU grouped-GEMM pipeline mechanism."""

    owner = "scheduling.grouped_gemm_tile_pipeline"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "sm80_bf16_grouped_gemm_two_slot_v1",
        "software_revision": "vllm_0.11.0_triton_sm80",
        "dtype": "bf16",
        "timer_domain": "cuda_event_gpu_timeline",
    }
    integer_fields = {
        "tile_m": 16,
        "tile_n": 32,
        "tile_k": 64,
        "group_size_m": 1,
        "threads_per_cta": 128,
        "registers_per_thread": 48,
        "dynamic_shared_memory_bytes_per_cta": 12288,
        "pipeline_depth": 2,
        "resident_ctas_per_sm": 10,
        "long_k_grid_ctas": 256,
        "long_k_anchor_tiles": 32,
        "sparse_refill_active_ctas": 1488,
        "sparse_refill_grid_ctas": 1536,
        "sparse_refill_k_tiles": 32,
        "sparse_refill_wave_count": 2,
    }
    variant_integer_fields = {
        "registers_per_thread_by_variant": {"w1": 40, "w2": 48},
        "resident_ctas_per_sm_by_variant": {"w1": 12, "w2": 10},
    }
    # The same Triton source compiled on another card has its own measured
    # compile facts; the set of admitted cards is closed.  On H200 the A800
    # sparse-refill geometry (1536 CTAs) fits one resident wave (132 x 12).
    compile_facts_by_revision = {
        "vllm_0.11.0_triton_sm80": {
            "registers_per_thread": 48,
            "sparse_refill_wave_count": 2,
            "registers_per_thread_by_variant": {"w1": 40, "w2": 48},
        },
        "vllm_0.24.0_triton_3.6_sm90": {
            "registers_per_thread": 45,
            "sparse_refill_wave_count": 1,
            "registers_per_thread_by_variant": {"w1": 39, "w2": 45},
        },
    }
    revision = value["software_revision"]
    if revision not in compile_facts_by_revision:
        raise ValueError(
            f"{owner}.software_revision {revision!r} has no measured "
            "compile facts")
    card_facts = compile_facts_by_revision[revision]
    text_fields = {**text_fields, "software_revision": revision}
    integer_fields = {
        **integer_fields,
        "registers_per_thread": card_facts["registers_per_thread"],
        "sparse_refill_wave_count": card_facts["sparse_refill_wave_count"],
    }
    variant_integer_fields = {
        **variant_integer_fields,
        "registers_per_thread_by_variant": card_facts[
            "registers_per_thread_by_variant"],
    }
    coefficient_fields = {
        "fixed_span_ns", "local_k_round_interval_ns", "empty_tail_round_ns",
        "long_k_anchor_completion_ns", "long_k_round_interval_ns"}
    fields = (
        set(text_fields) | set(integer_fields) | set(variant_integer_fields)
        | coefficient_fields
        | {
            "production_source_sha256", "reference_sha256",
            "empty_tail_reference_sha256", "long_k_reference_sha256",
            "sparse_refill_reference_sha256",
        })
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    normalized_integers = {}
    for name, expected in integer_fields.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
        normalized_integers[name] = item
    normalized_variant_integers = {}
    for name, expected in variant_integer_fields.items():
        item = value[name]
        if not isinstance(item, dict) or set(item) != {"w1", "w2"}:
            raise ValueError(f"{owner}.{name} must declare only W1 and W2")
        normalized = {}
        for variant, expected_value in expected.items():
            variant_value = item[variant]
            if isinstance(variant_value, bool) or not isinstance(
                    variant_value, int):
                raise TypeError(
                    f"{owner}.{name}.{variant} must be an integer")
            if variant_value != expected_value:
                raise ValueError(
                    f"{owner}.{name}.{variant} must equal {expected_value}")
            normalized[variant] = variant_value
        normalized_variant_integers[name] = normalized
    coefficients = {}
    for name in coefficient_fields:
        item = value[name]
        if (isinstance(item, bool)
                or not isinstance(item, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        normalized = float(item)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        coefficients[name] = normalized
    if coefficients["long_k_anchor_completion_ns"] <= (
            normalized_integers["long_k_anchor_tiles"]
            * coefficients["long_k_round_interval_ns"]):
        raise ValueError(
            f"{owner}.long_k_anchor_completion_ns must include a positive "
            "fixed span")
    hashes = {}
    for name in (
            "production_source_sha256", "reference_sha256",
            "empty_tail_reference_sha256", "long_k_reference_sha256",
            "sparse_refill_reference_sha256"):
        item = value[name]
        if (not isinstance(item, str) or len(item) != 64
                or any(character not in "0123456789abcdef"
                       for character in item)):
            raise ValueError(f"{owner}.{name} must be lowercase SHA256")
        hashes[name] = item
    return {
        **text_fields,
        **normalized_integers,
        **normalized_variant_integers,
        **coefficients,
        **hashes,
    }


# wp11_moe_fp8: H200 vLLM 0.24 FP8 block-quantized Triton fused-MoE tiles.
_FP8_BLOCK_TILE_PIPELINE_REFERENCE = (
    "microbench/nvidia_h200/references/"
    "fp8_block_grouped_gemm_tile_pipeline_sm90.json")
_FP8_BLOCK_TILE_WAVE_REFILL_DRAM_REFERENCE = (
    "microbench/nvidia_h200/references/"
    "fp8_block_grouped_gemm_tile_wave_refill_dram_sm90.json")
# Optional per-program DRAM-fed later-wave service (all three or none).
_FP8_BLOCK_DRAM_WAVE_FIELDS = frozenset({
    "dram_wave_refill_ns", "dram_wave_k_round_interval_ns",
    "dram_wave_refill_reference_sha256"})
_FP8_BLOCK_TILE_WAVE_REFILL_REFERENCE = (
    "microbench/nvidia_h200/references/"
    "fp8_block_grouped_gemm_tile_wave_refill_sm90.json")


def _normalize_grouped_gemm_fp8_block_tile_pipelines(value, repository_root):
    """Normalize the FP8 block grouped-GEMM tile-pipeline family.

    One row per compiled tile program of the vLLM 0.24 FP8 block-scaled
    Triton ``fused_moe_kernel``.  Every row carries the same resident-wave
    pipeline coefficients as ``grouped_gemm_tile_pipeline`` (fixed span,
    local K-round interval, empty-CTA tail round) measured by the FP8
    sparse-row probe, the per-wave refill span and full-residency K-round
    interval of later resident waves measured by the multi-wave probe, and
    the source/compile launch facts that fix its pipeline depth and
    residency.  Both reference files are content-addressed.
    """

    owner = "scheduling.grouped_gemm_fp8_block_tile_pipelines"
    if not isinstance(value, list) or not value:
        raise TypeError(f"{owner} must be a non-empty list")
    text_fields = {
        "model": "sm90_fp8_block_grouped_gemm_pipeline_v1",
        "software_revision": "vllm_0.24.0_triton_3.6_sm90",
        "dtype": "fp8_e4m3_w8a8",
        "timer_domain": "cuda_event_gpu_timeline",
        "reference_path": _FP8_BLOCK_TILE_PIPELINE_REFERENCE,
    }
    integer_fields = (
        "weight_block_n", "weight_block_k", "tile_m", "tile_n", "tile_k",
        "num_warps", "num_stages", "threads_per_cta", "pipeline_depth",
        "dynamic_shared_memory_bytes_per_cta")
    variant_integer_fields = (
        "registers_per_thread_by_variant", "resident_ctas_per_sm_by_variant")
    coefficient_fields = (
        "fixed_span_ns", "local_k_round_interval_ns", "empty_tail_round_ns",
        "wave_refill_ns", "wave_k_round_interval_ns")
    hash_fields = (
        "reference_sha256", "empty_tail_reference_sha256",
        "wave_refill_reference_sha256", "production_source_sha256")
    fields = (
        {"program"} | set(text_fields) | set(integer_fields)
        | set(variant_integer_fields) | set(coefficient_fields)
        | set(hash_fields))
    reference_path = repository_root / _FP8_BLOCK_TILE_PIPELINE_REFERENCE
    reference_payload = reference_path.read_bytes()
    reference_sha256 = hashlib.sha256(reference_payload).hexdigest()
    reference = json.loads(reference_payload)
    refill_payload = (
        repository_root / _FP8_BLOCK_TILE_WAVE_REFILL_REFERENCE).read_bytes()
    refill_sha256 = hashlib.sha256(refill_payload).hexdigest()
    refill_reference = json.loads(refill_payload)
    rows = {}
    for index, row in enumerate(value):
        section = f"{owner}[{index}]"
        if not isinstance(row, dict):
            raise TypeError(f"{section} must be a mapping")
        dram_wave = _FP8_BLOCK_DRAM_WAVE_FIELDS & set(row)
        if dram_wave and dram_wave != _FP8_BLOCK_DRAM_WAVE_FIELDS:
            raise ValueError(
                f"{section} DRAM-fed wave service must declare "
                f"{sorted(_FP8_BLOCK_DRAM_WAVE_FIELDS)} together")
        _require_exact_fields(section, row, fields | dram_wave)
        for name, expected in text_fields.items():
            if row[name] != expected:
                raise ValueError(f"{section}.{name} must be {expected!r}")
        normalized = {"program": row["program"], **text_fields}
        for name in integer_fields:
            item = row[name]
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"{section}.{name} must be a positive integer")
            normalized[name] = item
        for name in variant_integer_fields:
            item = row[name]
            if not isinstance(item, dict) or set(item) != {"w1", "w2"}:
                raise ValueError(
                    f"{section}.{name} must declare only W1 and W2")
            for variant in ("w1", "w2"):
                entry = item[variant]
                if (isinstance(entry, bool) or not isinstance(entry, int)
                        or entry <= 0):
                    raise ValueError(
                        f"{section}.{name}.{variant} must be a positive "
                        "integer")
            normalized[name] = {"w1": item["w1"], "w2": item["w2"]}
        for name in coefficient_fields:
            item = row[name]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{section}.{name} must be numeric")
            coefficient = float(item)
            if not math.isfinite(coefficient) or coefficient <= 0.0:
                raise ValueError(f"{section}.{name} must be finite and positive")
            normalized[name] = coefficient
        for name in hash_fields:
            item = row[name]
            if (not isinstance(item, str) or len(item) != 64
                    or any(character not in "0123456789abcdef"
                           for character in item)):
                raise ValueError(f"{section}.{name} must be lowercase SHA256")
            normalized[name] = item
        if normalized["reference_sha256"] != reference_sha256:
            raise ValueError(f"{section}.reference_sha256 mismatch")
        program = (
            f"bm{normalized['tile_m']}_bn{normalized['tile_n']}"
            f"_bk{normalized['tile_k']}_w{normalized['num_warps']}"
            f"_s{normalized['num_stages']}")
        if normalized["program"] != program:
            raise ValueError(f"{section}.program must be {program!r}")
        if program in rows:
            raise ValueError(f"{owner} repeats program {program!r}")
        if (normalized["weight_block_n"], normalized["weight_block_k"]) != (
                128, 128):
            raise ValueError(f"{section} weight block must be 128x128")
        if normalized["tile_k"] != normalized["weight_block_k"]:
            raise ValueError(f"{section}.tile_k must equal the weight block")
        if normalized["threads_per_cta"] != 32 * normalized["num_warps"]:
            raise ValueError(f"{section}.threads_per_cta must be 32 x warps")
        if normalized["pipeline_depth"] != normalized["num_stages"] - 1:
            raise ValueError(
                f"{section}.pipeline_depth must be num_stages - 1")
        measured = reference["programs"][program]["service_parameters"]
        for name in ("fixed_span_ns", "local_k_round_interval_ns"):
            if normalized[name] != float(measured[name]):
                raise ValueError(
                    f"{section}.{name} disagrees with its reference")
        if (normalized["empty_tail_round_ns"]
                != float(reference["empty_tail"]["empty_tail_round_ns"])
                or normalized["empty_tail_reference_sha256"]
                != reference["empty_tail"]["reused_reference_sha256"]):
            raise ValueError(
                f"{section}.empty_tail disagrees with its reference")
        if normalized["wave_refill_reference_sha256"] != refill_sha256:
            raise ValueError(f"{section}.wave_refill_reference_sha256 mismatch")
        refill_measured = refill_reference["programs"][program][
            "service_parameters"]
        for name in ("wave_refill_ns", "wave_k_round_interval_ns"):
            if normalized[name] != float(refill_measured[name]):
                raise ValueError(
                    f"{section}.{name} disagrees with its reference")
        if (normalized["production_source_sha256"]
                != reference["production_source_sha256"]):
            raise ValueError(
                f"{section}.production_source_sha256 disagrees with its "
                "reference")
        if dram_wave:
            # DRAM-streamed multi-wave probe of the same tile program
            # (sm90_fp8_block_tile_wave_refill_dram_reference.py).
            dram_payload = (
                repository_root
                / _FP8_BLOCK_TILE_WAVE_REFILL_DRAM_REFERENCE).read_bytes()
            if (row["dram_wave_refill_reference_sha256"]
                    != hashlib.sha256(dram_payload).hexdigest()):
                raise ValueError(
                    f"{section}.dram_wave_refill_reference_sha256 mismatch")
            dram_document = json.loads(dram_payload)
            dram_programs = dram_document["programs"]
            # The service is measured for K in the probe's k_values only;
            # it is not extrapolated to deeper K loops.
            normalized["dram_wave_max_k_tiles"] = (
                max(dram_document["k_values"]) // normalized["tile_k"])
            if program not in dram_programs:
                raise ValueError(
                    f"{section} DRAM-fed wave reference does not measure "
                    f"{program!r}")
            dram_measured = dram_programs[program]["dram"][
                "service_parameters"]
            for name in ("dram_wave_refill_ns",
                         "dram_wave_k_round_interval_ns"):
                item = row[name]
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise TypeError(f"{section}.{name} must be numeric")
                if float(item) != float(dram_measured[name]):
                    raise ValueError(
                        f"{section}.{name} disagrees with its reference")
                normalized[name] = float(item)
            normalized["dram_wave_refill_reference_sha256"] = (
                row["dram_wave_refill_reference_sha256"])
        rows[program] = normalized
    return rows


def _normalize_sglang_fp8_block_tile_pipelines(
        value, repository_root, hardware_id, vllm_rows):
    """Tile-program rows of SGLang 0.5.12's own FP8 block fused_moe_kernel.

    The reference (sglang_fp8_block_tile_pipeline_reference.py) carries, per
    compiled program of the H200 E=257, N=256 configs, the same coefficients
    as the vLLM rows: sub-wave fixed span and K-round interval, L2 multi-wave
    refill and, for the TMA W2 programs, the DRAM-fed wave at K <= 1024.
    K rounds are counted in the program's BLOCK_SIZE_K.  Residency and shared
    memory come from SGLang's own compile facts.  The early-return empty-CTA
    retire round is the vLLM rows' measured value: SGLang keeps the same
    ``pid_m * BLOCK_SIZE_M >= num_tokens_post_padded`` return before any
    operand load, at the same 128 threads per CTA.  Rows are keyed
    ``sgl_[tma_]bm.._bn.._bk.._w.._s..``.
    """

    owner = "scheduling.sglang_fp8_block_tile_pipelines"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed"})
    if value["model"] != "sm90_sglang_fp8_block_grouped_gemm_pipeline_v1":
        raise ValueError(f"{owner}.model is unsupported")
    payload = _read_owned_reference(
        value, owner, hardware_id, repository_root)
    reference = json.loads(payload)
    if (reference["schema"] != (
            f"{hardware_id}_sglang_fp8_block_grouped_gemm_tile_pipeline_"
            "reference_v1")
            or reference["hardware_id"] != hardware_id
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    if vllm_rows is None:
        raise ValueError(
            f"{owner} reuses the measured empty-tail round of "
            "grouped_gemm_fp8_block_tile_pipelines, which is absent")
    empty_tail = {row["empty_tail_round_ns"] for row in vllm_rows.values()}
    if len(empty_tail) != 1:
        raise ValueError(f"{owner} empty-tail rounds disagree")
    rows = {}
    for program, entry in reference["programs"].items():
        contract = entry["source_and_launch_contract"]
        production = entry["production_compile"]
        service = entry["service_parameters"]
        if not entry["probe_shared_memory_matches_production"]:
            raise ValueError(
                f"{owner}.{program} probe shared memory differs from "
                "production")
        expected = (
            f"sgl_{'tma_' if contract['use_tma'] else ''}"
            f"bm{contract['tile_m']}_bn{contract['tile_n']}"
            f"_bk{contract['tile_k']}_w{contract['num_warps']}"
            f"_s{contract['num_stages']}")
        if program != expected:
            raise ValueError(f"{owner} program {program!r} != {expected!r}")
        row = {
            "program": program,
            "tile_m": contract["tile_m"], "tile_n": contract["tile_n"],
            "tile_k": contract["tile_k"],
            "num_warps": contract["num_warps"],
            "num_stages": contract["num_stages"],
            "threads_per_cta": contract["threads_per_cta"],
            "pipeline_depth": contract["num_stages"] - 1,
            "dynamic_shared_memory_bytes_per_cta": (
                production["shared_memory_bytes"]),
            "registers_per_thread_by_variant": {
                "w1": production["registers_per_thread"],
                "w2": production["registers_per_thread"]},
            "resident_ctas_per_sm_by_variant": {
                "w1": production["resident_ctas_per_sm"],
                "w2": production["resident_ctas_per_sm"]},
            "empty_tail_round_ns": next(iter(empty_tail)),
        }
        for name in ("fixed_span_ns", "local_k_round_interval_ns",
                     "wave_refill_ns", "wave_k_round_interval_ns"):
            coefficient = float(service[name])
            if not math.isfinite(coefficient) or coefficient <= 0.0:
                raise ValueError(f"{owner}.{program}.{name} must be positive")
            row[name] = coefficient
        if contract["use_tma"]:
            for name in ("dram_wave_refill_ns",
                         "dram_wave_k_round_interval_ns"):
                row[name] = float(service[name])
            # Measured at K 256..1024 only; not extrapolated deeper.
            row["dram_wave_max_k_tiles"] = 1024 // contract["tile_k"]
        rows[program] = row
    return rows


# --- wp10 (H200 FA3 MLA): FlashAttention-3 mechanism services -------------
_FA3_WGMMA_SHAPES = (
    "ss_n64_kmajor", "ss_n128_kmajor", "rs_n128_mnmajor",
    "ss_n256_mnmajor", "rs_n256_mnmajor",
)
_FA3_SOFTMAX_GEOMETRIES = ("16x64", "32x32")


def _fa3_positive_cycles(owner, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{owner} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{owner} must be finite and positive")
    return number


def _normalize_fa3_mla_mechanisms(value, repository_root, hardware_id):
    """Normalize the measured FA3 MLA mechanism services.

    The reference is produced by
    ``microbench/<hw>/operators/flash_attn/fa3_mechanism_reference.py`` from
    three independent probes (WGMMA gemm service, online-softmax step
    service, paged-cache cp.async two-stage ring service).  No FlashAttention
    kernel is executed by any of them.
    """

    owner = "scheduling.fa3_mla_mechanisms"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "fa3_mla_mechanisms_v1":
        raise ValueError(f"{owner}.model is unsupported")
    if value["timer_domain"] != "clock64_sm_cycles":
        raise ValueError(f"{owner}.timer_domain must be clock64_sm_cycles")
    _require_no_duration_consumption(value, owner)
    reference_path = Path(value["reference_path"])
    _require_owned_reference_path(reference_path, owner, hardware_id)
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"]
            != f"{hardware_id}_fa3_mla_mechanisms_reference_v1"
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "fa3_mla_mechanisms"
            or reference["timer_domain"] != value["timer_domain"]
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    clock_mhz = _fa3_positive_cycles(
        f"{owner}.measured_sm_clock_mhz", reference["measured_sm_clock_mhz"])
    wgmma = {}
    for shape in _FA3_WGMMA_SHAPES:
        for ksteps, by_wg in reference["wgmma_gemm_cycles"][shape].items():
            for active, cycles in by_wg.items():
                wgmma[(shape, int(ksteps), int(active))] = (
                    _fa3_positive_cycles(
                        f"{owner}.wgmma[{shape}][{ksteps}][{active}]",
                        cycles))
    softmax = {}
    for geometry in _FA3_SOFTMAX_GEOMETRIES:
        for active, cycles in reference["softmax_step_cycles"][
                geometry].items():
            softmax[(geometry, int(active))] = _fa3_positive_cycles(
                f"{owner}.softmax[{geometry}][{active}]", cycles)
    ring_hold = reference["decode_ring"]["hold_cycles"]
    if isinstance(ring_hold, bool) or not isinstance(ring_hold, int):
        raise TypeError(f"{owner}.decode_ring.hold_cycles must be an integer")
    ring = {}
    for grid, cell in reference["paged_kv_stream"][str(ring_hold)].items():
        ring[int(grid)] = (
            _fa3_positive_cycles(
                f"{owner}.ring[{grid}].first_full_cycles",
                cell["first_full_cycles"]),
            _fa3_positive_cycles(
                f"{owner}.ring[{grid}].steady_ii_cycles",
                cell["steady_ii_cycles"]),
        )
    if sorted(ring) != list(range(1, max(ring) + 1)):
        raise ValueError(f"{owner} decode ring must cover every grid 1..N")
    return {
        "model": value["model"],
        "measured_sm_clock_hz": clock_mhz * 1e6,
        "wgmma_gemm_cycles": dict(wgmma),
        "softmax_step_cycles": dict(softmax),
        "decode_ring_hold_cycles": ring_hold,
        "decode_ring_by_grid": dict(ring),
    }
# --- end wp10 --------------------------------------------------------------


def _normalize_cooperative_grid_sync_service(
        value, repository_root, hardware_id):
    """Measured cooperative-group grid barrier inside a replayed CUDA graph."""

    owner = "scheduling.cooperative_grid_sync_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "lookup_rule", "reference_path",
        "reference_sha256", "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "cooperative_grid_sync_graph_node_v1":
        raise ValueError(f"{owner}.model is unsupported")
    if value["timer_domain"] != "cuda_event_around_graph_replay_divided_by_chain":
        raise ValueError(f"{owner}.timer_domain is unsupported")
    if value["lookup_rule"] != "step_down_largest_sampled_grid_not_exceeding_request":
        raise ValueError(f"{owner}.lookup_rule is unsupported")
    _require_no_duration_consumption(value, owner)
    reference_path = Path(value["reference_path"])
    _require_owned_reference_path(reference_path, owner, hardware_id)
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"]
            != f"{hardware_id}_cooperative_grid_sync_graph_node_reference_v1"
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "cooperative_grid_sync_in_cuda_graph"
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    entries = []
    for raw_grid, raw_ns in reference["grid_sync_ns_by_grid"].items():
        grid = int(raw_grid)
        service = float(raw_ns)
        if grid <= 0 or not math.isfinite(service) or service <= 0.0:
            raise ValueError(f"{owner} entries must be positive")
        entries.append((grid, service))
    if not entries:
        raise ValueError(f"{owner} table is empty")
    return tuple(sorted(entries))


def _normalize_split_merge_chunk_service(
        value, repository_root, hardware_id):
    """Measured per-chunk interval of a serial split-KV merge walk.

    One chunk = the LSE and partial-output loads of lines another SM wrote
    before a grid barrier, consumed by the online-softmax rescale chain
    before the next chunk's loads issue.  Returns seconds per chunk.
    """

    owner = "scheduling.split_merge_chunk_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "split_kv_merge_chunk_load_use_v1":
        raise ValueError(f"{owner}.model is unsupported")
    if value["timer_domain"] != (
            "clock64_cta_envelope_differential_43_minus_11_chunks"):
        raise ValueError(f"{owner}.timer_domain is unsupported")
    _require_no_duration_consumption(value, owner)
    reference_path = Path(value["reference_path"])
    _require_owned_reference_path(reference_path, owner, hardware_id)
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"]
            != f"{hardware_id}_split_merge_rescale_load_use_reference_v1"
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "split_kv_merge_chunk_load_use"
            or reference["timer_domain"] != value["timer_domain"]
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    published = reference["published"]
    if (published["writer"], published["consumer"]) != ("cross", "rescale"):
        raise ValueError(f"{owner} must publish the cross-SM rescale walk")
    cycles = float(published["per_chunk_cycles"])
    clock_hz = float(reference["reference_sm_clock_hz"])
    if (not math.isfinite(cycles) or cycles <= 0.0
            or not math.isfinite(clock_hz) or clock_hz <= 0.0):
        raise ValueError(f"{owner} per-chunk interval must be positive")
    return cycles / clock_hz


_STREAM_MEMCPY_DIRECTIONS = ("h2d_pinned", "h2d_pageable", "d2h_pinned", "d2d")


def _normalize_cuda_graph_side_stream_fork_node_boundary(
        value, repository_root, hardware_id):
    """Added boundary per dependent node of a graph with side-stream forks.

    A replayed graph whose capture forks a side-stream node from the main
    chain and joins it back every few nodes (SGLang's per-layer alt-stream
    RMSNorm) spaces its main-chain nodes further apart than a pure chain.
    Returns the per-node increment in seconds at the published fork spacing.
    """

    owner = "scheduling.cuda_graph_side_stream_fork_node_boundary"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "fork_spacing_nodes", "reference_path",
        "reference_sha256", "production_or_evaluation_duration_consumed"})
    if value["model"] != "cuda_graph_side_stream_fork_node_boundary_v1":
        raise ValueError(f"{owner}.model is unsupported")
    if value["timer_domain"] != (
            "torch_profiler_cupti_main_chain_start_to_start"):
        raise ValueError(f"{owner}.timer_domain is unsupported")
    payload = _read_owned_reference(
        value, owner, hardware_id, repository_root)
    reference = json.loads(payload)
    if (reference["schema"] != (
            f"{hardware_id}_cuda_graph_side_stream_fork_node_boundary_"
            "reference_v1")
            or reference["hardware_id"] != hardware_id
            or reference["timer_domain"] != value["timer_domain"]
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    spacing = value["fork_spacing_nodes"]
    if (isinstance(spacing, bool) or not isinstance(spacing, int)
            or spacing != reference["published_fork_spacing_nodes"]):
        raise ValueError(
            f"{owner}.fork_spacing_nodes disagrees with its reference")
    table = reference["by_fork_spacing_nodes"]
    increment_ns = (
        float(table[str(spacing)]["main_chain_cadence_mean_ns"])
        - float(table["chain"]["main_chain_cadence_mean_ns"]))
    if abs(increment_ns
           - float(reference["published_increment_ns_per_node"])) > 0.1:
        raise ValueError(f"{owner} published increment disagrees with table")
    if not math.isfinite(increment_ns) or increment_ns <= 0.0:
        raise ValueError(f"{owner} increment must be positive")
    return increment_ns * 1e-9


def _normalize_stream_memcpy_service(value, repository_root, hardware_id):
    """Measured cost of one eager cudaMemcpyAsync between kernels on a stream.

    Keyed by direction and host-memory kind; a payload steps down to the
    largest sampled payload at or below it (the smallest sample below it)
    and fails closed at or above ``max_payload_bytes``.
    """

    owner = "scheduling.stream_memcpy_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "lookup_rule", "max_payload_bytes",
        "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed"})
    if value["model"] != "eager_stream_memcpy_service_v1":
        raise ValueError(f"{owner}.model is unsupported")
    if value["timer_domain"] != "cuda_event_chain_difference_per_copy":
        raise ValueError(f"{owner}.timer_domain is unsupported")
    if value["lookup_rule"] != (
            "step_down_largest_sampled_payload_not_exceeding_request_"
            "floor_at_smallest_sample"):
        raise ValueError(f"{owner}.lookup_rule is unsupported")
    payload = _read_owned_reference(
        value, owner, hardware_id, repository_root)
    reference = json.loads(payload)
    if (reference["schema"]
            != f"{hardware_id}_stream_memcpy_service_reference_v1"
            or reference["hardware_id"] != hardware_id
            or reference["timer_domain"] != value["timer_domain"]
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    max_payload = value["max_payload_bytes"]
    if (isinstance(max_payload, bool) or not isinstance(max_payload, int)
            or max_payload <= 0):
        raise ValueError(f"{owner}.max_payload_bytes must be positive")
    tables = {}
    by_direction = reference["service_ns_by_direction"]
    if set(by_direction) != set(_STREAM_MEMCPY_DIRECTIONS):
        raise ValueError(f"{owner} directions must be {_STREAM_MEMCPY_DIRECTIONS}")
    for direction in _STREAM_MEMCPY_DIRECTIONS:
        rows = []
        for raw_bytes in sorted(by_direction[direction], key=int):
            size = int(raw_bytes)
            service = float(by_direction[direction][raw_bytes])
            if size <= 0 or size >= max_payload:
                raise ValueError(f"{owner} sampled payload out of range")
            if not math.isfinite(service) or service <= 0.0:
                raise ValueError(f"{owner} service must be positive")
            rows.append((size, service))
        tables[direction] = tuple(rows)
    return {"max_payload_bytes": max_payload, "tables": tables}


def _normalize_cuda_graph_small_node_span(
        value, repository_root, hardware_id):
    """Normalize the measured span floor of a replayed CUDA-graph node."""

    owner = "scheduling.cuda_graph_small_node_span"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "threads_per_cta", "nodes_per_batch",
        "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "cuda_graph_small_node_span_v1":
        raise ValueError(f"{owner}.model is unsupported")
    if value["timer_domain"] != "cuda_event_gpu_timeline":
        raise ValueError(f"{owner}.timer_domain must be a CUDA-event domain")
    for name in ("threads_per_cta", "nodes_per_batch"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{owner}.{name} must be a positive integer")
    _require_no_duration_consumption(value, owner)
    reference_path = Path(value["reference_path"])
    _require_owned_reference_path(reference_path, owner, hardware_id)
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"]
            != f"{hardware_id}_cuda_graph_small_node_span_reference_v1"
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "cuda_graph_small_node_span"
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    entries = []
    previous_grid = 0
    for point in reference["derived_model"]["span_ns_by_grid_ctas"]:
        grid = point["grid_ctas"]
        span_ns = float(point["minimum_graph_node_span_ns"])
        if (isinstance(grid, bool) or not isinstance(grid, int)
                or grid <= previous_grid):
            raise ValueError(f"{owner} grids must ascend and be positive")
        if not math.isfinite(span_ns) or span_ns <= 0.0:
            raise ValueError(f"{owner} span must be positive")
        entries.append((grid, span_ns))
        previous_grid = grid
    if not entries:
        raise ValueError(f"{owner} span table is empty")
    return {
        "model": value["model"],
        "timer_domain": value["timer_domain"],
        "threads_per_cta": value["threads_per_cta"],
        "span_ns_by_grid_ctas": tuple(entries),
    }


def _normalize_block_fp8_gemm_tma_ring_service(
        value, repository_root, hardware_id):
    """Normalize the measured TMA operand-ring service of block-FP8 GEMMs.

    The reference (``fp8_gemm_mechanisms_reference.py``) publishes the
    measured nsys durations of the MMA-free TMA producer/consumer ring (per
    ring geometry, streaming CTAs and k-blocks), the geometry-free ring
    service form fitted to those probe surfaces only (per-stage floor, chip
    DRAM stream rate, p-norm composition, fill), the store-free launch shell
    and the full-chip output-tile store rate.
    """

    owner = "scheduling.block_fp8_gemm_tma_ring_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "block_fp8_gemm_tma_ring_service_v1":
        raise ValueError(f"{owner}.model is unsupported")
    if value["timer_domain"] != "nsys_gpu_kernel":
        raise ValueError(f"{owner}.timer_domain must be nsys_gpu_kernel")
    _require_no_duration_consumption(value, owner)
    reference_path = Path(value["reference_path"])
    _require_owned_reference_path(reference_path, owner, hardware_id)
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"]
            != f"{hardware_id}_block_fp8_gemm_tma_ring_reference_v1"
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "block_fp8_gemm_tma_ring_service"
            or reference["timer_domain"] != value["timer_domain"]
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    model = reference["derived_model"]

    def positive_int(item, name):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{owner} {name} must be a positive integer")
        return item

    def positive_ns(item, name):
        result = float(item)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{owner} {name} must be positive")
        return result

    ring = model["ring_service_model"]
    ring_model = {
        name: positive_ns(ring[name], name) for name in (
            "stage_floor_ns", "dram_stream_bytes_per_ns",
            "combination_power", "fill_base_ns")
    }
    fill_per_store_byte = float(ring["fill_per_store_byte_ns"])
    if not math.isfinite(fill_per_store_byte) or fill_per_store_byte < 0.0:
        raise ValueError(
            f"{owner} fill_per_store_byte_ns must be finite and non-negative")
    ring_model["fill_per_store_byte_ns"] = fill_per_store_byte
    return {
        "model": value["model"],
        "timer_domain": value["timer_domain"],
        "ring": ring_model,
        "launch_shell_ns": positive_ns(
            model["launch_shell_ns"], "launch_shell_ns"),
        "full_chip_tile_store_bytes_per_ns": positive_ns(
            model["full_chip_tile_store_bytes_per_ns"],
            "full_chip_tile_store_bytes_per_ns"),
    }


# The kernel-duration timer each hardware's evaluation scores against, which
# is the only domain its span floor may be measured in.
_SMALL_KERNEL_SPAN_FLOOR_TIMER_DOMAIN = {
    "nvidia_a800": "nsys_gpu_kernel",
    "nvidia_h200": "nsys_gpu_kernel",
    "metax_c500": "mctracer_uninstrumented_kernel_duration",
}


def _normalize_small_kernel_span_floor(
        value, repository_root, hardware_id):
    """Normalize the measured duration floor of a memory-touching launch.

    The empty-task scheduler describes a launch that touches nothing.  A launch
    that performs any global access does not reach that duration, so this floor
    composes with an execution time by maximum.
    """

    owner = "scheduling.small_kernel_span_floor"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "small_kernel_span_floor_v1":
        raise ValueError(f"{owner}.model is unsupported")
    # The floor is a bound on the duration the evaluation scores against, so
    # each hardware must declare its own kernel-duration timer domain and no
    # other: nsys for A800, and for C500 the mcTracer uninstrumented duration
    # as the required source for C500 kernel-duration observations.
    if hardware_id not in _SMALL_KERNEL_SPAN_FLOOR_TIMER_DOMAIN:
        raise ValueError(
            f"{owner} has no declared timer domain for {hardware_id}")
    expected_timer_domain = _SMALL_KERNEL_SPAN_FLOOR_TIMER_DOMAIN[
        hardware_id]
    if value["timer_domain"] != expected_timer_domain:
        raise ValueError(
            f"{owner}.timer_domain must be {expected_timer_domain}")
    _require_no_duration_consumption(value, owner)
    reference_path = Path(value["reference_path"])
    if (reference_path.is_absolute()
            or ".." in reference_path.parts
            or reference_path.parts[:3] != (
                "microbench", _mechanism_reference_hardware_id(hardware_id), "references")):
        raise ValueError(f"{owner}.reference_path is outside its owner")
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"] != "small_kernel_span_floor_reference_v1"
            or reference["hardware_id"] != _mechanism_reference_hardware_id(hardware_id)
            or reference["mechanism"] != "small_kernel_span_floor"
            or reference["production_or_evaluation_duration_consumed"]
            is not False):
        raise ValueError(f"{owner} reference contract drifted")
    entries = []
    previous_grid = 0
    for point in reference["derived_model"]["floor_by_grid_ctas"]:
        grid = point["grid_ctas"]
        floor_ns = float(point["minimum_span_ns"])
        if (isinstance(grid, bool) or not isinstance(grid, int)
                or grid <= previous_grid):
            raise ValueError(f"{owner} grids must ascend and be positive")
        if not math.isfinite(floor_ns) or floor_ns <= 0.0:
            raise ValueError(f"{owner} floor must be positive")
        entries.append((grid, floor_ns))
        previous_grid = grid
    if not entries:
        raise ValueError(f"{owner} floor table is empty")
    return {
        "model": value["model"],
        "timer_domain": value["timer_domain"],
        "floor_ns_by_grid_ctas": tuple(entries),
    }


def _normalize_early_exit_cta_retire(value, repository_root, hardware_id):
    """Normalize the measured marginal cost of an early-exit CTA.

    A CTA whose query tile lies beyond its own request starts, reads the two
    integers describing its tile, and returns before the Q/K/V path.  This is
    the launch-wide time one such CTA adds; dividing by the AP count gives
    the serial cost one AP pays per early-exit CTA it retires.
    """

    owner = "scheduling.early_exit_cta_retire"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "software_revision", "timer_domain", "threads_per_cta",
        "dynamic_shared_bytes", "launch_wide_ns_per_early_exit_cta",
        "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "early_exit_cta_retire_v1":
        raise ValueError(f"{owner}.model is unsupported")
    for name in ("software_revision", "timer_domain"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise TypeError(f"{owner}.{name} must be a non-empty string")
    for name in ("threads_per_cta", "dynamic_shared_bytes"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{owner}.{name} must be a positive integer")
    rate = value["launch_wide_ns_per_early_exit_cta"]
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or rate <= 0.0):
        raise ValueError(f"{owner} rate must be finite and positive")
    _require_no_duration_consumption(value, owner)
    reference_path = Path(value["reference_path"])
    _require_owned_reference_path(reference_path, owner, hardware_id)
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"] != "metax_c500_early_exit_cta_retire_reference_v1"
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "early_exit_cta_retire"
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["production_predictor_consumption_allowed"]
            is not True):
        raise ValueError(f"{owner} reference contract drifted")
    if not math.isclose(
            float(reference["derived_model"][
                "launch_wide_ns_per_early_exit_cta"]),
            float(rate), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{owner} rate disagrees with its reference")
    scope = reference["measurement"]
    if (scope["threads_per_cta"] != value["threads_per_cta"]
            or scope["dynamic_shared_bytes"] != value["dynamic_shared_bytes"]):
        raise ValueError(f"{owner} reference scope differs from HWSpec")
    return {
        "model": value["model"],
        "software_revision": value["software_revision"],
        "timer_domain": value["timer_domain"],
        "threads_per_cta": value["threads_per_cta"],
        "dynamic_shared_bytes": value["dynamic_shared_bytes"],
        "launch_wide_ns_per_early_exit_cta": float(rate),
    }


def _derive_tma_quiescent_read_ramp(measurements, capacity, *,
        request_b_sizes=(4096, 8192, 16384)):
    """Optimistic monotone byte-service envelope of independent TMA controls."""
    if request_b_sizes not in ((4096, 8192, 16384), (4096, 8192, 16384, 32768)):
        raise ValueError("unsupported independent TMA request protocol")
    expected = {(grid, b, depth, rounds)
        for grid in (32, 64, 96, 132) for b in request_b_sizes
        for depth in (4, 6, 8, 12, 16) if depth * (1024 + b) <= 212992
        for rounds in (16, 32, 64, 128, 512)}
    seen = set()
    points = []
    if isinstance(capacity, bool) or not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("TMA ramp capacity must be positive")
    for row in measurements:
        _require_exact_fields("TMA ramp measurement", row, {
            "grid_ctas", "a_bytes", "b_bytes", "depth", "rounds", "span_ns"})
        grid, a, b, depth, rounds = (row[key] for key in (
            "grid_ctas", "a_bytes", "b_bytes", "depth", "rounds"))
        if any(isinstance(v, bool) or not isinstance(v, int)
               for v in (grid, a, b, depth, rounds)):
            raise ValueError("TMA ramp geometry must contain integers")
        key = (grid, b, depth, rounds)
        if a != 1024 or key not in expected or key in seen:
            raise ValueError("TMA ramp controls are invalid or duplicated")
        seen.add(key)
        span = row["span_ns"]
        if (isinstance(span, bool) or not isinstance(span, (int, float))
                or not math.isfinite(span) or span <= 0):
            raise ValueError("TMA ramp span must be positive")
        length = (a + b) * rounds
        points.append((grid, length, min(1.0, grid * length * 1e9 / span / capacity)))
    if seen != expected:
        raise ValueError("TMA ramp requires every independent control")
    grids = tuple(sorted({grid for grid, _, _ in points}))
    lengths = tuple(sorted({length for _, length, _ in points}))
    ramp = {}
    for grid in grids:
        for length in lengths:
            ramp[grid, length] = max(fraction for g, n, fraction in points
                if g <= grid and n <= length)
    return grids, lengths, ramp


def _validate_tma_capture_binding(binding, owner, hardware_id):
    """Check archived provenance; raw captures are not runtime inputs."""
    _require_exact_fields(owner, binding, {"path", "sha256"})
    if not isinstance(binding["path"], str) or not binding["path"]:
        raise ValueError(f"{owner} capture path must be a non-empty string")
    path = Path(binding["path"])
    if (path.is_absolute() or ".." in path.parts
            or path.parts[:3] != ("profiler_output", "microbench", hardware_id)):
        raise ValueError(f"{owner} capture outside owner")
    if (not isinstance(binding["sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", binding["sha256"]) is None):
        raise ValueError(f"{owner} capture SHA256 is invalid")


def _normalize_dram_finite_tma_read_stream_ramp(value, repository_root, hardware):
    owner = "scheduling.dram_finite_tma_read_stream_ramp"
    hardware_id = hardware.hardware_id
    _require_fields_with_optional(owner, value, {"model", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed"}, {"additional_profiles"})
    if value["model"] != "dram_finite_read_stream_ramp_v3_tma_quiescent":
        raise ValueError(f"{owner}.model is unsupported")
    reference = json.loads(_read_owned_reference(
        value, owner, _mechanism_reference_hardware_id(hardware_id), repository_root))
    _require_exact_fields(owner + ".reference", reference, {
        "schema", "hardware_id", "mechanism",
        "production_or_evaluation_duration_consumed", "selection_rule",
        "capture", "scope", "source_state_basis", "measurements",
        "derived_model", "allowed_use", "limits",
    })
    if (reference["schema"] != "dram_finite_read_stream_ramp_reference_v3_tma_quiescent"
            or reference["hardware_id"] != _mechanism_reference_hardware_id(hardware_id)
            or reference["mechanism"] != "dram_finite_read_stream_ramp"
            or reference["production_or_evaluation_duration_consumed"] is not False
            or reference["selection_rule"] != "quiescent_advancing_cumulative_grid_length_capacity_upper_envelope"):
        raise ValueError(f"{owner} independent reference contract drifted")
    scope = reference["scope"]
    maximum_request_bytes = scope["request_total_bytes_max"]
    if maximum_request_bytes not in (17408, 33792):
        raise ValueError(f"{owner} unmeasured request domain")
    if scope != {"transport": "tma", "threads_per_cta": 384, "tile_k": 64,
            "operand_element_bytes": 2, "memory_start_state": "quiescent_after_3ms_ALU",
            "request_total_bytes_min": 5120, "request_total_bytes_max": maximum_request_bytes}:
        raise ValueError(f"{owner} scope drifted")
    _validate_tma_capture_binding(
        reference["capture"], owner, _mechanism_reference_hardware_id(hardware_id))
    # Publication audits the raw paired controls. Runtime re-derives the
    # envelope from the complete measurements in the hash-bound reference.
    captured = reference["measurements"]
    derived = reference["derived_model"]
    capacity = derived["dram_read_capacity_bytes_per_s"]
    physical_capacity = hardware.dram_capacity_for_access(
        "read", active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz)
    if not math.isclose(capacity, physical_capacity, rel_tol=1e-12):
        raise ValueError(f"{owner} capacity differs from independent hardware ceiling")
    request_b_sizes = ((4096, 8192, 16384) if maximum_request_bytes == 17408
        else (4096, 8192, 16384, 32768))
    grids, lengths, ramp = _derive_tma_quiescent_read_ramp(captured, capacity,
        request_b_sizes=request_b_sizes)
    expected_ramp = [{"grid_ctas": grid, "bytes_per_cta": length,
        "achieved_fraction_of_dram_read_capacity": ramp[grid, length]}
        for grid in grids for length in lengths]
    if (derived["grids"] != list(grids) or derived["lengths_bytes_per_cta"] != list(lengths)
            or derived["ramp"] != expected_ramp):
        raise ValueError(f"{owner} derived capacity envelope drifted")
    return {**scope, "model": value["model"], "grids": grids,
        "lengths_bytes_per_cta": lengths, "ramp": ramp,
        "dram_read_capacity_bytes_per_s": capacity,
        "reference_path": value["reference_path"], "reference_sha256": value["reference_sha256"]}


def _derive_tma_balanced_read_ramp(measurements, capacity):
    """Frozen independent optimistic bound; all non-calibration cells validate it."""
    packets = (16384, 24576, 32768, 40960, 45056)
    expected = {(t, a, d, g, resident, r)
        for t in (256, 384) for a in packets for d in (1, 2, 3, 4, 6)
        for g in (32, 64, 96, 132) for resident in (0, 1)
        for r in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if 2 * a * d <= 212992 and r >= d}
    if isinstance(capacity, bool) or not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("balanced TMA requires positive DRAM capacity")
    cells, points, calibration = {}, [], set()
    for row in measurements:
        _require_exact_fields("balanced TMA measurement", row, {
            "threads", "a_bytes", "b_bytes", "depth", "grid", "resident", "rounds",
            "spans_ns", "span_ns_p50", "prelude_clock_mhz_p50"})
        key = tuple(row[k] for k in ("threads", "a_bytes", "depth", "grid", "resident", "rounds"))
        if (any(type(v) is not int for v in key) or key not in expected or key in cells
                or type(row["b_bytes"]) is not int or row["b_bytes"] != row["a_bytes"]):
            raise ValueError("balanced TMA geometry invalid or duplicated")
        spans = row["spans_ns"]
        if (not isinstance(spans, list) or len(spans) != 7
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in spans)
                or row["span_ns_p50"] != statistics.median(spans)):
            raise ValueError("balanced TMA requires all seven retained trials and exact median")
        clock = row["prelude_clock_mhz_p50"]
        if (type(clock) not in (int, float) or not math.isfinite(clock)
                or abs(clock * 1e6 / 1.98e9 - 1) > .05):
            raise ValueError("balanced TMA prelude clock outside frozen domain")
        cells[key] = spans
        t, a, d, g, resident, r = key
        if t == 256 and g in (32, 132) and a in (16384, 32768, 45056) and r in (1, 8, 64, 256) and resident == 0:
            calibration.add(key)
            length = 2 * a * r
            points.extend((g, length, min(1., g * length * 1e9 / span / capacity)) for span in spans)
    if set(cells) != expected:
        raise ValueError("balanced TMA requires every independent control")
    grids = tuple(sorted({g for g, _, _ in points}))
    lengths = tuple(sorted({n for _, n, _ in points}))
    ramp = {(g, n): max(f for grid, length, f in points if grid <= g and length <= n)
            for g in grids for n in lengths}
    for key, spans in cells.items():
        t, a, d, g, resident, r = key
        if resident != 0 or key in calibration:
            continue
        length = 2 * a * r
        g_key = next(v for v in grids if v >= g)
        n_key = next(v for v in lengths if v >= length)
        completion_ns = g * length * 1e9 / (capacity * ramp[g_key, n_key])
        if any(completion_ns > observed * 1.05 + 64 for observed in spans):
            raise ValueError("balanced TMA heldout completion lower bound failed")
    return grids, lengths, ramp


def _normalize_tma_balanced_read_profile(value, repository_root, hardware):
    owner = "scheduling.dram_finite_tma_read_stream_ramp.additional_profiles"
    _require_exact_fields(owner, value, {"model", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed"})
    if value["model"] != "dram_finite_read_stream_ramp_v3_tma_quiescent":
        raise ValueError(f"{owner} model unsupported")
    ref = json.loads(_read_owned_reference(value, owner, hardware.hardware_id, repository_root))
    _require_exact_fields(owner + ".reference", ref, {"schema", "hardware_id", "mechanism",
        "production_or_evaluation_duration_consumed", "selection_rule", "scope", "evidence", "derived_model", "limits"})
    if (hardware.hardware_id != "nvidia_h200" or ref["hardware_id"] != hardware.hardware_id
            or ref["schema"] != "dram_finite_read_stream_ramp_reference_v3_tma_quiescent"
            or ref["mechanism"] != "dram_finite_read_stream_ramp"
            or ref["production_or_evaluation_duration_consumed"] is not False
            or ref["selection_rule"] != "balanced_quiescent_all_trial_calibration_capacity_upper_envelope"):
        raise ValueError(f"{owner} independent reference contract drifted")
    scope = {"transport": "tma", "threads_per_cta": [256, 384],
        "balanced_operand_bytes": [16384, 24576, 32768, 40960, 45056],
        "native_slots_per_operand": 2, "maximum_shared_bytes": 212992,
        "memory_start_state": "quiescent_after_3ms_ALU", "minimum_producers": 32,
        "maximum_producers": 132}
    if ref["scope"] != scope:
        raise ValueError(f"{owner} scope drifted")
    _require_exact_fields(owner + ".evidence", ref["evidence"], {"capture", "source", "runner", "plan"})
    for name, binding in ref["evidence"].items():
        _validate_tma_capture_binding(
            binding, owner + ".evidence." + name, hardware.hardware_id)
    capacity = hardware.dram_capacity_for_access("read", active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz, dram_freq_hz=hardware.nominal_dram_freq_hz)
    # The published, hash-bound reference already contains the audited
    # envelope. Replay of raw trials belongs to publication, not prediction.
    derived = ref["derived_model"]
    _require_exact_fields(owner + ".derived_model", derived, {
        "dram_read_capacity_bytes_per_s", "grids", "lengths_bytes_per_cta", "ramp"})
    declared_capacity = derived["dram_read_capacity_bytes_per_s"]
    if (type(declared_capacity) not in (int, float)
            or not math.isfinite(declared_capacity)
            or declared_capacity <= 0 or declared_capacity != capacity):
        raise ValueError(f"{owner} capacity differs from independent hardware ceiling")
    grids = (32, 132)
    lengths = tuple(sorted({2 * a * rounds for a in (16384, 32768, 45056)
                            for rounds in (1, 8, 64, 256)}))
    for field, expected in (("grids", grids), ("lengths_bytes_per_cta", lengths)):
        values = derived[field]
        if (not isinstance(values, list) or tuple(values) != expected
                or any(type(value) is not int for value in values)):
            raise ValueError(f"{owner} published {field} domain drifted")
    rows = derived["ramp"]
    keys = [(g, n) for g in grids for n in lengths]
    if not isinstance(rows, list) or len(rows) != len(keys):
        raise ValueError(f"{owner} published envelope is incomplete")
    ramp = {}
    for row, key in zip(rows, keys):
        _require_exact_fields(owner + ".ramp", row, {
            "grid_ctas", "bytes_per_cta", "achieved_fraction_of_dram_read_capacity"})
        if (type(row["grid_ctas"]) is not int or type(row["bytes_per_cta"]) is not int
                or (row["grid_ctas"], row["bytes_per_cta"]) != key):
            raise ValueError(f"{owner} published envelope geometry drifted")
        fraction = row["achieved_fraction_of_dram_read_capacity"]
        if (type(fraction) not in (int, float) or not math.isfinite(fraction)
                or not 0 < fraction <= 1):
            raise ValueError(f"{owner} published capacity fraction is invalid")
        ramp[key] = fraction
    for i, grid in enumerate(grids):
        for j, length in enumerate(lengths):
            if ((i > 0 and ramp[grid, length] < ramp[grids[i - 1], length])
                    or (j > 0 and ramp[grid, length] < ramp[grid, lengths[j - 1]])):
                raise ValueError(f"{owner} published envelope must be monotone")
    return {**scope, "grids": grids, "lengths_bytes_per_cta": lengths, "ramp": ramp,
        "dram_read_capacity_bytes_per_s": capacity, "reference_path": value["reference_path"]}


def _normalize_dram_finite_read_stream_ramp(
        value, repository_root, hardware_id):
    """Normalize the measured ramp of a finite DRAM read stream.

    A finite read stream reaches only part of the published DRAM read
    capacity, and the achieved fraction rises with the bytes each CTA
    streams.  The table is an upper envelope on that fraction, so dividing
    compulsory read bytes by capacity times the fraction is a completion lower
    bound.
    """

    owner = "scheduling.dram_finite_read_stream_ramp"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    if value.get("model") == "dram_finite_read_stream_ramp_v2_threaded":
        return _normalize_dram_finite_read_stream_ramp_threaded(
            value, repository_root, hardware_id, owner)
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "threads_per_cta",
        "in_flight_bytes_per_cta", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "dram_finite_read_stream_ramp_v1":
        raise ValueError(f"{owner}.model is unsupported")
    if (not isinstance(value["timer_domain"], str)
            or not value["timer_domain"].strip()):
        raise TypeError(f"{owner}.timer_domain must be a non-empty string")
    for name in ("threads_per_cta", "in_flight_bytes_per_cta"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{owner}.{name} must be a positive integer")
    _require_no_duration_consumption(value, owner)
    if (not isinstance(value["reference_path"], str)
            or not value["reference_path"].strip()):
        raise TypeError(f"{owner}.reference_path must be a non-empty string")
    reference_path = Path(value["reference_path"])
    if (reference_path.is_absolute()
            or ".." in reference_path.parts
            or reference_path.parts[:3] != (
                "microbench", _mechanism_reference_hardware_id(hardware_id), "references")):
        raise ValueError(f"{owner}.reference_path is outside its owner")
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"] != "dram_finite_read_stream_ramp_reference_v1"
            or reference["hardware_id"] != _mechanism_reference_hardware_id(hardware_id)
            or reference["mechanism"] != "dram_finite_read_stream_ramp"
            or reference["production_or_evaluation_duration_consumed"]
            is not False):
        raise ValueError(f"{owner} reference contract drifted")
    scope = reference["scope"]
    if (scope["threads_per_cta"] != value["threads_per_cta"]
            or scope["in_flight_bytes_per_cta"]
            != value["in_flight_bytes_per_cta"]):
        raise ValueError(f"{owner} reference scope differs from HWSpec")
    derived = reference["derived_model"]
    capacity = float(derived["dram_read_capacity_bytes_per_s"])
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError(f"{owner} reference capacity must be positive")
    ramp = {}
    for entry in derived["ramp"]:
        grid = entry["grid_ctas"]
        per_cta = entry["bytes_per_cta"]
        fraction = entry["achieved_fraction_of_dram_read_capacity"]
        for name, item in (("grid_ctas", grid), ("bytes_per_cta", per_cta)):
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"{owner} ramp {name} must be positive")
        if (isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))):
            raise TypeError(f"{owner} ramp fraction must be numeric")
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(f"{owner} ramp fraction must lie within (0, 1]")
        if (grid, per_cta) in ramp:
            raise ValueError(f"{owner} repeats ramp cell {(grid, per_cta)}")
        ramp[(grid, per_cta)] = fraction
    if not ramp:
        raise ValueError(f"{owner} ramp is empty")
    return {
        "model": value["model"],
        "timer_domain": value["timer_domain"],
        "threads_per_cta": value["threads_per_cta"],
        "in_flight_bytes_per_cta": value["in_flight_bytes_per_cta"],
        "dram_read_capacity_bytes_per_s": capacity,
        "grids": tuple(sorted({grid for grid, _ in ramp})),
        "lengths_bytes_per_cta": tuple(sorted({
            per_cta for _, per_cta in ramp})),
        "ramp": ramp,
        "ramp_by_threads": None,
        "reference_path": value["reference_path"],
        "reference_sha256": reference_sha256,
    }


def _normalize_dram_finite_read_stream_ramp_threaded(
        value, repository_root, hardware_id, owner):
    """Normalize a finite read-stream ramp measured at several request depths.

    Schema v1 pins the probe to one threads-per-CTA, and its own reference
    forbids reading it away from that depth.  On a device whose deployed
    weight-streaming launches run several block sizes, one surface therefore
    cannot describe them all: a fixed 128-thread probe reproduces the deployed
    C500 GEMV rates to between 0.660 and 1.005, and the same probe run at each
    kernel's own depth lands between 0.874 and 1.005.  v2 keeps the depth as
    part of the key instead.  v1 references are untouched and keep their path.
    """

    _require_exact_fields(owner, value, {
        "model", "timer_domain", "threads_per_cta_axis",
        "in_flight_bytes_per_cta_by_threads", "reference_path",
        "reference_sha256", "production_or_evaluation_duration_consumed",
    })
    if (not isinstance(value["timer_domain"], str)
            or not value["timer_domain"].strip()):
        raise TypeError(f"{owner}.timer_domain must be a non-empty string")
    axis = value["threads_per_cta_axis"]
    if (not isinstance(axis, list) or not axis
            or any(isinstance(item, bool) or not isinstance(item, int)
                   or item <= 0 for item in axis)
            or sorted(axis) != axis or len(set(axis)) != len(axis)):
        raise ValueError(
            f"{owner}.threads_per_cta_axis must be ascending positive integers")
    depths = value["in_flight_bytes_per_cta_by_threads"]
    if not isinstance(depths, dict) or set(depths) != {str(v) for v in axis}:
        raise ValueError(
            f"{owner}.in_flight_bytes_per_cta_by_threads must cover the axis")
    for name, item in depths.items():
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{owner} in-flight bytes for {name} must be positive")
    _require_no_duration_consumption(value, owner)
    if (not isinstance(value["reference_path"], str)
            or not value["reference_path"].strip()):
        raise TypeError(f"{owner}.reference_path must be a non-empty string")
    reference_path = Path(value["reference_path"])
    if (reference_path.is_absolute() or ".." in reference_path.parts
            or reference_path.parts[:3] != (
                "microbench", _mechanism_reference_hardware_id(hardware_id),
                "references")):
        raise ValueError(f"{owner}.reference_path is outside its owner")
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    if (reference["schema"] != "dram_finite_read_stream_ramp_reference_v2_threaded"
            or reference["hardware_id"]
            != _mechanism_reference_hardware_id(hardware_id)
            or reference["mechanism"] != "dram_finite_read_stream_ramp"
            or reference["production_or_evaluation_duration_consumed"]
            is not False):
        raise ValueError(f"{owner} reference contract drifted")
    scope = reference["scope"]
    if (scope["threads_per_cta_axis"] != axis
            or scope["in_flight_bytes_per_cta_by_threads"] != depths):
        raise ValueError(f"{owner} reference scope differs from HWSpec")
    derived = reference["derived_model"]
    capacity = float(derived["dram_read_capacity_bytes_per_s"])
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError(f"{owner} reference capacity must be positive")
    by_threads = derived["ramp_by_threads"]
    if set(by_threads) != {str(v) for v in axis}:
        raise ValueError(f"{owner} ramp does not cover the declared axis")
    ramp_by_threads = {}
    for threads_text, block in by_threads.items():
        if not isinstance(block, dict) or set(block) != {"cells", "provenance"}:
            raise ValueError(
                f"{owner} depth {threads_text} must carry cells and provenance")
        # Every depth is its own capture: its own binary, its own raw report.
        # A depth missing any of that identity is refused rather than trusted.
        depth_provenance = block["provenance"]
        required = {
            "source_commit", "source_sha256", "snapshot_sha256",
            "binary_sha256", "report_sha256",
            "production_or_evaluation_duration_consumed",
        }
        if (not isinstance(depth_provenance, dict)
                or set(depth_provenance) != required):
            raise ValueError(
                f"{owner} depth {threads_text} provenance must be exactly "
                f"{sorted(required)}")
        for name in ("source_sha256", "snapshot_sha256", "binary_sha256",
                     "report_sha256"):
            item = depth_provenance[name]
            if (not isinstance(item, str)
                    or re.fullmatch(r"[0-9a-f]{64}", item) is None):
                raise ValueError(
                    f"{owner} depth {threads_text} {name} must be "
                    "lowercase SHA256")
        commit = depth_provenance["source_commit"]
        if (not isinstance(commit, str)
                or re.fullmatch(r"[0-9a-f]{7,40}", commit) is None):
            raise ValueError(
                f"{owner} depth {threads_text} source_commit must be a "
                "hexadecimal commit")
        if (depth_provenance["production_or_evaluation_duration_consumed"]
                is not False):
            raise ValueError(
                f"{owner} depth {threads_text} cannot consume production "
                "or evaluation duration")
        entries = block["cells"]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{owner} depth {threads_text} has no cells")
        surface = {}
        for entry in entries:
            grid = entry["grid_ctas"]
            per_cta = entry["bytes_per_cta"]
            fraction = entry["achieved_fraction_of_dram_read_capacity"]
            for name, item in (("grid_ctas", grid), ("bytes_per_cta", per_cta)):
                if (isinstance(item, bool) or not isinstance(item, int)
                        or item <= 0):
                    raise ValueError(f"{owner} ramp {name} must be positive")
            if (isinstance(fraction, bool)
                    or not isinstance(fraction, (int, float))):
                raise TypeError(f"{owner} ramp fraction must be numeric")
            fraction = float(fraction)
            if not math.isfinite(fraction) or fraction <= 0.0:
                raise ValueError(f"{owner} ramp fraction must be positive")
            # The published capacity is the ceiling; a cell that measures
            # marginally above it is consumed at the ceiling, and the raw
            # value stays in the reference.
            fraction = min(fraction, 1.0)
            if (grid, per_cta) in surface:
                raise ValueError(
                    f"{owner} repeats ramp cell {(grid, per_cta)}")
            surface[(grid, per_cta)] = fraction
        if not surface:
            raise ValueError(f"{owner} ramp for {threads_text} is empty")
        ramp_by_threads[int(threads_text)] = {
            "ramp": surface,
            "grids": tuple(sorted({grid for grid, _ in surface})),
            "lengths_bytes_per_cta": tuple(sorted({
                per_cta for _, per_cta in surface})),
        }
    return {
        "model": value["model"],
        "timer_domain": value["timer_domain"],
        "threads_per_cta": None,
        "in_flight_bytes_per_cta": None,
        "dram_read_capacity_bytes_per_s": capacity,
        "grids": (),
        "lengths_bytes_per_cta": (),
        "ramp": None,
        "ramp_by_threads": ramp_by_threads,
        "reference_path": value["reference_path"],
        "reference_sha256": reference_sha256,
    }


def _normalize_l2_operand_residency_curve(
        value, repository_root, hardware_id):
    """Normalize the measured L2 residency curve of a repeated stream.

    The curve gives the DRAM-served byte fraction of a repeated same-operand
    single-pass stream at a complete operand-and-result footprint.  It
    replaces a binary residency threshold at the L2 capacity, which the
    measurement falsifies: at capacity the stream is only partly L2-served.
    """

    owner = "scheduling.l2_operand_residency_curve"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "timer_domain", "stream", "threads_per_cta",
        "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    if value["model"] != "l2_operand_residency_curve_v1":
        raise ValueError(f"{owner}.model is unsupported")
    for name in ("timer_domain", "stream"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise TypeError(f"{owner}.{name} must be a non-empty string")
    threads_per_cta = value["threads_per_cta"]
    if isinstance(threads_per_cta, bool) or not isinstance(
            threads_per_cta, int) or threads_per_cta <= 0:
        raise ValueError(f"{owner}.threads_per_cta must be a positive integer")
    _require_no_duration_consumption(value, owner)
    if (not isinstance(value["reference_path"], str)
            or not value["reference_path"].strip()):
        raise TypeError(f"{owner}.reference_path must be a non-empty string")
    reference_path = Path(value["reference_path"])
    if (reference_path.is_absolute()
            or ".." in reference_path.parts
            or reference_path.parts[:3] != (
                "microbench", _mechanism_reference_hardware_id(hardware_id), "references")):
        raise ValueError(f"{owner}.reference_path is outside its owner")
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    _require_fields_with_optional(f"{owner} reference", reference, {
        "allowed_use", "derived_model", "forbidden_use", "hardware_id",
        "mechanism", "production_or_evaluation_duration_consumed",
        "provenance", "schema", "scope"}, _REFERENCE_OPTIONAL_FIELDS)
    if (reference["schema"] != "l2_operand_residency_curve_reference_v1"
            or reference["hardware_id"] != _mechanism_reference_hardware_id(hardware_id)
            or reference["mechanism"] != "l2_operand_residency_curve"
            or reference["production_or_evaluation_duration_consumed"]
            is not False):
        raise ValueError(f"{owner} reference contract drifted")
    scope = reference["scope"]
    if not isinstance(scope, dict):
        raise TypeError(f"{owner} reference scope must be a mapping")
    if (scope["stream"] != value["stream"]
            or scope["threads_per_cta"] != threads_per_cta):
        raise ValueError(f"{owner} reference scope differs from HWSpec")
    derived = reference["derived_model"]
    if not isinstance(derived, dict):
        raise TypeError(f"{owner} reference derived model must be a mapping")
    curve = derived["residency_curve"]
    if not isinstance(curve, list) or len(curve) < 2:
        raise TypeError(f"{owner} residency curve needs at least two points")
    points = []
    previous_bytes = None
    previous_fraction = None
    for entry in curve:
        if not isinstance(entry, dict):
            raise TypeError(f"{owner} residency curve entries must be mappings")
        footprint = entry["total_footprint_bytes"]
        fraction = entry["dram_served_byte_fraction"]
        if isinstance(footprint, bool) or not isinstance(footprint, int)                 or footprint <= 0:
            raise ValueError(
                f"{owner} residency footprint must be a positive integer")
        if isinstance(fraction, bool) or not isinstance(
                fraction, (int, float)):
            raise TypeError(f"{owner} residency fraction must be numeric")
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"{owner} residency fraction must lie within [0, 1]")
        if previous_bytes is not None and footprint <= previous_bytes:
            raise ValueError(
                f"{owner} residency curve must increase in footprint")
        if previous_fraction is not None and fraction < previous_fraction:
            raise ValueError(
                f"{owner} residency fraction must be non-decreasing")
        previous_bytes, previous_fraction = footprint, fraction
        points.append((footprint, fraction))
    if points[0][0] != scope["minimum_footprint_bytes"] \
            or points[-1][0] != scope["maximum_footprint_bytes"]:
        raise ValueError(f"{owner} residency curve does not span its scope")
    if points[-1][1] != 1.0:
        raise ValueError(
            f"{owner} residency curve must end fully DRAM-served")
    return {
        "model": value["model"],
        "timer_domain": value["timer_domain"],
        "stream": value["stream"],
        "threads_per_cta": threads_per_cta,
        "l2_capacity_bytes": scope["l2_capacity_bytes"],
        "residency_curve": tuple(points),
        "reference_path": value["reference_path"],
        "reference_sha256": reference_sha256,
    }


def _normalize_finite_async_operand_stream_completion(
        value, repository_root, hardware_id):
    """Normalize one topology-bounded finite async-copy completion floor."""

    owner = "scheduling.finite_async_operand_stream_completion"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "timer_domain", "dtype", "threads_per_cta",
        "grid_ctas", "resident_ctas_per_compute_unit",
        "operand_pair_bytes", "complete_operand_pair_slots",
        "declared_pipeline_depth", "minimum_rounds", "maximum_rounds",
        "fixed_completion_ns", "l2_resident_ns_per_round",
        "dram_unique_ns_per_round", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    if value["model"] != "finite_async_operand_stream_completion_v1":
        raise ValueError(f"{owner}.model is unsupported")
    for name in ("timer_domain", "dtype"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise TypeError(f"{owner}.{name} must be a non-empty string")
    integer_names = (
        "threads_per_cta", "grid_ctas",
        "resident_ctas_per_compute_unit", "operand_pair_bytes",
        "complete_operand_pair_slots", "declared_pipeline_depth",
        "minimum_rounds", "maximum_rounds",
    )
    integers = {}
    for name in integer_names:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        integers[name] = item
    if integers["complete_operand_pair_slots"] > integers[
            "declared_pipeline_depth"]:
        raise ValueError(
            f"{owner}.complete_operand_pair_slots exceeds the declared "
            "pipeline depth")
    if integers["minimum_rounds"] > integers["maximum_rounds"]:
        raise ValueError(f"{owner} round domain is inverted")
    numeric_names = (
        "fixed_completion_ns", "l2_resident_ns_per_round",
        "dram_unique_ns_per_round",
    )
    numerics = {}
    for name in numeric_names:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{owner}.{name} must be numeric")
        normalized = float(item)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        numerics[name] = normalized
    _require_no_duration_consumption(value, owner)
    if (not isinstance(value["reference_path"], str)
            or not value["reference_path"].strip()):
        raise TypeError(f"{owner}.reference_path must be a non-empty string")
    reference_path = Path(value["reference_path"])
    if (reference_path.is_absolute()
            or ".." in reference_path.parts
            or reference_path.parts[:3] != (
                "microbench", _mechanism_reference_hardware_id(hardware_id), "references")):
        raise ValueError(f"{owner}.reference_path is outside its owner")
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    reference_fields = {
        "allowed_use", "derived_model", "forbidden_use", "hardware_id",
        "mechanism", "production_or_evaluation_duration_consumed",
        "provenance", "schema", "scope",
    }
    _require_fields_with_optional(
        f"{owner} reference", reference, reference_fields,
        _REFERENCE_OPTIONAL_FIELDS)
    if (reference["schema"]
            != "finite_async_operand_stream_completion_reference_v1"
            or reference["hardware_id"] != _mechanism_reference_hardware_id(hardware_id)
            or reference["mechanism"] != "finite_async_operand_stream_completion"
            or reference["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError(f"{owner} reference contract drifted")
    scope = reference["scope"]
    if not isinstance(scope, dict):
        raise TypeError(f"{owner} reference scope must be a mapping")
    expected_scope = {
        "dtype": value["dtype"],
        "threads_per_cta": integers["threads_per_cta"],
        "grid_ctas": integers["grid_ctas"],
        "resident_ctas_per_compute_unit": integers[
            "resident_ctas_per_compute_unit"],
        "operand_pair_bytes": integers["operand_pair_bytes"],
        "complete_operand_pair_slots": integers[
            "complete_operand_pair_slots"],
        "minimum_rounds": integers["minimum_rounds"],
        "maximum_rounds": integers["maximum_rounds"],
    }
    if scope != expected_scope:
        raise ValueError(f"{owner} reference scope differs from HWSpec")
    derived = reference["derived_model"]
    if not isinstance(derived, dict):
        raise TypeError(f"{owner} reference derived model must be a mapping")
    _require_exact_fields(
        f"{owner} reference derived model", derived,
        {"cache_tiers", "fixed_completion_ns", "formula",
         "selection_rule", "verification"})
    if not isinstance(derived["cache_tiers"], dict):
        raise TypeError(f"{owner} reference cache tiers must be a mapping")
    cache_tiers = derived["cache_tiers"]
    if set(cache_tiers) != {"l2_resident", "dram_unique"}:
        raise ValueError(f"{owner} reference cache tiers are incomplete")
    expected_numerics = {
        "fixed_completion_ns": numerics["fixed_completion_ns"],
        "l2_resident_ns_per_round": numerics[
            "l2_resident_ns_per_round"],
        "dram_unique_ns_per_round": numerics[
            "dram_unique_ns_per_round"],
    }
    observed_numerics = {
        "fixed_completion_ns": derived["fixed_completion_ns"],
        "l2_resident_ns_per_round": cache_tiers["l2_resident"][
            "conservative_ns_per_round"],
        "dram_unique_ns_per_round": cache_tiers["dram_unique"][
            "conservative_ns_per_round"],
    }
    if observed_numerics != expected_numerics:
        raise ValueError(f"{owner} reference parameters differ from HWSpec")
    return {
        "model": value["model"],
        "timer_domain": value["timer_domain"],
        "dtype": value["dtype"],
        **integers,
        **numerics,
        "reference_path": value["reference_path"],
        "reference_sha256": reference_sha256,
    }


def _normalize_finite_mixed_operand_stream_startup(
        value, repository_root, hardware_id):
    """Normalize the launch-local mixed-tier operand-stream startup."""

    owner = "scheduling.finite_mixed_operand_stream_startup"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    if value["model"] == "finite_operand_stream_startup_profiles_v2":
        _require_exact_fields(owner, value, {"model", "profiles"})
        if not isinstance(value["profiles"], list) or not value["profiles"]:
            raise ValueError(f"{owner}.profiles must be a non-empty list")
        profiles = []
        identities = set()
        for profile in value["profiles"]:
            if not isinstance(profile, dict) or profile["model"] != "finite_mixed_operand_stream_startup_v1":
                raise ValueError(f"{owner} profiles must be v1 mechanism records")
            normalized, = _normalize_finite_mixed_operand_stream_startup(
                profile, repository_root, hardware_id)
            identity = tuple(normalized[name] for name in (
                "transport", "dtype_element_bytes", "threads_per_cta", "tile_k"))
            if identity in identities:
                raise ValueError(f"{owner} has ambiguous transport profiles")
            identities.add(identity)
            profiles.append(normalized)
        return tuple(profiles)
    fields = {
        "model", "timer_domain", "dtype_element_bytes", "threads_per_cta",
        "tile_k", "minimum_pipeline_depth", "maximum_pipeline_depth",
        "minimum_rounds", "maximum_rounds", "one_wave_grid_ctas",
        "mixed_tier_startup_ns", "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_fields_with_optional(
        owner, value, fields, {"transport", "minimum_grid_ctas"})
    if value["model"] != "finite_mixed_operand_stream_startup_v1":
        raise ValueError(f"{owner}.model is unsupported")
    _require_no_duration_consumption(value, owner)
    # v1 predates typed transports and describes the LDGSTS ring. Normalize
    # that schema default here, before any predictor dispatch.
    transport = value["transport"] if "transport" in value else "ldgsts"
    if transport not in ("ldgsts", "tma"):
        raise ValueError(f"{owner}.transport is unsupported")
    minimum_grid = value["minimum_grid_ctas"] if "minimum_grid_ctas" in value else 1
    if isinstance(minimum_grid, bool) or not isinstance(minimum_grid, int) or minimum_grid <= 0:
        raise ValueError(f"{owner}.minimum_grid_ctas must be a positive integer")
    if (not isinstance(value["timer_domain"], str)
            or not value["timer_domain"].strip()):
        raise TypeError(f"{owner}.timer_domain must be a non-empty string")
    integer_names = (
        "dtype_element_bytes", "threads_per_cta", "tile_k",
        "minimum_pipeline_depth", "maximum_pipeline_depth",
        "minimum_rounds", "maximum_rounds", "one_wave_grid_ctas",
    )
    integers = {}
    for name in integer_names:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        integers[name] = item
    if integers["minimum_pipeline_depth"] > integers[
            "maximum_pipeline_depth"]:
        raise ValueError(f"{owner} pipeline-depth domain is inverted")
    if integers["minimum_rounds"] > integers["maximum_rounds"]:
        raise ValueError(f"{owner} round domain is inverted")
    if minimum_grid > integers["one_wave_grid_ctas"]:
        raise ValueError(f"{owner} grid domain is inverted")
    item = value["mixed_tier_startup_ns"]
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise TypeError(f"{owner}.mixed_tier_startup_ns must be numeric")
    startup_ns = float(item)
    if not math.isfinite(startup_ns) or startup_ns <= 0.0:
        raise ValueError(
            f"{owner}.mixed_tier_startup_ns must be finite and positive")
    _require_no_duration_consumption(value, owner)
    if (not isinstance(value["reference_path"], str)
            or not value["reference_path"].strip()):
        raise TypeError(f"{owner}.reference_path must be a non-empty string")
    reference_path = Path(value["reference_path"])
    if (reference_path.is_absolute()
            or ".." in reference_path.parts
            or reference_path.parts[:3] != (
                "microbench", _mechanism_reference_hardware_id(hardware_id), "references")):
        raise ValueError(f"{owner}.reference_path is outside its owner")
    reference_sha256 = value["reference_sha256"]
    if (not isinstance(reference_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None):
        raise ValueError(f"{owner}.reference_sha256 must be lowercase SHA256")
    reference_bytes = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != reference_sha256:
        raise ValueError(f"{owner} reference SHA256 mismatch")
    reference = json.loads(reference_bytes)
    reference_fields = {
        "allowed_use", "captures", "derived_model", "device_name",
        "forbidden_use", "hardware_id", "mechanism", "probe_contract",
        "production_or_evaluation_duration_consumed", "runner",
        "runner_sha256", "schema", "scope", "source", "source_sha256",
    }
    _require_fields_with_optional(
        f"{owner} reference", reference, reference_fields,
        _REFERENCE_OPTIONAL_FIELDS)
    if (reference["schema"]
            != "finite_mixed_operand_stream_startup_reference_v1"
            or reference["hardware_id"] != _mechanism_reference_hardware_id(hardware_id)
            or reference["mechanism"] != "finite_mixed_operand_stream_startup"
            or reference["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError(f"{owner} reference contract drifted")
    scope = reference["scope"]
    if not isinstance(scope, dict):
        raise TypeError(f"{owner} reference scope must be a mapping")
    expected_scope = {
        "dtype_element_bytes": integers["dtype_element_bytes"],
        "threads_per_cta": integers["threads_per_cta"],
        "tile_k": integers["tile_k"],
        "minimum_pipeline_depth": integers["minimum_pipeline_depth"],
        "maximum_pipeline_depth": integers["maximum_pipeline_depth"],
        "minimum_rounds": integers["minimum_rounds"],
        "maximum_rounds": integers["maximum_rounds"],
        "one_wave_grid_ctas": integers["one_wave_grid_ctas"],
    }
    if "transport" in value:
        expected_scope["transport"] = transport
    if "minimum_grid_ctas" in value:
        expected_scope["minimum_grid_ctas"] = minimum_grid
    if scope != expected_scope:
        raise ValueError(f"{owner} reference scope differs from HWSpec")
    derived = reference["derived_model"]
    if not isinstance(derived, dict):
        raise TypeError(f"{owner} reference derived model must be a mapping")
    if (derived["one_wave_mixed_tier_startup_ns"] != startup_ns):
        raise ValueError(f"{owner} reference parameter differs from HWSpec")
    return ({
        "model": value["model"],
        "timer_domain": value["timer_domain"],
        "transport": transport,
        "minimum_grid_ctas": minimum_grid,
        **integers,
        "mixed_tier_startup_ns": startup_ns,
        "reference_path": value["reference_path"],
        "reference_sha256": reference_sha256,
    },)


def _normalize_ordinary_u16_store_completion(value, repository_root, hardware):
    """All-control store/no-store lower envelope after peak service subtraction."""
    owner = "scheduling.ordinary_u16_store_completion"
    hardware_id = hardware.hardware_id
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    reference = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", reference, {
        "schema_version", "hardware_id", "mechanism", "selection", "capture",
        "measurements", "completion_ns", "scope_limitations",
        "production_or_evaluation_duration_consumed"})
    if (type(reference["schema_version"]) is not int or reference["schema_version"] != 1
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "ordinary_u16_global_store_completion"
            or reference["selection"] !=
            "minimum_all_paired_medians_minus_peak_byte_and_active_sm_stg_issue_service"
            or reference["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError(f"{owner} evidence boundary is invalid")
    capture = reference["capture"]
    _require_exact_fields(owner + " capture", capture, {"path", "sha256"})
    if (not isinstance(capture["path"], str)
            or not capture["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in capture["path"].split("/")
            or not isinstance(capture["sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", capture["sha256"]) is None):
        raise ValueError(f"{owner} capture pin is invalid")
    limitations = reference["scope_limitations"]
    if (not isinstance(limitations, list) or not limitations
            or any(not isinstance(x, str) or not x for x in limitations)):
        raise ValueError(f"{owner} requires scope limitations")
    sms = hardware.num_compute_units
    bandwidth = min(hardware.resource_by_role(role).capacity
                    for role in ("l2_cache", "dram"))
    chip_issue_rate = hardware.resource_by_role("warp_issue").capacity
    expected = {(g, w, f) for g in (1, 32, 132, 896)
                for w in (1, 4, 16) for f in (0, 1)}
    seen = set()
    residuals = []
    for row in reference["measurements"]:
        _require_exact_fields(owner + " measurement", row, {
            "grid", "threads", "active_warps", "fresh_node_addresses",
            "graph_nodes", "registers_per_thread", "static_shared_bytes",
            "no_store_ns_p50", "store_ns_p50", "paired_increment_ns_p50",
            "all_output_and_baseline_sentinels_checked", "paired_samples_ns"})
        key = tuple(row[name] for name in ("grid", "active_warps", "fresh_node_addresses"))
        if (any(type(x) is not int for x in key) or key not in expected or key in seen
                or any(type(row[name]) is not int or row[name] != fixed
                       for name, fixed in (("threads", 512), ("graph_nodes", 128),
                           ("registers_per_thread", 16), ("static_shared_bytes", 8192)))
                or row["all_output_and_baseline_sentinels_checked"] is not True):
            raise ValueError(f"{owner} invalid or duplicate control")
        seen.add(key)
        samples = row["paired_samples_ns"]
        if (not isinstance(samples, list) or len(samples) != 9
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or x <= 0 for x in (
                           *samples, row["no_store_ns_p50"], row["store_ns_p50"],
                           row["paired_increment_ns_p50"]))
                or statistics.median(samples) != row["paired_increment_ns_p50"]):
            raise ValueError(f"{owner} requires all nine paired samples")
        grid, warps, _fresh = key
        # Warp-issue capacity is chip-shared in HWSpec; only active SMs issue
        # this probe's extra STG.  Keep the physical SM count, not SM subpipes.
        issue_rate = chip_issue_rate * min(grid, sms) / sms
        nominal_ns = (grid * warps * 32 * 2 / bandwidth
                      + grid * warps / issue_rate) * 1e9
        residuals.append(row["paired_increment_ns_p50"] - nominal_ns)
    if seen != expected or not residuals or min(residuals) <= 0:
        raise ValueError(f"{owner} requires positive residuals for every control")
    selected = reference["completion_ns"]
    if (isinstance(selected, bool) or not isinstance(selected, (int, float))
            or not math.isfinite(selected) or selected != min(residuals)):
        raise ValueError(f"{owner} completion differs from the all-control minimum")
    return {
        "completion_ns": selected,
        "maximum_grid_size": max(grid for grid, _warps, _fresh in expected),
        "threads_per_cta": 512,
        "element_bytes": 2,
    }


def _normalize_gemm_tensor_instruction_service(value, repository_root, hardware_id):
    """Independent same-compiler WGMMA cycle capacity, without ns prices."""
    owner = "scheduling.gemm_tensor_instruction_service"
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    reference = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", reference, {
        "schema_version", "hardware_id", "mechanism", "selection", "capture",
        "measurements", "instruction_rates", "scope_limitations",
        "production_or_evaluation_duration_consumed"})
    if (type(reference["schema_version"]) is not int or reference["schema_version"] != 1
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "shared_operand_fp16_wgmma_instruction_window"
            or reference["selection"] != "random_seed_median_secant_then_maximum_measured_window_capacity"
            or reference["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError(f"{owner} evidence boundary is invalid")
    capture = reference["capture"]
    _require_exact_fields(owner + " capture", capture, {"path", "sha256", "source_sha256", "runner_sha256"})
    if (not isinstance(capture["path"], str)
            or not capture["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in capture["path"].split("/")):
        raise ValueError(f"{owner} raw path is invalid")
    for name in ("sha256", "source_sha256", "runner_sha256"):
        if not isinstance(capture[name], str) or re.fullmatch("[0-9a-f]{64}", capture[name]) is None:
            raise ValueError(f"{owner} content-address pin is invalid")
    limitations = reference["scope_limitations"]
    if not isinstance(limitations, list) or not limitations or any(not isinstance(x, str) or not x for x in limitations):
        raise ValueError(f"{owner} requires scope limitations")
    observations = {}
    expected = {(n, a, w, p, s, r) for n in (8, 16, 32, 64) for a in (1, 2, 4)
        for w in (1, 2) for p in (0, 1, 3) for s in (1, 2, 3) for r in (128, 512)}
    for row in reference["measurements"]:
        names = ("instruction_n", "chains", "math_warpgroups", "pending_groups", "seed", "rounds")
        _require_exact_fields(owner + " measurement", row, {*names, "cycles"})
        key = tuple(row[name] for name in names)
        if any(type(row[name]) is not int for name in names) or key not in expected or key in observations:
            raise ValueError(f"{owner} duplicated or invalid control")
        cycles = row["cycles"]
        if isinstance(cycles, bool) or not isinstance(cycles, (int, float)) or not math.isfinite(cycles) or cycles <= 0:
            raise ValueError(f"{owner} cycles must be finite and positive")
        observations[key] = cycles
    if set(observations) != expected:
        raise ValueError(f"{owner} requires all random controls")
    rates = {}
    for n in (8, 16, 32, 64):
        windows = []
        for a in (1, 2, 4):
            for w in (1, 2):
                for p in (0, 1, 3):
                    seeds = []
                    for s in (1, 2, 3):
                        delta = observations[n, a, w, p, s, 512] - observations[n, a, w, p, s, 128]
                        if delta <= 0:
                            raise ValueError(f"{owner} cycle secant must be positive")
                        seeds.append(384 * w * 4 * a * 2 * 64 * n * 16 / delta)
                    windows.append(statistics.median(seeds))
        rates[n] = max(windows)
    published = [{"instruction_n": n, "flops_per_sm_cycle": rate} for n, rate in rates.items()]
    if reference["instruction_rates"] != published:
        raise ValueError(f"{owner} selected rates differ from all-control capacity envelope")
    return {"instruction_m": 64, "instruction_k": 16, "instruction_rates": rates}


def _normalize_gemm_tensor_activity_clock(value, repository_root, hardware_id):
    """Read an independent Tensor clock, optionally with same-probe service."""
    owner = "scheduling.gemm_tensor_activity_clock"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    if "additional_concurrency_profiles" in value:
        fields = {"reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"}
        _require_exact_fields(owner, value, fields | {"additional_concurrency_profiles"})
        primary = _normalize_gemm_tensor_activity_clock(
            {name: value[name] for name in fields}, repository_root, hardware_id)
        if (primary["flops_per_sm_cycle"] is None
                or primary["minimum_active_units"] != primary["maximum_active_units"]):
            raise ValueError(f"{owner} concurrency profiles require exact shared-Tensor controls")
        records = value["additional_concurrency_profiles"]
        if not isinstance(records, list) or not records:
            raise ValueError(f"{owner} additional profiles must be a nonempty list")
        profiles = {}
        for record in records:
            _require_exact_fields(owner + " additional profile", record, fields)
            profile = _normalize_gemm_tensor_activity_clock(record, repository_root, hardware_id)
            active = profile["minimum_active_units"]
            if (profile["flops_per_sm_cycle"] is None
                    or active != profile["maximum_active_units"]
                    or active in profiles
                    or primary["minimum_active_units"] <= active <= primary["maximum_active_units"]
                    or profile["physical_grid_ctas"] != primary["physical_grid_ctas"]
                    or profile["threads_per_cta"] != primary["threads_per_cta"]):
                raise ValueError(f"{owner} concurrency domains must be distinct exact controls")
            profiles[active] = profile
        return {**primary, "concurrency_profiles": profiles}
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    reference = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    if not isinstance(reference, dict) or type(reference["schema_version"]) is not int:
        raise TypeError(f"{owner} reference version must be an integer")
    if reference["schema_version"] in (4, 5, 6, 7, 8):
        return _normalize_gemm_accumulator_activity_service(reference, owner, hardware_id)
    if reference["schema_version"] in (2, 3):
        return _normalize_gemm_shared_tensor_service(reference, owner, hardware_id)
    _require_exact_fields(owner + " reference", reference, {
        "schema_version", "hardware_id", "mechanism", "selection", "measurements",
        "selected_sm_clock_hz", "raw", "source_sha256", "runner_sha256",
        "scope_limitations", "production_or_evaluation_duration_consumed"})
    if (reference["schema_version"] != 1 or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "register_tensor_activity_clock"
            or reference["selection"] != "median_of_all_eight_control_point_clock_medians"
            or reference["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError(f"{owner} evidence boundary is invalid")
    for name in ("source_sha256", "runner_sha256"):
        if not isinstance(reference[name], str) or re.fullmatch("[0-9a-f]{64}", reference[name]) is None:
            raise ValueError(f"{owner} source identity is invalid")
    raw = reference["raw"]
    if not isinstance(raw, dict):
        raise TypeError(f"{owner} raw pin must be a mapping")
    _require_exact_fields(owner + " raw", raw, {"path", "sha256"})
    if (not isinstance(raw["path"], str)
            or not raw["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in raw["path"].split("/")
            or not isinstance(raw["sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", raw["sha256"]) is None):
        raise ValueError(f"{owner} raw content-address pin is invalid")
    if (not isinstance(reference["scope_limitations"], list)
            or not reference["scope_limitations"]
            or any(not isinstance(item, str) or not item for item in reference["scope_limitations"])):
        raise ValueError(f"{owner} must disclose its scope limitations")
    rows = reference["measurements"]
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError(f"{owner} requires eight independent clock controls")
    identities, clocks, grids = set(), [], set()
    for row in rows:
        _require_exact_fields(owner + " measurement", row, {
            "grid_ctas", "rounds", "patterned", "clock_mhz"})
        identity = tuple(row[name] for name in ("grid_ctas", "rounds", "patterned"))
        if (any(isinstance(row[name], bool) or not isinstance(row[name], int)
                or row[name] <= 0 for name in ("grid_ctas", "rounds"))
                or isinstance(row["patterned"], bool)
                or row["patterned"] not in (0, 1) or identity in identities):
            raise ValueError(f"{owner} clock controls are invalid or duplicated")
        frequency = row["clock_mhz"]
        if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"{owner} clock must be finite and positive")
        identities.add(identity)
        clocks.append(float(frequency) * 1e6)
        grids.add(row["grid_ctas"])
    if len(grids) != 2 or len({row["rounds"] for row in rows}) != 2:
        raise ValueError(f"{owner} requires two grid and two length controls")
    clocks.sort()
    selected = (clocks[3] + clocks[4]) / 2
    if reference["selected_sm_clock_hz"] != selected:
        raise ValueError(f"{owner} selected clock is not the protocol median")
    return {"sm_freq_hz": selected, "minimum_active_units": min(grids),
        "maximum_active_units": max(grids), "flops_per_sm_cycle": None,
        "physical_grid_ctas": None,
        "activity_clock_points": None, "math_threads_per_cta": None}


def _normalize_gemm_shared_tensor_service(reference, owner, hardware_id):
    """Keep cycle throughput and operating frequency in consistent units."""
    _require_exact_fields(owner + " reference", reference, {
        "schema_version", "hardware_id", "mechanism", "selection", "grid_ctas",
        "threads_per_cta", "math_warpgroups", "instruction_shape",
        "instructions_per_round_per_warpgroup", "measurements",
        "selected_flops_per_sm_cycle", "selected_sm_clock_hz", "raw",
        "source_sha256", "runner_sha256", "scope_limitations",
        "production_or_evaluation_duration_consumed"})
    version = reference["schema_version"]
    if version == 2:
        selection = "median_of_all_four_length_secants_and_eight_clock_medians"
        axes = ((128, 0), (128, 1), (256, 0), (256, 1))
        axis_fields = ("copy_rows_control", "cached_control")
    elif version == 3:
        selection = "median_of_all_three_random_seed_length_secants_and_six_clock_medians"
        axes = ((1,), (2,), (3,))
        axis_fields = ("operand_seed",)
    else:
        raise ValueError(f"{owner} Tensor service version is invalid")
    if (reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "shared_operand_fp16_wgmma_activity_service"
            or reference["selection"] != selection
            or reference["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError(f"{owner} evidence boundary is invalid")
    for name in ("source_sha256", "runner_sha256"):
        if not isinstance(reference[name], str) or re.fullmatch("[0-9a-f]{64}", reference[name]) is None:
            raise ValueError(f"{owner} source identity is invalid")
    raw = reference["raw"]
    _require_exact_fields(owner + " raw", raw, {"path", "sha256"})
    if (not isinstance(raw["path"], str)
            or not raw["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in raw["path"].split("/")
            or not isinstance(raw["sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", raw["sha256"]) is None):
        raise ValueError(f"{owner} raw content-address pin is invalid")
    if (not isinstance(reference["scope_limitations"], list)
            or not reference["scope_limitations"]
            or any(not isinstance(item, str) or not item for item in reference["scope_limitations"])):
        raise ValueError(f"{owner} must disclose its scope limitations")
    for name, expected in (("grid_ctas", 132), ("threads_per_cta", 384),
            ("math_warpgroups", 2), ("instructions_per_round_per_warpgroup", 4)):
        if type(reference[name]) is not int or reference[name] != expected:
            raise ValueError(f"{owner} probe geometry is invalid")
    if reference["instruction_shape"] != [64, 128, 16]:
        raise ValueError(f"{owner} WGMMA instruction shape is invalid")
    rows = reference["measurements"]
    if not isinstance(rows, list) or len(rows) != 2 * len(axes):
        raise ValueError(f"{owner} requires all declared math-only controls")
    controls, clocks = {}, []
    for row in rows:
        _require_exact_fields(owner + " measurement", row, {
            *axis_fields, "rounds", "math_cycles", "clock_mhz"})
        identity = tuple(row[name] for name in (*axis_fields, "rounds"))
        if (any(type(row[name]) is not int for name in (*axis_fields, "rounds"))
                or identity[:-1] not in axes
                or identity[-1] not in (128, 512) or identity in controls):
            raise ValueError(f"{owner} math controls are invalid or duplicated")
        for name in ("math_cycles", "clock_mhz"):
            number = row[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ValueError(f"{owner} service observations must be finite and positive")
        controls[identity] = row["math_cycles"]
        clocks.append(row["clock_mhz"] * 1e6)
    m, n, k = reference["instruction_shape"]
    round_flops = 2 * m * n * k * reference["math_warpgroups"] * reference["instructions_per_round_per_warpgroup"]
    rates = []
    for axis in axes:
        cycles = controls[(*axis, 512)] - controls[(*axis, 128)]
        if cycles <= 0:
            raise ValueError(f"{owner} cycle secant must be positive")
        rates.append((512 - 128) * round_flops / cycles)
    rate, clock = statistics.median(rates), statistics.median(clocks)
    if (reference["selected_flops_per_sm_cycle"] != rate
            or reference["selected_sm_clock_hz"] != clock):
        raise ValueError(f"{owner} selected service is not the protocol median")
    return {"sm_freq_hz": clock, "flops_per_sm_cycle": rate,
        "minimum_active_units": reference["grid_ctas"],
        "maximum_active_units": reference["grid_ctas"],
        "physical_grid_ctas": reference["grid_ctas"],
        "threads_per_cta": reference["threads_per_cta"],
        "activity_clock_points": None, "math_threads_per_cta": 256}


def _gemm_accumulator_activity_statistics(rows, owner, *, measurement_repetitions=1):
    """Aggregate independent SM90 controls, never production operator times."""
    geometries = (
        ((64, 1), (64, 3), (96, 1), (96, 3), (128, 1), (128, 3)),
        ((64, 2), (64, 4), (64, 5), (96, 2), (96, 3), (96, 4), (128, 1), (128, 2)))
    if type(measurement_repetitions) is not int or measurement_repetitions <= 0:
        raise ValueError(f"{owner} requires positive integer measurement repetitions")
    geometries = geometries * measurement_repetitions
    expected = {(capture, n, chains, seed, rounds)
        for capture, group in enumerate(geometries) for n, chains in group
        for seed in (1, 2, 3) for rounds in (128, 512)}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f"{owner} requires all random activity controls")
    observations = {}
    for row in rows:
        _require_exact_fields(owner + " measurement", row, {
            "capture_index", "instruction_n", "accumulator_chains", "operand_seed",
            "rounds", "math_cycles", "clock_mhz"})
        identity_fields = ("capture_index", "instruction_n", "accumulator_chains", "operand_seed", "rounds")
        identity = tuple(row[name] for name in identity_fields)
        if (any(type(row[name]) is not int for name in identity_fields)
                or identity not in expected or identity in observations):
            raise ValueError(f"{owner} activity controls are invalid or duplicated")
        for name in ("math_cycles", "clock_mhz"):
            number = row[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ValueError(f"{owner} activity observations must be finite and positive")
        observations[identity] = row
    per_geometry = {}
    for capture, group in enumerate(geometries):
        for n, chains in group:
            clocks, rates = [], []
            round_flops = 2 * 4 * chains * 2 * 64 * n * 16
            for seed in (1, 2, 3):
                short, long = (observations[capture, n, chains, seed, rounds]
                    for rounds in (128, 512))
                cycles = long["math_cycles"] - short["math_cycles"]
                if cycles <= 0:
                    raise ValueError(f"{owner} cycle secant must be positive")
                rates.append(384 * round_flops / cycles)
                clocks.extend(row["clock_mhz"] * 1e6 for row in (short, long))
            geometry = (n, chains)
            if geometry not in per_geometry:
                per_geometry[geometry] = []
            per_geometry[geometry].append((statistics.median(clocks), statistics.median(rates)))
    footprints, rates = {}, []
    for (n, chains), captures in per_geometry.items():
        clock = statistics.median(point[0] for point in captures)
        rates.append(statistics.median(point[1] for point in captures))
        footprint = n * chains // 2
        if footprint not in footprints:
            footprints[footprint] = []
        footprints[footprint].append(clock)
    return statistics.median(rates), [
        {"accumulator_values_per_thread": footprint, "sm_freq_hz": statistics.median(clocks)}
        for footprint, clocks in sorted(footprints.items())]


def _normalize_gemm_accumulator_activity_service(reference, owner, hardware_id):
    version = reference["schema_version"]
    if type(version) is not int or version not in (4, 5, 6, 7, 8):
        raise ValueError(f"{owner} accumulator activity version is unsupported")
    fields = {
        "schema_version", "hardware_id", "mechanism", "selection", "grid_ctas",
        "threads_per_cta", "measurement_threads_per_cta", "math_warpgroups",
        "captures", "measurements", "selected_flops_per_sm_cycle", "activity_points",
        "scope_limitations", "production_or_evaluation_duration_consumed"}
    if version >= 5:
        fields.add("operand_activity_pattern")
    if version >= 6:
        fields.add("measurement_protocol")
    if version == 7:
        fields.add("measurement_repetitions")
    if version == 8:
        fields.add("active_ctas")
    _require_exact_fields(owner + " reference", reference, fields)
    if (version >= 5 and reference["operand_activity_pattern"]
            != "random_sign_exponent_and_mantissa"):
        raise ValueError(f"{owner} activity requires the declared random fraction pattern")
    if version >= 6:
        protocol = reference["measurement_protocol"]
        _require_exact_fields(owner + " measurement protocol", protocol, {
            "kind", "device_warmup_reference_cycles", "accumulator_reset_each_window"})
        if (protocol["kind"] != "host_gap_free_device_warmup"
                or type(protocol["device_warmup_reference_cycles"]) is not int
                or protocol["device_warmup_reference_cycles"] != 96 * 1024 * 1024
                or protocol["accumulator_reset_each_window"] is not True):
            raise ValueError(f"{owner} sustained device activity protocol is invalid")
    if (reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "shared_operand_fp16_wgmma_accumulator_activity_service"
            or reference["selection"] != "median_per_capture_then_geometry_then_accumulator_footprint"
            or reference["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError(f"{owner} accumulator activity evidence boundary is invalid")
    for name, expected in (("grid_ctas", 132), ("threads_per_cta", 384),
            ("measurement_threads_per_cta", 256), ("math_warpgroups", 2)):
        if type(reference[name]) is not int or reference[name] != expected:
            raise ValueError(f"{owner} activity probe geometry is invalid")
    captures = reference["captures"]
    repetitions = reference["measurement_repetitions"] if version == 7 else 1
    if (type(repetitions) is not int or repetitions <= 0
            or (version == 7 and repetitions < 2)):
        raise ValueError(f"{owner} replicated activity needs at least two complete repetitions")
    if not isinstance(captures, list) or len(captures) != 2 * repetitions:
        raise ValueError(f"{owner} requires both complete activity captures per repetition")
    for capture in captures:
        _require_exact_fields(owner + " capture", capture, {
            "path", "sha256", "source_sha256", "runner_sha256"})
        if (not isinstance(capture["path"], str)
                or not capture["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
                or ".." in capture["path"].split("/")):
            raise ValueError(f"{owner} activity raw path is invalid")
        for name in ("sha256", "source_sha256", "runner_sha256"):
            if not isinstance(capture[name], str) or re.fullmatch("[0-9a-f]{64}", capture[name]) is None:
                raise ValueError(f"{owner} activity content-address pin is invalid")
    limitations = reference["scope_limitations"]
    if (not isinstance(limitations, list) or not limitations
            or any(not isinstance(item, str) or not item for item in limitations)):
        raise ValueError(f"{owner} must disclose its activity scope limitations")
    if version in (7, 8):
        if (len({capture["path"] for capture in captures}) != len(captures)
                or len({capture["sha256"] for capture in captures}) != len(captures)
                or len({capture["source_sha256"] for capture in captures}) != 1):
            raise ValueError(f"{owner} repetitions need distinct captures of one unchanged probe")
    active_ctas = reference["grid_ctas"]
    measurements = reference["measurements"]
    if version == 8:
        active_ctas = reference["active_ctas"]
        if (type(active_ctas) is not int or not 0 < active_ctas < reference["grid_ctas"]):
            raise ValueError(f"{owner} active CTAs must be a measured partial launch")
        measurements = []
        names = {"capture_index", "instruction_n", "accumulator_chains", "operand_seed",
                 "rounds", "math_cycles", "clock_mhz"}
        for row in reference["measurements"]:
            _require_exact_fields(owner + " partial measurement", row, names | {"active_ctas"})
            if type(row["active_ctas"]) is not int or row["active_ctas"] != active_ctas:
                raise ValueError(f"{owner} active CTA control disagrees with its measurements")
            measurements.append({name: row[name] for name in names})
    rate, points = _gemm_accumulator_activity_statistics(
        measurements, owner, measurement_repetitions=repetitions)
    published = reference["activity_points"]
    if not isinstance(published, list):
        raise TypeError(f"{owner} activity points must be a list")
    for point in published:
        _require_exact_fields(owner + " activity point", point, {
            "accumulator_values_per_thread", "sm_freq_hz"})
        if type(point["accumulator_values_per_thread"]) is not int:
            raise ValueError(f"{owner} accumulator footprint must be an integer")
    if reference["selected_flops_per_sm_cycle"] != rate or published != points:
        raise ValueError(f"{owner} selected activity service is not the protocol median")
    return {"sm_freq_hz": None, "flops_per_sm_cycle": rate,
        "minimum_active_units": active_ctas,
        "maximum_active_units": active_ctas,
        "physical_grid_ctas": reference["grid_ctas"],
        "threads_per_cta": reference["threads_per_cta"],
        "math_threads_per_cta": reference["measurement_threads_per_cta"],
        "activity_clock_points": {point["accumulator_values_per_thread"]: point["sm_freq_hz"]
            for point in points}}


def _normalize_gemm_bf16_mma_saturation(value, repository_root):
    """Normalize the independent BF16 MMA wave/accumulator surface."""

    owner = "scheduling.gemm_bf16_mma_saturation"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "register_only_bf16_mma_accumulator_saturation_v2",
        "software_revision": "maca_3.7.2",
        "dtype": "bf16",
        "rate_unit": "flops_per_ap_cycle",
    }
    integer_fields = {
        "ap_count": 104,
        "wave_size": 64,
        "accumulator_vector_width": 4,
    }
    fields = (
        set(text_fields) | set(integer_fields)
        | {
            "architecture_peak_flops_per_ap_cycle",
            "result_path", "result_sha256", "reference_path",
            "reference_sha256",
            "production_or_evaluation_duration_consumed",
        })
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    normalized_integers = {}
    for name, expected in integer_fields.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
        normalized_integers[name] = item
    peak = value["architecture_peak_flops_per_ap_cycle"]
    if isinstance(peak, bool) or not isinstance(peak, (int, float)):
        raise TypeError(
            f"{owner}.architecture_peak_flops_per_ap_cycle must be numeric")
    peak = float(peak)
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError(
            f"{owner}.architecture_peak_flops_per_ap_cycle must be finite "
            "and positive")
    result_path = value["result_path"]
    if (not isinstance(result_path, str)
            or not result_path.startswith(
                "profiler_output/microbench/metax_c500/")
            or ".." in result_path.split("/")):
        raise ValueError(f"{owner}.result_path is outside its evidence owner")
    result_sha256 = value["result_sha256"]
    if (not isinstance(result_sha256, str) or len(result_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in result_sha256)):
        raise ValueError(f"{owner}.result_sha256 must be lowercase SHA256")
    _require_no_duration_consumption(value, owner)

    reference_path = value["reference_path"]
    if reference_path != str(_METAX_GEMM_BF16_MMA_ACCUMULATOR_REFERENCE):
        raise ValueError(f"{owner}.reference_path is not canonical")
    reference_sha256 = value["reference_sha256"]
    if reference_sha256 != (
            _METAX_GEMM_BF16_MMA_ACCUMULATOR_REFERENCE_SHA256):
        raise ValueError(f"{owner}.reference_sha256 is not canonical")
    path = repository_root / _METAX_GEMM_BF16_MMA_ACCUMULATOR_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference_sha256:
        raise ValueError("MetaX GEMM BF16 MMA reference SHA256 mismatch")
    reference = json.loads(payload)
    reference_fields = {
        "schema_version", "hardware_id", "software_revision", "compiler",
        "mechanism", "rate_unit", "architecture_peak_flops_per_ap_cycle",
        "ap_count", "wave_size", "mma_shape", "accumulator_dtype",
        "accumulator_vector_width", "iterations", "selected_repeat",
        "measurements", "stability", "provenance",
        "production_or_evaluation_duration_consumed",
        "parameter_fit_from_production_duration", "forward_eligible",
    }
    _require_exact_fields("MetaX GEMM BF16 MMA reference", reference,
                          reference_fields)
    if (isinstance(reference["schema_version"], bool)
            or not isinstance(reference["schema_version"], int)
            or reference["schema_version"] != 2
            or reference["hardware_id"] != "metax_c500"
            or reference["software_revision"] != text_fields[
                "software_revision"]
            or reference["compiler"] != "mxcc 1.0.0 (d9102a1572)"
            or reference["mechanism"] != (
                "register_only_bf16_mma_wave_accumulator_saturation")
            or reference["rate_unit"] != text_fields["rate_unit"]
            or reference["ap_count"] != integer_fields["ap_count"]
            or reference["wave_size"] != integer_fields["wave_size"]
            or reference["accumulator_vector_width"] != integer_fields[
                "accumulator_vector_width"]
            or float(reference["architecture_peak_flops_per_ap_cycle"])
            != peak
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["parameter_fit_from_production_duration"] is not False
            or reference["forward_eligible"] is not True):
        raise ValueError("MetaX GEMM BF16 MMA reference contract drifted")
    if (not isinstance(reference["mma_shape"], list)
            or any(isinstance(dimension, bool)
                   or not isinstance(dimension, int)
                   for dimension in reference["mma_shape"])
            or reference["mma_shape"] != [16, 16, 16]
            or reference["accumulator_dtype"] != "fp32"
            or isinstance(reference["iterations"], bool)
            or not isinstance(reference["iterations"], int)
            or reference["iterations"] != 16384
            or isinstance(reference["selected_repeat"], bool)
            or not isinstance(reference["selected_repeat"], int)
            or reference["selected_repeat"] != 2):
        raise ValueError(
            "MetaX GEMM BF16 MMA instruction or repetition contract drifted")

    stability = reference["stability"]
    if not isinstance(stability, dict):
        raise TypeError("MetaX GEMM BF16 MMA stability must be a mapping")
    _require_exact_fields(
        "MetaX GEMM BF16 MMA stability", stability,
        {"repeat_count", "maximum_relative_rate_delta", "comparison",
         "repeat_result_sha256"})
    maximum_delta = stability["maximum_relative_rate_delta"]
    if (isinstance(stability["repeat_count"], bool)
            or not isinstance(stability["repeat_count"], int)
            or stability["repeat_count"] != 2
            or isinstance(maximum_delta, bool)
            or not isinstance(maximum_delta, (int, float))
            or not math.isfinite(float(maximum_delta))
            or not 0.0 <= float(maximum_delta) <= 0.002
            or stability["comparison"] != (
                "absolute difference divided by the two-repeat mean")):
        raise ValueError("MetaX GEMM BF16 MMA stability contract drifted")
    repeat_sha256 = stability["repeat_result_sha256"]
    if (not isinstance(repeat_sha256, list)
            or len(repeat_sha256) != stability["repeat_count"]
            or any(not isinstance(digest, str) or len(digest) != 64
                   or any(character not in "0123456789abcdef"
                          for character in digest)
                   for digest in repeat_sha256)
            or repeat_sha256 != [
                "2e2f17196be576e637aa01bcaef5cf02decb4ce738bccb1908b5f9a6fb24fe5e",
                "1046e5db6a217ed883b7b99ddc0f8cd5a0af680dcdfb2fd4020524aea8f51db9",
            ]
            or repeat_sha256[reference["selected_repeat"] - 1]
            != result_sha256):
        raise ValueError("MetaX GEMM BF16 MMA repeat provenance drifted")

    provenance = reference["provenance"]
    if not isinstance(provenance, dict):
        raise TypeError("MetaX GEMM BF16 MMA provenance must be a mapping")
    _require_exact_fields(
        "MetaX GEMM BF16 MMA provenance", provenance,
        {"capture_baseline_commit", "capture_command", "source_sha256",
         "binary_sha256", "physical_device_id", "owner", "remote_snapshot",
         "recorded_pids", "result_paths"})
    if (provenance["capture_baseline_commit"] != "b0f6db466"
            or provenance["capture_command"] != (
                "MACA_VISIBLE_DEVICES=0 mma_accumulator_sweep")
            or provenance["source_sha256"] != (
                "9f40d19ba1bf6efa1fdd48dbf1d31175320c7e356cf99fc96455025267001e52")
            or provenance["binary_sha256"] != (
                "0066751e1d3ce0ec4da09afbf1a8f7d5d3a1064ba4d0da3f3d711338d1d0710c")
            or isinstance(provenance["physical_device_id"], bool)
            or not isinstance(provenance["physical_device_id"], int)
            or provenance["physical_device_id"] < 0
            or provenance["owner"] != "GEMM worker"
            or (not isinstance(provenance["remote_snapshot"], str)
                or not provenance["remote_snapshot"].strip()
                or ".." in provenance["remote_snapshot"].split("/"))
            or provenance["result_paths"] != [
                "profiler_output/microbench/metax_c500/"
                "mma_accumulator_chain32_b0f6db466_20260915/raw_r1.csv",
                "profiler_output/microbench/metax_c500/"
                "mma_accumulator_chain32_b0f6db466_20260915/raw_r2.csv"]):
        raise ValueError("MetaX GEMM BF16 MMA execution provenance drifted")
    pids = provenance["recorded_pids"]
    if (not isinstance(pids, list) or len(pids) != 2
            or any(isinstance(pid, bool) or not isinstance(pid, int)
                   or pid <= 0 for pid in pids)):
        raise ValueError("MetaX GEMM BF16 MMA recorded PIDs are invalid")
    expected_wave_points = (1, 2, 4, 8, 16, 32)
    expected_chain_points = (4, 8, 16, 32)
    rates_by_occupancy = {}
    for index, measurement in enumerate(reference["measurements"]):
        context = f"MetaX GEMM BF16 MMA reference.measurements[{index}]"
        if not isinstance(measurement, dict):
            raise TypeError(f"{context} must be a mapping")
        _require_exact_fields(
            context, measurement,
            {"active_waves_per_ap", "accumulator_chains",
             "flops_per_ap_cycle"})
        waves = measurement["active_waves_per_ap"]
        chains = measurement["accumulator_chains"]
        rate = measurement["flops_per_ap_cycle"]
        if (isinstance(waves, bool) or not isinstance(waves, int)
                or waves not in expected_wave_points):
            raise ValueError(f"{context}.active_waves_per_ap is invalid")
        if (isinstance(chains, bool) or not isinstance(chains, int)
                or chains not in expected_chain_points):
            raise ValueError(f"{context}.accumulator_chains is invalid")
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate)) or not 0.0 < rate <= peak):
            raise ValueError(f"{context}.flops_per_ap_cycle is invalid")
        key = (waves, chains)
        if key in rates_by_occupancy:
            raise ValueError(f"{context} duplicates occupancy point {key}")
        rates_by_occupancy[key] = float(rate)
    expected_keys = {
        (waves, chains)
        for waves in expected_wave_points
        for chains in expected_chain_points
    }
    if set(rates_by_occupancy) != expected_keys:
        raise ValueError("MetaX GEMM BF16 MMA reference grid is incomplete")
    for chains in expected_chain_points:
        rates = tuple(
            rates_by_occupancy[(waves, chains)]
            for waves in expected_wave_points)
        if any(left >= right for left, right in zip(rates, rates[1:])):
            raise ValueError(
                "MetaX GEMM BF16 MMA wave curve must be strictly increasing")
    for waves in expected_wave_points:
        rates = tuple(
            rates_by_occupancy[(waves, chains)]
            for chains in expected_chain_points)
        if any(left >= right for left, right in zip(rates, rates[1:])):
            raise ValueError(
                "MetaX GEMM BF16 MMA accumulator curve must be strictly "
                "increasing")
    return {
        **text_fields,
        **normalized_integers,
        "architecture_peak_flops_per_ap_cycle": peak,
        "active_wave_points": expected_wave_points,
        "accumulator_chain_points": expected_chain_points,
        "flops_per_ap_cycle_by_occupancy": rates_by_occupancy,
        "fallback_efficiency": rates_by_occupancy[(32, 4)] / peak,
        "result_path": result_path,
        "result_sha256": result_sha256,
        "reference_path": reference_path,
        "reference_sha256": reference_sha256,
        "production_or_evaluation_duration_consumed": False,
    }


def _normalize_fa3_gqa_compiled_dependencies(value, repository_root, hardware_id):
    """Discrete compiled wait order; never a time or kernel-name price."""
    owner = "scheduling.fa3_gqa_compiled_dependencies"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its retained SM90 programs")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "schema_version", "hardware_id", "model", "source_program", "selection",
        "entries", "evidence", "scope_limitations", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(ref, owner)
    if (type(ref["schema_version"]) is not int or ref["schema_version"] != 2
            or ref["hardware_id"] != hardware_id or ref["model"] != "fa3_gqa_compiled_wait_order_v2"
            or ref["source_program"] != "vllm_0_30_fa3_sm90_gqa_bf16"
            or ref["selection"] != "all_retained_recurrent_regions_wait_before_first_EX2"
            or not isinstance(ref["entries"], list) or not ref["entries"]):
        raise ValueError(f"{owner} compiled domain differs")
    result, phase_work = {}, {}
    for row in ref["entries"]:
        _require_exact_fields(owner + " entry", row, {
            "physical_key", "wait_before_exp", "source_variants", "online_prefix_work"})
        key = row["physical_key"]
        if (not isinstance(key, list) or len(key) != 7
                or any(type(x) is not int or x <= 0 for x in key[:3])
                or any(type(x) is not bool for x in key[3:])
                or tuple(key) in result or type(row["wait_before_exp"]) is not bool
                or not isinstance(row["source_variants"], list) or not row["source_variants"]):
            raise ValueError(f"{owner} invalid or duplicated physical key")
        expected = {"FMNMX": key[1] // 2 + 4, "EX2": key[1] // 2 + 2,
                    "FFMA": key[1] // 2 + 2, "FADD": key[1] // 2}
        prefix = {op: [] for op in expected}
        symbols = set()
        for variant in row["source_variants"]:
            _require_exact_fields(owner + " variant", variant, {
                "source_symbol", "export_path", "export_sha256", "regions"})
            symbol = variant["source_symbol"]
            if (not isinstance(symbol, str) or "bfloat16_t" not in symbol
                    or "FlashAttnFwdSm90" not in symbol or symbol in symbols
                    or not isinstance(variant["export_path"], str)
                    or not variant["export_path"].startswith("profiler_output/kernel_dataset/nvidia_h200/")
                    or ".." in variant["export_path"].split("/")
                    or not isinstance(variant["export_sha256"], str)
                    or not re.fullmatch("[0-9a-f]{64}", variant["export_sha256"])
                    or not isinstance(variant["regions"], list) or not variant["regions"]):
                raise ValueError(f"{owner} invalid static variant provenance")
            symbols.add(symbol)
            all_early, pcs = True, set()
            for region in variant["regions"]:
                _require_exact_fields(owner + " region", region, {
                    "wait1_pc", "wait0_pc", "first_exp_pc", "wait_before_exp",
                    "online_before_wait0", "online_totals", "next_mma_pc"})
                addresses = [region[k] for k in ("wait1_pc", "wait0_pc", "first_exp_pc")]
                if (any(not isinstance(x, str) or not re.fullmatch("0x[0-9a-f]+", x) for x in addresses)
                        or type(region["wait_before_exp"]) is not bool):
                    raise ValueError(f"{owner} invalid instruction order facts")
                wait1, wait0, exp = [int(x, 16) for x in addresses]
                if (wait1 in pcs or not wait1 < min(wait0, exp) or wait0 == exp
                        or region["wait_before_exp"] != (wait0 < exp)):
                    raise ValueError(f"{owner} instruction order contradicts its evidence")
                pcs.add(wait1)
                all_early = all_early and region["wait_before_exp"]
                _require_exact_fields(owner + " prefix counts", region["online_before_wait0"], set(expected))
                _require_exact_fields(owner + " total counts", region["online_totals"], set(expected))
                if (not isinstance(region["next_mma_pc"], str)
                        or not re.fullmatch("0x[0-9a-f]+", region["next_mma_pc"])
                        or int(region["next_mma_pc"], 16) <= max(wait0, exp)):
                    raise ValueError(f"{owner} invalid online region endpoint")
                for op, total in expected.items():
                    before = region["online_before_wait0"][op]
                    if (type(before) is not int or not 0 <= before <= total
                            or type(region["online_totals"][op]) is not int
                            or region["online_totals"][op] != total):
                        raise ValueError(f"{owner} invalid online work geometry")
                    prefix[op].append(before)
                if region["wait_before_exp"] != (region["online_before_wait0"]["EX2"] == 0):
                    raise ValueError(f"{owner} online prefix contradicts its first EX2")
            if row["wait_before_exp"] != all_early:
                raise ValueError(f"{owner} cannot discard a conflicting recurrent region")
        result[tuple(key)] = row["wait_before_exp"]
        _require_exact_fields(owner + " prefix selection", row["online_prefix_work"], set(expected))
        for op, total in expected.items():
            fact = row["online_prefix_work"][op]
            _require_exact_fields(owner + " prefix selection count", fact, {"before", "total"})
            if (type(fact["before"]) is not int or fact["before"] != min(prefix[op])
                    or type(fact["total"]) is not int or fact["total"] != total):
                raise ValueError(f"{owner} online prefix must retain every recurrent region")
        phase_work[tuple(key)] = {op: (min(prefix[op]), total) for op, total in expected.items()}
    if not isinstance(ref["evidence"], list) or len(ref["evidence"]) != 3:
        raise ValueError(f"{owner} requires retained manifest and order/work extractions")
    paths = set()
    for evidence in ref["evidence"]:
        _require_exact_fields(owner + " evidence", evidence, {"path", "sha256", "size"})
        path = evidence["path"]
        if (not isinstance(path, str) or not path.startswith("profiler_output/kernel_dataset/nvidia_h200/")
                or ".." in path.split("/") or path in paths
                or not isinstance(evidence["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", evidence["sha256"])
                or type(evidence["size"]) is not int or evidence["size"] <= 0):
            raise ValueError(f"{owner} invalid evidence binding")
        paths.add(path)
    if (not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(x, str) or not x for x in ref["scope_limitations"])):
        raise ValueError(f"{owner} must declare transfer and aggregation limits")
    return result, phase_work


def _normalize_warp_integer_dependencies(value, repository_root, hardware_id):
    """Independent signed32 quotient and five-level inclusive-scan latency."""
    owner = "scheduling.warp_integer_dependencies"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 domain")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "schema_version", "hardware_id", "mechanism", "selection", "validation_fraction",
        "measurements", "evidence", "scope_limitations", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(ref, owner)
    if (type(ref["schema_version"]) is not int or ref["schema_version"] != 1
            or ref["hardware_id"] != hardware_id or ref["mechanism"] != "signed32_divide_and_inclusive_warp_scan"
            or ref["selection"] != "grid132_onewarp64_256_d7_minus_matched_feedback"
            or ref["validation_fraction"] != .05):
        raise ValueError(f"{owner} mechanism boundary drifted")
    if not isinstance(ref["measurements"], list) or len(ref["measurements"]) != 28:
        raise ValueError(f"{owner} requires all independent cells")
    periods = {}
    for row in ref["measurements"]:
        _require_exact_fields(owner + " measurement", row, {
            "mode", "warps", "divisor", "grid", "cycles64", "cycles128", "cycles256"})
        key = tuple(row[k] for k in ("mode", "warps", "divisor", "grid"))
        times = tuple(row[f"cycles{n}"] for n in (64, 128, 256))
        if any(type(x) is not int for x in key) or key in periods:
            raise ValueError(f"{owner} duplicate or invalid cell")
        if (any(isinstance(x, bool) or not isinstance(x, (int, float))
                or not math.isfinite(x) or x <= 0 for x in times)
                or not times[0] < times[1] < times[2]):
            raise ValueError(f"{owner} requires positive finite ordered cycle measurements")
        period = (times[2] - times[0]) / 192
        if abs(times[1] - times[0] - 64 * period) / times[1] > .05:
            raise ValueError(f"{owner} independent repetition midpoint failed")
        periods[key] = period
    domain = {(m,w,d,g) for m in (0,1,2) for w in (1,4)
              for d in ((1,) if m == 2 else (1,7,257)) for g in (1,132)}
    if set(periods) != domain:
        raise ValueError(f"{owner} incomplete independent domain")
    divide = periods[1,1,7,132] - periods[0,1,7,132]
    scan = periods[2,1,1,132] - periods[0,1,1,132]
    if divide <= 0 or scan <= 0:
        raise ValueError(f"{owner} nonpositive incremental dependency service")
    for (_,w,d,g), period in periods.items():
        observed_divide = periods[1,w,d,g] - periods[0,w,d,g]
        observed_scan = periods[2,w,1,g] - periods[0,w,1,g]
        if abs(observed_divide / divide - 1) > .05 or abs(observed_scan / scan - 1) > .05:
            raise ValueError(f"{owner} independent divisor/warp/grid transfer failed")
    _require_exact_fields(owner + " evidence", ref["evidence"], {
        "raw", "source", "runner", "plan", "analysis", "analyzer"})
    for item in ref["evidence"].values():
        _require_exact_fields(owner + " evidence item", item, {"path", "sha256", "size"})
        if (not isinstance(item["path"], str)
                or not item["path"].startswith("profiler_output/microbench/nvidia_h200/")
                or ".." in item["path"].split("/")
                or not isinstance(item["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", item["sha256"])
                or type(item["size"]) is not int or item["size"] <= 0):
            raise ValueError(f"{owner} invalid evidence binding")
    if (not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(s, str) or not s for s in ref["scope_limitations"])):
        raise ValueError(f"{owner} must declare composition limits")
    return {"signed_divide_cycles": divide, "inclusive_scan_cycles": scan}


def _normalize_ss_ex2_busy_sharing(value, repository_root, hardware_id):
    """Asymmetric SS/EX2 capacity loss from independent queued primitives."""
    owner = "scheduling.ss_ex2_busy_sharing"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 domain")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "schema_version", "hardware_id", "mechanism", "instruction", "matrix_per_repeat",
        "tensor_busy_cycles_per_instruction", "ex2_lane_ops_per_cycle", "selection",
        "validation_fraction", "measurements", "evidence", "scope_limitations",
        "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(ref, owner)
    expected = {"schema_version": 1, "hardware_id": hardware_id,
        "mechanism": "asymmetric_ss_ex2_busy_capacity",
        "instruction": "bf16_ss_m64n128k16", "matrix_per_repeat": 8,
        "tensor_busy_cycles_per_instruction": 64, "ex2_lane_ops_per_cycle": 16,
        "selection": "grid132_queue4_width128_integrated_ex2_loss_over_tensor_busy",
        "validation_fraction": .05}
    if (any(ref[key] != item for key, item in expected.items())
            or any(type(ref[key]) is not int for key in ("schema_version", "matrix_per_repeat",
                "tensor_busy_cycles_per_instruction", "ex2_lane_ops_per_cycle"))):
        raise ValueError(f"{owner} mechanism boundary drifted")
    rows = ref["measurements"]
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError(f"{owner} requires all twelve independent cells")
    cells = {}
    for row in rows:
        _require_exact_fields(owner + " measurement", row, {
            "queue", "width", "grid", "tensor_alone", "ex2_alone",
            "tensor_mixed", "ex2_mixed", "join_mixed"})
        key = tuple(row[name] for name in ("queue", "width", "grid"))
        if any(type(x) is not int for x in key) or key in cells:
            raise ValueError(f"{owner} invalid or duplicate cell")
        for name in ("tensor_alone", "ex2_alone", "tensor_mixed", "ex2_mixed", "join_mixed"):
            item = row[name]
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(item) or item <= 0):
                raise ValueError(f"{owner} services must be finite and positive")
        cells[key] = row
    if set(cells) != {(q, w, g) for q in (1, 4) for w in (32, 64, 128) for g in (1, 132)}:
        raise ValueError(f"{owner} incomplete queued validation domain")
    busy = ref["matrix_per_repeat"] * ref["tensor_busy_cycles_per_instruction"]
    calibration = cells[4, 128, 132]
    loss = (calibration["ex2_mixed"] - calibration["ex2_alone"]) / busy
    if (not 0 < loss < 1 or calibration["ex2_alone"] <= calibration["tensor_mixed"]
            or calibration["ex2_mixed"] <= calibration["tensor_mixed"]):
        raise ValueError(f"{owner} integrated loss is not identifiable")
    for (queue, width, _), row in cells.items():
        tensor = row["tensor_alone"]
        work = 128 * width / ref["ex2_lane_ops_per_cycle"]
        overhead = row["ex2_alone"] - work
        rate = 1 - loss * min(1., busy / tensor)
        shared = min(work / rate, tensor)
        ex2 = overhead + shared + max(0., work - rate * shared)
        if overhead < 0 or (queue == 4 and abs(tensor / busy - 1) > .05):
            raise ValueError(f"{owner} standalone capacity control failed")
        for predicted, name in ((tensor, "tensor_mixed"), (ex2, "ex2_mixed"),
                                (max(tensor, ex2), "join_mixed")):
            if abs(predicted / row[name] - 1) > .05:
                raise ValueError(f"{owner} independent component/join validation failed")
    _require_exact_fields(owner + " evidence", ref["evidence"], {
        "raw", "source", "runner", "plan", "analysis", "busy_analysis", "derivation"})
    for item in ref["evidence"].values():
        _require_exact_fields(owner + " evidence item", item, {"path", "sha256", "size"})
        if (not isinstance(item["path"], str)
                or not item["path"].startswith("profiler_output/microbench/nvidia_h200/")
                or ".." in item["path"].split("/")
                or not isinstance(item["sha256"], str)
                or not re.fullmatch("[0-9a-f]{64}", item["sha256"])
                or type(item["size"]) is not int or item["size"] <= 0):
            raise ValueError(f"{owner} invalid independent evidence binding")
    if (not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(x, str) or not x for x in ref["scope_limitations"])):
        raise ValueError(f"{owner} must declare transfer approximations")
    return {"loss_fraction": loss}


def _normalize_packed_bf16_conversion(value, repository_root, hardware_id):
    """Independent packed conversion capacity, excluding integer feedback work."""
    owner = "scheduling.packed_bf16_conversion"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 domain")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "schema_version", "hardware_id", "mechanism", "instruction", "packed_results",
        "timer_domain", "selection", "validation_fraction", "measurements", "evidence",
        "scope_limitations", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(ref, owner)
    expected = {"schema_version": 1, "hardware_id": hardware_id,
        "mechanism": "packed_bf16_conversion_capacity_v1",
        "instruction": "F2FP.BF16.F32.PACK_AB", "packed_results": 2,
        "timer_domain": "same_CTA_clock64_envelope",
        "selection": "grid132_threads256_width16_64_repetition128_512_one_two_LOP3",
        "validation_fraction": .05}
    if (any(ref[key] != item for key, item in expected.items())
            or any(type(ref[key]) is not int for key in ("schema_version", "packed_results"))):
        raise ValueError(f"{owner} mechanism boundary drifted")

    def positive(item):
        return (not isinstance(item, bool) and isinstance(item, (int, float))
                and math.isfinite(item) and item > 0)

    if not isinstance(ref["measurements"], list) or len(ref["measurements"]) != 60:
        raise ValueError(f"{owner} requires all independent capacity cells")
    slopes = {}
    for row in ref["measurements"]:
        _require_exact_fields(owner + " measurement", row, {
            "pairs", "feedback", "threads", "grid", "cycles_128", "cycles_256", "cycles_512"})
        key = tuple(row[name] for name in ("pairs", "feedback", "threads", "grid"))
        times = tuple(row[f"cycles_{n}"] for n in (128, 256, 512))
        if (any(type(x) is not int for x in key) or key in slopes
                or not all(positive(x) for x in times) or not times[0] < times[1] < times[2]):
            raise ValueError(f"{owner} invalid or duplicate capacity cell")
        slope = (times[2] - times[0]) / 384
        if abs(times[1] - (times[0] + 128 * slope)) / times[1] > .05:
            raise ValueError(f"{owner} independent repetition midpoint failed")
        slopes[key] = slope
    if set(slopes) != {(p, f, t, g) for p in (16, 32, 44, 48, 64)
                       for f in (1, 2) for t in (128, 256) for g in (1, 32, 132)}:
        raise ValueError(f"{owner} capacity domain drifted")
    periods = {f: (slopes[64, f, 256, 132] - slopes[16, f, 256, 132]) / (48 * 8)
               for f in (1, 2)}
    conversion = 2 * periods[1] - periods[2]
    feedback = periods[2] - periods[1]
    if not positive(conversion) or not positive(feedback):
        raise ValueError(f"{owner} nonphysical separated capacity")
    for pairs in (32, 44, 48):
        for f in (1, 2):
            for threads in (128, 256):
                for grid in (1, 32, 132):
                    observed = slopes[pairs, f, threads, grid] - slopes[16, f, threads, grid]
                    predicted = (pairs - 16) * (threads / 32) * (conversion + f * feedback)
                    if not positive(observed) or abs(predicted - observed) / observed > .05:
                        raise ValueError(f"{owner} independent width capacity validation failed")
    _require_exact_fields(owner + " evidence", ref["evidence"], {
        "raw", "source", "runner", "analysis", "derivation", "plan"})
    paths = set()
    for evidence in ref["evidence"].values():
        _require_exact_fields(owner + " evidence item", evidence, {"path", "sha256", "size"})
        path = evidence["path"]
        if (not isinstance(path, str) or not path.startswith("profiler_output/microbench/nvidia_h200/")
                or ".." in path.split("/") or path in paths
                or not isinstance(evidence["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", evidence["sha256"])
                or type(evidence["size"]) is not int or evidence["size"] <= 0):
            raise ValueError(f"{owner} invalid independent evidence binding")
        paths.add(path)
    if (not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(x, str) or not x for x in ref["scope_limitations"])):
        raise ValueError(f"{owner} must declare transfer limits")
    return {"cycles_per_warp_instruction": conversion,
            "feedback_cycles_per_warp_instruction": feedback,
            "pairs_per_thread": (16, 32, 44, 48, 64), "active_warpgroups": (1, 2)}


def _normalize_moe_align_shared_atomic_e128(value, repository_root, hardware_id):
    """Independent one-CTA E128 shared-atomic capacity for ordinary alignment."""
    owner = "scheduling.moe_align_shared_atomic_e128"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 domain")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_no_duration_consumption(ref, owner)
    if not isinstance(ref["cases"], list):
        raise ValueError(f"{owner} requires independent cases")
    matches = [case for case in ref["cases"]
               if case["case_name"] == "shared_cta_e128_r262144"]
    if len(matches) != 1:
        raise ValueError(f"{owner} requires exactly one E128 shared-atomic case")
    case = matches[0]
    if (case["block"] != 1024 or case["grid"] != 1 or case["experts"] != 128
            or case["items"] != 262144):
        raise ValueError(f"{owner} physical case drifted")
    rate = case["median_atomic_ops_per_s"]
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or rate <= 0):
        raise ValueError(f"{owner} requires positive atomic capacity")
    return float(rate)


def _normalize_moe_align_fill_bytes_per_s(value, repository_root, hardware_id):
    """Independent one-CTA contiguous INT32 fill capacity."""
    owner = "scheduling.moe_align_fill_bytes_per_s"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 domain")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "hardware_id", "mechanism", "production_or_evaluation_duration_consumed",
        "mechanism_qualified", "threads_per_cta", "grid_ctas", "bytes_per_second",
        "selection", "points", "source_bindings", "scope_limitations"})
    _require_no_duration_consumption(ref, owner)
    if (ref["hardware_id"] != hardware_id
            or ref["mechanism"] != "independent_single_cta_int32_fill_capacity"
            or ref["mechanism_qualified"] is not True
            or type(ref["threads_per_cta"]) is not int or ref["threads_per_cta"] != 1024
            or type(ref["grid_ctas"]) is not int or ref["grid_ctas"] != 1
            or not isinstance(ref["selection"], str) or not ref["selection"]):
        raise ValueError(f"{owner} physical scope drifted")
    points = ref["points"]
    if not isinstance(points, list) or len(points) != 3:
        raise ValueError(f"{owner} requires all three independent capacities")
    rates = []
    domains = set()
    for point in points:
        _require_exact_fields(owner + " point", point, {
            "items", "repeat_median_cycles", "bytes_per_second"})
        items = point["items"]
        cycles = point["repeat_median_cycles"]
        rate = point["bytes_per_second"]
        if (type(items) is not int or items <= 0 or items in domains
                or not isinstance(cycles, list) or len(cycles) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0 for value in cycles)
                or isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate <= 0):
            raise ValueError(f"{owner} requires unique items and positive repeat cycles")
        domains.add(items)
        spread = max(cycles) / min(cycles) - 1
        expected = 4 * items * 1.98e9 / max(cycles)
        if spread > .03 or not math.isclose(rate, expected, rel_tol=1e-12):
            raise ValueError(f"{owner} repeat spread or byte/cycle unit mismatch")
        rates.append(float(rate))
    if domains != {262144, 524288, 1048576}:
        raise ValueError(f"{owner} independent item domain drifted")
    if (isinstance(ref["bytes_per_second"], bool)
            or not isinstance(ref["bytes_per_second"], (int, float))
            or not math.isfinite(ref["bytes_per_second"])
            or not math.isclose(ref["bytes_per_second"], min(rates), rel_tol=1e-12)):
        raise ValueError(f"{owner} conservative minimum selection mismatch")
    bindings = ref["source_bindings"]
    _require_exact_fields(owner + " source_bindings", bindings, {"raw", "plan", "source"})
    expected_paths = {
        "raw": "profiler_output/microbench/nvidia_h200/single_cta_fill_capacity_20261109/attempt_1/raw.json",
        "plan": "microbench/nvidia_h200/references/single_cta_fill_capacity_plan_sm90.json",
        "source": "microbench/nvidia_h200/memory/lsu/single_cta_int32_fill.cu",
    }
    for role, path in expected_paths.items():
        binding = bindings[role]
        _require_exact_fields(owner + f" {role} binding", binding, {"path", "sha256"})
        if (binding["path"] != path or not isinstance(binding["sha256"], str)
                or not re.fullmatch("[0-9a-f]{64}", binding["sha256"])):
            raise ValueError(f"{owner} independent source binding drifted")
    if (not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(item, str) or not item for item in ref["scope_limitations"])):
        raise ValueError(f"{owner} requires explicit scope limitations")
    return min(rates)


def _normalize_moe_activation_mixed_stream(value, repository_root, hardware_id):
    """Independent exact-geometry two-read/one-write logical stream capacity."""
    owner = "scheduling.moe_activation_mixed_stream"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 domain")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "hardware_id", "mechanism", "mechanism_qualified",
        "production_or_evaluation_duration_consumed", "selection", "points",
        "source_bindings", "scope_limitations"})
    _require_no_duration_consumption(ref, owner)
    if (ref["hardware_id"] != hardware_id
            or ref["mechanism"] != "independent_exact_geometry_two_read_one_write_stream_capacity"
            or ref["mechanism_qualified"] is not True
            or not isinstance(ref["selection"], str) or not ref["selection"]):
        raise ValueError(f"{owner} physical scope drifted")
    if not isinstance(ref["points"], list) or len(ref["points"]) != 18:
        raise ValueError(f"{owner} requires all eighteen exact stream cells")
    rates = {}
    for point in ref["points"]:
        _require_exact_fields(owner + " point", point, {
            "grid", "block", "process_median_ms", "logical_bytes_per_second"})
        grid, block = point["grid"], point["block"]
        medians = point["process_median_ms"]
        rate = point["logical_bytes_per_second"]
        if (type(grid) is not int or grid <= 0 or type(block) is not int or block <= 0
                or (grid, block) in rates or not isinstance(medians, list) or len(medians) != 2
                or any(isinstance(ms, bool) or not isinstance(ms, (int, float))
                       or not math.isfinite(ms) or ms <= 0 for ms in medians)
                or isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate <= 0):
            raise ValueError(f"{owner} requires unique cells and positive process medians")
        spread = max(medians) / min(medians) - 1
        expected = 48 * grid * block * 1e3 / max(medians)
        if spread > .03 or not math.isclose(rate, expected, rel_tol=1e-12):
            raise ValueError(f"{owner} process spread or logical byte unit mismatch")
        rates[grid, block] = float(rate)
    expected_cells = {(grid, block) for grid in (131072, 262144, 524288)
                      for block in (96, 128)} | {
        (65536,1024),(262144,256),(32768,1024),(262144,192),
        (131072,360),(327680,128),(131072,256),(16384,1024),
        (131072,192),(65536,360),(327680,64),(163840,128)}
    if set(rates) != expected_cells:
        raise ValueError(f"{owner} exact launch domain drifted")
    bindings = ref["source_bindings"]
    _require_exact_fields(owner + " source_bindings", bindings, {
        "raw", "analysis", "plan", "source",
        "extended_raw", "extended_analysis", "extended_plan", "extended_source"})
    expected_paths = {
        "raw": "profiler_output/microbench/nvidia_h200/vec16_mixed_stream_capacity_20261109/attempt_1/raw.json",
        "analysis": "profiler_output/microbench/nvidia_h200/vec16_mixed_stream_capacity_20261109/analysis_corrected.json",
        "plan": "microbench/nvidia_h200/references/vec16_mixed_stream_capacity_plan_sm90.json",
        "source": "microbench/nvidia_h200/memory/dram/vec16_two_read_one_write.cu",
        "extended_raw": "profiler_output/microbench/nvidia_h200/vec16_mixed_stream_capacity_extended_20261109/attempt_1/raw.json",
        "extended_analysis": "profiler_output/microbench/nvidia_h200/vec16_mixed_stream_capacity_extended_20261109/attempt_1/raw.json",
        "extended_plan": "microbench/nvidia_h200/references/vec16_mixed_stream_capacity_extended_plan_sm90.json",
        "extended_source": "microbench/nvidia_h200/memory/dram/vec16_mixed_stream_capacity_extended.cu",
    }
    for role, path in expected_paths.items():
        binding = bindings[role]
        _require_exact_fields(owner + f" {role} binding", binding, {"path", "sha256"})
        if (binding["path"] != path or not isinstance(binding["sha256"], str)
                or not re.fullmatch("[0-9a-f]{64}", binding["sha256"])):
            raise ValueError(f"{owner} independent source binding drifted")
    if (not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(item, str) or not item for item in ref["scope_limitations"])):
        raise ValueError(f"{owner} requires explicit scope limitations")
    return rates


def _normalize_fa3_parameter_reads(value, repository_root, hardware_id):
    """Independent eager argument-read service and static compiled footprint."""
    owner = "scheduling.fa3_parameter_reads"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 domain")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(value, owner)
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "schema_version", "hardware_id", "model", "timer_domain", "source_program",
        "region_bytes", "control_regions", "isolated_region_ns", "peer_region_ns",
        "calibration", "validation", "reuse_validation", "main_parameter_regions",
        "static_work", "combine_static_work", "evidence", "scope_limitations", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(ref, owner)
    expected = {"schema_version": 2, "hardware_id": hardware_id,
        "model": "eager_parameter_region_serial_queue_v1",
        "timer_domain": "nsys_kernel_span_difference_with_globaltimer_body_validation",
        "source_program": "vllm_0_30_fa3_sm90_gqa_bf16", "region_bytes": 64, "control_regions": 2}
    if (any(ref[key] != item for key, item in expected.items())
            or any(type(ref[key]) is not int for key in ("schema_version", "region_bytes", "control_regions"))):
        raise ValueError(f"{owner} evidence boundary drifted")

    def positive(item):
        return (not isinstance(item, bool) and isinstance(item, (int, float))
                and math.isfinite(item) and item > 0)

    if not positive(ref["isolated_region_ns"]) or not positive(ref["peer_region_ns"]):
        raise ValueError(f"{owner} requires positive physical service periods")
    calibration = {}
    if not isinstance(ref["calibration"], list) or len(ref["calibration"]) != 4:
        raise ValueError(f"{owner} requires four independent endpoint cells")
    for row in ref["calibration"]:
        _require_exact_fields(owner + " calibration", row, {"resident_sms", "hold_cycles", "regions", "control_ns", "read_ns"})
        if (any(type(row[k]) is not int for k in ("resident_sms", "hold_cycles", "regions"))
                or row["regions"] != 60 or not positive(row["control_ns"])
                or not positive(row["read_ns"]) or row["read_ns"] <= row["control_ns"]):
            raise ValueError(f"{owner} invalid calibration cell")
        key = (row["resident_sms"], row["hold_cycles"])
        if key in calibration:
            raise ValueError(f"{owner} duplicated calibration cell")
        calibration[key] = (row["read_ns"] - row["control_ns"]) / 58
    if set(calibration) != {(g, h) for g in (1, 132) for h in (0, 4096)}:
        raise ValueError(f"{owner} incomplete endpoint domain")
    isolated = statistics.median(calibration[1, h] for h in (0, 4096))
    peer = (statistics.median(calibration[132, h] for h in (0, 4096)) - isolated) / 131
    if (not math.isclose(ref["isolated_region_ns"], isolated, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(ref["peer_region_ns"], peer, rel_tol=0, abs_tol=1e-9)):
        raise ValueError(f"{owner} rates disagree with independent endpoint derivation")
    validation = set()
    if not isinstance(ref["validation"], list) or len(ref["validation"]) != 16:
        raise ValueError(f"{owner} incomplete independent validation")
    for row in ref["validation"]:
        _require_exact_fields(owner + " validation", row, {"resident_sms", "regions", "hold_cycles", "increment_ns"})
        key = (row["resident_sms"], row["regions"], row["hold_cycles"])
        if (any(type(x) is not int for x in key) or key in validation or not positive(row["increment_ns"])):
            raise ValueError(f"{owner} invalid validation cell")
        validation.add(key)
        predicted = (row["regions"] - 2) * (isolated + (row["resident_sms"] - 1) * peer)
        if abs(predicted - row["increment_ns"]) / row["increment_ns"] > .15:
            raise ValueError(f"{owner} independent primitive validation gate failed")
    if validation != {(g, n, h) for g in (1, 8, 32, 132) for n in (16, 60) for h in (0, 4096)}:
        raise ValueError(f"{owner} validation domain drifted")
    reuse = set()
    if not isinstance(ref["reuse_validation"], list) or len(ref["reuse_validation"]) != 4:
        raise ValueError(f"{owner} requires eager argument-reuse validation")
    for row in ref["reuse_validation"]:
        _require_exact_fields(owner + " reuse", row, {"resident_sms", "regions", "increments_ns_fixed_one_all"})
        key = (row["resident_sms"], row["regions"])
        times = row["increments_ns_fixed_one_all"]
        if (any(type(x) is not int for x in key) or key in reuse
                or not isinstance(times, list) or len(times) != 3 or not all(positive(x) for x in times)
                or any(abs(x - times[2]) / times[2] > .2 for x in times[:2])):
            raise ValueError(f"{owner} eager argument-reuse boundary failed")
        reuse.add(key)
    if reuse != {(g, n) for g in (1, 132) for n in (16, 60)}:
        raise ValueError(f"{owner} incomplete argument-reuse domain")
    regions = ref["main_parameter_regions"]
    if type(regions) is not int or regions <= ref["control_regions"]:
        raise ValueError(f"{owner} invalid source footprint")
    if not isinstance(ref["static_work"], list) or len(ref["static_work"]) != 12:
        raise ValueError(f"{owner} requires all retained GQA source variants")
    tiles, symbols, entries, counts = set(), set(), set(), []
    for row in ref["static_work"]:
        _require_exact_fields(owner + " static source", row, {"tile", "regions", "source_symbol", "manifest_entry"})
        tile, offsets, symbol = row["tile"], row["regions"], row["source_symbol"]
        if (not isinstance(tile, list) or len(tile) != 3 or any(type(x) is not int or x <= 0 for x in tile)
                or not isinstance(offsets, list) or not offsets
                or any(type(x) is not int or x < 0 or x % 64 for x in offsets)
                or offsets != sorted(set(offsets)) or not isinstance(symbol, str)
                or "bfloat16_t" not in symbol or symbol in symbols
                or type(row["manifest_entry"]) is not int or row["manifest_entry"] < 0
                or row["manifest_entry"] in entries):
            raise ValueError(f"{owner} invalid static source facts")
        tiles.add(tuple(tile))
        symbols.add(symbol)
        entries.add(row["manifest_entry"])
        counts.append(len(offsets))
    if min(counts) != regions or max(counts) - regions > 1:
        raise ValueError(f"{owner} common source footprint disagrees with static evidence")
    combine_work = ref["combine_static_work"]
    if not isinstance(combine_work, list) or len(combine_work) != 2:
        raise ValueError(f"{owner} incomplete combine source domain")
    combine_regions, combine_symbols = {}, set()
    for row in combine_work:
        _require_exact_fields(owner + " combine source", row, {
            "physical_key", "regions", "source_symbol", "manifest_entry"})
        key, offsets, symbol = row["physical_key"], row["regions"], row["source_symbol"]
        if (not isinstance(key, list) or len(key) != 4 or any(type(x) is not int for x in key)
                or tuple(key) not in ((8, 128, 32, 256), (16, 64, 16, 256))
                or tuple(key) in combine_regions or not isinstance(offsets, list) or not offsets
                or any(type(x) is not int or x < 0 or x % 64 for x in offsets)
                or offsets != sorted(set(offsets)) or len(offsets) <= ref["control_regions"]
                or not isinstance(symbol, str) or "FlashAttnFwdCombine" not in symbol
                or "bfloat16_t" not in symbol or symbol in combine_symbols
                or type(row["manifest_entry"]) is not int or row["manifest_entry"] < 0):
            raise ValueError(f"{owner} invalid combine source footprint")
        combine_regions[tuple(key)] = len(offsets)
        combine_symbols.add(symbol)
    if not isinstance(ref["evidence"], list) or len(ref["evidence"]) != 5:
        raise ValueError(f"{owner} incomplete evidence bindings")
    roles = set()
    for row in ref["evidence"]:
        _require_exact_fields(owner + " evidence", row, {"role", "path", "sha256", "size"})
        if (row["role"] in roles or not isinstance(row["path"], str)
                or not row["path"].startswith("profiler_output/") or ".." in row["path"].split("/")
                or not isinstance(row["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])
                or type(row["size"]) is not int or row["size"] <= 0):
            raise ValueError(f"{owner} invalid evidence provenance")
        roles.add(row["role"])
    if (roles != {"calibration", "cohort_validation", "reuse_validation", "static_manifest", "combine_static_manifest"}
            or not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(x, str) or not x for x in ref["scope_limitations"])):
        raise ValueError(f"{owner} missing provenance or scope")
    return {"isolated_region_ns": float(isolated), "peer_region_ns": float(peer),
            "control_regions": 2, "main_parameter_regions": regions,
            "supported_main_tiles": tuple(sorted(tiles)), "combine_parameter_regions": combine_regions}


def _normalize_cp_async_mbarrier_notification(value, repository_root, hardware_id):
    """Independent shared notification cycles, with no copy payload service."""
    owner = "scheduling.cp_async_mbarrier_notification"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 architecture")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "schema_version", "hardware_id", "mechanism", "timer_domain", "producer_threads",
        "arrival_kind", "residency_ctas_per_sm", "selection", "one_way_validation_fraction",
        "roundtrip_midpoint_validation_fraction", "services", "measurements", "raw",
        "source_sha256", "runner_sha256", "production_or_evaluation_duration_consumed", "scope_limitations"})
    _require_no_duration_consumption(ref, owner)
    expected = {"schema_version": 1, "hardware_id": hardware_id,
        "mechanism": "isolated_cp_async_mbarrier_notification",
        "timer_domain": "paired_same_CTA_clock64", "producer_threads": 128,
        "arrival_kind": "cp.async.mbarrier.arrive.noinc", "residency_ctas_per_sm": 1,
        "selection": "median_all_grid_hold_endpoint_first_events",
        "one_way_validation_fraction": .1, "roundtrip_midpoint_validation_fraction": .05}
    if (any(ref[k] != v for k, v in expected.items())
            or any(type(ref[k]) is not int for k in ("schema_version", "producer_threads", "residency_ctas_per_sm"))):
        raise ValueError(f"{owner} mechanism boundary drifted")

    def positive(x):
        return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0

    rows = ref["measurements"]
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError(f"{owner} incomplete notification measurements")
    cells = {}
    for row in rows:
        fields = ("consumer_threads", "signal", "grid_ctas", "producer_hold_cycles", "consumer_hold_cycles")
        _require_exact_fields(owner + " measurement", row, set(fields) | {"points", "roundtrip_midpoint_error"})
        key = tuple(row[k] for k in fields)
        if any(type(x) is not int for x in key) or key in cells:
            raise ValueError(f"{owner} invalid or duplicated notification cell")
        points = row["points"]
        if not isinstance(points, list) or len(points) != 3:
            raise ValueError(f"{owner} missing notification endpoints/midpoint")
        for point in points:
            _require_exact_fields(owner + " point", point, {
                "rounds", "total_cycles", "forward_cycles", "reverse_cycles"})
            if (type(point["rounds"]) is not int or any(not positive(point[k])
                    for k in ("total_cycles", "forward_cycles", "reverse_cycles"))):
                raise ValueError(f"{owner} invalid notification cycles")
        if [p["rounds"] for p in points] != [128, 256, 512]:
            raise ValueError(f"{owner} notification repetition domain drifted")
        times = [p["total_cycles"] for p in points]
        period = (times[2] - times[0]) / 384
        error = abs(times[1] - times[0] - 128 * period) / times[1]
        if (period <= 0 or error > .05 or isinstance(row["roundtrip_midpoint_error"], bool)
                or not isinstance(row["roundtrip_midpoint_error"], (int, float))
                or not math.isfinite(row["roundtrip_midpoint_error"])
                or not math.isclose(error, row["roundtrip_midpoint_error"], rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"{owner} independent roundtrip midpoint failed")
        cells[key] = points
    if set(cells) != {(c, 2, g, p, q) for c in (128, 256) for g in (1, 32, 132)
                      for p in (0, 512) for q in (0, 512)}:
        raise ValueError(f"{owner} notification physical domain drifted")
    services = ref["services"]
    if not isinstance(services, list) or len(services) != 2:
        raise ValueError(f"{owner} incomplete selected services")
    result = {}
    for row in services:
        _require_exact_fields(owner + " service", row, {"consumer_threads", "forward_cycles", "reverse_cycles"})
        count = row["consumer_threads"]
        if type(count) is not int or count not in (128, 256) or count in result:
            raise ValueError(f"{owner} invalid selected consumer domain")
        domain = [points for key, points in cells.items() if key[0] == count]
        selected = {}
        for field in ("forward_cycles", "reverse_cycles"):
            value = statistics.median(points[i][field] for points in domain for i in (0, 2))
            if (not positive(row[field]) or row[field] != value
                    or any(abs(p[field] / value - 1) > .1 for points in domain for p in points)):
                raise ValueError(f"{owner} one-way service or independent validation failed")
            selected[field] = float(value)
        result[count] = selected
    raw = ref["raw"]
    _require_exact_fields(owner + " raw", raw, {"path", "sha256"})
    if (not isinstance(raw["path"], str)
            or not raw["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in raw["path"].split("/")
            or any(not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h)
                   for h in (raw["sha256"], ref["source_sha256"], ref["runner_sha256"]))
            or not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(x, str) or not x for x in ref["scope_limitations"])):
        raise ValueError(f"{owner} invalid notification provenance")
    return result


def _normalize_bf16_wgmma_enqueue(value, repository_root, hardware_id):
    """Typed enqueue/suffix services and earliest coissued SS completion."""
    owner = "FA3 WGMMA enqueue evidence"
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner, ref, {"schema_version", "hardware_id", "mechanism", "timer_domain",
        "selection", "endpoint_repeats", "validation_repeats", "maximum_relative_tail_deviation",
        "producer_scalar_or_global_payload", "enqueue_services", "retirement_tails", "measurements",
        "raw", "source_sha256", "runner_sha256", "production_or_evaluation_duration_consumed", "scope_limitations",
        "coissued_ss_first_retirement"})
    _require_no_duration_consumption(ref, owner)
    expected = {"schema_version": 2, "hardware_id": "nvidia_h200",
        "mechanism": "isolated_bf16_wgmma_enqueue_and_retirement_suffix",
        "timer_domain": "same_WG_clock64", "selection": "median_three_grids_single_WG_endpoint_cells",
        "endpoint_repeats": [128, 512], "validation_repeats": 256, "maximum_relative_tail_deviation": .1}
    if (hardware_id != "nvidia_h200" or any(ref[k] != v for k, v in expected.items())
            or ref["producer_scalar_or_global_payload"] is not False
            or any(type(ref[k]) is not int for k in ("schema_version", "validation_repeats"))):
        raise ValueError(f"{owner} mechanism boundary drifted")

    def positive(x):
        return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0

    dimensions = ("ss_n", "ss_k16", "rs_n", "rs_k16", "grid_ctas", "active_warpgroups", "repeats")
    events = ("ss_enqueue_cycles", "rs_enqueue_cycles", "ss_retirement_tail_cycles",
              "rs_retirement_tail_cycles", "pair_rs_retirement_tail_cycles")
    rows = ref["measurements"]
    if not isinstance(rows, list) or len(rows) != 54:
        raise ValueError(f"{owner} incomplete event controls")
    cells = {}
    for row in rows:
        _require_exact_fields(owner + " event", row, set(dimensions) | set(events))
        key = tuple(row[k] for k in dimensions)
        if (any(type(x) is not int for x in key) or key in cells
                or any(not positive(row[k]) for k in events)):
            raise ValueError(f"{owner} invalid or duplicated event")
        cells[key] = row
    shapes = ((176, 8, 128, 11), (192, 4, 64, 12), (128, 8, 128, 8))
    if set(cells) != {s + (g, w, r) for s in shapes for g in (1, 32, 132)
                      for w in (1, 2) for r in (128, 256, 512)}:
        raise ValueError(f"{owner} physical control domain drifted")
    calibrated = [r for r in rows if r["active_warpgroups"] == 1 and r["repeats"] in (128, 512)]
    domain = {("ss", 176, 8), ("ss", 192, 4), ("ss", 128, 8),
              ("rs", 128, 11), ("rs", 64, 12), ("rs", 128, 8)}
    services, tails = ref["enqueue_services"], ref["retirement_tails"]
    if not isinstance(services, list) or len(services) != 6 or not isinstance(tails, list) or len(tails) != 2:
        raise ValueError(f"{owner} incomplete typed services")
    prefixes = {}
    for service in services:
        _require_exact_fields(owner + " prefix", service, {
            "operand_source", "instruction_n", "k_steps", "enqueue_cycles"})
        source, n, k = (service[x] for x in ("operand_source", "instruction_n", "k_steps"))
        if (not isinstance(source, str) or type(n) is not int or type(k) is not int
                or (source, n, k) not in domain or (source, n, k) in prefixes):
            raise ValueError(f"{owner} invalid prefix opcode")
        samples = [r for r in rows if (r[source + "_n"], r[source + "_k16"]) == (n, k)]
        selected = statistics.median(r[source + "_enqueue_cycles"] for r in calibrated
            if (r[source + "_n"], r[source + "_k16"]) == (n, k))
        if (not positive(service["enqueue_cycles"]) or service["enqueue_cycles"] != selected
                or any(selected > r[source + "_enqueue_cycles"] * 1.1 for r in samples)
                or any(abs(r[source + "_enqueue_cycles"] / selected - 1) > .1
                       for r in samples if r["active_warpgroups"] == 1)):
            raise ValueError(f"{owner} enqueue floor validation failed")
        prefixes[source, n, k] = float(selected)
    suffixes = {}
    for service in tails:
        _require_exact_fields(owner + " suffix", service, {"operand_source", "instruction_n", "retirement_tail_cycles"})
        n = service["instruction_n"]
        if service["operand_source"] != "rs" or type(n) is not int or n not in (64, 128) or n in suffixes:
            raise ValueError(f"{owner} only measured RS suffixes can be priced")
        selected = statistics.median(r[k] for r in calibrated if r["rs_n"] == n
            for k in ("rs_retirement_tail_cycles", "pair_rs_retirement_tail_cycles"))
        if (not positive(service["retirement_tail_cycles"]) or service["retirement_tail_cycles"] != selected
                or any(abs(r[k] / selected - 1) > .1 for r in rows if r["rs_n"] == n
                       for k in ("rs_retirement_tail_cycles", "pair_rs_retirement_tail_cycles"))):
            raise ValueError(f"{owner} RS suffix validation failed")
        suffixes[n] = float(selected)
    # An earliest completion is weaker than a whole-cohort barrier. The
    # existing instrumented SS-only burst records both WGs in one CTA epoch;
    # its three extra clock reads are removed with matched no-SS controls.
    cohort = ref["coissued_ss_first_retirement"]
    _require_exact_fields(owner + " first retirement", cohort, {
        "selection", "timer_domain", "active_warpgroups", "extra_clock_count",
        "validation_fraction", "measurements", "calibration", "services"})
    if (cohort["selection"] != "grid132_repeats512_trials0_1_first_WG_minus_matched_extra_clocks"
            or cohort["timer_domain"] != "same_CTA_clock64"
            or type(cohort["active_warpgroups"]) is not int or cohort["active_warpgroups"] != 2
            or type(cohort["extra_clock_count"]) is not int or cohort["extra_clock_count"] != 3
            or isinstance(cohort["validation_fraction"], bool) or cohort["validation_fraction"] != .1):
        raise ValueError(f"{owner} coissued first-retirement boundary drifted")
    cohort_fields = ("instruction_n", "k_steps", "grid_ctas", "repeats", "trial")
    cohort_cells = {}
    if not isinstance(cohort["measurements"], list) or len(cohort["measurements"]) != 135:
        raise ValueError(f"{owner} incomplete coissued first-retirement controls")
    for row in cohort["measurements"]:
        _require_exact_fields(owner + " first-retirement cell", row, set(cohort_fields) | {
            "raw_first_completion_cycles", "empty_clock_cycles"})
        key = tuple(row[k] for k in cohort_fields)
        if (any(type(x) is not int for x in key) or key in cohort_cells
                or not positive(row["raw_first_completion_cycles"])
                or not positive(row["empty_clock_cycles"])
                or row["raw_first_completion_cycles"] <= 3 * row["empty_clock_cycles"]):
            raise ValueError(f"{owner} invalid coissued first-retirement cell")
        cohort_cells[key] = row
    ss_shapes = ((176, 8), (192, 4), (128, 8))
    if set(cohort_cells) != {(n, k, g, r, t) for n, k in ss_shapes
                            for g in (1, 32, 132) for r in (128, 256, 512) for t in range(5)}:
        raise ValueError(f"{owner} coissued first-retirement domain drifted")
    calibration_cells = {}
    if not isinstance(cohort["calibration"], list) or len(cohort["calibration"]) != 3:
        raise ValueError(f"{owner} incomplete pooled first-retirement calibration")
    for row in cohort["calibration"]:
        _require_exact_fields(owner + " first-retirement calibration", row, {
            "instruction_n", "k_steps", "raw_first_completion_cycles", "empty_clock_cycles"})
        n, k = row["instruction_n"], row["k_steps"]
        if (type(n) is not int or type(k) is not int or (n, k) not in ss_shapes
                or (n, k) in calibration_cells):
            raise ValueError(f"{owner} invalid pooled first-retirement calibration")
        # The pooled median of two equal-size trial populations lies between
        # their medians; the raw-bound owner retains the exact pooled value.
        trials = [cohort_cells[n, k, 132, 512, t] for t in (0, 1)]
        for field in ("raw_first_completion_cycles", "empty_clock_cycles"):
            if (not positive(row[field])
                    or not min(x[field] for x in trials) <= row[field] <= max(x[field] for x in trials)):
                raise ValueError(f"{owner} pooled first-retirement median is inconsistent")
        calibration_cells[n, k] = row
    first_completions = {}
    if not isinstance(cohort["services"], list) or len(cohort["services"]) != 3:
        raise ValueError(f"{owner} incomplete coissued first-retirement services")
    for service in cohort["services"]:
        _require_exact_fields(owner + " first-retirement service", service, {
            "instruction_n", "k_steps", "first_completion_cycles"})
        n, k = service["instruction_n"], service["k_steps"]
        if (type(n) is not int or type(k) is not int or (n, k) not in ss_shapes
                or (n, k, 2) in first_completions):
            raise ValueError(f"{owner} invalid coissued first-retirement opcode")
        calibration = calibration_cells[n, k]
        selected = calibration["raw_first_completion_cycles"] - 3 * calibration["empty_clock_cycles"]
        samples = [x for key, x in cohort_cells.items() if key[:2] == (n, k)]
        if (not positive(service["first_completion_cycles"])
                or service["first_completion_cycles"] != selected
                or any(abs(selected / (x["raw_first_completion_cycles"] - 3 * x["empty_clock_cycles"]) - 1) > .1
                       for x in samples)):
            raise ValueError(f"{owner} coissued first-retirement validation failed")
        first_completions[n, k, 2] = float(selected)
    raw = ref["raw"]
    _require_exact_fields(owner + " raw", raw, {"path", "sha256"})
    if (not isinstance(raw["path"], str) or not raw["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in raw["path"].split("/")
            or any(not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h)
                   for h in (raw["sha256"], ref["source_sha256"], ref["runner_sha256"]))
            or not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(x, str) or not x for x in ref["scope_limitations"])):
        raise ValueError(f"{owner} invalid provenance")
    return {"enqueue_cycles": prefixes, "pv_retirement_tail_cycles": suffixes,
            "first_coissued_ss_completion_cycles": first_completions}


def _normalize_bf16_ss_wgmma_busy_capacity(value, repository_root, hardware_id, clock_profile):
    """Backlogged SS work/cycle, independent of its retirement latency."""
    owner = "BF16 SS WGMMA busy capacity"
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(value, owner)
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "schema_version", "hardware_id", "mechanism", "scope", "selection",
        "validation_fraction", "peak_flops_per_sm_cycle", "measurements",
        "qualification", "capture", "scope_limitations",
        "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(ref, owner)
    scope = {"dtype": "bf16", "operand_source": "ss", "instruction_m": 64,
        "instruction_k": 16, "instruction_n": [128, 176], "k_steps": 8,
        "active_warpgroups": [1, 2], "grids": [1, 132], "queues": [1, 4],
        "cta_threads": 256, "shared_bytes": 196608}
    if (hardware_id != "nvidia_h200" or ref["hardware_id"] != hardware_id
            or type(ref["schema_version"]) is not int or ref["schema_version"] != 1
            or ref["mechanism"] != "bf16_ss_backlogged_wgmma_busy_capacity"
            or ref["scope"] != scope
            or ref["selection"] != "maximum_all_queue4_work_per_core_cycle"
            or ref["validation_fraction"] != .05):
        raise ValueError(f"{owner} mechanism scope drifted")
    for key in ("instruction_m", "instruction_k", "k_steps", "cta_threads", "shared_bytes"):
        if type(ref["scope"][key]) is not int:
            raise ValueError(f"{owner} scope requires integer geometry")
    for key in ("instruction_n", "active_warpgroups", "grids", "queues"):
        if any(type(x) is not int for x in ref["scope"][key]):
            raise ValueError(f"{owner} scope requires integer geometry")
    qualification = ref["qualification"]
    _require_exact_fields(owner + " qualification", qualification, {
        "path", "sha256", "timing_launches", "numeric_pattern_launches",
        "records", "scientific_controls_passed"})
    if (qualification["scientific_controls_passed"] is not True
            or any(type(qualification[k]) is not int for k in (
                "timing_launches", "numeric_pattern_launches", "records"))
            or (qualification["timing_launches"], qualification["numeric_pattern_launches"],
                qualification["records"]) != (288, 8, 114976)):
        raise ValueError(f"{owner} incomplete scientific qualification")
    for evidence in (ref["capture"], {k: qualification[k] for k in ("path", "sha256")}):
        _require_exact_fields(owner + " evidence", evidence, {"path", "sha256"})
        if (not isinstance(evidence["path"], str)
                or not evidence["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
                or ".." in evidence["path"].split("/")
                or not isinstance(evidence["sha256"], str)
                or re.fullmatch("[0-9a-f]{64}", evidence["sha256"]) is None):
            raise ValueError(f"{owner} invalid provenance")
    def positive(x):
        return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0
    rows = ref["measurements"]
    if not isinstance(rows, list) or len(rows) != 96:
        raise ValueError(f"{owner} requires all96 independent endpoint periods")
    periods = {}
    for row in rows:
        _require_exact_fields(owner + " period", row, {
            "instruction_n", "queue", "grid", "active_warpgroups", "rotate", "trial",
            "cycles_per_instruction", "ns_per_instruction"})
        key = tuple(row[k] for k in (
            "instruction_n", "queue", "grid", "active_warpgroups", "rotate", "trial"))
        cycles, ns = row["cycles_per_instruction"], row["ns_per_instruction"]
        if (any(type(x) is not int for x in key) or key in periods
                or not positive(cycles) or not positive(ns)
                or abs(cycles / ns * 1e9 / clock_profile["tensor_clock_hz"] - 1) > .05):
            raise ValueError(f"{owner} invalid endpoint period or clock")
        periods[key] = cycles, ns
    expected = {(n, q, g, w, r, t) for n in (128, 176) for q in (1, 4)
                for g in (1, 132) for w in (1, 2) for r in (0, 1) for t in range(3)}
    if set(periods) != expected:
        raise ValueError(f"{owner} incomplete physical domain")
    peak = ref["peak_flops_per_sm_cycle"]
    selected = max(2 * 64 * n * 16 / values[0]
                   for (n, q, g, w, r, t), values in periods.items() if q == 4)
    if not positive(peak) or not math.isclose(peak, selected, rel_tol=1e-12):
        raise ValueError(f"{owner} selected capacity is not the independent maximum")
    for (n, q, g, w, r, t), (cycles, ns) in periods.items():
        rate = 2 * 64 * n * 16 / cycles
        if rate > peak * 1.05 or (q == 4 and abs(rate / peak - 1) > .05):
            raise ValueError(f"{owner} peak/control validation failed")
        for unit in (0, 1):
            samples = [periods[n, q, g, w, r, trial][unit] for trial in range(3)]
            if (max(samples) - min(samples)) / statistics.median(samples) > .05:
                raise ValueError(f"{owner} trial validation failed")
    if (not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(x, str) or not x for x in ref["scope_limitations"])):
        raise ValueError(f"{owner} missing transfer limitations")
    return {"flops_per_sm_cycle": float(peak), "instruction_n": (128, 176),
            "k_steps": 8, "active_warpgroups": (1, 2)}


def _normalize_fa3_gqa_wgmma_batches(value, repository_root, hardware_id, clock_profile):
    """Typed retirement/enqueue plus independently bounded SS busy work."""
    owner = "scheduling.fa3_gqa_wgmma_batches"
    if hardware_id != "nvidia_h200" or clock_profile is None:
        raise ValueError(f"{owner} requires H200 independent active-clock evidence")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(value, owner)
    ref = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", ref, {
        "schema_version", "hardware_id", "mechanism", "timer_domain", "instruction_m",
        "instruction_k", "selection", "midpoint_validation_fraction", "two_wg_validation_fraction",
        "mixed_clock_validation_fraction", "batch_services", "raw", "executed_source_sha256",
        "executed_runner_sha256", "generated_helper_sha256", "scope_limitations",
        "production_or_evaluation_duration_consumed", "enqueue_reference", "ss_busy_capacity_reference"})
    _require_no_duration_consumption(ref, owner)
    expected = {"schema_version": 3, "hardware_id": hardware_id,
        "mechanism": "isolated_bf16_wgmma_retirement_batches",
        "timer_domain": "same_WG_clock64_globaltimer", "instruction_m": 64, "instruction_k": 16,
        "selection": "median_three_grids_of_ns512_minus_ns128_over384",
        "midpoint_validation_fraction": .05, "two_wg_validation_fraction": .1,
        "mixed_clock_validation_fraction": .05}
    if (any(ref[key] != item for key, item in expected.items())
            or any(type(ref[key]) is not int for key in ("schema_version", "instruction_m", "instruction_k"))):
        raise ValueError(f"{owner} mechanism boundary drifted")

    def positive(item):
        return (not isinstance(item, bool) and isinstance(item, (int, float))
                and math.isfinite(item) and item > 0)

    domain = {("ss", 176, 8): 88, ("ss", 192, 4): 96, ("ss", 128, 8): 64,
              ("rs", 128, 11): 88, ("rs", 64, 12): 96, ("rs", 128, 8): 64}
    batches = ref["batch_services"]
    if not isinstance(batches, list) or len(batches) != len(domain):
        raise ValueError(f"{owner} incomplete typed batch domain")
    services, two_wg = {}, {}
    for batch in batches:
        _require_exact_fields(owner + " batch", batch, {
            "operand_source", "instruction_n", "k_steps", "register_a_words_per_thread",
            "accumulator_words_per_thread", "scalar_values_per_thread", "measurements",
            "mixed_clock_validation"})
        source, n, k = (batch[field] for field in ("operand_source", "instruction_n", "k_steps"))
        if (not isinstance(source, str) or type(n) is not int or type(k) is not int
                or (source, n, k) not in domain or (source, n, k) in services):
            raise ValueError(f"{owner} invalid or duplicated typed batch")
        key = source, n, k
        footprints = {"register_a_words_per_thread": 4 * k if source == "rs" else 0,
            "accumulator_words_per_thread": n // 2, "scalar_values_per_thread": domain[key]}
        if any(type(batch[field]) is not int or batch[field] != count for field, count in footprints.items()):
            raise ValueError(f"{owner} incompatible register footprint")
        rows = batch["measurements"]
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError(f"{owner} requires six grid/WG endpoints")
        periods = {}
        for row in rows:
            _require_exact_fields(owner + " endpoint", row, {
                "grid_ctas", "active_warpgroups", "ns_128", "ns_256", "ns_512"})
            cell = row["grid_ctas"], row["active_warpgroups"]
            times = tuple(row[f"ns_{repeats}"] for repeats in (128, 256, 512))
            if (any(type(x) is not int for x in cell) or cell in periods
                    or not all(positive(x) for x in times) or not times[0] < times[1] < times[2]):
                raise ValueError(f"{owner} invalid batch endpoints")
            period = (times[2] - times[0]) / 384
            if abs(times[1] - (times[0] + 128 * period)) / times[1] > .05:
                raise ValueError(f"{owner} independent midpoint validation failed")
            periods[cell] = period * 1e-9
        if set(periods) != {(g, w) for g in (1, 32, 132) for w in (1, 2)}:
            raise ValueError(f"{owner} incomplete grid/WG domain")
        services[key] = statistics.median(periods[g, 1] for g in (1, 32, 132))
        two_wg[key] = statistics.median(periods[g, 2] for g in (1, 32, 132))
        if not services[key] <= two_wg[key] <= 2 * services[key] * 1.1:
            raise ValueError(f"{owner} inconsistent two-WG retirement")
        clocks = batch["mixed_clock_validation"]
        if not isinstance(clocks, list) or len(clocks) != 18:
            raise ValueError(f"{owner} incomplete mixed-clock validation")
        cells = set()
        for row in clocks:
            _require_exact_fields(owner + " mixed clock", row, {
                "grid_ctas", "active_warpgroups", "repeats", "clock_hz_p50"})
            cell = tuple(row[field] for field in ("grid_ctas", "active_warpgroups", "repeats"))
            if (any(type(x) is not int for x in cell) or cell in cells
                    or not positive(row["clock_hz_p50"])
                    or abs(row["clock_hz_p50"] / clock_profile["tensor_clock_hz"] - 1) > .05):
                raise ValueError(f"{owner} invalid independent mixed clock")
            cells.add(cell)
        if cells != {(g, w, r) for g in (1, 32, 132) for w in (1, 2) for r in (128, 256, 512)}:
            raise ValueError(f"{owner} mixed-clock domain drifted")
    raw = ref["raw"]
    _require_exact_fields(owner + " raw", raw, {"path", "sha256"})
    if (not isinstance(raw["path"], str)
            or not raw["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in raw["path"].split("/")
            or any(not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h)
                   for h in (raw["sha256"], ref["executed_source_sha256"],
                             ref["executed_runner_sha256"], ref["generated_helper_sha256"]))
            or not isinstance(ref["scope_limitations"], list) or not ref["scope_limitations"]
            or any(not isinstance(x, str) or not x for x in ref["scope_limitations"])):
        raise ValueError(f"{owner} invalid mechanism provenance")
    return {"retirement_s": services, "two_wg_period_s": two_wg,
            "validated_active_warpgroups": (1, 2),
            "ss_busy_capacity": _normalize_bf16_ss_wgmma_busy_capacity(
                ref["ss_busy_capacity_reference"], repository_root, hardware_id, clock_profile),
            "enqueue": _normalize_bf16_wgmma_enqueue(ref["enqueue_reference"], repository_root, hardware_id)}


def _normalize_bf16_wgmma_tensor_service_clock(value, repository_root, hardware_id):
    """Independent active clock and validated core-cycle service scope."""
    owner = "scheduling.bf16_wgmma_tensor_service_clock"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} requires its measured H200 architecture")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256", "production_or_evaluation_duration_consumed"})
    reference = json.loads(_read_owned_reference(value, owner, hardware_id, repository_root))
    _require_exact_fields(owner + " reference", reference, {
        "schema_version", "hardware_id", "mechanism", "timer_domain", "source_boundary",
        "selection", "dtype", "active_warpgroups", "instruction_m", "instruction_n",
        "instruction_k", "measured_clock_hz", "measurements", "raw", "source_sha256",
        "production_or_evaluation_duration_consumed", "scope_limitations", "single_wg_validation",
        "mixed_core_cycle_validation"})
    _require_no_duration_consumption(reference, owner)
    expected = {"schema_version": 3, "hardware_id": hardware_id,
        "mechanism": "isolated_bf16_wgmma_tensor_service_clock",
        "timer_domain": "clock64_and_globaltimer_same_cta",
        "source_boundary": "tensor_only_shared_operand_wgmma_no_attention",
        "selection": "median_of_three_grid_clock_medians", "dtype": "bf16",
        "active_warpgroups": 2, "instruction_m": 64, "instruction_n": 128,
        "instruction_k": 16, "production_or_evaluation_duration_consumed": False}
    if (any(reference[key] != item for key, item in expected.items())
            or any(type(reference[key]) is not int for key in (
                "schema_version", "active_warpgroups", "instruction_m", "instruction_n", "instruction_k"))):
        raise ValueError(f"{owner} reference boundary drifted")
    rows = reference["measurements"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError(f"{owner} requires three independent grid points")
    clocks = []
    for row in rows:
        _require_exact_fields(owner + " measurement", row, {"grid_ctas", "clock_hz_p50"})
        clock = row["clock_hz_p50"]
        if (type(row["grid_ctas"]) is not int or row["grid_ctas"] <= 0
                or isinstance(clock, bool) or not isinstance(clock, (int, float))
                or not math.isfinite(clock) or clock <= 0):
            raise ValueError(f"{owner} clock must be finite and positive")
        clocks.append(float(clock))
    if [row["grid_ctas"] for row in rows] != [1, 32, 132]:
        raise ValueError(f"{owner} grid evidence is incomplete or duplicated")
    selected = reference["measured_clock_hz"]
    if (isinstance(selected, bool) or not isinstance(selected, (int, float))
            or not math.isfinite(selected) or selected <= 0
            or not math.isclose(selected, statistics.median(clocks), rel_tol=0, abs_tol=0.001)):
        raise ValueError(f"{owner} selected clock disagrees with mechanism measurements")
    validation = reference["single_wg_validation"]
    _require_exact_fields(owner + " single-WG validation", validation, {
        "source_boundary", "active_warpgroups", "selection", "measurements", "raw",
        "source_sha256", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(validation, owner)
    if (validation["source_boundary"] != "single_active_wg_tensor_only_no_attention"
            or type(validation["active_warpgroups"]) is not int or validation["active_warpgroups"] != 1
            or validation["selection"] != "all_36_pure_tensor_cells_validation_only"
            or not isinstance(validation["measurements"], list) or len(validation["measurements"]) != 36):
        raise ValueError(f"{owner} single-WG evidence boundary drifted")
    cells, single_clocks = set(), []
    for row in validation["measurements"]:
        dimensions = ("matrix_instructions", "reserved_scalar_values", "exp_variant", "grid_ctas")
        _require_exact_fields(owner + " single-WG cell", row, set(dimensions) | {"clock_hz_p50"})
        key = tuple(row[field] for field in dimensions)
        rate = row["clock_hz_p50"]
        if (any(type(row[field]) is not int for field in dimensions) or key in cells
                or isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate <= 0):
            raise ValueError(f"{owner} invalid single-WG cell")
        cells.add(key)
        single_clocks.append(rate)
    expected_cells = {(m, v, e, g) for m in (4, 8, 16) for v in (16, 64, 128)
                      for e in (0, 1) for g in (1, 132)}
    if cells != expected_cells or not min(single_clocks) <= selected <= max(single_clocks):
        raise ValueError(f"{owner} single-WG validation does not cover the existing clock")
    evidence_raw = validation["raw"]
    _require_exact_fields(owner + " single-WG raw", evidence_raw, {"path", "sha256", "size"})
    if (not isinstance(evidence_raw["path"], str)
            or not evidence_raw["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in evidence_raw["path"].split("/")
            or type(evidence_raw["size"]) is not int or evidence_raw["size"] <= 0
            or not isinstance(evidence_raw["sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", evidence_raw["sha256"])
            or not isinstance(validation["source_sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", validation["source_sha256"])):
        raise ValueError(f"{owner} invalid single-WG provenance")
    raw = reference["raw"]
    _require_exact_fields(owner + " raw", raw, {"path", "sha256", "size"})
    if (not isinstance(raw["path"], str)
            or not raw["path"].startswith(f"profiler_output/microbench/{hardware_id}/")
            or ".." in raw["path"].split("/")
            or isinstance(raw["size"], bool) or not isinstance(raw["size"], int) or raw["size"] <= 0
            or not isinstance(raw["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", raw["sha256"])
            or not isinstance(reference["source_sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", reference["source_sha256"])
            or not isinstance(reference["scope_limitations"], list)
            or not reference["scope_limitations"]
            or any(not isinstance(item, str) or not item for item in reference["scope_limitations"])):
        raise ValueError(f"{owner} provenance is invalid")
    mixed = reference["mixed_core_cycle_validation"]
    _require_exact_fields(owner + " mixed-core validation", mixed, {
        "selection", "maximum_relative_clock_deviation", "measurements",
        "core_cycle_roles", "source_records", "production_or_evaluation_duration_consumed"})
    _require_no_duration_consumption(mixed, owner)
    core_roles = ("compute_fma", "compute_sfu", "warp_issue", "compute_alu")
    if (mixed["selection"] != "all_same_wg_mixed_cells_validation_only"
            or mixed["maximum_relative_clock_deviation"] != .05
            or mixed["core_cycle_roles"] != list(core_roles)
            or mixed["source_records"] != ["raw", "single_wg_validation.raw"]
            or not isinstance(mixed["measurements"], list) or len(mixed["measurements"]) != 42):
        raise ValueError(f"{owner} mixed-core domain or evidence population drifted")
    mixed_cells = set()
    dimensions = ("active_warpgroups", "matrix_instructions", "scalar_values", "exp_variant", "grid_ctas")
    for row in mixed["measurements"]:
        _require_exact_fields(owner + " mixed-core cell", row, set(dimensions) | {"clock_hz_p50"})
        key = tuple(row[k] for k in dimensions)
        rate = row["clock_hz_p50"]
        if (any(type(x) is not int for x in key) or key in mixed_cells
                or isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate <= 0
                or abs(rate / selected - 1) > .05):
            raise ValueError(f"{owner} mixed core clock fails independent range validation")
        mixed_cells.add(key)
    expected_mixed = {(1, m, v, e, g) for m in (4, 8, 16) for v in (16, 64, 128)
                      for e in (0, 1) for g in (1, 132)}
    expected_mixed |= {(2, 16, v, 1, g) for v in (64, 128) for g in (1, 32, 132)}
    if mixed_cells != expected_mixed:
        raise ValueError(f"{owner} mixed core evidence is incomplete")
    return {"tensor_clock_hz": float(selected), "active_warpgroups": 2,
            "core_cycle_roles": core_roles,
            "validated_active_warpgroups": (1, 2),
            "instruction_n": 128, "dtype": "bf16"}


def _normalize_h200_fused_moe_w2_active_sm_clock(
        value, repository_root, hardware_id):
    """Load the independent, same-CTA clock64/globaltimer W2 clock point."""

    owner = "scheduling.h200_fused_moe_w2_active_sm_clock"
    if hardware_id != "nvidia_h200":
        raise ValueError(f"{owner} is H200-only")
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    })
    payload = _read_owned_reference(
        value, owner, hardware_id, repository_root)
    reference = json.loads(payload)
    _require_exact_fields(owner + " reference", reference, {
        "schema_version", "hardware_id", "mechanism", "probe_mode",
        "k_rounds", "static_probe_boundary", "selection", "measurements",
        "selected_sm_clock_hz", "raw", "source_sha256", "source_launch",
        "production_or_evaluation_duration_consumed",
        "parameter_fit_from_production_duration", "scope_limitations",
    })
    if (reference["schema_version"] != 1
            or reference["hardware_id"] != hardware_id
            or reference["mechanism"] != "bf16_w2_panel_wgmma_active_sm_clock"
            or reference["probe_mode"] != "gather_plus_wgmma"
            or reference["k_rounds"] != 12
            or reference["static_probe_boundary"] != (
                "synthetic_routed_panel_cp_async_and_wgmma_no_router_or_output")
            or reference["selection"] != (
                "median_of_three_grid_cta_clock_mhz_p50")
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["parameter_fit_from_production_duration"]
            is not False):
        raise ValueError(f"{owner} reference has an invalid evidence boundary")
    if (not isinstance(reference["source_sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}",
                                reference["source_sha256"])
            or not isinstance(reference["scope_limitations"], list)
            or not reference["scope_limitations"]
            or any(not isinstance(item, str) or not item
                   for item in reference["scope_limitations"])):
        raise ValueError(f"{owner} reference provenance is invalid")
    rows = reference["measurements"]
    if (not isinstance(rows, list) or len(rows) != 3
            or [row["grid_ctas"] for row in rows] != [132, 2112, 33784]):
        raise ValueError(f"{owner} reference lacks the three grid points")
    measured = []
    for row in rows:
        _require_exact_fields(owner + " measurement", row, {
            "grid_ctas", "cta_clock_mhz_p50"})
        clock_mhz = row["cta_clock_mhz_p50"]
        if (isinstance(clock_mhz, bool)
                or not isinstance(clock_mhz, (int, float))
                or not math.isfinite(clock_mhz) or clock_mhz <= 0):
            raise ValueError(f"{owner} measured clock must be positive")
        measured.append(float(clock_mhz))
    selected = reference["selected_sm_clock_hz"]
    if (isinstance(selected, bool) or not isinstance(selected, (int, float))
            or not math.isfinite(selected) or selected <= 0
            or not math.isclose(
                selected, statistics.median(measured) * 1e6,
                rel_tol=0, abs_tol=0.001)):
        raise ValueError(f"{owner} selected clock disagrees with probe")
    source = reference["source_launch"]
    required_source = {
        "algorithm": "triton_grouped_gemm:w2", "dtype": "bf16",
        "variant": "w2", "tile_m": 128, "tile_n": 256,
        "tile_k": 64, "group_size_m": 16,
        "threads_per_cta": 256,
        "dynamic_shared_memory_bytes": 196608, "experts": 128,
        "hidden": 2048, "intermediate": 768,
        "grid_min": 132, "grid_max": 33784,
    }
    _require_exact_fields(owner + " source launch", source,
                          set(required_source))
    if source != required_source:
        raise ValueError(f"{owner} source launch drifted")
    raw = reference["raw"]
    _require_exact_fields(owner + " raw", raw, {"path", "sha256", "size"})
    if (not isinstance(raw["path"], str)
            or not raw["path"].startswith(
                "profiler_output/microbench/nvidia_h200/")
            or ".." in raw["path"].split("/")
            or isinstance(raw["size"], bool)
            or not isinstance(raw["size"], int) or raw["size"] <= 0
            or not isinstance(raw["sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", raw["sha256"])):
        raise ValueError(f"{owner} raw pin is invalid")
    return {"sm_clock_hz": float(selected), "source_launch": source,
            "reference_path": value["reference_path"]}


def _normalize_metax_fused_moe_bm32_bf16_mma_occupancy(
        value, repository_root):
    """Normalize the source-closed C500 BM32 Fused-MoE MMA point."""

    owner = "scheduling.metax_fused_moe_bm32_bf16_mma_occupancy"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "metax_bm32_bf16_mma_occupancy_v1",
        "software_revision": "maca_3.7.2",
        "dtype": "bf16",
        "rate_unit": "flops_per_ap_cycle",
    }
    fields = set(text_fields) | {
        "result_path", "result_sha256", "reference_path",
        "reference_sha256", "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    result_path = value["result_path"]
    if (not isinstance(result_path, str)
            or not result_path.startswith(
                "profiler_output/microbench/metax_c500/")
            or ".." in result_path.split("/")):
        raise ValueError(f"{owner}.result_path is outside its evidence owner")
    result_sha256 = value["result_sha256"]
    if (not isinstance(result_sha256, str) or len(result_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in result_sha256)):
        raise ValueError(f"{owner}.result_sha256 must be lowercase SHA256")
    _require_no_duration_consumption(value, owner)
    if value["reference_path"] != str(
            _METAX_FUSED_MOE_BM32_BF16_MMA_OCCUPANCY_REFERENCE):
        raise ValueError(f"{owner}.reference_path is not canonical")
    if value["reference_sha256"] != (
            _METAX_FUSED_MOE_BM32_BF16_MMA_OCCUPANCY_REFERENCE_SHA256):
        raise ValueError(f"{owner}.reference_sha256 is not canonical")
    payload = (repository_root
               / _METAX_FUSED_MOE_BM32_BF16_MMA_OCCUPANCY_REFERENCE).read_bytes()
    if hashlib.sha256(payload).hexdigest() != value["reference_sha256"]:
        raise ValueError("MetaX BM32 Fused-MoE MMA reference SHA256 mismatch")
    reference = json.loads(payload)
    _require_exact_fields(
        "MetaX BM32 Fused-MoE MMA reference", reference,
        {
            "schema_version", "hardware_id", "software_revision",
            "compiler", "mechanism", "rate_unit",
            "architecture_peak_flops_per_ap_cycle", "ap_count",
            "wave_size", "mma_shape", "accumulator_dtype",
            "accumulator_vector_width", "selection", "measurement",
            "stability", "provenance",
            "production_or_evaluation_duration_consumed",
            "parameter_fit_from_production_duration", "forward_eligible",
        })
    if (
            reference["schema_version"] != 1
            or reference["hardware_id"] != "metax_c500"
            or reference["software_revision"] != text_fields[
                "software_revision"]
            or reference["compiler"] != "mxcc 1.0.0 (d9102a1572)"
            or reference["mechanism"] != (
                "register_only_bf16_mma_wave_accumulator_saturation")
            or reference["rate_unit"] != text_fields["rate_unit"]
            or reference["architecture_peak_flops_per_ap_cycle"] != 2048.0
            or reference["ap_count"] != 104
            or reference["wave_size"] != 64
            or reference["mma_shape"] != [16, 16, 16]
            or reference["accumulator_dtype"] != "fp32"
            or reference["accumulator_vector_width"] != 4
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["parameter_fit_from_production_duration"] is not False
            or reference["forward_eligible"] is not True):
        raise ValueError("MetaX BM32 Fused-MoE MMA reference contract drifted")
    selection = reference["selection"]
    _require_exact_fields(
        "MetaX BM32 Fused-MoE MMA selection", selection,
        {
            "active_waves_per_ap", "accumulator_chains", "tile_m",
            "tile_n", "tile_k", "threads_per_cta",
            "dynamic_shared_memory_bytes", "resident_ctas_per_ap",
        })
    expected_selection = {
        "active_waves_per_ap": 4,
        "accumulator_chains": 2,
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 128,
        "threads_per_cta": 256,
        "dynamic_shared_memory_bytes": 49_152,
        "resident_ctas_per_ap": 1,
    }
    for name, expected in expected_selection.items():
        item = selection[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"MetaX BM32 Fused-MoE MMA {name} must be int")
        if item != expected:
            raise ValueError(
                f"MetaX BM32 Fused-MoE MMA {name} must equal {expected}")
    measurement = reference["measurement"]
    _require_exact_fields(
        "MetaX BM32 Fused-MoE MMA measurement", measurement,
        {"iterations", "selected_repeat", "flops_per_ap_cycle"})
    if (measurement["iterations"] != 65_536
            or measurement["selected_repeat"] != 2):
        raise ValueError("MetaX BM32 Fused-MoE MMA measurement contract drifted")
    rate = measurement["flops_per_ap_cycle"]
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or not 0.0 < float(rate) <= reference[
                "architecture_peak_flops_per_ap_cycle"]):
        raise ValueError("MetaX BM32 Fused-MoE MMA rate is invalid")
    stability = reference["stability"]
    _require_exact_fields(
        "MetaX BM32 Fused-MoE MMA stability", stability,
        {"repeat_count", "maximum_relative_rate_delta", "comparison",
         "repeat_result_sha256"})
    delta = stability["maximum_relative_rate_delta"]
    repeat_hashes = stability["repeat_result_sha256"]
    if (
            stability["repeat_count"] != 2
            or isinstance(delta, bool) or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
            or not 0.0 <= float(delta) <= 0.005
            or stability["comparison"] != (
                "absolute difference divided by the two-repeat mean")
            or not isinstance(repeat_hashes, list)
            or len(repeat_hashes) != stability["repeat_count"]
            or any(
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)
                for digest in repeat_hashes)):
        raise ValueError("MetaX BM32 Fused-MoE MMA stability contract drifted")
    provenance = reference["provenance"]
    _require_exact_fields(
        "MetaX BM32 Fused-MoE MMA provenance", provenance,
        {
            "capture_baseline_commit", "capture_command", "source_sha256",
            "binary_sha256", "physical_device_id", "owner",
            "remote_snapshot", "recorded_pids", "result_paths",
        })
    for name in ("source_sha256", "binary_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"MetaX BM32 Fused-MoE MMA {name} is invalid")
    baseline = provenance["capture_baseline_commit"]
    if (not isinstance(baseline, str) or len(baseline) != 40
            or any(character not in "0123456789abcdef"
                   for character in baseline)):
        raise ValueError("MetaX BM32 Fused-MoE MMA baseline is invalid")
    if (provenance["capture_command"]
            != "MACA_VISIBLE_DEVICES=0 mma_accumulator_sweep"
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or not isinstance(provenance["owner"], str)
            or not provenance["owner"]
            or (not isinstance(provenance["remote_snapshot"], str)
                or not provenance["remote_snapshot"].strip()
                or ".." in provenance["remote_snapshot"].split("/"))
            or not isinstance(provenance["recorded_pids"], list)
            or len(provenance["recorded_pids"]) != 2
            or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                   for pid in provenance["recorded_pids"])
            or not isinstance(provenance["result_paths"], list)
            or len(provenance["result_paths"]) != 2
            or provenance["result_paths"][1] != result_path
            or repeat_hashes[1] != result_sha256):
        raise ValueError("MetaX BM32 Fused-MoE MMA provenance drifted")
    return {
        **text_fields,
        **expected_selection,
        "ap_count": reference["ap_count"],
        "wave_size": reference["wave_size"],
        "architecture_peak_flops_per_ap_cycle": float(
            reference["architecture_peak_flops_per_ap_cycle"]),
        "flops_per_ap_cycle": float(rate),
        "result_path": result_path,
        "result_sha256": result_sha256,
        "reference_path": value["reference_path"],
        "reference_sha256": value["reference_sha256"],
        "production_or_evaluation_duration_consumed": False,
    }


def _normalize_metax_fused_moe_grouped_operand_pipeline(
        value, repository_root, *, owner, model, tile_signature,
        reference_path, reference_sha256, declares_domain):
    """Load one source-closed C500 grouped-operand pipeline mechanism.

    Every deployed Triton tile signature carries its own compiled resources
    and therefore its own synthetic-routing microbenchmark; this loader is
    parameterized by that signature so one signature can never silently
    consume another's coefficients.
    """

    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": model,
        "software_revision": "maca_3.7.2",
        "dtype": "bf16",
        "timer_domain": "torch_cuda_event_gpu_timeline",
    }
    integer_fields = dict(tile_signature)
    fields = (
        set(text_fields) | set(integer_fields)
        | {
            "result_path", "result_sha256", "reference_path",
            "reference_sha256", "production_or_evaluation_duration_consumed",
        })
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    normalized_integers = {}
    for name, expected in integer_fields.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
        normalized_integers[name] = item
    result_path = value["result_path"]
    if (not isinstance(result_path, str)
            or not result_path.startswith(
                "profiler_output/microbench/metax_c500/")
            or ".." in result_path.split("/")):
        raise ValueError(f"{owner}.result_path is outside its evidence owner")
    result_sha256 = value["result_sha256"]
    if (not isinstance(result_sha256, str) or len(result_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in result_sha256)):
        raise ValueError(f"{owner}.result_sha256 must be lowercase SHA256")
    _require_no_duration_consumption(value, owner)
    if value["reference_path"] != str(reference_path):
        raise ValueError(f"{owner}.reference_path is not canonical")
    if value["reference_sha256"] != reference_sha256:
        raise ValueError(f"{owner}.reference_sha256 is not canonical")
    payload = (repository_root / reference_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != value["reference_sha256"]:
        raise ValueError(
            f"{owner} operand-pipeline reference SHA256 mismatch")
    reference = json.loads(payload)
    reference_fields = {
        "schema_version", "hardware_id", "software_revision",
        "mechanism", "timer_domain", "signature", "scope", "method",
        "cells", "extraction", "selected_parameters", "stability",
        "provenance", "allowed_conclusion",
        "production_or_evaluation_duration_consumed",
        "parameter_fit_from_production_duration", "forward_eligible",
    }
    # The first accepted signature (BM32) predates the machine-readable domain
    # bound and states its domain only in prose; every signature measured
    # afterwards must carry it so the predictor can fail closed outside it.
    if declares_domain:
        reference_fields.add("domain")
    _require_fields_with_optional(
        f"{owner} reference", reference, reference_fields,
        _REFERENCE_OPTIONAL_FIELDS)
    if (reference["schema_version"] != 1
            or reference["hardware_id"] != "metax_c500"
            or reference["software_revision"] != text_fields[
                "software_revision"]
            or reference["mechanism"] != "triton_grouped_operand_pipeline"
            or reference["timer_domain"] != text_fields["timer_domain"]
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["parameter_fit_from_production_duration"] is not False
            or reference["forward_eligible"] is not True):
        raise ValueError(f"{owner} reference contract drifted")
    signature = reference["signature"]
    _require_exact_fields(f"{owner} signature", signature, set(integer_fields))
    for name, expected in integer_fields.items():
        item = signature[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner} signature {name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner} signature {name} must equal {expected}")
    parameters = reference["selected_parameters"]
    _require_exact_fields(
        f"{owner} parameters", parameters,
        {
            "fixed_span_ns", "local_k_round_interval_ns",
            "empty_wave_slot_ns",
        })
    coefficients = {}
    for name, item in parameters.items():
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) or float(item) <= 0.0):
            raise ValueError(
                f"{owner} {name} must be finite and positive")
        coefficients[name] = float(item)
    maximum_expert_m_tiles = None
    if declares_domain:
        domain = reference["domain"]
        _require_exact_fields(
            f"{owner} domain", domain, {"maximum_expert_m_tiles", "reason"})
        maximum_expert_m_tiles = domain["maximum_expert_m_tiles"]
        if (isinstance(maximum_expert_m_tiles, bool)
                or not isinstance(maximum_expert_m_tiles, int)
                or maximum_expert_m_tiles <= 0):
            raise ValueError(
                f"{owner} domain.maximum_expert_m_tiles must be a positive "
                "integer")
        if (not isinstance(domain["reason"], str)
                or not domain["reason"].strip()):
            raise ValueError(f"{owner} domain.reason must be non-empty")
    return {
        **text_fields,
        **normalized_integers,
        **coefficients,
        "maximum_expert_m_tiles": maximum_expert_m_tiles,
        "result_path": result_path,
        "result_sha256": result_sha256,
        "reference_path": value["reference_path"],
        "reference_sha256": value["reference_sha256"],
        "production_or_evaluation_duration_consumed": False,
    }


def _metax_grouped_operand_pipeline_signature(
        *, tile_m, tile_n, tile_k, num_warps, num_stages, threads_per_cta,
        dynamic_shared_memory_bytes_per_cta):
    """Return one deployed compiled program's resource contract.

    ``pipeline_depth`` is the compiled stage count: the operand buffer the
    program allocates is what amortizes DRAM latency across a K round.
    """

    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "group_size_m": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "threads_per_cta": threads_per_cta,
        "dynamic_shared_memory_bytes_per_cta": (
            dynamic_shared_memory_bytes_per_cta),
        "resident_ctas_per_ap": 1,
        "pipeline_depth": num_stages,
    }


# vLLM-metax 0.24.0's BF16 non-batch-invariant default rule selects exactly
# these compiled programs by batch-token count.  Each one needs its own
# synthetic-routing sweep: they differ in tile shape, stage count and operand
# buffer size, and BLOCK_SIZE_M=64 even has two programs that share every
# resource fact except the warp count.  ``declares_domain`` is False only for
# the first accepted signature, which predates the machine-readable domain
# bound and states its domain in prose.
_METAX_GROUPED_OPERAND_PIPELINE_SPECS = {
    "metax_fused_moe_bm32_operand_pipeline": {
        "model": "metax_bm32_bf16_grouped_operand_pipeline_v1",
        "reference_path": _METAX_FUSED_MOE_BM32_OPERAND_PIPELINE_REFERENCE,
        "reference_sha256": (
            _METAX_FUSED_MOE_BM32_OPERAND_PIPELINE_REFERENCE_SHA256),
        "declares_domain": False,
        "signature": _metax_grouped_operand_pipeline_signature(
            tile_m=32, tile_n=64, tile_k=128, num_warps=4, num_stages=3,
            threads_per_cta=256,
            dynamic_shared_memory_bytes_per_cta=49_152),
    },
    "metax_fused_moe_bm16_operand_pipeline": {
        "model": "metax_bm16_bf16_grouped_operand_pipeline_v1",
        "reference_path": _METAX_FUSED_MOE_BM16_OPERAND_PIPELINE_REFERENCE,
        "reference_sha256": (
            _METAX_FUSED_MOE_BM16_OPERAND_PIPELINE_REFERENCE_SHA256),
        "declares_domain": True,
        "signature": _metax_grouped_operand_pipeline_signature(
            tile_m=16, tile_n=64, tile_k=128, num_warps=4, num_stages=4,
            threads_per_cta=256,
            dynamic_shared_memory_bytes_per_cta=40_960),
    },
    # The above-512 batch-token default program, which every chunked-prefill step of the retained deployment selects.
    "metax_fused_moe_bm128_bn128_operand_pipeline": {
        "model": "metax_bm128_bn128_bf16_grouped_operand_pipeline_v1",
        "reference_path": _METAX_FUSED_MOE_BM128_BN128_OPERAND_PIPELINE_REFERENCE,
        "reference_sha256": (
            _METAX_FUSED_MOE_BM128_BN128_OPERAND_PIPELINE_REFERENCE_SHA256),
        "declares_domain": True,
        "signature": _metax_grouped_operand_pipeline_signature(
            tile_m=128, tile_n=128, tile_k=64, num_warps=8,
            num_stages=3, threads_per_cta=512,
            dynamic_shared_memory_bytes_per_cta=65_536),
    },
    # The 129-512 batch-token default program, the eight-warp BM64 variant.
    "metax_fused_moe_bm64_bn128_warps8_operand_pipeline": {
        "model": "metax_bm64_bn128_warps8_bf16_grouped_operand_pipeline_v1",
        "reference_path": _METAX_FUSED_MOE_BM64_BN128_WARPS8_OPERAND_PIPELINE_REFERENCE,
        "reference_sha256": (
            _METAX_FUSED_MOE_BM64_BN128_WARPS8_OPERAND_PIPELINE_REFERENCE_SHA256),
        "declares_domain": True,
        "signature": _metax_grouped_operand_pipeline_signature(
            tile_m=64, tile_n=128, tile_k=64, num_warps=8,
            num_stages=3, threads_per_cta=512,
            dynamic_shared_memory_bytes_per_cta=49_152),
    },
    # The 97-128 batch-token default program.  BLOCK_SIZE_M=64 has two deployed programs that share tile shape, block threads and operand buffer; only the warp count separates them.
    "metax_fused_moe_bm64_bn128_warps4_operand_pipeline": {
        "model": "metax_bm64_bn128_warps4_bf16_grouped_operand_pipeline_v1",
        "reference_path": _METAX_FUSED_MOE_BM64_BN128_WARPS4_OPERAND_PIPELINE_REFERENCE,
        "reference_sha256": (
            _METAX_FUSED_MOE_BM64_BN128_WARPS4_OPERAND_PIPELINE_REFERENCE_SHA256),
        "declares_domain": True,
        "signature": _metax_grouped_operand_pipeline_signature(
            tile_m=64, tile_n=128, tile_k=64, num_warps=4,
            num_stages=3, threads_per_cta=512,
            dynamic_shared_memory_bytes_per_cta=49_152),
    },
    # The 65-96 batch-token default program: BM32/BN128/BK64 with four warps, three stages and a 40,960-B operand buffer.
    "metax_fused_moe_bm32_bn128_operand_pipeline": {
        "model": "metax_bm32_bn128_bf16_grouped_operand_pipeline_v1",
        "reference_path": _METAX_FUSED_MOE_BM32_BN128_OPERAND_PIPELINE_REFERENCE,
        "reference_sha256": (
            _METAX_FUSED_MOE_BM32_BN128_OPERAND_PIPELINE_REFERENCE_SHA256),
        "declares_domain": True,
        "signature": _metax_grouped_operand_pipeline_signature(
            tile_m=32, tile_n=128, tile_k=64, num_warps=4,
            num_stages=3, threads_per_cta=256,
            dynamic_shared_memory_bytes_per_cta=40_960),
    },
}


_SCHEDULING_OPTIONAL_FIELDS.update(_METAX_GROUPED_OPERAND_PIPELINE_SPECS)
# The contract module owns the key list that the predictor dispatches over;
# every configurable mechanism must appear there, and every declared key must
# resolve to an HWSpec attribute so the predictor never needs a defaulted
# lookup.
_UNDECLARED_PIPELINE_SPECS = (
    set(_METAX_GROUPED_OPERAND_PIPELINE_SPECS)
    - set(METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS))
if _UNDECLARED_PIPELINE_SPECS:
    raise ImportError(
        "MetaX grouped-operand pipeline specs are not declared in the model "
        f"contract: {sorted(_UNDECLARED_PIPELINE_SPECS)}")


def _normalize_metax_grouped_operand_pipeline(
        name, value, repository_root):
    """Load one source-closed C500 grouped-operand pipeline mechanism."""

    spec = _METAX_GROUPED_OPERAND_PIPELINE_SPECS[name]
    return _normalize_metax_fused_moe_grouped_operand_pipeline(
        value, repository_root,
        owner=f"scheduling.{name}",
        model=spec["model"],
        tile_signature=spec["signature"],
        reference_path=spec["reference_path"],
        reference_sha256=spec["reference_sha256"],
        declares_domain=spec["declares_domain"])


def _normalize_fa_tile16_operand_completion(value, repository_root):
    """Load the finite source-shaped MetaX tile16 operand completion."""

    owner = "scheduling.fa_tile16_operand_completion"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(
        owner,
        value,
        {
            "model", "software_revision", "reference_path",
            "reference_sha256", "production_or_evaluation_duration_consumed",
        })
    if value["model"] != (
            "fresh_operand_load_barrier_bf16_mma_completion_v1"):
        raise ValueError(f"{owner}.model is unsupported")
    if value["software_revision"] != "maca_3.7.2":
        raise ValueError(f"{owner}.software_revision must be 'maca_3.7.2'")
    if value["reference_path"] != str(
            _METAX_FA_TILE16_OPERAND_COMPLETION_REFERENCE):
        raise ValueError(f"{owner}.reference_path is not canonical")
    if value["reference_sha256"] != (
            _METAX_FA_TILE16_OPERAND_COMPLETION_REFERENCE_SHA256):
        raise ValueError(f"{owner}.reference_sha256 is not canonical")
    if value["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(f"{owner} cannot consume production duration")
    path = repository_root / _METAX_FA_TILE16_OPERAND_COMPLETION_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != value["reference_sha256"]:
        raise ValueError("MetaX tile16 operand reference SHA256 mismatch")
    reference = json.loads(payload)
    _require_exact_fields(
        "MetaX tile16 operand reference",
        reference,
        {
            "schema_version", "hardware_id", "software_revision",
            "mechanism", "duration_unit", "clock_hz", "ap_count",
            "active_ap_count", "wave_size", "panel_rows", "panel_count",
            "trial_count", "selected_statistic", "maximum_relative_span",
            "measurements", "provenance",
            "production_or_evaluation_duration_consumed",
            "parameter_fit_from_production_duration", "forward_eligible",
        })
    if (reference["schema_version"] != 1
            or reference["hardware_id"] != "metax_c500"
            or reference["software_revision"] != "maca_3.7.2"
            or reference["mechanism"] != (
                "fresh_operand_load_barrier_bf16_mma_completion")
            or reference["duration_unit"] != "ns_per_operand_completion"
            or reference["selected_statistic"] != (
                "median_block_cycles_per_operand")
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["parameter_fit_from_production_duration"] is not False
            or reference["forward_eligible"] is not True):
        raise ValueError("MetaX tile16 operand reference contract drifted")
    expected_integers = {
        "ap_count": 104,
        "active_ap_count": 64,
        "wave_size": 64,
        "panel_rows": 16,
        "panel_count": 256,
        "trial_count": 7,
    }
    for name, expected in expected_integers.items():
        item = reference[name]
        if (isinstance(item, bool) or not isinstance(item, int)
                or item != expected):
            raise ValueError(
                f"MetaX tile16 operand reference.{name} must equal "
                f"{expected}")
    clock_hz = reference["clock_hz"]
    maximum_span = reference["maximum_relative_span"]
    if (isinstance(clock_hz, bool)
            or not isinstance(clock_hz, (int, float))
            or float(clock_hz) != 1125000000.0
            or isinstance(maximum_span, bool)
            or not isinstance(maximum_span, (int, float))
            or float(maximum_span) != 0.05):
        raise ValueError("MetaX tile16 operand timing contract drifted")
    expected_ledger = {64: (2, 4), 128: (4, 8)}
    completion_ns_by_head_dim = {}
    measurements = reference["measurements"]
    if not isinstance(measurements, list) or len(measurements) != 2:
        raise ValueError(
            "MetaX tile16 operand reference requires D64 and D128 cells")
    for index, measurement in enumerate(measurements):
        context = f"MetaX tile16 operand measurements[{index}]"
        if not isinstance(measurement, dict):
            raise TypeError(f"{context} must be a mapping")
        _require_exact_fields(
            context,
            measurement,
            {
                "head_dim", "loads_128b_per_lane",
                "mma_16x16x16bf16_per_lane",
                "median_cycles_per_operand", "completion_ns",
                "relative_span", "trial_median_cycles",
            })
        head_dim = measurement["head_dim"]
        if (isinstance(head_dim, bool) or not isinstance(head_dim, int)
                or head_dim not in expected_ledger
                or head_dim in completion_ns_by_head_dim):
            raise ValueError(f"{context}.head_dim is invalid")
        expected_loads, expected_mma = expected_ledger[head_dim]
        if (measurement["loads_128b_per_lane"] != expected_loads
                or measurement["mma_16x16x16bf16_per_lane"] != expected_mma):
            raise ValueError(f"{context} source instruction ledger drifted")
        cycles = measurement["median_cycles_per_operand"]
        completion_ns = measurement["completion_ns"]
        relative_span = measurement["relative_span"]
        trials = measurement["trial_median_cycles"]
        for name, item in (
                ("median_cycles_per_operand", cycles),
                ("completion_ns", completion_ns),
                ("relative_span", relative_span)):
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item)) or float(item) <= 0.0):
                raise ValueError(f"{context}.{name} must be finite and positive")
        if float(relative_span) > float(maximum_span):
            raise ValueError(f"{context} failed its stability gate")
        if (not isinstance(trials, list)
                or len(trials) != reference["trial_count"]
                or any(isinstance(item, bool)
                       or not isinstance(item, (int, float))
                       or not math.isfinite(float(item))
                       or float(item) <= 0.0 for item in trials)):
            raise ValueError(f"{context}.trial_median_cycles is invalid")
        expected_ns = float(cycles) / float(clock_hz) * 1.0e9
        if not math.isclose(
                float(completion_ns), expected_ns,
                rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{context}.completion_ns disagrees with cycles")
        completion_ns_by_head_dim[head_dim] = float(completion_ns)
    provenance = reference["provenance"]
    if not isinstance(provenance, dict):
        raise TypeError("MetaX tile16 operand provenance must be a mapping")
    _require_exact_fields(
        "MetaX tile16 operand provenance",
        provenance,
        {
            "capture_commit", "physical_device_id", "owner",
            "remote_snapshot", "source_sha256", "binary_sha256",
            "disassembly_sha256", "raw_sha256",
        })
    if (provenance["capture_commit"] != (
            "4ec3eb2c00769931ef3fd60de39a60c4ecaa6878")
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or provenance["owner"] != "research/kernel-accuracy-fa"
            or (not isinstance(provenance["remote_snapshot"], str)
                or not provenance["remote_snapshot"].strip()
                or ".." in provenance["remote_snapshot"].split("/"))):
        raise ValueError("MetaX tile16 operand provenance drifted")
    for name in (
            "source_sha256", "binary_sha256", "disassembly_sha256",
            "raw_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(
                f"MetaX tile16 operand provenance.{name} is invalid")
    base_point = {
        "active_ap_count": expected_integers["active_ap_count"],
        "panel_count": expected_integers["panel_count"],
        "query_heads_per_workgroup": None,
        "completion_ns_by_head_dim": completion_ns_by_head_dim,
    }
    supplement = _load_metax_fa_tile16_operand_completion_d64_g16_p64(
        repository_root)
    return {
        "model": value["model"],
        "software_revision": value["software_revision"],
        **expected_integers,
        "completion_ns_by_head_dim": completion_ns_by_head_dim,
        "points": (base_point, supplement),
        "reference_path": value["reference_path"],
        "reference_sha256": value["reference_sha256"],
        "production_or_evaluation_duration_consumed": False,
    }


def _load_metax_fa_tile16_operand_completion_d64_g16_p64(repository_root):
    """Load the non-overlapping finite D64 tile16 completion point."""

    path = repository_root / (
        _METAX_FA_TILE16_OPERAND_COMPLETION_D64_G16_P64_REFERENCE)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            _METAX_FA_TILE16_OPERAND_COMPLETION_D64_G16_P64_REFERENCE_SHA256):
        raise ValueError("MetaX tile16 D64 G16/P64 reference SHA256 mismatch")
    reference = json.loads(payload)
    owner = "MetaX tile16 D64 G16/P64 operand reference"
    _require_exact_fields(owner, reference, {
        "schema_version", "hardware_id", "software_revision", "mechanism",
        "duration_unit", "clock_hz", "ap_count", "active_ap_count",
        "wave_size", "panel_rows", "panel_count",
        "query_heads_per_workgroup", "trial_count",
        "fresh_process_count", "selected_statistic",
        "maximum_relative_span", "measurements", "provenance",
        "production_or_evaluation_duration_consumed",
        "parameter_fit_from_production_duration", "forward_eligible",
    })
    if (reference["schema_version"] != 1
            or reference["hardware_id"] != "metax_c500"
            or reference["software_revision"] != "maca_3.7.2"
            or reference["mechanism"] != (
                "fresh_operand_load_barrier_bf16_mma_completion")
            or reference["duration_unit"] != "ns_per_operand_completion"
            or reference["selected_statistic"] != (
                "median_block_cycles_per_operand")
            or reference["production_or_evaluation_duration_consumed"]
            is not False
            or reference["parameter_fit_from_production_duration"] is not False
            or reference["forward_eligible"] is not True):
        raise ValueError(f"{owner} identity contract drifted")
    expected_integers = {
        "ap_count": 104,
        "active_ap_count": 16,
        "wave_size": 64,
        "panel_rows": 16,
        "panel_count": 64,
        "query_heads_per_workgroup": 4,
        "trial_count": 7,
        "fresh_process_count": 5,
    }
    for name, expected in expected_integers.items():
        item = reference[name]
        if (isinstance(item, bool) or not isinstance(item, int)
                or item != expected):
            raise ValueError(f"{owner}.{name} must equal {expected}")
    if (reference["clock_hz"] != 1125000000.0
            or reference["maximum_relative_span"] != 0.05):
        raise ValueError(f"{owner} timing contract drifted")
    measurements = reference["measurements"]
    if not isinstance(measurements, list) or len(measurements) != 1:
        raise ValueError(f"{owner} requires exactly one D64 measurement")
    measurement = measurements[0]
    _require_exact_fields(f"{owner}.measurements[0]", measurement, {
        "head_dim", "loads_128b_per_lane", "mma_16x16x16bf16_per_lane",
        "median_cycles_per_operand", "completion_ns", "relative_span",
        "trial_median_cycles",
    })
    if (measurement["head_dim"] != 64
            or measurement["loads_128b_per_lane"] != 2
            or measurement["mma_16x16x16bf16_per_lane"] != 4):
        raise ValueError(f"{owner} source instruction ledger drifted")
    for name in (
            "median_cycles_per_operand", "completion_ns", "relative_span"):
        item = measurement[name]
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) or float(item) <= 0.0):
            raise ValueError(f"{owner}.{name} must be finite and positive")
    if measurement["relative_span"] > reference["maximum_relative_span"]:
        raise ValueError(f"{owner} failed its stability gate")
    trials = measurement["trial_median_cycles"]
    if (not isinstance(trials, list)
            or len(trials) != reference["fresh_process_count"]
            or any(isinstance(item, bool)
                   or not isinstance(item, (int, float))
                   or not math.isfinite(float(item)) or item <= 0.0
                   for item in trials)):
        raise ValueError(f"{owner}.trial_median_cycles is invalid")
    expected_ns = (measurement["median_cycles_per_operand"]
                   / reference["clock_hz"] * 1.0e9)
    if not math.isclose(
            measurement["completion_ns"], expected_ns,
            rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"{owner}.completion_ns disagrees with cycles")
    provenance = reference["provenance"]
    _require_exact_fields(f"{owner}.provenance", provenance, {
        "capture_commit", "physical_device_id", "owner",
        "remote_snapshot", "source_sha256", "binary_sha256", "raw_files",
    })
    if (isinstance(provenance["physical_device_id"], bool)
            or not isinstance(provenance["physical_device_id"], int)
            or provenance["physical_device_id"] < 0
            or provenance["owner"] != "research/kernel-accuracy-fa"
            or (not isinstance(provenance["remote_snapshot"], str)
                or not provenance["remote_snapshot"].strip()
                or ".." in provenance["remote_snapshot"].split("/"))):
        raise ValueError(f"{owner} provenance scope drifted")
    for name in ("capture_commit", "source_sha256", "binary_sha256"):
        digest = provenance[name]
        expected_length = 40 if name == "capture_commit" else 64
        if (not isinstance(digest, str) or len(digest) != expected_length):
            raise ValueError(f"{owner}.provenance.{name} is invalid")
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{owner}.provenance.{name} is not lowercase hex")
    raw_files = provenance["raw_files"]
    if not isinstance(raw_files, list) or len(raw_files) != 5:
        raise ValueError(f"{owner}.provenance.raw_files must have five rows")
    raw_paths = set()
    for index, raw_file in enumerate(raw_files):
        raw_owner = f"{owner}.provenance.raw_files[{index}]"
        _require_exact_fields(raw_owner, raw_file, {"path", "sha256"})
        raw_path = raw_file["path"]
        if (not isinstance(raw_path, str)
                or not raw_path.startswith(
                    "profiler_output/microbench/metax_c500/")
                or ".." in raw_path.split("/") or raw_path in raw_paths):
            raise ValueError(f"{raw_owner}.path is invalid")
        raw_paths.add(raw_path)
        digest = raw_file["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{raw_owner}.sha256 is invalid")
    return {
        "active_ap_count": reference["active_ap_count"],
        "panel_count": reference["panel_count"],
        "query_heads_per_workgroup": (
            reference["query_heads_per_workgroup"]),
        "completion_ns_by_head_dim": {
            measurement["head_dim"]: float(measurement["completion_ns"]),
        },
    }


def _load_metax_gemv_wave8_completion_service(repository_root):
    """Load the source-shaped eight-output GEMV completion service."""

    path = repository_root / _METAX_GEMV_WAVE8_COMPLETION_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            _METAX_GEMV_WAVE8_COMPLETION_REFERENCE_SHA256):
        raise ValueError("MetaX GEMV wave8 completion reference SHA256 mismatch")
    value = json.loads(payload)
    owner = "MetaX GEMV wave8 completion reference"
    _require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "production_or_evaluation_duration_consumed", "target_operator_used",
         "baseline_commit", "source", "analysis", "forward_scope",
         "trial_stdout_sha256", "campaign", "resources", "parameters",
         "validation_rows", "validation", "thresholds", "publishable",
         "disposition"})
    expected = {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "software_revision": "maca_3.7.2",
        "mechanism": "wave64_eight_output_reduction_completion",
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "publishable": True,
        "disposition": "candidate_for_forward_evaluation",
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["forward_scope"]
    _require_exact_fields(
        f"{owner}.forward_scope", scope,
        {"dtype", "block_threads", "outputs_per_cta", "minimum_grid_size",
         "maximum_resident_grid_size", "requires_single_wave_cta",
         "requires_fully_resident_grid"})
    if scope != {
            "dtype": "bf16",
            "block_threads": 64,
            "outputs_per_cta": 8,
            "minimum_grid_size": 1,
            "maximum_resident_grid_size": 3328,
            "requires_single_wave_cta": True,
            "requires_fully_resident_grid": True,
    }:
        raise ValueError(f"{owner}.forward_scope changed")
    parameters = value["parameters"]
    _require_exact_fields(
        f"{owner}.parameters", parameters,
        {"block_threads", "outputs_per_cta", "resident_ctas_per_ap",
         "wave_completion_slots_per_ap",
         "first_wave_cohort_completion_ns_per_ap",
         "subsequent_wave_cohort_completion_ns_per_ap",
         "first_wave_cohort_completion_rate_per_ap_s",
         "subsequent_wave_cohort_completion_rate_per_ap_s"})
    for name, expected_value in {
            "block_threads": 64,
            "outputs_per_cta": 8,
            "resident_ctas_per_ap": 32,
            "wave_completion_slots_per_ap": 4,
    }.items():
        if parameters[name] != expected_value:
            raise ValueError(f"{owner}.parameters.{name} changed")
    first_ns = float(parameters["first_wave_cohort_completion_ns_per_ap"])
    subsequent_ns = float(
        parameters["subsequent_wave_cohort_completion_ns_per_ap"])
    if (not math.isfinite(first_ns) or first_ns <= 0.0
            or not math.isfinite(subsequent_ns) or subsequent_ns <= 0.0):
        raise ValueError(f"{owner} completion times must be positive")
    for path_field, expected_path in (
            ("source", "microbench/metax_c500/compute/reduction/"
             "gemv_wave8_completion.cpp"),
            ("analysis", "microbench/metax_c500/compute/reduction/"
             "gemv_wave8_completion_runner.py")):
        item = value[path_field]
        _require_exact_fields(f"{owner}.{path_field}", item, {"path", "sha256"})
        if item["path"] != expected_path:
            raise ValueError(f"{owner}.{path_field}.path changed")
        source_bytes = (repository_root / expected_path).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != item["sha256"]:
            raise ValueError(f"{owner}.{path_field}.sha256 mismatch")
    validation = value["validation"]
    if (validation["count"] != 11
            or float(validation["mape_percent"]) > 15.0
            or float(validation["max_ape_percent"]) > 25.0):
        raise ValueError(f"{owner} validation gate failed")
    return {
        **scope,
        "ap_count": 104,
        "resident_ctas_per_ap": parameters["resident_ctas_per_ap"],
        "wave_completion_slots_per_ap": parameters[
            "wave_completion_slots_per_ap"],
        "first_wave_cohort_completion_ns_per_ap": first_ns,
        "subsequent_wave_cohort_completion_ns_per_ap": subsequent_ns,
        "reference_path": str(_METAX_GEMV_WAVE8_COMPLETION_REFERENCE),
        "reference_sha256": _METAX_GEMV_WAVE8_COMPLETION_REFERENCE_SHA256,
    }


def _load_metax_gemv_block256_wave8_completion_service(repository_root):
    """Load the source-shaped four-wave eight-output GEMV completion."""

    path = repository_root / _METAX_GEMV_BLOCK256_WAVE8_COMPLETION_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            _METAX_GEMV_BLOCK256_WAVE8_COMPLETION_REFERENCE_SHA256):
        raise ValueError(
            "MetaX block256 GEMV wave8 completion reference SHA256 mismatch")
    value = json.loads(payload)
    owner = "MetaX block256 GEMV wave8 completion reference"
    _require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "production_or_evaluation_duration_consumed", "target_operator_used",
         "baseline_commit", "source", "analysis", "forward_scope",
         "trial_stdout_sha256", "campaign", "resources", "parameters",
         "validation_rows", "validation", "thresholds", "publishable",
         "disposition"})
    expected = {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "software_revision": "maca_3.7.2",
        "mechanism": "four_wave_eight_output_reduction_completion",
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "publishable": True,
        "disposition": "candidate_for_forward_evaluation",
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["forward_scope"]
    _require_exact_fields(
        f"{owner}.forward_scope", scope,
        {"dtype", "block_threads", "outputs_per_cta", "minimum_grid_size",
         "maximum_resident_grid_size", "requires_four_wave_cta",
         "requires_fully_resident_grid", "static_shared_bytes_per_cta",
         "waves_per_cta"})
    if scope != {
            "dtype": "bf16",
            "block_threads": 256,
            "outputs_per_cta": 8,
            "minimum_grid_size": 1,
            "maximum_resident_grid_size": 832,
            "requires_four_wave_cta": True,
            "requires_fully_resident_grid": True,
            "static_shared_bytes_per_cta": 144,
            "waves_per_cta": 4,
    }:
        raise ValueError(f"{owner}.forward_scope changed")
    parameters = value["parameters"]
    _require_exact_fields(
        f"{owner}.parameters", parameters,
        {"block_threads", "outputs_per_cta", "resident_ctas_per_ap",
         "wave_completion_slots_per_ap", "static_shared_bytes_per_cta",
         "waves_per_cta", "first_wave_cohort_completion_ns_per_ap",
         "subsequent_wave_cohort_completion_ns_per_ap",
         "first_wave_cohort_completion_rate_per_ap_s",
         "subsequent_wave_cohort_completion_rate_per_ap_s"})
    for name, expected_value in {
            "block_threads": 256,
            "outputs_per_cta": 8,
            "resident_ctas_per_ap": 8,
            "wave_completion_slots_per_ap": 1,
            "static_shared_bytes_per_cta": 144,
            "waves_per_cta": 4,
    }.items():
        if parameters[name] != expected_value:
            raise ValueError(f"{owner}.parameters.{name} changed")
    first_ns = float(parameters["first_wave_cohort_completion_ns_per_ap"])
    subsequent_ns = float(
        parameters["subsequent_wave_cohort_completion_ns_per_ap"])
    if (not math.isfinite(first_ns) or first_ns <= 0.0
            or not math.isfinite(subsequent_ns) or subsequent_ns <= 0.0):
        raise ValueError(f"{owner} completion times must be positive")
    for path_field, expected_path in (
            ("source", "microbench/metax_c500/compute/reduction/"
             "gemv_block256_wave8_completion.cpp"),
            ("analysis", "microbench/metax_c500/compute/reduction/"
             "gemv_block256_wave8_completion_runner.py")):
        item = value[path_field]
        _require_exact_fields(f"{owner}.{path_field}", item, {"path", "sha256"})
        if item["path"] != expected_path:
            raise ValueError(f"{owner}.{path_field}.path changed")
        source_bytes = (repository_root / expected_path).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != item["sha256"]:
            raise ValueError(f"{owner}.{path_field}.sha256 mismatch")
    validation = value["validation"]
    if (validation["count"] != 12
            or float(validation["mape_percent"]) > 15.0
            or float(validation["max_ape_percent"]) > 25.0):
        raise ValueError(f"{owner} validation gate failed")
    return {
        **scope,
        "ap_count": 104,
        "resident_ctas_per_ap": parameters["resident_ctas_per_ap"],
        "wave_completion_slots_per_ap": parameters[
            "wave_completion_slots_per_ap"],
        "first_wave_cohort_completion_ns_per_ap": first_ns,
        "subsequent_wave_cohort_completion_ns_per_ap": subsequent_ns,
        "reference_path": str(
            _METAX_GEMV_BLOCK256_WAVE8_COMPLETION_REFERENCE),
        "reference_sha256": (
            _METAX_GEMV_BLOCK256_WAVE8_COMPLETION_REFERENCE_SHA256),
    }


def _normalize_memory_dependency_task_latency_model(value):
    """Normalize a bounded source-selected GPU memory-dependency floor."""

    owner = "scheduling.memory_dependency_task_latency_model"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    if "model" not in value:
        raise ValueError(f"{owner} is missing required fields: ['model']")
    model = value["model"]
    if model == "isolated_warm_default_cache_memory_dependency_task_v1":
        text_fields = {
            "model": model,
            "contract": WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
            "duration_unit": "ns_per_complete_nonempty_task",
            "timer_domain": "nsys_gpu_kernel",
            "software_revision": "cuda_12.2_sm80",
            "cache_policy": "warm_default_cache",
        }
        model_integer_fields = {"launch_rank", "fanout_activation_grid"}
        model_coefficient_fields = {"fanout_activation_ns"}
    elif model == (
            "paired_batched_warm_default_cache_memory_dependency_task_v1"):
        text_fields = {
            "model": model,
            "contract": WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
            "duration_unit": "ns_per_complete_nonempty_task",
            "timer_domain": "mcevent_paired_default_stream",
            "software_revision": "maca_3.7.2",
            "cache_policy": "warm_default_cache",
            "pair_pattern": "abba",
        }
        model_integer_fields = {
            "maximum_launch_rank", "launches_per_batch"}
        model_coefficient_fields = set()
    elif model == (
            "paired_batched_multiround_warm_default_cache_"
            "memory_dependency_task_v1"):
        text_fields = {
            "model": model,
            "contract": WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
            "duration_unit": "ns_per_complete_nonempty_task",
            "timer_domain": "mcevent_paired_default_stream",
            "software_revision": "maca_3.7.2",
            "cache_policy": "warm_default_cache",
            "pair_pattern": "abba",
        }
        model_integer_fields = {
            "maximum_launch_rank", "launches_per_batch"}
        model_coefficient_fields = {
            "resident_cta_ns", "wave_round_ns"}
    elif model == (
            "isolated_mcevent_multiround_warm_default_cache_"
            "memory_dependency_task_v1"):
        text_fields = {
            "model": model,
            "contract": WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
            "duration_unit": "ns_per_complete_nonempty_task",
            "timer_domain": "mcevent_gpu_kernel",
            "software_revision": "maca_3.8.2.2",
            "cache_policy": "warm_default_cache",
        }
        model_integer_fields = {"maximum_launch_rank"}
        model_coefficient_fields = {
            "resident_cta_ns", "wave_round_ns"}
    elif model == (
            "paired_mcevent_active_unit_memory_dependency_increment_v1"):
        text_fields = {
            "model": model,
            "contract": WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
            "duration_unit": "ns_increment_over_paired_empty_task",
            "timer_domain": "mcevent_gpu_kernel",
            "software_revision": "maca_3.8.2.2",
            "cache_policy": "warm_default_cache",
            "pair_pattern": "abba",
        }
        model_integer_fields = {"maximum_launch_rank"}
        model_coefficient_fields = {
            "first_active_unit_increment_ns",
            "additional_active_unit_increment_ns",
        }
    else:
        raise ValueError(f"{owner}.model is unsupported: {model!r}")
    integer_fields = {
        "minimum_grid_size", "maximum_grid_size",
        "minimum_block_size", "maximum_block_size",
    } | model_integer_fields
    coefficient_fields = {
        "maximum_sm_clock_hz",
    } | model_coefficient_fields
    if model in (
            "isolated_warm_default_cache_memory_dependency_task_v1",
            "paired_batched_warm_default_cache_memory_dependency_task_v1"):
        coefficient_fields |= {"cta_ns", "warp_ns"}
    if model != (
            "paired_mcevent_active_unit_memory_dependency_increment_v1"):
        coefficient_fields.add("base_ns")
    _require_exact_fields(
        owner, value,
        set(text_fields) | integer_fields | coefficient_fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    integers = {}
    for name in integer_fields:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        integers[name] = item
    if "launch_rank" in integers and integers["launch_rank"] != 1:
        raise ValueError(f"{owner}.launch_rank must equal one")
    if ("maximum_launch_rank" in integers
            and integers["maximum_launch_rank"] not in (1, 2)):
        raise ValueError(
            f"{owner}.maximum_launch_rank must be one or two")
    if integers["minimum_grid_size"] != 1:
        raise ValueError(f"{owner}.minimum_grid_size must equal one")
    if integers["maximum_grid_size"] < integers["minimum_grid_size"]:
        raise ValueError(f"{owner} grid range is invalid")
    if integers["maximum_block_size"] < integers["minimum_block_size"]:
        raise ValueError(f"{owner} block range is invalid")
    if ("fanout_activation_grid" in integers
            and not (integers["minimum_grid_size"]
                     <= integers["fanout_activation_grid"]
                     <= integers["maximum_grid_size"])):
        raise ValueError(
            f"{owner}.fanout_activation_grid is outside the grid range")
    coefficients = {}
    for name in coefficient_fields:
        item = value[name]
        if (isinstance(item, bool)
                or not isinstance(item, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        normalized = float(item)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        coefficients[name] = normalized
    return {
        **text_fields,
        **integers,
        **coefficients,
    }


def _normalize_tensor_iterator_copy_task_latency_model(value):
    """Normalize source-shaped non-contiguous TensorIterator copy floors."""

    owner = "scheduling.tensor_iterator_copy_task_latency_model"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "isolated_tensor_iterator_strided_copy_task_v1",
        "cycle_unit": "sm_cycles_per_complete_task",
        "timer_domain": "cuda_event_gpu_timeline",
        "software_revision": "pytorch_2.8.0_cuda_12.8_sm80_source_semantics",
        "cache_policy": "warm_default_cache",
    }
    integer_fields = {
        "launch_rank", "threads_per_cta", "minimum_grid_size",
        "maximum_grid_size", "fanout_activation_grid",
        "resident_ctas_per_sm",
    }
    _require_exact_fields(
        owner, value,
        set(text_fields) | integer_fields | {"reference_sm_clock_hz", "entries"})
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    integers = {}
    for name in integer_fields:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        integers[name] = item
    if integers["launch_rank"] != 1:
        raise ValueError(f"{owner}.launch_rank must equal one")
    if integers["threads_per_cta"] != 128:
        raise ValueError(f"{owner}.threads_per_cta must equal 128")
    if integers["minimum_grid_size"] != 1:
        raise ValueError(f"{owner}.minimum_grid_size must equal one")
    if integers["maximum_grid_size"] < integers["minimum_grid_size"]:
        raise ValueError(f"{owner} grid range is invalid")
    if not (integers["minimum_grid_size"]
            <= integers["fanout_activation_grid"]
            <= integers["maximum_grid_size"]):
        raise ValueError(
            f"{owner}.fanout_activation_grid is outside the grid range")
    reference_clock = value["reference_sm_clock_hz"]
    if (isinstance(reference_clock, bool)
            or not isinstance(reference_clock, (int, float))):
        raise TypeError(f"{owner}.reference_sm_clock_hz must be numeric")
    reference_clock = float(reference_clock)
    if not math.isfinite(reference_clock) or reference_clock <= 0:
        raise ValueError(
            f"{owner}.reference_sm_clock_hz must be finite and positive")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise TypeError(f"{owner}.entries must be a non-empty list")
    expected_contracts = {
        TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR: ("bf16", 4, 4),
        TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR: ("float32", 3, 2),
    }
    entries = []
    for index, raw in enumerate(raw_entries):
        entry_owner = f"{owner}.entries[{index}]"
        if not isinstance(raw, dict):
            raise TypeError(f"{entry_owner} must be a mapping")
        _require_exact_fields(
            entry_owner, raw,
            {
                "contract", "dtype", "tensor_iterator_dimensions",
                "values_per_thread", "base_cycles",
                "fanout_activation_cycles", "cta_cycles",
            })
        contract = raw["contract"]
        if contract not in expected_contracts:
            raise ValueError(
                f"{entry_owner}.contract is unsupported: {contract!r}")
        expected_dtype, expected_dimensions, expected_values = (
            expected_contracts[contract])
        if raw["dtype"] != expected_dtype:
            raise ValueError(
                f"{entry_owner}.dtype must be {expected_dtype!r}")
        if raw["tensor_iterator_dimensions"] != expected_dimensions:
            raise ValueError(
                f"{entry_owner}.tensor_iterator_dimensions must equal "
                f"{expected_dimensions}")
        if raw["values_per_thread"] != expected_values:
            raise ValueError(
                f"{entry_owner}.values_per_thread must equal "
                f"{expected_values}")
        coefficients = {}
        for name in (
                "base_cycles", "fanout_activation_cycles", "cta_cycles"):
            item = raw[name]
            if (isinstance(item, bool)
                    or not isinstance(item, (int, float))):
                raise TypeError(f"{entry_owner}.{name} must be numeric")
            normalized = float(item)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(
                    f"{entry_owner}.{name} must be finite and positive")
            coefficients[name] = normalized
        entries.append({
            "contract": contract,
            "dtype": expected_dtype,
            "tensor_iterator_dimensions": expected_dimensions,
            "values_per_thread": expected_values,
            **coefficients,
        })
    contracts = tuple(entry["contract"] for entry in entries)
    if set(contracts) != set(expected_contracts) or len(contracts) != 2:
        raise ValueError(
            f"{owner}.entries must contain each supported contract once")
    return {
        **text_fields,
        **integers,
        "reference_sm_clock_hz": reference_clock,
        "entries": tuple(entries),
    }


def _normalize_register_reduction_dependency_cycles(value, repository_root, hardware_id):
    """Normalize isolated register-reduction and CTA-barrier mechanisms."""

    owner = "scheduling.register_reduction_dependency_cycles"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "isolated_register_reduction_and_cta_barrier_v1",
        "cycle_unit": "cycles_per_128_thread_cta_stage",
        "timer_domain": "clock64_same_cta_envelope_cycles",
        "software_revision": "cuda_12.2_sm80",
    }
    fields = set(text_fields) | {
        "threads_per_cta", "quad_reduction_width", "source_sha256",
        "report_sha256", "two_resident_reference_sha256", "entries",
    }
    _require_fields_with_optional(owner, value, fields, {"additional_register_chains"})
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    if value["threads_per_cta"] != 128:
        raise ValueError(f"{owner}.threads_per_cta must equal 128")
    if value["quad_reduction_width"] != 4:
        raise ValueError(f"{owner}.quad_reduction_width must equal four")
    for name in (
            "source_sha256", "report_sha256",
            "two_resident_reference_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise TypeError(f"{owner}.entries must be a non-empty list")
    entry_fields = {
        "lane_local_rows_per_thread", "score_values_per_lane_local_row",
        "fp32_max_quad4_cycles", "fp32_local_sum_cycles",
        "fp32_scale_exp2_sum_cycles",
        "cta_barrier_cycles", "fp32_quad4_sum_cycles",
        "integer_column_predicate_cycles",
        "fp32_to_bf16_conversion_cycles",
    }
    entry_optional_fields = {"two_resident_demand_factors"}
    factor_names = (
        "fp32_max_quad4", "cta_barrier", "fp32_quad4_sum",
        "fp32_scale_exp2_sum", "integer_column_predicate",
        "fp32_to_bf16_conversion",
    )
    factor_fields = {"resident_ctas_per_sm", *factor_names}
    normalized_entries = []
    identities = set()
    for index, entry in enumerate(entries):
        entry_owner = f"{owner}.entries[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{entry_owner} must be a mapping")
        _require_fields_with_optional(
            entry_owner, entry, entry_fields, entry_optional_fields)
        integers = {}
        for name in (
                "lane_local_rows_per_thread",
                "score_values_per_lane_local_row"):
            item = entry[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{entry_owner}.{name} must be an integer")
            if item <= 0:
                raise ValueError(f"{entry_owner}.{name} must be positive")
            integers[name] = item
        identity = (
            integers["lane_local_rows_per_thread"],
            integers["score_values_per_lane_local_row"],
        )
        if identity in identities:
            raise ValueError(f"{owner}.entries contains duplicate geometry")
        identities.add(identity)
        cycles = {}
        for name in (
                "fp32_max_quad4_cycles", "fp32_local_sum_cycles",
                "fp32_scale_exp2_sum_cycles",
                "cta_barrier_cycles", "fp32_quad4_sum_cycles",
                "integer_column_predicate_cycles",
                "fp32_to_bf16_conversion_cycles"):
            item = entry[name]
            if (isinstance(item, bool)
                    or not isinstance(item, (int, float))):
                raise TypeError(f"{entry_owner}.{name} must be numeric")
            normalized = float(item)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(
                    f"{entry_owner}.{name} must be finite and positive")
            cycles[name] = normalized
        normalized_entry = {**integers, **cycles}
        if "two_resident_demand_factors" in entry:
            factors = entry["two_resident_demand_factors"]
            if not isinstance(factors, dict):
                raise TypeError(
                    f"{entry_owner}.two_resident_demand_factors must be a "
                    "mapping")
            factor_owner = f"{entry_owner}.two_resident_demand_factors"
            _require_exact_fields(factor_owner, factors, factor_fields)
            if factors["resident_ctas_per_sm"] != 2:
                raise ValueError(
                    f"{factor_owner}.resident_ctas_per_sm must equal two")
            normalized_factors = {"resident_ctas_per_sm": 2}
            for name in factor_names:
                item = factors[name]
                if (isinstance(item, bool)
                        or not isinstance(item, (int, float))):
                    raise TypeError(f"{factor_owner}.{name} must be numeric")
                normalized = float(item)
                if (not math.isfinite(normalized)
                        or normalized <= 0.0 or normalized > 1.0):
                    raise ValueError(
                        f"{factor_owner}.{name} must be in (0, 1]")
                normalized_factors[name] = normalized
            normalized_entry["two_resident_demand_factors"] = (
                normalized_factors)
        normalized_entries.append(normalized_entry)
    additional_chains = []
    if "additional_register_chains" in value:
        binding = value["additional_register_chains"]
        _require_exact_fields(owner + ".additional_register_chains", binding, {
            "reference_path", "reference_sha256",
            "production_or_evaluation_duration_consumed"})
        reference = json.loads(_read_owned_reference(
            binding, owner, hardware_id, repository_root))
        if (reference["hardware_id"] != hardware_id
                or reference["mechanism"] != "two_row_fp32_register_chains"
                or reference["threads_per_cta"] != 128
                or reference["lane_local_rows_per_thread"] != 2
                or reference["timer_domain"] != "clock64_sm_cycles"
                or reference["production_or_evaluation_duration_consumed"] is not False):
            raise ValueError(f"{owner} additional register-chain domain differs")
        widths = set()
        if not isinstance(reference["entries"], list) or not reference["entries"]:
            raise ValueError(f"{owner} additional register chains must be nonempty")
        for entry in reference["entries"]:
            _require_exact_fields(owner + ".additional_register_chains.entry", entry, {
                "score_values_per_lane_local_row", "fp32_max_quad4_cycles",
                "fp32_local_sum_cycles"})
            width = entry["score_values_per_lane_local_row"]
            if type(width) is not int or width <= 0 or width in widths:
                raise ValueError(f"{owner} additional register-chain width invalid")
            widths.add(width)
            for field in ("fp32_max_quad4_cycles", "fp32_local_sum_cycles"):
                number = entry[field]
                if (isinstance(number, bool) or not isinstance(number, (int, float))
                        or not math.isfinite(number) or number <= 0):
                    raise ValueError(f"{owner} additional register-chain cycles invalid")
            additional_chains.append(dict(entry))
    return {
        **text_fields,
        "threads_per_cta": 128,
        "quad_reduction_width": 4,
        "source_sha256": value["source_sha256"],
        "report_sha256": value["report_sha256"],
        "two_resident_reference_sha256": value[
            "two_resident_reference_sha256"],
        "entries": tuple(normalized_entries),
        "additional_register_chains": tuple(additional_chains),
    }


def _normalize_fa2_local_spill_dependency_cycles(value):
    """Normalize the isolated SM80 FA2 local-stack dependency service."""

    owner = "scheduling.fa2_local_spill_dependency_cycles"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "isolated_fa2_local_spill_round_v1",
        "cycle_unit": "cycles_per_128_thread_cta_spill_round",
        "timer_domain": "clock64_same_cta_envelope_cycles",
        "software_revision": "flash_attn_2_8_3_cuda_12_2_sm80",
    }
    fields = set(text_fields) | {
        "threads_per_cta", "stack_bytes_per_thread",
        "scalar_store_sites", "scalar_load_sites",
        "spill_round_cycles", "source_sha256", "report_sha256",
        "production_binary_sha256", "production_sass_sha256",
    }
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "threads_per_cta": 128,
        "stack_bytes_per_thread": 56,
        "scalar_store_sites": 13,
        "scalar_load_sites": 13,
    }
    for name, expected in expected_integers.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    cycles = value["spill_round_cycles"]
    if isinstance(cycles, bool) or not isinstance(cycles, (int, float)):
        raise TypeError(f"{owner}.spill_round_cycles must be numeric")
    cycles = float(cycles)
    if not math.isfinite(cycles) or cycles <= 0:
        raise ValueError(
            f"{owner}.spill_round_cycles must be finite and positive")
    digests = {}
    for name in (
            "source_sha256", "report_sha256", "production_binary_sha256",
            "production_sass_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
        digests[name] = digest
    return {
        **text_fields,
        **expected_integers,
        "spill_round_cycles": cycles,
        **digests,
    }


def _normalize_fa2_dense_d128_causal_local_spill_dependency_cycles(value):
    """Normalize the isolated SM80 dense causal D128 stack service."""

    owner = (
        "scheduling."
        "fa2_dense_d128_causal_local_spill_dependency_cycles")
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "isolated_fa2_dense_d128_causal_local_spill_round_v1",
        "cycle_unit": "cycles_per_128_thread_cta_local_stack_round",
        "timer_domain": "clock64_same_cta_envelope_cycles",
        "software_revision": "flash_attn_2_8_3_cuda_12_2_sm80",
    }
    fields = set(text_fields) | {
        "threads_per_cta", "stack_bytes_per_thread",
        "scalar_store_sites", "scalar_load_sites",
        "spill_round_cycles", "source_sha256", "report_sha256",
        "production_binary_sha256",
        "production_sass_sha256",
    }
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "threads_per_cta": 128,
        "stack_bytes_per_thread": 176,
        "scalar_store_sites": 51,
        "scalar_load_sites": 53,
    }
    for name, expected in expected_integers.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    item = value["spill_round_cycles"]
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise TypeError(f"{owner}.spill_round_cycles must be numeric")
    cycles = float(item)
    if not math.isfinite(cycles) or cycles <= 0:
        raise ValueError(
            f"{owner}.spill_round_cycles must be finite and positive")
    digests = {}
    for name in (
            "source_sha256", "report_sha256", "production_binary_sha256",
            "production_sass_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
        digests[name] = digest
    return {
        **text_fields,
        **expected_integers,
        "spill_round_cycles": cycles,
        **digests,
    }


def _normalize_cta_barrier_service_cycles(
        value, *,
        owner="scheduling.cta_barrier_service_cycles",
        model="isolated_cta_barrier_service_v1",
        cycle_unit="cycles_per_256_thread_cta_barrier",
        shared_field="dynamic_shared_bytes",
        shared_bytes=32 * 1024,
        software_revision="maca_3.7.2",
        resident_ctas_per_unit=2):
    """Normalize an independently measured CTA-barrier cycle service."""

    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": model,
        "cycle_unit": cycle_unit,
        "timer_domain": "clock64_same_cta_cycles",
        "software_revision": software_revision,
    }
    fields = set(text_fields) | {
        "threads_per_cta", shared_field,
        "resident_ctas_per_unit", "barrier_cycles",
        "source_sha256", "report_sha256",
    }
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    integer_fields = {
        "threads_per_cta": 256,
        shared_field: shared_bytes,
        "resident_ctas_per_unit": resident_ctas_per_unit,
    }
    for name, expected in integer_fields.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    barrier_cycles = value["barrier_cycles"]
    if (isinstance(barrier_cycles, bool)
            or not isinstance(barrier_cycles, (int, float))):
        raise TypeError(f"{owner}.barrier_cycles must be numeric")
    barrier_cycles = float(barrier_cycles)
    if not math.isfinite(barrier_cycles) or barrier_cycles <= 0:
        raise ValueError(f"{owner}.barrier_cycles must be finite and positive")
    for name in ("source_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
    return {
        **text_fields,
        **integer_fields,
        "barrier_cycles": barrier_cycles,
        "source_sha256": value["source_sha256"],
        "report_sha256": value["report_sha256"],
    }


def _normalize_imcontinuous_vec4_reduction_source_service(value):
    """Normalize the independent C500 vec4 reduction source service."""

    owner = "scheduling.imcontinuous_vec4_reduction_source_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "isolated_imcontinuous_vec4_reduction_source_service_v1",
        "cycle_unit": "cycles_per_512_thread_cta_source_program",
        "timer_domain": "clock64_same_cta_cycles",
        "software_revision": "maca_3.7.2",
    }
    fields = set(text_fields) | {
        "threads_per_cta", "dynamic_shared_bytes", "programs",
        "source_sha256", "report_sha256",
    }
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    integer_fields = {
        "threads_per_cta": 512,
        "dynamic_shared_bytes": 4104,
    }
    for name, expected in integer_fields.items():
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if item != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    for name in ("source_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
    programs = value["programs"]
    if not isinstance(programs, list) or not programs:
        raise TypeError(f"{owner}.programs must be a non-empty list")
    expected_programs = {
        2: (256, 2),
        4: (128, 4),
        8: (64, 8),
        10: (64, 8),
        16: (32, 16),
    }
    normalized_programs = {}
    for index, program in enumerate(programs):
        program_owner = f"{owner}.programs[{index}]"
        if not isinstance(program, dict):
            raise TypeError(f"{program_owner} must be a mapping")
        _require_exact_fields(
            program_owner, program,
            {"topk", "block_x", "block_y", "service_cycles_per_cta"})
        integers = {}
        for name in ("topk", "block_x", "block_y"):
            item = program[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{program_owner}.{name} must be an integer")
            if item <= 0:
                raise ValueError(f"{program_owner}.{name} must be positive")
            integers[name] = item
        topk = integers["topk"]
        if topk in normalized_programs:
            raise ValueError(f"{owner}.programs repeats topk={topk}")
        if (topk not in expected_programs
                or (integers["block_x"], integers["block_y"])
                != expected_programs[topk]
                or integers["block_x"] * integers["block_y"] != 512):
            raise ValueError(
                f"{program_owner} has an unsupported source geometry")
        cycles = program["service_cycles_per_cta"]
        if (isinstance(cycles, bool)
                or not isinstance(cycles, (int, float))):
            raise TypeError(
                f"{program_owner}.service_cycles_per_cta must be numeric")
        cycles = float(cycles)
        if not math.isfinite(cycles) or cycles <= 0:
            raise ValueError(
                f"{program_owner}.service_cycles_per_cta must be finite "
                "and positive")
        normalized_programs[topk] = {
            **integers,
            "service_cycles_per_cta": cycles,
            "service": f"imcontinuous_vec4_reduce_topk{topk}",
        }
    if set(normalized_programs) != set(expected_programs):
        raise ValueError(
            f"{owner}.programs must cover top-k {sorted(expected_programs)}")
    return {
        **text_fields,
        **integer_fields,
        "programs": normalized_programs,
        "source_sha256": value["source_sha256"],
        "report_sha256": value["report_sha256"],
    }


def _normalize_fa2_combine_dependency_cycles(value):
    """Normalize source-matched FA2 split-combine dependency mechanisms."""

    owner = "scheduling.fa2_combine_dependency_cycles"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "isolated_fa2_combine_source_dependency_v2",
        "cycle_unit": "cycles_per_128_thread_cta_stage",
        "timer_domain": "clock64_same_cta_envelope_cycles",
        "software_revision": "cuda_12.2_sm80",
    }
    fields = set(text_fields) | {
        "threads_per_cta", "source_sha256", "report_sha256",
        "reduction_entries", "conversion_entries",
        "source_fragment_entries",
    }
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    if value["threads_per_cta"] != 128:
        raise ValueError(f"{owner}.threads_per_cta must equal 128")
    for name in ("source_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")

    reduction_entries = value["reduction_entries"]
    if not isinstance(reduction_entries, list):
        raise TypeError(f"{owner}.reduction_entries must be a list")
    normalized_reductions = []
    reduction_widths = set()
    for index, entry in enumerate(reduction_entries):
        entry_owner = f"{owner}.reduction_entries[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{entry_owner} must be a mapping")
        _require_exact_fields(entry_owner, entry, {
            "allreduce_width", "fp32_max_allreduce_cycles",
            "fp32_sum_allreduce_cycles",
        })
        width = entry["allreduce_width"]
        if isinstance(width, bool) or not isinstance(width, int):
            raise TypeError(f"{entry_owner}.allreduce_width must be an integer")
        if width not in (2, 4, 8, 16, 32):
            raise ValueError(f"{entry_owner}.allreduce_width is unsupported")
        if width in reduction_widths:
            raise ValueError(f"{owner}.reduction_entries repeats a width")
        reduction_widths.add(width)
        cycles = {}
        for name in (
                "fp32_max_allreduce_cycles", "fp32_sum_allreduce_cycles"):
            item = entry[name]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{entry_owner}.{name} must be numeric")
            normalized = float(item)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(
                    f"{entry_owner}.{name} must be finite and positive")
            cycles[name] = normalized
        normalized_reductions.append({"allreduce_width": width, **cycles})
    if reduction_widths != {2, 4, 8, 16, 32}:
        raise ValueError(
            f"{owner}.reduction_entries must cover widths 2/4/8/16/32")

    conversion_entries = value["conversion_entries"]
    if not isinstance(conversion_entries, list):
        raise TypeError(f"{owner}.conversion_entries must be a list")
    normalized_conversions = []
    conversion_geometries = set()
    for index, entry in enumerate(conversion_entries):
        entry_owner = f"{owner}.conversion_entries[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{entry_owner} must be a mapping")
        _require_exact_fields(entry_owner, entry, {
            "block_m", "head_dim", "output_values_per_lane",
            "fp32_to_bf16_output_conversion_cycles",
        })
        integers = {}
        for name in ("block_m", "head_dim", "output_values_per_lane"):
            item = entry[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{entry_owner}.{name} must be an integer")
            if item <= 0:
                raise ValueError(f"{entry_owner}.{name} must be positive")
            integers[name] = item
        identity = (
            integers["block_m"], integers["head_dim"],
            integers["output_values_per_lane"])
        if identity in conversion_geometries:
            raise ValueError(f"{owner}.conversion_entries repeats a geometry")
        conversion_geometries.add(identity)
        cycles = entry["fp32_to_bf16_output_conversion_cycles"]
        if isinstance(cycles, bool) or not isinstance(cycles, (int, float)):
            raise TypeError(
                f"{entry_owner}.fp32_to_bf16_output_conversion_cycles "
                "must be numeric")
        cycles = float(cycles)
        if not math.isfinite(cycles) or cycles <= 0:
            raise ValueError(
                f"{entry_owner}.fp32_to_bf16_output_conversion_cycles "
                "must be finite and positive")
        normalized_conversions.append({
            **integers,
            "fp32_to_bf16_output_conversion_cycles": cycles,
        })
    if conversion_geometries != {(4, 128, 4), (8, 64, 4)}:
        raise ValueError(
            f"{owner}.conversion_entries must cover the two source geometries")

    source_fragment_entries = value["source_fragment_entries"]
    if not isinstance(source_fragment_entries, list):
        raise TypeError(f"{owner}.source_fragment_entries must be a list")
    normalized_fragments = []
    fragment_geometries = set()
    expected_fragment_geometries = {
        (8, 64, 2, 2), (8, 64, 4, 4), (8, 64, 8, 8),
        (4, 128, 2, 2), (4, 128, 3, 4), (4, 128, 4, 4),
        (4, 128, 5, 8), (4, 128, 6, 8), (4, 128, 7, 8),
        (4, 128, 8, 8), (4, 128, 9, 16), (4, 128, 11, 16),
        (4, 128, 16, 16), (4, 128, 22, 32),
    }
    cycle_fields = (
        "slse_publish_transpose_cycles",
        "post_max_exp_local_sum_cycles",
        "post_sum_log_cycles",
        "scale_exp_smem_barrier_cycles",
    )
    for index, entry in enumerate(source_fragment_entries):
        entry_owner = f"{owner}.source_fragment_entries[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{entry_owner} must be a mapping")
        _require_exact_fields(entry_owner, entry, {
            "block_m", "head_dim", "num_splits", "max_splits",
            *cycle_fields,
        })
        integers = {}
        for name in ("block_m", "head_dim", "num_splits", "max_splits"):
            item = entry[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{entry_owner}.{name} must be an integer")
            if item <= 0:
                raise ValueError(f"{entry_owner}.{name} must be positive")
            integers[name] = item
        identity = tuple(integers[name] for name in (
            "block_m", "head_dim", "num_splits", "max_splits"))
        if identity in fragment_geometries:
            raise ValueError(
                f"{owner}.source_fragment_entries repeats a geometry")
        fragment_geometries.add(identity)
        cycles = {}
        for name in cycle_fields:
            item = entry[name]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{entry_owner}.{name} must be numeric")
            normalized = float(item)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(
                    f"{entry_owner}.{name} must be finite and positive")
            cycles[name] = normalized
        normalized_fragments.append({**integers, **cycles})
    # The pinned set is the A800 FA2 heuristic's split choices; a card whose
    # SM count lets the heuristic choose more splits carries those measured
    # rows too.  Every row is a measurement of its (kMaxSplits, full/partial)
    # source class, so extra rows never replace a pinned one.
    if not expected_fragment_geometries <= fragment_geometries:
        raise ValueError(
            f"{owner}.source_fragment_entries must cover the pinned source "
            "geometries")
    return {
        **text_fields,
        "threads_per_cta": 128,
        "source_sha256": value["source_sha256"],
        "report_sha256": value["report_sha256"],
        "reduction_entries": tuple(normalized_reductions),
        "conversion_entries": tuple(normalized_conversions),
        "source_fragment_entries": tuple(normalized_fragments),
    }


def _normalize_metax_fa2_combine_dependency_cycles(value):
    """Normalize source-matched C500 D128 split-reducer stage programs."""

    owner = "scheduling.metax_fa2_combine_dependency_cycles"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "isolated_metax_fa2_d128_combine_dependency_v2",
        "cycle_unit": "cycles_per_256_thread_cta_stage",
        "timer_domain": "clock64_same_cta_cycles",
        "software_revision": "maca_3.7.2",
    }
    fields = set(text_fields) | {"threads_per_cta", "entries"}
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    if value["threads_per_cta"] != 256:
        raise ValueError(f"{owner}.threads_per_cta must equal 256")
    entries = value["entries"]
    if not isinstance(entries, list):
        raise TypeError(f"{owner}.entries must be a list")
    expected_stages = {
        "lse_shared_transpose",
        "max_allreduce",
        "post_max_exp_sum",
        "sum_allreduce",
        "post_sum_log",
        "scale_shared_barrier",
        "output_conversion",
    }
    expected_entry_fields = {
        "block_m", "head_dim", "max_splits", "predicate_class",
        "stage_cycles", "source_sha256", "report_sha256",
    }
    normalized_entries = []
    entry_keys = set()
    for index, entry in enumerate(entries):
        entry_owner = f"{owner}.entries[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{entry_owner} must be a mapping")
        _require_exact_fields(entry_owner, entry, expected_entry_fields)
        integers = {
            "block_m": 8,
            "head_dim": 128,
        }
        for name, expected in integers.items():
            if entry[name] != expected:
                raise ValueError(f"{entry_owner}.{name} must equal {expected}")
        max_splits = entry["max_splits"]
        if max_splits not in (2, 8):
            raise ValueError(f"{entry_owner}.max_splits must be 2 or 8")
        predicate_class = entry["predicate_class"]
        if predicate_class not in ("full", "partial"):
            raise ValueError(
                f"{entry_owner}.predicate_class is unsupported")
        if ((max_splits, predicate_class) not in {
                (2, "full"), (8, "partial")}):
            raise ValueError(
                f"{entry_owner} has no pinned C500 source program")
        for name in ("source_sha256", "report_sha256"):
            digest = entry[name]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    f"{entry_owner}.{name} must be a lowercase SHA256")
        stage_cycles = entry["stage_cycles"]
        if not isinstance(stage_cycles, dict):
            raise TypeError(f"{entry_owner}.stage_cycles must be a mapping")
        _require_exact_fields(
            f"{entry_owner}.stage_cycles", stage_cycles, expected_stages)
        normalized_stages = {}
        for name in expected_stages:
            cycles = stage_cycles[name]
            if (isinstance(cycles, bool)
                    or not isinstance(cycles, (int, float))):
                raise TypeError(
                    f"{entry_owner}.stage_cycles.{name} must be numeric")
            normalized = float(cycles)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(
                    f"{entry_owner}.stage_cycles.{name} must be finite and "
                    "positive")
            normalized_stages[name] = normalized
        key = (max_splits, predicate_class)
        if key in entry_keys:
            raise ValueError(f"{owner}.entries has duplicate source programs")
        entry_keys.add(key)
        normalized_entries.append({
            "block_m": 8,
            "head_dim": 128,
            "max_splits": max_splits,
            "predicate_class": predicate_class,
            "stage_cycles": normalized_stages,
            "source_sha256": entry["source_sha256"],
            "report_sha256": entry["report_sha256"],
        })
    if entry_keys != {(2, "full"), (8, "partial")}:
        raise ValueError(
            f"{owner}.entries must cover the pinned C500 source programs")
    return {
        **text_fields,
        "threads_per_cta": 256,
        "entries": tuple(normalized_entries),
    }


def _normalize_fa2_oaccum_load_use_cycles(value):
    """Normalize the isolated FA2 Oaccum LDG128 load-use mechanism."""

    owner = "scheduling.fa2_oaccum_load_use_cycles"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    text_fields = {
        "model": "isolated_ldg128_fp32x4_load_to_operands_ready_v1",
        "cycle_unit": "sm_cycles_per_128_thread_cta_iteration",
        "timer_domain": "clock64_same_cta_envelope_differential",
        "software_revision": "cuda_12.2_sm80",
    }
    fields = set(text_fields) | {
        "threads_per_cta", "load_bytes_per_lane", "values_per_lane",
        "dependent_fp32_fma_per_lane", "source_sha256", "report_sha256",
        "cache_entries",
    }
    _require_exact_fields(owner, value, fields)
    for name, expected in text_fields.items():
        if not _text_field_matches(name, value[name], expected):
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    integer_expectations = {
        "threads_per_cta": 128,
        "load_bytes_per_lane": 16,
        "values_per_lane": 4,
        "dependent_fp32_fma_per_lane": 4,
    }
    for name, expected in integer_expectations.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    for name in ("source_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a lowercase SHA256")
    entries = value["cache_entries"]
    if not isinstance(entries, list):
        raise TypeError(f"{owner}.cache_entries must be a list")
    normalized_entries = []
    for index, entry in enumerate(entries):
        entry_owner = f"{owner}.cache_entries[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{entry_owner} must be a mapping")
        _require_exact_fields(entry_owner, entry, {
            "cache_tier", "model", "first_load_ready_cycles",
            "additional_load_interval_cycles",
        })
        if entry["cache_tier"] not in ("l2_warm_cg", "dram_cg"):
            raise ValueError(f"{entry_owner}.cache_tier is unknown")
        if entry["model"] != "first_plus_additional_load_interval_v1":
            raise ValueError(f"{entry_owner}.model is unsupported")
        normalized = {}
        for name in (
                "first_load_ready_cycles",
                "additional_load_interval_cycles"):
            item = entry[name]
            if (isinstance(item, bool)
                    or not isinstance(item, (int, float))):
                raise TypeError(f"{entry_owner}.{name} must be numeric")
            item = float(item)
            if not math.isfinite(item) or item <= 0:
                raise ValueError(
                    f"{entry_owner}.{name} must be finite and positive")
            normalized[name] = item
        normalized_entries.append({
            "cache_tier": entry["cache_tier"],
            "model": entry["model"],
            **normalized,
        })
    if {entry["cache_tier"] for entry in normalized_entries} != {
            "l2_warm_cg", "dram_cg"}:
        raise ValueError(
            f"{owner}.cache_entries must cover both cache tiers exactly")
    if len(normalized_entries) != 2:
        raise ValueError(f"{owner}.cache_entries cannot repeat a cache tier")
    return {
        **text_fields,
        **integer_expectations,
        "source_sha256": value["source_sha256"],
        "report_sha256": value["report_sha256"],
        "cache_entries": tuple(normalized_entries),
    }


def _normalize_runtime_task_latency_models(value):
    """Normalize bounded framework task-class lower bounds."""

    owner = "scheduling.runtime_task_latency_models"
    if not isinstance(value, dict) or not value:
        raise TypeError(f"{owner} must be a non-empty mapping")
    supported = {
        PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
        PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS,
    }
    unknown = set(value) - supported
    if unknown:
        raise ValueError(f"{owner} has unsupported task classes: {sorted(unknown)}")
    normalized_models = {}
    for task_class, model in value.items():
        model_owner = f"{owner}.{task_class}"
        if not isinstance(model, dict):
            raise TypeError(f"{model_owner} must be a mapping")
        if (task_class == PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS
                and model.get("model") == (
                    "warm_bf16_binary_tensor_iterator_task_floor_curve_v1")):
            # The floor is a measured curve over element count, not one
            # constant: a task that is too small to fill the device walks an
            # occupancy ramp, so its floor rises with size.  The samples are
            # the measurement's own element counts; no breakpoint or curve
            # shape is chosen here.
            fields = {
                "model", "duration_unit", "timer_domain",
                "software_revision", "derivation_rule", "lookup_rule",
                "probe_block_threads", "floor_ns_by_elements",
            }
            expected = {
                "duration_unit": "ns_per_task",
                "timer_domain": (
                    "paired_differential_events_around_cuda_graph_replay"),
                "model": (
                    "warm_bf16_binary_tensor_iterator_task_floor_curve_v1"),
                "derivation_rule": (
                    "minimum_all_binary_graph_replay_differentials"
                    "_per_element"),
                "lookup_rule": (
                    "step_down_largest_sampled_element_not_exceeding_request"),
            }
            expected_one_of = {
                "software_revision": _accepted_software_revisions(
                    "torch_2.8.0+cu128_cuda_sm80"),
            }
            _require_exact_fields(model_owner, model, fields)
            for name, expected_value in expected.items():
                if model[name] != expected_value:
                    raise ValueError(
                        f"{model_owner}.{name} must be {expected_value!r}")
            for name, allowed in expected_one_of.items():
                if model[name] not in allowed:
                    raise ValueError(
                        f"{model_owner}.{name} must be one of "
                        f"{list(allowed)!r}")
            curve = model["floor_ns_by_elements"]
            if not isinstance(curve, dict) or not curve:
                raise TypeError(
                    f"{model_owner}.floor_ns_by_elements must be a non-empty "
                    "mapping")
            normalized_curve = {}
            previous_elements = None
            previous_duration = None
            for key in sorted(curve, key=lambda item: int(item)):
                elements = int(key)
                duration = curve[key]
                if elements <= 0:
                    raise ValueError(
                        f"{model_owner} floor element count must be positive")
                if (isinstance(duration, bool)
                        or not isinstance(duration, (int, float))
                        or not math.isfinite(duration) or duration <= 0.0):
                    raise ValueError(
                        f"{model_owner} floor duration must be positive")
                if (previous_duration is not None
                        and duration < previous_duration):
                    raise ValueError(
                        f"{model_owner} floor curve must be non-decreasing "
                        "in element count")
                normalized_curve[elements] = float(duration)
                previous_elements = elements
                previous_duration = float(duration)
            probe_block = model["probe_block_threads"]
            if (isinstance(probe_block, bool)
                    or not isinstance(probe_block, int) or probe_block <= 0):
                raise ValueError(
                    f"{model_owner}.probe_block_threads must be positive")
            normalized_models[task_class] = {
                **{name: model[name] for name in
                   ("model", "duration_unit", "timer_domain",
                    "software_revision", "derivation_rule", "lookup_rule")},
                "probe_block_threads": probe_block,
                "floor_ns_by_elements": normalized_curve,
            }
            continue
        if task_class == PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS:
            fields = {
                "model", "duration_unit", "timer_domain",
                "software_revision", "derivation_rule", "minimum_elements",
                "maximum_elements", "duration_ns",
            }
            expected = {
                "duration_unit": "ns_per_task",
                "timer_domain": (
                    "paired_differential_events_around_cuda_graph_replay"),
                "model": "warm_bf16_binary_tensor_iterator_task_floor_v1",
                "derivation_rule": (
                    "minimum_all_binary_graph_replay_differentials"),
            }
            # Each publishing card measures this task class on its own stack,
            # so the revision is pinned per stack rather than globally.  The
            # set stays closed: a revision nobody has measured is rejected.
            expected_one_of = {
                "software_revision": (
                    "torch_2.4.0_metax_3.7.2.0",
                    "torch_2.8.0+cu128_cuda_sm80",
                ),
            }
            range_names = ("minimum_elements", "maximum_elements")
            coefficient_names = ("duration_ns",)
        elif task_class == (
                PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS):
            common_expected = {
                "duration_unit": "ns_per_task",
                "timer_domain": "torch_cuda_event",
                "software_revision": "torch_2.10.0_metax_3.8.2.2",
            }
            fields = {
                "model", "duration_unit", "timer_domain",
                "software_revision", "derivation_rule",
                "minimum_grid_size", "maximum_grid_size",
                "minimum_block_size", "maximum_block_size",
                "submit_ns", "cta_descriptor_ns", "wave_round_ns",
            }
            expected = {
                **common_expected,
                "model": "warm_pytorch_custom_op_scheduler_v1",
                "derivation_rule": (
                    "least_squares_physical_scheduler_geometry"),
            }
            expected_one_of = {}
            range_names = (
                "minimum_grid_size", "maximum_grid_size",
                "minimum_block_size", "maximum_block_size")
            coefficient_names = (
                "submit_ns", "cta_descriptor_ns", "wave_round_ns")
        else:
            raise AssertionError(f"unhandled runtime task class: {task_class}")
        _require_exact_fields(model_owner, model, fields)
        for name, expected_value in expected.items():
            if model[name] != expected_value:
                raise ValueError(
                    f"{model_owner}.{name} must be {expected_value!r}")
        for name, allowed in expected_one_of.items():
            if model[name] not in allowed:
                raise ValueError(
                    f"{model_owner}.{name} must be one of "
                    f"{list(allowed)!r}")
        normalized = dict(expected)
        for name in expected_one_of:
            normalized[name] = model[name]
        for name in range_names:
            item = model[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{model_owner}.{name} must be an integer")
            if item <= 0:
                raise ValueError(f"{model_owner}.{name} must be positive")
            normalized[name] = item
        for minimum_name, maximum_name in zip(
                range_names[::2], range_names[1::2]):
            if normalized[maximum_name] < normalized[minimum_name]:
                raise ValueError(
                    f"{model_owner} {minimum_name.removeprefix('minimum_')} "
                    "range is invalid")
        for name in coefficient_names:
            item = model[name]
            if (isinstance(item, bool)
                    or not isinstance(item, (int, float))):
                raise TypeError(f"{model_owner}.{name} must be numeric")
            item = float(item)
            if not math.isfinite(item) or item <= 0:
                raise ValueError(
                    f"{model_owner}.{name} must be finite and positive")
            normalized[name] = item
        normalized_models[task_class] = normalized
    return normalized_models


def _normalize_vector_reduce_sum_latency_lut(value):
    if not isinstance(value, dict):
        raise TypeError(
            "scheduling.vector_reduce_sum_latency_lut must be a mapping")
    _require_exact_fields(
        "scheduling.vector_reduce_sum_latency_lut", value,
        {"element_counts", "duration_ns"})
    element_counts = value["element_counts"]
    durations = value["duration_ns"]
    if (not isinstance(element_counts, list)
            or not isinstance(durations, list)
            or len(element_counts) < 2
            or len(element_counts) != len(durations)):
        raise ValueError(
            "vector reduction latency counts/durations must be equal-length "
            "lists with at least two points")
    if any(isinstance(count, bool) or not isinstance(count, int)
           for count in element_counts):
        raise TypeError("vector reduction element counts must be integers")
    if (element_counts[0] <= 0
            or any(left >= right for left, right in zip(
                element_counts, element_counts[1:]))):
        raise ValueError(
            "vector reduction element counts must be positive and increase")
    normalized_durations = []
    for duration in durations:
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))):
            raise TypeError("vector reduction durations must be numeric")
        normalized = float(duration)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(
                "vector reduction durations must be finite and positive")
        normalized_durations.append(normalized)
    return tuple(element_counts), tuple(normalized_durations)


def _normalize_fia_cross_domain_three_handoff(value):
    """Validate the source-bound CANN FIA whole-program span mechanism."""

    owner = "scheduling.fia_cross_domain_three_handoff"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "task_kind",
        "aic_per_group", "aiv_per_group", "minimum_task_count",
        "maximum_task_count", "minimum_active_aic_groups",
        "maximum_active_aic_groups", "startup_local_ns",
        "startup_shared_ns_per_active_group", "steady_local_ns",
        "steady_shared_ns_per_active_group", "source_commit",
        "source_snapshot_sha256", "binary_sha256", "report_path",
        "report_sha256", "duration_consumed",
        "single_task_active_aic_groups", "single_task_duration_ns",
        "supplemental_single_task_evidence", "extended_single_task_points",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "cann_fia_1_2_three_handoff_v5",
        "duration_unit": "ns_per_aic_group_program",
        "software_revision": "cann_8.5.1",
        "task_kind": "mixed",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(
                f"{owner}.{name} must be {expected!r}")
    integer_fields = {
        "aic_per_group": 1,
        "aiv_per_group": 2,
        "minimum_task_count": 1,
        "maximum_task_count": 128,
        "minimum_active_aic_groups": 1,
        "maximum_active_aic_groups": 24,
    }
    for name, expected in integer_fields.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(
                f"{owner}.{name} must equal the frozen value {expected}")
    numeric_fields = (
        "startup_local_ns", "startup_shared_ns_per_active_group",
        "steady_local_ns", "steady_shared_ns_per_active_group",
    )
    normalized = dict(value)
    for name in numeric_fields:
        actual = value[name]
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        actual = float(actual)
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        normalized[name] = actual
    active_groups = value["single_task_active_aic_groups"]
    durations = value["single_task_duration_ns"]
    if not isinstance(active_groups, list) or not isinstance(durations, list):
        raise TypeError(
            f"{owner} single-task LUT axes must be lists")
    expected_active_groups = (
        1, 4, 6, 8, 11, 12, 16, 18, 19, 21, 22, 23, 24)
    if tuple(active_groups) != expected_active_groups:
        raise ValueError(
            f"{owner}.single_task_active_aic_groups changed the frozen "
            "exact domain")
    if len(durations) != len(active_groups):
        raise ValueError(f"{owner} single-task LUT axes disagree")
    normalized_durations = []
    for duration in durations:
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))):
            raise TypeError(
                f"{owner}.single_task_duration_ns must be numeric")
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"{owner}.single_task_duration_ns must be finite and "
                "positive")
        normalized_durations.append(duration)
    normalized["published_single_task_active_aic_groups"] = (
        expected_active_groups)
    normalized["published_single_task_duration_ns"] = tuple(
        normalized_durations)
    evidence_fields = {
        "active_aic_groups", "duration_ns", "source_commit",
        "source_snapshot_sha256", "binary_sha256", "report_path",
        "report_sha256", "production_or_evaluation_duration_consumed",
    }

    def _require_point_provenance(entry_owner, entry):
        """Check one independently provenanced single-task point."""

        _require_exact_fields(entry_owner, entry, evidence_fields)
        if entry["production_or_evaluation_duration_consumed"] is not False:
            raise ValueError(
                f"{entry_owner} cannot consume production duration")
        for name in (
                "source_snapshot_sha256", "binary_sha256", "report_sha256"):
            digest = entry[name]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(f"{entry_owner}.{name} must be SHA256 hex")
        entry_commit = entry["source_commit"]
        if (not isinstance(entry_commit, str)
                or len(entry_commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in entry_commit)):
            raise ValueError(
                f"{entry_owner}.source_commit must be a full Git commit")
        entry_report_path = entry["report_path"]
        if (not isinstance(entry_report_path, str)
                or not entry_report_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in entry_report_path.split("/")):
            raise ValueError(
                f"{entry_owner}.report_path is outside its evidence owner")

    supplemental = value["supplemental_single_task_evidence"]
    if not isinstance(supplemental, list) or len(supplemental) != 2:
        raise TypeError(
            f"{owner}.supplemental_single_task_evidence must contain the "
            "group-4 and group-6 evidence")
    expected_supplemental_points = {4: 960.0, 6: 1320.0}
    observed_supplemental_groups = set()
    for index, entry in enumerate(supplemental):
        entry_owner = f"{owner}.supplemental_single_task_evidence[{index}]"
        _require_point_provenance(entry_owner, entry)
        supplemental_groups = entry["active_aic_groups"]
        if (supplemental_groups not in expected_supplemental_points
                or entry["duration_ns"]
                != expected_supplemental_points[supplemental_groups]):
            raise ValueError(
                f"{entry_owner} must bind one measured supplemental point")
        if supplemental_groups in observed_supplemental_groups:
            raise ValueError(f"{entry_owner} duplicates an evidence point")
        observed_supplemental_groups.add(supplemental_groups)
    if observed_supplemental_groups != set(expected_supplemental_points):
        raise ValueError(
            f"{owner}.supplemental_single_task_evidence lost an evidence point")
    normalized["supplemental_single_task_evidence"] = tuple(
        dict(item) for item in supplemental)
    extended = value["extended_single_task_points"]
    expected_extended_groups = (3, 5, 7, 9, 10, 13, 14, 15, 17, 20)
    if not isinstance(extended, list) or len(extended) != len(
            expected_extended_groups):
        raise TypeError(
            f"{owner}.extended_single_task_points must carry the gap-fill "
            "campaign's measured points")
    extended_durations = {}
    for index, entry in enumerate(extended):
        entry_owner = f"{owner}.extended_single_task_points[{index}]"
        _require_point_provenance(entry_owner, entry)
        extended_groups = entry["active_aic_groups"]
        if (isinstance(extended_groups, bool)
                or not isinstance(extended_groups, int)):
            raise TypeError(
                f"{entry_owner}.active_aic_groups must be an integer")
        if extended_groups in expected_active_groups:
            raise ValueError(
                f"{entry_owner} restates a published campaign point")
        if extended_groups in extended_durations:
            raise ValueError(f"{entry_owner} duplicates an extension point")
        extended_duration = entry["duration_ns"]
        if (isinstance(extended_duration, bool)
                or not isinstance(extended_duration, (int, float))):
            raise TypeError(f"{entry_owner}.duration_ns must be numeric")
        extended_duration = float(extended_duration)
        if not math.isfinite(extended_duration) or extended_duration <= 0:
            raise ValueError(
                f"{entry_owner}.duration_ns must be finite and positive")
        extended_durations[extended_groups] = extended_duration
    if tuple(sorted(extended_durations)) != expected_extended_groups:
        raise ValueError(
            f"{owner}.extended_single_task_points changed the frozen "
            "extension domain")
    normalized["extended_single_task_points"] = tuple(
        dict(item) for item in extended)
    merged_points = dict(zip(expected_active_groups, normalized_durations))
    merged_points.update(extended_durations)
    merged_groups = tuple(sorted(merged_points))
    normalized["single_task_active_aic_groups"] = merged_groups
    normalized["single_task_duration_ns"] = tuple(
        merged_points[groups] for groups in merged_groups)
    for name in (
            "source_snapshot_sha256", "binary_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    commit = value["source_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in commit)):
        raise ValueError(f"{owner}.source_commit must be a full Git commit")
    report_path = value["report_path"]
    if (not isinstance(report_path, str)
            or not report_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in report_path.split("/")):
        raise ValueError(f"{owner}.report_path is outside its evidence owner")
    if value["duration_consumed"] is not False:
        raise ValueError(
            f"{owner}.duration_consumed must be false for production or "
            "evaluation durations")
    return normalized


def _normalize_fia_aiv_instruction_cold_fill(value):
    """Validate exact CANN FIA AIV instruction cold-fill points."""

    owner = "scheduling.fia_aiv_instruction_cold_fill"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    _require_exact_fields(owner, value, {
        "model", "duration_unit", "software_revision", "regular_points",
        "split_fuse", "evidence_ref",
        "production_or_evaluation_duration_consumed",
    })
    expected_text = {
        "model": "cann_fia_aiv_instruction_cold_fill_v1",
        "duration_unit": "ns_per_resident_aiv_program_start",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    _require_no_duration_consumption(value, owner)
    evidence_ref = value["evidence_ref"]
    if (not isinstance(evidence_ref, str)
            or not evidence_ref.startswith(
                "microbench/ascend_910b/references/")
            or ".." in evidence_ref.split("/")):
        raise ValueError(f"{owner}.evidence_ref is outside its owner")
    raw_points = value["regular_points"]
    if not isinstance(raw_points, list) or not raw_points:
        raise TypeError(f"{owner}.regular_points must be a non-empty list")
    points = []
    keys = set()
    for index, point in enumerate(raw_points):
        point_owner = f"{owner}.regular_points[{index}]"
        if not isinstance(point, dict):
            raise TypeError(f"{point_owner} must be a mapping")
        _require_exact_fields(point_owner, point, {
            "active_aic_groups", "source_s2", "head_dims", "duration_ns",
        })
        for name in ("active_aic_groups", "source_s2"):
            item = point[name]
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"{point_owner}.{name} must be an integer")
            if item <= 0:
                raise ValueError(f"{point_owner}.{name} must be positive")
        head_dims = point["head_dims"]
        if (not isinstance(head_dims, list) or not head_dims
                or any(isinstance(item, bool) or not isinstance(item, int)
                       or item <= 0 for item in head_dims)
                or tuple(sorted(set(head_dims))) != tuple(head_dims)):
            raise ValueError(
                f"{point_owner}.head_dims must be unique positive integers")
        duration_ns = point["duration_ns"]
        if (isinstance(duration_ns, bool)
                or not isinstance(duration_ns, (int, float))
                or not math.isfinite(float(duration_ns))
                or duration_ns <= 0):
            raise ValueError(
                f"{point_owner}.duration_ns must be finite and positive")
        normalized = dict(point)
        normalized["head_dims"] = tuple(head_dims)
        normalized["duration_ns"] = float(duration_ns)
        for head_dim in head_dims:
            key = (point["active_aic_groups"], point["source_s2"], head_dim)
            if key in keys:
                raise ValueError(
                    f"{owner}.regular_points contains duplicate key {key}")
            keys.add(key)
        points.append(normalized)
    split_fuse = value["split_fuse"]
    if not isinstance(split_fuse, dict):
        raise TypeError(f"{owner}.split_fuse must be a mapping")
    _require_exact_fields(f"{owner}.split_fuse", split_fuse, {
        "active_aic_groups", "aiv_per_group", "duration_ns_per_aiv",
    })
    for name in ("active_aic_groups", "aiv_per_group"):
        item = split_fuse[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(
                f"{owner}.split_fuse.{name} must be a positive integer")
    duration_ns = split_fuse["duration_ns_per_aiv"]
    if (isinstance(duration_ns, bool)
            or not isinstance(duration_ns, (int, float))
            or not math.isfinite(float(duration_ns)) or duration_ns <= 0):
        raise ValueError(
            f"{owner}.split_fuse.duration_ns_per_aiv must be finite and "
            "positive")
    normalized = dict(value)
    normalized["regular_points"] = tuple(points)
    normalized_split_fuse = dict(split_fuse)
    normalized_split_fuse["duration_ns_per_aiv"] = float(duration_ns)
    normalized["split_fuse"] = normalized_split_fuse
    return normalized


def _load_fia_aiv_instruction_cold_fill_reference(
        repository_root, reference_path, reference_sha256):
    """Load the immutable source-exact FIA instruction cold-fill evidence."""

    path = repository_root / reference_path
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference_sha256:
        raise ValueError(
            "FIA AIV instruction cold-fill reference SHA256 mismatch")
    reference = json.loads(payload)
    owner = "FIA AIV instruction cold-fill reference"
    _require_exact_fields(owner, reference, {
        "schema_version", "hardware_id", "software_revision",
        "mechanism_id", "scope",
        "production_or_evaluation_duration_consumed", "regular_points",
        "split_fuse", "provenance",
    })
    if (reference["schema_version"] != 1
            or reference["hardware_id"] != "ascend_910b"
            or reference["software_revision"] != "cann_8.5.1"
            or reference["mechanism_id"]
            != "cann_fia_aiv_instruction_cold_fill_v1"
            or reference["production_or_evaluation_duration_consumed"]
            is not False):
        raise ValueError(f"{owner} identity or provenance contract changed")
    provenance = reference["provenance"]
    _require_exact_fields(f"{owner}.provenance", provenance, {
        "source_commit", "regular_source_dir", "regular_kernel_sha256",
        "regular_host_sha256", "regular_supplement_commit",
        "regular_supplement_source_dir",
        "regular_supplement_kernel_sha256",
        "regular_supplement_host_sha256",
        "regular_supplement_raw_files", "split_fuse_source_dir",
        "split_fuse_kernel_sha256", "split_fuse_host_sha256",
    })
    for name in (
            "regular_kernel_sha256", "regular_host_sha256",
            "regular_supplement_kernel_sha256",
            "regular_supplement_host_sha256",
            "split_fuse_kernel_sha256", "split_fuse_host_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.provenance.{name} must be SHA256 hex")
    supplement_commit = provenance["regular_supplement_commit"]
    if (not isinstance(supplement_commit, str)
            or len(supplement_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in supplement_commit)):
        raise ValueError(
            f"{owner}.provenance.regular_supplement_commit must be a full "
            "Git commit")
    supplement_dir = provenance["regular_supplement_source_dir"]
    if (not isinstance(supplement_dir, str)
            or not supplement_dir.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in supplement_dir.split("/")):
        raise ValueError(
            f"{owner}.provenance.regular_supplement_source_dir is outside "
            "its evidence owner")
    supplement_raw_files = provenance["regular_supplement_raw_files"]
    if (not isinstance(supplement_raw_files, list)
            or len(supplement_raw_files) != 5):
        raise ValueError(
            f"{owner}.provenance.regular_supplement_raw_files must contain "
            "five fresh-process captures")
    raw_paths = set()
    for index, raw_file in enumerate(supplement_raw_files):
        raw_owner = (
            f"{owner}.provenance.regular_supplement_raw_files[{index}]")
        if not isinstance(raw_file, dict):
            raise TypeError(f"{raw_owner} must be a mapping")
        _require_exact_fields(raw_owner, raw_file, {"path", "sha256"})
        raw_path = raw_file["path"]
        if (not isinstance(raw_path, str)
                or not raw_path.startswith(
                    (f"{supplement_dir}/results/", f"{supplement_dir}/raw/"))
                or ".." in raw_path.split("/")
                or raw_path in raw_paths):
            raise ValueError(f"{raw_owner}.path is invalid")
        raw_paths.add(raw_path)
        digest = raw_file["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{raw_owner}.sha256 must be SHA256 hex")
    return _normalize_fia_aiv_instruction_cold_fill({
        "model": reference["mechanism_id"],
        "duration_unit": "ns_per_resident_aiv_program_start",
        "software_revision": reference["software_revision"],
        "regular_points": reference["regular_points"],
        "split_fuse": reference["split_fuse"],
        "evidence_ref": str(reference_path),
        "production_or_evaluation_duration_consumed": reference[
            "production_or_evaluation_duration_consumed"],
    })


def _load_fia_aiv_instruction_cold_fill(repository_root):
    """Merge non-overlapping exact FIA instruction cold-fill references."""

    base = _load_fia_aiv_instruction_cold_fill_reference(
        repository_root,
        _FIA_AIV_INSTRUCTION_COLD_FILL_REFERENCE,
        _FIA_AIV_INSTRUCTION_COLD_FILL_REFERENCE_SHA256,
    )
    supplements = (
        _load_fia_aiv_instruction_cold_fill_reference(
            repository_root,
            _FIA_AIV_INSTRUCTION_COLD_FILL_SUPPLEMENT_REFERENCE,
            _FIA_AIV_INSTRUCTION_COLD_FILL_SUPPLEMENT_REFERENCE_SHA256,
        ),
        _load_fia_aiv_instruction_cold_fill_reference(
            repository_root,
            _FIA_AIV_INSTRUCTION_COLD_FILL_G21_SUPPLEMENT_REFERENCE,
            _FIA_AIV_INSTRUCTION_COLD_FILL_G21_SUPPLEMENT_REFERENCE_SHA256,
        ),
        _load_fia_aiv_instruction_cold_fill_reference(
            repository_root,
            _FIA_AIV_INSTRUCTION_COLD_FILL_G11_S512_SUPPLEMENT_REFERENCE,
            _FIA_AIV_INSTRUCTION_COLD_FILL_G11_S512_SUPPLEMENT_REFERENCE_SHA256,
        ),
        _load_fia_aiv_instruction_cold_fill_reference(
            repository_root,
            _FIA_AIV_INSTRUCTION_COLD_FILL_G22_S512_SUPPLEMENT_REFERENCE,
            _FIA_AIV_INSTRUCTION_COLD_FILL_G22_S512_SUPPLEMENT_REFERENCE_SHA256,
        ),
    )
    merged_keys = {
        (point["active_aic_groups"], point["source_s2"], head_dim)
        for point in base["regular_points"]
        for head_dim in point["head_dims"]
    }
    merged_points = base["regular_points"]
    for supplement in supplements:
        for name in (
                "model", "duration_unit", "software_revision", "split_fuse",
                "production_or_evaluation_duration_consumed"):
            if supplement[name] != base[name]:
                raise ValueError(
                    "FIA AIV instruction cold-fill supplement changed the "
                    f"shared {name} contract")
        supplement_keys = {
            (point["active_aic_groups"], point["source_s2"], head_dim)
            for point in supplement["regular_points"]
            for head_dim in point["head_dims"]
        }
        duplicates = merged_keys.intersection(supplement_keys)
        if duplicates:
            raise ValueError(
                "FIA AIV instruction cold-fill references contain duplicate "
                f"exact points: {sorted(duplicates)}")
        merged_keys.update(supplement_keys)
        merged_points += supplement["regular_points"]
    return {
        **base,
        "regular_points": merged_points,
    }


def _normalize_fia_aiv_sync_all_service(value):
    """Validate the independent CANN FIA AIV-wide barrier mechanism."""

    owner = "scheduling.fia_aiv_sync_all_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "task_kind",
        "aiv_per_group", "minimum_active_aiv_workers",
        "maximum_active_aiv_workers", "startup_ns",
        "per_active_aiv_worker_ns", "production_source_sha256",
        "source_commit", "source_snapshot_sha256", "binary_sha256",
        "raw_path", "raw_sha256", "report_path", "report_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "aiv_sync_all_affine_v1",
        "duration_unit": "ns_per_complete_sync_all",
        "software_revision": "cann_8.5.1",
        "task_kind": "mixed",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "aiv_per_group": 2,
        "minimum_active_aiv_workers": 2,
        "maximum_active_aiv_workers": 48,
    }
    normalized = dict(value)
    for name, expected in expected_integers.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(
                f"{owner}.{name} must equal the frozen value {expected}")
    for name in ("startup_ns", "per_active_aiv_worker_ns"):
        actual = value[name]
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        actual = float(actual)
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        normalized[name] = actual
    for name in (
            "production_source_sha256", "source_snapshot_sha256",
            "binary_sha256", "raw_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    commit = value["source_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in commit)):
        raise ValueError(f"{owner}.source_commit must be a full Git commit")
    for name in ("raw_path", "report_path"):
        path = value[name]
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.{name} is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    return normalized


def _normalize_fia_flash_decode_terminal_prelude_service(value):
    """Validate the source-defined mixed-core FlashDecode prelude."""

    owner = "scheduling.fia_flash_decode_terminal_prelude_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "task_kind",
        "aic_per_group", "aiv_per_group", "minimum_active_aic_groups",
        "maximum_active_aic_groups", "icache_preload_count",
        "aiv_buffer_count", "aiv_buffer_bytes", "startup_ns",
        "per_active_aic_group_ns", "production_source_commit",
        "fia_kernel_sha256", "flash_decode_source_sha256", "source_commit",
        "source_snapshot_sha256", "binary_sha256", "raw_path",
        "raw_sha256", "report_path", "report_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "fia_flash_decode_terminal_prelude_affine_v1",
        "duration_unit": "ns_per_complete_terminal_prelude",
        "software_revision": "cann_8.5.1",
        "task_kind": "mixed",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "aic_per_group": 1,
        "aiv_per_group": 2,
        "minimum_active_aic_groups": 1,
        "maximum_active_aic_groups": 24,
        "icache_preload_count": 2,
        "aiv_buffer_count": 14,
        "aiv_buffer_bytes": 97536,
    }
    normalized = dict(value)
    for name, expected in expected_integers.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(
                f"{owner}.{name} must equal the source value {expected}")
    for name in ("startup_ns", "per_active_aic_group_ns"):
        actual = value[name]
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        actual = float(actual)
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        normalized[name] = actual
    for name in (
            "fia_kernel_sha256", "flash_decode_source_sha256",
            "source_snapshot_sha256", "binary_sha256", "raw_sha256",
            "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    for name in ("production_source_commit", "source_commit"):
        commit = value[name]
        if (not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(f"{owner}.{name} must be a full Git commit")
    for name in ("raw_path", "report_path"):
        path = value[name]
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.{name} is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    return normalized


def _normalize_fia_tiling_data_materialization_service(value):
    """Validate the independent mixed-core FIA tiling prelude."""

    owner = "scheduling.fia_tiling_data_materialization_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "task_kind",
        "payload_bytes", "aiv_transfer_bytes", "aiv_per_group",
        "minimum_active_aic_groups", "maximum_active_aic_groups",
        "aic_completion_ns", "aiv_startup_ns",
        "aiv_per_active_worker_ns", "macro_generator_sha256",
        "fia_entry_sha256", "fia_kernel_sha256", "source_commit",
        "source_snapshot_sha256", "binary_sha256", "raw_path",
        "raw_sha256", "report_path", "report_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "fia_mixed_tiling_data_materialization_v1",
        "duration_unit": "ns_per_complete_actor_local_tiling_prelude",
        "software_revision": "cann_8.5.1",
        "task_kind": "mixed",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "payload_bytes": 1560,
        "aiv_transfer_bytes": 1568,
        "aiv_per_group": 2,
        "minimum_active_aic_groups": 1,
        "maximum_active_aic_groups": 24,
    }
    normalized = dict(value)
    for name, expected in expected_integers.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(
                f"{owner}.{name} must equal the frozen value {expected}")
    for name in (
            "aic_completion_ns", "aiv_startup_ns",
            "aiv_per_active_worker_ns"):
        actual = value[name]
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        actual = float(actual)
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        normalized[name] = actual
    for name in (
            "macro_generator_sha256", "fia_entry_sha256",
            "fia_kernel_sha256", "source_snapshot_sha256",
            "binary_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    raw_digests = value["raw_sha256"]
    if (not isinstance(raw_digests, list) or len(raw_digests) != 3
            or any(not isinstance(digest, str) or len(digest) != 64
                   or any(character not in "0123456789abcdef"
                          for character in digest)
                   for digest in raw_digests)):
        raise ValueError(f"{owner}.raw_sha256 must contain three SHA256s")
    normalized["raw_sha256"] = tuple(raw_digests)
    commit = value["source_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in commit)):
        raise ValueError(f"{owner}.source_commit must be a full Git commit")
    for name in ("raw_path", "report_path"):
        path = value[name]
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.{name} is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    return normalized


def _normalize_fia_mixed_task_outer_envelope(value):
    """Validate the independent FIA mixed-task outer envelope."""

    owner = "scheduling.fia_mixed_task_outer_envelope"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "timer_domain", "software_revision",
        "task_kind", "aiv_per_group", "minimum_active_aiv_workers",
        "maximum_active_aiv_workers", "startup_ns",
        "per_active_aiv_worker_ns", "production_source_sha256",
        "source_commit", "source_snapshot_sha256", "binary_sha256",
        "raw_path", "raw_sha256", "report_path", "report_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "fia_mixed_task_outer_envelope_affine_v1",
        "duration_unit": "ns_per_complete_mixed_task_outer_envelope",
        "timer_domain": "aclrt_event_minus_internal_system_cycle",
        "software_revision": "cann_8.5.1",
        "task_kind": "mixed",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "aiv_per_group": 2,
        "minimum_active_aiv_workers": 2,
        "maximum_active_aiv_workers": 48,
    }
    normalized = dict(value)
    for name, expected in expected_integers.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(
                f"{owner}.{name} must equal the frozen value {expected}")
    for name in ("startup_ns", "per_active_aiv_worker_ns"):
        actual = value[name]
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        actual = float(actual)
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        normalized[name] = actual
    for name in (
            "production_source_sha256", "source_snapshot_sha256",
            "binary_sha256", "raw_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    commit = value["source_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in commit)):
        raise ValueError(f"{owner}.source_commit must be a full Git commit")
    for name in ("raw_path", "report_path"):
        path = value[name]
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.{name} is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    return normalized


def _normalize_fia_aiv_workspace_transfer_service(value):
    """Validate source-shaped FIA AIV transfer-completion evidence."""

    owner = "scheduling.fia_aiv_workspace_transfer_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "task_kind",
        "active_aic_groups", "completion_ns_by_mode", "source_commit",
        "source_snapshot_sha256", "raw_path", "raw_sha256",
        "manifest_path", "manifest_sha256", "result_path", "result_sha256",
        "candidate_path", "candidate_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "fia_aiv_workspace_transfer_completion_v4",
        "duration_unit": "ns_per_source_task_critical_completion",
        "software_revision": "cann_8.5.1",
        "task_kind": "mixed",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    active_groups = value["active_aic_groups"]
    if active_groups != [8, 16]:
        raise ValueError(f"{owner}.active_aic_groups must equal [8, 16]")
    modes = {
        "vec1_m4_s512_ordered_chain",
        "vec1_m4_s1024_ordered_chain",
        "vec2_m4_final_ordered_chain",
    }
    completion = value["completion_ns_by_mode"]
    if not isinstance(completion, dict) or set(completion) != modes:
        raise ValueError(f"{owner}.completion_ns_by_mode has invalid modes")
    normalized = dict(value)
    normalized_completion = {}
    for mode in sorted(modes):
        by_group = completion[mode]
        if not isinstance(by_group, dict) or set(by_group) != {"8", "16"}:
            raise ValueError(
                f"{owner}.completion_ns_by_mode.{mode} has invalid groups")
        normalized_by_group = {}
        for group in active_groups:
            duration = by_group[str(group)]
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))):
                raise TypeError(
                    f"{owner}.completion_ns_by_mode.{mode}.{group} must be "
                    "numeric")
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"{owner}.completion_ns_by_mode.{mode}.{group} must be "
                    "finite and positive")
            normalized_by_group[str(group)] = duration
        normalized_completion[mode] = normalized_by_group
    normalized["completion_ns_by_mode"] = normalized_completion
    for name in (
            "source_snapshot_sha256", "raw_sha256", "manifest_sha256",
            "result_sha256", "candidate_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    commit = value["source_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in commit)):
        raise ValueError(f"{owner}.source_commit must be a full Git commit")
    for name in ("raw_path", "manifest_path", "result_path", "candidate_path"):
        path = value[name]
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.{name} is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    return normalized


def _normalize_aic_mte2_nd2nz_service(value):
    """Validate an independently measured AIC GM-to-L1 ND2NZ service."""

    owner = "scheduling.aic_mte2_nd2nz_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision",
        "descriptor_bytes", "active_aic_groups",
        "duration_ns_by_active_aic_group", "source_snapshot_sha256",
        "stream_descriptor_bytes",
        "stream_duration_ns_by_active_aic_group",
        "raw_path", "raw_sha256", "geometry_validation_raw_path",
        "geometry_validation_raw_sha256",
        "small_descriptor_raw_path", "small_descriptor_raw_sha256",
        "strided_active_aic_groups", "strided_geometries",
        "strided_source_snapshot_sha256", "strided_raw_path",
        "strided_raw_sha256",
        "cold_exact_geometries", "cold_exact_raw_path",
        "cold_exact_raw_sha256", "cold_exact_binary_sha256",
        "cold_exact_source_commit",
        "cold_completion_model", "cold_completion_active_aic_groups",
        "cold_completion_geometries",
        "cold_completion_minimum_extra_ns",
        "cold_completion_bandwidth_bytes_per_s",
        "cold_completion_resident_raw_files",
        "cold_completion_cold_raw_files",
        "cold_completion_label_correction_path",
        "cold_completion_label_correction_sha256",
        "cold_completion_measurement_baseline_commit",
        "cold_completion_source_commit",
        "cold_completion_source_kernel_sha256",
        "cold_completion_source_main_sha256",
        "cold_completion_source_runner_sha256",
        "cold_completion_binary_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "aic_gm_to_l1_nd2nz_serial_and_stream_v2",
        "duration_unit": "ns_per_descriptor_per_aic",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    descriptor_bytes = value["descriptor_bytes"]
    active_groups = value["active_aic_groups"]
    if (not isinstance(descriptor_bytes, list)
            or not isinstance(active_groups, list)):
        raise TypeError(f"{owner} axes must be lists")
    if tuple(descriptor_bytes) != (
            32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
            16384, 32768, 65536):
        raise ValueError(f"{owner}.descriptor_bytes changed its frozen domain")
    if tuple(active_groups) != (1, 8, 24):
        raise ValueError(f"{owner}.active_aic_groups changed its frozen domain")
    curves = value["duration_ns_by_active_aic_group"]
    if not isinstance(curves, dict) or set(curves) != set(active_groups):
        raise ValueError(f"{owner} duration curves must match active AIC groups")
    normalized_curves = {}
    for active_group in active_groups:
        durations = curves[active_group]
        if (not isinstance(durations, list)
                or len(durations) != len(descriptor_bytes)):
            raise ValueError(
                f"{owner} duration curve for {active_group} AICs has the "
                "wrong length")
        normalized_durations = []
        for duration in durations:
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))):
                raise TypeError(f"{owner} durations must be numeric")
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"{owner} durations must be finite and positive")
            normalized_durations.append(duration)
        normalized_curves[active_group] = tuple(normalized_durations)
    stream_descriptor_bytes = value["stream_descriptor_bytes"]
    if (not isinstance(stream_descriptor_bytes, list)
            or tuple(stream_descriptor_bytes) != (
                512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)):
        raise ValueError(
            f"{owner}.stream_descriptor_bytes changed its frozen domain")
    stream_curves = value["stream_duration_ns_by_active_aic_group"]
    if (not isinstance(stream_curves, dict)
            or set(stream_curves) != set(active_groups)):
        raise ValueError(
            f"{owner} stream duration curves must match active AIC groups")
    normalized_stream_curves = {}
    for active_group in active_groups:
        durations = stream_curves[active_group]
        if (not isinstance(durations, list)
                or len(durations) != len(stream_descriptor_bytes)):
            raise ValueError(
                f"{owner} stream duration curve for {active_group} AICs "
                "has the wrong length")
        normalized_durations = []
        for duration in durations:
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))):
                raise TypeError(f"{owner} stream durations must be numeric")
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"{owner} stream durations must be finite and positive")
            normalized_durations.append(duration)
        normalized_stream_curves[active_group] = tuple(
            normalized_durations)
    for name in (
            "source_snapshot_sha256", "raw_sha256",
            "geometry_validation_raw_sha256",
            "small_descriptor_raw_sha256",
            "strided_source_snapshot_sha256", "strided_raw_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    for name in (
            "raw_path", "geometry_validation_raw_path",
            "small_descriptor_raw_path", "strided_raw_path"):
        path = value[name]
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.{name} is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    strided_groups = value["strided_active_aic_groups"]
    if (not isinstance(strided_groups, list)
            or tuple(strided_groups) != (8, 16, 24)):
        raise ValueError(
            f"{owner}.strided_active_aic_groups changed its frozen domain")
    strided_geometries = value["strided_geometries"]
    if not isinstance(strided_geometries, list) or not strided_geometries:
        raise TypeError(f"{owner}.strided_geometries must be a non-empty list")
    normalized_strided = {}
    strided_fields = {
        "rows", "columns", "source_row_stride_elements",
        "serial_duration_ns", "stream_duration_ns",
    }
    for index, geometry in enumerate(strided_geometries):
        geometry_owner = f"{owner}.strided_geometries[{index}]"
        if not isinstance(geometry, dict):
            raise TypeError(f"{geometry_owner} must be a mapping")
        _require_exact_fields(geometry_owner, geometry, strided_fields)
        dimensions = tuple(
            geometry[name]
            for name in ("rows", "columns", "source_row_stride_elements"))
        if any(
                isinstance(item, bool) or not isinstance(item, int)
                or item <= 0 for item in dimensions):
            raise ValueError(
                f"{geometry_owner} dimensions must be positive integers")
        if dimensions[2] <= dimensions[1]:
            raise ValueError(
                f"{geometry_owner} must describe a genuinely strided copy")
        if dimensions in normalized_strided:
            raise ValueError(
                f"{owner}.strided_geometries contains a duplicate geometry")
        duration_curves = {}
        for duration_name in ("serial_duration_ns", "stream_duration_ns"):
            durations = geometry[duration_name]
            if (not isinstance(durations, list)
                    or len(durations) != len(strided_groups)):
                raise ValueError(
                    f"{geometry_owner}.{duration_name} must match the "
                    "strided active-group axis")
            normalized_durations = []
            for duration in durations:
                if (isinstance(duration, bool)
                        or not isinstance(duration, (int, float))):
                    raise TypeError(
                        f"{geometry_owner}.{duration_name} must be numeric")
                duration = float(duration)
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(
                        f"{geometry_owner}.{duration_name} must be finite "
                        "and positive")
                normalized_durations.append(duration)
            duration_curves[duration_name] = tuple(normalized_durations)
        normalized_strided[dimensions] = duration_curves
    cold_exact_geometries = value["cold_exact_geometries"]
    if not isinstance(cold_exact_geometries, list) or not cold_exact_geometries:
        raise TypeError(
            f"{owner}.cold_exact_geometries must be a non-empty list")
    cold_exact_fields = {
        "rows", "columns", "source_row_stride_elements",
        "active_aic_groups", "serial_duration_ns", "stream_duration_ns",
    }
    normalized_cold_exact = {}
    for index, geometry in enumerate(cold_exact_geometries):
        geometry_owner = f"{owner}.cold_exact_geometries[{index}]"
        if not isinstance(geometry, dict):
            raise TypeError(f"{geometry_owner} must be a mapping")
        _require_exact_fields(geometry_owner, geometry, cold_exact_fields)
        key = tuple(
            geometry[name]
            for name in (
                "rows", "columns", "source_row_stride_elements",
                "active_aic_groups"))
        if any(
                isinstance(item, bool) or not isinstance(item, int)
                or item <= 0 for item in key):
            raise ValueError(
                f"{geometry_owner} key must be positive integers")
        if key[2] < key[1]:
            raise ValueError(
                f"{geometry_owner} source stride is smaller than columns")
        if key in normalized_cold_exact:
            raise ValueError(
                f"{owner}.cold_exact_geometries repeats a geometry")
        durations = {}
        for duration_name in ("serial_duration_ns", "stream_duration_ns"):
            duration = geometry[duration_name]
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))):
                raise TypeError(
                    f"{geometry_owner}.{duration_name} must be numeric")
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(
                    f"{geometry_owner}.{duration_name} must be finite and "
                    "positive")
            durations[duration_name] = duration
        normalized_cold_exact[key] = durations
    cold_exact_path = value["cold_exact_raw_path"]
    if (not isinstance(cold_exact_path, str)
            or not cold_exact_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in cold_exact_path.split("/")):
        raise ValueError(f"{owner}.cold_exact_raw_path is outside its owner")
    for name in ("cold_exact_raw_sha256", "cold_exact_binary_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    cold_exact_commit = value["cold_exact_source_commit"]
    if (not isinstance(cold_exact_commit, str)
            or len(cold_exact_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in cold_exact_commit)):
        raise ValueError(
            f"{owner}.cold_exact_source_commit must be a full Git commit")
    cold_model = value["cold_completion_model"]
    if cold_model != "resident_plus_predicted_dram_miss_latency_bandwidth_v1":
        raise ValueError(
            f"{owner}.cold_completion_model has an unsupported value")
    cold_active_groups = value["cold_completion_active_aic_groups"]
    if (isinstance(cold_active_groups, bool)
            or not isinstance(cold_active_groups, int)
            or cold_active_groups != 22):
        raise ValueError(
            f"{owner}.cold_completion_active_aic_groups must be 22")
    cold_geometries = value["cold_completion_geometries"]
    if not isinstance(cold_geometries, list) or not cold_geometries:
        raise TypeError(
            f"{owner}.cold_completion_geometries must be a non-empty list")
    cold_geometry_fields = {
        "rows", "columns", "source_row_stride_elements",
        "resident_duration_ns", "cold_validation_duration_ns",
    }
    normalized_cold_geometries = {}
    for index, geometry in enumerate(cold_geometries):
        geometry_owner = f"{owner}.cold_completion_geometries[{index}]"
        if not isinstance(geometry, dict):
            raise TypeError(f"{geometry_owner} must be a mapping")
        _require_exact_fields(
            geometry_owner, geometry, cold_geometry_fields)
        dimensions = tuple(
            geometry[name]
            for name in ("rows", "columns", "source_row_stride_elements"))
        if any(
                isinstance(item, bool) or not isinstance(item, int)
                or item <= 0 for item in dimensions):
            raise ValueError(
                f"{geometry_owner} dimensions must be positive integers")
        if dimensions[2] < dimensions[1]:
            raise ValueError(
                f"{geometry_owner} source stride is smaller than columns")
        if dimensions in normalized_cold_geometries:
            raise ValueError(
                f"{owner}.cold_completion_geometries repeats a geometry")
        durations = {}
        for duration_name in (
                "resident_duration_ns", "cold_validation_duration_ns"):
            duration = geometry[duration_name]
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))):
                raise TypeError(
                    f"{geometry_owner}.{duration_name} must be numeric")
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(
                    f"{geometry_owner}.{duration_name} must be finite and "
                    "positive")
            durations[duration_name] = duration
        if (durations["cold_validation_duration_ns"]
                <= durations["resident_duration_ns"]):
            raise ValueError(
                f"{geometry_owner} cold completion must exceed resident")
        normalized_cold_geometries[dimensions] = durations
    expected_cold_geometries = {
        (4, 64, 64),
        (256, 64, 512),
        (4, 256, 512),
    }
    if set(normalized_cold_geometries) != expected_cold_geometries:
        raise ValueError(
            f"{owner}.cold_completion_geometries changed its exact domain")
    minimum_extra_ns = value["cold_completion_minimum_extra_ns"]
    bandwidth_bytes_per_s = value[
        "cold_completion_bandwidth_bytes_per_s"]
    for field_name, field_value in (
            ("cold_completion_minimum_extra_ns", minimum_extra_ns),
            ("cold_completion_bandwidth_bytes_per_s",
             bandwidth_bytes_per_s)):
        if (isinstance(field_value, bool)
                or not isinstance(field_value, (int, float))):
            raise TypeError(f"{owner}.{field_name} must be numeric")
        if not math.isfinite(field_value) or field_value <= 0.0:
            raise ValueError(
                f"{owner}.{field_name} must be finite and positive")
    minimum_extra_ns = float(minimum_extra_ns)
    bandwidth_bytes_per_s = float(bandwidth_bytes_per_s)
    q_geometry = normalized_cold_geometries[(4, 64, 64)]
    measured_q_extra_ns = (
        q_geometry["cold_validation_duration_ns"]
        - q_geometry["resident_duration_ns"])
    if not math.isclose(
            minimum_extra_ns, measured_q_extra_ns,
            rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"{owner}.cold_completion_minimum_extra_ns must come from Q")
    kv_geometry = normalized_cold_geometries[(256, 64, 512)]
    measured_kv_extra_ns = (
        kv_geometry["cold_validation_duration_ns"]
        - kv_geometry["resident_duration_ns"])
    derived_bandwidth = (
        256 * 64 * 2 / (measured_kv_extra_ns * 1.0e-9))
    if not math.isclose(
            bandwidth_bytes_per_s, derived_bandwidth,
            rel_tol=1.0e-12, abs_tol=0.0):
        raise ValueError(
            f"{owner}.cold_completion_bandwidth_bytes_per_s must come from "
            "K/V")
    p_geometry = normalized_cold_geometries[(4, 256, 512)]
    predicted_p_cold_ns = math.fsum((
        p_geometry["resident_duration_ns"],
        max(
            minimum_extra_ns,
            4 * 256 * 2 / bandwidth_bytes_per_s * 1.0e9),
    ))
    if not math.isclose(
            predicted_p_cold_ns,
            p_geometry["cold_validation_duration_ns"],
            rel_tol=1.0e-4, abs_tol=0.0):
        raise ValueError(
            f"{owner} cold-completion mechanism fails its P holdout")
    normalized_raw_files = {}
    for file_class in (
            "cold_completion_resident_raw_files",
            "cold_completion_cold_raw_files"):
        raw_files = value[file_class]
        if not isinstance(raw_files, list) or len(raw_files) != 3:
            raise ValueError(f"{owner}.{file_class} must contain three files")
        normalized_files = []
        for index, raw_file in enumerate(raw_files):
            file_owner = f"{owner}.{file_class}[{index}]"
            if not isinstance(raw_file, dict):
                raise TypeError(f"{file_owner} must be a mapping")
            _require_exact_fields(file_owner, raw_file, {"path", "sha256"})
            path = raw_file["path"]
            if (not isinstance(path, str)
                    or not path.startswith(
                        "profiler_output/microbench/ascend_910b/")
                    or ".." in path.split("/")):
                raise ValueError(f"{file_owner}.path is outside its owner")
            digest = raw_file["sha256"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(f"{file_owner}.sha256 must be SHA256 hex")
            normalized_files.append({"path": path, "sha256": digest})
        normalized_raw_files[file_class] = tuple(normalized_files)
    correction_path = value["cold_completion_label_correction_path"]
    if (not isinstance(correction_path, str)
            or not correction_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in correction_path.split("/")):
        raise ValueError(
            f"{owner}.cold_completion_label_correction_path is outside its "
            "owner")
    for name in (
            "cold_completion_label_correction_sha256",
            "cold_completion_source_kernel_sha256",
            "cold_completion_source_main_sha256",
            "cold_completion_source_runner_sha256",
            "cold_completion_binary_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    for name in (
            "cold_completion_measurement_baseline_commit",
            "cold_completion_source_commit"):
        commit = value[name]
        if (not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(f"{owner}.{name} must be a full Git commit")
    return {
        **expected_text,
        "descriptor_bytes": tuple(descriptor_bytes),
        "active_aic_groups": tuple(active_groups),
        "duration_ns_by_active_aic_group": normalized_curves,
        "stream_descriptor_bytes": tuple(stream_descriptor_bytes),
        "stream_duration_ns_by_active_aic_group": normalized_stream_curves,
        "source_snapshot_sha256": value["source_snapshot_sha256"],
        "raw_path": value["raw_path"],
        "raw_sha256": value["raw_sha256"],
        "geometry_validation_raw_path": value[
            "geometry_validation_raw_path"],
        "geometry_validation_raw_sha256": value[
            "geometry_validation_raw_sha256"],
        "small_descriptor_raw_path": value["small_descriptor_raw_path"],
        "small_descriptor_raw_sha256": value[
            "small_descriptor_raw_sha256"],
        "strided_active_aic_groups": tuple(strided_groups),
        "strided_active_aic_group_set": frozenset(strided_groups),
        "strided_geometries": normalized_strided,
        "strided_source_snapshot_sha256": value[
            "strided_source_snapshot_sha256"],
        "strided_raw_path": value["strided_raw_path"],
        "strided_raw_sha256": value["strided_raw_sha256"],
        "cold_exact_geometries": normalized_cold_exact,
        "cold_exact_raw_path": value["cold_exact_raw_path"],
        "cold_exact_raw_sha256": value["cold_exact_raw_sha256"],
        "cold_exact_binary_sha256": value["cold_exact_binary_sha256"],
        "cold_exact_source_commit": value["cold_exact_source_commit"],
        "cold_completion_model": cold_model,
        "cold_completion_active_aic_groups": cold_active_groups,
        "cold_completion_geometries": normalized_cold_geometries,
        "cold_completion_minimum_extra_ns": minimum_extra_ns,
        "cold_completion_bandwidth_bytes_per_s": bandwidth_bytes_per_s,
        **normalized_raw_files,
        "cold_completion_label_correction_path": correction_path,
        "cold_completion_label_correction_sha256": value[
            "cold_completion_label_correction_sha256"],
        "cold_completion_measurement_baseline_commit": value[
            "cold_completion_measurement_baseline_commit"],
        "cold_completion_source_commit": value[
            "cold_completion_source_commit"],
        "cold_completion_source_kernel_sha256": value[
            "cold_completion_source_kernel_sha256"],
        "cold_completion_source_main_sha256": value[
            "cold_completion_source_main_sha256"],
        "cold_completion_source_runner_sha256": value[
            "cold_completion_source_runner_sha256"],
        "cold_completion_binary_sha256": value[
            "cold_completion_binary_sha256"],
        "production_or_evaluation_duration_consumed": False,
    }


def _normalize_aic_mte2_operand_pair_stream_service(value):
    """Validate one source-shaped, L2-resident A/B MTE2 stream service."""

    owner = "scheduling.aic_mte2_operand_pair_stream_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "dtype",
        "source_residency", "a_rows", "a_columns", "b_rows",
        "b_columns", "active_aic_groups", "minimum_pairs_per_worker",
        "maximum_pairs_per_worker", "continuous_ns_per_pair",
        "source_commit", "source_snapshot_sha256", "binary_sha256",
        "manifest_path", "manifest_sha256", "raw_paths", "raw_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "aic_l2_to_l1_nd2nz_operand_pair_stream_v1",
        "duration_unit": "ns_per_a_b_operand_pair_per_aic",
        "software_revision": "cann_8.5.1",
        "dtype": "bf16",
        "source_residency": "l2_resident",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "a_rows": 128,
        "a_columns": 64,
        "b_rows": 64,
        "b_columns": 256,
        "active_aic_groups": 24,
        "minimum_pairs_per_worker": 1024,
        "maximum_pairs_per_worker": 4096,
    }
    normalized = dict(expected_text)
    for name, expected in expected_integers.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
        normalized[name] = actual
    duration = value["continuous_ns_per_pair"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise TypeError(f"{owner}.continuous_ns_per_pair must be numeric")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"{owner}.continuous_ns_per_pair must be finite and positive")
    normalized["continuous_ns_per_pair"] = duration
    for name in (
            "source_snapshot_sha256", "binary_sha256", "manifest_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
        normalized[name] = digest
    commit = value["source_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError(f"{owner}.source_commit must be a full Git commit")
    normalized["source_commit"] = commit
    manifest_path = value["manifest_path"]
    if (not isinstance(manifest_path, str)
            or not manifest_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in manifest_path.split("/")):
        raise ValueError(f"{owner}.manifest_path is outside its evidence owner")
    normalized["manifest_path"] = manifest_path
    raw_paths = value["raw_paths"]
    raw_sha256 = value["raw_sha256"]
    if (not isinstance(raw_paths, list) or not isinstance(raw_sha256, list)
            or len(raw_paths) != 3 or len(raw_sha256) != 3):
        raise ValueError(f"{owner} requires exactly three independent raw runs")
    normalized_paths = []
    normalized_hashes = []
    for index, (path, digest) in enumerate(zip(raw_paths, raw_sha256)):
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.raw_paths[{index}] is outside its evidence owner")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.raw_sha256[{index}] must be SHA256 hex")
        normalized_paths.append(path)
        normalized_hashes.append(digest)
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError(f"{owner}.raw_paths must name independent files")
    normalized["raw_paths"] = tuple(normalized_paths)
    normalized["raw_sha256"] = tuple(normalized_hashes)
    _require_no_duration_consumption(value, owner)
    normalized["production_or_evaluation_duration_consumed"] = False
    return normalized


def _normalize_aic_fixpipe_full_tile_service(value):
    """Validate completed full-width FIA Fixpipe primitive service."""

    owner = "scheduling.aic_fixpipe_full_tile_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "dtype", "layout",
        "column_count", "supported_rows",
        "large_row_minimum_rows",
        "destination_stride_alignment_elements",
        "maximum_destination_stride_elements", "call_startup_ns",
        "row_service_ns", "large_row_call_startup_ns",
        "large_row_service_ns", "row_address_phase_2_penalty_ns",
        "row_address_phase_4_penalty_ns", "source_snapshot_sha256",
        "raw_path", "raw_sha256", "report_path", "report_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "aic_l0c_to_gm_fixpipe_full_tile_v2",
        "duration_unit": "ns_per_completed_call_per_aic",
        "software_revision": "cann_8.5.1",
        "dtype": "fp32",
        "layout": "row_major",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "column_count": 128,
        "large_row_minimum_rows": 16,
        "destination_stride_alignment_elements": 32,
        "maximum_destination_stride_elements": 8192,
    }
    for name, expected in expected_integers.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(
                f"{owner}.{name} must equal the frozen value {expected}")
    supported_rows = value["supported_rows"]
    if not isinstance(supported_rows, list):
        raise TypeError(f"{owner}.supported_rows must be a list")
    if tuple(supported_rows) != (1, 4, 5, 8, 64):
        raise ValueError(f"{owner}.supported_rows changed its frozen domain")
    normalized = dict(value)
    for name in (
            "call_startup_ns", "row_service_ns",
            "large_row_call_startup_ns", "large_row_service_ns",
            "row_address_phase_2_penalty_ns",
            "row_address_phase_4_penalty_ns"):
        actual = value[name]
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        actual = float(actual)
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        normalized[name] = actual
    for name in (
            "source_snapshot_sha256", "raw_sha256", "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    for name in ("raw_path", "report_path"):
        path = value[name]
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.{name} is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    normalized["supported_rows"] = tuple(supported_rows)
    return normalized


def _normalize_aic_mte1_load_pair_completion_service(value):
    """Validate completed source-shaped AIC L1-to-L0 load pairs."""

    owner = "scheduling.aic_mte1_load_pair_completion_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "dtype",
        "a_rows", "stage_kinds", "head_dims", "mm1_b_load_calls",
        "mm1_b_repeat_times", "mm2_b_load_calls", "mm2_b_repeat_times",
        "completion_ns_by_stage", "active_aic_groups",
        "production_source_sha256",
        "measurement_baseline_commit", "source_commit",
        "source_snapshot_sha256", "binary_sha256", "raw_path",
        "raw_sha256", "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "aic_l1_to_l0_fia_load_pair_completion_v1",
        "duration_unit": "ns_per_completed_load_pair_per_aic",
        "software_revision": "cann_8.5.1",
        "dtype": "bf16",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    if value["a_rows"] != 16:
        raise ValueError(f"{owner}.a_rows must equal 16")
    if tuple(value["stage_kinds"]) != ("mm1", "mm2"):
        raise ValueError(f"{owner}.stage_kinds changed its frozen domain")
    if tuple(value["head_dims"]) != (64, 128):
        raise ValueError(f"{owner}.head_dims changed its frozen domain")
    expected_geometry = {
        "mm1_b_load_calls": (4, 8),
        "mm1_b_repeat_times": (8, 8),
        "mm2_b_load_calls": (8, 8),
        "mm2_b_repeat_times": (4, 8),
        "active_aic_groups": (1, 8, 16, 24),
    }
    normalized = dict(value)
    for name, expected in expected_geometry.items():
        actual = value[name]
        if not isinstance(actual, list) or tuple(actual) != expected:
            raise ValueError(f"{owner}.{name} changed its frozen domain")
        normalized[name] = expected
    curves = value["completion_ns_by_stage"]
    if (not isinstance(curves, dict)
            or set(curves) != set(value["stage_kinds"])):
        raise ValueError(f"{owner} completion curves must match stage kinds")
    normalized_curves = {}
    for stage_kind in value["stage_kinds"]:
        durations = curves[stage_kind]
        if (not isinstance(durations, list)
                or len(durations) != len(value["head_dims"])):
            raise ValueError(
                f"{owner} completion curve for {stage_kind} has the wrong "
                "length")
        normalized_durations = []
        for duration in durations:
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))):
                raise TypeError(f"{owner} completion durations must be numeric")
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"{owner} completion durations must be finite and positive")
            normalized_durations.append(duration)
        normalized_curves[stage_kind] = tuple(normalized_durations)
    normalized["completion_ns_by_stage"] = normalized_curves
    for name in (
            "production_source_sha256", "source_snapshot_sha256",
            "binary_sha256", "raw_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    for name in ("measurement_baseline_commit", "source_commit"):
        commit = value[name]
        if (not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(f"{owner}.{name} must be a full Git commit")
    raw_path = value["raw_path"]
    if (not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")):
        raise ValueError(f"{owner}.raw_path is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    normalized["stage_kinds"] = tuple(value["stage_kinds"])
    normalized["head_dims"] = tuple(value["head_dims"])
    return normalized


def _normalize_aic_grouped_matmul_mte1_pair_completion_service(value):
    """Validate completed source-shaped GroupedMatmul L1-to-L0 pairs."""

    owner = "scheduling.aic_grouped_matmul_mte1_pair_completion_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "dtype",
        "m_rows", "n_columns", "k_columns", "a_load_calls",
        "a_repeat_times", "b_load_calls", "b_repeat_times",
        "b_l1_k_extent", "completion_ns_by_m_rows", "active_aic_groups",
        "minimum_n_columns", "maximum_n_columns", "n_column_granularity",
        "n_column_service_ns", "n_geometry_reference_path",
        "n_geometry_reference_sha256",
        "scheduler_source_sha256", "load_to_l0a_source_sha256",
        "load_to_l0b_source_sha256", "measurement_baseline_commit",
        "source_snapshot_sha256", "binary_sha256", "raw_path",
        "raw_tree_sha256", "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "aic_l1_to_l0_grouped_matmul_pair_completion_v2",
        "duration_unit": "ns_per_completed_load_pair_per_aic",
        "software_revision": "cann_8.5.1",
        "dtype": "bf16",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_geometry = {
        "m_rows": (16, 32, 48, 64, 80, 96, 112, 128),
        "n_columns": 256,
        "k_columns": 64,
        "a_load_calls": (1, 2, 3, 4, 5, 6, 7, 8),
        "a_repeat_times": 4,
        "b_load_calls": 4,
        "b_repeat_times": 16,
        "b_l1_k_extent": 256,
        "active_aic_groups": (1, 8, 16, 24),
    }
    normalized = dict(value)
    for name, expected in expected_geometry.items():
        actual = value[name]
        if isinstance(expected, tuple):
            if not isinstance(actual, list) or tuple(actual) != expected:
                raise ValueError(f"{owner}.{name} changed its frozen domain")
            normalized[name] = expected
        elif actual != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    durations = value["completion_ns_by_m_rows"]
    if (not isinstance(durations, list)
            or len(durations) != len(value["m_rows"])):
        raise ValueError(f"{owner} completion curve has the wrong length")
    normalized_durations = []
    for duration in durations:
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))):
            raise TypeError(f"{owner} completion durations must be numeric")
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"{owner} completion durations must be finite and positive")
        normalized_durations.append(duration)
    normalized["completion_ns_by_m_rows"] = tuple(normalized_durations)
    expected_n_domain = {
        "minimum_n_columns": 112,
        "maximum_n_columns": 256,
        "n_column_granularity": 16,
    }
    for name, expected in expected_n_domain.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    n_column_service_ns = value["n_column_service_ns"]
    if (isinstance(n_column_service_ns, bool)
            or not isinstance(n_column_service_ns, (int, float))
            or not math.isfinite(float(n_column_service_ns))
            or n_column_service_ns <= 0):
        raise ValueError(f"{owner}.n_column_service_ns must be positive")
    normalized["n_column_service_ns"] = float(n_column_service_ns)
    for name in (
            "scheduler_source_sha256", "load_to_l0a_source_sha256",
            "load_to_l0b_source_sha256", "source_snapshot_sha256",
            "binary_sha256", "raw_tree_sha256",
            "n_geometry_reference_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    commit = value["measurement_baseline_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in commit)):
        raise ValueError(
            f"{owner}.measurement_baseline_commit must be a full Git commit")
    raw_path = value["raw_path"]
    if (not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")):
        raise ValueError(f"{owner}.raw_path is outside its evidence owner")
    n_geometry_reference_path = value["n_geometry_reference_path"]
    if (not isinstance(n_geometry_reference_path, str)
            or not n_geometry_reference_path.startswith(
                "microbench/ascend_910b/references/")
            or ".." in n_geometry_reference_path.split("/")):
        raise ValueError(
            f"{owner}.n_geometry_reference_path is outside its evidence "
            "owner")
    _require_no_duration_consumption(value, owner)
    return normalized


def _normalize_fia_block_reduce_service(value, *, owner, model):
    """Validate one isolated FIA source-level row-reduction service."""

    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "row_granularity",
        "maximum_row_chunk", "maximum_column_count",
        "maximum_source_elements", "special_256_coefficients_ns",
        "special_512_coefficients_ns", "tail_coefficients_ns",
        "production_source_sha256", "source_snapshot_sha256",
        "report_path", "report_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": model,
        "duration_unit": "ns_per_call_per_aiv",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    integer_fields = {
        "row_granularity": 8,
        "maximum_row_chunk": 64,
        "maximum_column_count": 1024,
        "maximum_source_elements": 8192,
    }
    for name, expected in integer_fields.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")

    coefficient_fields = {
        "special_256_coefficients_ns": ("base", "row"),
        "special_512_coefficients_ns": ("base", "row"),
        "tail_coefficients_ns": (
            "base", "row", "column_round", "row_column_round",
            "short_column", "row_short_column",
            "additional_full_segment", "row_additional_full_segment",
            "residual_segment", "row_residual_segment",
            "masked_alignment", "row_masked_alignment",
        ),
    }
    normalized_coefficients = {}
    for section, names in coefficient_fields.items():
        coefficients = value[section]
        if not isinstance(coefficients, dict):
            raise TypeError(f"{owner}.{section} must be a mapping")
        _require_exact_fields(f"{owner}.{section}", coefficients, set(names))
        normalized = {}
        for name in names:
            coefficient = coefficients[name]
            if (isinstance(coefficient, bool)
                    or not isinstance(coefficient, (int, float))):
                raise TypeError(
                    f"{owner}.{section}.{name} must be numeric")
            coefficient = float(coefficient)
            if not math.isfinite(coefficient):
                raise ValueError(
                    f"{owner}.{section}.{name} must be finite")
            normalized[name] = coefficient
        normalized_coefficients[section] = normalized

    for name in (
            "production_source_sha256", "source_snapshot_sha256",
            "report_sha256"):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    report_path = value["report_path"]
    if (not isinstance(report_path, str)
            or not report_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in report_path.split("/")):
        raise ValueError(f"{owner}.report_path is outside its evidence owner")
    _require_no_duration_consumption(value, owner)
    return {
        **expected_text,
        **integer_fields,
        **normalized_coefficients,
        "production_source_sha256": value["production_source_sha256"],
        "source_snapshot_sha256": value["source_snapshot_sha256"],
        "report_path": report_path,
        "report_sha256": value["report_sha256"],
        "production_or_evaluation_duration_consumed": False,
    }


def _normalize_fia_block_reduce_sum_service(value):
    return _normalize_fia_block_reduce_service(
        value,
        owner="scheduling.fia_block_reduce_sum_service",
        model="cann_fia_calc_local_row_sum_source_service_v1",
    )


def _normalize_fia_block_reduce_max_service(value):
    return _normalize_fia_block_reduce_service(
        value,
        owner="scheduling.fia_block_reduce_max_service",
        model="cann_fia_calc_local_row_max_source_service_v1",
    )


def _normalize_fia_kv_line_fill_span(value):
    """Validate the measured AIC ND2NZ row-slice L2 fill span."""

    owner = "scheduling.fia_kv_line_fill_span"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "software_revision", "fill_span_bytes",
        "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "cann_fia_kv_line_fill_span_v1",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    span = value["fill_span_bytes"]
    if isinstance(span, bool) or not isinstance(span, int) or span <= 0:
        raise ValueError(f"{owner}.fill_span_bytes must be a positive integer")
    if span % 128 != 0:
        raise ValueError(
            f"{owner}.fill_span_bytes must be a multiple of the 128 B msprof "
            "access granularity")
    if value["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(
            f"{owner} must not consume production or evaluation duration")
    for name in ("reference_path", "reference_sha256"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"{owner}.{name} must be a non-empty string")
    return dict(value)


def _normalize_fia_brcb_service(value):
    """Validate the isolated CalcExp Brcb plus PIPE_V service."""

    owner = "scheduling.fia_brcb_service"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision", "row_granularity",
        "minimum_row_chunk", "maximum_row_chunk", "startup_ns",
        "per_row_ns", "production_source_sha256",
        "source_snapshot_sha256", "binary_sha256",
        "measurement_baseline_commit", "raw_paths", "raw_sha256",
        "reference_path", "reference_sha256",
        "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "cann_fia_calc_exp_brcb_source_service_v1",
        "duration_unit": "ns_per_call_per_aiv",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    expected_integers = {
        "row_granularity": 8,
        "minimum_row_chunk": 8,
        "maximum_row_chunk": 64,
    }
    normalized = dict(value)
    for name, expected in expected_integers.items():
        actual = value[name]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if actual != expected:
            raise ValueError(f"{owner}.{name} must equal {expected}")
    for name in ("startup_ns", "per_row_ns"):
        actual = value[name]
        if (isinstance(actual, bool)
                or not isinstance(actual, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        actual = float(actual)
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        normalized[name] = actual
    for name, length in (
            ("production_source_sha256", 64),
            ("source_snapshot_sha256", 64),
            ("binary_sha256", 64),
            ("measurement_baseline_commit", 40),
            ("raw_sha256", 64),
            ("reference_sha256", 64)):
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != length
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be lowercase hex")
    raw_paths = value["raw_paths"]
    if not isinstance(raw_paths, list) or len(raw_paths) != 3:
        raise ValueError(f"{owner}.raw_paths must contain three captures")
    for path in raw_paths:
        if (not isinstance(path, str)
                or not path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in path.split("/")):
            raise ValueError(f"{owner}.raw_paths contains an invalid path")
    reference_path = value["reference_path"]
    if (not isinstance(reference_path, str)
            or not reference_path.startswith(
                "microbench/ascend_910b/references/")
            or ".." in reference_path.split("/")):
        raise ValueError(f"{owner}.reference_path is outside its owner")
    _require_no_duration_consumption(value, owner)
    normalized["raw_paths"] = tuple(raw_paths)
    return normalized


def _normalize_aic_empty_kernel_duration_lut(value):
    if not isinstance(value, dict):
        raise TypeError(
            "scheduling.aic_empty_kernel_duration_lut must be a mapping")
    _require_exact_fields(
        "scheduling.aic_empty_kernel_duration_lut", value,
        {"block_dims", "duration_ns"})
    block_dims = value["block_dims"]
    durations = value["duration_ns"]
    if (not isinstance(block_dims, list)
            or not isinstance(durations, list)
            or len(block_dims) < 2
            or len(block_dims) != len(durations)):
        raise ValueError(
            "AIC empty-kernel block dims/durations must be equal-length "
            "lists with at least two points")
    if any(isinstance(block_dim, bool) or not isinstance(block_dim, int)
           for block_dim in block_dims):
        raise TypeError("AIC empty-kernel block dims must be integers")
    if block_dims != list(range(1, block_dims[-1] + 1)):
        raise ValueError(
            "AIC empty-kernel block dims must contain every integer from "
            "one through the measured maximum")
    normalized_durations = []
    for duration in durations:
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))):
            raise TypeError("AIC empty-kernel durations must be numeric")
        normalized = float(duration)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(
                "AIC empty-kernel durations must be finite and positive")
        normalized_durations.append(normalized)
    return tuple(block_dims), tuple(normalized_durations)


def _normalize_aiv_empty_kernel_duration_lut(value):
    if not isinstance(value, dict):
        raise TypeError(
            "scheduling.aiv_empty_kernel_duration_lut must be a mapping")
    _require_exact_fields(
        "scheduling.aiv_empty_kernel_duration_lut", value,
        {"block_dims", "duration_ns"})
    block_dims = value["block_dims"]
    durations = value["duration_ns"]
    if (not isinstance(block_dims, list)
            or not isinstance(durations, list)
            or len(block_dims) < 2
            or len(block_dims) != len(durations)):
        raise ValueError(
            "AIV empty-kernel block dims/durations must be equal-length "
            "lists with at least two points")
    if any(isinstance(block_dim, bool) or not isinstance(block_dim, int)
           for block_dim in block_dims):
        raise TypeError("AIV empty-kernel block dims must be integers")
    if block_dims != list(range(1, block_dims[-1] + 1)):
        raise ValueError(
            "AIV empty-kernel block dims must contain every integer from "
            "one through the measured maximum")
    normalized_durations = []
    for duration in durations:
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))):
            raise TypeError("AIV empty-kernel durations must be numeric")
        normalized = float(duration)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(
                "AIV empty-kernel durations must be finite and positive")
        normalized_durations.append(normalized)
    return tuple(block_dims), tuple(normalized_durations)


def _normalize_aiv_nonempty_task_latency_model(value):
    owner = "scheduling.aiv_nonempty_task_latency_model"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "timer_domain", "software_revision",
        "minimum_aiv_block_dim", "maximum_aiv_block_dim", "submit_ns",
        "descriptor_ns_per_aiv",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "isolated_nonempty_aiv_task_scheduler_v1",
        "duration_unit": "ns_per_complete_nonempty_aiv_task",
        "timer_domain": "msprof_task_time_complete_task",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    bounds = {}
    for name in ("minimum_aiv_block_dim", "maximum_aiv_block_dim"):
        bound = value[name]
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if bound <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        bounds[name] = bound
    if bounds["minimum_aiv_block_dim"] != 1:
        raise ValueError(
            f"{owner}.minimum_aiv_block_dim must equal one")
    if (bounds["maximum_aiv_block_dim"]
            < bounds["minimum_aiv_block_dim"]):
        raise ValueError(f"{owner} block-dimension range is invalid")
    coefficients = {}
    for name in ("submit_ns", "descriptor_ns_per_aiv"):
        coefficient = value[name]
        if (isinstance(coefficient, bool)
                or not isinstance(coefficient, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        normalized = float(coefficient)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        coefficients[name] = normalized
    return {
        **expected_text,
        **bounds,
        **coefficients,
    }


def _normalize_mix_aiv_nonempty_task_latency_model(value):
    owner = "scheduling.mix_aiv_nonempty_task_latency_model"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    expected = {
        "model": "isolated_nonempty_mix_aiv_task_scheduler_v1",
        "duration_unit": "ns_per_complete_nonempty_mix_aiv_task",
    }
    for name, expected_value in expected.items():
        if name not in value:
            raise ValueError(f"{owner} is missing {name!r}")
        if value[name] != expected_value:
            raise ValueError(
                f"{owner}.{name} must be {expected_value!r}")
    normalized = _normalize_aiv_nonempty_task_latency_model({
        **value,
        "model": "isolated_nonempty_aiv_task_scheduler_v1",
        "duration_unit": "ns_per_complete_nonempty_aiv_task",
    })
    return {**normalized, **expected}


def _normalize_aiv_worker_start_fill_model(value):
    owner = "scheduling.aiv_worker_start_fill_model"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "timer_domain", "software_revision",
        "minimum_aiv_block_dim", "maximum_aiv_block_dim",
        "system_counter_tick_ns", "entry_interval_cycles", "evidence_ref",
        "evidence_sha256", "production_or_evaluation_duration_consumed",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "isolated_aiv_worker_start_fill_v1",
        "duration_unit": "ns_per_consecutive_aiv_entry",
        "timer_domain": "device_system_counter",
        "software_revision": "cann_8.5.1",
    }
    normalized = dict(value)
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    for name in ("minimum_aiv_block_dim", "maximum_aiv_block_dim"):
        bound = value[name]
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if bound <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        normalized[name] = bound
    if normalized["minimum_aiv_block_dim"] != 1:
        raise ValueError(
            f"{owner}.minimum_aiv_block_dim must equal one")
    if (normalized["maximum_aiv_block_dim"]
            < normalized["minimum_aiv_block_dim"]):
        raise ValueError(f"{owner} block-dimension range is invalid")
    for name in ("system_counter_tick_ns", "entry_interval_cycles"):
        coefficient = value[name]
        if (isinstance(coefficient, bool)
                or not isinstance(coefficient, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        normalized[name] = coefficient
    evidence_ref = value["evidence_ref"]
    if (not isinstance(evidence_ref, str)
            or not evidence_ref.startswith(
                "microbench/ascend_910b/references/")
            or ".." in evidence_ref.split("/")):
        raise ValueError(f"{owner}.evidence_ref is outside its owner")
    evidence_sha256 = value["evidence_sha256"]
    if (not isinstance(evidence_sha256, str)
            or len(evidence_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in evidence_sha256)):
        raise ValueError(f"{owner}.evidence_sha256 must be SHA256 hex")
    _require_no_duration_consumption(value, owner)
    return normalized


def _normalize_aiv_memory_task_latency_model(value):
    owner = "scheduling.aiv_memory_task_latency_model"
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "timer_domain", "software_revision",
        "minimum_aiv_block_dim", "maximum_aiv_block_dim", "submit_ns",
        "descriptor_ns_per_aiv",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "isolated_aiv_memory_task_floor_v1",
        "duration_unit": "ns_per_complete_nonempty_aiv_task",
        "timer_domain": "msprof_task_time_complete_task",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    bounds = {}
    for name in ("minimum_aiv_block_dim", "maximum_aiv_block_dim"):
        bound = value[name]
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if bound <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        bounds[name] = bound
    if bounds["minimum_aiv_block_dim"] != 1:
        raise ValueError(f"{owner}.minimum_aiv_block_dim must equal one")
    if (bounds["maximum_aiv_block_dim"]
            < bounds["minimum_aiv_block_dim"]):
        raise ValueError(f"{owner} block-dimension range is invalid")
    coefficients = {}
    for name in ("submit_ns", "descriptor_ns_per_aiv"):
        coefficient = value[name]
        if (isinstance(coefficient, bool)
                or not isinstance(coefficient, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        normalized = float(coefficient)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        coefficients[name] = normalized
    return {**expected_text, **bounds, **coefficients}


def _normalize_aiv_tiling_materialization_latency_model(value, owner):
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision",
        "minimum_active_aiv_workers", "maximum_active_aiv_workers",
        "startup_ns", "serialized_ns_per_aiv",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": "shared_aiv_tiling_materialization_affine_v1",
        "duration_unit": "ns_per_complete_tiling_materialization_phase",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    bounds = {}
    for name in (
            "minimum_active_aiv_workers", "maximum_active_aiv_workers"):
        bound = value[name]
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if bound <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        bounds[name] = bound
    if bounds["minimum_active_aiv_workers"] != 1:
        raise ValueError(
            f"{owner}.minimum_active_aiv_workers must equal one")
    if (bounds["maximum_active_aiv_workers"]
            < bounds["minimum_active_aiv_workers"]):
        raise ValueError(f"{owner} active-AIV range is invalid")
    coefficients = {}
    for name in ("startup_ns", "serialized_ns_per_aiv"):
        coefficient = value[name]
        if (isinstance(coefficient, bool)
                or not isinstance(coefficient, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        normalized = float(coefficient)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        coefficients[name] = normalized
    return {**expected_text, **bounds, **coefficients}


def _normalize_aiv_complete_max_tiling_materialization_latency_model(
        value, owner, expected_model):
    """Normalize a source-shaped complete-max AIV tiling prologue."""

    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a mapping")
    fields = {
        "model", "duration_unit", "software_revision",
        "minimum_active_aiv_workers", "maximum_active_aiv_workers",
        "single_worker_completion_ns", "serialized_ns_per_aiv",
    }
    _require_exact_fields(owner, value, fields)
    expected_text = {
        "model": expected_model,
        "duration_unit": "ns_per_complete_tiling_materialization_phase",
        "software_revision": "cann_8.5.1",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    bounds = {}
    for name in (
            "minimum_active_aiv_workers", "maximum_active_aiv_workers"):
        bound = value[name]
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise TypeError(f"{owner}.{name} must be an integer")
        if bound <= 0:
            raise ValueError(f"{owner}.{name} must be positive")
        bounds[name] = bound
    if bounds["minimum_active_aiv_workers"] != 1:
        raise ValueError(
            f"{owner}.minimum_active_aiv_workers must equal one")
    if (bounds["maximum_active_aiv_workers"]
            < bounds["minimum_active_aiv_workers"]):
        raise ValueError(f"{owner} active-AIV range is invalid")
    coefficients = {}
    for name in ("single_worker_completion_ns", "serialized_ns_per_aiv"):
        value_ns = value[name]
        if (isinstance(value_ns, bool)
                or not isinstance(value_ns, (int, float))):
            raise TypeError(f"{owner}.{name} must be numeric")
        normalized = float(value_ns)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(f"{owner}.{name} must be finite and positive")
        coefficients[name] = normalized
    return {**expected_text, **bounds, **coefficients}


def _normalize_aiv_rope_tiling_materialization_latency_model(value, owner):
    return _normalize_aiv_complete_max_tiling_materialization_latency_model(
        value, owner, "rope_aiv_tiling_materialization_complete_max_v1")


def _normalize_aiv_gather_v3_tiling_materialization_latency_model(
        value, owner):
    return _normalize_aiv_complete_max_tiling_materialization_latency_model(
        value, owner,
        "gather_v3_aiv_tiling_materialization_complete_max_v1")


def _normalize_aiv_broadcast_one_dim_tiling_materialization_latency_model(
        value, owner):
    return _normalize_aiv_complete_max_tiling_materialization_latency_model(
        value, owner,
        "broadcast_one_dim_tiling_materialization_complete_max_v1")


def _load_aiv_rope_tiling_materialization_latency_model(repository_root):
    """Load the pinned independent 48-byte RoPE prologue evidence."""

    path = repository_root / _AIV_ROPE_TILING_MATERIALIZATION_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            _AIV_ROPE_TILING_MATERIALIZATION_REFERENCE_SHA256):
        raise ValueError(
            "AIV RoPE tiling-materialization reference SHA256 mismatch")
    evidence = json.loads(payload_bytes)
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "tiling_struct_materialization",
        "payload_bytes": 48,
        "model": "rope_aiv_tiling_materialization_complete_max_v1",
        "formula": (
            "max(single_worker_completion_ns, serialized_ns_per_aiv * "
            "active_aiv_workers)"),
        "request_size_unit": "active_aiv_workers",
        "duration_unit": "ns_per_complete_tiling_materialization_phase",
        "minimum_active_aiv_workers": 1,
        "maximum_active_aiv_workers": 48,
        "measurement_gate_passed": True,
        "mechanism_publishable": True,
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if evidence[name] != expected_value:
            raise ValueError(
                f"AIV RoPE tiling-materialization reference {name} changed")
    model = _normalize_aiv_rope_tiling_materialization_latency_model(
        {
            "model": evidence["model"],
            "duration_unit": evidence["duration_unit"],
            "software_revision": evidence["software_revision"],
            "minimum_active_aiv_workers": evidence[
                "minimum_active_aiv_workers"],
            "maximum_active_aiv_workers": evidence[
                "maximum_active_aiv_workers"],
            "single_worker_completion_ns": evidence["parameters"][
                "single_worker_completion_ns"],
            "serialized_ns_per_aiv": evidence["parameters"][
                "serialized_ns_per_aiv"],
        },
        "AIV RoPE tiling-materialization reference",
    )
    return {
        **model,
        "reference_path": str(_AIV_ROPE_TILING_MATERIALIZATION_REFERENCE),
        "reference_sha256": (
            _AIV_ROPE_TILING_MATERIALIZATION_REFERENCE_SHA256),
    }


def _load_aiv_gather_v3_tiling_materialization_latency_model(
        repository_root):
    """Load the pinned independent 192-byte GatherV3 prologue evidence."""

    path = repository_root / _AIV_GATHER_V3_TILING_MATERIALIZATION_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            _AIV_GATHER_V3_TILING_MATERIALIZATION_REFERENCE_SHA256):
        raise ValueError(
            "AIV GatherV3 tiling-materialization reference SHA256 mismatch")
    evidence = json.loads(payload_bytes)
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "tiling_struct_materialization",
        "payload_bytes": 192,
        "model": "gather_v3_aiv_tiling_materialization_complete_max_v1",
        "formula": (
            "max(single_worker_completion_ns, serialized_ns_per_aiv * "
            "active_aiv_workers)"),
        "request_size_unit": "active_aiv_workers",
        "duration_unit": "ns_per_complete_tiling_materialization_phase",
        "minimum_active_aiv_workers": 1,
        "maximum_active_aiv_workers": 48,
        "measurement_gate_passed": True,
        "mechanism_publishable": True,
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if evidence[name] != expected_value:
            raise ValueError(
                "AIV GatherV3 tiling-materialization reference "
                f"{name} changed")
    model = _normalize_aiv_gather_v3_tiling_materialization_latency_model(
        {
            "model": evidence["model"],
            "duration_unit": evidence["duration_unit"],
            "software_revision": evidence["software_revision"],
            "minimum_active_aiv_workers": evidence[
                "minimum_active_aiv_workers"],
            "maximum_active_aiv_workers": evidence[
                "maximum_active_aiv_workers"],
            "single_worker_completion_ns": evidence["parameters"][
                "single_worker_completion_ns"],
            "serialized_ns_per_aiv": evidence["parameters"][
                "serialized_ns_per_aiv"],
        },
        "AIV GatherV3 tiling-materialization reference",
    )
    return {
        **model,
        "reference_path": str(
            _AIV_GATHER_V3_TILING_MATERIALIZATION_REFERENCE),
        "reference_sha256": (
            _AIV_GATHER_V3_TILING_MATERIALIZATION_REFERENCE_SHA256),
    }


def _load_aiv_broadcast_one_dim_tiling_materialization_latency_model(
        repository_root):
    """Load the pinned independent 56-byte BroadcastOneDim evidence."""

    path = repository_root / (
        _AIV_BROADCAST_ONE_DIM_TILING_MATERIALIZATION_REFERENCE)
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            _AIV_BROADCAST_ONE_DIM_TILING_MATERIALIZATION_REFERENCE_SHA256):
        raise ValueError(
            "AIV BroadcastOneDim tiling-materialization reference SHA256 "
            "mismatch")
    evidence = json.loads(payload_bytes)
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "tiling_struct_materialization",
        "payload_bytes": 56,
        "model": "broadcast_one_dim_tiling_materialization_complete_max_v1",
        "formula": (
            "max(single_worker_completion_ns, serialized_ns_per_aiv * "
            "active_aiv_workers)"),
        "request_size_unit": "active_aiv_workers",
        "duration_unit": "ns_per_complete_tiling_materialization_phase",
        "minimum_active_aiv_workers": 1,
        "maximum_active_aiv_workers": 48,
        "measurement_gate_passed": True,
        "mechanism_publishable": True,
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if evidence[name] != expected_value:
            raise ValueError(
                "AIV BroadcastOneDim tiling-materialization reference "
                f"{name} changed")
    model = _normalize_aiv_broadcast_one_dim_tiling_materialization_latency_model(
        {
            "model": evidence["model"],
            "duration_unit": evidence["duration_unit"],
            "software_revision": evidence["software_revision"],
            "minimum_active_aiv_workers": evidence[
                "minimum_active_aiv_workers"],
            "maximum_active_aiv_workers": evidence[
                "maximum_active_aiv_workers"],
            "single_worker_completion_ns": evidence["parameters"][
                "single_worker_completion_ns"],
            "serialized_ns_per_aiv": evidence["parameters"][
                "serialized_ns_per_aiv"],
        },
        "AIV BroadcastOneDim tiling-materialization reference",
    )
    return {
        **model,
        "reference_path": str(
            _AIV_BROADCAST_ONE_DIM_TILING_MATERIALIZATION_REFERENCE),
        "reference_sha256": (
            _AIV_BROADCAST_ONE_DIM_TILING_MATERIALIZATION_REFERENCE_SHA256),
    }


def _normalize_mixed_empty_kernel_duration_lut(value):
    if not isinstance(value, dict):
        raise TypeError(
            "scheduling.mixed_empty_kernel_duration_lut must be a mapping")
    _require_exact_fields(
        "scheduling.mixed_empty_kernel_duration_lut", value,
        {"aic_block_dims", "aiv_block_dim_ratio", "duration_ns"})
    block_dims = value["aic_block_dims"]
    aiv_block_dim_ratio = value["aiv_block_dim_ratio"]
    durations = value["duration_ns"]
    if (not isinstance(block_dims, list)
            or not isinstance(durations, list)
            or len(block_dims) < 2
            or len(block_dims) != len(durations)):
        raise ValueError(
            "mixed empty-kernel AIC block dims/durations must be "
            "equal-length lists with at least two points")
    if any(isinstance(block_dim, bool) or not isinstance(block_dim, int)
           for block_dim in block_dims):
        raise TypeError("mixed empty-kernel AIC block dims must be integers")
    if block_dims != list(range(1, block_dims[-1] + 1)):
        raise ValueError(
            "mixed empty-kernel AIC block dims must contain every integer "
            "from one through the measured maximum")
    if (isinstance(aiv_block_dim_ratio, bool)
            or not isinstance(aiv_block_dim_ratio, int)):
        raise TypeError(
            "mixed empty-kernel AIV block-dim ratio must be an integer")
    if aiv_block_dim_ratio != 2:
        raise ValueError(
            "mixed empty-kernel AIV block-dim ratio must be two for the "
            "KERNEL_TYPE_MIX_AIC_1_2 task contract")
    normalized_durations = []
    for duration in durations:
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))):
            raise TypeError("mixed empty-kernel durations must be numeric")
        normalized = float(duration)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError(
                "mixed empty-kernel durations must be finite and positive")
        normalized_durations.append(normalized)
    return (
        tuple(block_dims), aiv_block_dim_ratio,
        tuple(normalized_durations))


_MEMORY_REQUIRED_FIELDS = {
    "l2_conflict_efficiency_moe",
    "dram_protocol_efficiency", "dram_protocol_efficiency_l1cached",
    "profiler_per_kernel_traffic_reliable",
}

_MEMORY_OPTIONAL_FIELDS = {
    "dram_latency_ns", "l2_latency_ns", "cacheline_bytes",
    "l2_replacement_capacity_bytes",
    "l2_replacement_capacity_evidence_ref",
    "l2_replacement_capacity_evidence_sha256",
    "l2_dirty_replacement_capacity_bytes_per_s",
    "l2_dirty_replacement_evidence_ref",
    "l2_dirty_replacement_evidence_sha256",
    "dram_miss_extra_cycles", "dtype_bytes",
    "dram_transaction_bytes", "l2_transaction_bytes",
    "source_slots_per_unit",
    "gemm_l1_refetch_tile_rows", "l2_hit_latency_local_cycles",
    "l2_hit_latency_remote_cycles", "l2_local_fraction",
    "l2_partition_service_cycles_bw", "lsu_peak_issue_rate_per_sm",
    "gemv_scalar_half_issue_group_elements_per_lane",
    "gemv_scalar_half_resident_ctas_per_sm",
    "gemv_scalar_half_dram_transactions_per_sm_s",
    "gemv_scalar_half_source_fill_ns",
    "ldgsts_initial_32b_transactions_single_sm_cycle",
    "ldgsts_initial_32b_transactions_per_active_sm_cycle",
    "ldgsts_initial_32b_transactions_full_chip_per_cycle",
    "ldgsts_finite_resident_panel_bytes",
    "ldgsts_finite_resident_full_chip_ctas",
    "ldgsts_finite_resident_single_panel_cycles",
    "ldgsts_finite_resident_affine_startup_cycles",
    "ldgsts_finite_resident_affine_panel_cycles",
    "ldgsts_finite_resident_maximum_panels",
    "ldgsts_finite_resident_reference_sha256",
    "ldgsts_queued_steady_minimum_grid_ctas",
    "ldgsts_queued_steady_maximum_panels_per_cta",
    "ldgsts_queued_steady_average_panels_per_cta",
    "ldgsts_queued_steady_full_chip_initial_32b_transactions_per_cycle",
    "ldgsts_queued_steady_reference_sha256",
    "memcpy_dtod_floor_ns",
    "memcpy_dtod_native_flash_attention_copy_medium_min_bytes",
    "memcpy_dtod_native_flash_attention_copy_medium_max_bytes",
    "memcpy_dtod_native_flash_attention_copy_medium_floor_ns",
    "mte2_gemm_tile_bytes",
    "mte2_per_core_peak_gbs", "mte2_tile_latency_fixed_ns",
    "mte2_transfer_bw_gbs", "pim_accumulator_roundtrip_ns",
    "pim_burst_bytes", "pim_bus_turnaround_ns", "pim_column_access_ns",
    "pim_columns_per_row", "pim_pu_inbuf_bytes", "pim_pu_lat_ns",
    "pim_row_activate_ns", "pim_row_bytes", "pim_row_close_ns",
    "pim_write_read_row_cycle_ns",
    "pipeline_fill", "systolic_dma_bus_bpc",
    "dram_arbitration_mode",
    "dram_read_capacity_per_clk", "dram_write_capacity_per_clk",
    "memory_access_class_services",
}

_MEMORY_PREDICTOR_FIELDS = {
    "memcpy_dtod_floor_ns",
}

_MEMORY_PREDICTOR_FIELDS_BY_MODEL = {
    "gpu_mshr": {
        "dram_arbitration_mode",
        "dram_read_capacity_per_clk", "dram_write_capacity_per_clk",
    },
    "npu_tile": {
        "dram_arbitration_mode",
        "dram_read_capacity_per_clk", "dram_write_capacity_per_clk",
    },
    "pim_bank": set(),
    "systolic_scratchpad": set(),
}


def _normalize_memory_access_class_services(value):
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(
            "memory_subsystem.memory_access_class_services must be a list "
            "or null")
    services = []
    for index, raw_service in enumerate(value):
        if not isinstance(raw_service, dict):
            raise TypeError(
                "memory access-class service entries must be mappings")
        _require_fields_with_optional(
            f"memory access-class service entry {index}", raw_service,
            {
                "service", "access_kind", "ingress_resource_role",
                "signature", "scope", "logical_byte_capacity_per_s",
                "evidence_ref",
            },
            {
                "replay_to_dram_fraction",
                "logical_byte_capacity_per_s_by_grid_ctas",
            },
        )
        raw_signature = raw_service["signature"]
        if not isinstance(raw_signature, dict):
            raise TypeError(
                "memory access-class service signature must be a mapping")
        _require_fields_with_optional(
            f"memory access-class signature {index}", raw_signature,
            {
                "access_class", "dtype", "tile_m", "tile_k",
                "logical_row_bytes",
            },
            {
                "operand_role", "tile_n", "group_size_m",
                "weight_streams_per_group",
            },
        )
        operand_role = None
        if "operand_role" in raw_signature:
            operand_role = raw_signature["operand_role"]
        tile_n = None
        if "tile_n" in raw_signature:
            tile_n = raw_signature["tile_n"]
        group_size_m = None
        if "group_size_m" in raw_signature:
            group_size_m = raw_signature["group_size_m"]
        weight_streams_per_group = None
        if "weight_streams_per_group" in raw_signature:
            weight_streams_per_group = raw_signature[
                "weight_streams_per_group"]
        signature = MemoryAccessClassSignature(
            access_class=raw_signature["access_class"],
            dtype=raw_signature["dtype"],
            tile_m=raw_signature["tile_m"],
            tile_k=raw_signature["tile_k"],
            logical_row_bytes=raw_signature["logical_row_bytes"],
            operand_role=operand_role,
            tile_n=tile_n,
            group_size_m=group_size_m,
            weight_streams_per_group=weight_streams_per_group,
        )
        replay_to_dram_fraction = None
        if "replay_to_dram_fraction" in raw_service:
            replay_to_dram_fraction = raw_service[
                "replay_to_dram_fraction"]
        capacity_by_grid = None
        if "logical_byte_capacity_per_s_by_grid_ctas" in raw_service:
            raw_table = raw_service[
                "logical_byte_capacity_per_s_by_grid_ctas"]
            if not isinstance(raw_table, dict) or not raw_table:
                raise TypeError(
                    f"memory access-class service entry {index} measured "
                    "capacity table must be a non-empty mapping")
            entries = []
            for raw_grid, raw_capacity in raw_table.items():
                grid = int(raw_grid)
                if str(grid) != str(raw_grid).strip():
                    raise ValueError(
                        f"memory access-class service entry {index} measured "
                        "capacity grid must be an integer key")
                entries.append((grid, float(raw_capacity)))
            entries.sort()
            capacity_by_grid = tuple(entries)
            base = float(raw_service["logical_byte_capacity_per_s"])
            if base > max(capacity for _, capacity in capacity_by_grid):
                raise ValueError(
                    f"memory access-class service entry {index} entry "
                    "capacity exceeds every measured grid")
        services.append(MemoryAccessClassService(
            service=raw_service["service"],
            access_kind=raw_service["access_kind"],
            ingress_resource_role=raw_service["ingress_resource_role"],
            signature=signature,
            scope=raw_service["scope"],
            logical_byte_capacity_per_s=(
                raw_service["logical_byte_capacity_per_s"]),
            evidence_ref=raw_service["evidence_ref"],
            replay_to_dram_fraction=replay_to_dram_fraction,
            logical_byte_capacity_per_s_by_grid_ctas=capacity_by_grid,
        ))
    identities = tuple(
        (
            service.service,
            service.access_kind,
            service.ingress_resource_role,
            service.signature,
        )
        for service in services
    )
    if len(identities) != len(set(identities)):
        raise ValueError(
            "memory_subsystem.memory_access_class_services cannot repeat an "
            "exact service signature")
    metadata_by_name = {}
    for service in services:
        metadata = (
            service.access_kind,
            service.ingress_resource_role,
            service.scope,
        )
        if service.service in metadata_by_name:
            if metadata_by_name[service.service] != metadata:
                raise ValueError(
                    "memory access-class signatures with one service name "
                    "must share direction, ingress role, and scope")
        else:
            metadata_by_name[service.service] = metadata
    return tuple(services)


def _normalize_memory_subsystem(raw, memory_model):
    if memory_model not in _MEMORY_PREDICTOR_FIELDS_BY_MODEL:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")
    required_fields = (
        _MEMORY_REQUIRED_FIELDS
        | _MEMORY_PREDICTOR_FIELDS
        | _MEMORY_PREDICTOR_FIELDS_BY_MODEL[memory_model]
    )
    missing = required_fields - set(raw)
    if missing:
        raise ValueError(
            f"memory_subsystem missing required fields: {sorted(missing)}")
    unknown = set(raw) - _MEMORY_REQUIRED_FIELDS - _MEMORY_OPTIONAL_FIELDS
    if unknown:
        raise ValueError(
            f"memory_subsystem has unknown fields: {sorted(unknown)}")
    memory = dict.fromkeys(_MEMORY_OPTIONAL_FIELDS)
    memory.update({
        name: raw[name] for name in _MEMORY_OPTIONAL_FIELDS & set(raw)
    })
    memory.update({name: raw[name] for name in _MEMORY_REQUIRED_FIELDS})
    for name, value in memory.items():
        if name in {
                "memory_access_class_services",
                "ldgsts_finite_resident_reference_sha256",
                "ldgsts_queued_steady_reference_sha256",
                "l2_replacement_capacity_evidence_ref",
                "l2_replacement_capacity_evidence_sha256",
                "l2_dirty_replacement_evidence_ref",
                "l2_dirty_replacement_evidence_sha256"}:
            continue
        if name == "dram_arbitration_mode":
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    "memory_subsystem.dram_arbitration_mode must be a "
                    "string or null")
            continue
        if name == "profiler_per_kernel_traffic_reliable":
            if not isinstance(value, bool):
                raise TypeError(f"memory_subsystem.{name} must be boolean")
            continue
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"memory_subsystem.{name} must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"memory_subsystem.{name} must be finite, got {value!r}")
    integer_fields = {
        "cacheline_bytes",
        "l2_replacement_capacity_bytes",
        "dram_transaction_bytes", "l2_transaction_bytes",
        "source_slots_per_unit",
        "gemv_scalar_half_issue_group_elements_per_lane",
        "gemv_scalar_half_resident_ctas_per_sm",
        "dtype_bytes",
        "ldgsts_finite_resident_panel_bytes",
        "ldgsts_finite_resident_full_chip_ctas",
        "ldgsts_finite_resident_maximum_panels",
        "ldgsts_queued_steady_minimum_grid_ctas",
        "ldgsts_queued_steady_maximum_panels_per_cta",
        "mte2_gemm_tile_bytes",
        "memcpy_dtod_native_flash_attention_copy_medium_min_bytes",
        "memcpy_dtod_native_flash_attention_copy_medium_max_bytes",
        "pim_burst_bytes", "pim_columns_per_row", "pim_pu_inbuf_bytes",
        "pim_row_bytes", "pipeline_fill",
    }
    invalid_integer_fields = {
        name: memory[name] for name in integer_fields
        if memory[name] is not None and not isinstance(memory[name], int)
    }
    if invalid_integer_fields:
        raise TypeError(
            "memory_subsystem integer fields must be integers: "
            f"{invalid_integer_fields}")
    required_by_model = {
        "gpu_mshr": {
            "dram_latency_ns", "l2_latency_ns", "cacheline_bytes",
            "l2_hit_latency_local_cycles", "l2_hit_latency_remote_cycles",
            "dram_miss_extra_cycles",
            "dram_transaction_bytes", "l2_transaction_bytes",
        },
        # The current 910B contract is deliberately aggregate-only.  MTE tile
        # size, startup latency, and descriptor capacity are optional evidence,
        # not prerequisites silently reconstructed by the production Forward.
        "npu_tile": set(),
        "pim_bank": {
            "dtype_bytes", "pim_accumulator_roundtrip_ns",
            "pim_burst_bytes", "pim_bus_turnaround_ns",
            "pim_column_access_ns", "pim_columns_per_row",
            "pim_pu_inbuf_bytes", "pim_pu_lat_ns", "pim_row_activate_ns",
            "pim_row_bytes", "pim_row_close_ns",
            "pim_write_read_row_cycle_ns",
        },
        "systolic_scratchpad": {
            "dtype_bytes", "pipeline_fill", "systolic_dma_bus_bpc",
        },
    }
    incomplete = sorted(
        name for name in required_by_model[memory_model]
        if memory[name] is None)
    if incomplete:
        raise ValueError(
            f"{memory_model} memory_subsystem fields cannot be null: "
            f"{incomplete}")
    replacement_fields = (
        "l2_replacement_capacity_bytes",
        "l2_replacement_capacity_evidence_ref",
        "l2_replacement_capacity_evidence_sha256",
    )
    declared_replacement_fields = tuple(
        name for name in replacement_fields if memory[name] is not None)
    if declared_replacement_fields and len(declared_replacement_fields) != 3:
        raise ValueError(
            "L2 replacement capacity requires bytes, evidence ref, and "
            "evidence SHA256 together")
    if memory["l2_replacement_capacity_bytes"] is not None:
        if memory["l2_replacement_capacity_bytes"] <= 0:
            raise ValueError(
                "memory_subsystem.l2_replacement_capacity_bytes must be "
                "positive")
        evidence_ref = memory[
            "l2_replacement_capacity_evidence_ref"]
        if (not isinstance(evidence_ref, str) or not evidence_ref
                or evidence_ref.startswith("/")
                or ".." in evidence_ref.split("/")):
            raise ValueError(
                "L2 replacement-capacity evidence ref must be a relative "
                "repository path")
        evidence_sha256 = memory[
            "l2_replacement_capacity_evidence_sha256"]
        if (not isinstance(evidence_sha256, str)
                or len(evidence_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in evidence_sha256)):
            raise ValueError(
                "L2 replacement-capacity evidence SHA256 must be lowercase "
                "hex")
    native_copy_fields = (
        "memcpy_dtod_native_flash_attention_copy_medium_min_bytes",
        "memcpy_dtod_native_flash_attention_copy_medium_max_bytes",
        "memcpy_dtod_native_flash_attention_copy_medium_floor_ns",
    )
    declared_native_copy_fields = tuple(
        name for name in native_copy_fields if memory[name] is not None)
    if (declared_native_copy_fields
            and len(declared_native_copy_fields) != len(native_copy_fields)):
        raise ValueError(
            "native FlashAttention D2D copy plateau requires byte bounds "
            "and a floor together")
    if declared_native_copy_fields:
        minimum_bytes = memory[native_copy_fields[0]]
        maximum_bytes = memory[native_copy_fields[1]]
        floor_ns = memory[native_copy_fields[2]]
        if minimum_bytes <= 0 or maximum_bytes < minimum_bytes:
            raise ValueError(
                "native FlashAttention D2D copy plateau byte range is "
                "invalid")
        if floor_ns <= 0:
            raise ValueError(
                "native FlashAttention D2D copy plateau floor must be "
                "positive")
    dirty_replacement_fields = (
        "l2_dirty_replacement_capacity_bytes_per_s",
        "l2_dirty_replacement_evidence_ref",
        "l2_dirty_replacement_evidence_sha256",
    )
    declared_dirty_replacement_fields = tuple(
        name for name in dirty_replacement_fields if memory[name] is not None)
    if (declared_dirty_replacement_fields
            and len(declared_dirty_replacement_fields) != 3):
        raise ValueError(
            "L2 dirty-replacement service requires capacity, evidence ref, "
            "and evidence SHA256 together")
    if memory["l2_dirty_replacement_capacity_bytes_per_s"] is not None:
        if memory["l2_dirty_replacement_capacity_bytes_per_s"] <= 0:
            raise ValueError(
                "memory_subsystem.l2_dirty_replacement_capacity_bytes_per_s "
                "must be positive")
        evidence_ref = memory["l2_dirty_replacement_evidence_ref"]
        if (not isinstance(evidence_ref, str) or not evidence_ref
                or evidence_ref.startswith("/")
                or ".." in evidence_ref.split("/")):
            raise ValueError(
                "L2 dirty-replacement evidence ref must be a relative "
                "repository path")
        evidence_sha256 = memory[
            "l2_dirty_replacement_evidence_sha256"]
        if (not isinstance(evidence_sha256, str)
                or len(evidence_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in evidence_sha256)):
            raise ValueError(
                "L2 dirty-replacement evidence SHA256 must be lowercase hex")
    arbitration_mode = memory["dram_arbitration_mode"]
    read_capacity = memory["dram_read_capacity_per_clk"]
    write_capacity = memory["dram_write_capacity_per_clk"]
    if memory_model in ("gpu_mshr", "npu_tile"):
        valid_modes = {
            "shared_common_capacity", "shared_directional_capacity",
            "independent_directional_capacity"}
        if arbitration_mode not in valid_modes:
            raise ValueError(
                "target memory models require "
                "memory_subsystem.dram_arbitration_mode to be one of "
                f"{sorted(valid_modes)}, got {arbitration_mode!r}")
        if arbitration_mode == "shared_common_capacity":
            if read_capacity is not None or write_capacity is not None:
                raise ValueError(
                    "shared_common_capacity must not declare directional "
                    "DRAM capacities")
        elif arbitration_mode in (
                "shared_directional_capacity",
                "independent_directional_capacity"):
            if read_capacity is None or write_capacity is None:
                raise ValueError(
                    f"{arbitration_mode} requires explicit read and write "
                    "DRAM capacities")
            for name, value in (
                    ("dram_read_capacity_per_clk", read_capacity),
                    ("dram_write_capacity_per_clk", write_capacity)):
                if float(value) <= 0:
                    raise ValueError(
                        f"memory_subsystem.{name} must be positive")
        else:
            raise AssertionError(
                f"unhandled DRAM arbitration mode: {arbitration_mode!r}")
    elif any(value is not None for value in (
            arbitration_mode, read_capacity, write_capacity)):
        raise ValueError(
            f"{memory_model} must not declare target DRAM arbitration fields")
    memory["memory_access_class_services"] = (
        _normalize_memory_access_class_services(
            memory["memory_access_class_services"]))
    return memory


class Resource:
    """One hardware resource (compute pipe, memory BW, DMA engine, etc.).

    Required fields for all types:
        type, name, description, clock_domain, scope,
        capacity_per_unit_per_clk, num_units
    Additional required fields by type:
        memory: traffic_filter (lru | page_conflict | null)
        memory + traffic_filter=lru: size_bytes; num_partitions is optional
        memory + traffic_filter=page_conflict: size_bytes; num_partitions is optional
    """

    def __init__(self, cfg, clock_domains,
                 model_clock_domain, dram_clock_domain):
        # ── Common required fields ──
        self.type = cfg["type"]
        self.name = cfg["name"]
        self.role = cfg["role"]  # semantic role (e.g. "l2_cache", "dram", "compute_tensor")
        self.description = cfg["description"]
        self.scope = cfg["scope"]
        self.clock_domain = cfg["clock_domain"]  # "sm", "dram", or null
        self.service_rates_per_unit_s = cfg["service_rates_per_unit_s"]
        self.service_setup_time_s = cfg["service_setup_time_s"]
        self.active_unit_service_time_s = cfg[
            "active_unit_service_time_s"]
        self.exact_request_services = cfg["exact_request_services"]
        if "capacity_per_second" in cfg and cfg["capacity_per_second"] is not None:
            if (self.scope != "shared" or cfg["num_units"] != 1
                    or self.clock_domain is not None or cfg["capacity_per_unit_per_clk"] is not None
                    or self.service_rates_per_unit_s is not None or self.service_setup_time_s is not None
                    or self.active_unit_service_time_s is not None or self.exact_request_services is not None):
                raise ValueError("aggregate roofline capacity requires one whole-device resource without native services")
            self.capacity = float(cfg["capacity_per_second"])
            if not math.isfinite(self.capacity) or self.capacity <= 0:
                raise ValueError("aggregate roofline capacity must be finite and positive")
            if self.type not in ("compute", "memory"):
                raise ValueError("aggregate roofline resource must be compute or memory")
            self.size_bytes = cfg["size_bytes"]
            self.traffic_filter = None
            self.num_partitions = None
            self.total_units = 1  # One whole-device service, not one SM/CU.
            self.num_units = 0
            self.capacity_per_unit = 0
            self._nominal_clock_hz = None
            self._runtime_clock_source = "absolute"
            self._capacity_per_clk = None
            self._capacity_per_unit_per_clk = None
            return
        if self.exact_request_services is not None:
            self.exact_request_services = {
                service: {
                    **spec,
                    "service_time_s_by_request_size": dict(
                        spec["service_time_s_by_request_size"]),
                }
                for service, spec in self.exact_request_services.items()
            }

        # Resolve clock frequency from clock_domains section
        if self.clock_domain in clock_domains:
            self._nominal_clock_hz = float(clock_domains[self.clock_domain])
        else:
            raise ValueError(
                f"resource {self.name!r} references unknown clock domain "
                f"{self.clock_domain!r}")
        if self.clock_domain == model_clock_domain:
            self._runtime_clock_source = "model"
        elif self.clock_domain == dram_clock_domain:
            self._runtime_clock_source = "dram"
        elif self.clock_domain in clock_domains:
            # Auxiliary domains such as Ascend AIC/AIV are explicitly fixed at
            # their configured nominal frequency.
            self._runtime_clock_source = "fixed"
        else:
            raise AssertionError(
                f"unhandled validated clock domain: {self.clock_domain!r}")

        # ── Type-specific required fields ──
        if self.type == "compute":
            self.traffic_filter = None
            self.size_bytes = 0
            self.num_partitions = 1
            if self.service_rates_per_unit_s is not None:
                self.service_rates_per_unit_s = dict(
                    self.service_rates_per_unit_s)
            if self.service_setup_time_s is not None:
                self.service_setup_time_s = dict(self.service_setup_time_s)
            if self.active_unit_service_time_s is not None:
                self.active_unit_service_time_s = {
                    service: dict(model)
                    for service, model
                    in self.active_unit_service_time_s.items()
                }
        elif self.type == "memory":
            if self.service_rates_per_unit_s is not None:
                raise ValueError(
                    f"memory resource {self.name!r} cannot declare typed "
                    "compute service rates")
            if self.service_setup_time_s is not None:
                raise ValueError(
                    f"memory resource {self.name!r} cannot declare typed "
                    "compute service setup times")
            if self.active_unit_service_time_s is not None:
                raise ValueError(
                    f"memory resource {self.name!r} cannot declare typed "
                    "active-unit service times")
            self.traffic_filter = cfg["traffic_filter"]  # "lru", "page_conflict", or null
            if self.traffic_filter in ("lru", "page_conflict"):
                self.size_bytes = cfg["size_bytes"]
                self.num_partitions = cfg["num_partitions"]
            elif self.traffic_filter is None:
                self.size_bytes = (
                    0 if cfg["size_bytes"] is None else cfg["size_bytes"])
                self.num_partitions = (
                    1 if cfg["num_partitions"] is None
                    else cfg["num_partitions"])
            else:
                raise ValueError(f"Unknown traffic_filter '{self.traffic_filter}' for resource '{self.name}'")
        else:
            raise ValueError(f"Unknown type '{self.type}' for resource '{self.name}'")

        # ── Capacity: capacity_per_unit_per_clk × num_units × clock_hz ──
        cap_per_unit = float(cfg["capacity_per_unit_per_clk"])
        num_units = cfg["num_units"]
        if not math.isfinite(cap_per_unit) or cap_per_unit <= 0:
            raise ValueError(
                f"resource {self.name!r} capacity must be finite and positive")
        if num_units <= 0:
            raise ValueError(
                f"resource {self.name!r} num_units must be positive")
        # Preserve the physical unit count for launch selectors even when the
        # resource is modeled as one shared capacity envelope.  ``num_units``
        # below retains its historical per-unit-capacity meaning and is zero
        # for shared resources; ``total_units`` always means the flattened
        # hardware instance count declared by the component tree.
        self.total_units = num_units
        if self.active_unit_service_time_s is not None:
            for service, model in self.active_unit_service_time_s.items():
                if model["active_units"][-1] != self.total_units:
                    raise ValueError(
                        f"resource {self.name!r} active-unit service "
                        f"{service!r} must end at its physical unit count "
                        f"{self.total_units}")
        self.capacity = cap_per_unit * num_units * self._nominal_clock_hz

        # Clock-independent throughput for per-kernel clock scaling
        self._capacity_per_clk = cap_per_unit * num_units

        # ── Scope-specific fields ──
        if self.scope == "per_unit":
            cap_per_unit = float(cfg["capacity_per_unit_per_clk"])
            self.capacity_per_unit = cap_per_unit * self._nominal_clock_hz
            self._capacity_per_unit_per_clk = cap_per_unit
            self.num_units = cfg["num_units"]
        elif self.scope == "shared":
            self.capacity_per_unit = 0
            self._capacity_per_unit_per_clk = 0
            self.num_units = 0
        else:
            raise ValueError(f"Unknown scope '{self.scope}' for resource '{self.name}'")

    def effective_capacity(self, active_units: int,
                           sm_freq_hz: float, dram_freq_hz: float):
        """Return effective capacity, scaled by active units and per-kernel clock.

        Args:
            active_units: number of active SMs (for per_unit scope)
            sm_freq_hz: actual SM frequency for this kernel
            dram_freq_hz: actual DRAM frequency for this kernel
        """
        if self._runtime_clock_source == "absolute":
            raise NotImplementedError("aggregate roofline resource has no native clock or occupancy model")
        if self._runtime_clock_source == "model":
            actual_clock = sm_freq_hz
        elif self._runtime_clock_source == "dram":
            actual_clock = dram_freq_hz
        elif self._runtime_clock_source == "fixed":
            actual_clock = self._nominal_clock_hz
        else:
            raise RuntimeError(
                f"unknown runtime clock source: {self._runtime_clock_source!r}")
        
        if self.scope == "per_unit":
            return (
                self._capacity_per_unit_per_clk * actual_clock
                * min(active_units, self.num_units))
        elif self.scope == "shared":
            return self._capacity_per_clk * actual_clock
        else:
            raise RuntimeError(f"unknown resource scope: {self.scope!r}")

    def effective_service_capacity(
            self, service: str, *, active_units: int,
            execution_domain_units: int) -> float:
        """Return one typed service's active SI operation rate.

        Configured rates are per physical resource instance per second.  The
        active execution-domain fraction selects the corresponding fraction
        of the resource's flattened physical instances without converting
        through a guessed runtime clock.
        """

        if self.type != "compute":
            raise ValueError(
                f"resource {self.name!r} is not a compute resource")
        if (self.service_rates_per_unit_s is None
                and self.active_unit_service_time_s is None):
            raise ValueError(
                f"resource {self.name!r} has no typed service rates")
        if (isinstance(active_units, bool)
                or not isinstance(active_units, int)
                or active_units <= 0):
            raise ValueError("active_units must be a positive integer")
        if (isinstance(execution_domain_units, bool)
                or not isinstance(execution_domain_units, int)
                or execution_domain_units <= 0):
            raise ValueError(
                "execution_domain_units must be a positive integer")
        active_domain_units = min(active_units, execution_domain_units)
        active_resource_units = (
            self.total_units * active_domain_units / execution_domain_units)
        if not active_resource_units.is_integer():
            raise ValueError(
                f"resource {self.name!r} has a fractional active typed "
                "service worker count")
        active_resource_units = int(active_resource_units)
        if (self.service_rates_per_unit_s is not None
                and service in self.service_rates_per_unit_s):
            return (
                self.service_rates_per_unit_s[service]
                * active_resource_units)
        if (self.active_unit_service_time_s is not None
                and service in self.active_unit_service_time_s):
            model = self.active_unit_service_time_s[service]
            service_time_s = _linear_interpolate(
                model["active_units"], model["service_time_s"],
                active_resource_units)
            return active_resource_units / service_time_s
        raise ValueError(
            f"resource {self.name!r} has no typed service {service!r}")

    def has_typed_service(self, service: str) -> bool:
        """Return whether a fixed-rate or active-unit service is declared."""

        return (
            (self.service_rates_per_unit_s is not None
             and service in self.service_rates_per_unit_s)
            or (self.active_unit_service_time_s is not None
                and service in self.active_unit_service_time_s)
        )

    def service_rate_for_one_unit(self, service: str) -> float:
        """Return one physical resource instance's typed SI operation rate."""

        if self.type != "compute":
            raise ValueError(
                f"resource {self.name!r} is not a compute resource")
        if self.service_rates_per_unit_s is None:
            raise ValueError(
                f"resource {self.name!r} has no typed service rates")
        if service not in self.service_rates_per_unit_s:
            raise ValueError(
                f"resource {self.name!r} has no typed service {service!r}")
        return self.service_rates_per_unit_s[service]

    def setup_time_for_service(self, service: str) -> float:
        """Return seconds paid by one serial intrinsic invocation."""

        if self.type != "compute":
            raise ValueError(
                f"resource {self.name!r} is not a compute resource")
        if self.service_setup_time_s is None:
            raise ValueError(
                f"resource {self.name!r} has no typed service setup times")
        if service not in self.service_setup_time_s:
            raise ValueError(
                f"resource {self.name!r} has no setup time for typed "
                f"service {service!r}")
        return self.service_setup_time_s[service]

    def exact_request_service_time_s(
            self, service: str, request_size: int) -> float:
        """Return one exact measured mechanism service in seconds/unit."""
        if self.exact_request_services is None:
            raise ValueError(
                f"resource {self.name!r} has no exact request services")
        if service not in self.exact_request_services:
            raise ValueError(
                f"resource {self.name!r} has no exact request service "
                f"{service!r}")
        if (isinstance(request_size, bool)
                or not isinstance(request_size, int)
                or request_size <= 0):
            raise ValueError("request_size must be a positive integer")
        service_times = self.exact_request_services[service][
            "service_time_s_by_request_size"]
        if request_size not in service_times:
            raise ValueError(
                f"resource {self.name!r} exact request service {service!r} "
                f"has no request size {request_size}")
        return service_times[request_size]

    def __repr__(self):
        return f"Resource({self.name}, cap={self.capacity:.2e}, type={self.type})"


class ResourceGroup:
    """A group of resources sharing hardware. Demands within a group add up."""

    def __init__(self, name, members):
        self.name = name
        self.members = list(members)

    def __repr__(self):
        return f"ResourceGroup({self.name}, members={self.members})"


# Source admission is an explicit software-contract scope, not a physical
# architecture inference or a claim that a runtime/provider was selected.
# These are the existing lowering/source-template domains, owned here so
# common lowering and predictors consume the same hardware declaration.
_SOURCE_CONTRACT_HARDWARE = {
    # Existing source domains only: these declarations move literal device
    # guards out of generic consumers without admitting new source programs.
    'bf16_fp32_symbolic_matmul_source_math': ('nvidia_h200',),
    'bf16_fp8_expert_source_launch_binding': ('nvidia_h200',),
    'bf16_symbolic_matmul_source_math': ('nvidia_h200',),
    'bf16_symbolic_vector_source_math': ('nvidia_h200',),
    'dense_fp16_symbolic_source_math': ('nvidia_h200',),
    'h200_partial_source_programs': ('nvidia_h200',),
    'kimi_grouped_marlin_source_math': ('nvidia_h200',),
    'kimi_marlin_i256_source_math': ('nvidia_h200',),
    'marlin_routed_expert_source_launch_binding': ('nvidia_h200',),
    'mla_decode_source_launch_binding': ('nvidia_h200',),
    'multistage_attention_indexer_unbound_source_math': ('nvidia_h200',),
    'pytorch_2_8_grouped_strided_norm_copy_source': ('nvidia_h200',),
    'unbound_bf16_expert_source_math': ('nvidia_h200',),
    'unbound_grouped_bf16_expert_source_math': ('nvidia_h200',),
    'vllm_0_27_1_v3_l32_tp4_source_projection': ('nvidia_h200',),
    'vllm_0_30_indexer_source_launch_binding': ('nvidia_h200',),
    'vllm_source_vector_launch_binding': ('nvidia_h200',),
    'vllm_unquantized_prepared_expert_source_recipe': ('nvidia_h200',),
    'unbound_clamped_silu_source_math': ('nvidia_h200',),
    'unbound_materialized_gather_source_math': ('nvidia_h200',),
    # Only the validated native EP1 W4A16 public input is reused: its full
    # resource tuple is fixed by exact geometry/launch/other attributes and
    # routed rows, active experts and padded M rows. Live HW and source data
    # guards remain mandatory; this admission adds no rate or pricing rule.
    'vllm_0_30_marlin_w4a16_ep1_public_resource_reuse': ('nvidia_h200',),
    'kimi_ep_marlin_swiglu_math': ('nvidia_a800', 'nvidia_h200'),
    'ascend_0_18_moe_dispatch_math': ('ascend_910b',),
    'aggregate_gpu_source_roofline': ('hygon_bw1100', 'nvidia_rtx_pro_5500'),
    'programmed_mxfp4_fp8_expert_service': ('intel_agmf039r47',),
    'aten_advanced_indexing_long_math': ('nvidia_a800', 'nvidia_h200', 'metax_c500'),
    'vllm_0_11_embedding_index_select_math': ('nvidia_a800', 'nvidia_h200'),
    'vllm_0_30_v32_target_fusion_math': ('nvidia_a800', 'nvidia_h200'),
    'glm_tp8_pp4_ep8_source_projection': ('nvidia_a800',),
    'glm_tp8_pp2_ep1_source_projection': ('nvidia_h200',),
    'mla_tp8_ep8_source_projection': ('nvidia_h200',),
    'kimi_tp8_pp2_ep8_source_projection': ('nvidia_a800',),
    'kimi_tp1_source_projection': ('nvidia_a800', 'nvidia_h200'),
    'kimi_k3_pp16_source_projection': ('nvidia_h200',),
    'kimi_tp1_unbound_source_lowering': ('nvidia_a800',),
    'target_tp1_representative_source_projection': ('nvidia_a800',),
    'pytorch_2_8_embedding_index_select_launch': ('nvidia_a800', 'nvidia_h200'),
    'pytorch_2_8_sm90_row_gather_launch': ('nvidia_h200',),
    'pytorch_2_8_sm90_int32_int64_launch': ('nvidia_h200',),
    'pytorch_2_8_sm80_logits_cast_launch': ('nvidia_a800',),
    'pytorch_2_8_sm90_logits_cast_launch': ('nvidia_h200',),
    'pytorch_2_8_sm90_int64_int32_launch': ('nvidia_h200',),
    'pytorch_2_8_sm80_int64_int32_launch': ('nvidia_a800',),
    'pytorch_2_8_sm90_argmax_source_launch': ('nvidia_h200',),
    'pytorch_2_4_maca_materialized_gather_launch': ('metax_c500',),
    'pytorch_2_4_maca_int32_int64_launch': ('metax_c500',),
    'pytorch_2_4_maca_scalar_index_launch': ('metax_c500',),
    'pytorch_2_4_maca_logits_cast_launch': ('metax_c500',),
    'pytorch_2_4_maca_small_int64_int32_launch': ('metax_c500',),
    'aten_advanced_indexing_long_unbound_lowering': ('nvidia_h200', 'metax_c500'),
    'vllm_0_11_embedding_unbound_index_select': ('nvidia_h200',),
    'symbolic_fp16_unbound_source_lowering': ('metax_c500',),
    'vllm_0_11_fp32_softmax_topk': ('nvidia_a800', 'nvidia_h200'),
    'vllm_0_24_bf16_softmax_topk': ('metax_c500',),
    'vllm_0_18_bf16_softmax_topk': ('ascend_910b',),
    'vllm_0_18_full_fia_split_fuse': ('ascend_910b',),
    'vllm_0_18_eager_fia_direct_kv': ('ascend_910b',),
    'vllm_0_18_bf16_routed_expert_native_projection': ('ascend_910b',),
    'vllm_0_11_masked_embedding_pipeline_endpoint': ('nvidia_a800',),
    'pytorch_2_8_advanced_indexing_int64_launch': ('nvidia_a800',),
    'vllm_0_27_1_symbolic_embedding': ('nvidia_h200',),
    'vllm_0_27_1_symbolic_masked_embedding': ('nvidia_h200',),
    'sm90_tf32_hc_prenorm_gemm_impl': ('nvidia_h200',),
    'symbolic_glm4_sigmoid_topk': ('nvidia_a800', 'nvidia_h200'),
    'symbolic_grouped_routed_experts': ('nvidia_h200',),
    'symbolic_dsv3_routed_experts': ('nvidia_h200',),
    'symbolic_mla_cache_update': ('nvidia_h200',),
    'symbolic_mla_context_prefill_mathematical': ('nvidia_h200',),
    'symbolic_mla_decode_mathematical': ('nvidia_h200',),
    'symbolic_mla_fresh_prefill_mathematical': ('nvidia_h200',),
    'symbolic_mtp_eager': ('nvidia_a800',),
    'symbolic_mtp_position_mask': ('nvidia_a800',),
    'symbolic_softmax_topk': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'symbolic_unquantized_experts': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'vllm_0_11_symbolic_masked_embedding': ('nvidia_a800', 'nvidia_h200'),
    'vllm_0_18_symbolic_embedding': ('ascend_910b',),
    'vllm_0_18_symbolic_masked_embedding': ('ascend_910b',),
    'vllm_0_18_symbolic_logits_boundary': ('ascend_910b',),
    'vllm_0_18_symbolic_plain_greedy': ('ascend_910b',),
    'vllm_0_24_symbolic_embedding': ('metax_c500',),
    'vllm_0_24_symbolic_masked_embedding': ('metax_c500',),
    'vllm_0_24_symbolic_logits_boundary': ('metax_c500',),
    'vllm_0_24_symbolic_plain_greedy': ('metax_c500',),
    'sglang_0_5_19_symbolic_grouped_routing': ('nvidia_h200',),
    'sglang_0_5_19_symbolic_fused_shared_moe': ('nvidia_h200',),
    'sglang_0_5_19_symbolic_vocab_embedding': ('nvidia_h200',),
    'sglang_0_5_19_symbolic_logits_boundary': ('nvidia_h200',),
    'sglang_0_5_19_symbolic_one_shot_prefill': ('nvidia_h200',),
    'vllm_0_11_symbolic_embedding': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'vllm_0_11_symbolic_logits_boundary': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'vllm_0_11_symbolic_plain_greedy': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'vllm_0_30_group_fp8_fp32_scales': ('nvidia_h200',),
    'vllm_0_30_fa3_gqa_fresh_prefill': ('nvidia_h200',),
    'vllm_0_30_fa3_sparse_mla': ('nvidia_h200',),
    'vllm_0_27_1_symbolic_grouped_routing': ('nvidia_h200',),
    'vllm_0_27_1_symbolic_logits_boundary': ('nvidia_h200',),
    'vllm_0_27_1_symbolic_plain_greedy': ('nvidia_h200',),
    'vllm_0_27_1_symbolic_mtp_position_where': ('nvidia_h200',),
    'vllm_0_27_1_symbolic_mtp_eager': ('nvidia_h200',),
    'vllm_0_30_symbolic_mtp_eager': ('nvidia_h200',),
    'vllm_0_30_symbolic_v32_eh_norm': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_fresh_prefill_logits': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_marlin_experts': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_native_marlin_moe': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_nope_context_prefill': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_nope_decode': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_nope_fresh_prefill': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_nope_mixed': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_ordinary_mlp': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_plain_greedy': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_tp1_embedding': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_tp1_native_latent_output': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_triton_context_prefill': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_triton_decode': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_triton_fresh_prefill': ('nvidia_h200',),
    'vllm_0_30_kimi_k3_triton_mixed': ('nvidia_h200',),
    'vllm_0_30_symbolic_dsa': ('nvidia_h200',),
    'vllm_0_30_symbolic_dsa_indexer': ('nvidia_h200',),
    'vllm_0_30_symbolic_dsa_sparse': ('nvidia_h200',),
    'vllm_0_30_symbolic_embedding': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'vllm_0_30_symbolic_grouped_routing': ('nvidia_h200',),
    'vllm_0_30_symbolic_k3_prefill_attention': ('nvidia_h200',),
    'vllm_0_30_symbolic_logits_boundary': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'vllm_0_30_symbolic_masked_embedding': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'vllm_0_30_symbolic_mhc': ('nvidia_h200',),
    'vllm_0_30_symbolic_plain_greedy': ('nvidia_a800', 'nvidia_h200', 'metax_c500', 'ascend_910b'),
    'vllm_0_30_symbolic_v4_compressor': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_grouped_output': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_indexer_query': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_inverse_rope': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_marlin': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_qkv_norm': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_routing': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_scored_indexer': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_short_topk': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_sparse': ('nvidia_h200',),
    'vllm_0_30_symbolic_v4_swa_insert': ('nvidia_h200',),
    'vllm_0_30_symbolic_vocab_embedding': ('nvidia_a800', 'nvidia_h200'),
}

class HWSpec:
    """Hardware specification: resources + groups + scheduling."""

    roofline_only = False
    fpga_execution = None
    gpu_vector_memory_only = False

    def with_roofline_utilization(self, *, compute, bandwidth):
        """Return an effective-rate scenario, preserving clocks, delays and links.

        Whole-device aggregate profiles only. Memory includes local HBM and
        DDR staging, while network links and physical memory sizes stay fixed.
        This explicit scenario does not replace the canonical peak profile.
        """
        from copy import deepcopy
        if not self.roofline_only:
            raise ValueError("utilization scenarios require an aggregate roofline profile")
        for value in (compute, bandwidth):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("roofline utilization must be finite and in (0, 1]")
        scenario = deepcopy(self)
        for resource in scenario.resources:
            if resource.type == "compute":
                resource.capacity *= compute
            elif resource.type == "memory":
                resource.capacity *= bandwidth
            else:
                raise ValueError("aggregate utilization requires compute or memory resources")
        return scenario

    def _initialize_aggregate_roofline(self, cfg):
        from src.kernel.component_tree import ComponentTree
        from src.kernel.hardware_ids import canonical_hardware_id
        self.roofline_only = True
        self.kernel_memory_model = "roofline"
        self.hardware_id = canonical_hardware_id(cfg["hardware_id"])
        self.kernel_dtype = cfg["features"]["kernel_dtype"]
        self.parameter_source = cfg["features"]["parameter_source"]
        self.gpu_vector_memory_only = (cfg["features"]["gpu_vector_memory_only"]
            if "gpu_vector_memory_only" in cfg["features"] else False)
        self.fpga_execution = (dict(cfg["features"]["fpga_execution"])
            if "fpga_execution" in cfg["features"] else None)
        self.async_output_release = cfg["features"].get("async_output_release")
        tree = ComponentTree(cfg, Path(__file__).resolve().parents[2])
        self.resources = [Resource(resource, {}, None, None) for resource in tree.get_flat_resources()]
        self._res_map, self._role_map = {}, {}
        for resource in self.resources:
            if resource.name in self._res_map or resource.role in self._role_map:
                raise ValueError("aggregate roofline resource names and roles must be unique")
            if resource._runtime_clock_source != "absolute":
                raise ValueError("aggregate roofline profile requires SI capacities without invented clocks")
            if resource.type == "memory" and (type(resource.size_bytes) is not int or resource.size_bytes <= 0):
                raise ValueError("aggregate roofline memory requires an explicit positive byte capacity")
            self._res_map[resource.name] = resource
            self._role_map[resource.role] = resource
        if not {"compute_tensor", "dram"}.issubset(self._role_map):
            raise ValueError("aggregate roofline profile requires tensor and DRAM resources")
        if self.hardware_id == "intel_agmf039r47":
            if self.fpga_execution is None:
                raise ValueError("Intel FPGA requires its programmed execution parameters")
            execution = self.fpga_execution
            mac_s = (execution["clock_hz"] * execution["mm_m_parallel"]
                * execution["mm_k_parallel"] * execution["mm_n_parallel"])
            for role in ("compute_tensor", "compute_tensor_fp16", "compute_tensor_fp8"):
                if self.resource_by_role(role).capacity != 2 * mac_s:
                    raise ValueError("FPGA alternate matrix formats must share the same single MM core capacity")
            if self.resource_by_role("compute_vector").capacity != execution["clock_hz"] * execution["vector_elements_per_cycle"]:
                raise ValueError("FPGA vector element capacity must match its clock and parallelism")
            self.resource_by_role("ddr")
        self.resource_groups = []
        self.interconnect = {}
        if cfg["interconnect"]:
            self.interconnect = dict(cfg["interconnect"])
            self.interconnect["roce_bw_bytes_s"] = float(cfg["interconnect"]["roce_unidir_bw_GB_s"]) * 1e9

    @staticmethod
    def source_contract_supported(source_contract: str, hardware_id: str) -> bool:
        """Whether a named source program has an existing hardware binding.

        This preserves explicit integration domains. It does not infer them
        from throughput, memory architecture or the success of a selector.
        Unknown source programs fail closed rather than borrowing a binding.
        The static form also serves metadata-only kernel descriptions that
        intentionally do not instantiate or load an HWSpec.
        """
        from src.kernel.hardware_ids import canonical_hardware_id
        if source_contract not in _SOURCE_CONTRACT_HARDWARE:
            raise ValueError(f"unknown source contract: {source_contract!r}")
        return canonical_hardware_id(hardware_id) in _SOURCE_CONTRACT_HARDWARE[source_contract]

    def supports_source_contract(self, source_contract: str) -> bool:
        return self.source_contract_supported(source_contract, self.hardware_id)







    def require_v030_fa3_gqa_prefill_source_launch(self, kernel) -> None:
        """Validate a supplied source launch without selecting a new one."""

        if not self.supports_source_contract("vllm_0_30_fa3_gqa_fresh_prefill"):
            raise ValueError("hardware has no vLLM 0.30 FA3 GQA source binding")
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
            validate_v030_gqa_fresh_prefill_launch_contract)
        validate_v030_gqa_fresh_prefill_launch_contract(
            kernel.shape, kernel.dtype, kernel.variant, kernel.attributes, self,
            kernel.launch)

    def fa3_gqa_observed_source_splits(self, kernel, *, block_m, block_n):
        """Validate the hardware-owned FA3 program and derive its split facts."""
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
            validate_v030_gqa_observed_launch, v030_gqa_dynamic_splits)
        validate_v030_gqa_observed_launch(kernel, self)
        return v030_gqa_dynamic_splits(
            (kernel.attributes["model_input_q_lens"] if "source_program" in kernel.attributes
                and kernel.attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode" else kernel.attributes["q_lens"]),
            (kernel.attributes["model_input_kv_lens"] if "source_program" in kernel.attributes
                and kernel.attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode" else kernel.attributes["kv_lens"]),
            ratio=kernel.shape["num_query_heads"] // kernel.shape["num_kv_heads"],
            kv_heads=kernel.shape["num_kv_heads"], block_m=block_m, block_n=block_n,
            split_cap=kernel.launch.tiling.num_kv_splits,
            num_sms=int(self.num_compute_units), allow_source_padding=("source_program" in kernel.attributes
                and kernel.attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode"))

    def fa3_prepare_varlen_launch(self, batch):
        """Forward the public metadata launch to its bound FA3 source owner."""
        if not self.supports_source_contract("vllm_0_30_fa3_gqa_fresh_prefill"):
            raise NotImplementedError("FA3 prepare source is not bound on this hardware")
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import select_fa3_prepare_varlen_launch
        return select_fa3_prepare_varlen_launch(batch)

    def fa3_gqa_forward_specs(self, shape, dtype, attributes, *, page_size,
                              num_splits, scheduler_metadata_precomputed):
        """Forward explicit caller facts to the hardware-owned FA3 call."""
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import select_v030_gqa_forward_specs
        return select_v030_gqa_forward_specs(
            shape, dtype, attributes, self, page_size=page_size,
            num_splits=num_splits,
            scheduler_metadata_precomputed=scheduler_metadata_precomputed)

    def fa3_gqa_model_call_facts(self, version, *, page_size,
                               graph_configured_mode="NONE", max_capture_tokens=0, query_tokens=None):
        """Pinned ordinary eager default scenario, not observed runtime state.

        vLLM0.30 CacheConfig resolves an unspecified block size to16. Its
        ordinary FA3 metadata builder enables AOT, passes num_splits=0 and
        shares that metadata across layers. PIECEWISE with eager attention
        uses the same caller. A caller configured with FULL graphs passes
        the source-default fixed split count while logical rows fit its
        capture bound, even when this particular attention call runs eager.
        Captured attention, MTP and overrides require separate source facts.
        """
        if (not self.supports_source_contract("vllm_0_30_fa3_gqa_fresh_prefill")
                or version != "vllm_0_30_0"):
            raise ValueError("FA3 GQA source facts require the explicit ordinary vLLM0.30 scenario")
        if page_size is not None and (type(page_size) is not int or page_size < 16 or page_size % 16):
            raise ValueError("FA3 GQA requires the declared aligned page size")
        if graph_configured_mode not in ("NONE","PIECEWISE","FULL","FULL_DECODE_ONLY","FULL_AND_PIECEWISE"):
            raise ValueError("FA3 caller requires an explicit supported graph configuration")
        if type(max_capture_tokens) is not int or max_capture_tokens<0:
            raise ValueError("FA3 caller capture bound must be a nonnegative integer")
        if graph_configured_mode=="NONE":
            if max_capture_tokens!=0 or query_tokens is not None:
                raise ValueError("an eager caller cannot discard supplied graph facts")
        elif type(query_tokens) is not int or query_tokens<=0:
            raise ValueError("FA3 graph-configured caller requires its actual positive logical token count")
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
            FA3_GQA_PREFILL_030_SOURCE_REVISION, FA3_GQA_DEFAULT_GRAPH_SPLITS_030)
        full=graph_configured_mode in ("FULL","FULL_DECODE_ONLY","FULL_AND_PIECEWISE")
        splits=(FA3_GQA_DEFAULT_GRAPH_SPLITS_030 if full and query_tokens<=max_capture_tokens else 0)
        facts={"fa3_gqa_page_size": 16 if page_size is None else page_size,
            "fa3_gqa_page_origin": "vllm030_CacheConfig_default" if page_size is None else "declared_hbm_tier",
            "fa3_gqa_num_splits": splits, "fa3_gqa_scheduler_metadata_precomputed": True,
            "fa3_gqa_source_revision": FA3_GQA_PREFILL_030_SOURCE_REVISION,
            "fa3_gqa_caller_scenario": "vllm030_ordinary_eager_default_aot",
            "fa3_gqa_runtime_binary_bound": False}
        if graph_configured_mode!="NONE":
            facts.update({"fa3_gqa_caller_scenario":"vllm030_graph_configured_eager_attention_default_aot",
                "fa3_gqa_graph_configured_mode":graph_configured_mode,
                "fa3_gqa_max_capture_tokens":max_capture_tokens,"fa3_gqa_query_tokens":query_tokens})
        return facts

    def fa3_mla_forward_specs(self, shape, dtype, variant, attributes, *,
                              scheduler_metadata_precomputed):
        """Forward one FA3 MLA call, not another attention provider."""
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import select_fa3_mla_forward_specs
        return select_fa3_mla_forward_specs(
            shape, dtype, variant, attributes, self,
            scheduler_metadata_precomputed=scheduler_metadata_precomputed)

    def fa3_sparse_mla_forward_specs(self, shape, dtype, attributes, *, topk_capacity):
        """Forward actual sparse caller capacity separately from used lengths."""
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import select_v030_sparse_mla_forward_specs
        return select_v030_sparse_mla_forward_specs(
            shape, dtype, attributes, self, topk_capacity=topk_capacity)

    def require_fa3_mla_source_revision(self, revision):
        """Check caller provenance against the hardware launch source owner."""
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
            FA3_SOURCE_REVISION, FA3_GQA_PREFILL_030_SOURCE_REVISION, FA3_MLA_PREFILL_027_SOURCE_REVISION)
        if self.hardware_id != "nvidia_h200" or revision not in (
                FA3_SOURCE_REVISION, FA3_GQA_PREFILL_030_SOURCE_REVISION, FA3_MLA_PREFILL_027_SOURCE_REVISION):
            raise ValueError("FA3 MLA caller source revision does not match its hardware program")

    def fa3_mla_source_revision_for_model(self, version):
        """Identity for an explicit versioned default Hopper source scenario.

        Equal SM90 device sources and equivalent eager host arguments permit
        source interpretation, without claiming installed binary/rate identity.
        """
        from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
            FA3_GQA_PREFILL_030_SOURCE_REVISION, FA3_MLA_PREFILL_027_SOURCE_REVISION)
        if self.hardware_id != "nvidia_h200" or version not in ("vllm_0_30_0", "vllm_0_27_1"):
            raise ValueError("FA3 model source requires an explicit supported H200 version scenario")
        return {"vllm_0_30_0": FA3_GQA_PREFILL_030_SOURCE_REVISION,
            "vllm_0_27_1": FA3_MLA_PREFILL_027_SOURCE_REVISION}[version]

    def fa3_mla_model_backend_source(self, version):
        """Name the declared source caller; never infer an installed provider."""
        self.fa3_mla_source_revision_for_model(version)
        return {"vllm_0_30_0": "vllm_0_30_default_hopper_flash_attn3",
            "vllm_0_27_1": "vllm_0_27_1_default_hopper_flash_attn3"}[version]

    def source_embedding_index_select_launch_domain(self, rows, hidden, vocab, elements):
        """Domain of the already-bound Torch embedding launch, without rederivation."""
        if not self.supports_source_contract("pytorch_2_8_embedding_index_select_launch"):
            raise NotImplementedError("hardware has no source embedding index-select launch")
        from src.kernel.launch_config.selection_primitives import pytorch_2_8_embedding_gather_domain
        return pytorch_2_8_embedding_gather_domain(rows, hidden, vocab, elements, int(self.num_compute_units))

    def source_routing_variant(self, source_contract: str) -> str:
        """Discrete router name of the existing unquantized expert source."""
        if source_contract != "symbolic_unquantized_experts":
            raise ValueError(f"source has no declared routing variant: {source_contract!r}")
        if not self.supports_source_contract(source_contract):
            raise NotImplementedError("hardware has no unquantized expert source binding")
        return "moe_gating_topk" if self.hardware_id == "ascend_910b" else "moe_topk"

    def analytical_link_bandwidth_bytes_s(self, link: str) -> float:
        """Resolve physical link identity to its retained bandwidth field.

        The historical generic field stores independently measured MetaXLink
        and HCCS bandwidth on their own hardware. This mapping never selects
        NVLink-specific synchronization or collective services for them.
        """
        if self.roofline_only:
            if link == "roce" and self.interconnect:
                return self.interconnect["roce_bw_bytes_s"]
            if link == "pcie" and "pcie_bw_bytes_s" in self.interconnect:
                return float(self.interconnect["pcie_bw_bytes_s"])
            raise NotImplementedError("aggregate roofline profile has no declared interconnect bandwidth")
        if link == "nvlink":
            bandwidth_key = "nvlink_bw_bytes_s"
        elif link == "metaxlink":
            if self.hardware_id != "metax_c500":
                raise ValueError("MetaXLink bandwidth belongs to C500")
            bandwidth_key = "nvlink_bw_bytes_s"
        elif link == "hccs":
            if self.hardware_id != "ascend_910b":
                raise ValueError("HCCS bandwidth belongs to Ascend910B")
            bandwidth_key = "nvlink_bw_bytes_s"
        elif link == "infiniband":
            bandwidth_key = "infiniband_bw_bytes_s"
        else:
            raise ValueError(
                "analytical COMMUNICATION requires nvlink, metaxlink, hccs or infiniband, "
                f"got {link!r}")
        value = self.interconnect[bandwidth_key]
        if value is None:
            raise KeyError(f"COMMUNICATION requires interconnect.{bandwidth_key}")
        return float(value)

    def _declares_memory_model(self, hardware_id):
        """Whether this spec loads the named hardware's memory model.

        Every service/gate a hardware package owns is conditioned on the same
        pair: the canonical hardware id and the declared
        ``features.kernel_memory_model`` of that card.  Both live in the
        hardware's own YAML, so the condition is data, not a branch -- this
        helper only names it once instead of repeating the literal.
        """

        return (self.hardware_id == hardware_id
                and self._source_config["features"]["kernel_memory_model"]
                == _MEMORY_MODEL_BY_HARDWARE[hardware_id])

    def __init__(self, yaml_path):
        if isinstance(yaml_path, dict):
            from copy import deepcopy
            cfg = deepcopy(yaml_path)
        else:
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
        _validate_top_level_config(cfg)
        # Keep the validated source mapping available to launch-contract
        # consumers in this interpreter.  It prevents a second YAML parse of
        # the same immutable hardware contract before evaluation workers fork.
        self._source_config = cfg
        if cfg["features"]["kernel_memory_model"] == "roofline":
            self._initialize_aggregate_roofline(cfg)
            return

        from src.kernel.hardware_ids import canonical_hardware_id
        self.hardware_id = canonical_hardware_id(cfg["hardware_id"])

        # Clock domains (defined once, referenced by resources)
        clock_domains = cfg["clock_domains"]
        sched = cfg["scheduling"]
        model_clock_domain = sched["model_clock_domain"]
        dram_clock_domain = sched["dram_clock_domain"]
        for domain in (model_clock_domain, dram_clock_domain):
            if domain not in clock_domains:
                raise ValueError(
                    f"scheduling references unknown clock domain: {domain!r}")

        from src.kernel.component_tree import ComponentTree
        repository_root = Path(__file__).resolve().parents[2]
        tree = ComponentTree(cfg, repository_root)
        self.resources = [
            Resource(
                resource, clock_domains,
                model_clock_domain, dram_clock_domain)
            for resource in tree.get_flat_resources()
        ]
        if any(resource._runtime_clock_source == "absolute" for resource in self.resources):
            raise ValueError("absolute resource capacities require an aggregate roofline profile")
        resource_names = [resource.name for resource in self.resources]
        if len(resource_names) != len(set(resource_names)):
            raise ValueError("hardware config contains duplicate resource names")
        self._res_map = {r.name: r for r in self.resources}
        self._role_map = {}
        for r in self.resources:
            if r.role in self._role_map:
                raise ValueError(f"duplicate resource role: {r.role!r}")
            self._role_map[r.role] = r
        required_roles = {
            "compute_tensor", "compute_fma", "compute_sfu", "compute_alu",
            "smem", "l1_cache", "xbar", "l2_cache", "dram",
        }
        missing_roles = required_roles - set(self._role_map)
        if missing_roles:
            raise ValueError(
                f"hardware config missing resource roles: {sorted(missing_roles)}")

        # Resource groups
        self.resource_groups = []
        grouped_resources = set()
        group_names = set()
        for g in cfg["resource_groups"]:
            _require_exact_fields("resource_group", g, {"name", "members"})
            if g["name"] in group_names:
                raise ValueError(f"duplicate resource group: {g['name']!r}")
            group_names.add(g["name"])
            if not isinstance(g["members"], list) or not g["members"]:
                raise ValueError(
                    f"resource group {g['name']!r} requires members")
            if len(g["members"]) != len(set(g["members"])):
                raise ValueError(
                    f"resource group {g['name']!r} contains duplicate members")
            unknown_members = set(g["members"]) - set(self._res_map)
            if unknown_members:
                raise ValueError(
                    f"resource group {g['name']!r} has unknown members: "
                    f"{sorted(unknown_members)}")
            duplicate_members = grouped_resources.intersection(g["members"])
            if duplicate_members:
                raise ValueError(
                    "resources cannot belong to multiple groups: "
                    f"{sorted(duplicate_members)}")
            rg = ResourceGroup(g["name"], g["members"])
            self.resource_groups.append(rg)
            grouped_resources.update(rg.members)

        # Resources not in any explicit group become their own group
        for r in self.resources:
            if r.name not in grouped_resources:
                self.resource_groups.append(ResourceGroup(r.name, [r.name]))

        # Scheduling
        self.parallel_unit = sched["parallel_unit"]
        if self.parallel_unit not in ("warp", "thread"):
            raise ValueError(
                f"unknown scheduling.parallel_unit: {self.parallel_unit!r}")
        integer_scheduling_fields = (
            "parallel_width", "num_compute_units", "max_warps_per_unit",
            "max_blocks_per_unit", "max_regs_per_unit",
            "max_smem_per_unit_bytes", "max_smem_per_block_bytes",
            "max_threads_per_unit",
        )
        for name in integer_scheduling_fields:
            if (isinstance(sched[name], bool)
                    or not isinstance(sched[name], int)):
                raise TypeError(f"scheduling.{name} must be an integer")
        self.parallel_width = sched["parallel_width"]
        self.num_compute_units = sched["num_compute_units"]
        self.max_warps_per_unit = sched["max_warps_per_unit"]
        self.max_blocks_per_unit = sched["max_blocks_per_unit"]
        self.max_regs_per_unit = sched["max_regs_per_unit"]
        self.max_smem_per_unit_bytes = sched["max_smem_per_unit_bytes"]
        self.max_smem_per_block_bytes = sched["max_smem_per_block_bytes"]
        self.max_threads_per_unit = sched["max_threads_per_unit"]
        # Registers are allocated to a warp in fixed-size units (CUDA
        # occupancy calculator); cards that publish no unit allocate exactly.
        if "register_allocation_unit_per_warp" in sched:
            unit = sched["register_allocation_unit_per_warp"]
            if isinstance(unit, bool) or not isinstance(unit, int) or unit <= 0:
                raise ValueError(
                    "scheduling.register_allocation_unit_per_warp must be a "
                    "positive integer")
            self.register_allocation_unit_per_warp = unit
        else:
            self.register_allocation_unit_per_warp = 1
        if (isinstance(sched["dispatch_overhead_ns"], bool)
                or not isinstance(sched["dispatch_overhead_ns"], (int, float))):
            raise TypeError("scheduling.dispatch_overhead_ns must be numeric")
        self.dispatch_overhead_ns = float(sched["dispatch_overhead_ns"])
        if "empty_kernel_duration_ns" in sched:
            value = sched["empty_kernel_duration_ns"]
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))):
                raise TypeError(
                    "scheduling.empty_kernel_duration_ns must be numeric")
            self.empty_kernel_duration_ns = float(value)
        else:
            self.empty_kernel_duration_ns = None
        if "empty_kernel_duration_lut" in sched:
            self.empty_kernel_duration_lut = (
                _normalize_empty_kernel_duration_lut(
                    sched["empty_kernel_duration_lut"]))
        else:
            self.empty_kernel_duration_lut = None
        if "launch_latency_model" in sched:
            self.launch_latency_model = _normalize_launch_latency_model(
                sched["launch_latency_model"])
        else:
            self.launch_latency_model = None
        if "cta_dispatch_cost" in sched:
            self.cta_dispatch_cost = _normalize_cta_dispatch_cost(
                sched["cta_dispatch_cost"])
        else:
            self.cta_dispatch_cost = None
        if "fa2_source_task_outer_envelope" in sched:
            self.fa2_source_task_outer_envelope = (
                _normalize_fa2_source_task_outer_envelope(
                    sched["fa2_source_task_outer_envelope"]))
        else:
            self.fa2_source_task_outer_envelope = None
        if "memory_dependency_task_latency_model" in sched:
            self.memory_dependency_task_latency_model = (
                _normalize_memory_dependency_task_latency_model(
                    sched["memory_dependency_task_latency_model"]))
        else:
            self.memory_dependency_task_latency_model = None
        if "rmsnorm_shared_gamma_l1_residency" in sched:
            self.rmsnorm_shared_gamma_l1_residency = (
                _normalize_rmsnorm_shared_gamma_l1_residency(
                    sched["rmsnorm_shared_gamma_l1_residency"]))
        else:
            self.rmsnorm_shared_gamma_l1_residency = None
        if "add_rmsnorm_source_round_pipeline" in sched:
            self.add_rmsnorm_source_round_pipeline = (
                _normalize_add_rmsnorm_source_round_pipeline(
                    sched["add_rmsnorm_source_round_pipeline"]))
        else:
            self.add_rmsnorm_source_round_pipeline = None
        if "add_rmsnorm_block256_source_round_pipeline" in sched:
            self.add_rmsnorm_block256_source_round_pipeline = (
                _normalize_add_rmsnorm_block256_source_round_pipeline(
                    sched["add_rmsnorm_block256_source_round_pipeline"]))
        else:
            self.add_rmsnorm_block256_source_round_pipeline = None
        if "grouped_gemm_tile_pipeline" in sched:
            self.grouped_gemm_tile_pipeline = (
                _normalize_grouped_gemm_tile_pipeline(
                    sched["grouped_gemm_tile_pipeline"]))
        else:
            self.grouped_gemm_tile_pipeline = None
        if "grouped_gemm_fp8_block_tile_pipelines" in sched:
            self.grouped_gemm_fp8_block_tile_pipelines = (
                _normalize_grouped_gemm_fp8_block_tile_pipelines(
                    sched["grouped_gemm_fp8_block_tile_pipelines"],
                    repository_root))
        else:
            self.grouped_gemm_fp8_block_tile_pipelines = None
        if "sglang_fp8_block_tile_pipelines" in sched:
            sglang_rows = _normalize_sglang_fp8_block_tile_pipelines(
                sched["sglang_fp8_block_tile_pipelines"], repository_root,
                self.hardware_id, self.grouped_gemm_fp8_block_tile_pipelines)
            self.grouped_gemm_fp8_block_tile_pipelines = {
                **self.grouped_gemm_fp8_block_tile_pipelines, **sglang_rows}
        for _pipeline_key in METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS:
            if _pipeline_key in sched:
                setattr(self, _pipeline_key,
                        _normalize_metax_grouped_operand_pipeline(
                            _pipeline_key, sched[_pipeline_key],
                            repository_root))
            else:
                setattr(self, _pipeline_key, None)
        if "dram_finite_read_stream_ramp" in sched:
            self.dram_finite_read_stream_ramp = (
                _normalize_dram_finite_read_stream_ramp(
                    sched["dram_finite_read_stream_ramp"],
                    repository_root, self.hardware_id))
        else:
            self.dram_finite_read_stream_ramp = None
        if "early_exit_cta_retire" in sched:
            self.early_exit_cta_retire = _normalize_early_exit_cta_retire(
                sched["early_exit_cta_retire"], repository_root,
                self.hardware_id)
        else:
            self.early_exit_cta_retire = None
        # wp10 (H200 FA3 MLA): optional FlashAttention-3 mechanism services.
        if "fa3_mla_mechanisms" in sched:
            self.fa3_mla_mechanisms = _normalize_fa3_mla_mechanisms(
                sched["fa3_mla_mechanisms"], repository_root,
                self.hardware_id)
        else:
            self.fa3_mla_mechanisms = None
        self.bf16_wgmma_tensor_service_clock = (
            _normalize_bf16_wgmma_tensor_service_clock(
                sched["bf16_wgmma_tensor_service_clock"], repository_root, self.hardware_id)
            if "bf16_wgmma_tensor_service_clock" in sched else None)
        self.fa3_parameter_reads = (
            _normalize_fa3_parameter_reads(sched["fa3_parameter_reads"], repository_root, self.hardware_id)
            if "fa3_parameter_reads" in sched else None)
        self.fa3_gqa_wgmma_batches = (
            _normalize_fa3_gqa_wgmma_batches(sched["fa3_gqa_wgmma_batches"], repository_root,
                self.hardware_id, self.bf16_wgmma_tensor_service_clock)
            if "fa3_gqa_wgmma_batches" in sched else None)
        self.packed_bf16_conversion = (
            _normalize_packed_bf16_conversion(sched["packed_bf16_conversion"], repository_root,
                self.hardware_id) if "packed_bf16_conversion" in sched else None)
        self.moe_align_shared_atomic_e128_ops_per_s = (
            _normalize_moe_align_shared_atomic_e128(
                sched["moe_align_shared_atomic_e128"], repository_root, self.hardware_id)
            if "moe_align_shared_atomic_e128" in sched else None)
        self.moe_align_fill_bytes_per_s = (
            _normalize_moe_align_fill_bytes_per_s(
                sched["moe_align_fill_bytes_per_s"], repository_root, self.hardware_id)
            if "moe_align_fill_bytes_per_s" in sched else None)
        self.moe_activation_mixed_stream = (
            _normalize_moe_activation_mixed_stream(
                sched["moe_activation_mixed_stream"], repository_root, self.hardware_id)
            if "moe_activation_mixed_stream" in sched else None)
        self.ss_ex2_busy_sharing = (
            _normalize_ss_ex2_busy_sharing(sched["ss_ex2_busy_sharing"], repository_root,
                self.hardware_id) if "ss_ex2_busy_sharing" in sched else None)
        self.warp_integer_dependencies = (
            _normalize_warp_integer_dependencies(sched["warp_integer_dependencies"], repository_root,
                self.hardware_id) if "warp_integer_dependencies" in sched else None)
        self.cp_async_mbarrier_notification = (
            _normalize_cp_async_mbarrier_notification(sched["cp_async_mbarrier_notification"],
                repository_root, self.hardware_id)
            if "cp_async_mbarrier_notification" in sched else None)
        if "fa3_gqa_compiled_dependencies" in sched:
            self.fa3_gqa_compiled_dependencies, self.fa3_gqa_compiled_online_prefix = (
                _normalize_fa3_gqa_compiled_dependencies(
                    sched["fa3_gqa_compiled_dependencies"], repository_root, self.hardware_id))
        else:
            self.fa3_gqa_compiled_dependencies = self.fa3_gqa_compiled_online_prefix = None
        self.async_copy_cold_ready = (
            _normalize_async_copy_cold_ready(sched["async_copy_cold_ready"],
                repository_root, self.hardware_id)
            if "async_copy_cold_ready" in sched else None)
        if "block_fp8_gemm_tma_ring_service" in sched:
            self.block_fp8_gemm_tma_ring_service = (
                _normalize_block_fp8_gemm_tma_ring_service(
                    sched["block_fp8_gemm_tma_ring_service"],
                    repository_root, self.hardware_id))
        else:
            self.block_fp8_gemm_tma_ring_service = None
        if "cooperative_grid_sync_service" in sched:
            self.cooperative_grid_sync_service = (
                _normalize_cooperative_grid_sync_service(
                    sched["cooperative_grid_sync_service"], repository_root,
                    self.hardware_id))
        else:
            self.cooperative_grid_sync_service = None
        if "split_merge_chunk_service" in sched:
            self.split_merge_chunk_service_s = (
                _normalize_split_merge_chunk_service(
                    sched["split_merge_chunk_service"], repository_root,
                    self.hardware_id))
        else:
            self.split_merge_chunk_service_s = None
        if "stream_memcpy_service" in sched:
            self.stream_memcpy_service = _normalize_stream_memcpy_service(
                sched["stream_memcpy_service"], repository_root,
                self.hardware_id)
        else:
            self.stream_memcpy_service = None
        if "cuda_graph_side_stream_fork_node_boundary" in sched:
            self.cuda_graph_side_stream_fork_node_increment_s = (
                _normalize_cuda_graph_side_stream_fork_node_boundary(
                    sched["cuda_graph_side_stream_fork_node_boundary"],
                    repository_root, self.hardware_id))
        else:
            self.cuda_graph_side_stream_fork_node_increment_s = None
        if "cuda_graph_small_node_span" in sched:
            self.cuda_graph_small_node_span = (
                _normalize_cuda_graph_small_node_span(
                    sched["cuda_graph_small_node_span"], repository_root,
                    self.hardware_id))
        else:
            self.cuda_graph_small_node_span = None
        if "small_kernel_span_floor" in sched:
            self.small_kernel_span_floor = (
                _normalize_small_kernel_span_floor(
                    sched["small_kernel_span_floor"],
                    repository_root, self.hardware_id))
        else:
            self.small_kernel_span_floor = None
        if "l2_operand_residency_curve" in sched:
            self.l2_operand_residency_curve = (
                _normalize_l2_operand_residency_curve(
                    sched["l2_operand_residency_curve"],
                    repository_root, self.hardware_id))
        else:
            self.l2_operand_residency_curve = None
        if "finite_async_operand_stream_completion" in sched:
            self.finite_async_operand_stream_completion = (
                _normalize_finite_async_operand_stream_completion(
                    sched["finite_async_operand_stream_completion"],
                    repository_root, self.hardware_id))
        else:
            self.finite_async_operand_stream_completion = None
        if "finite_mixed_operand_stream_startup" in sched:
            self.finite_mixed_operand_stream_startup = (
                _normalize_finite_mixed_operand_stream_startup(
                    sched["finite_mixed_operand_stream_startup"],
                    repository_root, self.hardware_id))
        else:
            self.finite_mixed_operand_stream_startup = None
        if "gemm_bf16_mma_saturation" in sched:
            self.gemm_bf16_mma_saturation = (
                _normalize_gemm_bf16_mma_saturation(
                    sched["gemm_bf16_mma_saturation"], repository_root))
        else:
            self.gemm_bf16_mma_saturation = None
        if "gemm_tensor_activity_clock" in sched:
            self.gemm_tensor_activity_clock = _normalize_gemm_tensor_activity_clock(
                sched["gemm_tensor_activity_clock"], repository_root, self.hardware_id)
        else:
            self.gemm_tensor_activity_clock = None
        if "gemm_tensor_instruction_service" in sched:
            self.gemm_tensor_instruction_service = _normalize_gemm_tensor_instruction_service(
                sched["gemm_tensor_instruction_service"], repository_root, self.hardware_id)
        else:
            self.gemm_tensor_instruction_service = None
        if "ordinary_u16_store_completion" in sched:
            self.ordinary_u16_store_completion = _normalize_ordinary_u16_store_completion(
                sched["ordinary_u16_store_completion"], repository_root, self)
        else:
            self.ordinary_u16_store_completion = None
        if "metax_fused_moe_bm32_bf16_mma_occupancy" in sched:
            self.metax_fused_moe_bm32_bf16_mma_occupancy = (
                _normalize_metax_fused_moe_bm32_bf16_mma_occupancy(
                    sched["metax_fused_moe_bm32_bf16_mma_occupancy"],
                    repository_root))
        else:
            self.metax_fused_moe_bm32_bf16_mma_occupancy = None
        if "h200_fused_moe_w2_active_sm_clock" in sched:
            self.h200_fused_moe_w2_active_sm_clock = (
                _normalize_h200_fused_moe_w2_active_sm_clock(
                    sched["h200_fused_moe_w2_active_sm_clock"],
                    repository_root, self.hardware_id))
        else:
            self.h200_fused_moe_w2_active_sm_clock = None
        self.prepared_moe_operand_resources = (
            _normalize_prepared_moe_operand_resources(sched["prepared_moe_operand_resources"], repository_root, self.hardware_id)
            if "prepared_moe_operand_resources" in sched else None)
        if "fa_tile16_operand_completion" in sched:
            self.fa_tile16_operand_completion = (
                _normalize_fa_tile16_operand_completion(
                    sched["fa_tile16_operand_completion"], repository_root))
        else:
            self.fa_tile16_operand_completion = None
        self.metax_gemv_wave8_completion_service = (
            _load_metax_gemv_wave8_completion_service(repository_root)
            if (self._declares_memory_model("metax_c500"))
            else None)
        self.metax_gemv_block256_wave8_completion_service = (
            _load_metax_gemv_block256_wave8_completion_service(
                repository_root)
            if (self._declares_memory_model("metax_c500"))
            else None)
        if self.metax_gemv_wave8_completion_service is not None:
            service = self.metax_gemv_wave8_completion_service
            if (service["ap_count"] != self.num_compute_units
                    or service["resident_ctas_per_ap"]
                    != self.max_blocks_per_unit):
                raise ValueError(
                    "MetaX GEMV wave8 completion topology differs from "
                    "the hardware scheduling contract")
        if self.metax_gemv_block256_wave8_completion_service is not None:
            service = self.metax_gemv_block256_wave8_completion_service
            if (service["ap_count"] != self.num_compute_units
                    or service["resident_ctas_per_ap"]
                    * service["waves_per_cta"] != self.max_warps_per_unit):
                raise ValueError(
                    "MetaX block256 GEMV completion topology differs from "
                    "the hardware scheduling contract")
        if "tensor_iterator_copy_task_latency_model" in sched:
            self.tensor_iterator_copy_task_latency_model = (
                _normalize_tensor_iterator_copy_task_latency_model(
                    sched["tensor_iterator_copy_task_latency_model"]))
        else:
            self.tensor_iterator_copy_task_latency_model = None
        if "runtime_task_latency_models" in sched:
            self.runtime_task_latency_models = (
                _normalize_runtime_task_latency_models(
                    sched["runtime_task_latency_models"]))
        else:
            self.runtime_task_latency_models = {}
        if "vector_reduce_sum_latency_lut" in sched:
            self.vector_reduce_sum_latency_lut = (
                _normalize_vector_reduce_sum_latency_lut(
                    sched["vector_reduce_sum_latency_lut"]))
        else:
            self.vector_reduce_sum_latency_lut = None
        if "register_reduction_dependency_cycles" in sched:
            self.register_reduction_dependency_cycles = (
                _normalize_register_reduction_dependency_cycles(
                    sched["register_reduction_dependency_cycles"],
                    repository_root, self.hardware_id))
        else:
            self.register_reduction_dependency_cycles = None
        if "fa2_local_spill_dependency_cycles" in sched:
            self.fa2_local_spill_dependency_cycles = (
                _normalize_fa2_local_spill_dependency_cycles(
                    sched["fa2_local_spill_dependency_cycles"]))
        else:
            self.fa2_local_spill_dependency_cycles = None
        dense_d128_spill = (
            "fa2_dense_d128_causal_local_spill_dependency_cycles")
        if dense_d128_spill in sched:
            self.fa2_dense_d128_causal_local_spill_dependency_cycles = (
                _normalize_fa2_dense_d128_causal_local_spill_dependency_cycles(
                    sched[dense_d128_spill]))
        else:
            self.fa2_dense_d128_causal_local_spill_dependency_cycles = None
        if "cta_barrier_service_cycles" in sched:
            self.cta_barrier_service_cycles = (
                _normalize_cta_barrier_service_cycles(
                    sched["cta_barrier_service_cycles"]))
        else:
            self.cta_barrier_service_cycles = None
        if "sm80_depth3_cta_barrier_service_cycles" in sched:
            self.sm80_depth3_cta_barrier_service_cycles = (
                _normalize_cta_barrier_service_cycles(
                    sched["sm80_depth3_cta_barrier_service_cycles"],
                    owner=(
                        "scheduling."
                        "sm80_depth3_cta_barrier_service_cycles"),
                    model="isolated_sm80_depth3_cta_barrier_service_v1",
                    shared_field="total_shared_bytes",
                    shared_bytes=147456,
                    software_revision="cuda_12.2_sm80",
                    resident_ctas_per_unit=1))
        else:
            self.sm80_depth3_cta_barrier_service_cycles = None
        if "flash_mla_combine_barrier_service_cycles" in sched:
            self.flash_mla_combine_barrier_service_cycles = (
                _normalize_cta_barrier_service_cycles(
                    sched["flash_mla_combine_barrier_service_cycles"],
                    owner=(
                        "scheduling."
                        "flash_mla_combine_barrier_service_cycles"),
                    model=(
                        "isolated_flash_mla_combine_cta_barrier_service_v1"),
                    shared_field="static_shared_bytes",
                    shared_bytes=2560))
        else:
            self.flash_mla_combine_barrier_service_cycles = None
        if "imcontinuous_vec4_reduction_source_service" in sched:
            self.imcontinuous_vec4_reduction_source_service = (
                _normalize_imcontinuous_vec4_reduction_source_service(
                    sched["imcontinuous_vec4_reduction_source_service"]))
        else:
            self.imcontinuous_vec4_reduction_source_service = None
        if "fa2_combine_dependency_cycles" in sched:
            self.fa2_combine_dependency_cycles = (
                _normalize_fa2_combine_dependency_cycles(
                    sched["fa2_combine_dependency_cycles"]))
        else:
            self.fa2_combine_dependency_cycles = None
        if "metax_fa2_combine_dependency_cycles" in sched:
            self.metax_fa2_combine_dependency_cycles = (
                _normalize_metax_fa2_combine_dependency_cycles(
                    sched["metax_fa2_combine_dependency_cycles"]))
        else:
            self.metax_fa2_combine_dependency_cycles = None
        if "fa2_oaccum_load_use_cycles" in sched:
            self.fa2_oaccum_load_use_cycles = (
                _normalize_fa2_oaccum_load_use_cycles(
                    sched["fa2_oaccum_load_use_cycles"]))
        else:
            self.fa2_oaccum_load_use_cycles = None
        if "fia_cross_domain_three_handoff" in sched:
            self.fia_cross_domain_three_handoff = (
                _normalize_fia_cross_domain_three_handoff(
                    sched["fia_cross_domain_three_handoff"]))
        else:
            self.fia_cross_domain_three_handoff = None
        # ``CalcSinnerTopKBegin`` belongs to SparseFlashAttentionMla, not
        # FiaKernelNonQuantMla.  The former sparse-index microbenchmark is
        # retained only as historical evidence and is fail-closed for FIA.
        self.fia_mla_topk_scan_service = None
        self.fia_mla_topk_scan_steady_service = None
        self.fia_aiv_instruction_cold_fill = (
            _load_fia_aiv_instruction_cold_fill(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        if "fia_aiv_sync_all_service" in sched:
            self.fia_aiv_sync_all_service = (
                _normalize_fia_aiv_sync_all_service(
                    sched["fia_aiv_sync_all_service"]))
        else:
            self.fia_aiv_sync_all_service = None
        if "fia_flash_decode_terminal_prelude_service" in sched:
            self.fia_flash_decode_terminal_prelude_service = (
                _normalize_fia_flash_decode_terminal_prelude_service(
                    sched["fia_flash_decode_terminal_prelude_service"]))
        else:
            self.fia_flash_decode_terminal_prelude_service = None
        if "fia_tiling_data_materialization_service" in sched:
            self.fia_tiling_data_materialization_service = (
                _normalize_fia_tiling_data_materialization_service(
                    sched["fia_tiling_data_materialization_service"]))
        else:
            self.fia_tiling_data_materialization_service = None
        if "fia_mixed_task_outer_envelope" in sched:
            self.fia_mixed_task_outer_envelope = (
                _normalize_fia_mixed_task_outer_envelope(
                    sched["fia_mixed_task_outer_envelope"]))
        else:
            self.fia_mixed_task_outer_envelope = None
        if "fia_aiv_workspace_transfer_service" in sched:
            self.fia_aiv_workspace_transfer_service = (
                _normalize_fia_aiv_workspace_transfer_service(
                    sched["fia_aiv_workspace_transfer_service"]))
        else:
            self.fia_aiv_workspace_transfer_service = None
        if "aic_mte2_nd2nz_service" in sched:
            self.aic_mte2_nd2nz_service = (
                _normalize_aic_mte2_nd2nz_service(
                    sched["aic_mte2_nd2nz_service"]))
        else:
            self.aic_mte2_nd2nz_service = None
        self.aic_mte2_paged_gqa128_stride_service = (
            _load_aic_mte2_paged_gqa128_stride_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.split_fuse_mte2_short_tail_service = (
            _load_split_fuse_mte2_short_tail_service(repository_root)
            if self._declares_memory_model("ascend_910b") else None)
        if self.aic_mte2_paged_gqa128_stride_service is not None:
            if self.aic_mte2_nd2nz_service is None:
                raise ValueError(
                    "paged-GQA stride service requires the AIC MTE2 service")
            supplemental = self.aic_mte2_paged_gqa128_stride_service
            if (supplemental["active_aic_groups"]
                    != self.aic_mte2_nd2nz_service[
                        "strided_active_aic_groups"]):
                raise ValueError(
                    "paged-GQA stride active-group axis changed")
            geometry = supplemental["geometry"]
            geometries = dict(
                self.aic_mte2_nd2nz_service["strided_geometries"])
            if geometry in geometries:
                raise ValueError(
                    "paged-GQA stride geometry duplicates configured service")
            geometries[geometry] = {
                "serial_duration_ns": supplemental["serial_duration_ns"],
                "stream_duration_ns": supplemental["stream_duration_ns"],
            }
            self.aic_mte2_nd2nz_service = {
                **self.aic_mte2_nd2nz_service,
                "strided_geometries": geometries,
            }
        self.aic_mte2_fia_mla_strided_service = (
            _load_aic_mte2_fia_mla_strided_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        if (self.aic_mte2_fia_mla_strided_service is not None
                and self.aic_mte2_nd2nz_service is None):
            raise ValueError(
                "FIA MLA strided service requires the AIC MTE2 service")
        self.aic_mte2_nd2nz_33copy_group_service = (
            _load_aic_mte2_nd2nz_33copy_group_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_deterministic_split_k_source_queue_service = (
            _load_aic_deterministic_split_k_source_queue_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_deterministic_split_k_collective_barrier_service = (
            _load_aic_deterministic_split_k_collective_barrier_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_nd2nz_large_contiguous_service = (
            _load_aic_mte2_nd2nz_large_contiguous_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        if (self.aic_mte2_nd2nz_large_contiguous_service is not None
                and self.aic_mte2_nd2nz_service is None):
            raise ValueError(
                "large-contiguous ND2NZ service requires the base service")
        self.aic_mte2_gemm_large_resident_service = (
            _load_aic_mte2_gemm_large_resident_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        if (self.aic_mte2_gemm_large_resident_service is not None
                and self.aic_mte2_nd2nz_service is None):
            raise ValueError(
                "GEMM large-resident ND2NZ service requires the base service")
        self.aic_mte2_gemm_shared_source_queue_service = (
            _load_aic_mte2_gemm_shared_source_queue_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        if (self.aic_mte2_gemm_shared_source_queue_service is not None
                and self.aic_mte2_nd2nz_service is None):
            raise ValueError(
                "GEMM shared MTE2 queue requires the base ND2NZ service")
        if "aic_mte2_operand_pair_stream_service" in sched:
            self.aic_mte2_operand_pair_stream_service = (
                _normalize_aic_mte2_operand_pair_stream_service(
                    sched["aic_mte2_operand_pair_stream_service"]))
        else:
            self.aic_mte2_operand_pair_stream_service = None
        self.aic_mte2_cold_operand_pair_stream_service = (
            _load_aic_mte2_cold_operand_pair_stream_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_cold_n_major_b_stride_service = (
            _load_aic_mte2_cold_n_major_b_stride_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_cold_legacy_k14336_stride_service = (
            _load_aic_mte2_cold_legacy_k14336_stride_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_cold_v3_k1536_n5120_stride_service = (
            _load_aic_mte2_cold_v3_k1536_n5120_stride_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_cold_small_m_stride_service = (
            _load_aic_mte2_cold_small_m_stride_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_finite_stream_completion_service = (
            _load_aic_mte2_finite_stream_completion_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_gemm_source_order_completion_service = (
            _load_aic_mte2_gemm_source_order_completion_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_matmul_v3_split_k_resident_tile_service = (
            _load_aic_matmul_v3_split_k_resident_tile_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_gemm_finite_ingress_completion_service = (
            _load_aic_gemm_finite_ingress_completion_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_gemm_short_pipeline_completion_service = (
            _load_aic_gemm_short_pipeline_completion_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_gemm_persistent_issue_service = (
            _load_aic_mte2_gemm_persistent_issue_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_matmul_v2_dynamic_m_tail_control_service = (
            _load_aic_matmul_v2_dynamic_m_tail_control_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_matmul_v2_base_k_lifecycle_service = (
            _load_aic_matmul_v2_base_k_lifecycle_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_matmul_v2_small_m_base_k_control_service = (
            _load_aic_matmul_v2_small_m_base_k_control_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_gemm_region_completion_services = (
            _load_aic_mte2_gemm_region_completion_services(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else ())
        self.aic_mte2_gemm_region_completion_service = (
            self.aic_mte2_gemm_region_completion_services[0]
            if self.aic_mte2_gemm_region_completion_services else None)
        self.aic_mte2_gemm_region_pipeline_completion_service = (
            _load_aic_mte2_gemm_region_pipeline_completion_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_fia_mha_finite_drain_service = (
            _load_aic_mte2_fia_mha_finite_drain_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_fia_gqa_finite_drain_service = (
            _load_aic_mte2_fia_gqa_finite_drain_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_grouped_matmul_finite_stream_completion_service = (
            _load_aic_grouped_matmul_finite_stream_completion_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_grouped_matmul_k2880_wide_base_tail_completion_service = (
            _load_aic_grouped_matmul_k2880_wide_base_tail_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_grouped_matmul_k2880_output_tile_completion_service = (
            _load_aic_grouped_matmul_k2880_output_tile_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_grouped_matmul_step4_worker_sequence_completion_service = (
            _load_aic_grouped_matmul_step4_worker_sequence_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_grouped_matmul_short_source_prelude_control_service = (
            _load_aic_grouped_matmul_short_source_prelude_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_grouped_matmul_m32_short_source_prelude_control_service = (
            _load_aic_grouped_matmul_m32_short_source_prelude_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_grouped_matmul_entry_metadata_pair_service = (
            _load_aic_grouped_matmul_entry_metadata_pair_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_grouped_matmul_short_source_tail_prelude_control_service = (
            _load_aic_grouped_matmul_short_source_tail_prelude_service(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        legacy_l1_pair_reference = (
            _load_aic_mte2_legacy_l1_pair_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_legacy_double_l0c_pair_service = (
            None if legacy_l1_pair_reference is None
            else legacy_l1_pair_reference["double_l0c_service"])
        self.aic_mte2_legacy_single_l0c_pair_control = (
            None if legacy_l1_pair_reference is None
            else legacy_l1_pair_reference["single_l0c_control"])
        self.aic_mte2_operand_pair_overlap_domain = (
            _load_aic_mte2_operand_pair_overlap_domain(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_operand_pair_finite_overlap_domain = (
            _load_aic_mte2_operand_pair_finite_overlap_domain(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_mte2_mixed_ring_long_overlap_domain = (
            _load_aic_mte2_mixed_ring_long_overlap_domain(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_matmul_v3_asw_mte2_pair_stream_service = (
            _load_aic_matmul_v3_asw_mte2_pair_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        if "aic_mte1_load_pair_completion_service" in sched:
            self.aic_mte1_load_pair_completion_service = (
                _normalize_aic_mte1_load_pair_completion_service(
                    sched["aic_mte1_load_pair_completion_service"]))
        else:
            self.aic_mte1_load_pair_completion_service = None
        if "aic_grouped_matmul_mte1_pair_completion_service" in sched:
            self.aic_grouped_matmul_mte1_pair_completion_service = (
                _normalize_aic_grouped_matmul_mte1_pair_completion_service(
                    sched[
                        "aic_grouped_matmul_mte1_pair_completion_service"]))
        else:
            self.aic_grouped_matmul_mte1_pair_completion_service = None
        if "aic_fixpipe_full_tile_service" in sched:
            self.aic_fixpipe_full_tile_service = (
                _normalize_aic_fixpipe_full_tile_service(
                    sched["aic_fixpipe_full_tile_service"]))
        else:
            self.aic_fixpipe_full_tile_service = None
        self.aic_fixpipe_bf16_terminal_tile_service = (
            _load_aic_fixpipe_bf16_terminal_tile_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_fixpipe_bf16_finite_stream_service = (
            _load_aic_fixpipe_bf16_finite_stream_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aic_fixpipe_bf16_long_stream_service = (
            _load_aic_fixpipe_bf16_long_stream_service(repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        if "fia_block_reduce_sum_service" in sched:
            self.fia_block_reduce_sum_service = (
                _normalize_fia_block_reduce_sum_service(
                    sched["fia_block_reduce_sum_service"]))
        else:
            self.fia_block_reduce_sum_service = None
        if "fia_block_reduce_max_service" in sched:
            self.fia_block_reduce_max_service = (
                _normalize_fia_block_reduce_max_service(
                    sched["fia_block_reduce_max_service"]))
        else:
            self.fia_block_reduce_max_service = None
        if "fia_brcb_service" in sched:
            self.fia_brcb_service = _normalize_fia_brcb_service(
                sched["fia_brcb_service"])
        else:
            self.fia_brcb_service = None
        if "fia_kv_line_fill_span" in sched:
            self.fia_kv_line_fill_span = _normalize_fia_kv_line_fill_span(
                sched["fia_kv_line_fill_span"])
        else:
            self.fia_kv_line_fill_span = None
        if "aic_empty_kernel_duration_lut" in sched:
            self.aic_empty_kernel_duration_lut = (
                _normalize_aic_empty_kernel_duration_lut(
                    sched["aic_empty_kernel_duration_lut"]))
        else:
            self.aic_empty_kernel_duration_lut = None
        if "aiv_empty_kernel_duration_lut" in sched:
            self.aiv_empty_kernel_duration_lut = (
                _normalize_aiv_empty_kernel_duration_lut(
                    sched["aiv_empty_kernel_duration_lut"]))
        else:
            self.aiv_empty_kernel_duration_lut = None
        if "aiv_nonempty_task_latency_model" in sched:
            self.aiv_nonempty_task_latency_model = (
                _normalize_aiv_nonempty_task_latency_model(
                    sched["aiv_nonempty_task_latency_model"]))
        else:
            self.aiv_nonempty_task_latency_model = None
        if "mix_aiv_nonempty_task_latency_model" in sched:
            self.mix_aiv_nonempty_task_latency_model = (
                _normalize_mix_aiv_nonempty_task_latency_model(
                    sched["mix_aiv_nonempty_task_latency_model"]))
        else:
            self.mix_aiv_nonempty_task_latency_model = None
        if "aiv_worker_start_fill_model" in sched:
            self.aiv_worker_start_fill_model = (
                _normalize_aiv_worker_start_fill_model(
                    sched["aiv_worker_start_fill_model"]))
        else:
            self.aiv_worker_start_fill_model = None
        if "aiv_memory_task_latency_model" in sched:
            self.aiv_memory_task_latency_model = (
                _normalize_aiv_memory_task_latency_model(
                    sched["aiv_memory_task_latency_model"]))
        else:
            self.aiv_memory_task_latency_model = None
        if "aiv_tiling_materialization_latency_model" in sched:
            self.aiv_tiling_materialization_latency_model = (
                _normalize_aiv_tiling_materialization_latency_model(
                    sched["aiv_tiling_materialization_latency_model"],
                    "scheduling.aiv_tiling_materialization_latency_model"))
        else:
            self.aiv_tiling_materialization_latency_model = None
        self.aiv_tiling_materialization_48b_latency_model = (
            _load_aiv_rope_tiling_materialization_latency_model(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aiv_tiling_materialization_56b_latency_model = (
            _load_aiv_broadcast_one_dim_tiling_materialization_latency_model(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        self.aiv_tiling_materialization_192b_latency_model = (
            _load_aiv_gather_v3_tiling_materialization_latency_model(
                repository_root)
            if (self._declares_memory_model("ascend_910b"))
            else None)
        if "aiv_tiling_materialization_80b_latency_model" in sched:
            self.aiv_tiling_materialization_80b_latency_model = (
                _normalize_aiv_tiling_materialization_latency_model(
                    sched["aiv_tiling_materialization_80b_latency_model"],
                    "scheduling."
                    "aiv_tiling_materialization_80b_latency_model"))
        else:
            self.aiv_tiling_materialization_80b_latency_model = None
        if "mixed_empty_kernel_duration_lut" in sched:
            self.mixed_empty_kernel_duration_lut = (
                _normalize_mixed_empty_kernel_duration_lut(
                    sched["mixed_empty_kernel_duration_lut"]))
        else:
            self.mixed_empty_kernel_duration_lut = None
        if (self.vector_reduce_sum_latency_lut is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "vector reduction latency LUT requires a "
                "compute_reduction hardware resource")
        if (self.register_reduction_dependency_cycles is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "register reduction dependency cycles require a "
                "compute_reduction hardware resource")
        if (self.fa2_local_spill_dependency_cycles is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "FA2 local spill dependency cycles require a "
                "compute_reduction hardware resource")
        if (self.fa2_dense_d128_causal_local_spill_dependency_cycles
                is not None and "compute_reduction" not in self._role_map):
            raise ValueError(
                "FA2 dense D128 causal local spill dependency cycles require "
                "a compute_reduction hardware resource")
        if (self.cta_barrier_service_cycles is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "CTA barrier service cycles require a compute_reduction "
                "hardware resource")
        if (self.sm80_depth3_cta_barrier_service_cycles is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "SM80 depth-three CTA barrier service cycles require a "
                "compute_reduction hardware resource")
        if (self.imcontinuous_vec4_reduction_source_service is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "imcontinuous vec4 reduction source service requires a "
                "compute_reduction hardware resource")
        if self.imcontinuous_vec4_reduction_source_service is not None:
            reduction = self.resource_by_role("compute_reduction")
            for program in (
                    self.imcontinuous_vec4_reduction_source_service[
                        "programs"].values()):
                service = program["service"]
                if not reduction.has_typed_service(service):
                    raise ValueError(
                        "compute_reduction lacks imcontinuous vec4 service "
                        f"{service!r}")
                expected_rate = (
                    reduction.capacity_per_unit
                    / program["service_cycles_per_cta"])
                actual_rate = reduction.service_rate_for_one_unit(service)
                if not math.isclose(
                        actual_rate, expected_rate,
                        rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError(
                        f"compute_reduction service {service!r} rate does "
                        "not match its measured cycle service")
        if (self.fa2_combine_dependency_cycles is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "FA2 combine dependency cycles require a "
                "compute_reduction hardware resource")
        if (self.metax_fa2_combine_dependency_cycles is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "MetaX FA2 combine dependency cycles require a "
                "compute_reduction hardware resource")
        if (self.fa2_oaccum_load_use_cycles is not None
                and "compute_reduction" not in self._role_map):
            raise ValueError(
                "FA2 Oaccum load-use cycles require a "
                "compute_reduction hardware resource")
        scheduling_scalars = {
            "dispatch_overhead_ns": self.dispatch_overhead_ns,
        }
        if self.empty_kernel_duration_ns is not None:
            scheduling_scalars["empty_kernel_duration_ns"] = (
                self.empty_kernel_duration_ns)
        invalid_scheduling_scalars = {
            name: value for name, value in scheduling_scalars.items()
            if not math.isfinite(value)
        }
        if invalid_scheduling_scalars:
            raise ValueError(
                "scheduling scalar fields must be finite: "
                f"{invalid_scheduling_scalars}")
        positive_scheduling = (
            self.parallel_width, self.num_compute_units,
            self.max_warps_per_unit, self.max_blocks_per_unit,
            self.max_regs_per_unit, self.max_smem_per_unit_bytes,
            self.max_smem_per_block_bytes, self.max_threads_per_unit,
        )
        if any(value <= 0 for value in positive_scheduling):
            raise ValueError("scheduling capacity fields must be positive")
        if self.rmsnorm_shared_gamma_l1_residency is not None:
            residency = self.rmsnorm_shared_gamma_l1_residency
            threads = residency["threads_per_cta"]
            registers = residency["registers_per_thread"]
            static_smem = residency["static_shared_memory_bytes_per_cta"]
            physical_residency = min(
                self.max_blocks_per_unit,
                self.max_threads_per_unit // threads,
                self.max_regs_per_unit // (threads * registers),
                self.max_smem_per_unit_bytes // static_smem,
            )
            if physical_residency != residency["resident_ctas_per_sm"]:
                raise ValueError(
                    "RMSNorm shared-gamma residency disagrees with hardware "
                    "thread/register/shared-memory capacities")
        add_rmsnorm_pipelines = tuple(
            pipeline for pipeline in (
                self.add_rmsnorm_source_round_pipeline,
                self.add_rmsnorm_block256_source_round_pipeline)
            if pipeline is not None)
        for pipeline in add_rmsnorm_pipelines:
            threads = pipeline["threads_per_cta"]
            registers = pipeline["registers_per_thread"]
            static_smem = pipeline["static_shared_memory_bytes_per_cta"]
            physical_residency = min(
                self.max_blocks_per_unit,
                self.max_threads_per_unit // threads,
                self.max_warps_per_unit // math.ceil(
                    threads / self.parallel_width),
                self.max_regs_per_unit // (threads * registers),
                self.max_smem_per_unit_bytes // static_smem,
            )
            if physical_residency != pipeline["resident_ctas_per_sm"]:
                raise ValueError(
                    "Add-RMSNorm source-round residency disagrees with "
                    "hardware thread/warp/register/shared-memory capacities")
            if pipeline["model"] == (
                    "sm80_bf16_add_rmsnorm_block1024_one_wave_v8"):
                if (pipeline["maximum_one_wave_grid_blocks"]
                        != self.num_compute_units):
                    raise ValueError(
                        "Add-RMSNorm source-round one-wave domain must equal "
                        "the hardware compute-unit count")
                regimes = ("first",)
                active_widths = (
                    pipeline["active_threads_per_source_round"],
                    pipeline[
                        "additional_active_threads_per_source_round"],
                    pipeline[
                        "second_additional_active_threads_per_source_round"],
                    pipeline[
                        "third_additional_active_threads_per_source_round"],
                    pipeline[
                        "fourth_additional_active_threads_per_source_round"],
                )
            elif pipeline["model"] == (
                    "sm80_bf16_add_rmsnorm_block256_resident_v1"):
                if (pipeline["minimum_grid_blocks"] != 1
                        or pipeline["maximum_grid_blocks"]
                        != self.num_compute_units * physical_residency):
                    raise ValueError(
                        "block-256 Add-RMSNorm grid domain must equal the "
                        "statically resident full-chip capacity")
                regimes = ("first", "resident")
                active_widths = (256,)
            else:
                raise ValueError(
                    "unknown Add-RMSNorm source-round pipeline model")
            vector = self.resource_by_role("compute_vector")
            for function in ("add_square", "scale_multiply"):
                for regime in regimes:
                    for active_width in active_widths:
                        service = (
                            f"bf16_{function}_{regime}_{active_width}_"
                            f"block{threads}")
                        if not vector.has_typed_service(service):
                            raise ValueError(
                                "Add-RMSNorm source-round pipeline lacks "
                                f"service {service!r}")
            if pipeline["model"] == (
                    "sm80_bf16_add_rmsnorm_block1024_one_wave_v8"):
                curves = (
                    (
                        pipeline["active_threads_per_source_round"],
                        pipeline["active_unit_points"],
                    ),
                    (
                        pipeline[
                            "additional_active_threads_per_source_round"],
                        pipeline["additional_active_unit_points"],
                    ),
                    (
                        pipeline[
                            "second_additional_active_threads_per_source_round"],
                        pipeline["second_additional_active_unit_points"],
                    ),
                    (
                        pipeline[
                            "third_additional_active_threads_per_source_round"],
                        pipeline["third_additional_active_unit_points"],
                    ),
                    (
                        pipeline[
                            "fourth_additional_active_threads_per_source_round"],
                        pipeline["fourth_additional_active_unit_points"],
                    ),
                )
                for function in ("add_square", "scale_multiply"):
                    for width, active_unit_points in curves:
                        for active_units in active_unit_points:
                            active_suffix = (
                                "" if active_units == 1
                                else f"_active{active_units}")
                            service = (
                                f"bf16_{function}_first_{width}"
                                f"{active_suffix}_block{threads}")
                            if not vector.has_typed_service(service):
                                raise ValueError(
                                    "Add-RMSNorm active-unit curve lacks "
                                    f"service {service!r}")
        if self.grouped_gemm_tile_pipeline is not None:
            pipeline = self.grouped_gemm_tile_pipeline
            threads = pipeline["threads_per_cta"]
            shared_bytes = pipeline[
                "dynamic_shared_memory_bytes_per_cta"]
            for variant in ("w1", "w2"):
                registers = pipeline[
                    "registers_per_thread_by_variant"][variant]
                warps = math.ceil(threads / self.parallel_width)
                unit = self.register_allocation_unit_per_warp
                registers_per_warp = (
                    math.ceil(registers * self.parallel_width / unit) * unit)
                physical_residency = min(
                    self.max_blocks_per_unit,
                    self.max_threads_per_unit // threads,
                    self.max_warps_per_unit // warps,
                    self.max_regs_per_unit // (warps * registers_per_warp),
                    self.max_smem_per_unit_bytes // shared_bytes,
                )
                if physical_residency != pipeline[
                        "resident_ctas_per_sm_by_variant"][variant]:
                    raise ValueError(
                        "grouped-GEMM pipeline residency disagrees with "
                        "hardware thread/warp/register/shared-memory "
                        f"capacities for {variant}")
        if self.grouped_gemm_fp8_block_tile_pipelines is not None:
            for program, pipeline in (
                    self.grouped_gemm_fp8_block_tile_pipelines.items()):
                threads = pipeline["threads_per_cta"]
                shared_bytes = pipeline[
                    "dynamic_shared_memory_bytes_per_cta"]
                warps = math.ceil(threads / self.parallel_width)
                unit = self.register_allocation_unit_per_warp
                for variant in ("w1", "w2"):
                    registers = pipeline[
                        "registers_per_thread_by_variant"][variant]
                    registers_per_warp = (
                        math.ceil(registers * self.parallel_width / unit)
                        * unit)
                    physical_residency = min(
                        self.max_blocks_per_unit,
                        self.max_threads_per_unit // threads,
                        self.max_warps_per_unit // warps,
                        self.max_regs_per_unit // (warps * registers_per_warp),
                        self.max_smem_per_unit_bytes // shared_bytes,
                    )
                    if physical_residency != pipeline[
                            "resident_ctas_per_sm_by_variant"][variant]:
                        raise ValueError(
                            f"FP8 block grouped-GEMM pipeline {program} "
                            "residency disagrees with hardware thread/warp/"
                            "register/shared-memory capacities for "
                            f"{variant}")
        if self.gemm_bf16_mma_saturation is not None:
            saturation = self.gemm_bf16_mma_saturation
            if saturation["ap_count"] != self.num_compute_units:
                raise ValueError(
                    "GEMM BF16 MMA saturation AP count disagrees with "
                    "hardware compute-unit count")
            if saturation["wave_size"] != self.parallel_width:
                raise ValueError(
                    "GEMM BF16 MMA saturation wave size disagrees with "
                    "hardware parallel width")
            if saturation["active_wave_points"][-1] != self.max_warps_per_unit:
                raise ValueError(
                    "GEMM BF16 MMA saturation occupancy disagrees with "
                    "hardware maximum waves per unit")
        if self.metax_fused_moe_bm32_bf16_mma_occupancy is not None:
            occupancy = self.metax_fused_moe_bm32_bf16_mma_occupancy
            if self.hardware_id != "metax_c500":
                raise ValueError(
                    "MetaX BM32 Fused-MoE MMA occupancy is C500-only")
            if occupancy["ap_count"] != self.num_compute_units:
                raise ValueError(
                    "MetaX BM32 Fused-MoE MMA AP count disagrees with "
                    "hardware compute-unit count")
            if occupancy["wave_size"] != self.parallel_width:
                raise ValueError(
                    "MetaX BM32 Fused-MoE MMA wave size disagrees with "
                    "hardware parallel width")
            if occupancy["threads_per_cta"] % self.parallel_width:
                raise ValueError(
                    "MetaX BM32 Fused-MoE MMA CTA has partial waves")
            if occupancy["active_waves_per_ap"] != (
                    occupancy["resident_ctas_per_ap"]
                    * occupancy["threads_per_cta"] // self.parallel_width):
                raise ValueError(
                    "MetaX BM32 Fused-MoE MMA active-wave geometry drifted")
            if occupancy["dynamic_shared_memory_bytes"] > min(
                    self.max_smem_per_unit_bytes,
                    self.max_smem_per_block_bytes):
                raise ValueError(
                    "MetaX BM32 Fused-MoE MMA shared allocation exceeds "
                    "hardware capacity")
            if (self.max_smem_per_unit_bytes
                    // occupancy["dynamic_shared_memory_bytes"]
                    != occupancy["resident_ctas_per_ap"]):
                raise ValueError(
                    "MetaX BM32 Fused-MoE MMA shared-memory residency "
                    "disagrees with hardware capacity")
        if self.fa_tile16_operand_completion is not None:
            completion = self.fa_tile16_operand_completion
            if completion["ap_count"] != self.num_compute_units:
                raise ValueError(
                    "FA tile16 operand completion AP count disagrees with "
                    "hardware compute-unit count")
            if completion["wave_size"] != self.parallel_width:
                raise ValueError(
                    "FA tile16 operand completion wave size disagrees with "
                    "hardware parallel width")
        if self.memory_dependency_task_latency_model is not None:
            task_model = self.memory_dependency_task_latency_model
            expected_width_by_model = {
                "isolated_warm_default_cache_memory_dependency_task_v1": 32,
                "paired_batched_multiround_warm_default_cache_"
                "memory_dependency_task_v1": 64,
                "isolated_mcevent_multiround_warm_default_cache_"
                "memory_dependency_task_v1": 64,
                "paired_mcevent_active_unit_"
                "memory_dependency_increment_v1": 64,
            }
            expected_width = expected_width_by_model[task_model["model"]]
            if self.parallel_width != expected_width:
                raise ValueError(
                    "memory-dependency task floor requires a matching "
                    f"{expected_width}-thread physical warp")
            if (task_model["model"] not in (
                    "paired_batched_multiround_warm_default_cache_"
                    "memory_dependency_task_v1",
                    "isolated_mcevent_multiround_warm_default_cache_"
                    "memory_dependency_task_v1",
                    "paired_mcevent_active_unit_"
                    "memory_dependency_increment_v1")
                    and task_model["maximum_grid_size"]
                    > self.num_compute_units):
                raise ValueError(
                    "memory-dependency task floor cannot cover more than "
                    "one full-chip launch wave")
            maximum_block_threads = min(1024, self.max_threads_per_unit)
            if task_model["maximum_block_size"] > maximum_block_threads:
                raise ValueError(
                    "memory-dependency task floor block range exceeds the "
                    "hardware thread limit")
        if self.tensor_iterator_copy_task_latency_model is not None:
            copy_model = self.tensor_iterator_copy_task_latency_model
            if self.parallel_width != 32:
                raise ValueError(
                    "TensorIterator copy task floor requires a 32-thread warp")
            if copy_model["threads_per_cta"] > self.max_threads_per_unit:
                raise ValueError(
                    "TensorIterator copy task floor exceeds the hardware "
                    "thread limit")
            if copy_model["resident_ctas_per_sm"] > self.max_blocks_per_unit:
                raise ValueError(
                    "TensorIterator copy residency exceeds the hardware "
                    "block limit")
        if self.dispatch_overhead_ns < 0:
            raise ValueError("scheduling latency/bandwidth fields cannot be negative")
        if (self.empty_kernel_duration_ns is not None
                and self.empty_kernel_duration_ns <= 0):
            raise ValueError(
                "scheduling.empty_kernel_duration_ns must be positive")
        generic_launch_models = sum(value is not None for value in (
            self.empty_kernel_duration_ns,
            self.empty_kernel_duration_lut,
            self.launch_latency_model,
        ))
        if generic_launch_models > 1:
            raise ValueError(
                "configure either empty_kernel_duration_ns or "
                "empty_kernel_duration_lut or launch_latency_model, not "
                "more than one")

        # Features
        feat = cfg["features"]
        for name in (
                "cluster_supported", "multicast_supported", "hw_decompression"):
            if not isinstance(feat[name], bool):
                raise ValueError(f"features.{name} must be boolean")
        self.cluster_supported = feat["cluster_supported"]
        if (isinstance(feat["cluster_size"], bool)
                or not isinstance(feat["cluster_size"], int)):
            raise TypeError("features.cluster_size must be an integer")
        self.cluster_size = feat["cluster_size"]
        self.multicast_supported = feat["multicast_supported"]
        self.hw_decompression = feat["hw_decompression"]
        if (isinstance(feat["compression_ratio"], bool)
                or not isinstance(feat["compression_ratio"], (int, float))):
            raise TypeError("features.compression_ratio must be numeric")
        self.compression_ratio = float(feat["compression_ratio"])
        if (self.cluster_size < 0 or not math.isfinite(self.compression_ratio)
                or self.compression_ratio <= 0):
            raise ValueError(
                "features.cluster_size must be non-negative and "
                "compression_ratio must be positive")
        if self.cluster_supported != (self.cluster_size > 0):
            raise ValueError(
                "features.cluster_supported must agree with positive "
                "features.cluster_size")
        # Physical memory/execution path is an explicit hardware capability.
        # Predictors consume this semantic model and never infer architecture
        # behavior from a hardware identifier.
        if not isinstance(feat["kernel_memory_model"], str):
            raise TypeError("features.kernel_memory_model must be a string")
        self.kernel_memory_model = feat["kernel_memory_model"]
        valid_memory_models = {
            "gpu_mshr", "npu_tile", "pim_bank", "systolic_scratchpad"}
        if self.kernel_memory_model not in valid_memory_models:
            raise ValueError(
                "features.kernel_memory_model must be one of "
                f"{sorted(valid_memory_models)}, got {self.kernel_memory_model!r}")
        if not isinstance(feat["l1_cache_enabled"], bool):
            raise TypeError("features.l1_cache_enabled must be boolean")
        self.l1_cache_enabled = feat["l1_cache_enabled"]
        # When the async-scheduling engine takes a step's output: "self" when
        # the output is available once that step's own device work ends, and
        # "successor" when it only becomes available after the next submitted
        # step's device work ends.  Optional in the schema because the
        # accelerators that never drive the Serving engine have no such
        # semantics; the Serving boundary requires it and fails closed.
        self.async_output_release = None
        if "async_output_release" in feat:
            release = feat["async_output_release"]
            if not isinstance(release, str):
                raise TypeError(
                    "features.async_output_release must be a string")
            valid_releases = {"self", "successor"}
            if release not in valid_releases:
                raise ValueError(
                    "features.async_output_release must be one of "
                    f"{sorted(valid_releases)}, got {release!r}")
            self.async_output_release = release
        if "analytical_vector_dependency_paths" in feat:
            if not isinstance(
                    feat["analytical_vector_dependency_paths"], bool):
                raise TypeError(
                    "features.analytical_vector_dependency_paths must be "
                    "boolean")
            self.analytical_vector_dependency_paths = (
                feat["analytical_vector_dependency_paths"])
        else:
            self.analytical_vector_dependency_paths = False
        if "full_graph_physical_extent_dispatch" in feat:
            if not isinstance(
                    feat["full_graph_physical_extent_dispatch"], bool):
                raise TypeError(
                    "features.full_graph_physical_extent_dispatch must be "
                    "boolean")
            self.full_graph_physical_extent_dispatch = (
                feat["full_graph_physical_extent_dispatch"])
        else:
            self.full_graph_physical_extent_dispatch = False
        if "piecewise_graph_physical_extent_dispatch" in feat:
            if not isinstance(
                    feat["piecewise_graph_physical_extent_dispatch"], bool):
                raise TypeError(
                    "features.piecewise_graph_physical_extent_dispatch must "
                    "be boolean")
            if (feat["piecewise_graph_physical_extent_dispatch"]
                    and not self.full_graph_physical_extent_dispatch):
                raise ValueError(
                    "features.piecewise_graph_physical_extent_dispatch "
                    "extends full_graph_physical_extent_dispatch and "
                    "requires it")
            self.piecewise_graph_physical_extent_dispatch = (
                feat["piecewise_graph_physical_extent_dispatch"])
        else:
            self.piecewise_graph_physical_extent_dispatch = False
        if "full_graph_source_program_physical_extent" in feat:
            if not isinstance(
                    feat["full_graph_source_program_physical_extent"], bool):
                raise TypeError(
                    "features.full_graph_source_program_physical_extent "
                    "must be boolean")
            self.full_graph_source_program_physical_extent = (
                feat["full_graph_source_program_physical_extent"])
        else:
            self.full_graph_source_program_physical_extent = False
        if "gpu_store_completion_scope" in feat:
            if self.kernel_memory_model != "gpu_mshr":
                raise ValueError(
                    "features.gpu_store_completion_scope requires "
                    "kernel_memory_model='gpu_mshr'")
            if not isinstance(feat["gpu_store_completion_scope"], str):
                raise TypeError(
                    "features.gpu_store_completion_scope must be a string")
            self.gpu_store_completion_scope = feat[
                "gpu_store_completion_scope"].strip().lower()
            if self.gpu_store_completion_scope not in ("l2", "dram"):
                raise ValueError(
                    "features.gpu_store_completion_scope must be l2 or dram")
        else:
            if self.kernel_memory_model == "gpu_mshr":
                raise ValueError(
                    "features.gpu_store_completion_scope is required for "
                    "kernel_memory_model='gpu_mshr'")
            self.gpu_store_completion_scope = None
        if not isinstance(feat["kernel_dtype"], str):
            raise TypeError("features.kernel_dtype must be a string")
        self.kernel_dtype = feat["kernel_dtype"].strip().lower()
        if self.kernel_dtype not in {
                "bf16", "fp16", "fp8", "int8", "uint8", "float32",
                "fp32", "int32"}:
            raise ValueError(
                f"features.kernel_dtype is unsupported: {self.kernel_dtype!r}")
        # Compatibility properties keep individual predictor paths readable;
        # all three are derived from the single authoritative capability.
        self.aic_aiv_separated = self.kernel_memory_model == "npu_tile"
        self.pim_bank_compute = self.kernel_memory_model == "pim_bank"
        self.systolic_dma = self.kernel_memory_model == "systolic_scratchpad"
        if (self.memory_dependency_task_latency_model is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "memory-dependency task floor requires gpu_mshr hardware")
        if (self.tensor_iterator_copy_task_latency_model is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "TensorIterator copy task floor requires gpu_mshr hardware")
        if (self.launch_latency_model is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "isolated empty-task scheduler model currently requires "
                "kernel_memory_model='gpu_mshr'")
        if (self.fa2_source_task_outer_envelope is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "FA2 source task outer envelope requires "
                "kernel_memory_model='gpu_mshr'")
        if (self.register_reduction_dependency_cycles is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "register reduction dependency cycles require "
                "kernel_memory_model='gpu_mshr'")
        if (self.fa2_local_spill_dependency_cycles is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "FA2 local spill dependency cycles require "
                "kernel_memory_model='gpu_mshr'")
        if (self.fa2_dense_d128_causal_local_spill_dependency_cycles
                is not None and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "FA2 dense D128 causal local spill dependency cycles require "
                "kernel_memory_model='gpu_mshr'")
        if (self.cta_barrier_service_cycles is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "CTA barrier service cycles require "
                "kernel_memory_model='gpu_mshr'")
        if (self.sm80_depth3_cta_barrier_service_cycles is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "SM80 depth-three CTA barrier service cycles require "
                "kernel_memory_model='gpu_mshr'")
        if (self.flash_mla_combine_barrier_service_cycles is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "FlashMLA combine barrier service cycles require "
                "kernel_memory_model='gpu_mshr'")
        if (self.imcontinuous_vec4_reduction_source_service is not None
                and self.kernel_memory_model != "gpu_mshr"):
            raise ValueError(
                "imcontinuous vec4 reduction source service requires "
                "kernel_memory_model='gpu_mshr'")
        if self.aic_empty_kernel_duration_lut is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC empty-kernel duration LUT requires separated AIC/AIV")
            block_dims, _ = self.aic_empty_kernel_duration_lut
            if block_dims[-1] != self.num_compute_units:
                raise ValueError(
                    "AIC empty-kernel duration LUT must end at the physical "
                    f"AIC count {self.num_compute_units}")
        if self.aiv_empty_kernel_duration_lut is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV empty-kernel duration LUT requires separated AIC/AIV")
            vector_units = self.resource_by_role("compute_vector").total_units
            block_dims, _ = self.aiv_empty_kernel_duration_lut
            if block_dims[-1] != vector_units:
                raise ValueError(
                    "AIV empty-kernel duration LUT must end at the physical "
                    f"Vector-core count {vector_units}")
        if self.aiv_nonempty_task_latency_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV non-empty task latency model requires separated "
                    "AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.aiv_nonempty_task_latency_model[
                "maximum_aiv_block_dim"]
            if maximum != vector_units:
                raise ValueError(
                    "AIV non-empty task latency model must end at the "
                    f"physical Vector-core count {vector_units}")
        if self.mix_aiv_nonempty_task_latency_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "MIX_AIV non-empty task latency model requires "
                    "separated AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.mix_aiv_nonempty_task_latency_model[
                "maximum_aiv_block_dim"]
            if maximum != vector_units:
                raise ValueError(
                    "MIX_AIV non-empty task latency model must end at the "
                    f"physical Vector-core count {vector_units}")
        if self.aiv_worker_start_fill_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV worker-start fill model requires separated AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.aiv_worker_start_fill_model[
                "maximum_aiv_block_dim"]
            if maximum != vector_units:
                raise ValueError(
                    "AIV worker-start fill model must end at the physical "
                    f"Vector-core count {vector_units}")
        if self.aiv_memory_task_latency_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV memory-task latency model requires separated "
                    "AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.aiv_memory_task_latency_model[
                "maximum_aiv_block_dim"]
            if maximum != vector_units:
                raise ValueError(
                    "AIV memory-task latency model must end at the physical "
                    f"Vector-core count {vector_units}")
        if self.aiv_tiling_materialization_latency_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV tiling-materialization latency model requires "
                    "separated AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.aiv_tiling_materialization_latency_model[
                "maximum_active_aiv_workers"]
            if maximum != vector_units:
                raise ValueError(
                    "AIV tiling-materialization latency model must end at "
                    f"the physical Vector-core count {vector_units}")
        if self.aiv_tiling_materialization_48b_latency_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV 48-byte tiling-materialization latency model "
                    "requires separated AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.aiv_tiling_materialization_48b_latency_model[
                "maximum_active_aiv_workers"]
            if maximum != vector_units:
                raise ValueError(
                    "AIV 48-byte tiling-materialization latency model must "
                    "end at the physical Vector-core count "
                    f"{vector_units}")
        if self.aiv_tiling_materialization_56b_latency_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV 56-byte tiling-materialization latency model "
                    "requires separated AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.aiv_tiling_materialization_56b_latency_model[
                "maximum_active_aiv_workers"]
            if maximum != vector_units:
                raise ValueError(
                    "AIV 56-byte tiling-materialization latency model must "
                    "end at the physical Vector-core count "
                    f"{vector_units}")
        if self.aiv_tiling_materialization_80b_latency_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV 80-byte tiling-materialization latency model "
                    "requires separated AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.aiv_tiling_materialization_80b_latency_model[
                "maximum_active_aiv_workers"]
            if maximum != vector_units:
                raise ValueError(
                    "AIV 80-byte tiling-materialization latency model must "
                    "end at the physical Vector-core count "
                    f"{vector_units}")
        if self.aiv_tiling_materialization_192b_latency_model is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV 192-byte tiling-materialization latency model "
                    "requires separated AIC/AIV")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            maximum = self.aiv_tiling_materialization_192b_latency_model[
                "maximum_active_aiv_workers"]
            if maximum != vector_units:
                raise ValueError(
                    "AIV 192-byte tiling-materialization latency model must "
                    "end at the physical Vector-core count "
                    f"{vector_units}")
        if self.mixed_empty_kernel_duration_lut is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "mixed empty-kernel duration LUT requires separated "
                    "AIC/AIV")
            block_dims, _, _ = self.mixed_empty_kernel_duration_lut
            if block_dims[-1] != self.num_compute_units:
                raise ValueError(
                    "mixed empty-kernel duration LUT must end at the physical "
                    f"AIC count {self.num_compute_units}")
        if self.fia_cross_domain_three_handoff is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "FIA cross-domain mechanism requires separated AIC/AIV")
            if self.mixed_empty_kernel_duration_lut is None:
                raise ValueError(
                    "FIA cross-domain mechanism requires a mixed-task launch "
                    "floor")
            mechanism = self.fia_cross_domain_three_handoff
            if (mechanism["maximum_active_aic_groups"]
                    != self.num_compute_units):
                raise ValueError(
                    "FIA cross-domain active-group maximum must equal the "
                    "physical AIC count")
        if self.fia_aiv_instruction_cold_fill is not None:
            if (not self.aic_aiv_separated
                    or self.fia_cross_domain_three_handoff is None):
                raise ValueError(
                    "FIA AIV instruction cold fill requires the separated "
                    "source-bound FIA program")
            split_fuse = self.fia_aiv_instruction_cold_fill["split_fuse"]
            vector_units = self.resource_by_role("compute_vector").total_units
            if (split_fuse["active_aic_groups"]
                    * split_fuse["aiv_per_group"] != vector_units):
                raise ValueError(
                    "FIA SplitFuse cold-fill topology must cover all "
                    "physical Vector cores")
        if self.fia_aiv_sync_all_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "FIA AIV SyncAll service requires separated AIC/AIV")
            if self.fia_cross_domain_three_handoff is None:
                raise ValueError(
                    "FIA AIV SyncAll service requires the source-bound "
                    "cross-domain program")
            mechanism = self.fia_aiv_sync_all_service
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            if mechanism["maximum_active_aiv_workers"] != vector_units:
                raise ValueError(
                    "FIA AIV SyncAll maximum must equal the physical "
                    "Vector-core count")
        if self.fia_flash_decode_terminal_prelude_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "FIA FlashDecode prelude requires separated AIC/AIV")
            if self.fia_cross_domain_three_handoff is None:
                raise ValueError(
                    "FIA FlashDecode prelude requires the source-bound "
                    "cross-domain program")
            mechanism = self.fia_flash_decode_terminal_prelude_service
            if (mechanism["maximum_active_aic_groups"]
                    != self.num_compute_units):
                raise ValueError(
                    "FIA FlashDecode prelude maximum must equal the "
                    "physical AIC count")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            if (mechanism["aiv_per_group"] * self.num_compute_units
                    != vector_units):
                raise ValueError(
                    "FIA FlashDecode prelude topology disagrees with "
                    "Vector-core count")
        if self.fia_tiling_data_materialization_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "FIA tiling materialization requires separated AIC/AIV")
            if self.fia_cross_domain_three_handoff is None:
                raise ValueError(
                    "FIA tiling materialization requires the source-bound "
                    "cross-domain program")
            mechanism = self.fia_tiling_data_materialization_service
            if (mechanism["maximum_active_aic_groups"]
                    != self.num_compute_units):
                raise ValueError(
                    "FIA tiling materialization maximum must equal the "
                    "physical AIC count")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            if (mechanism["aiv_per_group"] * self.num_compute_units
                    != vector_units):
                raise ValueError(
                    "FIA tiling materialization topology disagrees with "
                    "Vector-core count")
        if self.fia_mixed_task_outer_envelope is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "FIA mixed-task outer envelope requires separated "
                    "AIC/AIV")
            if (self.fia_cross_domain_three_handoff is None
                    or self.fia_aiv_sync_all_service is None):
                raise ValueError(
                    "FIA mixed-task outer envelope requires the complete "
                    "source-bound FIA program")
            mechanism = self.fia_mixed_task_outer_envelope
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            if mechanism["maximum_active_aiv_workers"] != vector_units:
                raise ValueError(
                    "FIA mixed-task outer-envelope maximum must equal the "
                    "physical Vector-core count")
        if self.fia_aiv_workspace_transfer_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "FIA AIV workspace-transfer service requires separated "
                    "AIC/AIV")
            if self.fia_cross_domain_three_handoff is None:
                raise ValueError(
                    "FIA AIV workspace-transfer service requires the "
                    "source-bound FIA program")
        if self.aic_mte2_nd2nz_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC MTE2 ND2NZ service requires separated AIC/AIV")
            if (self.aic_mte2_nd2nz_service["active_aic_groups"][-1]
                    != self.num_compute_units):
                raise ValueError(
                    "AIC MTE2 ND2NZ active-group maximum must equal the "
                    "physical AIC count")
        if self.aic_mte2_nd2nz_33copy_group_service is not None:
            mechanism = self.aic_mte2_nd2nz_33copy_group_service
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC MTE2 33-copy group service requires separated "
                    "AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC MTE2 33-copy group service requires the full AIC "
                    "domain")
        if self.aic_deterministic_split_k_source_queue_service is not None:
            mechanism = self.aic_deterministic_split_k_source_queue_service
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC deterministic split-K source queue requires "
                    "separated AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC deterministic split-K source queue requires the "
                    "full AIC domain")
        if self.aic_mte2_nd2nz_large_contiguous_service is not None:
            mechanism = self.aic_mte2_nd2nz_large_contiguous_service
            if not self.aic_aiv_separated:
                raise ValueError(
                    "large-contiguous ND2NZ service requires separated "
                    "AIC/AIV")
            if mechanism["active_aic_groups"][-1] != self.num_compute_units:
                raise ValueError(
                    "large-contiguous ND2NZ maximum group count must equal "
                    "the physical AIC count")
        if self.aic_mte2_gemm_large_resident_service is not None:
            mechanism = self.aic_mte2_gemm_large_resident_service
            if not self.aic_aiv_separated:
                raise ValueError(
                    "GEMM large-resident ND2NZ service requires separated "
                    "AIC/AIV")
            if mechanism["active_aic_groups"][-1] > self.num_compute_units:
                raise ValueError(
                    "GEMM large-resident ND2NZ group count exceeds the "
                    "physical AIC count")
        if self.aic_mte2_operand_pair_stream_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC MTE2 operand-pair stream service requires "
                    "separated AIC/AIV")
            if (self.aic_mte2_operand_pair_stream_service[
                    "active_aic_groups"] != self.num_compute_units):
                raise ValueError(
                    "AIC MTE2 operand-pair active-group count must equal "
                    "the physical AIC count")
        if self.aic_mte2_cold_operand_pair_stream_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC MTE2 cold operand-pair stream service requires "
                    "separated AIC/AIV")
            if (self.aic_mte2_cold_operand_pair_stream_service[
                    "active_aic_groups"] != self.num_compute_units):
                raise ValueError(
                    "AIC MTE2 cold operand-pair active-group count must "
                    "equal the physical AIC count")
        if self.aic_mte2_cold_legacy_k14336_stride_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC MTE2 cold legacy stride service requires "
                    "separated AIC/AIV")
            if (self.aic_mte2_cold_legacy_k14336_stride_service[
                    "active_aic_groups"] > self.num_compute_units):
                raise ValueError(
                    "AIC MTE2 cold legacy stride active-group count exceeds "
                    "the physical AIC count")
        if (self.aic_mte2_finite_stream_completion_service is not None
                and max(
                    max(worker_class["active_aic_groups"])
                    for worker_class in self.aic_mte2_finite_stream_completion_service[
                        "worker_classes"].values()) > self.num_compute_units):
            raise ValueError(
                "AIC MTE2 finite-stream service exceeds physical AIC count")
        if (self.aic_mte2_fia_mha_finite_drain_service is not None
                and max(self.aic_mte2_fia_mha_finite_drain_service[
                    "active_aic_groups"]) > self.num_compute_units):
            raise ValueError(
                "AIC MTE2 FIA MHA finite-drain service exceeds physical "
                "AIC count")
        if (self.aic_mte2_fia_gqa_finite_drain_service is not None
                and max(self.aic_mte2_fia_gqa_finite_drain_service[
                    "active_aic_groups"]) > self.num_compute_units):
            raise ValueError(
                "AIC MTE2 FIA GQA finite-drain service exceeds physical "
                "AIC count")
        if self.aic_grouped_matmul_finite_stream_completion_service is not None:
            mechanism = self.aic_grouped_matmul_finite_stream_completion_service
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul finite-stream service requires "
                    "separated AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC GroupedMatmul finite-stream active-group count must "
                    "equal the physical AIC count")
        if (self.aic_grouped_matmul_k2880_wide_base_tail_completion_service
                is not None):
            mechanism = (
                self.aic_grouped_matmul_k2880_wide_base_tail_completion_service)
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul K2880 tail service requires separated "
                    "AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC GroupedMatmul K2880 tail active-group count must "
                    "equal the physical AIC count")
        if (self.aic_grouped_matmul_k2880_output_tile_completion_service
                is not None):
            mechanism = (
                self.aic_grouped_matmul_k2880_output_tile_completion_service)
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul K2880 output-tile service requires "
                    "separated AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC GroupedMatmul K2880 output-tile active-group count "
                    "must equal the physical AIC count")
        if (self.aic_grouped_matmul_step4_worker_sequence_completion_service
                is not None):
            mechanism = (
                self.aic_grouped_matmul_step4_worker_sequence_completion_service)
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul step-4 worker-sequence service requires "
                    "separated AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC GroupedMatmul step-4 worker-sequence active-group "
                    "count must equal the physical AIC count")
        if (self.aic_grouped_matmul_short_source_prelude_control_service
                is not None):
            mechanism = (
                self.aic_grouped_matmul_short_source_prelude_control_service)
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul short-source prelude service requires "
                    "separated AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC GroupedMatmul short-source prelude active-group "
                    "count must equal the physical AIC count")
        if (self.aic_grouped_matmul_m32_short_source_prelude_control_service
                is not None):
            mechanism = (
                self.aic_grouped_matmul_m32_short_source_prelude_control_service)
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul M32 short-source prelude service "
                    "requires separated AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC GroupedMatmul M32 short-source prelude active-group "
                    "count must equal the physical AIC count")
        if self.aic_grouped_matmul_entry_metadata_pair_service is not None:
            mechanism = self.aic_grouped_matmul_entry_metadata_pair_service
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul entry metadata pair requires separated "
                    "AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC GroupedMatmul entry metadata-pair active-group count "
                    "must equal the physical AIC count")
        if (self.aic_grouped_matmul_short_source_tail_prelude_control_service
                is not None):
            mechanism = (
                self.aic_grouped_matmul_short_source_tail_prelude_control_service)
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul short-source tail prelude service "
                    "requires separated AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC GroupedMatmul short-source tail prelude active-group "
                    "count must equal the physical AIC count")
        if (self.aic_mte2_legacy_double_l0c_pair_service is not None
                and self.aic_mte2_legacy_double_l0c_pair_service[
                    "active_aic_groups"] > self.num_compute_units):
            raise ValueError(
                "legacy double-L0C pair service exceeds physical AIC count")
        if self.aic_mte2_operand_pair_overlap_domain is not None:
            mechanism = self.aic_mte2_operand_pair_overlap_domain
            if self.aic_mte2_operand_pair_stream_service is None:
                raise ValueError(
                    "AIC MTE2 operand-pair overlap requires the long-stream "
                    "service")
            if (mechanism["maximum_active_aic_groups"]
                    != self.num_compute_units):
                raise ValueError(
                    "AIC MTE2 operand-pair overlap maximum must equal the "
                    "physical AIC count")
        if self.aic_mte2_operand_pair_finite_overlap_domain is not None:
            mechanism = self.aic_mte2_operand_pair_finite_overlap_domain
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "AIC MTE2 finite operand-pair overlap must cover all AICs")
        if self.aic_mte2_mixed_ring_long_overlap_domain is not None:
            mechanism = self.aic_mte2_mixed_ring_long_overlap_domain
            if (mechanism["active_aic_groups"] != self.num_compute_units
                    or mechanism["cold_source_ring_bytes_per_operand"]
                    <= self.resource_by_role("l2_cache").size_bytes):
                raise ValueError(
                    "AIC MTE2 mixed-ring overlap must cover all AICs and "
                    "revisit beyond L2 capacity")
        if self.aic_matmul_v3_asw_mte2_pair_stream_service is not None:
            mechanism = self.aic_matmul_v3_asw_mte2_pair_stream_service
            if not self.aic_aiv_separated:
                raise ValueError(
                    "MatMulV3 ASW MTE2 pair service requires separated "
                    "AIC/AIV")
            if mechanism["active_aic_groups"] != self.num_compute_units:
                raise ValueError(
                    "MatMulV3 ASW MTE2 pair active-group count must equal "
                    "the physical AIC count")
        if self.aic_mte1_load_pair_completion_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC MTE1 load-pair service requires separated AIC/AIV")
            if (self.aic_mte1_load_pair_completion_service[
                    "active_aic_groups"][-1] != self.num_compute_units):
                raise ValueError(
                    "AIC MTE1 load-pair active-group maximum must equal the "
                    "physical AIC count")
        if self.aic_grouped_matmul_mte1_pair_completion_service is not None:
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIC GroupedMatmul MTE1 pair service requires separated "
                    "AIC/AIV")
            if (self.aic_grouped_matmul_mte1_pair_completion_service[
                    "active_aic_groups"][-1] != self.num_compute_units):
                raise ValueError(
                    "AIC GroupedMatmul MTE1 pair active-group maximum must "
                    "equal the physical AIC count")
        if (self.aic_fixpipe_full_tile_service is not None
                and not self.aic_aiv_separated):
            raise ValueError(
                "AIC Fixpipe service requires separated AIC/AIV")
        if (self.aic_fixpipe_bf16_terminal_tile_service is not None
                and not self.aic_aiv_separated):
            raise ValueError(
                "AIC BF16 terminal Fixpipe service requires separated AIC/AIV")
        if (self.aic_fixpipe_bf16_finite_stream_service is not None
                and not self.aic_aiv_separated):
            raise ValueError(
                "AIC BF16 finite Fixpipe service requires separated AIC/AIV")
        if (self.fia_block_reduce_sum_service is not None
                or self.fia_block_reduce_max_service is not None
                or self.fia_brcb_service is not None
        ) and not self.aic_aiv_separated:
            raise ValueError(
                "FIA Vector services require separated AIC/AIV")
        # Derived: L2 size (from l2_cache role resource, for forward traffic model)
        l2_res = self.resource_by_role("l2_cache")
        self.l2_size_bytes = l2_res.size_bytes
        if (self.aic_mte2_nd2nz_33copy_group_service is not None
                and self.aic_mte2_nd2nz_33copy_group_service[
                    "shared_l2_bytes"] != self.l2_size_bytes):
            raise ValueError(
                "AIC MTE2 33-copy group evidence L2 size disagrees with "
                "the hardware")

        # Memory-subsystem mechanism parameters.  The production formula
        # consumes explicit byte/request capacities and finite-source slots;
        # it no longer derives an active-unit DRAM scaling factor here.
        mem_sub = _normalize_memory_subsystem(
            cfg["memory_subsystem"], self.kernel_memory_model)
        replacement_capacity = mem_sub["l2_replacement_capacity_bytes"]
        if replacement_capacity is None:
            self.l2_replacement_capacity_bytes = self.l2_size_bytes
            self.l2_replacement_capacity_evidence_ref = None
        else:
            if replacement_capacity > self.l2_size_bytes:
                raise ValueError(
                    "L2 replacement capacity cannot exceed nominal L2 size")
            evidence_ref = mem_sub[
                "l2_replacement_capacity_evidence_ref"]
            evidence_path = repository_root / evidence_ref
            evidence_bytes = evidence_path.read_bytes()
            if hashlib.sha256(evidence_bytes).hexdigest() != mem_sub[
                    "l2_replacement_capacity_evidence_sha256"]:
                raise ValueError(
                    "L2 replacement-capacity evidence SHA256 mismatch")
            evidence = json.loads(evidence_bytes)
            _require_exact_fields(
                "L2 replacement-capacity evidence", evidence,
                {
                    "schema_version", "hardware_id", "software_revision",
                    "mechanism", "nominal_capacity_bytes",
                    "replacement_capacity_bytes", "selection_rule",
                    "counter_sector_bytes", "source_program",
                    "measurements", "capture",
                    "production_or_evaluation_duration_consumed",
                    "parameter_fit_from_production_duration",
                })
            if (evidence["schema_version"]
                    != "midlayer.cache_replacement_capacity.v1"
                    or evidence["hardware_id"] != self.hardware_id
                    or evidence["nominal_capacity_bytes"]
                        != self.l2_size_bytes
                    or evidence["replacement_capacity_bytes"]
                        != replacement_capacity
                    or evidence[
                        "production_or_evaluation_duration_consumed"] is not False
                    or evidence[
                        "parameter_fit_from_production_duration"] is not False):
                raise ValueError(
                    "L2 replacement-capacity evidence contract drifted")
            source_program = evidence["source_program"]
            _require_exact_fields(
                "L2 replacement-capacity source program", source_program,
                {
                    "dtype", "active_aic_groups", "rows", "columns",
                    "source_row_stride_elements", "descriptor_bytes",
                    "worker_addressing", "passes",
                })
            if (source_program["active_aic_groups"]
                    != self.num_compute_units
                    or source_program["worker_addressing"]
                        != "disjoint_contiguous_partitions"
                    or source_program["passes"] != 4):
                raise ValueError(
                    "L2 replacement-capacity source topology drifted")
            counter_sector_bytes = evidence["counter_sector_bytes"]
            if (isinstance(counter_sector_bytes, bool)
                    or not isinstance(counter_sector_bytes, int)
                    or counter_sector_bytes <= 0):
                raise ValueError(
                    "L2 replacement-capacity sector size must be positive")
            measurements = evidence["measurements"]
            if not isinstance(measurements, list) or len(measurements) < 2:
                raise ValueError(
                    "L2 replacement-capacity evidence requires a boundary "
                    "sweep")
            resident_footprints = []
            conflicting_footprints = []
            for index, measurement in enumerate(measurements):
                owner = (
                    "L2 replacement-capacity measurements"
                    f"[{index}]")
                _require_exact_fields(
                    owner, measurement,
                    {
                        "nominal_footprint_mib", "actual_footprint_bytes",
                        "read_hit_sectors", "read_miss_sectors",
                        "read_hit_rate_pct", "raw_path", "raw_sha256",
                    })
                integer_names = (
                    "nominal_footprint_mib", "actual_footprint_bytes",
                    "read_hit_sectors", "read_miss_sectors",
                )
                if any(
                        isinstance(measurement[name], bool)
                        or not isinstance(measurement[name], int)
                        or measurement[name] <= 0
                        for name in integer_names):
                    raise ValueError(f"{owner} counters must be positive")
                hit_rate = measurement["read_hit_rate_pct"]
                if (isinstance(hit_rate, bool)
                        or not isinstance(hit_rate, (int, float))
                        or not math.isfinite(float(hit_rate))):
                    raise ValueError(f"{owner} hit rate must be finite")
                calculated_hit_rate = 100.0 * measurement[
                    "read_hit_sectors"] / (
                        measurement["read_hit_sectors"]
                        + measurement["read_miss_sectors"])
                if not math.isclose(
                        float(hit_rate), calculated_hit_rate,
                        rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError(f"{owner} hit rate is inconsistent")
                raw_path = measurement["raw_path"]
                raw_sha256 = measurement["raw_sha256"]
                if (not isinstance(raw_path, str)
                        or not raw_path.startswith(
                            "profiler_output/microbench/ascend_910b/")
                        or ".." in raw_path.split("/")
                        or not isinstance(raw_sha256, str)
                        or len(raw_sha256) != 64
                        or any(character not in "0123456789abcdef"
                               for character in raw_sha256)):
                    raise ValueError(f"{owner} raw provenance is invalid")
                miss_ratio = (
                    measurement["read_miss_sectors"]
                    * counter_sector_bytes
                    / measurement["actual_footprint_bytes"])
                if math.isclose(miss_ratio, 1.0, rel_tol=0.001):
                    resident_footprints.append(
                        measurement["nominal_footprint_mib"] * 1024 * 1024)
                elif miss_ratio > 1.1:
                    conflicting_footprints.append(
                        measurement["nominal_footprint_mib"] * 1024 * 1024)
                else:
                    raise ValueError(
                        f"{owner} is neither resident nor conflicting")
            if (max(resident_footprints, default=0)
                    != replacement_capacity
                    or min(conflicting_footprints, default=0)
                        <= replacement_capacity):
                raise ValueError(
                    "L2 replacement capacity does not match its counter "
                    "boundary")
            self.l2_replacement_capacity_bytes = replacement_capacity
            self.l2_replacement_capacity_evidence_ref = evidence_ref
        # Replacement grading is the shared L2-to-DRAM miss core's only other
        # per-hardware knob.  An empty profile is the exact capacity step of a
        # byte-capacity LRU, which is what every current target's replacement
        # evidence states; a target whose measured residency is graded rather
        # than a step declares that curve here as ascending
        # (occupancy_bytes, dram_served_fraction) points.
        self.l2_replacement_residency_profile: tuple[tuple[int, float], ...] = ()
        dirty_replacement_capacity = mem_sub[
            "l2_dirty_replacement_capacity_bytes_per_s"]
        if dirty_replacement_capacity is None:
            self.l2_dirty_replacement_capacity_bytes_per_s = None
            self.l2_dirty_replacement_evidence_ref = None
            self.l2_dirty_replacement_source_program = None
            self.l2_dirty_replacement_clean_completion_ns = None
        else:
            evidence_ref = mem_sub["l2_dirty_replacement_evidence_ref"]
            evidence_path = repository_root / evidence_ref
            evidence_bytes = evidence_path.read_bytes()
            if hashlib.sha256(evidence_bytes).hexdigest() != mem_sub[
                    "l2_dirty_replacement_evidence_sha256"]:
                raise ValueError(
                    "L2 dirty-replacement evidence SHA256 mismatch")
            evidence = json.loads(evidence_bytes)
            _require_exact_fields(
                "L2 dirty-replacement evidence", evidence,
                {
                    "schema_version", "hardware_id", "software_revision",
                    "mechanism", "capacity_bytes_per_s", "derivation",
                    "source_program", "capture",
                    "production_or_evaluation_duration_consumed",
                    "parameter_fit_from_production_duration",
                })
            derivation = evidence["derivation"]
            _require_exact_fields(
                "L2 dirty-replacement derivation", derivation,
                {
                    "initial_dirty_bytes", "displaced_predecessor_bytes",
                    "clean_mte2_only_ns", "dirty_mte2_only_ns",
                    "clean_mte2_mte1_completion_ns",
                    "dirty_mte2_mte1_completion_ns",
                    "additional_service_ns", "selection_rule",
                })
            source_program = evidence["source_program"]
            _require_exact_fields(
                "L2 dirty-replacement source program", source_program,
                {
                    "dtype", "active_aic_groups", "base_tile_m",
                    "base_tile_n", "base_tile_k",
                    "output_tiles_per_worker", "k_slabs",
                    "source_row_stride", "cache_preamble_kind",
                    "cache_preamble_bytes",
                })
            integer_source_fields = {
                "active_aic_groups", "base_tile_m", "base_tile_n",
                "base_tile_k", "output_tiles_per_worker", "k_slabs",
                "source_row_stride", "cache_preamble_bytes",
            }
            if (source_program["dtype"] != "bf16"
                    or source_program["cache_preamble_kind"]
                        != "disjoint_aiv_gm_write"
                    or any(
                        isinstance(source_program[name], bool)
                        or not isinstance(source_program[name], int)
                        or source_program[name] <= 0
                        for name in integer_source_fields)):
                raise ValueError(
                    "L2 dirty-replacement source program drifted")
            numeric_derivation_fields = {
                "initial_dirty_bytes", "displaced_predecessor_bytes",
                "clean_mte2_only_ns", "dirty_mte2_only_ns",
                "clean_mte2_mte1_completion_ns",
                "dirty_mte2_mte1_completion_ns", "additional_service_ns",
            }
            if any(
                    isinstance(derivation[name], bool)
                    or not isinstance(derivation[name], (int, float))
                    or not math.isfinite(float(derivation[name]))
                    or derivation[name] <= 0
                    for name in numeric_derivation_fields):
                raise ValueError(
                    "L2 dirty-replacement derivation must be positive")
            if (evidence["schema_version"]
                    != "midlayer.l2_dirty_replacement_backpressure.v1"
                    or evidence["hardware_id"] != self.hardware_id
                    or not math.isclose(
                        evidence["capacity_bytes_per_s"],
                        dirty_replacement_capacity,
                        rel_tol=1e-12, abs_tol=0.0)
                    or derivation["initial_dirty_bytes"]
                        != self.l2_replacement_capacity_bytes
                    or not math.isclose(
                        derivation["dirty_mte2_only_ns"]
                        - derivation["clean_mte2_only_ns"],
                        derivation["additional_service_ns"],
                        rel_tol=0.0, abs_tol=1e-9)
                    or derivation["dirty_mte2_mte1_completion_ns"]
                        <= derivation["clean_mte2_mte1_completion_ns"]
                    or evidence[
                        "production_or_evaluation_duration_consumed"] is not False
                    or evidence[
                        "parameter_fit_from_production_duration"] is not False):
                raise ValueError(
                    "L2 dirty-replacement evidence contract drifted")
            self.l2_dirty_replacement_capacity_bytes_per_s = float(
                dirty_replacement_capacity)
            self.l2_dirty_replacement_evidence_ref = evidence_ref
            self.l2_dirty_replacement_source_program = dict(source_program)
            self.l2_dirty_replacement_clean_completion_ns = float(
                derivation["clean_mte2_mte1_completion_ns"])
        self.dram_latency_ns = (
            None if mem_sub["dram_latency_ns"] is None
            else float(mem_sub["dram_latency_ns"]))
        for _pipeline_name in METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS:
            _pipeline = getattr(self, _pipeline_name)
            if _pipeline is None:
                continue
            if self.hardware_id != "metax_c500":
                raise ValueError(
                    f"{_pipeline_name} is C500-only")
            if self.dram_latency_ns is None:
                raise ValueError(
                    f"{_pipeline_name} requires DRAM latency")
            threads = _pipeline["threads_per_cta"]
            shared_bytes = _pipeline[
                "dynamic_shared_memory_bytes_per_cta"]
            physical_residency = min(
                self.max_blocks_per_unit,
                self.max_threads_per_unit // threads,
                self.max_warps_per_unit // math.ceil(
                    threads / self.parallel_width),
                self.max_smem_per_unit_bytes // shared_bytes,
            )
            if physical_residency != _pipeline["resident_ctas_per_ap"]:
                raise ValueError(
                    f"{_pipeline_name} residency disagrees with hardware "
                    "thread/warp/shared-memory capacities")
        self.l2_latency_ns = (
            None if mem_sub["l2_latency_ns"] is None
            else float(mem_sub["l2_latency_ns"]))
        self.cacheline_bytes = mem_sub["cacheline_bytes"]
        self.l2_transaction_bytes = mem_sub["l2_transaction_bytes"]
        self.dram_transaction_bytes = mem_sub["dram_transaction_bytes"]
        self.l2_conflict_efficiency_moe = float(
            mem_sub["l2_conflict_efficiency_moe"])
        self.dram_protocol_efficiency = float(mem_sub["dram_protocol_efficiency"])
        # Optional access-path efficiency relative to the configured streaming
        # DRAM capacity.  A value below one requires an independent mechanism
        # benchmark for that access path; target-kernel achieved bandwidth is not
        # a legal source.  Current production configs use one for this path.
        self.dram_protocol_efficiency_l1cached = float(
            mem_sub["dram_protocol_efficiency_l1cached"])
        self.source_slots_per_unit = mem_sub["source_slots_per_unit"]
        self.gemv_scalar_half_issue_group_elements_per_lane = mem_sub[
            "gemv_scalar_half_issue_group_elements_per_lane"]
        self.gemv_scalar_half_resident_ctas_per_sm = mem_sub[
            "gemv_scalar_half_resident_ctas_per_sm"]
        self.gemv_scalar_half_dram_transactions_per_sm_s = mem_sub[
            "gemv_scalar_half_dram_transactions_per_sm_s"]
        self.gemv_scalar_half_source_fill_ns = mem_sub[
            "gemv_scalar_half_source_fill_ns"]
        gemv_scalar_stream = (
            self.gemv_scalar_half_issue_group_elements_per_lane,
            self.gemv_scalar_half_resident_ctas_per_sm,
            self.gemv_scalar_half_dram_transactions_per_sm_s,
            self.gemv_scalar_half_source_fill_ns,
        )
        if any(value is None for value in gemv_scalar_stream):
            if not all(value is None for value in gemv_scalar_stream):
                raise ValueError(
                    "GEMV scalar-half DRAM source requires all four "
                    "geometry, residency, rate, and fill fields")
        else:
            if self.kernel_memory_model != "gpu_mshr":
                raise ValueError(
                    "GEMV scalar-half DRAM source requires gpu_mshr hardware")
            if self.l2_transaction_bytes != 32:
                raise ValueError(
                    "GEMV scalar-half DRAM source requires 32-byte L2 "
                    "transactions")
            if any(float(value) <= 0.0 for value in gemv_scalar_stream):
                raise ValueError(
                    "GEMV scalar-half DRAM source fields must be positive")
        self.ldgsts_initial_32b_transactions_single_sm_cycle = (
            mem_sub["ldgsts_initial_32b_transactions_single_sm_cycle"])
        self.ldgsts_initial_32b_transactions_per_active_sm_cycle = (
            mem_sub[
                "ldgsts_initial_32b_transactions_per_active_sm_cycle"])
        self.ldgsts_initial_32b_transactions_full_chip_per_cycle = (
            mem_sub[
                "ldgsts_initial_32b_transactions_full_chip_per_cycle"])
        self.ldgsts_finite_resident_panel_bytes = mem_sub[
            "ldgsts_finite_resident_panel_bytes"]
        self.ldgsts_finite_resident_full_chip_ctas = mem_sub[
            "ldgsts_finite_resident_full_chip_ctas"]
        self.ldgsts_finite_resident_single_panel_cycles = mem_sub[
            "ldgsts_finite_resident_single_panel_cycles"]
        self.ldgsts_finite_resident_affine_startup_cycles = mem_sub[
            "ldgsts_finite_resident_affine_startup_cycles"]
        self.ldgsts_finite_resident_affine_panel_cycles = mem_sub[
            "ldgsts_finite_resident_affine_panel_cycles"]
        self.ldgsts_finite_resident_maximum_panels = mem_sub[
            "ldgsts_finite_resident_maximum_panels"]
        self.ldgsts_finite_resident_reference_sha256 = mem_sub[
            "ldgsts_finite_resident_reference_sha256"]
        self.ldgsts_queued_steady_minimum_grid_ctas = mem_sub[
            "ldgsts_queued_steady_minimum_grid_ctas"]
        self.ldgsts_queued_steady_maximum_panels_per_cta = mem_sub[
            "ldgsts_queued_steady_maximum_panels_per_cta"]
        self.ldgsts_queued_steady_average_panels_per_cta = mem_sub[
            "ldgsts_queued_steady_average_panels_per_cta"]
        self.ldgsts_queued_steady_full_chip_initial_32b_transactions_per_cycle = (
            mem_sub[
                "ldgsts_queued_steady_full_chip_initial_32b_transactions_per_cycle"])
        self.ldgsts_queued_steady_reference_sha256 = mem_sub[
            "ldgsts_queued_steady_reference_sha256"]
        ldgsts_rates = (
            self.ldgsts_initial_32b_transactions_single_sm_cycle,
            self.ldgsts_initial_32b_transactions_per_active_sm_cycle,
            self.ldgsts_initial_32b_transactions_full_chip_per_cycle,
        )
        ldgsts_finite_resident = (
            self.ldgsts_finite_resident_panel_bytes,
            self.ldgsts_finite_resident_full_chip_ctas,
            self.ldgsts_finite_resident_single_panel_cycles,
            self.ldgsts_finite_resident_affine_startup_cycles,
            self.ldgsts_finite_resident_affine_panel_cycles,
            self.ldgsts_finite_resident_maximum_panels,
            self.ldgsts_finite_resident_reference_sha256,
        )
        ldgsts_queued_steady = (
            self.ldgsts_queued_steady_minimum_grid_ctas,
            self.ldgsts_queued_steady_maximum_panels_per_cta,
            self.ldgsts_queued_steady_average_panels_per_cta,
            self.ldgsts_queued_steady_full_chip_initial_32b_transactions_per_cycle,
            self.ldgsts_queued_steady_reference_sha256,
        )
        if any(value is None for value in ldgsts_rates):
            if not all(value is None for value in ldgsts_rates):
                raise ValueError(
                    "LDGSTS frontend service requires all three single-SM, "
                    "per-active-SM, and full-chip transaction rates")
        else:
            if self.kernel_memory_model != "gpu_mshr":
                raise ValueError(
                    "LDGSTS frontend service requires gpu_mshr hardware")
            if self.l2_transaction_bytes != 32:
                raise ValueError(
                    "LDGSTS frontend service is measured in initial 32-byte "
                    "transactions")
            if any(float(value) <= 0.0 for value in ldgsts_rates):
                raise ValueError(
                    "LDGSTS frontend transaction rates must be positive")
        if any(value is None for value in ldgsts_finite_resident):
            if not all(value is None for value in ldgsts_finite_resident):
                raise ValueError(
                    "finite resident LDGSTS service requires its complete "
                    "panel, cohort, endpoint, affine, and domain fields")
        else:
            if all(value is None for value in ldgsts_rates):
                raise ValueError(
                    "finite resident LDGSTS service requires the base "
                    "LDGSTS frontend service")
            if self.l2_transaction_bytes != 32:
                raise ValueError(
                    "finite resident LDGSTS service requires 32-byte "
                    "transactions")
            if any(float(value) <= 0.0
                   for value in ldgsts_finite_resident[:-1]):
                raise ValueError(
                    "finite resident LDGSTS service fields must be positive")
            reference_sha256 = (
                self.ldgsts_finite_resident_reference_sha256)
            if (not isinstance(reference_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", reference_sha256)
                    is None):
                raise ValueError(
                    "finite resident LDGSTS reference must be a lowercase "
                    "SHA256")
            if (self.ldgsts_finite_resident_panel_bytes
                    % self.l2_transaction_bytes):
                raise ValueError(
                    "finite resident LDGSTS panel bytes must be divisible "
                    "by the transaction size")
            if self.ldgsts_finite_resident_maximum_panels < 2:
                raise ValueError(
                    "finite resident LDGSTS affine domain must include at "
                    "least two panels")
        if any(value is None for value in ldgsts_queued_steady):
            if not all(value is None for value in ldgsts_queued_steady):
                raise ValueError(
                    "queued steady LDGSTS service requires its complete "
                    "grid, source-period, rate, and reference fields")
        else:
            if all(value is None for value in ldgsts_rates):
                raise ValueError(
                    "queued steady LDGSTS service requires the base LDGSTS "
                    "frontend service")
            if any(value is None for value in ldgsts_finite_resident):
                raise ValueError(
                    "queued steady LDGSTS service requires the finite "
                    "resident panel and cohort contract")
            if self.l2_transaction_bytes != 32:
                raise ValueError(
                    "queued steady LDGSTS service requires 32-byte "
                    "transactions")
            if any(float(value) <= 0.0
                   for value in ldgsts_queued_steady[:-1]):
                raise ValueError(
                    "queued steady LDGSTS service fields must be positive")
            steady_reference_sha256 = (
                self.ldgsts_queued_steady_reference_sha256)
            if (not isinstance(steady_reference_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", steady_reference_sha256)
                    is None):
                raise ValueError(
                    "queued steady LDGSTS reference must be a lowercase "
                    "SHA256")
            if (self.ldgsts_queued_steady_minimum_grid_ctas
                    <= self.ldgsts_finite_resident_full_chip_ctas):
                raise ValueError(
                    "queued steady LDGSTS grid must exceed the resident "
                    "full-chip cohort")
            if (self.ldgsts_queued_steady_average_panels_per_cta
                    > self.ldgsts_queued_steady_maximum_panels_per_cta):
                raise ValueError(
                    "queued steady LDGSTS average panels must not exceed "
                    "the source-period maximum")
        # Whether the profiler provides trustworthy per-kernel-invocation byte
        # counters.  This is an explicit observation capability, not inferred
        # from hardware identity.  C500 layer-1 numbered reports and NVIDIA NCU
        # reports satisfy the current contract; whole-shape reports do not.
        profiler_traffic_reliable = mem_sub[
            "profiler_per_kernel_traffic_reliable"]
        if not isinstance(profiler_traffic_reliable, bool):
            raise ValueError(
                "memory_subsystem.profiler_per_kernel_traffic_reliable must "
                "be boolean")
        self.profiler_per_kernel_traffic_reliable = profiler_traffic_reliable
        self.memory_subsystem = mem_sub
        self.memory_access_class_services = mem_sub[
            "memory_access_class_services"]
        self.dram_arbitration_mode = mem_sub["dram_arbitration_mode"]
        self.dram_read_capacity_per_clk = (
            None if mem_sub["dram_read_capacity_per_clk"] is None
            else float(mem_sub["dram_read_capacity_per_clk"]))
        self.dram_write_capacity_per_clk = (
            None if mem_sub["dram_write_capacity_per_clk"] is None
            else float(mem_sub["dram_write_capacity_per_clk"]))
        if self.kernel_memory_model in ("gpu_mshr", "npu_tile"):
            dram_resource = self.resource_by_role("dram")
            if dram_resource.scope != "shared":
                raise ValueError(
                    "target DRAM arbitration requires one shared resource")
            if dram_resource._runtime_clock_source != "dram":
                raise ValueError(
                    "target DRAM arbitration capacity must use the configured "
                    "DRAM clock domain")
            if self.dram_arbitration_mode in (
                    "shared_directional_capacity",
                    "independent_directional_capacity"):
                if not math.isclose(
                        dram_resource._capacity_per_clk,
                        self.dram_read_capacity_per_clk,
                        rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        "the canonical DRAM resource capacity must equal the "
                        "directional read capacity")
            elif self.dram_arbitration_mode == "shared_common_capacity":
                pass
            else:
                raise AssertionError(
                    "memory normalization admitted an unknown DRAM "
                    f"arbitration mode: {self.dram_arbitration_mode!r}")

        if self.kernel_memory_model == "gpu_mshr":
            if (self.l2_transaction_bytes is None
                    or self.dram_transaction_bytes is None
                    or self.l2_transaction_bytes <= 0
                    or self.dram_transaction_bytes <= 0):
                raise ValueError(
                    "GPU transaction sizes must be positive")
            local_cycles = float(mem_sub["l2_hit_latency_local_cycles"])
            remote_cycles = float(mem_sub["l2_hit_latency_remote_cycles"])
            local_fraction_raw = mem_sub["l2_local_fraction"]
            dram_extra_cycles = float(mem_sub["dram_miss_extra_cycles"])
            if local_cycles <= 0 or remote_cycles <= 0:
                raise ValueError(
                    "GPU MSHR L2 hit latencies must be positive")
            if local_fraction_raw is None:
                if not math.isclose(
                        local_cycles, remote_cycles,
                        rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        "GPU MSHR requires l2_local_fraction when local and "
                        "remote L2 latencies differ")
                modeled_l2_cycles = local_cycles
            else:
                local_fraction = float(local_fraction_raw)
                if not 0.0 <= local_fraction <= 1.0:
                    raise ValueError(
                        "GPU MSHR l2_local_fraction must be within [0, 1]")
                modeled_l2_cycles = (
                    local_fraction * local_cycles
                    + (1.0 - local_fraction) * remote_cycles)
            if dram_extra_cycles < 0:
                raise ValueError(
                    "GPU MSHR DRAM miss extra latency cannot be negative")

            cycles_per_ns = float(
                clock_domains[model_clock_domain]) * 1e-9
            measured_l2_cycles = self.l2_latency_ns * cycles_per_ns
            modeled_dram_cycles = modeled_l2_cycles + dram_extra_cycles
            measured_dram_cycles = self.dram_latency_ns * cycles_per_ns
            if not math.isclose(
                    modeled_l2_cycles, measured_l2_cycles,
                    rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    "GPU MSHR latency contract is inconsistent: weighted "
                    "L2 hit cycles must equal l2_latency_ns at the model clock")
            if not math.isclose(
                    modeled_dram_cycles, measured_dram_cycles,
                    rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    "GPU MSHR latency contract is inconsistent: L2 hit plus "
                    "DRAM miss extra cycles must equal dram_latency_ns at the "
                    "model clock")

        if (self.source_slots_per_unit is not None
                and self.source_slots_per_unit <= 0):
            raise ValueError(
                "source_slots_per_unit must be positive when configured")

        if (self.dram_latency_ns is not None
                and self.dram_latency_ns <= 0):
            raise ValueError(
                f"dram_latency_ns must be positive, got {self.dram_latency_ns}")

        self.nominal_sm_freq_hz = float(clock_domains[model_clock_domain])
        self.nominal_dram_freq_hz = float(clock_domains[dram_clock_domain])
        if "dram_finite_tma_read_stream_ramp" in sched:
            binding = sched["dram_finite_tma_read_stream_ramp"]
            self.dram_finite_tma_read_stream_ramp = _normalize_dram_finite_tma_read_stream_ramp(
                binding, repository_root, self)
            profiles = binding["additional_profiles"] if "additional_profiles" in binding else []
            if not isinstance(profiles, list):
                raise ValueError("TMA additional_profiles must be a list")
            self.dram_finite_tma_read_stream_profiles = tuple(
                _normalize_tma_balanced_read_profile(v, repository_root, self) for v in profiles)
            scopes = [(tuple(v["threads_per_cta"]), tuple(v["balanced_operand_bytes"]))
                for v in self.dram_finite_tma_read_stream_profiles]
            if len(set(scopes)) != len(scopes):
                raise ValueError("ambiguous balanced TMA profiles")
        else:
            self.dram_finite_tma_read_stream_ramp = None
            self.dram_finite_tma_read_stream_profiles = ()
        if (self.memory_dependency_task_latency_model is not None
                and self.nominal_sm_freq_hz
                > self.memory_dependency_task_latency_model[
                    "maximum_sm_clock_hz"]):
            raise ValueError(
                "memory-dependency task floor does not cover the nominal "
                "model clock")

        ic = cfg["interconnect"]

        def _optional_float(key, scale=1.0):
            value = ic[key]
            if value is None:
                return None
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))):
                raise TypeError(f"interconnect.{key} must be numeric or null")
            return float(value) * scale

        self.interconnect = {
            "nvlink_bw_bytes_s": _optional_float(
                "nvlink_unidir_bw_GB_s", 1e9),
            "nvlink_signal_latency_ns": _optional_float(
                "nvlink_signal_latency_ns"),
            "nccl_alpha_ns": _optional_float("nccl_alpha_ns"),
            "pcie_bw_bytes_s": _optional_float("pcie_bw_GB_s", 1e9),
            "pcie_latency_ns": _optional_float("pcie_latency_ns"),
            "infiniband_bw_bytes_s": _optional_float(
                "infiniband_unidir_bw_GB_s", 1e9),
        }
        if any(value is not None and (
                not math.isfinite(value) or value <= 0)
               for value in self.interconnect.values()):
            raise ValueError(
                "configured interconnect values must be finite and positive")
        if "custom_all_reduce_service" in ic:
            self.custom_all_reduce_service = (
                _normalize_custom_all_reduce_service(
                    ic["custom_all_reduce_service"]))
        else:
            self.custom_all_reduce_service = None
        if "sglang_custom_all_reduce_service" in ic:
            self.sglang_custom_all_reduce_service = (
                _normalize_custom_all_reduce_service(
                    ic["sglang_custom_all_reduce_service"],
                    "sglang_custom_all_reduce_service"))
        else:
            self.sglang_custom_all_reduce_service = None
        if "sglang_flashinfer_allreduce_fusion_service" in ic:
            self.sglang_flashinfer_allreduce_fusion_service = (
                _normalize_custom_all_reduce_service(
                    ic["sglang_flashinfer_allreduce_fusion_service"],
                    "sglang_flashinfer_allreduce_fusion_service"))
        else:
            self.sglang_flashinfer_allreduce_fusion_service = None

    def empty_kernel_duration_for_launch(
            self, grid_size, block_size, launch_grid=None, launch_block=None):
        if isinstance(grid_size, bool) or not isinstance(grid_size, int):
            raise TypeError("dispatch grid_size must be an integer")
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError("dispatch block_size must be an integer")
        if grid_size <= 0 or block_size <= 0:
            raise ValueError("dispatch grid_size and block_size must be positive")
        if self.launch_latency_model is not None:
            maximum_block_threads = min(1024, self.max_threads_per_unit)
            if block_size > maximum_block_threads:
                raise ValueError(
                    "launch block size exceeds the hardware thread limit: "
                    f"block_size={block_size}, maximum="
                    f"{maximum_block_threads}")
            dimensions = []
            for name, supplied, flattened in (
                    ("launch_grid", launch_grid, grid_size),
                    ("launch_block", launch_block, block_size)):
                if supplied is None:
                    dimensions.append((flattened, 1, 1))
                    continue
                if (not isinstance(supplied, (list, tuple))
                        or len(supplied) != 3
                        or any(isinstance(axis, bool)
                               or not isinstance(axis, int) or axis <= 0
                               for axis in supplied)):
                    raise ValueError(f"{name} must be a positive dim3")
                normalized = tuple(supplied)
                if math.prod(normalized) != flattened:
                    raise ValueError(
                        f"{name} disagrees with its flattened size")
                dimensions.append(normalized)
            normalized_grid, normalized_block = dimensions
            model = self.launch_latency_model
            waves_per_block = math.ceil(block_size / self.parallel_width)
            scheduling_rounds = math.ceil(
                grid_size / self.num_compute_units)
            grid_rank = max(
                1, sum(axis > 1 for axis in normalized_grid))
            block_rank = max(
                1, sum(axis > 1 for axis in normalized_block))
            if model["model"] == "isolated_empty_task_scheduler_v1":
                round_service_ns = max(
                    model["cta_round_ns"],
                    model["wave_round_ns"] * waves_per_block,
                )
                return (
                    model["submit_ns"]
                    + model["extra_grid_dimension_ns"] * (grid_rank - 1)
                    + model["extra_block_dimension_ns"] * (block_rank - 1)
                    + scheduling_rounds * round_service_ns
                )
            elif model["model"] == "isolated_nonempty_task_scheduler_v1":
                if waves_per_block == 1:
                    round_service_ns = model["single_wave_round_ns"]
                else:
                    round_service_ns = (
                        model["multi_wave_round_ns"] * waves_per_block)
                return (
                    model["submit_ns"]
                    + grid_size * model["cta_descriptor_ns"]
                    + scheduling_rounds * round_service_ns
                )
            else:
                raise ValueError(
                    "unsupported normalized launch latency model: "
                    f"{model['model']!r}")
        if self.empty_kernel_duration_lut is None:
            if self.empty_kernel_duration_ns is None:
                return 0.0
            return self.empty_kernel_duration_ns
        lut_kind = self.empty_kernel_duration_lut[0]
        if lut_kind == "exact_dim3":
            if launch_grid is None or launch_block is None:
                raise ValueError(
                    "exact empty-kernel LUT requires full launch grid and block"
                )
            dimensions = []
            for name, dim3 in (
                    ("launch_grid", launch_grid),
                    ("launch_block", launch_block)):
                if (not isinstance(dim3, (list, tuple)) or len(dim3) != 3
                        or any(isinstance(axis, bool)
                               or not isinstance(axis, int) or axis <= 0
                               for axis in dim3)):
                    raise ValueError(f"{name} must be a positive dim3")
                dimensions.append(tuple(dim3))
            normalized_grid, normalized_block = dimensions
            if math.prod(normalized_grid) != grid_size:
                raise ValueError(
                    "full launch grid disagrees with flattened grid_size"
                )
            if math.prod(normalized_block) != block_size:
                raise ValueError(
                    "full launch block disagrees with flattened block_size"
                )
            duration_by_geometry = self.empty_kernel_duration_lut[1]
            identity = (normalized_grid, normalized_block)
            if identity not in duration_by_geometry:
                raise ValueError(
                    "launch geometry is absent from the exact empty-kernel LUT: "
                    f"grid={normalized_grid}, block={normalized_block}"
                )
            return duration_by_geometry[identity]
        if lut_kind != "interpolated_flat":
            raise AssertionError(
                f"unknown empty-kernel LUT kind: {lut_kind!r}"
            )
        _, grid_points, block_points, duration_by_block = (
            self.empty_kernel_duration_lut)
        if block_size < block_points[0] or block_size > block_points[-1]:
            raise ValueError(
                "dispatch block size is outside the measured LUT: "
                f"block_size={block_size}, range="
                f"[{block_points[0]}, {block_points[-1]}]")
        durations_at_grid = {
            point: _linear_interpolate(
                grid_points, duration_by_block[point], grid_size)
            for point in block_points
        }
        return _linear_interpolate(
            block_points,
            tuple(durations_at_grid[point] for point in block_points),
            block_size)

    def fa2_source_task_outer_envelope_ns(self, grid_size, block_size):
        """Return the measured source-pipeline outer completion envelope."""

        model = self.fa2_source_task_outer_envelope
        if model is None:
            raise ValueError("hardware has no FA2 source task outer envelope")
        if (isinstance(grid_size, bool) or not isinstance(grid_size, int)
                or grid_size <= 0):
            raise ValueError("FA2 source task grid_size must be positive")
        if block_size != model["threads_per_cta"]:
            raise ValueError(
                "FA2 source task block size is outside the measured "
                "outer-envelope geometry")
        if not (model["minimum_grid_size"] <= grid_size
                <= model["maximum_grid_size"]):
            raise ValueError(
                "FA2 source task grid size is outside the measured "
                "outer-envelope range")
        return model["maximum_outer_envelope_ns"]

    def has_task_duration_floor_contract(self, contract):
        if contract == WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR:
            model = self.memory_dependency_task_latency_model
            return model is not None and model["contract"] == contract
        if contract in (
                TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
                TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR):
            model = self.tensor_iterator_copy_task_latency_model
            if model is None:
                return False
            return any(
                entry["contract"] == contract for entry in model["entries"])
        raise ValueError(
            f"unknown task-duration floor contract: {contract!r}")

    def _tensor_iterator_copy_task_entry(self, contract):
        model = self.tensor_iterator_copy_task_latency_model
        if model is None:
            raise ValueError(
                "hardware has no TensorIterator copy task floor")
        for entry in model["entries"]:
            if entry["contract"] == contract:
                return entry
        raise ValueError(
            f"unknown task-duration floor contract: {contract!r}")

    def task_duration_floor_contract_supports_launch(
            self, contract, grid_size, block_size, launch_grid,
            launch_block, sm_freq_hz):
        """Return whether a typed launch lies inside a measured mechanism."""

        if not self.has_task_duration_floor_contract(contract):
            return False
        for name, value in (
                ("grid_size", grid_size), ("block_size", block_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"task-duration floor {name} must be an integer")
            if value <= 0:
                raise ValueError(f"task-duration floor {name} must be positive")
        dimensions = []
        for name, supplied, flattened in (
                ("launch_grid", launch_grid, grid_size),
                ("launch_block", launch_block, block_size)):
            if (not isinstance(supplied, (list, tuple))
                    or len(supplied) != 3
                    or any(isinstance(axis, bool)
                           or not isinstance(axis, int) or axis <= 0
                           for axis in supplied)):
                raise ValueError(
                    f"task-duration floor {name} must be a positive dim3")
            normalized = tuple(supplied)
            if math.prod(normalized) != flattened:
                raise ValueError(
                    f"task-duration floor {name} disagrees with its "
                    "flattened size")
            dimensions.append(normalized)
        if (isinstance(sm_freq_hz, bool)
                or not isinstance(sm_freq_hz, (int, float))):
            raise TypeError("task-duration floor sm_freq_hz must be numeric")
        normalized_frequency = float(sm_freq_hz)
        if not math.isfinite(normalized_frequency) or normalized_frequency <= 0:
            raise ValueError(
                "task-duration floor sm_freq_hz must be finite and positive")
        normalized_grid, normalized_block = dimensions
        if contract == WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR:
            model = self.memory_dependency_task_latency_model
            if model["model"] == (
                    "isolated_warm_default_cache_memory_dependency_task_v1"):
                grid_rank_supported = normalized_grid[1:] == (1, 1)
            elif model["model"] in (
                    "paired_batched_warm_default_cache_memory_dependency_task_v1",
                    "paired_batched_multiround_warm_default_cache_"
                    "memory_dependency_task_v1",
                    "isolated_mcevent_multiround_warm_default_cache_"
                    "memory_dependency_task_v1",
                    "paired_mcevent_active_unit_"
                    "memory_dependency_increment_v1"):
                grid_rank_supported = (
                    normalized_grid[2] == 1
                    and model["maximum_launch_rank"] >= (
                        2 if normalized_grid[1] > 1 else 1))
            else:
                raise AssertionError(
                    "unhandled memory-dependency task model: "
                    f"{model['model']!r}")
            return (
                grid_rank_supported
                and normalized_block[1:] == (1, 1)
                and model["minimum_grid_size"] <= grid_size
                <= model["maximum_grid_size"]
                and model["minimum_block_size"] <= block_size
                <= model["maximum_block_size"]
                and normalized_frequency <= model["maximum_sm_clock_hz"]
            )
        self._tensor_iterator_copy_task_entry(contract)
        model = self.tensor_iterator_copy_task_latency_model
        return (
            normalized_grid[1:] == (1, 1)
            and normalized_block == (model["threads_per_cta"], 1, 1)
            and model["minimum_grid_size"] <= grid_size
            <= model["maximum_grid_size"]
        )

    def memory_dependency_task_duration_ns(
            self, contract, grid_size, block_size, launch_grid,
            launch_block, sm_freq_hz):
        """Return the bounded complete warm load-dependency-store floor."""

        if not self.has_task_duration_floor_contract(contract):
            raise ValueError(
                "hardware has no matching memory-dependency task floor")
        if not self.task_duration_floor_contract_supports_launch(
                contract, grid_size, block_size, launch_grid, launch_block,
                sm_freq_hz):
            raise ValueError(
                "memory-dependency task launch is outside the measured "
                "rank/geometry/clock support")
        model = self.memory_dependency_task_latency_model
        if model["model"] == (
                "isolated_warm_default_cache_memory_dependency_task_v1"):
            duration_ns = (
                model["base_ns"]
                + grid_size * model["cta_ns"]
                + math.ceil(block_size / self.parallel_width)
                * model["warp_ns"])
            if grid_size >= model["fanout_activation_grid"]:
                duration_ns += model["fanout_activation_ns"]
        elif model["model"] == (
                "paired_batched_warm_default_cache_memory_dependency_task_v1"):
            duration_ns = (
                model["base_ns"]
                + grid_size * model["cta_ns"]
                + math.ceil(block_size / self.parallel_width)
                * model["warp_ns"])
        elif model["model"] in (
                "paired_batched_multiround_warm_default_cache_"
                "memory_dependency_task_v1",
                "isolated_mcevent_multiround_warm_default_cache_"
                "memory_dependency_task_v1"):
            resident_ctas = min(grid_size, self.num_compute_units)
            full_chip_rounds = math.ceil(grid_size / self.num_compute_units)
            waves_per_cta = math.ceil(block_size / self.parallel_width)
            duration_ns = (
                model["base_ns"]
                + resident_ctas * model["resident_cta_ns"]
                + full_chip_rounds * waves_per_cta
                * model["wave_round_ns"])
        elif model["model"] == (
                "paired_mcevent_active_unit_"
                "memory_dependency_increment_v1"):
            duration_ns = self.empty_kernel_duration_for_launch(
                grid_size, block_size, launch_grid, launch_block)
            duration_ns += self.memory_dependency_task_increment_ns(
                contract, grid_size, block_size, launch_grid, launch_block,
                sm_freq_hz)
        else:
            raise AssertionError(
                f"unhandled memory-dependency task model: {model['model']!r}")
        return duration_ns

    def memory_dependency_task_increment_ns(
            self, contract, grid_size, block_size, launch_grid,
            launch_block, sm_freq_hz):
        """Return non-empty task service above its paired empty launch."""

        if not self.has_task_duration_floor_contract(contract):
            raise ValueError(
                "hardware has no matching memory-dependency task floor")
        if not self.task_duration_floor_contract_supports_launch(
                contract, grid_size, block_size, launch_grid, launch_block,
                sm_freq_hz):
            raise ValueError(
                "memory-dependency task launch is outside the measured "
                "rank/geometry/clock support")
        model = self.memory_dependency_task_latency_model
        if model["model"] == (
                "paired_mcevent_active_unit_"
                "memory_dependency_increment_v1"):
            active_units = min(grid_size, self.num_compute_units)
            return (
                model["first_active_unit_increment_ns"]
                + (active_units - 1)
                * model["additional_active_unit_increment_ns"])
        complete_ns = self.memory_dependency_task_duration_ns(
            contract, grid_size, block_size, launch_grid, launch_block,
            sm_freq_hz)
        empty_ns = self.empty_kernel_duration_for_launch(
            grid_size, block_size, launch_grid, launch_block)
        increment_ns = complete_ns - empty_ns
        if increment_ns <= 0:
            raise ValueError(
                "memory-dependency task increment must be positive")
        return increment_ns

    def tensor_iterator_copy_task_duration_ns(
            self, contract, grid_size, block_size, launch_grid,
            launch_block, sm_freq_hz):
        """Return a source-shaped TensorIterator copy floor in nanoseconds."""

        entry = self._tensor_iterator_copy_task_entry(contract)
        if not self.task_duration_floor_contract_supports_launch(
                contract, grid_size, block_size, launch_grid, launch_block,
                sm_freq_hz):
            raise ValueError(
                "TensorIterator copy task launch is outside the measured "
                "rank/geometry support")
        model = self.tensor_iterator_copy_task_latency_model
        cycles = (
            entry["base_cycles"]
            + (entry["fanout_activation_cycles"]
               if grid_size >= model["fanout_activation_grid"] else 0.0)
            + grid_size * entry["cta_cycles"]
        )
        return cycles / (float(sm_freq_hz) * 1.0e-9)

    def execution_units_for_task(self, launch_task_kind):
        """Return the physical execution-domain size for one launch task.

        ``num_compute_units`` is the primary SM/AP/AIC domain.  A separated
        AIV launch instead uses the independently declared Vector-core count.
        Mixed CANN tasks express ``active_units`` in AIC block-dim units; the
        fixed 1:2 AIC:AIV contract has the same active fraction in both
        domains, so the primary AIC count is the normalization denominator.
        """

        if launch_task_kind == "generic":
            if self.aic_aiv_separated:
                raise ValueError(
                    "separated AIC/AIV hardware cannot use a generic "
                    "execution domain")
            return self.num_compute_units
        elif launch_task_kind in ("aic", "mixed"):
            if not self.aic_aiv_separated:
                raise ValueError(
                    f"{launch_task_kind} execution domain requires "
                    "separated AIC/AIV hardware")
            return self.num_compute_units
        elif launch_task_kind in ("aiv", "mix_aiv"):
            if not self.aic_aiv_separated:
                raise ValueError(
                    "AIV execution domain requires separated AIC/AIV "
                    "hardware")
            if "compute_vector" not in self._role_map:
                raise ValueError(
                    "AIV execution domain requires a compute_vector "
                    "resource")
            vector_units = self.resource_by_role(
                "compute_vector").total_units
            if vector_units <= 0:
                raise ValueError(
                    "AIV execution domain requires positive physical "
                    "Vector-core count")
            return vector_units
        else:
            raise ValueError(
                f"unknown launch task kind: {launch_task_kind!r}")

    def vector_reduce_sum_latency_ns(self, element_count):
        if isinstance(element_count, bool) or not isinstance(element_count, int):
            raise TypeError("vector reduction element_count must be an integer")
        if element_count <= 0:
            raise ValueError("vector reduction element_count must be positive")
        if self.vector_reduce_sum_latency_lut is None:
            raise ValueError(
                "hardware has no vector ReduceSum latency measurement")
        points, durations = self.vector_reduce_sum_latency_lut
        return _linear_interpolate(points, durations, element_count)

    @staticmethod
    def _exact_empty_kernel_duration_ns(
            block_dim, points, durations, task_kind):
        if block_dim > points[-1]:
            raise ValueError(
                f"{task_kind} empty-kernel block_dim is outside the measured "
                f"range [1, {points[-1]}]: {block_dim}")
        point_index = block_dim - 1
        if points[point_index] != block_dim:
            raise AssertionError(
                f"{task_kind} empty-kernel LUT is not an exact integer sweep")
        return durations[point_index]

    def aic_empty_kernel_duration_ns(self, block_dim):
        if isinstance(block_dim, bool) or not isinstance(block_dim, int):
            raise TypeError("AIC empty-kernel block_dim must be an integer")
        if block_dim <= 0:
            raise ValueError("AIC empty-kernel block_dim must be positive")
        if self.aic_empty_kernel_duration_lut is None:
            raise ValueError(
                "hardware has no AIC empty-kernel duration measurement")
        points, durations = self.aic_empty_kernel_duration_lut
        return self._exact_empty_kernel_duration_ns(
            block_dim, points, durations, "AIC")

    def aiv_empty_kernel_duration_ns(self, block_dim):
        if isinstance(block_dim, bool) or not isinstance(block_dim, int):
            raise TypeError("AIV empty-kernel block_dim must be an integer")
        if block_dim <= 0:
            raise ValueError("AIV empty-kernel block_dim must be positive")
        if self.aiv_empty_kernel_duration_lut is None:
            raise ValueError(
                "hardware has no AIV empty-kernel duration measurement")
        points, durations = self.aiv_empty_kernel_duration_lut
        return self._exact_empty_kernel_duration_ns(
            block_dim, points, durations, "AIV")

    def aiv_nonempty_task_duration_ns(self, block_dim):
        if isinstance(block_dim, bool) or not isinstance(block_dim, int):
            raise TypeError(
                "AIV non-empty task block_dim must be an integer")
        if block_dim <= 0:
            raise ValueError(
                "AIV non-empty task block_dim must be positive")
        model = self.aiv_nonempty_task_latency_model
        if model is None:
            raise ValueError(
                "hardware has no AIV non-empty task latency model")
        minimum = model["minimum_aiv_block_dim"]
        maximum = model["maximum_aiv_block_dim"]
        if not minimum <= block_dim <= maximum:
            raise ValueError(
                "AIV non-empty task block_dim is outside the measured "
                f"range [{minimum}, {maximum}]: {block_dim}")
        return (
            model["submit_ns"]
            + block_dim * model["descriptor_ns_per_aiv"]
        )

    def mix_aiv_nonempty_task_duration_ns(self, block_dim):
        if isinstance(block_dim, bool) or not isinstance(block_dim, int):
            raise TypeError(
                "MIX_AIV non-empty task block_dim must be an integer")
        if block_dim <= 0:
            raise ValueError(
                "MIX_AIV non-empty task block_dim must be positive")
        model = self.mix_aiv_nonempty_task_latency_model
        if model is None:
            raise ValueError(
                "hardware has no MIX_AIV non-empty task latency model")
        minimum = model["minimum_aiv_block_dim"]
        maximum = model["maximum_aiv_block_dim"]
        if not minimum <= block_dim <= maximum:
            raise ValueError(
                "MIX_AIV non-empty task block_dim is outside the measured "
                f"range [{minimum}, {maximum}]: {block_dim}")
        return (
            model["submit_ns"]
            + block_dim * model["descriptor_ns_per_aiv"]
        )

    def aiv_worker_start_fill_ns(self, block_dim):
        if isinstance(block_dim, bool) or not isinstance(block_dim, int):
            raise TypeError(
                "AIV worker-start fill block_dim must be an integer")
        if block_dim <= 0:
            raise ValueError(
                "AIV worker-start fill block_dim must be positive")
        model = self.aiv_worker_start_fill_model
        if model is None:
            raise ValueError(
                "hardware has no AIV worker-start fill model")
        minimum = model["minimum_aiv_block_dim"]
        maximum = model["maximum_aiv_block_dim"]
        if not minimum <= block_dim <= maximum:
            raise ValueError(
                "AIV worker-start fill block_dim is outside the measured "
                f"range [{minimum}, {maximum}]: {block_dim}")
        return (
            (block_dim - 1)
            * model["entry_interval_cycles"]
            * model["system_counter_tick_ns"]
        )

    def aiv_memory_task_duration_ns(self, block_dim):
        if isinstance(block_dim, bool) or not isinstance(block_dim, int):
            raise TypeError(
                "AIV memory-task block_dim must be an integer")
        if block_dim <= 0:
            raise ValueError("AIV memory-task block_dim must be positive")
        model = self.aiv_memory_task_latency_model
        if model is None:
            raise ValueError(
                "hardware has no AIV memory-task latency model")
        minimum = model["minimum_aiv_block_dim"]
        maximum = model["maximum_aiv_block_dim"]
        if not minimum <= block_dim <= maximum:
            raise ValueError(
                "AIV memory-task block_dim is outside the measured range "
                f"[{minimum}, {maximum}]: {block_dim}")
        return (
            model["submit_ns"]
            + block_dim * model["descriptor_ns_per_aiv"]
        )

    def task_prologue_duration_ns(self, contract, active_units):
        if contract == CANN_AIV_TILING_MATERIALIZATION_PROLOGUE:
            model = self.aiv_tiling_materialization_latency_model
        elif contract == CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE:
            model = self.aiv_tiling_materialization_48b_latency_model
        elif contract == CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE:
            model = self.aiv_tiling_materialization_56b_latency_model
        elif contract == CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE:
            model = self.aiv_tiling_materialization_80b_latency_model
        elif contract == CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE:
            model = self.aiv_tiling_materialization_192b_latency_model
        else:
            raise ValueError(f"unknown task prologue contract: {contract!r}")
        if isinstance(active_units, bool) or not isinstance(active_units, int):
            raise TypeError("task prologue active_units must be an integer")
        if active_units <= 0:
            raise ValueError("task prologue active_units must be positive")
        if model is None:
            raise ValueError(
                f"hardware has no AIV tiling-materialization latency model "
                f"for {contract!r}")
        minimum = model["minimum_active_aiv_workers"]
        maximum = model["maximum_active_aiv_workers"]
        if not minimum <= active_units <= maximum:
            raise ValueError(
                "AIV tiling-materialization active_units is outside the "
                f"measured range [{minimum}, {maximum}]: {active_units}")
        if contract in (
                CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE,
                CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE,
                CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE):
            return max(
                model["single_worker_completion_ns"],
                active_units * model["serialized_ns_per_aiv"],
            )
        return (
            model["startup_ns"]
            + active_units * model["serialized_ns_per_aiv"]
        )

    def resident_cta_cohort_refill_ns(self, active_units, block_size):
        """Return scheduler work required to admit one full CTA cohort.

        The initial resident cohort can overlap the first task body after the
        host submission prefix.  A later cohort cannot begin until a resident
        slot is released, so its descriptor and wave-round issue span is an
        ordered refill.  The duration is derived entirely from the existing
        hardware-wide non-empty scheduler model and architectural residency.
        """

        for name, value in (
                ("active_units", active_units), ("block_size", block_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"resident CTA cohort {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"resident CTA cohort {name} must be positive")
        if active_units > self.num_compute_units:
            raise ValueError(
                "resident CTA cohort active units exceed hardware capacity")
        model = self.launch_latency_model
        if model is None or model["model"] not in (
                "isolated_empty_task_scheduler_v1",
                "isolated_nonempty_task_scheduler_v1"):
            raise ValueError(
                "resident CTA cohort refill requires a measured task "
                "scheduler model")
        waves_per_cta = math.ceil(block_size / self.parallel_width)
        resident_ctas = min(
            self.max_blocks_per_unit,
            self.max_threads_per_unit // block_size,
            self.max_warps_per_unit // waves_per_cta,
        )
        if resident_ctas <= 0:
            raise ValueError(
                "resident CTA cohort launch exceeds hardware capacity")
        if model["model"] == "isolated_empty_task_scheduler_v1":
            round_service_ns = max(
                model["cta_round_ns"],
                model["wave_round_ns"] * waves_per_cta,
            )
            return resident_ctas * round_service_ns
        wave_round_ns = (
            model["single_wave_round_ns"]
            if waves_per_cta == 1
            else model["multi_wave_round_ns"] * waves_per_cta)
        return (
            active_units * resident_ctas * model["cta_descriptor_ns"]
            + resident_ctas * wave_round_ns
        )

    def mixed_empty_kernel_duration_ns(self, aic_block_dim):
        if (isinstance(aic_block_dim, bool)
                or not isinstance(aic_block_dim, int)):
            raise TypeError(
                "mixed empty-kernel aic_block_dim must be an integer")
        if aic_block_dim <= 0:
            raise ValueError(
                "mixed empty-kernel aic_block_dim must be positive")
        if self.mixed_empty_kernel_duration_lut is None:
            raise ValueError(
                "hardware has no mixed empty-kernel duration measurement")
        points, aiv_block_dim_ratio, durations = (
            self.mixed_empty_kernel_duration_lut)
        if aiv_block_dim_ratio != 2:
            raise AssertionError(
                "mixed empty-kernel LUT lost its AIC:AIV 1:2 contract")
        return self._exact_empty_kernel_duration_ns(
            aic_block_dim, points, durations, "mixed")

    def fia_cross_domain_three_handoff_time_ns(self, task_counts):
        """Return the bounded FIA whole-program synchronization span."""

        if not isinstance(task_counts, tuple) or not task_counts:
            raise TypeError(
                "FIA cross-domain task_counts must be a non-empty tuple")
        if self.fia_cross_domain_three_handoff is None:
            raise ValueError(
                "hardware has no FIA cross-domain three-handoff mechanism")
        mechanism = self.fia_cross_domain_three_handoff
        for task_count in task_counts:
            if isinstance(task_count, bool) or not isinstance(task_count, int):
                raise TypeError(
                    "FIA cross-domain task counts must be integers")
            if not (mechanism["minimum_task_count"] <= task_count
                    <= mechanism["maximum_task_count"]):
                raise ValueError(
                    "FIA cross-domain task count is outside the validated "
                    "range")
        active_aic_groups = len(task_counts)
        if not (mechanism["minimum_active_aic_groups"]
                <= active_aic_groups
                <= mechanism["maximum_active_aic_groups"]):
            raise ValueError(
                "FIA cross-domain active_aic_groups is outside the validated "
                "range")
        try:
            lut_index = mechanism["single_task_active_aic_groups"].index(
                active_aic_groups)
        except ValueError as error:
            raise ValueError(
                "FIA cross-domain active_aic_groups lacks an exact "
                "single-task mechanism point") from error
        single_task_span = mechanism["single_task_duration_ns"][lut_index]
        multi_task_counts = tuple(
            value for value in task_counts if value >= 2)
        if not multi_task_counts:
            return single_task_span
        active_multi_groups = len(multi_task_counts)
        startup = max(
            mechanism["startup_local_ns"],
            active_multi_groups
            * mechanism["startup_shared_ns_per_active_group"],
        )
        multi_task_span = startup
        for task_depth in range(3, max(multi_task_counts) + 1):
            active_groups = sum(
                value >= task_depth for value in multi_task_counts)
            multi_task_span += max(
                mechanism["steady_local_ns"],
                active_groups
                * mechanism["steady_shared_ns_per_active_group"],
            )
        return max(single_task_span, multi_task_span)

    def metax_fa_tile16_operand_completion_time_ns(
            self, active_ap_count, panel_count, head_dim,
            query_heads_per_workgroup):
        """Return one exact MetaX tile16 operand-completion point, if covered."""

        for value, name in (
                (active_ap_count, "active AP count"),
                (panel_count, "panel count"),
                (head_dim, "head dimension")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"MetaX tile16 operand completion {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"MetaX tile16 operand completion {name} must be positive")
        if query_heads_per_workgroup is not None:
            if (isinstance(query_heads_per_workgroup, bool)
                    or not isinstance(query_heads_per_workgroup, int)):
                raise TypeError(
                    "MetaX tile16 operand completion query-heads per "
                    "workgroup must be an integer or None")
            if query_heads_per_workgroup <= 0:
                raise ValueError(
                    "MetaX tile16 operand completion query-heads per "
                    "workgroup must be positive")
        mechanism = self.fa_tile16_operand_completion
        if mechanism is None:
            return None
        for point in mechanism["points"]:
            if (point["active_ap_count"] == active_ap_count
                    and point["panel_count"] == panel_count
                    and (point["query_heads_per_workgroup"] is None
                         or point["query_heads_per_workgroup"]
                         == query_heads_per_workgroup)):
                for point_head_dim, completion_ns in (
                        point["completion_ns_by_head_dim"].items()):
                    if point_head_dim == head_dim:
                        return completion_ns
        return None

    def fia_regular_aiv_instruction_cold_fill_time_ns(
            self, active_aic_groups, source_s2, head_dim):
        """Return an exact regular-FIA cold-fill point, if covered."""

        for value, name in (
                (active_aic_groups, "active AIC group count"),
                (source_s2, "source S2"),
                (head_dim, "head dimension")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"FIA instruction cold-fill {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"FIA instruction cold-fill {name} must be positive")
        mechanism = self.fia_aiv_instruction_cold_fill
        if mechanism is None:
            raise ValueError(
                "hardware has no FIA AIV instruction cold-fill mechanism")
        for point in mechanism["regular_points"]:
            if (active_aic_groups == point["active_aic_groups"]
                    and source_s2 == point["source_s2"]
                    and head_dim in point["head_dims"]):
                return point["duration_ns"]
        return None

    def fia_mla_topk_scan_times_ns(
            self, active_aic_groups, source_s2, aic_pipeline_kind,
            maximum_task_count):
        """Return exact first/steady CalcSinnerTopKBegin scan service."""

        for value, name in (
                (active_aic_groups, "active AIC group count"),
                (source_s2, "source S2"),
                (maximum_task_count, "maximum task count")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"FIA MLA TopK scan {name} must be an integer")
            if value <= 0:
                raise ValueError(f"FIA MLA TopK scan {name} must be positive")
        if not isinstance(aic_pipeline_kind, str) or not aic_pipeline_kind:
            raise TypeError(
                "FIA MLA TopK scan pipeline kind must be a non-empty string")
        mechanisms = (
            self.fia_mla_topk_scan_service,
            self.fia_mla_topk_scan_steady_service,
        )
        for mechanism in mechanisms:
            if mechanism is None:
                continue
            if (active_aic_groups != mechanism["active_aic_groups"]
                    or source_s2 != mechanism["source_s2"]
                    or aic_pipeline_kind != mechanism["pipeline_kind"]):
                continue
            measured_maximum = mechanism["maximum_tasks_per_aic"]
            if (measured_maximum is not None
                    and maximum_task_count > measured_maximum):
                continue
            if mechanism["sparse_block_size"] != 1:
                raise ValueError(
                    "FIA MLA TopK scan requires sparse block size one")
            if mechanism["indices_per_task"] != source_s2:
                raise ValueError(
                    "FIA MLA TopK scan index count disagrees with source S2")
            return (
                mechanism["first_task_ns"],
                mechanism["subsequent_task_ns"],
            )
        return None

    def fia_split_fuse_aiv_instruction_cold_fill_time_ns(
            self, active_aic_groups):
        """Return the exact per-AIV SplitFuse cold-fill service."""

        if (isinstance(active_aic_groups, bool)
                or not isinstance(active_aic_groups, int)):
            raise TypeError(
                "FIA SplitFuse cold-fill active groups must be an integer")
        mechanism = self.fia_aiv_instruction_cold_fill
        if mechanism is None:
            raise ValueError(
                "hardware has no FIA AIV instruction cold-fill mechanism")
        split_fuse = mechanism["split_fuse"]
        if active_aic_groups != split_fuse["active_aic_groups"]:
            raise ValueError(
                "FIA SplitFuse cold fill requires its exact measured "
                "active-group topology")
        return split_fuse["duration_ns_per_aiv"]

    def fia_aiv_sync_all_time_ns(self, active_aiv_workers):
        """Return one source-mandatory FIA AIV-wide barrier span."""

        if (isinstance(active_aiv_workers, bool)
                or not isinstance(active_aiv_workers, int)):
            raise TypeError("FIA AIV SyncAll worker count must be an integer")
        mechanism = self.fia_aiv_sync_all_service
        if mechanism is None:
            raise ValueError("hardware has no FIA AIV SyncAll service")
        if (active_aiv_workers % mechanism["aiv_per_group"] != 0
                or not (
                    mechanism["minimum_active_aiv_workers"]
                    <= active_aiv_workers
                    <= mechanism["maximum_active_aiv_workers"])):
            raise ValueError(
                "FIA AIV SyncAll worker count is outside the measured "
                "mixed-task domain")
        return (
            mechanism["startup_ns"]
            + active_aiv_workers
            * mechanism["per_active_aiv_worker_ns"]
        )

    def fia_tiling_data_materialization_times_ns(
            self, active_aic_groups, payload_bytes, aiv_transfer_bytes):
        """Return concurrent AIC and AIV tiling-prelude completion times."""

        for value, name in (
                (active_aic_groups, "active AIC group count"),
                (payload_bytes, "payload byte count"),
                (aiv_transfer_bytes, "AIV transfer byte count")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"FIA tiling {name} must be an integer")
        mechanism = self.fia_tiling_data_materialization_service
        if mechanism is None:
            raise ValueError(
                "hardware has no FIA tiling materialization service")
        if not (
                mechanism["minimum_active_aic_groups"]
                <= active_aic_groups
                <= mechanism["maximum_active_aic_groups"]):
            raise ValueError(
                "FIA tiling active AIC group count is outside the measured "
                "domain")
        if payload_bytes != mechanism["payload_bytes"]:
            raise ValueError("FIA tiling payload byte count changed")
        if aiv_transfer_bytes != mechanism["aiv_transfer_bytes"]:
            raise ValueError("FIA tiling AIV transfer byte count changed")
        active_aiv_workers = (
            active_aic_groups * mechanism["aiv_per_group"])
        return (
            mechanism["aic_completion_ns"],
            mechanism["aiv_startup_ns"]
            + active_aiv_workers
            * mechanism["aiv_per_active_worker_ns"],
        )

    def fia_flash_decode_terminal_prelude_time_ns(self, active_aic_groups):
        """Return the complete source-defined terminal prelude span."""

        if (isinstance(active_aic_groups, bool)
                or not isinstance(active_aic_groups, int)):
            raise TypeError(
                "FIA FlashDecode active-group count must be an integer")
        mechanism = self.fia_flash_decode_terminal_prelude_service
        if mechanism is None:
            raise ValueError(
                "hardware has no FIA FlashDecode terminal-prelude service")
        if not (
                mechanism["minimum_active_aic_groups"]
                <= active_aic_groups
                <= mechanism["maximum_active_aic_groups"]):
            raise ValueError(
                "FIA FlashDecode active-group count is outside the measured "
                "domain")
        return (
            mechanism["startup_ns"]
            + active_aic_groups
            * mechanism["per_active_aic_group_ns"]
        )

    def fia_mixed_task_outer_envelope_time_ns(self, active_aiv_workers):
        """Return the device-side envelope outside one FIA mixed task."""

        if (isinstance(active_aiv_workers, bool)
                or not isinstance(active_aiv_workers, int)):
            raise TypeError(
                "FIA mixed-task outer-envelope worker count must be an "
                "integer")
        mechanism = self.fia_mixed_task_outer_envelope
        if mechanism is None:
            raise ValueError(
                "hardware has no FIA mixed-task outer-envelope service")
        if (active_aiv_workers % mechanism["aiv_per_group"] != 0
                or not (
                    mechanism["minimum_active_aiv_workers"]
                    <= active_aiv_workers
                    <= mechanism["maximum_active_aiv_workers"])):
            raise ValueError(
                "FIA mixed-task outer-envelope worker count is outside the "
                "measured domain")
        return (
            mechanism["startup_ns"]
            + active_aiv_workers
            * mechanism["per_active_aiv_worker_ns"]
        )

    def fia_aiv_workspace_transfer_time_ns(
            self, completion_service, active_aic_groups):
        """Return one measured source-shaped AIV transfer completion."""

        if not isinstance(completion_service, str) or not completion_service:
            raise TypeError(
                "FIA AIV workspace completion service must be a non-empty "
                "string")
        if (isinstance(active_aic_groups, bool)
                or not isinstance(active_aic_groups, int)):
            raise TypeError(
                "FIA AIV workspace active-group count must be an integer")
        mechanism = self.fia_aiv_workspace_transfer_service
        if mechanism is None:
            raise ValueError(
                "hardware has no FIA AIV workspace-transfer service")
        completion = mechanism["completion_ns_by_mode"]
        if completion_service not in completion:
            raise ValueError(
                "unknown FIA AIV workspace completion service: "
                f"{completion_service!r}")
        if active_aic_groups not in mechanism["active_aic_groups"]:
            raise ValueError(
                "FIA AIV workspace active-group count is outside the "
                "measured domain")
        return completion[completion_service][str(active_aic_groups)]

    def aic_mte2_nd2nz_serial_time_ns(
            self, descriptor_bytes, active_aic_groups):
        """Return measured serial GM-to-L1 ND2NZ service per descriptor."""

        for name, value in (
                ("descriptor_bytes", descriptor_bytes),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"AIC MTE2 ND2NZ {name} must be an integer")
            if value <= 0:
                raise ValueError(f"AIC MTE2 ND2NZ {name} must be positive")
        mechanism = self.aic_mte2_nd2nz_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MTE2 ND2NZ serial service")
        byte_points = mechanism["descriptor_bytes"]
        group_points = mechanism["active_aic_groups"]
        durations_at_group_points = tuple(
            _linear_interpolate(
                byte_points,
                mechanism["duration_ns_by_active_aic_group"][group],
                descriptor_bytes,
            )
            for group in group_points)
        return _linear_interpolate(
            group_points, durations_at_group_points, active_aic_groups)

    def aic_mte2_nd2nz_stream_time_ns(
            self, descriptor_bytes, active_aic_groups):
        """Return measured back-to-back GM-to-L1 ND2NZ issue interval."""

        for name, value in (
                ("descriptor_bytes", descriptor_bytes),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"AIC MTE2 ND2NZ {name} must be an integer")
            if value <= 0:
                raise ValueError(f"AIC MTE2 ND2NZ {name} must be positive")
        mechanism = self.aic_mte2_nd2nz_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MTE2 ND2NZ stream service")
        byte_points = mechanism["stream_descriptor_bytes"]
        group_points = mechanism["active_aic_groups"]
        durations_at_group_points = tuple(
            _linear_interpolate(
                byte_points,
                mechanism["stream_duration_ns_by_active_aic_group"][group],
                descriptor_bytes,
            )
            for group in group_points)
        return _linear_interpolate(
            group_points, durations_at_group_points, active_aic_groups)

    def split_fuse_mte2_short_tail_time_ns(
            self, descriptor_count, active_aic_groups,
            dram_miss_fraction, *, two_slot_overlap):
        """Finite source ingress service for the measured paged-KV geometry."""

        if (type(descriptor_count) is not int
                or type(active_aic_groups) is not int
                or active_aic_groups != 24):
            raise ValueError("SplitFuse short-tail is outside its AIC domain")
        service = self.split_fuse_mte2_short_tail_service
        if service is None or descriptor_count not in service:
            raise ValueError("SplitFuse short-tail count is unmeasured")
        if (isinstance(dram_miss_fraction, bool)
                or not isinstance(dram_miss_fraction, (int, float))
                or not math.isfinite(float(dram_miss_fraction))
                or not 0.0 <= dram_miss_fraction <= 1.0):
            raise ValueError("SplitFuse DRAM miss fraction must be in [0, 1]")
        if not isinstance(two_slot_overlap, bool):
            raise TypeError("SplitFuse overlap selector must be boolean")
        mode = "two_slot_overlap" if two_slot_overlap else "drained"
        warm, cold = service[descriptor_count][mode]
        return warm + float(dram_miss_fraction) * (cold - warm)

    def aic_mte2_nd2nz_large_contiguous_supports(
            self, rows, columns, active_aic_groups):
        """Return whether a resident large descriptor was measured."""

        for name, value in (
                ("rows", rows), ("columns", columns),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MTE2 large-contiguous {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MTE2 large-contiguous {name} must be positive")
        mechanism = self.aic_mte2_nd2nz_large_contiguous_service
        if mechanism is None:
            return False
        group_points = mechanism["active_aic_groups"]
        return (
            group_points[0] <= active_aic_groups <= group_points[-1]
            and (rows, columns) in mechanism["geometries"])

    def _aic_mte2_nd2nz_large_contiguous_time_ns(
            self, rows, columns, active_aic_groups, service_name):
        if not self.aic_mte2_nd2nz_large_contiguous_supports(
                rows, columns, active_aic_groups):
            raise ValueError(
                "AIC MTE2 large-contiguous descriptor is outside the "
                "measured domain")
        mechanism = self.aic_mte2_nd2nz_large_contiguous_service
        return _linear_interpolate(
            mechanism["active_aic_groups"],
            mechanism["geometries"][(rows, columns)][service_name],
            active_aic_groups)

    def aic_mte2_nd2nz_large_contiguous_serial_time_ns(
            self, rows, columns, active_aic_groups):
        """Return resident completion for one large contiguous descriptor."""

        return self._aic_mte2_nd2nz_large_contiguous_time_ns(
            rows, columns, active_aic_groups,
            "serial_by_active_aic_group")

    def aic_mte2_nd2nz_large_contiguous_stream_time_ns(
            self, rows, columns, active_aic_groups):
        """Return its measured two-buffer steady issue interval."""

        return self._aic_mte2_nd2nz_large_contiguous_time_ns(
            rows, columns, active_aic_groups,
            "stream_by_active_aic_group")

    def aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            self, descriptor_bytes, active_aic_groups, *, stream,
            dram_miss_fraction=None):
        """Extend the large-contiguous service over its measured byte span.

        The independent large-descriptor sweep measures two geometries at
        both 128 and 256 KiB and shows a geometry-stable affine byte service
        at 22--24 active AICs. Extend that service between those endpoints and
        downward by less than one measured byte interval through the generic
        sweep's final 32--64-KiB affine interval. Do not splice in those
        differently shaped generic cells: their geometry belongs to another
        descriptor regime. This keeps a source program's byte service
        continuous when CANN coalesces two adjacent descriptors.

        The cold sweep measures the same two byte endpoints at 22 and 24
        active AICs. A one-interval extension of that axis covers the adjacent
        20--21-AIC launch boundary without switching the complete descriptor
        to the unrelated generic geometry.
        """

        for name, value in (
                ("descriptor_bytes", descriptor_bytes),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "AIC MTE2 large-contiguous extension "
                    f"{name} must be an integer")
            if value <= 0:
                raise ValueError(
                    "AIC MTE2 large-contiguous extension "
                    f"{name} must be positive")
        if not isinstance(stream, bool):
            raise TypeError(
                "AIC MTE2 large-contiguous extension stream must be bool")
        if (dram_miss_fraction is not None
                and (isinstance(dram_miss_fraction, bool)
                     or not isinstance(dram_miss_fraction, (int, float)))):
            raise TypeError(
                "AIC MTE2 large-contiguous extension miss fraction must be "
                "numeric or None")
        if (dram_miss_fraction is not None
                and (not math.isfinite(dram_miss_fraction)
                     or not 0.0 <= dram_miss_fraction <= 1.0)):
            raise ValueError(
                "AIC MTE2 large-contiguous extension miss fraction must be "
                "in [0, 1]")
        generic = self.aic_mte2_nd2nz_service
        large = self.aic_mte2_nd2nz_large_contiguous_service
        if generic is None or large is None:
            return None
        generic_points = (
            generic["stream_descriptor_bytes"]
            if stream else generic["descriptor_bytes"])
        generic_affine_lower_bytes = generic_points[-2]
        large_byte_counts = tuple(sorted({
            2 * rows * columns for rows, columns in large["geometries"]}))
        lower_bytes, upper_bytes = large_byte_counts[:2]
        if not generic_affine_lower_bytes <= descriptor_bytes <= upper_bytes:
            return None
        resident_name = (
            "stream_by_active_aic_group"
            if stream else "serial_by_active_aic_group")
        resident_endpoints_ns = {
            byte_count: tuple(
                _linear_interpolate(
                    large["active_aic_groups"],
                    services[resident_name], active_aic_groups)
                for (rows, columns), services in large["geometries"].items()
                if 2 * rows * columns == byte_count)
            for byte_count in (lower_bytes, upper_bytes)
        }
        if any(not values for values in resident_endpoints_ns.values()):
            raise AssertionError(
                "AIC MTE2 large descriptor service has an incomplete byte "
                "endpoint")
        lower_resident_ns, upper_resident_ns = (
            statistics.median(resident_endpoints_ns[byte_count])
            for byte_count in (lower_bytes, upper_bytes))
        byte_fraction = (
            (descriptor_bytes - lower_bytes) / (upper_bytes - lower_bytes))
        resident_ns = lower_resident_ns + byte_fraction * (
            upper_resident_ns - lower_resident_ns)
        if dram_miss_fraction is None or dram_miss_fraction == 0.0:
            completion_ns = resident_ns
        else:
            cold_groups = large["cold_active_aic_groups"]
            cold_interval = cold_groups[1] - cold_groups[0]
            if not (
                    cold_groups[0] - cold_interval <= active_aic_groups
                    <= cold_groups[-1]):
                return None
            cold_name = (
                "cold_stream_by_active_aic_group"
                if stream else "cold_serial_by_active_aic_group")
            cold_endpoints_ns = {}
            for byte_count in (lower_bytes, upper_bytes):
                values = []
                for (rows, columns), services in large["geometries"].items():
                    if 2 * rows * columns != byte_count:
                        continue
                    group_fraction = (
                        (active_aic_groups - cold_groups[0])
                        / (cold_groups[1] - cold_groups[0]))
                    values.append(
                        services[cold_name][0] + group_fraction * (
                            services[cold_name][1]
                            - services[cold_name][0]))
                if not values:
                    raise AssertionError(
                        "AIC MTE2 large cold descriptor service has an "
                        "incomplete byte endpoint")
                cold_endpoints_ns[byte_count] = statistics.median(values)
            cold_ns = cold_endpoints_ns[lower_bytes] + byte_fraction * (
                cold_endpoints_ns[upper_bytes]
                - cold_endpoints_ns[lower_bytes])
            if cold_ns <= resident_ns:
                raise ValueError(
                    "AIC MTE2 large-contiguous cold extension must exceed "
                    "resident service")
            completion_ns = math.fsum((
                resident_ns,
                float(dram_miss_fraction) * (cold_ns - resident_ns),
            ))
        if completion_ns <= 0:
            raise ValueError(
                "AIC MTE2 large-contiguous extension must stay positive")
        return completion_ns

    def aic_mte2_nd2nz_large_contiguous_cold_supports(
            self, rows, columns, active_aic_groups):
        """Return whether resident and DRAM-cold services were measured."""

        if not self.aic_mte2_nd2nz_large_contiguous_supports(
                rows, columns, active_aic_groups):
            return False
        group_points = self.aic_mte2_nd2nz_large_contiguous_service[
            "cold_active_aic_groups"]
        return group_points[0] <= active_aic_groups <= group_points[-1]

    def aic_mte2_nd2nz_large_contiguous_completion_time_ns(
            self, rows, columns, active_aic_groups, dram_miss_fraction, *,
            stream):
        """Blend exact resident and cold service by LRU miss fraction."""

        if not self.aic_mte2_nd2nz_large_contiguous_cold_supports(
                rows, columns, active_aic_groups):
            raise ValueError(
                "AIC MTE2 large-contiguous cold descriptor is outside the "
                "measured domain")
        if not isinstance(stream, bool):
            raise TypeError(
                "AIC MTE2 large-contiguous stream selector must be boolean")
        if (isinstance(dram_miss_fraction, bool)
                or not isinstance(dram_miss_fraction, (int, float))):
            raise TypeError(
                "AIC MTE2 large-contiguous miss fraction must be numeric")
        if (not math.isfinite(dram_miss_fraction)
                or not 0.0 <= dram_miss_fraction <= 1.0):
            raise ValueError(
                "AIC MTE2 large-contiguous miss fraction must be in [0, 1]")
        mechanism = self.aic_mte2_nd2nz_large_contiguous_service
        services = mechanism["geometries"][(rows, columns)]
        resident_name = (
            "stream_by_active_aic_group"
            if stream else "serial_by_active_aic_group")
        cold_name = (
            "cold_stream_by_active_aic_group"
            if stream else "cold_serial_by_active_aic_group")
        resident_ns = _linear_interpolate(
            mechanism["active_aic_groups"], services[resident_name],
            active_aic_groups)
        cold_ns = _linear_interpolate(
            mechanism["cold_active_aic_groups"], services[cold_name],
            active_aic_groups)
        if cold_ns <= resident_ns:
            raise ValueError(
                "AIC MTE2 large-contiguous cold service must exceed resident")
        return math.fsum((
            resident_ns,
            float(dram_miss_fraction) * (cold_ns - resident_ns),
        ))

    def aic_mte2_gemm_large_resident_supports(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups):
        """Return whether this resident GEMM descriptor was measured."""

        for name, value in (
                ("rows", rows), ("columns", columns),
                ("source_row_stride_elements", source_row_stride_elements),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MTE2 GEMM large-resident {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MTE2 GEMM large-resident {name} must be positive")
        mechanism = self.aic_mte2_gemm_large_resident_service
        if mechanism is None:
            return False
        return (rows, columns, source_row_stride_elements, active_aic_groups) in (
            mechanism["geometries"])

    def _aic_mte2_gemm_large_resident_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, service_name):
        if not self.aic_mte2_gemm_large_resident_supports(
                rows, columns, source_row_stride_elements,
                active_aic_groups):
            raise ValueError(
                "AIC MTE2 GEMM large-resident descriptor is outside the "
                "measured domain")
        mechanism = self.aic_mte2_gemm_large_resident_service
        services = mechanism["geometries"][
            (rows, columns, source_row_stride_elements, active_aic_groups)]
        return services[service_name]

    def aic_mte2_gemm_large_resident_serial_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups):
        """Return one completed resident GEMM descriptor service."""

        return self._aic_mte2_gemm_large_resident_time_ns(
            rows, columns, source_row_stride_elements, active_aic_groups,
            "serial_ns")

    def aic_mte2_gemm_large_resident_stream_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups):
        """Return the two-buffer resident GEMM issue interval."""

        return self._aic_mte2_gemm_large_resident_time_ns(
            rows, columns, source_row_stride_elements, active_aic_groups,
            "stream_ns")

    def aic_mte2_gemm_large_resident_program_supports(
            self, *, active_aic_groups, a_rows, a_columns,
            a_source_row_stride_elements, a_buffer_depth,
            a_descriptors_per_worker, b_rows, b_columns,
            b_source_row_stride_elements, b_buffer_depth,
            b_descriptors_per_worker):
        """Return whether the finite credit topology has a holdout."""

        signature = {
            "active_aic_groups": active_aic_groups,
            "a_rows": a_rows,
            "a_columns": a_columns,
            "a_source_row_stride_elements": a_source_row_stride_elements,
            "a_buffer_depth": a_buffer_depth,
            "a_descriptors_per_worker": a_descriptors_per_worker,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "b_source_row_stride_elements": b_source_row_stride_elements,
            "b_buffer_depth": b_buffer_depth,
            "b_descriptors_per_worker": b_descriptors_per_worker,
        }
        for name, value in signature.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MTE2 GEMM resident program {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MTE2 GEMM resident program {name} must be positive")
        mechanism = self.aic_mte2_gemm_large_resident_service
        if mechanism is None:
            return False
        holdout = mechanism["full_program_holdout"]
        topology_matches = (
            signature["a_buffer_depth"] == holdout["a_buffer_depth"]
            and signature["b_buffer_depth"] == holdout["b_buffer_depth"]
            and signature["a_descriptors_per_worker"]
                == holdout["a_descriptors_per_worker"]
            and signature["b_descriptors_per_worker"] >= 2
            and signature["b_descriptors_per_worker"] % 2 == 0)
        measured_operand = (
            self.aic_mte2_gemm_large_resident_supports(
                signature["a_rows"], signature["a_columns"],
                signature["a_source_row_stride_elements"],
                signature["active_aic_groups"])
            or self.aic_mte2_gemm_large_resident_supports(
                signature["b_rows"], signature["b_columns"],
                signature["b_source_row_stride_elements"],
                signature["active_aic_groups"]))
        return topology_matches and measured_operand

    def aic_mte2_gemm_shared_source_queue_supports(
            self, *, dtype, active_aic_groups, operand_buffer_depth,
            traversal_order, a_rows, a_columns,
            a_source_row_stride_elements, b_rows, b_columns,
            b_source_row_stride_elements):
        """Return whether both descriptors belong to one measured queue."""

        mechanism = self.aic_mte2_gemm_shared_source_queue_service
        if mechanism is None:
            return False
        signature = {
            "dtype": dtype,
            "active_aic_groups": active_aic_groups,
            "operand_buffer_depth": operand_buffer_depth,
            "traversal_order": traversal_order,
        }
        for name, value in signature.items():
            if value != mechanism[name]:
                return False
        geometries = mechanism["geometries"]
        return (
            (a_rows, a_columns, a_source_row_stride_elements) in geometries
            and (b_rows, b_columns, b_source_row_stride_elements)
            in geometries)

    def aic_mte2_gemm_shared_source_descriptor_time_ns(
            self, rows, columns, source_row_stride_elements,
            dram_miss_fraction, *, stream):
        """Blend one queue descriptor's resident and DRAM-cold service."""

        if not isinstance(stream, bool):
            raise TypeError("AIC GEMM shared-queue stream selector is invalid")
        if (isinstance(dram_miss_fraction, bool)
                or not isinstance(dram_miss_fraction, (int, float))
                or not math.isfinite(float(dram_miss_fraction))
                or not 0.0 <= dram_miss_fraction <= 1.0):
            raise ValueError(
                "AIC GEMM shared-queue miss fraction must be in [0, 1]")
        mechanism = self.aic_mte2_gemm_shared_source_queue_service
        if mechanism is None:
            raise ValueError("hardware has no shared MTE2 source queue")
        geometry = (rows, columns, source_row_stride_elements)
        if geometry not in mechanism["geometries"]:
            raise ValueError(
                "AIC GEMM shared-queue descriptor is outside its domain")
        services = mechanism["geometries"][geometry]
        mode = "stream" if stream else "serial"
        resident_ns = services[f"resident_{mode}_ns"]
        cold_ns = services[f"cold_{mode}_ns"]
        if cold_ns <= resident_ns:
            raise ValueError(
                "AIC GEMM shared-queue cold service must exceed resident")
        return math.fsum((
            resident_ns,
            float(dram_miss_fraction) * (cold_ns - resident_ns),
        ))

    def aic_mte2_nd2nz_cold_completion_supports(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups):
        """Return whether resident and cold services own this descriptor."""

        for name, value in (
                ("rows", rows), ("columns", columns),
                ("source_row_stride_elements", source_row_stride_elements),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MTE2 cold completion {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MTE2 cold completion {name} must be positive")
        mechanism = self.aic_mte2_nd2nz_service
        if mechanism is None:
            return False
        descriptor_bytes = rows * columns * 2
        if not (
                mechanism["active_aic_groups"][0]
                <= active_aic_groups
                <= mechanism["active_aic_groups"][-1]
                and mechanism["descriptor_bytes"][0]
                <= descriptor_bytes
                <= mechanism["descriptor_bytes"][-1]):
            return False
        if source_row_stride_elements == columns or rows <= 16:
            return True
        return (
            mechanism["strided_active_aic_groups"][0]
            <= active_aic_groups
            <= mechanism["strided_active_aic_groups"][-1]
            and (rows, columns, source_row_stride_elements)
            in mechanism["strided_geometries"])

    def aic_mte2_nd2nz_cold_completion_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, dram_miss_fraction):
        """Blend resident completion with predicted DRAM-miss service."""

        if not self.aic_mte2_nd2nz_cold_completion_supports(
                rows, columns, source_row_stride_elements,
                active_aic_groups):
            raise ValueError(
                "AIC MTE2 cold completion is outside the measured domain")
        if (isinstance(dram_miss_fraction, bool)
                or not isinstance(dram_miss_fraction, (int, float))):
            raise TypeError(
                "AIC MTE2 cold completion miss fraction must be numeric")
        if (not math.isfinite(dram_miss_fraction)
                or not 0.0 <= dram_miss_fraction <= 1.0):
            raise ValueError(
                "AIC MTE2 cold completion miss fraction must be in [0, 1]")
        mechanism = self.aic_mte2_nd2nz_service
        geometry_key = (rows, columns, source_row_stride_elements)
        descriptor_bytes = rows * columns * 2
        if (active_aic_groups
                == mechanism["cold_completion_active_aic_groups"]
                and geometry_key
                in mechanism["cold_completion_geometries"]):
            resident_ns = mechanism["cold_completion_geometries"][
                geometry_key]["resident_duration_ns"]
        else:
            resident_ns = self.aic_mte2_nd2nz_serial_time_ns(
                descriptor_bytes, active_aic_groups)
            if source_row_stride_elements != columns and rows > 16:
                resident_ns = max(
                    resident_ns,
                    self.aic_mte2_nd2nz_strided_serial_time_ns(
                        rows, columns, source_row_stride_elements,
                        active_aic_groups))
        calibration_groups = mechanism[
            "cold_completion_active_aic_groups"]
        cold_extra_ns = max(
            mechanism["cold_completion_minimum_extra_ns"],
            descriptor_bytes * active_aic_groups
            / (mechanism["cold_completion_bandwidth_bytes_per_s"]
               * calibration_groups) * 1.0e9,
        )
        return math.fsum((
            resident_ns,
            float(dram_miss_fraction) * cold_extra_ns,
        ))

    def aic_mte2_nd2nz_cold_exact_supports(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups):
        """Return whether an exact decode-launch cold service was measured."""

        values = (
            rows, columns, source_row_stride_elements, active_aic_groups)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise TypeError(
                "AIC MTE2 cold exact inputs must be integers")
        if any(value <= 0 for value in values):
            raise ValueError(
                "AIC MTE2 cold exact inputs must be positive")
        mechanism = self.aic_mte2_nd2nz_service
        if mechanism is None:
            return False
        return (
            rows, columns, source_row_stride_elements, active_aic_groups
        ) in mechanism["cold_exact_geometries"]

    def aic_mte2_nd2nz_cold_exact_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, *, stream):
        """Return the measured cold service of one exact B1 descriptor."""

        if not self.aic_mte2_nd2nz_cold_exact_supports(
                rows, columns, source_row_stride_elements,
                active_aic_groups):
            raise ValueError(
                "AIC MTE2 cold exact service is outside its measured domain")
        if not isinstance(stream, bool):
            raise TypeError(
                "AIC MTE2 cold exact stream selector must be boolean")
        durations = self.aic_mte2_nd2nz_service["cold_exact_geometries"][
            (rows, columns, source_row_stride_elements, active_aic_groups)]
        return (durations["stream_duration_ns"]
                if stream else durations["serial_duration_ns"])

    def aic_mte2_fia_mla_strided_supports(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups):
        """Return whether an exact CANN MLA descriptor service was measured."""

        values = (
            rows, columns, source_row_stride_elements, active_aic_groups)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise TypeError(
                "AIC MTE2 FIA MLA strided inputs must be integers")
        if any(value <= 0 for value in values):
            raise ValueError(
                "AIC MTE2 FIA MLA strided inputs must be positive")
        mechanism = self.aic_mte2_fia_mla_strided_service
        if mechanism is None:
            return False
        return (
            active_aic_groups in mechanism["active_aic_groups"]
            and (rows, columns, source_row_stride_elements)
            in mechanism["geometries"])

    def aic_mte2_fia_mla_strided_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, dram_miss_fraction, *, stream):
        """Blend exact resident/cold MLA service by predicted miss fraction."""

        if not self.aic_mte2_fia_mla_strided_supports(
                rows, columns, source_row_stride_elements,
                active_aic_groups):
            raise ValueError(
                "AIC MTE2 FIA MLA strided service is outside its exact domain")
        if not isinstance(stream, bool):
            raise TypeError("AIC MTE2 FIA MLA stream selector must be boolean")
        if (isinstance(dram_miss_fraction, bool)
                or not isinstance(dram_miss_fraction, (int, float))):
            raise TypeError(
                "AIC MTE2 FIA MLA miss fraction must be numeric")
        if (not math.isfinite(dram_miss_fraction)
                or not 0.0 <= dram_miss_fraction <= 1.0):
            raise ValueError(
                "AIC MTE2 FIA MLA miss fraction must be in [0, 1]")
        mechanism = self.aic_mte2_fia_mla_strided_service
        group_index = mechanism["active_aic_groups"].index(
            active_aic_groups)
        services = mechanism["geometries"][(
            rows, columns, source_row_stride_elements)]
        mode = "stream" if stream else "serial"
        resident_ns = services[
            f"resident_{mode}_by_active_aic_group"][group_index]
        cold_ns = services[
            f"cold_{mode}_by_active_aic_group"][group_index]
        return math.fsum((
            resident_ns,
            float(dram_miss_fraction) * (cold_ns - resident_ns),
        ))

    def aic_mte2_fia_gqa_finite_drain_supports(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count):
        """Return whether an exact cold FIA GQA program was measured."""

        values = (
            rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise TypeError(
                "AIC MTE2 FIA GQA finite-drain inputs must be integers")
        if any(value <= 0 for value in values):
            raise ValueError(
                "AIC MTE2 FIA GQA finite-drain inputs must be positive")
        mechanism = self.aic_mte2_fia_gqa_finite_drain_service
        if mechanism is None:
            return False
        if values in mechanism["programs"]:
            return True
        # Outside the exact-count grid the measured count-independence law
        # licenses the same per-descriptor service, but only for a licensed
        # geometry/concurrency cell and only inside its validated range.
        cell = (
            rows, columns, source_row_stride_elements, active_aic_groups)
        return (
            cell in mechanism["count_free_service"]
            and mechanism["count_law_minimum_descriptor_count"]
            <= descriptor_count
            <= mechanism["count_law_maximum_descriptor_count"])

    def aic_mte2_fia_gqa_finite_drain_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count):
        """Return an exact cold FIA GQA finite-program completion."""

        if not self.aic_mte2_fia_gqa_finite_drain_supports(
                rows, columns, source_row_stride_elements,
                active_aic_groups, descriptor_count):
            raise ValueError(
                "AIC MTE2 FIA GQA finite drain is outside the measured domain")
        mechanism = self.aic_mte2_fia_gqa_finite_drain_service
        key = (
            rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count)
        if key in mechanism["programs"]:
            return mechanism["programs"][key]
        cell = (
            rows, columns, source_row_stride_elements, active_aic_groups)
        return mechanism["count_free_service"][cell] * descriptor_count

    def aic_mte2_fia_mha_finite_drain_supports(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count):
        """Return whether an exact cold FIA MHA finite program was measured."""

        values = (
            rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise TypeError(
                "AIC MTE2 FIA MHA finite-drain inputs must be integers")
        if any(value <= 0 for value in values):
            raise ValueError(
                "AIC MTE2 FIA MHA finite-drain inputs must be positive")
        mechanism = self.aic_mte2_fia_mha_finite_drain_service
        if mechanism is None:
            return False
        key = (
            rows, columns, source_row_stride_elements, active_aic_groups)
        return (
            active_aic_groups in mechanism["active_aic_groups"]
            and key in mechanism["geometries"]
            and descriptor_count in mechanism["geometries"][key])

    def aic_mte2_fia_mha_finite_drain_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count):
        """Return an exact cold FIA MHA finite-program completion time."""

        if not self.aic_mte2_fia_mha_finite_drain_supports(
                rows, columns, source_row_stride_elements,
                active_aic_groups, descriptor_count):
            raise ValueError(
                "AIC MTE2 FIA MHA finite drain is outside the measured domain")
        mechanism = self.aic_mte2_fia_mha_finite_drain_service
        return mechanism["geometries"][(
            rows, columns, source_row_stride_elements,
            active_aic_groups)][descriptor_count]

    def aic_mte2_nd2nz_cold_stream_time_ns(
            self, aggregate_dram_bytes, active_aic_groups):
        """Return the shared-DRAM constraint for a continuous MTE2 stream.

        The independent, non-parameterizing stream validation is recorded in
        ``microbench/ascend_910b/references/`` under
        ``aic_mte2_nd2nz_cold_stream_parallel_constraint_cann_8_5_1.json``.
        """

        if (isinstance(aggregate_dram_bytes, bool)
                or not isinstance(aggregate_dram_bytes, (int, float))):
            raise TypeError("AIC MTE2 cold stream bytes must be numeric")
        if (not math.isfinite(aggregate_dram_bytes)
                or aggregate_dram_bytes <= 0):
            raise ValueError(
                "AIC MTE2 cold stream bytes must be finite and positive")
        if (isinstance(active_aic_groups, bool)
                or not isinstance(active_aic_groups, int)):
            raise TypeError(
                "AIC MTE2 cold stream active groups must be an integer")
        mechanism = self.aic_mte2_nd2nz_service
        if mechanism is None:
            raise ValueError("hardware has no AIC MTE2 cold stream service")
        group_points = mechanism["active_aic_groups"]
        if not group_points[0] <= active_aic_groups <= group_points[-1]:
            raise ValueError(
                "AIC MTE2 cold stream active groups are outside the "
                "measured range")
        calibration_groups = mechanism[
            "cold_completion_active_aic_groups"]
        return max(
            mechanism["cold_completion_minimum_extra_ns"],
            float(aggregate_dram_bytes)
            / (mechanism["cold_completion_bandwidth_bytes_per_s"]
               * calibration_groups) * 1.0e9,
        )

    def aic_mte2_operand_pair_stream_time_ns(
            self, a_rows, a_columns, b_rows, b_columns,
            active_aic_groups, pair_count):
        """Return the exact steady A/B ND2NZ stream time for one AIC.

        This service is deliberately an exact geometry and stream-length
        lookup.  It must not be interpolated into unrelated descriptors or
        short tile programs.
        """

        for name, value in (
                ("a_rows", a_rows), ("a_columns", a_columns),
                ("b_rows", b_rows), ("b_columns", b_columns),
                ("active_aic_groups", active_aic_groups),
                ("pair_count", pair_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MTE2 operand-pair stream {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MTE2 operand-pair stream {name} must be positive")
        mechanism = self.aic_mte2_operand_pair_stream_service
        if mechanism is None:
            raise ValueError("hardware has no AIC MTE2 operand-pair stream service")
        observed_geometry = (a_rows, a_columns, b_rows, b_columns)
        expected_geometry = tuple(
            mechanism[name]
            for name in ("a_rows", "a_columns", "b_rows", "b_columns"))
        if observed_geometry != expected_geometry:
            raise ValueError(
                "AIC MTE2 operand-pair stream geometry is outside the "
                "measured domain")
        if active_aic_groups != mechanism["active_aic_groups"]:
            raise ValueError(
                "AIC MTE2 operand-pair stream active-group count is outside "
                "the measured domain")
        if not (mechanism["minimum_pairs_per_worker"] <= pair_count
                <= mechanism["maximum_pairs_per_worker"]):
            raise ValueError(
                "AIC MTE2 operand-pair stream length is outside the measured "
                "domain")
        return pair_count * mechanism["continuous_ns_per_pair"]

    def aic_mte2_finite_stream_completion_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, descriptor_count):
        """Return one cold finite B1 descriptor program's completion time."""

        values = {
            "rows": rows,
            "columns": columns,
            "source_row_stride_elements": source_row_stride_elements,
            "active_aic_groups": active_aic_groups,
            "descriptor_count": descriptor_count,
        }
        for name, item in values.items():
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"AIC MTE2 finite-stream {name} must be an integer")
            if item <= 0:
                raise ValueError(
                    f"AIC MTE2 finite-stream {name} must be positive")
        mechanism = self.aic_mte2_finite_stream_completion_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MTE2 finite-stream completion service")
        if rows != mechanism["rows"] or columns != mechanism["columns"]:
            raise ValueError(
                "AIC MTE2 finite-stream descriptor is outside the measured "
                "domain")
        if not (mechanism["minimum_descriptors_per_worker"]
                <= descriptor_count
                <= mechanism["maximum_descriptors_per_worker"]):
            raise ValueError(
                "AIC MTE2 finite-stream length is outside the measured domain")
        classes = mechanism["worker_classes"]
        if source_row_stride_elements not in classes:
            raise ValueError(
                "AIC MTE2 finite-stream stride is outside the measured domain")
        worker_class = classes[source_row_stride_elements]
        group_points = worker_class["active_aic_groups"]
        if not group_points[0] <= active_aic_groups <= group_points[-1]:
            raise ValueError(
                "AIC MTE2 finite-stream active-AIC count is outside the "
                "measured domain")
        if len(group_points) == 1:
            intercept_ns = worker_class[
                "duration_at_minimum_descriptors_ns"][0]
            slope_ns = worker_class["ns_per_additional_descriptor"][0]
        else:
            intercept_ns = _linear_interpolate(
                group_points,
                worker_class["duration_at_minimum_descriptors_ns"],
                active_aic_groups)
            slope_ns = _linear_interpolate(
                group_points,
                worker_class["ns_per_additional_descriptor"],
                active_aic_groups)
        return (
            intercept_ns
            + (descriptor_count
               - mechanism["minimum_descriptors_per_worker"]) * slope_ns)

    def aic_mte2_gemm_source_order_completion_time_ns(
            self, dtype, active_aic_groups, operand_buffer_depth,
            base_tile_m, base_tile_n, base_tile_k, mte1_base_k,
            a_rows, a_columns, a_source_row_stride_elements,
            b_rows, b_columns, b_source_row_stride_elements,
            m_groups, n_groups, output_tiles_per_worker, k_slabs,
            mte2_programs_per_worker, mte1_pairs_per_worker):
        """Return an exact measured MatMulV2 ingress-program completion."""

        mechanism = self.aic_mte2_gemm_source_order_completion_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GEMM source-order completion service")
        values = {
            "dtype": dtype,
            "active_aic_groups": active_aic_groups,
            "operand_buffer_depth": operand_buffer_depth,
            "base_tile_m": base_tile_m,
            "base_tile_n": base_tile_n,
            "base_tile_k": base_tile_k,
            "mte1_base_k": mte1_base_k,
            "a_rows": a_rows,
            "a_columns": a_columns,
            "a_source_row_stride_elements": a_source_row_stride_elements,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "b_source_row_stride_elements": b_source_row_stride_elements,
            "m_groups": m_groups,
            "n_groups": n_groups,
            "output_tiles_per_worker": output_tiles_per_worker,
            "k_slabs": k_slabs,
            "mte2_programs_per_worker": mte2_programs_per_worker,
            "mte1_pairs_per_worker": mte1_pairs_per_worker,
        }
        for name, value in values.items():
            if name == "dtype":
                if not isinstance(value, str) or not value:
                    raise TypeError(
                        "AIC GEMM source-order dtype must be a string")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GEMM source-order {name} must be an integer")
            elif value <= 0:
                raise ValueError(
                    f"AIC GEMM source-order {name} must be positive")
        signature = tuple(
            values[name]
            for name in _AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS)
        programs = mechanism["programs"]
        if signature not in programs:
            raise ValueError(
                "AIC GEMM source-order program is outside the measured domain")
        return programs[signature]["completion_ns"]

    def aic_mte2_gemm_source_order_service(self, **signature):
        """Return an exact ingress service for one physical source program."""

        mechanism = self.aic_mte2_gemm_source_order_completion_service
        if mechanism is None:
            return None
        if set(signature) != set(
                _AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS):
            raise ValueError(
                "AIC GEMM source-order service signature fields changed")
        for name, value in signature.items():
            if name == "dtype":
                if not isinstance(value, str) or not value:
                    raise TypeError(
                        "AIC GEMM source-order dtype must be a string")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GEMM source-order {name} must be an integer")
            elif value <= 0:
                raise ValueError(
                    f"AIC GEMM source-order {name} must be positive")
        key = tuple(
            signature[name]
            for name in _AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS)
        programs = mechanism["programs"]
        if key not in programs:
            return None
        return programs[key]

    def aic_mte2_gemm_capacity_constrained_source_order_time_ns(
            self, a_rows, a_columns, a_descriptor_count,
            b_rows, b_columns, b_descriptor_count, active_aic_groups):
        """Return shared issue time for one capacity-constrained A/B stream.

        Independent long source-program probes show one aggregate MTE2 issue
        capacity for the alternating A1/B1 sequence across descriptor
        geometries. The caller owns the L2-capacity and source-traversal
        domain check; descriptor startup and issue remain competing bounds.
        """

        values = {
            "a_rows": a_rows,
            "a_columns": a_columns,
            "a_descriptor_count": a_descriptor_count,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "b_descriptor_count": b_descriptor_count,
            "active_aic_groups": active_aic_groups,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GEMM large source-order {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC GEMM large source-order {name} must be positive")
        mechanism = self.aic_mte2_gemm_source_order_completion_service
        if mechanism is None:
            return None
        measured_groups = mechanism[
            "capacity_constrained_active_aic_groups"]
        if not measured_groups - 1 <= active_aic_groups <= measured_groups:
            return None
        # The aggregate capacity is a shared MTE2 issue limit, not a service
        # for one particular descriptor shape.  The two source-order probes
        # establish a geometry-stable byte rate. Descriptor-specific startup
        # and issue constraints remain separate competing bounds in the
        # caller. Restricting this shared byte limit to either probe geometry
        # creates discontinuities whenever CANN repacks the same long stream.
        source_bytes = 2 * (
            a_descriptor_count * a_rows * a_columns
            + b_descriptor_count * b_rows * b_columns)
        return (
            source_bytes
            / mechanism["capacity_constrained_bytes_per_s"]
            * 1.0e9)

    def aic_mte2_gemm_capacity_constrained_source_completion_time_ns(
            self, a_rows, a_columns, a_descriptor_count,
            b_rows, b_columns, b_descriptor_count, active_aic_groups):
        """Return MTE2-to-MTE1 completion for one ordered A/B stream.

        Two independent source programs with different K and descriptor
        counts establish the same per-AIC completion byte capacity.  The
        service is local after the shared memory path has supplied data and
        therefore applies to any measured non-empty AIC concurrency.
        """

        values = {
            "a_rows": a_rows,
            "a_columns": a_columns,
            "a_descriptor_count": a_descriptor_count,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "b_descriptor_count": b_descriptor_count,
            "active_aic_groups": active_aic_groups,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GEMM source completion {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC GEMM source completion {name} must be positive")
        mechanism = self.aic_mte2_gemm_source_order_completion_service
        if mechanism is None:
            return None
        if (not self.aic_mte2_nd2nz_large_contiguous_supports(
                    a_rows, a_columns, active_aic_groups)
                or not self.aic_mte2_nd2nz_large_contiguous_supports(
                    b_rows, b_columns, active_aic_groups)):
            return None
        source_bytes = 2 * (
            a_descriptor_count * a_rows * a_columns
            + b_descriptor_count * b_rows * b_columns)
        return (
            source_bytes
            / mechanism[
                "capacity_constrained_completion_bytes_per_s"]
            * 1.0e9)

    def aic_gemm_finite_ingress_completion_time_ns(
            self, *, completion_pairs_per_worker):
        """Return the common finite source-pair completion bound."""

        mechanism = self.aic_gemm_finite_ingress_completion_service
        if mechanism is None:
            return None
        if (isinstance(completion_pairs_per_worker, bool)
                or not isinstance(completion_pairs_per_worker, int)
                or completion_pairs_per_worker <= 0):
            raise ValueError(
                "AIC finite ingress completion pairs must be positive")
        return (
            mechanism["startup_ns"]
            + completion_pairs_per_worker
            * mechanism["ns_per_completion_pair"])

    def aic_gemm_finite_transition_service(self, **signature):
        """Return exact MTE2 service for one normalized finite program."""

        mechanism = self.aic_gemm_finite_ingress_completion_service
        if mechanism is None:
            return None
        if set(signature) != set(_AIC_GEMM_FINITE_TRANSITION_SIGNATURE_FIELDS):
            raise ValueError(
                "AIC GEMM finite transition signature fields changed")
        for name, value in signature.items():
            if name == "dtype":
                if not isinstance(value, str) or not value:
                    raise TypeError(
                        "AIC GEMM finite transition dtype must be a string")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GEMM finite transition {name} must be an integer")
            elif value <= 0:
                raise ValueError(
                    f"AIC GEMM finite transition {name} must be positive")
        key = tuple(
            signature[name]
            for name in _AIC_GEMM_FINITE_TRANSITION_SIGNATURE_FIELDS)
        services = mechanism["exact_transition_services"]
        if key not in services:
            return None
        return services[key]

    def aic_gemm_short_pipeline_completion_time_ns(
            self, *, output_tiles_per_worker):
        """Return the finite source pipeline completion for one AIC."""

        if (isinstance(output_tiles_per_worker, bool)
                or not isinstance(output_tiles_per_worker, int)
                or output_tiles_per_worker <= 0):
            raise ValueError(
                "AIC short GEMM output tiles per worker must be positive")
        mechanism = self.aic_gemm_short_pipeline_completion_service
        if mechanism is None:
            return None
        if not (mechanism["minimum_output_tiles_per_worker"]
                <= output_tiles_per_worker
                <= mechanism["maximum_output_tiles_per_worker"]):
            return None
        return (
            mechanism["startup_ns"]
            + output_tiles_per_worker * mechanism["ns_per_output_tile"])

    def aic_mte2_gemm_persistent_issue_time_ns(self, **signature):
        """Return an exact source-program MTE2 issue completion bound."""

        mechanism = self.aic_mte2_gemm_persistent_issue_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GEMM persistent issue service")
        expected_fields = {
            "dtype", "active_aic_groups", "worker_grid_m", "worker_grid_n",
            "operand_buffer_depth", "base_tile_m", "base_tile_n",
            "base_tile_k", "a_rows", "a_columns",
            "a_source_row_stride_elements", "b_rows", "b_columns",
            "b_source_row_stride_elements", "output_tiles_per_worker",
            "k_slabs", "mte2_programs_per_worker",
        }
        if set(signature) != expected_fields:
            raise ValueError(
                "AIC GEMM persistent issue signature fields changed")
        for name, value in signature.items():
            if name == "dtype":
                if not isinstance(value, str) or not value:
                    raise TypeError(
                        "AIC GEMM persistent issue dtype must be a string")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GEMM persistent issue {name} must be an integer")
            elif value <= 0:
                raise ValueError(
                    f"AIC GEMM persistent issue {name} must be positive")
        if signature != {
                name: mechanism[name] for name in expected_fields}:
            raise ValueError(
                "AIC GEMM persistent issue program is outside the measured "
                "domain")
        return mechanism["mte2_issue_ns"]

    def aic_matmul_v2_dynamic_m_tail_control_time_ns(self):
        """Return Scalar service for one partial-M output/K-tile pair."""

        mechanism = self.aic_matmul_v2_dynamic_m_tail_control_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MatMulV2 dynamic M-tail control service")
        return mechanism["service_ns_per_request"]

    def aic_matmul_v2_dynamic_m_tail_maximum_descriptor_k(self):
        """Return the measured maximum partial-M descriptor K span."""

        mechanism = self.aic_matmul_v2_dynamic_m_tail_control_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MatMulV2 dynamic M-tail control service")
        return mechanism["maximum_partial_m_descriptor_k"]

    def aic_matmul_v2_base_k_lifecycle_time_ns(
            self, base_k, output_tiles_per_worker,
            k_tiles_per_output_tile):
        """Return the independent MatMulV2 control-lifecycle lower bound."""

        for name, value in (
                ("base_k", base_k),
                ("output_tiles_per_worker", output_tiles_per_worker),
                ("k_tiles_per_output_tile", k_tiles_per_output_tile)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MatMulV2 base-K lifecycle {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MatMulV2 base-K lifecycle {name} must be positive")
        mechanism = self.aic_matmul_v2_base_k_lifecycle_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MatMulV2 base-K lifecycle service")
        if base_k not in mechanism["base_k_values"]:
            raise ValueError(
                "AIC MatMulV2 base-K lifecycle is outside its measured "
                "base-K domain")
        return (
            mechanism["startup_ns"]
            + output_tiles_per_worker * k_tiles_per_output_tile
            * mechanism["base_k_control_ns"])

    def aic_matmul_v2_base_k_lifecycle_supports(
            self, dtype, active_aic_workers, base_tile_m, base_tile_n,
            base_k, projection_k, output_tiles_per_worker,
            a1_descriptor_k, b1_descriptor_k, operand_buffer_depth,
            dynamic_m_tail):
        """Return whether a MatMulV2 program matches the measured source."""

        mechanism = self.aic_matmul_v2_base_k_lifecycle_service
        if mechanism is None:
            return False
        return (
            dtype == mechanism["dtype"]
            and active_aic_workers == mechanism["active_aic_workers"]
            and base_tile_m == mechanism["base_tile_m"]
            and base_tile_n == mechanism["base_tile_n"]
            and base_k in mechanism["base_k_values"]
            and projection_k in mechanism["projection_k_values"]
            and output_tiles_per_worker
                == mechanism["output_tiles_per_worker"]
            and a1_descriptor_k == mechanism["a1_descriptor_k"]
            and b1_descriptor_k == mechanism["b1_descriptor_k"]
            and operand_buffer_depth == mechanism["operand_buffer_depth"]
            and dynamic_m_tail == mechanism["dynamic_m_tail"])

    def aic_matmul_v2_small_m_base_k_control_time_ns(self):
        """Return one validated small-M Scalar base-K source request."""

        mechanism = self.aic_matmul_v2_small_m_base_k_control_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MatMulV2 small-M base-K control service")
        return mechanism["base_k_control_ns_per_pair"]

    def aic_matmul_v2_small_m_base_k_control_supports(
            self, dtype, active_aic_workers, base_tile_m, base_tile_n,
            base_k, projection_k, output_tiles_per_worker,
            a1_descriptor_k, b1_descriptor_k, operand_buffer_depth,
            dynamic_m_tail):
        """Return whether the source has the validated small-M controller."""

        mechanism = self.aic_matmul_v2_small_m_base_k_control_service
        if mechanism is None:
            return False
        workers = mechanism["active_aic_workers_by_base_n"]
        return (
            dtype == mechanism["dtype"]
            and base_tile_n in workers
            and active_aic_workers == workers[base_tile_n]
            and base_tile_m == mechanism["base_tile_m"]
            and base_k == mechanism["base_tile_k"]
            and projection_k > 0
            and projection_k % base_k == 0
            and output_tiles_per_worker
                == mechanism["output_tiles_per_worker"]
            and a1_descriptor_k == mechanism["a1_descriptor_k"]
            and b1_descriptor_k == mechanism["b1_descriptor_k"]
            and operand_buffer_depth == mechanism["operand_buffer_depth"]
            and dynamic_m_tail)

    def aic_mte2_gemm_region_completion_time_ns(
            self, dtype, active_aic_groups, operand_buffer_depth,
            base_tile_m, base_tile_n, base_tile_k,
            a_rows, a_columns, a_source_row_stride_elements,
            b_rows, b_columns, b_source_row_stride_elements,
            m_tiles, n_tiles, m_regions, n_regions,
            m_tiles_per_region, n_tiles_per_region,
            maximum_output_tiles_per_worker, k_slabs,
            mte2_programs_per_worker, mte1_pairs_per_worker):
        """Return an exact measured MatMulV3 region-program completion."""

        mechanisms = self.aic_mte2_gemm_region_completion_services
        if not mechanisms:
            raise ValueError("hardware has no AIC GEMM region service")
        values = {
            "dtype": dtype,
            "active_aic_groups": active_aic_groups,
            "operand_buffer_depth": operand_buffer_depth,
            "base_tile_m": base_tile_m,
            "base_tile_n": base_tile_n,
            "base_tile_k": base_tile_k,
            "a_rows": a_rows,
            "a_columns": a_columns,
            "a_source_row_stride_elements": a_source_row_stride_elements,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "b_source_row_stride_elements": b_source_row_stride_elements,
            "m_tiles": m_tiles,
            "n_tiles": n_tiles,
            "m_regions": m_regions,
            "n_regions": n_regions,
            "m_tiles_per_region": m_tiles_per_region,
            "n_tiles_per_region": n_tiles_per_region,
            "maximum_output_tiles_per_worker": (
                maximum_output_tiles_per_worker),
            "k_slabs": k_slabs,
            "mte2_programs_per_worker": mte2_programs_per_worker,
            "mte1_pairs_per_worker": mte1_pairs_per_worker,
        }
        for name, value in values.items():
            if name == "dtype":
                if not isinstance(value, str) or not value:
                    raise TypeError("AIC GEMM region dtype must be a string")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"AIC GEMM region {name} must be an integer")
            elif value <= 0:
                raise ValueError(f"AIC GEMM region {name} must be positive")
        for mechanism in mechanisms:
            if values == {name: mechanism[name] for name in values}:
                return mechanism["completion_ns"]
        raise ValueError(
            "AIC GEMM region program is outside the measured domain")

    def aic_grouped_matmul_finite_stream_completion_supports(
            self, pair_count):
        """Return whether a pair count lies in a measured stream segment."""

        if isinstance(pair_count, bool) or not isinstance(pair_count, int):
            raise TypeError(
                "AIC GroupedMatmul finite-stream pair_count must be an integer")
        if pair_count <= 0:
            raise ValueError(
                "AIC GroupedMatmul finite-stream pair_count must be positive")
        mechanism = self.aic_grouped_matmul_finite_stream_completion_service
        if mechanism is None:
            return False
        return any(
            segment["minimum_pairs_per_worker"] <= pair_count
            <= segment["maximum_pairs_per_worker"]
            for segment in mechanism["segments"])

    def aic_grouped_matmul_finite_stream_completion_time_ns(
            self, a_rows, a_columns, b_rows, b_columns,
            active_aic_groups, drain_interval_pairs, pair_count):
        """Return the finite GroupedMatmul operand-stream completion bound."""

        values = {
            "a_rows": a_rows,
            "a_columns": a_columns,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "active_aic_groups": active_aic_groups,
            "drain_interval_pairs": drain_interval_pairs,
            "pair_count": pair_count,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GroupedMatmul finite-stream {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul finite-stream {name} must be positive")
        mechanism = self.aic_grouped_matmul_finite_stream_completion_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul finite-stream service")
        expected = {
            "a_rows": mechanism["a_rows"],
            "a_columns": mechanism["a_columns"],
            "b_rows": mechanism["b_rows"],
            "b_columns": mechanism["b_columns"],
            "active_aic_groups": mechanism["active_aic_groups"],
            "drain_interval_pairs": mechanism["drain_interval_pairs"],
        }
        if any(values[name] != expected[name] for name in expected):
            raise ValueError(
                "AIC GroupedMatmul finite-stream geometry is outside the "
                "measured domain")
        if not self.aic_grouped_matmul_finite_stream_completion_supports(
                pair_count):
            raise ValueError(
                "AIC GroupedMatmul finite-stream length is outside the "
                "measured domain")
        for segment in mechanism["segments"]:
            if not (segment["minimum_pairs_per_worker"] <= pair_count
                    <= segment["maximum_pairs_per_worker"]):
                continue
            if segment["model"] == "constant_rate_upper_bound":
                return pair_count * segment["completion_ns_per_pair"]
            if segment["model"] == "piecewise_linear_completion_span":
                return _linear_interpolate(
                    segment["measured_pairs_per_worker"],
                    segment["completion_span_ns"], pair_count)
            raise ValueError(
                "unsupported AIC GroupedMatmul finite-stream segment")
        raise AssertionError("validated pair count has no completion segment")

    def aic_grouped_matmul_k2880_wide_base_tail_supports(
            self, base_m, active_m, projection_k, a_columns, b_rows,
            b_columns, active_aic_groups, drain_interval_pairs, pair_count):
        """Return whether a wide-base tail stream is in the measured domain."""

        values = {
            "base_m": base_m,
            "active_m": active_m,
            "projection_k": projection_k,
            "a_columns": a_columns,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "active_aic_groups": active_aic_groups,
            "drain_interval_pairs": drain_interval_pairs,
            "pair_count": pair_count,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GroupedMatmul K2880 tail {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul K2880 tail {name} must be positive")
        mechanism = (
            self.aic_grouped_matmul_k2880_wide_base_tail_completion_service)
        if mechanism is None:
            return False
        return (
            base_m in mechanism["base_m_values"]
            and active_m in mechanism["active_m_values"]
            and projection_k == mechanism["projection_k"]
            and a_columns == mechanism["a_columns"]
            and b_rows == mechanism["b_rows"]
            and b_columns == mechanism["b_columns"]
            and active_aic_groups == mechanism["active_aic_groups"]
            and drain_interval_pairs == mechanism["drain_interval_pairs"]
            and mechanism["minimum_pairs_per_worker"] <= pair_count
            <= mechanism["maximum_pairs_per_worker"])

    def aic_grouped_matmul_k2880_wide_base_tail_time_ns(
            self, base_m, active_m, projection_k, a_columns, b_rows,
            b_columns, active_aic_groups, drain_interval_pairs, pair_count):
        """Return the finite K2880 wide-base tail completion lower bound."""

        mechanism = (
            self.aic_grouped_matmul_k2880_wide_base_tail_completion_service)
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul K2880 tail service")
        if not self.aic_grouped_matmul_k2880_wide_base_tail_supports(
                base_m, active_m, projection_k, a_columns, b_rows,
                b_columns, active_aic_groups, drain_interval_pairs,
                pair_count):
            raise ValueError(
                "AIC GroupedMatmul K2880 tail program is outside the measured "
                "domain")
        return _linear_interpolate(
            mechanism["measured_pairs_per_worker"],
            mechanism["completion_span_ns"], pair_count)

    def aic_grouped_matmul_k2880_output_tile_supports(
            self, base_m, active_m, projection_k, a_columns, b_rows,
            b_columns, active_aic_groups, drain_interval_pairs,
            output_tile_pairs):
        """Return whether one K2880 IterateAll tile is measured."""

        values = {
            "base_m": base_m,
            "active_m": active_m,
            "projection_k": projection_k,
            "a_columns": a_columns,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "active_aic_groups": active_aic_groups,
            "drain_interval_pairs": drain_interval_pairs,
            "output_tile_pairs": output_tile_pairs,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GroupedMatmul K2880 output-tile {name} must be an "
                    "integer")
            if value <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul K2880 output-tile {name} must be "
                    "positive")
        mechanism = (
            self.aic_grouped_matmul_k2880_output_tile_completion_service)
        if mechanism is None:
            return False
        return (
            base_m in mechanism["base_m_values"]
            and active_m in mechanism["active_m_values"]
            and projection_k == mechanism["projection_k"]
            and a_columns == mechanism["a_columns"]
            and b_rows == mechanism["b_rows"]
            and b_columns == mechanism["b_columns"]
            and active_aic_groups == mechanism["active_aic_groups"]
            and drain_interval_pairs == mechanism["drain_interval_pairs"]
            and output_tile_pairs == mechanism["output_tile_pairs"])

    def aic_grouped_matmul_k2880_output_tile_time_ns(
            self, base_m, active_m, projection_k, a_columns, b_rows,
            b_columns, active_aic_groups, drain_interval_pairs,
            output_tile_pairs):
        """Return one source-bounded K2880 IterateAll completion span."""

        mechanism = (
            self.aic_grouped_matmul_k2880_output_tile_completion_service)
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul K2880 output-tile service")
        if not self.aic_grouped_matmul_k2880_output_tile_supports(
                base_m, active_m, projection_k, a_columns, b_rows,
                b_columns, active_aic_groups, drain_interval_pairs,
                output_tile_pairs):
            raise ValueError(
                "AIC GroupedMatmul K2880 output tile is outside the measured "
                "domain")
        return mechanism["completion_span_ns_by_active_m"][active_m]

    def aic_grouped_matmul_step4_worker_sequence_supports(
            self, role, dtype, active_aic_groups, base_m, active_m,
            projection_k, projection_n, tile_n, tile_k,
            depth_a1, depth_b1, step_ka, step_kb,
            db_l0a, db_l0b, db_l0c, output_tiles_per_worker):
        """Return whether one complete worker sequence is measured."""

        if not isinstance(role, str):
            raise TypeError(
                "AIC GroupedMatmul step-4 worker role must be a string")
        if role not in ("w1", "w2"):
            raise ValueError(
                "AIC GroupedMatmul step-4 worker role must be 'w1' or 'w2'")
        if not isinstance(dtype, str) or not dtype:
            raise TypeError(
                "AIC GroupedMatmul step-4 worker dtype must be a non-empty "
                "string")
        values = {
            "active_aic_groups": active_aic_groups,
            "base_m": base_m,
            "active_m": active_m,
            "projection_k": projection_k,
            "projection_n": projection_n,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "depth_a1": depth_a1,
            "depth_b1": depth_b1,
            "step_ka": step_ka,
            "step_kb": step_kb,
            "db_l0a": db_l0a,
            "db_l0b": db_l0b,
            "db_l0c": db_l0c,
            "output_tiles_per_worker": output_tiles_per_worker,
        }
        for name, item in values.items():
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"AIC GroupedMatmul step-4 worker {name} must be an "
                    "integer")
            if item <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul step-4 worker {name} must be positive")
        mechanism = (
            self.aic_grouped_matmul_step4_worker_sequence_completion_service)
        if mechanism is None:
            return False
        role_service = mechanism["roles"][role]
        return (
            dtype == mechanism["dtype"]
            and active_aic_groups == mechanism["active_aic_groups"]
            and base_m == mechanism["base_m"]
            and active_m == mechanism["active_m"]
            and projection_k == role_service["projection_k"]
            and projection_n == role_service["projection_n"]
            and tile_n == mechanism["tile_n"]
            and tile_k == mechanism["tile_k"]
            and depth_a1 == mechanism["depth_a1"]
            and depth_b1 == mechanism["depth_b1"]
            and step_ka == mechanism["step_ka"]
            and step_kb == mechanism["step_kb"]
            and db_l0a == mechanism["db_l0a"]
            and db_l0b == mechanism["db_l0b"]
            and db_l0c == mechanism["db_l0c"]
            and role_service["minimum_output_tiles_per_worker"]
            <= output_tiles_per_worker
            <= role_service["maximum_output_tiles_per_worker"])

    def aic_grouped_matmul_step4_worker_sequence_time_ns(
            self, role, dtype, active_aic_groups, base_m, active_m,
            projection_k, projection_n, tile_n, tile_k,
            depth_a1, depth_b1, step_ka, step_kb,
            db_l0a, db_l0b, db_l0c, output_tiles_per_worker):
        """Interpolate one complete source worker sequence in-domain."""

        mechanism = (
            self.aic_grouped_matmul_step4_worker_sequence_completion_service)
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul step-4 worker-sequence "
                "service")
        if not self.aic_grouped_matmul_step4_worker_sequence_supports(
                role, dtype, active_aic_groups, base_m, active_m,
                projection_k, projection_n, tile_n, tile_k,
                depth_a1, depth_b1, step_ka, step_kb,
                db_l0a, db_l0b, db_l0c, output_tiles_per_worker):
            raise ValueError(
                "AIC GroupedMatmul step-4 worker sequence is outside the "
                "measured domain")
        role_service = mechanism["roles"][role]
        return _linear_interpolate(
            role_service["measured_output_tiles_per_worker"],
            role_service["completion_span_ns"], output_tiles_per_worker)

    def aic_grouped_matmul_short_source_prelude_control_supports(
            self, dtype, active_aic_groups, base_m, active_m, projection_k,
            tile_n, tile_k, depth_a1, depth_b1, step_ka, step_kb,
            db_l0a, db_l0b, db_l0c):
        """Return whether one short-source tile has an exact control service."""

        if not isinstance(dtype, str) or not dtype:
            raise TypeError(
                "AIC GroupedMatmul short-source prelude dtype must be a "
                "non-empty string")
        values = {
            "active_aic_groups": active_aic_groups,
            "base_m": base_m,
            "active_m": active_m,
            "projection_k": projection_k,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "depth_a1": depth_a1,
            "depth_b1": depth_b1,
            "step_ka": step_ka,
            "step_kb": step_kb,
            "db_l0a": db_l0a,
            "db_l0b": db_l0b,
            "db_l0c": db_l0c,
        }
        for name, item in values.items():
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"AIC GroupedMatmul short-source prelude {name} must be "
                    "an integer")
            if item <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul short-source prelude {name} must be "
                    "positive")
        mechanism = self.aic_grouped_matmul_short_source_prelude_control_service
        if mechanism is None:
            return False
        if projection_k % tile_k:
            return False
        return (
            dtype == mechanism["dtype"]
            and active_aic_groups == mechanism["active_aic_groups"]
            and base_m == mechanism["base_m"]
            and active_m == mechanism["active_m"]
            and tile_n == mechanism["tile_n"]
            and tile_k == mechanism["tile_k"]
            and depth_a1 == mechanism["depth_a1"]
            and depth_b1 == mechanism["depth_b1"]
            and step_ka == mechanism["step_ka"]
            and step_kb == mechanism["step_kb"]
            and db_l0a == mechanism["db_l0a"]
            and db_l0b == mechanism["db_l0b"]
            and db_l0c == mechanism["db_l0c"]
            and projection_k // tile_k in mechanism["k_rounds"])

    def aic_grouped_matmul_short_source_prelude_control_time_ns(
            self, dtype, active_aic_groups, base_m, active_m, projection_k,
            tile_n, tile_k, depth_a1, depth_b1, step_ka, step_kb,
            db_l0a, db_l0b, db_l0c):
        """Return one exact source-controller span before IterateAll issue."""

        mechanism = self.aic_grouped_matmul_short_source_prelude_control_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul short-source prelude "
                "service")
        if not self.aic_grouped_matmul_short_source_prelude_control_supports(
                dtype, active_aic_groups, base_m, active_m, projection_k,
                tile_n, tile_k, depth_a1, depth_b1, step_ka, step_kb,
                db_l0a, db_l0b, db_l0c):
            raise ValueError(
                "AIC GroupedMatmul short-source prelude is outside the "
                "measured domain")
        return mechanism["service_time_ns_by_k_rounds"][
            projection_k // tile_k]

    def aic_grouped_matmul_entry_metadata_pair_supports(
            self, dtype, active_aic_groups, source_kind, projection_k,
            projection_n):
        """Return whether the exact two-member entry prelude is selected."""

        if not isinstance(dtype, str) or not dtype:
            raise TypeError(
                "AIC GroupedMatmul entry metadata dtype must be a non-empty "
                "string")
        if not isinstance(source_kind, str) or not source_kind:
            raise TypeError(
                "AIC GroupedMatmul entry metadata source kind must be a "
                "non-empty string")
        for name, value in (
                ("active_aic_groups", active_aic_groups),
                ("projection_k", projection_k),
                ("projection_n", projection_n)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GroupedMatmul entry metadata {name} must be an "
                    "integer")
            if value <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul entry metadata {name} must be "
                    "positive")
        mechanism = self.aic_grouped_matmul_entry_metadata_pair_service
        if mechanism is None:
            return False
        return (
            dtype == mechanism["dtype"]
            and source_kind == mechanism["source_kind"]
            and active_aic_groups == mechanism["active_aic_groups"]
            and projection_k not in mechanism["unsupported_projection_dimensions"]
            and projection_n not in mechanism["unsupported_projection_dimensions"])

    def aic_grouped_matmul_entry_metadata_pair_time_ns(
            self, dtype, active_aic_groups, source_kind, projection_k,
            projection_n):
        """Return one exact AIC entry pair before ``GMMProcess``."""

        mechanism = self.aic_grouped_matmul_entry_metadata_pair_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul entry metadata-pair "
                "service")
        if not self.aic_grouped_matmul_entry_metadata_pair_supports(
                dtype, active_aic_groups, source_kind, projection_k,
                projection_n):
            raise ValueError(
                "AIC GroupedMatmul entry metadata pair is outside the "
                "measured domain")
        return mechanism["service_time_ns"]

    def aic_grouped_matmul_m32_short_source_prelude_control_supports(
            self, dtype, active_aic_groups, base_m, active_m, logical_n,
            projection_k, tile_n, tile_k, depth_a1, depth_b1, step_ka,
            step_kb, db_l0a, db_l0b, db_l0c):
        """Return whether one M32 short-source tile has exact control data."""

        if not isinstance(dtype, str) or not dtype:
            raise TypeError(
                "AIC GroupedMatmul M32 short-source prelude dtype must be "
                "a non-empty string")
        values = {
            "active_aic_groups": active_aic_groups,
            "base_m": base_m,
            "active_m": active_m,
            "logical_n": logical_n,
            "projection_k": projection_k,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "depth_a1": depth_a1,
            "depth_b1": depth_b1,
            "step_ka": step_ka,
            "step_kb": step_kb,
            "db_l0a": db_l0a,
            "db_l0b": db_l0b,
            "db_l0c": db_l0c,
        }
        for name, item in values.items():
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"AIC GroupedMatmul M32 short-source prelude {name} "
                    "must be an integer")
            if item <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul M32 short-source prelude {name} "
                    "must be positive")
        mechanism = (
            self.aic_grouped_matmul_m32_short_source_prelude_control_service)
        if mechanism is None or projection_k % tile_k:
            return False
        return (
            dtype == mechanism["dtype"]
            and active_aic_groups == mechanism["active_aic_groups"]
            and base_m == mechanism["base_m"]
            and active_m == mechanism["active_m"]
            and logical_n in mechanism["logical_n_values"]
            and tile_n == mechanism["tile_n"]
            and tile_k == mechanism["tile_k"]
            and depth_a1 == mechanism["depth_a1"]
            and depth_b1 == mechanism["depth_b1"]
            and step_ka == mechanism["step_ka"]
            and step_kb == mechanism["step_kb"]
            and db_l0a == mechanism["db_l0a"]
            and db_l0b == mechanism["db_l0b"]
            and db_l0c == mechanism["db_l0c"]
            and projection_k // tile_k in mechanism["k_rounds"])

    def aic_grouped_matmul_m32_short_source_prelude_control_time_ns(
            self, dtype, active_aic_groups, base_m, active_m, logical_n,
            projection_k, tile_n, tile_k, depth_a1, depth_b1, step_ka,
            step_kb, db_l0a, db_l0b, db_l0c):
        """Return one exact M32 source-controller span before IterateAll."""

        mechanism = (
            self.aic_grouped_matmul_m32_short_source_prelude_control_service)
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul M32 short-source "
                "prelude service")
        if not self.\
                aic_grouped_matmul_m32_short_source_prelude_control_supports(
                    dtype, active_aic_groups, base_m, active_m, logical_n,
                    projection_k, tile_n, tile_k, depth_a1, depth_b1, step_ka,
                    step_kb, db_l0a, db_l0b, db_l0c):
            raise ValueError(
                "AIC GroupedMatmul M32 short-source prelude is outside the "
                "measured domain")
        return mechanism["service_time_ns_by_logical_n_and_k_rounds"][
            logical_n][projection_k // tile_k]

    def aic_grouped_matmul_short_source_tail_prelude_control_supports(
            self, dtype, active_aic_groups, base_m, active_m, logical_n,
            projection_k, tile_n, tile_k, depth_a1, depth_b1, step_ka,
            step_kb, db_l0a, db_l0b, db_l0c):
        """Return whether one terminal-N tile has an exact control service."""

        if not isinstance(dtype, str) or not dtype:
            raise TypeError(
                "AIC GroupedMatmul short-source tail prelude dtype must be "
                "a non-empty string")
        values = {
            "active_aic_groups": active_aic_groups,
            "base_m": base_m,
            "active_m": active_m,
            "logical_n": logical_n,
            "projection_k": projection_k,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "depth_a1": depth_a1,
            "depth_b1": depth_b1,
            "step_ka": step_ka,
            "step_kb": step_kb,
            "db_l0a": db_l0a,
            "db_l0b": db_l0b,
            "db_l0c": db_l0c,
        }
        for name, item in values.items():
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"AIC GroupedMatmul short-source tail prelude {name} "
                    "must be an integer")
            if item <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul short-source tail prelude {name} "
                    "must be positive")
        mechanism = (
            self.aic_grouped_matmul_short_source_tail_prelude_control_service)
        if mechanism is None or projection_k % tile_k:
            return False
        return (
            dtype == mechanism["dtype"]
            and active_aic_groups == mechanism["active_aic_groups"]
            and base_m == mechanism["base_m"]
            and active_m == mechanism["active_m"]
            and logical_n in mechanism["logical_n_values"]
            and tile_n == mechanism["tile_n"]
            and tile_k == mechanism["tile_k"]
            and depth_a1 == mechanism["depth_a1"]
            and depth_b1 == mechanism["depth_b1"]
            and step_ka == mechanism["step_ka"]
            and step_kb == mechanism["step_kb"]
            and db_l0a == mechanism["db_l0a"]
            and db_l0b == mechanism["db_l0b"]
            and db_l0c == mechanism["db_l0c"]
            and projection_k // tile_k in mechanism["k_rounds"])

    def aic_grouped_matmul_short_source_tail_prelude_control_time_ns(
            self, dtype, active_aic_groups, base_m, active_m, logical_n,
            projection_k, tile_n, tile_k, depth_a1, depth_b1, step_ka,
            step_kb, db_l0a, db_l0b, db_l0c):
        """Return one exact terminal-N controller span before IterateAll."""

        mechanism = (
            self.aic_grouped_matmul_short_source_tail_prelude_control_service)
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul short-source tail "
                "prelude service")
        if not self.aic_grouped_matmul_short_source_tail_prelude_control_supports(
                dtype, active_aic_groups, base_m, active_m, logical_n,
                projection_k, tile_n, tile_k, depth_a1, depth_b1, step_ka,
                step_kb, db_l0a, db_l0b, db_l0c):
            raise ValueError(
                "AIC GroupedMatmul short-source tail prelude is outside the "
                "measured domain")
        return mechanism["service_time_ns_by_logical_n_and_k_rounds"][
            logical_n][projection_k // tile_k]

    def aic_mte2_cold_operand_pair_stream_time_ns(
            self, a_rows, a_columns, b_rows, b_columns,
            active_aic_groups, pair_count):
        """Return the measured DRAM-cold A/B stream time for one AIC."""

        for name, value in (
                ("a_rows", a_rows), ("a_columns", a_columns),
                ("b_rows", b_rows), ("b_columns", b_columns),
                ("active_aic_groups", active_aic_groups),
                ("pair_count", pair_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MTE2 cold operand-pair {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MTE2 cold operand-pair {name} must be positive")
        mechanism = self.aic_mte2_cold_operand_pair_stream_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MTE2 cold operand-pair stream service")
        observed_geometry = (a_rows, a_columns, b_rows, b_columns)
        expected_geometry = tuple(
            mechanism[name]
            for name in ("a_rows", "a_columns", "b_rows", "b_columns"))
        if observed_geometry != expected_geometry:
            raise ValueError(
                "AIC MTE2 cold operand-pair geometry is outside the "
                "measured domain")
        if active_aic_groups != mechanism["active_aic_groups"]:
            raise ValueError(
                "AIC MTE2 cold operand-pair active-group count is outside "
                "the measured domain")
        if not (mechanism["minimum_pairs_per_worker"] <= pair_count
                <= mechanism["maximum_pairs_per_worker"]):
            raise ValueError(
                "AIC MTE2 cold operand-pair stream length is outside the "
                "measured domain")
        return pair_count * mechanism["continuous_ns_per_pair"]

    def aic_mte2_mixed_ring_pair_time_ns(
            self, a_rows, a_columns, b_rows, b_columns,
            active_aic_groups, pair_count, a_residency, b_residency):
        """Return the independent mixed-residency A/B pair service."""

        values = {
            "a_rows": a_rows,
            "a_columns": a_columns,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "active_aic_groups": active_aic_groups,
        }
        for name, value in (*values.items(), ("pair_count", pair_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"mixed-ring MTE2 {name} must be an integer")
            if value <= 0:
                raise ValueError(f"mixed-ring MTE2 {name} must be positive")
        mechanism = self.aic_mte2_mixed_ring_long_overlap_domain
        if mechanism is None:
            raise ValueError("hardware has no mixed-ring MTE2 pair service")
        expected = {
            name: mechanism[name] for name in values
        }
        if values != expected:
            raise ValueError("mixed-ring MTE2 geometry is outside its domain")
        if not (mechanism["minimum_service_pairs_per_worker"] <= pair_count
                <= mechanism["maximum_service_pairs_per_worker"]):
            raise ValueError("mixed-ring MTE2 pair count is outside its domain")
        residency = [a_residency, b_residency]
        if residency not in mechanism["source_residency_pairs"]:
            raise ValueError("mixed-ring MTE2 residency is outside its domain")
        service_label = f"{a_residency}_a_{b_residency}_b"
        return pair_count * mechanism[
            "published_service_ns_per_pair"][service_label]

    def aic_mte2_cold_n_major_b_pair_time_ns(
            self, a_rows, a_columns, b_rows, b_columns,
            b_source_row_stride_elements, active_aic_groups, pair_count):
        """Return the exact cold M16 pair service with N-major B stride."""

        values = {
            "a_rows": a_rows,
            "a_columns": a_columns,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "b_source_row_stride_elements": b_source_row_stride_elements,
            "active_aic_groups": active_aic_groups,
            "pairs_per_worker": pair_count,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"cold N-major B pair {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"cold N-major B pair {name} must be positive")
        mechanism = self.aic_mte2_cold_n_major_b_stride_service
        if mechanism is None:
            raise ValueError(
                "hardware has no cold N-major B-stride pair service")
        expected = {
            name: mechanism[name]
            for name in values
        }
        if values != expected:
            raise ValueError(
                "cold N-major B pair is outside the measured domain")
        contiguous_b_serial_ns = self.aic_mte2_nd2nz_serial_time_ns(
            b_rows * b_columns * 2, active_aic_groups)
        stride_delta_ns = (
            mechanism["strided_b_serial_ns"] - contiguous_b_serial_ns)
        if stride_delta_ns <= 0:
            raise ValueError(
                "cold N-major B stride must have a positive serial penalty")
        return pair_count * (
            mechanism["contiguous_pair_ns"] + stride_delta_ns)

    def aic_mte2_legacy_double_l0c_pair_time_ns(
            self, a_rows, a_columns, a_source_row_stride_elements,
            b_rows, b_columns, b_source_row_stride_elements,
            active_aic_groups, pair_count):
        """Return one exact cold MatMulV2 A1/B1 pair-program service."""

        values = {
            "a_rows": a_rows,
            "a_columns": a_columns,
            "a_source_row_stride_elements": a_source_row_stride_elements,
            "b_rows": b_rows,
            "b_columns": b_columns,
            "b_source_row_stride_elements": b_source_row_stride_elements,
            "active_aic_groups": active_aic_groups,
            "pairs_per_worker": pair_count,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"legacy double-L0C pair {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"legacy double-L0C pair {name} must be positive")
        mechanism = self.aic_mte2_legacy_double_l0c_pair_service
        if mechanism is None:
            raise ValueError(
                "hardware has no legacy double-L0C pair service")
        if any(values[name] != mechanism[name] for name in values):
            raise ValueError(
                "legacy double-L0C pair program is outside the measured "
                "domain")
        return pair_count * mechanism["continuous_ns_per_pair"]

    def aic_mte2_nd2nz_strided_serial_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups):
        """Return measured serial service for one strided ND2NZ copy."""

        return self._aic_mte2_nd2nz_strided_time_ns(
            rows, columns, source_row_stride_elements,
            active_aic_groups, "serial_duration_ns")

    def aic_mte2_nd2nz_strided_stream_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups):
        """Return measured stream interval for one strided ND2NZ copy."""

        return self._aic_mte2_nd2nz_strided_time_ns(
            rows, columns, source_row_stride_elements,
            active_aic_groups, "stream_duration_ns")

    def aic_mte2_nd2nz_33copy_group_time_ns(
            self, source_k_columns, active_aic_groups):
        """Return one cold large-stride 1-A plus 32-B CopyIn group."""

        for name, value in (
                ("source_k_columns", source_k_columns),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MTE2 33-copy group {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MTE2 33-copy group {name} must be positive")
        mechanism = self.aic_mte2_nd2nz_33copy_group_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MTE2 33-copy group service")
        if active_aic_groups != mechanism["active_aic_groups"]:
            raise ValueError(
                "AIC MTE2 33-copy group active AIC count is outside the "
                "measured domain")
        durations = mechanism["service_time_ns_by_request_size"]
        if source_k_columns not in durations:
            raise ValueError(
                "AIC MTE2 33-copy group K extent is outside the measured "
                "domain")
        return durations[source_k_columns]

    def aic_deterministic_split_k_source_queue_time_ns(
            self, aligned_a_rows, active_aic_groups):
        """Return the complete MTE2/MTE1 span of the longest source worker."""

        for name, value in (
                ("aligned_a_rows", aligned_a_rows),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC deterministic split-K source queue {name} must be "
                    "an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC deterministic split-K source queue {name} must be "
                    "positive")
        mechanism = self.aic_deterministic_split_k_source_queue_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC deterministic split-K source-queue "
                "service")
        if active_aic_groups != mechanism["active_aic_groups"]:
            raise ValueError(
                "AIC deterministic split-K source-queue active AIC count "
                "is outside the measured domain")
        durations = mechanism["completion_ns_by_aligned_a_rows"]
        if aligned_a_rows not in durations:
            raise ValueError(
                "AIC deterministic split-K source-queue A rows are outside "
                "the measured domain")
        return durations[aligned_a_rows]

    def aic_deterministic_split_k_collective_barrier_time_ns(
            self, active_aic_groups, active_aiv_workers, n_slabs):
        """Return the source-matched two-slab collective-only excess."""

        for name, value in (
                ("active_aic_groups", active_aic_groups),
                ("active_aiv_workers", active_aiv_workers),
                ("n_slabs", n_slabs)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC deterministic split-K collective {name} must be "
                    "an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC deterministic split-K collective {name} must be "
                    "positive")
        mechanism = self.aic_deterministic_split_k_collective_barrier_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC deterministic split-K collective "
                "barrier service")
        if (active_aic_groups != mechanism["active_aic_groups"]
                or active_aiv_workers != mechanism["active_aiv_workers"]
                or n_slabs != mechanism["n_slabs"]):
            raise ValueError(
                "AIC deterministic split-K collective is outside the "
                "measured source domain")
        return mechanism["collective_lifecycle_excess_ns"]

    def memory_exact_request_service_time_s(
            self, resource_role, service, request_size):
        """Resolve an exact memory completion through HWSpec ownership."""

        if service == _AIC_MTE2_ND2NZ_33COPY_GROUP_SERVICE:
            if resource_role != "dram":
                raise ValueError(
                    "AIC MTE2 33-copy group service requires the DRAM path")
            return self.aic_mte2_nd2nz_33copy_group_time_ns(
                request_size, self.num_compute_units) * 1e-9
        if service in _AIC_FIXPIPE_BF16_FINITE_STREAM_SIGNATURE_BY_SERVICE:
            if resource_role != "dram":
                raise ValueError(
                    "AIC BF16 finite Fixpipe service requires the DRAM path")
            rows, columns, active_aic_groups = (
                _AIC_FIXPIPE_BF16_FINITE_STREAM_SIGNATURE_BY_SERVICE[service])
            # The compact source program repeats one terminal store stage per
            # output tile.  Return each stage's equal share of the measured
            # finite stream so the repeated stages sum to the exact drained
            # completion while preserving every PIPE_ALL tile boundary.
            return (
                self.aic_fixpipe_bf16_finite_stream_time_ns(
                    rows, columns, active_aic_groups, request_size)
                / request_size * 1e-9)
        return self.resource_by_role(
            resource_role).exact_request_service_time_s(
                service, request_size)

    def _aic_mte2_nd2nz_strided_time_ns(
            self, rows, columns, source_row_stride_elements,
            active_aic_groups, duration_name):
        for name, value in (
                ("rows", rows),
                ("columns", columns),
                ("source_row_stride_elements", source_row_stride_elements),
                ("active_aic_groups", active_aic_groups)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC MTE2 strided ND2NZ {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC MTE2 strided ND2NZ {name} must be positive")
        mechanism = self.aic_mte2_nd2nz_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MTE2 ND2NZ strided service")
        geometry = (rows, columns, source_row_stride_elements)
        geometries = mechanism["strided_geometries"]
        if geometry not in geometries:
            raise ValueError(
                "AIC MTE2 strided ND2NZ geometry is outside the measured "
                "domain")
        return _linear_interpolate(
            mechanism["strided_active_aic_groups"],
            geometries[geometry][duration_name],
            active_aic_groups,
        )

    def aic_mte1_load_pair_time_ns(self, stage_kind, head_dim, pair_count):
        """Return completed L0A/L0B source-pair service on one AIC."""

        if not isinstance(stage_kind, str) or not stage_kind:
            raise TypeError("AIC MTE1 stage_kind must be a non-empty string")
        for name, value in (("head_dim", head_dim), ("pair_count", pair_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"AIC MTE1 {name} must be an integer")
            if value <= 0:
                raise ValueError(f"AIC MTE1 {name} must be positive")
        mechanism = self.aic_mte1_load_pair_completion_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC MTE1 load-pair completion service")
        if stage_kind not in mechanism["stage_kinds"]:
            raise ValueError(
                f"AIC MTE1 stage kind {stage_kind!r} is outside the measured "
                "domain")
        try:
            head_index = mechanism["head_dims"].index(head_dim)
        except ValueError as error:
            raise ValueError(
                f"AIC MTE1 head dimension {head_dim} is outside the measured "
                "domain") from error
        return (
            pair_count
            * mechanism["completion_ns_by_stage"][stage_kind][head_index]
        )

    def aic_grouped_matmul_mte1_pair_time_ns(
            self, m_rows, n_columns, k_columns, pair_count):
        """Return completed GroupedMatmul L0A/L0B pair service on one AIC."""

        for name, value in (
                ("m_rows", m_rows), ("n_columns", n_columns),
                ("k_columns", k_columns), ("pair_count", pair_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GroupedMatmul MTE1 {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul MTE1 {name} must be positive")
        mechanism = self.aic_grouped_matmul_mte1_pair_completion_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC GroupedMatmul MTE1 pair service")
        if (k_columns != mechanism["k_columns"]
                or n_columns < mechanism["minimum_n_columns"]
                or n_columns > mechanism["maximum_n_columns"]
                or n_columns % mechanism["n_column_granularity"]):
            raise ValueError(
                "AIC GroupedMatmul MTE1 N/K geometry is outside the measured "
                "domain")
        try:
            row_index = mechanism["m_rows"].index(m_rows)
        except ValueError as error:
            raise ValueError(
                f"AIC GroupedMatmul MTE1 M rows {m_rows} are outside the "
                "measured domain") from error
        pair_ns = (
            mechanism["completion_ns_by_m_rows"][row_index]
            + (n_columns - mechanism["n_columns"])
            * mechanism["n_column_service_ns"])
        if pair_ns <= 0:
            raise ValueError("AIC GroupedMatmul MTE1 pair service is invalid")
        return pair_count * pair_ns

    def aic_grouped_matmul_mte1_pair_affine_extension_time_ns(
            self, m_rows, n_columns, k_columns, pair_count):
        """Extend the measured affine N service by one source granule."""

        for name, value in (
                ("m_rows", m_rows), ("n_columns", n_columns),
                ("k_columns", k_columns), ("pair_count", pair_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC GroupedMatmul MTE1 extension {name} must be an "
                    "integer")
            if value <= 0:
                raise ValueError(
                    f"AIC GroupedMatmul MTE1 extension {name} must be "
                    "positive")
        mechanism = self.aic_grouped_matmul_mte1_pair_completion_service
        if mechanism is None:
            return None
        minimum_n_columns = (
            mechanism["minimum_n_columns"]
            - mechanism["n_column_granularity"])
        if n_columns != minimum_n_columns:
            return None
        try:
            row_index = mechanism["m_rows"].index(m_rows)
        except ValueError:
            return None
        if k_columns != mechanism["k_columns"]:
            return None
        pair_ns = (
            mechanism["completion_ns_by_m_rows"][row_index]
            + (n_columns - mechanism["n_columns"])
            * mechanism["n_column_service_ns"])
        if pair_ns <= 0:
            raise ValueError(
                "AIC GroupedMatmul MTE1 affine extension is invalid")
        return pair_count * pair_ns

    def aic_matmul_v3_split_k_resident_tile_time_ns(
            self, dtype, active_aic_workers, tile_m, tile_n, tile_k):
        """Return one source-isolated MTE1/Cube/FixPipe tile completion.

        The probe holds the complete local source body fixed and shows a
        zero-intercept service proportional to K for 256 and 384 columns.
        CANN's same body uses K0=512 on the retained 2x4 programs.  Since GM
        and shared L2 are excluded from the probe, apply that local per-K
        service to non-empty AIC subsets and the source-admitted 512 column
        group as well.
        """

        mechanism = self.aic_matmul_v3_split_k_resident_tile_service
        if mechanism is None:
            raise ValueError(
                "hardware has no MatMulV3 split-K resident-tile service")
        if dtype != mechanism["dtype"]:
            raise ValueError(
                "MatMulV3 split-K resident-tile dtype is unsupported")
        for name, value in (
                ("active_aic_workers", active_aic_workers),
                ("tile_m", tile_m), ("tile_n", tile_n),
                ("tile_k", tile_k)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"MatMulV3 split-K resident-tile {name} must be integer")
            if value <= 0:
                raise ValueError(
                    f"MatMulV3 split-K resident-tile {name} must be positive")
        measured_k = tuple(sorted(mechanism["duration_ns_by_tile_k"]))
        ns_per_k = tuple(
            mechanism["duration_ns_by_tile_k"][columns] / columns
            for columns in measured_k)
        if (active_aic_workers > mechanism["active_aic_workers"]
                or tile_m != mechanism["tile_m"]
                or tile_n != mechanism["tile_n"]
                or tile_k % math.gcd(*measured_k)
                or not measured_k[0] <= tile_k <= 512
                or max(ns_per_k) / min(ns_per_k) > 1.001):
            raise ValueError(
                "MatMulV3 split-K resident-tile geometry is outside the "
                "measured domain")
        if tile_k in mechanism["duration_ns_by_tile_k"]:
            return mechanism["duration_ns_by_tile_k"][tile_k]
        return tile_k * statistics.median(ns_per_k)

    def aic_matmul_v3_asw_mte2_pair_time_ns(
            self, tile_m, tile_n, tile_k, active_aic_groups, pair_count):
        """Return a complete 256x128 ASW operand-pair stream per AIC."""

        for name, value in (
                ("tile_m", tile_m), ("tile_n", tile_n),
                ("tile_k", tile_k),
                ("active_aic_groups", active_aic_groups),
                ("pair_count", pair_count)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"MatMulV3 ASW MTE2 {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"MatMulV3 ASW MTE2 {name} must be positive")
        mechanism = self.aic_matmul_v3_asw_mte2_pair_stream_service
        if mechanism is None:
            raise ValueError(
                "hardware has no MatMulV3 ASW MTE2 pair service")
        if (tile_m != mechanism["tile_m"]
                or tile_n != mechanism["tile_n"]
                or tile_k != mechanism["tile_k"]
                or active_aic_groups != mechanism["active_aic_groups"]):
            raise ValueError(
                "MatMulV3 ASW MTE2 launch is outside the measured domain")
        if pair_count < mechanism["minimum_pairs_per_worker"]:
            raise ValueError(
                "MatMulV3 ASW MTE2 stream is shorter than the measured "
                "steady domain")
        return pair_count * mechanism["continuous_ns_per_pair"]

    def aic_fixpipe_full_tile_time_ns(
            self, rows, columns, destination_stride_elements):
        """Return completed row-major L0C-to-GM service for one call."""

        for name, value in (
                ("rows", rows),
                ("columns", columns),
                ("destination_stride_elements",
                 destination_stride_elements)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"AIC Fixpipe {name} must be an integer")
            if value <= 0:
                raise ValueError(f"AIC Fixpipe {name} must be positive")
        mechanism = self.aic_fixpipe_full_tile_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC Fixpipe full-tile service")
        if rows not in mechanism["supported_rows"]:
            raise ValueError("AIC Fixpipe rows are outside the measured domain")
        if columns != mechanism["column_count"]:
            raise ValueError(
                "AIC Fixpipe columns are outside the measured domain")
        alignment = mechanism["destination_stride_alignment_elements"]
        if (destination_stride_elements < columns
                or destination_stride_elements
                > mechanism["maximum_destination_stride_elements"]
                or destination_stride_elements % alignment):
            raise ValueError(
                "AIC Fixpipe destination stride is outside the measured "
                "domain")
        if rows >= mechanism["large_row_minimum_rows"]:
            return (
                mechanism["large_row_call_startup_ns"]
                + rows * mechanism["large_row_service_ns"]
            )
        phases = min(
            rows,
            columns // math.gcd(destination_stride_elements, columns),
        )
        if phases == 1:
            phase_penalty_ns = 0.0
        elif phases == 2:
            phase_penalty_ns = mechanism[
                "row_address_phase_2_penalty_ns"]
        elif phases == 4:
            phase_penalty_ns = mechanism[
                "row_address_phase_4_penalty_ns"]
        else:
            raise ValueError(
                "AIC Fixpipe row-address phase count is outside the "
                "measured domain")
        return (
            mechanism["call_startup_ns"]
            + rows * mechanism["row_service_ns"]
            + phase_penalty_ns
        )

    def aic_fixpipe_bf16_terminal_tile_time_ns(self, rows, columns):
        """Return one non-overlapped BF16 terminal tile completion."""

        for name, value in (("rows", rows), ("columns", columns)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC BF16 terminal Fixpipe {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC BF16 terminal Fixpipe {name} must be positive")
        mechanism = self.aic_fixpipe_bf16_terminal_tile_service
        if mechanism is None:
            raise ValueError(
                "hardware has no AIC BF16 terminal Fixpipe service")
        if rows != mechanism["rows"] or columns != mechanism["columns"]:
            raise ValueError(
                "AIC BF16 terminal Fixpipe tile is outside the measured domain")
        return mechanism["completion_ns"]

    def aic_fixpipe_bf16_finite_stream_time_ns(
            self, rows, columns, active_aic_groups, tiles_per_worker):
        """Return the slowest-AIC finite BF16 FixPipe stream completion."""

        for name, value in (
                ("rows", rows), ("columns", columns),
                ("active_aic_groups", active_aic_groups),
                ("tiles_per_worker", tiles_per_worker)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AIC BF16 finite Fixpipe {name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"AIC BF16 finite Fixpipe {name} must be positive")
        finite = self.aic_fixpipe_bf16_finite_stream_service
        long_stream = self.aic_fixpipe_bf16_long_stream_service
        if finite is None:
            raise ValueError(
                "hardware has no AIC BF16 finite Fixpipe stream service")
        signature = (rows, columns, active_aic_groups)
        if (finite["minimum_tiles_per_worker"] <= tiles_per_worker
                <= finite["maximum_tiles_per_worker"]):
            if signature not in finite["cells"]:
                raise ValueError(
                    "AIC BF16 finite Fixpipe geometry is outside the measured "
                    "domain")
            mechanism = finite
            runs = mechanism["cells"][signature]
        elif (long_stream is not None
                and long_stream["minimum_tiles_per_worker"]
                    < tiles_per_worker
                    <= long_stream["maximum_tiles_per_worker"]
                and signature == (
                    long_stream["rows"], long_stream["columns"],
                    long_stream["active_aic_groups"])):
            mechanism = long_stream
            runs = mechanism["run_affine_envelopes"]
        else:
            raise ValueError(
                "AIC BF16 finite Fixpipe tile count or geometry is outside "
                "the measured affine domains")
        short_tiles = mechanism["short_tiles_per_worker"]
        span_tiles = mechanism["long_tiles_per_worker"] - short_tiles
        run_cycles = []
        for envelope in runs:
            block_cycles = []
            for _, short_cycles, long_cycles in envelope:
                slope = (long_cycles - short_cycles) / span_tiles
                block_cycles.append(
                    short_cycles + (tiles_per_worker - short_tiles) * slope)
            run_cycles.append(max(block_cycles))
        median_cycles = sorted(run_cycles)[len(run_cycles) // 2]
        return median_cycles / mechanism["aic_clock_hz"] * 1e9

    def _fia_block_reduce_time_ns(
            self, mechanism, operation,
            row_chunk_round, column_count, column_alignment):
        """Return one source-matched FIA row-reduction service time."""

        for name, value in (
                ("row_chunk_round", row_chunk_round),
                ("column_count", column_count),
                ("column_alignment", column_alignment)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"FIA {operation} {name} must be an integer")
        if mechanism is None:
            raise ValueError(
                f"hardware has no FIA {operation} source service")
        row_granularity = mechanism["row_granularity"]
        if (row_chunk_round <= 0
                or row_chunk_round > mechanism["maximum_row_chunk"]
                or row_chunk_round % row_granularity):
            raise ValueError(
                f"FIA {operation} row chunk is outside the measured "
                "8-row-aligned domain")
        if not 1 <= column_count <= mechanism["maximum_column_count"]:
            raise ValueError(
                f"FIA {operation} column count is outside the measured "
                "domain")
        if column_alignment not in (16, 32):
            raise ValueError(
                f"FIA {operation} column alignment must be 16 or 32")
        column_round = (
            math.ceil(column_count / column_alignment) * column_alignment)
        if column_count == 512 and row_chunk_round not in (8, 16):
            raise ValueError(
                f"FIA 512-column {operation} requires an exact measured "
                "row chunk of 8 or 16")
        if (row_chunk_round * column_round
                > mechanism["maximum_source_elements"]):
            raise ValueError(
                f"FIA {operation} geometry exceeds the measured UB tile")

        if column_count == 256:
            coefficients = mechanism["special_256_coefficients_ns"]
            service_ns = (
                coefficients["base"]
                + coefficients["row"] * row_chunk_round)
        elif column_count == 512:
            coefficients = mechanism["special_512_coefficients_ns"]
            service_ns = (
                coefficients["base"]
                + coefficients["row"] * row_chunk_round)
        else:
            coefficients = mechanism["tail_coefficients_ns"]
            short_column = int(column_count < 64)
            additional_full_segment = max(column_count // 64 - 1, 0)
            residual_segment = int(
                column_count >= 64 and column_count % 64 != 0)
            masked_alignment = int(column_alignment == 32)
            service_ns = (
                coefficients["base"]
                + coefficients["row"] * row_chunk_round
                + coefficients["column_round"] * column_round
                + coefficients["row_column_round"]
                * row_chunk_round * column_round
                + coefficients["short_column"] * short_column
                + coefficients["row_short_column"]
                * row_chunk_round * short_column
                + coefficients["additional_full_segment"]
                * additional_full_segment
                + coefficients["row_additional_full_segment"]
                * row_chunk_round * additional_full_segment
                + coefficients["residual_segment"] * residual_segment
                + coefficients["row_residual_segment"]
                * row_chunk_round * residual_segment
                + coefficients["masked_alignment"] * masked_alignment
                + coefficients["row_masked_alignment"]
                * row_chunk_round * masked_alignment
            )
        if not math.isfinite(service_ns) or service_ns <= 0:
            raise ValueError(
                f"FIA {operation} source service produced a non-positive "
                "duration")
        return service_ns

    def fia_block_reduce_sum_time_ns(
            self, row_chunk_round, column_count, column_alignment):
        """Return one source-matched CalcLocalRowSum service time."""

        return self._fia_block_reduce_time_ns(
            self.fia_block_reduce_sum_service,
            "BlockReduceSum",
            row_chunk_round,
            column_count,
            column_alignment,
        )

    def fia_block_reduce_max_time_ns(
            self, row_chunk_round, column_count, column_alignment):
        """Return one source-matched CalcLocalRowMax service time."""

        return self._fia_block_reduce_time_ns(
            self.fia_block_reduce_max_service,
            "BlockReduceMax",
            row_chunk_round,
            column_count,
            column_alignment,
        )

    def fia_brcb_time_ns(self, row_chunk_round):
        """Return one source-matched CalcExp Brcb plus PIPE_V service."""

        if (isinstance(row_chunk_round, bool)
                or not isinstance(row_chunk_round, int)):
            raise TypeError("FIA Brcb row chunk must be an integer")
        mechanism = self.fia_brcb_service
        if mechanism is None:
            raise ValueError("hardware has no FIA Brcb source service")
        if (row_chunk_round < mechanism["minimum_row_chunk"]
                or row_chunk_round > mechanism["maximum_row_chunk"]
                or row_chunk_round % mechanism["row_granularity"]):
            raise ValueError(
                "FIA Brcb row chunk is outside the measured 8-row-aligned "
                "domain")
        return (
            mechanism["startup_ns"]
            + row_chunk_round * mechanism["per_row_ns"])

    def has_empty_kernel_duration_for_task(self, launch_task_kind):
        if launch_task_kind == "generic":
            return (
                self.empty_kernel_duration_lut is not None
                or self.empty_kernel_duration_ns is not None
                or self.launch_latency_model is not None)
        elif launch_task_kind == "aic":
            return self.aic_empty_kernel_duration_lut is not None
        elif launch_task_kind == "aiv":
            return self.aiv_empty_kernel_duration_lut is not None
        elif launch_task_kind == "mixed":
            return self.mixed_empty_kernel_duration_lut is not None
        else:
            raise ValueError(
                f"unknown launch task kind: {launch_task_kind!r}")

    def has_task_duration_floor_for_task(self, launch_task_kind):
        if launch_task_kind == "aiv":
            return (
                self.aiv_empty_kernel_duration_lut is not None
                or self.aiv_nonempty_task_latency_model is not None
                or self.aiv_memory_task_latency_model is not None)
        if launch_task_kind == "mix_aiv":
            return self.mix_aiv_nonempty_task_latency_model is not None
        return self.has_empty_kernel_duration_for_task(launch_task_kind)

    def runtime_task_class_floor_supports(
            self, task_class, elements, grid_size=None, block_size=None):
        if not isinstance(task_class, str) or not task_class:
            raise TypeError("runtime task class must be a non-empty string")
        if (isinstance(elements, bool) or not isinstance(elements, int)
                or elements <= 0):
            raise ValueError(
                "runtime task-class element count must be positive")
        if task_class not in self.runtime_task_latency_models:
            return False
        model = self.runtime_task_latency_models[task_class]
        if model["model"] == (
                "warm_bf16_binary_tensor_iterator_task_floor_curve_v1"):
            # A floor only transfers at the geometry it was measured at:
            # block size fixes per-CTA occupancy.
            if (block_size is not None
                    and block_size != model["probe_block_threads"]):
                return False
            curve = model["floor_ns_by_elements"]
            return min(curve) <= elements <= max(curve)
        if model["model"] == (
                "warm_bf16_binary_tensor_iterator_task_floor_v1"):
            return (
                model["minimum_elements"]
                <= elements <= model["maximum_elements"])
        elif model["model"] == (
                "warm_pytorch_custom_op_scheduler_v1"):
            if grid_size is None or block_size is None:
                return False
            if (isinstance(grid_size, bool) or not isinstance(grid_size, int)
                    or isinstance(block_size, bool)
                    or not isinstance(block_size, int)
                    or grid_size <= 0 or block_size <= 0):
                raise ValueError(
                    "runtime custom-op task geometry must be positive")
            return (
                model["minimum_grid_size"]
                <= grid_size <= model["maximum_grid_size"]
                and model["minimum_block_size"]
                <= block_size <= model["maximum_block_size"])
        else:
            raise ValueError(
                "unsupported runtime task-class model: "
                f"{model['model']!r}")

    def runtime_task_class_duration_ns(
            self, task_class, elements, grid_size=None, block_size=None):
        if not self.runtime_task_class_floor_supports(
                task_class, elements, grid_size, block_size):
            raise ValueError(
                "runtime task-class floor is unavailable outside its "
                "measured support")
        model = self.runtime_task_latency_models[task_class]
        if model["model"] == (
                "warm_bf16_binary_tensor_iterator_task_floor_curve_v1"):
            # The curve is non-decreasing, so the sample at the largest
            # measured size at or below this request bounds it from below.
            curve = model["floor_ns_by_elements"]
            return curve[max(key for key in curve if key <= elements)]
        if model["model"] == (
                "warm_bf16_binary_tensor_iterator_task_floor_v1"):
            return model["duration_ns"]
        elif model["model"] == (
                "warm_pytorch_custom_op_scheduler_v1"):
            waves_per_cta = math.ceil(block_size / self.parallel_width)
            scheduling_rounds = math.ceil(
                grid_size / self.num_compute_units)
            return (
                model["submit_ns"]
                + grid_size * model["cta_descriptor_ns"]
                + scheduling_rounds * waves_per_cta
                * model["wave_round_ns"])
        else:
            raise ValueError(
                "unsupported runtime task-class model: "
                f"{model['model']!r}")

    def custom_all_reduce_service_ns(self, participants, payload_bytes):
        """Return the measured custom all-reduce service, or None.

        None means pinned vLLM would not select the custom kernel for this
        payload (at or above ``max_payload_bytes``) or the card publishes no
        measurement.  A published card that lacks the requested participant
        count fails closed rather than borrowing another group size.
        """

        model = self.custom_all_reduce_service
        if model is None or payload_bytes >= model["max_payload_bytes"]:
            return None
        if participants not in model["tables"]:
            raise KeyError(
                "custom all-reduce service has no measurement for "
                f"{participants} participants")
        rows = model["tables"][participants]
        resolved = rows[0][1]
        for payload, service in rows:
            if payload <= payload_bytes:
                resolved = service
        return resolved

    def cooperative_grid_sync_ns(self, grid):
        """Measured in-graph cooperative grid barrier at the largest sampled grid <= grid."""

        table = self.cooperative_grid_sync_service
        if table is None:
            raise NotImplementedError(
                f"{self.hardware_id} publishes no cooperative grid-sync service")
        resolved = None
        for sampled, service in table:
            if sampled <= grid:
                resolved = service
        if resolved is None:
            raise ValueError(
                f"grid {grid} is below the smallest sampled grid-sync grid")
        return resolved

    def stream_memcpy_service_ns(self, direction, payload_bytes):
        """Measured eager stream memcpy cost; fails closed outside its table."""

        model = self.stream_memcpy_service
        if model is None:
            raise NotImplementedError(
                f"{self.hardware_id} publishes no stream memcpy service")
        if direction not in model["tables"]:
            raise KeyError(f"stream memcpy service has no {direction!r}")
        if payload_bytes >= model["max_payload_bytes"]:
            raise ValueError(
                "stream memcpy service is measured only below "
                f"{model['max_payload_bytes']} bytes, got {payload_bytes}")
        rows = model["tables"][direction]
        resolved = rows[0][1]
        for size, service in rows:
            if size <= payload_bytes:
                resolved = service
        return resolved

    def side_stream_fork_graph_node_increment_s(self):
        """Measured added boundary per node of a side-stream-forking graph."""

        if self.cuda_graph_side_stream_fork_node_increment_s is None:
            raise NotImplementedError(
                f"{self.hardware_id} publishes no side-stream-fork graph "
                "node boundary")
        return self.cuda_graph_side_stream_fork_node_increment_s

    def split_merge_chunk_s(self):
        """Measured per-chunk interval of a serial split-KV merge walk."""

        if self.split_merge_chunk_service_s is None:
            raise NotImplementedError(
                f"{self.hardware_id} publishes no split-KV merge chunk "
                "service")
        return self.split_merge_chunk_service_s

    def sglang_custom_all_reduce_service_ns(self, participants, payload_bytes):
        """Measured SGLang CustomAllReduceV2 service, or None above its max size.

        Unlike the vLLM table, a card without the SGLang measurement fails
        closed: the SGLang stack must not borrow the vLLM kernel's service.
        """

        model = self.sglang_custom_all_reduce_service
        if model is None:
            raise NotImplementedError(
                f"{self.hardware_id} publishes no SGLang custom all-reduce "
                "service (interconnect.sglang_custom_all_reduce_service)")
        if payload_bytes >= model["max_payload_bytes"]:
            return None
        if participants not in model["tables"]:
            raise KeyError(
                "SGLang custom all-reduce service has no measurement for "
                f"{participants} participants")
        rows = model["tables"][participants]
        resolved = rows[0][1]
        for payload, service in rows:
            if payload <= payload_bytes:
                resolved = service
        return resolved

    def sglang_flashinfer_allreduce_fusion_ns(self, participants,
                                              payload_bytes):
        """Measured flashinfer all-reduce + residual RMSNorm fusion service.

        Fails closed outside the measured payload range and on cards
        without the measurement.
        """

        model = self.sglang_flashinfer_allreduce_fusion_service
        if model is None:
            raise NotImplementedError(
                f"{self.hardware_id} publishes no flashinfer all-reduce "
                "fusion service")
        if payload_bytes >= model["max_payload_bytes"]:
            raise ValueError(
                "flashinfer all-reduce fusion is measured only below "
                f"{model['max_payload_bytes']} bytes, got {payload_bytes}")
        if participants not in model["tables"]:
            raise KeyError(
                "flashinfer all-reduce fusion service has no measurement "
                f"for {participants} participants")
        rows = model["tables"][participants]
        resolved = rows[0][1]
        for payload, service in rows:
            if payload <= payload_bytes:
                resolved = service
        return resolved

    def cold_async_copy_ready_s(self, producers, path, payload_bytes):
        """Return first-burst completion from a copy-only readiness probe.

        Retain completion floors of measured cohorts no larger than the
        demand. Taking their maximum preserves nondecreasing contention
        without interpolating a producer-count curve or introducing a faster
        response at a table boundary. This is a finite initial burst, not a
        replacement steady-stream resource rate.
        """
        model = self.async_copy_cold_ready
        if model is None:
            raise NotImplementedError("cold async copy readiness has no mechanism evidence")
        if path not in model["profiles"]:
            raise ValueError("unknown cold async copy path")
        if (type(payload_bytes) is not int
                or not model["minimum_payload_bytes"] <= payload_bytes <= model["maximum_payload_bytes"]):
            raise ValueError("cold async copy payload exceeds measured domain")
        rows = model["profiles"][path]
        if (type(producers) is not int or producers < rows[0]["producers"]
                or producers > rows[-1]["producers"]):
            raise ValueError("cold async copy producer count exceeds measured domain")
        return max(row["latency_ns"] * 1e-9
                   + payload_bytes * row["producers"] / row["aggregate_bandwidth_bytes_s"]
                   for row in rows if row["producers"] <= producers)

    def cta_dispatch_cost_ns(self, grid_size):
        """Return the measured dispatch cost of this launch's CTA count.

        Zero when the card publishes no measurement, so such cards are
        unaffected.  The table is non-decreasing, so stepping down to the
        largest sampled grid at or below the request is a lower bound and
        chooses no interpolation shape.
        """

        model = self.cta_dispatch_cost
        if model is None:
            return 0.0
        if isinstance(grid_size, bool) or not isinstance(grid_size, int):
            raise TypeError("CTA dispatch cost needs an integer grid")
        if grid_size <= 0:
            raise ValueError("CTA dispatch cost needs a positive grid")
        table = model["cost_ns_by_grid"]
        return table[max(key for key in table if key <= grid_size)]

    def runtime_task_class_is_launch_scheduler(self, task_class):
        if not isinstance(task_class, str) or not task_class:
            raise TypeError("runtime task class must be a non-empty string")
        if task_class not in self.runtime_task_latency_models:
            return False
        model = self.runtime_task_latency_models[task_class]
        return model["model"] == "warm_pytorch_custom_op_scheduler_v1"

    def empty_kernel_duration_for_task(
            self, launch_task_kind, grid_size, block_size,
            launch_grid=None, launch_block=None):
        if launch_task_kind == "generic":
            return self.empty_kernel_duration_for_launch(
                grid_size, block_size, launch_grid, launch_block)
        elif launch_task_kind in ("aic", "aiv", "mixed"):
            if block_size != 1:
                raise ValueError(
                    f"{launch_task_kind} task launch must use the scalar "
                    f"block-size placeholder one, got {block_size}")
            if launch_task_kind == "aic":
                return self.aic_empty_kernel_duration_ns(grid_size)
            elif launch_task_kind == "aiv":
                return self.aiv_empty_kernel_duration_ns(grid_size)
            elif launch_task_kind == "mixed":
                return self.mixed_empty_kernel_duration_ns(grid_size)
            else:
                raise AssertionError(
                    f"unhandled launch task kind: {launch_task_kind!r}")
        else:
            raise ValueError(
                f"unknown launch task kind: {launch_task_kind!r}")

    def task_duration_floor_for_task(
            self, launch_task_kind, grid_size, block_size,
            launch_grid=None, launch_block=None, *,
            task_duration_floor_contract=None, sm_freq_hz=None,
            runtime_task_class=None, runtime_task_elements=None):
        floors = []
        if launch_task_kind == "aiv":
            if block_size != 1:
                raise ValueError(
                    "aiv task launch must use the scalar block-size "
                    f"placeholder one, got {block_size}")
            if self.aiv_empty_kernel_duration_lut is not None:
                floors.append(self.aiv_empty_kernel_duration_ns(grid_size))
            if task_duration_floor_contract not in (
                None, AIV_COMPLETE_TASK_FLOOR_ONLY,
                AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
                AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM):
                raise ValueError(
                    "unknown AIV task-duration floor contract: "
                    f"{task_duration_floor_contract!r}")
            if task_duration_floor_contract not in (
                    AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
                    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM):
                if self.aiv_nonempty_task_latency_model is not None:
                    floors.append(
                        self.aiv_nonempty_task_duration_ns(grid_size))
                if self.aiv_memory_task_latency_model is not None:
                    floors.append(
                        self.aiv_memory_task_duration_ns(grid_size))
            if not floors:
                raise ValueError("hardware has no AIV task duration floor")
        elif launch_task_kind == "mix_aiv":
            if block_size != 1:
                raise ValueError(
                    "MIX_AIV task launch must use the scalar block-size "
                    f"placeholder one, got {block_size}")
            if task_duration_floor_contract is not None:
                raise ValueError(
                    "MIX_AIV task does not support a separate "
                    "task-duration floor contract")
            if self.mix_aiv_nonempty_task_latency_model is None:
                raise ValueError(
                    "hardware has no MIX_AIV task duration floor")
            floors.append(
                self.mix_aiv_nonempty_task_duration_ns(grid_size))
        elif launch_task_kind in ("generic", "aic", "mixed"):
            # A measured framework task class is a more specific complete-task
            # floor than the hardware-wide synthetic nonempty-task model.  It
            # replaces that generic floor for its bounded source/runtime
            # domain; the analytical body still competes under max below.
            if runtime_task_class is None:
                floors.append(self.empty_kernel_duration_for_task(
                    launch_task_kind, grid_size, block_size,
                    launch_grid, launch_block))
        else:
            raise ValueError(
                f"unknown launch task kind: {launch_task_kind!r}")
        if (task_duration_floor_contract is not None
                and launch_task_kind != "aiv"):
            if launch_task_kind != "generic":
                raise ValueError(
                    "memory-dependency task floor requires a generic GPU "
                    "launch task")
            if sm_freq_hz is None:
                raise ValueError(
                    "memory-dependency task floor requires sm_freq_hz")
            if (task_duration_floor_contract
                    == GPU_SOURCE_PIPELINE_OUTER_ENVELOPE):
                self.fa2_source_task_outer_envelope_ns(grid_size, block_size)
            elif (task_duration_floor_contract
                    == WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR):
                floors.append(self.memory_dependency_task_duration_ns(
                    task_duration_floor_contract,
                    grid_size,
                    block_size,
                    launch_grid,
                    launch_block,
                    sm_freq_hz,
                ))
            else:
                floors.append(self.tensor_iterator_copy_task_duration_ns(
                    task_duration_floor_contract,
                    grid_size,
                    block_size,
                    launch_grid,
                    launch_block,
                    sm_freq_hz,
                ))
        if ((runtime_task_class is None)
                != (runtime_task_elements is None)):
            raise ValueError(
                "runtime task class and element count must be supplied "
                "together")
        if runtime_task_class is not None:
            if launch_task_kind != "generic":
                raise ValueError(
                    "runtime task-class floor requires a generic GPU task")
            floors.append(self.runtime_task_class_duration_ns(
                runtime_task_class, runtime_task_elements,
                grid_size, block_size))
        return max(floors)

    def resource(self, name):
        """Get a resource by name."""
        return self._res_map[name]

    def resource_by_role(self, role):
        """Get a resource by semantic role (e.g. 'l2_cache', 'dram')."""
        return self._role_map[role]

    def has_memory_access_class_service(self, service):
        """Return whether any exact signature exists for a typed service."""

        if not isinstance(service, str) or not service.strip():
            raise ValueError(
                "memory access-class service name must be a non-empty string")
        normalized = service.strip()
        return any(
            candidate.service == normalized
            for candidate in self.memory_access_class_services)

    def memory_access_class_service(
            self, service, access_kind, ingress_resource_role, signature):
        """Return one exact typed access-class capability or fail loud."""

        if not isinstance(signature, MemoryAccessClassSignature):
            raise TypeError(
                "memory access-class lookup requires an exact signature")
        identity = (
            service, access_kind, ingress_resource_role, signature)
        matches = tuple(
            candidate for candidate in self.memory_access_class_services
            if (
                candidate.service,
                candidate.access_kind,
                candidate.ingress_resource_role,
                candidate.signature,
            ) == identity)
        if len(matches) != 1:
            raise ValueError(
                "memory access-class lookup requires exactly one hardware "
                f"capability, found {len(matches)} for {identity!r}")
        return matches[0]

    def tma_balanced_finite_read_completion_s(self, *, threads_per_cta,
            a_bytes, b_bytes, slots_per_operand, producers, read_bytes):
        """Finite unique-read lower bound, or None outside measured physical scope.

        Optimistic cohort normalization ignores address ordering and all cache
        reuse beyond the caller's compulsory bytes. Fill/drain is included.
        """
        if any(type(v) is not int or v <= 0 for v in (threads_per_cta, a_bytes,
                b_bytes, slots_per_operand, producers, read_bytes)):
            raise ValueError("finite TMA geometry and bytes must be positive integers")
        matches = []
        for profile in self.dram_finite_tma_read_stream_profiles:
            if (threads_per_cta not in profile["threads_per_cta"] or a_bytes != b_bytes
                    or a_bytes not in profile["balanced_operand_bytes"]
                    or slots_per_operand != profile["native_slots_per_operand"]
                    or 2 * a_bytes * slots_per_operand > profile["maximum_shared_bytes"]
                    or not profile["minimum_producers"] <= producers <= profile["maximum_producers"]):
                continue
            length = read_bytes / producers
            lengths = profile["lengths_bytes_per_cta"]
            if not lengths[0] <= length <= lengths[-1]:
                continue
            grid_key = next(g for g in profile["grids"] if g >= producers)
            length_key = next(n for n in lengths if n >= length)
            matches.append(read_bytes / (profile["dram_read_capacity_bytes_per_s"]
                * profile["ramp"][grid_key, length_key]))
        if len(matches) > 1:
            raise ValueError("ambiguous balanced TMA completion scope")
        return matches[0] if matches else None

    def dram_capacity_for_access(
            self, access_kind, *, active_units, sm_freq_hz, dram_freq_hz):
        """Return the configured shared-DRAM capacity for one direction."""

        if access_kind not in ("read", "write"):
            raise ValueError(
                "DRAM access kind must be 'read' or 'write', got "
                f"{access_kind!r}")
        resource = self.resource_by_role("dram")
        common_capacity = resource.effective_capacity(
            active_units,
            sm_freq_hz=sm_freq_hz,
            dram_freq_hz=dram_freq_hz,
        )
        if self.dram_arbitration_mode == "shared_common_capacity":
            return common_capacity
        if self.dram_arbitration_mode in (
                "shared_directional_capacity",
                "independent_directional_capacity"):
            if access_kind == "read":
                capacity_per_clk = self.dram_read_capacity_per_clk
            elif access_kind == "write":
                capacity_per_clk = self.dram_write_capacity_per_clk
            else:
                raise AssertionError(
                    f"unhandled validated access kind: {access_kind!r}")
            return capacity_per_clk * dram_freq_hz
        raise ValueError(
            "hardware does not define a supported DRAM arbitration mode")

    def __repr__(self):
        res_names = ", ".join(r.name for r in self.resources)
        grp_names = ", ".join(g.name for g in self.resource_groups)
        return f"HWSpec(resources=[{res_names}], groups=[{grp_names}])"
