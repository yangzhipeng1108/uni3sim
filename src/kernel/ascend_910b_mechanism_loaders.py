"""Load independent Ascend AIC mechanism references without changing HWSpec values.

The exact payload digest binds recorded execution labels and process identities.
Validate their structure here; keep their historical values in the evidence owner.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from src.kernel.hardware_reference_validation import require_exact_fields, linear_interpolate


AIC_MTE2_ND2NZ_33COPY_GROUP_SERVICE = (
    "aic_mte2_nd2nz_33copy_large_stride_group_completion_24_workers")


AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_nd2nz_33copy_large_stride_group_completion_cann_8_5_1.json")


AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE_SHA256 = (
    "924f0a974ebe50280a7843fc4065a7a5610bdec2ed01ca4f037be056a02339f6")


AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_deterministic_split_k_source_queue_completion_cann_8_5_1.json")


AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE_SHA256 = (
    "e5ccb4906be411a8ad1be8cff5156031ba09b6616f6f58c4df3eda7ac791c4fa")


AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_deterministic_split_k_collective_barrier_cann_8_5_1.json")


AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE_SHA256 = (
    "431abf9510a8a66c992c7dd8fd70a36c3583c67fb879f4b5d6e653b72cfe2141")


AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_nd2nz_large_contiguous_resident_cann_8_5_1.json")


AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE_SHA256 = (
    "202229a047b1e9adb863e2d68be19e0e927bb203d2cbe4a4024282a66be6fc05")


AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_large_resident_descriptor_cann_8_5_1.json")


AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE_SHA256 = (
    "58608d2361243dd60b3042cc5ec8174a12798163753be274678c1281d4bd617b")


AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_k14336_source_queue_cann_8_5_1.json")


AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE_SHA256 = (
    "61a9186748d0be26e5cbe90b4dbcb39cdabc7f6d87e879fa086d19deed436fe2")


AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_operand_pair_overlap_expansion_cann_8_5_1.json")


AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE_SHA256 = (
    "8b93ef11b057fbf9409302a73da45d54cef9bcaa07f986906d75ea1fc9645b81")


AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_operand_pair_finite_overlap_cann_8_5_1.json")


AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE_SHA256 = (
    "ea2534e05047a38e1aa02f798add800654e136404c27ba11fa7e15c8f5582d42")


AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_mixed_ring_long_overlap_cann_8_5_1.json")


AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE_SHA256 = (
    "f685a4ca1c9a234bed4a1c0e1094c3080f624933a1148c930c28313537f67abf")


AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_cold_operand_pair_count_cann_8_5_1.json")


AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE_SHA256 = (
    "ff6feabc67f58e0b4aa32cc63de29c3ea0e17af6ba714fc14113d90357064736")


AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_cold_m16_operand_pair_count_cann_8_5_1.json")


AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE_SHA256 = (
    "eaf0f88555e09d4de6580adb1b96adb14a7360c11f0f383b8736662b294b1810")


AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_cold_n_major_b_stride_cann_8_5_1.json")


AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE_SHA256 = (
    "ed3e2f00a31a87a99869db209638788cddea1a45a2860d6988c81c67f43531d3")


AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_cold_legacy_k14336_stride_cann_8_5_1.json")


AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE_SHA256 = (
    "9c16dd8a863ca55a50f91c55e9bfa29ab4be969495740afd0afb09b771121446")


AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_cold_v3_k1536_n5120_stride_cann_8_5_1.json")


AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE_SHA256 = (
    "d69a024f81418047cb0090bbff5302656048f0c1d33656b82a027f62965f5da8")


AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_cold_small_m_stride_cann_8_5_1.json")


AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE_SHA256 = (
    "b9bf3f92edb7e91cf02b8a0eccbd050fc395b99a7f814dae841d5547981c85a0")


AIC_MTE2_FINITE_STREAM_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_nd2nz_finite_stream_completion_cann_8_5_1.json")


AIC_MTE2_FINITE_STREAM_REFERENCE_SHA256 = (
    "a14537926184610e898ce440ebadbcb755f2496453428d4ef7ee9cdc89d14aca")


AIC_MTE2_GEMM_SOURCE_ORDER_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_source_order_completion_cann_8_5_1.json")


AIC_MTE2_GEMM_SOURCE_ORDER_COMPLETION_REFERENCE_SHA256 = (
    "4da29ebc88c77db16fed4b985ed2d5d334a5ba7e8b9b452c7ff3cced53726dff")


AIC_MTE2_GEMM_K13824_SOURCE_ORDER_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_k13824_source_order_completion_cann_8_5_1.json")


AIC_MTE2_GEMM_K13824_SOURCE_ORDER_COMPLETION_REFERENCE_SHA256 = (
    "33aee72df0bf91efa5e6aed46d3d44b626877b37c3b9f939d1c297406d887a8b")


AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v3_split_k_resident_tile_completion_cann_8_5_1.json")


AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE_SHA256 = (
    "11ea1ff31c95df56847493f531c1ae85a45dd1417dd780d79798f80d16a8aadc")


AIC_MATMUL_V2_FINITE_INGRESS_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v2_finite_ingress_completion_cann_8_5_1.json")


AIC_MATMUL_V2_FINITE_INGRESS_COMPLETION_REFERENCE_SHA256 = (
    "69c68950e53910b9177adf599742f94b629d7377607f67af4777be46741748d7")


AIC_MATMUL_V2_K14336_M128_N128_FINITE_INGRESS_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v2_k14336_m128_n128_finite_ingress_completion_cann_8_5_1.json")


AIC_MATMUL_V2_K14336_M128_N128_FINITE_INGRESS_COMPLETION_REFERENCE_SHA256 = (
    "f09daba3554aa15aaec5284cafc23081ba5c969221c7ed2402e67545054381e2")


AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_k1536_short_pipeline_completion_cann_8_5_1.json")


AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE_SHA256 = (
    "6f29bc299650591283e2bcd07dcdcb2229406d27102db9ec2c9050a00dae85a2")


AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v2_k5120_finite_transition_cann_8_5_1.json")


AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE_SHA256 = (
    "e23ad44e82e8b3b2a27341e301f4dc07067f9e1c9144276ffd26b61131ee7152")


AIC_MATMUL_V2_K8192_FINITE_TRANSITION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v2_k8192_source_transition_cann_8_5_1.json")


AIC_MATMUL_V2_K8192_FINITE_TRANSITION_REFERENCE_SHA256 = (
    "c45d887115655d89b1813b58811e993add32e3e65b72aba60f555b902840877c")


AIC_MATMUL_V2_K13824_FINITE_TRANSITION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v2_k13824_source_transition_cann_8_5_1.json")


AIC_MATMUL_V2_K13824_FINITE_TRANSITION_REFERENCE_SHA256 = (
    "b58fba105b7417af0bc5b6dbfedb25769673eb3ade766d3f822f356df6641033")


AIC_GEMM_FINITE_TRANSITION_SIGNATURE_FIELDS = (
    "dtype", "operand_buffer_depth", "active_aic_groups", "m_groups",
    "n_groups", "base_tile_m", "base_tile_n", "base_tile_k", "a_rows",
    "a_columns", "b_rows", "b_columns", "source_row_stride_elements",
    "output_tiles_per_worker", "k_slabs",
    "source_descriptors_per_worker", "completion_pairs_per_worker",
)


AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS = (
    "dtype", "active_aic_groups", "operand_buffer_depth",
    "base_tile_m", "base_tile_n", "base_tile_k", "mte1_base_k",
    "a_rows", "a_columns", "a_source_row_stride_elements", "b_rows",
    "b_columns", "b_source_row_stride_elements", "m_groups", "n_groups",
    "output_tiles_per_worker", "k_slabs", "mte2_programs_per_worker",
    "mte1_pairs_per_worker",
)


AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_m2220_persistent_issue_cann_8_5_1.json")


AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE_SHA256 = (
    "b527ee44daa4c62cdfbd81c974df7f0a10da5b1c388429db09e324bf37ae01a2")


AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v2_dynamic_m_tail_control_cann_8_5_1.json")


AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE_SHA256 = (
    "4b04d17b1181b4c53dbc9044d5804b5b499d9cf20f7892d8fc3c1800bae8b241")


AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v2_base_k_lifecycle_cann_8_5_1.json")


AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE_SHA256 = (
    "250dd01f0adbbc0cac52239cac033fcf2ff58b4055a2885e6493139cf753f097")


AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_matmul_v2_small_m_base_k_control_cann_8_5_1.json")


AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE_SHA256 = (
    "b69d3e333b5e02190db10d7eafd0f214fd7b712fa176c728170174ffe0683104")


AIC_MTE2_GEMM_REGION_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_region_completion_cann_8_5_1.json")


AIC_MTE2_GEMM_REGION_COMPLETION_REFERENCE_SHA256 = (
    "d708b4be6ece48be0b004fea9cb3aa95c09426cf00ef966e38ec39db1dd7c8b2")


AIC_MTE2_GEMM_REGION_COMPLETION_8X27_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_region_completion_8x27_cann_8_5_1.json")


AIC_MTE2_GEMM_REGION_COMPLETION_8X27_REFERENCE_SHA256 = (
    "cfaa183860eb87729e4054e0ed53d1bd685d322761c61914ac3d4b96709b3879")


AIC_MTE2_GEMM_REGION_COMPLETION_256X112_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_region_completion_256x112_cann_8_5_1.json")


AIC_MTE2_GEMM_REGION_COMPLETION_256X112_REFERENCE_SHA256 = (
    "5122ac26ea58a55f7d1655283f2a04912608c436377d473a20f4b82125b05ec2")


AIC_MTE2_GEMM_REGION_COMPLETION_16X16_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_region_completion_16x16_cann_8_5_1.json")


AIC_MTE2_GEMM_REGION_COMPLETION_16X16_REFERENCE_SHA256 = (
    "8ae0f0544f0b973f8827ffb9e2a73a38dcce62eef1d09c21abb14171046186fe")


AIC_MTE2_GEMM_REGION_COMPLETION_16X32_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_region_completion_16x32_cann_8_5_1.json")


AIC_MTE2_GEMM_REGION_COMPLETION_16X32_REFERENCE_SHA256 = (
    "3c79dd98f9a49529fc6c28d4c20082d854b8c31e9c90604e0f001684377b7f15")


AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_region_tail_completion_cann_8_5_1.json")


AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE_SHA256 = (
    "0324a2fabfc3e37e6af32f992d3387d31ad287b8254b916f126c8a108f700dfa")


AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_gemm_region_pipeline_completion_cann_8_5_1.json")


AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE_SHA256 = (
    "a218a7fac43cc289071da80dd282623019976192d514429faa3ae486c26cb9b5")


AIC_GROUPED_MATMUL_SHORT_STREAM_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_short_stream_completion_cann_8_5_1.json")


AIC_GROUPED_MATMUL_SHORT_STREAM_COMPLETION_REFERENCE_SHA256 = (
    "310a90977a6afb6506f0f0116bfeb8304c4f7bd19f88eb31caac6fe93f239fd4")


AIC_GROUPED_MATMUL_LONG_STREAM_COMPLETION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_long_stream_completion_cann_8_5_1.json")


AIC_GROUPED_MATMUL_LONG_STREAM_COMPLETION_REFERENCE_SHA256 = (
    "8f42f7645217ddfbe5909518474a3670a3618dde4fdf1672fba59727e66f4210")


AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_k2880_wide_base_tail_completion_cann_8_5_1.json")


AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE_SHA256 = (
    "eaddbe04c13dfd86c0b3c596d182ae38d13f6fb48744e9cb99a8d956a870b68c")


AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_k2880_output_tile_completion_cann_8_5_1.json")


AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE_SHA256 = (
    "38e7c09d53132737a450535fbaf9f60802a76524f27ad2b2cf7f1d6952896007")


AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_step4_worker_sequence_completion_cann_8_5_1.json")


AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE_SHA256 = (
    "04b3189555f09f4002d1ef79b613a2ead63f460c98b1a0efc972c8101e7341a6")


AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_step4_long_worker_sequence_completion_cann_8_5_1.json")


AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE_SHA256 = (
    "52f244f941bb70efdd8f5e742ab0ae1cac229ca2a584ca691a260b6021c99239")


AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_step4_w2_bounded_b_ring_extension_cann_8_5_1.json")


AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE_SHA256 = (
    "c2127d9998ef7a2466bea46e21248d62461a5a473fb301d4c55d15260941349d")


AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_short_source_prelude_control_cann_8_5_1.json")


AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256 = (
    "d6533fbfd357fc906518a0f520778e5e976a8241994aca7aa29571a938331d1b")


AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_short_source_tail_prelude_control_cann_8_5_1.json")


AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE_SHA256 = (
    "f7734f5e3bedef8e9284ec49e69268c6bb5ba205e075e4e9eea63a9a2dbe2ee2")


AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_m32_short_source_prelude_control_cann_8_5_1.json")


AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256 = (
    "b2b05832db0c8edc70983071325d69be57dfc0ab882e152d0759f0d77015e9ba")


AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "grouped_matmul_entry_metadata_pair_cann_8_5_1.json")


AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE_SHA256 = (
    "848f63c56df4f9e1782b69519675a298f64e3ec941d1e469bb4a0e72d5d14692")


AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "matmul_v3_asw_mte2_pair_cann_8_5_1.json")


AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE_SHA256 = (
    "7b2de7f92748c62c6eb5d8bf31debb50fc6701b266c34954a2baf9c508e6c2a9")


AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_mha_finite_drain_cann_8_5_1.json")


AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE_SHA256 = (
    "d90c38edb592bb1e4b7e10aa959d1107b98516d2477e027d1b72dec2b6f5d692")


AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_mha_finite_drain_group16_extended_cann_8_5_1.json")


AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256 = (
    "81e92b4d6443080ec5f88fa2b4f079c89bdf2149de461d12020e48e16395d111")


AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_gqa_finite_drain_cann_8_5_1.json")


AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE_SHA256 = (
    "45c5cf9427f5e80a800f0bdf471865193bccc4cab195b7fdc1d11b5f4e62bb0b")


AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_gqa_finite_drain_group16_count8_cann_8_5_1.json")


AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256 = (
    "fa5bd89ad38a42eb086a63529eb41cf49b08530c5867208b65da1588de2979ad")


AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_gqa_finite_drain_group22_count16_cann_8_5_1.json")


AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE_SHA256 = (
    "41c2f398e6e814e53485572d52e59555123895567404ff179baa0aaeff2e6f0e")


AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_gqa_finite_drain_group16_count16_cann_8_5_1.json")


AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE_SHA256 = (
    "8ce6dc79b9b13040d230056cc48243930d9af1cf27810ddf30752d21a69d1f3c")


AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_gqa_interleaved_finite_drain_cann_8_5_1.json")


AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE_SHA256 = (
    "f2cf65203054bc8222846a1c839219b8853b75bb47e6a1b27bd80137611071b1")


AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_gqa_drained_descriptor_count_law_cann_8_5_1.json")


AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE_SHA256 = (
    "c4ba7356eb36ae67bca29b52f1314d585f6ca723fdcabf7ad149d2a56cc6c518")


AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_fixpipe_bf16_terminal_tile_cann_8_5_1.json")


AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE_SHA256 = (
    "e1c0a639a8d8aa2c24e4ef7d1980c4c6f8ad8a3ae762c350cb7254717a6e4b93")


AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_fixpipe_bf16_finite_stream_cann_8_5_1.json")


AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE_SHA256 = (
    "91d051dc7e1c7ce3d68c62b8d897641177074cd5d425251b16fd880d2d2aa03e")


AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_fixpipe_bf16_long_stream_cann_8_5_1.json")


AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE_SHA256 = (
    "5168d98940bce705070e229e17dd233bf2b4e036a1745eb6cd3ff6a15da7f014")


AIC_MTE2_LEGACY_L1_PAIR_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_legacy_l1_pair_cann_8_5_1.json")


AIC_MTE2_LEGACY_L1_PAIR_REFERENCE_SHA256 = (
    "dd7390395474b6d57a3789ca7901aa68f9a0c77bd6f7d206604a6211b7560f2b")


AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_paged_gqa128_stride_cann_8_5_1.json")


AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE_SHA256 = (
    "f5e824dd334152354c2cb3331180362e7796b6b2a03d3e319bd4d03cca6ee243")

SPLIT_FUSE_MTE2_SHORT_TAIL_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "split_fuse_mte2_short_tail_cann_8_5_1.json")
SPLIT_FUSE_MTE2_SHORT_TAIL_REFERENCE_SHA256 = (
    "3967ce6d45f04d027469ef318a70fa35a473ccccc0ba7e009bccf2ff0bc26398")


AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE = Path(
    "microbench/ascend_910b/references/"
    "aic_mte2_fia_mla_strided_service_cann_8_5_1.json")


AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE_SHA256 = (
    "2138efae094539741279145907c3a236e601b46f2b2434d24a1a1b025aa0883f")


def load_aic_mte2_nd2nz_33copy_group_service(repository_root):
    """Load the retained cold large-stride AIC group-completion service."""

    path = repository_root / AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE_SHA256):
        raise ValueError("AIC MTE2 33-copy group reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    fields = {
        "schema_version", "service", "request_size_unit",
        "service_time_unit", "lookup_policy",
        "production_or_evaluation_duration_consumed",
        "service_time_ns_by_request_size", "mechanism",
        "published_cells", "provenance",
    }
    require_exact_fields("AIC MTE2 33-copy group reference", value, fields)
    expected = {
        "schema_version": 1,
        "service": AIC_MTE2_ND2NZ_33COPY_GROUP_SERVICE,
        "request_size_unit": "source_k_columns",
        "service_time_unit": "ns_per_request_per_aic",
        "lookup_policy": "exact_only",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(
                f"AIC MTE2 33-copy group reference {name} changed")
    mechanism = value["mechanism"]
    expected_mechanism = {
        "dtype": "bf16",
        "rows": 128,
        "copies_per_group": 33,
        "serial_copies_per_group": 1,
        "stream_copies_per_group": 32,
        "active_aic_groups": 24,
        "source_row_stride_elements": 28672,
        "source_row_stride_control_elements": 4096,
        "footprint_bytes": 34359738368,
        "minimum_touched_bytes": 308281344,
        "shared_l2_bytes": 201326592,
        "estimator": (
            "median(serial_ns_per_copy) + 32 * "
            "median(stream_ns_per_copy)"),
    }
    if mechanism != expected_mechanism:
        raise ValueError(
            "AIC MTE2 33-copy group mechanism contract changed")
    if mechanism["minimum_touched_bytes"] <= mechanism["shared_l2_bytes"]:
        raise ValueError(
            "AIC MTE2 33-copy group evidence is not outside shared L2")
    raw_times = value["service_time_ns_by_request_size"]
    if not isinstance(raw_times, dict) or set(raw_times) != {"256", "384"}:
        raise ValueError("AIC MTE2 33-copy group service domain changed")
    published = value["published_cells"]
    if not isinstance(published, dict) or set(published) != {"256", "384"}:
        raise ValueError("AIC MTE2 33-copy group published cells changed")
    normalized_times = {}
    for text_columns in ("256", "384"):
        cell = published[text_columns]
        if (not isinstance(cell, dict)
                or set(cell) != {
                    "serial_ns_per_copy", "stream_ns_per_copy",
                    "service_time_ns"}):
            raise ValueError(
                "AIC MTE2 33-copy group published cell schema changed")
        components = tuple(
            cell[name] for name in (
                "serial_ns_per_copy", "stream_ns_per_copy",
                "service_time_ns"))
        if any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) or float(item) <= 0
                for item in components):
            raise ValueError(
                "AIC MTE2 33-copy group service times must be positive")
        derived = (
            float(cell["serial_ns_per_copy"])
            + 32.0 * float(cell["stream_ns_per_copy"]))
        declared = float(cell["service_time_ns"])
        inline = raw_times[text_columns]
        if (isinstance(inline, bool)
                or not isinstance(inline, (int, float))
                or not math.isclose(
                    float(inline), derived, rel_tol=1e-12, abs_tol=1e-9)
                or not math.isclose(
                    declared, derived, rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError(
                "AIC MTE2 33-copy group derived service time changed")
        normalized_times[int(text_columns)] = derived
    provenance = value["provenance"]
    required_provenance = {
        "hardware_id", "physical_device_id", "software_revision",
        "source_commit", "source_snapshot_sha256", "binary_sha256",
        "raw_path", "raw_sha256", "runner_stderr_path",
        "runner_stderr_sha256", "trial_count", "iterations", "order",
    }
    require_exact_fields(
        "AIC MTE2 33-copy group provenance", provenance,
        required_provenance)
    if (provenance["hardware_id"] != "ascend_910b"
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or provenance["software_revision"] != "cann_8.5.1"
            or provenance["trial_count"] != 4
            or provenance["iterations"] != 2000
            or provenance["order"] != "alternating_forward_reverse"):
        raise ValueError("AIC MTE2 33-copy group provenance changed")
    for name in (
            "source_snapshot_sha256", "binary_sha256", "raw_sha256",
            "runner_stderr_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(
                f"AIC MTE2 33-copy group {name} must be SHA256 hex")
    return {
        **expected,
        **expected_mechanism,
        "service_time_ns_by_request_size": normalized_times,
        "reference_path": str(AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE),
        "reference_sha256": (
            AIC_MTE2_ND2NZ_33COPY_GROUP_REFERENCE_SHA256),
    }


def load_aic_deterministic_split_k_source_queue_service(repository_root):
    """Load the complete deterministic split-K MTE2/MTE1 source span."""

    path = repository_root / AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE_SHA256):
        raise ValueError(
            "AIC deterministic split-K source-queue reference SHA256 "
            "mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC deterministic split-K source-queue reference"
    require_exact_fields(owner, value, {
        "schema_version", "hardware_id", "software_revision", "mechanism",
        "duration_unit", "production_or_evaluation_duration_consumed",
        "source_program", "completion_ns_by_aligned_a_rows",
        "measurement_quality", "exclusions", "source_binding", "provenance",
    })
    expected_identity = {
        "schema_version": (
            "midlayer.ascend_910b."
            "aic_deterministic_split_k_source_queue_completion.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_deterministic_split_k_mte2_mte1_source_queue_completion"),
        "duration_unit": "ns_per_longest_aic_source_program",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected in expected_identity.items():
        if value[name] != expected:
            raise ValueError(f"{owner} {name} changed")
    expected_program = {
        "dtype": "bf16",
        "active_aic_groups": 24,
        "n_slabs": 2,
        "n_tiles_per_slab": 32,
        "k_chunks": 75,
        "full_chunk_workers": 3,
        "full_chunks_per_worker": 4,
        "regular_chunks_per_worker": 3,
        "k_chunk": 384,
        "k_tail": 256,
        "physical_k": 28672,
        "physical_n": 8192,
        "a_descriptor_columns": 384,
        "b_descriptor_rows": 128,
        "b_descriptor_columns": 384,
        "b_source_row_stride": 28672,
        "a1_buffer_slots": 1,
        "b1_buffer_slots": 2,
        "l0a_buffer_slots": 2,
        "l0b_buffer_slots": 2,
        "preload_commands": 4,
        "traversal_order": "n_then_m",
    }
    if value["source_program"] != expected_program:
        raise ValueError(f"{owner} source program changed")
    raw_times = value["completion_ns_by_aligned_a_rows"]
    expected_rows = tuple(range(16, 129, 16))
    if not isinstance(raw_times, dict) or set(raw_times) != {
            str(rows) for rows in expected_rows}:
        raise ValueError(f"{owner} aligned-row domain changed")
    completion_ns = {}
    for rows in expected_rows:
        item = raw_times[str(rows)]
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) or float(item) <= 0.0):
            raise ValueError(f"{owner} completion must be finite and positive")
        completion_ns[rows] = float(item)
    quality = value["measurement_quality"]
    require_exact_fields(owner + " quality", quality, {
        "independent_runs", "trials_per_cell", "statistic",
        "validated_rows", "failed_rows",
        "maximum_deviation_from_cell_median_pct",
    })
    if (quality["independent_runs"] != 3
            or quality["trials_per_cell"] != 8
            or quality["statistic"] != "median"
            or quality["validated_rows"] != 192
            or quality["failed_rows"] != 0
            or quality["maximum_deviation_from_cell_median_pct"] >= 2.0):
        raise ValueError(f"{owner} quality gate changed")
    if value["exclusions"] != [
            "cube", "fixpipe", "workspace", "aiv_reduction", "output"]:
        raise ValueError(f"{owner} exclusion contract changed")
    source_binding = value["source_binding"]
    require_exact_fields(owner + " source binding", source_binding, {
        "cann_source_path", "cann_source_sha256",
        "iterate_order_payload_value", "a1_slot_formula", "b1_slot_formula",
    })
    if source_binding != {
            "cann_source_path": (
                "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/"
                "ascendc/3rd/mat_mul_v3/op_kernel/"
                "mat_mul_deterministic_splitk_kernel.h"),
            "cann_source_sha256": (
                "f88f88f9278f55cf2deeb34ea0c2dce0aac0d1f20525b7ab1bc71ad9f5201605"),
            "iterate_order_payload_value": 1,
            "a1_slot_formula": "depthA1/(stepM*stepKa)",
            "b1_slot_formula": "depthB1/(stepN*stepKb)"}:
        raise ValueError(f"{owner} source binding changed")
    provenance = value["provenance"]
    require_exact_fields(owner + " provenance", provenance, {
        "measurement_baseline_commit", "source_commit",
        "source_snapshot_sha256", "source_sha256", "binary_sha256",
        "physical_device_id", "container", "raw_path", "raw_sha256",
        "runner_stderr_path", "runner_stderr_sha256",
    })
    for name in (
            "source_snapshot_sha256", "binary_sha256", "raw_sha256",
            "runner_stderr_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner} {name} must be SHA256 hex")
    if (not isinstance(provenance["source_sha256"], dict)
            or not provenance["source_sha256"]):
        raise ValueError(f"{owner} source file hashes are missing")
    for digest in provenance["source_sha256"].values():
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner} source file SHA256 is invalid")
    return {
        **expected_identity,
        **expected_program,
        "completion_ns_by_aligned_a_rows": completion_ns,
        "reference_path": str(
            AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE),
        "reference_sha256": (
            AIC_DETERMINISTIC_SPLIT_K_SOURCE_QUEUE_REFERENCE_SHA256),
    }


def load_aic_deterministic_split_k_collective_barrier_service(
        repository_root):
    """Load the source-matched two-slab AIC/AIV collective excess."""

    path = (
        repository_root
        / AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE)
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE_SHA256):
        raise ValueError(
            "AIC deterministic split-K collective reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC deterministic split-K collective reference"
    require_exact_fields(owner, value, {
        "schema_version", "hardware_id", "software_revision", "mechanism",
        "duration_unit", "production_or_evaluation_duration_consumed",
        "source_program", "control_comparison", "measurement_quality",
        "source_binding", "exclusions", "provenance",
    })
    expected_identity = {
        "schema_version": (
            "midlayer.ascend_910b."
            "deterministic_split_k_collective_barrier.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "cann_deterministic_split_k_collective_barrier",
        "duration_unit": (
            "ns_per_two_slab_24aic_48aiv_control_sequence"),
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected in expected_identity.items():
        if value[name] != expected:
            raise ValueError(f"{owner} {name} changed")
    expected_program = {
        "dtype": "bf16",
        "active_aic_groups": 24,
        "active_aiv_workers": 48,
        "n_slabs": 2,
        "scheduler_policy": "l2_split_matmul",
        "sequence": (
            "AIC_PIPE_FIX_flag_to_paired_AIV; all_AIV_wait_then_SyncAll;"
            "PIPE_ALL; all_AIV_PIPE_MTE3_flag_back; next_AIC_wait_then_"
            "SyncAll; PIPE_ALL"),
    }
    if value["source_program"] != expected_program:
        raise ValueError(f"{owner} source program changed")
    comparison = value["control_comparison"]
    require_exact_fields(owner + " comparison", comparison, {
        "source_collective_complete_ns", "same_flag_and_pipe_control_ns",
        "collective_lifecycle_excess_ns", "interpretation",
    })
    timings = {}
    for name in (
            "source_collective_complete_ns",
            "same_flag_and_pipe_control_ns",
            "collective_lifecycle_excess_ns"):
        item = comparison[name]
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item)) or float(item) <= 0.0):
            raise ValueError(f"{owner} {name} must be finite and positive")
        timings[name] = float(item)
    if not math.isclose(
            timings["collective_lifecycle_excess_ns"],
            timings["source_collective_complete_ns"]
            - timings["same_flag_and_pipe_control_ns"],
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{owner} control excess does not conserve spans")
    if (comparison["interpretation"] != (
            "The excess measures only the source-mandatory all-AIV and "
            "all-AIC SyncAll lifecycle for the second N slab. The probe "
            "does not contain production GEMM arithmetic or traffic.")):
        raise ValueError(f"{owner} interpretation changed")
    quality = value["measurement_quality"]
    require_exact_fields(owner + " quality", quality, {
        "independent_runs", "trials_per_cell", "statistic",
        "validated_rows", "failed_rows", "system_counter", "cells",
    })
    if (quality["independent_runs"] != 3
            or quality["trials_per_cell"] != 8
            or quality["statistic"] != "median_of_run_medians"
            or quality["validated_rows"] != 72
            or quality["failed_rows"] != 0):
        raise ValueError(f"{owner} quality gate changed")
    if quality["system_counter"] != {
            "frequency_hz": 50_000_000, "tick_ns": 20.0}:
        raise ValueError(f"{owner} system-counter contract changed")
    if set(quality["cells"]) != {
            "one_slab_source_collective",
            "two_slab_source_collective",
            "two_slab_flags_and_pipe_only"}:
        raise ValueError(f"{owner} cell domain changed")
    source_binding = value["source_binding"]
    require_exact_fields(owner + " source binding", source_binding, {
        "cann_source_path", "cann_source_sha256", "aic_symbols",
        "aiv_symbols",
    })
    if source_binding != {
            "cann_source_path": (
                "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/"
                "ascendc/3rd/mat_mul_v3/op_kernel/"
                "mat_mul_deterministic_splitk_kernel.h"),
            "cann_source_sha256": (
                "f88f88f9278f55cf2deeb34ea0c2dce0aac0d1f20525b7ab1bc71ad9f5201605"),
            "aic_symbols": ["Mc2MatMulMultiCoreSplitKDivideL2cache"],
            "aiv_symbols": ["ReduceKInUbL2cache"]}:
        raise ValueError(f"{owner} source binding changed")
    if value["exclusions"] != [
            "operand traffic", "cube", "fixpipe", "workspace",
            "aiv reduction body", "output storage", "production duration"]:
        raise ValueError(f"{owner} exclusions changed")
    provenance = value["provenance"]
    require_exact_fields(owner + " provenance", provenance, {
        "measurement_baseline_commit", "run_id", "source_snapshot_sha256",
        "source_sha256", "binary_sha256", "raw_path", "raw_sha256",
        "runner_log_path", "runner_log_sha256", "physical_device_id",
        "container",
    })
    for name in (
            "source_snapshot_sha256", "binary_sha256", "raw_sha256",
            "runner_log_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", digest) is None):
            raise ValueError(f"{owner} {name} must be SHA256 hex")
    if (not isinstance(provenance["source_sha256"], dict)
            or not provenance["source_sha256"]):
        raise ValueError(f"{owner} source file hashes are missing")
    for digest in provenance["source_sha256"].values():
        if (not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError(f"{owner} source file SHA256 is invalid")
    if (isinstance(provenance["physical_device_id"], bool)
            or not isinstance(provenance["physical_device_id"], int)
            or provenance["physical_device_id"] < 0
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())):
        raise ValueError(f"{owner} execution provenance changed")
    return {
        **expected_identity,
        **expected_program,
        **timings,
        "reference_path": str(
            AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE),
        "reference_sha256": (
            AIC_DETERMINISTIC_SPLIT_K_COLLECTIVE_BARRIER_REFERENCE_SHA256),
    }


def load_aic_mte2_nd2nz_large_contiguous_service(repository_root):
    """Load measured resident service for large contiguous descriptors."""

    path = repository_root / AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 large-contiguous reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 large-contiguous reference"
    require_exact_fields(
        owner, value,
        {
            "schema_version", "hardware_id", "software_revision",
            "mechanism", "duration_unit",
            "production_or_evaluation_duration_consumed",
            "measurement_scope", "parameters_ns", "measurement_quality",
            "provenance",
        })
    expected = {
        "schema_version": (
            "midlayer.ascend_910b."
            "aic_mte2_nd2nz_large_contiguous_resident_cold.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_l2_dram_to_l1_nd2nz_large_contiguous_descriptor_service"),
        "duration_unit": "ns_per_descriptor_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")

    scope = value["measurement_scope"]
    expected_scope = {
        "dtype": "bf16",
        "resident_source_residency": "l2_resident",
        "cold_source_residency": "dram_streamed",
        "active_aic_groups": [1, 8, 22, 24],
        "cold_active_aic_groups": [22, 24],
        "modes": {"serial_completion": 0, "two_buffer_stream": 1},
        "resident_footprint_bytes": 128 * 1024 * 1024,
        "cold_footprint_bytes": 8 * 1024 * 1024 * 1024,
        "published_l2_capacity_bytes": 192 * 1024 * 1024,
        "trials": 4,
        "iterations": 224,
        "order": "alternating_forward_reverse",
        "statistic": "median",
    }
    if scope != expected_scope:
        raise ValueError(f"{owner}.measurement_scope changed")
    if (scope["resident_footprint_bytes"]
            >= scope["published_l2_capacity_bytes"]):
        raise ValueError(f"{owner} resident footprint exceeds shared L2")
    if (scope["cold_footprint_bytes"]
            <= scope["published_l2_capacity_bytes"]):
        raise ValueError(f"{owner} cold footprint fits shared L2")

    parameter_fields = {
        "rows", "columns", "serial_by_active_aic_group",
        "stream_by_active_aic_group", "cold_serial_by_active_aic_group",
        "cold_stream_by_active_aic_group",
    }
    expected_geometries = {
        (128, 512), (256, 256), (128, 1024), (256, 512),
    }
    parameters = value["parameters_ns"]
    if not isinstance(parameters, list) or len(parameters) != 4:
        raise ValueError(f"{owner}.parameters_ns must contain four cells")
    geometries = {}
    group_count = len(scope["active_aic_groups"])
    cold_group_count = len(scope["cold_active_aic_groups"])
    for index, parameter in enumerate(parameters):
        parameter_owner = f"{owner}.parameters_ns[{index}]"
        if not isinstance(parameter, dict):
            raise TypeError(f"{parameter_owner} must be a mapping")
        require_exact_fields(parameter_owner, parameter, parameter_fields)
        geometry = (parameter["rows"], parameter["columns"])
        if (any(isinstance(item, bool) or not isinstance(item, int)
                or item <= 0 for item in geometry)
                or geometry not in expected_geometries
                or geometry in geometries):
            raise ValueError(f"{parameter_owner} geometry is invalid")
        if geometry[0] * geometry[1] * 2 not in (131_072, 262_144):
            raise ValueError(
                f"{parameter_owner} is not a measured large descriptor")
        services = {}
        for field, expected_count in (
                ("serial_by_active_aic_group", group_count),
                ("stream_by_active_aic_group", group_count),
                ("cold_serial_by_active_aic_group", cold_group_count),
                ("cold_stream_by_active_aic_group", cold_group_count)):
            durations = parameter[field]
            if (not isinstance(durations, list)
                    or len(durations) != expected_count
                    or any(
                        isinstance(duration, bool)
                        or not isinstance(duration, (int, float))
                        or not math.isfinite(float(duration))
                        or float(duration) <= 0.0
                        for duration in durations)):
                raise ValueError(f"{parameter_owner}.{field} is invalid")
            services[field] = tuple(float(duration) for duration in durations)
        geometries[geometry] = services
    if set(geometries) != expected_geometries:
        raise ValueError(f"{owner} exact geometry domain changed")

    quality = value["measurement_quality"]
    require_exact_fields(
        f"{owner}.measurement_quality", quality,
        {
            "validated_rows", "failed_rows", "maximum_trial_deviation_pct",
            "maximum_trial_deviation_pct_active_groups_22_24",
            "maximum_same_byte_geometry_delta_pct_active_groups_22_24",
            "cold_maximum_trial_deviation_pct",
            "cold_maximum_same_byte_geometry_delta_pct",
        })
    if quality["validated_rows"] != 128 or quality["failed_rows"] != 0:
        raise ValueError(f"{owner} row validation did not pass")
    for name in (
            "maximum_trial_deviation_pct",
            "maximum_trial_deviation_pct_active_groups_22_24",
            "maximum_same_byte_geometry_delta_pct_active_groups_22_24",
            "cold_maximum_trial_deviation_pct",
            "cold_maximum_same_byte_geometry_delta_pct"):
        metric = quality[name]
        if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric)) or metric < 0.0):
            raise ValueError(f"{owner}.{name} must be non-negative")

    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {
            "measurement_baseline_commit", "source_commit",
            "source_snapshot_sha256", "source_sha256", "binary_sha256",
            "physical_device_id", "container", "raw_path", "raw_sha256",
            "runner_stderr_path", "runner_stderr_sha256",
            "cold_source_snapshot_sha256", "cold_raw_path",
            "cold_raw_sha256", "cold_runner_stderr_path",
            "cold_runner_stderr_sha256",
        })
    if (isinstance(provenance["physical_device_id"], bool)
            or not isinstance(provenance["physical_device_id"], int)
            or provenance["physical_device_id"] < 0
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())):
        raise ValueError(f"{owner} execution provenance changed")
    for name in ("measurement_baseline_commit", "source_commit"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 40
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be a full Git commit")
    source_sha256 = provenance["source_sha256"]
    expected_source_files = {
        "CMakeLists.txt", "aic_mte2_nd2nz_stride_kernel.cpp",
        "aic_mte2_nd2nz_stride_main.cpp",
        "aic_mte2_nd2nz_stride_runner.py",
    }
    if (not isinstance(source_sha256, dict)
            or set(source_sha256) != expected_source_files):
        raise ValueError(f"{owner}.source_sha256 changed")
    for name in (
            "source_snapshot_sha256", "binary_sha256", "raw_sha256",
            "runner_stderr_sha256", "cold_source_snapshot_sha256",
            "cold_raw_sha256", "cold_runner_stderr_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} must be SHA256 hex")
    if any(
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef"
                   for character in digest)
            for digest in source_sha256.values()):
        raise ValueError(f"{owner}.source_sha256 values must be SHA256 hex")
    for name in (
            "raw_path", "runner_stderr_path", "cold_raw_path",
            "cold_runner_stderr_path"):
        evidence_path = provenance[name]
        if (not isinstance(evidence_path, str)
                or not evidence_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in evidence_path.split("/")):
            raise ValueError(f"{owner}.{name} is outside its owner")
    return {
        "active_aic_groups": tuple(scope["active_aic_groups"]),
        "cold_active_aic_groups": tuple(scope["cold_active_aic_groups"]),
        "geometries": geometries,
        "evidence_ref": str(AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE),
        "evidence_sha256": (
            AIC_MTE2_ND2NZ_LARGE_CONTIGUOUS_REFERENCE_SHA256),
    }


def load_aic_mte2_gemm_large_resident_service(repository_root):
    """Load source-stride-aware resident GEMM descriptor services."""

    path = repository_root / AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 GEMM large-resident reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 GEMM large-resident reference"
    require_exact_fields(
        owner, value,
        {
            "schema_version", "hardware_id", "software_revision",
            "mechanism", "duration_unit",
            "production_or_evaluation_duration_consumed",
            "measurement_scope", "parameters_ns", "full_program_holdout",
            "supplemental_program_holdouts",
            "measurement_quality", "provenance",
        })
    expected = {
        "schema_version": (
            "midlayer.ascend_910b."
            "aic_mte2_gemm_large_resident_descriptor.v3"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_l2_to_l1_nd2nz_gemm_large_descriptor_service"),
        "duration_unit": "ns_per_descriptor_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")

    scope = value["measurement_scope"]
    expected_scope = {
        "dtype": "bf16",
        "source_residency": "l2_resident",
        "active_aic_groups": [18, 22, 23, 24],
        "modes": {"serial_completion": 0, "two_buffer_stream": 1},
        "resident_footprint_bytes": 128 * 1024 * 1024,
        "published_l2_capacity_bytes": 192 * 1024 * 1024,
        "trials": 4,
        "iterations": 224,
        "order": "alternating_forward_reverse",
        "statistic": "median",
    }
    # The small-M extension is a second independent campaign over the same
    # mechanism: it adds the descriptor geometries of the at-most-64-row
    # production GEMM launches.  It was captured on a shared card, so it
    # declares that fact and only its cells with a cross-round median spread
    # within ten percent are published here.
    require_exact_fields(
        f"{owner}.measurement_scope", scope,
        set(expected_scope) | {"serving_decode_extension", "small_m_extension"})
    serving_scope = scope["serving_decode_extension"]
    if (not isinstance(serving_scope, dict)
            or serving_scope["exclusive_card"] is not False
            or serving_scope["independent_runs"] != 3
            or serving_scope["trials_per_run"] != 4
            or serving_scope["iterations"] != 224
            or not (0 < serving_scope["resident_footprint_bytes"]
                    < scope["published_l2_capacity_bytes"])):
        raise ValueError(
            f"{owner}.measurement_scope Serving-decode extension is invalid")
    small_m_scope = scope["small_m_extension"]
    if (not isinstance(small_m_scope, dict)
            or small_m_scope["exclusive_card"] is not False
            or small_m_scope["independent_runs"] != 3
            or small_m_scope["trials_per_run"] != 4
            or small_m_scope["iterations"] != 224):
        raise ValueError(
            f"{owner}.measurement_scope small-M extension is invalid")
    if {name: scope[name] for name in expected_scope} != expected_scope:
        raise ValueError(f"{owner}.measurement_scope changed")

    parameter_fields = {
        "rows", "columns", "source_row_stride_elements",
        "active_aic_groups", "serial_ns", "stream_ns",
    }
    # The original eight cells, the small-M extension, and the
    # Serving-decode extension: the complete measured descriptor domain
    # of the at-most-64-row production GEMM launches and of the ShareGPT
    # qwen3-8b tp1_qps16 Serving projection.
    expected_geometries = {
        (16, 256, 11008, 22),
        (16, 256, 12288, 22),
        (16, 256, 13824, 23),
        (16, 256, 14336, 22),
        (16, 512, 4096, 23),
        (16, 512, 5120, 24),
        (16, 512, 8192, 22),
        (16, 512, 11008, 22),
        (16, 512, 13824, 23),
        (16, 512, 14336, 22),
        (16, 1536, 1536, 23),
        (16, 4096, 4096, 22),
        (16, 4096, 4096, 23),
        (16, 4096, 4096, 24),
        (16, 5120, 5120, 23),
        (16, 5120, 5120, 24),
        (16, 8192, 8192, 23),
        (32, 256, 5120, 22),
        (32, 256, 5120, 23),
        (32, 256, 8192, 22),
        (32, 256, 8192, 23),
        (32, 256, 11008, 22),
        (32, 256, 12288, 22),
        (32, 256, 13824, 23),
        (32, 256, 14336, 22),
        (32, 1536, 1536, 23),
        (32, 4096, 4096, 22),
        (32, 4096, 4096, 24),
        (48, 256, 4096, 22),
        (48, 256, 4096, 23),
        (48, 256, 4096, 24),
        (48, 256, 5120, 22),
        (48, 256, 5120, 23),
        (48, 256, 5120, 24),
        (48, 256, 8192, 22),
        (48, 256, 8192, 23),
        (48, 256, 11008, 22),
        (48, 256, 12288, 22),
        (48, 256, 13824, 23),
        (48, 1536, 1536, 23),
        (64, 256, 4096, 22),
        (64, 256, 4096, 23),
        (64, 256, 4096, 24),
        (64, 256, 5120, 22),
        (64, 256, 5120, 23),
        (64, 256, 5120, 24),
        (64, 256, 8192, 22),
        (64, 256, 8192, 23),
        (64, 256, 11008, 22),
        (64, 256, 13824, 23),
        (64, 256, 14336, 22),
        (64, 512, 5120, 24),
        (64, 1536, 1536, 23),
        (80, 256, 4096, 22),
        (80, 256, 4096, 24),
        (80, 256, 12288, 22),
        (80, 512, 4096, 22),
        (80, 512, 4096, 24),
        (96, 256, 4096, 22),
        (96, 256, 4096, 24),
        (96, 256, 11008, 22),
        (96, 256, 12288, 22),
        (96, 512, 4096, 22),
        (96, 512, 4096, 23),
        (96, 512, 4096, 24),
        (96, 512, 11008, 22),
        (96, 512, 12288, 22),
        (96, 512, 12288, 24),
        (96, 512, 14336, 22),
        (112, 256, 4096, 24),
        (112, 256, 5120, 23),
        (112, 256, 12288, 22),
        (112, 256, 13824, 23),
        (112, 512, 4096, 22),
        (112, 512, 4096, 24),
        (112, 512, 12288, 22),
        (112, 512, 12288, 24),
        (112, 512, 13824, 23),
        (128, 256, 4096, 22),
        (128, 256, 4096, 23),
        (128, 256, 4096, 24),
        (128, 256, 5120, 22),
        (128, 256, 12288, 24),
        (128, 512, 4096, 22),
        (128, 512, 4096, 23),
        (128, 512, 4096, 24),
        (128, 512, 12288, 24),
        (176, 256, 4096, 24),
        (192, 256, 5120, 24),
        (192, 256, 12288, 22),
        (192, 512, 4096, 24),
        (208, 256, 4096, 24),
        (208, 256, 12288, 24),
        (224, 256, 1536, 23),
        (224, 256, 4096, 24),
        (224, 256, 5120, 23),
        (224, 256, 12288, 24),
        (240, 256, 4096, 24),
        (240, 256, 12288, 24),
        (256, 256, 4096, 22),
        (256, 256, 4096, 24),
        (256, 256, 5120, 18),
        (256, 256, 8192, 23),
        (256, 256, 12288, 24),
        (320, 256, 4096, 24),
    }
    parameters = value["parameters_ns"]
    if (not isinstance(parameters, list)
            or len(parameters) != len(expected_geometries)):
        raise ValueError(
            f"{owner}.parameters_ns must contain one cell per measured "
            "geometry")
    geometries = {}
    for index, parameter in enumerate(parameters):
        parameter_owner = f"{owner}.parameters_ns[{index}]"
        if not isinstance(parameter, dict):
            raise TypeError(f"{parameter_owner} must be a mapping")
        require_exact_fields(parameter_owner, parameter, parameter_fields)
        geometry = tuple(parameter[name] for name in (
            "rows", "columns", "source_row_stride_elements",
            "active_aic_groups"))
        if (any(isinstance(item, bool) or not isinstance(item, int)
                or item <= 0 for item in geometry)
                or geometry[2] < geometry[1]
                or geometry not in expected_geometries
                or geometry in geometries):
            raise ValueError(f"{parameter_owner} geometry is invalid")
        services = {}
        for field in ("serial_ns", "stream_ns"):
            duration = parameter[field]
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(float(duration))
                    or float(duration) <= 0.0):
                raise ValueError(f"{parameter_owner}.{field} is invalid")
            services[field] = float(duration)
        geometries[geometry] = services
    if set(geometries) != expected_geometries:
        raise ValueError(f"{owner} exact geometry domain changed")

    holdout = value["full_program_holdout"]
    require_exact_fields(
        f"{owner}.full_program_holdout", holdout,
        {
            "active_aic_groups", "a_rows", "a_columns",
            "a_source_row_stride_elements", "a_buffer_depth",
            "a_descriptors_per_worker", "b_rows", "b_columns",
            "b_source_row_stride_elements", "b_buffer_depth",
            "b_descriptors_per_worker", "credit_release",
            "median_completion_ns", "minimum_completion_ns",
            "maximum_completion_ns", "samples", "forward_parameter",
        })
    expected_holdout_signature = {
        "active_aic_groups": 23,
        "a_rows": 16,
        "a_columns": 5120,
        "a_source_row_stride_elements": 5120,
        "a_buffer_depth": 1,
        "a_descriptors_per_worker": 1,
        "b_rows": 224,
        "b_columns": 256,
        "b_source_row_stride_elements": 5120,
        "b_buffer_depth": 2,
        "b_descriptors_per_worker": 20,
        "credit_release": (
            "zero_work_mte2_to_mte1_handoff_after_each_b1_pair"),
        "samples": 24,
        "forward_parameter": False,
    }
    for name, expected_value in expected_holdout_signature.items():
        if holdout[name] != expected_value:
            raise ValueError(
                f"{owner}.full_program_holdout.{name} changed")
    for name in (
            "minimum_completion_ns", "median_completion_ns",
            "maximum_completion_ns"):
        duration = holdout[name]
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration)) or duration <= 0.0):
            raise ValueError(
                f"{owner}.full_program_holdout.{name} is invalid")
    if not (holdout["minimum_completion_ns"]
            <= holdout["median_completion_ns"]
            <= holdout["maximum_completion_ns"]):
        raise ValueError(f"{owner} holdout order is invalid")

    quality = value["measurement_quality"]
    expected_quality_counts = {
        "validated_descriptor_rows": 64,
        "failed_descriptor_rows": 0,
        "validated_program_rows": 24,
        "failed_program_rows": 0,
        "small_m_extension_validated_descriptor_rows": 840,
        "small_m_extension_failed_descriptor_rows": 0,
        "small_m_program_validated_rows": 1320,
        "small_m_program_failed_rows": 0,
        "serving_decode_validated_descriptor_rows": 1080,
        "serving_decode_failed_descriptor_rows": 0,
    }
    require_exact_fields(
        f"{owner}.measurement_quality", quality,
        set(expected_quality_counts) | {
            "maximum_descriptor_range_over_median_pct",
            "program_range_over_median_pct",
            "small_m_extension_admitted_geometries",
            "small_m_extension_pending_exclusive_card_geometries",
            "small_m_extension_maximum_admitted_round_spread_pct",
            "small_m_program_admitted_topologies",
            "small_m_program_pending_exclusive_card_topologies",
            "small_m_program_cross_round_spread_pct_max",
            "serving_decode_admitted_geometries",
            "serving_decode_pending_exclusive_card_geometries",
        })
    for name, expected_value in expected_quality_counts.items():
        if quality[name] != expected_value:
            raise ValueError(f"{owner}.measurement_quality.{name} changed")
    for name in (
            "maximum_descriptor_range_over_median_pct",
            "program_range_over_median_pct"):
        metric = quality[name]
        if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric)) or metric < 0.0):
            raise ValueError(f"{owner}.measurement_quality.{name} is invalid")

    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {
            "measurement_baseline_commit", "physical_device_id", "container",
            "descriptor_source_snapshot_sha256", "descriptor_source_sha256",
            "descriptor_binary_sha256", "descriptor_raw_path",
            "descriptor_raw_sha256", "descriptor_runner_stderr_path",
            "descriptor_runner_stderr_sha256",
            "k4096_descriptor_raw_path", "k4096_descriptor_raw_sha256",
            "k4096_descriptor_runner_stderr_path",
            "k4096_descriptor_runner_stderr_sha256",
            "small_m_extension", "small_m_program_extension",
            "serving_decode_extension",
            "program_source_snapshot_sha256", "program_source_sha256",
            "program_binary_sha256", "program_raw_path",
            "program_raw_sha256", "program_runner_stderr_path",
            "program_runner_stderr_sha256",
        })
    if (isinstance(provenance["physical_device_id"], bool)
            or not isinstance(provenance["physical_device_id"], int)
            or provenance["physical_device_id"] < 0
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())):
        raise ValueError(f"{owner} execution provenance changed")
    commit = provenance["measurement_baseline_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError(f"{owner}.measurement_baseline_commit is invalid")
    for prefix, expected_files in (
            ("descriptor", {
                "CMakeLists.txt", "aic_mte2_nd2nz_stride_kernel.cpp",
                "aic_mte2_nd2nz_stride_main.cpp",
                "aic_mte2_nd2nz_stride_runner.py",
            }),
            ("program", {
                "CMakeLists.txt", "aic_mte2_gemm_a1_b20_stream_kernel.cpp",
                "aic_mte2_gemm_a1_b20_stream_main.cpp",
                "aic_mte2_gemm_a1_b20_stream_runner.py",
            })):
        source_sha256 = provenance[f"{prefix}_source_sha256"]
        if (not isinstance(source_sha256, dict)
                or set(source_sha256) != expected_files):
            raise ValueError(f"{owner}.{prefix}_source_sha256 changed")
        for digest in source_sha256.values():
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(f"{owner}.{prefix}_source_sha256 is invalid")
        for suffix in (
                "source_snapshot_sha256", "binary_sha256", "raw_sha256",
                "runner_stderr_sha256"):
            digest = provenance[f"{prefix}_{suffix}"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(f"{owner}.{prefix}_{suffix} is invalid")
        for suffix in ("raw_path", "runner_stderr_path"):
            evidence_path = provenance[f"{prefix}_{suffix}"]
            if (not isinstance(evidence_path, str)
                    or not evidence_path.startswith(
                        "profiler_output/microbench/ascend_910b/")
                    or ".." in evidence_path.split("/")):
                raise ValueError(f"{owner}.{prefix}_{suffix} is invalid")
    for name in (
            "k4096_descriptor_raw_sha256",
            "k4096_descriptor_runner_stderr_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.{name} is invalid")
    for name in (
            "k4096_descriptor_raw_path",
            "k4096_descriptor_runner_stderr_path"):
        evidence_path = provenance[name]
        if (not isinstance(evidence_path, str)
                or not evidence_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in evidence_path.split("/")):
            raise ValueError(f"{owner}.{name} is invalid")
    return {
        "active_aic_groups": tuple(scope["active_aic_groups"]),
        "geometries": geometries,
        "full_program_holdout": holdout,
        "evidence_ref": str(AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE),
        "evidence_sha256": AIC_MTE2_GEMM_LARGE_RESIDENT_REFERENCE_SHA256,
    }


def load_aic_mte2_gemm_shared_source_queue_service(repository_root):
    """Load source-stride-aware A/B services for one shared MTE2 queue."""

    path = repository_root / AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE_SHA256):
        raise ValueError("AIC GEMM shared-source-queue reference SHA mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC GEMM shared-source-queue reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "production_or_evaluation_duration_consumed",
         "domain", "descriptor_services", "source_program_validation",
         "provenance"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_mte2_gemm_source_queue.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_matmul_v2_shared_mte2_source_queue_v1",
        "duration_unit": "ns_per_descriptor_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    domain = value["domain"]
    expected_domain = {
        "dtype": "bf16",
        "active_aic_groups": 24,
        "operand_buffer_depth": 2,
        "traversal_order": "persistent_n_then_m",
        "source_residency_states": ["l2_resident", "dram_cold"],
    }
    if domain != expected_domain:
        raise ValueError(f"{owner}.domain changed")

    service_fields = {
        "rows", "columns", "source_row_stride_elements",
        "resident_serial_samples_ns", "resident_stream_samples_ns",
        "cold_serial_samples_ns", "cold_stream_samples_ns",
        "resident_serial_ns", "resident_stream_ns", "cold_serial_ns",
        "cold_stream_ns",
    }
    geometries = {}
    services = value["descriptor_services"]
    if not isinstance(services, list) or len(services) != 2:
        raise ValueError(f"{owner}.descriptor_services changed")
    for index, cell in enumerate(services):
        cell_owner = f"{owner}.descriptor_services[{index}]"
        require_exact_fields(cell_owner, cell, service_fields)
        geometry = tuple(cell[name] for name in (
            "rows", "columns", "source_row_stride_elements"))
        if (any(isinstance(item, bool) or not isinstance(item, int)
                or item <= 0 for item in geometry)
                or geometry[2] < geometry[1] or geometry in geometries):
            raise ValueError(f"{cell_owner} geometry is invalid")
        normalized = {}
        for residency in ("resident", "cold"):
            for mode in ("serial", "stream"):
                samples = cell[f"{residency}_{mode}_samples_ns"]
                median = cell[f"{residency}_{mode}_ns"]
                if (not isinstance(samples, list) or len(samples) != 4
                        or any(isinstance(sample, bool)
                               or not isinstance(sample, (int, float))
                               or not math.isfinite(float(sample))
                               or sample <= 0 for sample in samples)
                        or median != statistics.median(samples)):
                    raise ValueError(
                        f"{cell_owner}.{residency}_{mode} is invalid")
                normalized[f"{residency}_{mode}_ns"] = float(median)
        geometries[geometry] = normalized
    if set(geometries) != {
            (128, 512, 14336), (240, 256, 14336)}:
        raise ValueError(f"{owner} descriptor domain changed")

    validation = value["source_program_validation"]
    validation_fields = {
        "worker_grid_m", "worker_grid_n", "worker_m_tiles",
        "worker_n_tiles", "output_tiles_per_worker", "k_slabs",
        "a_descriptors_per_worker", "b_descriptors_per_worker",
        "a_dram_bytes", "b_dram_bytes", "dram_capacity_bytes_per_s",
        "mte2_only_run_medians_ns", "mte2_only_median_of_run_medians_ns",
        "analytical_completion_ns", "analytical_relative_error",
        "forward_parameter",
    }
    require_exact_fields(
        f"{owner}.source_program_validation", validation, validation_fields)
    samples = validation["mte2_only_run_medians_ns"]
    if (not isinstance(samples, list) or len(samples) != 3
            or validation["mte2_only_median_of_run_medians_ns"]
                != statistics.median(samples)
            or validation["forward_parameter"] is not False):
        raise ValueError(f"{owner} source-program validation is invalid")
    a_service = geometries[(128, 512, 14336)]
    b_service = geometries[(240, 256, 14336)]
    analytical_ns = math.fsum((
        a_service["resident_serial_ns"],
        (validation["a_descriptors_per_worker"] - 1)
        * a_service["resident_stream_ns"],
        validation["b_descriptors_per_worker"]
        * b_service["resident_stream_ns"],
        (validation["a_dram_bytes"] + validation["b_dram_bytes"])
        / validation["dram_capacity_bytes_per_s"] * 1.0e9,
    ))
    observed_ns = validation["mte2_only_median_of_run_medians_ns"]
    relative_error = abs(analytical_ns - observed_ns) / observed_ns
    if (not math.isclose(validation["analytical_completion_ns"], analytical_ns,
                         rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(validation["analytical_relative_error"],
                                relative_error, rel_tol=0.0, abs_tol=1e-12)
            or relative_error > 0.02):
        raise ValueError(f"{owner} analytical holdout changed")

    provenance = value["provenance"]
    provenance_fields = {
        "measurement_baseline_commit", "physical_device_id", "container",
        "descriptor_source_snapshot_sha256", "descriptor_binary_sha256",
        "descriptor_raw_paths", "descriptor_raw_sha256",
        "source_program_source_snapshot_sha256",
        "source_program_binary_sha256", "source_program_raw_paths",
        "source_program_raw_sha256",
    }
    require_exact_fields(f"{owner}.provenance", provenance, provenance_fields)
    commit = provenance["measurement_baseline_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())):
        raise ValueError(f"{owner} execution provenance is invalid")
    for prefix in ("descriptor", "source_program"):
        paths = provenance[f"{prefix}_raw_paths"]
        hashes = provenance[f"{prefix}_raw_sha256"]
        if (not isinstance(paths, list) or not paths
                or len(paths) != len(hashes)
                or any(not isinstance(item, str)
                       or not item.startswith(
                           "profiler_output/microbench/ascend_910b/")
                       or ".." in item.split("/") for item in paths)
                or any(not isinstance(item, str) or len(item) != 64
                       or any(character not in "0123456789abcdef"
                              for character in item) for item in hashes)):
            raise ValueError(f"{owner}.{prefix} raw provenance is invalid")
        for suffix in ("source_snapshot_sha256", "binary_sha256"):
            digest = provenance[f"{prefix}_{suffix}"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(f"{owner}.{prefix}_{suffix} is invalid")
    return {
        **domain,
        "geometries": geometries,
        "evidence_ref": str(
            AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE),
        "evidence_sha256": (
            AIC_MTE2_GEMM_K14336_SOURCE_QUEUE_REFERENCE_SHA256),
    }


def load_aic_mte2_cold_operand_pair_stream_service(repository_root):
    """Load the frozen source-shaped, DRAM-cold A/B stream service."""

    path = repository_root / AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE_SHA256):
        raise ValueError("AIC MTE2 cold operand-pair reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 cold operand-pair reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "service",
         "service_time_unit", "published_cold_pair_service_ns",
         "publication_scope", "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "aic_mte2_cold_operand_pair_stream",
        "service_time_unit": "ns_per_128x64_A_plus_64x256_B_pair_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"source_residency", "active_aic_groups", "a_rows", "a_columns",
         "a_bytes", "b_rows", "b_columns", "b_bytes", "pair_bytes",
         "drain_interval_pairs", "pair_count_per_worker_min",
         "pair_count_per_worker_max", "measured_pair_counts_per_worker",
         "source_ring_constraint"})
    expected_scope = {
        "source_residency": "dram_cold",
        "active_aic_groups": 24,
        "a_rows": 128,
        "a_columns": 64,
        "a_bytes": 16384,
        "b_rows": 64,
        "b_columns": 256,
        "b_bytes": 32768,
        "pair_bytes": 49152,
        "drain_interval_pairs": 4,
        "pair_count_per_worker_min": 512,
        "pair_count_per_worker_max": 2048,
        "measured_pair_counts_per_worker": [512, 1024, 2048],
    }
    for name, expected_value in expected_scope.items():
        if scope[name] != expected_value:
            raise ValueError(f"{owner}.publication_scope.{name} changed")
    measurement = value["measurement"]
    if measurement["mechanism_gate_passed"] is not True:
        raise ValueError(f"{owner} mechanism gate failed")
    duration = value["published_cold_pair_service_ns"]
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError(f"{owner} service time must be finite and positive")
    return {
        "model": "aic_dram_to_l1_nd2nz_operand_pair_stream_v1",
        "duration_unit": "ns_per_a_b_operand_pair_per_aic",
        "software_revision": value["software_revision"],
        "dtype": "bf16",
        "source_residency": scope["source_residency"],
        "a_rows": scope["a_rows"],
        "a_columns": scope["a_columns"],
        "b_rows": scope["b_rows"],
        "b_columns": scope["b_columns"],
        "active_aic_groups": scope["active_aic_groups"],
        "drain_interval_pairs": scope["drain_interval_pairs"],
        "minimum_pairs_per_worker": scope["pair_count_per_worker_min"],
        "maximum_pairs_per_worker": scope["pair_count_per_worker_max"],
        "measured_pair_counts_per_worker": tuple(
            scope["measured_pair_counts_per_worker"]),
        "continuous_ns_per_pair": float(duration),
        "reference_path": str(AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE),
        "reference_sha256": AIC_MTE2_COLD_OPERAND_PAIR_REFERENCE_SHA256,
    }


def load_aic_mte2_finite_stream_completion_service(repository_root):
    """Load the cold finite B1 program-completion lower envelope."""

    path = repository_root / AIC_MTE2_FINITE_STREAM_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_FINITE_STREAM_REFERENCE_SHA256):
        raise ValueError("AIC MTE2 finite-stream reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 finite-stream reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_nd2nz_finite_stream_completion",
        "duration_unit": "ns_per_finite_descriptor_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"source_residency", "dtype", "rows", "columns",
         "descriptor_bytes", "minimum_descriptors_per_worker",
         "maximum_descriptors_per_worker", "worker_classes",
         "source_program", "estimator"})
    frozen_scope = {
        "source_residency": "dram_cold",
        "dtype": "bf16",
        "rows": 256,
        "columns": 256,
        "descriptor_bytes": 131072,
        "minimum_descriptors_per_worker": 48,
        "maximum_descriptors_per_worker": 72,
    }
    for name, expected_value in frozen_scope.items():
        if scope[name] != expected_value:
            raise ValueError(f"{owner}.publication_scope.{name} changed")
    if scope["source_program"] != (
            "one B1 ND-to-NZ descriptor stream with no intermediate drain"):
        raise ValueError(f"{owner} source program changed")
    if scope["estimator"] != (
            "lower-envelope endpoint affine service; interpolate only within "
            "a published active-AIC axis"):
        raise ValueError(f"{owner} estimator changed")
    classes = scope["worker_classes"]
    if not isinstance(classes, list) or len(classes) != 2:
        raise ValueError(f"{owner} must publish exactly two stride classes")
    normalized_classes = {}
    expected_axes = ((8192, (16,)), (13824, (18, 22)))
    for raw_class, (expected_stride, expected_groups) in zip(
            classes, expected_axes):
        require_exact_fields(
            f"{owner}.worker_class", raw_class,
            {"source_row_stride_elements", "active_aic_groups",
             "duration_at_48_descriptors_ns",
             "ns_per_additional_descriptor"})
        groups = raw_class["active_aic_groups"]
        intercepts = raw_class["duration_at_48_descriptors_ns"]
        slopes = raw_class["ns_per_additional_descriptor"]
        if (raw_class["source_row_stride_elements"] != expected_stride
                or tuple(groups) != expected_groups):
            raise ValueError(f"{owner} physical worker class changed")
        if (not isinstance(intercepts, list)
                or not isinstance(slopes, list)
                or len(intercepts) != len(groups)
                or len(slopes) != len(groups)):
            raise ValueError(f"{owner} affine axes disagree")
        normalized_intercepts = tuple(float(item) for item in intercepts)
        normalized_slopes = tuple(float(item) for item in slopes)
        if any(not math.isfinite(item) or item <= 0.0
               for item in normalized_intercepts + normalized_slopes):
            raise ValueError(f"{owner} affine services must be positive")
        normalized_classes[expected_stride] = {
            "active_aic_groups": expected_groups,
            "duration_at_minimum_descriptors_ns": normalized_intercepts,
            "ns_per_additional_descriptor": normalized_slopes,
        }
    measurement = value["measurement"]
    if not isinstance(measurement, dict):
        raise TypeError(f"{owner}.measurement must be a mapping")
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"mode", "capture_commit", "physical_device_id", "sweep_bytes",
         "binary_sha256", "source_kernel_sha256", "source_main_sha256",
         "source_runner_sha256", "raw_paths", "raw_sha256",
         "mechanism_gate_passed"})
    expected_measurement = {
        "mode": "independent_finite_descriptor_stream_microbenchmark",
        "capture_commit": "efcf67d7dd02326e56fd036ad4f8898cb71d4d70",
        "sweep_bytes": 2147483648,
        "mechanism_gate_passed": True,
    }
    if (isinstance(measurement["physical_device_id"], bool)
            or not isinstance(measurement["physical_device_id"], int)
            or measurement["physical_device_id"] < 0
            or any(measurement[name] != expected_value
                   for name, expected_value in expected_measurement.items())):
        raise ValueError(f"{owner} mechanism gate failed")
    raw_paths = measurement["raw_paths"]
    raw_hashes = measurement["raw_sha256"]
    if (not isinstance(raw_paths, list) or len(raw_paths) != 6
            or not isinstance(raw_hashes, list) or len(raw_hashes) != 6):
        raise ValueError(f"{owner} requires six raw captures")
    for raw_path in raw_paths:
        if (not isinstance(raw_path, str)
                or not raw_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in raw_path.split("/")):
            raise ValueError(f"{owner} raw path is outside its evidence owner")
    evidence_hashes = tuple(raw_hashes) + tuple(
        measurement[name] for name in (
            "binary_sha256", "source_kernel_sha256", "source_main_sha256",
            "source_runner_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "only a full-M-tile cann_matmul_v3 B1 program with the exact "
            "BF16 descriptor geometry, published physical source-row stride, "
            "active-AIC interval, descriptor-count interval, and DRAM-cold "
            "residency"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": "aic_mte2_nd2nz_finite_stream_completion_v1",
        **frozen_scope,
        "worker_classes": normalized_classes,
        "reference_path": str(AIC_MTE2_FINITE_STREAM_REFERENCE),
        "reference_sha256": AIC_MTE2_FINITE_STREAM_REFERENCE_SHA256,
    }


def load_aic_mte2_gemm_source_order_completion_service(repository_root):
    """Load exact MTE2 and MTE2-to-MTE1 source-program services."""

    references = (
        (AIC_MTE2_GEMM_SOURCE_ORDER_COMPLETION_REFERENCE,
         AIC_MTE2_GEMM_SOURCE_ORDER_COMPLETION_REFERENCE_SHA256),
        (AIC_MTE2_GEMM_K13824_SOURCE_ORDER_COMPLETION_REFERENCE,
         AIC_MTE2_GEMM_K13824_SOURCE_ORDER_COMPLETION_REFERENCE_SHA256),
    )
    expected = {
        "schema_version": (
            "midlayer.ascend_910b."
            "aic_mte2_gemm_source_order_completion.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_matmul_v2_mte2_to_mte1_source_order_completion_v1"),
        "duration_unit": "ns_per_completed_source_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    programs = {}
    for reference_path, reference_sha256 in references:
        path = repository_root / reference_path
        payload_bytes = path.read_bytes()
        if hashlib.sha256(payload_bytes).hexdigest() != reference_sha256:
            raise ValueError(
                "AIC GEMM source-order completion reference SHA256 mismatch")
        value = json.loads(payload_bytes)
        owner = f"AIC GEMM source-order reference {reference_path.name}"
        require_exact_fields(
            owner, value,
            {"schema_version", "hardware_id", "software_revision",
             "mechanism", "duration_unit", "source_program", "capture",
             "measurements", "production_or_evaluation_duration_consumed"})
        for name, expected_value in expected.items():
            if value[name] != expected_value:
                raise ValueError(f"{owner}.{name} must be {expected_value!r}")
        source_program = value["source_program"]
        require_exact_fields(
            f"{owner}.source_program", source_program,
            set(AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS))
        for name in AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS:
            field = source_program[name]
            if name == "dtype":
                if field != "bf16":
                    raise ValueError(f"{owner}.source_program.dtype changed")
            elif isinstance(field, bool) or not isinstance(field, int):
                raise TypeError(
                    f"{owner}.source_program.{name} must be an integer")
            elif field <= 0:
                raise ValueError(
                    f"{owner}.source_program.{name} must be positive")
        capture = value["capture"]
        legacy_capture_fields = {
            "source_commit", "source_snapshot_sha256",
            "source_mte2_kernel_sha256", "source_ingress_kernel_sha256",
            "source_main_sha256", "binary_sha256", "physical_device_id",
            "run_count", "trials_per_run", "raw_paths", "raw_sha256",
        }
        capture_fields = set(capture)
        if capture_fields not in (
                legacy_capture_fields,
                legacy_capture_fields | {"source_runner_sha256"}):
            raise ValueError(f"{owner}.capture fields changed")
        if (isinstance(capture["physical_device_id"], bool)
                or not isinstance(capture["physical_device_id"], int)
                or capture["physical_device_id"] < 0
                or capture["run_count"] != 3
                or capture["trials_per_run"] != 8):
            raise ValueError(f"{owner}.capture campaign changed")
        raw_paths = capture["raw_paths"]
        raw_hashes = capture["raw_sha256"]
        if (not isinstance(raw_paths, list) or not raw_paths
                or not isinstance(raw_hashes, list)
                or len(raw_hashes) != len(raw_paths)):
            raise ValueError(f"{owner}.capture raw provenance changed")
        for raw_path in raw_paths:
            if (not isinstance(raw_path, str)
                    or not raw_path.startswith(
                        "profiler_output/microbench/ascend_910b/")
                    or ".." in raw_path.split("/")):
                raise ValueError(
                    f"{owner}.capture raw path is outside its owner")
        evidence_names = [
            "source_snapshot_sha256", "source_mte2_kernel_sha256",
            "source_ingress_kernel_sha256", "source_main_sha256",
            "binary_sha256",
        ]
        if "source_runner_sha256" in capture:
            evidence_names.append("source_runner_sha256")
        evidence_hashes = tuple(raw_hashes) + tuple(
            capture[name] for name in evidence_names)
        if any(not isinstance(digest, str) or len(digest) != 64
               or any(character not in "0123456789abcdef"
                      for character in digest)
               for digest in evidence_hashes):
            raise ValueError(f"{owner}.capture evidence SHA256 is invalid")
        measurements = value["measurements"]
        require_exact_fields(
            f"{owner}.measurements", measurements,
            {"mte2_only_run_medians_ns",
             "mte2_mte1_completion_run_medians_ns",
             "mte2_only_median_of_run_medians_ns",
             "mte2_mte1_completion_median_of_run_medians_ns"})
        mte2_ns = measurements["mte2_only_median_of_run_medians_ns"]
        completion_ns = measurements[
            "mte2_mte1_completion_median_of_run_medians_ns"]
        for name, duration_ns in (
                ("MTE2", mte2_ns), ("completion", completion_ns)):
            if (isinstance(duration_ns, bool)
                    or not isinstance(duration_ns, (int, float))
                    or not math.isfinite(float(duration_ns))
                    or duration_ns <= 0):
                raise ValueError(
                    f"{owner} {name} must be finite and positive")
        if completion_ns < mte2_ns:
            raise ValueError(f"{owner} completion precedes MTE2 issue")
        signature = tuple(
            source_program[name]
            for name in AIC_MTE2_GEMM_SOURCE_ORDER_SIGNATURE_FIELDS)
        if signature in programs:
            raise ValueError("AIC GEMM source-order signature is duplicated")
        programs[signature] = {
            **source_program,
            "mte2_only_ns": float(mte2_ns),
            "completion_ns": float(completion_ns),
            "reference_path": str(reference_path),
            "reference_sha256": reference_sha256,
        }
    capacity_cells = []
    for program in programs.values():
        if program["a_columns"] % program["b_columns"]:
            raise ValueError(
                "AIC GEMM source-order A/B descriptor grouping changed")
        b_descriptors_per_a = (
            program["a_columns"] // program["b_columns"])
        bytes_per_mte2_program = 2 * (
            program["a_rows"] * program["a_columns"]
            + b_descriptors_per_a
            * program["b_rows"] * program["b_columns"])
        source_bytes = (
            program["mte2_programs_per_worker"]
            * bytes_per_mte2_program)
        capacity_cells.append({
            "active_aic_groups": program["active_aic_groups"],
            "source_bytes_per_worker": source_bytes,
            "mte2_only_ns": program["mte2_only_ns"],
            "completion_ns": program["completion_ns"],
            "capacity_bytes_per_s": (
                source_bytes / program["mte2_only_ns"] * 1.0e9),
            "completion_capacity_bytes_per_s": (
                source_bytes / program["completion_ns"] * 1.0e9),
            "reference_path": program["reference_path"],
            "reference_sha256": program["reference_sha256"],
        })
    active_group_counts = {
        cell["active_aic_groups"] for cell in capacity_cells}
    if len(active_group_counts) != 1:
        raise ValueError(
            "AIC GEMM source-order capacity cells changed AIC topology")
    capacities = [
        cell["capacity_bytes_per_s"] for cell in capacity_cells]
    shared_capacity = statistics.median(capacities)
    maximum_relative_delta = max(
        abs(capacity - shared_capacity) / shared_capacity
        for capacity in capacities)
    if maximum_relative_delta > 0.06:
        raise ValueError(
            "AIC GEMM source-order shared capacity is not geometry-stable")
    completion_capacities = [
        cell["completion_capacity_bytes_per_s"]
        for cell in capacity_cells]
    shared_completion_capacity = statistics.median(completion_capacities)
    completion_maximum_relative_delta = max(
        abs(capacity - shared_completion_capacity)
        / shared_completion_capacity
        for capacity in completion_capacities)
    if completion_maximum_relative_delta > 0.02:
        raise ValueError(
            "AIC GEMM source-order completion capacity is not "
            "geometry-stable")
    return {
        "model": expected["mechanism"],
        "programs": programs,
        "capacity_constrained_active_aic_groups": (
            active_group_counts.pop()),
        "capacity_constrained_bytes_per_s": shared_capacity,
        "capacity_constrained_completion_bytes_per_s": (
            shared_completion_capacity),
        "capacity_constrained_maximum_relative_delta": (
            maximum_relative_delta),
        "capacity_constrained_completion_maximum_relative_delta": (
            completion_maximum_relative_delta),
        "capacity_constrained_cells": tuple(capacity_cells),
    }


def load_aic_matmul_v3_split_k_resident_tile_service(repository_root):
    """Load source-isolated key-65632 resident tile completion service."""

    path = repository_root / AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE_SHA256):
        raise ValueError(
            "AIC MatMulV3 split-K resident-tile reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MatMulV3 split-K resident-tile reference"
    require_exact_fields(
        owner, value,
        {"schema", "hardware_id", "software_revision", "source_commit",
         "cann_source", "source_artifacts", "capture", "scope", "result",
         "validation", "mechanism", "forward_eligible",
         "production_or_evaluation_duration_consumed"})
    expected = {
        "schema": (
            "ascend_910b_matmul_v3_split_k_resident_tile_reference_v2"),
        "hardware_id": "ascend_910b",
        "software_revision": "CANN 8.5.1",
        "mechanism": "matmul_v3_key65632_resident_base_tile_completion",
        "forward_eligible": True,
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["scope"]
    require_exact_fields(
        f"{owner}.scope", scope,
        {"active_aic_workers", "dtype", "tile_m", "tile_n",
         "tile_k_values", "included", "excluded"})
    if (scope["active_aic_workers"] != 24
            or scope["dtype"] != "bfloat16"
            or scope["tile_m"] != 128
            or scope["tile_n"] != 128
            or scope["tile_k_values"] != [384, 256]):
        raise ValueError(f"{owner}.scope changed its frozen geometry")
    result = value["result"]
    require_exact_fields(
        f"{owner}.result", result,
        {"median_cycles_per_tile_by_tile_k",
         "median_ns_per_tile_by_tile_k", "system_counter_tick_ns",
         "tensor_operations_per_tile_by_tile_k", "unit"})
    if result["unit"] != "ns_per_aic_resident_base_tile_completion":
        raise ValueError(f"{owner}.result.unit changed")
    duration_by_k = {}
    for tile_k in scope["tile_k_values"]:
        duration = result["median_ns_per_tile_by_tile_k"][str(tile_k)]
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0.0):
            raise ValueError(f"{owner} contains an invalid tile duration")
        duration_by_k[tile_k] = float(duration)
    return {
        "active_aic_workers": scope["active_aic_workers"],
        "dtype": "bf16",
        "tile_m": scope["tile_m"],
        "tile_n": scope["tile_n"],
        "duration_ns_by_tile_k": duration_by_k,
        "reference_path": str(
            AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE),
        "reference_sha256": (
            AIC_MATMUL_V3_SPLIT_K_RESIDENT_TILE_REFERENCE_SHA256),
    }


def load_aic_gemm_exact_transition_reference(
        repository_root, reference, reference_sha256):
    """Load exact finite-to-steady MTE2 programs from one campaign."""

    path = repository_root / reference
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference_sha256:
        raise ValueError(
            "AIC MatMulV2 finite transition reference SHA256 mismatch")
    value = json.loads(payload)
    owner = f"AIC MatMulV2 finite transition reference {reference.name}"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "validated_domain", "source_programs", "captures",
         "production_or_evaluation_duration_consumed"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_matmul_v2_finite_transition.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_matmul_v2_source_pair_finite_to_steady_validation_v1"),
        "duration_unit": "ns_per_completed_source_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    domain = value["validated_domain"]
    require_exact_fields(
        f"{owner}.validated_domain", domain,
        {"maximum_completion_pairs_per_worker",
         "maximum_source_descriptors_per_worker"})
    for name, field in domain.items():
        if isinstance(field, bool) or not isinstance(field, int) or field <= 0:
            raise ValueError(
                f"{owner}.validated_domain.{name} must be positive")
    program_fields = {
        "active_aic_groups", "m_groups", "n_groups", "base_tile_m",
        "base_tile_n", "base_tile_k", "a_rows", "a_columns", "b_rows",
        "b_columns", "source_row_stride_elements",
        "output_tiles_per_worker", "k_slabs",
        "source_descriptors_per_worker", "completion_pairs_per_worker",
        "mte2_only_run_medians_ns",
        "mte2_mte1_completion_run_medians_ns",
        "mte2_only_median_of_run_medians_ns",
        "mte2_mte1_completion_median_of_run_medians_ns",
    }
    programs = value["source_programs"]
    if not isinstance(programs, list) or not programs:
        raise TypeError(f"{owner}.source_programs must be a list")
    normalized_programs = []
    for index, program in enumerate(programs):
        program_owner = f"{owner}.source_programs[{index}]"
        require_exact_fields(program_owner, program, program_fields)
        for name in program_fields - {
                "mte2_only_run_medians_ns",
                "mte2_mte1_completion_run_medians_ns",
                "mte2_only_median_of_run_medians_ns",
                "mte2_mte1_completion_median_of_run_medians_ns"}:
            field = program[name]
            if (isinstance(field, bool) or not isinstance(field, int)
                    or field <= 0):
                raise ValueError(f"{program_owner}.{name} must be positive")
        for samples_name, median_name in (
                ("mte2_only_run_medians_ns",
                 "mte2_only_median_of_run_medians_ns"),
                ("mte2_mte1_completion_run_medians_ns",
                 "mte2_mte1_completion_median_of_run_medians_ns")):
            samples = program[samples_name]
            if (not isinstance(samples, list) or len(samples) < 2
                    or any(isinstance(sample, bool)
                           or not isinstance(sample, (int, float))
                           or not math.isfinite(float(sample)) or sample <= 0
                           for sample in samples)):
                raise ValueError(f"{program_owner}.{samples_name} is invalid")
            if program[median_name] != statistics.median(samples):
                raise ValueError(
                    f"{program_owner}.{median_name} is not the median")
        mte2_ns = program["mte2_only_median_of_run_medians_ns"]
        completion_ns = program[
            "mte2_mte1_completion_median_of_run_medians_ns"]
        if completion_ns < mte2_ns:
            raise ValueError(
                f"{program_owner} completion precedes MTE2 issue")
        normalized_programs.append({
            "dtype": "bf16",
            "operand_buffer_depth": 2,
            **program,
            "mte2_only_ns": float(mte2_ns),
            "completion_ns": float(completion_ns),
            "reference_path": str(reference),
            "reference_sha256": reference_sha256,
        })
    if domain["maximum_completion_pairs_per_worker"] != max(
            program["completion_pairs_per_worker"]
            for program in programs):
        raise ValueError(f"{owner} pair domain is inconsistent")
    if domain["maximum_source_descriptors_per_worker"] != max(
            program["source_descriptors_per_worker"]
            for program in programs):
        raise ValueError(f"{owner} descriptor domain is inconsistent")
    captures = value["captures"]
    if not isinstance(captures, list) or not captures:
        raise TypeError(f"{owner}.captures must be a list")
    for index, capture in enumerate(captures):
        capture_owner = f"{owner}.captures[{index}]"
        require_exact_fields(
            capture_owner, capture,
            {"measurement_baseline_commit", "source_snapshot_sha256",
             "binary_sha256", "raw_paths", "raw_sha256"})
        commit = capture["measurement_baseline_commit"]
        digests = (
            capture["source_snapshot_sha256"], capture["binary_sha256"],
            *capture["raw_sha256"])
        if (not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(
                f"{capture_owner}.measurement_baseline_commit is invalid")
        raw_paths = capture["raw_paths"]
        raw_hashes = capture["raw_sha256"]
        if (not isinstance(raw_paths, list) or not raw_paths
                or not isinstance(raw_hashes, list)
                or len(raw_paths) != len(raw_hashes)):
            raise ValueError(
                f"{capture_owner} raw evidence lists are inconsistent")
        if any(not isinstance(raw_path, str)
               or not raw_path.startswith(
                   "profiler_output/microbench/ascend_910b/")
               or ".." in raw_path.split("/") for raw_path in raw_paths):
            raise ValueError(f"{capture_owner}.raw_paths is outside its owner")
        if any(not isinstance(digest, str) or len(digest) != 64
               or any(character not in "0123456789abcdef"
                      for character in digest) for digest in digests):
            raise ValueError(f"{capture_owner} evidence SHA256 is invalid")
    return normalized_programs


def load_aic_gemm_finite_ingress_completion_service(repository_root):
    """Load one geometry-independent source-pair completion service."""

    references = (
        (AIC_MATMUL_V2_FINITE_INGRESS_COMPLETION_REFERENCE,
         AIC_MATMUL_V2_FINITE_INGRESS_COMPLETION_REFERENCE_SHA256),
        (AIC_MATMUL_V2_K14336_M128_N128_FINITE_INGRESS_COMPLETION_REFERENCE,
         AIC_MATMUL_V2_K14336_M128_N128_FINITE_INGRESS_COMPLETION_REFERENCE_SHA256),
    )
    expected = {
        "schema_version": (
            "midlayer.ascend_910b."
            "aic_matmul_v2_finite_ingress_completion.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_matmul_v2_source_pair_finite_completion_v1",
        "duration_unit": "ns_per_source_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    source_fields = {
        "dtype", "active_aic_groups", "operand_buffer_depth",
        "base_tile_m", "base_tile_n", "base_tile_k", "mte1_base_k",
        "a_rows", "a_columns", "b_rows", "b_columns",
        "minimum_completion_pairs_per_worker",
        "maximum_completion_pairs_per_worker",
    }
    loaded = []
    for reference, reference_sha256 in references:
        path = repository_root / reference
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference_sha256:
            raise ValueError(
                "AIC MatMulV2 finite ingress reference SHA256 mismatch")
        value = json.loads(payload)
        owner = f"AIC MatMulV2 finite ingress reference {reference.name}"
        require_exact_fields(
            owner, value,
            {"schema_version", "hardware_id", "software_revision",
             "mechanism", "duration_unit", "source_program",
             "affine_service", "validation_cells", "holdout", "capture",
             "production_or_evaluation_duration_consumed"})
        for name, expected_value in expected.items():
            if value[name] != expected_value:
                raise ValueError(
                    f"{owner}.{name} must be {expected_value!r}")
        source = value["source_program"]
        require_exact_fields(
            f"{owner}.source_program", source, source_fields)
        if source["dtype"] != "bf16":
            raise ValueError(f"{owner}.source_program.dtype must be 'bf16'")
        groups = source["active_aic_groups"]
        if (not isinstance(groups, list) or not groups
                or groups != sorted(set(groups))
                or any(isinstance(group, bool) or not isinstance(group, int)
                       or group <= 0 or group > 24 for group in groups)):
            raise ValueError(f"{owner}.active_aic_groups changed")
        normalized_source = {**source, "active_aic_groups": tuple(groups)}
        for name in source_fields - {"dtype", "active_aic_groups"}:
            field = source[name]
            if (isinstance(field, bool) or not isinstance(field, int)
                    or field <= 0):
                raise ValueError(
                    f"{owner}.source_program.{name} must be positive")
        if (source["minimum_completion_pairs_per_worker"]
                > source["maximum_completion_pairs_per_worker"]):
            raise ValueError(f"{owner} completion-pair interval is invalid")
        affine = value["affine_service"]
        require_exact_fields(
            f"{owner}.affine_service", affine,
            {"startup_ns", "ns_per_completion_pair", "fit_cells"})
        for name in ("startup_ns", "ns_per_completion_pair"):
            field = affine[name]
            if (isinstance(field, bool) or not isinstance(field, (int, float))
                    or not math.isfinite(float(field)) or field <= 0):
                raise ValueError(
                    f"{owner}.affine_service.{name} must be positive")
        fit_cells = affine["fit_cells"]
        if not isinstance(fit_cells, list) or len(fit_cells) != 3:
            raise ValueError(f"{owner}.fit_cells changed")
        if not isinstance(value["validation_cells"], list) or not value[
                "validation_cells"]:
            raise ValueError(f"{owner}.validation_cells changed")
        if not isinstance(value["holdout"], dict):
            raise TypeError(f"{owner}.holdout must be a mapping")
        capture = value["capture"]
        require_exact_fields(
            f"{owner}.capture", capture,
            {"measurement_baseline_commit", "source_snapshot_sha256",
             "binary_sha256", "sweep_raw_path", "sweep_raw_sha256",
             "holdout_raw_path", "holdout_raw_sha256"})
        commit = capture["measurement_baseline_commit"]
        if (not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(
                f"{owner}.capture.measurement_baseline_commit must be "
                "lowercase hex")
        for name in ("source_snapshot_sha256", "binary_sha256",
                     "sweep_raw_sha256", "holdout_raw_sha256"):
            digest = capture[name]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(f"{owner}.capture.{name} must be SHA256 hex")
        for name in ("sweep_raw_path", "holdout_raw_path"):
            raw_path = capture[name]
            if (not isinstance(raw_path, str)
                    or not raw_path.startswith(
                        "profiler_output/microbench/ascend_910b/")
                    or ".." in raw_path.split("/")):
                raise ValueError(
                    f"{owner}.capture.{name} is outside its owner")
        loaded.append({
            "source_program": normalized_source,
            "startup_ns": float(affine["startup_ns"]),
            "ns_per_completion_pair": float(
                affine["ns_per_completion_pair"]),
            "reference_path": str(reference),
            "reference_sha256": reference_sha256,
        })
    transition_path = (
        repository_root / AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE)
    transition_payload = transition_path.read_bytes()
    if hashlib.sha256(transition_payload).hexdigest() != (
            AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE_SHA256):
        raise ValueError(
            "AIC MatMulV2 finite transition reference SHA256 mismatch")
    transition = json.loads(transition_payload)
    transition_owner = "AIC MatMulV2 finite transition reference"
    require_exact_fields(
        transition_owner, transition,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "validated_domain", "source_programs", "captures",
         "production_or_evaluation_duration_consumed"})
    transition_expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_matmul_v2_finite_transition.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_matmul_v2_source_pair_finite_to_steady_validation_v1"),
        "duration_unit": "ns_per_completed_source_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in transition_expected.items():
        if transition[name] != expected_value:
            raise ValueError(
                f"{transition_owner}.{name} must be {expected_value!r}")
    domain = transition["validated_domain"]
    require_exact_fields(
        f"{transition_owner}.validated_domain", domain,
        {"maximum_completion_pairs_per_worker",
         "maximum_source_descriptors_per_worker"})
    for name, value in domain.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{transition_owner}.validated_domain.{name} must be "
                "positive")
    programs = transition["source_programs"]
    if not isinstance(programs, list) or not programs:
        raise TypeError(f"{transition_owner}.source_programs must be a list")
    program_fields = {
        "active_aic_groups", "m_groups", "n_groups", "base_tile_m",
        "base_tile_n", "base_tile_k", "a_rows", "a_columns", "b_rows",
        "b_columns", "source_row_stride_elements",
        "output_tiles_per_worker", "k_slabs",
        "source_descriptors_per_worker", "completion_pairs_per_worker",
        "mte2_only_run_medians_ns",
        "mte2_mte1_completion_run_medians_ns",
        "mte2_only_median_of_run_medians_ns",
        "mte2_mte1_completion_median_of_run_medians_ns",
    }
    for index, program in enumerate(programs):
        owner = f"{transition_owner}.source_programs[{index}]"
        require_exact_fields(owner, program, program_fields)
        for name in program_fields - {
                "mte2_only_run_medians_ns",
                "mte2_mte1_completion_run_medians_ns",
                "mte2_only_median_of_run_medians_ns",
                "mte2_mte1_completion_median_of_run_medians_ns"}:
            value = program[name]
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value <= 0:
                raise ValueError(f"{owner}.{name} must be positive")
        for samples_name, median_name in (
                ("mte2_only_run_medians_ns",
                 "mte2_only_median_of_run_medians_ns"),
                ("mte2_mte1_completion_run_medians_ns",
                 "mte2_mte1_completion_median_of_run_medians_ns")):
            samples = program[samples_name]
            if (not isinstance(samples, list) or len(samples) < 2
                    or any(isinstance(value, bool)
                           or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) or value <= 0
                           for value in samples)):
                raise ValueError(f"{owner}.{samples_name} is invalid")
            median = program[median_name]
            if median != statistics.median(samples):
                raise ValueError(f"{owner}.{median_name} is not the median")
        if (program["mte2_mte1_completion_median_of_run_medians_ns"]
                < program["mte2_only_median_of_run_medians_ns"]):
            raise ValueError(f"{owner} completion precedes MTE2 issue")
    if domain["maximum_completion_pairs_per_worker"] != max(
            program["completion_pairs_per_worker"] for program in programs):
        raise ValueError(f"{transition_owner} pair domain is inconsistent")
    if domain["maximum_source_descriptors_per_worker"] != max(
            program["source_descriptors_per_worker"] for program in programs):
        raise ValueError(
            f"{transition_owner} descriptor domain is inconsistent")
    captures = transition["captures"]
    if not isinstance(captures, list) or not captures:
        raise TypeError(f"{transition_owner}.captures must be a list")
    for index, capture in enumerate(captures):
        owner = f"{transition_owner}.captures[{index}]"
        require_exact_fields(
            owner, capture,
            {"measurement_baseline_commit", "source_snapshot_sha256",
             "binary_sha256", "raw_paths", "raw_sha256"})
        commit = capture["measurement_baseline_commit"]
        digests = (
            capture["source_snapshot_sha256"], capture["binary_sha256"],
            *capture["raw_sha256"])
        if (not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(f"{owner}.measurement_baseline_commit is invalid")
        if (not isinstance(capture["raw_paths"], list)
                or not capture["raw_paths"]
                or len(capture["raw_paths"]) != len(capture["raw_sha256"])):
            raise ValueError(f"{owner} raw evidence lists are inconsistent")
        if any(not isinstance(path, str)
               or not path.startswith("profiler_output/microbench/ascend_910b/")
               or ".." in path.split("/")
               for path in capture["raw_paths"]):
            raise ValueError(f"{owner}.raw_paths is outside its owner")
        if any(not isinstance(digest, str) or len(digest) != 64
               or any(character not in "0123456789abcdef"
                      for character in digest)
               for digest in digests):
            raise ValueError(f"{owner} evidence SHA256 is invalid")

    exact_transition_programs = []
    for program in programs:
        exact_transition_programs.append({
            "dtype": "bf16",
            "operand_buffer_depth": 2,
            **program,
            "mte2_only_ns": float(
                program["mte2_only_median_of_run_medians_ns"]),
            "completion_ns": float(
                program[
                    "mte2_mte1_completion_median_of_run_medians_ns"]),
            "reference_path": str(
                AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE),
            "reference_sha256": (
                AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE_SHA256),
        })
    exact_transition_programs.extend(
        load_aic_gemm_exact_transition_reference(
            repository_root,
            AIC_MATMUL_V2_K8192_FINITE_TRANSITION_REFERENCE,
            AIC_MATMUL_V2_K8192_FINITE_TRANSITION_REFERENCE_SHA256))
    exact_transition_programs.extend(
        load_aic_gemm_exact_transition_reference(
            repository_root,
            AIC_MATMUL_V2_K13824_FINITE_TRANSITION_REFERENCE,
            AIC_MATMUL_V2_K13824_FINITE_TRANSITION_REFERENCE_SHA256))
    exact_transition_services = {}
    for program in exact_transition_programs:
        signature = tuple(
            program[name]
            for name in AIC_GEMM_FINITE_TRANSITION_SIGNATURE_FIELDS)
        if signature in exact_transition_services:
            raise ValueError(
                "AIC MatMulV2 finite transition signature is duplicated")
        exact_transition_services[signature] = program

    calibration, cross_geometry_validation = loaded
    calibration_rate = calibration["ns_per_completion_pair"]
    validation_rate = cross_geometry_validation["ns_per_completion_pair"]
    relative_rate_delta = abs(
        validation_rate - calibration_rate) / calibration_rate
    if relative_rate_delta > 0.02:
        raise ValueError(
            "AIC finite ingress completion is not geometry-independent")
    return {
        "model": expected["mechanism"],
        "startup_ns": calibration["startup_ns"],
        "ns_per_completion_pair": calibration_rate,
        "calibration": calibration,
        "cross_geometry_validation": cross_geometry_validation,
        "maximum_cross_geometry_rate_delta": 0.02,
        "maximum_validated_completion_pairs_per_worker": domain[
            "maximum_completion_pairs_per_worker"],
        "maximum_validated_source_descriptors_per_worker": domain[
            "maximum_source_descriptors_per_worker"],
        "transition_reference_path": str(
            AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE),
        "transition_reference_sha256": (
            AIC_MATMUL_V2_K5120_FINITE_TRANSITION_REFERENCE_SHA256),
        "exact_transition_services": exact_transition_services,
    }


def load_aic_gemm_short_pipeline_completion_service(repository_root):
    """Load one source-shaped finite MTE2/MTE1/Cube completion service."""

    path = repository_root / AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE_SHA256):
        raise ValueError("AIC short GEMM pipeline reference SHA256 mismatch")
    value = json.loads(payload)
    owner = "AIC short GEMM pipeline reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "source_program", "affine_service",
         "measurements", "validation", "capture", "interpretation",
         "production_or_evaluation_duration_consumed", "forward_eligible"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_gemm_short_pipeline_completion.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_matmul_v2_mte2_mte1_cube_finite_completion_v1",
        "duration_unit": "ns_per_source_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    source = value["source_program"]
    source_fields = {
        "dtype", "active_aic_groups", "operand_buffer_depth",
        "base_tile_m", "base_tile_n", "base_tile_k", "a_rows",
        "a_columns", "b_rows", "b_columns", "source_row_stride_elements",
        "m_groups", "n_groups", "minimum_output_tiles_per_worker",
        "maximum_output_tiles_per_worker", "k_slabs",
        "a_descriptors_per_output_tile", "b_descriptors_per_output_tile",
        "completion_pairs_per_output_tile",
    }
    require_exact_fields(f"{owner}.source_program", source, source_fields)
    if source["dtype"] != "bf16":
        raise ValueError(f"{owner}.source_program.dtype must be 'bf16'")
    for name in source_fields - {"dtype"}:
        field = source[name]
        if isinstance(field, bool) or not isinstance(field, int) or field <= 0:
            raise ValueError(f"{owner}.source_program.{name} must be positive")
    if source["minimum_output_tiles_per_worker"] > source[
            "maximum_output_tiles_per_worker"]:
        raise ValueError(f"{owner} output-tile domain is invalid")
    affine = value["affine_service"]
    require_exact_fields(
        f"{owner}.affine_service", affine,
        {"startup_ns", "ns_per_output_tile", "derivation"})
    for name in ("startup_ns", "ns_per_output_tile"):
        field = affine[name]
        if (isinstance(field, bool) or not isinstance(field, (int, float))
                or not math.isfinite(float(field)) or field <= 0):
            raise ValueError(f"{owner}.affine_service.{name} must be positive")
    if affine["derivation"] != (
            "first_cell_minus_median_adjacent_delta_plus_"
            "median_adjacent_delta_per_output_tile"):
        raise ValueError(f"{owner}.affine_service.derivation changed")
    measurements = value["measurements"]
    expected_tiles = list(range(
        source["minimum_output_tiles_per_worker"],
        source["maximum_output_tiles_per_worker"] + 1))
    if (not isinstance(measurements, list)
            or [cell["output_tiles_per_worker"] for cell in measurements]
                != expected_tiles):
        raise ValueError(f"{owner}.measurements changed its finite domain")
    sample_fields = (
        "mte2_only", "mte2_mte1_completion",
        "mte2_mte1_cube_completion")
    cube_medians = []
    for index, cell in enumerate(measurements):
        cell_owner = f"{owner}.measurements[{index}]"
        require_exact_fields(
            cell_owner, cell,
            {"output_tiles_per_worker"}
            | {f"{name}_run_medians_ns" for name in sample_fields}
            | {f"{name}_median_of_run_medians_ns" for name in sample_fields})
        for name in sample_fields:
            samples = cell[f"{name}_run_medians_ns"]
            median = cell[f"{name}_median_of_run_medians_ns"]
            if (not isinstance(samples, list) or len(samples) < 2
                    or any(isinstance(sample, bool)
                           or not isinstance(sample, (int, float))
                           or not math.isfinite(float(sample)) or sample <= 0
                           for sample in samples)
                    or median != statistics.median(samples)):
                raise ValueError(f"{cell_owner}.{name} samples are invalid")
        cube_medians.append(float(
            cell["mte2_mte1_cube_completion_median_of_run_medians_ns"]))
    validation = value["validation"]
    require_exact_fields(
        f"{owner}.validation", validation,
        {"adjacent_cube_completion_deltas_ns",
         "maximum_affine_relative_error",
         "cube_increment_over_mte1_completion_ns"})
    adjacent = [new - old for old, new in zip(cube_medians, cube_medians[1:])]
    if validation["adjacent_cube_completion_deltas_ns"] != adjacent:
        raise ValueError(f"{owner} adjacent completion deltas are inconsistent")
    predicted = [
        float(affine["startup_ns"])
        + tiles * float(affine["ns_per_output_tile"])
        for tiles in expected_tiles]
    maximum_error = max(
        abs(prediction - observed) / observed
        for prediction, observed in zip(predicted, cube_medians))
    if not math.isclose(
            validation["maximum_affine_relative_error"], maximum_error,
            rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{owner} affine validation error is inconsistent")
    capture = value["capture"]
    require_exact_fields(
        f"{owner}.capture", capture,
        {"measurement_baseline_commit", "physical_device_id", "container",
         "source_snapshot_sha256", "binary_sha256", "raw_path",
         "raw_sha256", "runner_stderr_path", "runner_stderr_sha256"})
    commit = capture["measurement_baseline_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError(f"{owner}.capture commit is invalid")
    for name in (
            "source_snapshot_sha256", "binary_sha256", "raw_sha256",
            "runner_stderr_sha256"):
        digest = capture[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.capture.{name} is invalid")
    for name in ("raw_path", "runner_stderr_path"):
        evidence_path = capture[name]
        if (not isinstance(evidence_path, str)
                or not evidence_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in evidence_path.split("/")):
            raise ValueError(f"{owner}.capture.{name} is outside its owner")
    interpretation = value["interpretation"]
    require_exact_fields(
        f"{owner}.interpretation", interpretation,
        {"forward_use", "reason"})
    if interpretation["forward_use"] != (
            "max_floor_for_matching_normalized_source_program"):
        raise ValueError(f"{owner}.interpretation.forward_use changed")
    return {
        **source,
        "startup_ns": float(affine["startup_ns"]),
        "ns_per_output_tile": float(affine["ns_per_output_tile"]),
        "reference_path": str(
            AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE),
        "reference_sha256": (
            AIC_GEMM_K1536_SHORT_PIPELINE_COMPLETION_REFERENCE_SHA256),
    }


def load_aic_mte2_gemm_persistent_issue_service(repository_root):
    """Load one exact persistent MatMulV2 MTE2 issue program."""

    path = repository_root / AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE_SHA256):
        raise ValueError(
            "AIC GEMM persistent issue reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC GEMM persistent issue reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "source_program", "capture", "measurements",
         "production_or_evaluation_duration_consumed"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_mte2_gemm_persistent_issue.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_matmul_v2_persistent_mte2_issue_v1",
        "duration_unit": "ns_per_completed_source_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    expected_program = {
        "dtype": "bf16",
        "active_aic_groups": 24,
        "worker_grid_m": 3,
        "worker_grid_n": 8,
        "operand_buffer_depth": 2,
        "base_tile_m": 256,
        "base_tile_n": 128,
        "base_tile_k": 64,
        "a_rows": 256,
        "a_columns": 256,
        "a_source_row_stride_elements": 12288,
        "b_rows": 128,
        "b_columns": 256,
        "b_source_row_stride_elements": 12288,
        "output_tiles_per_worker": 12,
        "k_slabs": 48,
        "mte2_programs_per_worker": 576,
    }
    if value["source_program"] != expected_program:
        raise ValueError(f"{owner}.source_program changed")
    capture = value["capture"]
    require_exact_fields(
        f"{owner}.capture", capture,
        {"measurement_baseline_commit", "source_commit",
         "source_snapshot_sha256", "source_mte2_kernel_sha256",
         "source_ingress_kernel_sha256", "source_main_sha256",
         "source_runner_sha256", "binary_sha256", "physical_device_id",
         "run_count", "trials_per_run", "raw_path", "raw_sha256"})
    if (isinstance(capture["physical_device_id"], bool)
            or not isinstance(capture["physical_device_id"], int)
            or capture["physical_device_id"] < 0
            or capture["run_count"] != 3
            or capture["trials_per_run"] != 8):
        raise ValueError(f"{owner}.capture campaign changed")
    if (not isinstance(capture["raw_path"], str)
            or not capture["raw_path"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in capture["raw_path"].split("/")):
        raise ValueError(f"{owner}.capture raw path is outside its owner")
    evidence_hashes = tuple(
        capture[name] for name in (
            "measurement_baseline_commit", "source_commit",
            "source_snapshot_sha256", "source_mte2_kernel_sha256",
            "source_ingress_kernel_sha256", "source_main_sha256",
            "source_runner_sha256", "binary_sha256", "raw_sha256"))
    if any(not isinstance(digest, str) or len(digest) not in (40, 64)
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner}.capture evidence hash is invalid")
    measurements = value["measurements"]
    require_exact_fields(
        f"{owner}.measurements", measurements,
        {"mte2_only_run_medians_ns",
         "mte2_mte1_completion_run_medians_ns",
         "mte2_only_median_of_run_medians_ns",
         "mte2_mte1_completion_median_of_run_medians_ns"})
    issue_runs = measurements["mte2_only_run_medians_ns"]
    completion_runs = measurements[
        "mte2_mte1_completion_run_medians_ns"]
    if (not isinstance(issue_runs, list) or len(issue_runs) != 3
            or not isinstance(completion_runs, list)
            or len(completion_runs) != 3):
        raise ValueError(f"{owner}.measurements requires three run medians")
    issue_ns = measurements["mte2_only_median_of_run_medians_ns"]
    completion_ns = measurements[
        "mte2_mte1_completion_median_of_run_medians_ns"]
    measured_values = tuple(issue_runs) + tuple(completion_runs) + (
        issue_ns, completion_ns)
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(float(item)) or item <= 0
           for item in measured_values):
        raise ValueError(f"{owner}.measurements must be finite and positive")
    if (float(issue_ns) != float(sorted(issue_runs)[1])
            or float(completion_ns) != float(sorted(completion_runs)[1])
            or issue_ns >= completion_ns):
        raise ValueError(f"{owner}.measurements medians are inconsistent")
    return {
        "model": expected["mechanism"],
        **expected_program,
        "mte2_issue_ns": float(issue_ns),
        "ingress_completion_ns": float(completion_ns),
        "reference_path": str(
            AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE),
        "reference_sha256": (
            AIC_MTE2_GEMM_M2220_PERSISTENT_ISSUE_REFERENCE_SHA256),
    }


def load_aic_matmul_v2_dynamic_m_tail_control_service(repository_root):
    """Load the isolated MatMulV2 partial-M Scalar control service."""

    path = repository_root / AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE_SHA256):
        raise ValueError(
            "AIC MatMulV2 dynamic M-tail control reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MatMulV2 dynamic M-tail control reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "source_program", "capture", "measurements",
         "interpretation", "production_or_evaluation_duration_consumed"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b."
            "aic_matmul_v2_dynamic_m_tail_control.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_matmul_v2_dynamic_m_tail_scalar_control_v1",
        "duration_unit": "ns_per_tail_output_tile_k_tile_pair",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    source = value["source_program"]
    require_exact_fields(
        f"{owner}.source_program", source,
        {"dtype", "active_aic_workers", "base_tile_m", "base_tile_n",
         "base_tile_k", "projection_k", "output_tiles_per_worker",
         "k_tiles_per_output_tile", "maximum_partial_m_descriptor_k",
         "dynamic_control_logical_m",
         "static_control_logical_m", "dynamic_enable_set_tail",
         "static_enable_set_tail"})
    if source != {
            "dtype": "bf16", "active_aic_workers": 22,
            "base_tile_m": 48, "base_tile_n": 96, "base_tile_k": 64,
            "projection_k": 2048, "output_tiles_per_worker": 2,
            "k_tiles_per_output_tile": 32,
            "maximum_partial_m_descriptor_k": 256,
            "dynamic_control_logical_m": 48,
            "static_control_logical_m": 48,
            "dynamic_enable_set_tail": True,
            "static_enable_set_tail": False}:
        raise ValueError(f"{owner}.source_program changed")
    capture = value["capture"]
    require_exact_fields(
        f"{owner}.capture", capture,
        {"measurement_baseline_commit", "source_snapshot_sha256",
         "kernel_source_sha256", "host_source_sha256", "binary_sha256",
         "physical_device_id", "profiled_launch_index", "dynamic_pipe_path",
         "dynamic_pipe_sha256", "static_pipe_path", "static_pipe_sha256"})
    if (isinstance(capture["physical_device_id"], bool)
            or not isinstance(capture["physical_device_id"], int)
            or capture["physical_device_id"] < 0
            or capture["profiled_launch_index"] != 2):
        raise ValueError(f"{owner}.capture campaign changed")
    for name in (
            "measurement_baseline_commit", "source_snapshot_sha256",
            "kernel_source_sha256", "host_source_sha256", "binary_sha256",
            "dynamic_pipe_sha256", "static_pipe_sha256"):
        digest = capture[name]
        if (not isinstance(digest, str) or len(digest) not in (40, 64)
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.capture.{name} is invalid")
    for name in ("dynamic_pipe_path", "static_pipe_path"):
        raw_path = capture[name]
        if (not isinstance(raw_path, str)
                or not raw_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in raw_path.split("/")):
            raise ValueError(f"{owner}.capture.{name} is outside its owner")
    measurements = value["measurements"]
    require_exact_fields(
        f"{owner}.measurements", measurements,
        {"dynamic_scalar_worker_median_us",
         "static_scalar_worker_median_us", "dynamic_tail_scalar_delta_ns",
         "tail_output_tile_k_tile_pairs",
         "service_ns_per_tail_output_tile_k_tile_pair",
         "partial_m_step4_program_median_ns",
         "partial_m_step8_program_median_ns"})
    dynamic_us = measurements["dynamic_scalar_worker_median_us"]
    static_us = measurements["static_scalar_worker_median_us"]
    delta_ns = measurements["dynamic_tail_scalar_delta_ns"]
    requests = measurements["tail_output_tile_k_tile_pairs"]
    service_ns = measurements[
        "service_ns_per_tail_output_tile_k_tile_pair"]
    step4_ns = measurements["partial_m_step4_program_median_ns"]
    step8_ns = measurements["partial_m_step8_program_median_ns"]
    if (any(isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) or item <= 0
            for item in (dynamic_us, static_us, delta_ns, requests, service_ns,
                         step4_ns, step8_ns))
            or not isinstance(requests, int)):
        raise ValueError(f"{owner}.measurements must be finite and positive")
    if (not math.isclose(delta_ns, (dynamic_us - static_us) * 1000.0,
                         rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(service_ns, delta_ns / requests,
                                rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError(f"{owner}.measurements are inconsistent")
    if step8_ns < step4_ns:
        raise ValueError(
            f"{owner}.measurements must not report a step-8 speedup")
    interpretation = value["interpretation"]
    if interpretation != {
            "request": (
                "one dynamic partial-M output tile at one base-K tile"),
            "dependency": (
                "serialized Scalar control completes before the "
                "corresponding A-source MTE1 stage"),
            "full_or_aligned_tile_requests": 0,
            "descriptor_k_interpretation": (
                "the partial-M source program preserves at least one "
                "descriptor per 256 K elements when the A1 group widens "
                "from 256 to 512 K elements")}:
        raise ValueError(f"{owner}.interpretation changed")
    return {
        "model": expected["mechanism"],
        "service_ns_per_request": float(service_ns),
        "maximum_partial_m_descriptor_k": source[
            "maximum_partial_m_descriptor_k"],
        "reference_path": str(
            AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE),
        "reference_sha256": (
            AIC_MATMUL_V2_DYNAMIC_M_TAIL_CONTROL_REFERENCE_SHA256),
    }


def load_aic_matmul_v2_base_k_lifecycle_service(repository_root):
    """Load the isolated MatMulV2 base-K control-lifecycle bound."""

    path = repository_root / AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE_SHA256):
        raise ValueError(
            "AIC MatMulV2 base-K lifecycle reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MatMulV2 base-K lifecycle reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "source_program", "capture", "measurements",
         "interpretation", "production_or_evaluation_duration_consumed"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_matmul_v2_base_k_lifecycle.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_matmul_v2_base_k_control_lifecycle_v1",
        "duration_unit": "ns_per_worker",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    source = value["source_program"]
    require_exact_fields(
        f"{owner}.source_program", source,
        {"dtype", "active_aic_workers", "base_tile_m", "base_tile_n",
         "base_k_values", "projection_k_values", "output_tiles_per_worker",
         "a1_descriptor_k", "b1_descriptor_k", "operand_buffer_depth",
         "dynamic_m_tail"})
    if source != {
            "dtype": "bf16", "active_aic_workers": 24,
            "base_tile_m": 96, "base_tile_n": 128,
            "base_k_values": [32, 64, 128],
            "projection_k_values": [1024, 2048, 4096],
            "output_tiles_per_worker": 2,
            "a1_descriptor_k": 512, "b1_descriptor_k": 256,
            "operand_buffer_depth": 2, "dynamic_m_tail": False}:
        raise ValueError(f"{owner}.source_program changed")
    capture = value["capture"]
    require_exact_fields(
        f"{owner}.capture", capture,
        {"measurement_baseline_commit", "source_snapshot",
         "source_snapshot_sha256", "kernel_source_sha256",
         "host_source_sha256", "binary_sha256", "physical_device_id",
         "raw_directory", "raw_sha256_manifest_sha256",
         "independent_runs"})
    if (isinstance(capture["physical_device_id"], bool)
            or not isinstance(capture["physical_device_id"], int)
            or capture["physical_device_id"] < 0
            or capture["independent_runs"] != 3):
        raise ValueError(f"{owner}.capture campaign changed")
    for name in (
            "measurement_baseline_commit", "source_snapshot_sha256",
            "kernel_source_sha256", "host_source_sha256", "binary_sha256",
            "raw_sha256_manifest_sha256"):
        digest = capture[name]
        if (not isinstance(digest, str) or len(digest) not in (40, 64)
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.capture.{name} is invalid")
    raw_directory = capture["raw_directory"]
    if (not isinstance(raw_directory, str)
            or not raw_directory.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_directory.split("/")):
        raise ValueError(f"{owner}.capture.raw_directory is outside its owner")
    measurements = value["measurements"]
    require_exact_fields(
        f"{owner}.measurements", measurements,
        {"worker_program_median_ns", "affine_model", "startup_ns",
         "common_ns_per_k", "base_k_control_ns", "r_squared",
         "maximum_relative_residual_pct"})
    expected_medians = {
        "1024": {"32": 18040.0, "64": 11040.0, "128": 8560.0},
        "2048": {"32": 35240.0, "64": 20600.0, "128": 15480.0},
        "4096": {"32": 67860.0, "64": 40010.0, "128": 30370.0},
    }
    if measurements["worker_program_median_ns"] != expected_medians:
        raise ValueError(f"{owner}.measurements medians changed")
    expected_affine = (
        "startup_ns + common_ns_per_k * projection_k + "
        "base_k_control_ns * output_tiles_per_worker * projection_k / base_k")
    if measurements["affine_model"] != expected_affine:
        raise ValueError(f"{owner}.measurements affine model changed")
    expected_parameters = {
        "startup_ns": 1393.3333333333064,
        "common_ns_per_k": 3.5996403769841407,
        "base_k_control_ns": 200.57823129251713,
        "r_squared": 0.997336124040642,
        "maximum_relative_residual_pct": 4.8660370296986475,
    }
    for name, expected_value in expected_parameters.items():
        measured = measurements[name]
        if (isinstance(measured, bool)
                or not isinstance(measured, (int, float))
                or not math.isclose(
                    float(measured), expected_value,
                    rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError(f"{owner}.measurements.{name} changed")
    interpretation = value["interpretation"]
    require_exact_fields(
        f"{owner}.interpretation", interpretation,
        {"forward_bound", "composition", "excluded_conclusions"})
    if (not isinstance(interpretation["excluded_conclusions"], list)
            or len(interpretation["excluded_conclusions"]) != 3
            or any(not isinstance(item, str) or not item
                   for item in interpretation["excluded_conclusions"])):
        raise ValueError(f"{owner}.interpretation is invalid")
    return {
        "model": expected["mechanism"],
        "dtype": source["dtype"],
        "active_aic_workers": source["active_aic_workers"],
        "base_tile_m": source["base_tile_m"],
        "base_tile_n": source["base_tile_n"],
        "base_k_values": tuple(source["base_k_values"]),
        "projection_k_values": tuple(source["projection_k_values"]),
        "output_tiles_per_worker": source["output_tiles_per_worker"],
        "a1_descriptor_k": source["a1_descriptor_k"],
        "b1_descriptor_k": source["b1_descriptor_k"],
        "operand_buffer_depth": source["operand_buffer_depth"],
        "dynamic_m_tail": source["dynamic_m_tail"],
        "startup_ns": float(measurements["startup_ns"]),
        "base_k_control_ns": float(measurements["base_k_control_ns"]),
        "reference_path": str(AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE),
        "reference_sha256": (
            AIC_MATMUL_V2_BASE_K_LIFECYCLE_REFERENCE_SHA256),
    }


def load_aic_matmul_v2_small_m_base_k_control_service(repository_root):
    """Load the cross-geometry validation of MatMulV2 Scalar control."""

    path = repository_root / AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE_SHA256):
        raise ValueError(
            "AIC MatMulV2 small-M base-K control reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MatMulV2 small-M base-K control reference"
    require_exact_fields(owner, value, {
        "schema_version", "hardware_id", "software_revision", "mechanism",
        "duration_unit", "source_program", "capture", "measurements",
        "validated_services", "forward_scope", "excluded_conclusions",
        "production_or_evaluation_duration_consumed",
    })
    expected_identity = {
        "schema_version": (
            "midlayer.ascend_910b.aic_matmul_v2_small_m_base_k_control.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_matmul_v2_small_m_scalar_source_control_validation_v1"),
        "duration_unit": "ns_per_output_tile_base_k_pair",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected in expected_identity.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} changed")
    source = value["source_program"]
    expected_source = {
        "dtype": "bf16",
        "base_tile_m": 16,
        "base_tile_n_values": [96, 112],
        "active_aic_workers_by_base_n": {"96": 22, "112": 23},
        "base_tile_k": 64,
        "projection_k_values": [1024, 2048],
        "output_tiles_per_worker": 2,
        "a1_descriptor_k": 512,
        "b1_descriptor_k": 512,
        "operand_buffer_depth": 2,
        "static_logical_m": 16,
        "dynamic_logical_m": 1,
    }
    if source != expected_source:
        raise ValueError(f"{owner}.source_program changed")
    capture = value["capture"]
    require_exact_fields(f"{owner}.capture", capture, {
        "measurement_baseline_commit", "source_snapshot",
        "source_snapshot_sha256", "source_files_sha256", "binary_sha256",
        "physical_device_id", "raw_directory",
        "raw_sha256_manifest_sha256", "independent_launches_per_cell",
        "worker_statistic",
    })
    if (isinstance(capture["physical_device_id"], bool)
            or not isinstance(capture["physical_device_id"], int)
            or capture["physical_device_id"] < 0
            or capture["independent_launches_per_cell"] != 3
            or capture["worker_statistic"] != "median"):
        raise ValueError(f"{owner}.capture campaign changed")
    for name in (
            "measurement_baseline_commit", "source_snapshot_sha256",
            "binary_sha256", "raw_sha256_manifest_sha256"):
        digest = capture[name]
        expected_length = 40 if name == "measurement_baseline_commit" else 64
        if (not isinstance(digest, str) or len(digest) != expected_length
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.capture.{name} is invalid")
    source_files = capture["source_files_sha256"]
    expected_source_files = {
        "CMakeLists.txt", "README.md",
        "matmul_v2_small_m_lifecycle_kernel.cpp",
        "matmul_v2_small_m_lifecycle_main.cpp",
    }
    if not isinstance(source_files, dict) or set(source_files) != (
            expected_source_files):
        raise ValueError(f"{owner}.capture source file set changed")
    if any(
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef"
                   for character in digest)
            for digest in source_files.values()):
        raise ValueError(f"{owner}.capture source SHA256 is invalid")
    raw_directory = capture["raw_directory"]
    if (not isinstance(raw_directory, str)
            or not raw_directory.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_directory.split("/")):
        raise ValueError(f"{owner}.capture.raw_directory is outside owner")
    measurements = value["measurements"]
    require_exact_fields(f"{owner}.measurements", measurements, {
        "scalar_worker_median_ns",
        "static_base_k_control_ns_per_pair_by_base_n",
        "static_base_k_control_ns_per_pair_mean",
        "dynamic_tail_delta_ns_per_pair_by_base_n",
        "dynamic_tail_delta_ns_per_pair_mean", "derivation",
    })
    scalar = measurements["scalar_worker_median_ns"]
    if not isinstance(scalar, dict) or set(scalar) != {"96", "112"}:
        raise ValueError(f"{owner}.measurements Scalar N domain changed")
    static_slopes = {}
    dynamic_slopes = {}
    for base_n in (96, 112):
        cell = scalar[str(base_n)]
        if not isinstance(cell, dict) or set(cell) != {"static", "dynamic"}:
            raise ValueError(f"{owner}.measurements Scalar mode changed")
        for mode in ("static", "dynamic"):
            points = cell[mode]
            if not isinstance(points, dict) or set(points) != {"1024", "2048"}:
                raise ValueError(f"{owner}.measurements Scalar K domain changed")
            if any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item)) or float(item) <= 0.0
                    for item in points.values()):
                raise ValueError(f"{owner}.measurements Scalar time invalid")
            slope = (
                float(points["2048"]) - float(points["1024"])) / 32.0
            if slope <= 0.0:
                raise ValueError(f"{owner}.measurements Scalar slope invalid")
            (static_slopes if mode == "static" else dynamic_slopes)[
                base_n] = slope
    declared_static = measurements[
        "static_base_k_control_ns_per_pair_by_base_n"]
    declared_dynamic_delta = measurements[
        "dynamic_tail_delta_ns_per_pair_by_base_n"]
    if set(declared_static) != {"96", "112"} or set(
            declared_dynamic_delta) != {"96", "112"}:
        raise ValueError(f"{owner}.measurements derived N domain changed")
    for base_n in (96, 112):
        if not math.isclose(
                float(declared_static[str(base_n)]), static_slopes[base_n],
                rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{owner}.measurements static slope changed")
        dynamic_delta = dynamic_slopes[base_n] - static_slopes[base_n]
        if not math.isclose(
                float(declared_dynamic_delta[str(base_n)]), dynamic_delta,
                rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{owner}.measurements dynamic delta changed")
    static_mean = math.fsum(static_slopes.values()) / len(static_slopes)
    dynamic_delta_mean = math.fsum(
        dynamic_slopes[base_n] - static_slopes[base_n]
        for base_n in static_slopes) / len(static_slopes)
    if (not math.isclose(
            float(measurements["static_base_k_control_ns_per_pair_mean"]),
            static_mean, rel_tol=0.0, abs_tol=1.0e-12)
            or not math.isclose(
                float(measurements["dynamic_tail_delta_ns_per_pair_mean"]),
                dynamic_delta_mean, rel_tol=0.0, abs_tol=1.0e-12)):
        raise ValueError(f"{owner}.measurements derived mean changed")
    validated = value["validated_services"]
    require_exact_fields(f"{owner}.validated_services", validated, {
        "base_k_control_ns_per_pair", "base_k_control_reference",
        "base_k_control_mean_relative_deviation_pct",
        "base_k_control_maximum_cell_relative_deviation_pct",
        "dynamic_tail_control_ns_per_pair", "dynamic_tail_control_reference",
        "dynamic_tail_mean_relative_deviation_pct",
    })
    base_service = float(validated["base_k_control_ns_per_pair"])
    tail_service = float(validated["dynamic_tail_control_ns_per_pair"])
    base_mean_deviation = abs(static_mean - base_service) / base_service * 100.0
    base_max_deviation = max(
        abs(slope - base_service) / base_service * 100.0
        for slope in static_slopes.values())
    tail_mean_deviation = (
        abs(dynamic_delta_mean - tail_service) / tail_service * 100.0)
    for name, derived in (
            ("base_k_control_mean_relative_deviation_pct",
             base_mean_deviation),
            ("base_k_control_maximum_cell_relative_deviation_pct",
             base_max_deviation),
            ("dynamic_tail_mean_relative_deviation_pct",
             tail_mean_deviation)):
        if not math.isclose(
                float(validated[name]), derived,
                rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{owner}.validated_services.{name} changed")
    if base_max_deviation >= 5.0 or tail_mean_deviation >= 5.0:
        raise ValueError(f"{owner} does not validate the retained services")
    expected_scope = {
        "projection_k_rule": "positive multiple of base_tile_k",
        "output_tiles_per_worker": 2,
        "dynamic_m_tail_required": True,
        "source_dependency": (
            "base control and dynamic-tail control serialize on the source "
            "Scalar stream before the corresponding MTE2/MTE1 issue; "
            "previously issued MTE1, Cube, and FixPipe work may overlap"),
    }
    if value["forward_scope"] != expected_scope:
        raise ValueError(f"{owner}.forward_scope changed")
    excluded = value["excluded_conclusions"]
    if (not isinstance(excluded, list) or len(excluded) != 3
            or any(not isinstance(item, str) or not item for item in excluded)):
        raise ValueError(f"{owner}.excluded_conclusions changed")
    return {
        "model": expected_identity["mechanism"],
        "dtype": source["dtype"],
        "base_tile_m": source["base_tile_m"],
        "active_aic_workers_by_base_n": {
            int(base_n): workers
            for base_n, workers in source[
                "active_aic_workers_by_base_n"].items()
        },
        "base_tile_k": source["base_tile_k"],
        "output_tiles_per_worker": source["output_tiles_per_worker"],
        "a1_descriptor_k": source["a1_descriptor_k"],
        "b1_descriptor_k": source["b1_descriptor_k"],
        "operand_buffer_depth": source["operand_buffer_depth"],
        "base_k_control_ns_per_pair": base_service,
        "reference_path": str(
            AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE),
        "reference_sha256": (
            AIC_MATMUL_V2_SMALL_M_BASE_K_CONTROL_REFERENCE_SHA256),
    }


def load_aic_mte2_gemm_region_completion_reference(
        repository_root, reference_path, reference_sha256,
        expected_program, owner, publish_mte2_only):
    """Load one exact MatMulV3 L2-region ingress program."""

    path = repository_root / reference_path
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != reference_sha256:
        raise ValueError("AIC GEMM region completion reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "source_program", "capture", "measurements",
         "production_or_evaluation_duration_consumed"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_mte2_gemm_region_completion.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_matmul_v3_l2_region_mte2_to_mte1_completion_v1"),
        "duration_unit": "ns_per_completed_region_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    source_program = value["source_program"]
    require_exact_fields(
        f"{owner}.source_program", source_program,
        {"dtype", "active_aic_groups", "operand_buffer_depth",
         "base_tile_m", "base_tile_n", "base_tile_k", "a_rows",
         "a_columns", "a_source_row_stride_elements", "b_rows",
         "b_columns", "b_source_row_stride_elements", "m_tiles", "n_tiles",
         "m_regions", "n_regions", "m_tiles_per_region",
         "n_tiles_per_region", "maximum_output_tiles_per_worker", "k_slabs",
         "mte2_programs_per_worker", "mte1_pairs_per_worker"})
    if source_program != expected_program:
        raise ValueError(f"{owner}.source_program changed")
    capture = value["capture"]
    require_exact_fields(
        f"{owner}.capture", capture,
        {"source_commit", "source_snapshot_sha256", "source_schedule_sha256",
         "source_mte2_kernel_sha256", "source_ingress_kernel_sha256",
         "source_main_sha256", "source_runner_sha256", "binary_sha256",
         "physical_device_id", "run_count", "trials_per_run", "raw_path",
         "raw_sha256"})
    if (isinstance(capture["physical_device_id"], bool)
            or not isinstance(capture["physical_device_id"], int)
            or capture["physical_device_id"] < 0
            or capture["run_count"] != 3
            or capture["trials_per_run"] != 8):
        raise ValueError(f"{owner}.capture campaign changed")
    raw_path = capture["raw_path"]
    if (not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")):
        raise ValueError(f"{owner}.capture raw path is outside its owner")
    evidence_hashes = tuple(
        capture[name] for name in (
            "source_snapshot_sha256", "source_schedule_sha256",
            "source_mte2_kernel_sha256", "source_ingress_kernel_sha256",
            "source_main_sha256", "source_runner_sha256", "binary_sha256",
            "raw_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner}.capture evidence SHA256 is invalid")
    measurements = value["measurements"]
    require_exact_fields(
        f"{owner}.measurements", measurements,
        {"mte2_only_run_medians_ns",
         "mte2_mte1_completion_run_medians_ns",
         "mte2_only_median_of_run_medians_ns",
         "mte2_mte1_completion_median_of_run_medians_ns"})
    completion_ns = measurements[
        "mte2_mte1_completion_median_of_run_medians_ns"]
    mte2_only_ns = measurements["mte2_only_median_of_run_medians_ns"]
    if (isinstance(completion_ns, bool)
            or not isinstance(completion_ns, (int, float))
            or not math.isfinite(float(completion_ns))
            or completion_ns <= 0
            or isinstance(mte2_only_ns, bool)
            or not isinstance(mte2_only_ns, (int, float))
            or not math.isfinite(float(mte2_only_ns))
            or mte2_only_ns <= 0
            or mte2_only_ns >= completion_ns):
        raise ValueError(
            f"{owner} MTE2 and completion spans must be finite and ordered")
    service = {
        "model": expected["mechanism"],
        **expected_program,
        "completion_ns": float(completion_ns),
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha256,
    }
    if publish_mte2_only:
        service["mte2_only_ns"] = float(mte2_only_ns)
    return service


def load_aic_mte2_gemm_region_completion_services(repository_root):
    """Load every exact measured MatMulV3 L2-region ingress program."""

    shared = {
        "dtype": "bf16", "active_aic_groups": 24,
        "operand_buffer_depth": 2,
        "base_tile_m": 128, "base_tile_n": 256, "base_tile_k": 64,
        "a_rows": 128, "a_columns": 512,
        "b_rows": 256, "b_columns": 256,
    }
    programs = (
        ({
            **shared,
            "a_source_row_stride_elements": 8192,
            "b_source_row_stride_elements": 8192,
            "m_tiles": 32, "n_tiles": 32,
            "m_regions": 4, "n_regions": 2,
            "m_tiles_per_region": 8, "n_tiles_per_region": 16,
            "maximum_output_tiles_per_worker": 43, "k_slabs": 16,
            "mte2_programs_per_worker": 688,
            "mte1_pairs_per_worker": 5504,
        }, AIC_MTE2_GEMM_REGION_COMPLETION_REFERENCE,
         AIC_MTE2_GEMM_REGION_COMPLETION_REFERENCE_SHA256,
         "AIC GEMM 4x2-region completion reference", False),
        ({
            **shared,
            "a_source_row_stride_elements": 5120,
            "b_source_row_stride_elements": 5120,
            "m_tiles": 32, "n_tiles": 54,
            "m_regions": 4, "n_regions": 2,
            "m_tiles_per_region": 8, "n_tiles_per_region": 27,
            "maximum_output_tiles_per_worker": 72, "k_slabs": 10,
            "mte2_programs_per_worker": 720,
            "mte1_pairs_per_worker": 5760,
        }, AIC_MTE2_GEMM_REGION_COMPLETION_8X27_REFERENCE,
         AIC_MTE2_GEMM_REGION_COMPLETION_8X27_REFERENCE_SHA256,
         "AIC GEMM 8x27-region completion reference", False),
        ({
            **shared,
            "a_source_row_stride_elements": 8192,
            "b_source_row_stride_elements": 8192,
            "m_tiles": 256, "n_tiles": 112,
            "m_regions": 32, "n_regions": 8,
            "m_tiles_per_region": 8, "n_tiles_per_region": 14,
            "maximum_output_tiles_per_worker": 1195, "k_slabs": 16,
            "mte2_programs_per_worker": 19120,
            "mte1_pairs_per_worker": 152960,
        }, AIC_MTE2_GEMM_REGION_COMPLETION_256X112_REFERENCE,
         AIC_MTE2_GEMM_REGION_COMPLETION_256X112_REFERENCE_SHA256,
         "AIC GEMM 32x8-region long completion reference", True),
        ({
            **shared,
            "a_source_row_stride_elements": 4096,
            "b_source_row_stride_elements": 4096,
            "m_tiles": 16, "n_tiles": 16,
            "m_regions": 1, "n_regions": 1,
            "m_tiles_per_region": 16, "n_tiles_per_region": 16,
            "maximum_output_tiles_per_worker": 11, "k_slabs": 8,
            "mte2_programs_per_worker": 88,
            "mte1_pairs_per_worker": 704,
        }, AIC_MTE2_GEMM_REGION_COMPLETION_16X16_REFERENCE,
         AIC_MTE2_GEMM_REGION_COMPLETION_16X16_REFERENCE_SHA256,
         "AIC GEMM 1x1-region completion reference", False),
        ({
            **shared,
            "a_source_row_stride_elements": 8192,
            "b_source_row_stride_elements": 8192,
            "m_tiles": 16, "n_tiles": 32,
            "m_regions": 2, "n_regions": 2,
            "m_tiles_per_region": 8, "n_tiles_per_region": 16,
            "maximum_output_tiles_per_worker": 22, "k_slabs": 16,
            "mte2_programs_per_worker": 352,
            "mte1_pairs_per_worker": 2816,
        }, AIC_MTE2_GEMM_REGION_COMPLETION_16X32_REFERENCE,
         AIC_MTE2_GEMM_REGION_COMPLETION_16X32_REFERENCE_SHA256,
         "AIC GEMM 2x2-region completion reference", False),
    )
    measured_services = tuple(
        load_aic_mte2_gemm_region_completion_reference(
            repository_root, reference_path, reference_sha256,
            expected_program, owner, publish_mte2_only)
        for (expected_program, reference_path, reference_sha256, owner,
             publish_mte2_only)
        in programs
    )
    return measured_services + (
        load_aic_mte2_gemm_region_tail_completion_services(repository_root))


def load_aic_mte2_gemm_region_tail_completion_services(repository_root):
    """Load tail-aware MatMulV3 region ingress measurements."""

    path = repository_root / AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE_SHA256):
        raise ValueError("AIC GEMM tail-region reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC GEMM tail-region completion reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "source_program", "cells", "capture",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_mte2_gemm_region_tail_completion.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_matmul_v3_l2_region_tail_mte2_to_mte1_completion_v1"),
        "duration_unit": "ns_per_completed_region_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    source = value["source_program"]
    expected_source = {
        "dtype": "bf16", "active_aic_groups": 24,
        "operand_buffer_depth": 2,
        "base_tile_m": 128, "base_tile_n": 256, "base_tile_k": 64,
        "a_rows": 128, "a_columns": 512,
        "b_rows": 256, "b_columns": 256,
        "mte1_base_pairs_per_slab": 8,
        "tail_rule": (
            "skip region-scheduled tiles outside the logical M/N tile extent"),
    }
    require_exact_fields(owner + ".source_program", source,
                          set(expected_source))
    if source != expected_source:
        raise ValueError(f"{owner}.source_program changed")
    expected_cells = {
        (16, 20, 1, 1, 16, 20, 10, 14),
        (16, 54, 2, 2, 8, 27, 10, 36),
        (32, 16, 4, 2, 8, 8, 28, 22),
        (32, 20, 4, 2, 8, 10, 27, 27),
        (256, 16, 36, 2, 8, 8, 28, 171),
    }
    fields = {
        "m_tiles", "n_tiles", "m_regions", "n_regions",
        "m_tiles_per_region", "n_tiles_per_region", "k_slabs",
        "maximum_output_tiles_per_worker", "mte2_programs_per_worker",
        "mte1_pairs_per_worker", "mte2_only_run_medians_ns",
        "mte2_mte1_completion_run_medians_ns",
        "mte2_only_median_of_run_medians_ns",
        "mte2_mte1_completion_median_of_run_medians_ns",
    }
    services = []
    signatures = set()
    for index, cell in enumerate(value["cells"]):
        cell_owner = f"{owner}.cells[{index}]"
        require_exact_fields(cell_owner, cell, fields)
        signature = tuple(cell[name] for name in (
            "m_tiles", "n_tiles", "m_regions", "n_regions",
            "m_tiles_per_region", "n_tiles_per_region", "k_slabs",
            "maximum_output_tiles_per_worker"))
        if signature not in expected_cells or signature in signatures:
            raise ValueError(f"{cell_owner} signature is invalid")
        signatures.add(signature)
        maximum_tiles = cell["maximum_output_tiles_per_worker"]
        k_slabs = cell["k_slabs"]
        if (cell["mte2_programs_per_worker"] != maximum_tiles * k_slabs
                or cell["mte1_pairs_per_worker"]
                != maximum_tiles * k_slabs * 8):
            raise ValueError(f"{cell_owner} work counts changed")
        medians = {}
        for mode in ("mte2_only", "mte2_mte1_completion"):
            runs = cell[f"{mode}_run_medians_ns"]
            median = cell[f"{mode}_median_of_run_medians_ns"]
            if (not isinstance(runs, list) or len(runs) != 3
                    or any(isinstance(item, bool)
                           or not isinstance(item, (int, float))
                           or not math.isfinite(float(item)) or item <= 0
                           for item in runs)
                    or median != sorted(runs)[1]):
                raise ValueError(f"{cell_owner}.{mode} medians are invalid")
            medians[mode] = float(median)
        if medians["mte2_mte1_completion"] <= medians["mte2_only"]:
            raise ValueError(f"{cell_owner} MTE1 completion must add service")
        services.append({
            "model": expected["mechanism"],
            **{name: source[name] for name in (
                "dtype", "active_aic_groups", "operand_buffer_depth",
                "base_tile_m", "base_tile_n", "base_tile_k", "a_rows",
                "a_columns", "b_rows", "b_columns")},
            "a_source_row_stride_elements": 512 * k_slabs,
            "b_source_row_stride_elements": 512 * k_slabs,
            **{name: cell[name] for name in fields
               if name not in {
                   "mte2_only_run_medians_ns",
                   "mte2_mte1_completion_run_medians_ns",
                   "mte2_only_median_of_run_medians_ns",
                   "mte2_mte1_completion_median_of_run_medians_ns"}},
            "mte2_only_ns": medians["mte2_only"],
            "completion_ns": medians["mte2_mte1_completion"],
            "reference_path": str(
                AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE),
            "reference_sha256": (
                AIC_MTE2_GEMM_REGION_TAIL_COMPLETION_REFERENCE_SHA256),
        })
    if signatures != expected_cells:
        raise ValueError(f"{owner} cell set changed")
    capture = value["capture"]
    require_exact_fields(
        owner + ".capture", capture,
        {"source_commit", "source_snapshot_path", "source_snapshot_sha256",
         "source_schedule_sha256", "source_mte2_kernel_sha256",
         "source_ingress_kernel_sha256", "source_main_sha256",
         "source_runner_sha256", "binary_sha256", "physical_device_id",
         "run_count", "trials_per_run", "raw_path", "raw_sha256",
         "stderr_sha256"})
    if (capture["source_commit"]
            != "48ed43d779c436da4a77aff10f3907be6348616d"
            or (isinstance(capture["physical_device_id"], bool)
                or not isinstance(capture["physical_device_id"], int)
                or capture["physical_device_id"] < 0)
            or capture["run_count"] != 3
            or capture["trials_per_run"] != 8):
        raise ValueError(f"{owner}.capture campaign changed")
    if (not capture["raw_path"].startswith(
            "profiler_output/microbench/ascend_910b/")
            or ".." in capture["raw_path"].split("/")):
        raise ValueError(f"{owner}.capture raw path is outside its owner")
    digests = tuple(capture[name] for name in (
        "source_snapshot_sha256", "source_schedule_sha256",
        "source_mte2_kernel_sha256", "source_ingress_kernel_sha256",
        "source_main_sha256", "source_runner_sha256", "binary_sha256",
        "raw_sha256", "stderr_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in digests):
        raise ValueError(f"{owner}.capture evidence SHA256 is invalid")
    return tuple(services)


def load_aic_mte2_gemm_region_pipeline_completion_service(repository_root):
    """Load one region program with its dependent Cube consumer."""

    path = repository_root / AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE_SHA256):
        raise ValueError("AIC GEMM region pipeline reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC GEMM region pipeline completion reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "source_program", "measurements", "capture",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable"})
    expected = {
        "schema_version": (
            "midlayer.ascend_910b."
            "aic_mte2_gemm_region_pipeline_completion.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_matmul_v3_l2_region_mte2_to_cube_completion_v1"),
        "duration_unit": "ns_per_completed_region_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    source = value["source_program"]
    expected_source = {
        "dtype": "bf16", "active_aic_groups": 24,
        "operand_buffer_depth": 2,
        "base_tile_m": 128, "base_tile_n": 256, "base_tile_k": 64,
        "a_rows": 128, "a_columns": 512,
        "a_source_row_stride_elements": 14336,
        "b_rows": 256, "b_columns": 256,
        "b_source_row_stride_elements": 14336,
        "m_tiles": 256, "n_tiles": 16,
        "m_regions": 36, "n_regions": 2,
        "m_tiles_per_region": 8, "n_tiles_per_region": 8,
        "maximum_output_tiles_per_worker": 171, "k_slabs": 28,
        "mte2_programs_per_worker": 4788,
        "mte1_pairs_per_worker": 38304,
    }
    require_exact_fields(owner + ".source_program", source,
                          set(expected_source))
    if source != expected_source:
        raise ValueError(f"{owner}.source_program changed")
    measurements = value["measurements"]
    modes = (
        "mte2_only", "mte2_mte1_completion",
        "mte2_mte1_cube_completion")
    measurement_fields = {
        f"{mode}_{suffix}"
        for mode in modes
        for suffix in (
            "run_medians_ns", "median_of_run_medians_ns")
    }
    require_exact_fields(owner + ".measurements", measurements,
                          measurement_fields)
    medians = {}
    for mode in modes:
        runs = measurements[f"{mode}_run_medians_ns"]
        median = measurements[f"{mode}_median_of_run_medians_ns"]
        if (not isinstance(runs, list) or len(runs) != 3
                or any(isinstance(item, bool)
                       or not isinstance(item, (int, float))
                       or not math.isfinite(float(item)) or item <= 0
                       for item in runs)
                or median != sorted(runs)[1]):
            raise ValueError(f"{owner}.{mode} medians are invalid")
        medians[mode] = float(median)
    if not (medians["mte2_only"] < medians["mte2_mte1_completion"]
            < medians["mte2_mte1_cube_completion"]):
        raise ValueError(f"{owner} pipeline spans must be ordered")
    capture = value["capture"]
    capture_fields = {
        "source_commit", "source_snapshot_path", "source_snapshot_sha256",
        "source_schedule_sha256", "source_mte2_kernel_sha256",
        "source_ingress_kernel_sha256", "source_pipeline_kernel_sha256",
        "source_main_sha256", "source_runner_sha256", "binary_sha256",
        "physical_device_id", "run_count", "trials_per_run", "raw_path",
        "raw_sha256", "stderr_sha256",
    }
    require_exact_fields(owner + ".capture", capture, capture_fields)
    if (capture["source_commit"]
            != "aa782f857d066163b11d3be1ecb2e413743dc455"
            or (isinstance(capture["physical_device_id"], bool)
                or not isinstance(capture["physical_device_id"], int)
                or capture["physical_device_id"] < 0)
            or capture["run_count"] != 3
            or capture["trials_per_run"] != 8):
        raise ValueError(f"{owner}.capture campaign changed")
    if (not capture["raw_path"].startswith(
            "profiler_output/microbench/ascend_910b/")
            or ".." in capture["raw_path"].split("/")):
        raise ValueError(f"{owner}.capture raw path is outside its owner")
    digest_names = capture_fields - {
        "source_commit", "source_snapshot_path", "physical_device_id",
        "run_count", "trials_per_run", "raw_path",
    }
    if any(not isinstance(capture[name], str) or len(capture[name]) != 64
           or any(character not in "0123456789abcdef"
                  for character in capture[name])
           for name in digest_names):
        raise ValueError(f"{owner}.capture evidence SHA256 is invalid")
    return {
        "model": expected["mechanism"],
        **source,
        "mte2_only_ns": medians["mte2_only"],
        "ingress_completion_ns": medians["mte2_mte1_completion"],
        "pipeline_completion_ns": medians[
            "mte2_mte1_cube_completion"],
        "reference_path": str(
            AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE),
        "reference_sha256": (
            AIC_MTE2_GEMM_REGION_PIPELINE_COMPLETION_REFERENCE_SHA256),
    }


def load_aic_mte2_fia_mha_finite_drain_service(repository_root):
    """Load exact cold finite-drain services for the FIA MHA source."""

    path = repository_root / AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE_SHA256):
        raise ValueError("AIC MTE2 FIA MHA finite-drain reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 FIA MHA finite-drain reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_fia_mha_finite_drain",
        "duration_unit": "ns_per_finite_descriptor_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"source_residency", "dtype", "active_aic_groups",
         "completion_edge", "estimator", "geometries"})
    expected_scope = {
        "source_residency": "dram_cold_disjoint",
        "dtype": "bf16",
        "completion_edge": "MTE2_MTE1_after_every_descriptor",
        "estimator": "exact_measured_geometry_and_descriptor_count_only",
    }
    for name, expected_value in expected_scope.items():
        if scope[name] != expected_value:
            raise ValueError(
                f"{owner}.publication_scope.{name} changed")
    if scope["active_aic_groups"] != [16, 22, 24]:
        raise ValueError(
            f"{owner}.publication_scope.active_aic_groups changed")
    expected_counts = {
        (1, 128, 128, 16): (2,),
        (256, 128, 4096, 16): (4, 8),
        (1, 128, 128, 22): (2, 3, 5, 6),
        (1, 256, 512, 22): (4, 6),
        (1, 256, 1024, 22): (8, 12),
        (1, 256, 2048, 22): (16, 24),
        (256, 128, 4096, 22): (
            4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48),
        (1, 128, 128, 24): (10, 11, 21, 22),
        (256, 128, 4096, 24): (20, 22, 40, 42, 44),
    }
    geometries = scope["geometries"]
    if not isinstance(geometries, list) or len(geometries) != 9:
        raise ValueError(f"{owner} must publish exactly nine geometries")
    normalized_geometries = {}
    for index, geometry in enumerate(geometries):
        geometry_owner = f"{owner}.publication_scope.geometries[{index}]"
        require_exact_fields(
            geometry_owner, geometry,
            {"rows", "columns", "source_row_stride_elements",
             "active_aic_groups", "descriptor_counts", "duration_ns"})
        key = tuple(
            geometry[name] for name in (
                "rows", "columns", "source_row_stride_elements",
                "active_aic_groups"))
        if key not in expected_counts:
            raise ValueError(f"{geometry_owner} is outside the frozen domain")
        counts = geometry["descriptor_counts"]
        durations = geometry["duration_ns"]
        if (not isinstance(counts, list) or tuple(counts) != expected_counts[key]
                or not isinstance(durations, list)
                or len(durations) != len(counts)):
            raise ValueError(f"{geometry_owner} count axis changed")
        normalized_durations = []
        for duration in durations:
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(
                    f"{geometry_owner}.duration_ns must be positive")
            normalized_durations.append(float(duration))
        normalized_geometries[key] = dict(
            zip(tuple(counts), tuple(normalized_durations)))
    if set(normalized_geometries) != set(expected_counts):
        raise ValueError(f"{owner} geometry domain is incomplete")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"mode", "measurement_baseline_commit", "source_commit",
         "physical_device_id", "binary_sha256", "source_kernel_sha256",
         "source_main_sha256", "raw_paths", "raw_sha256",
         "mechanism_gate_passed"})
    if (measurement["mode"]
            != "independent_finite_drained_descriptor_microbenchmark"
            or (isinstance(measurement["physical_device_id"], bool)
                or not isinstance(measurement["physical_device_id"], int)
                or measurement["physical_device_id"] < 0)
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    for name in ("measurement_baseline_commit", "source_commit"):
        commit = measurement[name]
        if (not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(f"{owner}.measurement.{name} is invalid")
    raw_paths = measurement["raw_paths"]
    raw_hashes = measurement["raw_sha256"]
    if (not isinstance(raw_paths, list) or len(raw_paths) != 24
            or not isinstance(raw_hashes, list) or len(raw_hashes) != 24):
        raise ValueError(f"{owner} requires twenty-four raw captures")
    for raw_path in raw_paths:
        if (not isinstance(raw_path, str)
                or not raw_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in raw_path.split("/")):
            raise ValueError(f"{owner} raw path is outside its evidence owner")
    evidence_hashes = tuple(raw_hashes) + tuple(
        measurement[name] for name in (
            "binary_sha256", "source_kernel_sha256", "source_main_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    supplement_path = (
        repository_root
        / AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE)
    supplement_bytes = supplement_path.read_bytes()
    if hashlib.sha256(supplement_bytes).hexdigest() != (
            AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 FIA MHA finite-drain supplement reference SHA256 "
            "mismatch")
    supplement = json.loads(supplement_bytes)
    supplement_owner = "AIC MTE2 FIA MHA finite-drain supplement reference"
    require_exact_fields(
        supplement_owner, supplement,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected_supplement = {
        "schema_version": 2,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_fia_mha_finite_drain_supplement",
        "duration_unit": "ns_per_finite_descriptor_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected_supplement.items():
        if supplement[name] != expected_value:
            raise ValueError(
                f"{supplement_owner}.{name} must be {expected_value!r}")
    supplement_scope = supplement["publication_scope"]
    require_exact_fields(
        f"{supplement_owner}.publication_scope", supplement_scope,
        {"source_residency", "dtype", "completion_edge", "estimator",
         "programs"})
    for name, expected_value in expected_scope.items():
        if supplement_scope[name] != expected_value:
            raise ValueError(
                f"{supplement_owner}.publication_scope.{name} changed")
    supplement_programs = supplement_scope["programs"]
    expected_supplement_counts = {
        16: 28220.0,
        32: 56560.0,
        64: 111600.0,
    }
    if (not isinstance(supplement_programs, list)
            or len(supplement_programs) != 3):
        raise ValueError(f"{supplement_owner} must publish three programs")
    normalized_supplement = {}
    for index, program in enumerate(supplement_programs):
        program_owner = (
            f"{supplement_owner}.publication_scope.programs[{index}]")
        require_exact_fields(
            program_owner, program,
            {"rows", "columns", "source_row_stride_elements",
             "active_aic_groups", "descriptor_count", "duration_ns"})
        geometry = tuple(program[name] for name in (
            "rows", "columns", "source_row_stride_elements",
            "active_aic_groups"))
        descriptor_count = program["descriptor_count"]
        duration_ns = program["duration_ns"]
        if (geometry != (256, 128, 4096, 16)
                or descriptor_count not in expected_supplement_counts
                or descriptor_count in normalized_supplement
                or duration_ns != expected_supplement_counts[
                    descriptor_count]):
            raise ValueError(f"{program_owner} is outside its frozen domain")
        normalized_supplement[descriptor_count] = float(duration_ns)
    if set(normalized_supplement) != set(expected_supplement_counts):
        raise ValueError(f"{supplement_owner} program domain is incomplete")
    supplement_measurement = supplement["measurement"]
    require_exact_fields(
        f"{supplement_owner}.measurement", supplement_measurement,
        {"mode", "measurement_baseline_commit", "source_commit",
         "physical_device_id", "binary_sha256", "source_kernel_sha256",
         "source_main_sha256", "cmake_sha256", "raw_paths", "raw_sha256",
         "program_run_duration_ns", "maximum_program_deviation_pct",
         "mechanism_gate_passed"})
    if (supplement_measurement["mode"]
            != "independent_finite_drained_descriptor_microbenchmark"
            or (isinstance(supplement_measurement["physical_device_id"], bool)
                or not isinstance(supplement_measurement["physical_device_id"], int)
                or supplement_measurement["physical_device_id"] < 0)
            or supplement_measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{supplement_owner} mechanism gate failed")
    supplement_runs = supplement_measurement["program_run_duration_ns"]
    supplement_deviations = supplement_measurement[
        "maximum_program_deviation_pct"]
    if (not isinstance(supplement_runs, list)
            or len(supplement_runs) != len(supplement_programs)
            or not isinstance(supplement_deviations, list)
            or len(supplement_deviations) != len(supplement_programs)):
        raise ValueError(
            f"{supplement_owner} run groups do not match programs")
    for index, (program, run_group, deviation) in enumerate(zip(
            supplement_programs, supplement_runs, supplement_deviations,
            strict=True)):
        if (not isinstance(run_group, list) or len(run_group) != 3
                or any(isinstance(item, bool)
                       or not isinstance(item, (int, float))
                       or not math.isfinite(item) or item <= 0
                       for item in run_group)
                or float(sorted(run_group)[1])
                != float(program["duration_ns"])):
            raise ValueError(
                f"{supplement_owner} program {index} duration is not the "
                "run median")
        if (isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(deviation)
                or not 0.0 <= deviation <= 5.0):
            raise ValueError(
                f"{supplement_owner} program {index} deviation exceeds 5%")
    supplement_paths = supplement_measurement["raw_paths"]
    supplement_hashes = supplement_measurement["raw_sha256"]
    if (not isinstance(supplement_paths, list)
            or len(supplement_paths) != 9
            or not isinstance(supplement_hashes, list)
            or len(supplement_hashes) != 9):
        raise ValueError(f"{supplement_owner} requires nine raw captures")
    if any(
            not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")
            for raw_path in supplement_paths):
        raise ValueError(f"{supplement_owner} raw path is outside its owner")
    supplement_evidence_hashes = tuple(supplement_hashes) + tuple(
        supplement_measurement[name] for name in (
            "binary_sha256", "source_kernel_sha256", "source_main_sha256",
            "cmake_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in supplement_evidence_hashes):
        raise ValueError(f"{supplement_owner} evidence SHA256 is invalid")
    mha_geometry = (256, 128, 4096, 16)
    if set(normalized_supplement).intersection(
            normalized_geometries[mha_geometry]):
        raise ValueError(f"{supplement_owner} duplicates a primary program")
    normalized_geometries[mha_geometry].update(normalized_supplement)
    return {
        "model": "aic_mte2_fia_mha_finite_drain_v1",
        "active_aic_groups": tuple(scope["active_aic_groups"]),
        "geometries": normalized_geometries,
        "reference_path": str(AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE),
        "reference_sha256": AIC_MTE2_FIA_MHA_FINITE_DRAIN_REFERENCE_SHA256,
        "supplemental_reference_path": str(
            AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE),
        "supplemental_reference_sha256": (
            AIC_MTE2_FIA_MHA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256),
    }


def load_aic_mte2_fia_gqa_drained_count_law(repository_root):
    """Load the measured count-independence licence for drained GQA programs.

    The campaign establishes that the drained per-descriptor ND2NZ service at
    a fixed geometry and AIC concurrency does not depend on the descriptor
    count over the validated range.  It licenses applying an exact measured
    program per descriptor at any count in that range; it does not supply the
    absolute service level, which stays with the exact-count references.
    """

    path = repository_root / AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 FIA GQA drained count-law reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 FIA GQA drained count-law reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_fia_gqa_drained_descriptor_count_law",
        "duration_unit": "dimensionless_count_independence_licence",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"source_residency", "dtype", "completion_edge", "estimator",
         "minimum_descriptor_count", "maximum_descriptor_count",
         "count_span_limit_pct", "cells"})
    expected_scope = {
        "source_residency": "dram_cold_disjoint",
        "dtype": "bf16",
        "completion_edge": "MTE2_MTE1_after_every_descriptor",
        "estimator": "count_free_per_descriptor_service_over_measured_range",
    }
    for name, expected_value in expected_scope.items():
        if scope[name] != expected_value:
            raise ValueError(f"{owner}.publication_scope.{name} changed")
    minimum_count = scope["minimum_descriptor_count"]
    maximum_count = scope["maximum_descriptor_count"]
    span_limit = scope["count_span_limit_pct"]
    if (isinstance(minimum_count, bool) or not isinstance(minimum_count, int)
            or isinstance(maximum_count, bool)
            or not isinstance(maximum_count, int)
            or minimum_count < 2 or maximum_count <= minimum_count):
        raise ValueError(f"{owner} descriptor-count range is invalid")
    if (isinstance(span_limit, bool)
            or not isinstance(span_limit, (int, float))
            or not math.isfinite(span_limit) or span_limit <= 0.0):
        raise ValueError(f"{owner}.publication_scope.count_span_limit_pct")
    cells = scope["cells"]
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{owner} must publish at least one cell")
    normalized_cells = {}
    for index, cell in enumerate(cells):
        cell_owner = f"{owner}.publication_scope.cells[{index}]"
        require_exact_fields(
            cell_owner, cell,
            {"rows", "columns", "source_row_stride_elements",
             "active_aic_groups", "measured_per_descriptor_ns_by_count",
             "count_span_pct", "count_independent"})
        key = tuple(
            cell[name] for name in (
                "rows", "columns", "source_row_stride_elements",
                "active_aic_groups"))
        if any(isinstance(item, bool) or not isinstance(item, int)
               or item <= 0 for item in key):
            raise ValueError(f"{cell_owner} geometry must be positive")
        if key in normalized_cells:
            raise ValueError(f"{cell_owner} duplicates a cell")
        if cell["count_independent"] is not True:
            raise ValueError(
                f"{cell_owner} did not pass its count-independence gate")
        cell_span = cell["count_span_pct"]
        if (isinstance(cell_span, bool)
                or not isinstance(cell_span, (int, float))
                or not math.isfinite(cell_span) or cell_span < 0.0
                or cell_span > span_limit):
            raise ValueError(f"{cell_owner}.count_span_pct exceeds its gate")
        measured = cell["measured_per_descriptor_ns_by_count"]
        if not isinstance(measured, dict) or len(measured) < 2:
            raise ValueError(f"{cell_owner} needs at least two counts")
        measured_counts = set()
        for count_text, service_ns in measured.items():
            if not isinstance(count_text, str) or not count_text.isdigit():
                raise ValueError(f"{cell_owner} count keys must be digits")
            count_value = int(count_text)
            if not minimum_count <= count_value <= maximum_count:
                raise ValueError(
                    f"{cell_owner} count {count_value} is out of range")
            if (isinstance(service_ns, bool)
                    or not isinstance(service_ns, (int, float))
                    or not math.isfinite(service_ns) or service_ns <= 0.0):
                raise ValueError(f"{cell_owner} service must be positive")
            measured_counts.add(count_value)
        if (min(measured_counts) != minimum_count
                or max(measured_counts) != maximum_count):
            raise ValueError(
                f"{cell_owner} must span the declared count range")
        normalized_cells[key] = float(cell_span)
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"mode", "measurement_baseline_commit", "physical_device_id",
         "process_count", "timing_boundary", "registration_path",
         "report_path", "process_median_span_limit_pct", "excluded_cells",
         "mechanism_gate_passed"})
    if (measurement["mode"]
            != "independent_finite_drained_descriptor_microbenchmark"
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    return {
        "model": "aic_mte2_fia_gqa_drained_descriptor_count_law_v1",
        "minimum_descriptor_count": minimum_count,
        "maximum_descriptor_count": maximum_count,
        "cells": frozenset(normalized_cells),
        "reference_path": str(AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE),
        "reference_sha256": (
            AIC_MTE2_FIA_GQA_DRAINED_COUNT_LAW_REFERENCE_SHA256),
    }


def load_aic_mte2_fia_gqa_finite_drain_service(repository_root):
    """Load exact cold finite-drain services for FIA GQA programs."""

    path = repository_root / AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE_SHA256):
        raise ValueError("AIC MTE2 FIA GQA finite-drain reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 FIA GQA finite-drain reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 2,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_fia_gqa_finite_drain",
        "duration_unit": "ns_per_finite_descriptor_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"source_residency", "dtype", "completion_edge", "estimator",
         "programs"})
    expected_scope = {
        "source_residency": "dram_cold_disjoint",
        "dtype": "bf16",
        "completion_edge": "MTE2_MTE1_after_every_descriptor",
        "estimator": "exact_measured_geometry_and_descriptor_count_only",
    }
    for name, expected_value in expected_scope.items():
        if scope[name] != expected_value:
            raise ValueError(
                f"{owner}.publication_scope.{name} changed")
    expected_programs = {
        (256, 64, 512, 16, 2),
        (256, 128, 1024, 22, 4),
        (128, 128, 1024, 22, 8),
        (256, 128, 1024, 24, 2),
    }
    programs = scope["programs"]
    if not isinstance(programs, list) or len(programs) != 4:
        raise ValueError(f"{owner} must publish exactly four programs")
    normalized_programs = {}
    for index, program in enumerate(programs):
        program_owner = f"{owner}.publication_scope.programs[{index}]"
        require_exact_fields(
            program_owner, program,
            {"rows", "columns", "source_row_stride_elements",
             "active_aic_groups", "descriptor_count", "duration_ns"})
        key = tuple(
            program[name] for name in (
                "rows", "columns", "source_row_stride_elements",
                "active_aic_groups", "descriptor_count"))
        if key not in expected_programs or key in normalized_programs:
            raise ValueError(f"{program_owner} is outside the frozen domain")
        if any(isinstance(item, bool) or not isinstance(item, int)
               or item <= 0 for item in key):
            raise ValueError(f"{program_owner} geometry must be positive")
        duration_ns = program["duration_ns"]
        if (isinstance(duration_ns, bool)
                or not isinstance(duration_ns, (int, float))
                or not math.isfinite(duration_ns) or duration_ns <= 0):
            raise ValueError(f"{program_owner}.duration_ns must be positive")
        normalized_programs[key] = float(duration_ns)
    if set(normalized_programs) != expected_programs:
        raise ValueError(f"{owner} program domain is incomplete")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"mode", "measurement_baseline_commit", "source_commit",
         "physical_device_id", "binary_sha256", "source_kernel_sha256",
         "source_main_sha256", "raw_paths", "raw_sha256",
         "program_run_duration_ns", "maximum_program_deviation_pct",
         "mechanism_gate_passed"})
    if (measurement["mode"]
            != "independent_finite_drained_descriptor_microbenchmark"
            or (isinstance(measurement["physical_device_id"], bool)
                or not isinstance(measurement["physical_device_id"], int)
                or measurement["physical_device_id"] < 0)
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    run_groups = measurement["program_run_duration_ns"]
    deviations = measurement["maximum_program_deviation_pct"]
    if (not isinstance(run_groups, list) or len(run_groups) != len(programs)
            or not isinstance(deviations, list)
            or len(deviations) != len(programs)):
        raise ValueError(f"{owner} run groups do not match programs")
    for index, (program, run_group, deviation) in enumerate(zip(
            programs, run_groups, deviations, strict=True)):
        if (not isinstance(run_group, list) or len(run_group) != 3
                or any(isinstance(item, bool)
                       or not isinstance(item, (int, float))
                       or not math.isfinite(item) or item <= 0
                       for item in run_group)
                or float(sorted(run_group)[1])
                != float(program["duration_ns"])):
            raise ValueError(
                f"{owner} program {index} duration is not the run median")
        if (isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(deviation)
                or not 0.0 <= deviation <= 5.0):
            raise ValueError(
                f"{owner} program {index} cross-run deviation exceeds 5%")
    raw_paths = measurement["raw_paths"]
    raw_hashes = measurement["raw_sha256"]
    if (not isinstance(raw_paths, list) or len(raw_paths) != 12
            or not isinstance(raw_hashes, list) or len(raw_hashes) != 12):
        raise ValueError(f"{owner} requires twelve raw captures")
    for raw_path in raw_paths:
        if (not isinstance(raw_path, str)
                or not raw_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in raw_path.split("/")):
            raise ValueError(f"{owner} raw path is outside its evidence owner")
    evidence_hashes = tuple(raw_hashes) + tuple(
        measurement[name] for name in (
            "binary_sha256", "source_kernel_sha256", "source_main_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    supplement_path = (
        repository_root
        / AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE)
    supplement_bytes = supplement_path.read_bytes()
    if hashlib.sha256(supplement_bytes).hexdigest() != (
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 FIA GQA finite-drain supplement reference SHA256 "
            "mismatch")
    supplement = json.loads(supplement_bytes)
    supplement_owner = "AIC MTE2 FIA GQA finite-drain supplement reference"
    require_exact_fields(
        supplement_owner, supplement,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected_supplement = {
        "schema_version": 4,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_fia_gqa_finite_drain_supplement",
        "duration_unit": "ns_per_finite_descriptor_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected_supplement.items():
        if supplement[name] != expected_value:
            raise ValueError(
                f"{supplement_owner}.{name} must be {expected_value!r}")
    supplement_scope = supplement["publication_scope"]
    require_exact_fields(
        f"{supplement_owner}.publication_scope", supplement_scope,
        {"source_residency", "dtype", "completion_edge", "estimator",
         "programs"})
    for name, expected_value in expected_scope.items():
        if supplement_scope[name] != expected_value:
            raise ValueError(
                f"{supplement_owner}.publication_scope.{name} changed")
    expected_supplement_programs = {
        (256, 128, 1024, 16, 8),
        (256, 64, 512, 16, 4),
        (256, 128, 1024, 8, 16),
        (256, 128, 1024, 8, 32),
        (256, 128, 1024, 16, 32),
        (256, 128, 1024, 22, 8),
        (256, 128, 1024, 24, 20),
        (256, 128, 1024, 24, 22),
    }
    supplement_programs = supplement_scope["programs"]
    if (not isinstance(supplement_programs, list)
            or len(supplement_programs) != 8):
        raise ValueError(f"{supplement_owner} must publish eight programs")
    normalized_supplement_programs = {}
    for index, supplement_program in enumerate(supplement_programs):
        program_owner = (
            f"{supplement_owner}.publication_scope.programs[{index}]")
        require_exact_fields(
            program_owner, supplement_program,
            {"rows", "columns", "source_row_stride_elements",
             "active_aic_groups", "descriptor_count", "duration_ns"})
        supplement_key = tuple(
            supplement_program[name] for name in (
                "rows", "columns", "source_row_stride_elements",
                "active_aic_groups", "descriptor_count"))
        if (supplement_key not in expected_supplement_programs
                or supplement_key in normalized_supplement_programs):
            raise ValueError(f"{program_owner} is outside its domain")
        supplement_duration = supplement_program["duration_ns"]
        if (isinstance(supplement_duration, bool)
                or not isinstance(supplement_duration, (int, float))
                or not math.isfinite(supplement_duration)
                or supplement_duration <= 0):
            raise ValueError(f"{program_owner} duration must be positive")
        normalized_supplement_programs[supplement_key] = float(
            supplement_duration)
    if set(normalized_supplement_programs) != expected_supplement_programs:
        raise ValueError(f"{supplement_owner} program domain is incomplete")
    supplement_measurement = supplement["measurement"]
    require_exact_fields(
        f"{supplement_owner}.measurement", supplement_measurement,
        {"mode", "measurement_baseline_commit", "source_commit",
         "physical_device_id", "binary_sha256", "source_kernel_sha256",
         "source_main_sha256", "cmake_sha256", "raw_paths", "raw_sha256",
         "program_run_duration_ns", "maximum_program_deviation_pct",
         "mechanism_gate_passed"})
    if (supplement_measurement["mode"]
            != "independent_finite_drained_descriptor_microbenchmark"
            or (isinstance(supplement_measurement["physical_device_id"], bool)
                or not isinstance(supplement_measurement["physical_device_id"], int)
                or supplement_measurement["physical_device_id"] < 0)
            or supplement_measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{supplement_owner} mechanism gate failed")
    supplement_runs = supplement_measurement["program_run_duration_ns"]
    supplement_deviations = supplement_measurement[
        "maximum_program_deviation_pct"]
    if (not isinstance(supplement_runs, list)
            or len(supplement_runs) != len(supplement_programs)
            or not isinstance(supplement_deviations, list)
            or len(supplement_deviations) != len(supplement_programs)):
        raise ValueError(
            f"{supplement_owner} run groups do not match programs")
    for index, (program, run_group, deviation) in enumerate(zip(
            supplement_programs, supplement_runs, supplement_deviations,
            strict=True)):
        if (not isinstance(run_group, list) or len(run_group) != 3
                or any(isinstance(item, bool)
                       or not isinstance(item, (int, float))
                       or not math.isfinite(item) or item <= 0
                       for item in run_group)
                or float(sorted(run_group)[1])
                != float(program["duration_ns"])):
            raise ValueError(
                f"{supplement_owner} program {index} duration is not the "
                "run median")
        if (isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(deviation)
                or not 0.0 <= deviation <= 5.0):
            raise ValueError(
                f"{supplement_owner} program {index} deviation exceeds 5%")
    supplement_paths = supplement_measurement["raw_paths"]
    supplement_hashes = supplement_measurement["raw_sha256"]
    if (not isinstance(supplement_paths, list)
            or len(supplement_paths) != 24
            or not isinstance(supplement_hashes, list)
            or len(supplement_hashes) != 24):
        raise ValueError(f"{supplement_owner} requires twenty-four raw captures")
    if any(
            not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")
            for raw_path in supplement_paths):
        raise ValueError(f"{supplement_owner} raw path is outside its owner")
    supplement_evidence_hashes = tuple(supplement_hashes) + tuple(
        supplement_measurement[name] for name in (
            "binary_sha256", "source_kernel_sha256", "source_main_sha256",
            "cmake_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in supplement_evidence_hashes):
        raise ValueError(f"{supplement_owner} evidence SHA256 is invalid")
    if set(normalized_supplement_programs).intersection(normalized_programs):
        raise ValueError(f"{supplement_owner} duplicates a primary program")
    normalized_programs.update(normalized_supplement_programs)
    group22_count16_path = (
        repository_root
        / AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE)
    group22_count16_bytes = group22_count16_path.read_bytes()
    if hashlib.sha256(group22_count16_bytes).hexdigest() != (
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 FIA GQA group-22/count-16 reference SHA256 mismatch")
    group22_count16 = json.loads(group22_count16_bytes)
    group22_count16_owner = (
        "AIC MTE2 FIA GQA group-22/count-16 finite-drain reference")
    require_exact_fields(
        group22_count16_owner, group22_count16,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected_group22_count16 = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_mte2_fia_gqa_finite_drain_group22_count16_supplement"),
        "duration_unit": "ns_per_finite_descriptor_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected_group22_count16.items():
        if group22_count16[name] != expected_value:
            raise ValueError(
                f"{group22_count16_owner}.{name} must be {expected_value!r}")
    group22_count16_scope = group22_count16["publication_scope"]
    require_exact_fields(
        f"{group22_count16_owner}.publication_scope", group22_count16_scope,
        {"source_residency", "dtype", "completion_edge", "estimator",
         "programs"})
    for name, expected_value in expected_scope.items():
        if group22_count16_scope[name] != expected_value:
            raise ValueError(
                f"{group22_count16_owner}.publication_scope.{name} changed")
    expected_group22_count16_programs = {
        (256, 128, 1024, 22, 16),
    }
    group22_count16_programs = group22_count16_scope["programs"]
    if (not isinstance(group22_count16_programs, list)
            or len(group22_count16_programs) != 1):
        raise ValueError(
            f"{group22_count16_owner} must publish exactly one program")
    normalized_group22_count16_programs = {}
    for index, program in enumerate(group22_count16_programs):
        program_owner = (
            f"{group22_count16_owner}.publication_scope.programs[{index}]")
        require_exact_fields(
            program_owner, program,
            {"rows", "columns", "source_row_stride_elements",
             "active_aic_groups", "descriptor_count", "duration_ns"})
        key = tuple(
            program[name] for name in (
                "rows", "columns", "source_row_stride_elements",
                "active_aic_groups", "descriptor_count"))
        if (key not in expected_group22_count16_programs
                or key in normalized_group22_count16_programs):
            raise ValueError(f"{program_owner} is outside its domain")
        duration_ns = program["duration_ns"]
        if (isinstance(duration_ns, bool)
                or not isinstance(duration_ns, (int, float))
                or not math.isfinite(duration_ns) or duration_ns <= 0):
            raise ValueError(f"{program_owner} duration must be positive")
        normalized_group22_count16_programs[key] = float(duration_ns)
    if (set(normalized_group22_count16_programs)
            != expected_group22_count16_programs):
        raise ValueError(f"{group22_count16_owner} program domain is incomplete")
    group22_count16_measurement = group22_count16["measurement"]
    require_exact_fields(
        f"{group22_count16_owner}.measurement", group22_count16_measurement,
        {"mode", "measurement_baseline_commit", "source_commit",
         "physical_device_id", "binary_sha256", "source_kernel_sha256",
         "source_main_sha256", "cmake_sha256", "raw_paths", "raw_sha256",
         "program_run_duration_ns", "maximum_program_deviation_pct",
         "mechanism_gate_passed"})
    if (group22_count16_measurement["mode"]
            != "independent_finite_drained_descriptor_microbenchmark"
            or (isinstance(group22_count16_measurement["physical_device_id"], bool)
                or not isinstance(group22_count16_measurement["physical_device_id"], int)
                or group22_count16_measurement["physical_device_id"] < 0)
            or group22_count16_measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{group22_count16_owner} mechanism gate failed")
    group22_count16_runs = group22_count16_measurement[
        "program_run_duration_ns"]
    group22_count16_deviations = group22_count16_measurement[
        "maximum_program_deviation_pct"]
    if (not isinstance(group22_count16_runs, list)
            or len(group22_count16_runs) != len(group22_count16_programs)
            or not isinstance(group22_count16_deviations, list)
            or len(group22_count16_deviations) != len(group22_count16_programs)):
        raise ValueError(
            f"{group22_count16_owner} run groups do not match programs")
    for index, (program, run_group, deviation) in enumerate(zip(
            group22_count16_programs, group22_count16_runs,
            group22_count16_deviations, strict=True)):
        if (not isinstance(run_group, list) or len(run_group) != 3
                or any(isinstance(item, bool)
                       or not isinstance(item, (int, float))
                       or not math.isfinite(item) or item <= 0
                       for item in run_group)
                or float(sorted(run_group)[1])
                != float(program["duration_ns"])):
            raise ValueError(
                f"{group22_count16_owner} program {index} duration is not "
                "the run median")
        if (isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(deviation)
                or not 0.0 <= deviation <= 5.0):
            raise ValueError(
                f"{group22_count16_owner} program {index} deviation exceeds "
                "5%")
    group22_count16_paths = group22_count16_measurement["raw_paths"]
    group22_count16_hashes = group22_count16_measurement["raw_sha256"]
    if (not isinstance(group22_count16_paths, list)
            or len(group22_count16_paths) != 3
            or not isinstance(group22_count16_hashes, list)
            or len(group22_count16_hashes) != 3):
        raise ValueError(
            f"{group22_count16_owner} requires three raw captures")
    if any(
            not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")
            for raw_path in group22_count16_paths):
        raise ValueError(
            f"{group22_count16_owner} raw path is outside its owner")
    group22_count16_evidence_hashes = tuple(group22_count16_hashes) + tuple(
        group22_count16_measurement[name] for name in (
            "binary_sha256", "source_kernel_sha256", "source_main_sha256",
            "cmake_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in group22_count16_evidence_hashes):
        raise ValueError(f"{group22_count16_owner} evidence SHA256 is invalid")
    if (set(normalized_group22_count16_programs)
            .intersection(normalized_programs)):
        raise ValueError(f"{group22_count16_owner} duplicates a program")
    normalized_programs.update(normalized_group22_count16_programs)

    def load_exact_single_program_extension(
            path, expected_sha256, owner, mechanism, expected_program):
        """Load a separately-built one-program GQA finite-drain reference."""

        payload_bytes = path.read_bytes()
        if hashlib.sha256(payload_bytes).hexdigest() != expected_sha256:
            raise ValueError(f"{owner} SHA256 mismatch")
        value = json.loads(payload_bytes)
        require_exact_fields(
            owner, value,
            {"schema_version", "hardware_id", "software_revision",
             "mechanism", "duration_unit", "publication_scope",
             "measurement", "production_or_evaluation_duration_consumed",
             "parameter_fit_from_production_duration", "mechanism_publishable",
             "forward_eligible", "forward_eligibility_constraint"})
        expected = {
            "schema_version": 1,
            "hardware_id": "ascend_910b",
            "software_revision": "cann_8.5.1",
            "mechanism": mechanism,
            "duration_unit": "ns_per_finite_descriptor_program_per_aic",
            "production_or_evaluation_duration_consumed": False,
            "parameter_fit_from_production_duration": False,
            "mechanism_publishable": True,
            "forward_eligible": True,
        }
        for name, expected_value in expected.items():
            if value[name] != expected_value:
                raise ValueError(f"{owner}.{name} must be {expected_value!r}")
        scope = value["publication_scope"]
        require_exact_fields(
            f"{owner}.publication_scope", scope,
            {"source_residency", "dtype", "completion_edge", "estimator",
             "programs"})
        for name, expected_value in expected_scope.items():
            if scope[name] != expected_value:
                raise ValueError(f"{owner}.publication_scope.{name} changed")
        programs = scope["programs"]
        if not isinstance(programs, list) or len(programs) != 1:
            raise ValueError(f"{owner} must publish exactly one program")
        program = programs[0]
        require_exact_fields(
            f"{owner}.publication_scope.programs[0]", program,
            {"rows", "columns", "source_row_stride_elements",
             "active_aic_groups", "descriptor_count", "duration_ns"})
        key = tuple(
            program[name] for name in (
                "rows", "columns", "source_row_stride_elements",
                "active_aic_groups", "descriptor_count"))
        if key != expected_program:
            raise ValueError(f"{owner} program is outside its frozen domain")
        if any(isinstance(item, bool) or not isinstance(item, int)
               or item <= 0 for item in key):
            raise ValueError(f"{owner} program geometry must be positive")
        duration_ns = program["duration_ns"]
        if (isinstance(duration_ns, bool)
                or not isinstance(duration_ns, (int, float))
                or not math.isfinite(duration_ns) or duration_ns <= 0):
            raise ValueError(f"{owner} program duration must be positive")
        measurement = value["measurement"]
        require_exact_fields(
            f"{owner}.measurement", measurement,
            {"mode", "measurement_baseline_commit", "source_commit",
             "physical_device_id", "binary_sha256", "source_kernel_sha256",
             "source_main_sha256", "cmake_sha256", "raw_paths", "raw_sha256",
             "program_run_duration_ns", "maximum_program_deviation_pct",
             "mechanism_gate_passed"})
        if (measurement["mode"]
                != "independent_finite_drained_descriptor_microbenchmark"
                or (isinstance(measurement["physical_device_id"], bool)
                    or not isinstance(measurement["physical_device_id"], int)
                    or measurement["physical_device_id"] < 0)
                or measurement["mechanism_gate_passed"] is not True):
            raise ValueError(f"{owner} mechanism gate failed")
        runs = measurement["program_run_duration_ns"]
        deviations = measurement["maximum_program_deviation_pct"]
        if (not isinstance(runs, list) or len(runs) != 1
                or not isinstance(deviations, list) or len(deviations) != 1):
            raise ValueError(f"{owner} run groups do not match its program")
        run_group = runs[0]
        deviation = deviations[0]
        if (not isinstance(run_group, list) or len(run_group) != 3
                or any(isinstance(item, bool)
                       or not isinstance(item, (int, float))
                       or not math.isfinite(item) or item <= 0
                       for item in run_group)
                or float(sorted(run_group)[1]) != float(duration_ns)):
            raise ValueError(f"{owner} duration is not the run median")
        if (isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(deviation)
                or not 0.0 <= deviation <= 5.0):
            raise ValueError(f"{owner} deviation exceeds 5%")
        raw_paths = measurement["raw_paths"]
        raw_hashes = measurement["raw_sha256"]
        if (not isinstance(raw_paths, list) or len(raw_paths) != 3
                or not isinstance(raw_hashes, list) or len(raw_hashes) != 3):
            raise ValueError(f"{owner} requires three raw captures")
        if any(
                not isinstance(raw_path, str)
                or not raw_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in raw_path.split("/")
                for raw_path in raw_paths):
            raise ValueError(f"{owner} raw path is outside its owner")
        evidence_hashes = tuple(raw_hashes) + tuple(
            measurement[name] for name in (
                "binary_sha256", "source_kernel_sha256", "source_main_sha256",
                "cmake_sha256"))
        if any(not isinstance(digest, str) or len(digest) != 64
               or any(character not in "0123456789abcdef"
                      for character in digest)
               for digest in evidence_hashes):
            raise ValueError(f"{owner} evidence SHA256 is invalid")
        return {key: float(duration_ns)}

    group16_count16_owner = (
        "AIC MTE2 FIA GQA group-16/count-16 finite-drain reference")
    normalized_group16_count16_programs = (
        load_exact_single_program_extension(
            repository_root
            / AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE,
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE_SHA256,
            group16_count16_owner,
            "aic_mte2_fia_gqa_finite_drain_group16_count16_supplement",
            (256, 128, 1024, 16, 16)))
    if set(normalized_group16_count16_programs).intersection(
            normalized_programs):
        raise ValueError(f"{group16_count16_owner} duplicates a program")
    normalized_programs.update(normalized_group16_count16_programs)
    interleaved_path = (
        repository_root
        / AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE)
    interleaved_bytes = interleaved_path.read_bytes()
    if hashlib.sha256(interleaved_bytes).hexdigest() != (
            AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 FIA GQA interleaved finite-drain reference SHA256 "
            "mismatch")
    interleaved = json.loads(interleaved_bytes)
    interleaved_owner = (
        "AIC MTE2 FIA GQA interleaved finite-drain reference")
    require_exact_fields(
        interleaved_owner, interleaved,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected_interleaved = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_mte2_fia_gqa_interleaved_finite_drain_override"),
        "duration_unit": "ns_per_finite_descriptor_program_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected_interleaved.items():
        if interleaved[name] != expected_value:
            raise ValueError(
                f"{interleaved_owner}.{name} must be {expected_value!r}")
    interleaved_scope = interleaved["publication_scope"]
    require_exact_fields(
        f"{interleaved_owner}.publication_scope", interleaved_scope,
        {"source_residency", "source_address_layout", "dtype",
         "completion_edge", "estimator", "programs"})
    expected_interleaved_scope = {
        "source_residency": "dram_cold_disjoint",
        "source_address_layout": "bshd_eight_head_interleaved",
        "dtype": "bf16",
        "completion_edge": "MTE2_MTE1_after_every_descriptor",
        "estimator": "exact_measured_geometry_and_descriptor_count_only",
    }
    for name, expected_value in expected_interleaved_scope.items():
        if interleaved_scope[name] != expected_value:
            raise ValueError(
                f"{interleaved_owner}.publication_scope.{name} changed")
    expected_interleaved_programs = {
        (256, 128, 1024, 8, 16),
        (256, 128, 1024, 8, 32),
    }
    interleaved_programs = interleaved_scope["programs"]
    if (not isinstance(interleaved_programs, list)
            or len(interleaved_programs) != 2):
        raise ValueError(
            f"{interleaved_owner} must publish exactly two programs")
    normalized_interleaved_programs = {}
    for index, program in enumerate(interleaved_programs):
        program_owner = (
            f"{interleaved_owner}.publication_scope.programs[{index}]")
        require_exact_fields(
            program_owner, program,
            {"rows", "columns", "source_row_stride_elements",
             "active_aic_groups", "descriptor_count", "duration_ns"})
        key = tuple(
            program[name] for name in (
                "rows", "columns", "source_row_stride_elements",
                "active_aic_groups", "descriptor_count"))
        if (key not in expected_interleaved_programs
                or key in normalized_interleaved_programs):
            raise ValueError(f"{program_owner} is outside its domain")
        duration_ns = program["duration_ns"]
        if (isinstance(duration_ns, bool)
                or not isinstance(duration_ns, (int, float))
                or not math.isfinite(duration_ns) or duration_ns <= 0):
            raise ValueError(f"{program_owner} duration must be positive")
        normalized_interleaved_programs[key] = float(duration_ns)
    if set(normalized_interleaved_programs) != expected_interleaved_programs:
        raise ValueError(f"{interleaved_owner} program domain is incomplete")
    measurement = interleaved["measurement"]
    require_exact_fields(
        f"{interleaved_owner}.measurement", measurement,
        {"mode", "measurement_baseline_commit", "physical_device_id",
         "binary_sha256", "source_kernel_sha256", "source_main_sha256",
         "cmake_sha256", "raw_paths", "raw_sha256",
         "program_run_duration_ns", "maximum_program_deviation_pct",
         "mechanism_gate_passed"})
    if (measurement["mode"]
            != "independent_finite_drained_descriptor_microbenchmark"
            or (isinstance(measurement["physical_device_id"], bool)
                or not isinstance(measurement["physical_device_id"], int)
                or measurement["physical_device_id"] < 0)
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{interleaved_owner} mechanism gate failed")
    run_groups = measurement["program_run_duration_ns"]
    deviations = measurement["maximum_program_deviation_pct"]
    if (not isinstance(run_groups, list)
            or len(run_groups) != len(interleaved_programs)
            or not isinstance(deviations, list)
            or len(deviations) != len(interleaved_programs)):
        raise ValueError(
            f"{interleaved_owner} run groups do not match programs")
    for index, (program, run_group, deviation) in enumerate(zip(
            interleaved_programs, run_groups, deviations, strict=True)):
        if (not isinstance(run_group, list) or len(run_group) != 3
                or any(isinstance(item, bool)
                       or not isinstance(item, (int, float))
                       or not math.isfinite(item) or item <= 0
                       for item in run_group)
                or float(sorted(run_group)[1])
                != float(program["duration_ns"])):
            raise ValueError(
                f"{interleaved_owner} program {index} duration is not the "
                "run median")
        if (isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(deviation)
                or not 0.0 <= deviation <= 5.0):
            raise ValueError(
                f"{interleaved_owner} program {index} deviation exceeds 5%")
    raw_paths = measurement["raw_paths"]
    raw_hashes = measurement["raw_sha256"]
    if (not isinstance(raw_paths, list) or len(raw_paths) != 6
            or not isinstance(raw_hashes, list) or len(raw_hashes) != 6):
        raise ValueError(f"{interleaved_owner} requires six raw captures")
    if any(
            not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")
            for raw_path in raw_paths):
        raise ValueError(f"{interleaved_owner} raw path is outside its owner")
    evidence_hashes = tuple(raw_hashes) + tuple(
        measurement[name] for name in (
            "binary_sha256", "source_kernel_sha256", "source_main_sha256",
            "cmake_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{interleaved_owner} evidence SHA256 is invalid")
    if not set(normalized_interleaved_programs).issubset(
            normalized_supplement_programs):
        raise ValueError(
            f"{interleaved_owner} must refine an existing exact program")
    normalized_programs.update(normalized_interleaved_programs)
    # The count-independence campaign licenses applying an exact measured
    # program per descriptor at any count in its validated range.  The
    # absolute level stays with the exact-count references: for each licensed
    # cell take the published program with the largest measured descriptor
    # count, which is the one least affected by program-edge effects.
    count_law = load_aic_mte2_fia_gqa_drained_count_law(repository_root)
    count_free_anchor = {}
    for program_key, program_duration in normalized_programs.items():
        cell = program_key[:4]
        descriptor_count = program_key[4]
        if cell not in count_law["cells"]:
            continue
        if descriptor_count < count_law["minimum_descriptor_count"]:
            continue
        if (cell in count_free_anchor
                and count_free_anchor[cell][0] >= descriptor_count):
            continue
        count_free_anchor[cell] = (
            descriptor_count, program_duration / descriptor_count)
    count_free_service = {
        cell: per_descriptor_ns
        for cell, (_, per_descriptor_ns) in count_free_anchor.items()}
    if not count_free_service:
        raise ValueError(
            "AIC MTE2 FIA GQA count law licenses no measured program")
    return {
        "model": "aic_mte2_fia_gqa_finite_drain_v1",
        "count_free_service": count_free_service,
        "count_law_minimum_descriptor_count": (
            count_law["minimum_descriptor_count"]),
        "count_law_maximum_descriptor_count": (
            count_law["maximum_descriptor_count"]),
        "count_law_reference_path": count_law["reference_path"],
        "count_law_reference_sha256": count_law["reference_sha256"],
        "active_aic_groups": tuple(sorted({
            key[3] for key in normalized_programs})),
        "programs": normalized_programs,
        "reference_path": str(AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE),
        "reference_sha256": AIC_MTE2_FIA_GQA_FINITE_DRAIN_REFERENCE_SHA256,
        "supplemental_reference_path": str(
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE),
        "supplemental_reference_sha256": (
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_SUPPLEMENT_REFERENCE_SHA256),
        "group22_count16_reference_path": str(
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE),
        "group22_count16_reference_sha256": (
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP22_COUNT16_REFERENCE_SHA256),
        "group16_count16_reference_path": str(
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE),
        "group16_count16_reference_sha256": (
            AIC_MTE2_FIA_GQA_FINITE_DRAIN_GROUP16_COUNT16_REFERENCE_SHA256),
        "interleaved_reference_path": str(
            AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE),
        "interleaved_reference_sha256": (
            AIC_MTE2_FIA_GQA_INTERLEAVED_FINITE_DRAIN_REFERENCE_SHA256),
    }


def load_aic_grouped_matmul_stream_completion_reference(
        repository_root, reference_path, reference_sha256, mechanism_id,
        pair_counts, forward_eligibility_constraint):
    """Load one frozen M16/N256/K64 GroupedMatmul completion campaign."""

    path = repository_root / reference_path
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != reference_sha256:
        raise ValueError(
            f"AIC GroupedMatmul {mechanism_id} reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = f"AIC GroupedMatmul {mechanism_id} completion reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "publication_scope",
         "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": mechanism_id,
        "duration_unit": (
            "ns_per_completed_M16_N256_K64_operand_pair_per_aic"),
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"a_residency", "b_residency", "active_aic_groups", "a_rows",
         "a_columns", "b_rows", "b_columns", "pair_bytes",
         "drain_interval_pairs", "minimum_pairs_per_worker",
         "maximum_pairs_per_worker", "measured_pairs_per_worker",
         "eviction_bytes"})
    frozen_scope = {
        "a_residency": "l2_resident",
        "b_residency": "dram_cold",
        "active_aic_groups": 24,
        "a_rows": 16,
        "a_columns": 64,
        "b_rows": 64,
        "b_columns": 256,
        "pair_bytes": 34816,
        "drain_interval_pairs": 4,
        "minimum_pairs_per_worker": pair_counts[0],
        "maximum_pairs_per_worker": pair_counts[-1],
        "measured_pairs_per_worker": list(pair_counts),
        "eviction_bytes": 537919488,
    }
    for name, expected_value in frozen_scope.items():
        if scope[name] != expected_value:
            raise ValueError(f"{owner}.publication_scope.{name} changed")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"median_completion_span_ns", "median_ns_per_pair",
         "published_completion_ns_per_pair", "within_count_relative_spans",
         "rate_relative_span", "maximum_allowed_within_count_relative_span",
         "maximum_allowed_rate_relative_span", "stability_gate",
         "constant_rate_gate", "mechanism_gate_passed"})
    if tuple(scope["measured_pairs_per_worker"]) != pair_counts:
        raise ValueError(f"{owner} measured pair counts changed")
    expected_keys = {str(count) for count in pair_counts}
    for name in (
            "median_completion_span_ns", "median_ns_per_pair",
            "within_count_relative_spans"):
        mapping = measurement[name]
        if not isinstance(mapping, dict) or set(mapping) != expected_keys:
            raise ValueError(f"{owner}.measurement.{name} changed its axis")
    rates = []
    for pair_count in pair_counts:
        key = str(pair_count)
        span = measurement["median_completion_span_ns"][key]
        rate = measurement["median_ns_per_pair"][key]
        relative_span = measurement["within_count_relative_spans"][key]
        for name, duration in (("span", span), ("rate", rate)):
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(
                    f"{owner} {pair_count}-pair {name} must be positive")
        if not math.isclose(
                float(rate), float(span) / pair_count,
                rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{owner} {pair_count}-pair rate changed")
        if (isinstance(relative_span, bool)
                or not isinstance(relative_span, (int, float))
                or not math.isfinite(relative_span)
                or not 0.0 <= relative_span <= 0.1):
            raise ValueError(
                f"{owner} {pair_count}-pair stability changed")
        rates.append(float(rate))
    published_rate = measurement["published_completion_ns_per_pair"]
    if (isinstance(published_rate, bool)
            or not isinstance(published_rate, (int, float))
            or not math.isfinite(published_rate) or published_rate <= 0
            or not math.isclose(float(published_rate), max(rates),
                         rel_tol=0.0, abs_tol=1e-9)
            or measurement["maximum_allowed_within_count_relative_span"] != 0.1
            or measurement["maximum_allowed_rate_relative_span"] != 0.1
            or measurement["rate_relative_span"] > 0.1
            or measurement["stability_gate"] is not True
            or measurement["constant_rate_gate"] is not True
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    provenance = value["provenance"]
    common_provenance_fields = {
        "measurement_baseline_commit", "snapshot_path", "raw_root",
        "host", "container", "physical_device_id", "logical_device_id",
        "owner", "source_snapshot_sha256", "source_sha256",
        "binary_sha256", "raw_sha256", "stderr_sha256", "command",
    }
    if mechanism_id == "aic_grouped_matmul_short_stream_completion_v1":
        provenance_fields = common_provenance_fields | {
            "container_id", "container_image", "container_image_id",
            "report_sha256",
        }
        expected_source_names = {
            "CMakeLists.txt", "README.md",
            "grouped_pair_completion_kernel.cpp",
            "grouped_pair_completion_main.cpp",
            "grouped_pair_completion_runner.py",
            "grouped_pair_completion_report.py",
        }
        report_hashes = (provenance["report_sha256"],)
    elif mechanism_id == "aic_grouped_matmul_long_stream_completion_v1":
        provenance_fields = common_provenance_fields
        expected_source_names = {
            "CMakeLists.txt", "README.md",
            "grouped_pair_completion_kernel.cpp",
            "grouped_pair_completion_main.cpp",
            "grouped_pair_completion_runner.py",
        }
        report_hashes = ()
    else:
        raise ValueError(f"unsupported GroupedMatmul mechanism {mechanism_id}")
    require_exact_fields(
        f"{owner}.provenance", provenance, provenance_fields)
    if (provenance["measurement_baseline_commit"] != (
            "f1f021d3aac254992b16fd92b35534e016b196f5")
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "fused_moe_worker"
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in provenance["raw_root"].split("/")):
        raise ValueError(f"{owner} provenance changed")
    source_hashes = provenance["source_sha256"]
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != expected_source_names
            or not isinstance(provenance["raw_sha256"], list)
            or len(provenance["raw_sha256"]) != 3
            or not isinstance(provenance["stderr_sha256"], list)
            or len(provenance["stderr_sha256"]) != 3):
        raise ValueError(f"{owner} evidence files changed")
    evidence_hashes = tuple(source_hashes.values()) + tuple(
        provenance[name] for name in (
            "source_snapshot_sha256", "binary_sha256")) + report_hashes + tuple(
                provenance["raw_sha256"]) + tuple(provenance["stderr_sha256"])
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            forward_eligibility_constraint):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        **{
            name: (tuple(item) if isinstance(item, list) else item)
            for name, item in frozen_scope.items()
        },
        "completion_span_ns": tuple(
            float(measurement["median_completion_span_ns"][str(pair_count)])
            for pair_count in pair_counts),
        "completion_ns_per_pair": float(published_rate),
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha256,
    }


def load_aic_grouped_matmul_finite_stream_completion_service(
        repository_root):
    """Unify measured short and long GroupedMatmul completion domains."""

    short = load_aic_grouped_matmul_stream_completion_reference(
        repository_root,
        AIC_GROUPED_MATMUL_SHORT_STREAM_COMPLETION_REFERENCE,
        AIC_GROUPED_MATMUL_SHORT_STREAM_COMPLETION_REFERENCE_SHA256,
        "aic_grouped_matmul_short_stream_completion_v1",
        (16, 32, 64, 128),
        "use only as a lower bound for 24-AIC BF16 M16/N256/K64 operand "
        "streams with 16--128 pairs per worker, four-pair drain semantics, "
        "resident-or-slower A, and DRAM-cold B",
    )
    long = load_aic_grouped_matmul_stream_completion_reference(
        repository_root,
        AIC_GROUPED_MATMUL_LONG_STREAM_COMPLETION_REFERENCE,
        AIC_GROUPED_MATMUL_LONG_STREAM_COMPLETION_REFERENCE_SHA256,
        "aic_grouped_matmul_long_stream_completion_v1",
        (256, 1024, 4096, 8192),
        "use only as a lower bound for 24-AIC BF16 M16/N256/K64 operand "
        "streams with 256--8192 pairs per worker, four-pair drain semantics, "
        "resident-or-slower A, and DRAM-cold B",
    )
    geometry_fields = (
        "a_residency", "b_residency", "active_aic_groups", "a_rows",
        "a_columns", "b_rows", "b_columns", "pair_bytes",
        "drain_interval_pairs", "eviction_bytes",
    )
    if any(short[name] != long[name] for name in geometry_fields):
        raise ValueError(
            "AIC GroupedMatmul completion campaigns changed geometry")
    return {
        "model": "aic_grouped_matmul_finite_stream_completion_v2",
        "dtype": "bf16",
        **{name: short[name] for name in geometry_fields},
        "minimum_pairs_per_worker": short["minimum_pairs_per_worker"],
        "maximum_pairs_per_worker": long["maximum_pairs_per_worker"],
        "measured_pairs_per_worker": (
            short["measured_pairs_per_worker"]
            + long["measured_pairs_per_worker"]),
        "segments": ({
            "model": "piecewise_linear_completion_span",
            "minimum_pairs_per_worker": short[
                "minimum_pairs_per_worker"],
            "maximum_pairs_per_worker": long[
                "maximum_pairs_per_worker"],
            "measured_pairs_per_worker": (
                short["measured_pairs_per_worker"]
                + long["measured_pairs_per_worker"]),
            "completion_span_ns": (
                tuple(
                    pair_count * short["completion_ns_per_pair"]
                    for pair_count in short["measured_pairs_per_worker"])
                + long["completion_span_ns"]),
        },),
        "reference_paths": (
            short["reference_path"], long["reference_path"]),
        "reference_sha256s": (
            short["reference_sha256"], long["reference_sha256"]),
    }


def load_aic_grouped_matmul_k2880_wide_base_tail_service(repository_root):
    """Load the measured finite K2880 wide-base tail completion curve."""

    path = repository_root / (
        AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul K2880 wide-base tail reference SHA256 "
            "mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul K2880 wide-base tail reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "publication_scope",
         "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": (
            "aic_grouped_matmul_k2880_wide_base_tail_completion_v1"),
        "duration_unit": (
            "ns_per_completed_tail_N256_K64_operand_stream_per_aic"),
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"projection_k", "base_m_values", "active_m_values",
         "a_residency", "b_residency", "active_aic_groups", "a_columns",
         "b_rows", "b_columns", "b_source_row_stride",
         "b_source_tiles_per_worker", "copied_ring_bytes",
         "source_allocation_bytes", "drain_interval_pairs",
         "minimum_pairs_per_worker", "maximum_pairs_per_worker",
         "measured_pairs_per_worker"})
    pair_counts = (512, 1024, 3072, 6144)
    frozen_scope = {
        "projection_k": 2880,
        "base_m_values": [64, 128],
        "active_m_values": [16, 32],
        "a_residency": "l2_resident",
        "b_residency": "dram_cold",
        "active_aic_groups": 24,
        "a_columns": 64,
        "b_rows": 64,
        "b_columns": 256,
        "b_source_row_stride": 2880,
        "b_source_tiles_per_worker": 512,
        "copied_ring_bytes": 402653184,
        "source_allocation_bytes": 4465360896,
        "drain_interval_pairs": 4,
        "minimum_pairs_per_worker": pair_counts[0],
        "maximum_pairs_per_worker": pair_counts[-1],
        "measured_pairs_per_worker": list(pair_counts),
    }
    if scope != frozen_scope:
        raise ValueError(f"{owner} publication scope changed")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"campaigns", "conservative_completion_span_ns",
         "maximum_allowed_within_count_relative_span",
         "mechanism_gate_passed"})
    campaigns = measurement["campaigns"]
    expected_campaigns = {
        "m64_tail16": (64, 16),
        "m64_tail32": (64, 32),
        "m128_tail16": (128, 16),
        "m128_tail32": (128, 32),
    }
    if not isinstance(campaigns, dict) or set(campaigns) != set(
            expected_campaigns):
        raise ValueError(f"{owner} campaign axis changed")
    expected_pair_keys = {str(pair_count) for pair_count in pair_counts}
    for name, (base_m, active_m) in expected_campaigns.items():
        campaign = campaigns[name]
        require_exact_fields(
            f"{owner}.measurement.campaigns.{name}", campaign,
            {"base_m", "active_m", "median_completion_span_ns",
             "within_count_relative_spans", "stability_gate"})
        if campaign["base_m"] != base_m or campaign["active_m"] != active_m:
            raise ValueError(f"{owner} campaign geometry changed")
        for field in (
                "median_completion_span_ns", "within_count_relative_spans"):
            if (not isinstance(campaign[field], dict)
                    or set(campaign[field]) != expected_pair_keys):
                raise ValueError(f"{owner} campaign pair axis changed")
        for duration in campaign["median_completion_span_ns"].values():
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f"{owner} completion span is invalid")
        for relative_span in campaign["within_count_relative_spans"].values():
            if (isinstance(relative_span, bool)
                    or not isinstance(relative_span, (int, float))
                    or not 0.0 <= relative_span <= 0.1):
                raise ValueError(f"{owner} stability span is invalid")
        if campaign["stability_gate"] is not True:
            raise ValueError(f"{owner} campaign stability gate failed")
    curve = measurement["conservative_completion_span_ns"]
    if not isinstance(curve, dict) or set(curve) != expected_pair_keys:
        raise ValueError(f"{owner} conservative curve axis changed")
    expected_curve = {
        str(pair_count): min(
            campaigns[name]["median_completion_span_ns"][str(pair_count)]
            for name in campaigns)
        for pair_count in pair_counts
    }
    if curve != expected_curve:
        raise ValueError(f"{owner} conservative curve changed")
    if (measurement["maximum_allowed_within_count_relative_span"] != 0.1
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "snapshot_path", "raw_root",
         "host", "container", "physical_device_id", "logical_device_id",
         "owner", "source_snapshot_sha256", "source_sha256", "campaigns",
         "build_command", "run_command"})
    if (provenance["measurement_baseline_commit"] != (
            "9889e8c4e72c3d4220d97b77aa07f682496d26ae")
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in provenance["raw_root"].split("/")):
        raise ValueError(f"{owner} provenance changed")
    source_hashes = provenance["source_sha256"]
    expected_source_names = {
        "CMakeLists.txt", "README.md",
        "grouped_pair_completion_kernel.cpp",
        "grouped_pair_completion_main.cpp",
        "grouped_pair_completion_runner.py",
    }
    if not isinstance(source_hashes, dict) or set(source_hashes) != (
            expected_source_names):
        raise ValueError(f"{owner} source evidence changed")
    provenance_campaigns = provenance["campaigns"]
    if (not isinstance(provenance_campaigns, dict)
            or set(provenance_campaigns) != set(expected_campaigns)):
        raise ValueError(f"{owner} provenance campaign axis changed")
    evidence_hashes = list(source_hashes.values()) + [
        provenance["source_snapshot_sha256"]]
    for campaign in provenance_campaigns.values():
        require_exact_fields(
            f"{owner}.provenance.campaign", campaign,
            {"binary_sha256", "raw_sha256", "stderr_sha256"})
        if (not isinstance(campaign["raw_sha256"], list)
                or len(campaign["raw_sha256"]) != 3
                or not isinstance(campaign["stderr_sha256"], list)
                or len(campaign["stderr_sha256"]) != 3):
            raise ValueError(f"{owner} captured run count changed")
        evidence_hashes.extend([campaign["binary_sha256"]])
        evidence_hashes.extend(campaign["raw_sha256"])
        evidence_hashes.extend(campaign["stderr_sha256"])
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "use only as a lower bound for 24-AIC BF16 M16/M32 tail "
            "operand streams under source-selected base-M64/base-M128, "
            "projection-K2880/N256/K64, 512--6144 pairs per worker, "
            "four-pair drain semantics, resident-or-slower A, and "
            "DRAM-cold B"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": (
            "aic_grouped_matmul_k2880_wide_base_tail_completion_v1"),
        "dtype": "bf16",
        **{
            name: (tuple(item) if isinstance(item, list) else item)
            for name, item in frozen_scope.items()
        },
        "completion_span_ns": tuple(
            float(curve[str(pair_count)]) for pair_count in pair_counts),
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_K2880_WIDE_BASE_TAIL_REFERENCE_SHA256),
    }


def load_aic_grouped_matmul_k2880_output_tile_service(repository_root):
    """Load one-IterateAll K2880 output-tile completion spans."""

    path = repository_root / AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul K2880 output-tile reference SHA256 mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul K2880 output-tile reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "publication_scope",
         "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": (
            "aic_grouped_matmul_k2880_output_tile_completion_v1"),
        "duration_unit": (
            "ns_per_completed_output_tile_N256_K64_operand_stream_per_aic"),
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    pair_counts = (32, 44, 48, 64)
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"projection_k", "output_tile_pairs", "base_m_values",
         "active_m_values", "a_residency", "b_residency",
         "active_aic_groups", "a_columns", "b_rows", "b_columns",
         "b_source_row_stride",
         "b_source_tiles_per_worker_matches_pair_count",
         "maximum_copied_ring_bytes", "source_allocation_bytes",
         "drain_interval_pairs", "minimum_measured_pairs_per_worker",
         "maximum_measured_pairs_per_worker",
         "measured_pairs_per_worker"})
    frozen_scope = {
        "projection_k": 2880,
        "output_tile_pairs": 45,
        "base_m_values": [128],
        "active_m_values": [16, 32, 48],
        "a_residency": "l2_resident",
        "b_residency": "dram_cold",
        "active_aic_groups": 24,
        "a_columns": 64,
        "b_rows": 64,
        "b_columns": 256,
        "b_source_row_stride": 2880,
        "b_source_tiles_per_worker_matches_pair_count": True,
        "maximum_copied_ring_bytes": 50331648,
        "source_allocation_bytes": 4465360896,
        "drain_interval_pairs": 4,
        "minimum_measured_pairs_per_worker": pair_counts[0],
        "maximum_measured_pairs_per_worker": pair_counts[-1],
        "measured_pairs_per_worker": list(pair_counts),
    }
    if scope != frozen_scope:
        raise ValueError(f"{owner} publication scope changed")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"campaigns", "maximum_allowed_within_count_relative_span",
         "cross_active_m_relative_span",
         "maximum_allowed_cross_active_m_relative_span",
         "mechanism_gate_passed"})
    campaigns = measurement["campaigns"]
    expected_campaigns = {
        "m128_tail16": 16,
        "m128_tail32": 32,
        "m128_tail48": 48,
    }
    if not isinstance(campaigns, dict) or set(campaigns) != set(
            expected_campaigns):
        raise ValueError(f"{owner} campaign axis changed")
    pair_keys = {str(pair_count) for pair_count in pair_counts}
    completion_by_active_m = {}
    for name, active_m in expected_campaigns.items():
        campaign = campaigns[name]
        require_exact_fields(
            f"{owner}.measurement.campaigns.{name}", campaign,
            {"base_m", "active_m", "median_completion_span_ns",
             "within_count_relative_spans",
             "output_tile_completion_span_ns", "stability_gate"})
        if campaign["base_m"] != 128 or campaign["active_m"] != active_m:
            raise ValueError(f"{owner} campaign geometry changed")
        medians = campaign["median_completion_span_ns"]
        spans = campaign["within_count_relative_spans"]
        if (not isinstance(medians, dict) or set(medians) != pair_keys
                or not isinstance(spans, dict) or set(spans) != pair_keys):
            raise ValueError(f"{owner} campaign pair axis changed")
        if any(isinstance(duration, bool)
               or not isinstance(duration, (int, float))
               or not math.isfinite(duration) or duration <= 0
               for duration in medians.values()):
            raise ValueError(f"{owner} completion span is invalid")
        if any(isinstance(span, bool) or not isinstance(span, (int, float))
               or not 0.0 <= span <= 0.1 for span in spans.values()):
            raise ValueError(f"{owner} stability span is invalid")
        interpolated = linear_interpolate(
            pair_counts,
            tuple(float(medians[str(count)]) for count in pair_counts),
            scope["output_tile_pairs"])
        if (not math.isclose(
                campaign["output_tile_completion_span_ns"], interpolated)
                or campaign["stability_gate"] is not True):
            raise ValueError(f"{owner} output-tile derivation changed")
        completion_by_active_m[active_m] = interpolated
    if (measurement["maximum_allowed_within_count_relative_span"] != 0.1
            or measurement[
                "maximum_allowed_cross_active_m_relative_span"] != 0.1
            or not 0.0 <= measurement[
                "cross_active_m_relative_span"] <= 0.1
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "raw_root",
         "host", "container", "physical_device_ids", "logical_device_id",
         "owner", "source_snapshot_sha256", "source_sha256",
         "cann_source_path", "cann_source_sha256", "cann_source_boundary",
         "campaigns", "build_command", "run_command"})
    if (provenance["measurement_baseline_commit"] != (
            "8496d80eb0a48c9e7bee527c0904749f1bda0560")
            or not isinstance(provenance["physical_device_ids"], list)
            or len(provenance["physical_device_ids"]) != 1
            or any(isinstance(device_id, bool)
                   or not isinstance(device_id, int) or device_id < 0
                   for device_id in provenance["physical_device_ids"])
            or len(set(provenance["physical_device_ids"])) != 1
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in provenance["raw_root"].split("/")
            or provenance["cann_source_sha256"] != (
                "622409b7f5ce03b0c23c018eb6e3a360018867eb764fa261ab83a25d39456c32")):
        raise ValueError(f"{owner} provenance changed")
    source_hashes = provenance["source_sha256"]
    expected_source_names = {
        "CMakeLists.txt", "README.md", "grouped_pair_completion_kernel.cpp",
        "grouped_pair_completion_main.cpp",
        "grouped_pair_completion_runner.py",
    }
    if not isinstance(source_hashes, dict) or set(source_hashes) != (
            expected_source_names):
        raise ValueError(f"{owner} source evidence changed")
    provenance_campaigns = provenance["campaigns"]
    if (not isinstance(provenance_campaigns, dict)
            or set(provenance_campaigns) != set(expected_campaigns)):
        raise ValueError(f"{owner} provenance campaign axis changed")
    evidence_hashes = list(source_hashes.values()) + [
        provenance["source_snapshot_sha256"], provenance["cann_source_sha256"]]
    for name, campaign in provenance_campaigns.items():
        require_exact_fields(
            f"{owner}.provenance.campaigns.{name}", campaign,
            {"physical_device_id", "binary_sha256", "cmake_cache_sha256",
             "raw_sha256", "stderr_sha256"})
        if (isinstance(campaign["physical_device_id"], bool)
                or not isinstance(campaign["physical_device_id"], int)
                or campaign["physical_device_id"] < 0
                or not isinstance(campaign["raw_sha256"], list)
                or len(campaign["raw_sha256"]) != 3
                or not isinstance(campaign["stderr_sha256"], list)
                or len(campaign["stderr_sha256"]) != 3):
            raise ValueError(f"{owner} captured run provenance changed")
        evidence_hashes.extend((
            campaign["binary_sha256"], campaign["cmake_cache_sha256"]))
        evidence_hashes.extend(campaign["raw_sha256"])
        evidence_hashes.extend(campaign["stderr_sha256"])
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "use only as a per-output-tile lower bound for 24-AIC BF16 "
            "M16/M32/M48 tails under source-selected base-M128, "
            "projection-K2880/N256/K64, 45 K steps per IterateAll call, "
            "four-pair drain semantics, resident-or-slower A, and "
            "DRAM-cold B"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": "aic_grouped_matmul_k2880_output_tile_completion_v1",
        "dtype": "bf16",
        **{
            name: (tuple(item) if isinstance(item, list) else item)
            for name, item in frozen_scope.items()
        },
        "completion_span_ns_by_active_m": completion_by_active_m,
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_K2880_OUTPUT_TILE_REFERENCE_SHA256),
    }


def load_aic_grouped_matmul_step4_worker_sequence_base_service(
        repository_root):
    """Load the immutable first step-4 GroupedMatmul sequence campaign."""

    path = repository_root / (
        AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul step-4 worker-sequence reference SHA256 "
            "mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul step-4 worker-sequence reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "publication_scope",
         "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": (
            "aic_grouped_matmul_step4_worker_sequence_completion_v1"),
        "duration_unit": "ns_per_completed_worker_sequence",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"dtype", "active_aic_groups", "base_m", "active_m", "tile_n",
         "tile_k", "depth_a1", "depth_b1", "step_ka", "step_kb",
         "db_l0a", "db_l0b", "db_l0c", "source_operand_layout",
         "roles"})
    frozen_scope = {
        "dtype": "bf16", "active_aic_groups": 24, "base_m": 128,
        "active_m": 16, "tile_n": 256, "tile_k": 64,
        "depth_a1": 8, "depth_b1": 8, "step_ka": 4, "step_kb": 4,
        "db_l0a": 2, "db_l0b": 2, "db_l0c": 1,
        "source_operand_layout": "disjoint_a_b_c_per_iterate_all_call",
    }
    if any(scope[name] != expected_value
           for name, expected_value in frozen_scope.items()):
        raise ValueError(f"{owner} publication scope changed")
    expected_roles = {
        "w1": {
            "projection_k": 4096, "projection_n": 1024,
            "minimum_output_tiles_per_worker": 47,
            "maximum_output_tiles_per_worker": 80,
            "measured_output_tiles_per_worker": [47, 64, 80],
        },
        "w2": {
            "projection_k": 1024, "projection_n": 4096,
            "minimum_output_tiles_per_worker": 95,
            "maximum_output_tiles_per_worker": 160,
            "measured_output_tiles_per_worker": [95, 128, 160],
        },
    }
    roles = scope["roles"]
    if roles != expected_roles:
        raise ValueError(f"{owner} role domains changed")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"completion_span_ns", "median_completion_span_ns",
         "within_count_relative_spans",
         "maximum_allowed_within_count_relative_span",
         "mechanism_gate_passed"})
    if (set(measurement["completion_span_ns"]) != set(expected_roles)
            or set(measurement["median_completion_span_ns"])
            != set(expected_roles)
            or set(measurement["within_count_relative_spans"])
            != set(expected_roles)
            or measurement["maximum_allowed_within_count_relative_span"]
            != 0.05
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate changed")
    completion_by_role = {}
    for role, role_scope in expected_roles.items():
        counts = tuple(role_scope["measured_output_tiles_per_worker"])
        keys = {str(count) for count in counts}
        samples = measurement["completion_span_ns"][role]
        medians = measurement["median_completion_span_ns"][role]
        spans = measurement["within_count_relative_spans"][role]
        if set(samples) != keys or set(medians) != keys or set(spans) != keys:
            raise ValueError(f"{owner} {role} measurement axis changed")
        normalized_medians = []
        for count in counts:
            key = str(count)
            runs = samples[key]
            median = medians[key]
            relative_span = spans[key]
            if (not isinstance(runs, list) or len(runs) != 3
                    or any(isinstance(run, bool)
                           or not isinstance(run, (int, float))
                           or not math.isfinite(run) or run <= 0
                           for run in runs)
                    or isinstance(median, bool)
                    or not isinstance(median, (int, float))
                    or not math.isfinite(median) or median <= 0
                    or float(sorted(runs)[1]) != float(median)
                    or isinstance(relative_span, bool)
                    or not isinstance(relative_span, (int, float))
                    or not math.isfinite(relative_span)
                    or not math.isclose(
                        float(relative_span),
                        (max(runs) - min(runs)) / float(median),
                        rel_tol=0.0, abs_tol=1e-12)
                    or relative_span > 0.05):
                raise ValueError(
                    f"{owner} {role} {count}-tile measurement is invalid")
            normalized_medians.append(float(median))
        completion_by_role[role] = {
            **{
                name: (tuple(item) if isinstance(item, list) else item)
                for name, item in role_scope.items()
            },
            "completion_span_ns": tuple(normalized_medians),
        }
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "snapshot_path", "raw_root",
         "host", "container", "physical_device_id", "logical_device_id",
         "owner", "source_sha256", "binary_sha256", "raw_sha256",
         "cann_source_path", "cann_source_sha256"})
    if (provenance["measurement_baseline_commit"] != (
            "4be298d2bb0f78b125d047ac5093beffb8c79e30")
            or (not isinstance(provenance["host"], str)
                or not provenance["host"].strip())
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in provenance["raw_root"].split("/")
            or provenance["cann_source_sha256"] != (
                "622409b7f5ce03b0c23c018eb6e3a360018867eb764fa261ab83a25d39456c32")):
        raise ValueError(f"{owner} provenance changed")
    source_hashes = provenance["source_sha256"]
    binary_hashes = provenance["binary_sha256"]
    raw_hashes = provenance["raw_sha256"]
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != {
                "CMakeLists.txt", "README.md",
                "grouped_matmul_tile_completion_kernel.cpp.in",
                "grouped_matmul_tile_completion_main.cpp"}
            or not isinstance(binary_hashes, dict)
            or set(binary_hashes) != set(expected_roles)
            or not isinstance(raw_hashes, dict)
            or set(raw_hashes) != set(expected_roles)):
        raise ValueError(f"{owner} captured evidence axis changed")
    evidence_hashes = list(source_hashes.values()) + [
        provenance["cann_source_sha256"]] + list(binary_hashes.values())
    for role, role_scope in expected_roles.items():
        role_hashes = raw_hashes[role]
        expected_keys = {
            str(count)
            for count in role_scope["measured_output_tiles_per_worker"]}
        if not isinstance(role_hashes, dict) or set(role_hashes) != expected_keys:
            raise ValueError(f"{owner} {role} raw evidence axis changed")
        for hashes in role_hashes.values():
            if not isinstance(hashes, list) or len(hashes) != 3:
                raise ValueError(f"{owner} {role} captured run count changed")
            evidence_hashes.extend(hashes)
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "replace the analytical worker span only for 24-AIC BF16 "
            "base-M128/active-M16/N256/K64 GroupedMatmul programs with "
            "depthA1/depthB1 8/8, stepKa/stepKb 4/4, double-buffered "
            "L0A/L0B, single-buffered L0C, DRAM-cold B, only padded-M16 "
            "classes, and a per-worker output-tile count inside the measured "
            "role-specific interval"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": (
            "aic_grouped_matmul_step4_worker_sequence_completion_v1"),
        **frozen_scope,
        "roles": completion_by_role,
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE_SHA256),
    }


def load_aic_grouped_matmul_step4_worker_sequence_service(repository_root):
    """Extend the immutable short curve with a separately captured long span."""

    base = load_aic_grouped_matmul_step4_worker_sequence_base_service(
        repository_root)
    path = repository_root / (
        AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul long worker-sequence reference SHA256 "
            "mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul long worker-sequence reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "base_reference_path",
         "base_reference_sha256", "publication_scope", "measurement",
         "provenance", "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": (
            "aic_grouped_matmul_step4_long_worker_sequence_completion_v1"),
        "duration_unit": "ns_per_completed_worker_sequence",
        "base_reference_path": str(
            AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE),
        "base_reference_sha256": (
            AIC_GROUPED_MATMUL_STEP4_WORKER_SEQUENCE_REFERENCE_SHA256),
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"dtype", "active_aic_groups", "base_m", "active_m", "tile_n",
         "tile_k", "depth_a1", "depth_b1", "step_ka", "step_kb",
         "db_l0a", "db_l0b", "db_l0c", "source_operand_layout",
         "roles"})
    frozen_scope = {
        "dtype": "bf16", "active_aic_groups": 24, "base_m": 128,
        "active_m": 16, "tile_n": 256, "tile_k": 64,
        "depth_a1": 8, "depth_b1": 8, "step_ka": 4, "step_kb": 4,
        "db_l0a": 2, "db_l0b": 2, "db_l0c": 1,
        "source_operand_layout": "disjoint_a_b_c_per_iterate_all_call",
    }
    if any(scope[name] != expected_value
           for name, expected_value in frozen_scope.items()):
        raise ValueError(f"{owner} publication scope changed")
    expected_roles = {
        "w1": {
            "projection_k": 4096, "projection_n": 1024,
            "base_measured_output_tiles_per_worker": [47, 64, 80],
            "extension_measured_output_tiles_per_worker": [96, 128, 192],
            "minimum_output_tiles_per_worker": 47,
            "maximum_output_tiles_per_worker": 192,
        },
        "w2": {
            "projection_k": 1024, "projection_n": 4096,
            "base_measured_output_tiles_per_worker": [95, 128, 160],
            "extension_measured_output_tiles_per_worker": [176, 192, 224, 240],
            "minimum_output_tiles_per_worker": 95,
            "maximum_output_tiles_per_worker": 240,
        },
    }
    if scope["roles"] != expected_roles:
        raise ValueError(f"{owner} role domains changed")
    for role, expected_role in expected_roles.items():
        base_role = base["roles"][role]
        if (base_role["projection_k"] != expected_role["projection_k"]
                or base_role["projection_n"] != expected_role["projection_n"]
                or tuple(expected_role[
                    "base_measured_output_tiles_per_worker"])
                != base_role["measured_output_tiles_per_worker"]
                or expected_role["minimum_output_tiles_per_worker"]
                != base_role["minimum_output_tiles_per_worker"]):
            raise ValueError(f"{owner} {role} base curve disagrees")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"completion_span_ns", "median_completion_span_ns",
         "within_count_relative_spans",
         "maximum_allowed_within_count_relative_span",
         "mechanism_gate_passed"})
    if (set(measurement["completion_span_ns"]) != set(expected_roles)
            or set(measurement["median_completion_span_ns"])
            != set(expected_roles)
            or set(measurement["within_count_relative_spans"])
            != set(expected_roles)
            or measurement["maximum_allowed_within_count_relative_span"]
            != 0.05
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate changed")
    extension_medians = {}
    for role, role_scope in expected_roles.items():
        counts = tuple(role_scope[
            "extension_measured_output_tiles_per_worker"])
        keys = {str(count) for count in counts}
        samples = measurement["completion_span_ns"][role]
        medians = measurement["median_completion_span_ns"][role]
        spans = measurement["within_count_relative_spans"][role]
        if set(samples) != keys or set(medians) != keys or set(spans) != keys:
            raise ValueError(f"{owner} {role} measurement axis changed")
        normalized = []
        for count in counts:
            key = str(count)
            runs = samples[key]
            median = medians[key]
            relative_span = spans[key]
            if (not isinstance(runs, list) or len(runs) != 3
                    or any(isinstance(run, bool)
                           or not isinstance(run, (int, float))
                           or not math.isfinite(run) or run <= 0
                           for run in runs)
                    or isinstance(median, bool)
                    or not isinstance(median, (int, float))
                    or not math.isfinite(median) or median <= 0
                    or float(sorted(runs)[1]) != float(median)
                    or isinstance(relative_span, bool)
                    or not isinstance(relative_span, (int, float))
                    or not math.isfinite(relative_span)
                    or not math.isclose(
                        float(relative_span),
                        (max(runs) - min(runs)) / float(median),
                        rel_tol=0.0, abs_tol=1e-12)
                    or relative_span > 0.05):
                raise ValueError(
                    f"{owner} {role} {count}-tile measurement is invalid")
            normalized.append(float(median))
        extension_medians[role] = tuple(normalized)
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "raw_root", "run_plan_sha256",
         "host", "container", "physical_device_id", "logical_device_id",
         "owner", "initial_source_snapshot_path",
         "initial_binary_sha256", "extended_source_snapshot_path",
         "extended_source_sha256", "extended_w2_binary_sha256",
         "raw_sha256", "cann_source_path", "cann_source_sha256"})
    if (provenance["measurement_baseline_commit"] != (
            "68a1210325ec05a00b78da512911d9f4d150f28f")
            or (not isinstance(provenance["host"], str)
                or not provenance["host"].strip())
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in provenance["raw_root"].split("/")
            or provenance["cann_source_sha256"] != (
                "622409b7f5ce03b0c23c018eb6e3a360018867eb764fa261ab83a25d39456c32")):
        raise ValueError(f"{owner} provenance changed")
    initial_binaries = provenance["initial_binary_sha256"]
    extended_source = provenance["extended_source_sha256"]
    raw_hashes = provenance["raw_sha256"]
    required_source_names = {
        "CMakeLists.txt", "README.md",
        "grouped_matmul_tile_completion_kernel.cpp.in",
        "grouped_matmul_tile_completion_main.cpp"}
    if (not isinstance(initial_binaries, dict)
            or set(initial_binaries) != set(expected_roles)
            or not isinstance(extended_source, dict)
            or set(extended_source) != required_source_names
            or not isinstance(raw_hashes, dict)
            or set(raw_hashes) != set(expected_roles)):
        raise ValueError(f"{owner} captured evidence axis changed")
    evidence_hashes = [
        provenance["run_plan_sha256"], provenance["cann_source_sha256"],
        provenance["extended_w2_binary_sha256"],
        *initial_binaries.values(), *extended_source.values()]
    for role, role_scope in expected_roles.items():
        role_hashes = raw_hashes[role]
        expected_keys = {
            str(count)
            for count in role_scope[
                "extension_measured_output_tiles_per_worker"]}
        if not isinstance(role_hashes, dict) or set(role_hashes) != expected_keys:
            raise ValueError(f"{owner} {role} raw evidence axis changed")
        for hashes in role_hashes.values():
            if not isinstance(hashes, list) or len(hashes) != 3:
                raise ValueError(f"{owner} {role} captured run count changed")
            evidence_hashes.extend(hashes)
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "extend the source-complete worker span only for the v1 24-AIC "
            "BF16 base-M128/active-M16/N256/K64 GroupedMatmul program, "
            "DRAM-cold B, only padded-M16 classes, and a per-worker "
            "output-tile count inside the v2 role-specific measured interval"):
        raise ValueError(f"{owner} forward eligibility changed")
    completion_by_role = {}
    for role, role_scope in expected_roles.items():
        base_role = base["roles"][role]
        completion_by_role[role] = {
            "projection_k": base_role["projection_k"],
            "projection_n": base_role["projection_n"],
            "minimum_output_tiles_per_worker": role_scope[
                "minimum_output_tiles_per_worker"],
            "maximum_output_tiles_per_worker": role_scope[
                "maximum_output_tiles_per_worker"],
            "measured_output_tiles_per_worker": (
                base_role["measured_output_tiles_per_worker"]
                + tuple(role_scope[
                    "extension_measured_output_tiles_per_worker"])),
            "completion_span_ns": (
                base_role["completion_span_ns"] + extension_medians[role]),
        }
    long_service = {
        "model": (
            "aic_grouped_matmul_step4_long_worker_sequence_completion_v1"),
        **frozen_scope,
        "roles": completion_by_role,
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE_SHA256),
    }
    return load_aic_grouped_matmul_step4_w2_bounded_b_ring_extension_service(
        repository_root, long_service)


def load_aic_grouped_matmul_step4_w2_bounded_b_ring_extension_service(
        repository_root, base):
    """Extend only the bridged W2 sequence domain through 384 tiles."""

    path = repository_root / (
        AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul step-4 W2 bounded-B extension reference "
            "SHA256 mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul step-4 W2 bounded-B extension reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "base_reference_path",
         "base_reference_sha256", "publication_scope", "measurement",
         "provenance", "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": (
            "aic_grouped_matmul_step4_w2_bounded_b_ring_extension_v1"),
        "duration_unit": "ns_per_completed_worker_sequence",
        "base_reference_path": str(
            AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE),
        "base_reference_sha256": (
            AIC_GROUPED_MATMUL_STEP4_LONG_WORKER_SEQUENCE_REFERENCE_SHA256),
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"dtype", "active_aic_groups", "base_m", "active_m", "tile_n",
         "tile_k", "depth_a1", "depth_b1", "step_ka", "step_kb",
         "db_l0a", "db_l0b", "db_l0c", "source_operand_layout",
         "role"})
    expected_scope = {
        "dtype": "bf16", "active_aic_groups": 24, "base_m": 128,
        "active_m": 16, "tile_n": 256, "tile_k": 64,
        "depth_a1": 8, "depth_b1": 8, "step_ka": 4, "step_kb": 4,
        "db_l0a": 2, "db_l0b": 2, "db_l0c": 1,
        "source_operand_layout": (
            "bounded_b_ring_128_bridged_to_disjoint_a_b_c_per_"
            "iterate_all_call"),
        "role": {
            "name": "w2", "projection_k": 1024, "projection_n": 4096,
            "base_maximum_output_tiles_per_worker": 240,
            "extended_maximum_output_tiles_per_worker": 384,
            "bridge_output_tiles_per_worker": [128, 176, 240],
            "extension_output_tiles_per_worker": [288, 320, 352, 384],
        },
    }
    if scope != expected_scope:
        raise ValueError(f"{owner} publication scope changed")
    base_role = base["roles"]["w2"]
    if (base["model"]
            != "aic_grouped_matmul_step4_long_worker_sequence_completion_v1"
            or base_role["projection_k"] != scope["role"]["projection_k"]
            or base_role["projection_n"] != scope["role"]["projection_n"]
            or base_role["maximum_output_tiles_per_worker"]
            != scope["role"]["base_maximum_output_tiles_per_worker"]):
        raise ValueError(f"{owner} base service disagrees")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"bridge_completion_span_ns",
         "bridge_reference_median_completion_span_ns",
         "bridge_median_completion_span_ns", "bridge_relative_deltas",
         "bridge_within_count_relative_spans", "extension_completion_span_ns",
         "extension_median_completion_span_ns",
         "extension_within_count_relative_spans",
         "maximum_allowed_within_count_relative_span",
         "maximum_allowed_bridge_relative_delta", "mechanism_gate_passed"})
    bridge_counts = tuple(scope["role"]["bridge_output_tiles_per_worker"])
    extension_counts = tuple(
        scope["role"]["extension_output_tiles_per_worker"])
    bridge_keys = {str(count) for count in bridge_counts}
    extension_keys = {str(count) for count in extension_counts}
    if (set(measurement["bridge_completion_span_ns"]) != bridge_keys
            or set(measurement[
                "bridge_reference_median_completion_span_ns"]) != bridge_keys
            or set(measurement["bridge_median_completion_span_ns"])
            != bridge_keys
            or set(measurement["bridge_relative_deltas"]) != bridge_keys
            or set(measurement["bridge_within_count_relative_spans"])
            != bridge_keys
            or set(measurement["extension_completion_span_ns"])
            != extension_keys
            or set(measurement["extension_median_completion_span_ns"])
            != extension_keys
            or set(measurement["extension_within_count_relative_spans"])
            != extension_keys
            or measurement["maximum_allowed_within_count_relative_span"]
            != 0.05
            or measurement["maximum_allowed_bridge_relative_delta"] != 0.05
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} measurement axis changed")
    base_count_axis = base_role["measured_output_tiles_per_worker"]
    base_span_axis = base_role["completion_span_ns"]
    for count in bridge_counts:
        key = str(count)
        runs = measurement["bridge_completion_span_ns"][key]
        reference = measurement["bridge_reference_median_completion_span_ns"][key]
        median = measurement["bridge_median_completion_span_ns"][key]
        relative_delta = measurement["bridge_relative_deltas"][key]
        relative_span = measurement["bridge_within_count_relative_spans"][key]
        if (not isinstance(runs, list) or len(runs) != 3
                or any(isinstance(run, bool)
                       or not isinstance(run, (int, float))
                       or not math.isfinite(run) or run <= 0 for run in runs)
                or isinstance(reference, bool)
                or not isinstance(reference, (int, float))
                or not math.isfinite(reference) or reference <= 0
                or isinstance(median, bool)
                or not isinstance(median, (int, float))
                or not math.isfinite(median) or median <= 0
                or float(sorted(runs)[1]) != float(median)
                or not math.isclose(
                    float(reference), linear_interpolate(
                        base_count_axis, base_span_axis, count),
                    rel_tol=0.0, abs_tol=1e-12)
                or isinstance(relative_delta, bool)
                or not isinstance(relative_delta, (int, float))
                or not math.isclose(
                    float(relative_delta),
                    (float(median) - float(reference)) / float(reference),
                    rel_tol=0.0, abs_tol=1e-12)
                or abs(relative_delta) > 0.05
                or isinstance(relative_span, bool)
                or not isinstance(relative_span, (int, float))
                or not math.isclose(
                    float(relative_span),
                    (max(runs) - min(runs)) / float(median),
                    rel_tol=0.0, abs_tol=1e-12)
                or relative_span > 0.05):
            raise ValueError(f"{owner} {count}-tile bridge is invalid")
    extension_medians = []
    for count in extension_counts:
        key = str(count)
        runs = measurement["extension_completion_span_ns"][key]
        median = measurement["extension_median_completion_span_ns"][key]
        relative_span = measurement["extension_within_count_relative_spans"][key]
        if (not isinstance(runs, list) or len(runs) != 3
                or any(isinstance(run, bool)
                       or not isinstance(run, (int, float))
                       or not math.isfinite(run) or run <= 0 for run in runs)
                or isinstance(median, bool)
                or not isinstance(median, (int, float))
                or not math.isfinite(median) or median <= 0
                or float(sorted(runs)[1]) != float(median)
                or isinstance(relative_span, bool)
                or not isinstance(relative_span, (int, float))
                or not math.isclose(
                    float(relative_span),
                    (max(runs) - min(runs)) / float(median),
                    rel_tol=0.0, abs_tol=1e-12)
                or relative_span > 0.05):
            raise ValueError(f"{owner} {count}-tile extension is invalid")
        extension_medians.append(float(median))
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "raw_root", "run_plan_sha256",
         "host", "container", "physical_device_id", "logical_device_id",
         "owner", "source_basis_snapshot_path",
         "bounded_b_ring_source_snapshot_path",
         "bounded_b_ring_source_sha256", "bounded_b_ring_w2_binary_sha256",
         "raw_sha256", "cann_source_path", "cann_source_sha256"})
    if (provenance["measurement_baseline_commit"] != (
            "c1589b778e05b688bcb87e8a12cf640c30c046fa")
            or (not isinstance(provenance["host"], str)
                or not provenance["host"].strip())
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in provenance["raw_root"].split("/")
            or provenance["cann_source_sha256"] != (
                "622409b7f5ce03b0c23c018eb6e3a360018867eb764fa261ab83a25d39456c32")):
        raise ValueError(f"{owner} provenance changed")
    source_hashes = provenance["bounded_b_ring_source_sha256"]
    raw_hashes = provenance["raw_sha256"]
    expected_source_names = {
        "CMakeLists.txt", "README.md",
        "grouped_matmul_tile_completion_kernel.cpp.in",
        "grouped_matmul_tile_completion_main.cpp"}
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != expected_source_names
            or not isinstance(raw_hashes, dict)
            or set(raw_hashes) != bridge_keys | extension_keys):
        raise ValueError(f"{owner} captured evidence axis changed")
    evidence_hashes = [
        provenance["run_plan_sha256"], provenance["cann_source_sha256"],
        provenance["bounded_b_ring_w2_binary_sha256"], *source_hashes.values()]
    for hashes in raw_hashes.values():
        if not isinstance(hashes, list) or len(hashes) != 3:
            raise ValueError(f"{owner} captured run count changed")
        evidence_hashes.extend(hashes)
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "extend only the existing 24-AIC BF16 base-M128/active-M16/"
            "N256/K64 step-4 W2 worker-sequence service from 241 through "
            "384 tiles per worker after the bounded 128-slot B ring bridges "
            "the fully disjoint B reference at 128, 176 and 240 tiles"):
        raise ValueError(f"{owner} forward eligibility changed")
    roles = dict(base["roles"])
    roles["w2"] = {
        **base_role,
        "maximum_output_tiles_per_worker": scope["role"][
            "extended_maximum_output_tiles_per_worker"],
        "measured_output_tiles_per_worker": (
            base_role["measured_output_tiles_per_worker"] + extension_counts),
        "completion_span_ns": (
            base_role["completion_span_ns"] + tuple(extension_medians)),
    }
    return {
        **base,
        "model": "aic_grouped_matmul_step4_w2_bounded_b_ring_extension_v1",
        "roles": roles,
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_STEP4_W2_BOUNDED_B_RING_EXTENSION_REFERENCE_SHA256),
    }


def load_aic_grouped_matmul_short_source_prelude_service(repository_root):
    """Load the exact base-M16 short-program controller prelude service."""

    path = repository_root / AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul short-source prelude reference SHA256 mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul short-source prelude reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "publication_scope",
         "source_scope", "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann-8.5.1",
        "mechanism_id": (
            "aic_grouped_matmul_short_source_prelude_control_v1"),
        "duration_unit": "ns_per_nonempty_output_tile_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"dtype", "active_aic_groups", "base_m", "active_m", "tile_n",
         "tile_k", "depth_a1", "depth_b1", "step_ka", "step_kb",
         "db_l0a", "db_l0b", "db_l0c", "k_rounds"})
    expected_scope = {
        "dtype": "bf16", "active_aic_groups": 24, "base_m": 16,
        "active_m": 16, "tile_n": 256, "tile_k": 64,
        "depth_a1": 120, "depth_b1": 8, "step_ka": 60, "step_kb": 4,
        "db_l0a": 2, "db_l0b": 2, "db_l0c": 1,
        "k_rounds": [45, 96, 256],
    }
    if scope != expected_scope:
        raise ValueError(f"{owner} publication scope changed")
    source_scope = value["source_scope"]
    require_exact_fields(
        f"{owner}.source_scope", source_scope,
        {"upstream_file", "source_sequence", "excluded_source_work"})
    if (source_scope["upstream_file"] != "grouped_matmul/grouped_matmul.h"
            or source_scope["source_sequence"] != (
                "GMMProcess::MNBlockIdxCompute followed by "
                "GMMCompute::MMCompute M/N-tail and X/W/Y-offset setup plus "
                "the base-K A1/B1/L0 scheduler counters before "
                "IterateAll<false>")
            or source_scope["excluded_source_work"] != [
                "GetTensorAddr list loads", "MTE transfer", "Cube arithmetic",
                "FixPipe output", "vector work", "routing", "activation",
                "reduction", "complete production operator"]):
        raise ValueError(f"{owner} source scope changed")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"model", "repeats_per_launch", "fresh_process_trials",
         "tile_counts", "system_counter_tick_ns", "selection_policy",
         "service_time_ns_by_k_rounds", "completion_span_ns",
         "maximum_fresh_process_relative_span",
         "non_tiny_per_tile_relative_span_by_k_rounds",
         "maximum_allowed_relative_span", "mechanism_gate_passed"})
    if (measurement["model"] != (
            "exact_k_round_lookup_with_minimum_non_tiny_per_tile_service_v1")
            or measurement["repeats_per_launch"] != 256
            or measurement["fresh_process_trials"] != 3
            or measurement["tile_counts"] != [1, 4, 16, 64]
            or measurement["system_counter_tick_ns"] != 20.0
            or measurement["selection_policy"] != (
                "for each exact K-round, select the minimum median per-tile "
                "service among non-tiny tile counts 4, 16 and 64; no "
                "production/evaluation timing or APE selects a value")
            or measurement["maximum_allowed_relative_span"] != 0.05
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} measurement contract changed")
    k_rounds = tuple(scope["k_rounds"])
    tile_counts = tuple(measurement["tile_counts"])
    keys = {str(rounds) for rounds in k_rounds}
    if (set(measurement["service_time_ns_by_k_rounds"]) != keys
            or set(measurement["completion_span_ns"]) != keys
            or set(measurement[
                "non_tiny_per_tile_relative_span_by_k_rounds"]) != keys):
        raise ValueError(f"{owner} measurement axis changed")
    service_time_ns_by_k_rounds = {}
    observed_fresh_spans = []
    for rounds in k_rounds:
        key = str(rounds)
        published = measurement["service_time_ns_by_k_rounds"][key]
        completion_by_tile_count = measurement["completion_span_ns"][key]
        if set(completion_by_tile_count) != {str(count) for count in tile_counts}:
            raise ValueError(f"{owner} {rounds}-round tile axis changed")
        medians = {}
        for tile_count in tile_counts:
            runs = completion_by_tile_count[str(tile_count)]
            if (not isinstance(runs, list) or len(runs) != 3
                    or any(isinstance(run, bool)
                           or not isinstance(run, (int, float))
                           or not math.isfinite(run) or run <= 0
                           for run in runs)):
                raise ValueError(
                    f"{owner} {rounds}-round {tile_count}-tile capture is invalid")
            median = float(sorted(runs)[1])
            relative_span = (max(runs) - min(runs)) / median
            if relative_span > measurement["maximum_allowed_relative_span"]:
                raise ValueError(
                    f"{owner} {rounds}-round {tile_count}-tile capture is unstable")
            observed_fresh_spans.append(relative_span)
            medians[tile_count] = median
        selected = min(
            medians[tile_count] / tile_count
            for tile_count in tile_counts if tile_count != 1)
        if (isinstance(published, bool)
                or not isinstance(published, (int, float))
                or not math.isfinite(published) or published <= 0
                or not math.isclose(
                    float(published), selected,
                    rel_tol=0.0, abs_tol=1.0e-12)):
            raise ValueError(f"{owner} {rounds}-round selected service is invalid")
        non_tiny_span = measurement[
            "non_tiny_per_tile_relative_span_by_k_rounds"][key]
        per_tile_values = tuple(
            medians[tile_count] / tile_count
            for tile_count in tile_counts if tile_count != 1)
        if (isinstance(non_tiny_span, bool)
                or not isinstance(non_tiny_span, (int, float))
                or not math.isclose(
                    float(non_tiny_span),
                    (max(per_tile_values) - min(per_tile_values)) / selected,
                    rel_tol=0.0, abs_tol=1.0e-12)
                or non_tiny_span > measurement["maximum_allowed_relative_span"]):
            raise ValueError(f"{owner} {rounds}-round per-tile span is invalid")
        service_time_ns_by_k_rounds[rounds] = float(published)
    if not math.isclose(
            measurement["maximum_fresh_process_relative_span"],
            max(observed_fresh_spans), rel_tol=0.0, abs_tol=1.0e-18):
        raise ValueError(f"{owner} maximum fresh-process span is invalid")
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "worker_head", "raw_root",
         "source_snapshot_path", "host", "container", "physical_device_id",
         "logical_device_id", "owner", "source_sha256", "binary_sha256",
         "raw_sha256", "cann_source_sha256"})
    if (provenance["measurement_baseline_commit"] != (
            "c1589b778e05b688bcb87e8a12cf640c30c046fa")
            or provenance["worker_head"] != (
                "aee79a74b7fb06f535cf08c28d6ad6261e7301e2")
            or (not isinstance(provenance["host"], str)
                or not provenance["host"].strip())
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in provenance["raw_root"].split("/")
            or provenance["cann_source_sha256"] != {
                "grouped_matmul.h": (
                    "d06679b4a391c032665f11fb3700c7478726c62e4230a0029d4286e086b209ed"),
                "grouped_matmul_utils.h": (
                    "377bd1fc2e9858e1809c6dca5320fe2e25150749c6644a41b958c2562c6149e4"),
            }):
        raise ValueError(f"{owner} provenance changed")
    source_hashes = provenance["source_sha256"]
    raw_hashes = provenance["raw_sha256"]
    expected_source_names = {
        "CMakeLists.txt", "README.md",
        "grouped_matmul_prelude_control_kernel.cpp",
        "grouped_matmul_prelude_control_main.cpp",
    }
    expected_raw_names = {
        "trial1_forward.csv", "trial2_reverse.csv", "trial3_interleaved.csv"}
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != expected_source_names
            or not isinstance(raw_hashes, dict)
            or set(raw_hashes) != expected_raw_names):
        raise ValueError(f"{owner} captured evidence axis changed")
    evidence_hashes = [provenance["binary_sha256"], *source_hashes.values(),
                       *raw_hashes.values(),
                       *provenance["cann_source_sha256"].values()]
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "an exact-source consumer may use only one exact listed K-round "
            "with the full base-M16/active-M16/N256/K64/depth-120-8/"
            "step-60-4/24-AIC scope, and must compose the prelude as an AIC "
            "controller dependency before the same tile's IterateAll issue; "
            "it must not use this as an OP residual or together with "
            "unclassified GetTensorAddr service"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": "aic_grouped_matmul_short_source_prelude_control_v1",
        **{
            name: (tuple(item) if isinstance(item, list) else item)
            for name, item in scope.items()
        },
        "service_time_ns_by_k_rounds": service_time_ns_by_k_rounds,
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256),
    }


def load_aic_grouped_matmul_short_source_tail_prelude_service(
        repository_root):
    """Load exact terminal-N controller spans for the short CANN program."""

    path = repository_root / (
        AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul short-source tail prelude reference SHA256 "
            "mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul short-source tail prelude reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "publication_scope",
         "source_scope", "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann-8.5.1",
        "mechanism_id": (
            "aic_grouped_matmul_short_source_tail_prelude_control_v1"),
        "duration_unit": "ns_per_nonempty_output_tile_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"dtype", "active_aic_groups", "base_m", "active_m",
         "logical_n_values", "tile_n", "tile_k", "depth_a1", "depth_b1",
         "step_ka", "step_kb", "db_l0a", "db_l0b", "db_l0c",
         "k_rounds"})
    expected_scope = {
        "dtype": "bf16", "active_aic_groups": 24, "base_m": 16,
        "active_m": 16, "logical_n_values": [64, 128], "tile_n": 256,
        "tile_k": 64, "depth_a1": 120, "depth_b1": 8, "step_ka": 60,
        "step_kb": 4, "db_l0a": 2, "db_l0b": 2, "db_l0c": 1,
        "k_rounds": [45],
    }
    if scope != expected_scope:
        raise ValueError(f"{owner} publication scope changed")
    source_scope = value["source_scope"]
    require_exact_fields(
        f"{owner}.source_scope", source_scope,
        {"upstream_file", "source_sequence", "excluded_source_work"})
    if (source_scope["upstream_file"] != "grouped_matmul/grouped_matmul.h"
            or source_scope["source_sequence"] != (
                "GMMProcess::MNBlockIdxCompute followed by "
                "GMMCompute::MMCompute terminal-N current_n and X/W/Y-offset "
                "setup plus the base-K A1/B1/L0 scheduler counters before "
                "IterateAll<false>")
            or source_scope["excluded_source_work"] != [
                "GetTensorAddr list loads", "MTE transfer", "Cube arithmetic",
                "FixPipe output", "vector work", "routing", "activation",
                "reduction", "complete production operator"]):
        raise ValueError(f"{owner} source scope changed")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"model", "repeats_per_launch", "fresh_process_trials",
         "tile_counts", "system_counter_tick_ns", "selection_policy",
         "service_time_ns_by_logical_n_and_k_rounds",
         "completion_span_ns_by_logical_n_and_k_rounds",
         "maximum_fresh_process_relative_span_by_logical_n",
         "non_tiny_per_tile_relative_span_by_logical_n_and_k_rounds",
         "maximum_allowed_relative_span", "mechanism_gate_passed"})
    if (measurement["model"] != (
            "exact_logical_n_k_round_lookup_with_minimum_non_tiny_per_tile_"
            "service_v1")
            or measurement["repeats_per_launch"] != 256
            or measurement["fresh_process_trials"] != 3
            or measurement["tile_counts"] != [1, 4, 16, 64]
            or measurement["system_counter_tick_ns"] != 20.0
            or measurement["selection_policy"] != (
                "for each exact logical-N and K-round, select the minimum "
                "median per-tile service among non-tiny tile counts 4, 16 "
                "and 64; no production/evaluation timing or APE selects a "
                "value")
            or measurement["maximum_allowed_relative_span"] != 0.05
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} measurement contract changed")
    logical_n_values = tuple(scope["logical_n_values"])
    k_rounds = tuple(scope["k_rounds"])
    tile_counts = tuple(measurement["tile_counts"])
    logical_n_keys = {str(item) for item in logical_n_values}
    round_keys = {str(item) for item in k_rounds}
    service_by_logical_n = measurement[
        "service_time_ns_by_logical_n_and_k_rounds"]
    completion_by_logical_n = measurement[
        "completion_span_ns_by_logical_n_and_k_rounds"]
    fresh_span_by_logical_n = measurement[
        "maximum_fresh_process_relative_span_by_logical_n"]
    non_tiny_span_by_logical_n = measurement[
        "non_tiny_per_tile_relative_span_by_logical_n_and_k_rounds"]
    if (set(service_by_logical_n) != logical_n_keys
            or set(completion_by_logical_n) != logical_n_keys
            or set(fresh_span_by_logical_n) != logical_n_keys
            or set(non_tiny_span_by_logical_n) != logical_n_keys):
        raise ValueError(f"{owner} logical-N axis changed")
    normalized_services = {}
    for logical_n in logical_n_values:
        logical_n_key = str(logical_n)
        services = service_by_logical_n[logical_n_key]
        completions = completion_by_logical_n[logical_n_key]
        non_tiny_spans = non_tiny_span_by_logical_n[logical_n_key]
        if (not isinstance(services, dict) or not isinstance(completions, dict)
                or not isinstance(non_tiny_spans, dict)
                or set(services) != round_keys
                or set(completions) != round_keys
                or set(non_tiny_spans) != round_keys):
            raise ValueError(f"{owner} {logical_n}-N K-round axis changed")
        normalized_services[logical_n] = {}
        observed_fresh_spans = []
        for k_round_count in k_rounds:
            k_rounds_key = str(k_round_count)
            completion_by_tile_count = completions[k_rounds_key]
            if (not isinstance(completion_by_tile_count, dict)
                    or set(completion_by_tile_count)
                    != {str(count) for count in tile_counts}):
                raise ValueError(
                    f"{owner} {logical_n}-N {k_round_count}-round tile axis changed")
            medians = {}
            for tile_count in tile_counts:
                runs = completion_by_tile_count[str(tile_count)]
                if (not isinstance(runs, list) or len(runs) != 3
                        or any(isinstance(run, bool)
                               or not isinstance(run, (int, float))
                               or not math.isfinite(run) or run <= 0.0
                               for run in runs)):
                    raise ValueError(
                        f"{owner} {logical_n}-N {k_round_count}-round "
                        f"{tile_count}-tile capture is invalid")
                median = float(sorted(runs)[1])
                fresh_span = (max(runs) - min(runs)) / median
                if fresh_span > measurement["maximum_allowed_relative_span"]:
                    raise ValueError(
                        f"{owner} {logical_n}-N {k_round_count}-round "
                        f"{tile_count}-tile capture is unstable")
                medians[tile_count] = median
                observed_fresh_spans.append(fresh_span)
            selected = min(
                medians[tile_count] / tile_count
                for tile_count in tile_counts if tile_count != 1)
            published = services[k_rounds_key]
            if (isinstance(published, bool)
                    or not isinstance(published, (int, float))
                    or not math.isfinite(published) or published <= 0.0
                    or not math.isclose(
                        float(published), selected,
                        rel_tol=0.0, abs_tol=1.0e-12)):
                raise ValueError(
                    f"{owner} {logical_n}-N {k_round_count}-round selected "
                    "service is invalid")
            per_tile_values = tuple(
                medians[tile_count] / tile_count
                for tile_count in tile_counts if tile_count != 1)
            published_non_tiny_span = non_tiny_spans[k_rounds_key]
            expected_non_tiny_span = (
                (max(per_tile_values) - min(per_tile_values)) / selected)
            if (isinstance(published_non_tiny_span, bool)
                    or not isinstance(published_non_tiny_span, (int, float))
                    or not math.isclose(
                        float(published_non_tiny_span),
                        expected_non_tiny_span, rel_tol=0.0, abs_tol=1.0e-12)
                    or published_non_tiny_span
                    > measurement["maximum_allowed_relative_span"]):
                raise ValueError(
                    f"{owner} {logical_n}-N {k_round_count}-round per-tile "
                    "span is invalid")
            normalized_services[logical_n][k_round_count] = float(published)
        published_fresh_span = fresh_span_by_logical_n[logical_n_key]
        if (isinstance(published_fresh_span, bool)
                or not isinstance(published_fresh_span, (int, float))
                or not math.isclose(
                    float(published_fresh_span), max(observed_fresh_spans),
                    rel_tol=0.0, abs_tol=1.0e-18)
                or published_fresh_span
                > measurement["maximum_allowed_relative_span"]):
            raise ValueError(
                f"{owner} {logical_n}-N fresh-process span is invalid")
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "worker_head",
         "raw_root_by_logical_n", "source_snapshot_path_by_logical_n",
         "host", "container", "physical_device_id", "logical_device_id",
         "owner", "source_sha256", "binary_sha256_by_logical_n",
         "run_plan_sha256_by_logical_n", "raw_sha256_by_logical_n",
         "cann_source_sha256"})
    if (provenance["measurement_baseline_commit"] != (
            "c1589b778e05b688bcb87e8a12cf640c30c046fa")
            or provenance["worker_head"] != (
                "60e321b987346f6ed72eb0dd3f6fd5675ab7285e")
            or (not isinstance(provenance["host"], str)
                or not provenance["host"].strip())
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"
            or provenance["cann_source_sha256"] != {
                "grouped_matmul.h": (
                    "d06679b4a391c032665f11fb3700c7478726c62e4230a0029d4286e086b209ed"),
                "grouped_matmul_utils.h": (
                    "377bd1fc2e9858e1809c6dca5320fe2e25150749c6644a41b958c2562c6149e4"),
            }):
        raise ValueError(f"{owner} provenance changed")
    raw_roots = provenance["raw_root_by_logical_n"]
    snapshots = provenance["source_snapshot_path_by_logical_n"]
    source_hashes = provenance["source_sha256"]
    binary_hashes = provenance["binary_sha256_by_logical_n"]
    plan_hashes = provenance["run_plan_sha256_by_logical_n"]
    raw_hashes = provenance["raw_sha256_by_logical_n"]
    expected_source_names = {
        "CMakeLists.txt", "README.md",
        "grouped_matmul_prelude_control_kernel.cpp",
        "grouped_matmul_prelude_control_main.cpp",
    }
    expected_raw_names = {
        "trial1_forward.csv", "trial2_reverse.csv", "trial3_interleaved.csv"}
    if (not isinstance(raw_roots, dict) or not isinstance(snapshots, dict)
            or not isinstance(source_hashes, dict)
            or not isinstance(binary_hashes, dict)
            or not isinstance(plan_hashes, dict)
            or not isinstance(raw_hashes, dict)
            or set(raw_roots) != logical_n_keys
            or set(snapshots) != logical_n_keys
            or set(source_hashes) != expected_source_names
            or set(binary_hashes) != logical_n_keys
            or set(plan_hashes) != logical_n_keys
            or set(raw_hashes) != logical_n_keys):
        raise ValueError(f"{owner} captured evidence axis changed")
    evidence_hashes = [*source_hashes.values(),
                       *provenance["cann_source_sha256"].values()]
    for logical_n in logical_n_values:
        logical_n_key = str(logical_n)
        raw_root = raw_roots[logical_n_key]
        snapshot = snapshots[logical_n_key]
        if (not isinstance(raw_root, str)
                or not raw_root.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in raw_root.split("/")
                or (not isinstance(snapshot, str)
                or not snapshot.strip()
                or ".." in snapshot.split("/"))):
            raise ValueError(f"{owner} {logical_n}-N evidence path is invalid")
        raw_by_trial = raw_hashes[logical_n_key]
        if (not isinstance(raw_by_trial, dict)
                or set(raw_by_trial) != expected_raw_names):
            raise ValueError(
                f"{owner} {logical_n}-N raw evidence axis changed")
        evidence_hashes.extend((binary_hashes[logical_n_key],
                                plan_hashes[logical_n_key],
                                *raw_by_trial.values()))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "an exact-source consumer may use only one exact listed logical-N "
            "and K-round with the full base-M16/active-M16/N256/K64/"
            "depth-120-8/step-60-4/24-AIC scope, and must compose the tail "
            "prelude as an AIC controller dependency before the same terminal "
            "tile's IterateAll issue; it must not use this as an OP residual "
            "or together with unclassified GetTensorAddr service"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": (
            "aic_grouped_matmul_short_source_tail_prelude_control_v1"),
        **{name: (tuple(item) if isinstance(item, list) else item)
           for name, item in scope.items()},
        "service_time_ns_by_logical_n_and_k_rounds": normalized_services,
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_SHORT_SOURCE_TAIL_PRELUDE_REFERENCE_SHA256),
    }


def load_aic_grouped_matmul_m32_short_source_prelude_service(
        repository_root):
    """Load exact M32 short-source controller spans for CANN GroupedMatmul."""

    path = repository_root / (
        AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul M32 short-source prelude reference SHA256 "
            "mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul M32 short-source prelude reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "publication_scope",
         "source_scope", "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann-8.5.1",
        "mechanism_id": (
            "aic_grouped_matmul_m32_short_source_prelude_control_v1"),
        "duration_unit": "ns_per_nonempty_output_tile_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    expected_scope = {
        "dtype": "bf16", "active_aic_groups": 24, "base_m": 32,
        "active_m": 32, "logical_n_values": [64, 128, 256],
        "tile_n": 256, "tile_k": 64, "depth_a1": 56, "depth_b1": 8,
        "step_ka": 28, "step_kb": 4, "db_l0a": 2, "db_l0b": 2,
        "db_l0c": 1, "k_rounds": [45],
    }
    require_exact_fields(
        f"{owner}.publication_scope", scope, set(expected_scope))
    if scope != expected_scope:
        raise ValueError(f"{owner} publication scope changed")
    source_scope = value["source_scope"]
    require_exact_fields(
        f"{owner}.source_scope", source_scope,
        {"upstream_file", "source_sequence", "excluded_source_work"})
    if (source_scope["upstream_file"] != "grouped_matmul/grouped_matmul.h"
            or source_scope["source_sequence"] != (
                "GMMProcess::MNBlockIdxCompute followed by "
                "GMMCompute::MMCompute M32 current-M/current-N and X/W/Y-offset "
                "setup plus the base-K A1/B1/L0 scheduler counters before "
                "IterateAll<false>")
            or source_scope["excluded_source_work"] != [
                "GetTensorAddr list loads", "MTE transfer", "Cube arithmetic",
                "FixPipe output", "vector work", "routing", "activation",
                "reduction", "complete production operator"]):
        raise ValueError(f"{owner} source scope changed")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"model", "repeats_per_launch", "fresh_process_trials",
         "tile_counts", "system_counter_tick_ns", "selection_policy",
         "service_time_ns_by_logical_n_and_k_rounds",
         "completion_span_ns_by_logical_n_and_k_rounds",
         "maximum_fresh_process_relative_span_by_logical_n",
         "non_tiny_per_tile_relative_span_by_logical_n_and_k_rounds",
         "maximum_allowed_relative_span", "mechanism_gate_passed"})
    if (measurement["model"] != (
            "exact_logical_n_k_round_lookup_with_minimum_non_tiny_per_tile_"
            "service_v1")
            or measurement["repeats_per_launch"] != 256
            or measurement["fresh_process_trials"] != 3
            or measurement["tile_counts"] != [1, 4, 16, 64]
            or measurement["system_counter_tick_ns"] != 20.0
            or measurement["selection_policy"] != (
                "for each exact logical-N and K-round, select the minimum "
                "median per-tile service among non-tiny tile counts 4, 16 "
                "and 64; no production/evaluation timing or APE selects a "
                "value")
            or measurement["maximum_allowed_relative_span"] != 0.05
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} measurement contract changed")
    logical_n_values = (64, 128, 256)
    tile_counts = (1, 4, 16, 64)
    service_by_logical_n = measurement[
        "service_time_ns_by_logical_n_and_k_rounds"]
    completion_by_logical_n = measurement[
        "completion_span_ns_by_logical_n_and_k_rounds"]
    fresh_spans = measurement[
        "maximum_fresh_process_relative_span_by_logical_n"]
    non_tiny_spans = measurement[
        "non_tiny_per_tile_relative_span_by_logical_n_and_k_rounds"]
    logical_n_keys = {str(item) for item in logical_n_values}
    if (set(service_by_logical_n) != logical_n_keys
            or set(completion_by_logical_n) != logical_n_keys
            or set(fresh_spans) != logical_n_keys
            or set(non_tiny_spans) != logical_n_keys):
        raise ValueError(f"{owner} logical-N axis changed")
    normalized_services = {}
    for logical_n in logical_n_values:
        logical_n_key = str(logical_n)
        services = service_by_logical_n[logical_n_key]
        completions = completion_by_logical_n[logical_n_key]
        spans = non_tiny_spans[logical_n_key]
        if (set(services) != {"45"} or set(completions) != {"45"}
                or set(spans) != {"45"}):
            raise ValueError(f"{owner} {logical_n}-N K-round axis changed")
        captures = completions["45"]
        if set(captures) != {str(count) for count in tile_counts}:
            raise ValueError(f"{owner} {logical_n}-N tile axis changed")
        medians = {}
        observed_fresh_spans = []
        for tile_count in tile_counts:
            runs = captures[str(tile_count)]
            if (not isinstance(runs, list) or len(runs) != 3
                    or any(isinstance(run, bool)
                           or not isinstance(run, (int, float))
                           or not math.isfinite(run) or run <= 0.0
                           for run in runs)):
                raise ValueError(
                    f"{owner} {logical_n}-N {tile_count}-tile capture invalid")
            median = float(sorted(runs)[1])
            fresh_span = (max(runs) - min(runs)) / median
            if fresh_span > measurement["maximum_allowed_relative_span"]:
                raise ValueError(f"{owner} {logical_n}-N capture is unstable")
            medians[tile_count] = median
            observed_fresh_spans.append(fresh_span)
        selected = min(medians[count] / count for count in tile_counts
                       if count != 1)
        published = services["45"]
        if (isinstance(published, bool)
                or not isinstance(published, (int, float))
                or not math.isfinite(published) or published <= 0.0
                or not math.isclose(float(published), selected,
                                    rel_tol=0.0, abs_tol=1.0e-12)):
            raise ValueError(f"{owner} {logical_n}-N service is invalid")
        per_tile = tuple(medians[count] / count for count in tile_counts
                         if count != 1)
        expected_span = (max(per_tile) - min(per_tile)) / selected
        if (not math.isclose(float(spans["45"]), expected_span,
                            rel_tol=0.0, abs_tol=1.0e-12)
                or spans["45"] > measurement["maximum_allowed_relative_span"]
                or not math.isclose(float(fresh_spans[logical_n_key]),
                                    max(observed_fresh_spans),
                                    rel_tol=0.0, abs_tol=1.0e-18)):
            raise ValueError(f"{owner} {logical_n}-N stability is invalid")
        normalized_services[logical_n] = {45: float(published)}
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "worker_head", "raw_root",
         "source_snapshot_path", "host", "container", "physical_device_id",
         "logical_device_id", "owner", "source_sha256",
         "binary_sha256_by_logical_n", "run_plan_sha256",
         "raw_sha256_by_logical_n", "cann_source_sha256"})
    if (provenance["measurement_baseline_commit"] != (
            "c1589b778e05b688bcb87e8a12cf640c30c046fa")
            or provenance["worker_head"] != (
                "8952b6c7faed03f1a80ca0ff9b9721fdcdb808cc")
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or (not isinstance(provenance["source_snapshot_path"], str)
                or not provenance["source_snapshot_path"].strip()
                or ".." in provenance["source_snapshot_path"].split("/"))
            or (not isinstance(provenance["host"], str)
                or not provenance["host"].strip())
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"):
        raise ValueError(f"{owner} provenance changed")
    expected_source_names = {
        "CMakeLists.txt", "README.md",
        "grouped_matmul_prelude_control_kernel.cpp",
        "grouped_matmul_prelude_control_main.cpp"}
    expected_cann_hashes = {
        "grouped_matmul.h": (
            "d06679b4a391c032665f11fb3700c7478726c62e4230a0029d4286e086b209ed"),
        "grouped_matmul_utils.h": (
            "377bd1fc2e9858e1809c6dca5320fe2e25150749c6644a41b958c2562c6149e4")}
    binary_hashes = provenance["binary_sha256_by_logical_n"]
    raw_hashes = provenance["raw_sha256_by_logical_n"]
    if (not isinstance(provenance["source_sha256"], dict)
            or set(provenance["source_sha256"]) != expected_source_names
            or not isinstance(binary_hashes, dict)
            or set(binary_hashes) != logical_n_keys
            or not isinstance(raw_hashes, dict)
            or set(raw_hashes) != logical_n_keys
            or provenance["cann_source_sha256"] != expected_cann_hashes):
        raise ValueError(f"{owner} evidence axis changed")
    evidence_hashes = [*provenance["source_sha256"].values(),
                       provenance["run_plan_sha256"], *binary_hashes.values(),
                       *expected_cann_hashes.values()]
    for logical_n in logical_n_values:
        names = {
            f"m32_n{logical_n}_trial1.csv",
            f"m32_n{logical_n}_trial2.csv",
            f"m32_n{logical_n}_trial3.csv"}
        raw_by_trial = raw_hashes[str(logical_n)]
        if not isinstance(raw_by_trial, dict) or set(raw_by_trial) != names:
            raise ValueError(f"{owner} {logical_n}-N raw evidence changed")
        evidence_hashes.extend(raw_by_trial.values())
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "an exact-source consumer may use only one exact listed logical-N "
            "and K-round with the full base-M32/active-M32/N256/K64/"
            "depth-56-8/step-28-4/24-AIC scope, and must compose the prelude "
            "as an AIC controller dependency before the same tile's "
            "IterateAll issue; it must not use this as an OP residual or "
            "together with unclassified GetTensorAddr service"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": "aic_grouped_matmul_m32_short_source_prelude_control_v1",
        **{name: (tuple(item) if isinstance(item, list) else item)
           for name, item in scope.items()},
        "service_time_ns_by_logical_n_and_k_rounds": normalized_services,
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_M32_SHORT_SOURCE_PRELUDE_REFERENCE_SHA256),
    }


def load_aic_grouped_matmul_entry_metadata_pair_service(repository_root):
    """Load the exact launch-wide GroupedMatmul entry metadata pair."""

    path = repository_root / AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE_SHA256):
        raise ValueError(
            "AIC GroupedMatmul entry metadata-pair reference SHA256 mismatch")
    value = json.loads(payload)
    owner = "AIC GroupedMatmul entry metadata-pair reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision",
         "mechanism_id", "duration_unit", "publication_scope",
         "source_scope", "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann-8.5.1",
        "mechanism_id": "aic_grouped_matmul_entry_metadata_pair_v1",
        "duration_unit": (
            "ns_per_aic_for_one_launch_wide_typed_metadata_pair"),
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    expected_scope = {
        "dtype": "bf16",
        "source_kind": "grouped_matmul",
        "entry_path": "GMM_FLOAT/GMM_CUBE_IMP",
        "active_aic_groups": 24,
        "gmm_base_params_bytes": 1672,
        "tcube_tiling_bytes": 200,
        "unsupported_projection_dimensions": [7168],
    }
    require_exact_fields(
        f"{owner}.publication_scope", scope, set(expected_scope))
    if scope != expected_scope:
        raise ValueError(f"{owner} publication scope changed")
    source_scope = value["source_scope"]
    require_exact_fields(
        f"{owner}.source_scope", source_scope,
        {"upstream_file", "source_sequence", "included_source_work",
         "excluded_source_work"})
    if (source_scope["upstream_file"] != (
            "ops_transformer/ascendc/grouped_matmul/grouped_matmul.cpp")
            or source_scope["source_sequence"] != [
                "GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, "
                "gmmBaseParams_, tiling)",
                "GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, "
                "mmTilingData_, tiling)",
                "GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, "
                "gmmArrayAddr_, tiling)",
            ]
            or source_scope["included_source_work"] != [
                "typed gmmBaseParams materialization",
                "typed mmTilingData materialization",
            ]
            or source_scope["excluded_source_work"] != [
                "gmmArray address formation without a payload copy",
                "GroupedMatmul tile processing",
                "routing",
                "matrix arithmetic",
                "output work",
                "complete production operator",
            ]):
        raise ValueError(f"{owner} source scope changed")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"model", "iterations_per_launch", "fresh_process_trials",
         "block_dims", "system_counter_tick_ns",
         "service_time_ns_by_run_at_24_aic", "selection_policy",
         "selected_service_time_ns", "maximum_fresh_process_relative_span",
         "maximum_all_24_aic_sample_relative_span",
         "maximum_allowed_relative_span", "mechanism_gate_passed"})
    if (measurement["model"] != (
            "exact_two_typed_member_aic_materialization_v1")
            or measurement["iterations_per_launch"] != 4096
            or measurement["fresh_process_trials"] != 3
            or measurement["block_dims"] != [1, 24, 48]
            or measurement["system_counter_tick_ns"] != 20.0
            or measurement["selection_policy"] != (
                "median of the three independent 24-AIC fresh-process "
                "medians; no production or evaluation duration and no APE "
                "selects the value")
            or measurement["maximum_allowed_relative_span"] != 0.05
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} measurement contract changed")
    by_run = measurement["service_time_ns_by_run_at_24_aic"]
    if not isinstance(by_run, dict) or set(by_run) != {"r2", "r3", "r4"}:
        raise ValueError(f"{owner} fresh-process axis changed")
    values = tuple(by_run[run] for run in ("r2", "r3", "r4"))
    if any(isinstance(item, bool)
           or not isinstance(item, (int, float))
           or not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError(f"{owner} fresh-process sample is invalid")
    selected = float(statistics.median(values))
    fresh_span = (max(values) - min(values)) / selected
    all_sample_span = measurement["maximum_all_24_aic_sample_relative_span"]
    if (not math.isclose(
                float(measurement["selected_service_time_ns"]), selected,
                rel_tol=0.0, abs_tol=1.0e-12)
            or not math.isclose(
                float(measurement["maximum_fresh_process_relative_span"]),
                fresh_span, rel_tol=0.0, abs_tol=1.0e-18)
            or isinstance(all_sample_span, bool)
            or not isinstance(all_sample_span, (int, float))
            or not math.isfinite(all_sample_span)
            or all_sample_span < fresh_span
            or all_sample_span > measurement["maximum_allowed_relative_span"]):
        raise ValueError(f"{owner} stability publication is invalid")
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {"measurement_baseline_commit", "worker_head", "raw_root",
         "source_snapshot_path", "host", "container",
         "physical_device_id", "logical_device_id", "owner",
         "source_sha256", "binary_sha256", "raw_sha256",
         "cann_source_sha256"})
    if (provenance["measurement_baseline_commit"] != (
            "871f7b1622f5e1fb0f6937f89748e9dc44721a02")
            or provenance["worker_head"] != "bcad610c"
            or not provenance["raw_root"].startswith(
                "profiler_output/microbench/ascend_910b/")
            or (not isinstance(provenance["source_snapshot_path"], str)
                or not provenance["source_snapshot_path"].strip()
                or ".." in provenance["source_snapshot_path"].split("/"))
            or (not isinstance(provenance["host"], str)
                or not provenance["host"].strip())
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())
            or (isinstance(provenance["physical_device_id"], bool)
                or not isinstance(provenance["physical_device_id"], int)
                or provenance["physical_device_id"] < 0)
            or (isinstance(provenance["logical_device_id"], bool)
                or not isinstance(provenance["logical_device_id"], int)
                or provenance["logical_device_id"] < 0)
            or provenance["owner"] != "kernel-accuracy-fusedmoe"):
        raise ValueError(f"{owner} provenance changed")
    source_hashes = provenance["source_sha256"]
    raw_hashes = provenance["raw_sha256"]
    cann_hashes = provenance["cann_source_sha256"]
    expected_source_names = {
        "CMakeLists.txt", "tiling_data_materialization_kernel.cpp",
        "tiling_data_materialization_main.cpp",
        "tiling_data_materialization_runner.py",
    }
    expected_raw_names = {"raw_r2.csv", "raw_r3.csv", "raw_r4.csv"}
    expected_cann_hashes = {
        "grouped_matmul.h": (
            "d06679b4a391c032665f11fb3700c7478726c62e4230a0029d4286e086b209ed"),
        "grouped_matmul.cpp": (
            "8bafe5cb9007f1a6b6fa52521504e2251dde2bfd02c81f8a1aa24d01fb1c408c"),
    }
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != expected_source_names
            or not isinstance(raw_hashes, dict)
            or set(raw_hashes) != expected_raw_names
            or cann_hashes != expected_cann_hashes):
        raise ValueError(f"{owner} evidence axis changed")
    evidence_hashes = [*source_hashes.values(), provenance["binary_sha256"],
                       *raw_hashes.values(), *cann_hashes.values()]
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "only the CANN 8.5.1 BF16 GroupedMatmul GMM_FLOAT/GMM_CUBE_IMP "
            "entry with 24 AICs and the exact 1672-byte gmmBaseParams then "
            "200-byte TCubeTiling typed member sequence may consume this "
            "once per launch per AIC before GMMProcess; dimensions with a "
            "nonmatching captured descriptor delta, including H7168, are "
            "fail-closed, and gmmArray address formation must not be charged"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": "aic_grouped_matmul_entry_metadata_pair_v1",
        **{name: (tuple(item) if isinstance(item, list) else item)
           for name, item in scope.items()},
        "service_time_ns": selected,
        "reference_path": str(path),
        "reference_sha256": (
            AIC_GROUPED_MATMUL_ENTRY_METADATA_PAIR_REFERENCE_SHA256),
    }


def load_aic_fixpipe_bf16_terminal_tile_service(repository_root):
    """Load one stable BF16 terminal FixPipe tile completion."""

    path = repository_root / AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE_SHA256):
        raise ValueError("AIC BF16 terminal FixPipe reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC BF16 terminal FixPipe reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_fixpipe_bf16_terminal_tile_completion",
        "duration_unit": "ns_per_terminal_tile_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"dtype", "rows", "columns", "active_aic_groups",
         "terminal_tiles_per_program", "completion_ns", "samples_ns",
         "estimator"})
    frozen = {
        "dtype": "bf16",
        "rows": 128,
        "columns": 256,
        "active_aic_groups": 1,
        "terminal_tiles_per_program": 1,
    }
    for name, expected_value in frozen.items():
        if scope[name] != expected_value:
            raise ValueError(f"{owner}.publication_scope.{name} changed")
    if scope["estimator"] != (
            "median of six independently launched one-AIC long-minus-short "
            "completion deltas"):
        raise ValueError(f"{owner} estimator changed")
    samples = scope["samples_ns"]
    if not isinstance(samples, list) or len(samples) != 6:
        raise ValueError(f"{owner} requires six one-AIC samples")
    normalized_samples = tuple(float(item) for item in samples)
    completion_ns = float(scope["completion_ns"])
    if (any(not math.isfinite(item) or item <= 0.0
            for item in normalized_samples)
            or not math.isclose(
                completion_ns, float(sorted(normalized_samples)[2]
                                     + sorted(normalized_samples)[3]) / 2.0,
                rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError(f"{owner} completion must be the sample median")
    measurement = value["measurement"]
    if not isinstance(measurement, dict):
        raise TypeError(f"{owner}.measurement must be a mapping")
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"mode", "capture_commit", "aic_clock_hz", "raw_paths",
         "raw_sha256", "provenance_sha256", "binary_sha256",
         "mechanism_gate_passed"})
    expected_measurement = {
        "mode": "independent_l0c_to_gm_fixpipe_completion_microbenchmark",
        "capture_commit": "efcf67d7dd02326e56fd036ad4f8898cb71d4d70",
        "aic_clock_hz": 3200000000,
        "mechanism_gate_passed": True,
    }
    if any(measurement[name] != expected_value
           for name, expected_value in expected_measurement.items()):
        raise ValueError(f"{owner} mechanism gate failed")
    raw_paths = measurement["raw_paths"]
    if (not isinstance(raw_paths, list) or len(raw_paths) != 2
            or not isinstance(measurement["raw_sha256"], list)
            or len(measurement["raw_sha256"]) != 2
            or not isinstance(measurement["provenance_sha256"], list)
            or len(measurement["provenance_sha256"]) != 2):
        raise ValueError(f"{owner} requires two raw captures")
    evidence_hashes = (
        tuple(measurement["raw_sha256"])
        + tuple(measurement["provenance_sha256"])
        + (measurement["binary_sha256"],))
    if any(not isinstance(raw_path, str)
           or not raw_path.startswith(
               "profiler_output/microbench/ascend_910b/")
           or ".." in raw_path.split("/") for raw_path in raw_paths):
        raise ValueError(f"{owner} raw path is outside its evidence owner")
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    if value["forward_eligibility_constraint"] != (
            "one non-overlapped terminal BF16 128x256 FixPipe completion "
            "after a source-matched MatMulV3 ASW ingress program; multi-AIC "
            "measurements are excluded"):
        raise ValueError(f"{owner} forward eligibility changed")
    return {
        "model": "aic_fixpipe_bf16_terminal_tile_completion_v1",
        **frozen,
        "completion_ns": completion_ns,
        "reference_path": str(AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE),
        "reference_sha256": AIC_FIXPIPE_BF16_TERMINAL_TILE_REFERENCE_SHA256,
    }


def load_aic_fixpipe_bf16_finite_stream_service(repository_root):
    """Load source-shaped BF16 FixPipe finite-stream completion envelopes."""

    path = repository_root / AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE_SHA256):
        raise ValueError("AIC BF16 finite FixPipe reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC BF16 finite FixPipe reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_fixpipe_bf16_finite_stream_completion",
        "duration_unit": "ns_per_finite_tile_stream_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    if value["forward_eligibility_constraint"] != (
            "only source-matched BF16 row-major MatMulV3 output-tile "
            "streams whose exact rows, columns, non-empty AIC count, and "
            "per-worker tile count are inside the published finite affine "
            "domain"):
        raise ValueError(f"{owner} forward eligibility changed")

    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"dtype", "layout", "minimum_tiles_per_worker",
         "maximum_tiles_per_worker", "short_tiles_per_worker",
         "long_tiles_per_worker", "estimator", "cells"})
    if scope["dtype"] != "bf16" or scope["layout"] != "row_major":
        raise ValueError(f"{owner} dtype/layout changed")
    minimum_tiles = scope["minimum_tiles_per_worker"]
    maximum_tiles = scope["maximum_tiles_per_worker"]
    short_tiles = scope["short_tiles_per_worker"]
    long_tiles = scope["long_tiles_per_worker"]
    if (minimum_tiles != 1 or maximum_tiles != 96
            or short_tiles != 32 or long_tiles != 96):
        raise ValueError(f"{owner} finite tile-count domain changed")
    if scope["estimator"] != (
            "for each run and AIC, linearly interpolate the drained "
            "32-to-96-tile endpoints; evaluate at the source tile count, "
            "take the slowest AIC, then take the median across three runs"):
        raise ValueError(f"{owner} estimator changed")
    expected_cells = {
        (16, 176, 22),
        (32, 256, 16), (32, 256, 20), (32, 256, 22),
        (48, 256, 22),
        (64, 256, 20), (64, 256, 22),
        (128, 256, 24),
    }
    cells = {}
    for index, cell in enumerate(scope["cells"]):
        cell_owner = f"{owner}.publication_scope.cells[{index}]"
        require_exact_fields(
            cell_owner, cell,
            {"rows", "columns", "active_aic_groups",
             "run_affine_envelopes"})
        signature = tuple(
            cell[name] for name in (
                "rows", "columns", "active_aic_groups"))
        if (any(isinstance(item, bool) or not isinstance(item, int)
                or item <= 0 for item in signature)
                or signature not in expected_cells
                or signature in cells):
            raise ValueError(f"{cell_owner} signature is invalid")
        runs = cell["run_affine_envelopes"]
        if not isinstance(runs, list) or len(runs) != 3:
            raise ValueError(f"{cell_owner} requires three run envelopes")
        normalized_runs = []
        for run_index, envelope in enumerate(runs):
            if not isinstance(envelope, list) or not envelope:
                raise ValueError(
                    f"{cell_owner} run {run_index} envelope is empty")
            normalized_lines = []
            block_indices = set()
            for line in envelope:
                if (not isinstance(line, list) or len(line) != 3
                        or any(isinstance(item, bool)
                               or not isinstance(item, int) for item in line)):
                    raise TypeError(
                        f"{cell_owner} affine entries must be integer triples")
                block_index, short_cycles, long_cycles = line
                if (block_index < 0
                        or block_index >= signature[2]
                        or block_index in block_indices
                        or short_cycles <= 0
                        or long_cycles <= short_cycles):
                    raise ValueError(
                        f"{cell_owner} affine entry is invalid: {line!r}")
                block_indices.add(block_index)
                normalized_lines.append(tuple(line))
            normalized_runs.append(tuple(normalized_lines))
        cells[signature] = tuple(normalized_runs)
    if set(cells) != expected_cells:
        raise ValueError(f"{owner} finite geometry cells changed")

    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"mode", "capture_commit", "aic_clock_hz", "physical_device_id",
         "run_count", "raw_path", "raw_sha256", "stderr_sha256",
         "source_snapshot_sha256", "binary_sha256", "validation_rows",
         "validation", "mechanism_gate_passed"})
    expected_measurement = {
        "mode": "independent_l0c_to_gm_fixpipe_finite_stream_microbenchmark",
        "capture_commit": "d707fd012d755a48974d599dbd0af3d97972ce0b",
        "aic_clock_hz": 3_200_000_000,
        "run_count": 3,
        "validation_rows": 24,
        "validation": "exact",
        "mechanism_gate_passed": True,
    }
    if (isinstance(measurement["physical_device_id"], bool)
            or not isinstance(measurement["physical_device_id"], int)
            or measurement["physical_device_id"] < 0
            or any(measurement[name] != expected_value
                   for name, expected_value in expected_measurement.items())):
        raise ValueError(f"{owner} measurement contract changed")
    raw_path = measurement["raw_path"]
    if (not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")):
        raise ValueError(f"{owner} raw path is outside its evidence owner")
    evidence_hashes = tuple(
        measurement[name] for name in (
            "raw_sha256", "stderr_sha256", "source_snapshot_sha256",
            "binary_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef"
                  for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    return {
        "model": "aic_fixpipe_bf16_finite_stream_completion_v1",
        "minimum_tiles_per_worker": minimum_tiles,
        "maximum_tiles_per_worker": maximum_tiles,
        "short_tiles_per_worker": short_tiles,
        "long_tiles_per_worker": long_tiles,
        "aic_clock_hz": measurement["aic_clock_hz"],
        "cells": cells,
        "reference_path": str(AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE),
        "reference_sha256": AIC_FIXPIPE_BF16_FINITE_STREAM_REFERENCE_SHA256,
    }


def load_aic_fixpipe_bf16_long_stream_service(repository_root):
    """Load the source-matched BF16 128x256 long-stream envelope."""

    path = repository_root / AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE_SHA256):
        raise ValueError("AIC BF16 long FixPipe reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC BF16 long FixPipe reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "measurement",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_fixpipe_bf16_long_stream_completion",
        "duration_unit": "ns_per_finite_tile_stream_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    if value["forward_eligibility_constraint"] != (
            "only source-matched BF16 row-major MatMulV3 key-65536 "
            "output-tile streams with exactly 128 rows, 256 columns, 24 "
            "non-empty AICs, and per-worker tile counts inside the published "
            "96-to-1195-tile affine domain"):
        raise ValueError(f"{owner} forward eligibility changed")

    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"dtype", "layout", "rows", "columns", "active_aic_groups",
         "minimum_tiles_per_worker", "maximum_tiles_per_worker",
         "short_tiles_per_worker", "long_tiles_per_worker", "estimator",
         "run_affine_envelopes"})
    frozen = {
        "dtype": "bf16",
        "layout": "row_major",
        "rows": 128,
        "columns": 256,
        "active_aic_groups": 24,
        "minimum_tiles_per_worker": 96,
        "maximum_tiles_per_worker": 1195,
        "short_tiles_per_worker": 96,
        "long_tiles_per_worker": 1195,
        "estimator": (
            "for each run and AIC, linearly interpolate the drained "
            "96-to-1195-tile endpoints; evaluate at the source tile count, "
            "take the slowest AIC, then take the median across three runs"),
    }
    for name, expected_value in frozen.items():
        if scope[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    runs = scope["run_affine_envelopes"]
    if not isinstance(runs, list) or len(runs) != 3:
        raise ValueError(f"{owner} requires three run envelopes")
    normalized_runs = []
    for run_index, envelope in enumerate(runs):
        if not isinstance(envelope, list) or len(envelope) != 24:
            raise ValueError(
                f"{owner} run {run_index} requires all 24 AICs")
        normalized_lines = []
        block_indices = set()
        for line in envelope:
            if (not isinstance(line, list) or len(line) != 3
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           for item in line)):
                raise TypeError(
                    f"{owner} affine entries must be integer triples")
            block_index, short_cycles, long_cycles = line
            if (block_index < 0 or block_index >= 24
                    or block_index in block_indices
                    or short_cycles <= 0 or long_cycles <= short_cycles):
                raise ValueError(
                    f"{owner} affine entry is invalid: {line!r}")
            block_indices.add(block_index)
            normalized_lines.append(tuple(line))
        normalized_runs.append(tuple(normalized_lines))

    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"mode", "capture_commit", "aic_clock_hz", "physical_device_id",
         "run_count", "raw_path", "raw_sha256", "stderr_sha256",
         "source_snapshot_sha256", "binary_sha256", "validation_rows",
         "validation", "mechanism_gate_passed"})
    expected_measurement = {
        "mode": "independent_l0c_to_gm_fixpipe_long_stream_microbenchmark",
        "capture_commit": "17e0f61f45a2d0f7e15b60cd794a35a9d8ad8ab3",
        "aic_clock_hz": 3_200_000_000,
        "run_count": 3,
        "validation_rows": 3,
        "validation": "exact",
        "mechanism_gate_passed": True,
    }
    if (isinstance(measurement["physical_device_id"], bool)
            or not isinstance(measurement["physical_device_id"], int)
            or measurement["physical_device_id"] < 0
            or any(measurement[name] != expected_value
                   for name, expected_value in expected_measurement.items())):
        raise ValueError(f"{owner} measurement contract changed")
    raw_path = measurement["raw_path"]
    if (not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")):
        raise ValueError(f"{owner} raw path is outside its evidence owner")
    evidence_hashes = tuple(
        measurement[name] for name in (
            "raw_sha256", "stderr_sha256", "source_snapshot_sha256",
            "binary_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    return {
        "model": "aic_fixpipe_bf16_long_stream_completion_v1",
        **frozen,
        "aic_clock_hz": measurement["aic_clock_hz"],
        "run_affine_envelopes": tuple(normalized_runs),
        "reference_path": str(AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE),
        "reference_sha256": AIC_FIXPIPE_BF16_LONG_STREAM_REFERENCE_SHA256,
    }


def load_aic_matmul_v3_asw_mte2_pair_service(repository_root):
    """Load the source-shaped 256x128 ASW operand-pair stream."""

    path = repository_root / AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE_SHA256):
        raise ValueError("MatMulV3 ASW MTE2 pair reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "MatMulV3 ASW MTE2 pair reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "service",
         "scope", "l2_resident_256x128_pair_service", "measurement",
         "mechanism", "production_or_evaluation_duration_consumed",
         "measurement_gate_passed", "mechanism_publishable"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "CANN 8.5.1",
        "service": "matmul_v3_asw_mte2_operand_pair",
        "scope": "24_aic_full_chip_l2_resident",
        "production_or_evaluation_duration_consumed": False,
        "measurement_gate_passed": True,
        "mechanism_publishable": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    service = value["l2_resident_256x128_pair_service"]
    require_exact_fields(
        f"{owner}.l2_resident_256x128_pair_service", service,
        {"active_aic_groups", "tile_m", "tile_n", "tile_k", "a_bytes",
         "b_bytes", "pair_bytes", "a_median_ns", "b_median_ns",
         "pair_median_ns", "pair_to_isolated_sum_ratio",
         "pair_max_relative_median_deviation",
         "logical_byte_capacity_per_s"})
    frozen = {
        "active_aic_groups": 24,
        "tile_m": 256,
        "tile_n": 128,
        "tile_k": 64,
        "a_bytes": 32768,
        "b_bytes": 16384,
        "pair_bytes": 49152,
    }
    for name, expected_value in frozen.items():
        if service[name] != expected_value:
            raise ValueError(f"{owner}.{name} changed")
    pair_ns = float(service["pair_median_ns"])
    logical_capacity = float(service["logical_byte_capacity_per_s"])
    if (not math.isfinite(pair_ns) or pair_ns <= 0.0
            or not math.isfinite(logical_capacity)
            or logical_capacity <= 0.0
            or not math.isclose(
                service["pair_bytes"] * service["active_aic_groups"]
                / (pair_ns * 1.0e-9), logical_capacity,
                rel_tol=1.0e-12, abs_tol=0.0)):
        raise ValueError(f"{owner} pair duration and capacity disagree")
    measurement = value["measurement"]
    require_exact_fields(
        f"{owner}.measurement", measurement,
        {"mode", "source_commit", "source_snapshot_sha256",
         "binary_sha256", "independent_processes",
         "replications_per_process", "order", "active_aic_groups",
         "epochs_per_aic", "raw_paths", "raw_sha256",
         "full_production_gemm_executed"})
    expected_measurement = {
        "mode": "independent_source_shaped_mte2_pair_microbenchmark",
        "independent_processes": 5,
        "replications_per_process": 4,
        "order": "alternating_forward_reverse",
        "active_aic_groups": 24,
        "epochs_per_aic": 1366,
        "full_production_gemm_executed": False,
    }
    if any(measurement[name] != expected_value
           for name, expected_value in expected_measurement.items()):
        raise ValueError(f"{owner} measurement contract changed")
    raw_paths = measurement["raw_paths"]
    raw_hashes = measurement["raw_sha256"]
    if (not isinstance(raw_paths, list) or len(raw_paths) != 5
            or not isinstance(raw_hashes, list) or len(raw_hashes) != 5):
        raise ValueError(f"{owner} requires five independent captures")
    if any(not isinstance(raw_path, str)
           or not raw_path.startswith(
               "profiler_output/microbench/ascend_910b/")
           or ".." in raw_path.split("/") for raw_path in raw_paths):
        raise ValueError(f"{owner} raw path is outside its evidence owner")
    evidence_hashes = tuple(raw_hashes) + tuple(
        measurement[name] for name in (
            "source_commit", "source_snapshot_sha256", "binary_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 40
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes[-3:-2]):
        raise ValueError(f"{owner} source commit is invalid")
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in (*evidence_hashes[:-3], *evidence_hashes[-2:])):
        raise ValueError(f"{owner} evidence SHA256 is invalid")
    mechanism = value["mechanism"]
    require_exact_fields(
        f"{owner}.mechanism", mechanism,
        {"descriptor", "copy_in_order", "a1_b1_banks_independent",
         "consumer_rule"})
    if (mechanism["descriptor"] != "BF16_ND_to_NZ"
            or mechanism["copy_in_order"] != ["A.LoadData", "B.LoadData"]
            or mechanism["a1_b1_banks_independent"] is not True
            or mechanism["consumer_rule"] != (
                "use_joint_pair_completion_not_isolated_duration_sum")):
        raise ValueError(f"{owner} source mechanism changed")
    return {
        "model": "aic_matmul_v3_asw_mte2_pair_stream_v1",
        **frozen,
        "minimum_pairs_per_worker": measurement["epochs_per_aic"],
        "continuous_ns_per_pair": pair_ns,
        "reference_path": str(AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE),
        "reference_sha256": AIC_MATMUL_V3_ASW_MTE2_PAIR_REFERENCE_SHA256,
    }


def load_aic_mte2_cold_n_major_b_stride_service(repository_root):
    """Load the exact cold M16 pair and N-major B-stride services."""

    stride_path = repository_root / AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE
    stride_bytes = stride_path.read_bytes()
    if hashlib.sha256(stride_bytes).hexdigest() != (
            AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE_SHA256):
        raise ValueError("cold N-major B-stride reference SHA256 mismatch")
    stride = json.loads(stride_bytes)
    owner = "cold N-major B-stride reference"
    require_exact_fields(
        owner, stride,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "parameters_ns",
         "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_cold_n_major_b_nd2nz_stride",
        "duration_unit": "ns_per_64x256_b_descriptor_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if stride[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = stride["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"source_residency", "dtype", "rows", "columns",
         "source_row_stride_elements", "source_tile_span_bytes",
         "copied_bytes_per_descriptor", "active_aic_groups",
         "iterations_per_worker", "footprint_bytes",
         "source_tiles_per_worker", "aggregate_copied_ring_bytes",
         "source_ring_constraint"})
    expected_scope = {
        "source_residency": "dram_cold",
        "dtype": "bf16",
        "rows": 64,
        "columns": 256,
        "source_row_stride_elements": 28672,
        "source_tile_span_bytes": 3613184,
        "copied_bytes_per_descriptor": 32768,
        "active_aic_groups": 23,
        "iterations_per_worker": 640,
        "footprint_bytes": 53687091200,
        "source_tiles_per_worker": 646,
        "aggregate_copied_ring_bytes": 486866944,
    }
    for name, expected_value in expected_scope.items():
        if scope[name] != expected_value:
            raise ValueError(f"{owner}.publication_scope.{name} changed")
    if stride["measurement"]["mechanism_gate_passed"] is not True:
        raise ValueError(f"{owner} mechanism gate failed")
    serial_ns = stride["parameters_ns"]["serial_completion"]
    stream_ns = stride["parameters_ns"]["two_buffer_stream"]
    for name, value in (
            ("serial_completion", serial_ns),
            ("two_buffer_stream", stream_ns)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{owner}.{name} must be finite and positive")

    pair_path = repository_root / AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE
    pair_bytes = pair_path.read_bytes()
    if hashlib.sha256(pair_bytes).hexdigest() != (
            AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE_SHA256):
        raise ValueError("cold M16 operand-pair reference SHA256 mismatch")
    pair = json.loads(pair_bytes)
    pair_scope = pair["publication_scope"]
    pair_expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "aic_mte2_cold_operand_pair_stream",
        "service_time_unit": "ns_per_16x64_A_plus_64x256_B_pair_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in pair_expected.items():
        if pair[name] != expected_value:
            raise ValueError(f"cold M16 operand-pair reference.{name} changed")
    expected_pair_scope = {
        "source_residency": "dram_cold",
        "active_aic_groups": 23,
        "a_rows": 16,
        "a_columns": 64,
        "b_rows": 64,
        "b_columns": 256,
        "drain_interval_pairs": 4,
        "pair_count_per_worker_min": 512,
        "pair_count_per_worker_max": 2048,
    }
    for name, expected_value in expected_pair_scope.items():
        if pair_scope[name] != expected_value:
            raise ValueError(
                f"cold M16 operand-pair reference.publication_scope."
                f"{name} changed")
    pair_ns = pair["published_cold_pair_service_ns"]
    if (isinstance(pair_ns, bool) or not isinstance(pair_ns, (int, float))
            or not math.isfinite(pair_ns) or pair_ns <= 0
            or pair["measurement"]["mechanism_gate_passed"] is not True):
        raise ValueError("cold M16 operand-pair service is invalid")
    return {
        "model": "cold_m16_pair_plus_n_major_b_serial_stride_delta_v1",
        "dtype": scope["dtype"],
        "source_residency": scope["source_residency"],
        "a_rows": pair_scope["a_rows"],
        "a_columns": pair_scope["a_columns"],
        "b_rows": scope["rows"],
        "b_columns": scope["columns"],
        "b_source_row_stride_elements": scope[
            "source_row_stride_elements"],
        "active_aic_groups": scope["active_aic_groups"],
        "pairs_per_worker": scope["iterations_per_worker"],
        "contiguous_pair_ns": float(pair_ns),
        "strided_b_serial_ns": float(serial_ns),
        "strided_b_stream_ns": float(stream_ns),
        "pair_reference_path": str(
            AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE),
        "pair_reference_sha256": (
            AIC_MTE2_COLD_M16_OPERAND_PAIR_REFERENCE_SHA256),
        "stride_reference_path": str(
            AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE),
        "stride_reference_sha256": (
            AIC_MTE2_COLD_N_MAJOR_B_STRIDE_REFERENCE_SHA256),
    }


def load_aic_mte2_legacy_l1_pair_service(repository_root):
    """Load the frozen cold legacy MatMulV2 pair-program service."""

    path = repository_root / AIC_MTE2_LEGACY_L1_PAIR_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_LEGACY_L1_PAIR_REFERENCE_SHA256):
        raise ValueError("legacy L1 pair reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "legacy L1 pair reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "source_residency", "capture", "cells",
         "production_or_evaluation_duration_consumed"})
    expected_text = {
        "schema_version": (
            "midlayer.ascend_910b.aic_mte2_legacy_l1_pair_reference.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_legacy_matmul_v2_cold_pair_program_v1",
        "duration_unit": "ns_per_a_b_operand_pair_per_aic",
        "source_residency": "dram_cold_disjoint",
    }
    for name, expected in expected_text.items():
        if value[name] != expected:
            raise ValueError(f"{owner}.{name} must be {expected!r}")
    if value["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(f"{owner} cannot consume production duration")
    capture = value["capture"]
    require_exact_fields(
        f"{owner}.capture", capture,
        {"source_commit", "source_snapshot_sha256", "binary_sha256",
         "footprint_mb_per_operand", "short_programs", "long_programs",
         "run_count", "raw_paths", "raw_sha256"})
    if (capture["footprint_mb_per_operand"] != 16384
            or capture["short_programs"] != 1
            or capture["long_programs"] != 4
            or capture["run_count"] != 3):
        raise ValueError(f"{owner} capture geometry changed")
    for name in ("source_commit", "source_snapshot_sha256", "binary_sha256"):
        digest = capture[name]
        if name == "source_commit":
            expected_length = 40
        elif name in {"source_snapshot_sha256", "binary_sha256"}:
            expected_length = 64
        else:
            raise ValueError(f"{owner}.capture has an unknown digest field")
        if (not isinstance(digest, str) or len(digest) != expected_length
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.capture.{name} is invalid")
    raw_paths = capture["raw_paths"]
    raw_hashes = capture["raw_sha256"]
    if (not isinstance(raw_paths, list) or not isinstance(raw_hashes, list)
            or len(raw_paths) != 3 or len(raw_hashes) != 3):
        raise ValueError(f"{owner} requires three raw captures")
    for index, raw_path in enumerate(raw_paths):
        if (not isinstance(raw_path, str)
                or not raw_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in raw_path.split("/")):
            raise ValueError(f"{owner}.raw_paths[{index}] is invalid")
    for index, digest in enumerate(raw_hashes):
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner}.raw_sha256[{index}] is invalid")
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError(f"{owner} requires four validation cells")
    cell_fields = {
        "l0c_pipeline_depth", "a_rows", "a_columns",
        "a_source_row_stride_elements", "b_rows", "b_columns",
        "b_source_row_stride_elements", "active_aic_groups",
        "pairs_per_worker", "drain_after_each_pair", "ns_per_pair_samples",
        "production_consumed",
    }
    expected_geometries = {
        2: (128, 256, 14336, 96, 256, 14336, 22, 112),
        1: (256, 256, 5120, 112, 256, 5120, 24, 80),
    }
    consumed = []
    single_l0c_controls = []
    identities = set()
    for index, cell in enumerate(cells):
        require_exact_fields(f"{owner}.cells[{index}]", cell, cell_fields)
        depth = cell["l0c_pipeline_depth"]
        drain = cell["drain_after_each_pair"]
        if depth not in expected_geometries or drain not in (0, 1):
            raise ValueError(f"{owner}.cells[{index}] mode is invalid")
        identity = (depth, drain)
        if identity in identities:
            raise ValueError(f"{owner} repeats a validation cell")
        identities.add(identity)
        geometry = tuple(
            cell[name] for name in (
                "a_rows", "a_columns", "a_source_row_stride_elements",
                "b_rows", "b_columns", "b_source_row_stride_elements",
                "active_aic_groups", "pairs_per_worker"))
        if geometry != expected_geometries[depth]:
            raise ValueError(f"{owner}.cells[{index}] geometry changed")
        samples = cell["ns_per_pair_samples"]
        if (not isinstance(samples, list) or len(samples) != 3
                or any(isinstance(sample, bool)
                       or not isinstance(sample, (int, float))
                       or not math.isfinite(sample) or sample <= 0
                       for sample in samples)):
            raise ValueError(f"{owner}.cells[{index}] samples are invalid")
        median = sorted(float(sample) for sample in samples)[1]
        if (max(samples) - min(samples)) / median > 0.05:
            raise ValueError(f"{owner}.cells[{index}] is unstable")
        if cell["production_consumed"] is True:
            consumed.append({
                **dict(zip((
                    "a_rows", "a_columns", "a_source_row_stride_elements",
                    "b_rows", "b_columns", "b_source_row_stride_elements",
                    "active_aic_groups", "pairs_per_worker"), geometry)),
                "continuous_ns_per_pair": median,
            })
        elif cell["production_consumed"] is not False:
            raise ValueError(
                f"{owner}.cells[{index}].production_consumed must be boolean")
        elif identity == (1, 1):
            single_l0c_controls.append(dict(zip((
                "a_rows", "a_columns", "a_source_row_stride_elements",
                "b_rows", "b_columns", "b_source_row_stride_elements",
                "active_aic_groups", "pairs_per_worker"), geometry)))
    if identities != {(1, 0), (1, 1), (2, 0), (2, 1)}:
        raise ValueError(f"{owner} validation matrix is incomplete")
    if len(consumed) != 1 or len(single_l0c_controls) != 1:
        raise ValueError(f"{owner} must consume only the double-L0C cell")
    return {
        "double_l0c_service": consumed[0],
        "single_l0c_control": single_l0c_controls[0],
    }


def load_aic_mte2_cold_legacy_k14336_stride_service(repository_root):
    """Load cold serial services for exact legacy K14336 descriptors."""

    path = repository_root / AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE_SHA256):
        raise ValueError("cold legacy K14336 stride reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "cold legacy K14336 stride reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "parameters_ns",
         "measurement", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_cold_legacy_k14336_nd2nz_stride",
        "duration_unit": "ns_per_descriptor_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")

    scope = value["publication_scope"]
    require_exact_fields(
        f"{owner}.publication_scope", scope,
        {"source_residency", "dtype", "active_aic_groups",
         "iterations_per_worker", "footprint_bytes", "descriptors",
         "source_ring_constraint"})
    expected_scope = {
        "source_residency": "dram_cold",
        "dtype": "bf16",
        "active_aic_groups": 22,
        "iterations_per_worker": 224,
        "footprint_bytes": 8589934592,
    }
    for name, expected_value in expected_scope.items():
        if scope[name] != expected_value:
            raise ValueError(
                f"{owner}.publication_scope.{name} changed")
    descriptors = scope["descriptors"]
    if not isinstance(descriptors, list) or len(descriptors) != 2:
        raise ValueError(f"{owner} requires two descriptor cells")
    descriptor_fields = {
        "operand", "rows", "columns", "source_row_stride_elements",
        "source_tile_span_bytes", "copied_bytes_per_descriptor",
        "source_tiles_per_worker", "aggregate_copied_ring_bytes",
    }
    expected_descriptors = {
        "a": (16, 4096, 14336, 438272, 131072, 890, 2566389760),
        "b": (96, 640, 14336, 2725120, 122880, 143, 386580480),
    }
    normalized_descriptors = {}
    for index, descriptor in enumerate(descriptors):
        require_exact_fields(
            f"{owner}.publication_scope.descriptors[{index}]",
            descriptor, descriptor_fields)
        operand = descriptor["operand"]
        if operand not in expected_descriptors or operand in normalized_descriptors:
            raise ValueError(f"{owner} descriptor operand is invalid")
        geometry = tuple(descriptor[name] for name in (
            "rows", "columns", "source_row_stride_elements",
            "source_tile_span_bytes", "copied_bytes_per_descriptor",
            "source_tiles_per_worker", "aggregate_copied_ring_bytes"))
        if geometry != expected_descriptors[operand]:
            raise ValueError(f"{owner} {operand} descriptor changed")
        if descriptor["aggregate_copied_ring_bytes"] <= 192 * 1024 * 1024:
            raise ValueError(f"{owner} {operand} descriptor is not L2-cold")
        normalized_descriptors[operand] = descriptor

    parameters = value["parameters_ns"]
    require_exact_fields(f"{owner}.parameters_ns", parameters, {"a", "b"})
    services = {}
    for operand in ("a", "b"):
        parameter = parameters[operand]
        require_exact_fields(
            f"{owner}.parameters_ns.{operand}", parameter,
            {"serial_completion", "two_buffer_stream"})
        for name in ("serial_completion", "two_buffer_stream"):
            duration = parameter[name]
            if (isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(
                    f"{owner}.{operand}.{name} must be finite and positive")
        descriptor = normalized_descriptors[operand]
        services[operand] = {
            "rows": descriptor["rows"],
            "columns": descriptor["columns"],
            "source_row_stride_elements": descriptor[
                "source_row_stride_elements"],
            "copied_bytes_per_descriptor": descriptor[
                "copied_bytes_per_descriptor"],
            "serial_completion_ns": float(parameter["serial_completion"]),
            "two_buffer_stream_ns": float(parameter["two_buffer_stream"]),
        }

    measurement = value["measurement"]
    required_measurement_fields = {
        f"{operand}_{mode}_{suffix}"
        for operand in ("a", "b")
        for mode in ("serial", "two_buffer_stream")
        for suffix in ("trial_ns", "relative_span", "order_balance_error")
    }
    required_measurement_fields.update({
        "maximum_allowed_relative_span",
        "maximum_allowed_order_balance_error", "stability_gate",
        "mechanism_gate_passed",
    })
    require_exact_fields(owner + ".measurement", measurement,
                          required_measurement_fields)
    if (measurement["stability_gate"] is not True
            or measurement["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    span_limit = measurement["maximum_allowed_relative_span"]
    balance_limit = measurement["maximum_allowed_order_balance_error"]
    if span_limit != 0.1 or balance_limit != 0.1:
        raise ValueError(f"{owner} stability thresholds changed")
    for operand in ("a", "b"):
        for mode in ("serial", "two_buffer_stream"):
            samples = measurement[f"{operand}_{mode}_trial_ns"]
            if (not isinstance(samples, list) or len(samples) != 4
                    or any(isinstance(sample, bool)
                           or not isinstance(sample, (int, float))
                           or not math.isfinite(sample) or sample <= 0
                           for sample in samples)):
                raise ValueError(
                    f"{owner} {operand} {mode} samples are invalid")
            ordered = sorted(float(sample) for sample in samples)
            median = (ordered[1] + ordered[2]) / 2
            if mode == "serial":
                parameter_name = "serial_completion"
            elif mode == "two_buffer_stream":
                parameter_name = "two_buffer_stream"
            else:
                raise ValueError(f"{owner} has an unknown measurement mode")
            if not math.isclose(
                    median, float(parameters[operand][parameter_name]),
                    rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"{owner} {operand} {mode} median changed")
            relative_span = measurement[f"{operand}_{mode}_relative_span"]
            order_balance = measurement[
                f"{operand}_{mode}_order_balance_error"]
            if (not isinstance(relative_span, (int, float))
                    or not math.isfinite(relative_span)
                    or relative_span < 0 or relative_span > span_limit):
                raise ValueError(
                    f"{owner} {operand} {mode} span is unstable")
            if (not isinstance(order_balance, (int, float))
                    or not math.isfinite(order_balance)
                    or order_balance < 0 or order_balance > balance_limit):
                raise ValueError(
                    f"{owner} {operand} {mode} order balance is unstable")
    return {
        "model": "cold_contiguous_envelope_plus_serial_stride_delta_v1",
        "dtype": scope["dtype"],
        "source_residency": scope["source_residency"],
        "active_aic_groups": scope["active_aic_groups"],
        "descriptors": services,
        "reference_path": str(
            AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE),
        "reference_sha256": (
            AIC_MTE2_COLD_LEGACY_K14336_STRIDE_REFERENCE_SHA256),
    }


def load_aic_mte2_cold_v3_k1536_n5120_stride_service(repository_root):
    """Load exact cold serial services for K1536/N5120 B descriptors."""

    path = repository_root / AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE_SHA256):
        raise ValueError("cold V3 K1536/N5120 stride reference SHA mismatch")
    value = json.loads(payload_bytes)
    owner = "cold V3 K1536/N5120 stride reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_cold_v3_k1536_n5120_nd2nz_stride",
        "duration_unit": "ns_per_descriptor_per_aic",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} must be {expected_value!r}")
    scope = value["publication_scope"]
    require_exact_fields(
        owner + ".publication_scope", scope,
        {"source_residency", "dtype", "iterations_per_worker",
         "footprint_bytes", "descriptors", "source_ring_constraint",
         "maximum_allowed_relative_span",
         "maximum_allowed_order_balance_error", "stability_gate",
         "mechanism_gate_passed"})
    if (scope["source_residency"] != "dram_cold"
            or scope["dtype"] != "bf16"
            or scope["iterations_per_worker"] != 224
            or scope["footprint_bytes"] != 8589934592
            or scope["maximum_allowed_relative_span"] != 0.1
            or scope["maximum_allowed_order_balance_error"] != 0.1
            or scope["stability_gate"] is not True
            or scope["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} publication scope changed")
    descriptor_fields = {
        "rows", "columns", "source_row_stride_elements",
        "active_aic_groups", "source_tile_span_bytes",
        "copied_bytes_per_descriptor", "source_tiles_per_worker",
        "aggregate_copied_ring_bytes", "serial_completion_ns",
        "two_buffer_stream_ns", "serial_trial_ns",
        "two_buffer_stream_trial_ns", "serial_relative_span",
        "two_buffer_stream_relative_span", "serial_order_balance_error",
        "two_buffer_stream_order_balance_error",
    }
    expected_geometries = {
        (224, 256, 5120, 23): (2284032, 114688, 163, 429965312),
        (256, 256, 5120, 20): (2611712, 131072, 164, 429916160),
        (256, 256, 5120, 24): (2611712, 131072, 137, 430964736),
    }
    services = {}
    descriptors = scope["descriptors"]
    if not isinstance(descriptors, list) or len(descriptors) != 3:
        raise ValueError(f"{owner} requires three descriptor cells")
    for index, descriptor in enumerate(descriptors):
        require_exact_fields(
            f"{owner}.descriptors[{index}]", descriptor, descriptor_fields)
        signature = tuple(descriptor[name] for name in (
            "rows", "columns", "source_row_stride_elements",
            "active_aic_groups"))
        if signature not in expected_geometries or signature in services:
            raise ValueError(f"{owner} descriptor signature is invalid")
        captured = tuple(descriptor[name] for name in (
            "source_tile_span_bytes", "copied_bytes_per_descriptor",
            "source_tiles_per_worker", "aggregate_copied_ring_bytes"))
        if captured != expected_geometries[signature]:
            raise ValueError(f"{owner} descriptor capture geometry changed")
        for mode in ("serial", "two_buffer_stream"):
            samples = descriptor[f"{mode}_trial_ns"]
            if mode == "serial":
                duration_name = "serial_completion_ns"
            elif mode == "two_buffer_stream":
                duration_name = "two_buffer_stream_ns"
            else:
                raise ValueError(f"{owner} has an unknown descriptor mode")
            duration = descriptor[duration_name]
            if (not isinstance(samples, list) or len(samples) != 4
                    or any(not isinstance(sample, (int, float))
                           or isinstance(sample, bool)
                           or not math.isfinite(sample) or sample <= 0
                           for sample in samples)
                    or not isinstance(duration, (int, float))
                    or isinstance(duration, bool)
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f"{owner} descriptor samples are invalid")
        services[signature] = {
            "copied_bytes_per_descriptor": descriptor[
                "copied_bytes_per_descriptor"],
            "serial_completion_ns": float(
                descriptor["serial_completion_ns"]),
            "two_buffer_stream_ns": float(
                descriptor["two_buffer_stream_ns"]),
        }
    return {
        "model": "cold_contiguous_envelope_plus_serial_stride_delta_v1",
        "dtype": scope["dtype"],
        "source_residency": scope["source_residency"],
        "descriptors": services,
        "reference_path": str(
            AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE),
        "reference_sha256": (
            AIC_MTE2_COLD_V3_K1536_N5120_STRIDE_REFERENCE_SHA256),
    }


def load_aic_mte2_cold_small_m_stride_service(repository_root):
    """Load exact cold serial services for remaining small-M descriptors."""

    path = repository_root / AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE_SHA256):
        raise ValueError("cold small-M stride reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "cold small-M stride reference"
    require_exact_fields(
        owner, value,
        {"schema_version", "hardware_id", "software_revision", "mechanism",
         "duration_unit", "publication_scope", "cells", "provenance",
         "production_or_evaluation_duration_consumed",
         "parameter_fit_from_production_duration", "mechanism_publishable",
         "forward_eligible", "forward_eligibility_constraint"})
    if (value["schema_version"] != 1
            or value["hardware_id"] != "ascend_910b"
            or value["software_revision"] != "cann_8.5.1"
            or value["mechanism"] != "aic_mte2_cold_small_m_nd2nz_stride"
            or value["duration_unit"] != "ns_per_descriptor_per_aic"
            or value["production_or_evaluation_duration_consumed"] is not False
            or value["parameter_fit_from_production_duration"] is not False
            or value["mechanism_publishable"] is not True
            or value["forward_eligible"] is not True):
        raise ValueError(f"{owner} top-level contract changed")
    scope = value["publication_scope"]
    if (scope["source_residency"] != "dram_cold"
            or scope["dtype"] != "bf16"
            or scope["iterations_per_worker"] != 224
            or scope["footprint_bytes"] != 8589934592
            or scope["stability_gate"] is not True
            or scope["mechanism_gate_passed"] is not True):
        raise ValueError(f"{owner} publication scope changed")
    cell_fields = {
        "rows", "columns", "source_row_stride_elements",
        "active_aic_groups", "copied_bytes_per_descriptor",
        "serial_completion_ns", "serial_relative_span",
        "serial_order_balance_error", "production_consumed",
    }
    services = {}
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != 40:
        raise ValueError(f"{owner} requires forty descriptor cells")
    for index, cell in enumerate(cells):
        require_exact_fields(f"{owner}.cells[{index}]", cell, cell_fields)
        signature = tuple(cell[name] for name in (
            "rows", "columns", "source_row_stride_elements",
            "active_aic_groups"))
        if signature in services:
            raise ValueError(f"{owner} repeats a descriptor signature")
        if (cell["serial_relative_span"] > 0.1
                or cell["serial_order_balance_error"] > 0.1):
            raise ValueError(f"{owner} descriptor cell is unstable")
        if cell["production_consumed"] is True:
            services[signature] = {
                "copied_bytes_per_descriptor": cell[
                    "copied_bytes_per_descriptor"],
                "serial_completion_ns": float(cell["serial_completion_ns"]),
            }
        elif cell["production_consumed"] is not False:
            raise ValueError(
                f"{owner}.cells[{index}].production_consumed is invalid")
    provenance = value["provenance"]
    observation_fields = {
        "tile_m", "tile_n", "tile_k", "active_aic_groups", "a_columns",
        "b_columns", "a_buffer_depth", "b_buffer_depth",
        "kernel_tiling_key", "result_path", "result_sha256",
        "tiling_payload_sha256",
    }
    l1_layouts = {}
    asymmetric_descriptor_signatures = set()

    def record_l1_layout(observation_owner, source_shape, observation):
        require_exact_fields(
            observation_owner, observation, observation_fields)
        integer_fields = observation_fields - {
            "result_path", "result_sha256", "tiling_payload_sha256"}
        if any(
                isinstance(observation[name], bool)
                or not isinstance(observation[name], int)
                or observation[name] <= 0
                for name in integer_fields):
            raise ValueError(
                f"{owner} dry-launch observation integers must be positive")
        if any(
                not isinstance(observation[name], str)
                or not observation[name]
                for name in ("result_path", "result_sha256",
                             "tiling_payload_sha256")):
            raise ValueError(
                f"{owner} dry-launch observation provenance is invalid")
        tile_k = observation["tile_k"]
        if (observation["a_columns"] % tile_k
                or observation["b_columns"] % tile_k
                or observation["a_columns"] > source_shape["k"]
                or observation["b_columns"] > source_shape["k"]):
            raise ValueError(
                f"{owner} dry-launch L1 columns are not K-tile aligned")
        signature = (
            source_shape["k"], source_shape["n"], observation["tile_m"],
            observation["tile_n"], tile_k,
            observation["active_aic_groups"],
        )
        if signature in l1_layouts:
            raise ValueError(f"{owner} repeats a dry-launch signature")
        a_signature = (
            observation["tile_m"], observation["a_columns"],
            source_shape["k"], observation["active_aic_groups"])
        b_signature = (
            observation["tile_n"], observation["b_columns"],
            source_shape["n"], observation["active_aic_groups"])
        a_stride_service = None
        if a_signature in services:
            a_stride_service = services[a_signature]
        b_stride_service = None
        if b_signature in services:
            b_stride_service = services[b_signature]
        l1_layouts[signature] = {
            "a_columns": observation["a_columns"],
            "b_columns": observation["b_columns"],
            "a_buffer_depth": observation["a_buffer_depth"],
            "b_buffer_depth": observation["b_buffer_depth"],
            "kernel_tiling_key": observation["kernel_tiling_key"],
            "a_stride_service": a_stride_service,
            "b_stride_service": b_stride_service,
        }
        if observation["a_buffer_depth"] != observation["b_buffer_depth"]:
            asymmetric_descriptor_signatures.update(
                (a_signature, b_signature))

    capture_fields = {
        "capture_commit", "snapshot_path", "raw_path", "raw_sha256",
        "stderr_path", "stderr_sha256", "host", "container",
        "physical_device_id", "logical_device_id", "owner", "runner_pid",
        "child_pids", "source_snapshot_sha256", "source_sha256",
        "binary_sha256", "command", "source_shape",
        "dry_launch_observations",
    }
    capture_names = (
        ("matmul_v2_k5120_n5120_capture", 4, False),
        ("matmul_v2_k5120_n1536_capture", 1, True),
        ("matmul_v2_key98499_k4096_n4096_capture", 1, False),
        ("matmul_v2_key98499_k1536_n5120_capture", 1, False),
    )
    for capture_name, observation_count, has_corroboration in capture_names:
        capture = provenance[capture_name]
        expected_capture_fields = set(capture_fields)
        if has_corroboration:
            expected_capture_fields.add("corroborating_dry_launch_results")
        require_exact_fields(
            f"{owner}.provenance.{capture_name}", capture,
            expected_capture_fields)
        source_shape = capture["source_shape"]
        require_exact_fields(
            f"{owner}.{capture_name}.source_shape", source_shape, {"k", "n"})
        if any(
                isinstance(source_shape[name], bool)
                or not isinstance(source_shape[name], int)
                or source_shape[name] <= 0
                for name in ("k", "n")):
            raise ValueError(
                f"{owner} source shape must contain positive integers")
        observations = capture["dry_launch_observations"]
        if (not isinstance(observations, list)
                or len(observations) != observation_count):
            raise ValueError(
                f"{owner}.{capture_name} has the wrong observation count")
        for index, observation in enumerate(observations):
            record_l1_layout(
                f"{owner}.{capture_name}.dry_launch_observations[{index}]",
                source_shape, observation)
        if has_corroboration:
            corroborations = capture["corroborating_dry_launch_results"]
            if not isinstance(corroborations, list) or len(corroborations) != 1:
                raise ValueError(f"{owner} requires one corroborating launch")
            corroboration = corroborations[0]
            require_exact_fields(
                f"{owner}.{capture_name}.corroborating_dry_launch_results[0]",
                corroboration,
                {"batch_size", "kernel_tiling_key", "result_path",
                 "result_sha256", "tiling_payload_sha256"})
            if (corroboration["batch_size"] != 4
                    or corroboration["kernel_tiling_key"]
                    != observations[0]["kernel_tiling_key"]):
                raise ValueError(
                    f"{owner} corroborating dry launch does not match")
    descriptor_capture = provenance["p90_boundary_descriptor_capture"]
    descriptor_capture_fields = {
        "capture_commit", "snapshot_path", "raw_path", "raw_sha256",
        "stderr_path", "stderr_sha256", "npu_smi_before_sha256",
        "npu_smi_after_sha256", "host", "container", "physical_device_id",
        "logical_device_id", "owner", "runner_pid", "child_pids",
        "source_snapshot_sha256", "source_sha256", "binary_sha256",
        "command",
    }
    require_exact_fields(
        f"{owner}.provenance.p90_boundary_descriptor_capture",
        descriptor_capture, descriptor_capture_fields)
    if (descriptor_capture["capture_commit"]
            != "e3d8579455cd96f812eb2555ced7ccf4fac3e70c"
            or (not isinstance(descriptor_capture["host"], str)
                or not descriptor_capture["host"].strip())
            or (not isinstance(descriptor_capture["container"], str)
                or not descriptor_capture["container"].strip())
            or (isinstance(descriptor_capture["physical_device_id"], bool)
                or not isinstance(descriptor_capture["physical_device_id"], int)
                or descriptor_capture["physical_device_id"] < 0)
            or (isinstance(descriptor_capture["logical_device_id"], bool)
                or not isinstance(descriptor_capture["logical_device_id"], int)
                or descriptor_capture["logical_device_id"] < 0)
            or descriptor_capture["owner"] != "kernel_accuracy_gemm"
            or descriptor_capture["runner_pid"] <= 0
            or (not isinstance(descriptor_capture["child_pids"], list)
                or len(descriptor_capture["child_pids"]) != 4
                or any(isinstance(pid, bool) or not isinstance(pid, int)
                       or pid <= 0 for pid in descriptor_capture["child_pids"]))):
        raise ValueError(f"{owner} P90 descriptor capture identity changed")
    source_hashes = descriptor_capture["source_sha256"]
    if set(source_hashes) != {
            "CMakeLists.txt", "aic_mte2_nd2nz_stride_kernel.cpp",
            "aic_mte2_nd2nz_stride_main.cpp",
            "aic_mte2_nd2nz_stride_runner.py"}:
        raise ValueError(f"{owner} P90 descriptor capture sources changed")
    evidence_hashes = tuple(source_hashes.values()) + tuple(
        descriptor_capture[name] for name in (
            "raw_sha256", "stderr_sha256", "npu_smi_before_sha256",
            "npu_smi_after_sha256", "source_snapshot_sha256",
            "binary_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in evidence_hashes):
        raise ValueError(f"{owner} P90 descriptor evidence SHA256 is invalid")
    for path_name in ("raw_path", "stderr_path"):
        evidence_path = descriptor_capture[path_name]
        if (not isinstance(evidence_path, str)
                or not evidence_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in evidence_path.split("/")):
            raise ValueError(
                f"{owner} P90 descriptor path is outside its owner")
    tail_capture = provenance["tail_descriptor_capture"]
    tail_capture_fields = descriptor_capture_fields | {
        "binary_source_commit"}
    require_exact_fields(
        f"{owner}.provenance.tail_descriptor_capture",
        tail_capture, tail_capture_fields)
    if (tail_capture["capture_commit"]
            != "e3d8579455cd96f812eb2555ced7ccf4fac3e70c"
            or tail_capture["binary_source_commit"]
                != "e3d8579455cd96f812eb2555ced7ccf4fac3e70c"
            or (not isinstance(tail_capture["host"], str)
                or not tail_capture["host"].strip())
            or (not isinstance(tail_capture["container"], str)
                or not tail_capture["container"].strip())
            or (isinstance(tail_capture["physical_device_id"], bool)
                or not isinstance(tail_capture["physical_device_id"], int)
                or tail_capture["physical_device_id"] < 0)
            or (isinstance(tail_capture["logical_device_id"], bool)
                or not isinstance(tail_capture["logical_device_id"], int)
                or tail_capture["logical_device_id"] < 0)
            or tail_capture["owner"] != "kernel_accuracy_gemm"
            or tail_capture["runner_pid"] <= 0
            or (not isinstance(tail_capture["child_pids"], list)
                or len(tail_capture["child_pids"]) != 4
                or any(isinstance(pid, bool) or not isinstance(pid, int)
                       or pid <= 0 for pid in tail_capture["child_pids"]))):
        raise ValueError(f"{owner} tail descriptor capture identity changed")
    tail_source_hashes = tail_capture["source_sha256"]
    if set(tail_source_hashes) != {
            "CMakeLists.txt", "aic_mte2_nd2nz_stride_kernel.cpp",
            "aic_mte2_nd2nz_stride_main.cpp",
            "aic_mte2_nd2nz_stride_runner.py"}:
        raise ValueError(f"{owner} tail descriptor capture sources changed")
    tail_evidence_hashes = tuple(tail_source_hashes.values()) + tuple(
        tail_capture[name] for name in (
            "raw_sha256", "stderr_sha256", "npu_smi_before_sha256",
            "npu_smi_after_sha256", "source_snapshot_sha256",
            "binary_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in tail_evidence_hashes):
        raise ValueError(f"{owner} tail descriptor evidence SHA256 is invalid")
    for path_name in ("raw_path", "stderr_path"):
        evidence_path = tail_capture[path_name]
        if (not isinstance(evidence_path, str)
                or not evidence_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in evidence_path.split("/")):
            raise ValueError(
                f"{owner} tail descriptor path is outside its owner")
    max_ape_capture = provenance["max_ape_descriptor_capture"]
    require_exact_fields(
        f"{owner}.provenance.max_ape_descriptor_capture",
        max_ape_capture, descriptor_capture_fields)
    if (max_ape_capture["capture_commit"]
            != "842f2e5770ef036d3a2842ec3d0b72b174f9257d"
            or (not isinstance(max_ape_capture["host"], str)
                or not max_ape_capture["host"].strip())
            or (not isinstance(max_ape_capture["container"], str)
                or not max_ape_capture["container"].strip())
            or (isinstance(max_ape_capture["physical_device_id"], bool)
                or not isinstance(max_ape_capture["physical_device_id"], int)
                or max_ape_capture["physical_device_id"] < 0)
            or (isinstance(max_ape_capture["logical_device_id"], bool)
                or not isinstance(max_ape_capture["logical_device_id"], int)
                or max_ape_capture["logical_device_id"] < 0)
            or max_ape_capture["owner"] != "kernel_accuracy_gemm"
            or (isinstance(max_ape_capture["runner_pid"], bool) or not isinstance(max_ape_capture["runner_pid"], int)
                or max_ape_capture["runner_pid"] <= 0)
            or (not isinstance(max_ape_capture["child_pids"], list)
                or len(max_ape_capture["child_pids"]) != 4
                or any(isinstance(pid, bool) or not isinstance(pid, int)
                       or pid <= 0 for pid in max_ape_capture["child_pids"]))):
        raise ValueError(f"{owner} MaxAPE descriptor capture identity changed")
    max_ape_source_hashes = max_ape_capture["source_sha256"]
    if set(max_ape_source_hashes) != {
            "CMakeLists.txt", "aic_mte2_nd2nz_stride_kernel.cpp",
            "aic_mte2_nd2nz_stride_main.cpp",
            "aic_mte2_nd2nz_stride_runner.py"}:
        raise ValueError(f"{owner} MaxAPE descriptor sources changed")
    max_ape_evidence_hashes = tuple(max_ape_source_hashes.values()) + tuple(
        max_ape_capture[name] for name in (
            "raw_sha256", "stderr_sha256", "npu_smi_before_sha256",
            "npu_smi_after_sha256", "source_snapshot_sha256",
            "binary_sha256"))
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in max_ape_evidence_hashes):
        raise ValueError(f"{owner} MaxAPE descriptor evidence SHA256 is invalid")
    for path_name in ("raw_path", "stderr_path"):
        evidence_path = max_ape_capture[path_name]
        if (not isinstance(evidence_path, str)
                or not evidence_path.startswith(
                    "profiler_output/microbench/ascend_910b/")
                or ".." in evidence_path.split("/")):
            raise ValueError(
                f"{owner} MaxAPE descriptor path is outside its owner")
    supplemental_fields = {
        "result_schema_version", "baseline_commit", "snapshot_path",
        "source_snapshot_marker_sha256", "host", "container",
        "physical_device_id", "logical_device_id", "owner", "runner_pid",
        "command", "result_path", "result_sha256", "cann_version",
        "dtype", "operator", "mode", "source_shape", "kernel_family",
        "kernel_stub", "block_dim", "device_kernel_submitted",
        "duration_consumed", "observation",
    }
    supplemental = provenance["supplemental_dry_launch_observations"]
    if not isinstance(supplemental, list) or len(supplemental) != 1:
        raise ValueError(f"{owner} requires one supplemental dry launch")
    supplemental_observation = supplemental[0]
    require_exact_fields(
        f"{owner}.provenance.supplemental_dry_launch_observations[0]",
        supplemental_observation, supplemental_fields)
    supplemental_shape = supplemental_observation["source_shape"]
    require_exact_fields(
        f"{owner}.supplemental.source_shape", supplemental_shape,
        {"b", "m", "n", "k"})
    if any(
            isinstance(supplemental_shape[name], bool)
            or not isinstance(supplemental_shape[name], int)
            or supplemental_shape[name] <= 0
            for name in ("b", "m", "n", "k")):
        raise ValueError(f"{owner} supplemental shape must be positive")
    observation = supplemental_observation["observation"]
    if any(
            not isinstance(supplemental_observation[name], str)
            or re.fullmatch(r"[0-9a-f]{40}", supplemental_observation[name])
                is None
            for name in ("baseline_commit",)):
        raise ValueError(f"{owner} supplemental commit is invalid")
    if any(
            not isinstance(supplemental_observation[name], str)
            or re.fullmatch(r"[0-9a-f]{64}", supplemental_observation[name])
                is None
            for name in ("source_snapshot_marker_sha256", "result_sha256")):
        raise ValueError(f"{owner} supplemental SHA256 is invalid")
    if any(
            not isinstance(observation[name], str)
            or re.fullmatch(r"[0-9a-f]{64}", observation[name]) is None
            for name in ("result_sha256", "tiling_payload_sha256")):
        raise ValueError(f"{owner} supplemental observation SHA256 is invalid")
    if (supplemental_observation["result_schema_version"] != 3
            or supplemental_observation["cann_version"] != "8.5.1"
            or supplemental_observation["dtype"] != "bf16"
            or supplemental_observation["operator"] != "aclnnMatmul"
            or supplemental_observation["mode"] != "single_shape"
            or supplemental_observation["kernel_family"] != "matmul_v2"
            or supplemental_observation["kernel_stub"]
                != ("MatMulV2_ND_ND_FP16_FP16_false_true_all_"
                    f"{observation['kernel_tiling_key']}")
            or supplemental_observation["block_dim"]
                != observation["active_aic_groups"]
            or supplemental_observation["device_kernel_submitted"] is not False
            or supplemental_observation["duration_consumed"] is not False
            or supplemental_observation["result_path"]
                != observation["result_path"]
            or supplemental_observation["result_sha256"]
                != observation["result_sha256"]
            or supplemental_shape["b"] * supplemental_shape["m"]
                > observation["tile_m"]):
        raise ValueError(f"{owner} supplemental dry-launch contract changed")
    if any(
            not isinstance(supplemental_observation[name], str)
            or not supplemental_observation[name]
            for name in ("snapshot_path", "host", "container", "owner",
                         "command", "result_path")):
        raise ValueError(f"{owner} supplemental execution provenance is invalid")
    if (not supplemental_observation["result_path"].startswith(
            "profiler_output/microbench/ascend_910b/")
            or ".." in supplemental_observation["result_path"].split("/")
            or (not isinstance(supplemental_observation["host"], str)
                or not supplemental_observation["host"].strip())
            or (not isinstance(supplemental_observation["container"], str)
                or not supplemental_observation["container"].strip())
            or supplemental_observation["owner"] != "kernel_accuracy_gemm"):
        raise ValueError(f"{owner} supplemental execution identity is invalid")
    if any(
            isinstance(supplemental_observation[name], bool)
            or not isinstance(supplemental_observation[name], int)
            or supplemental_observation[name] < 0
            for name in ("physical_device_id", "logical_device_id")):
        raise ValueError(f"{owner} supplemental device IDs are invalid")
    if (isinstance(supplemental_observation["runner_pid"], bool)
            or not isinstance(supplemental_observation["runner_pid"], int)
            or supplemental_observation["runner_pid"] <= 0):
        raise ValueError(f"{owner} supplemental runner PID is invalid")
    record_l1_layout(
        f"{owner}.supplemental_dry_launch_observations[0].observation",
        supplemental_shape, observation)
    for signature in asymmetric_descriptor_signatures:
        if signature in services:
            del services[signature]
    if len(services) != 27:
        raise ValueError(
            f"{owner} must publish twenty-seven generic positive-penalty cells")
    return {
        "model": "cold_contiguous_envelope_plus_serial_stride_delta_v1",
        "descriptors": services,
        "l1_layouts": l1_layouts,
        "reference_path": str(AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE),
        "reference_sha256": AIC_MTE2_COLD_SMALL_M_STRIDE_REFERENCE_SHA256,
    }


def load_aic_mte2_operand_pair_overlap_domain(repository_root):
    """Load the frozen short-program A1/B1 overlap eligibility domain."""

    path = repository_root / AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 operand-pair overlap reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    require_exact_fields(
        "AIC MTE2 operand-pair overlap reference", value,
        {
            "schema_version", "hardware_id", "software_revision", "service",
            "production_or_evaluation_duration_consumed", "forward_change",
            "mechanism_validation", "provenance",
        })
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "aic_mte2_operand_pair_overlap_domain_expansion",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(
                f"AIC MTE2 operand-pair overlap reference {name} changed")
    forward = value["forward_change"]
    require_exact_fields(
        "AIC MTE2 operand-pair overlap forward change", forward,
        {
            "parameter_fit_from_this_validation", "duration_parameter_published",
            "overlap_active_aic_group_range", "overlap_pair_count_range",
            "interpretation",
        })
    if (forward["parameter_fit_from_this_validation"] is not False
            or forward["duration_parameter_published"] is not False):
        raise ValueError(
            "AIC MTE2 operand-pair overlap evidence cannot publish duration")
    active_groups = forward["overlap_active_aic_group_range"]
    pair_counts = forward["overlap_pair_count_range"]
    if active_groups != [22, 24] or pair_counts != [512, 4096]:
        raise ValueError("AIC MTE2 operand-pair overlap domain changed")
    mechanism = value["mechanism_validation"]
    if mechanism["mechanism_gate_passed"] is not True:
        raise ValueError("AIC MTE2 operand-pair overlap mechanism gate failed")
    expected_geometry = {
        "dtype": "bf16",
        "source_residency": "l2_resident",
        "a_rows": 128,
        "a_columns": 64,
        "b_rows": 64,
        "b_columns": 256,
        "drain_interval": 4,
        "iterations_per_worker": 512,
        "active_aic_groups": [22, 24],
    }
    if any(mechanism[name] != expected_value
           for name, expected_value in expected_geometry.items()):
        raise ValueError("AIC MTE2 operand-pair overlap geometry changed")
    return {
        **expected_geometry,
        "minimum_active_aic_groups": active_groups[0],
        "maximum_active_aic_groups": active_groups[1],
        "minimum_pairs_per_worker": pair_counts[0],
        "maximum_pairs_per_worker": pair_counts[1],
        "reference_path": str(AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE),
        "reference_sha256": AIC_MTE2_OPERAND_PAIR_OVERLAP_REFERENCE_SHA256,
        "production_or_evaluation_duration_consumed": False,
    }


def load_aic_mte2_operand_pair_finite_overlap_domain(repository_root):
    """Load the frozen finite resident A1/B1 overlap eligibility domain."""

    path = repository_root / AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 finite operand-pair reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 finite operand-pair overlap reference"
    require_exact_fields(
        owner, value,
        {
            "schema_version", "hardware_id", "software_revision", "service",
            "production_or_evaluation_duration_consumed",
            "parameter_fit_from_production_duration", "mechanism_publishable",
            "forward_eligible", "forward_eligibility_constraint",
            "forward_change", "publication_scope", "measurement", "provenance",
        })
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "aic_mte2_operand_pair_finite_overlap_domain",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner} {name} changed")
    forward = value["forward_change"]
    require_exact_fields(
        f"{owner}.forward_change", forward,
        {
            "parameter_fit_from_this_validation", "duration_parameter_published",
            "overlap_pair_count_range", "interpretation",
        })
    if (forward["parameter_fit_from_this_validation"] is not False
            or forward["duration_parameter_published"] is not False
            or forward["overlap_pair_count_range"] != [16, 512]):
        raise ValueError(f"{owner} forward eligibility changed")
    scope = value["publication_scope"]
    expected_scope = {
        "a_columns": 64,
        "a_rows": 128,
        "active_aic_groups": 24,
        "b_columns": 256,
        "b_rows": 64,
        "drain_interval_pairs": 4,
        "dtype": "bf16",
        "maximum_pairs_per_worker": 512,
        "minimum_pairs_per_worker": 16,
        "source_residency": "l2_resident",
    }
    require_exact_fields(
        owner + ".publication_scope", scope, set(expected_scope))
    if scope != expected_scope:
        raise ValueError(f"{owner} publication scope changed")
    measurement = value["measurement"]
    if (measurement["mechanism_gate_passed"] is not True
            or measurement["stability_gate"] is not True
            or measurement["validation_gate"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    return {
        "dtype": scope["dtype"],
        "source_residency": scope["source_residency"],
        "a_rows": scope["a_rows"],
        "a_columns": scope["a_columns"],
        "b_rows": scope["b_rows"],
        "b_columns": scope["b_columns"],
        "drain_interval": scope["drain_interval_pairs"],
        "active_aic_groups": scope["active_aic_groups"],
        "minimum_pairs_per_worker": scope["minimum_pairs_per_worker"],
        "maximum_pairs_per_worker": scope["maximum_pairs_per_worker"],
        "reference_path": str(
            AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE),
        "reference_sha256": (
            AIC_MTE2_OPERAND_PAIR_FINITE_OVERLAP_REFERENCE_SHA256),
        "production_or_evaluation_duration_consumed": False,
    }


def load_aic_mte2_mixed_ring_long_overlap_domain(repository_root):
    """Load the frozen mixed-residency long overlap eligibility domain."""

    path = repository_root / AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE_SHA256):
        raise ValueError(
            "AIC MTE2 mixed-ring overlap reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 mixed-ring overlap reference"
    require_exact_fields(
        owner, value,
        {
            "schema_version", "hardware_id", "software_revision", "service",
            "production_or_evaluation_duration_consumed",
            "parameter_fit_from_production_duration", "mechanism_publishable",
            "forward_eligible", "forward_eligibility_constraint",
            "forward_change", "publication_scope", "measurement", "provenance",
            "published_service_ns_per_pair",
        })
    expected = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "aic_mte2_mixed_ring_long_overlap_domain",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": True,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner} {name} changed")
    forward = value["forward_change"]
    require_exact_fields(
        f"{owner}.forward_change", forward,
        {
            "parameter_fit_from_this_validation", "duration_parameter_published",
            "overlap_pair_count_range", "published_service_pair_count_range",
            "interpretation",
        })
    if (forward["parameter_fit_from_this_validation"] is not False
            or forward["duration_parameter_published"] is not True
            or forward["overlap_pair_count_range"] != [512, 12288]
            or forward["published_service_pair_count_range"] != [512, 2048]):
        raise ValueError(f"{owner} forward eligibility changed")
    scope = value["publication_scope"]
    expected_scope = {
        "a_columns": 64,
        "a_rows": 128,
        "active_aic_groups": 24,
        "b_columns": 256,
        "b_rows": 64,
        "drain_interval_pairs": 4,
        "dtype": "bf16",
        "maximum_pairs_per_worker": 12288,
        "minimum_pairs_per_worker": 512,
        "source_residency_pairs": [
            ["resident", "resident"],
            ["resident", "cold"],
            ["cold", "resident"],
            ["cold", "cold"],
        ],
        "cold_source_ring_bytes_per_operand": 536870912,
    }
    require_exact_fields(
        owner + ".publication_scope", scope, set(expected_scope))
    if scope != expected_scope:
        raise ValueError(f"{owner} publication scope changed")
    measurement = value["measurement"]
    if (measurement["mechanism_gate_passed"] is not True
            or measurement["stability_gate"] is not True
            or measurement["validation_gate"] is not True):
        raise ValueError(f"{owner} mechanism gate failed")
    residency_labels = {
        "resident_a_resident_b", "resident_a_cold_b",
        "cold_a_resident_b", "cold_a_cold_b",
    }
    published_service_ns = value["published_service_ns_per_pair"]
    require_exact_fields(
        owner + ".published_service_ns_per_pair",
        published_service_ns, residency_labels)
    for label, service_ns in published_service_ns.items():
        if (isinstance(service_ns, bool)
                or not isinstance(service_ns, (int, float))
                or not math.isfinite(service_ns)
                or service_ns <= 0):
            raise ValueError(
                f"{owner}.published_service_ns_per_pair.{label} must be "
                "finite and positive")
    return {
        "dtype": scope["dtype"],
        "source_residency_pairs": scope["source_residency_pairs"],
        "a_rows": scope["a_rows"],
        "a_columns": scope["a_columns"],
        "b_rows": scope["b_rows"],
        "b_columns": scope["b_columns"],
        "drain_interval": scope["drain_interval_pairs"],
        "active_aic_groups": scope["active_aic_groups"],
        "cold_source_ring_bytes_per_operand": scope[
            "cold_source_ring_bytes_per_operand"],
        "minimum_pairs_per_worker": scope["minimum_pairs_per_worker"],
        "maximum_pairs_per_worker": scope["maximum_pairs_per_worker"],
        "minimum_service_pairs_per_worker": forward[
            "published_service_pair_count_range"][0],
        "maximum_service_pairs_per_worker": forward[
            "published_service_pair_count_range"][1],
        "published_service_ns_per_pair": dict(published_service_ns),
        "reference_path": str(AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE),
        "reference_sha256": (
            AIC_MTE2_MIXED_RING_LONG_OVERLAP_REFERENCE_SHA256),
        "production_or_evaluation_duration_consumed": False,
    }


def load_aic_mte2_paged_gqa128_stride_service(repository_root):
    """Load the exact paged-GQA D128 GM-to-L1 descriptor service."""

    path = repository_root / AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE_SHA256):
        raise ValueError("AIC MTE2 paged-GQA stride reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 paged-GQA stride reference"
    require_exact_fields(
        owner, value,
        {
            "schema_version", "hardware_id", "software_revision",
            "mechanism", "duration_unit",
            "production_or_evaluation_duration_consumed",
            "measurement_scope", "parameters_ns", "stability_gate",
            "validation_controls", "provenance",
        })
    expected = {
        "schema_version": "midlayer.ascend.aic_mte2_paged_gqa128_stride.v1",
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_gm_to_l1_nd2nz_strided_completion",
        "duration_unit": "ns_per_descriptor_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner} {name} changed")
    scope = value["measurement_scope"]
    require_exact_fields(
        f"{owner}.measurement_scope", scope,
        {
            "dtype", "rows", "columns", "source_row_stride_elements",
            "active_aic_groups", "modes", "footprint_bytes", "iterations",
            "trials", "order",
        })
    expected_scope = {
        "dtype": "bf16",
        "rows": 128,
        "columns": 128,
        "source_row_stride_elements": 1024,
        "active_aic_groups": [8, 16, 24],
        "modes": {"serial_completion": 0, "two_buffer_stream": 1},
        "footprint_bytes": 536870912,
        "iterations": 20000,
        "trials": 4,
        "order": "alternating_forward_reverse",
    }
    if scope != expected_scope:
        raise ValueError(f"{owner} measurement scope changed")
    parameters = value["parameters_ns"]
    require_exact_fields(
        f"{owner}.parameters_ns", parameters,
        {
            "serial_duration_by_active_aic_group",
            "stream_duration_by_active_aic_group",
        })
    normalized = {}
    for name in (
            "serial_duration_by_active_aic_group",
            "stream_duration_by_active_aic_group"):
        durations = parameters[name]
        if (not isinstance(durations, list)
                or len(durations) != len(scope["active_aic_groups"])
                or any(
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(float(duration))
                    or float(duration) <= 0
                    for duration in durations)):
            raise ValueError(f"{owner} {name} must contain positive durations")
        normalized[name] = tuple(float(duration) for duration in durations)
    stability = value["stability_gate"]
    require_exact_fields(
        f"{owner}.stability_gate", stability,
        {
            "maximum_cross_trial_spread_pct",
            "maximum_position_balance_error_pct", "limit_pct", "passed",
        })
    for name in (
            "maximum_cross_trial_spread_pct",
            "maximum_position_balance_error_pct", "limit_pct"):
        metric = stability[name]
        if (isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
                or float(metric) < 0):
            raise ValueError(f"{owner} {name} must be finite and non-negative")
    if (stability["passed"] is not True
            or stability["maximum_cross_trial_spread_pct"]
            > stability["limit_pct"]
            or stability["maximum_position_balance_error_pct"]
            > stability["limit_pct"]):
        raise ValueError(f"{owner} stability gate failed")
    controls = value["validation_controls"]
    if not isinstance(controls, list) or len(controls) != 2:
        raise ValueError(f"{owner} must contain two validation controls")
    for index, (control, expected_stride) in enumerate(
            zip(controls, (128, 4096))):
        require_exact_fields(
            f"{owner}.validation_controls[{index}]", control,
            {
                "source_row_stride_elements", "serial_duration_ns",
                "stream_duration_ns",
            })
        if control["source_row_stride_elements"] != expected_stride:
            raise ValueError(f"{owner} validation control stride changed")
        for name in ("serial_duration_ns", "stream_duration_ns"):
            durations = control[name]
            if (not isinstance(durations, list)
                    or len(durations) != len(scope["active_aic_groups"])
                    or any(
                        isinstance(duration, bool)
                        or not isinstance(duration, (int, float))
                        or not math.isfinite(float(duration))
                        or float(duration) <= 0
                        for duration in durations)):
                raise ValueError(
                    f"{owner} validation control {name} is invalid")
    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {
            "measurement_baseline_commit", "source_snapshot_sha256",
            "binary_sha256", "raw_path", "raw_sha256",
            "physical_device_id", "container",
        })
    if (isinstance(provenance["physical_device_id"], bool)
            or not isinstance(provenance["physical_device_id"], int)
            or provenance["physical_device_id"] < 0
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())):
        raise ValueError(f"{owner} execution provenance changed")
    commit = provenance["measurement_baseline_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError(f"{owner} baseline commit must be full Git hex")
    for name in ("source_snapshot_sha256", "binary_sha256", "raw_sha256"):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner} {name} must be SHA256 hex")
    raw_path = provenance["raw_path"]
    if (not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")):
        raise ValueError(f"{owner} raw_path is outside its evidence owner")
    return {
        "geometry": (
            scope["rows"], scope["columns"],
            scope["source_row_stride_elements"]),
        "active_aic_groups": tuple(scope["active_aic_groups"]),
        "serial_duration_ns": normalized[
            "serial_duration_by_active_aic_group"],
        "stream_duration_ns": normalized[
            "stream_duration_by_active_aic_group"],
        "evidence_ref": str(AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE),
        "evidence_sha256": AIC_MTE2_PAGED_GQA128_STRIDE_REFERENCE_SHA256,
        "production_or_evaluation_duration_consumed": False,
    }


def load_split_fuse_mte2_short_tail_service(repository_root):
    """Load finite paged-KV ingress times from the independent 24-AIC probe."""

    path = repository_root / SPLIT_FUSE_MTE2_SHORT_TAIL_REFERENCE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
            SPLIT_FUSE_MTE2_SHORT_TAIL_REFERENCE_SHA256):
        raise ValueError("SplitFuse short-tail reference SHA256 mismatch")
    value = json.loads(payload)
    owner = "SplitFuse short-tail reference"
    require_exact_fields(owner, value, {
        "schema_version", "hardware_id", "software_revision", "mechanism",
        "production_or_evaluation_duration_consumed", "scope",
        "services_ns", "provenance",
    })
    expected = {
        "schema_version": "midlayer.ascend.split_fuse_mte2_short_tail.v1",
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "bf16_paged_kv_128x128_stride1024_finite_ingress",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner}.{name} changed")
    scope = value["scope"]
    if scope != {
            "active_aic_groups": 24,
            "rows": 128,
            "columns": 128,
            "source_row_stride_elements": 1024,
            "descriptor_counts": [3, 4, 5, 7],
            "issue_modes": ["drained", "two_slot_overlap"],
            "timed_launches_per_process": 256,
            "process_count": 3}:
        raise ValueError(f"{owner}.scope changed")
    services = value["services_ns"]
    if not isinstance(services, dict) or set(services) != {"3", "4", "5", "7"}:
        raise ValueError(f"{owner}.services_ns has the wrong counts")
    normalized = {}
    for count in (3, 4, 5, 7):
        arms = services[str(count)]
        require_exact_fields(f"{owner}.services_ns[{count}]", arms,
                             {"drained", "two_slot_overlap"})
        normalized_arms = {}
        for mode, values in arms.items():
            require_exact_fields(f"{owner}.{count}.{mode}", values,
                                 {"first_touch_ns", "same_address_replay_ns"})
            cold = values["first_touch_ns"]
            warm = values["same_address_replay_ns"]
            if (any(isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item)) or item <= 0
                    for item in (cold, warm)) or cold <= warm):
                raise ValueError(f"{owner}.{count}.{mode} has invalid service")
            normalized_arms[mode] = (float(warm), float(cold))
        if (normalized_arms["two_slot_overlap"][1]
                > 1.05 * normalized_arms["drained"][1]):
            raise ValueError(f"{owner}.{count} overlap gate failed")
        normalized[count] = normalized_arms
    provenance = value["provenance"]
    require_exact_fields(f"{owner}.provenance", provenance,
                         {"raw_path", "raw_sha256", "binary_sha256",
                          "source_sha256"})
    raw_path = provenance["raw_path"]
    if (not isinstance(raw_path, str)
            or not raw_path.startswith(
                "profiler_output/microbench/ascend_910b/")
            or ".." in raw_path.split("/")):
        raise ValueError(f"{owner}.raw_path is invalid")
    digests = (provenance["raw_sha256"], provenance["binary_sha256"],
               *provenance["source_sha256"].values())
    if any(not isinstance(digest, str) or len(digest) != 64
           or any(character not in "0123456789abcdef" for character in digest)
           for digest in digests):
        raise ValueError(f"{owner}.provenance has invalid SHA256")
    return normalized


def load_aic_mte2_fia_mla_strided_service(repository_root):
    """Load exact resident/cold CANN MLA GM-to-L1 descriptor service."""

    path = repository_root / AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != (
            AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE_SHA256):
        raise ValueError("AIC MTE2 FIA MLA strided reference SHA256 mismatch")
    value = json.loads(payload_bytes)
    owner = "AIC MTE2 FIA MLA strided reference"
    require_exact_fields(
        owner, value,
        {
            "schema_version", "hardware_id", "software_revision",
            "mechanism", "duration_unit",
            "production_or_evaluation_duration_consumed",
            "source_semantics", "measurement_scope", "parameters_ns",
            "stability_gate", "provenance",
        })
    expected = {
        "schema_version": (
            "midlayer.ascend.aic_mte2_fia_mla_strided_service.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "aic_gm_to_l1_nd2nz_mla_strided_resident_cold_completion"),
        "duration_unit": "ns_per_descriptor_per_aic",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{owner} {name} changed")

    source_semantics = value["source_semantics"]
    descriptor_names = (
        "q_main", "q_rope", "k_main", "k_rope", "probability", "value")
    require_exact_fields(
        f"{owner}.source_semantics", source_semantics,
        {
            "production_kernel_name", "production_binary_sha256",
            "production_metadata_sha256", *descriptor_names,
        })
    if source_semantics["production_kernel_name"] != (
            "FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da"):
        raise ValueError(f"{owner} production kernel changed")
    for name in ("production_binary_sha256", "production_metadata_sha256"):
        digest = source_semantics[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner} {name} must be SHA256 hex")
    expected_source_descriptors = {
        "q_main": (256, 512),
        "q_rope": (64, 64),
        "k_main": (256, 512),
        "k_rope": (32, 64),
        "probability": (256, 512),
        "value": (128, 512),
    }
    for name, expected_descriptor in expected_source_descriptors.items():
        descriptor = source_semantics[name]
        require_exact_fields(
            f"{owner}.source_semantics.{name}", descriptor,
            {"columns", "source_row_stride_elements"})
        if (descriptor["columns"], descriptor[
                "source_row_stride_elements"]) != expected_descriptor:
            raise ValueError(f"{owner} {name} source geometry changed")

    scope = value["measurement_scope"]
    require_exact_fields(
        f"{owner}.measurement_scope", scope,
        {
            "dtype", "active_aic_groups", "modes",
            "resident_footprint_bytes", "cold_footprint_bytes",
            "published_l2_capacity_bytes",
            "minimum_cold_long_interval_unique_bytes", "resident_trials",
            "cold_trials", "statistic",
        })
    expected_scope = {
        "dtype": "bf16",
        "active_aic_groups": [1, 4, 8, 16, 22, 24],
        "modes": {"serial_completion": 0, "two_buffer_stream": 1},
        "resident_footprint_bytes": 67108864,
        "cold_footprint_bytes": 536870912,
        "published_l2_capacity_bytes": 201326592,
        "minimum_cold_long_interval_unique_bytes": 266338304,
        "resident_trials": 3,
        "cold_trials": 7,
        "statistic": "median",
    }
    if scope != expected_scope:
        raise ValueError(f"{owner} measurement scope changed")
    if (scope["minimum_cold_long_interval_unique_bytes"]
            <= scope["published_l2_capacity_bytes"]):
        raise ValueError(f"{owner} cold working set does not exceed L2")

    parameter_fields = {
        "rows", "columns", "source_row_stride_elements",
        "resident_serial_by_active_aic_group",
        "resident_stream_by_active_aic_group",
        "cold_serial_by_active_aic_group",
        "cold_stream_by_active_aic_group",
    }
    expected_geometries = {
        (64, 256, 512), (128, 256, 512),
        (128, 32, 64), (128, 128, 512),
    }
    raw_parameters = value["parameters_ns"]
    if not isinstance(raw_parameters, list) or len(raw_parameters) != 4:
        raise ValueError(f"{owner} must contain four exact geometries")
    geometries = {}
    for index, parameter in enumerate(raw_parameters):
        require_exact_fields(
            f"{owner}.parameters_ns[{index}]", parameter, parameter_fields)
        geometry = tuple(
            parameter[name]
            for name in ("rows", "columns", "source_row_stride_elements"))
        if (any(isinstance(item, bool) or not isinstance(item, int)
                or item <= 0 for item in geometry)
                or geometry not in expected_geometries
                or geometry in geometries):
            raise ValueError(f"{owner} contains an invalid geometry")
        services = {}
        for name in parameter_fields - {
                "rows", "columns", "source_row_stride_elements"}:
            durations = parameter[name]
            if (not isinstance(durations, list)
                    or len(durations) != len(scope["active_aic_groups"])
                    or any(
                        isinstance(duration, bool)
                        or not isinstance(duration, (int, float))
                        or not math.isfinite(float(duration))
                        or float(duration) <= 0.0
                        for duration in durations)):
                raise ValueError(f"{owner} {name} is invalid")
            services[name] = tuple(float(duration) for duration in durations)
        for mode in ("serial", "stream"):
            resident = services[f"resident_{mode}_by_active_aic_group"]
            cold = services[f"cold_{mode}_by_active_aic_group"]
            if any(cold_ns < resident_ns
                   for resident_ns, cold_ns in zip(resident, cold)):
                raise ValueError(
                    f"{owner} cold {mode} service precedes resident service")
        geometries[geometry] = services
    if set(geometries) != expected_geometries:
        raise ValueError(f"{owner} exact geometry domain changed")

    stability = value["stability_gate"]
    require_exact_fields(
        f"{owner}.stability_gate", stability,
        {
            "metric", "resident_value_pct", "cold_value_pct", "limit_pct",
            "passed",
        })
    if stability["metric"] != (
            "maximum_cross_trial_median_absolute_deviation_pct"):
        raise ValueError(f"{owner} stability metric changed")
    for name in ("resident_value_pct", "cold_value_pct", "limit_pct"):
        metric = stability[name]
        if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric)) or float(metric) < 0.0):
            raise ValueError(f"{owner} {name} must be non-negative")
    if (stability["passed"] is not True
            or stability["resident_value_pct"] > stability["limit_pct"]
            or stability["cold_value_pct"] > stability["limit_pct"]):
        raise ValueError(f"{owner} stability gate failed")

    provenance = value["provenance"]
    require_exact_fields(
        f"{owner}.provenance", provenance,
        {
            "measurement_baseline_commit", "source_snapshot_sha256",
            "binary_sha256", "physical_device_id", "container",
            "resident_raw", "cold_raw",
        })
    if (isinstance(provenance["physical_device_id"], bool)
            or not isinstance(provenance["physical_device_id"], int)
            or provenance["physical_device_id"] < 0
            or (not isinstance(provenance["container"], str)
                or not provenance["container"].strip())):
        raise ValueError(f"{owner} execution provenance changed")
    for name, length in (
            ("measurement_baseline_commit", 40),
            ("source_snapshot_sha256", 64), ("binary_sha256", 64)):
        digest = provenance[name]
        if (not isinstance(digest, str) or len(digest) != length
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise ValueError(f"{owner} {name} must be lowercase hex")
    for name, expected_count in (("resident_raw", 3), ("cold_raw", 7)):
        raw_files = provenance[name]
        if not isinstance(raw_files, list) or len(raw_files) != expected_count:
            raise ValueError(f"{owner} {name} count changed")
        for index, raw_file in enumerate(raw_files):
            require_exact_fields(
                f"{owner}.provenance.{name}[{index}]", raw_file,
                {"path", "sha256"})
            raw_path = raw_file["path"]
            digest = raw_file["sha256"]
            if (not isinstance(raw_path, str)
                    or not raw_path.startswith(
                        "profiler_output/microbench/ascend_910b/")
                    or ".." in raw_path.split("/")):
                raise ValueError(f"{owner} raw path is outside its owner")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(f"{owner} raw SHA256 is invalid")
    return {
        "active_aic_groups": tuple(scope["active_aic_groups"]),
        "geometries": geometries,
        "evidence_ref": str(AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE),
        "evidence_sha256": (
            AIC_MTE2_FIA_MLA_STRIDED_SERVICE_REFERENCE_SHA256),
        "production_or_evaluation_duration_consumed": False,
    }
