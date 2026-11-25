"""Structured contracts for the unified analytical kernel model.

The contracts contain only workload semantics, hardware mechanisms, and model
diagnostics.  They deliberately contain no measured production duration or
kernel-name latency correction.
"""

from dataclasses import dataclass, field, replace
from functools import cached_property
import math
from types import MappingProxyType
from typing import Mapping, Optional, Tuple, Union


WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR = (
    "warm_default_cache_memory_dependency_task_v1")
CANN_AIV_TILING_MATERIALIZATION_PROLOGUE = (
    "cann_aiv_tiling_materialization_v1")
CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE = (
    "cann_aiv_tiling_materialization_48b_v1")
CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE = (
    "cann_aiv_tiling_materialization_56b_v1")
CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE = (
    "cann_aiv_tiling_materialization_80b_v1")
CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE = (
    "cann_aiv_tiling_materialization_192b_v1")
AIV_COMPLETE_TASK_FLOOR_ONLY = "aiv_complete_task_floor_only_v1"
AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM = (
    "aiv_empty_task_shell_plus_source_program_v1")
AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR = (
    "aiv_empty_task_shell_competing_source_floor_v1")
AIV_SOURCE_PIPELINE_LAUNCH_ONLY = "aiv_source_pipeline_launch_only_v1"
AIV_WORKER_START_FILL = "aiv_worker_start_fill_v1"
AIV_SOURCE_ENTRY_FILL = "aiv_source_entry_fill_v1"
GPU_SOURCE_PIPELINE_OUTER_ENVELOPE = (
    "gpu_source_pipeline_outer_envelope_v1")
MEMORY_DEPENDENCY_TASK_INCREMENT_SERVICE = (
    "task:memory_dependency_increment")
GLOBAL_STORE_COMPLETION_SERVICE = "memory_completion:global_write"
SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER = (
    "source_bound_memory_dependency_task")
SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER = (
    "source_bound_l2_read_transactions_per_warp")
SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER = (
    "source_bound_tensor_iterator_copy_task")
TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR = (
    "torch_tensor_iterator_strided_copy_bf16_rank4_v1")
TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR = (
    "torch_tensor_iterator_strided_copy_fp32_rank3_v1")
SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER = (
    "source_bound_l2_read_warps_per_cta")
SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER = (
    "source_bound_gpu_norm_cta_pipeline")
SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER = (
    "source_rmsnorm_kernel_vllm_0_8_5_maca_3_7_2")
SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER = (
    "source_bound_silu_nonrecurrent_cta_pipeline")
SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER = (
    "source_bound_warp_slot2_instructions")
SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER = (
    "source_vector_elements_per_thread")
SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER = (
    "source_bf16_conversion_operations")
SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER = (
    "source_cann_bf16_elementwise_dag")
SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER = (
    "source_cann_rope_index_select_count")
SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER = (
    "source_cann_rope_gather_v3_program")
SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER = (
    "source_vllm_ascend_split_qkv_rmsnorm_rope_program")
SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER = (
    "source_pytorch_bf16_binary_tensor_iterator_task")
PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS = (
    "pytorch_bf16_binary_tensor_iterator")
SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER = (
    "source_pytorch_cpp_extension_independent_cta_memory_task")
SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER = (
    "source_metax_bf16_grouped_strided_copy")
PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS = (
    "pytorch_cpp_extension_independent_cta_memory_task")
SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER = (
    "source_fused_moe_w2_routed_weight_program")
SOURCE_SCALAR_KV_LOOP_PARAMETER = "source_scalar_kv_loop"
SOURCE_VECTORIZED_KV_COPY_PARAMETER = "source_vectorized_kv_copy"
SOURCE_QK_ROPE_LOOP_PARAMETER = "source_qk_rope_loop"
SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER = (
    "source_metax_rope_vllm_0_8_5_abi10_program")
SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER = (
    "source_metax_rope_vllm_0_9_1_abi11_program")
SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER = (
    "source_vllm_0_24_0_act_and_mul_program")
SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER = (
    "source_metax_rope_vllm_0_24_0_abi13_program")


TYPED_COMPUTE_SERVICES_BY_ROLE = MappingProxyType({
    "compute_vector": frozenset({
        "generic", "exp", "div",
        "cast_bf16_to_fp32_serialized", "cast_fp32_to_bf16_serialized",
        "muls_serialized", "adds_serialized", "mul_serialized",
        # CANN 8.5.1 KernelRmsNormMergeN (RmsNorm tiling key 2) row-group
        # primitives, measured as whole source calls by the independent AIV
        # opcode campaign.
        "merge_n_reduce_sum_multi_n",
        "merge_n_broadcast_gamma",
        "merge_n_broadcast_rstd",
        "merge_n_performance_rstd_brcb",
        "merge_n_performance_mul_row_rstd",
        "merge_n_performance_mul_row_gamma",
        "integer_div_u32_dependency", "integer_rem_u32_dependency",
        "integer_div_u64_dependency", "integer_rem_u64_dependency",
        "bf16_roundtrip_pair_dependency",
        "address_multiply_add_u64_dependency",
        "vector_alu_issue_slot_instruction",
        "rmsnorm_input_square_first_1024",
        "rmsnorm_input_square_resident_1024",
        "rmsnorm_input_square_queued_1024",
        "rmsnorm_output_scale_first_1024",
        "rmsnorm_output_scale_resident_1024",
        "rmsnorm_output_scale_queued_1024",
        "rmsnorm_input_square_first_128",
        "rmsnorm_input_square_first_128_active2",
        "rmsnorm_input_square_first_128_active4",
        "rmsnorm_input_square_first_128_active8",
        "rmsnorm_input_square_first_128_active16",
        "rmsnorm_input_square_first_128_active32",
        "rmsnorm_input_square_first_128_active64",
        "rmsnorm_input_square_first_128_active104",
        "rmsnorm_input_square_resident_128",
        "rmsnorm_input_square_queued_128",
        "rmsnorm_output_scale_first_128",
        "rmsnorm_output_scale_first_128_active2",
        "rmsnorm_output_scale_first_128_active4",
        "rmsnorm_output_scale_first_128_active8",
        "rmsnorm_output_scale_first_128_active16",
        "rmsnorm_output_scale_first_128_active32",
        "rmsnorm_output_scale_first_128_active64",
        "rmsnorm_output_scale_first_128_active104",
        "rmsnorm_output_scale_resident_128",
        "rmsnorm_output_scale_queued_128",
        "rmsnorm_phase_chain_excess_first_1024",
        "rmsnorm_phase_chain_excess_first_1024_active2",
        "rmsnorm_phase_chain_excess_first_1024_active4",
        "rmsnorm_phase_chain_excess_first_1024_active8",
        "rmsnorm_phase_chain_complete_first_1024",
        "rmsnorm_phase_entry_first_1024",
        "rmsnorm_phase_entry_first_1024_active2",
        "rmsnorm_phase_entry_first_1024_active4",
        "rmsnorm_phase_entry_first_1024_active8",
        "rmsnorm_phase_entry_first_1024_active16",
        "rmsnorm_phase_entry_first_1024_active64",
        "add_rmsnorm_phase_chain_excess_first_1024_active2",
        "add_rmsnorm_phase_chain_excess_first_1024_active4",
        "add_rmsnorm_phase_chain_excess_first_1024_active8",
        "add_rmsnorm_phase_chain_excess_first_1024_active16",
        "add_rmsnorm_phase_chain_excess_first_1024_active32",
        "add_rmsnorm_phase_chain_complete_first_1024",
        "add_rmsnorm_scalar_input_add_square_first_256",
        "add_rmsnorm_scalar_input_add_square_resident_256",
        "add_rmsnorm_input_add_square_first_1024",
        "add_rmsnorm_output_scale_first_1024",
        "add_rmsnorm_grid64_fixed_input_loop_depth4_1024",
        "add_rmsnorm_grid64_fixed_input_loop_depth5_1024",
        "add_rmsnorm_grid64_fixed_input_loop_depth8_1024",
        "add_rmsnorm_grid64_fixed_input_loop_depth16_1024",
        "add_rmsnorm_grid64_fixed_input_loop_depth32_1024",
        "add_rmsnorm_h4096_grid64_cold_source_chain_entry",
        "silu_scalar_round",
        "silu_nonrecurrent_first_entry_block1024",
        "silu_nonrecurrent_first_round_block1024",
        "silu_nonrecurrent_resident_entry_block1024",
        "silu_nonrecurrent_resident_round_block1024",
        "silu_nonrecurrent_queued_entry_block1024",
        "silu_nonrecurrent_queued_round_block1024",
        "bf16_add_square_first_256",
        "bf16_add_square_resident_256",
        "bf16_add_square_queued_256",
        "bf16_add_square_first_512",
        "bf16_add_square_resident_512",
        "bf16_add_square_queued_512",
        "bf16_add_square_first_1024",
        "bf16_add_square_resident_1024",
        "bf16_add_square_queued_1024",
        "bf16_add_square_first_256_block256",
        "bf16_add_square_resident_256_block256",
        "bf16_add_square_first_64_block256",
        "bf16_add_square_resident_64_block256",
        "bf16_add_square_first_128_block256",
        "bf16_add_square_resident_128_block256",
        "bf16_add_square_first_256_block1024",
        "bf16_add_square_first_256_active4_block1024",
        "bf16_add_square_first_256_active16_block1024",
        "bf16_add_square_first_256_active108_block1024",
        "bf16_add_square_first_256_active132_block1024",
        "bf16_add_square_first_320_block1024",
        "bf16_add_square_first_320_active4_block1024",
        "bf16_add_square_first_320_active16_block1024",
        "bf16_add_square_first_320_active108_block1024",
        "bf16_add_square_first_320_active132_block1024",
        "bf16_add_square_first_512_block1024",
        "bf16_add_square_first_512_active4_block1024",
        "bf16_add_square_first_512_active16_block1024",
        "bf16_add_square_first_512_active108_block1024",
        "bf16_add_square_first_512_active132_block1024",
        "bf16_add_square_first_640_block1024",
        "bf16_add_square_first_640_active4_block1024",
        "bf16_add_square_first_640_active8_block1024",
        "bf16_add_square_first_640_active16_block1024",
        "bf16_add_square_first_640_active108_block1024",
        "bf16_add_square_first_640_active132_block1024",
        "bf16_add_square_first_1024_block1024",
        "bf16_add_square_first_1024_active4_block1024",
        "bf16_add_square_first_1024_active8_block1024",
        "bf16_add_square_first_1024_active16_block1024",
        "bf16_add_square_first_1024_active108_block1024",
        "bf16_add_square_first_1024_active132_block1024",
        "bf16_scale_multiply_first_256",
        "bf16_scale_multiply_resident_256",
        "bf16_scale_multiply_queued_256",
        "bf16_scale_multiply_first_512",
        "bf16_scale_multiply_resident_512",
        "bf16_scale_multiply_queued_512",
        "bf16_scale_multiply_first_1024",
        "bf16_scale_multiply_resident_1024",
        "bf16_scale_multiply_queued_1024",
        "bf16_scale_multiply_first_256_block256",
        "bf16_scale_multiply_resident_256_block256",
        "bf16_scale_multiply_first_64_block256",
        "bf16_scale_multiply_resident_64_block256",
        "bf16_scale_multiply_first_128_block256",
        "bf16_scale_multiply_resident_128_block256",
        "bf16_scale_multiply_first_256_block1024",
        "bf16_scale_multiply_first_256_active4_block1024",
        "bf16_scale_multiply_first_256_active16_block1024",
        "bf16_scale_multiply_first_256_active108_block1024",
        "bf16_scale_multiply_first_256_active132_block1024",
        "bf16_scale_multiply_first_320_block1024",
        "bf16_scale_multiply_first_320_active4_block1024",
        "bf16_scale_multiply_first_320_active16_block1024",
        "bf16_scale_multiply_first_320_active108_block1024",
        "bf16_scale_multiply_first_320_active132_block1024",
        "bf16_scale_multiply_first_512_block1024",
        "bf16_scale_multiply_first_512_active4_block1024",
        "bf16_scale_multiply_first_512_active16_block1024",
        "bf16_scale_multiply_first_512_active108_block1024",
        "bf16_scale_multiply_first_512_active132_block1024",
        "bf16_scale_multiply_first_640_block1024",
        "bf16_scale_multiply_first_640_active4_block1024",
        "bf16_scale_multiply_first_640_active8_block1024",
        "bf16_scale_multiply_first_640_active16_block1024",
        "bf16_scale_multiply_first_640_active108_block1024",
        "bf16_scale_multiply_first_640_active132_block1024",
        "bf16_scale_multiply_first_1024_block1024",
        "bf16_scale_multiply_first_1024_active4_block1024",
        "bf16_scale_multiply_first_1024_active8_block1024",
        "bf16_scale_multiply_first_1024_active16_block1024",
        "bf16_scale_multiply_first_1024_active108_block1024",
        "bf16_scale_multiply_first_1024_active132_block1024",
        "rmsnorm_half_interval_reduce",
        "add_rmsnorm_reduce_sum_fp32",
        "notequal_int32_bool_serialized",
        "cast_bool_bf16_serialized",
    }),
    "compute_sfu": frozenset({
        "exp_dependency", "reciprocal_one_plus_dependency",
    }),
    "compute_alu": frozenset({
        "prmt_b32_bf16_expand",
        "moe_routing_global_atomic_e8",
        "moe_routing_global_atomic_e32",
        "moe_routing_global_atomic_e128",
    }),
    "compute_reduction": frozenset({
        "imcontinuous_vec4_reduce_topk2",
        "imcontinuous_vec4_reduce_topk4",
        "imcontinuous_vec4_reduce_topk8",
        "imcontinuous_vec4_reduce_topk10",
        "imcontinuous_vec4_reduce_topk16",
        "block_reduce_first_128",
        "block_reduce_first_128_active2",
        "block_reduce_first_128_active4",
        "block_reduce_first_128_active8",
        "block_reduce_first_128_active16",
        "block_reduce_first_128_active32",
        "block_reduce_first_128_active64",
        "block_reduce_first_128_active104",
        "block_reduce_resident_128",
        "block_reduce_queued_128",
        "block_reduce_first_256",
        "block_reduce_first_256_active2",
        "block_reduce_first_256_active4",
        "block_reduce_first_256_active8",
        "block_reduce_first_256_active16",
        "block_reduce_first_256_active32",
        "block_reduce_first_256_active64",
        "block_reduce_first_256_active104",
        "block_reduce_resident_256",
        "block_reduce_queued_256",
        "block_reduce_first_1024",
        "block_reduce_first_1024_active2",
        "block_reduce_first_1024_active4",
        "block_reduce_first_1024_active8",
        "block_reduce_first_1024_active16",
        "block_reduce_first_1024_active32",
        "block_reduce_first_1024_active64",
        "block_reduce_first_1024_active104",
        "block_reduce_resident_1024",
        "block_reduce_queued_1024",
        "cann_add_rmsnorm_split_d_block_reduce_full64",
        "cann_add_rmsnorm_split_d_block_reduce_tail8",
    }),
})
_TYPED_COMPUTE_SERVICES = frozenset().union(
    *TYPED_COMPUTE_SERVICES_BY_ROLE.values())
COMPUTE_EXACT_REQUEST_SERVICES_BY_ROLE = MappingProxyType({
    "compute_fma": frozenset({
        "aic_scalar_global_get_value_serialized",
        "cann_grouped_matmul_finite_group_scan_control",
        "cann_grouped_matmul_zero_expert_control",
    }),
    "compute_vector": frozenset({
        "local_scalar_fill_s_v",
        "local_ub_copy_contiguous",
        "local_ub_copy_stride_64b",
        "local_ub_copy_stride_128b",
        "aiv_elementwise_mul_prefix_46_workers",
        "aiv_elementwise_mul_suffix_46_workers",
        "aiv_elementwise_mul_prefix_47_workers",
        "aiv_elementwise_mul_suffix_47_workers",
        "norm_sqrt_duplicate_div_serialized",
        "moe_compute_expert_tokens_key1001_init_sync_phase",
        "rmsnorm_local_scalar_dependency",
        "scalar_global_get_value_serialized",
        "rmsnorm_block128_wave_lifetime_completion",
    }),
    "scalar_aiv": frozenset({
        "add_rmsnorm_local_scalar_handoff",
        "add_rmsnorm_split_d_local_scalar_handoff",
        "vector_to_mte2_fence",
        "aiv_four_queue_control_chain",
        "aiv_four_queue_three_tbuf_lifecycle",
        "aiv_queue_control_cycle",
        "aiv_scalar_u64_divrem_dependency",
        "aiv_swiglu_tile_address_control",
        "aiv_three_queue_control_chain",
        "aiv_three_queue_two_tbuf_lifecycle",
        "aiv_single_tbuf_lifecycle",
        "aiv_gather_v3_two_tque_lifecycle",
        "aiv_elementwise_tbuf_lifecycle_110080b",
    }),
})
_COMPUTE_EXACT_REQUEST_SERVICES = frozenset().union(
    *COMPUTE_EXACT_REQUEST_SERVICES_BY_ROLE.values())
MEMORY_EXACT_REQUEST_SERVICES_BY_ROLE = MappingProxyType({
    "dram": frozenset({
        "aic_mte2_nd2nz_33copy_large_stride_group_completion_24_workers",
        "aic_fixpipe_bf16_finite_stream_16x176_22_workers",
        "aic_fixpipe_bf16_finite_stream_32x256_16_workers",
        "aic_fixpipe_bf16_finite_stream_32x256_20_workers",
        "aic_fixpipe_bf16_finite_stream_32x256_22_workers",
        "aic_fixpipe_bf16_finite_stream_48x256_22_workers",
        "aic_fixpipe_bf16_finite_stream_64x256_20_workers",
        "aic_fixpipe_bf16_finite_stream_64x256_22_workers",
        "aic_fixpipe_bf16_finite_stream_128x256_24_workers",
    }),
    "mte2": frozenset({
        "add_rmsnorm_key31_residual_reload_completion_chain",
        "aiv_mte2_request_completion",
        "aiv_rope_mte2_to_vector_completion",
        "aiv_rope_mte2_to_vector_completion_32_workers",
        "aiv_rope_mte2_to_vector_completion_43_workers",
        "aiv_rope_mte2_to_vector_completion_47_workers",
        "aiv_rope_key24_mte2_steady_completion_47_workers_4096b",
        "aiv_rope_key22_first_q_slice_mte2_to_vector_completion",
        "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_4_workers",
        "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_8_workers",
        "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_16_workers",
        "aiv_datacopypad_mte2_to_vector_completion",
        "aiv_datacopypad_mte2_to_vector_completion_two_workers",
        "aiv_add_rmsnorm_row_group_input_pair_completion_32_workers",
        "aiv_add_rmsnorm_single_n_input_pair_completion_one_worker",
        "aiv_rmsnorm_key0_gamma_input_sequence_completion",
        "aiv_rmsnorm_key0_h2048_mte2_to_vector_completion_4_workers",
        "aiv_rmsnorm_key0_h2048_mte2_to_vector_completion_8_workers",
        "aiv_rmsnorm_key0_first_mte2_completion_16_workers",
        "aiv_datacopypad_mte2_to_vector_completion_32_workers",
        "aiv_datacopypad_mte2_to_vector_completion_43_workers",
        "aiv_datacopypad_mte2_to_vector_completion_47_workers",
        "aiv_rmsnorm_key0_steady_mte2_completion_32_workers",
        "aiv_rmsnorm_key0_steady_mte2_completion_43_workers",
        "aiv_rmsnorm_key0_steady_mte2_completion_47_workers",
        "aiv_swiglu_tque_mte2_to_vector_completion_24_workers",
        "aiv_swiglu_tque_mte2_to_vector_completion_40_workers",
        "aiv_swiglu_tque_mte2_to_vector_completion_48_workers",
        "aiv_swiglu_tque_input_pair_completion_24_workers_q2",
        "aiv_swiglu_tque_input_pair_completion_40_workers_q2",
        "aiv_swiglu_tque_input_pair_completion_48_workers_q1",
        "aiv_swiglu_tque_input_pair_completion_48_workers_q2",
    }),
    "mte3": frozenset({
        "aiv_rmsnorm_key0_finite_source_envelope",
        "aiv_elementwise_single_tile_finite_worker_completion_47_workers",
        "aiv_mul_q4q1_finite_worker_completion_48_workers",
        "aiv_mul_q4q1_finite_worker_completion_48_workers_11008b_x2",
        "aiv_mul_q4q1_finite_worker_completion_48_workers_13824b_x2",
        "aiv_mul_q4q1_finite_worker_completion_48_workers_18432b_x3",
        "aiv_mul_q4q1_finite_worker_completion_48_workers_17664b_x5",
        "aiv_mte3_request_completion",
        "aiv_rope_vector_to_mte3_completion",
        "aiv_rope_vector_to_mte3_completion_43_workers",
        "aiv_rope_vector_to_mte3_completion_47_workers",
        "aiv_rope_key22_first_q_slice_vector_to_mte3_completion",
        "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_4_workers",
        "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_8_workers",
        "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_16_workers",
        "aiv_datacopypad_vector_to_mte3_completion",
        "aiv_datacopypad_vector_to_mte3_completion_two_workers",
        "aiv_rmsnorm_key0_h2048_vector_to_mte3_completion_4_workers",
        "aiv_rmsnorm_key0_h2048_vector_to_mte3_completion_8_workers",
        "aiv_rmsnorm_key0_h2048_first_mte3_completion_32_workers",
        "aiv_rmsnorm_key0_h5120_first_mte3_completion_32_workers",
        "aiv_add_rmsnorm_key30_terminal_output_sequence_32_workers",
        "aiv_datacopypad_vector_to_mte3_completion_32_workers",
        "aiv_datacopypad_vector_to_mte3_completion_43_workers",
        "aiv_datacopypad_vector_to_mte3_completion_47_workers",
        "aiv_swiglu_tque_vector_to_mte3_completion_24_workers",
        "aiv_swiglu_tque_vector_to_mte3_completion_40_workers",
        "aiv_swiglu_tque_vector_to_mte3_completion_48_workers",
        "aiv_reshape_cache_equal_chain_completion",
        "aiv_reshape_cache_equal_chain_completion_48_workers",
        "aiv_reshape_cache_equal_warm_chain_completion_2_workers",
        "aiv_reshape_cache_equal_warm_chain_completion_4_workers",
        "aiv_reshape_cache_equal_warm_chain_completion_8_workers",
        "aiv_reshape_cache_equal_warm_chain_completion_16_workers",
        "aiv_reshape_cache_equal_repeated_chain_completion_48_workers_1024b",
        "aiv_reshape_cache_equal_repeated_chain_completion_48_workers_2048b",
        "aiv_reshape_cache_unequal_repeated_pair_completion_48_workers_2048b",
    }),
    "l0a_buffer": frozenset({
        "cann_nbuffer33_l1_to_l0_pair_completion",
    }),
})


def _require_name(owner: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string")
    return value.strip()


def _require_finite(owner: str, value: float, *, positive: bool) -> float:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise TypeError(f"{owner} must be a finite number")
    normalized = float(value)
    if positive and normalized <= 0:
        raise ValueError(f"{owner} must be positive")
    if not positive and normalized < 0:
        raise ValueError(f"{owner} cannot be negative")
    return normalized


def _immutable_float_mapping(
        owner: str, values: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    normalized = {}
    for name, value in values.items():
        key = _require_name(f"{owner} key", name)
        normalized[key] = _require_finite(
            f"{owner}[{key!r}]", value, positive=False)
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class ComputeServiceDemand:
    """Operations assigned to one typed service on a physical resource.

    A service identifies a hardware mechanism such as a generic vector lane,
    Exp, or Div.  It is not a kernel name or an empirical latency class.
    Multiple services on one ``ComputeDemand`` remain serialized on that
    demand's single physical resource.
    """

    service: str
    operations: float
    serial_invocations: int = 0
    operations_by_worker: Tuple[float, ...] = ()
    serial_invocations_by_worker: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "service",
            _require_name(
                "ComputeServiceDemand.service", self.service))
        if self.service not in _TYPED_COMPUTE_SERVICES:
            raise ValueError(
                "ComputeServiceDemand.service must be one of "
                f"{sorted(_TYPED_COMPUTE_SERVICES)}, got {self.service!r}")
        object.__setattr__(
            self, "operations",
            _require_finite(
                "ComputeServiceDemand.operations", self.operations,
                positive=True))
        if (isinstance(self.serial_invocations, bool)
                or not isinstance(self.serial_invocations, int)):
            raise TypeError(
                "ComputeServiceDemand.serial_invocations must be an integer")
        if self.serial_invocations < 0:
            raise ValueError(
                "ComputeServiceDemand.serial_invocations cannot be negative")
        if not isinstance(self.operations_by_worker, tuple):
            raise TypeError(
                "ComputeServiceDemand.operations_by_worker must be a tuple")
        if not isinstance(self.serial_invocations_by_worker, tuple):
            raise TypeError(
                "ComputeServiceDemand.serial_invocations_by_worker must be "
                "a tuple")
        has_worker_operations = bool(self.operations_by_worker)
        has_worker_invocations = bool(self.serial_invocations_by_worker)
        if has_worker_operations != has_worker_invocations:
            raise ValueError(
                "ComputeServiceDemand worker operations and invocations must "
                "either both be present or both be absent")
        if not has_worker_operations:
            return
        if (len(self.operations_by_worker)
                != len(self.serial_invocations_by_worker)):
            raise ValueError(
                "ComputeServiceDemand worker operations and invocations must "
                "have equal length")
        normalized_operations = tuple(
            _require_finite(
                f"ComputeServiceDemand.operations_by_worker[{index}]",
                value, positive=False)
            for index, value in enumerate(self.operations_by_worker)
        )
        normalized_invocations = []
        for index, value in enumerate(
                self.serial_invocations_by_worker):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "ComputeServiceDemand.serial_invocations_by_worker"
                    f"[{index}] must be an integer")
            if value < 0:
                raise ValueError(
                    "ComputeServiceDemand.serial_invocations_by_worker"
                    f"[{index}] cannot be negative")
            normalized_invocations.append(value)
        worker_operations = math.fsum(normalized_operations)
        conservation_tolerance = (
            max(
                math.ulp(worker_operations),
                math.ulp(self.operations),
            )
            * max(1, len(normalized_operations)))
        if abs(worker_operations - self.operations) > conservation_tolerance:
            raise ValueError(
                "ComputeServiceDemand worker operations must conserve the "
                f"aggregate: workers={worker_operations}, "
                f"aggregate={self.operations}")
        if max(normalized_invocations) != self.serial_invocations:
            raise ValueError(
                "ComputeServiceDemand serial_invocations must equal the "
                "critical-worker invocation count")
        object.__setattr__(
            self, "operations_by_worker", normalized_operations)
        object.__setattr__(
            self, "serial_invocations_by_worker",
            tuple(normalized_invocations))


@dataclass(frozen=True)
class ComputeExactRequestDemand:
    """Exact mechanism requests assigned to concrete physical workers.

    Request sizes are source-defined mechanism units, not workload-duration
    classes.  The formula performs exact-only HWSpec lookup and prices the
    maximum serialized sum over workers; it never interpolates this schedule.
    """

    service: str
    request_sizes_by_worker: Tuple[Tuple[int, ...], ...]
    request_size_runs_by_worker: Tuple[
        Tuple[Tuple[int, int], ...], ...] = ()
    pipelined_completion: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "service",
            _require_name(
                "ComputeExactRequestDemand.service", self.service))
        if self.service not in _COMPUTE_EXACT_REQUEST_SERVICES:
            raise ValueError(
                "ComputeExactRequestDemand.service must be one of "
                f"{sorted(_COMPUTE_EXACT_REQUEST_SERVICES)}, got "
                f"{self.service!r}")
        if not isinstance(self.request_sizes_by_worker, tuple):
            raise TypeError(
                "ComputeExactRequestDemand.request_sizes_by_worker must be "
                "a tuple")
        if not isinstance(self.request_size_runs_by_worker, tuple):
            raise TypeError(
                "ComputeExactRequestDemand.request_size_runs_by_worker must "
                "be a tuple")
        if not isinstance(self.pipelined_completion, bool):
            raise TypeError(
                "ComputeExactRequestDemand.pipelined_completion must be "
                "boolean")
        has_request_sizes = bool(self.request_sizes_by_worker)
        has_request_runs = bool(self.request_size_runs_by_worker)
        if has_request_sizes == has_request_runs:
            raise ValueError(
                "ComputeExactRequestDemand requires exactly one request "
                "schedule representation")
        if has_request_sizes:
            normalized_workers = []
            total_request_size = 0
            for worker_requests in self.request_sizes_by_worker:
                if not isinstance(worker_requests, tuple):
                    raise TypeError(
                        "ComputeExactRequestDemand worker requests must be "
                        "tuples")
                normalized_requests = []
                for request_size in worker_requests:
                    if (isinstance(request_size, bool)
                            or not isinstance(request_size, int)
                            or request_size <= 0):
                        raise ValueError(
                            "ComputeExactRequestDemand request sizes must be "
                            "positive integers")
                    normalized_requests.append(request_size)
                    total_request_size += request_size
                normalized_workers.append(tuple(normalized_requests))
            normalized_runs = ()
        else:
            normalized_workers = ()
            normalized_runs = []
            total_request_size = 0
            for worker_runs in self.request_size_runs_by_worker:
                if not isinstance(worker_runs, tuple):
                    raise TypeError(
                        "ComputeExactRequestDemand worker request runs must "
                        "be tuples")
                counts_by_size = {}
                for run in worker_runs:
                    if (not isinstance(run, tuple) or len(run) != 2):
                        raise TypeError(
                            "ComputeExactRequestDemand request runs must be "
                            "(size, count) pairs")
                    request_size, count = run
                    if (isinstance(request_size, bool)
                            or not isinstance(request_size, int)
                            or request_size <= 0):
                        raise ValueError(
                            "ComputeExactRequestDemand request-run sizes must "
                            "be positive integers")
                    if (isinstance(count, bool) or not isinstance(count, int)
                            or count <= 0):
                        raise ValueError(
                            "ComputeExactRequestDemand request-run counts must "
                            "be positive integers")
                    if request_size in counts_by_size:
                        counts_by_size[request_size] += count
                    else:
                        counts_by_size[request_size] = count
                    total_request_size += request_size * count
                normalized_runs.append(tuple(sorted(counts_by_size.items())))
        if total_request_size <= 0:
            raise ValueError(
                "ComputeExactRequestDemand requires at least one request")
        object.__setattr__(
            self, "request_sizes_by_worker", tuple(normalized_workers))
        object.__setattr__(
            self, "request_size_runs_by_worker", tuple(normalized_runs))

    @property
    def worker_count(self) -> int:
        """Return the source-defined physical-worker population."""

        schedules = (self.request_sizes_by_worker
                     if self.request_sizes_by_worker
                     else self.request_size_runs_by_worker)
        return len(schedules)

    @property
    def request_size_runs(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        """Return canonical per-worker ``(request_size, count)`` schedules."""

        if self.request_size_runs_by_worker:
            return self.request_size_runs_by_worker
        runs = []
        for worker_requests in self.request_sizes_by_worker:
            counts_by_size = {}
            for request_size in worker_requests:
                if request_size in counts_by_size:
                    counts_by_size[request_size] += 1
                else:
                    counts_by_size[request_size] = 1
            runs.append(tuple(sorted(counts_by_size.items())))
        return tuple(runs)

    @property
    def operations(self) -> float:
        """Return conserved logical request elements."""

        return float(math.fsum(
            request_size * count
            for worker_runs in self.request_size_runs
            for request_size, count in worker_runs))


@dataclass(frozen=True)
class MemoryExactRequestDemand:
    """Exact completion requests issued to one physical memory endpoint.

    This is a latency floor for the same bytes already present in the memory
    path, not additional traffic.  Pipeline composition therefore takes the
    maximum of byte-capacity time and this exact completion service.
    """

    resource_role: str
    service: str
    request_sizes_by_worker: Tuple[Tuple[int, ...], ...]
    pipelined_completion: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_role",
            _require_name(
                "MemoryExactRequestDemand.resource_role",
                self.resource_role))
        object.__setattr__(
            self, "service",
            _require_name("MemoryExactRequestDemand.service", self.service))
        if not isinstance(self.pipelined_completion, bool):
            raise TypeError(
                "MemoryExactRequestDemand.pipelined_completion must be "
                "boolean")
        if self.resource_role not in MEMORY_EXACT_REQUEST_SERVICES_BY_ROLE:
            raise ValueError(
                "MemoryExactRequestDemand.resource_role must be one of "
                f"{sorted(MEMORY_EXACT_REQUEST_SERVICES_BY_ROLE)}, got "
                f"{self.resource_role!r}")
        allowed = MEMORY_EXACT_REQUEST_SERVICES_BY_ROLE[self.resource_role]
        if self.service not in allowed:
            raise ValueError(
                f"MemoryExactRequestDemand service for {self.resource_role!r} "
                f"must be one of {sorted(allowed)}, got {self.service!r}")
        if (not isinstance(self.request_sizes_by_worker, tuple)
                or not self.request_sizes_by_worker):
            raise TypeError(
                "MemoryExactRequestDemand.request_sizes_by_worker must be "
                "a non-empty tuple")
        normalized_workers = []
        request_count = 0
        for worker_requests in self.request_sizes_by_worker:
            if not isinstance(worker_requests, tuple):
                raise TypeError(
                    "MemoryExactRequestDemand worker requests must be tuples")
            normalized_requests = []
            for request_size in worker_requests:
                if (isinstance(request_size, bool)
                        or not isinstance(request_size, int)
                        or request_size <= 0):
                    raise ValueError(
                        "MemoryExactRequestDemand request sizes must be "
                        "positive integers")
                normalized_requests.append(request_size)
                request_count += 1
            normalized_workers.append(tuple(normalized_requests))
        if request_count == 0:
            raise ValueError(
                "MemoryExactRequestDemand requires at least one request")
        object.__setattr__(
            self, "request_sizes_by_worker", tuple(normalized_workers))


@dataclass(frozen=True)
class ComputeDemand:
    """Operations assigned to one physical compute resource role.

    Typed steady services and exact worker-request services are optional so
    hardware and operators without a source-closed schedule retain aggregate
    resource capacity.  If present, together they exactly conserve the
    aggregate amount.
    """

    resource_role: str
    operations: float
    shape_efficiency: float
    schedule_efficiency: float
    service_demands: Tuple[ComputeServiceDemand, ...] = ()
    exact_request_demands: Tuple[ComputeExactRequestDemand, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_role",
            _require_name("ComputeDemand.resource_role", self.resource_role))
        object.__setattr__(
            self, "operations",
            _require_finite(
                "ComputeDemand.operations", self.operations, positive=False))
        for field_name in ("shape_efficiency", "schedule_efficiency"):
            value = _require_finite(
                f"ComputeDemand.{field_name}", getattr(self, field_name),
                positive=True)
            if value > 1.0:
                raise ValueError(
                    f"ComputeDemand.{field_name} cannot exceed one")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.service_demands, tuple):
            raise TypeError(
                "ComputeDemand.service_demands must be a tuple")
        if any(not isinstance(value, ComputeServiceDemand)
               for value in self.service_demands):
            raise TypeError(
                "ComputeDemand.service_demands must contain "
                "ComputeServiceDemand")
        service_names = tuple(
            value.service for value in self.service_demands)
        if len(service_names) != len(set(service_names)):
            raise ValueError(
                "ComputeDemand.service_demands cannot repeat services")
        worker_counts = set()
        if self.service_demands:
            if self.resource_role not in TYPED_COMPUTE_SERVICES_BY_ROLE:
                raise ValueError(
                    "typed compute services are supported only on declared "
                    "typed resource roles such as compute_vector; got "
                    f"{self.resource_role!r}")
            allowed_services = TYPED_COMPUTE_SERVICES_BY_ROLE[
                self.resource_role]
            unknown_services = set(service_names) - allowed_services
            if unknown_services:
                raise ValueError(
                    f"typed compute services for {self.resource_role!r} "
                    f"must be one of {sorted(allowed_services)}, got "
                    f"{sorted(unknown_services)}")
            if self.operations == 0.0:
                raise ValueError(
                    "typed compute services require positive aggregate "
                    "operations")
            worker_counts = {
                len(value.operations_by_worker)
                for value in self.service_demands
                if value.operations_by_worker
            }
            if worker_counts:
                if len(worker_counts) != 1:
                    raise ValueError(
                        "typed compute worker schedules must use one worker "
                        "count")
                if any(not value.operations_by_worker
                       for value in self.service_demands):
                    raise ValueError(
                        "aggregate and worker-scheduled typed compute services "
                        "cannot be mixed")
        if not isinstance(self.exact_request_demands, tuple):
            raise TypeError(
                "ComputeDemand.exact_request_demands must be a tuple")
        if any(not isinstance(value, ComputeExactRequestDemand)
               for value in self.exact_request_demands):
            raise TypeError(
                "ComputeDemand.exact_request_demands must contain "
                "ComputeExactRequestDemand")
        exact_service_names = tuple(
            value.service for value in self.exact_request_demands)
        if len(exact_service_names) != len(set(exact_service_names)):
            raise ValueError(
                "ComputeDemand.exact_request_demands cannot repeat services")
        if set(service_names) & set(exact_service_names):
            raise ValueError(
                "ComputeDemand typed and exact request services cannot repeat "
                "a service")
        if self.exact_request_demands:
            if self.resource_role not in (
                    COMPUTE_EXACT_REQUEST_SERVICES_BY_ROLE):
                raise ValueError(
                    "exact request services are unsupported on compute "
                    f"role {self.resource_role!r}")
            allowed_exact_services = (
                COMPUTE_EXACT_REQUEST_SERVICES_BY_ROLE[self.resource_role])
            unknown_exact_services = (
                set(exact_service_names) - allowed_exact_services)
            if unknown_exact_services:
                raise ValueError(
                    f"exact request services for {self.resource_role!r} "
                    f"must be one of {sorted(allowed_exact_services)}, got "
                    f"{sorted(unknown_exact_services)}")
            if self.operations == 0.0:
                raise ValueError(
                    "exact request services require positive aggregate "
                    "operations")
            exact_worker_counts = {
                value.worker_count
                for value in self.exact_request_demands
            }
            if len(exact_worker_counts) != 1:
                raise ValueError(
                    "exact request services must use one worker count")
            if worker_counts and worker_counts != exact_worker_counts:
                raise ValueError(
                    "typed and exact request worker schedules must use the "
                    "same worker count")
        if self.service_demands or self.exact_request_demands:
            typed_operations = math.fsum((
                *(value.operations for value in self.service_demands),
                *(value.operations for value in self.exact_request_demands),
            ))
            conservation_tolerance = (
                max(
                    math.ulp(typed_operations),
                    math.ulp(self.operations),
                )
                * max(
                    1,
                    len(self.service_demands)
                    + len(self.exact_request_demands),
                ))
            if abs(typed_operations - self.operations) > (
                    conservation_tolerance):
                raise ValueError(
                    "ComputeDemand typed and exact service operations must "
                    "conserve "
                    f"aggregate operations: typed={typed_operations}, "
                    f"aggregate={self.operations}")


@dataclass(frozen=True)
class MemoryNodeDemand:
    """Work reaching one ordered memory service node.

    Transaction count and size are present only when both are independently
    established.  A byte-only node keeps both fields ``None`` instead of
    turning a buffer size or cache capacity into a fabricated transaction.
    """

    resource_role: str
    transactions: Optional[float]
    bytes: float
    transaction_bytes: Optional[float]
    independent_service_streams: int = 1
    capacity_ceiling_resource_role: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_role",
            _require_name(
                "MemoryNodeDemand.resource_role", self.resource_role))
        byte_count = _require_finite(
            "MemoryNodeDemand.bytes", self.bytes, positive=True)
        if (isinstance(self.independent_service_streams, bool)
                or not isinstance(self.independent_service_streams, int)):
            raise TypeError(
                "MemoryNodeDemand.independent_service_streams must be an "
                "integer")
        if self.independent_service_streams <= 0:
            raise ValueError(
                "MemoryNodeDemand.independent_service_streams must be "
                "positive")
        if self.capacity_ceiling_resource_role is not None:
            object.__setattr__(
                self,
                "capacity_ceiling_resource_role",
                _require_name(
                    "MemoryNodeDemand.capacity_ceiling_resource_role",
                    self.capacity_ceiling_resource_role),
            )
        if (self.independent_service_streams > 1
                and self.capacity_ceiling_resource_role is None):
            raise ValueError(
                "parallel memory service streams require an architectural "
                "capacity ceiling resource role")
        has_transactions = self.transactions is not None
        has_transaction_bytes = self.transaction_bytes is not None
        if has_transactions != has_transaction_bytes:
            raise ValueError(
                "MemoryNodeDemand transactions and transaction_bytes must "
                "either both be present or both be None")
        if has_transactions:
            transactions = _require_finite(
                "MemoryNodeDemand.transactions", self.transactions,
                positive=True)
            transaction_bytes = _require_finite(
                "MemoryNodeDemand.transaction_bytes", self.transaction_bytes,
                positive=True)
            if not math.isclose(
                    byte_count, transactions * transaction_bytes,
                    rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(
                    "MemoryNodeDemand.bytes must equal transactions * "
                    "transaction_bytes")
            object.__setattr__(self, "transactions", transactions)
            object.__setattr__(
                self, "transaction_bytes", transaction_bytes)
        object.__setattr__(self, "bytes", byte_count)

    @property
    def rate_unit(self) -> str:
        return "transactions" if self.transactions is not None else "bytes"


@dataclass(frozen=True)
class MemorySourceDemand:
    """Workload-side producer and independent-request availability."""

    active_units: int
    producers_per_unit: float
    independent_requests_per_producer: Optional[float]
    frontend_rate_per_unit_s: Optional[float]

    def __post_init__(self) -> None:
        if (isinstance(self.active_units, bool)
                or not isinstance(self.active_units, int)):
            raise TypeError("MemorySourceDemand.active_units must be an integer")
        if self.active_units <= 0:
            raise ValueError("MemorySourceDemand.active_units must be positive")
        object.__setattr__(
            self, "producers_per_unit",
            _require_finite(
                "MemorySourceDemand.producers_per_unit",
                self.producers_per_unit, positive=True))
        if self.independent_requests_per_producer is not None:
            object.__setattr__(
                self, "independent_requests_per_producer",
                _require_finite(
                    "MemorySourceDemand.independent_requests_per_producer",
                    self.independent_requests_per_producer, positive=True))
        if self.frontend_rate_per_unit_s is not None:
            object.__setattr__(
                self, "frontend_rate_per_unit_s",
                _require_finite(
                    "MemorySourceDemand.frontend_rate_per_unit_s",
                    self.frontend_rate_per_unit_s, positive=True))

    @property
    def workload_slots_per_unit(self) -> Optional[float]:
        if self.independent_requests_per_producer is None:
            return None
        return self.producers_per_unit * self.independent_requests_per_producer


@dataclass(frozen=True)
class QueueUse:
    """Workload allocation rate and entry lifetime for one local structure."""

    name: str
    allocations_per_initial_request: float
    lifetime_weights: Tuple[Tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name("QueueUse.name", self.name))
        object.__setattr__(
            self, "allocations_per_initial_request",
            _require_finite(
                "QueueUse.allocations_per_initial_request",
                self.allocations_per_initial_request, positive=True))
        if (not isinstance(self.lifetime_weights, tuple)
                or not self.lifetime_weights):
            raise TypeError("QueueUse.lifetime_weights must be a non-empty tuple")
        normalized = []
        seen = set()
        for role, weight in self.lifetime_weights:
            normalized_role = _require_name("QueueUse lifetime role", role)
            if normalized_role in seen:
                raise ValueError(
                    f"QueueUse has duplicate lifetime role {normalized_role!r}")
            seen.add(normalized_role)
            normalized.append((
                normalized_role,
                _require_finite(
                    f"QueueUse lifetime weight for {normalized_role!r}",
                    weight, positive=True),
            ))
        object.__setattr__(self, "lifetime_weights", tuple(normalized))


@dataclass(frozen=True)
class MemoryAccessClassSignature:
    """Exact workload-independent signature for one memory access class."""

    access_class: str
    dtype: str
    tile_m: int
    tile_k: int
    logical_row_bytes: int
    operand_role: Optional[str] = None
    tile_n: Optional[int] = None
    group_size_m: Optional[int] = None
    weight_streams_per_group: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "access_class",
            _require_name(
                "MemoryAccessClassSignature.access_class",
                self.access_class))
        dtype = _require_name(
            "MemoryAccessClassSignature.dtype", self.dtype).lower()
        object.__setattr__(self, "dtype", dtype)
        for field_name in ("tile_m", "tile_k", "logical_row_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"MemoryAccessClassSignature.{field_name} must be an "
                    "integer")
            if value <= 0:
                raise ValueError(
                    f"MemoryAccessClassSignature.{field_name} must be "
                    "positive")
        grouped_values = (
            self.operand_role, self.tile_n, self.group_size_m)
        if all(value is None for value in grouped_values):
            if self.weight_streams_per_group is not None:
                raise ValueError(
                    "MemoryAccessClassSignature weight-stream topology "
                    "requires grouped operand fields")
            return
        if any(value is None for value in grouped_values):
            raise ValueError(
                "MemoryAccessClassSignature grouped operand fields must be "
                "all present or all null")
        if self.operand_role not in ("activation", "weight", "combined"):
            raise ValueError(
                "MemoryAccessClassSignature.operand_role must be "
                "'activation', 'weight', or 'combined'")
        for field_name in ("tile_n", "group_size_m"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"MemoryAccessClassSignature.{field_name} must be an "
                    "integer")
            if value <= 0:
                raise ValueError(
                    f"MemoryAccessClassSignature.{field_name} must be "
                    "positive")
        if self.weight_streams_per_group is not None:
            value = self.weight_streams_per_group
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "MemoryAccessClassSignature.weight_streams_per_group "
                    "must be an integer")
            if value <= 0 or value > self.group_size_m:
                raise ValueError(
                    "MemoryAccessClassSignature.weight_streams_per_group "
                    "must be within group_size_m")


@dataclass(frozen=True)
class MemoryAccessClassDemand:
    """Logical ingress bytes served by one typed access mechanism.

    A source-program service may replace the complete generic ingress path
    service while leaving downstream hierarchy traffic independently priced.
    Partial replacement is intentionally unsupported.
    """

    service: str
    ingress_resource_role: str
    logical_bytes: float
    signature: MemoryAccessClassSignature
    replaces_ingress_path_service: bool = False

    @classmethod
    def bind_validated_logical_bytes(cls, template, *, logical_bytes):
        """Bind dynamic byte work to an already-checked source service.

        The exact access signature and ingress policy describe immutable
        source-program facts.  Predictors may reuse that checked contract for
        another shape only after validating the new logical byte count.
        """

        if not isinstance(template, cls):
            raise TypeError(
                "memory-access template must be MemoryAccessClassDemand")
        instance = object.__new__(cls)
        instance.__dict__.update(template.__dict__)
        object.__setattr__(
            instance,
            "logical_bytes",
            _require_finite(
                "MemoryAccessClassDemand.logical_bytes",
                logical_bytes,
                positive=True),
        )
        return instance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "service",
            _require_name("MemoryAccessClassDemand.service", self.service))
        object.__setattr__(
            self, "ingress_resource_role",
            _require_name(
                "MemoryAccessClassDemand.ingress_resource_role",
                self.ingress_resource_role))
        object.__setattr__(
            self, "logical_bytes",
            _require_finite(
                "MemoryAccessClassDemand.logical_bytes",
                self.logical_bytes, positive=True))
        if not isinstance(self.signature, MemoryAccessClassSignature):
            raise TypeError(
                "MemoryAccessClassDemand.signature must be "
                "MemoryAccessClassSignature")
        if self.service != self.signature.access_class:
            raise ValueError(
                "MemoryAccessClassDemand.service must equal the signature "
                "access_class")
        if not isinstance(self.replaces_ingress_path_service, bool):
            raise TypeError(
                "MemoryAccessClassDemand.replaces_ingress_path_service "
                "must be a boolean")


@dataclass(frozen=True)
class MemoryAccessClassService:
    """Full-chip logical-byte capacity for one exact access signature."""

    service: str
    access_kind: str
    ingress_resource_role: str
    signature: MemoryAccessClassSignature
    scope: str
    logical_byte_capacity_per_s: float
    evidence_ref: str
    replay_to_dram_fraction: Optional[float] = None
    # The same evidence measured at several launch CTA counts, ascending.  A
    # full-chip service is a rate the launch has to reach, and a launch with
    # one CTA per multiprocessor does not reach the same rate as one that fills
    # them.  When the evidence swept the grid, the capacity is read at the
    # launch's own grid instead of at the sweep's minimum, which belongs to
    # whichever grid happened to be least concurrent.
    logical_byte_capacity_per_s_by_grid_ctas: Optional[
        Tuple[Tuple[int, float], ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "service",
            _require_name("MemoryAccessClassService.service", self.service))
        table = self.logical_byte_capacity_per_s_by_grid_ctas
        if table is not None:
            if not isinstance(table, tuple) or not table:
                raise TypeError(
                    "MemoryAccessClassService."
                    "logical_byte_capacity_per_s_by_grid_ctas must be a "
                    "non-empty tuple")
            previous_grid = 0
            for entry in table:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise TypeError(
                        "measured capacity entries must be (grid, capacity)")
                grid, capacity = entry
                if (isinstance(grid, bool) or not isinstance(grid, int)
                        or grid <= previous_grid):
                    raise ValueError(
                        "measured capacity grids must ascend and be positive")
                if (isinstance(capacity, bool)
                        or not isinstance(capacity, (int, float))
                        or not float(capacity) > 0.0):
                    raise ValueError(
                        "measured capacities must be positive")
                previous_grid = grid
        if self.access_kind not in ("read", "write"):
            raise ValueError(
                "MemoryAccessClassService.access_kind must be 'read' or "
                "'write'")
        object.__setattr__(
            self, "ingress_resource_role",
            _require_name(
                "MemoryAccessClassService.ingress_resource_role",
                self.ingress_resource_role))
        if not isinstance(self.signature, MemoryAccessClassSignature):
            raise TypeError(
                "MemoryAccessClassService.signature must be "
                "MemoryAccessClassSignature")
        if self.service != self.signature.access_class:
            raise ValueError(
                "MemoryAccessClassService.service must equal the signature "
                "access_class")
        if self.scope != "full_chip":
            raise ValueError(
                "MemoryAccessClassService.scope must be 'full_chip'")
        object.__setattr__(
            self, "logical_byte_capacity_per_s",
            _require_finite(
                "MemoryAccessClassService.logical_byte_capacity_per_s",
                self.logical_byte_capacity_per_s, positive=True))
        object.__setattr__(
            self, "evidence_ref",
            _require_name(
                "MemoryAccessClassService.evidence_ref",
                self.evidence_ref))
        if self.replay_to_dram_fraction is not None:
            fraction = _require_finite(
                "MemoryAccessClassService.replay_to_dram_fraction",
                self.replay_to_dram_fraction, positive=False)
            if fraction < 0 or fraction > 1:
                raise ValueError(
                    "MemoryAccessClassService.replay_to_dram_fraction must "
                    "be within [0, 1]")
            object.__setattr__(
                self, "replay_to_dram_fraction", fraction)


@dataclass(frozen=True)
class MemoryPathDemand:
    """One ordered path solved in transaction/s or byte/s units."""

    name: str
    access_kind: str
    initial_transactions: Optional[float]
    nodes: Tuple[MemoryNodeDemand, ...]
    source: MemorySourceDemand
    queues: Tuple[QueueUse, ...] = ()
    access_class_demands: Tuple[MemoryAccessClassDemand, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("MemoryPathDemand.name", self.name))
        if self.access_kind not in ("read", "write"):
            raise ValueError(
                "MemoryPathDemand.access_kind must be 'read' or 'write'")
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise TypeError("MemoryPathDemand.nodes must be a non-empty tuple")
        if any(not isinstance(node, MemoryNodeDemand) for node in self.nodes):
            raise TypeError("MemoryPathDemand.nodes must contain MemoryNodeDemand")
        roles = tuple(node.resource_role for node in self.nodes)
        if len(roles) != len(set(roles)):
            raise ValueError("MemoryPathDemand node roles must be unique")
        rate_units = {node.rate_unit for node in self.nodes}
        if len(rate_units) != 1:
            raise ValueError(
                "MemoryPathDemand cannot mix transaction and byte-only nodes")
        rate_unit = next(iter(rate_units))
        if rate_unit == "transactions":
            if self.initial_transactions is None:
                raise ValueError(
                    "transaction paths require initial_transactions")
            initial = _require_finite(
                "MemoryPathDemand.initial_transactions",
                self.initial_transactions, positive=True)
            if not math.isclose(
                    self.nodes[0].transactions, initial,
                    rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    "MemoryPathDemand initial_transactions must equal "
                    "first-node transactions")
            object.__setattr__(self, "initial_transactions", initial)
        elif self.initial_transactions is not None:
            raise ValueError(
                "byte-only paths must set initial_transactions to None")
        if not isinstance(self.source, MemorySourceDemand):
            raise TypeError("MemoryPathDemand.source must be MemorySourceDemand")
        if not isinstance(self.queues, tuple):
            raise TypeError("MemoryPathDemand.queues must be a tuple")
        if any(not isinstance(queue, QueueUse) for queue in self.queues):
            raise TypeError("MemoryPathDemand.queues must contain QueueUse")
        queue_names = tuple(queue.name for queue in self.queues)
        if len(queue_names) != len(set(queue_names)):
            raise ValueError("MemoryPathDemand queue names must be unique")
        if rate_unit == "bytes":
            if self.source.independent_requests_per_producer is not None:
                raise ValueError(
                    "byte-only paths cannot declare independent request slots")
            if self.source.frontend_rate_per_unit_s is not None:
                raise ValueError(
                    "byte-only paths cannot declare a request frontend rate")
            if self.queues:
                raise ValueError(
                    "byte-only paths cannot declare request queues")
        unknown_lifetime_roles = {
            role
            for queue in self.queues
            for role, _ in queue.lifetime_weights
            if role not in roles
        }
        if unknown_lifetime_roles:
            raise ValueError(
                "MemoryPathDemand queue lifetimes reference unknown roles: "
                f"{sorted(unknown_lifetime_roles)}")
        if not isinstance(self.access_class_demands, tuple):
            raise TypeError(
                "MemoryPathDemand.access_class_demands must be a tuple")
        if any(not isinstance(value, MemoryAccessClassDemand)
               for value in self.access_class_demands):
            raise TypeError(
                "MemoryPathDemand.access_class_demands must contain "
                "MemoryAccessClassDemand")
        service_identities = tuple(
            (
                value.service,
                value.ingress_resource_role,
                value.signature,
            )
            for value in self.access_class_demands)
        if len(service_identities) != len(set(service_identities)):
            raise ValueError(
                "MemoryPathDemand cannot repeat an exact memory access-class "
                "demand identity")
        ingress_role = self.nodes[0].resource_role
        invalid_ingress = {
            value.ingress_resource_role
            for value in self.access_class_demands
            if value.ingress_resource_role != ingress_role
        }
        if invalid_ingress:
            raise ValueError(
                "MemoryPathDemand access-class demands must bind the first "
                f"ingress node {ingress_role!r}, got "
                f"{sorted(invalid_ingress)}")
        typed_ingress_bytes = math.fsum(
            value.logical_bytes for value in self.access_class_demands)
        ingress_bytes = self.nodes[0].bytes
        conservation_tolerance = max(
            math.ulp(typed_ingress_bytes), math.ulp(ingress_bytes), 1e-9)
        if typed_ingress_bytes > ingress_bytes + conservation_tolerance:
            raise ValueError(
                "MemoryPathDemand access-class logical bytes cannot exceed "
                f"ingress bytes: typed={typed_ingress_bytes}, "
                f"ingress={ingress_bytes}")

    @property
    def rate_unit(self) -> str:
        return self.nodes[0].rate_unit


@dataclass(frozen=True)
class PhaseComputeDemand:
    """Source-defined semantic work assigned to one compute resource.

    ``operation`` names the static semantic or compiler-lowered operation;
    it is not a latency class or a fitted kernel correction.  Multiple
    operations may share one physical resource inside a phase.
    """

    resource_role: str
    operation: str
    operations: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_role",
            _require_name(
                "PhaseComputeDemand.resource_role", self.resource_role))
        object.__setattr__(
            self, "operation",
            _require_name("PhaseComputeDemand.operation", self.operation))
        object.__setattr__(
            self, "operations",
            _require_finite(
                "PhaseComputeDemand.operations", self.operations,
                positive=True))


@dataclass(frozen=True)
class PhaseMemoryDemand:
    """Byte work assigned to one node of an existing typed memory path."""

    path_name: str
    resource_role: str
    bytes: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path_name",
            _require_name("PhaseMemoryDemand.path_name", self.path_name))
        object.__setattr__(
            self, "resource_role",
            _require_name(
                "PhaseMemoryDemand.resource_role", self.resource_role))
        object.__setattr__(
            self, "bytes",
            _require_finite(
                "PhaseMemoryDemand.bytes", self.bytes, positive=True))


@dataclass(frozen=True)
class AnalyticalDependencyPath:
    """One coarse semantic critical path over priced resource services.

    Services use the canonical ``launch:submit``, ``task:<mechanism>``,
    ``compute:<role>``, ``local_memory:<role>``, ``memory:<path>`` or
    ``memory_latency:<path>`` namespace.  A memory latency service represents
    one additional source-defined dependent access level without duplicating
    the path's bytes.  A task mechanism represents a separately characterized
    operator-independent stage increment; its serial count comes from static
    source stages and launch concurrency.  Each service contributes its
    analytical time exactly once.  This contract deliberately has no
    instructions, buffers, or warp/scoreboard state.
    """

    name: str
    services: Tuple[str, ...]
    concurrent_service_groups: Tuple[Tuple[str, ...], ...] = ()
    serial_service_counts: Tuple[Tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name",
            _require_name("AnalyticalDependencyPath.name", self.name))
        if not isinstance(self.services, tuple) or not self.services:
            raise TypeError(
                "AnalyticalDependencyPath.services must be a non-empty tuple")
        normalized = tuple(
            _require_name("AnalyticalDependencyPath service", service)
            for service in self.services)
        invalid = tuple(
            service for service in normalized
            if (service not in (
                    "launch:submit",
                    MEMORY_DEPENDENCY_TASK_INCREMENT_SERVICE,
                    GLOBAL_STORE_COMPLETION_SERVICE)
                and not service.startswith(
                    ("compute:", "local_memory:", "memory:",
                     "memory_latency:"))))
        if invalid:
            raise ValueError(
                "AnalyticalDependencyPath services must use launch:submit, "
                "a supported task:/memory_completion:, compute:, local_memory:, memory:, or "
                "memory_latency: "
                f"names: {invalid}")
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "AnalyticalDependencyPath services must be unique")
        object.__setattr__(self, "services", normalized)
        if not isinstance(self.concurrent_service_groups, tuple):
            raise TypeError(
                "AnalyticalDependencyPath.concurrent_service_groups must "
                "be a tuple")
        grouped = set()
        normalized_groups = []
        for group in self.concurrent_service_groups:
            if not isinstance(group, tuple) or len(group) < 2:
                raise TypeError(
                    "each analytical dependency concurrent group must be a "
                    "tuple with at least two services")
            normalized_group = tuple(
                _require_name(
                    "AnalyticalDependencyPath concurrent service", service)
                for service in group)
            if len(normalized_group) != len(set(normalized_group)):
                raise ValueError(
                    "analytical dependency concurrent groups cannot repeat "
                    "a service")
            unknown = set(normalized_group) - set(normalized)
            if unknown:
                raise ValueError(
                    "analytical dependency concurrent group references "
                    f"unknown services: {sorted(unknown)}")
            overlap = grouped.intersection(normalized_group)
            if overlap:
                raise ValueError(
                    "analytical dependency service belongs to multiple "
                    f"concurrent groups: {sorted(overlap)}")
            grouped.update(normalized_group)
            normalized_groups.append(normalized_group)
        object.__setattr__(
            self, "concurrent_service_groups", tuple(normalized_groups))
        if not isinstance(self.serial_service_counts, tuple):
            raise TypeError(
                "AnalyticalDependencyPath.serial_service_counts must be a "
                "tuple")
        normalized_counts = []
        counted = set()
        for service, count in self.serial_service_counts:
            normalized_service = _require_name(
                "AnalyticalDependencyPath serial service", service)
            if normalized_service not in normalized:
                raise ValueError(
                    "analytical dependency serial count references unknown "
                    f"service: {normalized_service!r}")
            if normalized_service in counted:
                raise ValueError(
                    "analytical dependency serial counts cannot repeat a "
                    f"service: {normalized_service!r}")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(
                    "analytical dependency serial service count must be an "
                    "integer")
            if count <= 0:
                raise ValueError(
                    "analytical dependency serial service count must be "
                    "positive")
            counted.add(normalized_service)
            normalized_counts.append((normalized_service, count))
        object.__setattr__(
            self, "serial_service_counts", tuple(normalized_counts))


@dataclass(frozen=True)
class AnalyticalPhase:
    """One non-timing phase in a source-defined dependency graph."""

    name: str
    predecessors: Tuple[str, ...]
    compute_demands: Tuple[PhaseComputeDemand, ...]
    memory_demands: Tuple[PhaseMemoryDemand, ...]
    completion_barrier: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("AnalyticalPhase.name", self.name))
        if not isinstance(self.predecessors, tuple):
            raise TypeError("AnalyticalPhase.predecessors must be a tuple")
        normalized_predecessors = tuple(
            _require_name("AnalyticalPhase predecessor", value)
            for value in self.predecessors)
        if len(normalized_predecessors) != len(set(normalized_predecessors)):
            raise ValueError(
                "AnalyticalPhase predecessors must be unique")
        object.__setattr__(
            self, "predecessors", normalized_predecessors)
        if not isinstance(self.compute_demands, tuple):
            raise TypeError(
                "AnalyticalPhase.compute_demands must be a tuple")
        if any(not isinstance(value, PhaseComputeDemand)
               for value in self.compute_demands):
            raise TypeError(
                "AnalyticalPhase.compute_demands must contain "
                "PhaseComputeDemand")
        compute_keys = tuple(
            (value.resource_role, value.operation)
            for value in self.compute_demands)
        if len(compute_keys) != len(set(compute_keys)):
            raise ValueError(
                "AnalyticalPhase compute role/operation pairs must be unique")
        if not isinstance(self.memory_demands, tuple):
            raise TypeError(
                "AnalyticalPhase.memory_demands must be a tuple")
        if any(not isinstance(value, PhaseMemoryDemand)
               for value in self.memory_demands):
            raise TypeError(
                "AnalyticalPhase.memory_demands must contain "
                "PhaseMemoryDemand")
        memory_keys = tuple(
            (value.path_name, value.resource_role)
            for value in self.memory_demands)
        if len(memory_keys) != len(set(memory_keys)):
            raise ValueError(
                "AnalyticalPhase memory path/node pairs must be unique")
        if not isinstance(self.completion_barrier, bool):
            raise TypeError(
                "AnalyticalPhase.completion_barrier must be boolean")


@dataclass(frozen=True)
class AnalyticalPhaseLedger:
    """Non-timing phase graph that exactly conserves structured demand.

    Whole-kernel static work whose phase assignment is not independently
    known remains in ``unassigned_compute_demands``.  Keeping it explicit
    prevents a guessed phase split while retaining exact conservation.
    Phase operations retain source-semantic names and conserve physical
    resource roles; typed service conservation stays inside ``ComputeDemand``
    because a semantic operation need not map one-to-one to a hardware
    service.  The two checks are therefore deliberately transitive rather
    than a second service-by-phase schema.
    """

    phases: Tuple[AnalyticalPhase, ...]
    unassigned_compute_demands: Tuple[PhaseComputeDemand, ...] = ()
    limitations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.phases, tuple) or not self.phases:
            raise TypeError(
                "AnalyticalPhaseLedger.phases must be a non-empty tuple")
        if any(not isinstance(value, AnalyticalPhase)
               for value in self.phases):
            raise TypeError(
                "AnalyticalPhaseLedger.phases must contain AnalyticalPhase")
        names = tuple(value.name for value in self.phases)
        if len(names) != len(set(names)):
            raise ValueError("Analytical phase names must be unique")
        prior_names = set()
        for phase in self.phases:
            unknown = set(phase.predecessors) - prior_names
            if unknown:
                raise ValueError(
                    f"Analytical phase {phase.name!r} has non-prior "
                    f"predecessors: {sorted(unknown)}")
            prior_names.add(phase.name)
        if not isinstance(self.unassigned_compute_demands, tuple):
            raise TypeError(
                "AnalyticalPhaseLedger.unassigned_compute_demands must be a "
                "tuple")
        if any(not isinstance(value, PhaseComputeDemand)
               for value in self.unassigned_compute_demands):
            raise TypeError(
                "AnalyticalPhaseLedger.unassigned_compute_demands must "
                "contain PhaseComputeDemand")
        unassigned_keys = tuple(
            (value.resource_role, value.operation)
            for value in self.unassigned_compute_demands)
        if len(unassigned_keys) != len(set(unassigned_keys)):
            raise ValueError(
                "unassigned compute role/operation pairs must be unique")
        if not isinstance(self.limitations, tuple):
            raise TypeError(
                "AnalyticalPhaseLedger.limitations must be a tuple")
        normalized_limitations = tuple(
            _require_name("AnalyticalPhaseLedger limitation", value)
            for value in self.limitations)
        if len(normalized_limitations) != len(set(normalized_limitations)):
            raise ValueError(
                "AnalyticalPhaseLedger limitations must be unique")
        if self.unassigned_compute_demands and not normalized_limitations:
            raise ValueError(
                "unassigned phase work requires an explicit limitation")
        if not self.unassigned_compute_demands and normalized_limitations:
            raise ValueError(
                "phase-ledger limitations require unassigned compute work")
        object.__setattr__(self, "limitations", normalized_limitations)

    def validate_conservation(
            self, compute_demands: Tuple[ComputeDemand, ...],
            memory_paths: Tuple[MemoryPathDemand, ...]) -> None:
        """Fail if phase and unassigned work drift from typed demand."""

        expected_compute = {
            demand.resource_role: demand.operations
            for demand in compute_demands
        }
        actual_compute = {}
        phase_compute = tuple(
            demand
            for phase in self.phases
            for demand in phase.compute_demands)
        for demand in phase_compute + self.unassigned_compute_demands:
            if demand.resource_role in actual_compute:
                current = actual_compute[demand.resource_role]
            else:
                current = 0.0
            actual_compute[demand.resource_role] = (
                current + demand.operations)
        if set(actual_compute) != set(expected_compute):
            raise ValueError(
                "analytical phase compute roles do not match structured "
                f"demand: phase={sorted(actual_compute)}, "
                f"structured={sorted(expected_compute)}")
        for role, expected in expected_compute.items():
            if not math.isclose(
                    actual_compute[role], expected,
                    rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(
                    "analytical phase compute work does not conserve "
                    f"{role!r}: phase={actual_compute[role]}, "
                    f"structured={expected}")

        expected_memory = {
            (path.name, node.resource_role): node.bytes
            for path in memory_paths
            for node in path.nodes
        }
        actual_memory = {}
        for phase in self.phases:
            for demand in phase.memory_demands:
                key = (demand.path_name, demand.resource_role)
                if key in actual_memory:
                    current = actual_memory[key]
                else:
                    current = 0.0
                actual_memory[key] = current + demand.bytes
        if set(actual_memory) != set(expected_memory):
            raise ValueError(
                "analytical phase memory nodes do not match structured "
                f"demand: phase={sorted(actual_memory)}, "
                f"structured={sorted(expected_memory)}")
        for key, expected in expected_memory.items():
            if not math.isclose(
                    actual_memory[key], expected,
                    rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(
                    "analytical phase memory work does not conserve "
                    f"{key!r}: phase={actual_memory[key]}, "
                    f"structured={expected}")

    def rescaled(
            self, compute_demands: Tuple[ComputeDemand, ...],
            memory_paths: Tuple[MemoryPathDemand, ...]
            ) -> "AnalyticalPhaseLedger":
        """Preserve phase shares across an explicit resource ablation."""

        old_compute = {}
        phase_compute = tuple(
            demand
            for phase in self.phases
            for demand in phase.compute_demands)
        for demand in phase_compute + self.unassigned_compute_demands:
            if demand.resource_role in old_compute:
                current = old_compute[demand.resource_role]
            else:
                current = 0.0
            old_compute[demand.resource_role] = current + demand.operations
        new_compute = {
            demand.resource_role: demand.operations
            for demand in compute_demands
        }
        unknown_compute = set(new_compute) - set(old_compute)
        if unknown_compute:
            raise ValueError(
                "cannot introduce compute roles into an analytical phase "
                f"ledger: {sorted(unknown_compute)}")

        old_memory = {}
        for phase in self.phases:
            for demand in phase.memory_demands:
                key = (demand.path_name, demand.resource_role)
                if key in old_memory:
                    current = old_memory[key]
                else:
                    current = 0.0
                old_memory[key] = current + demand.bytes
        new_memory = {
            (path.name, node.resource_role): node.bytes
            for path in memory_paths
            for node in path.nodes
        }
        unknown_memory = set(new_memory) - set(old_memory)
        if unknown_memory:
            raise ValueError(
                "cannot introduce memory nodes into an analytical phase "
                f"ledger: {sorted(unknown_memory)}")

        def scale_compute(demand: PhaseComputeDemand):
            role = demand.resource_role
            if role not in new_compute or new_compute[role] == 0.0:
                return None
            return replace(
                demand,
                operations=(
                    demand.operations * new_compute[role]
                    / old_compute[role]),
            )

        def scale_memory(demand: PhaseMemoryDemand):
            key = (demand.path_name, demand.resource_role)
            if key not in new_memory or new_memory[key] == 0.0:
                return None
            return replace(
                demand,
                bytes=demand.bytes * new_memory[key] / old_memory[key],
            )

        phases = tuple(
            replace(
                phase,
                compute_demands=tuple(
                    scaled for demand in phase.compute_demands
                    for scaled in (scale_compute(demand),)
                    if scaled is not None),
                memory_demands=tuple(
                    scaled for demand in phase.memory_demands
                    for scaled in (scale_memory(demand),)
                    if scaled is not None),
            )
            for phase in self.phases)
        unassigned = tuple(
            scaled for demand in self.unassigned_compute_demands
            for scaled in (scale_compute(demand),)
            if scaled is not None)
        limitations = self.limitations if unassigned else ()
        result = AnalyticalPhaseLedger(
            phases=phases,
            unassigned_compute_demands=unassigned,
            limitations=limitations,
        )
        result.validate_conservation(compute_demands, memory_paths)
        return result


@dataclass(frozen=True)
class MemoryService:
    """Hardware byte/request capacity and unloaded latency for one node."""

    resource_role: str
    byte_capacity_per_s: float
    request_capacity_per_s: Optional[float]
    unloaded_latency_s: float
    queue_model: str
    service_time_s: Optional[float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_role",
            _require_name("MemoryService.resource_role", self.resource_role))
        object.__setattr__(
            self, "byte_capacity_per_s",
            _require_finite(
                "MemoryService.byte_capacity_per_s",
                self.byte_capacity_per_s, positive=True))
        if self.request_capacity_per_s is not None:
            object.__setattr__(
                self, "request_capacity_per_s",
                _require_finite(
                    "MemoryService.request_capacity_per_s",
                    self.request_capacity_per_s, positive=True))
        object.__setattr__(
            self, "unloaded_latency_s",
            _require_finite(
                "MemoryService.unloaded_latency_s",
                self.unloaded_latency_s, positive=False))
        if self.queue_model not in ("none", "md1"):
            raise ValueError(
                "MemoryService.queue_model must be 'none' or 'md1'")
        if self.queue_model == "md1":
            if self.service_time_s is None:
                raise ValueError("M/D/1 service requires service_time_s")
            object.__setattr__(
                self, "service_time_s",
                _require_finite(
                    "MemoryService.service_time_s", self.service_time_s,
                    positive=True))
        elif self.service_time_s is not None:
            raise ValueError(
                "queue_model='none' must not declare service_time_s")


@dataclass(frozen=True)
class QueueCapacity:
    """Hardware entry capacity for one explicitly named local structure."""

    name: str
    max_entries: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("QueueCapacity.name", self.name))
        object.__setattr__(
            self, "max_entries",
            _require_finite(
                "QueueCapacity.max_entries", self.max_entries, positive=True))


@dataclass(frozen=True)
class MemoryHierarchy:
    """Hardware mechanisms participating in one memory path."""

    services: Tuple[MemoryService, ...]
    frontend_rate_per_unit_s: Optional[float]
    source_slots_per_unit: Optional[float]
    queues: Tuple[QueueCapacity, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.services, tuple) or not self.services:
            raise TypeError("MemoryHierarchy.services must be a non-empty tuple")
        if any(not isinstance(value, MemoryService) for value in self.services):
            raise TypeError("MemoryHierarchy.services must contain MemoryService")
        roles = tuple(value.resource_role for value in self.services)
        if len(roles) != len(set(roles)):
            raise ValueError("MemoryHierarchy service roles must be unique")
        for field_name in (
                "frontend_rate_per_unit_s", "source_slots_per_unit"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name,
                    _require_finite(
                        f"MemoryHierarchy.{field_name}", value,
                        positive=True))
        if not isinstance(self.queues, tuple):
            raise TypeError("MemoryHierarchy.queues must be a tuple")
        if any(not isinstance(value, QueueCapacity) for value in self.queues):
            raise TypeError("MemoryHierarchy.queues must contain QueueCapacity")
        names = tuple(value.name for value in self.queues)
        if len(names) != len(set(names)):
            raise ValueError("MemoryHierarchy queue names must be unique")


@dataclass(frozen=True)
class PipelineResource:
    """Exclusive pipeline service slots, shared globally or pinned per worker."""

    name: str
    scope: str
    slots: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("PipelineResource.name", self.name))
        if self.scope not in ("shared", "worker"):
            raise ValueError("PipelineResource.scope must be 'shared' or 'worker'")
        if isinstance(self.slots, bool) or not isinstance(self.slots, int):
            raise TypeError("PipelineResource.slots must be an integer")
        if self.slots <= 0:
            raise ValueError("PipelineResource.slots must be positive")


@dataclass(frozen=True)
class PipelineStage:
    """One topologically ordered stage of a tile task.

    ``predecessors=None`` preserves the legacy linear-stage contract: the
    first stage has no intra-task predecessor and every later stage depends on
    the immediately preceding stage.  An explicit tuple publishes a source
    DAG; an empty tuple therefore starts an independent branch.  Tuple order
    is topological, so predecessors may only name earlier stages.
    ``prior_resources`` waits for the named resources' state at logical-task
    entry, which carries a fixed-size recurrence across repeated task bodies.
    """

    name: str
    resource: str
    duration_s: float
    predecessors: Optional[Tuple[str, ...]] = None
    start_predecessors: Tuple[str, ...] = ()
    prior_resources: Tuple[str, ...] = ()
    occupancy_s: Optional[float] = None

    @classmethod
    def _from_validated_demand(
            cls, demand, duration_s, occupancy_s=None, *, resource=None):
        """Resolve one already validated demand without rechecking metadata."""

        if not isinstance(demand, PipelineStageDemand):
            raise TypeError("resolved pipeline stage requires stage demand")
        instance = object.__new__(cls)
        instance.__dict__.update({
            "name": demand.name,
            "resource": demand.service if resource is None else resource,
            "duration_s": _require_finite(
                "PipelineStage.duration_s", duration_s, positive=True),
            "occupancy_s": _require_finite(
                "PipelineStage.occupancy_s",
                duration_s if occupancy_s is None else occupancy_s,
                positive=False),
            "predecessors": demand.predecessors,
            "start_predecessors": demand.start_predecessors,
            "prior_resources": demand.prior_resources,
        })
        return instance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("PipelineStage.name", self.name))
        object.__setattr__(
            self, "resource",
            _require_name("PipelineStage.resource", self.resource))
        object.__setattr__(
            self, "duration_s",
            _require_finite(
                "PipelineStage.duration_s", self.duration_s, positive=True))
        object.__setattr__(
            self, "occupancy_s",
            _require_finite(
                "PipelineStage.occupancy_s",
                self.duration_s if self.occupancy_s is None
                else self.occupancy_s,
                positive=False))
        if self.occupancy_s > self.duration_s:
            raise ValueError(
                "PipelineStage occupancy cannot exceed completion latency")
        if self.predecessors is not None:
            if (not isinstance(self.predecessors, tuple)
                    or any(not isinstance(value, str)
                           for value in self.predecessors)):
                raise TypeError(
                    "PipelineStage.predecessors must be a tuple of names or "
                    "None")
            normalized = tuple(
                _require_name("PipelineStage.predecessors", value)
                for value in self.predecessors)
            if len(normalized) != len(set(normalized)):
                raise ValueError(
                    "PipelineStage.predecessors cannot contain duplicates")
            object.__setattr__(self, "predecessors", normalized)
        if (not isinstance(self.start_predecessors, tuple)
                or any(not isinstance(value, str)
                       for value in self.start_predecessors)):
            raise TypeError(
                "PipelineStage.start_predecessors must be a tuple of names")
        normalized_start = tuple(
            _require_name("PipelineStage.start_predecessors", value)
            for value in self.start_predecessors)
        if len(normalized_start) != len(set(normalized_start)):
            raise ValueError(
                "PipelineStage.start_predecessors cannot contain duplicates")
        if (self.predecessors is not None
                and set(normalized_start) & set(self.predecessors)):
            raise ValueError(
                "PipelineStage predecessor cannot require both start and end")
        object.__setattr__(self, "start_predecessors", normalized_start)
        if (not isinstance(self.prior_resources, tuple)
                or any(not isinstance(value, str)
                       for value in self.prior_resources)):
            raise TypeError(
                "PipelineStage.prior_resources must be a tuple of names")
        normalized_prior = tuple(
            _require_name("PipelineStage.prior_resources", value)
            for value in self.prior_resources)
        if len(normalized_prior) != len(set(normalized_prior)):
            raise ValueError(
                "PipelineStage.prior_resources cannot contain duplicates")
        object.__setattr__(self, "prior_resources", normalized_prior)


def _validate_pipeline_stage_order(label: str, stages: tuple) -> None:
    """Validate the implicit-linear or explicit-DAG predecessor contract."""

    prior_names = []
    prior_name_set = set()
    for index, stage in enumerate(stages):
        predecessors = (
            (() if index == 0 else (prior_names[-1],))
            if stage.predecessors is None else stage.predecessors)
        unknown = set(predecessors) - prior_name_set
        unknown_start = set(stage.start_predecessors) - prior_name_set
        if unknown or unknown_start:
            raise ValueError(
                f"{label} stage {stage.name!r} predecessors must name "
                "earlier stages: "
                f"end={sorted(unknown)}, start={sorted(unknown_start)}")
        prior_names.append(stage.name)
        prior_name_set.add(stage.name)


@dataclass(frozen=True)
class PipelineTask:
    """One tile pattern and its worker assignment in launch order.

    ``repeat_count`` compactly represents consecutive identical logical tasks
    on the same worker.  It is a schedule fact, not an aggregate-time shortcut:
    the pipeline evaluator applies the same resource and buffer recurrence once
    per logical repetition (with an exact max-plus fast-forward for long runs).
    """

    name: str
    worker: int
    stages: Tuple[PipelineStage, ...]
    repeat_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("PipelineTask.name", self.name))
        if isinstance(self.worker, bool) or not isinstance(self.worker, int):
            raise TypeError("PipelineTask.worker must be an integer")
        if self.worker < 0:
            raise ValueError("PipelineTask.worker cannot be negative")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError("PipelineTask.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError("PipelineTask.repeat_count must be positive")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise TypeError("PipelineTask.stages must be a non-empty tuple")
        if any(not isinstance(value, PipelineStage) for value in self.stages):
            raise TypeError("PipelineTask.stages must contain PipelineStage")
        names = tuple(value.name for value in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("PipelineTask stage names must be unique")
        _validate_pipeline_stage_order("PipelineTask", self.stages)


@dataclass(frozen=True)
class PipelineTaskSequence:
    """A consecutive task pattern repeated on one worker.

    This represents schedules such as a FlashAttention CTA's repeated KV body
    followed by one final store.  Repeating the whole sequence preserves store
    placement and buffer state exactly; it must not be flattened into separate
    aggregate stage totals.
    """

    name: str
    worker: int
    tasks: Tuple[PipelineTask, ...]
    repeat_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("PipelineTaskSequence.name", self.name))
        if isinstance(self.worker, bool) or not isinstance(self.worker, int):
            raise TypeError("PipelineTaskSequence.worker must be an integer")
        if self.worker < 0:
            raise ValueError("PipelineTaskSequence.worker cannot be negative")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError(
                "PipelineTaskSequence.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError(
                "PipelineTaskSequence.repeat_count must be positive")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise TypeError(
                "PipelineTaskSequence.tasks must be a non-empty tuple")
        if any(not isinstance(value, PipelineTask) for value in self.tasks):
            raise TypeError(
                "PipelineTaskSequence.tasks must contain PipelineTask")
        if any(value.worker != self.worker for value in self.tasks):
            raise ValueError(
                "PipelineTaskSequence tasks must use the sequence worker")
        names = tuple(value.name for value in self.tasks)
        if len(names) != len(set(names)):
            raise ValueError(
                "PipelineTaskSequence task names must be unique")


@dataclass(frozen=True)
class PipelineTaskProgram:
    """One worker's repeated ordered references to task sequences.

    ``repeat_count`` repeats the complete ``order`` pattern, then
    ``tail_order`` executes once.  Worker assignment and intervening
    resource/buffer state are preserved exactly; this is a launch-schedule
    compression, not an aggregate-time approximation.
    """

    name: str
    worker: int
    patterns: Tuple[PipelineTaskSequence, ...]
    order: Tuple[int, ...]
    repeat_count: int = 1
    tail_order: Tuple[int, ...] = ()
    worker_multiplicity: int = 1
    physical_workers: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("PipelineTaskProgram.name", self.name))
        if isinstance(self.worker, bool) or not isinstance(self.worker, int):
            raise TypeError("PipelineTaskProgram.worker must be an integer")
        if self.worker < 0:
            raise ValueError("PipelineTaskProgram.worker cannot be negative")
        if not isinstance(self.patterns, tuple) or not self.patterns:
            raise TypeError(
                "PipelineTaskProgram.patterns must be a non-empty tuple")
        if any(not isinstance(value, PipelineTaskSequence)
               for value in self.patterns):
            raise TypeError(
                "PipelineTaskProgram.patterns must contain "
                "PipelineTaskSequence")
        if any(value.worker != self.worker for value in self.patterns):
            raise ValueError(
                "PipelineTaskProgram patterns must use the program worker")
        pattern_names = tuple(value.name for value in self.patterns)
        if len(pattern_names) != len(set(pattern_names)):
            raise ValueError(
                "PipelineTaskProgram pattern names must be unique")
        if not isinstance(self.order, tuple) or not self.order:
            raise TypeError("PipelineTaskProgram.order must be a non-empty tuple")
        invalid_order = {
            value for value in self.order
            if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value >= len(self.patterns))
        }
        if invalid_order:
            raise ValueError(
                "PipelineTaskProgram.order contains invalid pattern indexes: "
                f"{sorted(invalid_order, key=str)}")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError(
                "PipelineTaskProgram.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError(
                "PipelineTaskProgram.repeat_count must be positive")
        if (isinstance(self.worker_multiplicity, bool)
                or not isinstance(self.worker_multiplicity, int)):
            raise TypeError(
                "PipelineTaskProgram.worker_multiplicity must be an integer")
        if self.worker_multiplicity <= 0:
            raise ValueError(
                "PipelineTaskProgram.worker_multiplicity must be positive")
        if not isinstance(self.physical_workers, tuple):
            raise TypeError(
                "PipelineTaskProgram.physical_workers must be a tuple")
        if self.physical_workers:
            if len(self.physical_workers) != self.worker_multiplicity:
                raise ValueError(
                    "PipelineTaskProgram.physical_workers must match "
                    "worker_multiplicity")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in self.physical_workers):
                raise ValueError(
                    "PipelineTaskProgram.physical_workers must contain "
                    "non-negative integers")
            if len(set(self.physical_workers)) != len(self.physical_workers):
                raise ValueError(
                    "PipelineTaskProgram.physical_workers must be unique")
        if not isinstance(self.tail_order, tuple):
            raise TypeError("PipelineTaskProgram.tail_order must be a tuple")
        invalid_tail_order = {
            value for value in self.tail_order
            if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value >= len(self.patterns))
        }
        if invalid_tail_order:
            raise ValueError(
                "PipelineTaskProgram.tail_order contains invalid pattern "
                f"indexes: {sorted(invalid_tail_order, key=str)}")


@dataclass(frozen=True)
class PipelineBuffer:
    """Finite producer-to-consumer buffer slots for overlapping tile stages.

    ``release`` is the source-visible event that returns one slot.  Most
    queue lifetimes retain the consumer's completion.  A source may instead
    permit the next producer to issue when its predecessor's DMA is accepted;
    that distinct ``start`` event is explicit rather than approximated as a
    full completion edge.
    """

    name: str
    producer_stage: str
    consumer_stage: str
    slots: int
    scope: str
    release: str = "completion"

    def __post_init__(self) -> None:
        for field_name in ("name", "producer_stage", "consumer_stage"):
            object.__setattr__(
                self, field_name,
                _require_name(
                    f"PipelineBuffer.{field_name}", getattr(self, field_name)))
        if isinstance(self.slots, bool) or not isinstance(self.slots, int):
            raise TypeError("PipelineBuffer.slots must be an integer")
        if self.slots <= 0:
            raise ValueError("PipelineBuffer.slots must be positive")
        if self.scope not in ("shared", "worker"):
            raise ValueError("PipelineBuffer.scope must be 'shared' or 'worker'")
        if self.release not in ("start", "completion"):
            raise ValueError(
                "PipelineBuffer.release must be 'start' or 'completion'")
        if self.producer_stage == self.consumer_stage:
            raise ValueError("PipelineBuffer stages must be distinct")


@dataclass(frozen=True)
class AttentionPipelinePhase:
    """One resolved source phase in a closed-form attention envelope."""

    name: str
    stages: Tuple[PipelineStage, ...]
    repeat_count: int = 1

    @classmethod
    def _from_validated_demand(cls, demand, stages):
        """Resolve one already validated phase demand and resolved stages."""

        if not isinstance(demand, AttentionPipelinePhaseDemand):
            raise TypeError("resolved attention phase requires phase demand")
        instance = object.__new__(cls)
        instance.__dict__.update({
            "name": demand.name,
            "stages": stages,
            "repeat_count": demand.repeat_count,
        })
        return instance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name",
            _require_name("AttentionPipelinePhase.name", self.name))
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError(
                "AttentionPipelinePhase.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError(
                "AttentionPipelinePhase.repeat_count must be positive")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise TypeError(
                "AttentionPipelinePhase.stages must be a non-empty tuple")
        if any(not isinstance(value, PipelineStage) for value in self.stages):
            raise TypeError(
                "AttentionPipelinePhase.stages must contain PipelineStage")
        names = tuple(value.name for value in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("AttentionPipelinePhase stage names must be unique")
        _validate_pipeline_stage_order("AttentionPipelinePhase", self.stages)


@dataclass(frozen=True)
class AttentionPipelineClass:
    """One resolved CTA-class transition and its analytical multiplicity.

    ``query_rows == panel_count == 0`` with no phases is a source-ordinal
    marker.  It advances direct-launch worker assignment and the resident CTA
    lifecycle without adding service time; the whole-launch issue lower bound
    remains responsible for the early-exit CTA itself.
    """

    name: str
    count: int
    query_rows: int
    panel_count: int
    phases: Tuple[AttentionPipelinePhase, ...]

    @classmethod
    def _from_validated_demand(cls, demand, phases):
        """Resolve one already validated class demand and phase program."""

        if not isinstance(demand, AttentionPipelineClassDemand):
            raise TypeError("resolved attention class requires class demand")
        instance = object.__new__(cls)
        instance.__dict__.update({
            "name": demand.name,
            "count": demand.count,
            "query_rows": demand.query_rows,
            "panel_count": demand.panel_count,
            "phases": phases,
        })
        return instance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name",
            _require_name("AttentionPipelineClass.name", self.name))
        for field_name in ("count", "query_rows", "panel_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AttentionPipelineClass.{field_name} must be an integer")
        source_ordinal_only = (
            self.query_rows == 0
            and self.panel_count == 0
            and not self.phases)
        if self.count <= 0 or (
                not source_ordinal_only
                and (self.query_rows <= 0 or self.panel_count < 0)):
            raise ValueError(
                "resolved attention class requires positive count/query_rows "
                "and non-negative panel_count, or the zero-sized "
                "source-ordinal marker")
        if not isinstance(self.phases, tuple):
            raise TypeError(
                "AttentionPipelineClass.phases must be a tuple")
        if not source_ordinal_only and not self.phases:
            raise TypeError(
                "active AttentionPipelineClass.phases must be non-empty")
        if any(not isinstance(value, AttentionPipelinePhase)
               for value in self.phases):
            raise TypeError(
                "AttentionPipelineClass.phases must contain "
                "AttentionPipelinePhase")


@dataclass(frozen=True)
class AttentionPipelineEnvelope:
    """Resolved constant-size launch envelope for analytical evaluation."""

    active_lane_count: int
    classes: Tuple[AttentionPipelineClass, ...]
    buffers: Tuple[PipelineBuffer, ...]
    source_repeat_count: int = 1
    cta_admission_policy: str = "round_robin"
    resident_tail_latency_s_per_panel: float = 0.0
    resident_tail_group_interval_panels: int = 0
    resident_tail_latency_s_per_group: float = 0.0

    @classmethod
    def _from_validated_demand(
            cls, demand, classes, *, tail_latency_s_per_panel,
            tail_latency_s_per_group):
        """Resolve one validated demand without rescanning its class stream."""

        if not isinstance(demand, AttentionPipelineEnvelopeDemand):
            raise TypeError("resolved attention envelope requires demand")
        instance = object.__new__(cls)
        instance.__dict__.update({
            "active_lane_count": demand.active_lane_count,
            "classes": classes,
            "buffers": demand.buffers,
            "source_repeat_count": demand.source_repeat_count,
            "cta_admission_policy": demand.cta_admission_policy,
            "resident_tail_latency_s_per_panel": _require_finite(
                "AttentionPipelineEnvelope."
                "resident_tail_latency_s_per_panel",
                tail_latency_s_per_panel, positive=False),
            "resident_tail_group_interval_panels": (
                demand.resident_tail_group_interval_panels),
            "resident_tail_latency_s_per_group": _require_finite(
                "AttentionPipelineEnvelope."
                "resident_tail_latency_s_per_group",
                tail_latency_s_per_group, positive=False),
        })
        return instance

    def __post_init__(self) -> None:
        if (isinstance(self.active_lane_count, bool)
                or not isinstance(self.active_lane_count, int)
                or self.active_lane_count <= 0):
            raise ValueError(
                "AttentionPipelineEnvelope.active_lane_count must be a "
                "positive integer")
        if not isinstance(self.classes, tuple) or not self.classes:
            raise TypeError(
                "AttentionPipelineEnvelope.classes must be a non-empty tuple")
        if any(not isinstance(value, AttentionPipelineClass)
               for value in self.classes):
            raise TypeError(
                "AttentionPipelineEnvelope.classes must contain "
                "AttentionPipelineClass")
        if (isinstance(self.source_repeat_count, bool)
                or not isinstance(self.source_repeat_count, int)
                or self.source_repeat_count <= 0):
            raise ValueError(
                "AttentionPipelineEnvelope.source_repeat_count must be a "
                "positive integer")
        if not isinstance(self.cta_admission_policy, str):
            raise TypeError(
                "AttentionPipelineEnvelope.cta_admission_policy must be a "
                "string")
        if self.cta_admission_policy not in (
                "round_robin", "paired_resident_fifo",
                "bounded_resident_suffix_fifo"):
            raise ValueError(
                "AttentionPipelineEnvelope.cta_admission_policy must be "
                "'round_robin', 'paired_resident_fifo', or "
                "'bounded_resident_suffix_fifo'")
        names = tuple(value.name for value in self.classes)
        if len(names) != len(set(names)):
            raise ValueError(
                "AttentionPipelineEnvelope class names must be unique")
        if not isinstance(self.buffers, tuple):
            raise TypeError(
                "AttentionPipelineEnvelope.buffers must be a tuple")
        if any(not isinstance(value, PipelineBuffer) for value in self.buffers):
            raise TypeError(
                "AttentionPipelineEnvelope.buffers must contain "
                "PipelineBuffer")
        if any(value.scope != "worker" for value in self.buffers):
            raise ValueError(
                "resolved attention pipeline buffers must be lane-local")
        object.__setattr__(
            self,
            "resident_tail_latency_s_per_panel",
            _require_finite(
                "AttentionPipelineEnvelope."
                "resident_tail_latency_s_per_panel",
                self.resident_tail_latency_s_per_panel,
                positive=False,
            ),
        )
        if (isinstance(self.resident_tail_group_interval_panels, bool)
                or not isinstance(
                    self.resident_tail_group_interval_panels, int)):
            raise TypeError(
                "AttentionPipelineEnvelope."
                "resident_tail_group_interval_panels must be an integer")
        if self.resident_tail_group_interval_panels < 0:
            raise ValueError(
                "AttentionPipelineEnvelope."
                "resident_tail_group_interval_panels cannot be negative")
        object.__setattr__(
            self,
            "resident_tail_latency_s_per_group",
            _require_finite(
                "AttentionPipelineEnvelope."
                "resident_tail_latency_s_per_group",
                self.resident_tail_latency_s_per_group,
                positive=False,
            ),
        )
        if ((self.resident_tail_group_interval_panels == 0)
                != (self.resident_tail_latency_s_per_group == 0.0)):
            raise ValueError(
                "attention resident-tail group interval and latency must be "
                "configured together")
        _validate_attention_pipeline_buffers(self.classes, self.buffers)


def _pipeline_entry_patterns(entry):
    """Return the independently repeatable task patterns in one entry."""

    if isinstance(entry, (PipelineTaskProgram, PipelineTaskProgramDemand)):
        return tuple(pattern.tasks for pattern in entry.patterns)
    if isinstance(entry, (PipelineTaskSequence, PipelineTaskSequenceDemand)):
        return (entry.tasks,)
    return ((entry,),)


def _validate_pipeline_pattern_contract(entries, buffers) -> None:
    """Validate finite-buffer lifetime pairing in each repeated pattern."""

    for entry in entries:
        for pattern in _pipeline_entry_patterns(entry):
            for buffer in buffers:
                events = []
                for task in pattern:
                    stage_names = tuple(stage.name for stage in task.stages)
                    has_producer = buffer.producer_stage in stage_names
                    has_consumer = buffer.consumer_stage in stage_names
                    if has_producer and has_consumer:
                        if (stage_names.index(buffer.producer_stage)
                                >= stage_names.index(buffer.consumer_stage)):
                            raise ValueError(
                                f"PipelineBuffer {buffer.name!r} producer "
                                "must precede consumer")
                        events.extend(("producer", "consumer"))
                    elif has_producer:
                        if task.repeat_count != 1:
                            raise ValueError(
                                f"PipelineBuffer {buffer.name!r} producer-only "
                                "task cannot repeat")
                        events.append("producer")
                    elif has_consumer:
                        if task.repeat_count != 1:
                            raise ValueError(
                                f"PipelineBuffer {buffer.name!r} consumer-only "
                                "task cannot repeat")
                        events.append("consumer")
                if not events:
                    raise ValueError(
                        f"PipelineBuffer {buffer.name!r} stages must occur in "
                        "every independently repeatable task pattern")
                if (len(events) % 2
                        or any(events[index:index + 2]
                               != ["producer", "consumer"]
                               for index in range(0, len(events), 2))):
                    raise ValueError(
                        f"PipelineBuffer {buffer.name!r} lifetime must be "
                        "paired producer then consumer within every pattern")


@dataclass(frozen=True)
class PipelinePlan:
    """Complete worker/task/resource plan for a Load/Compute/Store schedule."""

    worker_count: int
    resources: Tuple[PipelineResource, ...]
    tasks: Tuple[
        Union[PipelineTask, PipelineTaskSequence, PipelineTaskProgram], ...]
    buffers: Tuple[PipelineBuffer, ...] = ()
    initial_resource_availability_s: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.worker_count, bool) or not isinstance(
                self.worker_count, int):
            raise TypeError("PipelinePlan.worker_count must be an integer")
        if self.worker_count <= 0:
            raise ValueError("PipelinePlan.worker_count must be positive")
        if not isinstance(self.resources, tuple) or not self.resources:
            raise TypeError("PipelinePlan.resources must be a non-empty tuple")
        if any(not isinstance(value, PipelineResource) for value in self.resources):
            raise TypeError("PipelinePlan.resources must contain PipelineResource")
        resource_names = tuple(value.name for value in self.resources)
        if len(resource_names) != len(set(resource_names)):
            raise ValueError("PipelinePlan resource names must be unique")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise TypeError("PipelinePlan.tasks must be a non-empty tuple")
        if any(not isinstance(
                value,
                (PipelineTask, PipelineTaskSequence, PipelineTaskProgram))
               for value in self.tasks):
            raise TypeError(
                "PipelinePlan.tasks must contain PipelineTask or "
                "PipelineTaskSequence or PipelineTaskProgram")
        task_names = tuple(value.name for value in self.tasks)
        if len(task_names) != len(set(task_names)):
            raise ValueError("PipelinePlan task names must be unique")
        invalid_workers = {
            task.worker for task in self.tasks
            if task.worker >= self.worker_count
        }
        if invalid_workers:
            raise ValueError(
                "PipelinePlan tasks reference workers outside worker_count: "
                f"{sorted(invalid_workers)}")
        logical_tasks = tuple(
            logical
            for task in self.tasks
            for pattern in _pipeline_entry_patterns(task)
            for logical in pattern)
        unknown_resources = {
            stage.resource
            for task in logical_tasks
            for stage in task.stages
            if stage.resource not in resource_names
        }
        unknown_resources.update(
            resource
            for task in logical_tasks
            for stage in task.stages
            for resource in stage.prior_resources
            if resource not in resource_names)
        if unknown_resources:
            raise ValueError(
                "PipelinePlan stages reference unknown resources: "
                f"{sorted(unknown_resources)}")
        if not isinstance(self.buffers, tuple):
            raise TypeError("PipelinePlan.buffers must be a tuple")
        if any(not isinstance(value, PipelineBuffer) for value in self.buffers):
            raise TypeError("PipelinePlan.buffers must contain PipelineBuffer")
        buffer_names = tuple(value.name for value in self.buffers)
        if len(buffer_names) != len(set(buffer_names)):
            raise ValueError("PipelinePlan buffer names must be unique")
        if not isinstance(self.initial_resource_availability_s, tuple):
            raise TypeError(
                "PipelinePlan initial resource availability must be a tuple")
        initial_names = []
        for name, duration_s in self.initial_resource_availability_s:
            initial_names.append(_require_name(
                "PipelinePlan initial resource", name))
            _require_finite(
                "PipelinePlan initial resource availability",
                duration_s, positive=True)
            if name not in resource_names:
                raise ValueError(
                    "PipelinePlan initial availability references an unknown "
                    f"resource: {name!r}")
        if len(initial_names) != len(set(initial_names)):
            raise ValueError(
                "PipelinePlan initial resource availability cannot repeat "
                "resources")
        _validate_pipeline_pattern_contract(self.tasks, self.buffers)


@dataclass(frozen=True)
class PipelineStageDemand:
    """One tile stage on a structured physical service.

    Memory and aggregate compute stages use ``work_fraction`` to partition an
    already structured whole-kernel demand.  A source-closed typed compute
    stage instead carries its absolute per-tile ``compute_service_demands``
    and/or ``compute_exact_request_demands``.  A source-closed byte-only
    memory stage carries absolute per-worker ``memory_node_demands`` and may
    additionally carry ``launch_shared_memory_node_demands`` for traffic
    issued identically by every worker but reaching a path node only once per
    launch.  ``memory_access_class_demands`` binds the same per-worker stage
    to an exact typed memory service; the formula requires its logical bytes
    to conserve the aggregate path annotation.  An exact completion may
    replace the generic byte-capacity time of its own resource only for the
    source-local dependency by setting
    ``memory_exact_request_replaces_resource_capacity``; aggregate path bytes
    remain conserved and priced independently.
    ``local_memory_source_completion`` may be ``"sequence_start"`` or
    ``"steady"`` and makes a transaction-resolved absolute memory stage carry
    its own finite-source completion dependency.  A sequence start pays one
    source fill; steady stages continue the same request source without
    repeating that fill.  The
    formula derives the stage-local transaction ratios from its node bytes and
    the aggregate path transaction geometry instead of reusing the
    launch-wide solved request rate; the aggregate path remains an independent
    resource envelope.  ``completion_resource_roles`` may identify the path
    nodes whose
    completion releases a source-local dependency while later path nodes
    continue to drain; every node remains conserved and priced as an aggregate
    resource lower bound. A fractional memory capacity stage may select the
    same completion nodes without repeating first-request latency; its full
    downstream work fraction remains conserved. Per-worker bytes may occur asymmetrically in a
    repeated source program.  A launch-shared stage must still occur
    identically and exactly once on every worker.  The formula prices these
    demands on one physical worker and verifies conservation against the
    aggregate compute and memory contracts.
    """

    name: str
    service: str
    work_fraction: Optional[float]
    include_first_request_latency: bool = False
    # A stage whose transactions are issued together as one request burst by
    # every thread of the task, rather than streamed against the launch's own
    # solved rate.  Its remaining transactions then drain at the path's
    # service capacity: the solved rate is derived from this very dependency
    # and already contains the idle the dependency creates between bursts, so
    # reusing it on top of the full first-request latency charges that idle
    # twice. The consumer joins every node's return endpoint, while the byte
    # port is occupied only by node capacity work. Only meaningful together
    # with include_first_request_latency.
    burst_drains_at_path_capacity: bool = False
    compute_service_demands: Tuple[ComputeServiceDemand, ...] = ()
    compute_exact_request_demands: Tuple[
        ComputeExactRequestDemand, ...] = ()
    memory_node_demands: Tuple[PhaseMemoryDemand, ...] = ()
    launch_shared_memory_node_demands: Tuple[PhaseMemoryDemand, ...] = ()
    memory_access_class_demands: Tuple[MemoryAccessClassDemand, ...] = ()
    memory_exact_request_demands: Tuple[
        MemoryExactRequestDemand, ...] = ()
    memory_exact_request_replaces_resource_capacity: bool = False
    aiv_ingress_group_size: Optional[int] = None
    predecessors: Optional[Tuple[str, ...]] = None
    start_predecessors: Tuple[str, ...] = ()
    prior_resources: Tuple[str, ...] = ()
    completion_resource_roles: Optional[Tuple[str, ...]] = None
    absolute_duration_s: Optional[float] = None
    local_memory_source_completion: Optional[str] = None
    # An asynchronous fractional memory stage releases its service after the
    # allocated byte/transaction work, while its consumers wait for completion.
    # Each such stage includes its own first-request latency; finite pipeline
    # buffers, rather than a guessed hardware credit count, limit overlap.
    pipelined_memory_completion: bool = False

    @classmethod
    def bind_validated_dynamic_value(
            cls, template, *, work_fraction=None, absolute_duration_s=None):
        """Bind one checked dynamic scalar to an immutable stage template.

        Source-program predictors frequently repeat an identical stage
        topology while only its normalized work or a documented mechanism
        latency changes.  The template has already completed the full
        contract validation; this method retains the validation applicable to
        the new scalar while avoiding repeated validation of its immutable
        name, service, dependency, and demand metadata.
        """

        if not isinstance(template, cls):
            raise TypeError("pipeline stage template must be PipelineStageDemand")
        is_absolute = template.absolute_duration_s is not None
        if is_absolute:
            if work_fraction is not None or absolute_duration_s is None:
                raise ValueError(
                    "absolute pipeline stage templates require exactly one "
                    "absolute duration")
            normalized_fraction = None
            normalized_duration = _require_finite(
                "PipelineStageDemand.absolute_duration_s",
                absolute_duration_s, positive=True)
        else:
            if work_fraction is None or absolute_duration_s is not None:
                raise ValueError(
                    "fractional pipeline stage templates require exactly one "
                    "work fraction")
            normalized_fraction = _require_finite(
                "PipelineStageDemand.work_fraction", work_fraction,
                positive=True)
            if normalized_fraction > 1.0:
                raise ValueError(
                    "PipelineStageDemand.work_fraction cannot exceed one")
            normalized_duration = None
        instance = object.__new__(cls)
        instance.__dict__.update(template.__dict__)
        object.__setattr__(instance, "work_fraction", normalized_fraction)
        object.__setattr__(instance, "absolute_duration_s", normalized_duration)
        return instance

    @classmethod
    def bind_validated_typed_compute_operations(
            cls, template, operations):
        """Bind source-local typed-compute work to a checked stage template.

        Typed source stages often retain the same service/dependency program
        while only the current tile's operation counts change.  This method
        keeps the checked source contract intact and validates only those
        dynamic counts.  It intentionally excludes worker-distributed demand
        vectors: their aggregate/concurrency contract is not a scalar binding.
        """

        if not isinstance(template, cls):
            raise TypeError("pipeline stage template must be PipelineStageDemand")
        if (template.work_fraction is not None
                or template.absolute_duration_s is not None
                or not template.compute_service_demands
                or template.compute_exact_request_demands
                or template.memory_node_demands
                or template.launch_shared_memory_node_demands
                or template.memory_access_class_demands
                or template.memory_exact_request_demands):
            raise ValueError(
                "typed-compute pipeline template must contain only typed "
                "compute demands")
        if (not isinstance(operations, tuple)
                or len(operations) != len(template.compute_service_demands)):
            raise ValueError(
                "typed-compute pipeline binding operations disagree with "
                "the template")
        bound_demands = []
        for demand, value in zip(template.compute_service_demands, operations):
            if demand.operations_by_worker or demand.serial_invocations_by_worker:
                raise ValueError(
                    "typed-compute scalar binding cannot alter "
                    "worker-distributed demands")
            bound = object.__new__(ComputeServiceDemand)
            bound.__dict__.update(demand.__dict__)
            object.__setattr__(
                bound, "operations", _require_finite(
                    "ComputeServiceDemand.operations", value, positive=True))
            bound_demands.append(bound)
        instance = object.__new__(cls)
        instance.__dict__.update(template.__dict__)
        object.__setattr__(
            instance, "compute_service_demands", tuple(bound_demands))
        return instance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name",
            _require_name("PipelineStageDemand.name", self.name))
        object.__setattr__(
            self, "service",
            _require_name("PipelineStageDemand.service", self.service))
        if self.aiv_ingress_group_size is not None:
            if (isinstance(self.aiv_ingress_group_size, bool)
                    or not isinstance(self.aiv_ingress_group_size, int)):
                raise TypeError(
                    "PipelineStageDemand.aiv_ingress_group_size must be "
                    "an integer or None")
            if self.aiv_ingress_group_size <= 1:
                raise ValueError(
                    "PipelineStageDemand.aiv_ingress_group_size must be "
                    "greater than one")
            if self.service != "memory:mte2_read":
                raise ValueError(
                    "AIV ingress grouping is valid only for memory:mte2_read")
        if not isinstance(self.burst_drains_at_path_capacity, bool):
            raise TypeError(
                "PipelineStageDemand.burst_drains_at_path_capacity must be "
                "a boolean")
        if (self.burst_drains_at_path_capacity
                and not self.include_first_request_latency):
            raise ValueError(
                "PipelineStageDemand.burst_drains_at_path_capacity requires "
                "include_first_request_latency")
        if not isinstance(self.include_first_request_latency, bool):
            raise TypeError(
                "PipelineStageDemand.include_first_request_latency must be "
                "boolean")
        if not isinstance(self.pipelined_memory_completion, bool):
            raise TypeError(
                "PipelineStageDemand.pipelined_memory_completion must be "
                "boolean")
        if (self.pipelined_memory_completion
                and not self.include_first_request_latency):
            raise ValueError(
                "pipelined memory completion requires "
                "include_first_request_latency")
        if (self.local_memory_source_completion is not None
                and self.local_memory_source_completion
                not in ("sequence_start", "steady")):
            raise ValueError(
                "PipelineStageDemand.local_memory_source_completion must "
                "be 'sequence_start', 'steady', or None")
        if not isinstance(self.compute_service_demands, tuple):
            raise TypeError(
                "PipelineStageDemand.compute_service_demands must be a tuple")
        if any(not isinstance(value, ComputeServiceDemand)
               for value in self.compute_service_demands):
            raise TypeError(
                "PipelineStageDemand.compute_service_demands must contain "
                "ComputeServiceDemand")
        service_names = tuple(
            value.service for value in self.compute_service_demands)
        if len(service_names) != len(set(service_names)):
            raise ValueError(
                "PipelineStageDemand.compute_service_demands cannot repeat "
                "services")
        if not isinstance(self.compute_exact_request_demands, tuple):
            raise TypeError(
                "PipelineStageDemand.compute_exact_request_demands must be "
                "a tuple")
        if any(not isinstance(value, ComputeExactRequestDemand)
               for value in self.compute_exact_request_demands):
            raise TypeError(
                "PipelineStageDemand.compute_exact_request_demands must "
                "contain ComputeExactRequestDemand")
        exact_service_names = tuple(
            value.service for value in self.compute_exact_request_demands)
        if len(exact_service_names) != len(set(exact_service_names)):
            raise ValueError(
                "PipelineStageDemand.compute_exact_request_demands cannot "
                "repeat services")
        if set(service_names) & set(exact_service_names):
            raise ValueError(
                "PipelineStageDemand typed and exact compute demands cannot "
                "repeat a service")
        if any(value.worker_count != 1
               for value in self.compute_exact_request_demands):
            raise ValueError(
                "pipeline-stage exact requests must describe exactly one "
                "local worker")
        if not isinstance(self.memory_node_demands, tuple):
            raise TypeError(
                "PipelineStageDemand.memory_node_demands must be a tuple")
        if any(not isinstance(value, PhaseMemoryDemand)
               for value in self.memory_node_demands):
            raise TypeError(
                "PipelineStageDemand.memory_node_demands must contain "
                "PhaseMemoryDemand")
        memory_keys = tuple(
            (value.path_name, value.resource_role)
            for value in self.memory_node_demands)
        if len(memory_keys) != len(set(memory_keys)):
            raise ValueError(
                "PipelineStageDemand.memory_node_demands cannot repeat path "
                "nodes")
        if not isinstance(self.launch_shared_memory_node_demands, tuple):
            raise TypeError(
                "PipelineStageDemand.launch_shared_memory_node_demands must "
                "be a tuple")
        if any(not isinstance(value, PhaseMemoryDemand)
               for value in self.launch_shared_memory_node_demands):
            raise TypeError(
                "PipelineStageDemand.launch_shared_memory_node_demands must "
                "contain PhaseMemoryDemand")
        shared_memory_keys = tuple(
            (value.path_name, value.resource_role)
            for value in self.launch_shared_memory_node_demands)
        if len(shared_memory_keys) != len(set(shared_memory_keys)):
            raise ValueError(
                "PipelineStageDemand.launch_shared_memory_node_demands "
                "cannot repeat path nodes")
        if set(memory_keys) & set(shared_memory_keys):
            raise ValueError(
                "pipeline-stage per-worker and launch-shared memory demands "
                "cannot repeat a path node")
        if not isinstance(self.memory_access_class_demands, tuple):
            raise TypeError(
                "PipelineStageDemand.memory_access_class_demands must be a "
                "tuple")
        if any(not isinstance(value, MemoryAccessClassDemand)
               for value in self.memory_access_class_demands):
            raise TypeError(
                "PipelineStageDemand.memory_access_class_demands must contain "
                "MemoryAccessClassDemand")
        access_class_keys = tuple(
            (
                value.service,
                value.ingress_resource_role,
                value.signature,
            )
            for value in self.memory_access_class_demands)
        if len(access_class_keys) != len(set(access_class_keys)):
            raise ValueError(
                "PipelineStageDemand.memory_access_class_demands cannot "
                "repeat exact services")
        if self.completion_resource_roles is not None:
            if not isinstance(self.completion_resource_roles, tuple):
                raise TypeError(
                    "PipelineStageDemand.completion_resource_roles must be "
                    "a tuple or None")
            if not self.completion_resource_roles:
                raise ValueError(
                    "PipelineStageDemand.completion_resource_roles cannot "
                    "be empty")
            has_node_demands = bool(
                self.memory_node_demands or self.launch_shared_memory_node_demands)
            if not has_node_demands and not (
                    self.service.startswith("memory:")
                    and self.work_fraction is not None
                    and not self.include_first_request_latency):
                raise ValueError(
                    "PipelineStageDemand completion roles require memory node "
                    "demands or a fractional memory capacity stage")
            normalized_completion_roles = tuple(
                _require_name(
                    "PipelineStageDemand.completion_resource_roles", role)
                for role in self.completion_resource_roles)
            if len(normalized_completion_roles) != len(
                    set(normalized_completion_roles)):
                raise ValueError(
                    "PipelineStageDemand.completion_resource_roles cannot "
                    "repeat roles")
            available_memory_roles = {
                demand.resource_role
                for demand in (
                    self.memory_node_demands
                    + self.launch_shared_memory_node_demands)
            }
            unknown_completion_roles = (
                set(normalized_completion_roles) - available_memory_roles)
            if has_node_demands and unknown_completion_roles:
                raise ValueError(
                    "PipelineStageDemand completion roles must belong to "
                    "the stage memory path: "
                    f"{sorted(unknown_completion_roles)}")
            object.__setattr__(
                self, "completion_resource_roles",
                normalized_completion_roles)
        if not isinstance(self.memory_exact_request_demands, tuple):
            raise TypeError(
                "PipelineStageDemand.memory_exact_request_demands must be "
                "a tuple")
        if any(not isinstance(value, MemoryExactRequestDemand)
               for value in self.memory_exact_request_demands):
            raise TypeError(
                "PipelineStageDemand.memory_exact_request_demands must "
                "contain MemoryExactRequestDemand")
        memory_exact_keys = tuple(
            (value.resource_role, value.service)
            for value in self.memory_exact_request_demands)
        if len(memory_exact_keys) != len(set(memory_exact_keys)):
            raise ValueError(
                "PipelineStageDemand.memory_exact_request_demands cannot "
                "repeat services")
        if len(self.memory_exact_request_demands) > 1:
            raise ValueError(
                "a pipeline stage may declare at most one exact memory "
                "completion service")
        if any(len(value.request_sizes_by_worker) != 1
               for value in self.memory_exact_request_demands):
            raise ValueError(
                "pipeline-stage exact memory requests must describe exactly "
                "one local worker")
        if not isinstance(
                self.memory_exact_request_replaces_resource_capacity, bool):
            raise TypeError(
                "PipelineStageDemand.memory_exact_request_replaces_resource_"
                "capacity must be boolean")
        if (self.memory_exact_request_replaces_resource_capacity
                and len(self.memory_exact_request_demands) != 1):
            raise ValueError(
                "replacing stage-local memory capacity requires exactly one "
                "exact memory completion")
        if (self.completion_resource_roles is not None
                and any(
                    value.resource_role not in self.completion_resource_roles
                    for value in self.memory_exact_request_demands)):
            raise ValueError(
                "pipeline-stage exact memory completion must belong to a "
                "completion resource role")
        if self.aiv_ingress_group_size is not None:
            if any(
                    value.resource_role == "mte2"
                    for value in self.memory_exact_request_demands):
                raise ValueError(
                    "AIV ingress grouping cannot stack with an exact MTE2 "
                    "completion")
        has_compute_demands = bool(
            self.compute_service_demands
            or self.compute_exact_request_demands)
        has_memory_node_demands = bool(
            self.memory_node_demands
            or self.launch_shared_memory_node_demands)
        has_memory_access_class_demands = bool(
            self.memory_access_class_demands)
        has_absolute_duration = self.absolute_duration_s is not None
        if has_absolute_duration:
            duration = _require_finite(
                "PipelineStageDemand.absolute_duration_s",
                self.absolute_duration_s,
                positive=True)
            if not self.service.startswith("mechanism:"):
                raise ValueError(
                    "absolute-duration pipeline stages require a mechanism "
                    "service")
            if (self.work_fraction is not None
                    or has_compute_demands
                    or has_memory_node_demands
                    or has_memory_access_class_demands
                    or self.memory_exact_request_demands):
                raise ValueError(
                    "absolute-duration pipeline stages cannot mix aggregate "
                    "or exact resource demands")
            object.__setattr__(self, "absolute_duration_s", duration)
        if self.memory_exact_request_replaces_resource_capacity:
            exact_demand, = self.memory_exact_request_demands
            stage_memory_roles = {
                demand.resource_role
                for demand in (
                    self.memory_node_demands
                    + self.launch_shared_memory_node_demands)
            }
            if exact_demand.resource_role not in stage_memory_roles:
                raise ValueError(
                    "replacing stage-local memory capacity requires bytes "
                    "on the exact completion resource")
        if has_absolute_duration:
            if self.completion_resource_roles is not None:
                raise ValueError(
                    "absolute-duration pipeline stages cannot declare "
                    "memory completion roles")
        elif has_compute_demands and (
                has_memory_node_demands
                or has_memory_access_class_demands
                or self.memory_exact_request_demands):
            raise ValueError(
                "pipeline stages cannot mix compute and memory-node demands")
        if has_absolute_duration:
            pass
        elif has_compute_demands:
            if self.work_fraction is not None:
                raise ValueError(
                    "typed compute pipeline stages must not declare a "
                    "work_fraction")
            if not self.service.startswith("compute:"):
                raise ValueError(
                    "typed compute pipeline stages require a compute service")
        elif has_memory_node_demands or has_memory_access_class_demands:
            if self.work_fraction is not None:
                raise ValueError(
                    "absolute memory pipeline stages must not declare a "
                    "work_fraction")
            if not self.service.startswith("memory:"):
                raise ValueError(
                    "absolute memory pipeline stages require a memory service")
            path_name = self.service.removeprefix("memory:")
            mismatched_paths = {
                value.path_name
                for value in (
                    self.memory_node_demands
                    + self.launch_shared_memory_node_demands)
                if value.path_name != path_name
            }
            if mismatched_paths:
                raise ValueError(
                    "pipeline-stage memory nodes must belong to service path "
                    f"{path_name!r}, got {sorted(mismatched_paths)}")
        else:
            if (self.completion_resource_roles is not None
                    and not self.service.startswith("memory:")):
                raise ValueError(
                    "PipelineStageDemand completion roles require memory "
                    "node demands")
            if self.work_fraction is None:
                raise ValueError(
                    "fractional pipeline stages require work_fraction")
            fraction = _require_finite(
                "PipelineStageDemand.work_fraction", self.work_fraction,
                positive=True)
            if fraction > 1.0:
                raise ValueError(
                    "PipelineStageDemand.work_fraction cannot exceed one")
            object.__setattr__(self, "work_fraction", fraction)
        if (self.memory_exact_request_demands
                and not self.service.startswith("memory:")):
            raise ValueError(
                "exact memory requests require a memory service")
        if self.include_first_request_latency and has_absolute_duration:
            raise ValueError(
                "absolute pipeline stages cannot include first-request "
                "latency")
        if (self.include_first_request_latency
                and not self.service.startswith("memory:")):
            raise ValueError(
                "first-request latency applies only to memory services")
        if (self.include_first_request_latency
                and (has_compute_demands
                     or has_memory_node_demands
                     or has_memory_access_class_demands
                     or self.memory_exact_request_demands)):
            raise ValueError(
                "absolute pipeline stages cannot include first-request "
                "latency")
        if (self.local_memory_source_completion is not None
                and not has_memory_node_demands):
            raise ValueError(
                "local memory source completion requires absolute memory "
                "node demands")
        if (self.local_memory_source_completion is not None
                and (self.include_first_request_latency
                     or has_compute_demands
                     or has_memory_access_class_demands
                     or self.memory_exact_request_demands
                     or has_absolute_duration)):
            raise ValueError(
                "local memory source completion cannot mix fractional, "
                "compute, exact, access-class, or absolute-duration "
                "completion")
        if self.predecessors is not None:
            if (not isinstance(self.predecessors, tuple)
                    or any(not isinstance(value, str)
                           for value in self.predecessors)):
                raise TypeError(
                    "PipelineStageDemand.predecessors must be a tuple of "
                    "names or None")
            normalized = tuple(
                _require_name("PipelineStageDemand.predecessors", value)
                for value in self.predecessors)
            if len(normalized) != len(set(normalized)):
                raise ValueError(
                    "PipelineStageDemand.predecessors cannot contain "
                    "duplicates")
            object.__setattr__(self, "predecessors", normalized)
        if (not isinstance(self.start_predecessors, tuple)
                or any(not isinstance(value, str)
                       for value in self.start_predecessors)):
            raise TypeError(
                "PipelineStageDemand.start_predecessors must be a tuple of "
                "names")
        normalized_start = tuple(
            _require_name(
                "PipelineStageDemand.start_predecessors", value)
            for value in self.start_predecessors)
        if len(normalized_start) != len(set(normalized_start)):
            raise ValueError(
                "PipelineStageDemand.start_predecessors cannot contain "
                "duplicates")
        if (self.predecessors is not None
                and set(normalized_start) & set(self.predecessors)):
            raise ValueError(
                "PipelineStageDemand predecessor cannot require both start "
                "and end")
        object.__setattr__(
            self, "start_predecessors", normalized_start)
        if (not isinstance(self.prior_resources, tuple)
                or any(not isinstance(value, str)
                       for value in self.prior_resources)):
            raise TypeError(
                "PipelineStageDemand.prior_resources must be a tuple of "
                "names")
        normalized_prior = tuple(
            _require_name(
                "PipelineStageDemand.prior_resources", value)
            for value in self.prior_resources)
        if len(normalized_prior) != len(set(normalized_prior)):
            raise ValueError(
                "PipelineStageDemand.prior_resources cannot contain "
                "duplicates")
        object.__setattr__(
            self, "prior_resources", normalized_prior)


@dataclass(frozen=True)
class AttentionPipelinePhaseDemand:
    """One finite source phase in an analytical attention CTA class.

    A phase is a source-program template, not a scheduled CTA or worker task.
    ``repeat_count`` represents the source KV-panel loop compactly.
    """

    name: str
    stages: Tuple[PipelineStageDemand, ...]
    repeat_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name",
            _require_name("AttentionPipelinePhaseDemand.name", self.name))
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError(
                "AttentionPipelinePhaseDemand.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError(
                "AttentionPipelinePhaseDemand.repeat_count must be positive")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise TypeError(
                "AttentionPipelinePhaseDemand.stages must be a non-empty tuple")
        if any(not isinstance(value, PipelineStageDemand)
               for value in self.stages):
            raise TypeError(
                "AttentionPipelinePhaseDemand.stages must contain "
                "PipelineStageDemand")
        names = tuple(value.name for value in self.stages)
        if len(names) != len(set(names)):
            raise ValueError(
                "AttentionPipelinePhaseDemand stage names must be unique")
        _validate_pipeline_stage_order(
            "AttentionPipelinePhaseDemand", self.stages)


@dataclass(frozen=True)
class AttentionPipelineClassDemand:
    """One analytically counted CTA source-program class.

    The exact zero-sized/no-phase form is reserved for an early-exit source
    ordinal that changes direct-launch worker assignment but contributes no
    pipeline work.
    """

    name: str
    count: int
    query_rows: int
    panel_count: int
    phases: Tuple[AttentionPipelinePhaseDemand, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name",
            _require_name("AttentionPipelineClassDemand.name", self.name))
        for field_name in ("count", "query_rows", "panel_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AttentionPipelineClassDemand.{field_name} must be an "
                    "integer")
        source_ordinal_only = (
            self.query_rows == 0
            and self.panel_count == 0
            and not self.phases)
        if self.count <= 0 or (
                not source_ordinal_only
                and (self.query_rows <= 0 or self.panel_count < 0)):
            raise ValueError(
                "attention pipeline class requires positive count/query_rows "
                "and non-negative panel_count, or the zero-sized "
                "source-ordinal marker")
        if not isinstance(self.phases, tuple):
            raise TypeError(
                "AttentionPipelineClassDemand.phases must be a tuple")
        if not source_ordinal_only and not self.phases:
            raise TypeError(
                "active AttentionPipelineClassDemand.phases must be non-empty")
        if any(not isinstance(value, AttentionPipelinePhaseDemand)
               for value in self.phases):
            raise TypeError(
                "AttentionPipelineClassDemand.phases must contain "
                "AttentionPipelinePhaseDemand")
        names = tuple(value.name for value in self.phases)
        if len(names) != len(set(names)):
            raise ValueError(
                "AttentionPipelineClassDemand phase names must be unique")


def _validate_attention_pipeline_buffers(classes, buffers) -> None:
    """Require one complete resident-credit lifetime in every CTA class."""

    events_by_program = {}
    for work_class in classes:
        if not work_class.phases:
            if work_class.query_rows != 0 or work_class.panel_count != 0:
                raise ValueError(
                    "phase-free attention classes must be source-ordinal "
                    "markers")
            if len(buffers) != 1:
                raise ValueError(
                    "attention source-ordinal markers require exactly one "
                    "resident lifecycle buffer")
            continue
        for buffer in buffers:
            if buffer.release != "completion":
                raise ValueError(
                    "attention pipeline buffers require completion release")
            program = (id(work_class.phases), id(buffer))
            if program not in events_by_program:
                events = []
                for phase in work_class.phases:
                    stage_names = tuple(stage.name for stage in phase.stages)
                    if buffer.producer_stage in stage_names:
                        if phase.repeat_count != 1:
                            raise ValueError(
                                f"PipelineBuffer {buffer.name!r} producer "
                                "phase cannot repeat")
                        events.append("producer")
                    if buffer.consumer_stage in stage_names:
                        if phase.repeat_count != 1:
                            raise ValueError(
                                f"PipelineBuffer {buffer.name!r} consumer "
                                "phase cannot repeat")
                        events.append("consumer")
                events_by_program[program] = tuple(events)
            events = events_by_program[program]
            if events != ("producer", "consumer"):
                raise ValueError(
                    f"PipelineBuffer {buffer.name!r} must have one ordered "
                    "producer/consumer lifetime in every attention class")


@dataclass(frozen=True)
class AttentionPipelineEnvelopeDemand:
    """Constant-size attention launch envelope with no CTA/worker objects."""

    active_lane_count: int
    classes: Tuple[AttentionPipelineClassDemand, ...]
    buffers: Tuple["PipelineBuffer", ...]
    source_repeat_count: int = 1
    cta_admission_policy: str = "round_robin"
    resident_tail_latency_service: Optional[str] = None
    resident_tail_requests_per_panel: int = 0
    resident_tail_group_interval_panels: int = 0
    resident_tail_requests_per_group: int = 0

    def __post_init__(self) -> None:
        if (isinstance(self.active_lane_count, bool)
                or not isinstance(self.active_lane_count, int)):
            raise TypeError(
                "AttentionPipelineEnvelopeDemand.active_lane_count must be "
                "an integer")
        if self.active_lane_count <= 0:
            raise ValueError(
                "AttentionPipelineEnvelopeDemand.active_lane_count must be "
                "positive")
        if not isinstance(self.classes, tuple) or not self.classes:
            raise TypeError(
                "AttentionPipelineEnvelopeDemand.classes must be a non-empty "
                "tuple")
        if any(not isinstance(value, AttentionPipelineClassDemand)
               for value in self.classes):
            raise TypeError(
                "AttentionPipelineEnvelopeDemand.classes must contain "
                "AttentionPipelineClassDemand")
        if (isinstance(self.source_repeat_count, bool)
                or not isinstance(self.source_repeat_count, int)
                or self.source_repeat_count <= 0):
            raise ValueError(
                "AttentionPipelineEnvelopeDemand.source_repeat_count must "
                "be a positive integer")
        if not isinstance(self.cta_admission_policy, str):
            raise TypeError(
                "AttentionPipelineEnvelopeDemand.cta_admission_policy must "
                "be a string")
        if self.cta_admission_policy not in (
                "round_robin", "paired_resident_fifo",
                "bounded_resident_suffix_fifo"):
            raise ValueError(
                "AttentionPipelineEnvelopeDemand.cta_admission_policy must "
                "be 'round_robin', 'paired_resident_fifo', or "
                "'bounded_resident_suffix_fifo'")
        names = tuple(value.name for value in self.classes)
        if len(names) != len(set(names)):
            raise ValueError(
                "AttentionPipelineEnvelopeDemand class names must be unique")
        if not isinstance(self.buffers, tuple):
            raise TypeError(
                "AttentionPipelineEnvelopeDemand.buffers must be a tuple")
        if any(not isinstance(value, PipelineBuffer) for value in self.buffers):
            raise TypeError(
                "AttentionPipelineEnvelopeDemand.buffers must contain "
                "PipelineBuffer")
        if any(value.scope != "worker" for value in self.buffers):
            raise ValueError(
                "attention pipeline envelope buffers must be lane-local")
        if self.resident_tail_latency_service is not None:
            object.__setattr__(
                self,
                "resident_tail_latency_service",
                _require_name(
                    "AttentionPipelineEnvelopeDemand."
                    "resident_tail_latency_service",
                    self.resident_tail_latency_service,
                ),
            )
        for field_name in (
                "resident_tail_requests_per_panel",
                "resident_tail_group_interval_panels",
                "resident_tail_requests_per_group"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"AttentionPipelineEnvelopeDemand.{field_name} must be "
                    "an integer")
            if value < 0:
                raise ValueError(
                    f"AttentionPipelineEnvelopeDemand.{field_name} cannot "
                    "be negative")
        if ((self.resident_tail_group_interval_panels == 0)
                != (self.resident_tail_requests_per_group == 0)):
            raise ValueError(
                "attention resident-tail group interval and requests must be "
                "configured together")
        has_tail_requests = (
            self.resident_tail_requests_per_panel > 0
            or self.resident_tail_requests_per_group > 0)
        if ((self.resident_tail_latency_service is None)
                != (not has_tail_requests)):
            raise ValueError(
                "attention resident-tail latency service and positive tail "
                "requests must be configured together")
        _validate_attention_pipeline_buffers(self.classes, self.buffers)


@dataclass(frozen=True)
class PipelineTaskDemand:
    """One launch-ordered tile pattern with unresolved service durations."""

    name: str
    worker: int
    stages: Tuple[PipelineStageDemand, ...]
    repeat_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_name("PipelineTaskDemand.name", self.name))
        if isinstance(self.worker, bool) or not isinstance(self.worker, int):
            raise TypeError("PipelineTaskDemand.worker must be an integer")
        if self.worker < 0:
            raise ValueError("PipelineTaskDemand.worker cannot be negative")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError(
                "PipelineTaskDemand.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError(
                "PipelineTaskDemand.repeat_count must be positive")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise TypeError(
                "PipelineTaskDemand.stages must be a non-empty tuple")
        if any(not isinstance(value, PipelineStageDemand)
               for value in self.stages):
            raise TypeError(
                "PipelineTaskDemand.stages must contain PipelineStageDemand")
        names = tuple(value.name for value in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("PipelineTaskDemand stage names must be unique")
        _validate_pipeline_stage_order("PipelineTaskDemand", self.stages)


@dataclass(frozen=True)
class PipelineTaskSequenceDemand:
    """Predictor-side repeated consecutive task pattern on one worker."""

    name: str
    worker: int
    tasks: Tuple[PipelineTaskDemand, ...]
    repeat_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name",
            _require_name("PipelineTaskSequenceDemand.name", self.name))
        if isinstance(self.worker, bool) or not isinstance(self.worker, int):
            raise TypeError(
                "PipelineTaskSequenceDemand.worker must be an integer")
        if self.worker < 0:
            raise ValueError(
                "PipelineTaskSequenceDemand.worker cannot be negative")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError(
                "PipelineTaskSequenceDemand.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError(
                "PipelineTaskSequenceDemand.repeat_count must be positive")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise TypeError(
                "PipelineTaskSequenceDemand.tasks must be a non-empty tuple")
        if any(not isinstance(value, PipelineTaskDemand)
               for value in self.tasks):
            raise TypeError(
                "PipelineTaskSequenceDemand.tasks must contain "
                "PipelineTaskDemand")
        if any(value.worker != self.worker for value in self.tasks):
            raise ValueError(
                "PipelineTaskSequenceDemand tasks must use the sequence worker")
        names = tuple(value.name for value in self.tasks)
        if len(names) != len(set(names)):
            raise ValueError(
                "PipelineTaskSequenceDemand task names must be unique")


@dataclass(frozen=True)
class PipelineTaskProgramDemand:
    """Predictor-side repeated references to task-sequence patterns.

    ``repeat_count`` is the exact number of consecutive repetitions of the
    complete ``order`` pattern on this worker; ``tail_order`` then executes
    once.
    """

    name: str
    worker: int
    patterns: Tuple[PipelineTaskSequenceDemand, ...]
    order: Tuple[int, ...]
    repeat_count: int = 1
    tail_order: Tuple[int, ...] = ()
    worker_multiplicity: int = 1
    physical_workers: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name",
            _require_name("PipelineTaskProgramDemand.name", self.name))
        if isinstance(self.worker, bool) or not isinstance(self.worker, int):
            raise TypeError(
                "PipelineTaskProgramDemand.worker must be an integer")
        if self.worker < 0:
            raise ValueError(
                "PipelineTaskProgramDemand.worker cannot be negative")
        if not isinstance(self.patterns, tuple) or not self.patterns:
            raise TypeError(
                "PipelineTaskProgramDemand.patterns must be a non-empty tuple")
        if any(not isinstance(value, PipelineTaskSequenceDemand)
               for value in self.patterns):
            raise TypeError(
                "PipelineTaskProgramDemand.patterns must contain "
                "PipelineTaskSequenceDemand")
        if any(value.worker != self.worker for value in self.patterns):
            raise ValueError(
                "PipelineTaskProgramDemand patterns must use the program worker")
        pattern_names = tuple(value.name for value in self.patterns)
        if len(pattern_names) != len(set(pattern_names)):
            raise ValueError(
                "PipelineTaskProgramDemand pattern names must be unique")
        if not isinstance(self.order, tuple) or not self.order:
            raise TypeError(
                "PipelineTaskProgramDemand.order must be a non-empty tuple")
        invalid_order = {
            value for value in self.order
            if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value >= len(self.patterns))
        }
        if invalid_order:
            raise ValueError(
                "PipelineTaskProgramDemand.order contains invalid pattern "
                f"indexes: {sorted(invalid_order, key=str)}")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError(
                "PipelineTaskProgramDemand.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError(
                "PipelineTaskProgramDemand.repeat_count must be positive")
        if (isinstance(self.worker_multiplicity, bool)
                or not isinstance(self.worker_multiplicity, int)):
            raise TypeError(
                "PipelineTaskProgramDemand.worker_multiplicity must be an "
                "integer")
        if self.worker_multiplicity <= 0:
            raise ValueError(
                "PipelineTaskProgramDemand.worker_multiplicity must be "
                "positive")
        if not isinstance(self.physical_workers, tuple):
            raise TypeError(
                "PipelineTaskProgramDemand.physical_workers must be a tuple")
        if self.physical_workers:
            if len(self.physical_workers) != self.worker_multiplicity:
                raise ValueError(
                    "PipelineTaskProgramDemand.physical_workers must match "
                    "worker_multiplicity")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in self.physical_workers):
                raise ValueError(
                    "PipelineTaskProgramDemand.physical_workers must contain "
                    "non-negative integers")
            if len(set(self.physical_workers)) != len(self.physical_workers):
                raise ValueError(
                    "PipelineTaskProgramDemand.physical_workers must be "
                    "unique")
        if not isinstance(self.tail_order, tuple):
            raise TypeError(
                "PipelineTaskProgramDemand.tail_order must be a tuple")
        invalid_tail_order = {
            value for value in self.tail_order
            if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value >= len(self.patterns))
        }
        if invalid_tail_order:
            raise ValueError(
                "PipelineTaskProgramDemand.tail_order contains invalid "
                f"pattern indexes: {sorted(invalid_tail_order, key=str)}")


@dataclass(frozen=True)
class BarrierPipelineEnvelopeDemand:
    """Tile-class histogram for worker-local programs with full barriers."""

    worker_count: int
    classes: Tuple[PipelineTaskSequenceDemand, ...]
    counts_by_worker: Tuple[Tuple[int, ...], ...]
    buffers: Tuple["PipelineBuffer", ...] = ()
    resolved_envelope: Optional["BarrierPipelineEnvelope"] = None

    def __post_init__(self) -> None:
        if (isinstance(self.worker_count, bool)
                or not isinstance(self.worker_count, int)):
            raise TypeError(
                "BarrierPipelineEnvelopeDemand.worker_count must be an integer")
        if self.worker_count <= 0:
            raise ValueError(
                "BarrierPipelineEnvelopeDemand.worker_count must be positive")
        if not isinstance(self.classes, tuple) or not self.classes:
            raise TypeError(
                "BarrierPipelineEnvelopeDemand.classes must be non-empty")
        if any(not isinstance(value, PipelineTaskSequenceDemand)
               for value in self.classes):
            raise TypeError(
                "BarrierPipelineEnvelopeDemand.classes must contain "
                "PipelineTaskSequenceDemand")
        if any(value.worker != 0 for value in self.classes):
            raise ValueError(
                "BarrierPipelineEnvelopeDemand classes must use template "
                "worker zero")
        names = tuple(value.name for value in self.classes)
        if len(names) != len(set(names)):
            raise ValueError(
                "BarrierPipelineEnvelopeDemand class names must be unique")
        if (not isinstance(self.counts_by_worker, tuple)
                or len(self.counts_by_worker) != self.worker_count):
            raise TypeError(
                "BarrierPipelineEnvelopeDemand counts must contain one row "
                "per worker")
        class_count = len(self.classes)
        for counts in self.counts_by_worker:
            if not isinstance(counts, tuple) or len(counts) != class_count:
                raise TypeError(
                    "BarrierPipelineEnvelopeDemand count rows must match "
                    "the class count")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in counts):
                raise ValueError(
                    "BarrierPipelineEnvelopeDemand counts must be "
                    "non-negative integers")
        if any(sum(row[index] for row in self.counts_by_worker) == 0
               for index in range(class_count)):
            raise ValueError(
                "BarrierPipelineEnvelopeDemand cannot contain an unused class")
        if not isinstance(self.buffers, tuple):
            raise TypeError(
                "BarrierPipelineEnvelopeDemand.buffers must be a tuple")
        if any(not isinstance(value, PipelineBuffer) for value in self.buffers):
            raise TypeError(
                "BarrierPipelineEnvelopeDemand.buffers must contain "
                "PipelineBuffer")
        if (self.resolved_envelope is not None
                and not isinstance(
                    self.resolved_envelope, BarrierPipelineEnvelope)):
            raise TypeError(
                "BarrierPipelineEnvelopeDemand resolved envelope must be "
                "typed or None")
        _validate_pipeline_pattern_contract(self.classes, self.buffers)
        services = {
            stage.service
            for sequence in self.classes
            for task in sequence.tasks
            for stage in task.stages}
        for sequence in self.classes:
            first_stage = sequence.tasks[0].stages[0]
            predecessors = (
                () if first_stage.predecessors is None
                else first_stage.predecessors)
            if predecessors or set(first_stage.prior_resources) != (
                    services - {first_stage.service}):
                raise ValueError(
                    "BarrierPipelineEnvelopeDemand classes must begin with a "
                    "full worker-resource barrier")


@dataclass(frozen=True)
class CannGroupedMatmulEnvelopeDemand:
    """CANN GEMM source tile program with analytical K recurrence.

    ``counts_by_worker`` conserves aggregate tile work while
    ``source_class_runs_by_worker`` retains the source-issued, compact RLE
    order needed by the controller-level asynchronous pipeline.  Ordinary
    MatMul instead consumes a phase-aware output-tile recurrence.  Neither
    representation materializes CTAs, and ``source_kind`` selects only the
    source-defined control boundary.
    """

    worker_count: int
    zero_expert_count: int
    class_shapes: Tuple[Tuple[int, int], ...]
    counts_by_worker: Tuple[Tuple[int, ...], ...]
    source_class_runs_by_worker: Tuple[Tuple[Tuple[int, int], ...], ...]
    projection_k: int
    projection_n: int
    tile_m: int
    tile_n: int
    tile_k: int
    a_step_k_tiles: int
    b_step_k_tiles: int
    a1_buffer_slots: int
    b1_buffer_slots: int
    l0a_buffer_slots: int
    l0b_buffer_slots: int
    l0c_buffer_slots: int
    dtype: str
    element_bytes: int
    tensor_operations: float
    a_l2_hit_fraction: float
    b_l2_hit_fraction: float
    source_kind: str = "grouped_matmul"
    output_path_name: str = "aic_output_l2_write"

    def __post_init__(self) -> None:
        integer_fields = (
            "worker_count", "projection_k", "projection_n", "tile_m",
            "tile_n", "tile_k",
            "a_step_k_tiles", "b_step_k_tiles", "a1_buffer_slots",
            "b1_buffer_slots", "l0a_buffer_slots", "l0b_buffer_slots",
            "l0c_buffer_slots", "element_bytes")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"CannGroupedMatmulEnvelopeDemand.{name} must be integer")
            if value <= 0:
                raise ValueError(
                    f"CannGroupedMatmulEnvelopeDemand.{name} must be positive")
        if (isinstance(self.zero_expert_count, bool)
                or not isinstance(self.zero_expert_count, int)):
            raise TypeError(
                "CannGroupedMatmulEnvelopeDemand.zero_expert_count must be "
                "integer")
        if self.zero_expert_count < 0:
            raise ValueError(
                "CannGroupedMatmulEnvelopeDemand.zero_expert_count cannot be "
                "negative")
        if self.a_step_k_tiles % self.b_step_k_tiles:
            raise ValueError(
                "CANN GroupedMatmul A K step must contain complete B steps")
        if self.l0a_buffer_slots != 2 or self.l0b_buffer_slots != 2:
            raise ValueError(
                "CANN GroupedMatmul envelope requires captured L0 ping-pong")
        _require_name("CannGroupedMatmulEnvelopeDemand.dtype", self.dtype)
        if not isinstance(self.class_shapes, tuple) or not self.class_shapes:
            raise TypeError(
                "CannGroupedMatmulEnvelopeDemand class shapes must be non-empty")
        if any(
                not isinstance(shape, tuple) or len(shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in shape)
                or shape[0] > self.tile_m or shape[1] > self.tile_n
                for shape in self.class_shapes):
            raise ValueError(
                "CANN GroupedMatmul envelope class shapes are invalid")
        if len(self.class_shapes) != len(set(self.class_shapes)):
            raise ValueError(
                "CANN GroupedMatmul envelope class shapes cannot repeat")
        if (not isinstance(self.counts_by_worker, tuple)
                or len(self.counts_by_worker) != self.worker_count):
            raise TypeError(
                "CANN GroupedMatmul envelope requires one count row per worker")
        for counts in self.counts_by_worker:
            if (not isinstance(counts, tuple)
                    or len(counts) != len(self.class_shapes)
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           or value < 0 for value in counts)):
                raise ValueError(
                    "CANN GroupedMatmul envelope class counts are invalid")
        if (not isinstance(self.source_class_runs_by_worker, tuple)
                or len(self.source_class_runs_by_worker) != self.worker_count):
            raise TypeError(
                "CANN GroupedMatmul envelope requires one source RLE per "
                "worker")
        for worker, (counts, runs) in enumerate(zip(
                self.counts_by_worker, self.source_class_runs_by_worker)):
            if not isinstance(runs, tuple):
                raise ValueError(
                    "CANN GroupedMatmul worker source RLE must be a tuple")
            if not runs:
                if any(counts):
                    raise ValueError(
                        "CANN GroupedMatmul populated worker has an empty "
                        "source RLE")
                continue
            replay_counts = [0] * len(self.class_shapes)
            previous_class_index = None
            for run in runs:
                if (not isinstance(run, tuple) or len(run) != 2
                        or any(isinstance(value, bool)
                               or not isinstance(value, int)
                               for value in run)):
                    raise TypeError(
                        "CANN GroupedMatmul source RLE entries must be "
                        "integer pairs")
                class_index, repeat_count = run
                if class_index < 0 or class_index >= len(self.class_shapes):
                    raise ValueError(
                        "CANN GroupedMatmul source RLE class is out of range")
                if repeat_count <= 0:
                    raise ValueError(
                        "CANN GroupedMatmul source RLE repeat must be positive")
                if class_index == previous_class_index:
                    raise ValueError(
                        "CANN GroupedMatmul source RLE has adjacent equal "
                        "classes")
                replay_counts[class_index] += repeat_count
                previous_class_index = class_index
            if tuple(replay_counts) != counts:
                raise ValueError(
                    "CANN GroupedMatmul source RLE disagrees with worker "
                    f"histogram for worker {worker}")
        if any(sum(row[index] for row in self.counts_by_worker) == 0
               for index in range(len(self.class_shapes))):
            raise ValueError(
                "CANN GroupedMatmul envelope contains an unused class")
        if (isinstance(self.tensor_operations, bool)
                or not isinstance(self.tensor_operations, (int, float))
                or not math.isfinite(self.tensor_operations)
                or self.tensor_operations <= 0.0):
            raise ValueError(
                "CANN GroupedMatmul envelope Tensor work must be positive")
        for name in ("a_l2_hit_fraction", "b_l2_hit_fraction"):
            value = getattr(self, name)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0.0 <= value <= 1.0):
                raise ValueError(
                    f"CANN GroupedMatmul envelope {name} must be in [0,1]")
        if self.source_kind not in (
                "grouped_matmul", "matmul_v2", "matmul_v3"):
            raise ValueError(
                "CANN GEMM envelope source_kind must identify GroupedMatmul, "
                "MatMulV2, or MatMulV3")
        _require_name(
            "CannGroupedMatmulEnvelopeDemand.output_path_name",
            self.output_path_name)


@dataclass(frozen=True)
class CannSingleCoreSplitKEnvelopeDemand:
    """Compact source program for CANN's single-core split-K GEMM.

    Each class is one physical AIC output block.  The formula retains the
    source A/B prefetch order and buffer credits while folding repeated
    inner-M stripes analytically; no production tile stream is materialized.
    """

    worker_count: int
    class_shapes: Tuple[Tuple[int, int], ...]
    class_multiplicities: Tuple[int, ...]
    projection_k: int
    single_core_k: int
    base_m: int
    base_n: int
    base_k: int
    step_m: int
    step_n: int
    a1_buffer_slots: int
    b1_buffer_slots: int
    partial_element_bytes: int
    dtype: str
    element_bytes: int

    def __post_init__(self) -> None:
        integer_fields = (
            "worker_count", "projection_k", "single_core_k", "base_m",
            "base_n", "base_k", "step_m", "step_n",
            "a1_buffer_slots", "b1_buffer_slots",
            "partial_element_bytes", "element_bytes")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"CannSingleCoreSplitKEnvelopeDemand.{name} must be "
                    "integer")
            if value <= 0:
                raise ValueError(
                    f"CannSingleCoreSplitKEnvelopeDemand.{name} must be "
                    "positive")
        if self.single_core_k % self.base_k:
            raise ValueError(
                "CANN single-core split-K K group must contain base tiles")
        if (self.step_n != 1
                or self.a1_buffer_slots != self.step_m
                or self.b1_buffer_slots != 2):
            raise ValueError(
                "CANN single-core split-K envelope requires the captured "
                "A-slot and B ping-pong topology")
        if not isinstance(self.class_shapes, tuple) or not self.class_shapes:
            raise TypeError(
                "CANN single-core split-K class shapes must be non-empty")
        if (not isinstance(self.class_multiplicities, tuple)
                or len(self.class_multiplicities) != len(self.class_shapes)):
            raise TypeError(
                "CANN single-core split-K multiplicities must match classes")
        for shape in self.class_shapes:
            if (not isinstance(shape, tuple) or len(shape) != 2
                    or any(isinstance(value, bool)
                           or not isinstance(value, int) or value <= 0
                           for value in shape)):
                raise ValueError(
                    "CANN single-core split-K class shape is invalid")
            if shape[0] % self.base_m or shape[1] % self.base_n:
                raise ValueError(
                    "CANN single-core split-K classes must contain complete "
                    "resident tiles")
        if len(self.class_shapes) != len(set(self.class_shapes)):
            raise ValueError(
                "CANN single-core split-K class shapes cannot repeat")
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value <= 0 for value in self.class_multiplicities):
            raise ValueError(
                "CANN single-core split-K multiplicities must be positive")
        if sum(self.class_multiplicities) != self.worker_count:
            raise ValueError(
                "CANN single-core split-K classes must cover every worker")
        _require_name("CannSingleCoreSplitKEnvelopeDemand.dtype", self.dtype)


@dataclass(frozen=True)
class GpuGroupedGemmPipelineEnvelopeDemand:
    """Constant-size grouped-GEMM resident-wave pipeline demand."""

    active_ctas: int
    k_tiles: int
    resident_ctas_per_sm: int
    dram_read_bytes: float
    # Names a ``grouped_gemm_fp8_block_tile_pipelines`` HWSpec row; null keeps
    # the single BF16 ``grouped_gemm_tile_pipeline`` mechanism.
    fp8_block_program: Optional[str] = None

    def __post_init__(self) -> None:
        if self.fp8_block_program is not None and (
                not isinstance(self.fp8_block_program, str)
                or not self.fp8_block_program):
            raise ValueError(
                "GpuGroupedGemmPipelineEnvelopeDemand.fp8_block_program "
                "must be a non-empty string or null")
        for name in ("active_ctas", "k_tiles", "resident_ctas_per_sm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"GpuGroupedGemmPipelineEnvelopeDemand.{name} must be "
                    "an integer")
            if value <= 0:
                raise ValueError(
                    f"GpuGroupedGemmPipelineEnvelopeDemand.{name} must be "
                    "positive")
        if (isinstance(self.dram_read_bytes, bool)
                or not isinstance(self.dram_read_bytes, (int, float))
                or not math.isfinite(self.dram_read_bytes)
                or self.dram_read_bytes <= 0.0):
            raise ValueError(
                "GpuGroupedGemmPipelineEnvelopeDemand.dram_read_bytes must "
                "be finite and positive")


# One HWSpec mechanism per independently measured Triton grouped-MoE tile
# signature.  Each key names the ``scheduling`` entry that carries that
# signature's synthetic-routing microbenchmark coefficients; a launch without
# such a key has no measured operand pipeline and must not borrow another
# signature's coefficients.
METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS = (
    "metax_fused_moe_bm16_operand_pipeline",
    "metax_fused_moe_bm32_operand_pipeline",
    "metax_fused_moe_bm32_bn128_operand_pipeline",
    "metax_fused_moe_bm64_bn128_warps4_operand_pipeline",
    "metax_fused_moe_bm64_bn128_warps8_operand_pipeline",
    "metax_fused_moe_bm128_bn128_operand_pipeline",
)


@dataclass(frozen=True)
class MetaxGroupedOperandPipelineDemand:
    """C500 grouped-GEMM per-CTA serial operand-pipeline demand."""

    active_ctas: int
    k_tiles: int
    dram_read_bytes: float
    mechanism_key: str

    def __post_init__(self) -> None:
        if self.mechanism_key not in (
                METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS):
            raise ValueError(
                "MetaxGroupedOperandPipelineDemand.mechanism_key must be one "
                f"of {METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS}")
        for name in ("active_ctas", "k_tiles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"MetaxGroupedOperandPipelineDemand.{name} must "
                    "be an integer")
            if value <= 0:
                raise ValueError(
                    f"MetaxGroupedOperandPipelineDemand.{name} must "
                    "be positive")
        if (isinstance(self.dram_read_bytes, bool)
                or not isinstance(self.dram_read_bytes, (int, float))
                or not math.isfinite(self.dram_read_bytes)
                or self.dram_read_bytes <= 0.0):
            raise ValueError(
                "MetaxGroupedOperandPipelineDemand.dram_read_bytes "
                "must be finite and positive")


@dataclass(frozen=True)
class BarrierPipelineEnvelope:
    """Resolved constant-size worker/tile-class pipeline envelope."""

    worker_count: int
    classes: Tuple[PipelineTaskSequence, ...]
    counts_by_worker: Tuple[Tuple[int, ...], ...]
    resources: Tuple[PipelineResource, ...]
    buffers: Tuple["PipelineBuffer", ...] = ()
    class_durations_s: Tuple[float, ...] = ()
    class_stage_service_s: Tuple[Tuple[Tuple[str, float], ...], ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.class_durations_s, tuple)
                or (self.class_durations_s
                    and len(self.class_durations_s) != len(self.classes))):
            raise TypeError(
                "BarrierPipelineEnvelope class durations must match classes")
        if any(not isinstance(value, (int, float))
               or isinstance(value, bool)
               or not math.isfinite(value)
               or value <= 0.0
               for value in self.class_durations_s):
            raise ValueError(
                "BarrierPipelineEnvelope class durations must be positive")
        if (not isinstance(self.class_stage_service_s, tuple)
                or (self.class_stage_service_s
                    and len(self.class_stage_service_s) != len(self.classes))):
            raise TypeError(
                "BarrierPipelineEnvelope class stage services must match "
                "classes")


@dataclass(frozen=True)
class PipelineWorkPlan:
    """Predictor-side worker assignment and per-tile service shares."""

    worker_count: int
    tasks: Tuple[PipelineTaskDemand, ...]
    buffers: Tuple[PipelineBuffer, ...] = ()
    initial_resource_availability_s: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.worker_count, bool) or not isinstance(
                self.worker_count, int):
            raise TypeError("PipelineWorkPlan.worker_count must be an integer")
        if self.worker_count <= 0:
            raise ValueError("PipelineWorkPlan.worker_count must be positive")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise TypeError("PipelineWorkPlan.tasks must be a non-empty tuple")
        if any(not isinstance(
                value, (
                    PipelineTaskDemand, PipelineTaskSequenceDemand,
                    PipelineTaskProgramDemand))
               for value in self.tasks):
            raise TypeError(
                "PipelineWorkPlan.tasks must contain PipelineTaskDemand or "
                "PipelineTaskSequenceDemand or PipelineTaskProgramDemand")
        task_names = tuple(value.name for value in self.tasks)
        if len(task_names) != len(set(task_names)):
            raise ValueError("PipelineWorkPlan task names must be unique")
        invalid_workers = {
            task.worker for task in self.tasks
            if task.worker >= self.worker_count
        }
        if invalid_workers:
            raise ValueError(
                "PipelineWorkPlan tasks reference workers outside "
                f"worker_count: {sorted(invalid_workers)}")
        if not isinstance(self.buffers, tuple):
            raise TypeError("PipelineWorkPlan.buffers must be a tuple")
        if any(not isinstance(value, PipelineBuffer) for value in self.buffers):
            raise TypeError(
                "PipelineWorkPlan.buffers must contain PipelineBuffer")
        buffer_names = tuple(value.name for value in self.buffers)
        if len(buffer_names) != len(set(buffer_names)):
            raise ValueError("PipelineWorkPlan buffer names must be unique")
        if not isinstance(self.initial_resource_availability_s, tuple):
            raise TypeError(
                "PipelineWorkPlan initial resource availability must be a "
                "tuple")
        initial_names = []
        prior_resources = {
            resource
            for entry in self.tasks
            for pattern in _pipeline_entry_patterns(entry)
            for task in pattern
            for stage in task.stages
            for resource in stage.prior_resources
        }
        for name, duration_s in self.initial_resource_availability_s:
            initial_names.append(_require_name(
                "PipelineWorkPlan initial resource", name))
            _require_finite(
                "PipelineWorkPlan initial resource availability",
                duration_s, positive=True)
            if name not in prior_resources:
                raise ValueError(
                    "PipelineWorkPlan initial availability must feed a "
                    f"source dependency: {name!r}")
        if len(initial_names) != len(set(initial_names)):
            raise ValueError(
                "PipelineWorkPlan initial resource availability cannot "
                "repeat resources")
        _validate_pipeline_pattern_contract(self.tasks, self.buffers)


@dataclass(frozen=True)
class FiaBlockReduceSumDescriptorDemand:
    """One source-level CalcLocalRowSum invocation on an AIV."""

    row_chunk_round: int
    column_count: int
    column_alignment: int
    count: int = 1

    def __post_init__(self) -> None:
        for name in (
                "row_chunk_round", "column_count", "column_alignment",
                "count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"FiaBlockReduceSumDescriptorDemand.{name} must be "
                    "integer")
            if value <= 0:
                raise ValueError(
                    f"FiaBlockReduceSumDescriptorDemand.{name} must be "
                    "positive")


@dataclass(frozen=True)
class FiaComputeStageDemand:
    """Physical work owned by one stage of a source-closed FIA task."""

    resource_role: str
    physical_worker: int
    logical_operations: float
    physical_operations: float
    typed_services: Tuple[ComputeServiceDemand, ...] = ()

    def __post_init__(self) -> None:
        if self.resource_role not in ("compute_tensor", "compute_vector"):
            raise ValueError(
                "FiaComputeStageDemand.resource_role must be compute_tensor "
                "or compute_vector")
        if (isinstance(self.physical_worker, bool)
                or not isinstance(self.physical_worker, int)):
            raise TypeError(
                "FiaComputeStageDemand.physical_worker must be an integer")
        if self.physical_worker < 0:
            raise ValueError(
                "FiaComputeStageDemand.physical_worker cannot be negative")
        for name in ("logical_operations", "physical_operations"):
            object.__setattr__(
                self, name,
                _require_finite(
                    f"FiaComputeStageDemand.{name}", getattr(self, name),
                    positive=False))
        if self.physical_operations < self.logical_operations:
            raise ValueError(
                "FiaComputeStageDemand physical work cannot be smaller than "
                "logical work")
        if not isinstance(self.typed_services, tuple):
            raise TypeError(
                "FiaComputeStageDemand.typed_services must be a tuple")
        if any(not isinstance(value, ComputeServiceDemand)
               for value in self.typed_services):
            raise TypeError(
                "FiaComputeStageDemand.typed_services must contain "
                "ComputeServiceDemand")
        names = tuple(value.service for value in self.typed_services)
        if len(names) != len(set(names)):
            raise ValueError(
                "FiaComputeStageDemand.typed_services cannot repeat services")
        if any(value.operations_by_worker for value in self.typed_services):
            raise ValueError(
                "FIA stage-local typed services cannot contain another "
                "worker schedule")
        if self.resource_role == "compute_tensor":
            if self.typed_services:
                raise ValueError(
                    "FIA Tensor stage must use untyped physical work")
        elif self.resource_role == "compute_vector":
            typed_operations = math.fsum(
                value.operations for value in self.typed_services)
            if not math.isclose(
                    typed_operations, self.logical_operations,
                    rel_tol=0.0,
                    abs_tol=max(
                        math.ulp(typed_operations),
                        math.ulp(self.logical_operations))):
                raise ValueError(
                    "FIA Vector stage typed services must conserve logical "
                    "work")
            if self.physical_operations != self.logical_operations:
                raise ValueError(
                    "FIA Vector stage physical and logical work must agree")
        else:
            raise AssertionError(
                "FIA stage resource validation lost its closed domain")


@dataclass(frozen=True)
class FiaMte2Nd2NzDescriptorDemand:
    """Repeated source-level AIC GM-to-L1 ND2NZ descriptor geometry."""

    rows: int
    columns: int
    element_bytes: int
    count: int
    source_row_stride_elements: Optional[int] = None
    block_table_lookup_count: int = 0

    def __post_init__(self) -> None:
        for name in ("rows", "columns", "element_bytes", "count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"FiaMte2Nd2NzDescriptorDemand.{name} must be integer")
            if value <= 0:
                raise ValueError(
                    f"FiaMte2Nd2NzDescriptorDemand.{name} must be positive")
        if self.rows > 512 or self.columns > 256:
            raise ValueError(
                "FIA MTE2 ND2NZ descriptor is outside the measured geometry")
        if self.element_bytes != 2:
            raise ValueError("FIA MTE2 ND2NZ service currently requires BF16")
        if self.source_row_stride_elements is not None:
            if (isinstance(self.source_row_stride_elements, bool)
                    or not isinstance(
                        self.source_row_stride_elements, int)):
                raise TypeError(
                    "FiaMte2Nd2NzDescriptorDemand."
                    "source_row_stride_elements must be an integer or None")
            if self.source_row_stride_elements < self.columns:
                raise ValueError(
                    "FIA MTE2 ND2NZ source row stride cannot be smaller "
                    "than the copied column count")
        if (isinstance(self.block_table_lookup_count, bool)
                or not isinstance(self.block_table_lookup_count, int)):
            raise TypeError(
                "FIA MTE2 ND2NZ block_table_lookup_count must be integer")
        if self.block_table_lookup_count < 0:
            raise ValueError(
                "FIA MTE2 ND2NZ block_table_lookup_count cannot be negative")

    @property
    def bytes_per_descriptor(self) -> int:
        return self.rows * self.columns * self.element_bytes

    @property
    def total_bytes(self) -> int:
        return self.count * self.bytes_per_descriptor


@dataclass(frozen=True)
class FiaMte2StageDemand:
    """Compressed AIC-local MTE2 descriptor sequence for one FIA stage."""

    physical_worker: int
    descriptors: Tuple[FiaMte2Nd2NzDescriptorDemand, ...]

    def __post_init__(self) -> None:
        if (isinstance(self.physical_worker, bool)
                or not isinstance(self.physical_worker, int)
                or self.physical_worker < 0):
            raise ValueError(
                "FiaMte2StageDemand.physical_worker must be a "
                "non-negative integer")
        if not isinstance(self.descriptors, tuple) or not self.descriptors:
            raise TypeError(
                "FiaMte2StageDemand.descriptors must be a non-empty tuple")
        if any(not isinstance(value, FiaMte2Nd2NzDescriptorDemand)
               for value in self.descriptors):
            raise TypeError(
                "FiaMte2StageDemand.descriptors must contain typed demands")

    @property
    def total_bytes(self) -> int:
        return sum(value.total_bytes for value in self.descriptors)


@dataclass(frozen=True)
class FiaMte1LoadPairDemand:
    """Repeated source-shaped L0A/L0B load pair completed on one AIC."""

    stage_kind: str
    head_dim: int
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.stage_kind, str) or not self.stage_kind:
            raise TypeError(
                "FiaMte1LoadPairDemand.stage_kind must be a non-empty string")
        for name in ("head_dim", "count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"FiaMte1LoadPairDemand.{name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"FiaMte1LoadPairDemand.{name} must be positive")


@dataclass(frozen=True)
class FiaMte1StageDemand:
    """Source-derived L1-to-L0A/L0B traffic for one FIA MM stage."""

    physical_worker: int
    l0a_bytes: int
    l0b_bytes: int
    load_pair: Optional[FiaMte1LoadPairDemand] = None

    def __post_init__(self) -> None:
        for name in ("physical_worker", "l0a_bytes", "l0b_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"FiaMte1StageDemand.{name} must be integer")
            if value < 0:
                raise ValueError(
                    f"FiaMte1StageDemand.{name} cannot be negative")
        if (self.load_pair is not None
                and not isinstance(self.load_pair, FiaMte1LoadPairDemand)):
            raise TypeError(
                "FiaMte1StageDemand.load_pair must be a typed demand")


@dataclass(frozen=True)
class FiaAivMemoryStageDemand:
    """AIV-local GM/L2-to-UB ingress and UB-to-GM/L2 egress."""

    physical_worker: int
    mte2_bytes: int
    mte3_bytes: int
    completion_service: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("physical_worker", "mte2_bytes", "mte3_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"FiaAivMemoryStageDemand.{name} must be integer")
            if value < 0:
                raise ValueError(
                    f"FiaAivMemoryStageDemand.{name} cannot be negative")
        if (self.completion_service is not None
                and (not isinstance(self.completion_service, str)
                     or not self.completion_service)):
            raise TypeError(
                "FiaAivMemoryStageDemand.completion_service must be None "
                "or a non-empty string")


@dataclass(frozen=True)
class FiaFixpipeDescriptorDemand:
    """One completed row-major L0C-to-GM Fixpipe call geometry."""

    rows: int
    columns: int
    destination_stride_elements: int
    count: int

    def __post_init__(self) -> None:
        for name in (
                "rows", "columns", "destination_stride_elements", "count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"FiaFixpipeDescriptorDemand.{name} must be integer")
            if value <= 0:
                raise ValueError(
                    f"FiaFixpipeDescriptorDemand.{name} must be positive")
        if self.destination_stride_elements < self.columns:
            raise ValueError(
                "FiaFixpipeDescriptorDemand destination stride cannot be "
                "smaller than the row width")


@dataclass(frozen=True)
class FiaFixpipeStageDemand:
    """Source-issued Fixpipe calls completed before one FIA handoff."""

    physical_worker: int
    descriptors: Tuple[FiaFixpipeDescriptorDemand, ...]

    def __post_init__(self) -> None:
        if (isinstance(self.physical_worker, bool)
                or not isinstance(self.physical_worker, int)
                or self.physical_worker < 0):
            raise ValueError(
                "FiaFixpipeStageDemand.physical_worker must be a "
                "non-negative integer")
        if not isinstance(self.descriptors, tuple):
            raise TypeError(
                "FiaFixpipeStageDemand.descriptors must be a tuple")
        if any(not isinstance(value, FiaFixpipeDescriptorDemand)
               for value in self.descriptors):
            raise TypeError(
                "FiaFixpipeStageDemand.descriptors must contain typed "
                "demands")

    @property
    def total_calls(self) -> int:
        return sum(value.count for value in self.descriptors)


@dataclass(frozen=True)
class FiaGqaSourcePanelDemand:
    """One literal 128-column-or-short regular-GQA Cube panel.

    This is a structural source view, not a time model.  It preserves the
    exact L1-to-L0, Tensor, and Fixpipe work which CANN issues after a K/P/V
    L1 round, including a short final panel that has no full-width Fixpipe.
    """

    source_columns: int
    mm1_mte1: FiaMte1StageDemand
    mm1: FiaComputeStageDemand
    mm1_fixpipe: FiaFixpipeStageDemand
    mm2_mte1: FiaMte1StageDemand
    mm2: FiaComputeStageDemand

    def __post_init__(self) -> None:
        if (isinstance(self.source_columns, bool)
                or not isinstance(self.source_columns, int)):
            raise TypeError(
                "FIA GQA source panel columns must be an integer")
        if self.source_columns <= 0 or self.source_columns > 128:
            raise ValueError(
                "FIA GQA source panel columns must be in [1, 128]")
        for name, demand_type in (
                ("mm1_mte1", FiaMte1StageDemand),
                ("mm1", FiaComputeStageDemand),
                ("mm1_fixpipe", FiaFixpipeStageDemand),
                ("mm2_mte1", FiaMte1StageDemand),
                ("mm2", FiaComputeStageDemand)):
            if not isinstance(getattr(self, name), demand_type):
                raise TypeError(
                    f"FIA GQA source panel {name} must be a typed demand")
        if (self.mm1.resource_role != "compute_tensor"
                or self.mm2.resource_role != "compute_tensor"):
            raise ValueError(
                "FIA GQA source panels require Tensor MM1/MM2 work")
        workers = (
            self.mm1_mte1.physical_worker,
            self.mm1.physical_worker,
            self.mm1_fixpipe.physical_worker,
            self.mm2_mte1.physical_worker,
            self.mm2.physical_worker,
        )
        if len(set(workers)) != 1:
            raise ValueError(
                "FIA GQA source panel stages must belong to one AIC")
        if self.mm1_mte1.load_pair is not None or self.mm2_mte1.load_pair is not None:
            raise ValueError(
                "FIA GQA source panels must expose literal bandwidth work, "
                "not aggregate load-pair completion")


@dataclass(frozen=True)
class FiaGqaSourceRoundDemand:
    """One literal regular-GQA K/P/V L1 round and its Cube panels.

    ``FiaMte2StageDemand`` compresses consecutive equal descriptors for the
    aggregate service ledger.  The CANN source program releases a KP or V L1
    slot after each logical K round, including a final short round.  This view
    retains both that order and the exact non-uniform panel work without a
    duration parameter.
    """

    key_descriptors: Tuple[FiaMte2Nd2NzDescriptorDemand, ...]
    probability_descriptor: FiaMte2Nd2NzDescriptorDemand
    value_descriptors: Tuple[FiaMte2Nd2NzDescriptorDemand, ...]
    panels: Tuple[FiaGqaSourcePanelDemand, ...]

    def __post_init__(self) -> None:
        for name in ("key_descriptors", "value_descriptors"):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values
                    or any(not isinstance(value, FiaMte2Nd2NzDescriptorDemand)
                           for value in values)):
                raise TypeError(
                    f"FIA GQA source round {name} must be a non-empty "
                    "descriptor tuple")
            if any(value.count != 1 for value in values):
                raise ValueError(
                    f"FIA GQA source round {name} must retain literal "
                    "one-copy descriptors")
        if not isinstance(
                self.probability_descriptor, FiaMte2Nd2NzDescriptorDemand):
            raise TypeError(
                "FIA GQA source round probability must be an MTE2 "
                "descriptor")
        if self.probability_descriptor.count != 1:
            raise ValueError(
                "FIA GQA source round probability must retain one literal "
                "copy")
        if (not isinstance(self.panels, tuple) or not self.panels
                or any(not isinstance(value, FiaGqaSourcePanelDemand)
                       for value in self.panels)):
            raise TypeError(
                "FIA GQA source round panels must be a non-empty typed "
                "tuple")
        source_columns = self.probability_descriptor.columns
        if sum(value.rows for value in self.key_descriptors) != source_columns:
            raise ValueError(
                "FIA GQA source round K descriptors must cover its P "
                "columns exactly")
        if sum(value.rows for value in self.value_descriptors) != source_columns:
            raise ValueError(
                "FIA GQA source round V descriptors must cover its P "
                "columns exactly")
        if sum(value.source_columns for value in self.panels) != source_columns:
            raise ValueError(
                "FIA GQA source panels must cover their P columns exactly")


@dataclass(frozen=True)
class FiaGqaSourceOrderDemand:
    """Literal regular-GQA ingress order extracted from CANN source loops."""

    query_descriptor: Optional[FiaMte2Nd2NzDescriptorDemand]
    rounds: Tuple[FiaGqaSourceRoundDemand, ...]

    def __post_init__(self) -> None:
        if (self.query_descriptor is not None
                and not isinstance(
                    self.query_descriptor, FiaMte2Nd2NzDescriptorDemand)):
            raise TypeError(
                "FIA GQA source-order query must be an MTE2 descriptor or "
                "None")
        if (self.query_descriptor is not None
                and self.query_descriptor.count != 1):
            raise ValueError(
                "FIA GQA source-order query must retain one literal copy")
        if (not isinstance(self.rounds, tuple) or not self.rounds
                or any(not isinstance(value, FiaGqaSourceRoundDemand)
                       for value in self.rounds)):
            raise TypeError(
                "FIA GQA source-order rounds must be a non-empty typed "
                "tuple")


@dataclass(frozen=True)
class FiaMlaSourceOrderDemand:
    """Literal CANN MLA MTE2 source calls before aggregate coalescing.

    The CANN MLA program has heterogeneous K/no-PE and K/RoPE copies.  Its
    aggregate MTE2 stage may coalesce adjacent equal copies, but that loses
    the MTE2->MTE1 pair boundary needed to resolve the source pipeline.
    """

    mm1_descriptors: Tuple[FiaMte2Nd2NzDescriptorDemand, ...]
    mm2_descriptors: Tuple[FiaMte2Nd2NzDescriptorDemand, ...]

    def __post_init__(self) -> None:
        for name in ("mm1_descriptors", "mm2_descriptors"):
            descriptors = getattr(self, name)
            if (not isinstance(descriptors, tuple) or not descriptors
                    or any(not isinstance(
                        descriptor, FiaMte2Nd2NzDescriptorDemand)
                           for descriptor in descriptors)):
                raise TypeError(
                    f"FIA MLA source {name} must be a non-empty typed tuple")
            if any(descriptor.count != 1 for descriptor in descriptors):
                raise ValueError(
                    f"FIA MLA source {name} must retain literal copies")


@dataclass(frozen=True)
class FiaCrossDomainTaskDemand:
    """One run of identical typed AIC/AIV source tasks."""

    ordinal: int
    mm1_mte2: FiaMte2StageDemand
    mm1_mte1: FiaMte1StageDemand
    mm1: FiaComputeStageDemand
    mm1_fixpipe: FiaFixpipeStageDemand
    vec1: FiaComputeStageDemand
    mm2_mte2: FiaMte2StageDemand
    mm2_mte1: FiaMte1StageDemand
    mm2: FiaComputeStageDemand
    mm2_fixpipe: FiaFixpipeStageDemand
    vec2: FiaComputeStageDemand
    vec1_memory: Optional[FiaAivMemoryStageDemand] = None
    vec2_memory: Optional[FiaAivMemoryStageDemand] = None
    vec1_update_memory: Optional[FiaAivMemoryStageDemand] = None
    vec1_peer_memory: Optional[FiaAivMemoryStageDemand] = None
    vec1_peer_update_memory: Optional[FiaAivMemoryStageDemand] = None
    vec2_peer_memory: Optional[FiaAivMemoryStageDemand] = None
    vec1_block_reduce_sum: Tuple[
        FiaBlockReduceSumDescriptorDemand, ...] = ()
    vec1_update: Optional[FiaComputeStageDemand] = None
    vec1_peer: Optional[FiaComputeStageDemand] = None
    vec1_peer_update: Optional[FiaComputeStageDemand] = None
    vec2_peer: Optional[FiaComputeStageDemand] = None
    vec1_peer_block_reduce_sum: Tuple[
        FiaBlockReduceSumDescriptorDemand, ...] = ()
    gqa_source_order: Optional[FiaGqaSourceOrderDemand] = None
    mla_source_order: Optional[FiaMlaSourceOrderDemand] = None
    aic_pipeline_kind: str = "serial_stages"
    repeat_count: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("FiaCrossDomainTaskDemand.ordinal must be integer")
        if self.ordinal < 0:
            raise ValueError(
                "FiaCrossDomainTaskDemand.ordinal cannot be negative")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)
                or self.repeat_count <= 0):
            raise ValueError(
                "FiaCrossDomainTaskDemand.repeat_count must be a positive "
                "integer")
        if self.aic_pipeline_kind not in (
                "serial_stages",
                "fia_block_cube_nonquant_mla_multibuffer"):
            raise ValueError(
                "FiaCrossDomainTaskDemand.aic_pipeline_kind is unknown")
        for name in ("mm1", "vec1", "mm2", "vec2"):
            if not isinstance(getattr(self, name), FiaComputeStageDemand):
                raise TypeError(
                    f"FiaCrossDomainTaskDemand.{name} must be a stage demand")
        for name in (
                "vec1_update", "vec1_peer", "vec1_peer_update",
                "vec2_peer"):
            stage = getattr(self, name)
            if stage is not None and not isinstance(
                    stage, FiaComputeStageDemand):
                raise TypeError(
                    f"FiaCrossDomainTaskDemand.{name} must be a stage demand "
                    "or None")
        if (self.vec1_peer is None) != (self.vec2_peer is None):
            raise ValueError(
                "FIA peer Vec1 and Vec2 stages must be present together")
        if self.vec1_peer_update is not None and self.vec1_peer is None:
            raise ValueError(
                "FIA peer update requires a peer Vec1/Vec2 lane")
        for name in ("mm1_mte2", "mm2_mte2"):
            if not isinstance(getattr(self, name), FiaMte2StageDemand):
                raise TypeError(
                    f"FiaCrossDomainTaskDemand.{name} must be an MTE2 stage")
        if (self.gqa_source_order is not None
                and not isinstance(
                    self.gqa_source_order, FiaGqaSourceOrderDemand)):
            raise TypeError(
                "FiaCrossDomainTaskDemand.gqa_source_order must be a "
                "typed demand or None")
        if (self.mla_source_order is not None
                and not isinstance(
                    self.mla_source_order, FiaMlaSourceOrderDemand)):
            raise TypeError(
                "FiaCrossDomainTaskDemand.mla_source_order must be a "
                "typed demand or None")
        if (self.gqa_source_order is not None
                and self.aic_pipeline_kind != "serial_stages"):
            raise ValueError(
                "FIA GQA source order requires the regular serial pipeline")
        if (self.mla_source_order is not None
                and self.aic_pipeline_kind
                != "fia_block_cube_nonquant_mla_multibuffer"):
            raise ValueError(
                "FIA MLA source order requires the MLA multibuffer pipeline")
        if (self.gqa_source_order is not None
                and self.mla_source_order is not None):
            raise ValueError("FIA task cannot mix GQA and MLA source orders")
        for name in ("mm1_mte1", "mm2_mte1"):
            if not isinstance(getattr(self, name), FiaMte1StageDemand):
                raise TypeError(
                    f"FiaCrossDomainTaskDemand.{name} must be an MTE1 stage")
        for name in ("mm1_fixpipe", "mm2_fixpipe"):
            if not isinstance(getattr(self, name), FiaFixpipeStageDemand):
                raise TypeError(
                    f"FiaCrossDomainTaskDemand.{name} must be a Fixpipe "
                    "stage")
        if (self.mm1.resource_role, self.vec1.resource_role,
                self.mm2.resource_role, self.vec2.resource_role) != (
                    "compute_tensor", "compute_vector",
                    "compute_tensor", "compute_vector"):
            raise ValueError(
                "FIA task stages must be MM1, Vec1, MM2, Vec2")
        if self.mm1.physical_worker != self.mm2.physical_worker:
            raise ValueError("FIA MM1 and MM2 must share one AIC worker")
        if (self.mm1_mte2.physical_worker != self.mm1.physical_worker
                or self.mm2_mte2.physical_worker
                != self.mm1.physical_worker):
            raise ValueError("FIA MTE2 and MM stages must share one AIC worker")
        if (self.mm1_mte1.physical_worker != self.mm1.physical_worker
                or self.mm2_mte1.physical_worker
                != self.mm1.physical_worker):
            raise ValueError("FIA MTE1 and MM stages must share one AIC worker")
        if (self.mm1_fixpipe.physical_worker != self.mm1.physical_worker
                or self.mm2_fixpipe.physical_worker
                != self.mm1.physical_worker):
            raise ValueError(
                "FIA Fixpipe and MM stages must share one AIC worker")
        if self.vec1.physical_worker != self.vec2.physical_worker:
            raise ValueError("FIA Vec1 and Vec2 must share one AIV worker")
        for compute_name, memory_name in (
                ("vec1", "vec1_memory"),
                ("vec2", "vec2_memory"),
                ("vec1_update", "vec1_update_memory"),
                ("vec1_peer", "vec1_peer_memory"),
                ("vec1_peer_update", "vec1_peer_update_memory"),
                ("vec2_peer", "vec2_peer_memory")):
            compute = getattr(self, compute_name)
            memory = getattr(self, memory_name)
            if memory is not None and not isinstance(
                    memory, FiaAivMemoryStageDemand):
                raise TypeError(
                    f"FiaCrossDomainTaskDemand.{memory_name} must be an "
                    "AIV memory stage or None")
            if compute is None and memory is not None:
                raise ValueError(
                    f"FIA {memory_name} requires its compute stage")
            if (compute is not None and memory is not None
                    and compute.physical_worker != memory.physical_worker):
                raise ValueError(
                    f"FIA {compute_name} compute and memory stages must use "
                    "one AIV worker")
        if (self.vec1_update is not None
                and (self.vec1_update.resource_role != "compute_vector"
                     or self.vec1_update.physical_worker
                     != self.vec1.physical_worker)):
            raise ValueError(
                "FIA primary update must use the primary AIV worker")
        if self.vec1_peer is not None:
            if (self.vec1_peer.resource_role,
                    self.vec2_peer.resource_role) != (
                        "compute_vector", "compute_vector"):
                raise ValueError("FIA peer stages must use Vector resources")
            if (self.vec1_peer.physical_worker
                    != self.vec2_peer.physical_worker):
                raise ValueError(
                    "FIA peer Vec1 and Vec2 must share one AIV worker")
        if (self.vec1_peer_update is not None
                and (self.vec1_peer_update.resource_role != "compute_vector"
                     or self.vec1_peer_update.physical_worker
                     != self.vec1_peer.physical_worker)):
            raise ValueError("FIA peer update must use the peer AIV worker")
        if (not isinstance(self.vec1_block_reduce_sum, tuple)
                or not isinstance(self.vec1_peer_block_reduce_sum, tuple)):
            raise TypeError(
                "FIA Vec1 block-reduce demands must be tuples")
        if any(
                not isinstance(value, FiaBlockReduceSumDescriptorDemand)
                for values in (
                    self.vec1_block_reduce_sum,
                    self.vec1_peer_block_reduce_sum)
                for value in values):
            raise TypeError(
                "FIA Vec1 block-reduce tuples must contain typed demands")
        if self.gqa_source_order is not None:
            def descriptor_signature(descriptor):
                return (
                    descriptor.rows, descriptor.columns,
                    descriptor.element_bytes,
                    (-1 if descriptor.source_row_stride_elements is None
                     else descriptor.source_row_stride_elements),
                    descriptor.block_table_lookup_count,
                )

            def expanded_stage(stage):
                return tuple(
                    descriptor_signature(descriptor)
                    for descriptor in stage.descriptors
                    for _ in range(descriptor.count))

            source_order = self.gqa_source_order
            mm1_source = (() if source_order.query_descriptor is None else (
                descriptor_signature(source_order.query_descriptor),)) + tuple(
                    descriptor_signature(descriptor)
                    for round_demand in source_order.rounds
                    for descriptor in round_demand.key_descriptors)
            mm2_source = tuple(
                descriptor_signature(descriptor)
                for round_demand in source_order.rounds
                for descriptor in (
                    round_demand.probability_descriptor,
                    *round_demand.value_descriptors))
            if tuple(sorted(expanded_stage(self.mm1_mte2))) != tuple(sorted(mm1_source)):
                raise ValueError(
                    "FIA GQA source order must conserve the MM1 MTE2 "
                    "descriptor multiset")
            if tuple(sorted(expanded_stage(self.mm2_mte2))) != tuple(sorted(mm2_source)):
                raise ValueError(
                    "FIA GQA source order must conserve the MM2 MTE2 "
                    "descriptor multiset")

            panels = tuple(
                panel
                for round_demand in source_order.rounds
                for panel in round_demand.panels)
            if any(
                    panel.mm1.physical_worker != self.mm1.physical_worker
                    for panel in panels):
                raise ValueError(
                    "FIA GQA source panels must use the task AIC worker")
            for source_name, aggregate, panel_stages in (
                    ("MM1 MTE1", self.mm1_mte1,
                     tuple(panel.mm1_mte1 for panel in panels)),
                    ("MM2 MTE1", self.mm2_mte1,
                     tuple(panel.mm2_mte1 for panel in panels))):
                if (sum(stage.l0a_bytes for stage in panel_stages)
                        != aggregate.l0a_bytes
                        or sum(stage.l0b_bytes for stage in panel_stages)
                        != aggregate.l0b_bytes):
                    raise ValueError(
                        f"FIA GQA source panels must conserve {source_name} "
                        "traffic")
            for source_name, aggregate, panel_stages in (
                    ("MM1 Tensor", self.mm1,
                     tuple(panel.mm1 for panel in panels)),
                    ("MM2 Tensor", self.mm2,
                     tuple(panel.mm2 for panel in panels))):
                if (not math.isclose(
                        math.fsum(stage.logical_operations
                                  for stage in panel_stages),
                        aggregate.logical_operations, rel_tol=0.0,
                        abs_tol=max(math.ulp(aggregate.logical_operations),
                                    1e-12))
                        or not math.isclose(
                            math.fsum(stage.physical_operations
                                      for stage in panel_stages),
                            aggregate.physical_operations, rel_tol=0.0,
                            abs_tol=max(math.ulp(aggregate.physical_operations),
                                        1e-12))):
                    raise ValueError(
                        f"FIA GQA source panels must conserve {source_name} "
                        "work")

            def fixpipe_signature(descriptor):
                return (
                    descriptor.rows, descriptor.columns,
                    descriptor.destination_stride_elements,
                )

            panel_fixpipe = tuple(
                fixpipe_signature(descriptor)
                for panel in panels
                for descriptor in panel.mm1_fixpipe.descriptors
                for _ in range(descriptor.count))
            aggregate_fixpipe = tuple(
                fixpipe_signature(descriptor)
                for descriptor in self.mm1_fixpipe.descriptors
                for _ in range(descriptor.count))
            if tuple(sorted(panel_fixpipe)) != tuple(sorted(aggregate_fixpipe)):
                raise ValueError(
                    "FIA GQA source panels must conserve MM1 Fixpipe calls")

        if self.mla_source_order is not None:
            def mla_descriptor_signature(descriptor):
                return (
                    descriptor.rows,
                    descriptor.columns,
                    descriptor.element_bytes,
                    (-1 if descriptor.source_row_stride_elements is None
                     else descriptor.source_row_stride_elements),
                    descriptor.block_table_lookup_count,
                )

            def expanded_mla_stage(stage):
                return tuple(
                    mla_descriptor_signature(descriptor)
                    for descriptor in stage.descriptors
                    for _ in range(descriptor.count))

            for source_name, aggregate, source in (
                    ("MM1", self.mm1_mte2,
                     self.mla_source_order.mm1_descriptors),
                    ("MM2", self.mm2_mte2,
                     self.mla_source_order.mm2_descriptors)):
                if tuple(sorted(expanded_mla_stage(aggregate))) != tuple(
                        sorted(mla_descriptor_signature(descriptor)
                               for descriptor in source)):
                    raise ValueError(
                        f"FIA MLA source order must conserve {source_name} "
                        "MTE2 descriptor multiset")


@dataclass(frozen=True)
class FiaCrossDomainAicGroupDemand:
    """One typed FIA program shared by equivalent physical AIC groups."""

    aic_worker: int
    aiv_workers: Tuple[int, int]
    tasks: Tuple[FiaCrossDomainTaskDemand, ...]
    equivalent_aic_workers: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (isinstance(self.aic_worker, bool)
                or not isinstance(self.aic_worker, int)
                or self.aic_worker < 0):
            raise ValueError(
                "FiaCrossDomainAicGroupDemand.aic_worker must be a "
                "non-negative integer")
        if self.aiv_workers != (
                2 * self.aic_worker, 2 * self.aic_worker + 1):
            raise ValueError(
                "FIA group must preserve the source 1:2 AIC/AIV mapping")
        if (not isinstance(self.equivalent_aic_workers, tuple)
                or any(isinstance(worker, bool)
                       or not isinstance(worker, int)
                       or worker <= self.aic_worker
                       for worker in self.equivalent_aic_workers)
                or self.equivalent_aic_workers
                != tuple(sorted(set(self.equivalent_aic_workers)))):
            raise ValueError(
                "FIA equivalent AIC workers must be unique ordered integers "
                "after the representative")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise TypeError(
                "FiaCrossDomainAicGroupDemand.tasks must be non-empty tuple")
        if any(not isinstance(value, FiaCrossDomainTaskDemand)
               for value in self.tasks):
            raise TypeError(
                "FiaCrossDomainAicGroupDemand.tasks must contain typed tasks")
        expected_ordinal = 0
        for task in self.tasks:
            if task.ordinal != expected_ordinal:
                raise ValueError(
                    "FIA task-run ordinals must be consecutive per group")
            expected_ordinal += task.repeat_count
        for task in self.tasks:
            if task.mm1.physical_worker != self.aic_worker:
                raise ValueError("FIA task Tensor stages use the wrong AIC")
            if task.vec1.physical_worker != self.aiv_workers[0]:
                raise ValueError(
                    "FIA task Vector work must use the source-active AIV lane")
            if (task.vec1_peer is not None
                    and task.vec1_peer.physical_worker
                    != self.aiv_workers[1]):
                raise ValueError(
                    "FIA task peer Vector work must use the second AIV lane")

    @property
    def aic_workers(self) -> Tuple[int, ...]:
        return (self.aic_worker,) + self.equivalent_aic_workers

    @property
    def class_count(self) -> int:
        return len(self.aic_workers)


@dataclass(frozen=True)
class FiaTilingDataPreludeDemand:
    """Source-mandatory mixed-core tiling materialization before FIA Init."""

    mechanism_id: str
    payload_bytes: int
    aiv_transfer_bytes: int
    active_aic_groups: int

    def __post_init__(self) -> None:
        if self.mechanism_id != "fia_mixed_tiling_data_materialization_v1":
            raise ValueError("FIA tiling prelude has an unknown mechanism")
        expected = {
            "payload_bytes": 1560,
            "aiv_transfer_bytes": 1568,
        }
        for name, required in expected.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"FIA tiling {name} must be an integer")
            if value != required:
                raise ValueError(
                    f"FIA tiling {name} must equal {required}")
        if (isinstance(self.active_aic_groups, bool)
                or not isinstance(self.active_aic_groups, int)):
            raise TypeError(
                "FIA tiling active_aic_groups must be an integer")
        if self.active_aic_groups <= 0:
            raise ValueError(
                "FIA tiling active_aic_groups must be positive")


@dataclass(frozen=True)
class FiaCrossDomainWorkPlan:
    """Source-closed FIA main DAG and an optional barrier-ordered AIV drain."""

    mechanism_id: str
    groups: Tuple[FiaCrossDomainAicGroupDemand, ...]
    terminal_vector_stages: Tuple[FiaComputeStageDemand, ...]
    aiv_initialization_stages: Tuple[FiaComputeStageDemand, ...] = ()
    tiling_data_prelude: Optional[FiaTilingDataPreludeDemand] = None
    source_s2_base_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mechanism_id != "cann_fia_1_2_three_handoff_v5":
            raise ValueError("FIA work plan has an unknown mechanism")
        if not isinstance(self.groups, tuple) or not self.groups:
            raise TypeError("FiaCrossDomainWorkPlan.groups must be non-empty")
        if any(not isinstance(value, FiaCrossDomainAicGroupDemand)
               for value in self.groups):
            raise TypeError(
                "FiaCrossDomainWorkPlan.groups must contain typed groups")
        physical_workers = tuple(
            worker for group in self.groups for worker in group.aic_workers)
        if tuple(sorted(physical_workers)) != tuple(
                range(len(physical_workers))):
            raise ValueError(
                "FIA work-plan physical groups must be consecutive")
        if any(sum(task.repeat_count for task in group.tasks) > 128
               for group in self.groups):
            raise ValueError(
                "FIA whole-program span exceeds the measured task-count "
                "domain")
        if not isinstance(self.terminal_vector_stages, tuple):
            raise TypeError(
                "FiaCrossDomainWorkPlan.terminal_vector_stages must be a "
                "tuple")
        if any(not isinstance(stage, FiaComputeStageDemand)
               for stage in self.terminal_vector_stages):
            raise TypeError(
                "FiaCrossDomainWorkPlan.terminal_vector_stages must contain "
                "FiaComputeStageDemand")
        terminal_workers = tuple(
            stage.physical_worker for stage in self.terminal_vector_stages)
        if terminal_workers != tuple(sorted(set(terminal_workers))):
            raise ValueError(
                "FIA terminal Vector workers must be unique and ordered")
        if any(
                stage.resource_role != "compute_vector"
                or stage.logical_operations <= 0
                or len(stage.typed_services) != 1
                or stage.typed_services[0].service != "generic"
                or stage.typed_services[0].serial_invocations != 0
                for stage in self.terminal_vector_stages):
            raise ValueError(
                "FIA terminal stages must be positive generic Vector work")
        if any(worker >= 2 * self.active_group_count
               for worker in terminal_workers):
            raise ValueError(
                "FIA terminal stage exceeds the source AIV launch")
        if not isinstance(self.aiv_initialization_stages, tuple):
            raise TypeError(
                "FiaCrossDomainWorkPlan.aiv_initialization_stages must be a "
                "tuple")
        if any(not isinstance(stage, FiaComputeStageDemand)
               for stage in self.aiv_initialization_stages):
            raise TypeError(
                "FIA AIV initialization must contain compute stages")
        initialization_workers = tuple(
            stage.physical_worker
            for stage in self.aiv_initialization_stages)
        if initialization_workers and initialization_workers != tuple(
                range(2 * self.active_group_count)):
            raise ValueError(
                "FIA AIV initialization must cover every source AIV once")
        if any(
                stage.resource_role != "compute_vector"
                or stage.logical_operations <= 0
                or len(stage.typed_services) != 1
                or stage.typed_services[0].service != "generic"
                or stage.typed_services[0].serial_invocations != 0
                for stage in self.aiv_initialization_stages):
            raise ValueError(
                "FIA AIV initialization must be positive generic Vector "
                "work")
        if (self.tiling_data_prelude is not None
                and not isinstance(
                    self.tiling_data_prelude, FiaTilingDataPreludeDemand)):
            raise TypeError(
                "FIA tiling_data_prelude must be a typed demand or None")
        if (self.tiling_data_prelude is not None
                and self.tiling_data_prelude.active_aic_groups
                != self.active_group_count):
            raise ValueError(
                "FIA tiling prelude active groups disagree with the plan")
        if self.source_s2_base_size is not None:
            if (isinstance(self.source_s2_base_size, bool)
                    or not isinstance(self.source_s2_base_size, int)):
                raise TypeError(
                    "FIA source S2 base size must be an integer or None")
            if self.source_s2_base_size <= 0:
                raise ValueError("FIA source S2 base size must be positive")

    @cached_property
    def active_group_count(self) -> int:
        return sum(group.class_count for group in self.groups)

    @cached_property
    def task_counts(self) -> Tuple[int, ...]:
        counts = [0] * self.active_group_count
        for group in self.groups:
            task_count = sum(
                task.repeat_count for task in group.tasks)
            for worker in group.aic_workers:
                counts[worker] = task_count
        return tuple(counts)

    def validate_conservation(
            self, compute_demands: Tuple[ComputeDemand, ...]) -> None:
        """Require the staged physical program to own aggregate compute once."""

        aggregate = {
            demand.resource_role: demand for demand in compute_demands}
        if set(aggregate) != {"compute_tensor", "compute_vector"}:
            raise ValueError(
                "FIA work plan requires Tensor and Vector aggregate demands")
        stage_by_role = {
            "compute_tensor": [],
            "compute_vector": [],
        }
        for group in self.groups:
            for task in group.tasks:
                stage_by_role["compute_tensor"].extend((
                    (task.mm1, group.class_count * task.repeat_count),
                    (task.mm2, group.class_count * task.repeat_count),
                ))
                stage_by_role["compute_vector"].extend((
                    (task.vec1, group.class_count * task.repeat_count),
                    (task.vec2, group.class_count * task.repeat_count),
                ))
                if task.vec1_update is not None:
                    stage_by_role["compute_vector"].append(
                        (task.vec1_update,
                         group.class_count * task.repeat_count))
                if task.vec1_peer is not None:
                    stage_by_role["compute_vector"].extend((
                        (task.vec1_peer,
                         group.class_count * task.repeat_count),
                        (task.vec2_peer,
                         group.class_count * task.repeat_count),
                    ))
                if task.vec1_peer_update is not None:
                    stage_by_role["compute_vector"].append(
                        (task.vec1_peer_update,
                         group.class_count * task.repeat_count))
        stage_by_role["compute_vector"].extend(
            (stage, 1) for stage in self.terminal_vector_stages)
        stage_by_role["compute_vector"].extend(
            (stage, 1) for stage in self.aiv_initialization_stages)
        for role, stages in stage_by_role.items():
            staged = math.fsum(
                multiplier * stage.logical_operations
                for stage, multiplier in stages)
            expected = aggregate[role].operations
            if not math.isclose(
                    staged, expected, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(
                    f"FIA {role} stages do not conserve aggregate work: "
                    f"staged={staged}, aggregate={expected}")
        tensor = aggregate["compute_tensor"]
        tensor_physical = math.fsum(
            multiplier * stage.physical_operations
            for stage, multiplier in stage_by_role["compute_tensor"])
        expected_physical = tensor.operations / tensor.shape_efficiency
        if not math.isclose(
                tensor_physical, expected_physical,
                rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "FIA Tensor physical stages disagree with aggregate padding")
        physical_by_aic = []
        for group in self.groups:
            physical = math.fsum(
                task.repeat_count * stage.physical_operations
                for task in group.tasks for stage in (task.mm1, task.mm2))
            physical_by_aic.extend(
                physical for _ in range(group.class_count))
        expected_schedule_efficiency = (
            tensor_physical
            / (self.active_group_count * max(physical_by_aic)))
        if not math.isclose(
                tensor.schedule_efficiency, expected_schedule_efficiency,
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "FIA Tensor worker program disagrees with aggregate schedule "
                "efficiency")
        vector = aggregate["compute_vector"]
        service_by_name = {
            demand.service: demand for demand in vector.service_demands}
        if not service_by_name:
            raise ValueError("FIA Vector aggregate requires typed services")
        worker_count = 2 * self.active_group_count
        staged_operations = {
            name: [0.0] * worker_count for name in service_by_name}
        staged_invocations = {
            name: [0] * worker_count for name in service_by_name}
        for group in self.groups:
            for task in group.tasks:
                stages = [task.vec1, task.vec2]
                for optional_stage in (
                        task.vec1_update, task.vec1_peer,
                        task.vec1_peer_update, task.vec2_peer):
                    if optional_stage is not None:
                        stages.append(optional_stage)
                for stage in stages:
                    lane = stage.physical_worker - 2 * group.aic_worker
                    if lane not in (0, 1):
                        raise ValueError(
                            "FIA Vector stage is outside its AIC group")
                    for aic_worker in group.aic_workers:
                        worker = 2 * aic_worker + lane
                        for service in stage.typed_services:
                            if service.service not in service_by_name:
                                raise ValueError(
                                    "FIA stage contains an unknown Vector "
                                    "service")
                            staged_operations[service.service][worker] += (
                                task.repeat_count * service.operations)
                            staged_invocations[service.service][worker] += (
                                task.repeat_count
                                * service.serial_invocations)
        for stage in self.terminal_vector_stages:
            for service in stage.typed_services:
                if service.service not in service_by_name:
                    raise ValueError(
                        "FIA terminal contains an unknown Vector service")
                worker = stage.physical_worker
                staged_operations[service.service][worker] += (
                    service.operations)
                staged_invocations[service.service][worker] += (
                    service.serial_invocations)
        for stage in self.aiv_initialization_stages:
            for service in stage.typed_services:
                if service.service not in service_by_name:
                    raise ValueError(
                        "FIA AIV initialization contains an unknown Vector "
                        "service")
                worker = stage.physical_worker
                staged_operations[service.service][worker] += (
                    service.operations)
                staged_invocations[service.service][worker] += (
                    service.serial_invocations)
        for name, demand in service_by_name.items():
            if len(demand.operations_by_worker) != worker_count:
                raise ValueError(
                    "FIA aggregate Vector worker count disagrees with plan")
            for worker, staged in enumerate(staged_operations[name]):
                if not math.isclose(
                        staged, demand.operations_by_worker[worker],
                        rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(
                        "FIA staged Vector operations disagree with the "
                        f"aggregate worker schedule for {name}/{worker}")
            if tuple(staged_invocations[name]) != (
                    demand.serial_invocations_by_worker):
                raise ValueError(
                    "FIA staged Vector invocations disagree with aggregate "
                    f"worker schedule for {name}")


@dataclass(frozen=True)
class FiaSplitFuseRequestRunDemand:
    """A consecutive run of identical requests in a SplitFuse source queue."""

    query_length: int
    key_value_length: int
    repeat_count: int

    def __post_init__(self) -> None:
        for name in ("query_length", "key_value_length", "repeat_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"FiaSplitFuseRequestRunDemand.{name} must be integer")
            if value <= 0:
                raise ValueError(
                    f"FiaSplitFuseRequestRunDemand.{name} must be positive")
        if self.query_length > self.key_value_length:
            raise ValueError(
                "SplitFuse bottom-right requests require Q length <= KV "
                "length")


@dataclass(frozen=True)
class FiaSplitFuseWorkPlan:
    """Compressed CANN SplitFuse AIC-to-dual-AIV source program.

    Request runs retain source order without materializing one Python object
    per task.  ``iter_task_runs`` maps each query-tile class onto the fixed AIC
    ring with quotient/remainder counts for the three-handoff recurrence.
    Each task retains both its sequence rows and its packed query-head rows:
    CANN uses the former for the causal boundary and the latter for MMAD,
    Vector, and workspace work.
    """

    mechanism_id: str
    aic_worker_count: int
    tile_q: int
    kv_stack_ceiling: int
    head_dim: int
    query_head_count: int
    kv_head_count: int
    request_runs: Tuple[FiaSplitFuseRequestRunDemand, ...]
    aic_l2_bytes: float
    aiv_l2_bytes: float
    aiv_mte3_bytes: float
    paged_kv_cache: bool
    kv_page_size: Optional[int]
    pv_v_ingress_before_softmax: bool = False
    tensor_operation_scale: float = 1.0
    vector_operation_scale: float = 1.0
    l2_byte_scale: float = 1.0
    mte3_byte_scale: float = 1.0
    # One independent cached FULL-graph request, appended after real
    # requests. KV=0 issues Q loading plus the local zero-output epilogue,
    # without QK/PV, workspace, or cross-domain handoffs. Zero rows is inert.
    empty_kv_tail_query_length: int = 0

    def __post_init__(self) -> None:
        if self.mechanism_id != "cann_split_fuse_aic_dual_aiv_v1":
            raise ValueError("SplitFuse work plan has an unknown mechanism")
        if (type(self.empty_kv_tail_query_length) is not int
                or self.empty_kv_tail_query_length < 0):
            raise ValueError("SplitFuse empty-KV tail rows must be a nonnegative integer")
        if self.empty_kv_tail_query_length and not self.paged_kv_cache:
            raise ValueError("SplitFuse empty-KV tail requires the cached source ABI")
        for name in (
                "aic_worker_count", "tile_q", "kv_stack_ceiling",
                "head_dim", "query_head_count", "kv_head_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"FiaSplitFuseWorkPlan.{name} must be integer")
            if value <= 0:
                raise ValueError(
                    f"FiaSplitFuseWorkPlan.{name} must be positive")
        if self.aic_worker_count > 24:
            raise ValueError("SplitFuse work plan exceeds the 24-AIC launch")
        if (self.tile_q, self.kv_stack_ceiling) != (128, 512):
            raise ValueError(
                "SplitFuse work plan requires the source Q128/KV512 tile")
        if self.head_dim % 16:
            raise ValueError("SplitFuse head dimension must be 16-aligned")
        if self.query_head_count % self.kv_head_count:
            raise ValueError("SplitFuse requires integral GQA head groups")
        if not isinstance(self.paged_kv_cache, bool):
            raise TypeError("FiaSplitFuseWorkPlan.paged_kv_cache must be bool")
        if not isinstance(self.pv_v_ingress_before_softmax, bool):
            raise TypeError(
                "FiaSplitFuseWorkPlan.pv_v_ingress_before_softmax "
                "must be bool")
        if self.paged_kv_cache:
            if (isinstance(self.kv_page_size, bool)
                    or not isinstance(self.kv_page_size, int)
                    or self.kv_page_size != 128):
                raise ValueError(
                    "paged SplitFuse requires the source 128-token page")
        elif self.kv_page_size is not None:
            raise ValueError(
                "contiguous SplitFuse cannot publish a KV page size")
        if not isinstance(self.request_runs, tuple) or not self.request_runs:
            raise TypeError(
                "FiaSplitFuseWorkPlan.request_runs must be non-empty tuple")
        if any(not isinstance(value, FiaSplitFuseRequestRunDemand)
               for value in self.request_runs):
            raise TypeError(
                "FiaSplitFuseWorkPlan.request_runs must contain typed runs")
        for previous, current in zip(
                self.request_runs, self.request_runs[1:]):
            if (previous.query_length, previous.key_value_length) == (
                    current.query_length, current.key_value_length):
                raise ValueError(
                    "adjacent identical SplitFuse request runs must be "
                    "coalesced")
        for name in (
                "tensor_operation_scale", "vector_operation_scale",
                "l2_byte_scale", "mte3_byte_scale", "aic_l2_bytes",
                "aiv_l2_bytes", "aiv_mte3_bytes"):
            value = _require_finite(
                f"FiaSplitFuseWorkPlan.{name}", getattr(self, name),
                positive=True)
            object.__setattr__(self, name, value)

    def paged_descriptor_lookup_count(self, attended_kv: int) -> int:
        """Return source QK plus PV block-table reads for one task."""

        if isinstance(attended_kv, bool) or not isinstance(attended_kv, int):
            raise TypeError("SplitFuse attended KV length must be integer")
        if attended_kv <= 0:
            raise ValueError("SplitFuse attended KV length must be positive")
        if not self.paged_kv_cache:
            return 0
        if self.kv_page_size is None:
            raise AssertionError("paged SplitFuse lost its page size")
        return 2 * math.ceil(attended_kv / self.kv_page_size)

    def query_head_bundle_sizes(self, query_length: int) -> Tuple[int, ...]:
        """Translate CANN ``GetQNBlockTile`` for one request."""

        if isinstance(query_length, bool) or not isinstance(query_length, int):
            raise TypeError("SplitFuse query length must be integer")
        if query_length <= 0:
            raise ValueError("SplitFuse query length must be positive")
        group_size = self.query_head_count // self.kv_head_count
        even_heads = 2 * ((self.tile_q // query_length) // 2)
        heads_per_task = max(1, min(group_size, even_heads))
        full_tasks, tail_heads = divmod(group_size, heads_per_task)
        per_kv_head = (heads_per_task,) * full_tasks
        if tail_heads:
            per_kv_head += (tail_heads,)
        return per_kv_head * self.kv_head_count

    @property
    def task_count(self) -> int:
        return sum(
            run.repeat_count
            * math.ceil(run.query_length / self.tile_q)
            * len(self.query_head_bundle_sizes(run.query_length))
            for run in self.request_runs)

    @property
    def task_counts(self) -> Tuple[int, ...]:
        quotient, remainder = divmod(self.task_count, self.aic_worker_count)
        return tuple(
            quotient + (1 if worker < remainder else 0)
            for worker in range(self.aic_worker_count))

    @cached_property
    def empty_kv_tail_worker_geometry(self):
        """Source-ordered zero-KV tasks, separate from the handoff ring."""

        counts = [dict() for _ in range(self.aic_worker_count)]
        query = self.empty_kv_tail_query_length
        if query:
            ordinal = self.task_count
            bundles = self.query_head_bundle_sizes(query)
            for start in range(0, query, self.tile_q):
                query_rows = min(self.tile_q, query - start)
                for heads in bundles:
                    worker = ordinal % self.aic_worker_count
                    key = (query_rows, heads)
                    counts[worker][key] = (
                        counts[worker][key] + 1 if key in counts[worker] else 1)
                    ordinal += 1
        return tuple(tuple((rows, heads, count)
            for (rows, heads), count in worker.items()) for worker in counts)

    @cached_property
    def empty_kv_tail_task_count(self) -> int:
        return sum(count for worker in self.empty_kv_tail_worker_geometry
            for _, _, count in worker)

    @cached_property
    def physical_task_counts(self) -> Tuple[int, ...]:
        return tuple(attention + sum(count for _, _, count in empty)
            for attention, empty in zip(self.task_counts, self.empty_kv_tail_worker_geometry))

    @cached_property
    def worker_program_blocks(self):
        """Return bounded periodic worker programs without request expansion.

        Each result is ``(repeat_count, worker_programs)``.  A worker program
        contains source-ordered ``(query_rows, packed_rows, attended_kv,
        count)`` runs for one complete assignment period.  Repeating an
        identical request can only rotate the first AIC by
        ``tasks_per_request % worker_count``; therefore its full assignment
        period is at most ``worker_count``.
        """

        def append_run(
                program, query_rows, packed_rows, attended_kv, count):
            if count <= 0:
                raise ValueError(
                    "SplitFuse worker program requires a positive run")
            if (program
                    and program[-1][:3]
                    == (query_rows, packed_rows, attended_kv)):
                previous = program[-1]
                program[-1] = (
                    query_rows, packed_rows, attended_kv,
                    previous[3] + count)
            else:
                program.append((
                    query_rows, packed_rows, attended_kv, count))

        def append_cyclic_run(
                worker_programs, first_ordinal, query_rows, packed_rows,
                attended_kv, count):
            """Append one source-contiguous task run without CTA expansion."""

            if count <= 0:
                raise ValueError(
                    "SplitFuse cyclic task-run count must be positive")
            for worker in range(self.aic_worker_count):
                first = (worker - first_ordinal) % self.aic_worker_count
                if first >= count:
                    continue
                append_run(
                    worker_programs[worker],
                    query_rows,
                    packed_rows,
                    attended_kv,
                    1 + (count - 1 - first) // self.aic_worker_count,
                )

        blocks = []
        first_worker = 0
        for request_run in self.request_runs:
            query_tiles = math.ceil(request_run.query_length / self.tile_q)
            head_group_size = self.query_head_count // self.kv_head_count
            even_heads = 2 * (
                (self.tile_q // request_run.query_length) // 2)
            heads_per_task = max(1, min(head_group_size, even_heads))
            full_head_tasks, tail_heads = divmod(
                head_group_size, heads_per_task)
            tasks_per_request = query_tiles * self.kv_head_count * (
                full_head_tasks + int(tail_heads > 0))
            period = (
                self.aic_worker_count
                // math.gcd(tasks_per_request, self.aic_worker_count))
            phase_programs = []
            phase_first_worker = first_worker
            causal_offset = (
                request_run.key_value_length - request_run.query_length)
            for _ in range(period):
                worker_programs = [
                    [] for _ in range(self.aic_worker_count)]
                ordinal = phase_first_worker
                for start in range(
                        0, request_run.query_length, self.tile_q):
                    query_rows = min(
                        self.tile_q, request_run.query_length - start)
                    attended_kv = min(
                        request_run.key_value_length,
                        start + self.tile_q + causal_offset,
                    )
                    for _ in range(self.kv_head_count):
                        if full_head_tasks:
                            append_cyclic_run(
                                worker_programs,
                                ordinal,
                                query_rows,
                                query_rows * heads_per_task,
                                attended_kv,
                                full_head_tasks,
                            )
                            ordinal += full_head_tasks
                        if tail_heads:
                            append_cyclic_run(
                                worker_programs,
                                ordinal,
                                query_rows,
                                query_rows * tail_heads,
                                attended_kv,
                                1,
                            )
                            ordinal += 1
                phase_first_worker = (
                    phase_first_worker + tasks_per_request
                ) % self.aic_worker_count
                phase_programs.append(tuple(
                    tuple(program) for program in worker_programs))
            if phase_first_worker != first_worker:
                raise AssertionError(
                    "SplitFuse request assignment period did not close")

            cycle_count, prefix_count = divmod(
                request_run.repeat_count, period)
            if cycle_count:
                cycle_programs = [
                    [] for _ in range(self.aic_worker_count)]
                for phase in phase_programs:
                    for worker, program in enumerate(phase):
                        for (query_rows, packed_rows, attended_kv,
                             count) in program:
                            append_run(
                                cycle_programs[worker],
                                query_rows,
                                packed_rows,
                                attended_kv,
                                count,
                            )
                blocks.append((cycle_count, tuple(
                    tuple(program) for program in cycle_programs)))
            if prefix_count:
                prefix_programs = [
                    [] for _ in range(self.aic_worker_count)]
                for phase in phase_programs[:prefix_count]:
                    for worker, program in enumerate(phase):
                        for (query_rows, packed_rows, attended_kv,
                             count) in program:
                            append_run(
                                prefix_programs[worker],
                                query_rows,
                                packed_rows,
                                attended_kv,
                                count,
                            )
                blocks.append((1, tuple(
                    tuple(program) for program in prefix_programs)))
            first_worker = (
                first_worker
                + request_run.repeat_count * tasks_per_request
            ) % self.aic_worker_count
        return tuple(blocks)

    def iter_worker_program_blocks(self):
        """Iterate the cached periodic worker programs."""

        yield from self.worker_program_blocks

    @cached_property
    def worker_geometry_counts(self):
        """Aggregate periodic programs by worker and task geometry."""

        counts = [dict() for _ in range(self.aic_worker_count)]
        for block_repeat, worker_programs in self.worker_program_blocks:
            for worker, program in enumerate(worker_programs):
                for (query_rows, packed_rows, attended_kv,
                     run_count) in program:
                    key = (query_rows, packed_rows, attended_kv)
                    increment = block_repeat * run_count
                    counts[worker][key] = (
                        counts[worker][key] + increment
                        if key in counts[worker] else increment)
        return tuple(tuple(
            (query_rows, packed_rows, attended_kv, count)
            for (query_rows, packed_rows, attended_kv), count
            in sorted(worker.items())
        ) for worker in counts)

    @property
    def task_classes(self) -> Tuple[Tuple[int, int, int, int], ...]:
        """Expand source runs for tests and diagnostics, never for Forward."""

        runs = []
        for request_run in self.request_runs:
            for _ in range(request_run.repeat_count):
                causal_offset = (
                    request_run.key_value_length
                    - request_run.query_length)
                head_bundles = self.query_head_bundle_sizes(
                    request_run.query_length)
                for start in range(
                        0, request_run.query_length, self.tile_q):
                    query_rows = min(
                        self.tile_q, request_run.query_length - start)
                    attended_kv = min(
                        request_run.key_value_length,
                        start + self.tile_q + causal_offset,
                    )
                    for bundled_heads in head_bundles:
                        packed_rows = query_rows * bundled_heads
                        if (runs
                                and runs[-1][:3]
                                == (query_rows, packed_rows, attended_kv)):
                            previous = runs[-1]
                            runs[-1] = (
                                query_rows, packed_rows, attended_kv,
                                previous[3] + 1)
                        else:
                            runs.append((
                                query_rows, packed_rows, attended_kv, 1))
        return tuple(runs)

    def iter_task_runs(self):
        """Expand source AIC runs for tests and diagnostics only."""

        ordinal = 0
        for (query_rows, packed_rows, attended_kv,
             class_count) in self.task_classes:
            quotient, remainder = divmod(
                class_count, self.aic_worker_count)
            first_worker = ordinal % self.aic_worker_count
            for offset in range(self.aic_worker_count):
                repeat_count = quotient + int(offset < remainder)
                if repeat_count:
                    yield (
                        (first_worker + offset) % self.aic_worker_count,
                        query_rows,
                        packed_rows,
                        attended_kv,
                        repeat_count,
                    )
            ordinal += class_count
        if ordinal != self.task_count:
            raise AssertionError("SplitFuse compressed task count drifted")

    def iter_task_geometry(self):
        """Expand task runs for tests and diagnostics, never for Forward."""

        for (worker, query_rows, packed_rows, attended_kv,
             repeat_count) in self.iter_task_runs():
            for _ in range(repeat_count):
                yield worker, query_rows, packed_rows, attended_kv

    def validate_conservation(
            self, compute_demands: Tuple[ComputeDemand, ...]) -> None:
        """Require compressed task geometry to conserve aggregate compute."""

        aggregate = {
            demand.resource_role: demand for demand in compute_demands}
        if set(aggregate) != {"compute_tensor", "compute_vector"}:
            raise ValueError(
                "SplitFuse work plan requires Tensor and Vector demands")
        tensor_by_aic = [0.0] * self.aic_worker_count
        vector_services = {
            "generic": [0.0] * (2 * self.aic_worker_count),
            "exp": [0.0] * (2 * self.aic_worker_count),
            "div": [0.0] * (2 * self.aic_worker_count),
        }
        vector_invocations = {
            "generic": [0] * (2 * self.aic_worker_count),
            "exp": [0] * (2 * self.aic_worker_count),
            "div": [0] * (2 * self.aic_worker_count),
        }
        embed_round = math.ceil(self.head_dim / 16) * 16
        vec2_row_tile = (8192 // embed_round // 8) * 8
        div_calls_per_loop = math.ceil(self.head_dim / 64)
        if vec2_row_tile <= 0:
            raise ValueError("SplitFuse Vec2 row tile is empty")
        compressed_task_count = 0
        for aic_worker, geometries in enumerate(self.worker_geometry_counts):
            for (query_rows, packed_rows, attended_kv,
                 repeat_count) in geometries:
                compressed_task_count += repeat_count
                mm_rows = math.ceil(packed_rows / 16) * 16
                tensor_by_aic[aic_worker] += (
                    repeat_count * 4 * self.head_dim
                    * mm_rows * attended_kv)
                for lane in range(2):
                    rows = (
                        packed_rows // 2 if lane == 0
                        else packed_rows - packed_rows // 2)
                    if rows == 0:
                        continue
                    worker = 2 * aic_worker + lane
                    scores = rows * attended_kv
                    outputs = rows * self.head_dim
                    exp_invocations = 0
                    for stack_start in range(
                            0, attended_kv, self.kv_stack_ceiling):
                        stack_columns = min(
                            self.kv_stack_ceiling,
                            attended_kv - stack_start)
                        stack_end = stack_start + stack_columns
                        masked = (
                            attended_kv - query_rows < stack_end - 1)
                        alignment = 32 if masked else 16
                        rounded_columns = (
                            math.ceil(stack_columns / alignment)
                            * alignment)
                        row_tile = min(
                            (8192 // rounded_columns // 8) * 8, 64)
                        if row_tile <= 0:
                            raise ValueError(
                                "SplitFuse Vec1 row tile is empty")
                        exp_invocations += math.ceil(rows / row_tile)
                    vector_services["generic"][worker] += (
                        repeat_count * (4 * scores + outputs
                            + 2 * outputs * (math.ceil(
                                attended_kv / self.kv_stack_ceiling) - 1)))
                    vector_services["exp"][worker] += (
                        repeat_count * scores)
                    vector_services["div"][worker] += (
                        repeat_count * outputs)
                    vector_invocations["exp"][worker] += (
                        repeat_count * exp_invocations)
                    vector_invocations["div"][worker] += (
                        repeat_count * math.ceil(rows / vec2_row_tile)
                        * div_calls_per_loop)
        if compressed_task_count != self.task_count:
            raise AssertionError(
                "SplitFuse periodic worker programs lost tasks")

        # block_epilogue_init_outputs.hpp duplicates one shared zero column
        # per participating subblock, then writes it separately to its heads.
        # No Exp/Div or tensor operation belongs to an empty-KV task.
        for worker, geometries in enumerate(self.empty_kv_tail_worker_geometry):
            for rows, heads, count in geometries:
                for lane in range(2):
                    lane_rows = (rows // 2 if lane == 0 else rows - rows // 2) if heads == 1 else rows
                    lane_heads = 1 if heads == 1 else (heads // 2 if lane == 0 else heads - heads // 2)
                    if lane_rows and lane_heads:
                        vector_services["generic"][2 * worker + lane] += count * lane_rows * embed_round

        tensor = aggregate["compute_tensor"]
        tensor_total = math.fsum(tensor_by_aic)
        if not math.isclose(
                tensor_total * self.tensor_operation_scale,
                tensor.operations, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "SplitFuse Tensor program does not conserve aggregate work")
        expected_schedule_efficiency = (
            tensor_total
            / (self.aic_worker_count * max(tensor_by_aic)))
        if not math.isclose(
                tensor.schedule_efficiency,
                expected_schedule_efficiency,
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "SplitFuse Tensor program disagrees with worker schedule")
        vector = aggregate["compute_vector"]
        service_by_name = {
            service.service: service for service in vector.service_demands}
        if set(service_by_name) != set(vector_services):
            raise ValueError(
                "SplitFuse Vector program requires generic/exp/div services")
        for name, operations_by_worker in vector_services.items():
            service = service_by_name[name]
            scaled = tuple(
                value * self.vector_operation_scale
                for value in operations_by_worker)
            if len(service.operations_by_worker) != len(scaled) or any(
                    not math.isclose(actual, expected, rel_tol=1e-12,
                                     abs_tol=1e-9)
                    for actual, expected in zip(
                        service.operations_by_worker, scaled)):
                raise ValueError(
                    "SplitFuse staged Vector operations disagree with "
                    f"aggregate worker schedule for {name}")
            if tuple(vector_invocations[name]) != (
                    service.serial_invocations_by_worker):
                raise ValueError(
                    "SplitFuse staged Vector invocations disagree with "
                    f"aggregate worker schedule for {name}")

    def validate_memory_conservation(
            self, memory_paths: Tuple[MemoryPathDemand, ...]) -> None:
        """Require the compressed staged bytes to own L2 and MTE3 once."""

        # The AscendC MIX entry's KFC message clear is framework work, not
        # part of the operator's staged plan, so it is not one of the bytes
        # the staging is required to own exactly once.
        bytes_by_role = {
            role: math.fsum(
                node.bytes
                for path in memory_paths
                if path.name != "kfc_workspace_clear"
                for node in path.nodes
                if node.resource_role == role)
            for role in ("l2_cache", "mte3")
        }
        expected_l2 = (
            (self.aic_l2_bytes + self.aiv_l2_bytes) * self.l2_byte_scale)
        expected_mte3 = self.aiv_mte3_bytes * self.mte3_byte_scale
        for role, expected in (
                ("l2_cache", expected_l2), ("mte3", expected_mte3)):
            if not math.isclose(
                    bytes_by_role[role], expected,
                    rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError(
                    "SplitFuse staged memory does not conserve aggregate "
                    f"{role} bytes: staged={expected}, "
                    f"aggregate={bytes_by_role[role]}")


@dataclass(frozen=True)
class MemoryNodeDiagnostic:
    resource_role: str
    transactions: Optional[float]
    bytes: float
    byte_capacity_per_s: float
    request_capacity_per_s: Optional[float]
    service_capacity_per_s: float
    arrival_rate_per_s: float
    rho: float
    queue_wait_s: Optional[float]
    capacity_time_s: float
    saturated: bool
    unloaded_latency_s: float


@dataclass(frozen=True)
class QueueDiagnostic:
    name: str
    allocation_rate_per_s: float
    lifetime_s: float
    used_entries: float
    max_entries: float
    initial_request_rate_capacity_per_s: float


@dataclass(frozen=True)
class MemoryDiagnostics:
    path_name: str
    access_kind: str
    rate_unit: str
    initial_transactions: Optional[float]
    lambda_initial_per_s: float
    lambda_capacity_per_s: float
    workload_slots_per_unit: Optional[float]
    hardware_source_slots_per_unit: Optional[float]
    source_slots_per_unit: Optional[float]
    source_lifetime_s: float
    source_rate_capacity_per_s: Optional[float]
    frontend_rate_capacity_per_s: Optional[float]
    steady_time_s: float
    first_request_latency_s: Optional[float]
    bottleneck: str
    capacity_bound: bool
    nodes: Tuple[MemoryNodeDiagnostic, ...]
    queues: Tuple[QueueDiagnostic, ...]


@dataclass(frozen=True)
class MemoryAccessClassDiagnostic:
    """Resolved logical-byte lower bound for one typed memory access."""

    path_name: str
    service: str
    access_kind: str
    ingress_resource_role: str
    signature: MemoryAccessClassSignature
    scope: str
    logical_bytes: float
    logical_byte_capacity_per_s: float
    service_time_s: float
    evidence_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path_name",
            _require_name(
                "MemoryAccessClassDiagnostic.path_name", self.path_name))
        object.__setattr__(
            self, "service",
            _require_name(
                "MemoryAccessClassDiagnostic.service", self.service))
        if self.access_kind not in ("read", "write"):
            raise ValueError(
                "MemoryAccessClassDiagnostic.access_kind must be 'read' or "
                "'write'")
        object.__setattr__(
            self, "ingress_resource_role",
            _require_name(
                "MemoryAccessClassDiagnostic.ingress_resource_role",
                self.ingress_resource_role))
        if not isinstance(self.signature, MemoryAccessClassSignature):
            raise TypeError(
                "MemoryAccessClassDiagnostic.signature must be "
                "MemoryAccessClassSignature")
        if self.service != self.signature.access_class:
            raise ValueError(
                "MemoryAccessClassDiagnostic.service must equal the signature "
                "access_class")
        if self.scope != "full_chip":
            raise ValueError(
                "MemoryAccessClassDiagnostic.scope must be 'full_chip'")
        for field_name in (
                "logical_bytes", "logical_byte_capacity_per_s",
                "service_time_s"):
            object.__setattr__(
                self, field_name,
                _require_finite(
                    f"MemoryAccessClassDiagnostic.{field_name}",
                    getattr(self, field_name), positive=True))
        object.__setattr__(
            self, "evidence_ref",
            _require_name(
                "MemoryAccessClassDiagnostic.evidence_ref",
                self.evidence_ref))
        expected_time = (
            self.logical_bytes / self.logical_byte_capacity_per_s)
        if not math.isclose(
                self.service_time_s, expected_time,
                rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                "MemoryAccessClassDiagnostic.service_time_s must equal "
                "logical_bytes / logical_byte_capacity_per_s")


@dataclass(frozen=True)
class MemoryArbitrationDiagnostic:
    """Joint service paid once for read/write paths sharing one resource."""

    resource_role: str
    arbitration_mode: str
    read_bytes: float
    write_bytes: float
    read_capacity_per_s: float
    write_capacity_per_s: float
    read_service_time_s: float
    write_service_time_s: float
    analytical_time_s: float
    effective_time_s: float
    switch_count: Optional[int]
    read_to_write_turnaround_s: Optional[float]
    write_to_read_turnaround_s: Optional[float]
    bound_kind: str
    schedule_status: str
    overridden: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_role",
            _require_name(
                "MemoryArbitrationDiagnostic.resource_role",
                self.resource_role))
        if self.arbitration_mode not in (
                "shared_common_capacity", "shared_directional_capacity",
                "independent_directional_capacity"):
            raise ValueError(
                "MemoryArbitrationDiagnostic.arbitration_mode is invalid")
        for field_name in ("read_bytes", "write_bytes"):
            object.__setattr__(
                self, field_name,
                _require_finite(
                    f"MemoryArbitrationDiagnostic.{field_name}",
                    getattr(self, field_name), positive=False))
        for field_name in ("read_capacity_per_s", "write_capacity_per_s"):
            object.__setattr__(
                self, field_name,
                _require_finite(
                    f"MemoryArbitrationDiagnostic.{field_name}",
                    getattr(self, field_name), positive=True))
        if (self.arbitration_mode == "shared_common_capacity"
                and not math.isclose(
                    self.read_capacity_per_s,
                    self.write_capacity_per_s,
                    rel_tol=1e-12, abs_tol=0.0)):
            raise ValueError(
                "shared_common_capacity requires equal read/write capacity")
        for field_name in (
                "read_service_time_s", "write_service_time_s",
                "analytical_time_s", "effective_time_s"):
            object.__setattr__(
                self, field_name,
                _require_finite(
                    f"MemoryArbitrationDiagnostic.{field_name}",
                    getattr(self, field_name), positive=False))
        expected_read = self.read_bytes / self.read_capacity_per_s
        expected_write = self.write_bytes / self.write_capacity_per_s
        if not math.isclose(
                self.read_service_time_s, expected_read,
                rel_tol=1e-12, abs_tol=1e-18):
            raise ValueError(
                "read service time must equal read bytes / capacity")
        if not math.isclose(
                self.write_service_time_s, expected_write,
                rel_tol=1e-12, abs_tol=1e-18):
            raise ValueError(
                "write service time must equal write bytes / capacity")
        expected_analytical_by_mode = {
            "independent_directional_capacity": max(
                expected_read, expected_write),
            "shared_common_capacity": expected_read + expected_write,
            "shared_directional_capacity": expected_read + expected_write,
        }
        expected_analytical = expected_analytical_by_mode[
            self.arbitration_mode]
        if not math.isclose(
                self.analytical_time_s, expected_analytical,
                rel_tol=1e-12, abs_tol=1e-18):
            raise ValueError(
                "analytical arbitration time disagrees with the declared "
                "directional composition")
        if not isinstance(self.overridden, bool):
            raise TypeError(
                "MemoryArbitrationDiagnostic.overridden must be boolean")
        if (not self.overridden and not math.isclose(
                self.effective_time_s, self.analytical_time_s,
                rel_tol=1e-12, abs_tol=1e-18)):
            raise ValueError(
                "non-overridden arbitration must use its analytical time")
        if self.schedule_status != "unavailable":
            raise ValueError(
                "current memory arbitration schedule status must be "
                "'unavailable'")
        if self.bound_kind != "lower_bound":
            raise ValueError(
                "current memory arbitration result must be a lower bound")
        if self.switch_count is not None:
            raise ValueError(
                "switch_count requires a future typed global epoch schedule")
        if (self.read_to_write_turnaround_s is not None
                or self.write_to_read_turnaround_s is not None):
            raise ValueError(
                "turnaround time requires a future typed global epoch "
                "schedule")


@dataclass(frozen=True)
class PipelineTaskDiagnostic:
    task_name: str
    worker: int
    stage_start_s: Mapping[str, float]
    stage_end_s: Mapping[str, float]
    repeat_count: int = 1

    def __post_init__(self) -> None:
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)):
            raise TypeError(
                "PipelineTaskDiagnostic.repeat_count must be an integer")
        if self.repeat_count <= 0:
            raise ValueError(
                "PipelineTaskDiagnostic.repeat_count must be positive")
        object.__setattr__(
            self, "stage_start_s",
            _immutable_float_mapping(
                "PipelineTaskDiagnostic.stage_start_s", self.stage_start_s))
        object.__setattr__(
            self, "stage_end_s",
            _immutable_float_mapping(
                "PipelineTaskDiagnostic.stage_end_s", self.stage_end_s))


@dataclass(frozen=True)
class PipelineDiagnostics:
    execution_time_s: float
    initiation_interval_s: float
    worker_makespan_s: Mapping[str, float]
    stage_service_s: Mapping[str, float]
    critical_path: Tuple[str, ...]
    tasks: Tuple[PipelineTaskDiagnostic, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worker_makespan_s",
            _immutable_float_mapping(
                "PipelineDiagnostics.worker_makespan_s",
                self.worker_makespan_s))
        object.__setattr__(
            self, "stage_service_s",
            _immutable_float_mapping(
                "PipelineDiagnostics.stage_service_s", self.stage_service_s))


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Structured explanation emitted by the canonical production predictor."""

    compute_time_s: Mapping[str, float]
    memory: Tuple[MemoryDiagnostics, ...]
    memory_access_classes: Tuple[MemoryAccessClassDiagnostic, ...]
    memory_arbitration: Tuple[MemoryArbitrationDiagnostic, ...]
    pipeline: Optional[PipelineDiagnostics]
    critical_path: Tuple[str, ...]
    support_level: str
    limitations: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "compute_time_s",
            _immutable_float_mapping(
                "PredictionDiagnostics.compute_time_s", self.compute_time_s))
        if not isinstance(self.memory, tuple):
            raise TypeError("PredictionDiagnostics.memory must be a tuple")
        if any(not isinstance(value, MemoryDiagnostics) for value in self.memory):
            raise TypeError(
                "PredictionDiagnostics.memory must contain MemoryDiagnostics")
        if not isinstance(self.memory_access_classes, tuple):
            raise TypeError(
                "PredictionDiagnostics.memory_access_classes must be a tuple")
        if any(not isinstance(value, MemoryAccessClassDiagnostic)
               for value in self.memory_access_classes):
            raise TypeError(
                "PredictionDiagnostics.memory_access_classes must contain "
                "MemoryAccessClassDiagnostic")
        if not isinstance(self.memory_arbitration, tuple):
            raise TypeError(
                "PredictionDiagnostics.memory_arbitration must be a tuple")
        if any(not isinstance(value, MemoryArbitrationDiagnostic)
               for value in self.memory_arbitration):
            raise TypeError(
                "PredictionDiagnostics.memory_arbitration must contain "
                "MemoryArbitrationDiagnostic")
        arbitration_roles = tuple(
            value.resource_role for value in self.memory_arbitration)
        if len(arbitration_roles) != len(set(arbitration_roles)):
            raise ValueError(
                "PredictionDiagnostics.memory_arbitration cannot repeat "
                "resource roles")
        if (self.pipeline is not None
                and not isinstance(self.pipeline, PipelineDiagnostics)):
            raise TypeError(
                "PredictionDiagnostics.pipeline must be PipelineDiagnostics or None")
        if not isinstance(self.critical_path, tuple):
            raise TypeError("PredictionDiagnostics.critical_path must be a tuple")
        if self.support_level not in (
                "complete", "partial", "aggregate_only", "compatibility"):
            raise ValueError(
                "PredictionDiagnostics.support_level must be 'complete', "
                "'partial', 'aggregate_only', or 'compatibility'")
        if not isinstance(self.limitations, tuple):
            raise TypeError("PredictionDiagnostics.limitations must be a tuple")
        normalized_limitations = tuple(
            _require_name("PredictionDiagnostics limitation", value)
            for value in self.limitations)
        if len(normalized_limitations) != len(set(normalized_limitations)):
            raise ValueError(
                "PredictionDiagnostics.limitations must be unique")
        if self.support_level == "complete" and normalized_limitations:
            raise ValueError(
                "complete PredictionDiagnostics cannot declare limitations")
        if self.support_level == "complete" and self.memory_arbitration:
            raise ValueError(
                "complete PredictionDiagnostics cannot contain lower-bound "
                "memory arbitration with an unavailable schedule")
        if self.support_level != "complete" and not normalized_limitations:
            raise ValueError(
                "non-complete PredictionDiagnostics must declare limitations")
        object.__setattr__(self, "limitations", normalized_limitations)


@dataclass
class PredictionInput:
    """Canonical resource-formula input.

    ``resource_time_overrides_ns`` is an optional formula-intermediate
    replacement keyed by hardware resource role.  Production predictors leave
    it empty.  Evaluation can use independently observed per-pipe execution
    times (for example Ascend AIC/AIV pipe times) without introducing another
    predictor or reimplementing the formula outside ``src``.
    """

    demand_per_resource: dict
    eta: float
    wave_efficiency: float
    active_units: int
    sm_freq_hz: float
    dram_freq_hz: float
    grid_size: int
    block_size: int
    variant: str
    mem_model: str
    compute_demands: Tuple[ComputeDemand, ...]
    memory_paths: Tuple[MemoryPathDemand, ...]
    pipeline_plan: Optional[
        Union[
            AttentionPipelineEnvelopeDemand, BarrierPipelineEnvelopeDemand,
            CannGroupedMatmulEnvelopeDemand,
            CannSingleCoreSplitKEnvelopeDemand,
            GpuGroupedGemmPipelineEnvelopeDemand,
            MetaxGroupedOperandPipelineDemand,
            PipelineWorkPlan,
            FiaCrossDomainWorkPlan,
            FiaSplitFuseWorkPlan]]
    launch_grid: Optional[Tuple[int, int, int]] = None
    launch_block: Optional[Tuple[int, int, int]] = None
    launch_task_kind: str = "generic"
    dram_capacity_multiplier: float = 1.0
    include_dispatch_overhead: bool = True
    support_level: str = "complete"
    limitations: Tuple[str, ...] = ()
    resource_time_overrides_ns: dict = field(default_factory=dict)
    phase_ledger: Optional[AnalyticalPhaseLedger] = None
    task_duration_floor_contract: Optional[str] = None
    aiv_worker_start_contract: Optional[str] = None
    aiv_source_entry_fill_contract: Optional[str] = None
    aiv_descriptor_prologue_pipeline: bool = False
    task_prologue_contract: Optional[str] = None
    task_prologue_repetitions: Optional[int] = None
    resident_cta_cohort_refills: Optional[int] = None
    runtime_task_class: Optional[str] = None
    runtime_task_elements: Optional[int] = None
    dependency_paths: Tuple[AnalyticalDependencyPath, ...] = ()
    competing_pipeline_floor: Optional[PipelineWorkPlan] = None
    execution_time_floor_ns: Optional[float] = None
    finite_mixed_operand_stream_startup_ns: Optional[float] = None
    # Independent launch-wide completion after ordinary global-write bytes.
    # Coarse dependency paths only: worker pipelines price their own store.
    global_store_completion_ns: Optional[float] = None
    # Launch-wide time added by CTAs that start, reject their tile and return
    # without entering the operator body.  Measured per such CTA and
    # multiplied by their count; it is additive because those CTAs occupy
    # their AP's resident slots in series with the real work.
    early_exit_cta_retire_ns: Optional[float] = None
    # Whether this launch executes as a node of a replayed CUDA graph rather
    # than as a standalone stream launch.  The two have different measured
    # span floors, and the canonical kernel dataset captures the standalone
    # form while a serving decode step replays a captured graph.
    graph_replayed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.demand_per_resource, Mapping):
            raise TypeError("PredictionInput.demand_per_resource must be a mapping")
        self.demand_per_resource = dict(self.demand_per_resource)
        valid_memory_models = {
            "gpu_mshr", "npu_tile", "pim_bank", "systolic_scratchpad"}
        if self.mem_model not in valid_memory_models:
            raise ValueError(f"unknown PredictionInput.mem_model: {self.mem_model!r}")
        if not isinstance(self.variant, str):
            raise TypeError("PredictionInput.variant must be a string")
        if not self.variant:
            raise ValueError("PredictionInput.variant must not be empty")
        if not isinstance(self.compute_demands, tuple):
            raise TypeError("PredictionInput.compute_demands must be a tuple")
        if any(not isinstance(value, ComputeDemand)
               for value in self.compute_demands):
            raise TypeError(
                "PredictionInput.compute_demands must contain ComputeDemand")
        compute_roles = tuple(
            value.resource_role for value in self.compute_demands)
        if len(compute_roles) != len(set(compute_roles)):
            raise ValueError(
                "PredictionInput.compute_demands cannot repeat resource roles")
        if not isinstance(self.memory_paths, tuple):
            raise TypeError("PredictionInput.memory_paths must be a tuple")
        if any(not isinstance(value, MemoryPathDemand)
               for value in self.memory_paths):
            raise TypeError(
                "PredictionInput.memory_paths must contain MemoryPathDemand")
        path_names = tuple(value.name for value in self.memory_paths)
        if len(path_names) != len(set(path_names)):
            raise ValueError(
                "PredictionInput.memory_paths cannot repeat path names")
        if self.pipeline_plan is not None and not isinstance(
                self.pipeline_plan,
                (
                    AttentionPipelineEnvelopeDemand,
                    BarrierPipelineEnvelopeDemand,
                    CannGroupedMatmulEnvelopeDemand,
                    CannSingleCoreSplitKEnvelopeDemand,
                    GpuGroupedGemmPipelineEnvelopeDemand,
                    MetaxGroupedOperandPipelineDemand,
                    PipelineWorkPlan,
                    FiaCrossDomainWorkPlan,
                    FiaSplitFuseWorkPlan)):
            raise TypeError(
                "PredictionInput.pipeline_plan must be "
                "AttentionPipelineEnvelopeDemand, "
                "BarrierPipelineEnvelopeDemand, PipelineWorkPlan, "
                "CannGroupedMatmulEnvelopeDemand, "
                "CannSingleCoreSplitKEnvelopeDemand, "
                "GpuGroupedGemmPipelineEnvelopeDemand, "
                "MetaxGroupedOperandPipelineDemand, "
                "FiaCrossDomainWorkPlan, FiaSplitFuseWorkPlan, or None")
        if (self.competing_pipeline_floor is not None
                and not isinstance(
                    self.competing_pipeline_floor, PipelineWorkPlan)):
            raise TypeError(
                "PredictionInput.competing_pipeline_floor must be "
                "PipelineWorkPlan or None")
        if (self.execution_time_floor_ns is not None
                and (isinstance(self.execution_time_floor_ns, bool)
                     or not isinstance(
                         self.execution_time_floor_ns, (int, float))
                     or not math.isfinite(self.execution_time_floor_ns)
                     or self.execution_time_floor_ns < 0)):
            raise ValueError(
                "PredictionInput.execution_time_floor_ns must be a finite "
                "non-negative number or None")
        if isinstance(
                self.pipeline_plan,
                (AttentionPipelineEnvelopeDemand,
                 BarrierPipelineEnvelopeDemand, PipelineWorkPlan)):
            requires_exact_requests = any(
                demand.exact_request_demands
                for demand in self.compute_demands)
            requires_worker_services = any(
                service.operations_by_worker
                for demand in self.compute_demands
                for service in demand.service_demands)
            if requires_exact_requests or requires_worker_services:
                if isinstance(
                        self.pipeline_plan, AttentionPipelineEnvelopeDemand):
                    unique_phases = {}
                    for work_class in self.pipeline_plan.classes:
                        for phase in work_class.phases:
                            unique_phases[id(phase)] = phase
                    pipeline_stages = tuple(
                        stage
                        for phase in unique_phases.values()
                        for stage in phase.stages)
                elif isinstance(
                        self.pipeline_plan, BarrierPipelineEnvelopeDemand):
                    pipeline_stages = tuple(
                        stage
                        for sequence in self.pipeline_plan.classes
                        for task in sequence.tasks
                        for stage in task.stages)
                else:
                    pipeline_stages = tuple(
                        stage
                        for entry in self.pipeline_plan.tasks
                        for sequence in (
                            entry.patterns
                            if isinstance(entry, PipelineTaskProgramDemand)
                            else (entry,))
                        for task in (
                            sequence.tasks
                            if isinstance(sequence, PipelineTaskSequenceDemand)
                            else (sequence,))
                        for stage in task.stages
                    )
            if requires_exact_requests and not any(
                        stage.compute_exact_request_demands
                        for stage in pipeline_stages):
                raise ValueError(
                    "exact request services cannot be placed in a "
                    "PipelineWorkPlan without a source-closed timed worker "
                    "lifecycle")
            if requires_worker_services and not any(
                        stage.compute_service_demands
                        for stage in pipeline_stages):
                raise ValueError(
                    "worker-scheduled typed services cannot be placed in a "
                    "PipelineWorkPlan without a source-closed timed worker "
                    "lifecycle")
        elif isinstance(
                self.pipeline_plan,
                (FiaCrossDomainWorkPlan, FiaSplitFuseWorkPlan)):
            self.pipeline_plan.validate_conservation(self.compute_demands)
            if isinstance(self.pipeline_plan, FiaSplitFuseWorkPlan):
                self.pipeline_plan.validate_memory_conservation(
                    self.memory_paths)
        if self.phase_ledger is not None:
            if not isinstance(self.phase_ledger, AnalyticalPhaseLedger):
                raise TypeError(
                    "PredictionInput.phase_ledger must be "
                    "AnalyticalPhaseLedger or None")
            self.phase_ledger.validate_conservation(
                self.compute_demands, self.memory_paths)
        if not isinstance(self.dependency_paths, tuple):
            raise TypeError("PredictionInput.dependency_paths must be a tuple")
        if any(not isinstance(value, AnalyticalDependencyPath)
               for value in self.dependency_paths):
            raise TypeError(
                "PredictionInput.dependency_paths must contain "
                "AnalyticalDependencyPath")
        path_names = tuple(path.name for path in self.dependency_paths)
        if len(path_names) != len(set(path_names)):
            raise ValueError(
                "PredictionInput dependency path names must be unique")
        if self.dependency_paths and self.pipeline_plan is not None:
            raise ValueError(
                "coarse analytical dependency paths cannot combine with a "
                "worker pipeline plan")
        completion_paths = tuple(path for path in self.dependency_paths
            if GLOBAL_STORE_COMPLETION_SERVICE in path.services)
        if self.global_store_completion_ns is None:
            if completion_paths:
                raise ValueError("global store completion requires a measured service")
        else:
            value = self.global_store_completion_ns
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("global store completion must be finite and positive")
            if (self.mem_model != "gpu_mshr" or self.launch_task_kind != "generic"
                    or self.pipeline_plan is not None or len(completion_paths) != 1):
                raise ValueError("global store completion requires one coarse GPU dependency path")
            path, = completion_paths
            if (path.services[-2:] != (
                    "memory:global_write", GLOBAL_STORE_COMPLETION_SERVICE)
                    or any(GLOBAL_STORE_COMPLETION_SERVICE in group
                           for group in path.concurrent_service_groups)
                    or any(service == GLOBAL_STORE_COMPLETION_SERVICE
                           for service, _ in path.serial_service_counts)
                    or not any(p.name == "global_write" for p in self.memory_paths)):
                raise ValueError("global store completion must follow its write exactly once")
        required_resources = {
            "l1_cache", "smem", "xbar", "l2_cache", "dram",
            "mte3",
            "compute_tensor", "compute_vector", "compute_reduction",
            "compute_fma",
            "compute_sfu", "compute_alu", "warp_issue", "scalar_aiv",
        }
        supplied_resources = set(self.demand_per_resource)
        missing = required_resources - supplied_resources
        unknown = supplied_resources - required_resources
        if missing or unknown:
            raise ValueError(
                "PredictionInput resource-demand schema mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}")
        invalid_demands = {
            role: value
            for role, value in self.demand_per_resource.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0)
        }
        if invalid_demands:
            raise ValueError(
                "resource demands must be finite non-negative numbers: "
                f"{invalid_demands}")
        if not isinstance(self.resource_time_overrides_ns, Mapping):
            raise TypeError(
                "PredictionInput.resource_time_overrides_ns must be a mapping")
        self.resource_time_overrides_ns = dict(
            self.resource_time_overrides_ns)
        invalid_override_keys = [
            role for role in self.resource_time_overrides_ns
            if not isinstance(role, str) or not role
        ]
        if invalid_override_keys:
            raise TypeError(
                "resource-time override roles must be non-empty strings: "
                f"{invalid_override_keys}")
        invalid_override_values = {
            role: value
            for role, value in self.resource_time_overrides_ns.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0)
        }
        if invalid_override_values:
            raise ValueError(
                "resource-time overrides must be finite non-negative "
                f"nanoseconds: {invalid_override_values}")
        if self.launch_task_kind not in (
                "generic", "aic", "aiv", "mix_aiv", "mixed"):
            raise ValueError(
                "PredictionInput.launch_task_kind must be 'generic', 'aic', "
                "'aiv', 'mix_aiv', or 'mixed'")
        if self.task_duration_floor_contract is not None:
            if not isinstance(self.task_duration_floor_contract, str):
                raise TypeError(
                    "PredictionInput.task_duration_floor_contract must be "
                    "a string or None")
            supported_task_floors = {
                AIV_COMPLETE_TASK_FLOOR_ONLY,
                AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
                AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
                AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
                GPU_SOURCE_PIPELINE_OUTER_ENVELOPE,
                WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
                TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
                TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR,
            }
            if self.task_duration_floor_contract not in supported_task_floors:
                raise ValueError(
                    "PredictionInput.task_duration_floor_contract is "
                    "unsupported: "
                    f"{self.task_duration_floor_contract!r}")
            if self.task_duration_floor_contract in (
                    AIV_COMPLETE_TASK_FLOOR_ONLY,
                    AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
                    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
                    AIV_SOURCE_PIPELINE_LAUNCH_ONLY):
                if (self.mem_model != "npu_tile"
                        or self.launch_task_kind != "aiv"):
                    raise ValueError(
                        "AIV task composition contract requires an aiv "
                        "npu_tile launch")
                if (self.task_duration_floor_contract in (
                        AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
                        AIV_SOURCE_PIPELINE_LAUNCH_ONLY)
                        and self.pipeline_plan is None):
                    raise ValueError(
                        "AIV source-pipeline composition contract requires a "
                        "pipeline plan")
            elif (self.task_duration_floor_contract
                  == GPU_SOURCE_PIPELINE_OUTER_ENVELOPE):
                if (self.mem_model != "gpu_mshr"
                        or self.launch_task_kind != "generic"
                        or self.pipeline_plan is None):
                    raise ValueError(
                        "GPU source-pipeline outer envelope requires a "
                        "generic gpu_mshr pipeline")
            elif (self.mem_model != "gpu_mshr"
                  or self.launch_task_kind != "generic"):
                raise ValueError(
                    "task-duration floor contract requires a "
                    "generic gpu_mshr launch")
            if not self.include_dispatch_overhead:
                raise ValueError(
                    "task-duration floor contract requires launch terms to "
                    "be enabled")
            if self.launch_grid is None or self.launch_block is None:
                raise ValueError(
                    "task-duration floor contract requires full launch "
                    "geometry")
        if self.aiv_worker_start_contract is not None:
            if self.aiv_worker_start_contract != AIV_WORKER_START_FILL:
                raise ValueError(
                    "PredictionInput.aiv_worker_start_contract is "
                    "unsupported: "
                    f"{self.aiv_worker_start_contract!r}")
            if (self.mem_model != "npu_tile"
                    or self.launch_task_kind != "aiv"
                    or not isinstance(self.pipeline_plan, PipelineWorkPlan)):
                raise ValueError(
                    "AIV worker-start fill requires an AIV npu_tile worker "
                    "pipeline")
            if self.task_duration_floor_contract not in (
                    AIV_COMPLETE_TASK_FLOOR_ONLY,
                    AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
                    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
                    AIV_SOURCE_PIPELINE_LAUNCH_ONLY):
                raise ValueError(
                    "AIV worker-start fill requires a source pipeline, an "
                    "empty task shell plus source program, or the "
                    "independent complete task floor")
            if not self.include_dispatch_overhead:
                raise ValueError(
                    "AIV worker-start fill requires launch terms to be "
                    "enabled")
        if self.aiv_source_entry_fill_contract is not None:
            if self.aiv_source_entry_fill_contract != AIV_SOURCE_ENTRY_FILL:
                raise ValueError(
                    "PredictionInput.aiv_source_entry_fill_contract is "
                    "unsupported: "
                    f"{self.aiv_source_entry_fill_contract!r}")
            if self.aiv_worker_start_contract is not None:
                raise ValueError(
                    "AIV source-entry fill cannot replace worker-start "
                    "composition")
            if (self.mem_model != "npu_tile"
                    or self.launch_task_kind != "aiv"
                    or not isinstance(self.pipeline_plan, PipelineWorkPlan)
                    or self.task_duration_floor_contract is not None
                    or not self.include_dispatch_overhead):
                raise ValueError(
                    "AIV source-entry fill requires a legacy AIV npu_tile "
                    "worker pipeline with enabled launch terms")
        if ((self.task_prologue_contract is None)
                != (self.task_prologue_repetitions is None)):
            raise ValueError(
                "task prologue contract and repetitions must be declared "
                "together")
        if self.task_prologue_contract is not None:
            if not isinstance(self.task_prologue_contract, str):
                raise TypeError(
                    "PredictionInput.task_prologue_contract must be a "
                    "string or None")
            if self.task_prologue_contract not in (
                    CANN_AIV_TILING_MATERIALIZATION_PROLOGUE,
                    CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE,
                    CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE,
                    CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE,
                    CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE):
                raise ValueError(
                    "PredictionInput.task_prologue_contract is unsupported: "
                    f"{self.task_prologue_contract!r}")
            if self.mem_model != "npu_tile" or self.launch_task_kind != "aiv":
                raise ValueError(
                    "CANN AIV task prologue requires an aiv npu_tile launch")
            if not self.include_dispatch_overhead:
                raise ValueError(
                    "CANN AIV task prologue requires launch terms to be "
                    "enabled")
            if (isinstance(self.task_prologue_repetitions, bool)
                    or not isinstance(self.task_prologue_repetitions, int)
                    or self.task_prologue_repetitions <= 0):
                raise ValueError(
                    "task prologue repetitions must be a positive integer")
        if not isinstance(self.aiv_descriptor_prologue_pipeline, bool):
            raise TypeError(
                "PredictionInput.aiv_descriptor_prologue_pipeline must be "
                "boolean")
        if self.aiv_descriptor_prologue_pipeline:
            if (self.mem_model != "npu_tile"
                    or self.launch_task_kind != "aiv"
                    or not isinstance(self.pipeline_plan, PipelineWorkPlan)
                    or not self.include_dispatch_overhead):
                raise ValueError(
                    "AIV descriptor/prologue pipelining requires an enabled "
                    "AIV npu_tile worker pipeline")
            if (self.task_duration_floor_contract is not None
                    or self.aiv_worker_start_contract is not None
                    or self.task_prologue_contract is None
                    or self.task_prologue_repetitions != 1):
                raise ValueError(
                    "AIV descriptor/prologue pipelining requires the legacy "
                    "descriptor scheduler and exactly one typed prologue")
        if self.resident_cta_cohort_refills is not None:
            if (isinstance(self.resident_cta_cohort_refills, bool)
                    or not isinstance(self.resident_cta_cohort_refills, int)
                    or self.resident_cta_cohort_refills <= 0):
                raise ValueError(
                    "resident CTA cohort refills must be a positive integer")
            if (self.mem_model != "gpu_mshr"
                    or self.launch_task_kind != "generic"
                    or not isinstance(self.pipeline_plan, PipelineWorkPlan)):
                raise ValueError(
                    "resident CTA cohort refills require a generic GPU "
                    "worker pipeline")
            if not self.include_dispatch_overhead:
                raise ValueError(
                    "resident CTA cohort refills require launch terms to be "
                    "enabled")
        if self.finite_mixed_operand_stream_startup_ns is not None:
            if (isinstance(self.finite_mixed_operand_stream_startup_ns, bool)
                    or not isinstance(
                        self.finite_mixed_operand_stream_startup_ns,
                        (int, float))
                    or not math.isfinite(
                        self.finite_mixed_operand_stream_startup_ns)
                    or self.finite_mixed_operand_stream_startup_ns < 0):
                raise ValueError(
                    "PredictionInput.finite_mixed_operand_stream_startup_ns "
                    "must be a finite non-negative number")
            if not self.include_dispatch_overhead:
                raise ValueError(
                    "finite mixed operand-stream startup requires launch "
                    "terms to be enabled")
        if not isinstance(self.graph_replayed, bool):
            raise TypeError("PredictionInput.graph_replayed must be boolean")
        if self.early_exit_cta_retire_ns is not None:
            value = self.early_exit_cta_retire_ns
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(
                    "PredictionInput.early_exit_cta_retire_ns must be a "
                    "finite non-negative number or None")
            if not self.include_dispatch_overhead:
                raise ValueError(
                    "early-exit CTA retire requires launch terms to be "
                    "enabled")
        if ((self.runtime_task_class is None)
                != (self.runtime_task_elements is None)):
            raise ValueError(
                "runtime task class and element count must be declared "
                "together")
        if self.runtime_task_class is not None:
            if self.runtime_task_class not in (
                    PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
                    PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS):
                raise ValueError(
                    "PredictionInput.runtime_task_class is unsupported: "
                    f"{self.runtime_task_class!r}")
            if (isinstance(self.runtime_task_elements, bool)
                    or not isinstance(self.runtime_task_elements, int)
                    or self.runtime_task_elements <= 0):
                raise ValueError(
                    "PredictionInput.runtime_task_elements must be a "
                    "positive integer")
            if self.mem_model != "gpu_mshr" or self.launch_task_kind != "generic":
                raise ValueError(
                    "PyTorch runtime task class requires a generic gpu_mshr "
                    "launch")
        if self.support_level not in (
                "complete", "partial", "aggregate_only", "compatibility"):
            raise ValueError(
                "PredictionInput.support_level must be 'complete', "
                "'partial', 'aggregate_only', or 'compatibility'")
        if not isinstance(self.limitations, tuple):
            raise TypeError("PredictionInput.limitations must be a tuple")
        if any(not isinstance(value, str) or not value.strip()
               for value in self.limitations):
            raise ValueError(
                "PredictionInput.limitations must contain non-empty strings")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("PredictionInput.limitations must be unique")
        if self.support_level == "complete" and self.limitations:
            raise ValueError(
                "complete PredictionInput cannot declare limitations")
        if self.support_level != "complete" and not self.limitations:
            raise ValueError(
                "non-complete PredictionInput must declare limitations")
        source_independence_missing = any(
            path.source.independent_requests_per_producer is None
            for path in self.memory_paths)
        source_limitation = "workload_producer_independence_unavailable"
        if source_independence_missing and source_limitation not in self.limitations:
            raise ValueError(
                "memory paths without workload producer independence must "
                f"declare limitation {source_limitation!r}")
        for name in ("include_dispatch_overhead",):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"PredictionInput.{name} must be boolean")
        if self.mem_model == "npu_tile":
            if self.support_level not in ("aggregate_only", "partial"):
                raise ValueError(
                    "npu_tile prediction must be aggregate_only or partial")
            if (self.support_level == "aggregate_only"
                    and self.pipeline_plan is not None):
                raise ValueError(
                    "npu_tile aggregate-only prediction cannot attach a "
                    "worker pipeline")
            if (self.support_level == "partial"
                    and self.pipeline_plan is None
                    and (self.phase_ledger is None
                         or not self.dependency_paths)):
                raise ValueError(
                    "npu_tile partial prediction requires a source-backed "
                    "worker pipeline or analytical dependency envelope")
            if self.launch_task_kind == "generic":
                raise ValueError(
                    "npu_tile prediction must declare an AIC, AIV, or mixed "
                    "launch task kind")
            launch_limitation = (
                "launch_task_duration_evidence_unavailable")
            if (self.include_dispatch_overhead
                    and launch_limitation in self.limitations):
                raise ValueError(
                    "npu_tile prediction cannot consume a launch floor while "
                    "declaring its evidence unavailable")
            if any(path.rate_unit != "bytes" for path in self.memory_paths):
                raise ValueError(
                    "npu_tile prediction requires byte-only memory paths")
        numeric_values = {
            "eta": self.eta,
            "wave_efficiency": self.wave_efficiency,
            "sm_freq_hz": self.sm_freq_hz,
            "dram_freq_hz": self.dram_freq_hz,
            "dram_capacity_multiplier": self.dram_capacity_multiplier,
        }
        non_finite = {
            name: value for name, value in numeric_values.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value))
        }
        if non_finite:
            raise ValueError(
                f"PredictionInput numeric fields must be finite: {non_finite}")
        for name in ("active_units", "grid_size", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"PredictionInput.{name} must be an integer")
        if (self.active_units <= 0 or self.grid_size <= 0 or self.block_size <= 0
                ):
            raise ValueError(
                "PredictionInput active_units, grid_size and block_size must "
                "be positive")
        if (self.launch_grid is None) != (self.launch_block is None):
            raise ValueError(
                "PredictionInput launch_grid and launch_block must both be "
                "present or both be absent"
            )
        if self.launch_grid is not None:
            normalized_dimensions = []
            for name in ("launch_grid", "launch_block"):
                dim3 = getattr(self, name)
                if (not isinstance(dim3, (list, tuple)) or len(dim3) != 3
                        or any(isinstance(axis, bool)
                               or not isinstance(axis, int) or axis <= 0
                               for axis in dim3)):
                    raise ValueError(
                        f"PredictionInput.{name} must be a positive dim3"
                    )
                normalized_dimensions.append(tuple(dim3))
            self.launch_grid, self.launch_block = normalized_dimensions
            if math.prod(self.launch_grid) != self.grid_size:
                raise ValueError(
                    "PredictionInput launch_grid disagrees with grid_size"
                )
            if math.prod(self.launch_block) != self.block_size:
                raise ValueError(
                    "PredictionInput launch_block disagrees with block_size"
                )
        if self.eta <= 0:
            raise ValueError("PredictionInput.eta must be positive")
        if not 0 < self.wave_efficiency <= 1:
            raise ValueError(
                "PredictionInput.wave_efficiency must be within (0, 1]")
        if self.sm_freq_hz <= 0 or self.dram_freq_hz <= 0:
            raise ValueError("PredictionInput clock frequencies must be positive")
        if self.dram_capacity_multiplier <= 0:
            raise ValueError(
                "PredictionInput.dram_capacity_multiplier must be positive")
