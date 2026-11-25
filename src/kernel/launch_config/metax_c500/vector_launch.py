"""MACA 3.7.2 MetaX C500 source-bound Vector launch selection.

The vLLM-MetaX plugin source is pinned to commit
``11b156053ec0344b8637c621c49b25c45f0a2331``.  The matching editable
checkout and loaded ``_C.abi3.so`` are retained under the versioned C500
kernel-observation artifact.  The embedding-gather evidence is independently
bound to the pinned torch ``libtorch_cuda.so`` identified below; launch facts
use no kernel duration.
"""

from typing import Mapping, Optional


def moe_count_sort_source_program():
    """Return the deployed MetaX histogram interface identity."""
    return "metax_routed_token_histogram"

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import (
    ceil_div,
    launch_policy,
    require_boolean,
    require_dtype,
    require_gpu_vector_variant,
    require_non_empty_string,
    require_paged_kv_cache_contract,
    require_shape,
    select_gpu_vector_launch,
)
from src.kernel.model_contract import (
    SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
    SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER,
    SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER,
    SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
    SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
    SOURCE_QK_ROPE_LOOP_PARAMETER,
    SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER,
    SOURCE_VECTORIZED_KV_COPY_PARAMETER,
    SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER,
)


_SILU_VALUES_PER_WAVE = 512
_SILU_VALUES_PER_BLOCK = 4096
_SILU_WAVES_PER_BLOCK = 8
_SILU_ACTIVE_WAVE_INSTRUCTIONS = 260
_SILU_INACTIVE_WAVE_INSTRUCTIONS = 10
_SILU_ACTIVE_WAVE_SLOT2_INSTRUCTIONS = 247
_SILU_INACTIVE_WAVE_SLOT2_INSTRUCTIONS = 3
_INDEX_SELECT_SMALL_ROW_LIMIT = 16
_INDEX_SELECT_THREADS = 128
_INDEX_SELECT_SMALL_BLOCKS_PER_COMPUTE_UNIT = 32
_INDEX_SELECT_LARGE_BLOCKS_PER_COMPUTE_UNIT = 64
_INDEX_SELECT_REGISTERS_PER_THREAD = 14
_INDEX_SELECT_OBSERVED_ROWS = 8192
_INDEX_SELECT_INT32_LIMIT = 1 << 31
# Retained for documentation: these are the token-embedding widths the
# launch-trace corroboration covered.  The pinned binary proof above states
# that host launch construction has no hidden-specific branch, so the geometry
# below is evaluated for any positive slice width; the C500 llama alignment
# capture corroborates head_dim=128 directly (small path grid=(1,1,1)
# block=(128,1,1), large path grid=tokens).

# torch 2.4.0+metax3.7.2.0 ATen/native/cuda/maca_kernels/CUDALoops_maca_arity2.cuh.
# The arity-2 dispatcher selects nt=128 for both three-dimensional rotate-half
# programs (L7152, L7218).  launch_legacy_kernel_maca_3_2_dim0_contiguous
# (L3466) takes vec=8 only when the innermost extent reaches 128 and otherwise
# vec=4, with grid=ceil(N/(nt*vec)); the
# ..._arg1_dim1_broadcast launcher (L3605) pins vec=4 and uses
# grid=ceil(N/(nt*vec*vec)).  The first branch additionally requires N>=5120.
_MACA_ARITY2_NT = 128
_MACA_ARITY2_DIM0_CONTIGUOUS_MIN_NUMEL = 5120
_MACA_ARITY2_WIDE_INNER = 128
# launch_legacy_kernel_maca_2_2_broadcast (L3068) runs when the iterator
# collapses to two dimensions: block_dim_x=64, load_num=inner/64, vec bounded
# by load_num, grid=(load_num/vec, getMaxGridSize(...)->rows, 1).
_MACA_ARITY2_2D_BLOCK = 64
# MACA_CatArrayBatchedCopyNoPartialWrite is compiled into libtorch_cuda and is
# neither a shipped header nor an exported dynamic symbol, so unlike every
# other entry here its geometry is a versioned duration-free launch
# observation: block 128 with eight values per thread, invariant across the 245
# distinct shapes of the pinned C500 llama alignment capture.
_CAT_ARRAY_THREADS = 128
_CAT_ARRAY_VALUES_PER_THREAD = 8
_FILL_OPT_MIN_ELEMENTS = 1024 * 64
_FILL_STANDARD_THREADS = 256
_FILL_OPT_THREADS = 1024
_FILL_STANDARD_ELEMENTS_PER_THREAD = 4
_FILL_MAX_ELEMENTS = (1 << 31) - 1
# csrc/libtorch_stable/cuda_vec_utils.cuh VecTraits<false>::ARCH_MAX_VEC_SIZE
# and the 1024-thread cap in LAUNCH_ACTIVATION_GATE_KERNEL.
_ACTIVATION_GATE_VECTOR_BYTES = 16
_BF16_ELEMENT_BYTES = 2
_ACTIVATION_GATE_MAX_THREADS = 1024
# topk_softmax_kernels.cu launcher constants.
_TOPK_SOFTMAX_WARPS_PER_BLOCK = 4
_TOPK_SOFTMAX_MAX_BYTES_PER_LDG = 16
_TOPK_SOFTMAX_GATING_ELEMENT_BYTES = 4
# Observed template instantiation in the retained C500 decode step.
_TOPK_SOFTMAX_COMPILED_WARP_WIDTH = 32
# Observed instantiation of the closed MetaX alignment kernel.
_MOE_ALIGN_OPT_SPECIALIZED_EXPERTS = 128
_MOE_ALIGN_OPT_THREADS = 512
# Observed grid/block of the routed-token histogram.
_MOE_HISTOGRAM_THREADS = 512
# ATen Reduce.cuh: mnt_wrapper<float>::MAX_NUM_THREADS.
_REDUCE_MAX_THREADS = 512
# Observed block of the routing-weight rescale.
_ROUTING_SCALE_THREADS = 128
# Observed block of the expert-map gather.
_EXPERT_MAP_LOOKUP_THREADS = 128
# Observed geometry of the router-logit widening copy.
_LOGIT_CAST_THREADS = 256
_LOGIT_CAST_VALUES_PER_THREAD = 4


def _last_power_of_two(value: int) -> int:
    """ATen `last_pow2`: the largest power of two not exceeding `value`."""

    if value < 1:
        raise ValueError("last_pow2 requires a positive extent")
    return 1 << (value.bit_length() - 1)
# `act_and_mul_kernel` reads exactly two operands per element.
_ACT_AND_MUL_INDEPENDENT_OPERAND_LOADS = 2
_MOE_ELEMENTWISE_THREADS = 128
_MOE_ELEMENTWISE_VALUES_PER_THREAD = 8
_MOE_ELEMENTWISE_TENSOR_ITERATOR_INDEX_LIMIT = 1 << 29
_MOE_SUM_THREADS = 512
_MOE_SUM_OUTPUT_VECTOR_WIDTH = 4
_MOE_SUM_TENSOR_ITERATOR_INDEX_LIMIT = 1 << 30
# The pinned reshape_and_cache_flash body loads the key and the value of one
# loop iteration back to back from two distinct tensors.  Neither address
# depends on the other's result, so one iteration keeps both requests
# outstanding.
_KV_CACHE_CONCURRENT_READ_TENSORS = 2
_GROUPED_STRIDED_COPY_THREADS = 64
_GROUPED_STRIDED_COPY_VALUES_PER_THREAD = 8
# The pinned torch 2.4.0 + MACA 3.7.2 TensorIterator symbol records the
# complete static launch program as ``elementwise_kernel_2_1_dim0_contiguous
# <64, 8, bf16, bf16, ...direct_copy...>``.  Canonical duration-free launch
# observations confirm grid=ceil(elements/(64*8)) for both Q and K slices.
# The template arguments and launch geometry are program facts; no observed
# kernel duration is used here or by the corresponding memory-service model.
# Source identity and addressing-domain proof:
# microbench/metax_c500/references/embedding_index_select_torch_2_4_maca_3_7_2.json.
# The BF16 index_select dispatcher at 0x1eeff40 uses the 16-index boundary;
# its caller at 0x1deefa3 constructs 128-thread launches at
# 0x1deeaa1...0x1deeb4c, with small/large caps of 32/64 blocks per
# compute unit, after checking all operand index extents. Construction has no
# hidden-specific branch: hidden enters only through sliceSize for the small
# path and totalSize=rows*hidden for the large path, so the binary proof covers
# all four enumerated ShareGPT hidden sizes. Launch traces independently
# corroborate rows 1...8192 and hidden 2560/4096 via the path-sorted 37-file set
# containing sharegpt_phase1.json plus each dense MetaX case's manifest.json,
# scheduler.jsonl, operators.jsonl, and launches.jsonl; SHA256 of that set's
# path-sorted `sha256sum` output is
# 527871f7b7f95812c330a039531d3dcae21f5cdbd120355ac3fc026a85ca0093.


def _source_bound_silu_wave_instruction_lower_bound(
        rows: int, intermediate: int) -> float:
    """Return non-SNOP dynamic instructions for the fixed BF16 program.

    MACA 3.7.2 ``_C.abi3.so`` SHA256
    ``a8715e7f7fe1790b46586fd8780036177c083504b24556ae1d661d0f0b569955``
    contains the BF16 ``act_and_mul_kernel_bd_opt`` program at 0x27600.  Its
    complete disassembly has a 260-instruction active path and a 10-instruction
    bounds-check/return path.  Each wave owns 512 output values and each
    512-thread block dispatches eight waves for a 4096-value chunk.  This is an
    auditable issue lower bound, not a latency estimate: load dependencies,
    SFU dependencies, scheduler interleaving, and pipeline fill/drain are not
    derivable by adding opcode initiation intervals to the instruction count.
    """

    active_waves_per_row = ceil_div(intermediate, _SILU_VALUES_PER_WAVE)
    dispatched_waves_per_row = (
        ceil_div(intermediate, _SILU_VALUES_PER_BLOCK)
        * _SILU_WAVES_PER_BLOCK)
    inactive_waves_per_row = (
        dispatched_waves_per_row - active_waves_per_row)
    return float(rows * (
        _SILU_ACTIVE_WAVE_INSTRUCTIONS * active_waves_per_row
        + _SILU_INACTIVE_WAVE_INSTRUCTIONS * inactive_waves_per_row
    ))


def _source_bound_silu_wave_slot2_instruction_lower_bound(
        rows: int, intermediate: int) -> float:
    """Return dynamic instructions statically assigned to issue slot 2.

    The fixed production code object has 247 active-path and three
    inactive-path MTE/Vector-ALU instructions. Canonical non-duration counters
    independently conserve those exact per-wave counts. The result is a
    physical slot demand, not an empirical kernel-time correction.
    """

    active_waves_per_row = ceil_div(intermediate, _SILU_VALUES_PER_WAVE)
    dispatched_waves_per_row = (
        ceil_div(intermediate, _SILU_VALUES_PER_BLOCK)
        * _SILU_WAVES_PER_BLOCK)
    inactive_waves_per_row = (
        dispatched_waves_per_row - active_waves_per_row)
    return float(rows * (
        _SILU_ACTIVE_WAVE_SLOT2_INSTRUCTIONS * active_waves_per_row
        + _SILU_INACTIVE_WAVE_SLOT2_INSTRUCTIONS * inactive_waves_per_row
    ))


# The deployed `rotary_embedding_kernel` issues exactly four loads per pair,
# and their addresses all come from the loop index alone:
#
#     cos_f = VLLM_LDG(cos_ptr + x_index);
#     sin_f = VLLM_LDG(sin_ptr + x_index);
#     x_f   = arr[x_index];
#     y_f   = arr[y_index];
#
# None of the four addresses depends on another load's value, so a warp can
# hold all four in flight.  `positions[token_idx]` is read once per CTA before
# the loop and is not part of the per-pair request set.
_ROPE_INDEPENDENT_PAIR_LOADS = 4


def _source_bound_rope_read_contract(
        hardware, query_heads, rotary_dim, block_size):
    """Per-warp rope request depth from the pinned NeoX source program.

    Returns `None` when a warp would straddle a head, because the four loads
    then split across the head stride and their tail-free coalescing is no
    longer the single contiguous run the count below assumes.
    """

    embed_dim = rotary_dim // 2
    width = hardware.parallel_width
    if embed_dim % width:
        return None
    warp_bytes = width * 2
    if warp_bytes % hardware.l2_transaction_bytes:
        return None
    sectors_per_load = warp_bytes // hardware.l2_transaction_bytes
    # The Q loop owns `num_heads * embed_dim` pairs and is the phase that
    # sizes the CTA; the shorter K loop reuses a subset of the same warps.
    query_pairs = query_heads * embed_dim
    active_threads = min(query_pairs, block_size)
    if active_threads % width:
        return None
    return (
        float(_ROPE_INDEPENDENT_PAIR_LOADS * sectors_per_load),
        float(active_threads // width),
    )


def pytorch_small_id_narrow_launch(shape, dtype, attributes):
    """Source-invariant small Long-to-Int Copy entry (N<2048)."""
    require_dtype(dtype, ("int64",))
    elements, rows = require_shape(shape, ("element_count", "rows"))
    if (attributes["sampling_source"] != "vllm_0_24_plain_greedy_bf16"
            or attributes["stage"] != "output_int32" or elements != rows):
        raise ValueError("C500 ID narrowing requires its0.24 greedy INT64 output source")
    if not 0 < elements < 2048:
        raise NotImplementedError("C500 ID narrowing requires its small iterator domain below2048")
    return LaunchConfig(algorithm="pytorch_2_4_metax_3_7_2_int64_to_int32",
        grid=(ceil_div(elements, 1024), 1, 1), block=(256, 1, 1),
        parameters={"registers_per_thread": 15.0, "values_per_thread": 4.0},
        execution_variant="vllm_plain_greedy_sample")


def _pytorch_copy_dispatch(attributes):
    """Validate supplied Copy branch facts without reading the simulator env."""
    if "copy_dispatch" not in attributes:
        raise ValueError("C500 Copy requires explicit copy dispatch facts")
    dispatch = attributes["copy_dispatch"]
    if not isinstance(dispatch, Mapping) or set(dispatch) != {
            "disable_unroll_float_opt", "input_alignment_bytes", "output_alignment_bytes"}:
        raise ValueError("C500 copy dispatch requires exactly its flag and two pointer alignments")
    if type(dispatch["disable_unroll_float_opt"]) is not bool:
        raise TypeError("C500 copy optimization flag must be boolean")
    for name in ("input_alignment_bytes", "output_alignment_bytes"):
        alignment = dispatch[name]
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("C500 copy pointer alignment must be a positive power of two")
    return dispatch


def pytorch_int32_int64_cast_launch(elements, attributes, variant):
    """Pinned generic and explicitly selected optimized integer Copy entries.

    The pinned host branch at0x16df749 jumps to0x16e1869 when the cached
    disable flag is true, independently of N/alignment. Small N reaches the
    same entry without needing that runtime fact. Enabled large Copy with
    proven4B alignment uses the matching vt2 entries, both with10 registers.
    """
    if elements <= 0 or 8 * elements >= 2**31:
        raise NotImplementedError("C500 integer cast requires a contiguous unsplit32bit byte extent")
    dispatch = _pytorch_copy_dispatch(attributes) if "copy_dispatch" in attributes else None
    if elements >= 2048:
        if dispatch is None:
            raise NotImplementedError("C500 integer cast >=2048 requires its optimized-copy branch binding or an explicit disabled flag")
        if not dispatch["disable_unroll_float_opt"]:
            if min(dispatch["input_alignment_bytes"], dispatch["output_alignment_bytes"]) < 4:
                raise NotImplementedError("C500 optimized integer Copy requires proven4B input/output alignment")
            return LaunchConfig(algorithm=("pytorch_2_4_metax_3_7_2_int32_to_int64_optimized_tail"
                if elements % 512 else "pytorch_2_4_metax_3_7_2_int32_to_int64_optimized"),
                grid=(ceil_div(elements, 512), 1, 1), block=(256, 1, 1),
                parameters={"registers_per_thread": 10.0, "values_per_thread": 2.0},
                execution_variant=variant)
    return LaunchConfig(algorithm="pytorch_2_4_metax_3_7_2_int32_to_int64",
        grid=(ceil_div(elements, 1024), 1, 1), block=(256, 1, 1),
        parameters={"registers_per_thread": 22.0, "values_per_thread": 4.0},
        execution_variant=variant)


def pytorch_logits_cast_launch(shape, dtype, attributes):
    """Conditional Torch Copy binding; dispatch facts must be supplied.

    CUDALoops_copy.cuh:860-866,5072-5095 and the matching xcore1000
    entries distinguish generic/full/tail programs. Small N unconditionally
    selects generic Copy; large N requires its actual cached flag/alignment.
    """
    require_dtype(dtype, ("bf16",))
    elements, = require_shape(shape, ("element_count",))
    if attributes["sampling_source"] != "vllm_0_24_plain_sampler_bf16":
        raise ValueError("C500 logits cast requires its0.24 BF16 source")
    if elements <= 0 or 4 * elements >= 2**31:
        raise NotImplementedError("C500 logits cast requires its contiguous unsplit32bit domain")
    dispatch = _pytorch_copy_dispatch(attributes) if "copy_dispatch" in attributes else None
    if "copy_input_contiguous" in attributes:
        if type(attributes["copy_input_contiguous"]) is not bool:
            raise TypeError("C500 Copy input contiguity must be boolean")
        if not attributes["copy_input_contiguous"]:
            raise NotImplementedError("C500 logits cast requires contiguous input")
    elif dispatch is None:
        raise ValueError("C500 flag-independent small Copy requires declared input contiguity")
    if elements < 2048 or dispatch is not None and dispatch["disable_unroll_float_opt"]:
        return LaunchConfig(algorithm="pytorch_2_4_metax_3_7_2_bfloat_to_float_generic",
            grid=(ceil_div(elements, 1024), 1, 1), block=(256, 1, 1),
            parameters={"registers_per_thread": 17.0, "values_per_thread": 4.0},
            execution_variant="vllm_logits_fp32_cast")
    if dispatch is None:
        raise ValueError("C500 logits cast requires explicit copy dispatch facts for N>=2048")
    if min(dispatch["input_alignment_bytes"], dispatch["output_alignment_bytes"]) < 4:
        raise NotImplementedError("C500 optimized logits cast requires enabled optimization,4B alignment and its contiguous32bit domain")
    tail = elements % 1024 != 0
    return LaunchConfig(algorithm=("pytorch_2_4_metax_3_7_2_bfloat_to_float_tail" if tail else
        "pytorch_2_4_metax_3_7_2_bfloat_to_float"),
        grid=(ceil_div(elements, 1024), 1, 1), block=(256, 1, 1),
        parameters={"registers_per_thread": 14.0 if tail else 10.0, "values_per_thread": 4.0},
        execution_variant="vllm_logits_fp32_cast")


def pytorch_scalar_index_launch(shape, dtype, attributes):
    """Pinned Torch advanced-indexing scalar copy, not index_select.

    IndexKernelMaca.cuh collapses a single selected row to one iterator
    dimension. Multi-row selection uses the 2D scalar entry only when the
    target runtime's broadcast optimization is explicitly disabled. Neither
    scalar entry needs vector pointer alignment; see the indexing reference.
    """
    require_dtype(dtype, ("bf16",))
    elements, rows, hidden, source_rows = require_shape(
        shape, ("element_count", "rows", "hidden", "source_rows"))
    if (attributes["model_execution_version"] != "vllm_0_24_0"
            or attributes["index_source"] != "query_start_loc_int32_to_int64"
            or ("dtype_bytes" in shape and shape["dtype_bytes"] != 2)
            or source_rows <= 0 or hidden <= 0 or elements != rows * hidden):
        raise ValueError("C500 scalar indexing requires its BF16/INT64 source")
    if rows <= 0 or max(2 * source_rows * hidden, 2 * elements, 8 * rows) >= 2**31:
        raise NotImplementedError("C500 scalar advanced indexing requires positive rows and 32-bit contiguous addressing")
    if "disable_index_broadcast" in attributes and type(attributes["disable_index_broadcast"]) is not bool:
        raise TypeError("C500 index broadcast flag must be boolean")
    two_dimensional = rows != 1 and hidden != 1
    if two_dimensional and ("disable_index_broadcast" not in attributes or not attributes["disable_index_broadcast"]):
        raise NotImplementedError("C500 multi-row scalar indexing requires explicit disabled broadcast optimization; only one row is flag-independent")
    return LaunchConfig(algorithm=("pytorch_2_4_metax_3_7_2_index_elementwise_1_2" if two_dimensional
        else "pytorch_2_4_metax_3_7_2_index_elementwise_1_1"),
        grid=(ceil_div(elements, 512), 1, 1), block=(128, 1, 1),
        parameters={"registers_per_thread": 22.0 if two_dimensional else 14.0, "values_per_thread": 4.0},
        execution_variant="vllm_logits_row_select")


def select_vector_launch(
    shape: Mapping[str, int],
    dtype: Optional[str],
    variant: Optional[str],
    attributes: Mapping[str, object],
    hardware,
) -> LaunchConfig:
    """Select launches bound to the retained MACA 3.7.2 plugin artifact."""

    if variant == "vllm_plain_greedy_sample":
        return pytorch_small_id_narrow_launch(shape, dtype, attributes)
    if variant == "vllm_logits_fp32_cast":
        return pytorch_logits_cast_launch(shape, dtype, attributes)
    if variant == "vllm_logits_row_select":
        return pytorch_scalar_index_launch(shape, dtype, attributes)
    integer_cast = (variant == "vllm_embedding_id_widen" or (
        variant == "elementwise" and "operation" in attributes
        and attributes["operation"] == "cast_int32_int64"))
    if integer_cast:
        require_dtype(dtype, ("int32",))
        if variant == "vllm_embedding_id_widen":
            rows, elements = require_shape(shape, ("rows", "element_count"))
            if rows != elements or attributes["embedding_source"] != "vllm_0_24_unquantized_tp1":
                raise ValueError("C500 ID widening requires its ordinary0.24 INT32 source")
        else:
            elements, = require_shape(shape, ("element_count",))
            if attributes["tensor_iterator_source"] != "pytorch_2_4_maca_3_7_2":
                raise ValueError("C500 integer cast requires its explicit Torch source")
        return pytorch_int32_int64_cast_launch(elements, attributes, variant)
    normalized_dtype = require_dtype(dtype, ("bf16",))
    policy = launch_policy(attributes)
    execution_variant = require_gpu_vector_variant(variant)
    if policy == "analytical_launch":
        return select_gpu_vector_launch(shape, execution_variant, attributes)
    if policy != "injected_launch":
        raise AssertionError(f"unhandled launch policy: {policy!r}")
    algorithm = execution_variant
    parameters = {}

    if execution_variant == "rmsnorm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        grid = (rows, 1, 1)
        block_size = min(hidden, 1024)
        parameters[
            SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER
        ] = 1.0
        # The vLLM-MetaX source reads the input once for the reduction and once
        # again for normalization.  This is a source-defined logical reread;
        # compulsory DRAM remains one input tensor.
        parameters["input_reread_count"] = 1.0
        # The fixed MACA 3.7.2 BF16 device code lowers variance / hidden to
        # one RCP per row before the source-level RSQ.  The semantic Vector
        # demand already accounts for RSQ, so publish only the extra RCP.
        parameters["lowered_sfu_operations"] = float(rows)
        parameters["norm_reduction_tree_flops"] = float(
            rows * (block_size - 1))
        parameters["norm_epsilon_add_flops"] = float(rows)
        # vLLM-MetaX 0.8.5 commit 11b156053ec0344b8637c621c49b25c45f0a2331
        # layernorm_kernels.cu (SHA256
        # 056986926c196168a2edd654ef79e07aac704404ef092fd7019a496d0ea0188a)
        # contains scalar BF16 load/convert and output conversion loops around
        # one CUB block reduction. Publish semantic work only; service time is
        # supplied by independent MACA 3.7.2 mechanism microbenchmarks.
        parameters[SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER] = float(
            6 * rows * hidden)
        parameters[
            SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
        ] = 1.0
        parameters[SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER] = float(
            ceil_div(block_size, hardware.parallel_width))
        if hidden == 128 or hidden % 256 == 0:
            parameters["source_bound_block_reduction"] = 1.0
            parameters[SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] = 1.0
    elif execution_variant == "add_rmsnorm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        grid = (rows, 1, 1)
        maximum_block_size = 1024 if rows < 256 else 256
        block_size = min(hidden, maximum_block_size)
        # The source writes the updated residual during its reduction pass and
        # reads that value again for normalization.
        parameters["input_reread_count"] = 1.0
        parameters["lowered_sfu_operations"] = float(rows)
        parameters["norm_reduction_tree_flops"] = float(
            rows * (block_size - 1))
        parameters["norm_epsilon_add_flops"] = float(rows)
        # The same pinned source uses two width-eight passes around one CUB
        # reduction. Only active vec8 producers contribute memory requests;
        # all physical threads still participate in the reduction.
        parameters[SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER] = float(
            3 * rows * hidden)
        active_threads = min(block_size, ceil_div(hidden, 8))
        parameters[SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER] = float(
            ceil_div(active_threads, hardware.parallel_width))
        parameters[
            SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
        ] = 16.0
        if hidden % 256 == 0:
            parameters["source_bound_block_reduction"] = 1.0
            parameters[SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] = 1.0
    elif execution_variant in ("activation", "silu"):
        rows, intermediate = require_shape(shape, ("tok", "intermediate"))
        if intermediate < 4096 or intermediate % 8:
            raise ValueError(
                "MetaX BF16 SwiGLU source-bound launch requires "
                "intermediate >= 4096 and divisible by 8")
        grid = (ceil_div(intermediate, 4096), rows, 1)
        block_size = 512
        parameters["warp_instructions_lower_bound"] = (
            _source_bound_silu_wave_instruction_lower_bound(
                rows, intermediate))
        parameters[SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER] = (
            _source_bound_silu_wave_slot2_instruction_lower_bound(
                rows, intermediate))
        # The pinned act_and_mul kernel assigns one packed BF16x8 vector to
        # each active thread. Its source program converts two packed inputs
        # to FP32 and converts the packed result back to BF16. Publish these
        # static semantics so the predictor can compose the CTA dependency
        # path from independently characterized hardware services.
        parameters[SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER] = 8.0
        parameters[SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER] = float(
            3 * rows * intermediate)
    elif execution_variant == "rotary":
        seq_len, query_heads, kv_heads, head_dim = require_shape(
            shape,
            ("seq_len", "num_query_heads", "num_kv_heads", "head_dim"))
        if head_dim % 2:
            raise ValueError("MetaX C500 RoPE requires an even head_dim")
        if query_heads % kv_heads:
            raise ValueError(
                "MetaX C500 RoPE requires num_query_heads to be divisible "
                "by num_kv_heads")
        grid = (seq_len, 1, 1)
        block_size = min(query_heads * head_dim // 2, 512)
        parameters["rotary_cache_path_bytes"] = float(
            seq_len * (query_heads + kv_heads) * head_dim * 2)
        # The retained C500 capture names this launch outright:
        #   vllm::rotary_embedding_kernel<c10::BFloat16, true>(
        #       long const*, BFloat16*, BFloat16*, BFloat16 const*,
        #       int, long, long, long, int, int, int)
        # Two template arguments and eleven runtime arguments: three strides,
        # so the explicit head stride is present, but no `rope_dim_offset`
        # and no `inverse`.  That is the ABI11 program.  vLLM 0.24.0's ABI13
        # takes a third `cache_t` template argument and thirteen runtime
        # arguments, so the capture is NOT running the 0.24.0 kernel, and the
        # vendored `upstream/vllm/v0.24.0_c500_sdist` tree does not describe
        # this dataset's program.  Argument counts survive demangling, so this
        # is decisive rather than suggestive.
        parameters[
            SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER] = 1.0
        parameters[SOURCE_QK_ROPE_LOOP_PARAMETER] = 1.0
        # The pinned BF16 source converts x, y, cos, and sin to FP32 for each
        # rotary pair, then converts both outputs back to BF16: six scalar
        # conversions per pair, or three per active output element.
        parameters[SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER] = float(
            3 * seq_len * (query_heads + kv_heads) * head_dim)
        read_contract = _source_bound_rope_read_contract(
            hardware, query_heads, head_dim, block_size)
        if read_contract is not None:
            parameters[
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
            ] = read_contract[0]
            parameters[
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
            ] = read_contract[1]
    elif execution_variant == "rope":
        (seq_len, query_heads, kv_heads, head_dim,
         rotary_dim) = require_shape(
             shape,
             ("seq_len", "num_query_heads", "num_kv_heads", "head_dim",
              "rotary_dim"))
        if rotary_dim % 2 or rotary_dim > head_dim:
            raise ValueError(
                "MetaX C500 typed RoPE requires an even rotary_dim no "
                "larger than head_dim")
        if query_heads % kv_heads:
            raise ValueError(
                "MetaX C500 RoPE requires num_query_heads to be divisible "
                "by num_kv_heads")
        if not require_boolean(attributes, "is_neox"):
            raise NotImplementedError(
                "MetaX C500 production RoPE is binary-bound only for the "
                "NeoX template")
        if not require_boolean(attributes, "key_present"):
            raise NotImplementedError(
                "MetaX C500 production RoPE requires the source-bound Q+K "
                "path")
        grid = (seq_len, 1, 1)
        block_size = min(query_heads * rotary_dim // 2, 512)
        parameters["rotary_cache_path_bytes"] = float(
            seq_len * (query_heads + kv_heads) * rotary_dim * 2)
        # Canonical Dataset observations bind the vLLM-MetaX 0.8.5 ABI10
        # program at commit 11b15605.  It has the same source-ordered Q and K
        # loop ownership as the versioned ABI11 Serving program above, while
        # retaining a distinct program identity.  The predictor derives the
        # shared loop-layer demand and HWSpec supplies independently measured
        # integer dependency services.
        parameters[
            SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER] = 1.0
        parameters[SOURCE_QK_ROPE_LOOP_PARAMETER] = 1.0
        parameters[SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER] = float(
            3 * seq_len * (query_heads + kv_heads) * rotary_dim)
        read_contract = _source_bound_rope_read_contract(
            hardware, query_heads, rotary_dim, block_size)
        if read_contract is not None:
            parameters[
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
            ] = read_contract[0]
            parameters[
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
            ] = read_contract[1]
    elif execution_variant in ("kv_cache", "kvcache"):
        require_paged_kv_cache_contract(attributes)
        seq_len, kv_heads, head_dim = require_shape(
            shape, ("seq_len", "num_kv_heads", "head_dim"))
        elements_per_token = kv_heads * head_dim
        grid = (seq_len, 1, 1)
        block_size = min(elements_per_token, 512)
        source_bound_kv_cache = (
            normalized_dtype == "bf16"
            and attributes["cache_head_layout"] == "nhd"
            and attributes["kv_cache_dtype"] == "auto"
            and attributes["vector_copy_base_alignment_bytes"] >= 16
            and elements_per_token % 8 == 0
            and elements_per_token // 8 <= block_size
            and (elements_per_token // 8) % hardware.parallel_width == 0
        )
        if source_bound_kv_cache:
            # Read-producer geometry of the historical flash-layout source.
            # The vLLM-MetaX 0.8.5 ABI10
            # reshape_and_cache_flash_kernel (csrc/cache_kernels.cu, SHA256
            # 04d4d6c399240279470c16f0a5d55aac3cdd86b040afbe52a20ffb086b99163c),
            # has the observed 15-parameter signature.  The vLLM 0.24
            # sdist has a sixteenth kv_scale_stride argument.  Public
            # mcoplib 0.4.9 source also has 15 arguments but uses a different
            # vectorized body; the installed g8b5bd8b build is not bound to
            # either source body by this ABI.  The old body is one
            # block-strided scalar loop
            #
            #     for (int i = threadIdx.x; i < n; i += blockDim.x) {
            #       scalar_t tgt_key   = key[src_key_idx];
            #       scalar_t tgt_value = value[src_value_idx];
            #       ...
            #     }
            #
            # so every physical thread of the block is a read producer, and
            # each iteration issues the key and the value load back to back.
            # Neither load's address depends on the other's result, so both
            # are outstanding together; charging only one tensor's requests
            # halved the memory-level parallelism this path actually reaches.
            # The serial depth of the loop itself is a separate fact and stays
            # with the dependency path that prices it.
            #
            # Publish only that physical geometry and the load/address/store
            # dependency marker; hardware services price the path
            # independently of this production workload.
            active_read_threads = min(elements_per_token, block_size)
            # The admission gate above restricts this path to BF16.
            dtype_bytes = {"bf16": 2}[normalized_dtype]
            warp_read_bytes = hardware.parallel_width * dtype_bytes
            if warp_read_bytes % hardware.l2_transaction_bytes:
                raise ValueError(
                    "MetaX C500 source-bound KV copy requires one warp's "
                    "contiguous scalar reads to fill whole L2 transactions")
            parameters[SOURCE_VECTORIZED_KV_COPY_PARAMETER] = 1.0
            parameters[
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
            ] = float(
                _KV_CACHE_CONCURRENT_READ_TENSORS
                * (warp_read_bytes // hardware.l2_transaction_bytes))
            parameters[
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
            ] = float(ceil_div(
                active_read_threads, hardware.parallel_width))
    elif execution_variant == "elementwise":
        elements, = require_shape(shape, ("element_count",))
        operation = require_non_empty_string(attributes, "operation")
        if operation == "copy":
            if "input_materialization" not in attributes:
                raise NotImplementedError(
                    "MetaX C500 production elementwise launch is "
                    "source-bound only for add/mul or the grouped-strided "
                    "Norm input copy")
            materialization = require_non_empty_string(
                attributes, "input_materialization")
            if materialization == "concatenated_last_dim":
                grid = (ceil_div(
                    elements,
                    _CAT_ARRAY_THREADS * _CAT_ARRAY_VALUES_PER_THREAD,
                ), 1, 1)
                block_size = _CAT_ARRAY_THREADS
                algorithm = (
                    "pytorch_2_4_metax_3_7_2_cat_array_batched_copy")
            elif materialization != "grouped_strided_to_contiguous":
                raise NotImplementedError(
                    "MetaX C500 direct copy is source-bound only for the "
                    "grouped-strided Norm input and last-dimension concat "
                    "programs")
            else:
                grid, block_size, algorithm = _grouped_strided_copy_launch(
                    attributes, elements, parameters)
        elif operation not in ("add", "mul"):
            raise NotImplementedError(
                "MetaX C500 production elementwise launch is source-bound "
                "only for add/mul; copy/fill/read and other operations are "
                "unsupported")
        elif "operand_layout" in attributes:
            grid, block_size, algorithm = _maca_arity2_strided_launch(
                attributes, elements)
        else:
            grid = (ceil_div(elements, 1024), 1, 1)
            block_size = 256
    elif (execution_variant == "gather" and "materialization_source" in attributes
            and attributes["materialization_source"] == "pytorch_2_4_maca_cat_last_dim"):
        elements, rows, hidden = require_shape(shape, ("element_count", "rows", "hidden"))
        if (elements != rows * hidden or attributes["gather_factor"] != 1
                or attributes["index_count"] != 0 or attributes["index_bytes"] != 8
                or require_boolean(attributes, "scattered")
                or not require_boolean(attributes, "materialized_output")):
            raise ValueError("C500 source cat requires an unindexed BF16 materialization")
        if 2 * elements >= _INDEX_SELECT_INT32_LIMIT:
            raise NotImplementedError("C500 source cat exceeds the retained single-launch addressing domain")
        grid = (ceil_div(elements, _CAT_ARRAY_THREADS * _CAT_ARRAY_VALUES_PER_THREAD), 1, 1)
        block_size = _CAT_ARRAY_THREADS
        algorithm = "pytorch_2_4_metax_3_7_2_cat_array_batched_copy"
    elif execution_variant == "gather":
        # The pinned torch index_select dispatcher keeps the 16-index
        # small/large boundary, 128-thread geometry, and 32/64-block-per-CU
        # caps. Canonical MetaX alignment launches corroborate those facts.
        elements, rows, hidden = require_shape(
            shape, ("element_count", "rows", "hidden"))
        if elements != rows * hidden:
            raise ValueError(
                "MetaX embedding gather element_count must equal rows * "
                "hidden")
        if hidden <= 0:
            raise ValueError("MetaX embedding gather hidden must be positive")
        if "source_rows" in shape:
            source_rows, = require_shape(shape, ("source_rows",))
            # The same-version host dispatcher checks all three tensors with
            # canUse32BitIndexMath before selecting the unsigned-index launch.
            # Bounding contiguous byte extents is sufficient (and stricter
            # than its element-offset check).  Observed row counts do not
            # determine that source branch.  See the index-select reference.
            if max(2 * source_rows * hidden, 2 * elements, 8 * rows) >= _INDEX_SELECT_INT32_LIMIT:
                raise NotImplementedError(
                    "MetaX C500 index_select requires its proven contiguous "
                    "32-bit addressing domain")
        elif rows > _INDEX_SELECT_OBSERVED_ROWS:
            raise NotImplementedError(
                "MetaX C500 gather requires explicit contiguous source_rows "
                f"outside the observed row domain: rows={rows}, hidden={hidden}, "
                f"index_count={attributes['index_count']}, scattered={attributes['scattered']}")
        gather_factor = attributes["gather_factor"]
        index_count = attributes["index_count"]
        index_bytes = attributes["index_bytes"]
        scattered = require_boolean(attributes, "scattered")
        for name, value in (
                ("gather_factor", gather_factor),
                ("index_count", index_count),
                ("index_bytes", index_bytes)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"MetaX embedding gather {name} must be an integer")
        if (gather_factor != 1 or index_count != rows or index_bytes != 8
                or not scattered):
            raise NotImplementedError(
                "MetaX C500 injected gather is source-bound only for the "
                "BF16 token-embedding index_select contract")
        compute_units = int(hardware.num_compute_units)
        if compute_units <= 0:
            raise ValueError(
                "MetaX C500 gather requires positive num_compute_units")
        if rows <= _INDEX_SELECT_SMALL_ROW_LIMIT:
            block_size = min(hidden, _INDEX_SELECT_THREADS)
            grid = (
                min(
                    ceil_div(hidden, _INDEX_SELECT_THREADS),
                    compute_units
                    * _INDEX_SELECT_SMALL_BLOCKS_PER_COMPUTE_UNIT,
                ),
                1,
                1,
            )
            algorithm = "aten_index_select_small_index"
        else:
            block_size = min(elements, _INDEX_SELECT_THREADS)
            grid = (
                min(
                    ceil_div(elements, _INDEX_SELECT_THREADS),
                    compute_units
                    * _INDEX_SELECT_LARGE_BLOCKS_PER_COMPUTE_UNIT,
                ),
                1,
                1,
            )
            algorithm = "aten_index_select_large_index"
        registers_per_thread = _INDEX_SELECT_REGISTERS_PER_THREAD
    elif execution_variant == "moe_fill":
        tokens, experts, topk = require_shape(
            shape, ("tokens", "experts", "topk"))
        fill_role = require_non_empty_string(attributes, "fill_role")
        block_m = 16 if tokens <= experts else 64
        max_padded = tokens * topk + experts * (block_m - 1)
        if fill_role == "sorted_token_ids":
            elements = max_padded
        elif fill_role == "expert_ids":
            elements = ceil_div(max_padded, block_m)
        elif fill_role == "num_tokens_post_padded":
            elements = 1
        else:
            raise ValueError(
                "MetaX C500 moe_fill fill_role must be one of "
                "['expert_ids', 'num_tokens_post_padded', "
                f"'sorted_token_ids'], got {fill_role!r}")
        if elements > _FILL_MAX_ELEMENTS:
            raise NotImplementedError(
                "MetaX C500 int32 FillFunctor requires at most 2**31 - 1 "
                "elements")

        # Fixed torch 2.4.0+metax3.7.2.0 CUDALoops.cuh SHA256
        # 11a161b3c899e7e76baa3f53b32226a3fd797ce5cfbfaddc6eb34e4addd2e448
        # selects a 1024-thread nullary kernel at N >= 64K.  Its int32 vector
        # width starts at four and is reduced until it divides N.  Smaller
        # fills use thread_constants.h's 256 threads and four items/thread
        # (SHA256 4e0e049710b1b94523637d03fa634a34549e9f794236d55d462122865a540b23).
        if elements >= _FILL_OPT_MIN_ELEMENTS:
            vector_width = 4
            while elements % vector_width:
                vector_width //= 2
            block_size = _FILL_OPT_THREADS
            grid = (ceil_div(elements, block_size * vector_width), 1, 1)
            algorithm = "pytorch_2_4_metax_int32_fill_nullary_opt"
        else:
            block_size = _FILL_STANDARD_THREADS
            grid = (
                ceil_div(
                    elements,
                    block_size * _FILL_STANDARD_ELEMENTS_PER_THREAD,
                ),
                1,
                1,
            )
            algorithm = "pytorch_2_4_metax_int32_fill"
    elif execution_variant == "moe_topk":
        tokens, experts = require_shape(shape, ("tokens", "experts"))
        # vLLM 0.24.0 csrc/libtorch_stable/moe/topk_softmax_kernels.cu,
        # topkGatingLauncherHelper.  The constant chain is compile-time and
        # depends only on the expert count, the gating element width and the
        # wavefront width, so it is a static launch fact:
        #
        #   BYTES_PER_LDG   = min(16, sizeof(InputType) * EXPERTS)
        #   ELTS_PER_LDG    = BYTES_PER_LDG / sizeof(InputType)
        #   VECs_PER_THREAD = max(1, EXPERTS / (ELTS_PER_LDG * WARP))
        #   VPT             = VECs_PER_THREAD * ELTS_PER_LDG
        #   THREADS_PER_ROW = EXPERTS / VPT
        #   ROWS_PER_WARP   = WARP / THREADS_PER_ROW
        #
        # and the launch is
        #   grid  = ceil(ceil(rows / ROWS_PER_WARP) / WARPS_PER_TB)
        #   block = dim3(WARP, WARPS_PER_TB)
        # with WARPS_PER_TB fixed at four by the launcher.
        #
        # WARP is the compiled template parameter, not the hardware
        # wavefront.  The retained decode step names the instantiation
        # outright -- `vllm::moe::topkGatingSoftmax<4, 128, 4, 16, 32, int>`,
        # i.e. VPT 4, 128 experts, four warps per block, 16 bytes per load
        # and WARP_SIZE_PARAM 32 -- so the MACA build compiles the 32-lane
        # CUDA branch even though the C500 wavefront is 64 wide.  Taking
        # parallel_width here would halve the grid and double the block.
        width = _TOPK_SOFTMAX_COMPILED_WARP_WIDTH
        bytes_per_load = min(
            _TOPK_SOFTMAX_MAX_BYTES_PER_LDG,
            _TOPK_SOFTMAX_GATING_ELEMENT_BYTES * experts)
        elements_per_load = (
            bytes_per_load // _TOPK_SOFTMAX_GATING_ELEMENT_BYTES)
        vectors_per_thread = max(1, experts // (elements_per_load * width))
        values_per_thread = vectors_per_thread * elements_per_load
        if values_per_thread <= 0 or experts % values_per_thread:
            raise NotImplementedError(
                "MetaX C500 specialized topk softmax requires an expert "
                "count divisible by its per-thread vector width")
        threads_per_row = experts // values_per_thread
        if threads_per_row <= 0 or width % threads_per_row:
            raise NotImplementedError(
                "MetaX C500 specialized topk softmax requires the threads "
                "per row to divide the wavefront")
        rows_per_warp = width // threads_per_row
        return LaunchConfig(
            algorithm="vllm_0_24_metax_topk_gating_softmax",
            grid=(ceil_div(
                ceil_div(tokens, rows_per_warp),
                _TOPK_SOFTMAX_WARPS_PER_BLOCK), 1, 1),
            block=(width, _TOPK_SOFTMAX_WARPS_PER_BLOCK, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant == "moe_routing_weight_norm":
        tokens, topk = require_shape(shape, ("tokens", "topk"))
        # The captured `topkGatingSoftmax<4,128,4,16,32,int>` takes nine
        # runtime arguments and has neither a `renormalize` flag nor a bias
        # pointer, so that build does not fuse renormalisation: vLLM divides
        # the routed weights by their own row sum, and this is the sum.
        #
        # torch reduces `[tokens, topk]` FP32 over its contiguous last
        # dimension, so ATen's ReduceConfig constructor fixes the launch:
        #
        #   block_width  = min(last_pow2(topk), warp)
        #   block_height = min(last_pow2(tokens), max_threads / block_width)
        #   block_width  = min(last_pow2(topk), max_threads / block_height)
        #   grid         = ceil(tokens / block_height)
        #
        # The canonical raw device-launch projection records grid (1,1,1)
        # block (8,1,1) at one token and grid (2,1,1) block (8,16,1) at
        # twenty-two, which is what this evaluates to at both.
        width = min(
            _last_power_of_two(topk), hardware.parallel_width)
        height = min(
            _last_power_of_two(tokens), _REDUCE_MAX_THREADS // width)
        width = min(
            _last_power_of_two(topk), _REDUCE_MAX_THREADS // height)
        values_per_thread = ceil_div(topk, width)
        if values_per_thread >= height * 16 or values_per_thread >= 256:
            raise NotImplementedError(
                "MetaX C500 routing-weight normalisation is observed only in "
                "the output-split reduction regime")
        return LaunchConfig(
            algorithm="torch_metax_routing_weight_row_sum",
            grid=(ceil_div(tokens, height), 1, 1),
            block=(width, height, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant == "moe_routing_weight_scale":
        tokens, topk = require_shape(shape, ("tokens", "topk"))
        # torch's TensorIterator over the [tokens, topk] FP32 weights, one
        # element per thread.  The canonical raw device-launch projection
        # records block (128,1,1) throughout, with grid 1, 2 and 3 at the
        # element counts ceil(tokens*topk / 128) gives.
        grid = (ceil_div(tokens * topk, _ROUTING_SCALE_THREADS), 1, 1)
        block_size = _ROUTING_SCALE_THREADS
        algorithm = "torch_metax_routing_weight_scale"
    elif execution_variant == "moe_routed_id_extent":
        tokens, topk = require_shape(shape, ("tokens", "topk"))
        # A full reduction of the routed id tensor to one scalar, which
        # ReduceConfig launches as a single block whose width follows the
        # reduced extent:  block_width = min(last_pow2(tokens*topk), warp),
        # block_height 1, grid 1 because there is exactly one output.
        # The canonical raw device-launch projection records block widths
        # 8, 16, 32 and 64 -- which is this expression at one, two, four and
        # twenty-two tokens -- always at grid (1,1,1).
        routed = tokens * topk
        width = min(_last_power_of_two(routed), hardware.parallel_width)
        return LaunchConfig(
            algorithm="torch_metax_routed_id_extent_reduce",
            grid=(1, 1, 1),
            block=(width, 1, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant == "moe_expert_map_lookup":
        tokens, experts, topk = require_shape(
            shape, ("tokens", "experts", "topk"))
        # One gathered int32 per padded block of the alignment output, which
        # is the same `expert_ids` extent the moe_fill roles already derive.
        # torch runs the gather as a TensorIterator at 128 threads; the
        # canonical raw device-launch projection records block (128,1,1).
        block_m = 16 if tokens <= experts else 64
        padded = tokens * topk + experts * (block_m - 1)
        grid = (ceil_div(
            ceil_div(padded, block_m), _EXPERT_MAP_LOOKUP_THREADS), 1, 1)
        block_size = _EXPERT_MAP_LOOKUP_THREADS
        algorithm = "torch_metax_expert_map_lookup"
    elif execution_variant == "moe_router_logit_cast":
        tokens, experts = require_shape(shape, ("tokens", "experts"))
        # torch's vectorised copy-cast: 256 threads, four elements each.
        # The canonical raw device-launch projection records block (256,1,1)
        # with grids 3, 4 and 5, which is this expression at the logit counts
        # of twenty-two, thirty-two and forty tokens.
        grid = (ceil_div(
            tokens * experts,
            _LOGIT_CAST_THREADS * _LOGIT_CAST_VALUES_PER_THREAD), 1, 1)
        block_size = _LOGIT_CAST_THREADS
        algorithm = "torch_metax_router_logit_widen"
    elif execution_variant == "moe_routed_id_widen":
        tokens, topk = require_shape(shape, ("tokens", "topk"))
        # torch's unrolled copy-cast over the routed ids, 256 threads and
        # four elements each.  The canonical raw device-launch projection
        # records block (256,1,1) at grid (1,1,1) across this cell's batch
        # range, which is what this expression gives below 1024 ids.
        grid = (ceil_div(
            tokens * topk,
            _LOGIT_CAST_THREADS * _LOGIT_CAST_VALUES_PER_THREAD), 1, 1)
        block_size = _LOGIT_CAST_THREADS
        algorithm = "torch_metax_routed_id_widen"
    elif execution_variant == "moe_count_and_sort":
        _, experts = require_shape(shape, ("tokens", "experts"))
        # `at::cuda::kernelHistogram1D`, the per-expert routed-token count the
        # deployment runs immediately before the alignment kernel.  Its source
        # is inside the closed MetaX extension, so the geometry comes from the
        # canonical dataset's own raw device-launch projection, which records
        # grid (1,1,1) block (512,1,1) with 1032 bytes of shared memory: one
        # cooperative block over the expert histogram, independent of tokens.
        if experts != _MOE_ALIGN_OPT_SPECIALIZED_EXPERTS:
            raise NotImplementedError(
                "MetaX C500 routed-token histogram is observed only for the "
                f"{_MOE_ALIGN_OPT_SPECIALIZED_EXPERTS}-expert deployment")
        grid = (1, 1, 1)
        block_size = _MOE_HISTOGRAM_THREADS
        algorithm = "torch_metax_kernel_histogram1d_routed_token_count"
    elif execution_variant == "moe_align_opt_128":
        _, experts = require_shape(shape, ("tokens", "experts"))
        # `vllm_metax` ships no CUDA source, so this kernel's geometry cannot
        # be read.  It is taken from launch observation instead, which is what
        # the contract admits for exactly this case: the retained decode step
        # names the instantiation
        # `moe_align_block_size_opt_128_expert_kernel<int, unsigned short,
        # 128, 512>`, whose last two template arguments are the specialized
        # expert count and the thread count, and an mcTracer capture of the
        # deployed block at the production routing geometry records grid
        # (1,1,1) block (512,1,1).  One cooperative block owns the whole
        # expert-offset scan, so the grid does not move with tokens.
        if experts != _MOE_ALIGN_OPT_SPECIALIZED_EXPERTS:
            raise NotImplementedError(
                "MetaX C500 moe_align_opt_128 is observed only for the "
                f"{_MOE_ALIGN_OPT_SPECIALIZED_EXPERTS}-expert specialization")
        grid = (1, 1, 1)
        block_size = _MOE_ALIGN_OPT_THREADS
        algorithm = "vllm_metax_0_24_moe_align_block_size_opt_128_expert"
    elif execution_variant == "moe_act_and_mul":
        tokens, _, topk, _, intermediate = require_shape(
            shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
        # vLLM-metax 0.24.0 csrc/libtorch_stable/activation_kernels.cu
        # LAUNCH_ACTIVATION_GATE_KERNEL: one block per routed row, and one
        # thread per vector of ARCH_MAX_VEC_SIZE bytes of the output half.
        # C500 is not the CUDA>=12.9 / cc>=10 path, so ARCH_MAX_VEC_SIZE is
        # VecTraits<false>::ARCH_MAX_VEC_SIZE = 16 bytes; the vector path is
        # taken only when the output half divides by that vector width.
        vector_elements = (
            _ACTIVATION_GATE_VECTOR_BYTES // _BF16_ELEMENT_BYTES)
        if intermediate % vector_elements:
            threads = min(intermediate, _ACTIVATION_GATE_MAX_THREADS)
        else:
            threads = min(
                intermediate // vector_elements,
                _ACTIVATION_GATE_MAX_THREADS)
        grid = (tokens * topk, 1, 1)
        block_size = threads
        algorithm = "vllm_0_24_metax_bf16_act_and_mul"
        parameters[SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER] = 1.0
        # Both branches of the loop body read the gate and the up operand at
        # the same loop index, from two halves of one row:
        #
        #     ld128(x, &x_vec[i]);   /  x = VLLM_LDG(&x_ptr[idx]);
        #     ld128(y, &y_vec[i]);   /  y = VLLM_LDG(&y_ptr[idx]);
        #
        # Neither address depends on the other's value, so a wavefront holds
        # both in flight.  Each is one contiguous run over the wavefront.
        load_bytes = (
            _ACTIVATION_GATE_VECTOR_BYTES
            if intermediate % vector_elements == 0 else _BF16_ELEMENT_BYTES)
        wavefront_bytes = hardware.parallel_width * load_bytes
        active_threads = min(block_size, hardware.parallel_width)
        if wavefront_bytes % hardware.l2_transaction_bytes == 0:
            parameters[
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
            ] = float(
                _ACT_AND_MUL_INDEPENDENT_OPERAND_LOADS
                * (wavefront_bytes // hardware.l2_transaction_bytes))
            parameters[
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
            ] = float(ceil_div(block_size, hardware.parallel_width))
        elif (active_threads * load_bytes
              % hardware.l2_transaction_bytes == 0):
            # Fewer active lanes than one wavefront: the single partial
            # wavefront still issues one contiguous run per operand.
            parameters[
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
            ] = float(
                _ACT_AND_MUL_INDEPENDENT_OPERAND_LOADS
                * (active_threads * load_bytes
                   // hardware.l2_transaction_bytes))
            parameters[
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER] = 1.0
    elif execution_variant in ("moe_silu", "moe_mul"):
        tokens, _, topk, _, intermediate = require_shape(
            shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
        if intermediate % _MOE_ELEMENTWISE_VALUES_PER_THREAD:
            raise NotImplementedError(
                "MetaX C500 BF16 MoE elementwise source path requires "
                "intermediate divisible by 8")
        elements = tokens * topk * intermediate
        work_fraction = attributes["work_fraction"]
        if (isinstance(work_fraction, bool)
                or not isinstance(work_fraction, (int, float))):
            raise TypeError(
                f"MetaX C500 {execution_variant} work_fraction must be "
                "numeric")
        expected_work_fraction = (
            0.5
            if elements > _MOE_ELEMENTWISE_TENSOR_ITERATOR_INDEX_LIMIT
            else 1.0)
        if float(work_fraction) != expected_work_fraction:
            raise ValueError(
                f"MetaX C500 {execution_variant} work_fraction must match "
                "the fixed TensorIterator partition: expected "
                f"{expected_work_fraction}, got {work_fraction!r}")
        invocation_elements = int(elements * expected_work_fraction)
        block_size = _MOE_ELEMENTWISE_THREADS
        grid = (ceil_div(
            invocation_elements,
            _MOE_ELEMENTWISE_THREADS * _MOE_ELEMENTWISE_VALUES_PER_THREAD,
        ), 1, 1)
        # Fixed torch 2.4.0+metax3.7.2.0 headers select 128 threads and eight
        # BF16 values/thread: CUDALoops_maca_arity1.cuh SHA256
        # 712048d7f795eacea3994f72805674d5fd9870becb55415ae246987c19da040a
        # for SiLU and CUDALoops_maca_arity2.cuh SHA256
        # 504c666c7179f18d7874a63147fbd400aeb7d5deff542b6b3b7cd8cbc027d09a
        # for multiply.
        if execution_variant == "moe_silu":
            algorithm = "pytorch_2_4_metax_bf16_silu_elementwise"
        elif execution_variant == "moe_mul":
            algorithm = "pytorch_2_4_metax_bf16_mul_elementwise"
        else:
            raise AssertionError(
                "unhandled MetaX MoE elementwise variant: "
                f"{execution_variant!r}")
    elif execution_variant == "moe_sum":
        tokens, _, topk, hidden, _ = require_shape(
            shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
        if hidden % _MOE_SUM_OUTPUT_VECTOR_WIDTH:
            raise NotImplementedError(
                "MetaX C500 BF16 moe_sum source path requires hidden "
                "divisible by 4")
        work_fraction = attributes["work_fraction"]
        if (isinstance(work_fraction, bool)
                or not isinstance(work_fraction, (int, float))):
            raise TypeError("MetaX C500 moe_sum work_fraction must be numeric")
        input_elements = tokens * topk * hidden
        expected_work_fraction = (
            0.5
            if input_elements > _MOE_SUM_TENSOR_ITERATOR_INDEX_LIMIT
            else 1.0
        )
        if float(work_fraction) != expected_work_fraction:
            raise ValueError(
                "MetaX C500 moe_sum work_fraction must match the fixed "
                "TensorIterator partition: expected "
                f"{expected_work_fraction}, got {work_fraction!r}")

        invocation_output_elements = tokens * hidden
        if expected_work_fraction == 0.5:
            if invocation_output_elements % 2:
                raise NotImplementedError(
                    "MetaX C500 moe_sum split requires an even output "
                    "element count")
            invocation_output_elements //= 2

        # For TOK=1 TensorIterator removes the singleton token dimension.  In
        # the remaining 2-D iterator, setReduceConfigMaca maps power-of-two
        # TOPK lanes (capped at 16) to reduction threads.  For TOK>1 the 3-D
        # iterator maps all 512 threads to output elements.  Both paths use
        # four BF16 outputs/thread.  The fixed sources are Reduce.cuh SHA256
        # e257dee1ce74c709cbfe9f132fc0b70a2db169d1a99f9bf091e0261b559913f4,
        # reduce_utils.cuh SHA256
        # 8ef5b116fd10511e18dc751bc2ced55076f85495fb4dcad8b245f77472fdc4b5,
        # and reduce_imcontinuous_kernels.cuh SHA256
        # 4413e4f33e09e3793999731f724d2cfd8162d4b19043d0e66bca4d73966ae553.
        if tokens == 1:
            topk_reduction_threads = min(
                1 << (topk.bit_length() - 1), 16)
            output_threads = _MOE_SUM_THREADS // topk_reduction_threads
        else:
            output_threads = _MOE_SUM_THREADS
        block_size = _MOE_SUM_THREADS
        grid = (ceil_div(
            invocation_output_elements,
            _MOE_SUM_OUTPUT_VECTOR_WIDTH * output_threads,
        ), 1, 1)
        algorithm = "pytorch_2_4_metax_bf16_sum_imcontinuous_reduce"
    else:
        raise NotImplementedError(
            "MetaX C500 production Vector launch is unsupported for "
            f"{execution_variant!r}: no MACA 3.7.2 device source or complete "
            "versioned launch fact is retained")

    if normalized_dtype != "bf16":
        raise AssertionError(
            f"unhandled MetaX Vector dtype: {normalized_dtype!r}")

    return LaunchConfig(
        algorithm=algorithm,
        grid=grid,
        block=(block_size, 1, 1),
        parameters=parameters,
        execution_variant=execution_variant,
    )


def _grouped_strided_copy_launch(attributes, elements, parameters):
    """Launch facts for the pinned BF16 grouped-strided Norm input copy."""

    layout_names = (
        "input_outer_groups", "input_rows_per_group",
        "input_group_stride_elements", "input_row_stride_elements",
    )
    missing_layout = tuple(
        name for name in layout_names if name not in attributes)
    if missing_layout:
        raise ValueError(
            "MetaX C500 Norm input copy layout is incomplete: "
            f"missing={missing_layout!r}")
    layout = {}
    for name in layout_names:
        value = attributes[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"MetaX C500 Norm input copy {name} must be an "
                "integer")
        if value <= 0:
            raise ValueError(
                f"MetaX C500 Norm input copy {name} must be positive")
        layout[name] = value
    if (layout["input_outer_groups"]
            * layout["input_rows_per_group"]
            * layout["input_row_stride_elements"] != elements):
        raise ValueError(
            "MetaX C500 Norm input copy layout disagrees with "
            "element_count")
    if layout["input_group_stride_elements"] <= (
            layout["input_rows_per_group"]
            * layout["input_row_stride_elements"]):
        raise ValueError(
            "MetaX C500 Norm input copy requires an inter-group gap")
    grid = (ceil_div(
        elements,
        _GROUPED_STRIDED_COPY_THREADS
        * _GROUPED_STRIDED_COPY_VALUES_PER_THREAD,
    ), 1, 1)
    block_size = _GROUPED_STRIDED_COPY_THREADS
    algorithm = (
        "pytorch_2_4_metax_3_7_2_bf16_grouped_strided_copy")
    parameters[
        SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER
    ] = 1.0
    parameters[
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
    ] = 8.0
    parameters[
        SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
    ] = 1.0
    return grid, block_size, algorithm


def _maca_arity2_strided_launch(attributes, elements):
    """Launch facts for the pinned MACA arity-2 rotate-half multiplies.

    ``out[tokens, rows, inner] = strided_view * broadcast_row``.  The iterator
    orders dimensions fastest-first as (inner, rows, tokens); neither pair
    coalesces, because the strided operand steps by the full head width and the
    broadcast operand carries a zero stride, so ndim is three whenever there is
    more than one token and two otherwise.
    """

    layout = require_non_empty_string(attributes, "operand_layout")
    if layout != "dim0_contiguous_dim1_broadcast":
        raise NotImplementedError(
            "MetaX C500 strided elementwise launch is source-bound only for "
            f"the rotate-half broadcast program, got {layout!r}")
    dimensions = {}
    for name in ("iteration_tokens", "iteration_rows", "iteration_inner"):
        value = attributes[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"MetaX C500 arity-2 {name} must be an integer")
        if value <= 0:
            raise ValueError(f"MetaX C500 arity-2 {name} must be positive")
        dimensions[name] = value
    tokens = dimensions["iteration_tokens"]
    rows = dimensions["iteration_rows"]
    inner = dimensions["iteration_inner"]
    if tokens * rows * inner != elements:
        raise ValueError(
            "MetaX C500 arity-2 iteration shape disagrees with element_count")
    if tokens == 1:
        # launch_legacy_kernel_maca_2_2_broadcast: block_dim_x = 64 and
        # load_num = inner / 64.  getVectorizedAlignment bounds vec by
        # load_num, and getMaxGridSize(grid_x, rows) returns rows for every row
        # count below multiProcessorCount * 32, leaving y_t = 1.
        if inner % _MACA_ARITY2_2D_BLOCK:
            raise NotImplementedError(
                "MetaX C500 two-dimensional arity-2 launch requires an "
                "innermost extent that is a multiple of 64")
        load_num = inner // _MACA_ARITY2_2D_BLOCK
        return (
            (load_num, rows, 1),
            _MACA_ARITY2_2D_BLOCK,
            "pytorch_2_4_metax_3_7_2_elementwise_2_2_broadcast",
        )
    if inner % _CAT_ARRAY_THREADS and inner % 64:
        raise NotImplementedError(
            "MetaX C500 arity-2 rotate-half launch requires a warp-aligned "
            "innermost extent")
    if elements >= _MACA_ARITY2_DIM0_CONTIGUOUS_MIN_NUMEL:
        vec = 8 if inner >= _MACA_ARITY2_WIDE_INNER else 4
        return (
            (ceil_div(elements, _MACA_ARITY2_NT * vec), 1, 1),
            _MACA_ARITY2_NT,
            "pytorch_2_4_metax_3_7_2_elementwise_3_2_dim0_contiguous",
        )
    if rows % 4:
        raise NotImplementedError(
            "MetaX C500 arity-2 dim1-broadcast launch requires a row count "
            "that is a multiple of four")
    vec = 4
    return (
        (ceil_div(elements, _MACA_ARITY2_NT * vec * vec), 1, 1),
        _MACA_ARITY2_NT,
        "pytorch_2_4_metax_3_7_2_elementwise_3_2_dim0_contiguous_"
        "arg1_dim1_broadcast",
    )
