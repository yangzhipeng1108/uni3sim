"""NVIDIA A800 vector-kernel launch geometry."""

from typing import Mapping, Optional


def moe_count_sort_source_program():
    """Return the public CUDA atomic-rank sorting interface identity."""
    return "cuda_moe_count_sort_int32_unmasked"

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import (
    ceil_div, require_boolean, require_dtype, require_gpu_vector_variant,
    require_non_empty_string, require_paged_kv_cache_contract, require_shape,
    require_variant, select_gpu_vector_launch)
from src.kernel.model_contract import (
    SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
    SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
)
from src.kernel.launch_config.nvidia_a800.fused_moe_launch import (
    pytorch_2_8_moe_reduce_output_element_slices,
    uses_specialized_topk_softmax,
)


_SOURCE_BOUND_MOE_VARIANTS = {
    "moe_topk",
    "moe_align_small_batch",
    "moe_align",
    "moe_count_and_sort",
    "moe_act_and_mul",
    "moe_sum",
    "moe_reduce",
}

_A800_SM_COUNT = 108
_A800_KV_CACHE_REGISTERS_PER_THREAD = 54
_SOURCE_BOUND_RESIDENT_CTAS_PER_UNIT_PARAMETER = (
    "source_bound_resident_ctas_per_unit")


def _last_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("power-of-two input must be positive")
    return 1 << (value.bit_length() - 1)


def select_sparse_attn_indexer_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Project the source-defined DSA indexer launch onto A800."""

    del attributes, hardware
    require_dtype(dtype, ("int32",))
    execution_variant = require_variant(
        variant,
        (
            "dsa_indexer_prefill", "dsa_indexer_decode",
            "dsa_indexer_mixed", "dsa_index_reuse",
            "dsv4_indexer_prefill", "dsv4_indexer_decode",
        ),
        "A800 SPARSE_ATTN_INDEXER",
    )
    if execution_variant == "dsa_index_reuse":
        require_shape(shape, ("element_count",))
        return LaunchConfig(
            algorithm="dsa_index_alias",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant in (
            "dsa_indexer_prefill", "dsa_indexer_decode",
            "dsa_indexer_mixed"):
        batch, q_len, heads = require_shape(
            shape, ("batch", "q_len", "num_query_heads"))
        return LaunchConfig(
            algorithm="vllm_dsa_indexer_analytical",
            grid=(batch * q_len, heads, 1),
            block=(128, 1, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant in (
            "dsv4_indexer_prefill", "dsv4_indexer_decode"):
        batch, q_len, heads = require_shape(
            shape, ("batch", "q_len", "num_query_heads"))
        return LaunchConfig(
            algorithm="vllm_dsv4_indexer_analytical",
            grid=(batch * q_len, heads, 1),
            block=(128, 1, 1),
            execution_variant=execution_variant,
        )
    else:
        raise AssertionError(
            f"unhandled DSA indexer variant: {execution_variant!r}")


def select_sparse_attn_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Project the FlashMLA sparse-attention launch onto A800."""

    del attributes, hardware
    require_dtype(dtype, ("bf16",))
    execution_variant = require_variant(
        variant,
        ("dsa_sparse_prefill", "dsa_sparse_decode"),
        "A800 SPARSE_ATTN",
    )
    batch, q_len, heads = require_shape(
        shape, ("batch", "q_len", "num_query_heads"))
    return LaunchConfig(
        algorithm="flashmla_sparse_analytical",
        grid=(batch * q_len, heads, 1),
        block=(128, 1, 1),
        execution_variant=execution_variant,
    )


def select_dsv4_decode_split_counts(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> tuple[int, ...]:
    """A800 has no DeepSeek-V4 split-KV decode launch evidence."""

    del shape, dtype, variant, attributes, hardware
    raise NotImplementedError(
        "DSV4 split-KV decode split counts are only evidenced on SM90 "
        "(nvidia_h200); the A800 package owns no dsv4_ sparse variant")


def select_kv_cache_update_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select source-derived MLA and DSA cache-update geometry."""

    del hardware
    execution_variant = require_variant(
        variant,
        ("mla_kv_cache", "dsa_fp8_index_cache",
         "dsv4_compressed_state_cache", "dsv4_compressor_state_cache"),
        "A800 KV_CACHE",
    )
    if execution_variant == "mla_kv_cache":
        require_dtype(dtype, ("bf16",))
        seq_len, compressed_dim = require_shape(
            shape, ("seq_len", "compressed_dim"))
        return LaunchConfig(
            algorithm="vllm_mla_cache_append_analytical",
            grid=(seq_len, 1, 1),
            block=(min(compressed_dim, 512), 1, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant == "dsa_fp8_index_cache":
        require_dtype(dtype, ("uint8",))
        seq_len, key_dim, scale_bytes = require_shape(
            shape, ("seq_len", "key_dim", "scale_bytes_per_token"))
        scale_count = scale_bytes // 4
        if scale_count <= 0 or key_dim % scale_count:
            raise ValueError(
                "DSA FP8 index cache scale bytes do not define an integral "
                "quantization block")
        quant_block_size = key_dim // scale_count
        return LaunchConfig(
            algorithm="vllm_indexer_k_quant_and_cache",
            grid=(seq_len, ceil_div(key_dim, quant_block_size * 4), 1),
            block=(32, 4, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant == "dsv4_compressed_state_cache":
        # CUDA main-cache finish uses CuTeDSL; FP8 indexer uses explicitly
        # one-warp Triton. save_partial is a separate producer in both paths.
        stage = attributes["finish_stage"] if "finish_stage" in attributes else None
        require_dtype(dtype, ("fp32",) if stage == "compress" else ("uint8",))
        state_bytes, seq_len = require_shape(
            shape, ("state_bytes", "seq_len"))
        if state_bytes not in (584, 132):
            raise ValueError("DSV4 finish has an invalid packed record width")
        element_count = shape["element_count"]
        output_extent = 512 if stage == "compress" else state_bytes
        if (isinstance(element_count, bool)
                or not isinstance(element_count, int) or element_count < 0
                or element_count % output_extent):
            raise ValueError(
                "A800 DSV4 compressor state count must be a non-negative "
                "integral multiple of its typed output extent")
        ratio, = require_shape(shape, ("compress_ratio",))
        if ratio == 4 and stage is None and state_bytes in (584, 132):
            return LaunchConfig(
                algorithm=("vllm_dsv4_c4_norm_rope_store_cutedsl" if state_bytes == 584
                           else "vllm_dsv4_c4_indexer_norm_rope_store_triton"),
                grid=(seq_len, 1, 1), block=(64 if state_bytes == 584 else 32, 1, 1),
                execution_variant=execution_variant)
        if ratio == 128 and state_bytes == 584:
            if stage == "compress":
                return LaunchConfig("vllm_dsv4_c128_compress_block8_cutedsl",
                    (seq_len * 8, 1, 1), (256, 1, 1), execution_variant=execution_variant)
            if stage == "store":
                return LaunchConfig("vllm_dsv4_c128_norm_rope_store_cutedsl",
                    (seq_len, 1, 1), (256, 1, 1), execution_variant=execution_variant)
            if stage is None:
                # Standalone logical operator work only. Serving expands this
                # into both physical launches; never claim a single C128 kernel.
                return LaunchConfig("vllm_dsv4_c128_two_stage_aggregate_projection",
                    (seq_len * 8, 1, 1), (256, 1, 1), execution_variant=execution_variant,
                    parameters={"dsv4_finish_store_grid": float(seq_len),
                                "dsv4_finish_store_block": 256.0})
        raise NotImplementedError("DSV4 finish launch requires C4 main/indexer or C128 main storage")
    elif execution_variant == "dsv4_compressor_state_cache":
        # deepseek_v4/compressor.py _SAVE_PARTIAL_STATES_KERNEL: one
        # per-token FP32 kv/score state write, one CTA per token.
        require_dtype(dtype, ("fp32",))
        seq_len, state_bytes = require_shape(
            shape, ("seq_len", "state_bytes"))
        return LaunchConfig(
            algorithm="vllm_dsv4_save_partial_states_analytical",
            grid=(seq_len, 1, 1),
            block=(min(state_bytes // 4, 1024), 1, 1),
            execution_variant=execution_variant,
        )
    else:
        raise AssertionError(
            f"unhandled KV cache update variant: {execution_variant!r}")


def select_target_vector_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select source-derived GLM vector launch geometry."""

    del attributes, hardware
    execution_variant = require_variant(
        variant,
        ("layernorm", "mla_rope",
         "mhc_pre", "mhc_post_pre", "mhc_post", "hc_head",
         "dsv4_inv_rope_fp8_quant", "dsv32_embed_norm"),
        "A800 target VECTOR")
    if execution_variant == "dsv4_inv_rope_fp8_quant":
        # deepseek_v4 fused_inv_rope_fp8_quant: one warp-level pass per
        # token/head row (inverse RoPE on the rope_dim slice, then 1x128
        # e4m3 quantisation with FP32 scales) feeding the wo_a einsum.
        require_dtype(dtype, ("fp8",))
        seq_len, query_heads = require_shape(
            shape, ("seq_len", "num_query_heads"))
        return LaunchConfig(
            algorithm="dsv4_fused_inv_rope_fp8_quant_analytical",
            grid=(seq_len, query_heads, 1),
            block=(32, 1, 1),
            execution_variant=execution_variant,
        )
    require_dtype(dtype, ("bf16",))
    if execution_variant == "dsv32_embed_norm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        block = 1 << (hidden - 1).bit_length()
        # fused_embed_norm.py uses the same explicit shape rule on CUDA.
        warps = min(32, max(4, block // 512))
        return LaunchConfig("vllm_v32_fused_embed_norm_triton",
            (rows, 1, 1), (32 * warps, 1, 1), execution_variant=execution_variant)
    if execution_variant == "layernorm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        return LaunchConfig(
            algorithm="pytorch_layer_norm_analytical",
            grid=(rows, 1, 1),
            block=(min(hidden, 1024), 1, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant == "mla_rope":
        seq_len, query_heads, key_heads = require_shape(
            shape, ("seq_len", "num_query_heads", "num_key_heads"))
        return LaunchConfig(
            algorithm="vllm_mla_rope_triton_analytical",
            grid=(seq_len, query_heads + key_heads, 1),
            block=(32, 1, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant in (
            "mhc_pre", "mhc_post_pre", "mhc_post", "hc_head"):
        # TileLang MHC kernels launch one CTA per token row (kc_v4attn
        # launch evidence: grid (T, 1, 1), block up to 128).
        element_count, hidden, hc_mult = require_shape(
            shape, ("element_count", "hidden", "hc_mult"))
        row_elements = hc_mult * hidden
        if element_count % row_elements:
            raise ValueError(
                "DSV4 MHC element_count is not divisible by hc_mult*hidden")
        rows = element_count // row_elements
        return LaunchConfig(
            algorithm=f"dsv4_{execution_variant}_tilelang_analytical",
            grid=(rows, 1, 1),
            block=(128, 1, 1),
            execution_variant=execution_variant,
        )
    else:
        raise AssertionError(
            f"unhandled target VECTOR variant: {execution_variant!r}")


def _select_source_bound_moe_launch(
    shape: Mapping[str, int],
    execution_variant: str,
    dtype: str,
    attributes: Mapping[str, object],
    hardware,
) -> LaunchConfig:
    """Select source-defined vLLM/PyTorch CUDA MoE helper launches."""

    if execution_variant == "moe_topk":
        tokens, experts = require_shape(shape, ("tokens", "experts"))
        if not uses_specialized_topk_softmax(experts):
            raise NotImplementedError(
                "vLLM 0.11.0 non-specialized topk_softmax requires a "
                "separate softmax launch")
        bytes_per_load = min(16, 4 * experts)
        elements_per_load = bytes_per_load // 4
        vectors_per_thread = max(
            1, experts // (elements_per_load * 32))
        values_per_thread = vectors_per_thread * elements_per_load
        if experts % values_per_thread:
            raise ValueError(
                "vLLM specialized topk expert count is not divisible by "
                "its per-thread vector width")
        threads_per_row = experts // values_per_thread
        if threads_per_row <= 0 or 32 % threads_per_row:
            raise ValueError(
                "vLLM specialized topk expert count has invalid warp "
                "partitioning")
        rows_per_warp = 32 // threads_per_row
        grid = ceil_div(ceil_div(tokens, rows_per_warp), 4)
        return LaunchConfig(
            algorithm=execution_variant,
            grid=(grid, 1, 1),
            # vLLM 0.11.0 topk_softmax.cu launches four warps as a
            # two-dimensional CUDA block: one warp along x and four along y.
            block=(32, 4, 1),
            execution_variant=execution_variant,
        )
    elif execution_variant == "moe_align_small_batch":
        experts, = require_shape(shape, ("experts",))
        grid, block = 1, max(experts, 32)
    elif execution_variant == "moe_align":
        experts, = require_shape(shape, ("experts",))
        padded_experts = ceil_div(experts, 32) * 32
        if padded_experts >= 1024:
            raise ValueError(
                "vLLM moe_align requires padded experts below 1024")
        grid, block = 1, 1024
    elif execution_variant == "moe_count_and_sort":
        tokens, topk = require_shape(shape, ("tokens", "topk"))
        grid, block = min(ceil_div(tokens * topk, 256), 65535), 256
    elif execution_variant == "moe_act_and_mul":
        tokens, topk, intermediate = require_shape(
            shape, ("tokens", "topk", "intermediate"))
        # vLLM 0.11.0 csrc/activation_kernels.cu launches one block per
        # routed row and one thread per output element, capped at 1024.
        grid, block = tokens * topk, min(intermediate, 1024)
    elif execution_variant == "moe_sum":
        tokens, topk, hidden = require_shape(
            shape, ("tokens", "topk", "hidden"))
        if topk not in (2, 3, 4):
            raise NotImplementedError(
                "vLLM 0.11.0 moe_sum CUDA kernel supports top-k 2, 3 or 4")
        grid, block = tokens, min(hidden, 1024)
    elif execution_variant == "moe_reduce":
        tokens, topk, hidden = require_shape(
            shape, ("tokens", "topk", "hidden"))
        if topk in (2, 3, 4):
            raise ValueError(
                "PyTorch moe_reduce is not selected for vLLM's specialized "
                "top-k 2/3/4 CUDA kernels")
        if dtype not in ("bf16", "fp16"):
            raise NotImplementedError(
                "source-bound PyTorch MoE reduction supports BF16/FP16")
        if "work_fraction" not in attributes:
            raise ValueError(
                "source-bound PyTorch moe_reduce requires work_fraction")
        work_fraction = attributes["work_fraction"]
        if (isinstance(work_fraction, bool)
                or not isinstance(work_fraction, (int, float))):
            raise TypeError("PyTorch moe_reduce work_fraction must be numeric")
        work_fraction = float(work_fraction)
        slices = pytorch_2_8_moe_reduce_output_element_slices(shape)
        total_output_elements = tokens * hidden
        matching_slices = tuple(
            (offset, count) for offset, count in slices
            if count / total_output_elements == work_fraction
        )
        if not matching_slices:
            raise ValueError(
                "PyTorch moe_reduce work_fraction does not identify one "
                "source-derived output slice: "
                f"fraction={work_fraction}, slices={slices}")

        def source_geometry(output_offset: int, output_elements: int):
            # CUDA allocations are suitably aligned for the widest candidate;
            # sub-iterator offsets and extents may still reduce vec4 to vec2
            # or scalar, exactly as Reduce.cuh::get_output_vec_size does.
            output_vector_extent = (
                output_elements if topk == 1 else hidden)
            output_vector_size = 4
            for value in (output_offset, output_vector_extent):
                while value % output_vector_size:
                    output_vector_size //= 2
            if output_vector_size not in (1, 2, 4):
                raise AssertionError(
                    "PyTorch reduction selected an invalid output vector")
            output_vectors = output_elements // output_vector_size
            max_threads = 512 // output_vector_size
            dim_x = min(_last_power_of_two(output_vectors), max_threads)
            dim_y = min(_last_power_of_two(topk), max_threads)
            block_x = min(dim_x, 32)
            block_y = min(dim_y, max_threads // block_x)
            block_x = min(dim_x, max_threads // block_y)

            # For enough inputs per output, block.y participates in the
            # reduction.  Otherwise each y lane owns another output.  Only the
            # latter multiplies ReduceConfig.step_output and shrinks grid.x.
            split_threshold = min(block_y * 16, 256)
            block_y_reduces = topk >= split_threshold
            step_output = block_x
            values_per_thread = topk
            if block_y_reduces:
                values_per_thread = ceil_div(topk, block_y)
            else:
                step_output *= block_y
            grid_x = ceil_div(output_vectors, step_output)

            # PyTorch may further split very large reductions across grid.y,
            # which also creates a global accumulation-buffer data path.  That
            # path is outside the current Forward demand contract, so reject it
            # instead of publishing an incomplete launch/demand pair.
            if block_y_reduces and values_per_thread >= 256:
                blocks_per_sm = (
                    int(hardware.max_threads_per_unit)
                    // (block_x * block_y))
                target_grid_size = (
                    int(hardware.num_compute_units) * blocks_per_sm)
                if grid_x <= target_grid_size:
                    ctas_per_output = max(
                        min(
                            ceil_div(target_grid_size, grid_x),
                            ceil_div(values_per_thread, 16),
                        ),
                        ceil_div(values_per_thread, 256),
                    )
                    if ctas_per_output > 1:
                        raise NotImplementedError(
                            "evidence_pending: PyTorch moe_reduce global "
                            "accumulation-buffer traffic is not represented")

            return (
                (grid_x, 1, 1),
                (block_x, block_y, 1),
            )

        geometries = {
            source_geometry(output_offset, output_elements)
            for output_offset, output_elements in matching_slices
        }
        if len(geometries) != 1:
            raise NotImplementedError(
                "evidence_pending: work_fraction does not distinguish "
                "PyTorch output slices with different launch geometry")
        grid, block = next(iter(geometries))
        return LaunchConfig(
            algorithm="pytorch_2_8_sum_out",
            grid=grid,
            block=block,
            execution_variant=execution_variant,
        )
    else:
        raise ValueError(
            "unsupported vLLM native MoE vector variant: "
            f"{execution_variant!r}")
    return LaunchConfig(
        algorithm=execution_variant,
        grid=(grid, 1, 1),
        block=(block, 1, 1),
        execution_variant=execution_variant,
    )


def source_bound_add_rmsnorm_width8_issue_lower_bound(
        shape: Mapping[str, int], block_size: int) -> float:
    """Return the retained SM80 vec8 SASS issue lower bound.

    The exact vLLM 0.11 binary proves a 39-instruction input loop and a
    49-instruction output loop for every tail-free vec8 work-warp iteration.
    A retained CFG also proves a disjoint 22-instruction mandatory non-loop
    subset for the first iteration on each participating warp.  Shape alone
    cannot prove the three-pointer alignment and stride predicates selecting
    this specialization, so only an exact runtime-symbol observation may call
    this helper; the ordinary Serving selector deliberately does not.
    """

    rows, hidden = require_shape(shape, ("rows", "hidden"))
    if block_size not in (256, 1024):
        raise ValueError(
            "typed A800 Add-RMS vec8 issue requires block_size 256 or 1024")
    if hidden % 256 != 0:
        raise ValueError(
            "typed A800 Add-RMS vec8 issue requires tail-free H % 256 == 0")
    vector_iterations = hidden // 256
    physical_warps = block_size // 32
    return float(rows * (
        110 * min(vector_iterations, physical_warps)
        + 88 * max(vector_iterations - physical_warps, 0)
    ))


def source_bound_kv_cache_bf16_auto_nhd_issue(
        shape: Mapping[str, int], block_size: int) -> float:
    """Return exact warp issue for the retained aligned vec8 KV path.

    The fixed vLLM 0.11.0 SM80 binary executes 114 instructions on every
    physical warp.  Its two aligned K/V copy regions each execute 54
    instructions on every warp containing at least one vec8 pack.  This
    helper is valid only when the caller has proved BF16 input/cache, auto
    cache dtype, NHD layout, valid slots, 16-byte base alignment, a tail-free
    vec8 row, and at most one pack per launched thread.
    """

    seq_len, kv_heads, head_dim = require_shape(
        shape, ("seq_len", "num_kv_heads", "head_dim"))
    elements_per_token = kv_heads * head_dim
    if block_size != min(elements_per_token, 512):
        raise ValueError(
            "typed A800 KV-cache issue requires the source launch block")
    if elements_per_token % 8 != 0:
        raise ValueError(
            "typed A800 KV-cache issue requires a tail-free BF16 vec8 row")
    vector_packs = elements_per_token // 8
    if vector_packs > block_size:
        raise ValueError(
            "typed A800 KV-cache issue requires at most one pack per thread")
    physical_warps = ceil_div(block_size, 32)
    active_copy_warps = ceil_div(vector_packs, 32)
    return float(seq_len * (
        114 * physical_warps + 108 * active_copy_warps))


def _source_bound_kv_cache_resident_ctas(hardware, block_size: int) -> int:
    """Return SM80 KV-copy residency from the fixed binary resources."""

    warps_per_cta = ceil_div(block_size, hardware.parallel_width)
    resident_ctas = min(
        hardware.max_blocks_per_unit,
        hardware.max_threads_per_unit // block_size,
        hardware.max_warps_per_unit // warps_per_cta,
        hardware.max_regs_per_unit
        // (_A800_KV_CACHE_REGISTERS_PER_THREAD * block_size),
    )
    if resident_ctas <= 0:
        raise ValueError(
            "typed A800 KV-cache launch exceeds SM80 resident resources")
    return resident_ctas


def source_bound_pytorch_vec4_elementwise_issue(
        shape: Mapping[str, int], block_size: int, operation: str) -> float:
    """Return exact SM80 warp work for pinned BF16 vec4 add/mul.

    The pinned PyTorch 2.8.0 CUDA source maps this three-BF16-operand
    specialization to 128 threads and eight scalar elements per thread.  The
    ``vec_size=4`` policy therefore has *two* vector iterations, not eight.
    The fixed sm80 full-CTA body executes 50 warp instructions on each of its
    four physical warps, hence 200 per 1024-element CTA.  The scalarized
    512-element remainder executes 175 instructions on each of the same four
    warps.  This source count is also conserved by the non-duration Nsight
    executed-instruction counter across the version-bound full and half-tail
    launch programs.  Other remainders have a different predicated CFG and
    are deliberately rejected.
    """

    elements, = require_shape(shape, ("element_count",))
    if block_size != 128:
        raise ValueError(
            "typed A800 PyTorch vec4 issue requires block_size 128")
    if operation not in ("add", "mul"):
        raise ValueError(
            "typed A800 PyTorch vec4 issue requires add or mul")
    full_blocks, remainder = divmod(elements, 1024)
    if remainder == 0:
        tail_instructions = 0
    elif remainder == 512:
        tail_instructions = 700
    else:
        raise ValueError(
            "typed A800 PyTorch vec4 issue supports only remainder 0 or 512")
    return float(200 * full_blocks + tail_instructions)


def _source_bound_rmsnorm_issue_lower_bound(
        shape: Mapping[str, int], block_size: int) -> float:
    rows, hidden = require_shape(shape, ("rows", "hidden"))
    if block_size != min(hidden, 1024) or block_size != 1024:
        raise ValueError(
            "typed A800 RMSNorm issue requires the source launch block")
    return float(rows * (1499 + 34 * ceil_div(hidden, 32)))


def _source_bound_rmsnorm_gamma_l2_copies(
        rows: int, hidden: int, dtype: str, block_size: int,
        hardware) -> Optional[int]:
    """Return conservative shared-gamma fills for resident CTA groups."""

    model = hardware.rmsnorm_shared_gamma_l1_residency
    if model is None:
        return None
    gamma_bytes = hidden * 2
    if (hidden % 256 != 0
            or dtype != model["dtype"]
            or block_size != model["threads_per_cta"]
            or gamma_bytes > model["maximum_evidenced_gamma_bytes"]):
        return None
    if gamma_bytes <= model["launch_lifetime_maximum_gamma_bytes"]:
        return min(rows, hardware.num_compute_units)
    return _resident_cta_group_copies(
        rows, hardware.num_compute_units, model["resident_ctas_per_sm"])


def _resident_cta_group_copies(
        rows: int, num_compute_units: int, resident_ctas: int) -> int:
    group_ctas = num_compute_units * resident_ctas
    full_groups, tail_ctas = divmod(rows, group_ctas)
    # Do not assume persistence between resident groups.  Within each group,
    # every active SM fetches the common gamma once for up to two resident
    # CTAs, as fixed by the source resource signature.
    return (full_groups * num_compute_units
            + min(tail_ctas, num_compute_units))


def _source_bound_rope_issue(
        shape: Mapping[str, int], attributes: Mapping[str, object],
        block_size: int) -> float:
    seq_len, query_heads, kv_heads, head_dim, rotary_dim = require_shape(
        shape,
        ("seq_len", "num_query_heads", "num_kv_heads", "head_dim",
         "rotary_dim"))
    is_neox = require_boolean(attributes, "is_neox")
    key_present = require_boolean(attributes, "key_present")
    query_pairs = query_heads * rotary_dim // 2
    key_pairs = kv_heads * rotary_dim // 2 if key_present else 0
    if (block_size != min(query_pairs, 512)
            or block_size != 512
            or not is_neox
            or not key_present
            or rotary_dim != head_dim):
        raise ValueError(
            "typed A800 RoPE issue requires the exact NeoX source launch")
    return float(seq_len * (
        784
        + 69 * ceil_div(query_pairs, 32)
        + 85 * ceil_div(key_pairs, 32)))


def _source_bound_silu_issue_lower_bound(
        shape: Mapping[str, int], block_size: int) -> float:
    tokens, intermediate = require_shape(shape, ("tok", "intermediate"))
    if block_size != min(intermediate, 1024) or block_size != 1024:
        raise ValueError(
            "typed A800 SiLU issue requires the source launch block")
    return float(tokens * (480 + 47 * ceil_div(intermediate, 32)))


def source_bound_flash_attention_copy_issue(
        shape: Mapping[str, int], dtype: str,
        grid_size: int, block_size: int) -> float:
    """Return the pinned PyTorch SM80 direct-copy instruction program."""

    elements, = require_shape(shape, ("element_count",))
    if block_size != 128:
        raise ValueError(
            "A800 FLASH_ATTN direct-copy requires 128 threads per CTA")
    if dtype == "bf16":
        elements_per_thread = 4
        expected_grid = ceil_div(elements, block_size * elements_per_thread)
        if elements != expected_grid * block_size * elements_per_thread:
            raise ValueError(
                "A800 BF16 FLASH_ATTN direct-copy requires full CTAs")
        instructions = 836 * expected_grid
    elif dtype == "float32":
        elements_per_thread = 2
        cta_elements = block_size * elements_per_thread
        full_ctas, tail_elements = divmod(elements, cta_elements)
        expected_grid = full_ctas + int(tail_elements > 0)
        instructions = 364 * full_ctas
        if tail_elements:
            live_warps = ceil_div(tail_elements, 32)
            instructions += 44 + 40 * live_warps
            if live_warps == 2:
                instructions += 2
    else:
        raise ValueError(
            "A800 FLASH_ATTN direct-copy supports BF16 and FP32 templates")
    if grid_size != expected_grid:
        raise ValueError(
            "A800 FLASH_ATTN direct-copy grid disagrees with its source "
            "instruction program")
    return float(instructions)


def validate_source_bound_vector_launch(
        hardware_id: str, dtype: str, variant: str,
        shape: Mapping[str, int], attributes: Mapping[str, object],
        launch: LaunchConfig) -> None:
    """Validate the A800 source semantics and launch behind a task floor."""

    memory_parameter = SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
    tensor_iterator_parameter = (
        SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER)
    has_memory_task = memory_parameter in launch.parameters
    has_tensor_iterator_task = tensor_iterator_parameter in launch.parameters
    if not has_memory_task and not has_tensor_iterator_task:
        return
    if has_memory_task and has_tensor_iterator_task:
        raise ValueError(
            "source-bound Vector launch cannot select two task floors")
    is_flash_attention_copy = (
        variant == "elementwise"
        and launch.algorithm
        == "observed_flash_attention_copy:elementwise")
    is_norm_input_materialization_copy = (
        variant == "elementwise"
        and launch.algorithm
        == "observed_norm_input_materialization_copy:elementwise")
    if has_tensor_iterator_task:
        if (hardware_id not in ("nvidia_a800", "nvidia_h200")
                or not (is_flash_attention_copy
                        or is_norm_input_materialization_copy)
                or require_non_empty_string(attributes, "operation")
                != "copy"):
            raise ValueError(
                "source-bound TensorIterator task requires the observed "
                "NVIDIA A800 FlashAttention or Norm input-materialization "
                "copy")
        if (is_norm_input_materialization_copy
                and require_non_empty_string(
                    attributes, "input_materialization")
                != "grouped_strided_to_contiguous"):
            raise ValueError(
                "source-bound TensorIterator Norm input copy requires the "
                "grouped-strided input materialization contract")
        if launch.grid[1:] != (1, 1) or launch.block != (128, 1, 1):
            raise ValueError(
                "source-bound TensorIterator task requires a one-dimensional "
                "128-thread launch")
        elements, = require_shape(shape, ("element_count",))
        expected_rank_and_width = {
            "bf16": (4.0, 4),
            "float32": (3.0, 2),
        }
        if dtype not in expected_rank_and_width:
            raise ValueError(
                "source-bound TensorIterator task requires BF16 or FP32")
        expected_rank, values_per_thread = expected_rank_and_width[dtype]
        expected_grid = ceil_div(elements, 128 * values_per_thread)
        if (launch.grid[0] != expected_grid
                or launch.parameters[tensor_iterator_parameter]
                != expected_rank):
            raise ValueError(
                "source-bound TensorIterator task disagrees with its "
                "source rank or launch geometry")
        return
    if (hardware_id not in ("nvidia_a800", "nvidia_h200")
            or (dtype != "bf16" and not is_flash_attention_copy)):
        raise ValueError(
            "source-bound memory-dependency task requires NVIDIA A800 BF16")
    if launch.grid[1:] != (1, 1) or launch.block[1:] != (1, 1):
        raise ValueError(
            "source-bound memory-dependency task requires one-dimensional "
            "launch geometry")

    parameters = launch.parameters
    read_parameter = SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
    grid = launch.grid[0]
    block = launch.block[0]
    if variant == "rmsnorm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        expected_grid = rows
        expected_block = min(hidden, 1024)
        # The source program is shared with A800, but its common-gamma L2
        # fill count is a physical residency census.  H200 therefore uses
        # its 132 SMs rather than borrowing A800's 108-SM endpoint.
        source_compute_units = 132 if hardware_id == "nvidia_h200" else _A800_SM_COUNT
        tail_free_source_domain = hidden % 256 == 0
        expected_weight_l2_copies = None
        if (tail_free_source_domain and dtype == "bf16"
                and expected_block == 1024
                and hidden * 2 <= 32768):
            if hidden * 2 <= 4096:
                expected_weight_l2_copies = min(rows, source_compute_units)
            else:
                expected_weight_l2_copies = _resident_cta_group_copies(
                    rows, source_compute_units, 2)
        has_weight_l2_copies = "norm_weight_l2_copies" in parameters
        has_pipeline = (
            SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER in parameters)
        valid = (
            parameters["input_reread_count"] == 1.0
            and parameters["norm_reduction_tree_flops"]
            == float(rows * (expected_block - 1))
            and parameters["norm_epsilon_add_flops"] == float(rows)
            and SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
            not in parameters
            and has_pipeline == tail_free_source_domain
            and (
                not has_pipeline
                or parameters[
                    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] == 1.0
            )
            and (
                not tail_free_source_domain
                or parameters["source_bound_block_reduction"] == 1.0
            )
            and (
                tail_free_source_domain
                or (
                    "source_bound_block_reduction" not in parameters
                )
            )
            and has_weight_l2_copies
            == (expected_weight_l2_copies is not None)
            and (
                not has_weight_l2_copies
                or parameters["norm_weight_l2_copies"]
                == float(expected_weight_l2_copies)
            ))
    elif variant == "add_rmsnorm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        expected_grid = rows
        expected_block = min(hidden, 256 if rows >= 256 else 1024)
        has_pipeline = (
            SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER in parameters)
        source_program_domain = dtype == "bf16" and hidden % 256 == 0
        valid = (
            parameters["input_reread_count"] == 1.0
            and parameters["norm_reduction_tree_flops"]
            == float(rows * (expected_block - 1))
            and parameters["norm_epsilon_add_flops"] == float(rows)
            and has_pipeline == source_program_domain
            and (
                not has_pipeline
                or parameters[
                    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] == 1.0
            )
            and (
                hidden % 256 != 0
                or parameters[
                    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
                ] == float(ceil_div(
                    min(expected_block, ceil_div(hidden, 8)), 32))
            ))
    elif variant == "rope":
        seq_len, query_heads, _, _, _ = require_shape(
            shape,
            ("seq_len", "num_query_heads", "num_kv_heads", "head_dim",
             "rotary_dim"))
        expected_grid = seq_len
        expected_block = min(
            query_heads * shape["rotary_dim"] // 2, 512)
        valid = (
            require_boolean(attributes, "is_neox")
            and require_boolean(attributes, "key_present")
            and shape["rotary_dim"] == shape["head_dim"])
    elif variant == "silu":
        tokens, intermediate = require_shape(shape, ("tok", "intermediate"))
        expected_grid = tokens
        expected_block = min(intermediate, 1024)
        has_issue_lower_bound = "warp_instructions_lower_bound" in parameters
        has_segmented_pipeline = (
            SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER
            in parameters)
        # The independently characterized first/resident/queued services
        # cover the fixed 1024-thread source program at full-device grids
        # 108, 216, and 1024, with grid 256 held out for validation.  Once
        # every SM has an initial CTA, the launch grid contributes only the
        # number of resident/queued cohorts; it does not select a different
        # source program.
        physical_compute_units = (
            132 if hardware_id == "nvidia_h200" else _A800_SM_COUNT)
        segmented_pipeline_domain = (
            tokens >= physical_compute_units and expected_block == 1024)
        valid = (
            has_issue_lower_bound
            and has_segmented_pipeline == segmented_pipeline_domain
            and (
                not has_segmented_pipeline
                or parameters[
                    SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER
                ] == 1.0
            )
            and (
                not has_issue_lower_bound
                or parameters["warp_instructions_lower_bound"]
                == _source_bound_silu_issue_lower_bound(
                    shape, expected_block)
            ))
    elif variant == "elementwise_pytorch_vec4":
        elements, = require_shape(shape, ("element_count",))
        expected_grid = ceil_div(elements, 1024)
        expected_block = 128
        operation = require_non_empty_string(attributes, "operation")
        valid = (
            operation in ("add", "mul")
            and elements % 1024 in (0, 512)
            and parameters["warp_instructions"]
            == source_bound_pytorch_vec4_elementwise_issue(
                shape, expected_block, operation)
            and (
                elements % 1024 != 0
                or parameters[
                    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER] == 4.0
            ))
    elif is_flash_attention_copy:
        expected_grid = grid
        expected_block = 128
        expected_issue = source_bound_flash_attention_copy_issue(
            shape, dtype, grid, block)
        valid = parameters["warp_instructions"] == expected_issue
    elif variant in ("kv_cache", "kvcache", "paged_kv_cache"):
        seq_len, kv_heads, head_dim = require_shape(
            shape, ("seq_len", "num_kv_heads", "head_dim"))
        elements_per_token = kv_heads * head_dim
        vector_packs = elements_per_token // 8
        expected_grid = seq_len
        expected_block = min(elements_per_token, 512)
        has_residency = (
            _SOURCE_BOUND_RESIDENT_CTAS_PER_UNIT_PARAMETER in parameters)
        valid = (
            elements_per_token % 8 == 0
            and vector_packs <= expected_block
            and vector_packs % 32 == 0
            and parameters[
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
            ] == float(ceil_div(vector_packs, 32))
            and has_residency == (expected_block == 512)
            and (
                not has_residency
                or parameters[
                    _SOURCE_BOUND_RESIDENT_CTAS_PER_UNIT_PARAMETER] == 2.0
            ))
    else:
        raise ValueError(
            "source-bound memory-dependency task has an unsupported Vector "
            f"variant: {variant!r}")
    if grid != expected_grid or block != expected_block or not valid:
        raise ValueError(
            "source-bound memory-dependency task disagrees with its exact "
            "source launch identity")
    read_source_required = (
        (variant == "rmsnorm" and shape["hidden"] % 256 == 0)
        or (variant == "add_rmsnorm" and shape["hidden"] % 256 == 0)
        or variant == "silu"
        or (variant == "elementwise_pytorch_vec4"
            and shape["element_count"] % 1024 == 0)
        or variant in ("rope", "kv_cache", "kvcache", "paged_kv_cache")
    )
    if read_source_required and read_parameter not in parameters:
        raise ValueError(
            "source-bound memory-dependency task exact source identity "
            "requires L2 read transaction geometry")
    if read_parameter in parameters:
        if variant == "rmsnorm":
            expected_read_transactions = 2.0
            read_domain_valid = shape["hidden"] % 256 == 0
        elif variant == "add_rmsnorm":
            expected_read_transactions = 32.0
            read_domain_valid = shape["hidden"] % 256 == 0
        elif variant == "silu":
            expected_read_transactions = 4.0
            read_domain_valid = block == 1024
        elif variant == "rope":
            _, query_heads, kv_heads, _, rotary_dim = require_shape(
                shape,
                ("seq_len", "num_query_heads", "num_kv_heads",
                 "head_dim", "rotary_dim"))
            key_present = require_boolean(attributes, "key_present")
            query_pairs = query_heads * rotary_dim // 2
            key_pairs = (
                kv_heads * rotary_dim // 2 if key_present else 0)
            expected_read_transactions = 8.0
            read_domain_valid = (
                query_pairs % block == 0
                and (not key_present
                     or key_pairs % 32 == 0))
        elif variant == "elementwise_pytorch_vec4":
            expected_read_transactions = 32.0
            read_domain_valid = shape["element_count"] % 1024 == 0
        elif is_flash_attention_copy:
            expected_read_transactions = 8.0
            read_domain_valid = True
        elif variant in ("kv_cache", "kvcache", "paged_kv_cache"):
            expected_read_transactions = 16.0
            read_domain_valid = True
        else:
            raise ValueError(
                "source-bound L2 read transactions have an unsupported "
                f"Vector variant: {variant!r}")
        if (not read_domain_valid
                or parameters[read_parameter]
                != expected_read_transactions):
            raise ValueError(
                "source-bound L2 read transactions disagree with the exact "
                "tail-free source transaction geometry")


# (operation, dtype) pairs of SGLang's PyTorch/Triton glue launches.
_SGLANG_GLUE_ELEMENTWISE_DTYPES = frozenset({
    ("fill", "int32"), ("fill", "int64"), ("fill", "float32"),
    ("fill", "bool"),
    ("copy", "int32"), ("copy", "int64"),
    ("unary", "int32"), ("unary", "int64"), ("unary", "float32"),
    ("add", "int32"), ("add", "int64"),
    ("cast_bf16_float32", "float32"),
})


def _pytorch_int32_int64_cast_launch(elements: int, variant: str) -> LaunchConfig:
    from src.kernel.launch_config.selection_primitives import pytorch_2_8_integer_copy_launch
    return pytorch_2_8_integer_copy_launch(elements, variant, "sm80")


def _pytorch_vectorized_row_gather_launch(rows: int, hidden: int, variant: str) -> LaunchConfig:
    # IndexKernelUtils.h eligibility: aligned contiguous slices, broadcast
    # INT64 IDs. Both index_select's large branch and advanced row indexing
    # use vectorized_gather_kernel<16,long>; the latter permits negative IDs.
    from src.kernel.launch_config.selection_primitives import pytorch_2_8_row_gather_launch
    return pytorch_2_8_row_gather_launch(rows, hidden, variant, "sm80", small_index=False)


def pytorch_embedding_index_select_launch_domain(rows, hidden, vocab, elements):
    """Static domain covered by this source-bound embedding launch."""
    from src.kernel.launch_config.selection_primitives import pytorch_2_8_embedding_gather_domain
    return pytorch_2_8_embedding_gather_domain(rows, hidden, vocab, elements, 108)


def pytorch_logits_cast_launch(shape, dtype, attributes):
    """Torch2.8 SM80 Float-output Copy, one exact PRMT bit expansion/value."""
    require_dtype(dtype, ("bf16",))
    elements, = require_shape(shape, ("element_count",))
    if attributes["sampling_source"] != "vllm_0_11_plain_sampler_bf16" or "copy_dispatch" in attributes:
        raise ValueError("A800 logits Copy requires its0.11 BF16 source")
    if "copy_input_contiguous" not in attributes or attributes["copy_input_contiguous"] is not True:
        raise NotImplementedError("A800 logits Copy requires proven contiguous input")
    if elements <= 0 or 4 * (elements - 1) > 2**31 - 1:
        raise NotImplementedError("A800 logits Copy requires an unsplit32-bit iterator")
    return LaunchConfig(algorithm="pytorch_2_8_sm80_bfloat_to_float_prmt",
        grid=((-(-elements // 512)), 1, 1), block=(128, 1, 1),
        parameters={"registers_per_thread": 30.0, "values_per_thread": 4.0},
        execution_variant="vllm_logits_fp32_cast")


def pytorch_id_narrow_launch(shape, dtype, attributes):
    """SM80 Int-output Copy loads only the low word at each Long address."""
    require_dtype(dtype, ("int64",))
    elements, rows = require_shape(shape, ("element_count", "rows"))
    if (attributes["sampling_source"] != "vllm_0_11_plain_greedy_bf16"
            or attributes["stage"] != "output_int32" or elements != rows):
        raise ValueError("A800 ID narrowing requires its0.11 greedy INT64 output source")
    if elements <= 0 or 8 * (elements - 1) > 2**31 - 1:
        raise NotImplementedError("A800 ID narrowing requires one unsplit32-bit iterator")
    return LaunchConfig(algorithm="pytorch_2_8_sm80_int64_to_int32",
        grid=((-(-elements // 512)), 1, 1), block=(128, 1, 1),
        parameters={"registers_per_thread": 32.0, "values_per_thread": 4.0},
        execution_variant="vllm_plain_greedy_sample")


def select_vector_launch(shape: Mapping[str, int], dtype: Optional[str],
                         variant: Optional[str], attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    if variant == "vllm_plain_greedy_sample" and attributes["stage"] == "output_int32":
        if hardware.hardware_id != "nvidia_a800":
            raise ValueError("SM80 output Copy requires the A800 source")
        return pytorch_id_narrow_launch(shape, dtype, attributes)
    if variant == "vllm_logits_fp32_cast":
        if hardware.hardware_id != "nvidia_a800":
            raise ValueError("SM80 logits Copy requires the A800 source")
        return pytorch_logits_cast_launch(shape, dtype, attributes)
    if (variant == "elementwise" and "operation" in attributes
            and attributes["operation"] == "cast_int32_int64"):
        require_dtype(dtype, ("int32",))
        if (hardware.hardware_id != "nvidia_a800"
                or attributes["tensor_iterator_source"] != "pytorch_2_8_sm80"):
            raise NotImplementedError("integer cast requires its explicit A800 Torch source")
        elements, = require_shape(shape, ("element_count",))
        return _pytorch_int32_int64_cast_launch(elements, variant)
    if variant == "vllm_logits_row_select":
        require_dtype(dtype, ("bf16",))
        rows, hidden, source_rows, elements = require_shape(shape, ("rows", "hidden", "source_rows", "element_count"))
        if (hardware.hardware_id != "nvidia_a800"
                or attributes["model_execution_version"] != "vllm_0_11_0"
                or attributes["index_source"] != "query_start_loc_int32_to_int64"):
            raise NotImplementedError("row selection requires its materialized A8000.11 Long indices")
        if (rows <= 0 or source_rows <= 0 or hidden <= 0 or hidden % 128
                or elements != rows * hidden
                or max(2 * source_rows * hidden, 2 * elements, 8 * rows) >= 2**31):
            raise NotImplementedError("row selection requires contiguous aligned unsplit rows")
        return _pytorch_vectorized_row_gather_launch(rows, hidden, variant)
    if variant == "gather" and "index_select_source" in attributes:
        require_dtype(dtype, ("bf16",))
        rows, hidden, vocab, elements = require_shape(shape, ("rows", "hidden", "source_rows", "element_count"))
        if (hardware.hardware_id != "nvidia_a800"
                or attributes["index_select_source"] != "vllm_0_11_embedding"
                or attributes["index_bytes"] != 8
                or attributes["index_count"] != rows
                or attributes["gather_factor"] != 1
                or attributes["materialized_output"] is not True):
            raise NotImplementedError("embedding index_select requires its A800 BF16/INT64 source")
        # The installed Torch 2.8 small-index program is used only for <=16
        # IDs and 32-bit addressable operands. Above 16 IDs index_select now
        # lowers to a broadcast-index gather_out, not indexSelectLargeIndex.
        if not pytorch_embedding_index_select_launch_domain(rows, hidden, vocab, elements):
            raise NotImplementedError("embedding index_select requires its contiguous unsplit 32-bit domain")
        small = rows <= 16
        if not small:
            return _pytorch_vectorized_row_gather_launch(rows, hidden, "gather")
        from src.kernel.launch_config.selection_primitives import pytorch_2_8_row_gather_launch
        return pytorch_2_8_row_gather_launch(rows, hidden, "gather", "sm80", small_index=True)
    if variant == "vllm_embedding_id_widen":
        require_dtype(dtype, ("int32",))
        rows, elements = require_shape(shape, ("rows", "element_count"))
        if (hardware.hardware_id != "nvidia_a800"
                or attributes["embedding_source"] != "vllm_0_11_unquantized_tp1"):
            raise NotImplementedError("embedding ID widen binary binding requires A800 vLLM 0.11")
        # Torch 2.8 sm_80 direct_copy's dynamic-cast (not nocast) branch:
        # CUDALoops.cuh launch_unrolled_kernel, nt=128, vt=4. A Long
        # output determines the largest contiguous iterator byte offset.
        # Larger iterators split recursively and require multiple launches.
        if rows != elements or rows <= 0 or 8 * (rows - 1) > 2**31 - 1:
            raise NotImplementedError("embedding ID widen requires one contiguous 32-bit-indexable iterator")
        return _pytorch_int32_int64_cast_launch(rows, variant)
    if (variant == "elementwise" and "operation" in attributes
            and attributes["operation"] == "copy" and dtype == "float32"):
        normalized_dtype = "float32"
    elif (variant == "elementwise" and "operation" in attributes
            and (attributes["operation"], dtype)
            in _SGLANG_GLUE_ELEMENTWISE_DTYPES):
        # SGLang's small index/metadata glue (MoE top-k shared-expert append,
        # EAGLE V2 verify/draft bookkeeping, FP32 logits materialization).
        normalized_dtype = dtype
    elif variant == "reduce" and dtype == "float32":
        normalized_dtype = dtype
    else:
        normalized_dtype = require_dtype(dtype, ("bf16", "fp16"))
    execution_variant = require_gpu_vector_variant(variant)

    if execution_variant in _SOURCE_BOUND_MOE_VARIANTS:
        return _select_source_bound_moe_launch(
            shape, execution_variant, normalized_dtype, attributes, hardware)

    # Exact vLLM 0.11.0 launch geometry used by the serving-aligned A800
    # microbenchmark. Keep this in the NVIDIA package: MetaX has a separate
    # runtime even though both packages share the generic fallback below.
    exact_variants = {
        "rmsnorm", "add_rmsnorm", "rotary", "rope", "activation", "silu",
        "kv_cache", "kvcache", "elementwise",
        "elementwise_pytorch_vec4",
    }
    if execution_variant not in exact_variants:
        return select_gpu_vector_launch(
            shape, execution_variant, attributes)
    # Publish source-level semantics, launch geometry, resources, transaction
    # contracts, and explicitly bounded SASS work where the execution domain
    # is closed below.
    parameters = {}
    algorithm = execution_variant
    source_bound_rope = False
    source_bound_kv_cache = False
    source_bound_elementwise = False
    if execution_variant == "rmsnorm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        grid, block = rows, min(hidden, 1024)
        # The vLLM kernel reads the input once for the reduction and once
        # again after the block-wide reduction.  This is one source-defined
        # reread per logical input element, not a fitted traffic multiplier.
        parameters["input_reread_count"] = 1.0
        parameters["norm_reduction_tree_flops"] = float(
            rows * (block - 1))
        parameters["norm_epsilon_add_flops"] = float(rows)
        if normalized_dtype == "bf16" and (
                block == 1024 or hidden % 256 != 0):
            # The scalar vLLM rms_norm_kernel runs one CTA per row with a
            # dependent post-reduction load -> normalize -> store round; that
            # complete memory-dependency task binding is source semantics of
            # the same kernel template, not a fitted correction.  The typed
            # vec8 tail-free extras remain closed to the block-1024 domain
            # whose reduction and read-round services are independently
            # characterized; tail-free widths below 1024 stay unmarked until
            # those services cover their block sizes.
            parameters[SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = 1.0
            if block == 1024 and hidden % 256 == 0:
                # One BF16 source load per lane becomes two independent
                # 32-byte L2 transactions on each active warp.  The predictor
                # separately restores a partially populated final loop round,
                # so the same source contract covers the vec-width-aligned
                # tail instead of dropping it to a bandwidth-only path.
                parameters[
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                ] = 2.0
                # The scalar RMS kernel uses the same CUB float reduction,
                # scalar variance normalization, rsqrt publication, and block
                # barrier as the independently measured reduction mechanism.
                parameters["source_bound_block_reduction"] = 1.0
                weight_l2_copies = _source_bound_rmsnorm_gamma_l2_copies(
                    rows, hidden, normalized_dtype, block, hardware)
                if weight_l2_copies is not None:
                    parameters["norm_weight_l2_copies"] = float(
                        weight_l2_copies)
                # Every tail-free block1024 launch uses the same one-CTA-per-
                # row source program.  Grid size changes worker and wave
                # geometry, not this program identity.
                parameters[
                    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] = 1.0
    elif execution_variant == "add_rmsnorm":
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        grid = rows
        max_block = 256 if rows >= 256 else 1024
        block = min(hidden, max_block)
        # The first pass writes input + residual to residual; the normalized
        # output pass rereads that residual once.
        parameters["input_reread_count"] = 1.0
        parameters["norm_reduction_tree_flops"] = float(
            rows * (block - 1))
        parameters["norm_epsilon_add_flops"] = float(rows)
        if normalized_dtype == "bf16" and hidden % 256 == 0:
            # The vec8 source executes two dependency phases around the block
            # reduction. In each phase an active lane consumes two aligned
            # 16-byte packs, which form 32 independent 32-byte sectors per
            # full warp. The active producer count follows directly from the
            # vec8 extent and the selected block size.
            active_threads = min(block, ceil_div(hidden, 8))
            parameters[
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
            ] = 32.0
            parameters[
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
            ] = float(ceil_div(active_threads, hardware.parallel_width))
            parameters[SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER] = float(
                6 * rows * hidden)
            # The pinned source executes one CUB block reduction and
            # publication barrier between its two width-eight passes.  The
            # hardware service is characterized independently from any
            # production-kernel duration.
            parameters["source_bound_block_reduction"] = 1.0
            parameters[SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = 1.0
            # Every tail-free BF16 launch executes the same two-pass
            # AddRMSNorm CTA program.  Grid size changes only the number and
            # residency class of those CTAs; it cannot change the program to
            # an opaque complete-task dependency.  Publish the program marker
            # for one-wave and multi-wave launches alike.  Exact source-round
            # services remain optional evidence inside this common pipeline.
            parameters[
                SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] = 1.0
    elif execution_variant == "rotary":
        seq_len, query_heads, kv_heads, head_dim = require_shape(
            shape,
            ("seq_len", "num_query_heads", "num_kv_heads", "head_dim"))
        if query_heads % kv_heads != 0:
            raise ValueError(
                "NVIDIA RoPE requires num_query_heads to be divisible by "
                "num_kv_heads")
        query_pairs = query_heads * (head_dim // 2)
        grid, block = seq_len, min(query_pairs, 512)
        # vLLM loads one cos and one sin scalar for every rotated Q/K pair in
        # every head.  This is cache-path traffic over the complete logical
        # Q+K tensor, while compulsory DRAM only fetches the unique cache row
        # once per token.  Both supported dtypes are two bytes per scalar.
        dtype_bytes = {"bf16": 2, "fp16": 2}[normalized_dtype]
        parameters["rotary_cache_path_bytes"] = float(
            seq_len * (query_heads + kv_heads) * head_dim * dtype_bytes)
        parameters["rotary_position_path_bytes"] = float(
            seq_len
            * ceil_div(block, hardware.parallel_width)
            * hardware.l2_transaction_bytes)
    elif execution_variant == "rope":
        (seq_len, query_heads, kv_heads, head_dim,
         rotary_dim) = require_shape(
             shape,
             ("seq_len", "num_query_heads", "num_kv_heads", "head_dim",
              "rotary_dim"))
        if query_heads % kv_heads != 0:
            raise ValueError(
                "NVIDIA RoPE requires num_query_heads to be divisible by "
                "num_kv_heads")
        if rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError(
                "NVIDIA typed RoPE requires an even rotary_dim no larger "
                "than head_dim")
        is_neox = require_boolean(attributes, "is_neox")
        key_present = require_boolean(attributes, "key_present")
        query_pairs = query_heads * (rotary_dim // 2)
        key_pairs = kv_heads * (rotary_dim // 2) if key_present else 0
        grid, block = seq_len, min(query_pairs, 512)
        dtype_bytes = {"bf16": 2, "fp16": 2}[normalized_dtype]
        parameters["rotary_cache_path_bytes"] = float(
            seq_len
            * (query_heads + (kv_heads if key_present else 0))
            * rotary_dim
            * dtype_bytes)
        parameters["rotary_position_path_bytes"] = float(
            seq_len
            * ceil_div(block, hardware.parallel_width)
            * hardware.l2_transaction_bytes)
        source_bound_rope = (
            normalized_dtype == "bf16"
            and is_neox
            and key_present
            and block == 512
            and rotary_dim == head_dim
        )
        if source_bound_rope:
            parameters[SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = 1.0
        if (query_pairs % block == 0
                and (not key_present or key_pairs % hardware.parallel_width == 0)):
            # In each source loop iteration a full warp independently issues
            # x, y, cos and sin BF16 loads before either output can be formed.
            # Each 32-lane scalar load spans two 32-byte L2 sectors, so the
            # source exposes eight independent read transactions per active
            # warp.  This is derived from the C++ access pattern and launch
            # geometry; it is independent of compiler instruction counts.
            parameters[
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
            ] = 8.0
    elif execution_variant in ("activation", "silu"):
        tokens, intermediate = require_shape(shape, ("tok", "intermediate"))
        grid, block = tokens, min(intermediate, 1024)
        if normalized_dtype == "bf16" and block == 1024:
            if execution_variant == "silu":
                if tokens > 0:
                    # The pinned sm_80 BF16 specialization uses 24 registers
                    # per thread.  With a 1024-thread block, a one-wave launch
                    # has exactly one CTA on each active SM, so no peer CTA can
                    # hide the source program's warp-issue work.  The ordinary
                    # SASS path has a 480-instruction fixed body plus 47
                    # executed instructions per active warp-round; exceptional
                    # FCHK branches only add work. The independently validated
                    # first/resident/queued CTA schedule below supplies the
                    # corresponding multi-wave overlap contract, so this
                    # conserved launch-wide issue bound applies to every
                    # launch of the same source program.
                    # Upstream v0.11.0 activation source SHA256:
                    # ba96abce691ac8335243b763ff755d007cc82f1cab6a3edf4e438ce085bc1691
                    # Installed _C.abi3.so SHA256:
                    # 984de21ce38cc8083d207663a380ca75acc6ad3dc230858092f59e363c4eebd5
                    parameters["warp_instructions_lower_bound"] = (
                        _source_bound_silu_issue_lower_bound(shape, block))
                parameters[SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = (
                    1.0)
                if tokens >= hardware.num_compute_units:
                    # The independently frozen block-1024 campaign derives
                    # the first/resident/queued services at full-device grids
                    # 108, 216, and 1024, and validates the resulting
                    # resident/queued composition at grid 256.  A full-SM
                    # launch of this identical source program has the same
                    # per-worker CTA sequence; only the number of queued
                    # cohorts changes.  Preserve that source-program
                    # composition for every full-device grid instead of
                    # treating one token count as an operator variant.
                    parameters[
                        SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER
                    ] = 1.0
                # Every launched thread owns at least one output because the
                # source chooses block=min(intermediate, 1024).  The two
                # adjacent BF16 input loads therefore expose four independent
                # 32-byte sectors for each full warp.  A later partial loop
                # round changes only the aggregate byte/request count; it
                # does not remove the 32 source warps established by the
                # first full round.
                parameters[
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                ] = 4.0
    elif execution_variant in ("kv_cache", "kvcache"):
        require_paged_kv_cache_contract(attributes)
        seq_len, kv_heads, head_dim = require_shape(
            shape, ("seq_len", "num_kv_heads", "head_dim"))
        elements_per_token = kv_heads * head_dim
        grid, block = seq_len, min(elements_per_token, 512)
        # Every thread executes the same 64-bit slot_mapping load before the
        # element loop.  CUDA coalesces one warp's accesses to that single
        # address into one 32-byte transaction on CC >= 6.0, so this is the
        # physical L1 request traffic: one sector per participating warp and
        # token.  It is deliberately separate from the unique slot footprint
        # that can continue to L2/DRAM.
        parameters["kv_cache_slot_l1_bytes"] = float(
            seq_len * ceil_div(block, 32) * 32)
        if normalized_dtype == "bf16":
            source_bound_kv_cache = (
                attributes["cache_head_layout"] == "nhd"
                and attributes["kv_cache_dtype"] == "auto"
                and attributes["vector_copy_base_alignment_bytes"] >= 16
                and elements_per_token % 8 == 0
                and elements_per_token // 8 <= block
                and (elements_per_token // 8) % hardware.parallel_width == 0
            )
            if source_bound_kv_cache:
                # Each participating BF16 vec8 thread owns one 16-byte pack.
                # One full copy warp therefore produces sixteen 32-byte L2
                # sectors in either the K or V source loop.  Threads beyond
                # the vec8 extent execute control work but produce no payload
                # requests, so publish the active copy-warp count separately.
                vector_packs = elements_per_token // 8
                parameters[
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                ] = float(
                    hardware.parallel_width * 16
                    // hardware.l2_transaction_bytes)
                parameters[
                    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
                ] = float(ceil_div(
                    vector_packs, hardware.parallel_width))
                # The pinned sm_80 binary uses 54 registers per thread and
                # zero static shared memory. Derive its CTA residency from
                # that binary resource identity and the HWSpec limits rather
                # than the generic thread-only upper bound.
                if block == 512:
                    parameters[
                        _SOURCE_BOUND_RESIDENT_CTAS_PER_UNIT_PARAMETER
                    ] = float(_source_bound_kv_cache_resident_ctas(
                        hardware, block))
                parameters[SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = (
                    1.0)
    elif execution_variant in (
            "elementwise", "elementwise_pytorch_vec4"):
        elements, = require_shape(shape, ("element_count",))
        operation = require_non_empty_string(attributes, "operation")
        if operation not in ("add", "mul", "unary", "cast_bf16_float32",
                             "copy", "fill", "read"):
            raise ValueError(
                f"unsupported NVIDIA elementwise operation: {operation!r}")
        if execution_variant == "elementwise" and operation == "copy":
            # PyTorch's non-vectorized direct_copy template launches nt=128
            # with vt=4 for 16-bit dtypes and vt=2 for FP32; the generic
            # eight-values-per-thread geometry does not describe it.
            if normalized_dtype in ("bf16", "fp16"):
                values_per_thread = 4
            elif normalized_dtype in ("float32", "int32", "int64"):
                # gpu_kernel_impl_nocast: unroll 2 when sizeof >= 4.
                values_per_thread = 2
            else:
                raise NotImplementedError(
                    "NVIDIA direct-copy launch is source-backed only for "
                    f"16-bit and 4-byte dtypes, got {normalized_dtype!r}")
            grid, block = ceil_div(elements, 128 * values_per_thread), 128
            if normalized_dtype == "bf16" and "input_materialization" in attributes:
                # vLLM 0.11.0 QK-norm materializes the strided QKV split view
                # via input.contiguous() (_custom_ops.py rms_norm), which
                # dispatches the same PyTorch TensorIterator direct_copy
                # program as the observed FlashAttention copies:
                # elementwise_kernel<128, 4, gpu_kernel_impl_nocast<
                # direct_copy_kernel_cuda>>.  The grouped strided source
                # coalesces to a lower-rank iterator than the rank-4 floor
                # probe, and its measured duration matches the FA copy at
                # equal grid, so the fixed-cost-dominated BF16 vt4 complete-
                # task floor bounds it.  Grids beyond the measured floor
                # support keep the analytical path through the predictor's
                # fail-soft limitation.
                if (attributes["input_materialization"]
                        != "grouped_strided_to_contiguous"):
                    raise NotImplementedError(
                        "NVIDIA A800 direct copy is source-bound only for "
                        "the grouped-strided Norm input program, got "
                        f"{attributes['input_materialization']!r}")
                algorithm = (
                    "observed_norm_input_materialization_copy:elementwise")
                parameters[
                    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER] = 4.0
        else:
            grid, block = ceil_div(elements, 1024), 128
        if execution_variant == "elementwise_pytorch_vec4":
            if operation not in ("add", "mul"):
                raise ValueError(
                    "typed PyTorch vec4 elementwise requires add or mul")
            source_bound_elementwise = (
                normalized_dtype == "bf16"
                and elements % 1024 in (0, 512)
            )
            if source_bound_elementwise:
                parameters[SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = (
                    1.0)
                parameters["warp_instructions"] = (
                    source_bound_pytorch_vec4_elementwise_issue(
                        shape, block, operation))
                if elements % 1024 == 0:
                    # The full vec4 body has four independent LDG.E.64 sites;
                    # every full warp issues eight 32-byte sectors per site.
                    parameters[
                        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                    ] = 32.0
                    # A tail-free CTA has 128 active threads, hence four
                    # independent warp producers. Publish this source/launch
                    # fact so the generic finite-source memory model may use
                    # the architectural thread-limited resident-CTA bound.
                    parameters[
                        SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
                    ] = 4.0
    else:
        raise AssertionError(
            f"unhandled exact NVIDIA Vector variant: {execution_variant!r}")
    return LaunchConfig(
        algorithm=algorithm,
        grid=(grid, 1, 1),
        block=(block, 1, 1),
        parameters=parameters,
        execution_variant=execution_variant,
    )
