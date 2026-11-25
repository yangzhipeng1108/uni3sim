"""H200 BF16/FP16 dense FlashMLA decode from exact versioned sources.

Source identity: vLLM0.30/0.27.1 cmake/external_projects/flashmla.cmake.
Each revision was independently inspected. FlashMLA's
api/dense_decode.{h,cpp} and sm90/decode/dense/{config.h,traits.h,splitkv_mla.cuh}
fix head dimensions, packed query/head tiles, threads and shared layout.
smxx/decode owns metadata and combine. This is the non-FA3 dense operator;
the query and cache are already prepared by their actual caller producers.
The0.27.1 combine grid is(B,Q,H/8);0.30 packs(B*Q,1,H/8).
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _source_geometry(shape, dtype, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 dense FlashMLA requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    version = attributes["model_execution_version"]
    snapshot = attributes["source_flashmla_snapshot"]
    if (version, snapshot) == ("vllm_0_30_0", "vllm_0_30_cmake_pin"):
        source_version = "0_30"
    elif (version, snapshot) == ("vllm_0_27_1", "vllm_0_27_1_cmake_pin"):
        source_version = "0_27_1"
    else:
        raise NotImplementedError("dense FlashMLA requires its exact version and matching pinned source")
    if (attributes["source_attention_backend"] != "flashmla_dense"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_query_layout"] != "contiguous_b_s_h_d"
            or attributes["source_cache_layout"] != "paged_contiguous_b64_h1_d576"
            or attributes["source_cache_dtype"] != dtype
            or attributes["source_context_lens_layout"] != "contiguous_int32"
            or attributes["source_block_table_inner_stride"] != 1
            or type(attributes["source_block_table_inner_stride"]) is not int
            or attributes["source_batch_invariant"] is not False
            or attributes["causal"] is not True
            or type(attributes["context_parallel_size"]) is not int
            or attributes["context_parallel_size"] != 1):
        raise NotImplementedError("dense FlashMLA requires its prepared query, matching cache dtype and CP1 causal source")
    fp16_enabled = attributes["source_flashmla_fp16_enabled"]
    if type(fp16_enabled) is not bool:
        raise TypeError("FlashMLA FP16 build availability must be an explicit boolean")
    if dtype == "fp16" and not fp16_enabled:
        raise NotImplementedError("the declared FlashMLA build disables FP16 kernels")
    batch, query, heads, kv_heads, dim, value_dim, page = require_shape(shape, (
        "batch", "q_len", "num_heads", "num_kv_heads", "qk_head_dim", "v_head_dim", "block_size"))
    # Although the API's early check mentions512, the pinned SM90 run()
    # asserts Config::HEAD_DIM_K==576. Do not admit an uncompiled variant.
    if (kv_heads != 1 or dim != 576 or value_dim != 512 or page != 64
            or query > 128 or batch >= 2**31 or query * heads >= 2**31
            or batch * query >= 2**31 or ceil_div(heads, 8) > 65535):
        raise NotImplementedError("dense FlashMLA binds uniform q1..128,one latent KV head and576/512-wide64-row pages")
    packed_rows = query * heads
    m_blocks = ceil_div(packed_rows, 64)
    parts = max(int(hardware.num_compute_units) // m_blocks, 1)
    return batch, query, heads, packed_rows, m_blocks, parts, source_version


def _dense_shared_bytes():
    """sizeof Traits<InputT>::SharedMemoryPlan; both bound types are2B.

    Three64x576 BF16/FP16 arrays,64x64 probability array,five64-float
    vectors (the reduction workspace counts twice),and19 eight-byte TMA
    barriers. cute::array_aligned makes the aggregate16-byte aligned.
    """
    arrays = 3 * 64 * 576 * 2 + 64 * 64 * 2 + 5 * 64 * 4
    barriers = (2 * (576 // 64) + 1) * 8
    return ceil_div(arrays + barriers, 16) * 16


def select_dense_flashmla_decode_recipe(shape, dtype, attributes, hardware):
    """Complete prepared-query dense MLA call: metadata/main/combine.

    The one-KV-head source transpose/reshape remains a view. Q concatenation
    or absorption is an explicit upstream producer, not silently folded into
    this operator. Metadata is generated only when requested. The source
    always submits combine,including parts=1;no zero-Fill is added for fresh
    empty scratch allocations. Physical helpers do not bind an aggregate
    KernelSpec or a source-aware Midlayer price.
    """
    batch, query, heads, packed, m_blocks, parts, version = _source_geometry(shape, dtype, attributes, hardware)
    del packed
    scheduler = attributes["source_scheduler_state"]
    if scheduler == "build_current_context":
        shared = 4 * (5 * batch + 1)
        if shared > 232448:
            raise NotImplementedError("dense FlashMLA metadata exceeds SM90 opt-in shared capacity")
        setup = (("schedule_metadata", LaunchConfig(f"flashmla_vllm_{version}_sm90_dense_metadata",
            (1, 1, 1), (32, 1, 1), parameters={"dynamic_shared_bytes": float(shared)})),)
    elif scheduler == "provided_current_context":
        metadata = attributes["source_tile_scheduler_shape"]
        splits = attributes["source_num_splits_shape"]
        if (not isinstance(metadata, (tuple, list)) or not isinstance(splits, (tuple, list))
                or any(type(value) is not int for value in (*metadata, *splits))
                or tuple(metadata) != (parts, 8) or tuple(splits) != (batch + 1,)):
            raise ValueError("dense FlashMLA scheduler shape must match packed query/head tiles and actual batch")
        setup = ()
    else:
        raise NotImplementedError("dense FlashMLA requires explicit current-context scheduler ownership")
    # dense_decode.cpp normalizes causal=False for one query per request.
    causal = "causal" if query > 1 else "single_query"
    main = LaunchConfig(f"flashmla_vllm_{version}_sm90_dense_{dtype}_{causal}",
        (m_blocks, 1, parts), (256, 1, 1), parameters={
            "dynamic_shared_bytes": float(_dense_shared_bytes()),
            "source_programmatic_stream_serialization": 1.0,
            "source_causal": float(query > 1)})
    combine_grid = ((batch * query, 1, ceil_div(heads, 8)) if version == "0_30"
                    else (batch, query, ceil_div(heads, 8)))
    combine = LaunchConfig(f"flashmla_vllm_{version}_sm90_dense_combine_{dtype}_s{ceil_div(parts, 32) * 32}",
        combine_grid, (256, 1, 1), parameters={
            "dynamic_shared_bytes": 0.0, "source_programmatic_stream_serialization": 1.0})
    return (*setup, ("attention_main", main), ("attention_combine", combine))


def dense_flashmla_buffer_extents(shape, dtype, attributes, hardware):
    """Exact source allocation extents, not traffic or runtime allocator capacity.

    The dense C++ API allocates FP32 accumulation slots batch+parts, even
    when some slots are inactive. Empty allocations do not imply a Fill or
    that every allocated byte is accessed. The output/LSE return reshapes
    only create views in the declared one-KV-head domain.
    """
    batch, query, heads, packed, m_blocks, parts, version = _source_geometry(shape, dtype, attributes, hardware)
    del m_blocks, version
    return {
        "output": {"shape": (batch, 1, packed, 512), "dtype": dtype, "bytes": batch * packed * 512 * 2},
        "lse": {"shape": (batch, 1, packed), "dtype": "fp32", "bytes": batch * packed * 4},
        "output_accum": {"shape": (batch + parts, 1, packed, 512), "dtype": "fp32",
                         "bytes": (batch + parts) * packed * 512 * 4},
        "lse_accum": {"shape": (batch + parts, 1, packed), "dtype": "fp32",
                      "bytes": (batch + parts) * packed * 4},
        "tile_scheduler": {"shape": (parts, 8), "dtype": "int32", "bytes": parts * 8 * 4},
        "num_splits": {"shape": (batch + 1,), "dtype": "int32", "bytes": (batch + 1) * 4},
        "returned_output_shape": (batch, query, heads, 512),
        "returned_lse_shape": (batch, heads, query),
    }
