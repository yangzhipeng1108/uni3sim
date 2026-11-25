"""DSA V3.2 sparse FP8 FlashMLA source program on H200, excluding FA3.

The FlashMLA revision is owned by vLLM0.30's
cmake/external_projects/flashmla.cmake. Its sparse_decode.cpp,
sm90/decode/sparse/{config.h,splitkv_mla.cuh} and smxx/decode/combine
fix the rules below. vLLM's flashmla_sparse.py and sparse_utils.py own
concat, index conversion and query-head padding. Compiler registers and
static shared are separately unbound. No observed duration is consumed.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _require_source(dtype, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 sparse FlashMLA requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_flashmla_snapshot"] != "vllm_0_30_cmake_pin"
            or attributes["source_kv_cache_layout"] != "fp8_ds_mla_656_aos"
            or attributes["source_dynamic_topk"] is not False
            or attributes["source_attn_sink"] != "none"
            or attributes["source_extra_cache"] != "none"
            or type(attributes["context_parallel_size"]) is not int
            or attributes["context_parallel_size"] != 1):
        raise NotImplementedError("DSA FlashMLA requires its explicit0.30 SM90 V3.2 FP8 cache and CP1 source")


def _geometry(shape, attributes, hardware):
    batch, q_len, heads, topk, page = require_shape(shape, ("batch", "q_len", "num_heads", "topk", "block_size"))
    if (heads > 128 or topk not in (512, 1024, 2048, 4096)
            or batch * q_len * heads >= 2**31 or page != 64):
        raise NotImplementedError("DSA FlashMLA source head/topk/query extent is outside the bound domain")
    layout = attributes["source_sequence_layout"]
    if layout == "flattened_index_group":
        kernel_batch, kernel_q_len = 1, batch * q_len
    elif layout == "native_batch":
        kernel_batch, kernel_q_len = batch, q_len
    else:
        raise NotImplementedError("sparse FlashMLA requires its actual batch/query source layout")
    if kernel_q_len > 65535:
        raise NotImplementedError("SM90 sparse FlashMLA query extent exceeds CUDA grid.y capacity")
    padded_heads = 64 if heads <= 64 else 128
    sms = int(hardware.num_compute_units)
    parts = max(sms // kernel_q_len // (padded_heads // 64), 1)
    return batch * q_len, heads, padded_heads, kernel_batch, kernel_q_len, parts


def _v32_shared_bytes():
    """SharedMemoryPlan sizeof, from source's dense swizzled layout extents.

    Q=64x576 BF16; union=max(two K buffers,BF16 O,FP32 OAccum stride520);
    S=64x64 BF16 aligned1024; two64-bool validity vectors; four64-float
    vectors; seven8-byte ClusterTransactionBarrier objects. The aggregate
    has1024-byte alignment. This is host layout, not a register estimate.
    """
    q = 64 * 576 * 2
    union = max(2 * q, 64 * 512 * 2, ((64 - 1) * 520 + 512) * 4)
    before_s = ceil_div(q + union, 1024) * 1024
    used = before_s + 64 * 64 * 2 + 2 * 64 + 4 * 64 * 4 + 7 * 8
    return ceil_div(used, 1024) * 1024


def select_dsa_flashmla_core_recipe(shape, dtype, attributes, hardware):
    """Complete sparse FlashMLA call: optional metadata, main and combine.

    The SM90 path always submits combine even for one partition. Its host
    sets an opt-in shared limit but launches combine with zero dynamic
    shared bytes; the static compiler allocation remains a separate fact.
    """
    _require_source(dtype, attributes, hardware)
    rows, heads, padded, batch, q_len, parts = _geometry(shape, attributes, hardware)
    del rows, heads
    scheduler = attributes["source_scheduler_state"]
    if scheduler == "build_current_context":
        shared = 4 * (batch * 5 + 1)
        if shared > 232448:
            raise NotImplementedError("FlashMLA metadata exceeds the SM90 opt-in shared-memory domain")
        setup = (("schedule_metadata", LaunchConfig("flashmla_vllm_0_30_sm90_sparse_metadata",
            (1, 1, 1), (32, 1, 1), parameters={"dynamic_shared_bytes": float(shared)})),)
    elif scheduler == "provided_current_context":
        meta = attributes["source_tile_scheduler_shape"]
        splits = attributes["source_num_splits_shape"]
        if (not isinstance(meta, (tuple, list)) or not isinstance(splits, (tuple, list))
                or any(type(value) is not int for value in (*meta, *splits))
                or tuple(meta) != (parts, 8) or tuple(splits) != (batch + 1,)):
            raise ValueError("FlashMLA scheduler tensors must match actual SM90 batch/query partition geometry")
        setup = ()
    else:
        raise NotImplementedError("FlashMLA requires explicit current-context scheduler ownership")
    main = LaunchConfig(f"flashmla_vllm_0_30_sm90_v32_sparse_fp8_h{padded}",
        (padded // 64, q_len, parts), (384, 1, 1), cluster=(padded // 64, 1, 1),
        parameters={"dynamic_shared_bytes": float(_v32_shared_bytes()),
                    "source_programmatic_stream_serialization": 0.0})
    split_bucket = ceil_div(parts, 32) * 32
    combine = LaunchConfig(f"flashmla_vllm_0_30_sm90_v32_sparse_combine_s{split_bucket}",
        (batch * q_len, 1, padded // 8), (256, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0, "source_programmatic_stream_serialization": 1.0})
    return (*setup, ("attention_main", main), ("attention_combine", combine))


def select_dsa_flashmla_attention_recipe(shape, dtype, attributes, hardware):
    """Prepared logical indices and Q halves through the sparse FP8 output.

    Return existing LaunchConfig/PhysicalTransferConfig stages. This does
    not attach one kernel to a complete SPARSE_ATTN KernelSpec. Concat,
    remap, optional padded Q Fill/Copy, main and combine remain separate.
    Inputs are already contiguous INT32 indices and the actual BMM-NoPE/
    contiguous-RoPE views. Metadata/setup producers outside this call are
    represented only when explicitly requested by source_scheduler_state.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import (
        select_torch213_fill_launch, select_torch213_strided_copy_launch)
    _require_source(dtype, attributes, hardware)
    rows, heads, padded, batch, q_len, parts = _geometry(shape, attributes, hardware)
    del batch, q_len, parts
    if (attributes["source_q_layout"] != "bmm_transposed_nope_contiguous_rope"
            or attributes["source_index_inputs_layout"] != "contiguous_int32"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"):
        raise NotImplementedError("DSA sparse caller requires its actual Q/index/Torch2.13 source layouts")
    count_valid = attributes["source_return_valid_counts"]
    if type(count_valid) is not bool:
        raise TypeError("sparse index remap requires the actual valid-count branch")
    remap_stream = attributes["source_index_remap_stream"]
    if remap_stream not in ("caller_stream", "index_group_side_stream"):
        raise NotImplementedError("sparse index remap requires its actual source stream ownership")
    topk, = require_shape(shape, ("topk",))
    concat = LaunchConfig("vllm_0_30_sm90_concat_mla_q_d512_r64",
        (ceil_div(rows * heads, 8), 1, 1), (256, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0})
    if count_valid:
        remap_fill = select_torch213_fill_launch({"element_count": rows * topk}, "int32",
            {**attributes, "source_input_layout": "contiguous_aligned_16"}, hardware)
        remap_setup = (("index_sentinel_fill", remap_fill),)
        remap_grid, remap_threads = (rows, 1, 1), 256
        remap_algorithm = "vllm_0_30_sm90_sparse_index_remap_count_compact"
    else:
        remap_setup = ()
        remap_grid, remap_threads = (rows, topk // 128, 1), 128
        remap_algorithm = "vllm_0_30_sm90_sparse_index_remap_tiled"
    # sparse_utils.py::_remap_tiling explicitly sets8/4 warps. No local
    # compiler default or observation-derived block is needed.
    remap = LaunchConfig(remap_algorithm, remap_grid, (remap_threads, 1, 1),
        parameters={"source_index_group_side_stream": float(remap_stream == "index_group_side_stream")})
    if heads < padded:
        elements = rows * heads * 576
        zero = select_torch213_fill_launch({"element_count": rows * padded * 576}, "bf16",
            {**attributes, "source_input_layout": "contiguous_aligned_16"}, hardware)
        if rows == 1:
            # All outer size-one dimensions disappear in TensorIterator;
            # both live views are contiguous, distinct allocations.
            copy = PhysicalTransferConfig(algorithm="stream_copy", bytes=elements * 2,
                direction="d2d", link="dram")
        else:
            maximum_offset = 2 * ((rows - 1) * padded * 576 + heads * 576 - 1)
            copy = select_torch213_strided_copy_launch({"element_count": elements}, "bf16",
                {**attributes, "source_input_layout": "strided_nocast",
                 "source_copy_flags": "same_nonnegated_real_dtype",
                 "source_maximum_offset_bytes": maximum_offset}, hardware)
        padding = (("q_padding_fill", zero), ("q_padding_copy", copy))
    else:
        padding = ()
    core = select_dsa_flashmla_core_recipe(shape, dtype, attributes, hardware)
    return (("q_concat", concat), *remap_setup, ("index_remap", remap), *padding, *core)


def select_dsa_bf16_flashmla_attention_recipe(shape, dtype, attributes, hardware):
    """BF16 sparse wrapper from prepared concatenated Q and logical indices.

    vLLM0.30 _forward_bf16_kv uses sparse_prefill_fwd for both phases.
    The CMake-pinned SM90 phase1 kernel takes BF16 D576 KV directly;
    there is no cache quantization, decode scheduler, split or combine.
    Q already produced into the caller's padded arena needs no copy.
    The standalone tensor-Q branch uses new_empty plus a live-head copy,
    never a padding fill. Register/shared compiler facts remain unbound.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig
    from src.kernel.launch_config import TilingSpec
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import (
        select_torch213_fill_launch, select_torch213_strided_copy_launch)
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("BF16 sparse FlashMLA source requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_flashmla_snapshot"] != "vllm_0_30_cmake_pin"
            or attributes["source_component"] != "vllm_0_30_flashmla_sparse_bf16"
            or attributes["source_attention_backend"] != "flashmla_sparse"
            or attributes["source_kv_cache_layout"] != "paged_bf16_rows_576"
            or attributes["source_index_inputs_layout"] != "contiguous_int32"
            or attributes["source_metadata_layout"] != "contiguous_int32_request_ids_and_block_table"
            or attributes["source_attn_sink"] != "none"
            or attributes["source_extra_cache"] != "none"
            or attributes["source_hisparse"] is not False
            or attributes["source_parallel_mode"] != "pcp1_dcp1"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"):
        raise NotImplementedError("BF16 sparse FlashMLA requires its exact source, cache and CP1 metadata")
    rows, heads, dim, value, topk, page, stride = require_shape(shape, (
        "rows", "num_heads", "qk_head_dim", "v_head_dim", "topk", "block_size", "block_stride_rows"))
    padded = ceil_div(heads, 64) * 64
    if (padded not in (64, 128) or (dim, value) != (576, 512)
            or topk % 128 or topk >= 2**31 or rows * (padded // 64) >= 2**31
            or page % 16 or stride < page):
        raise NotImplementedError("BF16 sparse FlashMLA requires head576/value512, SM90 padded heads and native index width")
    layout = attributes["source_q_input_layout"]
    if layout == "caller_padded_head_backing":
        if attributes["source_q_backing_heads"] != padded:
            raise ValueError("BF16 sparse caller backing must cover the actual padded head count")
        padding = ()
    elif layout == "contiguous_unpadded":
        if heads == padded:
            padding = ()
        elif rows == 1:
            padding = (("q_padding_copy", PhysicalTransferConfig(algorithm="stream_copy",
                bytes=rows * heads * dim * 2, direction="d2d", link="dram")),)
        else:
            maximum_offset = 2 * ((rows - 1) * padded * dim + heads * dim - 1)
            padding = (("q_padding_copy", select_torch213_strided_copy_launch(
                {"element_count": rows * heads * dim}, "bf16", {**attributes,
                    "source_input_layout": "strided_nocast", "source_copy_flags": "same_nonnegated_real_dtype",
                    "source_maximum_offset_bytes": maximum_offset}, hardware)),)
    else:
        raise NotImplementedError("BF16 sparse Q requires its actual concatenated tensor/backing layout")
    state = attributes["source_index_remap_state"]
    if state == "build_current_indices":
        stream = attributes["source_index_remap_stream"]
        if stream not in ("caller_stream", "index_group_side_stream"):
            raise NotImplementedError("BF16 sparse index remap requires actual stream ownership")
        fill = select_torch213_fill_launch({"element_count": rows * topk}, "int32",
            {**attributes, "source_input_layout": "contiguous_aligned_16"}, hardware)
        single_tile = topk & (topk - 1) == 0
        if single_tile:
            counter_setup = ()
            remap_grid, remap_block = (rows, 1, 1), (256, 1, 1)
            algorithm = "vllm_0_30_sm90_sparse_index_remap_count_compact"
        else:
            zero = select_torch213_fill_launch({"element_count": rows}, "int32",
                {**attributes, "source_input_layout": "contiguous_aligned_16"}, hardware)
            counter_setup = (("valid_count_zero", zero),)
            remap_grid, remap_block = (rows, topk // 128, 1), (128, 1, 1)
            algorithm = "vllm_0_30_sm90_sparse_index_remap_count_compact_tiled"
        remap = LaunchConfig(algorithm, remap_grid, remap_block,
            parameters={"source_index_group_side_stream": float(stream == "index_group_side_stream")})
        setup = (("index_sentinel_fill", fill), *counter_setup, ("index_remap", remap))
    elif state == "provided_current_indices_and_counts":
        if (attributes["source_physical_indices_shape"] != (rows, topk)
                or attributes["source_valid_counts_shape"] != (rows,)
                or attributes["source_index_producer_state"] != "same_current_logical_indices_ready"):
            raise ValueError("BF16 sparse index reuse requires actual current-index producer and matching buffers")
        setup = ()
    else:
        raise NotImplementedError("BF16 sparse attention requires explicit current-index remap ownership")
    main = LaunchConfig("flashmla_vllm_0_30_sm90_v32_sparse_prefill_bf16_d576_topk_length",
        (rows * (padded // 64), 1, 1), (384, 1, 1), cluster=(1, 1, 1),
        tiling=TilingSpec({"heads": 64, "selected_kv": 64, "qk_head_dim": 576, "value_head_dim": 512}))
    return (*setup, *padding, ("attention_main", main))
