"""Prepared V4 sparse FlashMLA bodies from vLLM0.30's CMake pin.

The pin's SM90 decode/prefill config and host launchers own these rules.
Cache insertion,outer metadata and query padding are caller producers. Registers,
static shared and deployment binary identity remain unbound; source shared
layout must not be replaced with a different revision's observed value.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _source(dtype, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("V4 sparse FlashMLA requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_flashmla_snapshot"] != "vllm_0_30_cmake_pin"
            or attributes["source_component"] != "vllm_0_30_v4_sparse_attention"
            or attributes["source_parallel_mode"] != "pcp1_dcp1"
            or attributes["source_q_output_layout"] != "prepared_contiguous_bf16_padded_heads"
            or attributes["source_attn_sink"] != "contiguous_fp32_per_padded_head"
            or attributes["source_index_inputs_layout"] != "prepared_contiguous_int32_with_lengths"):
        raise NotImplementedError("V4 sparse attention requires its exact0.30 prepared source and CP1 layouts")


def _decode_shared_bytes():
    # config.h: dense64x512 Q; union(two K,BF16 O,FP32 OAccum
    # with row stride520); S aligned1024; validity, four float vectors,
    # seven8-byte barriers. Whole struct has1024-byte alignment.
    q = 64 * 512 * 2
    union = max(2 * q, q, (63 * 520 + 512) * 4)
    before_s = ceil_div(q + union, 1024) * 1024
    return ceil_div(before_s + 64 * 64 * 2 + 128 + 4 * 64 * 4 + 7 * 8, 1024) * 1024


def _prefill_shared_bytes():
    # config.h: Q/O union, two K and two S buffers; bool[2][64],
    # float2[32],float2[64],two float[64],ten8-byte barriers.
    # array_aligned and the aggregate have16-byte alignment.
    used = 3 * 64 * 512 * 2 + 2 * 64 * 64 * 2 + 128 + 32 * 8 + 64 * 8 + 2 * 64 * 4 + 10 * 8
    return ceil_div(used, 16) * 16


def _cache(attributes, key, page):
    record = attributes[key]
    row_stride, page_stride = record["row_stride_bytes"], record["page_stride_bytes"]
    if (record["layout"] != "fp8_v4_584_paged_data_scales" or type(row_stride) is not int or row_stride != 584
            or type(page_stride) is not int or not page * 584 <= page_stride < 2**31):
        raise NotImplementedError("V4 packed cache requires actual584-byte rows and valid independent page strides")


def select_v4_flashmla_decode_core_recipe(shape, dtype, attributes, hardware):
    """Full FlashMLA call for already prepared V4 decode inputs.

    Each logical query is a separate kernel batch row,with s_q=1. All
    SM90 calls include combine,even when each request has one split. A
    provided scheduler belongs to this current context and layer type;
    absent scheduler means a real planner submission,not a fake fill.
    """
    _source(dtype, attributes, hardware)
    rows, heads, topk, page = require_shape(shape, ("rows", "num_heads", "swa_topk", "swa_page_tokens"))
    if heads not in (64, 128) or topk % 64 or topk >= 2**31 or rows * heads * 512 >= 2**31:
        raise NotImplementedError("V4 decode requires actual64/128 padded heads and64-aligned SWA indices")
    if attributes["source_sequence_layout"] != "flattened_query_batch_sq1":
        raise NotImplementedError("V4 decode requires the source flattened-query batch layout")
    _cache(attributes, "source_swa_cache", page)
    layer = attributes["source_layer_type"]
    if layer == "swaonly":
        if attributes["source_extra_cache"] != "none":
            raise ValueError("V4 SWA-only call cannot carry compressed cache")
    elif layer in ("c4", "c128"):
        extra_topk, extra_page = require_shape(shape, ("extra_topk", "extra_page_tokens"))
        if extra_topk >= 2**31:
            raise NotImplementedError("V4 extra index width exceeds source INT32 metadata")
        _cache(attributes, "source_extra_cache", extra_page)
    else:
        raise NotImplementedError("V4 decode binds only SWA-only,C4 andC128 source layer types")
    parts = max(int(hardware.num_compute_units) // (heads // 64), 1)
    scheduler = attributes["source_scheduler_state"]
    if scheduler == "build_current_context":
        shared = 4 * (rows * 5 + 1)
        if shared > 232448:
            raise NotImplementedError("V4 scheduler exceeds SM90 opt-in shared-memory capacity")
        setup = (("schedule_metadata", LaunchConfig("flashmla_vllm_0_30_sm90_sparse_metadata",
            (1, 1, 1), (32, 1, 1), parameters={"dynamic_shared_bytes": float(shared)})),)
    elif scheduler == "provided_current_context":
        meta, splits = attributes["source_tile_scheduler_shape"], attributes["source_num_splits_shape"]
        if (not isinstance(meta, (tuple, list)) or not isinstance(splits, (tuple, list))
                or any(type(value) is not int for value in (*meta, *splits))
                or tuple(meta) != (parts, 8) or tuple(splits) != (rows + 1,)):
            raise ValueError("V4 scheduler tensors must match actual flattened batch andSM90 partitions")
        setup = ()
    else:
        raise NotImplementedError("V4 requires actual current-context scheduler ownership")
    main = LaunchConfig(f"flashmla_vllm_0_30_sm90_v4_sparse_fp8_h{heads}",
        (heads // 64, 1, parts), (384, 1, 1), cluster=(heads // 64, 1, 1),
        parameters={"dynamic_shared_bytes": float(_decode_shared_bytes()),
                    "source_programmatic_stream_serialization": 0.0})
    combine = LaunchConfig(f"flashmla_vllm_0_30_sm90_v4_sparse_combine_s{ceil_div(parts, 32) * 32}",
        (rows, 1, heads // 8), (256, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0, "source_programmatic_stream_serialization": 1.0})
    return (*setup, ("attention_main", main), ("attention_combine", combine))


def select_v4_prepared_decode_recipe(shape, dtype, attributes, hardware):
    """Complete layer decode body after Q/cache/SWA metadata producers.

    C4's local-to-global/count helper executes every layer. C128 indices
    are already prepared by metadata; neither they nor SWA are rebuilt.
    Query unsqueeze,cache unsqueeze and output view submit no work.
    """
    core = select_v4_flashmla_decode_core_recipe(shape, dtype, attributes, hardware)
    if attributes["source_layer_type"] == "c4":
        rows, = require_shape(shape, ("rows",))
        remap = LaunchConfig("vllm_0_30_sm90_v4_global_topk_indices_and_lens_triton",
            (rows, 1, 1), (128, 1, 1))
        return (("compressed_index_remap", remap), *core)
    return core


def select_v4_flashmla_prefill_core_recipe(shape, dtype, attributes, hardware):
    """One source sparse BF16 prefill call after complete gather/index body.

    No decode scheduler or combine is submitted by this API. This is a
    prepared core,not a claim that cache gathers and index producers have
    been covered by one attention kernel.
    """
    _source(dtype, attributes, hardware)
    rows, heads, topk = require_shape(shape, ("rows", "num_heads", "topk"))
    if heads not in (64, 128) or topk % 128 or topk >= 2**31 or rows * heads * 512 >= 2**31:
        raise NotImplementedError("V4 prefill requires64/128 padded heads and128-aligned combined indices")
    if attributes["source_kv_input_layout"] != "prepared_contiguous_bf16_rows_1_512":
        raise NotImplementedError("V4 prefill consumes actual gathered BF16 K,not packed decode cache")
    return (("attention_main", LaunchConfig("flashmla_vllm_0_30_sm90_v4_sparse_prefill_topk_length",
        (rows * (heads // 64), 1, 1), (384, 1, 1), cluster=(1, 1, 1),
        parameters={"dynamic_shared_bytes": float(_prefill_shared_bytes())})),)


def select_v4_prepared_prefill_recipe(shape, dtype, attributes, hardware):
    """Complete pure-text prefill after Q,cache and actual chunk metadata.

    Each original chunk gathers compressed K (including its INT32 floor
    divide),gathers SWA,clears shared indices,combines indices,and attends.
    Workspace M/N and request boundaries come from actual CPU metadata;
    they are allocation extents,not inferred valid traffic. Div geometry
    describes a physical submission only;no typed integer-div price exists.
    """
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_fill_launch

    _source(dtype, attributes, hardware)
    rows, heads, batch, window, swa_page = require_shape(shape,
        ("rows", "num_heads", "batch", "window", "swa_page_tokens"))
    if (attributes["source_kv_input_layout"] != "prepared_contiguous_bf16_rows_1_512"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_image_visibility"] != "none"):
        raise NotImplementedError("V4 prefill binds pure-text prepared workspace andTorch2.13 aligned metadata")
    cutedsl = attributes["source_has_cutedsl"]
    if type(cutedsl) is not bool:
        raise TypeError("V4 gather requires the actual has_cutedsl boolean")
    _cache(attributes, "source_swa_cache", swa_page)
    layer = attributes["source_layer_type"]
    if layer == "swaonly":
        ratio, topk = 1, 0
        if attributes["source_extra_cache"] != "none":
            raise ValueError("V4 SWA-only prefill cannot carry compressed cache")
    elif layer in ("c4", "c128"):
        ratio = 4 if layer == "c4" else 128
        topk, extra_page = require_shape(shape, ("extra_topk", "extra_page_tokens"))
        _cache(attributes, "source_extra_cache", extra_page)
    else:
        raise NotImplementedError("V4 prefill binds only SWA-only,C4 andC128 source layer types")
    combined = ceil_div(topk + window, 128) * 128
    starts = attributes["source_query_start_loc"]
    seqs, gathers = attributes["source_seq_lens"], attributes["source_gather_lens"]
    if (not all(isinstance(values, (tuple, list)) for values in (starts, seqs, gathers))
            or len(starts) != batch + 1 or len(seqs) != batch or len(gathers) != batch
            or any(type(value) is not int for value in (*starts, *seqs, *gathers))
            or starts[0] != 0 or starts[-1] != rows
            or any(not 0 < starts[i+1] - starts[i] <= seqs[i] < 2**31 for i in range(batch))
            or any(not 0 < gathers[i] <= seqs[i] for i in range(batch))):
        raise ValueError("V4 prefill needs actual request query offsets,seq lengths andgather lengths")
    chunks = attributes["source_prefill_chunks"]
    if not isinstance(chunks, (tuple, list)) or not chunks:
        raise ValueError("V4 prefill needs its actual nonempty CPU chunk plan")
    stages, cursor = [], 0
    for chunk in chunks:
        start, end = chunk["request_start"], chunk["request_end"]
        m, n = chunk["workspace_rows"], chunk["compressed_region_rows"]
        if (any(type(value) is not int for value in (start, end, m, n))
                or start != cursor or not start < end <= batch or not 0 <= n < m
                or (layer == "swaonly" and n != 0)
                or any(gathers[i] > m - n or (layer != "swaonly" and seqs[i] // ratio > n)
                       for i in range(start, end))):
            raise ValueError("V4 chunks must partition requests andcover actual compressed/SWA view extents")
        cursor = end
        reqs, tokens = end - start, starts[end] - starts[start]
        if reqs * m * 512 >= 2**31:
            raise NotImplementedError("V4 gathered workspace exceeds the source INT32 stride domain")
        def gather(role, offset):
            if cutedsl:
                workers = max(1, min(ceil_div(m - offset, 4), ceil_div(8192, reqs)))
                algorithm = "vllm_0_30_sm90_v4_dequant_gather_cutedsl"
            else:
                workers = 128
                algorithm = "vllm_0_30_sm90_v4_dequant_gather_triton"
            stages.append((role, LaunchConfig(algorithm, (reqs, workers, 1), (128, 1, 1))))
        if layer != "swaonly":
            # Scalar INT32 BUnaryFunctor,with one tensor input,has8-byte
            # IO and8 values/thread in Torch2.13 CUDALoops;vector width4.
            # Slice offsets can reduce the full metadata's16B alignment;
            # vector width1 selects the separate four-values/thread path.
            vector = 4 if start % 4 == 0 else 2 if start % 2 == 0 else 1
            values = 4 if vector == 1 else 8
            stages.append(("compressed_seq_floor_divide", LaunchConfig(
                f"pytorch_2_13_sm90_int32_floor_divide_scalar_v{vector}",
                (ceil_div(reqs, 128 * values), 1, 1), (128, 1, 1),
                parameters={"dynamic_shared_bytes": 0.0})))
            gather("compressed_cache_gather", 0)
        gather("swa_cache_gather", n)
        stages.append(("combined_index_sentinel_fill", select_torch213_fill_launch(
            {"element_count": tokens * combined}, "int32", attributes, hardware)))
        stages.append(("combine_topk_swa_indices", LaunchConfig(
            "vllm_0_30_sm90_v4_combine_topk_swa_indices_triton", (reqs, 256, 1), (128, 1, 1))))
        stages.extend(select_v4_flashmla_prefill_core_recipe(
            {"rows": tokens, "num_heads": heads, "topk": combined}, dtype, attributes, hardware))
    if cursor != batch:
        raise ValueError("V4 chunk plan does not cover every actual prefill request")
    return tuple(stages)
