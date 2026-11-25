"""Prepared vLLM0.30 V4 indexer programs on H200,without duration inputs.

Attention.DeepseekV4Indexer owns short-context dispatch and Q preparation;
layers.sparse_attn_indexer owns clear,prefill chunks and decode scoring.
DeepGEMM/TopK host algorithms are shared with DSA through the H200 source
owner,not a different hardware package. Projection GEMMs and compressor
are separate producers,which must join before this prepared scoring body.
Compiler resources,CP/FP4 and padded decode packing are not approximated.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape
from src.kernel.launch_config.nvidia_h200.dsa_launch import (
    select_dsa_decode_indexer_recipe, select_dsa_prefill_indexer_chunk_recipe)
from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_fill_launch


def _source(attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("V4 indexer source requires H200 HWSpec")
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_deepseek_v4_indexer"):
        raise NotImplementedError("V4 indexer requires its exact0.30 source identity")


def select_v4_indexer_q_launch(shape, dtype, attributes, hardware):
    """Complete fused GPT-J RoPE/quant/FP32 weight-fold,after Q projection.

    CuTeDSL is selected by the actual has_cutedsl result,never shape or an
    error-driven fallback. Its coarsening boundary is512 tokens,not heads.
    Both paths produce FP8 Q with its scale folded into FP32 weights; no
    separate Q-scale tensor or pre/post cast kernel is invented.
    """
    _source(attributes, hardware)
    require_dtype(dtype, ("bf16",))
    rows, heads, dim, rope = require_shape(shape, ("rows", "num_heads", "head_dim", "rope_dim"))
    if (heads, dim, rope) != (64, 128, 64):
        raise NotImplementedError("V4 indexer Q binds the replicated64-head128/GPT-J64 source domain")
    if (attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_position_dtype"] != "int64"
            or attributes["source_rope_cache_dtype"] != "fp32"
            or attributes["source_weights_dtype"] != "bf16"
            or attributes["source_quantization"] != "fp8_e4m3fn"):
        raise NotImplementedError("V4 indexer Q requires actual contiguousBF16 Q/weights,INT64 positions andFP32 rotary cache")
    available = attributes["source_has_cutedsl"]
    if type(available) is not bool:
        raise TypeError("V4 source_has_cutedsl must be the actual runtime boolean")
    if available:
        coarsen = 1 if rows < 512 else 4
        threads_per_token = heads // coarsen * (dim // 16)
        return LaunchConfig(f"vllm_0_30_sm90_v4_indexer_q_fp8_cutedsl_coarsen{coarsen}",
            (ceil_div(rows * threads_per_token, 128), 1, 1), (128, 1, 1))
    return LaunchConfig("vllm_0_30_sm90_v4_indexer_q_fp8_triton", (rows, heads, 1), (32, 1, 1))


def select_v4_short_context_indexer_recipe(shape, dtype, attributes, hardware):
    """The complete candidate-fill body,with compressor already joined.

    Metadata max_seq_len and stream capture are actual caller inputs. This
    route skips Q projection/quant and scoring. Existing shared topk storage
    is fully overwritten including-1 tails; no new copy/clear is submitted.
    """
    _source(attributes, hardware)
    require_dtype(dtype, ("int32",))
    rows, topk, ratio, max_seq = require_shape(shape, ("rows", "topk", "compress_ratio", "max_seq_len"))
    if ratio != 4 or topk not in (512, 1024):
        raise NotImplementedError("V4 short-context source binds C4 Flash/Pro top-k domains")
    capture = attributes["source_stream_capturing"]
    if type(capture) is not bool:
        raise TypeError("V4 stream capture state must be an actual boolean")
    if capture or max_seq // ratio > topk:
        raise NotImplementedError("V4 source selects scoring rather than the short-context fill")
    if (attributes["source_positions_dtype"] != "int64"
            or attributes["source_topk_buffer_layout"] != "shared_contiguous_int32"
            or attributes["source_compressor_state"] != "cache_ready"):
        raise NotImplementedError("V4 short-context fill requires its real sharedbuffer,positions andcache producer")
    return (("short_context_fill", LaunchConfig("vllm_dsv4_short_context_topk_triton",
        (rows, 1, 1), (256, 1, 1))),)


def _count(mapping, key):
    value = mapping[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"V4 indexer {key} must be a non-negative integer")
    return value


def select_v4_prepared_indexer_recipe(shape, dtype, attributes, hardware):
    """Complete mixed/prefill/decode scorer after Q quant and cache update.

    CPU chunk records come from actual metadata,not a shape-chosen planner.
    Each chunk carries its maximum K view extent,actual local row count and
    skip-gather proof. The latter controls the empty branch; it cannot be
    replaced by the allocated view extent. Empty local K chunks still
    submit their explicit redundant fill. Source decode schedule is already
    prepared by metadata ownership. Native contiguous next_n1/2/4 only.
    """
    _source(attributes, hardware)
    require_dtype(dtype, ("fp8",))
    rows, heads, dim, topk, ratio = require_shape(shape, ("rows", "num_heads", "head_dim", "topk", "compress_ratio"))
    if (heads, dim, ratio) != (64, 128, 4) or topk not in (512, 1024):
        raise NotImplementedError("V4 prepared scorer binds replicated64-head128 C4 Flash/Pro FP8 domains")
    if (attributes["source_indexer_input_stage"] != "quantized_q_weights_and_cache_ready"
            or attributes["source_parallel_mode"] != "pcp1_dcp1"
            or attributes["source_topk_buffer_layout"] != "shared_contiguous_int32"
            or attributes["source_dense_mha_metadata_layer"] != "none"
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"):
        raise NotImplementedError("V4 scorer requires actual prepared FP8/cache/metadata andTorch2.13 sharedoutput")
    decode_rows, prefill_rows = (_count(shape, field) for field in ("num_decode_tokens", "num_prefill_tokens"))
    if decode_rows + prefill_rows != rows:
        raise NotImplementedError("V4 scorer currently requires native rows without extra hidden-state graph padding")
    chunks = attributes["source_prefill_chunks"]
    if not isinstance(chunks, (tuple, list)):
        raise TypeError("V4 source prefill chunks must be explicit metadata records")
    fill_attrs = {"tensor_iterator_source": attributes["tensor_iterator_source"],
                  "source_input_layout": "contiguous_aligned_16"}
    def fill(count):
        return select_torch213_fill_launch({"element_count": count * topk}, "int32", fill_attrs, hardware)
    clear_fields = {"source_topk_clear_rows", "source_topk_clear_state", "source_topk_phase_token_start"}
    present = clear_fields & set(attributes)
    if present and present != clear_fields:
        raise ValueError("V4 shared clear requires its complete invocation binding")
    if present:
        clear_rows = _count(attributes, "source_topk_clear_rows")
        token_start = _count(attributes, "source_topk_phase_token_start")
        if clear_rows <= rows:
            raise ValueError("V4 shared clear must cover both actual phase slices")
        clear_state = attributes["source_topk_clear_state"]
        if clear_state == "owned_current_invocation":
            if decode_rows or not prefill_rows or token_start != clear_rows - rows:
                raise ValueError("V4 shared clear belongs to the first prefill phase")
            stages = [("clear_topk", fill(clear_rows))]
        elif clear_state == "provided_current_invocation":
            if prefill_rows or not decode_rows or token_start != 0:
                raise ValueError("V4 shared clear consumer must be the following decode phase")
            stages = []
        else:
            raise ValueError("V4 shared clear state must bind this exact invocation")
    else:
        token_start = 0
        stages = [("clear_topk", fill(rows))]
    cursor = token_start + decode_rows
    for chunk in chunks:
        start, end, kv_rows = (_count(chunk, field) for field in ("token_start", "token_end", "kv_rows"))
        local_rows = _count(chunk, "local_total_seq_lens")
        if local_rows > kv_rows:
            raise ValueError("V4 local K rows exceed the actual maximum K view extent")
        if start != cursor or not start < end <= token_start + rows:
            raise ValueError("V4 source chunks must partition the actual contiguous prefill suffix")
        cursor = end
        if local_rows == 0:
            stages.append(("empty_chunk_fill", fill(end - start)))
            continue
        chunk_shape = {"rows": end - start, "kv_rows": kv_rows, "num_heads": heads, "head_dim": dim, "topk": topk}
        chunk_attrs = {**attributes, "source_schedule_metadata": "none", "source_kv_scale_dtype": "fp32",
            "source_skip_kv_gather": chunk["skip_kv_gather"], "source_gather_input_state": chunk["gather_input_state"],
            "quant_block_size": 128}
        stages.extend(select_dsa_prefill_indexer_chunk_recipe(chunk_shape, dtype, chunk_attrs, hardware))
    if cursor != token_start + rows:
        raise ValueError("V4 source chunks do not cover all actual prefill rows")
    if decode_rows:
        if (attributes["source_decode_layout"] != "native_contiguous"
                or attributes["source_schedule_metadata"] != "provided_current_context"):
            raise NotImplementedError("V4 padded decode pack/unpack requires its complete source program")
        batch, next_n = require_shape(shape, ("batch", "next_n"))
        if batch * next_n != decode_rows:
            raise ValueError("V4 native decode rows disagree with actual batch andnext_n")
        stages.extend(select_dsa_decode_indexer_recipe(shape, dtype, attributes, hardware))
    return tuple(stages)
