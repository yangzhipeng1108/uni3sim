"""DSV4 kernel-evaluation case construction: report axes and the
V4-attention / WQ fused-MoE case-spec builders."""

from __future__ import annotations

from typing import Mapping

from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config

from evaluation.json_shape import require_positive_metric as _positive_int

from evaluation.kernel.observation_readers import (
    _non_empty_string,
)


_DSV4_CASE_HARDWARE_ID = "nvidia_h200"
_DSV4_LAYER_COMPRESS_RATIOS = {"swa": 1, "c4": 4, "c128": 128}
# fp8_ds_mla V4 layout: 448 FP8 NoPE + 64 BF16 RoPE + 8 UE8M0 scales per
# compressed state; the SWA cache holds a K and a V row per token.
_DSV4_COMPRESSED_STATE_BYTES = 584
_DSV4_SWA_TOKEN_BYTES = 584
_DSV4_HEAD_DIM = 512
_DSV4_QK_HEAD_DIM = 512  # V4 head_dim includes its 64 RoPE coordinates.
_DSV4_KV_BLOCK_SIZE = 256
_DSV4_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")


def _require_dsv4_case_hardware(hardware_id: str) -> None:
    if hardware_id != _DSV4_CASE_HARDWARE_ID:
        raise NotImplementedError(
            "DeepSeek-V4 campaign case normalization is bound to the H200 "
            f"kcampaign evidence, got {hardware_id!r}")


def _dsv4_case_record(op_attributes: Mapping, context: str):
    if not isinstance(op_attributes, Mapping):
        raise TypeError(f"{context} Attr must be a mapping")
    if set(op_attributes) != {"runner_args", "launch_evidence"}:
        raise ValueError(
            f"{context} Attr fields are invalid: expected "
            f"['launch_evidence', 'runner_args'], "
            f"got={sorted(op_attributes)}")
    runner_args = op_attributes["runner_args"]
    evidence = op_attributes["launch_evidence"]
    if not isinstance(runner_args, Mapping):
        raise TypeError(f"{context} Attr.runner_args must be a mapping")
    if not isinstance(evidence, Mapping):
        raise TypeError(f"{context} Attr.launch_evidence must be a mapping")
    return runner_args, evidence


def _require_non_negative_int(mapping: Mapping, name: str, context: str) -> int:
    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.{name} must be an integer")
    if value < 0:
        raise ValueError(f"{context}.{name} must be non-negative, got {value}")
    return value


def _require_int_sequence(value, name: str, context: str) -> tuple:
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{context}.{name} must be a non-empty sequence")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int)
           for item in result):
        raise TypeError(f"{context}.{name} must contain integers")
    return result


def _require_evidence_int(evidence: Mapping, name: str, expected: int,
                          context: str) -> None:
    value = evidence[name]
    if value != expected:
        raise ValueError(
            f"{context} launch evidence {name}={value!r} disagrees with "
            f"the runner arguments ({expected!r})")


def _require_c128a_topk_width(evidence: Mapping, context: str) -> int:
    width = evidence["c128a_topk_width"]
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError(f"{context}.c128a_topk_width must be an integer")
    if width <= 0 or width % 128:
        raise ValueError(
            f"{context} evidence c128a_topk_width {width} is not a positive "
            "multiple of the 128-wide C128A topk buffer alignment")
    return width


def _observed_kernel_names(kernel_records) -> list:
    names = []
    for ordinal, record in enumerate(kernel_records):
        if not isinstance(record, Mapping):
            raise TypeError(f"campaign kernel record {ordinal} must be a mapping")
        metadata = record["Metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"campaign kernel record {ordinal} Metadata must be a mapping")
        names.append(_non_empty_string(
            metadata, "Kernel Name", f"campaign kernel record {ordinal}"))
    return names


def _require_observed_kernel(kernel_records, fragment: str, context: str) -> None:
    if not any(fragment in name
               for name in _observed_kernel_names(kernel_records)):
        raise ValueError(
            f"{context} observed kernels contain no {fragment!r} launch; "
            "the case did not measure the operator the spec would model")


def _selected_spec(kind: KernelKind, hardware_id: str, shape: Mapping,
                   dtype: str, variant: str, attributes: Mapping,
                   hardware) -> KernelSpec:
    launch = select_launch_config(
        hardware_id, kind, shape, dtype, variant, attributes, hardware)
    return KernelSpec(
        kind=kind, hardware_id=hardware_id, shape=shape, dtype=dtype,
        variant=variant, launch=launch, attributes=attributes)


def dsv4_case_report_axes(op_type: str, op_attributes: Mapping) -> Mapping:
    """Return the report grouping axes of one DeepSeek-V4 campaign case.

    Axes are derived from the runner arguments only (model semantics), so a
    case whose full mapping later fails still groups explicitly.
    """

    runner_args, _ = _dsv4_case_record(op_attributes, f"{op_type} Attr")
    if op_type == "V4ATTN":
        stage = _non_empty_string(runner_args, "op", "V4ATTN runner_args")
        if stage in ("decode", "prefill", "compressor"):
            layer_type = _non_empty_string(
                runner_args, "layer-type", "V4ATTN runner_args")
            if layer_type not in _DSV4_LAYER_COMPRESS_RATIOS:
                raise ValueError(
                    f"V4ATTN layer_type {layer_type!r} is unknown")
            if stage == "compressor":
                # Compressor runner arguments carry no heads/hidden fields,
                # and both models run C4 and C128 layers, so the case cannot
                # be attributed to one model from its own record.
                model = "unattributed"
            else:
                heads = _positive_int(runner_args, "padded-heads", "V4ATTN")
                if heads == 64:
                    model = "deepseek-v4-flash"
                elif heads == 128:
                    model = "deepseek-v4-pro"
                else:
                    raise ValueError(
                        f"V4ATTN padded_heads {heads} is not a DeepSeek-V4 "
                        "model")
        elif stage in ("mhc_pre", "mhc_post_pre"):
            hidden = _positive_int(runner_args, "hidden-size", "V4ATTN")
            if hidden == 4096:
                model = "deepseek-v4-flash"
            elif hidden == 7168:
                model = "deepseek-v4-pro"
            else:
                raise ValueError(
                    f"V4ATTN hidden_size {hidden} is not a DeepSeek-V4 model")
            layer_type = None
        else:
            raise ValueError(f"V4ATTN stage {stage!r} is unknown")
        return {
            "dsv4_stage": stage,
            "dsv4_model": model,
            "dsv4_layer_type": layer_type,
        }
    if op_type == "WQ_FUSEDMOE":
        hidden = _positive_int(runner_args, "hidden_size", "WQ_FUSEDMOE")
        if hidden == 4096:
            model = "deepseek-v4-flash"
        elif hidden == 7168:
            model = "deepseek-v4-pro"
        else:
            raise ValueError(
                f"WQ_FUSEDMOE hidden_size {hidden} is not a DeepSeek-V4 model")
        return {
            "dsv4_stage": "routed_experts",
            "dsv4_model": model,
            "dsv4_layer_type": None,
        }
    raise ValueError(f"no DeepSeek-V4 campaign op type {op_type!r}")


def _v4attn_attention_spec(runner_args: Mapping, evidence: Mapping,
                           kernel_records, hardware_id: str,
                           hardware) -> KernelSpec:
    context = "V4ATTN attention case"
    stage = _non_empty_string(runner_args, "op", context)
    if stage not in ("decode", "prefill"):
        raise AssertionError(f"unhandled V4ATTN attention stage {stage!r}")
    layer_type = _non_empty_string(runner_args, "layer-type", context)
    if layer_type not in _DSV4_LAYER_COMPRESS_RATIOS:
        raise ValueError(f"{context} layer_type {layer_type!r} is unknown")
    ratio = _DSV4_LAYER_COMPRESS_RATIOS[layer_type]
    batch = _positive_int(runner_args, "batch", context)
    q_len = _positive_int(runner_args, "q-len", context)
    prefix = _require_non_negative_int(runner_args, "context", context)
    heads = _positive_int(runner_args, "padded-heads", context)
    local_heads = _positive_int(runner_args, "local-heads", context)
    if local_heads > heads:
        raise ValueError(f"{context} local_heads exceeds padded_heads")
    index_topk = _positive_int(runner_args, "index-topk", context)
    window = _positive_int(runner_args, "window", context)
    if _positive_int(runner_args, "head-dim", context) != _DSV4_HEAD_DIM:
        raise ValueError(f"{context} head_dim must be {_DSV4_HEAD_DIM}")
    if _positive_int(runner_args, "kv-block-size", context) != (
            _DSV4_KV_BLOCK_SIZE):
        raise ValueError(
            f"{context} kv_block_size must be {_DSV4_KV_BLOCK_SIZE}")
    if stage == "decode":
        if prefix <= 0:
            raise ValueError(f"{context} decode context must be positive")
        kv_len = prefix
        _require_evidence_int(evidence, "op", stage, context)
    else:
        # The prefill runner executes one request per chunk.
        if batch != 1:
            raise ValueError(f"{context} prefill batch must be one")
        kv_len = prefix + q_len
        _require_evidence_int(evidence, "op", stage, context)
    q_lens = (q_len,) * batch
    kv_lens = (kv_len,) * batch
    if ratio == 128:
        # C128 layers keep one 128-token full-resolution window alongside
        # the compressed cache (deepseek_v4 sparse_mla.py c128a path).
        if window != 128:
            raise ValueError(
                f"{context} C128 window must equal the 128-token "
                "compression window")
        swa_kv_lens = tuple(min(128, length) for length in kv_lens)
    else:
        swa_kv_lens = tuple(min(window, length) for length in kv_lens)
    if ratio == 1:
        capacity = index_topk
        selected = (0,) * batch
    elif ratio == 4:
        # Only complete C4 windows are addressable. The prefill combine
        # helper caps each row by (pos+1)//4, including scored selection.
        capacity = index_topk
        selected = tuple(
            min(index_topk, length // ratio) for length in kv_lens)
    else:
        # C128 attends the FULL compressed range: floor(kv_len / 128)
        # states per request (sparse_mla.py:404 num_compressed), with no
        # index_topk cap (flashmla.py:193-207 c128a metadata path).  The
        # capacity is the 128-aligned C128A topk buffer width
        # (sparse_mla.py:39 _C128A_TOPK_ALIGNMENT, :155-170
        # c128a_max_compressed), not the model index_topk.  kv_len < 128
        # yields an all-zero selection, which is legal upstream.
        floors = tuple(length // ratio for length in kv_lens)
        if stage == "decode":
            capacity = _require_c128a_topk_width(evidence, context)
        else:
            capacity = max(128, -(-max(floors) // 128) * 128)
        selected = tuple(min(capacity, length) for length in floors)
    # Launch-evidence cross-checks (selection/launch facts, never durations).
    _require_evidence_int(evidence, "batch", batch, context)
    _require_evidence_int(evidence, "q_len", q_len, context)
    _require_evidence_int(evidence, "context", prefix, context)
    _require_evidence_int(evidence, "layer_type", layer_type, context)
    _require_evidence_int(evidence, "window", window, context)
    if _require_int_sequence(evidence["q_lens"], "q_lens", context) != q_lens:
        raise ValueError(f"{context} evidence q_lens disagree with the case")
    if stage == "decode":
        if _require_int_sequence(
                evidence["seq_lens"], "seq_lens", context) != kv_lens:
            raise ValueError(
                f"{context} evidence seq_lens disagree with the case")
        swa_length = _require_int_sequence(
            evidence["swa_topk_length"], "swa_topk_length", context)
        expected_swa = tuple(
            length for length in swa_kv_lens for _ in range(q_len))
        if swa_length != expected_swa:
            raise ValueError(
                f"{context} evidence swa_topk_length disagrees with "
                "min(window, kv_len) per token")
        if ratio > 1:
            _require_evidence_int(evidence, "compress_ratio", ratio, context)
            extra = _require_int_sequence(
                evidence["extra_topk_length"], "extra_topk_length", context)
            expected_extra = tuple(
                length for length in selected for _ in range(q_len))
            if extra != expected_extra:
                raise ValueError(
                    f"{context} evidence extra_topk_length disagrees with "
                    "the per-token selected compressed state count")
        if ratio == 4:
            indexer = evidence["indexer"]
            if not isinstance(indexer, Mapping):
                raise TypeError(f"{context} evidence indexer must be a mapping")
            _require_evidence_int(
                indexer, "index_topk", index_topk, context)
        # The FlashMLA tile-scheduler plan is a discrete launch fact: the
        # cumulative num_splits array yields the per-request split-KV count
        # the combine traffic depends on (never a duration).
        sched_meta = evidence["flash_mla_sched_meta"]
        if not isinstance(sched_meta, Mapping):
            raise TypeError(
                f"{context} evidence flash_mla_sched_meta must be a mapping")
        cumulative = _require_int_sequence(
            sched_meta["num_splits"], "num_splits", context)
        if len(cumulative) != batch + 1 or cumulative[0] != 0:
            raise ValueError(
                f"{context} evidence num_splits must be the cumulative "
                "per-request split plan")
        split_counts = tuple(
            cumulative[index + 1] - cumulative[index]
            for index in range(batch))
        if any(count <= 0 for count in split_counts):
            raise ValueError(
                f"{context} evidence num_splits must be non-decreasing "
                "with at least one split per request")
    else:
        if _require_int_sequence(
                evidence["seq_lens"], "seq_lens", context) != kv_lens:
            raise ValueError(
                f"{context} evidence seq_lens disagree with the case")
        if _require_int_sequence(
                evidence["prefix_lens"], "prefix_lens", context) != (prefix,):
            raise ValueError(
                f"{context} evidence prefix_lens disagree with the case")
        _require_evidence_int(evidence, "compress_ratio", ratio, context)
    _require_observed_kernel(
        kernel_records,
        "flash_fwd_splitkv_mla_fp8_sparse" if stage == "decode"
        else "sparse_attn_fwd",
        context)
    shape = {
        "batch": batch,
        "q_len": q_len,
        "kv_len": kv_len,
        "num_query_heads": heads,
        "qk_head_dim": _DSV4_QK_HEAD_DIM,
        "v_head_dim": _DSV4_HEAD_DIM,
        "topk": max(selected),
        "window_size": window,
        "compress_ratio": ratio,
        "compressed_state_bytes": _DSV4_COMPRESSED_STATE_BYTES,
        "swa_token_bytes": _DSV4_SWA_TOKEN_BYTES,
        "dtype_bytes": 2,
    }
    attributes = {
        "causal": True,
        "indexer_type": "shared",
        "kv_layout": "mla",
        "kv_dist": "uniform",
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "selected_kv_lens": selected,
        "swa_kv_lens": swa_kv_lens,
        "topk_capacity": capacity,
        "topk_index_dtype": "int32",
    }
    if stage == "decode":
        attributes["split_counts"] = split_counts
    return _selected_spec(
        KernelKind.SPARSE_ATTN, hardware_id, shape, "bf16",
        f"dsv4_sparse_{stage}", attributes, hardware)


def _v4attn_compressor_spec(runner_args: Mapping, evidence: Mapping,
                            kernel_records, hardware_id: str,
                            hardware) -> tuple[KernelSpec, ...]:
    context = "V4ATTN compressor case"
    layer_type = _non_empty_string(runner_args, "layer-type", context)
    if layer_type not in _DSV4_LAYER_COMPRESS_RATIOS:
        raise ValueError(f"{context} layer_type {layer_type!r} is unknown")
    ratio = _DSV4_LAYER_COMPRESS_RATIOS[layer_type]
    if ratio == 1:
        raise ValueError(f"{context} SWA-only layers have no compressor")
    batch = _positive_int(runner_args, "batch", context)
    q_len = _positive_int(runner_args, "q-len", context)
    _require_non_negative_int(runner_args, "context", context)
    if q_len == 1:
        # Decode step: one new token per request at staggered positions.
        new_tokens = batch
    else:
        if batch != 1:
            raise ValueError(f"{context} prefill batch must be one")
        new_tokens = q_len
    _require_evidence_int(evidence, "op", "compressor", context)
    _require_evidence_int(evidence, "compress_ratio", ratio, context)
    _require_evidence_int(evidence, "batch", batch, context)
    evidence_q_lens = _require_int_sequence(
        evidence["q_lens"], "q_lens", context)
    if sum(evidence_q_lens) != new_tokens:
        raise ValueError(
            f"{context} evidence q_lens sum {sum(evidence_q_lens)} disagrees "
            f"with the {new_tokens} new tokens")
    # Per-request cache state boundaries: old_len is the cached context,
    # new_len the length after this step's tokens (deepseek_v4/compressor.py
    # DeepseekCompressor).  A compressed state is finished only when a
    # request crosses a compress_ratio window boundary, so the append
    # traffic is floor(new/ratio) - floor(old/ratio) states per request;
    # a partial window stays in the FP32 partial-state cache.
    old_lens = _require_int_sequence(
        evidence["prefix_lens"], "prefix_lens", context)
    new_lens = _require_int_sequence(
        evidence["seq_lens"], "seq_lens", context)
    if len(old_lens) != len(evidence_q_lens) or len(new_lens) != len(old_lens):
        raise ValueError(
            f"{context} evidence prefix_lens/seq_lens must cover one entry "
            "per request")
    deltas = tuple(new - old for old, new in zip(old_lens, new_lens))
    if any(delta <= 0 for delta in deltas):
        raise ValueError(
            f"{context} evidence seq_lens must exceed prefix_lens on every "
            "request")
    if deltas != evidence_q_lens:
        raise ValueError(
            f"{context} evidence prefix_lens/seq_lens deltas disagree with "
            "q_lens per request")
    boundary_states = sum(
        new // ratio - old // ratio for old, new in zip(old_lens, new_lens))
    evidence_boundary = evidence["boundary_tokens"]
    if isinstance(evidence_boundary, bool) or not isinstance(
            evidence_boundary, int):
        raise TypeError(f"{context}.boundary_tokens must be an integer")
    if evidence_boundary != boundary_states:
        raise ValueError(
            f"{context} evidence boundary_tokens {evidence_boundary} "
            "disagrees with the per-request floor(seq_lens/ratio) - "
            f"floor(prefix_lens/ratio) boundary count {boundary_states}")
    compressed_cache = evidence["compressed_cache"]
    if not isinstance(compressed_cache, Mapping):
        raise TypeError(f"{context}.compressed_cache must be a mapping")
    _require_evidence_int(
        compressed_cache, "token_bytes", _DSV4_COMPRESSED_STATE_BYTES,
        context)
    if evidence["compress_kernels_executed"] is not True:
        raise ValueError(f"{context} compress kernels were not executed")
    shape = {
        "element_count": boundary_states * _DSV4_COMPRESSED_STATE_BYTES,
        "seq_len": new_tokens,
        "compress_ratio": ratio,
        "state_bytes": _DSV4_COMPRESSED_STATE_BYTES,
        "dtype_bytes": 1,
    }
    attributes = {
        "cache_layout": "paged_dsv4_compressed",
        "cache_write_semantics": "append",
        "quantization": "fp8_ds_mla",
        "state_old_lens": old_lens,
        "state_new_lens": new_lens,
    }
    if ratio == 128:
        # The source always submits compress and store, including a decode
        # step that completes no state. Keep the full input-token grids and
        # the empty initialized scratch/output payloads as separate facts.
        compress = _selected_spec(KernelKind.KV_CACHE, hardware_id,
            {**shape, "element_count": boundary_states * _DSV4_HEAD_DIM, "dtype_bytes": 4},
            "fp32", "dsv4_compressed_state_cache",
            {**attributes, "finish_stage": "compress"}, hardware)
        store = _selected_spec(KernelKind.KV_CACHE, hardware_id, shape,
            "uint8", "dsv4_compressed_state_cache",
            {**attributes, "finish_stage": "store"}, hardware)
        return (compress, store)
    return (_selected_spec(
        KernelKind.KV_CACHE, hardware_id, shape, "uint8",
        "dsv4_compressed_state_cache", attributes, hardware),)


def _v4attn_mhc_spec(runner_args: Mapping, evidence: Mapping,
                     kernel_records, hardware_id: str,
                     hardware) -> KernelSpec:
    context = "V4ATTN mhc case"
    stage = _non_empty_string(runner_args, "op", context)
    if stage not in ("mhc_pre", "mhc_post_pre"):
        raise AssertionError(f"unhandled V4ATTN mhc stage {stage!r}")
    tokens = _positive_int(runner_args, "batch", context)
    hidden = _positive_int(runner_args, "hidden-size", context)
    hc_mult = _positive_int(runner_args, "hc-mult", context)
    _positive_int(runner_args, "sinkhorn-iters", context)
    _require_evidence_int(evidence, "op", stage, context)
    _require_evidence_int(evidence, "num_tokens", tokens, context)
    _require_evidence_int(evidence, "hidden_size", hidden, context)
    _require_evidence_int(evidence, "hc_mult", hc_mult, context)
    _require_observed_kernel(kernel_records, "mhc", context)
    raise NotImplementedError(
        "legacy mHC macro evidence lacks per-stage prenorm binding and launch "
        "semantics; it cannot be mapped to one VECTOR kernel")


def build_v4attn_case_specs(op_attributes: Mapping, kernel_records,
                            hardware_id: str, hardware) -> list:
    """Map one V4ATTN candidate to the production spec of what it measured.

    The decode/prefill cases measure the FlashMLA sparse-attention call
    (plus unmodeled combine/gather glue); the compressor case measures the
    fused compress+store; the mhc cases measure the tilelang mHC kernels.
    The lightning indexer is not part of any measured case (the runner
    replaces it with seeded random top-k), so no dsv4_indexer spec is built.
    """

    _require_dsv4_case_hardware(hardware_id)
    runner_args, evidence = _dsv4_case_record(op_attributes, "V4ATTN Attr")
    stage = _non_empty_string(runner_args, "op", "V4ATTN runner_args")
    if stage in ("decode", "prefill"):
        spec = _v4attn_attention_spec(
            runner_args, evidence, kernel_records, hardware_id, hardware)
    elif stage == "compressor":
        return list(_v4attn_compressor_spec(
            runner_args, evidence, kernel_records, hardware_id, hardware))
    elif stage in ("mhc_pre", "mhc_post_pre"):
        spec = _v4attn_mhc_spec(
            runner_args, evidence, kernel_records, hardware_id, hardware)
    else:
        raise ValueError(f"V4ATTN op {stage!r} is unknown")
    return [spec]


def build_wq_fused_moe_case_specs(op_attributes: Mapping, kernel_records,
                                  hardware_id: str, hardware) -> list:
    """Map one WQ_FUSEDMOE candidate to its MXFP4 routed-experts spec.

    The case measures one ``fused_marlin_moe`` call (two Marlin projections
    plus unmodeled align/activation/moe_sum glue) of one EP rank.
    """

    _require_dsv4_case_hardware(hardware_id)
    context = "WQ_FUSEDMOE case"
    runner_args, evidence = _dsv4_case_record(op_attributes, "WQ_FUSEDMOE Attr")
    weight_format = _non_empty_string(runner_args, "weight_format", context)
    if weight_format != "mxfp4":
        raise NotImplementedError(
            f"{context} weight_format {weight_format!r} has no production "
            "routed_experts variant (only mxfp4)")
    tokens = _positive_int(runner_args, "num_tokens", context)
    experts = _positive_int(runner_args, "num_experts", context)
    topk = _positive_int(runner_args, "top_k", context)
    hidden = _positive_int(runner_args, "hidden_size", context)
    intermediate = _positive_int(runner_args, "intermediate_size", context)
    group_size = _positive_int(runner_args, "group_size", context)
    if group_size != 32:
        raise ValueError(f"{context} group_size must be 32, got {group_size}")
    ep_size = _positive_int(runner_args, "ep_size", context)
    ep_rank = _require_non_negative_int(runner_args, "ep_rank", context)
    if ep_rank >= ep_size or experts % ep_size:
        raise ValueError(f"{context} EP placement is invalid")
    local_experts = experts // ep_size
    first_local = ep_rank * local_experts
    _require_evidence_int(evidence, "weight_format", weight_format, context)
    _require_evidence_int(evidence, "num_tokens", tokens, context)
    _require_evidence_int(evidence, "num_experts", experts, context)
    _require_evidence_int(evidence, "local_experts", local_experts, context)
    _require_evidence_int(evidence, "ep_size", ep_size, context)
    _require_evidence_int(evidence, "ep_rank", ep_rank, context)
    _require_evidence_int(evidence, "first_local_expert", first_local, context)
    _require_evidence_int(evidence, "top_k", topk, context)
    _require_evidence_int(evidence, "hidden_size", hidden, context)
    _require_evidence_int(
        evidence, "intermediate_size_per_partition", intermediate, context)
    _require_evidence_int(evidence, "group_size", group_size, context)
    global_counts = _require_int_sequence(
        evidence["tok_per_expert"], "tok_per_expert", context)
    if len(global_counts) != experts:
        raise ValueError(
            f"{context} evidence tok_per_expert has {len(global_counts)} "
            f"entries, expected {experts}")
    if any(count < 0 for count in global_counts):
        raise ValueError(f"{context} evidence tok_per_expert cannot be negative")
    if sum(global_counts) != tokens * topk:
        raise ValueError(
            f"{context} evidence tok_per_expert sums to "
            f"{sum(global_counts)}, expected TOK*TOPK={tokens * topk}")
    local_counts = _require_int_sequence(
        evidence["local_tok_per_expert"], "local_tok_per_expert", context)
    if len(local_counts) != local_experts:
        raise ValueError(
            f"{context} evidence local_tok_per_expert has "
            f"{len(local_counts)} entries, expected {local_experts}")
    if any(count < 0 for count in local_counts):
        raise ValueError(
            f"{context} evidence local_tok_per_expert cannot be negative")
    if local_counts != global_counts[first_local:first_local + local_experts]:
        raise ValueError(
            f"{context} evidence local_tok_per_expert disagrees with the "
            "global routing histogram")
    if sum(local_counts) != evidence["local_routed_pairs"]:
        raise ValueError(
            f"{context} evidence local_routed_pairs disagrees with the "
            "local routing histogram")
    local_global = (
        (0,) * first_local + local_counts
        + (0,) * (experts - first_local - local_experts))
    _require_observed_kernel(kernel_records, "Marlin", context)
    shape = {
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "hidden": hidden,
        "intermediate": intermediate,
    }
    attributes = {
        "tokens_per_expert": global_counts,
        "local_tokens_per_expert": local_global,
        "expert_parallel_size": ep_size,
        "expert_rank": ep_rank,
        "owned_experts": tuple(range(first_local, first_local + local_experts)),
        "expert_weight_tile_rows": None,
        "routed_scaling_factor": 1.0,
        "routed_weight_dtype": "mxfp4",
        "weight_group_size": group_size,
    }
    spec = _selected_spec(
        KernelKind.FUSED_MOE, hardware_id, shape, "bf16", "routed_experts",
        attributes, hardware)
    # The M-block heuristic is a launch decision: the production selector
    # must reproduce the observed one for the case to be in domain.
    tiling = spec.launch.tiling
    if tiling is None:
        raise ValueError(f"{context} production launch carries no tiling")
    observed_block_m = evidence["marlin_block_size_m"]
    if tiling.tile_shape["m"] != observed_block_m:
        raise ValueError(
            f"{context} production Marlin block_size_m "
            f"{tiling.tile_shape['m']} disagrees with the observed "
            f"{observed_block_m}")
    return [spec]


def build_dsv4_case_specs(op_type: str, op_attributes: Mapping,
                          kernel_records, hardware_id: str,
                          hardware) -> list:
    """Dispatch one DeepSeek-V4 campaign candidate to its case builder."""

    if op_type == "V4ATTN":
        return build_v4attn_case_specs(
            op_attributes, kernel_records, hardware_id, hardware)
    if op_type == "WQ_FUSEDMOE":
        return build_wq_fused_moe_case_specs(
            op_attributes, kernel_records, hardware_id, hardware)
    raise NotImplementedError(
        f"no DeepSeek-V4 campaign case builder for {op_type!r}")
