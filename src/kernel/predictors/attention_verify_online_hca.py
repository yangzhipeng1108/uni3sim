"""Explicit online-C128 HCA T4 operator scenario from pinned SGLang source.

Both online/MTP source flags are declared true here; this module never changes
process environment, model settings or a Serving CLI. The program differs from
raw-history HCA and is not a fallback or a native-runtime acceptance claim.
"""
from src.kernel.predictors.attention_reference_compressed import compressed_state
from src.kernel.predictors.attention_verify_compressed import verify_compressed_work

_ROOT = "sglang/v0.5.19/python/sglang/"
_CUDA = _ROOT + "kernels/jit/csrc/deepseek_v4/"
_CONTROL = _ROOT + "kernels/ops/attention/dsv4/online_c128_mtp.py"
_COMPRESS = _ROOT + "kernels/ops/attention/dsv4/compress.py"
_RESOURCES = ("compute_tensor", "compute_vector", "compute_reduction",
              "compute_fma", "compute_sfu", "compute_alu")
_STAGES = ("online_previous_pending_commit", "online_mark_pending", "online_compressor_plan",
           "swa_kv_quant_store", "online_emit_segments", "online_tail_segments",
           "online_norm_rope_quant_store", "online_prefix_snapshots")
_DIM, _STATE_BYTES = 512, 3 * 512 * 4


def online_hca_stages():
    return _STAGES


def _segments(context):
    if type(context) is not int or context < 136:
        raise ValueError("online HCA T4 requires integer ending context >=136")
    start, end = context - 4, context
    segments = []
    while start < end:
        stop = min(end, (start // 128 + 1) * 128)
        segments.append({"tokens": stop - start, "chunk_offset": start % 128,
                         "end": stop, "emit": stop % 128 == 0})
        start = stop
    return segments


def _new(source):
    return {"demand": dict.fromkeys(_RESOURCES, 0), "operands": [], "source": list(source),
            "limitations": [
                "Explicit online-C128 MTP single-layer operator scenario; raw HCA refusal remains unchanged and no model/process MTP setting is enabled.",
                "No H200 native execution or numerical accuracy claim; pinned generic CUDA source has no SM100-only gate in these entry points.",
                "Source operand-role bytes are not physical HBM transactions; shared-memory traffic, comparisons, conversions, integer/atomic/barrier and compiled issue services unpriced.",
                "Previous pending verify base is N-8, exactly four tokens accepted, current prefix N-4; no unspecified acceptance distribution is assumed.",
                "Prepared projected KV/score, APE and norm parameters are FP32; output cache uses mixed FP8/BF16 packed records and online state is FP32.",
            ]}


def _operand(work, role, byte_count, access):
    work["operands"].append({"role": role, "bytes": int(byte_count), "access": access})


def online_hca_work(stage, context):
    """One four-token chain ending at context, including pending lifecycle.

    The separate prefix-snapshot kernel recomputes the four-step online update;
    its inputs and arithmetic are charged again rather than credited as reuse.
    Emit/tail passes retain the real padded 128-position source arithmetic.
    """
    segments = _segments(context)
    if stage not in _STAGES:
        raise ValueError("unknown online HCA T4 source stage")
    prefix = context - 4
    if stage == "swa_kv_quant_store":
        # This prepared BF16 row quant/scatter source is shared by CSA/HCA.
        work = verify_compressed_work("CSA", stage, context)
        # Reuse the common device work, not CSA's ring/planner assertions.
        work = {**work, "limitations": _new(work["source"])["limitations"]}
        for part in work["substeps"]:
            part["limitations"] = list(work["limitations"])
    elif stage == "online_previous_pending_commit":
        work = _new((_CONTROL + ":100-139,184-225", _CUDA + "online_c128_mtp.cuh:79-104"))
        _operand(work, "current_prefix_and_request_int64", 16, "read")
        _operand(work, "previous_pending_base_int64", 8, "read")
        copy = int(prefix % 128 != 0)
        _operand(work, "previous_verify_accepted_bank4", copy * _STATE_BYTES, "read")
        _operand(work, "committed_online_main_bank0", copy * _STATE_BYTES, "write")
        work["commit_copy_executed"] = bool(copy)
        work["accepted_previous_tokens"] = 4
        work["previous_pending_base"] = context - 8
    elif stage == "online_mark_pending":
        work = _new((_CONTROL + ":69-95", _CUDA + "online_c128_mtp.cuh:69-77"))
        _operand(work, "current_prefix_and_request_int64", 16, "read")
        _operand(work, "pending_verify_base_int64", 8, "write")
        work["pending_mark_count"] = 1
    elif stage == "online_compressor_plan":
        work = _new((_COMPRESS + ":335-374", _CUDA + "c128_online_v2.cuh:668-757,834-891"))
        # Eager CPU stage0 builds compact views; backing device buffers are
        # nevertheless allocated as [4,16] for BOTH C and W in the wrapper.
        count = len(segments)
        _operand(work, "host_plan_upload_device_payload", count * 16, "write")
        _operand(work, "plan_stage0", count * 16, "read")
        _operand(work, "request_id_int64_role_reads", count * 8, "read")
        _operand(work, "translated_plan_stage1", count * 16, "write")
        work["host_plan_bytes"] = 128
        work["device_plan_backing_bytes"] = 128
        work["compact_emit_plans"] = sum(s["emit"] for s in segments)
        work["compact_tail_plans"] = sum(not s["emit"] for s in segments)
        work["limitations"].append("Host stage0 metadata construction and host-to-device transfer latency unpriced; explicit device writes retained, no GPU raw-ring planner substituted.")
    elif stage in ("online_emit_segments", "online_tail_segments"):
        emit = stage == "online_emit_segments"
        selected = [s for s in segments if s["emit"] is emit]
        work = _new((_CUDA + "c128_online_v2.cuh:187-262,310-438,544-559",))
        for i, segment in enumerate(selected):
            valid = segment["tokens"]
            merge = int(segment["chunk_offset"] != 0)
            _operand(work, f"segment_{i}_compress_plan", 16, "read")
            _operand(work, f"segment_{i}_fp32_projected_kv_score", valid * 2 * _DIM * 4, "read")
            _operand(work, f"segment_{i}_fp32_APE", valid * _DIM * 4, "read")
            _operand(work, f"segment_{i}_committed_online_state", merge * _STATE_BYTES, "read")
            if emit:
                _operand(work, f"segment_{i}_emitted_fp32_compressed_kv", _DIM * 4, "write")
            else:
                _operand(work, f"segment_{i}_temporary_online_bank1", _STATE_BYTES, "write")
            # All 128 source positions execute local padded softmax; only
            # valid input positions load payload and receive APE additions.
            # Sixteen partials merge by rescaling and normalized weighting.
            fma = (4 * 128 + valid + 4 * 16 + 2 * 15) * _DIM
            sfu = (128 + 16) * _DIM
            # One online state/segment merge: two shifted exp weights,
            # weighted mean and denominator; absent for a fresh chunk.
            fma += merge * 9 * _DIM
            sfu += merge * 2 * _DIM
            work["demand"]["compute_fma"] += fma
            work["demand"]["compute_sfu"] += sfu
        work["segments"] = selected
        work["active_segment_count"] = len(selected)
        work["typed_actions"] = {"padded_softmax_and_merge_max_comparisons": len(selected) * 127 * _DIM}
        work["limitations"].append("Emit and tail are distinct ordered passes; old committed state stays read-only and the tail uses the offset temporary bank.")
    elif stage == "online_norm_rope_quant_store":
        count = sum(s["emit"] for s in segments)
        work = _new((_CUDA + "fused_norm_rope_v2.cuh:379-513", _COMPRESS + ":421-473"))
        _operand(work, "compact_emit_plans", count * 16, "read")
        _operand(work, "emitted_fp32_compressed_kv", count * _DIM * 4, "read")
        _operand(work, "fp32_norm_weight", count * _DIM * 4, "read")
        _operand(work, "fp32_rotary_cache", count * 64 * 4, "read")
        _operand(work, "destination_slot_int64", count * 8, "read")
        _operand(work, "mixed_fp8_bf16_cache_active_payload", count * 583, "write")
        work["demand"]["compute_fma"] = count * (4 * _DIM + 1 + 3 * 64 + 448 + 7)
        work["demand"]["compute_sfu"] = count
        work["active_compressed_rows"] = count
    elif stage == "online_prefix_snapshots":
        work = _new((_CONTROL + ":139-182", _CUDA + "online_c128_mtp.cuh:106-201,229-262"))
        _operand(work, "current_prefix_and_request_int64", 16, "read")
        _operand(work, "committed_online_bank0", int(prefix % 128 != 0) * _STATE_BYTES, "read")
        _operand(work, "four_fp32_projected_kv_score_rows_recomputed", 4 * 2 * _DIM * 4, "read")
        _operand(work, "four_fp32_APE_rows_recomputed", 4 * _DIM * 4, "read")
        nonreset = sum(pos % 128 != 0 for pos in range(prefix, context))
        stored = sum(end % 128 != 0 for end in range(prefix + 1, context + 1))
        _operand(work, "online_prefix_banks_1_to_4_active", stored * _STATE_BYTES, "write")
        work["demand"]["compute_fma"] = (4 + 8 * nonreset) * _DIM
        work["demand"]["compute_sfu"] = 2 * nonreset * _DIM
        work["active_prefix_snapshots"] = stored
        work["reserved_prefix_banks"] = 4
        work["ordered_token_loop"] = 4
        work["typed_actions"] = {"online_max_comparisons": nonreset * _DIM}
        work["limitations"].append("This kernel repeats online gate/APE/update after emit/tail; boundary snapshots are skipped and next-chunk accumulators reset, but all four banks remain reserved.")
    else:
        raise AssertionError("non-exhaustive online HCA source dispatch")
    work["mode"] = "sglang_0_5_19_online_c128_mtp_hca_t4"
    work["family"] = "HCA"
    work["source_geometry"] = {"global_page_tokens": 256, "swa_page_tokens": 128,
                               "live_attention_window_tokens": 128,
                               "online_history_ring_rows": 1}
    work["source_flags"] = {"SGLANG_OPT_USE_ONLINE_COMPRESS": True,
                            "SGLANG_EXPERIMENTAL_ONLINE_C128_MTP": True,
                            "backend_mtp_enabled_in_operator_scenario": True,
                            "operator_scenario_only": True, "model_cli_mtp_enabled": False,
                            "online_history_ring_rows": 1, "state_dtype": "fp32"}
    work["causal_context_lengths"] = tuple(range(prefix + 1, context + 1))
    return work


def online_hca_state(context):
    """Active request inventory; excludes pool/null reserves and core scratch."""
    segments = _segments(context)
    raw = compressed_state("HCA", context)
    base = {key: value for key, value in raw["components"].items()
            if key != "bf16_compressor_history"}
    base["committed_fp32_online_state"] = _STATE_BYTES
    swa_pages = (context - 1) // 128 - (context - 131) // 128 + 1
    swa_page_bytes = ((128 * 584 + 575) // 576) * 576
    spec = {"reserved_four_fp32_prefix_banks": 4 * _STATE_BYTES,
            "pending_previous_verify_base_int64": 8,
            "extra_verify_swa_pages": swa_pages * swa_page_bytes - base["swa_cache_pages"]}
    emitted = sum(s["emit"] for s in segments)
    temporary = {"online_device_plan_backing_C_and_W_4x16": 128,
                 "compact_emitted_fp32_kv": emitted * _DIM * 4,
                 "four_swa_quantization_temporary_records": 4 * 583}
    persistent = sum(base.values())
    speculative = persistent + sum(spec.values())
    workspace = sum(temporary.values())
    return {"base_persistent_bytes": persistent, "base_persistent_components": base,
            "spec_persistent_bytes": speculative, "spec_extra_components": spec,
            "workspace_bytes": workspace, "workspace_components": temporary,
            "total_bytes": speculative + workspace,
            "host_pinned_plan_bytes": 128,
            "online_main_state_bytes": _STATE_BYTES,
            "reserved_prefix_banks": 4,
            "active_prefix_snapshots": sum(end % 128 != 0 for end in range(context - 3, context + 1)),
            "source_geometry": {"global_page_tokens": 256, "swa_page_tokens": 128,
                                "live_attention_window_tokens": 128, "verify_window_union_tokens": 131,
                                "allocated_swa_pages": swa_pages, "head_dim": 512,
                                "query_heads": 64, "query_tokens": 4},
            "source": [_ROOT + "srt/mem_cache/deepseek_v4_compress_state.py:108-117",
                       _ROOT + "srt/mem_cache/deepseek_v4_memory_pool.py:31-44,537-544",
                       _COMPRESS + ":335-374", _CONTROL],
            "limitations": ["Separate online operator profile; no raw-HCA fallback, process env or model MTP change.",
                            "Active banks and explicit simultaneous workspace inventory; prepared inputs/parameters, pool padding, host allocation and attention-core/output scratch excluded.",
                            "Contiguous-slot SWA131 union uses two or three pages; actual allocator mapping/native peak is not attested."]}
