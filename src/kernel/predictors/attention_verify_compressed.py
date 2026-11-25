"""Isolated SGLang v0.5.19 linear verify-T4 CSA source workload.

This is a distinct current-source program, not the reference HTML's missing
verify adapter. BS1, Q64/D512, non-online BF16 state; projected gate inputs,
APE, pooled temporaries and norm weights use their caller's FP32 dtype.
No kernel times or fitted coefficients occur here.

Raw-history HCA is fail-closed: core metadata requires SWA page 128,
whereas the raw speculative C128 planner requires a page divisible by
its 256-row ring. Online C128 is a different unsupported program here.
"""
from src.kernel.predictors.attention_reference_compressed import compressed_state

_PREFIX = "sglang/v0.5.19/python/sglang/"
_CUDA = _PREFIX + "kernels/jit/csrc/deepseek_v4/"
_COMPRESS = _PREFIX + "kernels/ops/attention/dsv4/compress.py"
_BACKEND = _PREFIX + "srt/layers/attention/dsv4/compressor_v2.py"
_ROLES = ("compute_tensor", "compute_vector", "compute_reduction",
          "compute_fma", "compute_sfu", "compute_alu")


def _ratio(family):
    if family not in ("CSA", "HCA"):
        raise ValueError("verify compressed family must be CSA or HCA")
    if family == "HCA":
        raise NotImplementedError(
            "raw-history HCA verify-T4 has incompatible source guards: "
            "deepseek_v4_backend.py:99,2199 requires swa_page_size=128; "
            "deepseek_v4_memory_pool.py:34-49,667-669 gives speculative ring_size=256; "
            "c_plan.cuh:508 requires swa_page_size % ring_size == 0. "
            "Online C128 is a distinct source program, not a fallback.")
    return 4 if family == "CSA" else 128


def _context(family, context):
    ratio = _ratio(family)
    if type(context) is not int or context < 132:
        raise ValueError("verify compressed context must be integer >=132")
    return ratio


def verify_compressed_stages(family):
    _ratio(family)
    common = ("verify_compressor_plan", "swa_kv_quant_store",
              "core_compressor_pool", "core_history_write", "core_norm_rope_quant_store")
    if family == "CSA":
        return common + ("index_compressor_pool", "index_history_write", "index_norm_rope_quant_store")
    return common


def _work(source):
    return {"demand": dict.fromkeys(_ROLES, 0), "operands": [], "source": list(source),
            "limitations": [
                "Isolated CSA source composition for linear verify-T4, not historical reference adapter or four T1 calls.",
                "Operator adapter builds only its C4 plan. The unmodified whole-model backend also builds an unrelated C128 plan, whose raw-speculative page guard is incompatible with SWA128.",
                "Logical source operand bytes, not observed physical HBM/cache traffic; shared memory and address/control instruction service unpriced.",
                "Useful FP32 arithmetic/transcendentals; floating comparison, type-conversion and integer bit services unpriced.",
                "GPU metadata plans have four allocated slots with invalid padding; only completed compression plans produce active rows.",
            ]}


def _op(work, role, byte_count, access):
    work["operands"].append({"role": role, "bytes": byte_count, "access": access})


def _summed(parts):
    result = _work(tuple(source for part in parts for source in part["source"]))
    result["substeps"] = parts
    result["operands"] = [op for part in parts for op in part["operands"]]
    for role in _ROLES:
        result["demand"][role] = sum(part["demand"][role] for part in parts)
    return result


def verify_compressed_work(family, stage, context):
    """Source work for four causally ordered queries ending at ``context``.

    Plan consumption and history writes follow compress-before-write, preserving
    the pre-write ring. All four speculative token records are retained; no
    acceptance-dependent rollback is invented inside this operator window.
    """
    ratio = _context(family, context)
    if stage not in verify_compressed_stages(family):
        raise ValueError("unknown verify compressed stage for family")
    prefix = context - 4
    ends = tuple(end for end in range(prefix + 1, context + 1) if end % ratio == 0)
    count = len(ends)
    if stage == "verify_compressor_plan":
        # GPU input plan: fixed [4,16] C and [4,8] W, including invalid slots.
        make = _work((_CUDA + "c_plan.cuh:145-247,522-560",))
        _op(make, "final_context_int64", 8, "read")
        _op(make, "extend_four_int64", 8, "read")
        _op(make, "compress_plan_uint8_4x16", 64, "write")
        _op(make, "write_plan_uint8_4x8", 32, "write")
        translate = _work((_CUDA + "c_plan.cuh:250-395,540-560",))
        _op(translate, "plans_stage0", 96, "read")
        _op(translate, "request_id_int64", 8, "read")
        # Each valid compressed read page and every write slot is translated
        # through req_to_token int32 then full_to_state int64. C4 has two
        # overlapping read pages; C128 has one history page.
        mapped = 4 + count * (2 if ratio == 4 else 1)
        _op(translate, "req_to_token_mapping_roles", mapped * 4, "read")
        _op(translate, "full_to_state_mapping_roles", mapped * 8, "read")
        _op(translate, "plans_translated", 96, "write")
        result = _summed([make, translate])
        result["limitations"].append("Plan integer/atomic work is represented by explicit metadata payload; compiler instruction counts and host setup are not priced.")
    elif stage == "swa_kv_quant_store":
        quant = _work((_PREFIX + "kernels/ops/attention/dsv4/quant_k_cache.py:12-116",))
        _op(quant, "four_prepared_bf16_swa_rows", 4 * 512 * 2, "read")
        _op(quant, "fp8_rope_ue8m0_temporary", 4 * 583, "write")
        quant["demand"]["compute_fma"] = 4 * (448 + 14)
        quant["demand"]["compute_sfu"] = 4 * 14
        scatter = _work((_PREFIX + "kernels/ops/attention/dsv4/index_buf_accessor.py:120-186",))
        _op(scatter, "fp8_rope_ue8m0_temporary", 4 * 583, "read")
        _op(scatter, "four_slots_int64", 32, "read")
        _op(scatter, "four_cache_active_records", 4 * 583, "write")
        result = _summed([quant, scatter])
    else:
        is_index = stage.startswith("index_")
        dim = 128 if is_index else 512
        local_ratio = 4 if is_index else ratio
        coff = 2 if local_ratio == 4 else 1
        window = coff * local_ratio
        local_ends = tuple(end for end in range(prefix + 1, context + 1) if end % local_ratio == 0)
        rows = len(local_ends)
        result = _work((_CUDA + f"c{local_ratio}_v2.cuh", _COMPRESS, _BACKEND))
        if stage.endswith("compressor_pool"):
            old_rows = sum(window - min(end - prefix, window) for end in local_ends)
            new_rows = rows * window - old_rows
            _op(result, "padded_compress_plan", 64, "read")
            _op(result, "old_bf16_kv_score_history", old_rows * 2 * dim * 2, "read")
            _op(result, "new_fp32_projected_kv_score_roles", new_rows * 2 * dim * 4, "read")
            _op(result, "fp32_APE", rows * window * dim * 4, "read")
            _op(result, "active_fp32_pooled_output", rows * dim * 4, "write")
            pool = 5 * window * dim
            exp = window * dim
            if local_ratio == 4:
                pool += dim
            else:
                pool += (4 * 16 + 2 * 15) * dim
                exp += 16 * dim
            result["demand"]["compute_fma"] = rows * pool
            result["demand"]["compute_sfu"] = rows * exp
            result["history_rows_read"] = old_rows
            result["new_segment_rows_read"] = new_rows
            result["limitations"].append("Pool executes before history write; mixed BF16 history/FP32 ragged-input source branches are retained.")
        elif stage.endswith("history_write"):
            _op(result, "padded_write_plan", 32, "read")
            _op(result, "four_fp32_projected_records", 4 * 2 * coff * dim * 4, "read")
            _op(result, "four_bf16_persistent_records", 4 * 2 * coff * dim * 2, "write")
            result["typed_conversions"] = {"fp32_to_bf16": 4 * 2 * coff * dim}
            result["limitations"].append("Speculative ring mtp_pad retains all four writes independently of future acceptance; write/convert arithmetic has no invented issue price.")
        elif stage.endswith("norm_rope_quant_store"):
            result["source"] = [_CUDA + "fused_norm_rope_v2.cuh:63-232,379-513", _COMPRESS + ":421-473"]
            _op(result, "padded_compress_plan", 64, "read")
            _op(result, "active_fp32_pooled_output", rows * dim * 4, "read")
            _op(result, "fp32_norm_weight", rows * dim * 4, "read")
            _op(result, "fp32_rotary_cache", rows * 64 * 4, "read")
            _op(result, "active_output_slots_int64", rows * 8, "read")
            fma = 4 * dim + 1 + 3 * 64
            if is_index:
                # Seven 128-point butterfly stages, normalization scale,
                # then per-row FP8 activation scale and quantization.
                fma += 7 * dim + dim + dim + 2
                _op(result, "fp8_index_key_and_fp32_scale", rows * 132, "write")
            else:
                # Seven groups of 64 NoPE, BF16 RoPE tail, UE8M0 scale.
                # Fused path constructs scales by integer exponent bits,
                # not the separate Triton log2/ceil/exp2 program.
                fma += 448 + 7
                _op(result, "packed_kv_active_payload", rows * 583, "write")
            result["demand"]["compute_fma"] = rows * fma
            result["demand"]["compute_sfu"] = rows  # RMS rsqrt
            result["limitations"].append("Fused norm/RoPE/quant/store has no global normalized temporary; integer scale construction, FP8 rounding and comparisons remain unpriced.")
        else:
            raise AssertionError("unhandled verify compressed source stage")
        result["completed_compressed_rows"] = rows
        result["active_pooled_rows"] = rows
        result["allocated_pooled_rows"] = 4
        result["projected_gate_dtype"] = "fp32"
        result["history_dtype"] = "bf16"
        result["pooled_output_dtype"] = "fp32"
    result["mode"] = "sglang_0_5_19_isolated_csa_source_linear_verify_t4"
    result["source_geometry"] = {"global_page_tokens": 256, "swa_page_tokens": 128,
                                 "live_attention_window_tokens": 128,
                                 "source_history_ring_rows": 16 if ratio == 4 else 256}
    result["query_lengths"] = (4,)
    result["causal_context_lengths"] = tuple(range(prefix + 1, context + 1))
    return result


def verify_compressed_state(family, context):
    """Persistent storage and simultaneous padded GPU-plan workspaces.

    Cache inventory grows with completed/allocated compressed rows, while
    speculative history doubles the raw ring independently of context. Scratch
    allocation and active scratch traffic are reported separately.
    """
    ratio = _context(family, context)
    reference = compressed_state(family, context)
    base_components = dict(reference["components"])
    index_completed_rows = context // 4
    index_reserved_rows = (context + 3) // 4
    index_page_rows = 64
    index_allocated_pages = (index_reserved_rows + index_page_rows - 1) // index_page_rows
    if family == "CSA":
        # Current source stores the FP8 index key and FP32 scale in full
        # 64-row pages; do not inherit the historical active-row payload as
        # an allocator extent. The in-progress C4 group reserves a row but
        # does not produce a useful key or score candidate.
        base_components["fp8_index_cache_and_fp32_scales"] = index_allocated_pages * index_page_rows * 132
        base_components["bf16_index_compressor_history"] = 8 * 4 * 128 * 2
    base_bytes = sum(base_components.values())
    # The core requires 128-row SWA pages. Four queries jointly need 131
    # consecutive rows, not just the final query's 128. In this explicit
    # contiguous-slot scenario they touch two or three aligned pages.
    first_slot = context - 131
    last_slot = context - 1
    spec_swa_pages = last_slot // 128 - first_slot // 128 + 1
    swa_page_bytes = ((128 * 584 + 575) // 576) * 576
    spec_swa_page_bytes = spec_swa_pages * swa_page_bytes
    extra_history = {
        "extra_core_bf16_history": (16 if ratio == 4 else 256) // 2 * 2 * (2 if ratio == 4 else 1) * 512 * 2,
        "extra_swa_verify_page_storage": spec_swa_page_bytes - base_components["swa_cache_pages"],
    }
    if family == "CSA":
        extra_history["extra_index_bf16_history"] = 8 * 4 * 128 * 2
    workspace = {"shared_padded_gpu_compress_and_write_plans": 96,
                 "core_fp32_pooled_padded_four_rows": 4 * 512 * 4,
                 "swa_quantization_temporary_four_rows": 4 * 583}
    if family == "CSA":
        workspace["index_fp32_pooled_padded_four_rows"] = 4 * 128 * 4
    speculative = base_bytes + sum(extra_history.values())
    return {"base_persistent_bytes": base_bytes, "base_persistent_components": base_components,
            "spec_persistent_bytes": speculative, "spec_extra_components": extra_history,
            "workspace_bytes": sum(workspace.values()), "workspace_components": workspace,
            "total_bytes": speculative + sum(workspace.values()),
            "active_compressed_rows": context // ratio - (context - 4) // ratio,
            "allocated_pooled_rows": 4, "core_history_ring_rows": 16 if ratio == 4 else 256,
            "global_page_tokens": 256, "swa_page_tokens": 128, "live_attention_window_tokens": 128,
            "verify_swa_live_union_tokens": 131, "allocated_swa_pages": spec_swa_pages,
            "index_cache_completed_rows": index_completed_rows,
            "index_cache_reserved_rows": index_reserved_rows,
            "index_cache_allocated_rows": index_allocated_pages * index_page_rows,
            "index_cache_allocated_pages": index_allocated_pages,
            "index_cache_page_rows": index_page_rows,
            "index_history_ring_rows": 16 if family == "CSA" else 0,
            "source": [_PREFIX + "srt/mem_cache/deepseek_v4_memory_pool.py:34-49,291-309", _COMPRESS + ":226-293,380-419", _CUDA + "c_plan.cuh:498-560"],
            "limitations": ["Isolated CSA source verify composition, not original reference T4 adapter or unmodified whole-model backend.",
                            "SWA128 pages cover the 131-row union under explicitly contiguous slot ownership; arbitrary allocator mappings are not reconstructed.",
                            "Workspace sum is a conservative simultaneous inventory, not native allocator-observed peak; excludes prepared inputs, parameters, indexer/core scratch and host allocations."]}
