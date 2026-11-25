"""Source work for the reference SGLang CSA/HCA prepared-input path.

This module contains no timing constants. It describes the BF16-input,
BF16-history C4/C128 kernels named in the reference trace, not the V4 FP32
history implementation. Returned bytes are source operand-role payloads,
not measured HBM transactions. Native verify-T4 is deliberately rejected.
"""

_SOURCE = "sglang/v0.5.19/python/sglang/"
_KERNEL = _SOURCE + "kernels/"
_POOL = _SOURCE + "srt/mem_cache/deepseek_v4_memory_pool.py"
_RESOURCES = ("compute_tensor", "compute_vector", "compute_reduction",
              "compute_fma", "compute_sfu", "compute_alu")
_STAGES = ("swa_kv_quant_store", "compressor_update_norm_rope",
           "compressed_kv_quant_store", "compressed_index_cache_store")


def _geometry(family, context):
    if family not in ("CSA", "HCA"):
        raise ValueError("compressed reference family must be CSA or HCA")
    if type(context) is not int or context < 128:
        raise ValueError("compressed reference context must be an integer >=128")
    return 4 if family == "CSA" else 128


def compressed_work(family, stage, context, query_tokens):
    """Describe one T1 append ending at ``context`` (a token count).

    Boundary work executes only when context is divisible by the ratio.
    BF16 projected KV/score records enter the compressor directly; no
    projection GEMM or separate V4 save-partial kernel is invented. The
    reference's index store accepts an already prepared BF16 128-wide key.
    """
    ratio = _geometry(family, context)
    if stage not in _STAGES:
        raise ValueError("unknown compressed reference stage")
    if type(query_tokens) is not int or query_tokens != 1:
        raise NotImplementedError("reference compressed path supports T1 only; native verify-T4 compressor plan is unbound")
    if family == "HCA" and stage == "compressed_index_cache_store":
        raise ValueError("HCA has no index cache stage")
    complete = int(context % ratio == 0)
    dim, rope = 512, 64
    operands = []
    demand = dict.fromkeys(_RESOURCES, 0)
    typed = {}
    source = []
    substeps = None
    limitations = [
        "Source operand-role bytes assume all declared slots valid; physical HBM, cache residency and transactions unbound.",
        "Useful FP32 arithmetic and transcendental counts, not compiled instruction issue or native launch timing.",
        "Typed conversions, floating comparisons/clamps and address/control work are reported but have no separate service rate.",
    ]

    def operand(role, byte_count, access):
        operands.append({"role": role, "bytes": byte_count, "access": access})

    if stage in ("swa_kv_quant_store", "compressed_kv_quant_store"):
        rows = 1 if stage == "swa_kv_quant_store" else complete
        # Seven 64-element NoPE quant blocks plus a 64-element BF16 RoPE copy.
        # The scale allocation stores seven UE8M0 bytes, whereas cache records
        # reserve an eighth padding byte which this scatter does not write.
        operand("prepared_bf16_kv", rows * 1024, "read")
        operand("quantized_nope_rope_scales_temporary", rows * 583, "write")
        operand("quantized_nope_rope_scales_temporary", rows * 583, "read")
        operand("destination_slots_int64", rows * 8, "read")
        operand("packed_cache_active_payload", rows * 583, "write")
        demand["compute_fma"] = rows * (448 + 7 + 7)
        demand["compute_sfu"] = rows * 14  # log2 and exp2 per quant block
        typed = {"fp32_absolute": rows * 448, "fp32_max_comparisons": rows * (7 * 63 + 7),
                 "fp32_clamp_comparisons": rows * 896, "fp32_ceil": rows * 7,
                 "fp32_to_fp8": rows * 448, "scale_uint8_encode": rows * 7}
        source = [_KERNEL + "ops/attention/dsv4/quant_k_cache.py:12-116",
                  _KERNEL + "ops/attention/dsv4/index_buf_accessor.py:120-186"]
        limitations.append("Quantization temporary and scatter are distinct serial submissions; the stage totals are accounting summaries, and pricing must sum substeps.")
        substeps = [
            {"stage": "quantize", "demand": dict(demand), "operands": operands[:2],
             "source": source[:1], "limitations": list(limitations),
             "typed_actions": dict(typed)},
            {"stage": "scatter", "demand": dict.fromkeys(_RESOURCES, 0),
             "operands": operands[2:], "source": source[1:],
             "limitations": list(limitations), "typed_actions": {}},
        ]
    elif stage == "compressor_update_norm_rope":
        coff = 2 if ratio == 4 else 1
        record_bytes = 2 * coff * dim * 2
        window = coff * ratio
        operand("prepared_bf16_kv_score_record", record_bytes, "read")
        operand("persistent_bf16_kv_score_history", record_bytes, "write")
        # indices, seq_lens and C4 Page4Align's previous page index.
        operand("compressor_int32_metadata", 12 if ratio == 4 else 8, "read")
        operand("persistent_bf16_history_pool_input", complete * window * dim * 4, "read")
        operand("bf16_APE", complete * window * dim * 2, "read")
        operand("pooled_bf16_output", complete * dim * 2, "write")
        operand("pooled_bf16_output", complete * dim * 2, "read")
        operand("bf16_norm_weight", complete * dim * 2, "read")
        operand("fp32_rotary_cache", complete * rope * 4, "read")
        operand("norm_rope_bf16_output", complete * dim * 2, "write")
        operand("norm_boundary_int32_seq_len", 4, "read")
        # Every window score first receives its APE, then max-shift exp;
        # products and denominator sums count the source zero accumulators.
        pool_fma = 5 * window * dim
        pool_exp = window * dim
        if ratio == 4:
            pool_fma += dim  # final division of the weighted sum
            typed["pool_max_comparisons"] = complete * 7 * dim
        else:
            # c128.cuh merges sixteen eight-row partials. Each partial has
            # rescale subtraction, numerator rescale/divide, denominator
            # rescale; two 16-way sums take 15 additions per output.
            partials = 16
            pool_fma += (4 * partials + 2 * (partials - 1)) * dim
            pool_exp += partials * dim
            typed["pool_max_comparisons"] = complete * ((window - partials) + partials - 1) * dim
        # RMS square+sum, mean+eps, two normalize multiplies; 64 RoPE
        # elements each have two multiplies and one add/subtract.
        norm_fma = 4 * dim + 1 + 3 * rope
        demand["compute_fma"] = complete * (pool_fma + norm_fma)
        demand["compute_sfu"] = complete * (pool_exp + 1)
        typed.update({"bf16_pool_output_rounding": complete * dim,
                      "bf16_norm_rounding": complete * dim})
        source = [_KERNEL + f"jit/csrc/deepseek_v4/c{ratio}.cuh",
                  _KERNEL + "jit/csrc/deepseek_v4/fused_norm_rope.cuh:55-169",
                  _KERNEL + "ops/attention/dsv4/compress_old.py"]
        limitations.append("C4 Page4Align BF16 history and C128 raw BF16 ring explicitly selected from reference template facts; online C128 and FP32-state variants excluded.")
        limitations.append("C128 shared partial reductions and synchronization use mathematical work counts; shared-memory traffic and barriers are unpriced.")
        limitations.append("BF16 APE and norm weights are prepared parameter inputs, matching the reference kernel pointer types; model-loader/runtime parameter conversion is outside this prepared-input scope.")
        pool_demand = dict.fromkeys(_RESOURCES, 0)
        pool_demand["compute_fma"] = complete * pool_fma
        pool_demand["compute_sfu"] = complete * pool_exp
        norm_demand = dict.fromkeys(_RESOURCES, 0)
        norm_demand["compute_fma"] = complete * norm_fma
        norm_demand["compute_sfu"] = complete
        substeps = [
            {"stage": "history_update_and_pool", "demand": pool_demand,
             "operands": operands[:6], "source": [source[0], source[2]],
             "limitations": list(limitations),
             "typed_actions": {"pool_max_comparisons": typed["pool_max_comparisons"],
                               "bf16_pool_output_rounding": typed["bf16_pool_output_rounding"]}},
            {"stage": "norm_rope", "demand": norm_demand,
             "operands": operands[6:], "source": source[1:2],
             "limitations": list(limitations),
             "typed_actions": {"bf16_norm_rounding": typed["bf16_norm_rounding"]}},
        ]
    elif stage == "compressed_index_cache_store":
        # Reference trace has exactly _act_quant_kernel; its already projected
        # input key is 128 BF16 values, not another compressor subgraph.
        operand("prepared_bf16_index_key", complete * 128 * 2, "read")
        operand("fp8_index_cache", complete * 128, "write")
        operand("fp32_index_scale", complete * 4, "write")
        demand["compute_fma"] = complete * (128 + 1)
        typed = {"fp32_absolute": complete * 128, "fp32_max_comparisons": complete * 128,
                 "fp32_to_fp8": complete * 128, "fp32_clamp_comparisons": complete * 256}
        source = [_KERNEL + "ops/attention/dsa/triton_kernel.py:10-85"]
        limitations.append("Already prepared index key is an input, as in reference projected-input adapter; no unobserved separate index compressor or key projection is added. act_quant uses default scale_fmt=None (unrounded FP32 scale).")
    else:
        raise AssertionError("non-exhaustive compressed stage dispatch")
    result = {"demand": demand, "operands": operands, "limitations": limitations,
              "source": source, "typed_actions": typed,
              "completed_compressed_rows": complete,
              "context_counting": "query segment ends at context; old_context=context-1"}
    if substeps is not None:
        result["substeps"] = substeps
    return result


def compressed_state(family, context):
    """Reference raw storage plus BF16 persistent compressor history.

    Cache page geometries are reference-case facts: 64 SWA rows/page,
    64 C4 rows/page, two C128 rows/page, and 576-byte page alignment.
    This is active storage for one request/layer, not whole-pool reservation,
    transient peak memory, parameters or a model-level capacity claim.
    """
    ratio = _geometry(family, context)
    compressed_rows = context // ratio
    # The reference reserves a cache row for an in-progress compression
    # group. That row is storage only: it is not a completed attention key.
    allocated_compressed_rows = (context + ratio - 1) // ratio
    coff = 2 if ratio == 4 else 1
    ring_rows = 8 if ratio == 4 else 128
    compressed_page_rows = 64 if ratio == 4 else 2

    def page_bytes(rows):
        return ((rows * 584 + 575) // 576) * 576

    components = {
        "swa_cache_pages": ((min(context, 128) + 63) // 64) * page_bytes(64),
        "compressed_kv_cache_pages": ((allocated_compressed_rows + compressed_page_rows - 1) // compressed_page_rows)
            * page_bytes(compressed_page_rows),
        "bf16_compressor_history": ring_rows * 2 * coff * 512 * 2,
    }
    if family == "CSA":
        components["fp8_index_cache_and_fp32_scales"] = allocated_compressed_rows * 132
    cache_only = sum(value for name, value in components.items() if name != "bf16_compressor_history")
    return {"active_persistent_bytes": sum(components.values()), "components": components,
            "reference_core_cache_storage_bytes": cache_only,
            "active_raw_cache_payload_bytes": (min(context, 128) + compressed_rows) * 584
                + (compressed_rows * 132 if family == "CSA" else 0),
            "compressed_rows": compressed_rows,
            "allocated_compressed_rows": allocated_compressed_rows, "compress_ratio": ratio,
            "history_dtype": "bf16", "swa_window": 128,
            "compressed_page_rows": compressed_page_rows, "cache_page_alignment_bytes": 576,
            "source": [_POOL + ":34-49,106-130,291-294",
                       _KERNEL + f"jit/csrc/deepseek_v4/c{ratio}.cuh"],
            "limitations": ["Reference raw cache page geometry; excludes whole-pool reserve/null slots, transient workspace and parameters.",
                            "Non-speculative raw BF16 history; verify-T4 requires a separate enlarged-state/source-plan contract."]}
