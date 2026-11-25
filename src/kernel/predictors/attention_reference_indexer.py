"""Reference-shaped indexer source work, independent of measured durations.

These ledgers describe explicit algorithms and logical role bytes.  They do
not select a compiled provider.  In particular ``selection_passes`` exposes
the one-to-four byte-radix tie-refinement scenarios in SGLang's source; it
must not be fitted to production timings or called a native latency bound.
"""

_SGL = "sglang/v0.5.19/python/sglang/kernels/jit/"
_VLLM = "vllm/v0.30.0/vllm/"
_FAMILIES = ("DSA", "CSA", "GLM-5.3-Flash DSA")
_STAGES = ("indexer_q_prepare", "indexer_logits", "topk_page_transform",
           "kpool_cache_update", "index_pad_for_tilelang",
           "compressed_index_cache_store")


def _demand(*, tensor=0, fma=0, alu=0, sfu=0):
    return {"compute_tensor": tensor, "compute_vector": 0,
            "compute_fma": fma, "compute_alu": alu,
            "compute_reduction": 0, "compute_sfu": sfu}


def _operand(role, count, width, access):
    return {"role": role, "bytes": count * width, "access": access}


def _work(demand, operands, source, limitations, **details):
    return {"demand": demand, "operands": operands, "source": source,
            "limitations": ["logical role bytes are not measured HBM traffic",
                "native launch, compiler issue, cache residency and atomic latency unbound",
                *limitations], **details}


def _prepare(family, tokens):
    # The HTML trace identifies DSA's fused rope-first / no-Hadamard entry,
    # CSA's same no-Hadamard producer, and Flash-v1's act_quant + weights.
    groups, values = tokens * 64, tokens * 64 * 128
    rope = 0 if family == "GLM-5.3-Flash DSA" else 64
    hadamard = False
    arithmetic = groups * (3 * rope + 128 + 4)
    if hadamard:
        arithmetic += 8 * values  # seven butterfly add/sub stages + scaling.
    operands = [_operand("projected_index_query_bf16", values, 2, "read"),
        _operand("index_weights_fp32" if family == "GLM-5.3-Flash DSA" else "index_weights_bf16",
                 groups, 4 if family == "GLM-5.3-Flash DSA" else 2, "read"),
        _operand("prepared_index_query_fp8", values, 1, "write"),
        _operand("folded_index_weights_fp32", groups, 4, "write")]
    if rope:
        operands.extend((_operand("position_i32", tokens, 4, "read"),
            _operand("rotary_cache_bf16_unique_rows", tokens * rope, 2, "read")))
    else:
        # The unfused quantizer writes a global scale tensor consumed by
        # the distinct weights kernel. Fused DSA/CSA do not materialize it.
        operands.extend((_operand("query_quant_scales_fp32", groups, 4, "write"),
            _operand("query_quant_scales_fp32", groups, 4, "read")))
    result = _work(_demand(fma=arithmetic, sfu=2 * groups if not rope else 0), operands,
        [_SGL + "csrc/deepseek_v4/main_norm_rope.cuh:fused_q_indexer_rope_hadamard_quant",
         "sglang/v0.5.19/python/sglang/kernels/ops/attention/dsa/triton_kernel.py:_act_quant_kernel"],
        ["FP32 abs/max/ceil/clamp and casts retained as typed actions; no invented integer price",
         "Flash assumes explicit ue8m0 round_scale=True; unrounded alternative removes log2/exp2/ceil",
         "DSA and CSA trace templates declare rope-first and Hadamard=False; fused scales are unrounded"],
        typed_actions={"absmax_comparisons": groups * 127,
            "abs": values, "ceil": groups if not rope else 0, "fp8_conversions": values,
            "quantization_clamp_comparisons": 2 * values},
        geometry={"heads": 64, "dim": 128, "rope_dim": rope,
            "hadamard": hadamard})
    if not rope:
        # The reference submits quantization before scale/weight folding.
        # Keep the stage aggregate as a ledger; price its serial substeps
        # separately so the two submissions cannot overlap through one max.
        result["substeps"] = [
            {"stage": "act_quant", "demand": _demand(fma=groups * 129, sfu=2 * groups),
             "operands": [operands[0], operands[2], operands[4]],
             "source": result["source"][1:], "limitations": list(result["limitations"]),
             "typed_actions": dict(result["typed_actions"])},
            {"stage": "fold_weights", "demand": _demand(fma=3 * groups),
             "operands": [operands[1], operands[5], operands[3]],
             "source": ["sglang/v0.5.19/python/sglang/srt/layers/attention/dsa/dsa_indexer.py:_apply_q_scale_and_softmax_scale"],
             "limitations": list(result["limitations"]), "typed_actions": {}},
        ]
    return result


def _logits(family, context, tokens):
    ratio = 1 if family == "DSA" else 4
    pairs = sum(length // ratio for length in range(context - tokens + 1, context + 1))
    values, groups = tokens * 64 * 128, tokens * 64
    return _work(_demand(tensor=2 * pairs * 64 * 128, fma=2 * pairs * 64),
        [_operand("prepared_query_fp8", values, 1, "read"),
         _operand("index_cache_key_role_uses_fp8", pairs * 128, 1, "read"),
         _operand("index_cache_scale_role_uses_fp32", pairs, 4, "read"),
         _operand("folded_weights_fp32", groups, 4, "read"),
         _operand("effective_context_lengths_i32", tokens, 4, "read"),
         _operand("active_logits_fp32", pairs, 4, "write")],
        ["src/kernel/predictors/sparse_attn_indexer.py:_indexer_arithmetic",
         "src/kernel/predictors/sparse_attn_indexer.py:dsa_indexer_logical_work"],
        ["per-query key role uses expose reuse assumption, not observed unique device reads",
         "active causal logits only; inactive allocation backing is not charged as traffic"],
        typed_actions={"relu_comparisons": pairs * 64},
        geometry={"index_heads": 64, "index_dim": 128,
            "compression_ratio": ratio, "active_pairs": pairs})


def _selection(family, context, tokens, passes):
    ratio = 1 if family == "DSA" else 4
    capacity = 512 if family in ("CSA", "GLM-5.3-Flash DSA") else 2048
    lengths = [length // ratio for length in range(context - tokens + 1, context + 1)]
    selected = sum(min(capacity, length) for length in lengths)
    # Register4 holds <=16384 scores; streaming/cluster explicitly scan twice.
    global_values = sum(length * (1 if length <= 16384 else 2)
                        for length in lengths if length > capacity)
    active = sum(length for length in lengths if length > capacity)
    rows = sum(length > capacity for length in lengths)
    hist_rows = [(4096, 1) if length <= 65536 else (1024, 8)
                 for length in lengths if length > capacity]
    # Source arithmetic, not instruction counts: coarse sign/key/shift(3),
    # histogram increment(1), collect-counter increment(<=1) per candidate.
    # The refinement scenario retains maximal 2048 ties and 8-bit radix;
    # actual tie cardinality and early-exit decisions are dynamic inputs.
    ties = rows * 2048
    key_arithmetic = 5 * active
    # find_threshold visits each bin for local sum, prefix update, above
    # subtraction and comparison-bound sum (4*B). Its 32 warps perform
    # inclusive shuffle adds (32*129), warp_exc subtraction (1024), a
    # mathematical 32-lane sum (32*31), and prefix+warp_exc (1024).
    # Cluster has eight replicated histograms/threshold programs, plus a
    # seven-add reduction per global histogram bin across the eight ranks.
    histogram_arithmetic = sum(ranks * (4 * bins + 32 * 129 + 1024 + 32 * 31 + 1024)
        + (ranks - 1) * bins for bins, ranks in hist_rows)
    # Each exact round: shift/mask, histogram increment, active count and
    # output counter; 256-bin inclusive suffix scan has 8 add stages.
    refinement_arithmetic = 3 * ties + passes * (5 * ties + rows * 256 * (8 + 5))
    is_pool_expansion = family == "GLM-5.3-Flash DSA"
    # Flash materializes raw pool IDs for identity expansion; it neither
    # loads a page table nor executes the ordinary page-bit transform.
    transform_arithmetic = 0 if is_pool_expansion else 2 * selected
    operands = [_operand("score_source_scans_fp32", global_values, 4, "read"),
        _operand("effective_context_lengths_i32", tokens, 4, "read")]
    if not is_pool_expansion:
        operands.append(_operand("selected_page_table_i32", selected, 4, "read"))
    operands.append(_operand("selected_pool_indices_i32" if is_pool_expansion else "selected_indices_i32",
                             tokens * capacity, 4, "write"))
    if is_pool_expansion:
        # Current fused identity expansion reads pool IDs per output token.
        # This is an explicit current recipe, not the HTML's many-op v1.
        expanded = tokens * 2051
        operands.extend((_operand("pool_ids_i32_to_i64", tokens * 512, 4, "read"),
            _operand("pool_ids_i64_materialized", tokens * 512, 8, "write"),
            _operand("pool_ids_i64_expansion_role_uses", tokens * 2048, 8, "read"),
            _operand("expanded_history_plus_tail_i32", expanded, 4, "write")))
        transform_arithmetic += 5 * expanded
    return _work(_demand(alu=key_arithmetic + histogram_arithmetic
                        + refinement_arithmetic + transform_arithmetic), operands,
        [_SGL + "include/sgl_kernel/deepseek_v4/topk_impl.cuh:TopKStreaming/TopKCluster/radix_tie_select",
         _SGL + "csrc/deepseek_v4/topk_v2.cuh:topk_small_batch_kernel",
         _VLLM + "models/glm5next/nvidia/ops/kpool_compress.py:expand_pools_and_append_tail"],
        ["maximal-2048-tie byte-radix work scenario; passes1..4 is not a native latency interval",
         "FP32 boundary/rank comparisons and atomic/barrier service are unpriced typed actions",
         "shared histogram/tie traffic is reported separately from global operands",
         "current SGLang generic selector and fused kpool recipe; original reference-v1 kernel identity not asserted"],
        typed_actions={"candidate_boundary_comparisons_upper": 2 * active,
            "small_tie_rank_comparisons_upper": rows * 128 * 128},
        shared_role_bytes={"tie_candidate_single_materialization": 2 * ties * 8,
            "histogram_initialized_and_final_payload": 2 * (sum(bins * ranks for bins, ranks in hist_rows)
                + passes * rows * 256) * 4},
        selection_scenario={"max_ties_per_row": 2048, "radix_byte_rounds": passes,
            "source_max_rounds": 4, "active_candidates": active,
            "selected_pool_or_token_entries": selected, "output_capacity": capacity})


def _kpool(context, tokens):
    # Sequential positions preserve completion-before-current-stash ordering.
    completions = context // 4 - (context - tokens) // 4
    values, dim, pool = tokens * 128, 128, 4
    # Two score+APE passes; softmax shift/sum/weighted accumulator, normalize;
    # Hadamard seven add/sub stages + scale; quantization divide + scale.
    fma_per_completion = 2 * pool * dim + 4 * pool * dim + dim + 8 * dim + dim + 1
    operands = [_operand("current_index_keys_bf16", values, 2, "read"),
        _operand("current_pool_gate_scores_bf16", values, 2, "read"),
        _operand("pool_slot_and_positions_i32", tokens * 3, 4, "read"),
        _operand("tail_keys_and_scores_bf16", 2 * values, 2, "write"),
        _operand("prior_tail_scores_two_passes_bf16", completions * 2 * pool * dim, 2, "read"),
        _operand("prior_tail_keys_bf16", completions * pool * dim, 2, "read"),
        _operand("pool_ape_two_passes_fp32", completions * 2 * pool * dim, 4, "read"),
        _operand("completed_pooled_keys_fp8", completions * dim, 1, "write"),
        _operand("completed_pooled_scales_fp32", completions, 4, "write")]
    return _work(_demand(fma=completions * fma_per_completion,
                         sfu=completions * (pool * dim + 2)), operands,
        [_VLLM + "models/glm5next/nvidia/ops/kpool_compress.py:_kpool_decode_update_batched_kernel"],
        ["all masked source loads retained including current-slot loads before tl.where",
         "known completed-group count; no amortized fractional pool-completion",
         "pool gate input is already projected; its excluded input GEMM is not priced here",
         "current sequential batched source recipe differs from reference-v1 launch sequence"],
        geometry={"pool_size": 4, "index_dim": 128, "completions": completions},
        typed_actions={"softmax_max_comparisons": completions * pool * dim,
            "quant_absmax_comparisons": completions * 127,
            "quant_clamp_comparisons": completions * 2 * dim})


def indexer_work(family, stage, context, query_tokens, selection_passes):
    """Return a fail-closed ledger for one explicit reference-shaped stage.

    Only T1/T4 and the three named indexer families are accepted.  The source
    selector exposes refinement scenarios rather than hiding its value
    dependence in a single assumed price.  Callers must retain limitations.
    """
    if family not in _FAMILIES or stage not in _STAGES:
        raise ValueError("unknown reference indexer family or stage")
    if (type(context) is not int or type(query_tokens) is not int
            or context < query_tokens or query_tokens not in (1, 4)
            or type(selection_passes) is not int or not 1 <= selection_passes <= 4):
        raise ValueError("reference indexer requires positive context, T1/T4 and radix rounds1..4")
    if stage == "indexer_q_prepare":
        return _prepare(family, query_tokens)
    if stage == "indexer_logits":
        return _logits(family, context, query_tokens)
    if stage == "topk_page_transform":
        return _selection(family, context, query_tokens, selection_passes)
    if stage == "kpool_cache_update":
        if family != "GLM-5.3-Flash DSA":
            raise ValueError("only Flash DSA uses kpool")
        return _kpool(context, query_tokens)
    if stage == "index_pad_for_tilelang":
        if family != "GLM-5.3-Flash DSA":
            raise ValueError("only Flash DSA uses TileLang2051-to2112 padding")
        return _work(_demand(), [
            _operand("expanded_indices_i32", query_tokens * 2051, 4, "read"),
            _operand("padding_fill_i32", query_tokens * 61, 4, "write"),
            _operand("padding_input_i32", query_tokens * 61, 4, "read"),
            _operand("padded_indices_i32", query_tokens * 2112, 4, "write")],
            ["reference HTML index_pad_for_tilelang trace:fill+CatArrayBatchedCopy"],
            ["records explicit fill followed by concatenation; empty allocation has no initialization"])
    if stage == "compressed_index_cache_store":
        if family != "CSA":
            raise ValueError("only CSA owns compressed index cache insertion")
        completions = context // 4 - (context - query_tokens) // 4
        return _work(_demand(fma=completions * 129, sfu=completions * 2), [
            _operand("completed_index_key_bf16", completions * 128, 2, "read"),
            _operand("completed_index_key_fp8", completions * 128, 1, "write"),
            _operand("completed_index_scale_fp32", completions, 4, "write")],
            [_VLLM + "models/deepseek_v4/common/ops/fused_indexer_q.py"],
            ["completed compressed/index-normalized key is an input; compressor producer priced separately",
             "per128 FP8 quantization math; absmax/clamp/casts retained as typed actions"],
            typed_actions={"absmax_comparisons": completions * 127,
                "fp8_conversions": completions * 128})
    raise AssertionError("exhaustive reference indexer stage dispatch")
