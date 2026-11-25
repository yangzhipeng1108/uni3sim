"""Explicit current-source verification operators, separate from the reference.

Only the isolated CSA composition and KDA non-fused chain fallback are
admitted. The raw HCA source's incompatible SWA/ring constraints remain a
refusal, rather than silently altering a source launch or page contract.
"""

from src.kernel.kernel_spec import KernelKind, KernelSpec


VERIFY_PROGRAM = "sglang_0_5_19_isolated_chain_verify_source_v1"
ONLINE_HCA_PROGRAM = "sglang_0_5_19_online_c128_chain_verify_source_v1"
_FIELDS = {"family", "stage", "source_program", "split_count", "selector_passes"}


def verify_stages_for(family):
    if family == "KDA":
        from src.kernel.predictors.attention_verify_kda import verify_kda_stages
        return verify_kda_stages()
    from src.kernel.predictors.attention_verify_compressed import verify_compressed_stages
    if family == "CSA":
        return (("verify_attention_metadata",) + verify_compressed_stages(family)
                + ("indexer_q_prepare", "indexer_logits", "topk_page_transform",
                   "attention_core", "combine"))
    if family == "HCA":
        # The helper reports the exact conflicting pinned source guards.
        verify_compressed_stages(family)
        raise AssertionError("raw HCA T4 source guard must refuse")
    raise ValueError("current source verification supports isolated CSA and KDA only")


def _kind(family, stage):
    if stage not in verify_stages_for(family):
        raise ValueError("unknown current-source verify stage")
    if stage == "attention_core":
        return KernelKind.SPARSE_ATTN
    if stage in ("indexer_logits", "topk_page_transform"):
        return KernelKind.SPARSE_ATTN_INDEXER
    return KernelKind.VECTOR


def normalize_verify_attributes(kind, attributes, shape, dtype):
    if set(attributes) != _FIELDS or attributes["source_program"] not in (VERIFY_PROGRAM, ONLINE_HCA_PROGRAM):
        raise ValueError("verify source requires its closed program and complete attributes")
    family, stage = attributes["family"], attributes["stage"]
    expected = "int32" if kind is KernelKind.SPARSE_ATTN_INDEXER else "bf16"
    if attributes["source_program"] == ONLINE_HCA_PROGRAM:
        if (family != "HCA" or shape["q_len"] != 4 or shape["kv_len"] < 136
                or shape["head_dim"] != 512 or stage not in online_hca_stages_for()
                or kind is not (KernelKind.SPARSE_ATTN if stage == "attention_core" else KernelKind.VECTOR)
                or dtype != "bf16" or type(attributes["split_count"]) is not int
                or attributes["split_count"] != 1 or type(attributes["selector_passes"]) is not int
                or attributes["selector_passes"] != 1):
            raise ValueError("online HCA requires its explicit FP32-state T4 source scenario")
        return dict(attributes)
    if (shape["q_len"] != 4 or shape["kv_len"] < 132
            or shape["head_dim"] != (128 if family == "KDA" else 512)
            or _kind(family, stage) is not kind or dtype != expected):
        raise ValueError("current source verify requires its T4 shape, stage, kind and dtype")
    if (type(attributes["split_count"]) is not int or attributes["split_count"] != 1
            or type(attributes["selector_passes"]) is not int
            or attributes["selector_passes"] not in (1, 4)):
        raise ValueError("verify source uses explicit unsplit math and radix1/4 work scenarios")
    return dict(attributes)


def verify_stage_spec(family, stage, context, *, hardware_id, selector_passes):
    kind = _kind(family, stage)
    return KernelSpec(kind=kind, hardware_id=hardware_id,
        dtype="int32" if kind is KernelKind.SPARSE_ATTN_INDEXER else "bf16",
        variant="projected_attention_path", execution_support="unsupported",
        unsupported_reason="launch_selector_unavailable",
        shape={"batch": 1, "q_len": 4, "kv_len": context, "num_query_heads": 64,
               "num_kv_heads": 1, "head_dim": 128 if family == "KDA" else 512},
        attributes={"family": family, "stage": stage, "source_program": VERIFY_PROGRAM,
                    "split_count": 1, "selector_passes": selector_passes})


def online_hca_stages_for():
    from src.kernel.predictors.attention_verify_online_hca import online_hca_stages
    return ("verify_attention_metadata",) + online_hca_stages() + ("attention_core", "combine")


def online_hca_stage_spec(stage, context, *, hardware_id):
    if stage not in online_hca_stages_for():
        raise ValueError("unknown online HCA source stage")
    return KernelSpec(kind=KernelKind.SPARSE_ATTN if stage == "attention_core" else KernelKind.VECTOR,
        hardware_id=hardware_id, dtype="bf16", variant="projected_attention_path",
        execution_support="unsupported", unsupported_reason="launch_selector_unavailable",
        shape={"batch": 1, "q_len": 4, "kv_len": context, "num_query_heads": 64,
               "num_kv_heads": 1, "head_dim": 512},
        attributes={"family": "HCA", "stage": stage, "source_program": ONLINE_HCA_PROGRAM,
                    "split_count": 1, "selector_passes": 1})


def _metadata(context):
    """Logical GPU boundary payloads for the isolated C4 metadata program.

    BuildPageTablePositions and BuildCausalSwaPageIndices are sequenced.
    Compression metadata requests compute_page_indices=False: an isolated
    C4 operator does not create the unrelated C128 planner/indices.
    """
    if type(context) is not int or context < 132:
        raise ValueError("verify compressed context must be integer >=132")
    resources = ("compute_tensor", "compute_vector", "compute_reduction",
                 "compute_fma", "compute_sfu", "compute_alu")
    demand = dict.fromkeys(resources, 0)
    pages = -(-context // 256)
    parts = []

    def part(stage, operands):
        return {"stage": stage, "demand": dict(demand),
            "operands": [{"role": role, "bytes": count, "access": access}
                         for role, count, access in operands],
            "source": ["sglang/v0.5.19/python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py",
                       "sglang/v0.5.19/python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py"],
            "limitations": ["Logical metadata tensor payload; repeated CTA loads, address/host/launch services unpriced"]}

    parts.append(part("page_table_and_positions", (
        ("request_ids_i64", 4 * 8, "read"), ("causal_lengths_i32", 4 * 4, "read"),
        ("request_page_mapping_i32", 4 * pages * 4, "read"),
        ("page_table_i32", 4 * pages * 4, "write"),
        ("positions_lengths_swa_lengths_i32", 4 * 3 * 4, "write"))))
    parts.append(part("causal_swa_indices", (
        ("request_and_causal_lengths", 4 * (8 + 4), "read"),
        ("request_to_token_i32", 4 * 128 * 4, "read"),
        ("full_to_swa_i32", 4 * 128 * 4, "read"),
        ("swa_indices_i32", 4 * 128 * 4, "write"))))
    parts.append(part("compression_metadata_without_c128_indices", (
        ("lengths_positions_out_locations", 4 * (4 + 4 + 8), "read"),
        # Both C4 and C128 return outloc i64, position i32, raw/clamp i32.
        ("compression_locations_and_lengths", 4 * 2 * (8 + 4 + 4 + 4), "write"))))
    return {"demand": demand, "operands": [op for p in parts for op in p["operands"]],
        "source": [source for p in parts for source in p["source"]], "substeps": parts,
        "limitations": ["Standalone single-layer GPU metadata composition, not unmodified whole-backend initialization",
            "Only C4 plan and indices are built; raw C128 verify guard is not weakened",
            "Logical boundary payload priced through memory bandwidth; integer/atomic scheduling and host construction are unpriced"],
        "metadata_workspace_bytes": 4 * pages * 4 + 4 * 3 * 4 + 4 * 128 * 4 + 4 * 40,
        "source_geometry": {"global_page_tokens": 256, "swa_page_tokens": 128,
                            "live_swa_window": 128, "compute_c128_page_indices": False}}


def verify_source_work(kernel):
    family, stage = kernel.attributes["family"], kernel.attributes["stage"]
    context = kernel.shape["kv_len"]
    if kernel.attributes["source_program"] == ONLINE_HCA_PROGRAM:
        return online_hca_source_work(stage, context)
    if family == "KDA":
        from src.kernel.predictors.attention_verify_kda import verify_kda_work
        return verify_kda_work(stage, context, hardware_id=kernel.hardware_id)
    if family != "CSA":
        raise ValueError("unhandled current-source verify family")
    if stage == "verify_attention_metadata":
        return _metadata(context)
    from src.kernel.predictors.attention_verify_compressed import verify_compressed_stages, verify_compressed_work
    if stage in verify_compressed_stages(family):
        return verify_compressed_work(family, stage, context)
    if stage in ("attention_core", "combine"):
        from src.kernel.predictors.attention_reference_mla import attention_work
        work = attention_work(family, stage, context, 4, 1)
        if stage == "attention_core":
            # Useful normalized-exp baseline with a real per-head sink.
            # Sink adds a denominator score, but no value contribution.
            rows = 4 * 64
            work["operands"].append({"role": "fp32_attention_sink_per_query", "bytes": rows * 4, "access": "read"})
            work["demand"]["compute_fma"] += 4 * rows
            work["demand"]["compute_sfu"] += rows
            work["max_comparisons"] += rows
            work["attention_sink_included"] = True
        work["limitations"].append("Explicit current-source C4 causal verification; unsplit useful softmax with sink, not a selected native split")
        return work
    from src.kernel.predictors.attention_reference_indexer import indexer_work
    work = indexer_work(family, stage, context, 4, kernel.attributes["selector_passes"])
    if stage == "indexer_q_prepare":
        # Current V4 Q preparation has Hadamard after RoPE. The historical
        # reference trace selected the distinct rope-first/no-Hadamard entry.
        work["demand"]["compute_fma"] += 8 * 4 * 64 * 128
        work["geometry"]["hadamard"] = True
        work["geometry"]["rope_first"] = False
        for operand in work["operands"]:
            if operand["role"] == "rotary_cache_bf16_unique_rows":
                operand["role"] = "rotary_cache_fp32_unique_rows"
                operand["bytes"] = 4 * 64 * 4
        work["limitations"] = [item for item in work["limitations"]
            if "trace templates" not in item and "Flash assumes" not in item]
        work["limitations"].append("Current source rotary cache is FP32; unique-row payload, not inferred physical reuse")
        work["limitations"].append("Current V4 Hadamard preparation, not historical reference no-Hadamard variant")
    return work


def online_hca_source_work(stage, context):
    from src.kernel.predictors.attention_verify_online_hca import online_hca_stages, online_hca_work
    if stage in online_hca_stages():
        return online_hca_work(stage, context)
    if stage == "verify_attention_metadata":
        if type(context) is not int or context < 136:
            raise ValueError("online HCA T4 requires integer ending context >=136")
        work = _metadata(context)
        pages = -(-context // 256)
        columns = (256 // 128) * pages
        allocated = 4 * columns
        # metadata_kernel.py:135-145 allocates exactly page_size/128 times
        # page_table columns. BLOCK_SIZE=64 pads execution, not backing.
        # Its mapping load mask uses page_idx/max_pages; the causal mask is
        # applied only afterwards to output values, so every column loads.
        extra = {"stage": "c128_causal_indices", "demand": dict.fromkeys(work["demand"], 0),
            "operands": [{"role": "compressed_page_table_mapping_roles_i32", "bytes": allocated * 4, "access": "read"},
                         {"role": "c128_indices_i32", "bytes": allocated * 4, "access": "write"}],
            "source": ["sglang/v0.5.19/python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py:69-92,135-145"],
            "limitations": ["Mapping roles cover all backing columns before causal invalidation; repeated page-table roles are not deduplicated into physical HBM traffic. Address/launch service unpriced"]}
        work["substeps"].append(extra)
        work["operands"].extend(extra["operands"])
        work["demand"] = {role: sum(part["demand"][role] for part in work["substeps"])
                          for role in work["demand"]}
        work["metadata_workspace_bytes"] += allocated * 4
        work["source_geometry"]["compute_c128_page_indices"] = True
        work["source_geometry"]["c128_index_columns"] = columns
        work["limitations"] = [item for item in work["limitations"] if "Only C4 plan" not in item]
        work["limitations"].append("Distinct online C128 ring1 profile; both online flags explicit, no raw128/256 guard bypass")
        return work
    if stage not in ("attention_core", "combine"):
        raise ValueError("unknown online HCA source stage")
    from src.kernel.predictors.attention_reference_mla import attention_work
    work = attention_work("HCA", stage, context, 4, 1)
    if stage == "attention_core":
        rows = 4 * 64
        work["operands"].append({"role": "fp32_attention_sink_per_query", "bytes": rows * 4, "access": "read"})
        work["demand"]["compute_fma"] += 4 * rows
        work["demand"]["compute_sfu"] += rows
        work["max_comparisons"] += rows
        work["attention_sink_included"] = True
    work["limitations"].append("Online-C128 T4 source with sink; unsplit mathematical core, not an observed native split/provider")
    return work


def online_hca_inventory(context):
    from src.kernel.predictors.attention_verify_online_hca import online_hca_state
    state = online_hca_state(context)
    extra = {"metadata_page_table_swa_c128_indices": online_hca_source_work("verify_attention_metadata", context)["metadata_workspace_bytes"],
             "attention_output_bf16_and_lse_fp32": 4 * 64 * (512 * 2 + 4)}
    return {**state, "additional_operator_workspace_components": extra,
        "additional_operator_workspace_bytes": sum(extra.values()),
        "peak_including_operator_temporaries_bytes": state["total_bytes"] + sum(extra.values()),
        "limitations": [*state["limitations"], "Includes declared metadata/output inventory; not an observed allocator peak or model-level MTP deployment"]}


def verify_state(family, context):
    if family == "KDA":
        from src.kernel.predictors.attention_verify_kda import verify_kda_state
        return verify_kda_state()
    from src.kernel.predictors.attention_verify_compressed import verify_compressed_state
    state = verify_compressed_state(family, context)
    metadata = _metadata(context)["metadata_workspace_bytes"]
    positions = range(context - 3, context + 1)
    active_logit_bytes = sum(pos // 4 for pos in positions) * 4
    allocated_logit_columns = -(-context // 256) * 64
    extra = {"metadata_page_table_and_swa_indices": metadata,
        "rectangular_fp32_indexer_logits": 4 * allocated_logit_columns * 4,
        "prepared_fp8_index_queries_and_fp32_weights": 4 * 64 * (128 + 4),
        "topk_indices_int32": 4 * 512 * 4,
        "attention_output_bf16_and_lse_fp32": 4 * 64 * (512 * 2 + 4)}
    return {**state, "additional_operator_workspace_components": extra,
        "additional_operator_workspace_bytes": sum(extra.values()),
        "indexer_logits": {"active_role_bytes": active_logit_bytes,
                           "allocated_rows": 4, "allocated_columns": allocated_logit_columns,
                           "allocated_bytes": extra["rectangular_fp32_indexer_logits"]},
        "peak_including_operator_temporaries_bytes": state["total_bytes"] + sum(extra.values()),
        "limitations": [*state["limitations"],
            "Added indexer/attention/metadata buffers are an explicit simultaneous inventory; a conservative operator workspace budget, not a measured liveness peak or whole-model admission"]}
