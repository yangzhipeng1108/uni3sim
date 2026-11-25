"""Independent useful V4 indexer math and typed stage payloads."""


def indexer_baseline_reference(raw, requests, mode):
    if mode not in ("prefill", "decode"):
        raise ValueError("V4 indexer source phase must be prefill or decode")
    heads, width, capacity = raw["index_n_heads"], raw["index_head_dim"], raw["index_topk"]
    ends = [[position // 4 for position in range(request["computed"] + 1,
              request["computed"] + request["query"] + 1)] for request in requests]
    pairs = sum(end for sequence in ends for end in sequence)
    rows = sum(request["query"] for request in requests)
    empty_prefill = mode == "prefill" and all((request["computed"] + request["query"]) // 4 == 0
        for request in requests)
    selection = None
    selection_pairs = pairs
    if mode == "prefill":
        shortcuts = [len([end for end in sequence if end <= capacity]) for sequence in ends]
        selection_pairs = sum(end for sequence in ends for end in sequence if end > capacity)
        selection = {"source": "vllm_0_30_top_k_per_row_prefill",
            "shortcut_rows_per_request": shortcuts, "nontrivial_rows": rows - sum(shortcuts),
            "shortcut_score_read_values": 0, "nontrivial_useful_score_values": selection_pairs,
            "shortcut_indices": "0..rowLen-1 followed by -1 to topk_capacity; not score-sorted",
            "shortcut_ignores_score_values_nan_and_ties": True,
            "row_bound_dtype": "int32", "row_bound_elements": 2 * rows,
            "refinement_passes_and_scratch": ({"passes": 0, "global_scratch_elements": 0,
                "declared_shared_storage_accessed": False} if selection_pairs == 0 else None),
            "compiler_integer_instruction_cost_priced": False, "actual_provider_bound": False}

    def demand(tensor, scalar):
        return {"compute_tensor": tensor, "compute_vector": 0, "compute_fma": scalar,
                "compute_alu": 0, "compute_reduction": 0, "compute_sfu": 0}

    def typed(role, dtype, count, access):
        return {"role": role, "dtype": dtype, "elements": count,
                "bytes": count * {"fp8": 1, "uint8": 1, "fp32": 4, "int32": 4}[dtype], "access": access}

    stages = [{"stage": "topk_clear", "demand": demand(0, 0), "operands": [
        typed("topk_indices", "int32", rows * capacity, "write")]},
        {"stage": "scores", "demand": demand(2 * pairs * heads * width, 2 * pairs * heads), "operands": [
            typed("prepared_q", "fp8", rows * heads * width, "read"),
            typed("key_role_uses", "fp8", pairs * width, "read"),
            typed("key_scale_role_uses", "fp32", pairs, "read"),
            typed("folded_weights", "fp32", rows * heads, "read"),
            typed("active_scores", "fp32", pairs, "write")]},
        {"stage": "select", "demand": demand(0, 0), "operands": [
            typed("active_scores", "fp32", selection_pairs, "read"),
            typed("topk_indices", "int32", rows * capacity, "write")]}]
    if selection is not None:
        stages[-1]["operands"].insert(0, typed("row_starts_and_ends", "int32", 2 * rows, "read"))
    if mode == "prefill" and all((request["computed"] + request["query"]) // 4 > 0 for request in requests):
        score_metadata = {"role": "score_row_starts_and_ends", "dtype": "int32", "elements": 2 * rows,
            "bytes": rows * 8, "scope": "unique score-input payload over all query subchunks; separate from selector reads"}
    elif mode == "decode" and all(request["query"] == 1 for request in requests):
        score_metadata = {"role": "score_effective_context_lengths", "dtype": "int32", "elements": rows,
            "bytes": rows * 4, "scope": "one effective context length per unpadded single-token query; not block tables or schedule metadata"}
    else:
        score_metadata = None
    if score_metadata is not None:
        stages[1]["operands"].append(typed(score_metadata["role"], "int32", score_metadata["elements"], "read"))
    # Request grouping changes the sum of rectangular tensor areas; splitting
    # queries inside an existing group does not. These are semantic tensor
    # extents, never an assertion of contiguous backing or initialized cells.
    own_rectangles = sum(request["query"] * ((request["computed"] + request["query"]) // 4) for request in requests)
    full_rectangle = rows * sum((request["computed"] + request["query"]) // 4 for request in requests)
    materialization = {
        "dtype": "fp32", "element_bytes": 4, "clean_logits": False,
        "producer": "fp8_fp4_mqa_logits" if mode == "prefill" else "fp8_fp4_paged_mqa_logits",
        "consumer": "top_k_per_row_prefill" if mode == "prefill" else "get_indexer_topk(topk_backend)",
        "consumer_uses_same_tensor": True, "score_cast_or_copy_before_selection": False,
        "shape_rule": "[chunk_query_rows, chunk_total_compressed_context_rows]" if mode == "prefill" else "[batch * next_n, runtime_indexer_max_model_len]",
        "query_subchunks_keep_group_context_extent": True if mode == "prefill" else None,
        "total_semantic_output_elements": own_rectangles if mode == "prefill" and len(requests) == 1 else None,
        "total_semantic_output_element_bounds": {"lower": own_rectangles, "upper": full_rectangle} if mode == "prefill" else None,
        "runtime_indexer_max_model_len": None, "selected_chunk_partition": None,
        "physical_strides_and_backing_bytes": None, "inactive_cell_store_count": None,
        "unfilled_cell_cleanup_requested": False, "allocation_elements_counted_as_traffic": False,
        "actual_allocator_binding_verified": False,
        "scope": "wrapper logical tensor shape/FP32 handoff; inactive stores, physical backing, lifetime overlap and provider remain unbound"}
    gather = None
    if mode == "prefill" and not empty_prefill:
        counts = [(request["computed"] + request["query"]) // 4 for request in requests]
        total = sum(counts)
        gather = {"copied_rows_per_request": counts, "total_copied_rows": total,
            "packed_cache_row_bytes": width + 4, "query_subchunks_repeat_gather": False,
            "values_dtype": "fp8", "scale_backing_dtype": "uint8", "scale_bytes_per_row": 4,
            "scale_consumer_view_dtype": "fp32", "scale_consumer_view_allocates": False,
            "grouping_and_peak_live_rows": None, "workspace_allocation_extent": None,
            "metadata_reads_and_integer_address_work_complete": False,
            "source_order": ["paged_cache_copy", "scores_read_contiguous_values_and_scale_view"],
            "actual_provider_bound": False}
        stages.insert(1, {"stage": "prefill_gather", "demand": demand(0, 0), "operands": [
            typed("packed_indexer_cache_role_uses", "uint8", total * (width + 4), "read"),
            typed("gathered_keys", "fp8", total * width, "write"),
            typed("gathered_scale_bytes", "uint8", total * 4, "write")]})
    if empty_prefill:
        stages = [{"stage": stage, "demand": demand(0, 0), "operands": [
            typed("topk_indices", "int32", rows * capacity, "write")]}
            for stage in ("topk_clear", "empty_prefill_topk_fill")]
        selection = {"source": "vllm_0_30_empty_prefill_group", "selector_invoked": False,
            "rows": rows, "refinement_passes_and_scratch": {"passes": 0, "global_scratch_elements": 0},
            "actual_provider_bound": False}
        materialization.update({"producer": "q_slice.new_empty", "consumer": None,
            "consumer_uses_same_tensor": False, "total_semantic_output_elements": 0, "payload_allocation_bytes": 0,
            "scope": "empty FP32 [query_rows,0] descriptor; no score/selector and no payload allocation"})
    for stage in stages:
        stage["read_write_bytes"] = {access: sum(value["bytes"] for value in stage["operands"]
                                                  if value["access"] == access) for access in ("read", "write")}
    byte_account = {access: sum(stage["read_write_bytes"][access] for stage in stages) for access in ("read", "write")}
    remaining = [
        "score_rectangular_allocation_stride_padding_and_inactive_rows",
        "metadata_page_tables_and_schedule_buffers"] + (
            [] if selection is not None and selection_pairs == 0 else
            ["selection_backend_value_dependent_refinement_and_scratch"])
    return {"compressed_row_ends": ends, "active_pairs": pairs,
        "arithmetic": {"qk_dot_flops": 2 * pairs * heads * width, "relu_comparisons": pairs * heads,
            "head_weight_multiplications": pairs * heads, "head_sum_additions": pairs * (heads - 1),
            "key_scale_multiplications": pairs,
            "topk_selection_comparisons": 0 if mode == "prefill" and selection_pairs == 0 else None},
        "stages": stages, "demand": demand(2 * pairs * heads * width, 2 * pairs * heads),
        "prefill_gather": gather, "prefill_selection": selection,
        "score_metadata": score_metadata, "score_materialization": materialization,
        "read_write_bytes": byte_account, "logical_bytes": sum(byte_account.values()),
        "source_semantics": {"score_dtype": "fp32", "index_dtype": "int32", "sentinel": -1,
            "clear_and_selection_write_same_buffer": True, "q_scale_folded_into_weights": True,
            "context_parallel_size": 1,
            "key_scale_is_positive_quantization_scale": True,
            "relu_and_integer_selection_instruction_cost_priced": False, "selected_topk_backend": None,
            "actual_provider_bound": False, "finite_precision_equivalence_asserted": False},
        "remaining_source_services": remaining,
        "scope": "serial useful mathematical FLOPs/role-use bytes; not full indexer source/native acceptance or a hardware latency bound"}
