"""Independent pure-text CUDA-hash/Triton routing source accounts.

Expected values use config and declared rows, never production formulas.
These are scalar source groups and typed logical payloads, not a claim that
one source expression maps to one hardware instruction or a DRAM request.
"""


def routing_tensor_reference(rows, experts, hashed):
    """Independent Python allocations and no-copy router return semantics."""
    return {"condition": "modular pure-text top6, EP-off, no external capture hook, INT32 expert IDs",
        "logits": {"dtype": "fp32", "shape": [rows, experts], "read_only": True},
        "hidden_states_payload_read": False,
        "outputs": [{"role": role, "dtype": dtype, "shape": [rows, 6], "strides_elements": [6, 1],
                     "fresh": True, "aliases_inputs_or_each_other": False}
                    for role, dtype in (("topk_weights", "fp32"), ("topk_ids", "int32"))],
        "unused_temporary": ({"role": "token_expert_indices", "dtype": "int32", "shape": [rows, 6],
            "logical_storage_bytes": 4 * rows * 6, "initialized": False, "read_bytes": 0, "write_bytes": 0,
            "passed_to_selected_kernel": False, "returned": False, "lifetime": "wrapper call; not returned"} if hashed else None),
        "input_id_cast": ({"input_dtype": "int32", "hash_table_dtype": "int32", "allocates": False,
                           "kernel_widening": "register int64; no global copy"} if hashed else None),
        "unused_input_id_argument": None if hashed else {"dtype": "int32", "shape": [rows], "read_by_kernel": False},
        "bias_view": None if hashed else {"dtype": "fp32", "shape": [experts], "data_aliases_parameter": True},
        "postprocess": {"eplb_mapping": False, "ids_dtype_copy": False, "fused_shared_columns": 0, "capture_hook": None, "routing_replay_copy": False},
        "kernel_global_tensor_temporaries": [], "actual_provider_or_allocator_bound": False,
        "scope": "declared source route objects and returned buffers, not checkpoint assignments, compiled spills or runtime dispatch"}


def routing_source_reference(raw, rows, hashed):
    experts, k = raw["n_routed_experts"], raw["num_experts_per_tok"]
    if k != 6 or experts not in (256, 384):
        raise ValueError("V4 source routing requires the upstream top6 expert domain")
    tile = 1 << (experts - 1).bit_length()
    actions = {}

    def typed(name, dtype, operation, per_row):
        actions[name] = {"dtype": dtype, "operation": operation, "elements": rows * per_row}

    # Count per-token source expressions, then scale by live token rows.
    if hashed:
        groups = {
            "compute_fma": {"stable_softplus_additions": 2 * k, "output_scale_multiply": k,
                            "output_divide": k, "full_warp_sum_lane_additions": 32 * 5},
            "compute_sfu": {"exp": k, "log": k, "sqrt": k},
            "compute_reduction": {}, "compute_alu": {}}
        for name, operation in (("softplus_max", "maximum"), ("softplus_abs", "absolute"),
                ("softplus_negate", "negate"), ("nan_test", "isnan"),
                ("nan_replacement_site", "conditional_zero_assignment"),
                ("denominator_compare", "greater_than"), ("denominator_select", "select")):
            typed(name, "fp32", operation, k)
        conditional = {"nan_replacement_site": {"condition": "isnan(weight)",
            "active_elements": None, "active_element_bounds": [0, rows * k],
            "bound_to_input_values": False}}
        inputs = [("input_ids", "int32", rows), ("selected_table_entries", "int32", rows * k),
                  ("selected_logits", "fp32", rows * k)]
        program = {
            "softplus": "sqrt(max(x,0)+log(1+exp(-abs(x)))); NaN weight replaced by zero",
            "selection": "six checkpoint tid2eid entries; uniqueness and expert assignment unbound",
            "normalization": "weight * scale / (weight_sum > 0 ? weight_sum : 1)",
            "reduction": "five XOR shuffle/add stages on every lane of each live token warp",
            "bias_affects_exported_weight": False, "input_id_widening": "register int64, no global copy"}
        source = "vllm_0_30_cuda_hash_top6_pure_text"
    else:
        groups = {
            "compute_fma": {"exp_plus_one": tile, "bias_add": tile, "padded_output_scale_multiply": 8,
                            "normalization_divide": 1, "selected_weight_gather_additions": k * (tile - 1),
                            "eight_slot_weight_sum": 7},
            "compute_sfu": {"exp": tile, "log": tile, "sqrt": tile},
            "compute_reduction": {}, "compute_alu": {}}
        for name, dtype, operation, count in (
                ("softplus_threshold_compare", "fp32", "greater_than", tile),
                ("softplus_threshold_select", "fp32", "select", tile),
                ("expert_mask_compare", "int32", "less_than", tile),
                ("expert_mask_select", "fp32", "select", tile),
                ("nan_test", "fp32", "equal_self", tile),
                ("nan_select", "fp32", "select", tile),
                ("selection_max_tree", "fp32", "maximum", k * (tile - 1)),
                ("candidate_compare", "fp32", "equal", k * tile),
                ("candidate_select", "int32", "select", k * tile),
                ("selection_min_tree", "int32", "minimum", k * (tile - 1)),
                ("gather_compare", "int32", "equal", k * tile),
                ("gather_select", "fp32", "select", k * tile),
                ("slot_compare", "int32", "equal", k * 8),
                ("selected_weight_select", "fp32", "select", k * 8),
                ("selected_id_select", "int32", "select", k * 8),
                ("invalidate_compare", "int32", "equal", k * tile),
                ("invalidate_select", "fp32", "select", k * tile),
                ("denominator_compare", "fp32", "greater_than", 1),
                ("denominator_select", "fp32", "select", 1),
                ("output_mask", "int32", "less_than", 8)):
            typed(name, dtype, operation, count)
        conditional = {}
        inputs = [("logits", "fp32", rows * experts), ("bias_per_row", "fp32", rows * experts)]
        program = {
            "softplus": "sqrt(where(x>20,x,log(1+exp(x)))); padded expressions retained",
            "selection": "six max/min/gather rounds; ties select smallest expert id; selected scores set to -inf",
            "normalization": "selected_weights *= scale / where(weight_sum>0,weight_sum,1)",
            "reduction": "three padded-block trees per slot and one eight-slot sum",
            "bias_affects_exported_weight": False,
            "nan_selection_score": "NaN biased score replaced by -1e30; original weights retained"}
        source = "vllm_0_30_triton_top6_pure_text"
    groups = {role: {name: count * rows for name, count in events.items()} for role, events in groups.items()}
    operands = [{"role": role, "dtype": dtype, "elements": count, "bytes": 4 * count, "access": "read"}
                for role, dtype, count in inputs]
    operands += [{"role": role, "dtype": dtype, "elements": rows * k, "bytes": 4 * rows * k, "access": "write"}
                 for role, dtype in (("weights", "fp32"), ("ids", "int32"))]
    requested_read = sum(value["bytes"] for value in operands if value["access"] == "read")
    compulsory_read = requested_read if hashed else 4 * experts * (rows + 1)
    write = 8 * rows * k
    return {"source": source, "arithmetic_groups": groups,
            "unpriced_typed_actions": actions, "conditional_action_counts": conditional,
            "source_action_count_scope": "source expression extents and full logical reduction trees; conditional sites are not dynamic executions or compiled instructions",
            "demand": {"compute_tensor": 0, **{role: sum(events.values()) for role, events in groups.items()}},
            "operands": operands, "source_program": program,
            "source_role_read_write_bytes": {"read": requested_read, "write": write},
            "baseline_read_write_bytes": {"read": compulsory_read, "write": write},
            "baseline_read_scope": ("selected role uses; selected table/logit address aliasing unbound" if hashed
                                    else "complete logits matrix plus one bias parameter object; not observed DRAM or proven cache reuse"),
            "logical_bytes": compulsory_read + write, "actual_provider_bound": False,
            "numerical_equivalence_between_backends_asserted": False,
            "scope": "declared source expressions and typed payloads, not native instructions or observed memory traffic"}
