"""Independent conditional Marlin scratch census; no production imports."""

from functools import lru_cache


def marlin_weight_loads_reference(source, histogram):
    """Count source-global parameter visits independently of cache outcomes."""
    if (not histogram or any(type(count) is not int or count < 0 for count in histogram)
            or sum(histogram) != source["routed_rows"]):
        raise ValueError("Marlin weight-load reference requires a valid declared histogram")
    block = source["routing_alignment"]["block_size_m"]
    blocks = sum((count + block - 1) // block for count in histogram)
    active = sum(count > 0 for count in histogram)
    views = {view["role"]: view["shape"] for view in source["source_views"]}
    h, i = views["weighted_expert_output"][1], views["activated"][1]
    stages = []
    for name, values in (("w13", 2 * h * i), ("w2_with_router_weight", i * h)):
        stages.append({"stage": name,
            "distinct_packed_parameter_bytes": active * values // 2,
            "source_global_packed_weight_read_bytes": blocks * values // 2,
            "distinct_group_scale_parameter_bytes": active * values // 32,
            "source_global_scale_read_bytes": blocks * values // 32})
    return {"declared_M_blocks": blocks, "declared_active_experts": active, "stages": stages,
        "source_global_visits_used_for_prediction": True,
        "projection": "source-global weight/scales reads; cache reuse and physical transaction amplification unmodeled",
        "physical_traffic_or_cache_outcomes_bound": False}


def marlin_dequant_work_reference(source, histogram):
    """Count register dequant sites; group-scale decode repeats across lanes.

    Two scale-decode calls produce8 BF16 values and eight E2M1-decode
    calls produce32 values per thread/matmul iteration. The original
    thread-count*iteration-count is N*K/32 per matrix M-block.
    """
    if (not histogram or any(type(count) is not int or count < 0 for count in histogram)
            or sum(histogram) != source["routed_rows"]):
        raise ValueError("Marlin dequant reference requires a valid declared histogram")
    block = source["routing_alignment"]["block_size_m"]
    blocks = sum((count + block - 1) // block for count in histogram)
    views = {view["role"]: view["shape"] for view in source["source_views"]}
    h, i = views["weighted_expert_output"][1], views["activated"][1]
    stages = []
    for name, nk in (("w13", h * 2 * i), ("w2_with_router_weight", i * h)):
        weights = blocks * nk
        stages.append({"stage": name, "weight_decode_values": weights,
            "bf16_bias_multiply_values": weights, "bf16_group_scale_multiply_values": weights,
            "source_scale_decode_values": weights // 4,
            "unique_logical_group_scale_uses": weights // 32})
    return {"declared_M_blocks": blocks, "stages": stages,
        "counts_are_source_sites_not_compiler_instructions": True, "added_global_tensor_bytes": 0,
        "mapped_resource_role": None, "native_issue_service_priced": False,
        "compiler_spill_or_actual_traffic_bound": False}


def marlin_tensor_work_reference(source, histogram):
    """Count all source BF16 MMA sites, including per-expert M tails.

    The M8 path transposes operands for one m16n8k16 instruction, while
    the normal M16 path issues two. Both compute exactly the declared
    logical M-block; M8 does not acquire an extra factor of two.
    """
    if (not histogram or any(type(count) is not int or count < 0 for count in histogram)
            or sum(histogram) != source["routed_rows"]):
        raise ValueError("Marlin tensor reference requires a valid declared histogram")
    block = source["routing_alignment"]["block_size_m"]
    rows = sum(((count + block - 1) // block) * block for count in histogram)
    views = {view["role"]: view["shape"] for view in source["source_views"]}
    h, i = views["weighted_expert_output"][1], views["activated"][1]
    stages = []
    for name, per_row in (("w13", 4 * h * i), ("w2_with_router_weight", 2 * i * h)):
        useful, issued = source["routed_rows"] * per_row, rows * per_row
        stages.append({"stage": name, "useful_tensor_flops": useful,
            "source_tensor_flops": issued, "padding_tensor_flops": issued - useful})
    return {"live_rows": source["routed_rows"], "source_padded_rows": rows, "source_block_m": block,
        "M8_uses_transposed_operands_without_extra_row_padding": True,
        "stages": stages, "actual_routing_or_compiled_provider_bound": False}


@lru_cache(maxsize=256)
def _tail_slice_visits(m_blocks, n_tiles, k_tiles, ctas):
    """Sweep each CTA's linear K interval, rather than count tile boundaries."""
    global_tiles = m_blocks * n_tiles
    tail = global_tiles
    if global_tiles > ctas:
        remainder = global_tiles % ctas
        tail = remainder + ctas if remainder <= ctas // 3 else remainder
    length = (tail * k_tiles + ctas - 1) // ctas
    visits = [0] * tail
    for block in range(ctas):
        start = block * length
        stop = min(start + length, tail * k_tiles)
        if start >= stop:
            continue
        for tile in range(start // k_tiles, (stop - 1) // k_tiles + 1):
            visits[tile] += 1
    return global_tiles - tail, tuple(visits)


def marlin_metadata_reference(live, block, n_tiles, k_tiles, ctas, threads, weighted):
    """Independent CTA/tile traversal of init_slice/read_moe_block_data.

    Visit tiles, rather than derive the first/last M range of each stripe.
    Every DP tile reloads; adjacent tail tiles on one CTA reuse the header.
    """
    offset, _ = _tail_slice_visits(len(live), n_tiles, k_tiles, ctas)
    tail_iterations = (len(live) * n_tiles - offset) * k_tiles
    stripe = (tail_iterations + ctas - 1) // ctas
    updates = [0] * len(live)
    for cta in range(ctas):
        for tile in range(cta, offset, ctas):
            updates[tile // n_tiles] += 1
        previous_m = None
        position = cta * stripe
        end = min(position + stripe, tail_iterations)
        while position < end:
            column = position // k_tiles
            m = (offset + column) // n_tiles
            if m != previous_m:
                updates[m] += 1
                previous_m = m
            position = min((column + 1) * k_tiles, end)
    headers, slots = sum(updates), block * sum(updates)
    return {"header_updates_by_m_block": updates, "header_updates": headers,
        "sorted_ids_source_global_read_bytes": 4 * slots,
        "topk_weights_source_global_read_bytes": 4 * slots if weighted else 0,
        "padding_weight_zero_index_reads": sum(count * (block - rows) for count, rows in
            zip(updates, live, strict=True)) if weighted else 0,
        "sorted_id_validity_comparisons": slots,
        "weight_slot_validity_comparisons": slots if weighted else 0,
        "route_to_activation_integer_division_sites": slots,
        "expert_id_uniform_role_read_bytes": 4 * headers,
        "expert_id_source_lane_read_bytes": 4 * threads * headers,
        "post_pad_uniform_entry_read_bytes": 4 * ctas,
        "post_pad_source_lane_entry_read_bytes": 4 * threads * ctas,
        "source_reads_are_physical_DRAM": False, "counts_used_for_point_prediction": False}


def marlin_scratch_bounds_reference(source, histogram, sm_count):
    """Conditional on declared histogram, not checkpoint assignments or launch.

    Keep every potentially selected auto-config. A missing numRegs/kernel
    availability fact may shrink this superset, never justify choosing one.
    """
    if (type(sm_count) is not int or sm_count <= 0 or not histogram
            or any(type(count) is not int or count < 0 for count in histogram)
            or sum(histogram) != source["routed_rows"]):
        raise ValueError("conditional Marlin reference requires a valid declared histogram and SM count")
    block = source["routing_alignment"]["block_size_m"]
    live = []
    for count in histogram:
        remaining = count
        while remaining:
            rows = min(remaining, block)
            live.append(rows)
            remaining -= rows
    views = {view["role"]: view["shape"] for view in source["source_views"]}
    hidden, width = views["weighted_expert_output"][1], views["activated"][1]
    templates = {template["stage"]: template for template in source["fp32_scratch_templates"]}
    candidates = ((128, 128), (64, 128), (128, 64)) if block in (8, 16) else ((64, 256), (64, 128), (128, 64))
    stages = []
    for stage, n, k in (("w13", 2 * width, hidden), ("w2_with_router_weight", hidden, width)):
        coefficients = templates[stage]["fp32_values_per_n_column_by_valid_rows"]
        configurations = []
        for tk, tn in candidates:
            if n % tn or k % tk:
                continue
            slot_bound = (n // tn) * source["routed_rows"] * 4 // sm_count
            limit = min(4 if block in (8, 16) else 2, max(slot_bound, 1))
            for resident in range(1, limit + 1):
                ctas = sm_count * resident
                offset, visits = _tail_slice_visits(len(live), n // tn, k // tk, ctas)
                additions = sum((visits[index] - 1) * tn * coefficients[live[(offset + index) // (n // tn)]]
                                for index in range(len(visits)))
                # Count the shared reduction's actual source addition sites:
                # inner writers combine two shared operands with their own
                # accumulator; group zero then adds one shared operand.
                threads = {(128, 128): 256, (64, 128): 128, (128, 64): 128, (64, 256): 256}[tk, tn]
                group_width = tn // 2
                first = threads // group_width // 2
                per_slice_adds = 0
                step = first
                values = 4 * (4 if block == 8 else 8) * ((block + 15) // 16)
                while step:
                    if step < first:
                        per_slice_adds += step * group_width * values * 2
                    step //= 2
                if first:
                    per_slice_adds += group_width * values
                slices = offset + sum(visits)
                configurations.append({"thread_k": tk, "thread_n": tn, "grid_blocks": ctas,
                    "read_bytes": additions * 4, "write_bytes": additions * 4, "fp32_additions": additions,
                    "extra_partial_count": sum(count - 1 for count in visits),
                    "metadata_source_work": marlin_metadata_reference(live, block, n // tn, k // tk,
                        ctas, threads, stage == "w2_with_router_weight"),
                    "tile_source_work": {"threads": threads, "source_matrix_slices": slices,
                        "activation_global_read_bytes": sum(2 * rows * k for rows in live) * (n // tn),
                        "threadblock_fp32_additions": slices * per_slice_adds,
                        "activation_read_is_physical_DRAM": False,
                        "shared_reduction_counts_padded_rows": True,
                        "counts_used_for_point_prediction": False}})
        if not configurations:
            raise ValueError("independent Marlin source configuration domain is empty")
        stages.append({"stage": stage, "configuration_superset": configurations,
            "bounds": {field: {"lower": min(item[field] for item in configurations),
                               "upper": max(item[field] for item in configurations)}
                       for field in ("read_bytes", "write_bytes", "fp32_additions")}})
    return {"stages": stages, "declared_routing_is_checkpoint_assignment": False,
        "selected_source_configuration_bound": False, "bounds_used_for_prediction": False,
        "source_domain": "BF16/MXFP4 group32, valid histogram; auto candidate tiles and possible blocks_per_sm; kernel availability/numRegs/shared-memory selection remain unbound",
        "scope": "conditional source access census and enclosing bounds, not point traffic, duration, peak allocation or native provider acceptance"}
