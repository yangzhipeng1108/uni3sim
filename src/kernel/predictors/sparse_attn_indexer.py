"""Source-work analytical prediction for the DSA indexer pipeline."""

import math

from src.kernel.formula import PerformanceFormula
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.common import (
    build_aggregate_gpu_input,
    build_forward_demand,
    causal_attention_pairs,
    prediction_from_ns,
    roofline_ns,
    source_stage_roofline_ns,
)


def _indexer_arithmetic(pairs, heads, dim):
    return {"qk_dot_flops": 2 * pairs * heads * dim, "relu_comparisons": pairs * heads,
        "head_weight_multiplications": pairs * heads, "head_sum_additions": pairs * (heads - 1),
        "key_scale_multiplications": pairs, "topk_selection_comparisons": None}


def _indexer_demand(tensor=0, scalar=0):
    return {"compute_tensor": tensor, "compute_vector": 0, "compute_fma": scalar,
        "compute_alu": 0, "compute_reduction": 0, "compute_sfu": 0}


def _indexer_operand(role, dtype, count, access):
    width = {"fp8": 1, "uint8": 1, "fp32": 4, "int32": 4}[dtype]
    return {"role": role, "dtype": dtype, "elements": count, "bytes": width * count, "access": access}


def _prefill_gather_work(counts, dim):
    """CP1 request groups partition K; later query slices skip its gather."""
    copied_rows = sum(counts)
    gather = {"copied_rows_per_request": counts, "total_copied_rows": copied_rows,
        "packed_cache_row_bytes": dim + 4, "query_subchunks_repeat_gather": False,
        "values_dtype": "fp8", "scale_backing_dtype": "uint8", "scale_bytes_per_row": 4,
        "scale_consumer_view_dtype": "fp32", "scale_consumer_view_allocates": False,
        "grouping_and_peak_live_rows": None, "workspace_allocation_extent": None,
        "metadata_reads_and_integer_address_work_complete": False,
        "source_order": ["paged_cache_copy", "scores_read_contiguous_values_and_scale_view"],
        "actual_provider_bound": False}
    stage = {"stage": "prefill_gather", "demand": _indexer_demand(), "operands": [
        _indexer_operand("packed_indexer_cache_role_uses", "uint8", copied_rows * (dim + 4), "read"),
        _indexer_operand("gathered_keys", "fp8", copied_rows * dim, "write"),
        _indexer_operand("gathered_scale_bytes", "uint8", copied_rows * 4, "write")]}
    return gather, stage


def _prefill_selection_work(shortcuts, score_values, rows):
    """sampler.cu's unsplit topKPerRowJob reads no logits for rowLen<=K."""
    return {"source": "vllm_0_30_top_k_per_row_prefill",
        "shortcut_rows_per_request": shortcuts, "nontrivial_rows": rows - sum(shortcuts),
        "shortcut_score_read_values": 0, "nontrivial_useful_score_values": score_values,
        "shortcut_indices": "0..rowLen-1 followed by -1 to topk_capacity; not score-sorted",
        "shortcut_ignores_score_values_nan_and_ties": True,
        "row_bound_dtype": "int32", "row_bound_elements": 2 * rows,
        "refinement_passes_and_scratch": ({"passes": 0, "global_scratch_elements": 0,
            "declared_shared_storage_accessed": False} if score_values == 0 else None),
        "compiler_integer_instruction_cost_priced": False, "actual_provider_bound": False}


def dsa_indexer_logical_work(kernel: KernelSpec):
    """Uncompressed CP1 decode/forced-MQA useful score and selection payload.

    The declared mathematical program requires distinct valid causal top-k
    cardinality; it does not derive IDs or population from finite logits.
    Norm/RoPE owns the clear in V32's skip_topk_buffer_clear=True call.
    Score materialization precedes selection, which has no log2(K) proxy.
    This is not complete selector service, backing extent or actual traffic.
    """
    if (kernel.kind is not KernelKind.SPARSE_ATTN_INDEXER or kernel.variant not in (
            "dsa_indexer_decode", "dsa_indexer_prefill", "dsa_indexer_mixed")
            or kernel.attributes["mathematical_program"] not in (
                "dsa_weighted_relu_topk_v1", "dsa_weighted_relu_force_mqa_v1", "dsa_weighted_relu_dense_mha_v1", "dsa_weighted_relu_specdecode_v1")):
        raise ValueError("DSA mathematical work requires its explicit weighted-ReLU source program")
    lengths, queries = kernel.attributes["kv_lens"], kernel.attributes["q_lens"]
    rows, capacity = sum(queries), kernel.attributes["topk_capacity"]
    pairs = causal_attention_pairs(queries, lengths)
    heads, dim = kernel.shape["num_query_heads"], kernel.shape["head_dim"]
    arithmetic = _indexer_arithmetic(pairs, heads, dim)
    selected = causal_attention_pairs(queries, lengths, capacity=capacity)
    operand, demand = _indexer_operand, _indexer_demand
    prefill = [(q, kv) for q, kv, phase in zip(queries, lengths, kernel.attributes["request_phases"], strict=True)
        if phase == "prefill"]
    prefill_rows = sum(q for q, _ in prefill)
    selection_pairs, selection, gather, gather_stage = pairs, None, None, None
    sentinel_elements = 0
    if prefill:
        # Each request owns one K group. Query subchunks only slice Q and
        # reuse that group; its unique data-copy payload needs no budget.
        gather, gather_stage = _prefill_gather_work([kv for _, kv in prefill], dim)
        shortcuts = [max(0, min(q, capacity - kv + q)) for q, kv in prefill]
        skipped_pairs = sum(short * (2 * (kv - q + 1) + short - 1) // 2
            for (q, kv), short in zip(prefill, shortcuts, strict=True))
        prefill_pairs = causal_attention_pairs(tuple(q for q, _ in prefill), tuple(kv for _, kv in prefill))
        selection_pairs -= skipped_pairs
        selection = _prefill_selection_work(shortcuts, prefill_pairs - skipped_pairs, prefill_rows)
        selected_prefill = causal_attention_pairs(tuple(q for q, _ in prefill),
            tuple(kv for _, kv in prefill), capacity=capacity)
        sentinel_elements = prefill_rows * capacity - selected_prefill
        if prefill_rows == rows and selection_pairs == 0:
            arithmetic["topk_selection_comparisons"] = 0
    stages = []
    if not kernel.attributes["skip_topk_buffer_clear"]:
        stages.append({"stage": "topk_clear", "demand": demand(), "operands": [
            operand("topk_indices", "int32", rows * capacity, "write")]})
    if gather_stage is not None:
        stages.append(gather_stage)
    score_metadata = ([operand("score_row_starts_and_ends", "int32", 2 * prefill_rows, "read")]
        if prefill_rows else [])
    if rows > prefill_rows:
        score_metadata.append(operand("score_effective_context_lengths", "int32", rows - prefill_rows, "read"))
    select_metadata = ([operand("row_starts_and_ends", "int32", 2 * prefill_rows, "read")]
        if prefill_rows else [])
    select_padding = ([operand("prefill_sentinel_padding", "int32", sentinel_elements, "write")]
        if sentinel_elements else [])
    stages.extend((
        {"stage": "scores", "demand": demand(arithmetic["qk_dot_flops"],
            arithmetic["head_weight_multiplications"] + arithmetic["head_sum_additions"]
            + arithmetic["key_scale_multiplications"]), "operands": [
                operand("prepared_q", "fp8", rows * heads * dim, "read"),
                operand("key_role_uses", "fp8", pairs * dim, "read"),
                operand("key_scale_role_uses", "fp32", pairs, "read"),
                operand("folded_weights", "fp32", rows * heads, "read"),
                *score_metadata,
                operand("active_scores", "fp32", pairs, "write")]},
        {"stage": "select_useful_payload", "demand": demand(), "operands": [
            *select_metadata,
            operand("active_scores", "fp32", selection_pairs, "read"),
            *select_padding,
            operand("selected_indices", "int32", selected, "write")]},
    ))
    for stage in stages:
        stage["read_write_bytes"] = {access: sum(op["bytes"] for op in stage["operands"] if op["access"] == access)
            for access in ("read", "write")}
    return {"active_pairs": pairs, "arithmetic": arithmetic, "stages": stages,
        "prefill_gather_chunks_bound": False if "prefill" in kernel.attributes["request_phases"] else None,
        "prefill_gather": gather, "prefill_selection": selection,
        "prefill_gather_materialization_bytes": (None if gather_stage is None else
            sum(gather_stage["read_write_bytes"].values())),
        "selection_cardinality": "declared_causal_topk_not_observed",
        "topk_clear_owned_by_caller": kernel.attributes["skip_topk_buffer_clear"],
        "logical_bytes": sum(sum(stage["read_write_bytes"].values()) for stage in stages),
        "remaining_source_services": ("selector_comparison_refinement_and_scratch_unbound",
            "prefill_chunk_partition_and_full_workspace_unbound",
            "score_backing_and_inactive_cells_unbound", "block_table_and_schedule_metadata_unbound",
            "logical_role_uses_not_observed_physical_traffic", "actual_provider_and_launch_unbound")}


def dsv4_indexer_logical_work(kernel: KernelSpec):
    """Useful mathematical/typed stages, not an implemented top-k schedule.

    vLLM 0.30 tests/kernels/test_deepseek_v4_cpu_kernels.py's independent
    _reference_paged_mqa_logits specifies ReLU(dot)*weight, sum(heads), then
    KV scale. Its CUDA attention references specify the same useful weighted
    ReLU sum, with scale prepared separately. No finite-precision/provider
    equivalence is asserted. sparse_attn_indexer.py clears the INT32 top-k
    output before computing/consuming FP32 scores. The source dispatch has
    cooperative/persistent/per-row and optional backends, not log2(K) rounds.
    CP1 prefill gather payload is known independently of query subchunking.
    Gather metadata, workspace extent, scratch and value-dependent selection
    services remain explicit gaps outside this useful byte baseline.
    """
    if kernel.kind is not KernelKind.SPARSE_ATTN_INDEXER or kernel.variant not in (
            "dsv4_indexer_prefill", "dsv4_indexer_decode"):
        raise ValueError("V4 indexer work requires its sparse indexer contract")
    h, d, ratio = (kernel.shape[name] for name in ("num_query_heads", "head_dim", "compress_ratio"))
    q_lens, kv_lens = kernel.attributes["q_lens"], kernel.attributes["kv_lens"]
    row_ends = [[(kv - q + offset + 1) // ratio for offset in range(q)]
                for q, kv in zip(q_lens, kv_lens, strict=True)]
    pairs, rows = sum(sum(ends) for ends in row_ends), sum(q_lens)
    capacity = kernel.attributes["topk_capacity"]
    cp = kernel.attributes["context_parallel_size"]
    empty_prefill = (kernel.variant == "dsv4_indexer_prefill" and cp == 1
        and all(length // ratio == 0 for length in kv_lens))
    arithmetic = _indexer_arithmetic(pairs, h, d)
    selection = None
    selection_pairs = pairs
    if kernel.variant == "dsv4_indexer_prefill" and cp == 1:
        # sampler.cu's unsplit topKPerRowJob returns natural indices and -1
        # for rowLen<=topK without loading logits, including NaNs and ties.
        shortcuts = [sum(end <= capacity for end in ends) for ends in row_ends]
        selection_pairs = sum(end for ends in row_ends for end in ends if end > capacity)
        if selection_pairs == 0:
            arithmetic["topk_selection_comparisons"] = 0
        selection = _prefill_selection_work(shortcuts, selection_pairs, rows)

    demand, operand = _indexer_demand, _indexer_operand
    stages = [
        {"stage": "topk_clear", "demand": demand(), "operands": [
            operand("topk_indices", "int32", rows * capacity, "write")]},
        {"stage": "scores", "demand": demand(arithmetic["qk_dot_flops"],
            arithmetic["head_weight_multiplications"] + arithmetic["head_sum_additions"] + arithmetic["key_scale_multiplications"]),
         "operands": [operand("prepared_q", "fp8", rows * h * d, "read"),
            operand("key_role_uses", "fp8", pairs * d, "read"),
            operand("key_scale_role_uses", "fp32", pairs, "read"),
            operand("folded_weights", "fp32", rows * h, "read"),
            operand("active_scores", "fp32", pairs, "write")]},
        {"stage": "select", "demand": demand(), "operands": [
            operand("active_scores", "fp32", selection_pairs, "read"),
            operand("topk_indices", "int32", rows * capacity, "write")]},
    ]
    if empty_prefill:
        # sparse_attn_indexer.py clears the full output slice first, then
        # each zero-context query chunk fills its own slice again. No score
        # kernel, selector, Q/weights read or row-bound read occurs in CP1.
        stages = [{"stage": stage, "demand": demand(), "operands": [
            operand("topk_indices", "int32", rows * capacity, "write")]}
            for stage in ("topk_clear", "empty_prefill_topk_fill")]
        selection = {"source": "vllm_0_30_empty_prefill_group", "selector_invoked": False,
            "rows": rows, "refinement_passes_and_scratch": {"passes": 0, "global_scratch_elements": 0},
            "actual_provider_bound": False}
    if selection is not None and not empty_prefill:
        stages[-1]["operands"].insert(0, operand("row_starts_and_ends", "int32", 2 * rows, "read"))
    score_metadata = None
    if cp == 1 and kernel.variant == "dsv4_indexer_prefill" and all(kv // ratio > 0 for kv in kv_lens):
        score_metadata = {"role": "score_row_starts_and_ends", "dtype": "int32", "elements": 2 * rows,
            "bytes": 8 * rows, "scope": "unique score-input payload over all query subchunks; separate from selector reads"}
    elif cp == 1 and kernel.variant == "dsv4_indexer_decode" and all(q == 1 for q in q_lens):
        score_metadata = {"role": "score_effective_context_lengths", "dtype": "int32", "elements": rows,
            "bytes": 4 * rows, "scope": "one effective context length per unpadded single-token query; not block tables or schedule metadata"}
    if score_metadata is not None:
        stages[1]["operands"].append(operand(score_metadata["role"], "int32", score_metadata["elements"], "read"))
    prefill = kernel.variant == "dsv4_indexer_prefill"
    own_rectangles = sum(q * (kv // ratio) for q, kv in zip(q_lens, kv_lens, strict=True))
    full_rectangle = rows * sum(kv // ratio for kv in kv_lens)
    score_materialization = {
        "dtype": "fp32", "element_bytes": 4, "clean_logits": False,
        "producer": "fp8_fp4_mqa_logits" if prefill else "fp8_fp4_paged_mqa_logits",
        "consumer": "top_k_per_row_prefill" if prefill else "get_indexer_topk(topk_backend)",
        "consumer_uses_same_tensor": True, "score_cast_or_copy_before_selection": False,
        "shape_rule": "[chunk_query_rows, chunk_total_compressed_context_rows]" if prefill else "[batch * next_n, runtime_indexer_max_model_len]",
        "query_subchunks_keep_group_context_extent": True if prefill else None,
        "total_semantic_output_elements": own_rectangles if prefill and cp == 1 and len(q_lens) == 1 else None,
        "total_semantic_output_element_bounds": {"lower": own_rectangles, "upper": full_rectangle} if prefill and cp == 1 else None,
        "runtime_indexer_max_model_len": None, "selected_chunk_partition": None,
        "physical_strides_and_backing_bytes": None, "inactive_cell_store_count": None,
        "unfilled_cell_cleanup_requested": False, "allocation_elements_counted_as_traffic": False,
        "actual_allocator_binding_verified": False,
        "scope": "wrapper logical tensor shape/FP32 handoff; inactive stores, physical backing, lifetime overlap and provider remain unbound"}
    if empty_prefill:
        score_materialization = {**score_materialization, "producer": "q_slice.new_empty",
            "consumer": None, "consumer_uses_same_tensor": False,
            "total_semantic_output_elements": 0, "payload_allocation_bytes": 0,
            "scope": "empty FP32 [query_rows,0] descriptor; no score/selector and no payload allocation"}
    gather = None
    if kernel.variant == "dsv4_indexer_prefill" and cp == 1 and not empty_prefill:
        # _split_indexer_prefill_chunks partitions request groups; query
        # subchunks retain their group and set skip_kv_gather=True. Every
        # request's complete compressed context is copied exactly once,
        # independently of the unknown logits/query chunk budget.
        counts = [length // ratio for length in kv_lens]
        gather, gather_stage = _prefill_gather_work(counts, d)
        stages.insert(1, gather_stage)
    aggregate = demand()
    for stage in stages:
        for role, value in stage["demand"].items():
            aggregate[role] += value
        stage["read_write_bytes"] = {
            access: sum(value["bytes"] for value in stage["operands"] if value["access"] == access)
            for access in ("read", "write")}
    read_write = {access: sum(stage["read_write_bytes"][access] for stage in stages) for access in ("read", "write")}
    return {"compressed_row_ends": row_ends, "active_pairs": pairs, "arithmetic": arithmetic,
            "stages": stages, "demand": aggregate, "read_write_bytes": read_write,
            "prefill_gather": gather,
            "prefill_selection": selection,
            "score_metadata": score_metadata,
            "score_materialization": score_materialization,
            "logical_bytes": sum(read_write.values()),
            "source_semantics": {"score_dtype": "fp32", "index_dtype": "int32", "sentinel": -1,
                "clear_and_selection_write_same_buffer": True, "q_scale_folded_into_weights": True,
                "context_parallel_size": cp,
                "key_scale_is_positive_quantization_scale": True,
                "relu_and_integer_selection_instruction_cost_priced": False,
                "selected_topk_backend": None, "actual_provider_bound": False,
                "finite_precision_equivalence_asserted": False},
            "remaining_source_services": (["workspace_entry_capacity_and_native_fill_launch"] if empty_prefill else
                (["prefill_paged_to_contiguous_gather_and_chunk_reloads"]
                                          if kernel.variant == "dsv4_indexer_prefill" and cp is None else []) + [
                "score_rectangular_allocation_stride_padding_and_inactive_rows",
                "metadata_page_tables_and_schedule_buffers"] + (
                    [] if selection is not None and selection_pairs == 0 else
                    ["selection_backend_value_dependent_refinement_and_scratch"])),
            "scope": "serial useful mathematical FLOPs/role-use bytes; not full indexer source/native acceptance or a hardware latency bound"}


def _dsa_decode_selection_actions(kernel, stride, cluster):
    topk = kernel.attributes["topk_capacity"]
    actions = {name: 0 for name in ("histogram_atomic_updates", "scatter_bin_comparisons",
        "threshold_prefix_values", "refinement_histogram_scatter_actions",
        "refinement_prefix_values", "scatter_output_atomic_updates")}
    routes = {"trivial": 0, "single_cta_histogram": 0,
        "cluster_fused": 0, "cluster_streaming_two_pass": 0}
    score_reads = tie_values = 0
    per_block = -(-stride // cluster)
    fused_capacity = (2 if cluster == 8 else 3) * 16384
    # Each causal query row of a request runs the selector over its own
    # visible score range (cooperative_topk.cu is invoked per logits row);
    # a multi-row verify window therefore expands selection work by row.
    for q, length in zip(kernel.attributes["q_lens"], kernel.attributes["kv_lens"],
            strict=True):
        for offset in range(q):
            visible = length - q + offset + 1
            if visible <= topk:
                routes["trivial"] += 1
                continue
            score_reads += visible
            actions["histogram_atomic_updates"] += visible
            actions["scatter_bin_comparisons"] += visible
            actions["threshold_prefix_values"] += 4096 if visible <= 16384 else 1024
            actions["refinement_histogram_scatter_actions"] += 8 * topk
            actions["refinement_prefix_values"] += 4 * 256
            actions["scatter_output_atomic_updates"] += 2 * topk
            if visible <= 16384:
                routes["single_cta_histogram"] += 1
            else:
                tie_values += topk
                if per_block <= fused_capacity:
                    routes["cluster_fused"] += 1
                else:
                    routes["cluster_streaming_two_pass"] += 1
                    score_reads += visible
    return actions, routes, score_reads, tie_values


def _dsa_filtered_selection_actions(kernel, vector):
    """FilteredTopK source actions from persistent_topk.cuh/topk_histogram_4096.cuh.

    One CTA per row. Routes follow the source exactly: trivial fill for
    rowLen<=topk; the register-resident histogram_4096 single-pass path for
    rowLen<=32768; otherwise the coarse 8-bit histogram plus up to four
    8-bit refinement rounds over a shared-memory candidate set capped at
    16K entries. Counts are capacity bounds on source loops; shared-memory
    traffic is reported separately because the kernel services it from
    dynamic/static smem, not DRAM.
    """
    topk = kernel.attributes["topk_capacity"]
    actions = {name: 0 for name in ("trivial_fill_output_writes",
        "histogram_atomic_updates", "threshold_prefix_values",
        "scatter_bin_comparisons", "refinement_score_values",
        "refinement_histogram_scatter_actions", "scatter_output_atomic_updates")}
    routes = {"trivial": 0, "filtered_short_histogram": 0, "filtered_long_refined": 0}
    score_reads = tie_values = smem_traffic = 0
    for q, length in zip(kernel.attributes["q_lens"], kernel.attributes["kv_lens"],
            strict=True):
        for offset in range(q):
            visible = length - q + offset + 1
            if visible <= topk:
                routes["trivial"] += 1
                actions["trivial_fill_output_writes"] += topk
                continue
            if visible <= 32768:
                # histogram_4096_topk: one RF-resident pass (load, histogram,
                # threshold scan, scatter, bounded tie sort over tie_buffer).
                routes["filtered_short_histogram"] += 1
                score_reads += visible
                actions["histogram_atomic_updates"] += visible
                actions["threshold_prefix_values"] += 4096
                actions["scatter_bin_comparisons"] += visible
                actions["scatter_output_atomic_updates"] += 2 * topk
                ties = topk  # capacity bound: min(num_equal, TopK)
                tie_values += ties
                actions["refinement_histogram_scatter_actions"] += 8 * ties
                smem_traffic += (4 * visible                      # histogram updates
                    + 4 * 4096                                 # threshold scan
                    + 4 * visible                              # scatter bin compares
                    + 8 * 2 * ties)                            # tie buffer rw bound
            else:
                # Coarse pass + filter pass (two full score reads), then up to
                # four refinement rounds over <=16K shared candidates.
                routes["filtered_long_refined"] += 1
                candidates = min(visible - topk, 16384)
                score_reads += visible                    # coarse histogram pass
                score_reads += visible                    # filter pass
                score_reads += 4 * candidates             # refinement score[idx]
                actions["histogram_atomic_updates"] += visible
                actions["scatter_bin_comparisons"] += visible
                actions["threshold_prefix_values"] += 2560   # 8x256 cumsum + 512 scan
                actions["refinement_score_values"] += 4 * candidates
                actions["refinement_histogram_scatter_actions"] += 8 * candidates
                actions["scatter_output_atomic_updates"] += 2 * topk
                tie_values += candidates
                smem_traffic += (4 * visible               # coarse histogram updates
                    + 4 * 2048                            # cumsum read/write traffic
                    + 4 * visible                         # filter pass compares
                    + 4 * 8 * candidates)                 # refinement buffer rw bound
    return actions, routes, score_reads, tie_values, smem_traffic


def dsa_decode_source_stage_work(kernel: KernelSpec, hardware):
    """Serial DeepGEMM/decode-TopK source work with bounded tie refinement."""
    if (kernel.kind is not KernelKind.SPARSE_ATTN_INDEXER
            or kernel.variant != "dsa_indexer_decode"
            or kernel.attributes["mathematical_program"] not in ("dsa_weighted_relu_topk_v1",
                "dsa_weighted_relu_specdecode_v1")
            or kernel.attributes["source_cost_model"] != "vllm_0_30_bounded_stage_roofline_v1"
            or kernel.attributes["source_tie_refinement_policy"] != "capacity_bound"):
        raise ValueError("DSA decode source work requires its explicit bounded GLM program")
    spec_decode = kernel.attributes["mathematical_program"] == "dsa_weighted_relu_specdecode_v1"
    if spec_decode and (type(kernel.attributes["num_speculative_tokens"]) is not int
            or kernel.attributes["num_speculative_tokens"] < 1
            or type(kernel.attributes["decode_threshold"]) is not int
            or kernel.attributes["decode_threshold"] != kernel.attributes["num_speculative_tokens"] + 1):
        raise ValueError("DSA specdecode source work requires its explicit positive depth+1 threshold")
    rows = sum(kernel.attributes["q_lens"])
    heads, dim = kernel.shape["num_query_heads"], kernel.shape["head_dim"]
    topk = kernel.attributes["topk_capacity"]
    stride = -(-kernel.attributes["source_max_model_len"] // 256) * 256
    pairs = causal_attention_pairs(kernel.attributes["q_lens"], kernel.attributes["kv_lens"])
    arithmetic = _indexer_arithmetic(pairs, heads, dim)
    score = {"stage": "scores", "demand": _indexer_demand(
        arithmetic["qk_dot_flops"], arithmetic["head_weight_multiplications"]
        + arithmetic["head_sum_additions"] + arithmetic["key_scale_multiplications"]),
        "operands": [
            _indexer_operand("prepared_q", "fp8", rows * heads * dim, "read"),
            _indexer_operand("index_cache_values", "fp8", pairs * dim, "read"),
            _indexer_operand("index_cache_scales", "fp32", pairs, "read"),
            _indexer_operand("folded_weights", "fp32", rows * heads, "read"),
            _indexer_operand("score_effective_context_lengths", "int32", rows, "read"),
            _indexer_operand("score_backing", "fp32", rows * stride, "write")]}
    route = kernel.attributes["source_topk_route"]
    if not isinstance(route, dict) or route["route"] not in ("cooperative", "filtered_persistent"):
        raise ValueError("DSA decode source work requires its launch-bound TopK route facts")
    if route["route"] == "cooperative":
        cluster = route["cluster"]
        actions, routes, score_reads, tie_values = _dsa_decode_selection_actions(kernel, stride, cluster)
        selection_smem_bytes = 0
        workspace_traffic_bytes = 16 * tie_values
        tie_operands = (
            _indexer_operand("tie_workspace", "uint8", 8 * tie_values, "write"),
            _indexer_operand("tie_workspace", "uint8", 8 * tie_values, "read"))
        action_sources = {
            "histogram_atomic_updates": "cooperative_topk.cuh:tma_stream_pass/fused histogram; topk_histogram_4096.cuh:Phase1",
            "scatter_bin_comparisons": "cooperative_topk.cuh:fused/scatter pass; topk_histogram_4096.cuh:Phase3",
            "threshold_prefix_values": "cooperative_topk.cuh:find_threshold/DSMEM; topk_histogram_4096.cuh:Phase2",
            "refinement_histogram_scatter_actions": "topk_histogram_4096.cuh:tie_handle_large four rounds, two actions per active tie",
            "refinement_prefix_values": "topk_histogram_4096.cuh:tie_handle_large four 256-bin prefix scans",
            "scatter_output_atomic_updates": "cooperative_topk.cuh:counter_gt/counter_eq capacity-bounded output cohort"}
    else:
        cluster = None
        vector = route["vector_width"]
        (actions, routes, score_reads, tie_values,
            selection_smem_bytes) = _dsa_filtered_selection_actions(kernel, vector)
        # FilteredTopKUnifiedKernel/histogram_4096_topk take no global
        # workspace argument and service every tie/refinement buffer from
        # static/dynamic shared memory; the caller's arena stays resident
        # but unaccessed, so it contributes zero traffic on this route.
        workspace_traffic_bytes = 0
        tie_operands = ()
        action_sources = {
            "trivial_fill_output_writes": "persistent_topk.cuh:FilteredTopKUnifiedKernel trivial length<=topk output fill",
            "histogram_atomic_updates": "persistent_topk.cuh:s_histogram_buf coarse updates; topk_histogram_4096.cuh:Phase1",
            "scatter_bin_comparisons": "persistent_topk.cuh:coarse/filter passes; topk_histogram_4096.cuh:Phase3",
            "threshold_prefix_values": "persistent_topk.cuh:run_cumsum; topk_histogram_4096.cuh:Phase2",
            "refinement_score_values": "persistent_topk.cuh:four refinement rounds over <=16K shared candidates",
            "refinement_histogram_scatter_actions": "persistent_topk.cuh:refinement rounds; topk_histogram_4096.cuh:tie_handle(_large)",
            "scatter_output_atomic_updates": "persistent_topk.cuh:s_counter/s_indices output cohort"}
    comparisons = sum(actions.values())
    selection_demand = _indexer_demand()
    selection_demand["compute_alu"] = comparisons
    select = {"stage": "select", "demand": selection_demand, "operands": [
            _indexer_operand("active_scores", "fp32", score_reads, "read"),
            _indexer_operand("topk_context_lengths", "int32", rows, "read"),
            *tie_operands,
            _indexer_operand("selected_indices", "int32", rows * topk, "write")]}
    if selection_smem_bytes:
        # FilteredTopK services histogram/refinement traffic from shared
        # memory (static + 128 KiB dynamic), not L2/DRAM. The helper returns
        # one total service byte figure; price it once against the smem
        # capacity, never as a read+write pair.
        select["shared_memory_service_bytes"] = selection_smem_bytes
    stages = (score, select)
    for stage in stages:
        stage["read_write_bytes"] = {access: sum(op["bytes"] for op in stage["operands"]
            if op["access"] == access) for access in ("read", "write")}
    arithmetic["topk_selection_comparisons"] = comparisons
    smem = hardware.resource_by_role("smem")
    if smem.capacity <= 0:
        raise ValueError("decode selection shared-memory service requires a positive smem capacity")
    stage_ns = []
    for stage in stages:
        l2_dram_ns = source_stage_roofline_ns(stage["demand"], hardware,
            sum(stage["read_write_bytes"].values()), sum(stage["read_write_bytes"].values()))
        if "shared_memory_service_bytes" in stage:
            smem_ns = stage["shared_memory_service_bytes"] / smem.capacity * 1e9
            l2_dram_ns = max(l2_dram_ns, smem_ns)
        stage_ns.append(l2_dram_ns)
    return {"stages": stages, "predicted_stage_ns": tuple(stage_ns), "predicted_ns": sum(stage_ns),
        "arithmetic": arithmetic, "active_pairs": pairs,
        "score_stride": stride, "score_backing_bytes": 4 * rows * stride,
        "index_cache_read_bytes": pairs * (dim + 4), "topk_routes": routes,
        "selection_actions": actions,
        "selection_action_sources": action_sources,
        "selection_tie_values_capacity_bound": tie_values,
        "selection_workspace_allocated_bytes": 1024 * 1024,
        "selection_workspace_traffic_bytes": workspace_traffic_bytes,
        "selection_shared_memory_service_bytes": selection_smem_bytes,
        "analytical_live_capacity_bytes": 4 * rows * stride + 1024 * 1024 + 4 * rows * topk,
        "capacity_scope": "per-layer score backing, fixed TopK arena and selected-index output; not allocator peak",
        "source_topk_cluster_size": cluster,
        "source_topk_route": route,
        "limitations": ("serial_source_stage_roofline_not_native_pipeline_overlap",
            "capacity_bound_tie_refinement_not_observed_score_values",
            "logical_role_traffic_with_full_score_backing_not_measured_dram",
            "compiler_instruction_issue_and_launch_latency_not_replayed")}


def dsa_prefill_source_stage_work(kernel: KernelSpec, hardware):
    """Chunked prefill gather/logits/TopK plus the optional native decode suffix."""
    if (kernel.kind is not KernelKind.SPARSE_ATTN_INDEXER
            or kernel.variant not in ("dsa_indexer_prefill", "dsa_indexer_mixed")
            or kernel.attributes["mathematical_program"] not in (
                "dsa_weighted_relu_force_mqa_v1", "dsa_weighted_relu_dense_mha_v1")
            or kernel.attributes["source_cost_model"] != "vllm_0_30_bounded_stage_roofline_v1"
            or kernel.attributes["source_tie_refinement_policy"] != "capacity_bound"
            or kernel.attributes["source_chunk_policy"] != "one_request_per_chunk_analytical"):
        raise ValueError("DSA prefill source work requires its explicit bounded chunk program")
    from dataclasses import replace
    q_lens, kv_lens = kernel.attributes["q_lens"], kernel.attributes["kv_lens"]
    phases = kernel.attributes["request_phases"]
    rows = sum(q_lens)
    decode_rows = sum(q for q, phase in zip(q_lens, phases, strict=True) if phase == "decode")
    heads, dim = kernel.shape["num_query_heads"], kernel.shape["head_dim"]
    topk = kernel.attributes["topk_capacity"]
    stages = []
    active_pairs = score_backing = index_reads = comparisons = tie_traffic = 0
    selection_actions = {name: 0 for name in ("histogram_atomic_updates", "scatter_bin_comparisons",
        "threshold_prefix_values", "insertion_sort_comparisons", "radix_sort_actions")}
    routes = {"trivial": 0, "insertion": 0, "radix": 0}
    cursor = decode_rows
    for chunk_index, chunk in enumerate(kernel.attributes["source_prefill_chunks"]):
        start, end, kv = (chunk[field] for field in ("token_start", "token_end", "kv_rows"))
        if start != cursor or not start < end <= rows or kv <= 0:
            raise ValueError("DSA source chunks must partition the actual prefill suffix")
        cursor = end
        q = end - start
        pairs = causal_attention_pairs((q,), (kv,))
        active_pairs += pairs
        arithmetic = _indexer_arithmetic(pairs, heads, dim)
        if not chunk["skip_kv_gather"]:
            _, gather = _prefill_gather_work((kv,), dim)
            gather["stage"] = f"prefill_{chunk_index}_cache_gather"
            gather["read_write_bytes"] = {access: sum(op["bytes"] for op in gather["operands"]
                if op["access"] == access) for access in ("read", "write")}
            stages.append(gather)
            index_reads += sum(gather["read_write_bytes"].values())
        block_q = 128 // heads
        padded_rows = -(-q // block_q) * block_q
        stride = -(-(kv + 256) // 256) * 256
        query_tiles = padded_rows // block_q
        cache_role_rows = kv * query_tiles
        index_reads += cache_role_rows * (dim + 4)
        score_backing += 4 * padded_rows * stride
        stages.append({"stage": f"prefill_{chunk_index}_scores", "demand": _indexer_demand(
            arithmetic["qk_dot_flops"], arithmetic["head_weight_multiplications"]
            + arithmetic["head_sum_additions"] + arithmetic["key_scale_multiplications"]),
            "operands": [
                _indexer_operand("prepared_q", "fp8", q * heads * dim, "read"),
                _indexer_operand("index_cache_values", "fp8", cache_role_rows * dim, "read"),
                _indexer_operand("index_cache_scales", "fp32", cache_role_rows, "read"),
                _indexer_operand("folded_weights", "fp32", q * heads, "read"),
                _indexer_operand("score_row_starts_and_ends", "int32", 2 * q, "read"),
                _indexer_operand("score_backing", "fp32", padded_rows * stride, "write")]})
        score_reads = 0
        chunk_comparisons = 0
        shared_scratch_rows = 0
        for offset in range(q):
            length = kv - q + offset + 1
            if length <= topk:
                routes["trivial"] += 1
                continue
            score_reads += 8 * length
            selection_actions["histogram_atomic_updates"] += 4 * length
            selection_actions["scatter_bin_comparisons"] += 4 * length
            selection_actions["threshold_prefix_values"] += 4 * 2048
            shared_scratch_rows += 1
            if start + offset < 12288:
                routes["insertion"] += 1
                selection_actions["insertion_sort_comparisons"] += topk * topk
                chunk_comparisons += 8 * length + 4 * 2048 + topk * topk
            else:
                routes["radix"] += 1
                selection_actions["radix_sort_actions"] += 4 * topk
                chunk_comparisons += 8 * length + 4 * 2048 + 4 * topk
        comparisons += chunk_comparisons
        demand = _indexer_demand()
        demand["compute_alu"] = chunk_comparisons
        shared_scratch_bytes = 16 * topk * shared_scratch_rows
        select_stage = {"stage": f"prefill_{chunk_index}_select", "demand": demand,
            "operands": [
                _indexer_operand("active_scores", "fp32", score_reads, "read"),
                _indexer_operand("row_starts_and_ends", "int32", 2 * q, "read"),
                _indexer_operand("selected_indices", "int32", q * topk, "write")],
            # Sort/insertion scratch is a shared-memory service (DSMEM on
            # Hopper), not an L2/DRAM traffic flow; it is priced separately
            # below against the smem resource.
            "shared_memory_bytes": {"read": shared_scratch_bytes, "write": shared_scratch_bytes},
            "read_write_bytes": {"read": score_reads * 4 + 8 * q,
                "write": q * topk * 4}}
        stages.append(select_stage)
    if cursor != rows:
        raise ValueError("DSA source chunks do not cover the actual prefill suffix")
    decode_work = None
    if decode_rows:
        indexes = tuple(index for index, phase in enumerate(phases) if phase == "decode")
        decode_q = tuple(q_lens[index] for index in indexes)
        decode_kv = tuple(kv_lens[index] for index in indexes)
        decode_kernel = replace(kernel, variant="dsa_indexer_decode",
            shape={**dict(kernel.shape), "batch": len(indexes), "q_len": 1,
                "kv_len": max(decode_kv), "topk": min(topk, max(decode_kv))},
            attributes={**dict(kernel.attributes), "q_lens": decode_q, "kv_lens": decode_kv,
                "request_phases": ("decode",) * len(indexes),
                "mathematical_program": "dsa_weighted_relu_topk_v1"})
        decode_work = dsa_decode_source_stage_work(decode_kernel, hardware)
        for stage in decode_work["stages"]:
            stages.append({**stage, "stage": "decode_" + stage["stage"]})
        active_pairs += decode_work["active_pairs"]
        score_backing += decode_work["score_backing_bytes"]
        index_reads += decode_work["index_cache_read_bytes"]
        comparisons += decode_work["arithmetic"]["topk_selection_comparisons"]
        tie_traffic += decode_work["selection_workspace_traffic_bytes"]
    for stage in stages:
        if "read_write_bytes" not in stage:
            stage["read_write_bytes"] = {access: sum(op["bytes"] for op in stage["operands"]
                if op["access"] == access) for access in ("read", "write")}
    smem = hardware.resource_by_role("smem")
    if smem.capacity <= 0:
        raise ValueError("prefill selection shared-memory service requires a positive smem capacity")
    stage_ns = []
    for stage in stages:
        l2_dram_ns = source_stage_roofline_ns(stage["demand"], hardware,
            sum(stage["read_write_bytes"].values()), sum(stage["read_write_bytes"].values()))
        if "shared_memory_bytes" in stage:
            smem_bytes = stage["shared_memory_bytes"]
            # HWSpec exposes one smem capacity resource; price the total
            # service bytes (16*topk*rows per direction, 32*topk*rows total).
            smem_ns = (smem_bytes["read"] + smem_bytes["write"]) / smem.capacity * 1e9
            l2_dram_ns = max(l2_dram_ns, smem_ns)
        stage_ns.append(l2_dram_ns)
    gather_capacity = sum(stage["read_write_bytes"]["write"] for stage in stages
        if stage["stage"].endswith("cache_gather"))
    live_capacity = (score_backing + gather_capacity + rows * topk * 4
        + (1024 * 1024 if decode_work is not None else 0))
    return {"stages": stages, "predicted_stage_ns": stage_ns, "predicted_ns": sum(stage_ns),
        "analytical_live_capacity_bytes": live_capacity,
        "capacity_scope": "per-layer score/gather/TopK/decode arena; not allocator peak",
        "active_pairs": active_pairs, "score_backing_bytes": score_backing,
        "index_cache_and_gather_traffic_bytes": index_reads,
        "topk_routes": routes, "selection_comparisons_capacity_bound": comparisons,
        "prefill_selection_actions": selection_actions,
        "decode_selection_actions": None if decode_work is None else decode_work["selection_actions"],
        "selection_shared_scratch_read_write_bytes": 32 * topk * (routes["insertion"] + routes["radix"]),
        "selection_shared_scratch_service_bytes": 32 * topk * (routes["insertion"] + routes["radix"]),
        "selection_workspace_traffic_bytes": tie_traffic,
        "decode_topk_routes": None if decode_work is None else decode_work["topk_routes"],
        "limitations": ("serial_source_stage_roofline_not_native_pipeline_overlap",
            "one_request_per_chunk_analytical_not_automatic_runtime_chunk_choice",
            "four_refinement_passes_and_sort_capacity_bounds_not_observed_score_values",
            "query_tile_cache_role_traffic_not_measured_l2_or_dram",
            "decode_tie_workspace_capacity_bound_carried_from_decode_path")}


def _model(kernel: KernelSpec, hardware):
    if kernel.variant == "dsa_index_reuse":
        return None, 0.0
    q_lens = tuple(kernel.attributes["q_lens"])
    kv_lens = tuple(kernel.attributes["kv_lens"])
    heads = int(kernel.shape["num_query_heads"])
    head_dim = int(kernel.shape["head_dim"])
    topk = int(kernel.shape["topk"])
    query_tokens = sum(q_lens)
    if kernel.variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
        work = dsv4_indexer_logical_work(kernel)
        read_bytes, write_bytes = (float(work["read_write_bytes"][access]) for access in ("read", "write"))
        flops = work["demand"]
        limitations = (
            "dsv4_indexer_useful_math_byte_baseline_only_not_complete_source_service",
            "relu_comparisons_and_topk_integer_instruction_cost_unpriced_not_log2_estimated",
            "source_role_use_bytes_not_observed_dram_or_proven_cache_residency",
            *work["remaining_source_services"],
            "actual_provider_compiler_launch_and_finite_precision_order_unbound",
            "production_duration_calibration_not_used",
        )
    else:
        from src.kernel.memory_access_builder import sparse_indexer_memory_access_program
        from src.kernel.memory_path_solver import sparse_indexer_source_demands
        source = sparse_indexer_source_demands(
            sparse_indexer_memory_access_program(kernel, hardware), expected_kernel=kernel)
        query_tokens, heads, head_dim, pairs, _ = source["geometry"]
        read_bytes, write_bytes = float(source["read_bytes"]), float(source["write_bytes"])
        candidate_rounds = max(1, math.ceil(math.log2(topk)))
        flops = {
            "compute_tensor": float(2 * pairs * heads * head_dim),
            "compute_fma": float(pairs * heads),
            "compute_alu": float(pairs * candidate_rounds + query_tokens * topk),
            "compute_sfu": 0.0,
        }
        limitations = (
            "dsa_indexer_multi_kernel_overlap_unavailable",
            "dsa_topk_selector_schedule_unavailable",
            "sm90_source_execution_projected_on_sm80",
            "production_duration_calibration_not_used",
        )
    total_bytes = read_bytes + write_bytes
    demand = build_forward_demand(total_bytes, total_bytes, True, flops)
    inp = build_aggregate_gpu_input(
        kernel,
        hardware,
        demand,
        write_l2_bytes=write_bytes,
        write_dram_bytes=write_bytes,
        limitations=limitations,
    )
    return inp, total_bytes


class SparseAttentionIndexerPredictor(KernelPredictor):
    kind = KernelKind.SPARSE_ATTN_INDEXER

    def analytical_without_launch(self, kernel: KernelSpec, hardware):
        return self.roofline_without_launch(kernel, hardware)

    def roofline_without_launch(self, kernel: KernelSpec, hardware):
        if kernel.variant == "projected_attention_path":
            from src.kernel.predictors.projected_attention import source_prediction
            return source_prediction(kernel, hardware)
        if HWSpec.source_contract_supported("multistage_attention_indexer_unbound_source_math", hardware.hardware_id) and kernel.variant in (
                "dsv4_indexer_prefill", "dsv4_indexer_decode"):
            work = dsv4_indexer_logical_work(kernel)
            ns = sum(roofline_ns(stage["demand"], hardware, sum(stage["read_write_bytes"].values()))
                for stage in work["stages"])
            return prediction_from_ns(ns, "roofline")
        if HWSpec.source_contract_supported("aggregate_gpu_source_roofline", hardware.hardware_id) and kernel.variant in (
                "dsv4_indexer_prefill", "dsv4_indexer_decode"):
            work = dsv4_indexer_logical_work(kernel)
            if hardware.gpu_vector_memory_only:
                # Scores retain matrix Roofline. Clear/gather/select retain
                # their bytes without inventing a scalar or selector rate.
                ns = sum(roofline_ns({role: amount if role == "compute_tensor" else 0
                    for role, amount in stage["demand"].items()}, hardware,
                    sum(stage["read_write_bytes"].values())) for stage in work["stages"])
                return prediction_from_ns(ns, "roofline")
            missing = sorted({role for stage in work["stages"] for role, amount in stage["demand"].items()
                if amount > 0 and role not in hardware._role_map})
            if missing:
                raise NotImplementedError(f"BW1100 indexer source Roofline requires {tuple(missing)}; native launch and complete selection service remain separate")
            return self.roofline(kernel, hardware)
        if (kernel.variant in ("dsa_indexer_prefill", "dsa_indexer_mixed")
                and kernel.attributes["source_cost_model"] == "vllm_0_30_bounded_stage_roofline_v1"):
            work = dsa_prefill_source_stage_work(kernel, hardware)
            return prediction_from_ns(work["predicted_ns"], "roofline")
        if (kernel.variant == "dsa_indexer_decode"
                and kernel.attributes["source_cost_model"] == "vllm_0_30_bounded_stage_roofline_v1"):
            work = dsa_decode_source_stage_work(kernel, hardware)
            return prediction_from_ns(work["predicted_ns"], "roofline")
        if (kernel.variant in ("dsa_indexer_decode", "dsa_indexer_prefill", "dsa_indexer_mixed")
                and kernel.attributes["mathematical_program"] in (
                    "dsa_weighted_relu_topk_v1", "dsa_weighted_relu_force_mqa_v1", "dsa_weighted_relu_dense_mha_v1", "dsa_weighted_relu_specdecode_v1")):
            work = dsa_indexer_logical_work(kernel)
            ns = sum(roofline_ns(stage["demand"], hardware, sum(stage["read_write_bytes"].values()))
                for stage in work["stages"])
            return prediction_from_ns(ns, "roofline")
        return super().roofline_without_launch(kernel, hardware)

    def resource_input(self, kernel: KernelSpec, hardware):
        if (HWSpec.source_contract_supported("multistage_attention_indexer_unbound_source_math", kernel.hardware_id) and kernel.launch is None
                and kernel.variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode")):
            raise NotImplementedError("V4 logical indexer provides stage Roofline without a native aggregate resource program")
        if (kernel.variant in ("dsa_indexer_prefill", "dsa_indexer_decode", "dsa_indexer_mixed")
                and kernel.attributes["mathematical_program"] is not None):
            raise NotImplementedError("DSA mathematical program provides useful stage Roofline only")
        inp, compulsory = _model(kernel, hardware)
        if inp is None:
            raise NotImplementedError(
                "dsa_index_reuse is a zero-work alias without resource input")
        return inp, compulsory

    def midlayer(self, kernel: KernelSpec, hardware):
        if kernel.variant == "dsa_index_reuse":
            from src.kernel.memory_access_builder import sparse_indexer_memory_access_program
            from src.kernel.memory_path_solver import sparse_indexer_source_demands
            sparse_indexer_source_demands(
                sparse_indexer_memory_access_program(kernel, hardware), expected_kernel=kernel)
            return prediction_from_ns(0.0, "alias")
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(inp)

    def roofline(self, kernel: KernelSpec, hardware):
        if kernel.variant == "dsa_index_reuse":
            from src.kernel.memory_access_builder import sparse_indexer_memory_access_program
            from src.kernel.memory_path_solver import sparse_indexer_source_demands
            sparse_indexer_source_demands(
                sparse_indexer_memory_access_program(kernel, hardware), expected_kernel=kernel)
            return prediction_from_ns(0.0, "alias")
        if kernel.variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
            work = dsv4_indexer_logical_work(kernel)
            # Clear precedes scores; scores are materialized before selection.
            # Sum stage Rooflines rather than overlapping producer/consumer
            # into one global max. Integer selection work is explicitly not
            # priced, just as predicate/index work in the argmax baseline.
            ns = sum(roofline_ns(stage["demand"], hardware, sum(stage["read_write_bytes"].values()))
                     for stage in work["stages"])
            return prediction_from_ns(ns, "roofline")
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory),
            "roofline",
        )
