"""Source-work analytical prediction for DSA sparse MLA attention."""

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


def dsa_sparse_attention_logical_work(kernel: KernelSpec):
    """BF16 latent attention: ideal math and shared-KV role payload.

    The 512 latent values serve as both the NoPE key and the value; they are
    one cache input, not two independent tensors. Logical query/KV role uses
    do not prove cross-query reuse, physical traffic or native padding work.
    """
    prefill_program = (kernel.variant in ("dsa_sparse_prefill", "dsa_sparse_mixed")
        and "mathematical_program" in kernel.attributes
        and kernel.attributes["mathematical_program"] == "bf16_latent_force_mqa_v1")
    spec_decode = (kernel.variant == "dsa_sparse_decode" and "mathematical_program" in kernel.attributes
        and kernel.attributes["mathematical_program"] == "bf16_latent_specdecode_v1")
    if (kernel.kind is not KernelKind.SPARSE_ATTN or (kernel.variant != "dsa_sparse_decode" and not prefill_program)
            or kernel.dtype != "bf16" or kernel.execution_support != "unsupported" or kernel.launch is not None
            or kernel.shape["qk_head_dim"] != 576 or kernel.shape["v_head_dim"] != 512
            or (not prefill_program and not spec_decode and any(q != 1 for q in kernel.attributes["q_lens"]))):
        raise ValueError("DSA useful sparse math requires unbound BF16 576/512 single-token decode")
    tokens, heads = sum(kernel.attributes["q_lens"]), kernel.shape["num_query_heads"]
    selected = (causal_attention_pairs(kernel.attributes["q_lens"], kernel.attributes["kv_lens"],
        capacity=kernel.attributes["topk_capacity"]) if prefill_program or spec_decode
        else sum(kernel.attributes["selected_kv_lens"]))
    score_pairs, rows = selected * heads, tokens * heads
    payload = {"queries": 2 * tokens * heads * 576, "shared_latent_and_rope_role_uses": 2 * selected * 576,
        "selected_indices": 4 * selected, "output": 2 * tokens * heads * 512}
    return {"selected_pairs": selected, "score_pairs": score_pairs,
        "tensor_flops": 2 * score_pairs * (576 + 512), "scalar_flops": 4 * score_pairs - rows,
        "exp_operations": score_pairs, "reciprocal_operations": rows, "max_comparisons": score_pairs - rows,
        "payload_bytes": payload, "logical_bytes": sum(payload.values()),
        "limitations": ("declared_causal_topk_cardinality_not_observed_IDs",
            "ideal_softmax_not_native_online_split_or_padded_head_work",
            "logical_role_uses_not_physical_traffic_or_cache_residency",
            "index_conversion_metadata_and_backend_scratch_unbound")}


def dsa_sparse_source_stage_work(kernel: KernelSpec, hardware):
    """BF16 sparse FlashMLA source tiles, index remap and public output backing."""
    if (kernel.kind is not KernelKind.SPARSE_ATTN
            or kernel.variant not in ("dsa_sparse_decode", "dsa_sparse_prefill", "dsa_sparse_mixed")
            or kernel.dtype != "bf16" or kernel.attributes["source_cost_model"]
            != "vllm_0_30_bounded_stage_roofline_v1"
            or kernel.attributes["model_execution_version"] != "vllm_0_30_0"
            or kernel.attributes["source_component"] != "vllm_0_30_flashmla_sparse_bf16"
            or kernel.attributes["source_attention_backend"] != "flashmla_sparse"
            or kernel.attributes["source_kv_cache_layout"] != "paged_bf16_rows_576"
            or kernel.attributes["source_parallel_mode"] != "pcp1_dcp1"):
        raise ValueError("BF16 sparse source work requires its explicit GLM phase geometry")
    spec_decode = (kernel.variant == "dsa_sparse_decode" and "mathematical_program" in kernel.attributes
        and kernel.attributes["mathematical_program"] == "bf16_latent_specdecode_v1")
    if spec_decode:
        # K draft rows are priced as real attention rows; acceptance/rollback
        # policy is a serving-level fact, not a cheaper sparse decode.
        if any(q != shape_q for q, shape_q in zip(kernel.attributes["q_lens"],
                (kernel.shape["q_len"],) * len(kernel.attributes["q_lens"]))):
            raise ValueError("BF16 sparse specdecode requires uniform verify rows")
    elif kernel.variant == "dsa_sparse_decode" and any(q != 1 for q in kernel.attributes["q_lens"]):
        raise ValueError("BF16 sparse source work requires its explicit GLM phase geometry")
    q_lens, kv_lens = kernel.attributes["q_lens"], kernel.attributes["kv_lens"]
    rows = sum(q_lens)
    heads, qk, value = (kernel.shape[name] for name in
        ("num_query_heads", "qk_head_dim", "v_head_dim"))
    topk = kernel.attributes["topk_capacity"]
    if (qk, value) != (576, 512):
        raise ValueError("BF16 sparse source work requires latent576/value512 geometry")
    padded = -(-heads // 64) * 64
    packs = padded // 64
    # selected_kv_lens is the selected cardinality per request; every causal
    # query row of the request attends its own selected set (source wrapper
    # pads and loops rows). Specdecode and ordinary decode therefore use the
    # same per-query causal count with top-k capacity; the q1 reused-topk
    # program keeps its explicit historical selected_kv_lens.
    if kernel.variant == "dsa_sparse_decode" and spec_decode:
        selected = causal_attention_pairs(q_lens, kv_lens, capacity=topk)
    elif kernel.variant == "dsa_sparse_decode":
        selected = sum(kernel.attributes["selected_kv_lens"])
    else:
        selected = causal_attention_pairs(q_lens, kv_lens, capacity=topk)
    pairs = selected * padded
    output_rows = rows * padded
    stages = []
    layout = kernel.attributes["source_q_input_layout"]
    if layout not in ("contiguous_unpadded", "caller_padded_head_backing"):
        raise ValueError(f"unknown DSA sparse Q input layout: {layout!r}")
    if layout == "caller_padded_head_backing" and kernel.attributes["source_q_backing_heads"] != padded:
        raise ValueError("BF16 sparse caller backing must cover the actual padded head count")
    # The verify-window q_concat_buffer already carries the 64-head backing,
    # so the padded layout needs no extra Q padding copy stage.
    q_padding_copy = (heads < padded and layout == "contiguous_unpadded")
    build_indices = kernel.attributes["source_index_remap_state"] == "build_current_indices"
    if q_padding_copy:
        stages.append({"stage": "q_padding_copy", "demand": _zero_demand(),
            "read_write_bytes": {"read": rows * heads * qk * 2,
                "write": rows * heads * qk * 2}})
    if build_indices:
        stages.append({"stage": "index_sentinel_fill", "demand": _zero_demand(),
            "read_write_bytes": {"read": 0, "write": rows * topk * 4}})
        stages.append({"stage": "index_remap", "demand": _zero_demand(),
            "read_write_bytes": {"read": rows * topk * 4 + rows * 4 + selected * 4,
                "write": rows * topk * 4 + rows * 4}})
    stages.extend((
        {"stage": "qk_and_max", "demand": _attention_demand(
            tensor=2 * pairs * qk, alu=pairs - output_rows),
         "read_write_bytes": {"read": rows * padded * qk * 2
             + packs * selected * qk * 2 + selected * 4 + rows * 4,
             "write": output_rows * 4}},
        {"stage": "softmax_and_lse", "demand": _attention_demand(
            fma=4 * pairs, sfu=pairs + 2 * output_rows),
         "read_write_bytes": {"read": 0, "write": output_rows * 4}},
        {"stage": "pv_and_output", "demand": _attention_demand(
            tensor=2 * pairs * value),
         "read_write_bytes": {"read": 0, "write": output_rows * value * 2}},
    ))
    stage_ns = tuple(source_stage_roofline_ns(stage["demand"], hardware,
        sum(stage["read_write_bytes"].values()), sum(stage["read_write_bytes"].values())) for stage in stages)
    tiles = (sum(-(-min(topk, kv - q + offset + 1) // 64) * packs
            for q, kv in zip(q_lens, kv_lens, strict=True) for offset in range(q))
        if kernel.variant == "dsa_sparse_decode" and spec_decode else
        sum(-(-count // 64) * packs for count in kernel.attributes["selected_kv_lens"])
        if kernel.variant == "dsa_sparse_decode" else sum(
            -(-min(topk, kv - q + offset + 1) // 64) * packs
            for q, kv in zip(q_lens, kv_lens, strict=True) for offset in range(q)))
    return {"stages": stages, "predicted_stage_ns": stage_ns, "predicted_ns": sum(stage_ns),
        "selected_pairs": selected,
        "padded_score_pairs": pairs, "padded_heads": padded,
        "selected_kv_read_bytes": packs * selected * qk * 2,
        "index_state_read_bytes": selected * 4,
        "score_and_lse_write_bytes": 2 * output_rows * 4,
        "output_backing_write_bytes": output_rows * value * 2,
        "analytical_live_capacity_bytes": (rows * padded * qk * 2 + rows * topk * 4
            + 2 * output_rows * 4 + output_rows * value * 2),
        "capacity_scope": "per-layer padded Q/index/LSE/output backing; not allocator peak",
        "selected_kv_tiles": tiles,
        "limitations": ("serial_source_stage_roofline_not_native_tile_overlap",
            "one_shared_bf16_cache_payload_read_per_selected_row_and_head_pack",
            "cache_residency_and_paged_transaction_counts_not_observed",
            "capacity_bound_padded_heads_not_live_head_only_math")}


def dsa_sparse_decode_source_stage_work(kernel: KernelSpec, hardware):
    return dsa_sparse_source_stage_work(kernel, hardware)


def _zero_demand():
    return {role: 0 for role in ("compute_tensor", "compute_vector", "compute_fma",
        "compute_alu", "compute_reduction", "compute_sfu")}


def _attention_demand(tensor=0, fma=0, sfu=0, alu=0):
    return {"compute_tensor": tensor, "compute_vector": 0, "compute_fma": fma,
        "compute_alu": alu, "compute_reduction": 0, "compute_sfu": sfu}


def _dsa_model(kernel: KernelSpec, hardware):
    from src.kernel.memory_access_builder import sparse_attention_memory_access_program
    from src.kernel.memory_path_solver import sparse_attention_source_demands
    source = sparse_attention_source_demands(
        sparse_attention_memory_access_program(kernel, hardware), expected_kernel=kernel)
    query_tokens, heads, qk_dim, value_dim, pairs = source["geometry"]
    read_bytes, output_bytes = float(source["read_bytes"]), float(source["write_bytes"])
    tensor_flops = float(
        2 * pairs * heads * (qk_dim + value_dim))
    score_elements = float(pairs * heads)
    flops = {
        "compute_tensor": tensor_flops,
        "compute_fma": 3.0 * score_elements,
        "compute_reduction": score_elements,
        "compute_sfu": score_elements,
        "compute_alu": float(pairs * heads),
    }
    total_bytes = read_bytes + output_bytes
    demand = build_forward_demand(total_bytes, total_bytes, True, flops)
    inp = build_aggregate_gpu_input(
        kernel,
        hardware,
        demand,
        write_l2_bytes=output_bytes,
        write_dram_bytes=output_bytes,
        limitations=(
            "flashmla_sparse_tile_pipeline_schedule_unavailable",
            "selected_kv_cache_reuse_unavailable",
            "sm90_source_execution_projected_on_sm80",
            "production_duration_calibration_not_used",
        ),
    )
    return inp, total_bytes


def _dsv4_pair_counts(kernel: KernelSpec):
    """Per-row causal capacity, preserving the selected-ID evidence scope."""
    ratio = kernel.shape["compress_ratio"]
    q_lens, kv_lens = kernel.attributes["q_lens"], kernel.attributes["kv_lens"]
    if ("mathematical_program" in kernel.attributes and kernel.attributes["mathematical_program"]
            == "vllm_0_30_dspark_swa_noncausal_softmax_v1"):
        # ComputeDSparkNoncausalSWAIndicesKernel uses one block-anchored
        # trailing window plus EVERY draft query, including future rows.
        return {"compressed": 0, "swa": sum(q * live
            for q, live in zip(q_lens, kernel.attributes["swa_kv_lens"], strict=True))}
    selected = kernel.attributes["selected_kv_lens"]
    if kernel.attributes["compressed_selection"] == "all_completed_states":
        capacities = (kernel.attributes["topk_capacity"],) * len(q_lens)
    else:
        capacities = selected
    compressed = sum(min(capacity, position // ratio)
        for q, kv, capacity in zip(q_lens, kv_lens, capacities, strict=True)
        for position in range(kv - q + 1, kv + 1)) if ratio > 1 else 0
    swa = causal_attention_pairs(q_lens, kv_lens, capacity=kernel.shape["window_size"])
    return {"compressed": compressed, "swa": swa}


def dsv4_attention_source_inputs(kernel: KernelSpec):
    """Known V4 wrapper inputs, without inventing the FlashMLA callee work."""
    if kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant not in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
        raise ValueError("V4 attention source inputs require sparse prefill or decode")
    heads = kernel.shape["num_query_heads"]
    return {"attention_sink": {"dtype": "fp32", "elements": heads, "bytes": 4 * heads, "access": "read",
            "scope": "unique per-layer/per-rank parameter input, not per-query reload count"},
        "sink_initialization": "padded heads initialized to -inf; live slots loaded from checkpoint",
        "checkpoint_content_bound": False, "numerical_callee_work_bound": False, "actual_provider_bound": False}


def dsv4_attention_roofline_work(kernel: KernelSpec):
    """Useful scaled-QK, sink softmax and PV mathematical program.

    The independent vLLM _flash_mla_eager reference concatenates one sink
    before softmax, then excludes its probability from PV. The explicit
    baseline decomposes softmax into max, shift, exp, sum and division; it
    does not select an online/native implementation or FP32 reduction order.
    Floating max comparisons are typed actions, not INT32 ALU instructions.
    """
    if (kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant not in (
            "dsv4_sparse_prefill", "dsv4_sparse_decode") or kernel.dtype != "bf16"):
        raise ValueError("V4 attention math requires its BF16 sparse source program")
    counts = _dsv4_pair_counts(kernel)
    heads = kernel.shape["num_query_heads"]
    rows = sum(kernel.attributes["q_lens"]) * heads
    pairs = sum(counts.values()) * heads
    qk, value = kernel.shape["qk_head_dim"], kernel.shape["v_head_dim"]
    softmax_values = pairs + rows
    gather_values = (dsv4_prefill_gather_logical_work(kernel)["common_useful_fp_multiply_values"]
        if kernel.variant == "dsv4_sparse_prefill" else 0)
    primitives = {"qk_dot_flops": 2 * pairs * qk, "qk_scale_multiply": pairs,
        "softmax_shift_subtract": softmax_values, "softmax_exp": softmax_values,
        "softmax_sum_add": pairs, "softmax_normalize_divide": softmax_values,
        "pv_dot_flops": 2 * pairs * value, "gather_dequant_multiply": gather_values}
    return {"pair_capacities": counts, "score_pairs": pairs, "softmax_rows": rows,
        "softmax_values_including_sink": softmax_values, "mathematical_primitives": primitives,
        "typed_actions": {"fp32_softmax_max_comparisons": pairs},
        "demand": {"compute_tensor": primitives["qk_dot_flops"] + primitives["pv_dot_flops"],
            "compute_vector": 0, "compute_reduction": 0,
            "compute_fma": primitives["qk_scale_multiply"] + primitives["softmax_shift_subtract"]
                + primitives["softmax_sum_add"] + primitives["softmax_normalize_divide"] + gather_values,
            "compute_sfu": primitives["softmax_exp"], "compute_alu": 0},
        "softmax_decomposition": "stable normalized exponentials with one division per output; mathematical baseline, not selected library instructions",
        "actual_provider_bound": False,
        "limitations": (("declared DSpark all-valid block-anchored window plus all query rows; physical indices unbound"
                if "mathematical_program" in kernel.attributes and kernel.attributes["mathematical_program"]
                    == "vllm_0_30_dspark_swa_noncausal_softmax_v1"
                else "per-row selected causal capacities, not checkpoint top-k IDs"),
            "padded query-head source geometry, not compiler masked/online/split work",
            "typed max and address/selection services unpriced; no finite-precision equivalence asserted",
            "existing memory role-use projection, not observed physical traffic")}


def dsv4_attention_mathematical_work(kernel: KernelSpec):
    """Ideal sparse math and logical operand-role bytes, without native splits.

    Uses the existing useful softmax and source gather/index ledgers. No
    split/combine temporary or physical DRAM/head-pack reload is inferred.
    Packed cache records project584 bytes (the source gather alternatives
    retain their583/584-byte payload distinction in the gather ledger).
    """
    work = dsv4_attention_roofline_work(kernel)
    heads, qk, value = (kernel.shape[key] for key in ("num_query_heads", "qk_head_dim", "v_head_dim"))
    tokens = sum(kernel.attributes["q_lens"])
    payload = {"query": 2 * tokens * heads * qk, "output": 2 * tokens * heads * value,
        "sink": dsv4_attention_source_inputs(kernel)["attention_sink"]["bytes"]}
    pairs = sum(work["pair_capacities"].values())
    if kernel.variant == "dsv4_sparse_prefill":
        gather = dsv4_prefill_gather_logical_work(kernel)
        rows = sum(gather["gathered_rows"].values())
        indices = gather["index_materialization"]
        payload.update(cache_records=rows * 584, gathered_write=gather["workspace"]["write_bytes"],
            shared_kv_role_reads=2 * pairs * (qk + value),
            indices=indices["source_read_role_bytes"] + indices["source_write_role_bytes"])
    else:
        indices = dsv4_decode_attention_index_inputs(kernel)
        payload.update(cache_record_role_reads=pairs * 584,
            indices=indices["source_logical_consumer_read_bytes"])
        if kernel.shape["compress_ratio"] == 4:
            materialization = dsv4_decode_index_materialization_work(kernel)
            payload["indices"] += (materialization["source_read_role_bytes"]
                + materialization["source_write_role_bytes"]
                - indices["already_in_c4_materialization_consumer_bytes"])
    return {**work, "payload_bytes": payload, "logical_bytes": sum(payload.values()),
        "native_splits_bound": False, "physical_traffic_bound": False}


def dsv4_prefill_index_materialization_work(kernel: KernelSpec):
    """The source fill, combine producer, and attention consumer are distinct.

    cache_utils.combine_topk_swa_indices always fills the entire aligned
    INT32 output with -1, then overwrites only the live compressed/SWA
    entries and writes one length per query. The 256-worker/request Triton
    source reads five INT32 metadata scalars per worker before its loop.
    These are source role uses, not unique footprint or observed DRAM.
    """
    if kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant != "dsv4_sparse_prefill":
        raise ValueError("V4 prefill index materialization requires sparse prefill")
    tokens = sum(kernel.attributes["q_lens"])
    requests = len(kernel.attributes["q_lens"])
    ratio = kernel.shape["compress_ratio"]
    topk = 0 if ratio == 1 else kernel.attributes["topk_capacity"]
    capacity = -(-(topk + kernel.shape["window_size"]) // 128) * 128
    pairs = _dsv4_pair_counts(kernel)
    entries = pairs["compressed"] + pairs["swa"]
    fill_bytes = 4 * tokens * capacity
    metadata_bytes = requests * 256 * 5 * 4
    producer_read = 4 * pairs["compressed"] + metadata_bytes
    producer_write = 4 * (entries + tokens)
    consumer_read = 4 * (entries + tokens)
    return {
        "source": "vllm/models/deepseek_v4/common/ops/cache_utils.py:combine_topk_swa_indices/CombineTopkSwaIndicesKernel.kernel",
        "combined_capacity_per_query": capacity, "active_entries": pairs,
        "fill": {"dtype": "int32", "value": -1, "write_bytes": fill_bytes},
        "combine": {"metadata_dtype": "int32", "workers_per_request": 256,
                    "metadata_read_role_bytes": metadata_bytes,
                    "input_topk_read_bytes": 4 * pairs["compressed"],
                    "active_indices_write_bytes": 4 * entries, "lengths_write_bytes": 4 * tokens},
        "attention_consumer": {"indices_dtype": "int32", "indices_read_bytes": 4 * entries,
                               "lengths_dtype": "int32", "lengths_read_bytes": 4 * tokens},
        "source_read_role_bytes": producer_read + consumer_read,
        "source_write_role_bytes": fill_bytes + producer_write,
        "dependencies": ("fill_all_combined_indices", "combine_topk_and_swa", "flash_mla_reads_indices_and_lengths"),
        "indices_fill_and_active_store_alias": True,
        "integer_control_and_actual_launch_services_complete": False,
        "scope": "pure-text causal all-valid source program; fill and overwrite are both global writes, compiler/native traffic and integer issue service separate",
    }


def dsv4_decode_index_materialization_work(kernel: KernelSpec):
    """C4 local-to-global index producer and its logical attention consumer.

    The helper reads/stores the entire top-k width, including -1 entries.
    Only nonnegative local indices load the block table. Token validity
    changes the final length, not the index transformation or its stores.
    Short context and the explicit declared-cardinality domain have known
    counts. Unspecified scored-path masks/page reads remain symbolic; their
    existing capacity projection is separately labeled, not made exact.
    C128 builds its indices outside the per-layer wrapper and must not
    receive this C4 per-layer charge.
    """
    if kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant != "dsv4_sparse_decode" or kernel.shape["compress_ratio"] != 4:
        raise ValueError("V4 decode index materialization requires C4 sparse decode")
    tokens = sum(kernel.attributes["q_lens"])
    capacity = kernel.attributes["topk_capacity"]
    active = _dsv4_pair_counts(kernel)["compressed"]
    selection = kernel.attributes["compressed_selection"]
    exact_count = (active if selection in ("all_completed_states", "declared_causal_topk") else None)
    block_read_bytes = None if exact_count is None else 4 * exact_count
    full_entries = tokens * capacity
    # Candidate capacity alone does not assert the nonnegative mask count.
    # The separate declared-cardinality domain supplies that condition,
    # while retaining unknown actual IDs, page locations and physical traffic.
    producer_read = 4 * full_entries + 5 * tokens + (0 if block_read_bytes is None else block_read_bytes)
    producer_write = 4 * (full_entries + tokens)
    consumer_read = 4 * (active + tokens)
    return {
        "source": "vllm/models/deepseek_v4/common/ops/cache_utils.py:compute_global_topk_indices_and_lens/ComputeGlobalTopkIndicesAndLensKernel.kernel",
        "condition": "C4 decode; pure text; all tokens valid; short-context, declared causal top-k cardinality, or candidate-capacity projection",
        "selection_cardinality_source": selection,
        "tokens": tokens, "topk_capacity": capacity, "nonnegative_input_entries": exact_count,
        "nonnegative_input_entry_capacity": active,
        "producer": {"local_indices_dtype": "int32", "local_indices_read_bytes": 4 * full_entries,
                     "token_to_request_dtype": "int32", "token_to_request_read_bytes": 4 * tokens,
                     "is_valid_token_dtype": "bool", "is_valid_token_read_bytes": tokens,
                     "block_table_dtype": "int32", "block_table_read_role_bytes": block_read_bytes,
                     "block_table_read_role_bytes_per_nonnegative_entry": 4,
                     "global_indices_dtype": "int32", "global_indices_write_bytes": 4 * full_entries,
                     "lengths_dtype": "int32", "lengths_write_bytes": 4 * tokens},
        "views": {"global_indices_shape": [tokens, capacity], "global_indices_strides_elements": [capacity, 1],
                  "consumer_shape": [tokens, 1, capacity], "consumer_view_allocates": False,
                  "local_and_global_indices_alias": False, "lengths_are_fresh": True,
                  "empty_allocations_are_zero_initialized": False},
        "source_program": {"slot": "block_table[request, local_index // block_size] * block_size + local_index % block_size",
                           "negative_index_output": -1, "block_table_load_mask": "offset < topk and local_index >= 0",
                           "invalid_token_masks_only_final_length": True,
                           "padded_source_lane_extent": tokens * (-(-capacity // 1024) * 1024),
                           "nonnegative_comparisons_per_padded_source_lane": 1,
                           "length": "count(local_index >= 0) if is_valid_token else 0"},
        "attention_consumer": {"indices_dtype": "int32", "indices_read_bytes": 4 * active,
                               "lengths_dtype": "int32", "lengths_read_bytes": 4 * tokens},
        "source_read_role_bytes": producer_read + consumer_read,
        "source_write_role_bytes": producer_write,
        "scored_block_table_reads_priced": selection == "declared_causal_topk",
        "consumer_index_read_scope": ("declared causal top-k cardinality; not observed IDs"
            if selection == "declared_causal_topk" else "existing causal selected-capacity projection; not observed IDs"),
        "dependencies": ["map_full_local_index_rows_and_count_lengths", "flash_mla_reads_global_indices_and_lengths"],
        "actual_selected_ids_or_page_table_bound": False, "integer_control_or_native_services_complete": False,
        "scope": "C4 full-width materialization with short-context/declared-cardinality role counts or an explicitly incomplete candidate projection; actual IDs/physical traffic and complete attention service remain unbound"}


def dsv4_decode_attention_index_inputs(kernel: KernelSpec):
    """Typed index/length inputs of the two-tier decode attention call.

    This is the existing all-valid causal active-prefix projection, not a
    claim about FlashMLA's masked loads, replay or physical traffic. The C4
    local-to-global helper already charges its compressed consumer inputs;
    those bytes must not be added twice when exposing both input tiers.
    """
    if kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant != "dsv4_sparse_decode":
        raise ValueError("V4 decode index inputs require sparse decode")
    tokens = sum(kernel.attributes["q_lens"])
    ratio, pairs = kernel.shape["compress_ratio"], _dsv4_pair_counts(kernel)
    operands = [{"role": "swa_indices", "dtype": "int32", "elements": pairs["swa"],
                 "bytes": 4 * pairs["swa"], "access": "read"},
                {"role": "swa_lengths", "dtype": "int32", "elements": tokens,
                 "bytes": 4 * tokens, "access": "read"}]
    if ratio > 1:
        operands.extend([
            {"role": "compressed_indices", "dtype": "int32", "elements": pairs["compressed"],
             "bytes": 4 * pairs["compressed"], "access": "read"},
            {"role": "compressed_lengths", "dtype": "int32", "elements": tokens,
             "bytes": 4 * tokens, "access": "read"}])
    overlap = 4 * (pairs["compressed"] + tokens) if ratio == 4 else 0
    total = sum(operand["bytes"] for operand in operands)
    return {"operands": operands, "source_logical_consumer_read_bytes": total,
        "already_in_c4_materialization_consumer_bytes": overlap,
        "indices_shared_across_query_heads": True, "lengths_shared_across_query_heads": True,
        "selected_ids_or_actual_callee_loads_bound": False,
        "scope": ("declared DSpark all-valid block-anchored noncausal indices/length consumer payload; producer and physical indices unbound"
            if "mathematical_program" in kernel.attributes and kernel.attributes["mathematical_program"]
                == "vllm_0_30_dspark_swa_noncausal_softmax_v1" else
            "all-valid causal active-prefix input payload projection; not native loads, cache traffic or complete attention work")}


def dsv4_prefill_gather_logical_work(kernel: KernelSpec):
    """Source-conditional gather work, without inventing provider selection.

    vLLM0.30 cache_utils.py::dequantize_and_gather_k_cache chooses CuTe
    when has_cutedsl(), otherwise Triton. Both materialize one BF16 row
    shared by the mathematical K/V roles. CuTe converts/multiplies512
    values then overwrites64 tail values; Triton only dequantizes448.
    Actual compiler elimination, backend availability and local services
    are separate from the common useful448 scalar multiplications.
    """
    if kernel.kind is not KernelKind.SPARSE_ATTN or kernel.variant != "dsv4_sparse_prefill":
        raise ValueError("V4 prefill gather source work requires sparse prefill")
    q_lens, kv_lens = kernel.attributes["q_lens"], kernel.attributes["kv_lens"]
    ratio, window = kernel.shape["compress_ratio"], kernel.shape["window_size"]
    compressed = sum(kv // ratio for kv in kv_lens) if ratio > 1 else 0
    swa = sum(q + min(kv - q, window - 1)
              for q, kv in zip(q_lens, kv_lens, strict=True))
    rows = compressed + swa
    pairs = _dsv4_pair_counts(kernel)
    return {
        "gathered_rows": {"compressed": compressed, "swa": swa},
        "pair_capacities": pairs,
        "non_gather_scalar_fma_group": 3 * sum(pairs.values()) * kernel.shape["num_query_heads"],
        "common_useful_fp_multiply_values": rows * 448,
        "workspace": {"dtype": "bf16", "written_elements": rows * 512,
                      "write_bytes": rows * 512 * 2,
                      "score_and_value_alias_same_storage": True,
                      "separate_k_and_v_allocations": False},
        "backend_alternatives": [
            {"condition": "has_cutedsl()", "backend": "cutedsl",
             "input_record_bytes": rows * 584, "scale_bytes_loaded": rows * 8,
             "source_multiply_dtype": "bf16", "source_multiply_values": rows * 512,
             "overwritten_tail_multiply_values": rows * 64, "scale_exp2_values": 0,
             "scale_construction": "UE8M0 integer bits into BF16x2"},
            {"condition": "not has_cutedsl()", "backend": "triton",
             "input_record_bytes": rows * 583, "scale_bytes_loaded": rows * 7,
             "source_multiply_dtype": "fp32", "source_multiply_values": rows * 448,
             "overwritten_tail_multiply_values": 0, "scale_exp2_values": rows * 7,
             "scale_construction": "FP32 exp2(encoded_scale-127)"}],
        "source_operation_order": (["gather_compressed"] if ratio > 1 else [])
            + ["gather_swa", "combine_topk_swa_indices", "flash_mla_sparse_fwd"],
        "selected_backend": None, "numerical_equivalence_asserted": False,
        "index_materialization": dsv4_prefill_index_materialization_work(kernel),
        "scope": "common useful multiply count and conditional source payloads; metadata, compiler dead-work elimination, library/provider and launch services separate",
    }


def _dsv4_model(kernel: KernelSpec, hardware):
    """DeepSeek-V4 sparse MLA over its two-tier KV layout.

    One ``flash_mla_with_kvcache`` (decode) or chunked ``flash_mla_sparse_fwd``
    (prefill) call attends, per request, the SWA window from the
    full-resolution cache and the selected states from the compressed FP8
    cache (vLLM 0.30.0 deepseek_v4/nvidia/flashmla.py): indexer top-k on C4
    layers, the full compressed range ``floor(kv_len/128)`` on C128 layers
    (sparse_mla.py:404).  Both tiers are
    MQA-shared across the padded query heads, unlike the per-head GQA
    traffic of the DSA model above. Prefill gathers each cache union into
    one BF16 workspace, then accounts for query K/V roles separately;
    physical read replication and local/cache reuse are unbound. Each cache
    row has a single packed584B payload shared by the K/V roles. Decode's
    per-query payload projection does not infer two global reads just
    because the same values participate in QK and PV.
    """

    q_lens = tuple(kernel.attributes["q_lens"])
    kv_lens = tuple(kernel.attributes["kv_lens"])
    query_tokens = sum(q_lens)
    heads = int(kernel.shape["num_query_heads"])
    qk_dim = int(kernel.shape["qk_head_dim"])
    value_dim = int(kernel.shape["v_head_dim"])
    dtype_bytes = int(kernel.shape["dtype_bytes"])
    state_bytes = int(kernel.shape["compressed_state_bytes"])
    swa_token_bytes = int(kernel.shape["swa_token_bytes"])

    # A scored top-k row is also bounded by its own causal compressed
    # prefix. The final request's selected count cannot be multiplied by
    # every prefill row: earlier rows have fewer completed states.
    pair_counts = _dsv4_pair_counts(kernel)
    selected_pairs, swa_pairs = pair_counts["compressed"], pair_counts["swa"]
    pairs = selected_pairs + swa_pairs

    query_bytes = float(query_tokens * heads * qk_dim * dtype_bytes)
    compressed_bytes = float(selected_pairs * state_bytes)
    swa_bytes = float(swa_pairs * swa_token_bytes)
    # The shared top-k index buffer carries one int32 per (token, state).
    index_bytes = float(selected_pairs * 4)
    output_bytes = float(query_tokens * heads * value_dim * dtype_bytes)
    limitations = [
        "flashmla_sparse_tile_pipeline_schedule_unavailable",
        "selected_kv_cache_reuse_unavailable",
        "production_duration_calibration_not_used",
        "per_row_causal_selected_capacity_not_observed_topk_ids",
        "shared_KV_packed_payload_not_observed_global_reload_count",
    ]
    gather_multiply_values = 0
    if kernel.variant == "dsv4_sparse_prefill":
        # flashmla.py:319-349 gathers each request's full completed-state
        # range and SWA window union once. Both write into one BF16
        # workspace[...,512], not separate K/V allocations. sparse_swa.py
        # derives the union as q + min(prefix, window-1).
        ratio = kernel.shape["compress_ratio"]
        gather_work = dsv4_prefill_gather_logical_work(kernel)
        compressed_union = gather_work["gathered_rows"]["compressed"]
        swa_union = gather_work["gathered_rows"]["swa"]
        gather_multiply_values = gather_work["common_useful_fp_multiply_values"]
        indices = gather_work["index_materialization"]
        compressed_bytes = float(compressed_union * state_bytes)
        swa_bytes = float(swa_union * state_bytes)  # one packed584B record
        gathered_request_bytes = float((compressed_union + swa_union) * qk_dim * 2)
        # Score and value use the same workspace in two mathematical roles.
        # This role-use projection asserts neither a physical single read
        # nor local/cache reuse between those roles.
        gathered_reread_bytes = float((selected_pairs + swa_pairs) * (qk_dim + value_dim) * 2)
        # Launch fact (selector + observation grid): one CTA covers one
        # (query token, 64-head pack), so the selected states, the SWA
        # window and the gathered workspace are MQA-shared only inside one
        # 64-head pack; every additional pack re-reads the same bytes.
        # Whether those re-reads are L2-served is not decidable from any
        # duration-independent evidence available to this model
        # (review-08): the only measured H200 L2 residency curve is
        # contracted to a repeated same-operand single-pass stream and
        # forbids application to a different access mix, so no residency
        # deduction is taken.  The whole workspace re-read stream is
        # charged to DRAM — an explicit unmodeled-residency DRAM-traffic
        # upper bound; any real L2 service of the re-reads is unmodeled
        # upside, reported as residual.
        if heads % 64:
            raise ValueError(
                "dsv4_sparse_prefill requires num_query_heads divisible "
                "by the 64-head kernel tile")
        head_packs = heads // 64
        if kernel.launch.grid[0] != query_tokens * head_packs:
            raise ValueError(
                "dsv4_sparse_prefill launch grid disagrees with the "
                "64-head CTA pack count")
        # Replace the old compressed-ID-only read with the fill/combine
        # materialization and its INT32 consumer, including length rows.
        index_bytes = float(indices["source_read_role_bytes"])
        index_write_bytes = float(indices["source_write_role_bytes"])
        l2_read_bytes = (query_bytes + index_bytes + compressed_bytes + swa_bytes
                         + head_packs * gathered_reread_bytes)
        l2_write_bytes = output_bytes + gathered_request_bytes + index_write_bytes
        dram_read_bytes = (query_bytes + index_bytes + compressed_bytes
                           + swa_bytes
                           + head_packs * gathered_reread_bytes)
        dram_write_bytes = output_bytes + gathered_request_bytes + index_write_bytes
        limitations.append(
            "dsv4_prefill_workspace_residency_unmodeled_dram_upper_bound")
        limitations.append("dsv4_prefill_gather_workspace_schedule_unavailable")
        limitations.append("single_bf16_workspace_write_distinct_from_score_value_role_reads")
        limitations.append("prefill_common_useful_dequant_multiplications_counted_not_compiler_lane_work")
        limitations.append("prefill_gather_583_or584_byte_source_loads_depend_on_backend_584_byte_record_footprint_projected")
        limitations.append("prefill_gather_backend_specific_exp2_conversion_dead_tail_and_helper_launch_services_unpriced")
        limitations.append("prefill_index_fill_combine_and_consumer_source_role_bytes_not_native_transactions")
    else:
        # The split-KV decode writes one FP32 partial (value + 4-byte LSE
        # scratch per head) per (request, split) that the combine kernel
        # re-reads.  The split counts are the FlashMLA tile-scheduler plan
        # (flash_mla_sched_meta.num_splits, cumulative per request) — a
        # discrete launch fact carried by the KernelSpec contract, NOT the
        # grid z extent (the grid z is the total SM-slot count the planner
        # distributes work over, which overstates per-request splits by up
        # to two orders of magnitude at batch 32+ where the plan is one
        # split per request).
        decode_indices = dsv4_decode_attention_index_inputs(kernel)
        split_counts = tuple(kernel.attributes["split_counts"])
        split_rows = sum(
            q * s for q, s in zip(q_lens, split_counts))
        partial_bytes = float(
            2 * split_rows * heads * (value_dim * 4 + 4))
        # C4 performs a fresh full-width local-to-global transformation in
        # this layer before attention. C128 metadata is prepared elsewhere.
        if kernel.shape["compress_ratio"] == 4:
            indices = dsv4_decode_index_materialization_work(kernel)
            index_bytes = float(indices["source_read_role_bytes"]
                + decode_indices["source_logical_consumer_read_bytes"]
                - decode_indices["already_in_c4_materialization_consumer_bytes"])
            index_write_bytes = float(indices["source_write_role_bytes"])
            limitations.append("c4_decode_index_mapping_full_width_source_roles_not_native_transactions")
            if indices["nonnegative_input_entries"] is None:
                limitations.append("c4_scored_decode_page_table_reads_value_dependent_and_unpriced")
        else:
            index_bytes = float(decode_indices["source_logical_consumer_read_bytes"])
            index_write_bytes = 0.0
        limitations.append("decode_swa_and_compressed_index_length_payloads_not_native_reload_counts")
        l2_read_bytes = (query_bytes + compressed_bytes + swa_bytes
                         + index_bytes + partial_bytes)
        l2_write_bytes = output_bytes + partial_bytes + index_write_bytes
        dram_read_bytes = l2_read_bytes
        dram_write_bytes = l2_write_bytes
        limitations.append("dsv4_splitkv_combine_schedule_unavailable")
    # Both wrapper calls pass the same padded FP32 sink parameter. Its
    # unique input payload is compulsory in this logical operand account;
    # neither per-query physical reloads nor the callee's sink arithmetic
    # can be inferred from that fact.
    sink_bytes = dsv4_attention_source_inputs(kernel)["attention_sink"]["bytes"]
    l2_read_bytes += sink_bytes
    dram_read_bytes += sink_bytes
    limitations.append("attention_sink_fp32_input_role_callee_arithmetic_and_reload_count_unbound")
    from src.kernel.memory_access_builder import sparse_attention_memory_access_program
    from src.kernel.memory_path_solver import v4_sparse_source_demands
    source = v4_sparse_source_demands(
        sparse_attention_memory_access_program(kernel, hardware), expected_kernel=kernel)
    # Current operator arithmetic and launch limits are retained. The validated
    # source supplies actual bound operand roles and their pricing projection.
    pairs = source["geometry"][4]
    l2_read_bytes = dram_read_bytes = float(source["read_bytes"])
    l2_write_bytes = dram_write_bytes = float(source["write_bytes"])
    tensor_flops = float(2 * pairs * heads * (qk_dim + value_dim))
    score_elements = float(pairs * heads)
    flops = {
        "compute_tensor": tensor_flops,
        "compute_fma": 3.0 * score_elements + gather_multiply_values,
        "compute_reduction": score_elements,
        "compute_sfu": score_elements,
        "compute_alu": score_elements,
    }
    l2_total_bytes = l2_read_bytes + l2_write_bytes
    dram_total_bytes = dram_read_bytes + dram_write_bytes
    demand = build_forward_demand(l2_total_bytes, dram_total_bytes, True,
                                  flops)
    inp = build_aggregate_gpu_input(
        kernel,
        hardware,
        demand,
        write_l2_bytes=l2_write_bytes,
        write_dram_bytes=dram_write_bytes,
        limitations=tuple(limitations),
    )
    return inp, dram_total_bytes


def _model(kernel: KernelSpec, hardware):
    if kernel.variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
        return _dsv4_model(kernel, hardware)
    return _dsa_model(kernel, hardware)


class SparseAttentionPredictor(KernelPredictor):
    kind = KernelKind.SPARSE_ATTN

    def analytical_without_launch(self, kernel: KernelSpec, hardware):
        return self.roofline_without_launch(kernel, hardware)

    def roofline_without_launch(self, kernel: KernelSpec, hardware):
        if kernel.variant == "projected_attention_path":
            from src.kernel.predictors.projected_attention import source_prediction
            return source_prediction(kernel, hardware)
        if (kernel.variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode")
                and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in ("dsv4_sparse_softmax_flops_bytes_v1",
                    "vllm_0_30_dspark_swa_noncausal_softmax_v1")):
            work = dsv4_attention_mathematical_work(kernel)
            if (HWSpec.source_contract_supported("aggregate_gpu_source_roofline", hardware.hardware_id)
                    and hardware.gpu_vector_memory_only):
                # Explicit user-authorized approximation: preserve QK/PV
                # matrix work and all operand-role bytes; omit vector issue.
                demand = {role: amount if role == "compute_tensor" else 0
                    for role, amount in work["demand"].items()}
                return prediction_from_ns(roofline_ns(demand, hardware,
                    work["logical_bytes"]), "roofline")
            missing = tuple(role for role, amount in work["demand"].items()
                if amount > 0 and role not in hardware._role_map)
            if missing:
                raise NotImplementedError(f"V4 source sparse Roofline requires {missing}; native launch/splits are not required")
            return prediction_from_ns(roofline_ns(work["demand"], hardware, work["logical_bytes"]), "roofline")
        if (kernel.variant in ("dsa_sparse_decode", "dsa_sparse_prefill", "dsa_sparse_mixed")
                and kernel.attributes["source_cost_model"] == "vllm_0_30_bounded_stage_roofline_v1"):
            work = dsa_sparse_source_stage_work(kernel, hardware)
            return prediction_from_ns(work["predicted_ns"], "roofline")
        if (kernel.dtype == "bf16" and (kernel.variant == "dsa_sparse_decode"
                or (kernel.variant in ("dsa_sparse_prefill", "dsa_sparse_mixed")
                    and "mathematical_program" in kernel.attributes))):
            work = dsa_sparse_attention_logical_work(kernel)
            demand = {"compute_tensor": work["tensor_flops"], "compute_fma": work["scalar_flops"],
                "compute_sfu": work["exp_operations"] + work["reciprocal_operations"],
                "compute_vector": 0, "compute_reduction": 0, "compute_alu": 0}
            return prediction_from_ns(roofline_ns(demand, hardware, work["logical_bytes"]), "roofline")
        return super().roofline_without_launch(kernel, hardware)

    def resource_input(self, kernel: KernelSpec, hardware):
        return _model(kernel, hardware)

    def midlayer(self, kernel: KernelSpec, hardware):
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(inp)

    def roofline(self, kernel: KernelSpec, hardware):
        inp, compulsory = self.resource_input(kernel, hardware)
        if kernel.variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
            work = dsv4_attention_roofline_work(kernel)
            return prediction_from_ns(roofline_ns(work["demand"], hardware, compulsory), "roofline")
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory),
            "roofline",
        )
