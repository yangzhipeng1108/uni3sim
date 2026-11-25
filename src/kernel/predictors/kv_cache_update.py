"""Source-work analytical prediction for typed device-cache writes."""

from dataclasses import replace

from src.kernel.formula import PerformanceFormula
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.dsv4_c128 import dsv4_c128_stage_launches
from src.kernel.predictors.common import (
    build_aggregate_gpu_input,
    build_forward_demand,
    prediction_from_ns,
    roofline_ns,
    sum_serial_kernel_predictions,
)
from src.kernel.predictors.vector import (
    dense_fp16_source_logical_work, dense_bf16_source_logical_work,
    forward_analytical_resource_input,
    paged_kv_cache_resource_input,
)


def dsv4_swa_insert_tensor_objects(kernel: KernelSpec):
    """Source tensor storage of the packed V4 insert; no physical page binding."""
    if kernel.kind is not KernelKind.KV_CACHE or kernel.variant != "dsv4_qnorm_rope_kv_insert":
        raise ValueError("V4 SWA insert objects require their typed KV_CACHE contract")
    rows, heads, padded = (kernel.shape[name] for name in ("seq_len", "num_query_heads", "padded_query_heads"))
    partial = ("mathematical_program" in kernel.attributes and kernel.attributes["mathematical_program"]
        == "vllm_0_30_dspark_partial_context_insert_v1")
    valid = kernel.attributes["valid_token_count"] if partial else rows
    return {"program": "vllm_0_30_dspark_partial_context_insert_v1" if partial else "vllm_0_30_packed_swa_insert_objects",
        "condition": ("eager BF16 full six Q/KV rows; slots[0:S]>=0 and slots[S:6]=-1; only cache writes masked"
            if partial else "eager dict metadata, BF16 Q/KV, uint8 fp8_ds_mla cache, no DP padding, all slots nonnegative"),
        "q_input": {"dtype": "bf16", "shape": [rows, heads, 512], "read_only": True},
        "kv_input": {"dtype": "bf16", "shape": [rows, 512], "read_only": True},
        "metadata": {"positions": {"dtype": "int64", "shape": [rows]},
            "slots": {"dtype": "int64", "shape": [rows], "nonnegative": valid == rows,
                **({"valid_prefix_rows": valid, "invalid_tail_rows": [valid, rows], "invalid_tail_value": -1,
                    "actual_slot_values_bound": False} if partial else {})},
            "cos_sin": {"dtype": "fp32", "row_width": 64}},
        "q_output": {"dtype": "bf16", "shape": [rows, padded, 512], "strides_elements": [padded * 512, 512, 1],
            "fresh": True, "aliases_inputs_or_cache": False, "every_element_written": True,
            "zero_padding_heads": [heads, padded], "allocation_initialization_bytes": 0},
        "cache": {"dtype": "uint8", "existing_storage": True, "wrapper_view_allocates": False,
            "wrapper_view": "[num_blocks, -1]", "read_payload_bytes": 0, "write_payload_bytes": valid * 584,
            "block_base_bytes": "(slot // B) * S", "data_offset_bytes": "(slot % B) * 576",
            "rope_offset_bytes": "(slot % B) * 576 + 448",
            "scale_offset_bytes": "B * 576 + (slot % B) * 8", "B": "cache_block_size", "S": "cache.stride(0) in bytes",
            "nope_bytes": 448, "rope_bf16_bytes": 128, "real_scale_bytes": 7, "zero_scale_padding_bytes": 1,
            "source_prequant_rounding": "BF16 roundtrip before absmax/quantization",
            "allocated_or_zeroed_by_this_call": False, "physical_slots_or_page_stride_bound": False},
        "apply_q_norm": True, "kv_mxfp8": False, "global_tensor_temporaries": [],
        "kernel_local_arrays_are_global_temporaries": False,
        "scope": "source caller objects and conditional cache address program; not page allocation, cross-step ownership, compiler spills or runtime provider"}


def dsv4_swa_insert_source_work(kernel: KernelSpec):
    """Typed FP32 work of the declared BF16 Q/packed-SWA source branch."""
    objects = dsv4_swa_insert_tensor_objects(kernel)
    tokens, heads, dim, rope, padded = (kernel.shape[name] for name in
        ("seq_len", "num_query_heads", "head_dim", "rope_dim", "padded_query_heads"))
    if kernel.dtype != "bf16" or (dim, rope, kernel.shape["cache_token_bytes"]) != (512, 64, 584):
        raise ValueError("V4 SWA source arithmetic requires BF16/512/64 and packed584 records")
    values = tokens * heads * dim
    valid = kernel.attributes["valid_token_count"] if ("mathematical_program" in kernel.attributes
        and kernel.attributes["mathematical_program"] == "vllm_0_30_dspark_partial_context_insert_v1") else tokens
    # All N Q and KV inputs are decoded and RoPE'd before the slot guard.
    # Only valid KV rows execute the post-RoPE BF16 roundtrip/quant/store.
    # Q warpSum adds are floating-point additions. KV fmaxf is a typed
    # comparison, and the encoded exponent has28 FP32 +127 expressions.
    fma = (3 * values + 2 * tokens * heads + 160 * tokens * heads
        + 3 * tokens * (heads + 1) * rope + valid * (448 + 32 + 28))
    return {"objects": objects,
        "demand": {"compute_tensor": 0, "compute_reduction": 0, "compute_fma": fma,
            "compute_alu": 0, "compute_sfu": tokens * heads + 64 * valid},
        "typed_actions": {"bf16_to_fp32": values + 512 * tokens + 512 * valid,
            "fp32_to_bf16": values + 576 * valid, "fp32_to_fp8": 448 * valid,
            "fp32_to_uint8": 7 * valid, "fp32_abs": 512 * valid,
            "fp32_maximum": 1084 * valid, "fp32_minimum": 476 * valid,
            "fp32_ceil": 32 * valid, "fp32_negate": 32 * valid},
        "read_write_bytes": {"read": 2 * tokens * (heads + 1) * dim + 16 * tokens + 4 * tokens * rope,
            "write": 2 * tokens * padded * dim + 584 * valid},
        "unpriced": ("typed conversions, abs/min/max/ceil/negate and address/control instructions",
            "Q RMS scalar lane replication beyond logical per-head mean/epsilon/rsqrt",
            "native compiler/provider, physical cache/DRAM and allocator"),
        "scope": "source scalar Q square-accumulate and full warpSum, live RoPE and all32 KV scale lanes; not compiled instruction issue"}


def dsv4_partial_state_tensor_objects(kernel: KernelSpec):
    """FP32 split views and persistent writes; no physical-page alias assertion."""
    if kernel.kind is not KernelKind.KV_CACHE or kernel.variant != "dsv4_compressor_state_cache":
        raise ValueError("V4 partial state objects require their KV_CACHE producer")
    rows, record_bytes = kernel.shape["seq_len"], kernel.shape["state_bytes"]
    width = record_bytes // 8
    return {"projection_dtype": "fp32", "projection_shape": [rows, 2 * width],
        "split_views": [{"role": role, "dtype": "fp32", "shape": [rows, width],
            "strides_elements": [2 * width, 1], "offset_elements": offset, "allocates": False}
            for role, offset in (("kv", 0), ("score", width))],
        "ape_read_dtype": "fp32", "positions_dtype": "int64", "slot_dtype": "int64",
        "source_read_bytes": 12 * rows * width + 16 * rows,
        "source_write_bytes": 8 * rows * width, "score_ape_fp32_additions": rows * width,
        "writes": [{"role": "kv_state", "dtype": "fp32", "record_offset_elements": 0, "elements_per_row": width},
                   {"role": "score_state", "dtype": "fp32", "record_offset_elements": width, "elements_per_row": width}],
        "state_write_in_place": True, "allocates_state_cache": False, "returns_tensor": False,
        "global_tensor_temporaries": [], "pdl_grid_dependency_used": False,
        "physical_page_alias_or_cross_step_reuse_bound": False,
        "scope": "all-valid source payload and FP32 caller objects only; persistent page alias, reuse, history and actual provider remain unchecked"}


def dsv4_finish_partial_state_input(kernel: KernelSpec):
    """C128 store reads its fresh compressed tensor, not persistent state."""
    if kernel.kind is not KernelKind.KV_CACHE or kernel.variant != "dsv4_compressed_state_cache":
        raise ValueError("V4 finish state input requires its KV_CACHE consumer")
    if kernel.attributes["finish_stage"] == "store":
        return None
    dimension = 128 if kernel.shape["state_bytes"] == 132 else 512
    ratio = kernel.shape["compress_ratio"]
    width = dimension * (2 if ratio == 4 else 1)
    work = dsv4_finish_logical_work(kernel)
    return {"dtype": "fp32", "head_dim": dimension, "compress_ratio": ratio,
        "state_record_elements": 2 * width, "state_width": width,
        "state_address_program": {
            "block_rows_source": "state_metadata.block_size", "block_rows": None if ratio == 4 else 8,
            "constructor_default": 4 if ratio == 4 else 8,
            "C4_domain": "positive multiple of 4", "C128_supported_program_requires_8": True,
            "page_index": "absolute_position // block_rows", "row_offset": "absolute_position % block_rows",
            "same_block_rows_as_save_partial": True},
        "value_columns": "(window_row // 4) * head_dim + column" if ratio == 4 else "column",
        "score_columns": "state_width + value_column", "valid_window_rows": work["valid_window_rows"],
        "payload_read_bytes": work["valid_window_rows"] * dimension * 8,
        "cache_mutated_by_finish": False, "global_cast_or_copy_before_read": False,
        "scope": "source persistent KV/score reads only; C128 store-stage compressed_kv and cache metadata are separate operands"}


def _dsv4_finish_metadata_reads(tokens, states, rows, dimension, ratio, stage):
    """Typed baseline operand roles; replicated source/native loads separate."""
    roles = []

    def read(part, role, dtype, count, condition):
        roles.append({"stage": part, "role": role, "dtype": dtype, "elements": count,
            "bytes": count * (4 if dtype == "int32" else 8), "condition": condition})

    if stage != "store":
        read("compress", "state_slots", "int64", tokens, "all valid scheduled tokens")
        read("compress", "positions", "int64", tokens, "all valid scheduled tokens")
        if ratio == 4 and dimension == 512:
            read("compress", "destination_slots", "int64", tokens, "before boundary guard")
        read("compress", "request_indices", "int32", tokens if ratio == 128 else states,
             "before boundary guard" if ratio == 128 else "completed boundaries only")
        read("compress", "state_block_table", "int32", rows, "existing valid-window row/split operand projection")
    if stage != "compress":
        if ratio == 128:
            read("store", "state_slots", "int64", tokens, "all valid scheduled tokens")
            read("store", "positions", "int64", tokens, "all valid scheduled tokens")
            read("store", "destination_slots", "int64", tokens, "before boundary guard")
        elif dimension == 128:
            read("store", "destination_slots", "int64", states, "completed boundaries only")
    return {"roles": roles, "baseline_read_bytes": sum(role["bytes"] for role in roles),
        "scope": "all-valid baseline metadata operand projection; source load replication, physical pages and observed traffic separate"}


def dsv4_finish_logical_work(kernel: KernelSpec):
    """V4 finish value arithmetic, typed program and source temporary.

    This is an active-value Roofline projection. Source vector slots and
    explicit BF16/FP8 conversions are described separately, without assigning
    compiler instructions or a throughput to them. Slot mappings are assumed
    valid as declared by the existing finish contract, not GPU observations.
    """
    if kernel.kind is not KernelKind.KV_CACHE or kernel.variant != "dsv4_compressed_state_cache":
        raise ValueError("V4 finish work requires its typed cache KernelSpec")
    width = 128 if kernel.shape["state_bytes"] == 132 else 512
    ratio = kernel.shape["compress_ratio"]
    stage = kernel.attributes["finish_stage"]
    compress, store = stage in (None, "compress"), stage in (None, "store")
    ends = [list(range((old // ratio + 1) * ratio, new + 1, ratio)) for old, new in zip(
        kernel.attributes["state_old_lens"], kernel.attributes["state_new_lens"], strict=True)]
    states = sum(len(values) for values in ends)
    window = 2 * ratio if ratio == 4 else ratio
    rows = sum(min(window, end) for values in ends for end in values)
    values, outputs = rows * width, states * width
    groups = {name: {} for name in ("compute_fma", "compute_alu", "compute_reduction", "compute_sfu", "compute_tensor")}
    # Basic Roofline counts ordinary FP32 FLOPs, including scalar divisions
    # and reduction additions. Neither INT32 issue throughput nor a measured
    # block-reduction dependency service is their mathematical FLOP ceiling.
    # Comparisons/abs/ceil/sign retain typed counts without an invented rate.
    fma, sfu = (groups[name] for name in ("compute_fma", "compute_sfu"))
    typed_actions = {}

    def action(name, operation, elements):
        typed_actions[name] = {"dtype": "fp32", "operation": operation, "elements": elements}

    if compress:
        if width == 512 and ratio == 4:
            action("online_max", "maximum", values)
            fma.update({"old_new_score_subtract": 2 * values,
                        "old_new_log2e_multiply": 2 * values, "compressed_value_divide": outputs})
            fma.update({"online_sum_multiply_add": 2 * values, "online_product_rescale_multiply": values,
                        "weighted_value_multiply_add": 2 * values})
            sfu["online_old_new_exp2"] = 2 * values
        elif ratio == 128:
            action("local_max", "maximum", values)
            action("eight_lane_merge_max", "maximum", 24 * outputs)
            fma.update({"score_subtract": values, "score_log2e_multiply": values,
                        "merge_max_subtract": 8 * outputs, "merge_log2e_multiply": 8 * outputs,
                        "compressed_value_divide": outputs})
            fma.update({"local_exp_sum_add": values, "weighted_value_multiply_add": 2 * values,
                        "merge_sum_product_rescale_multiply": 16 * outputs})
            fma.update({"eight_lane_merge_sum": 24 * outputs,
                        "eight_lane_merge_product": 24 * outputs})
            sfu.update({"local_score_exp2": values, "merge_rescale_exp2": 8 * outputs})
        else:
            fma.update({"softmax_score_subtract": values, "softmax_normalize_divide": values})
            fma["weighted_value_multiply"] = values
            action("softmax_max", "maximum", values - outputs)
            fma.update({"softmax_sum": values - outputs,
                        "weighted_value_sum": values - outputs})
            sfu["softmax_exp"] = values
    if store:
        quant_values, blocks = (outputs, states) if width == 128 else (448 * states, 7 * states)
        fma.update({"norm_square": outputs, "norm_rrms_multiply": outputs, "norm_gamma_multiply": outputs,
                    "rope_multiply": 128 * states, "rope_add_subtract": 64 * states,
                    "quant_value_scale_multiply": quant_values})
        fma.update({"norm_sum": outputs - states,
                    "norm_variance_divide": states, "norm_eps_add": states})
        for name, operation, elements in (
                ("quant_absmax", "maximum", quant_values - blocks),
                ("quant_abs", "absolute", quant_values),
                ("quant_lower_clamp", "maximum", quant_values),
                ("quant_upper_clamp", "minimum", quant_values),
                ("quant_scale_minimum_max", "maximum", blocks)):
            action(name, operation, elements)
        sfu["norm_rsqrt"] = states
        if width == 128:
            fma["quant_raw_scale_multiply"] = blocks
            action("quant_exponent_ceil", "ceil", blocks)
            action("quant_inverse_exponent_negate", "negate", blocks)
            sfu.update({"quant_scale_log2": blocks, "quant_inverse_scale_exp2": blocks,
                        "quant_stored_scale_exp2": blocks})
        else:
            fma["quant_raw_scale_divide"] = blocks
    source_temporary = None
    if ratio == 128:
        written = []
        offset = 0
        for old, new, boundaries in zip(kernel.attributes["state_old_lens"], kernel.attributes["state_new_lens"], ends, strict=True):
            written.extend(offset + end - old - 1 for end in boundaries)
            offset += new - old
        source_temporary = {"role": "compressed_kv", "dtype": "fp32", "shape": [kernel.shape["seq_len"], width],
            "source_extent_bytes": 4 * kernel.shape["seq_len"] * width, "initialized_by_allocation": False,
            "written_token_rows": written, "read_token_rows": written, "producer_stage": "compress", "consumer_stage": "store",
            "source_alias": "same compressed_kv tensor argument", "actual_allocator_binding_verified": False}
    source_replication = None
    if store and width == 512:
        replicas = 32 if ratio == 128 else 8
        source_replication = {
            "norm_rsqrt_expressions": states * (256 if ratio == 128 else 1),
            # C128: pair sum, five warp sums, three eight-lane sums on
            # all width/2 threads. C4: initialized per-value sums, five
            # warp sums on width/8 threads and a two-warp final sum.
            "norm_sum_add_expressions": states * ((width // 2) * (1 + 5 + 3) if ratio == 128
                                                 else width + (width // 8) * 5 + width // 256),
            "quant_scale_threads_per_block": replicas,
            "raw_scale_divide_expressions": 7 * states * replicas,
            "quant_scale_minimum_max_expressions": 7 * states * replicas,
            "ue8m0_integer_expressions": 5 * 7 * states * replicas,
            "quant_max_expressions": 7 * states * (32 * (1 + 5) if ratio == 128 else 8 * (8 + 3)),
            "scope": "source scalar expressions, not compiler instructions or baseline replicated service"}
    return {"stage": stage, "state_dimension": width, "completed_states": states, "valid_window_rows": rows,
        "boundary_ends_by_request": ends, "window_rows_per_boundary": window,
        "state_half_selector": "window_row // 4" if ratio == 4 else "0",
        "arithmetic_groups": groups, "demand": {name: sum(counts.values()) for name, counts in groups.items()},
        "unpriced_typed_actions": typed_actions,
        "store_program": None if not store else {
            "normalization_input_dtype": "fp32", "gamma_storage_dtype": "bf16", "normalization_math_dtype": "fp32",
            "rope_math_dtype": "fp32", "rotary_position": "boundary_end_exclusive - compress_ratio",
            "quant_pre_cast": "fp32 -> bf16 -> fp32", "fp8_value_dtype": "e4m3",
            "quantized_values_per_state": 128 if width == 128 else 448,
            "quant_group_size": 128 if width == 128 else 64,
            "quantization_position": "after RoPE on all 128 values" if width == 128 else "NoPE branch only; RoPE branch stored BF16",
            "raw_scale_expression": "max(absmax, 1e-4) * (1 / 448)" if width == 128 else "max(1e-4, absmax / 448)",
            "scale_rounding": "ceil(log2(raw_scale))" if width == 128 else "((fp32_bits + 0x7fffff) >> 23) & 0xff",
            "inverse_scale_expression": "exp2(-exponent)" if width == 128 else "reinterpret_fp32((254 - ue8m0) << 23)",
            "stored_scale_expression": "exp2(exponent)" if width == 128 else "ue8m0 byte",
            "fp8_conversion": "tl.float8e4nv" if width == 128 else "_fp32x2_to_fp8e4m3x2",
            "scale_storage_dtype": "fp32" if width == 128 else "uint8_ue8m0",
            "ue8m0_bit_operations_per_block": 0 if width == 128 else 5,
            "cache_address_program": {
                "condition": "all valid completed boundaries; uint8 packed main/indexer cache",
                "slot_source": "k_cache_metadata.slot_mapping", "slot_dtype": "int64",
                "state_slot_tensor_used_for_output_address": False,
                "block_rows_source": "kv_cache.shape[1]", "block_stride_source": "kv_cache.stride(0) in bytes",
                "block_rows_symbol": "B", "block_stride_symbol": "S",
                "data_stride_bytes": 128 if width == 128 else 576, "scale_stride_bytes": 4 if width == 128 else 8,
                "page_index": "destination_slot // B", "row_offset": "destination_slot % B",
                "page_base_bytes": "(destination_slot // B) * S",
                "data_base_bytes": "page_base + (destination_slot % B) * D",
                "scale_base_bytes": "page_base + B * D + (destination_slot % B) * Q",
                "per_token_AoS_layout_assumed": False,
                "per_state_regions": ([
                    {"role": "fp8_values", "base": "data_base", "offset_bytes": 0, "bytes": 128, "dtype": "e4m3"},
                    {"role": "scale", "base": "scale_base", "offset_bytes": 0, "bytes": 4, "dtype": "fp32"}]
                    if width == 128 else [
                    {"role": "nope_values", "base": "data_base", "offset_bytes": 0, "bytes": 448, "dtype": "e4m3"},
                    {"role": "rope_values", "base": "data_base", "offset_bytes": 448, "bytes": 128, "dtype": "bf16"},
                    {"role": "scales", "base": "scale_base", "offset_bytes": 0, "bytes": 7, "dtype": "uint8_ue8m0"},
                    {"role": "scale_padding", "base": "scale_base", "offset_bytes": 7, "bytes": 1, "dtype": "uint8", "value": 0}]),
                "page_domain": "B positive; S >= B*(D+Q); byte-addressable cache with source store alignment",
                "metadata_values_or_physical_pages_bound": False}},
        "conversion_value_counts": None if not store else {
            "gamma_bf16_to_fp32": outputs, "pre_quant_fp32_to_bf16": outputs if width == 128 else 448 * states,
            "pre_quant_bf16_to_fp32": outputs if width == 128 else 448 * states,
            "fp32_to_fp8": outputs if width == 128 else 448 * states,
            "rope_fp32_to_bf16": 0 if width == 128 else states * 64},
        "metadata_reads": _dsv4_finish_metadata_reads(kernel.shape["seq_len"], states, rows, width, ratio, stage),
        "source_replication": source_replication,
        "source_temporary": source_temporary,
        "indexer_vector_domain": None if width != 128 or not compress else {
            "score_cells": states * window * width, "active_score_cells": values,
            "masked_negative_position_cells": states * window * width - values,
            "nonrotary_identity_rope_values": states * 64, "nonrotary_identity_rope_arithmetic": states * 64 * 3},
        "scope": "active-value arithmetic/byte Roofline; minimum norm/quant trees, explicit C128 merge; not native issue counts",
        "unpriced_source_work": ["FP32 maximum/minimum, absolute, ceil and sign actions; not INT32 ALU operations",
            "compiler mapping and dependency service of floating reductions and division", "BF16/FP8 typed conversions", "main UE8M0 integer bit construction",
            "normalization/quantizer lane replication", "masked indexer vector work and identity RoPE",
            "address, guards, barriers and shared-memory instructions"]}


def target_cache_logical_work(kernel: KernelSpec):
    """Existing cache operand/arithmetic ledger without a launch or HW model."""
    if kernel.kind is not KernelKind.KV_CACHE:
        raise ValueError("cache logical work requires a KV_CACHE KernelSpec")
    variant = kernel.variant
    tokens = int(kernel.shape["seq_len"])
    slot_bytes = float(8 * tokens)
    if variant == "mla_kv_cache":
        elements = int(kernel.shape["element_count"])
        tensor_bytes = float(elements * int(kernel.shape["dtype_bytes"]))
        read_bytes = tensor_bytes + slot_bytes
        write_bytes = tensor_bytes
        flops = {
            "compute_tensor": 0.0,
            "compute_fma": 0.0,
            "compute_alu": float(elements + tokens),
            "compute_sfu": 0.0,
        }
        limitations = (
            "mla_cache_append_cta_pipeline_schedule_unavailable",
            "production_duration_calibration_not_used",
        )
    elif variant == "dsa_fp8_index_cache":
        key_dim = int(kernel.shape["key_dim"])
        scale_bytes = int(kernel.shape["scale_bytes_per_token"])
        key_elements = tokens * key_dim
        scale_count = tokens * scale_bytes // 4
        read_bytes = float(2 * key_elements) + slot_bytes
        write_bytes = float(key_elements + tokens * scale_bytes)
        # The fixed CUDA source reduces one absolute maximum per quantization
        # block, derives one UE8M0 scale, then converts every BF16 key element
        # to E4M3. Count each source-level action once.
        flops = {
            "compute_tensor": 0.0,
            "compute_reduction": float(key_elements),
            "compute_fma": float(key_elements),
            "compute_alu": float(key_elements + tokens),
            "compute_sfu": float(scale_count),
        }
        limitations = (
            "fp8_quantization_warp_pipeline_schedule_unavailable",
            "production_duration_calibration_not_used",
        )
    elif variant == "dsv4_compressed_state_cache":
        # save_partial_states is a separate producer. Finish reads its FP32
        # KV/score window, softmax-weights KV, then norms/rotates/quantizes
        # the completed record. C4 overlaps two coefficient windows; C128
        # uses two source launches with a materialized FP32 scratch row.
        state_bytes = int(kernel.shape["state_bytes"])
        state_dim = 128 if state_bytes == 132 else 512
        ratio = kernel.shape["compress_ratio"]
        states = sum(new // ratio - old // ratio for old, new in zip(
            kernel.attributes["state_old_lens"], kernel.attributes["state_new_lens"], strict=True))
        window = 2 * ratio if ratio == 4 else ratio
        rows = sum(
            min(window, end)
            for old, new in zip(kernel.attributes["state_old_lens"],
                                kernel.attributes["state_new_lens"], strict=True)
            for end in range((old // ratio + 1) * ratio, new + 1, ratio))
        window_elements = rows * state_dim
        state_elements = states * state_dim
        stage = kernel.attributes["finish_stage"]
        compress = stage in (None, "compress")
        store = stage in (None, "store")
        work = dsv4_finish_logical_work(kernel)
        read_bytes = work["metadata_reads"]["baseline_read_bytes"]
        write_bytes = 0.0
        if compress:
            read_bytes += 8 * window_elements
            if ratio == 128:
                write_bytes += 4 * state_elements
        if store:
            if ratio == 128:
                read_bytes += 4 * state_elements
            read_bytes += states * (2 * state_dim + 4 * 64)
            write_bytes += states * state_bytes
        flops = {name: float(count) for name, count in work["demand"].items()}
        limitations = (
            "dsv4_finish_roofline_only",
            "fp32_state_window_reads_save_partial_producer_not_repeated",
            "all_rows_valid_slots_projection_actual_mapping_unvalidated",
            "per_boundary_operand_roles_not_observed_dram_or_cache_reuse",
            "source_value_arithmetic_and_exp2_not_compiler_padded_instruction_counts",
            "fp32_math_not_int32_alu_or_block_reduction_service_counts",
            "norm_and_quant_reduction_logical_work_warp_replication_unpriced",
            "bf16_roundtrip_and_fp8_conversion_intrinsic_work_unpriced",
            "main_cute_scale_integer_bit_construction_not_sfu_work",
            "ue8m0_integer_bit_construction_typed_service_not_instruction_priced",
            "local_smem_and_address_control_work_unpriced",
            "bf16_norm_gamma_and_fp32_rope_source_config_projection",
            "cuda_vllm_0_30_source_launch_not_executed_or_compiler_verified",
            "production_duration_calibration_not_used",
        )
        if ratio == 128 and stage is None:
            limitations += ("c128_two_launch_aggregate_work_not_one_physical_kernel",)
    elif variant == "dsv4_qnorm_rope_kv_insert":
        work = dsv4_swa_insert_source_work(kernel)
        read_bytes, write_bytes = (float(work["read_write_bytes"][access]) for access in ("read", "write"))
        flops = {role: float(value) for role, value in work["demand"].items()}
        limitations = (
            "dsv4_fused_qnorm_rope_kv_insert_roofline_only",
            "logical_unique_traffic_not_observed_dram",
            "cos_sin_traffic_upper_bound_distinct_token_positions",
            "scalar_source_operations_exclude_index_address_arithmetic",
            "production_duration_calibration_not_used",
        )
    elif variant == "dsv4_compressor_state_cache":
        # deepseek_v4/compressor.py _SAVE_PARTIAL_STATES_KERNEL: split the
        # per-token FP32 kv/score projection, add the FP32 APE table and write
        # one FP32 kv_state+score_state record per token.
        state_bytes = int(kernel.shape["state_bytes"])
        state_elements = tokens * state_bytes // 4
        input_bytes = 4 * state_elements
        # Every active row reads its score-width APE row, int64 position and
        # int64 slot. Cross-token APE reuse/invalid-slot skipping is unbound.
        ape_bytes = 2 * state_elements
        read_bytes = float(input_bytes + ape_bytes + 16 * tokens)
        write_bytes = float(tokens * state_bytes)
        flops = {
            "compute_tensor": 0.0,
            "compute_reduction": 0.0,
            "compute_fma": float(state_elements // 2),
            "compute_alu": float(tokens),
            "compute_sfu": 0.0,
        }
        limitations = (
            "dsv4_save_partial_states_pipeline_schedule_unavailable",
            "ape_row_read_per_active_token_cross_token_reuse_unmodeled",
            "slot_address_and_mask_control_integer_work_uncounted",
            "all_rows_valid_slots_projection_actual_mapping_unvalidated",
            "production_duration_calibration_not_used",
        )
    else:
        raise ValueError(
            f"unknown KV_CACHE variant: {kernel.variant!r}")
    return {"read_bytes": read_bytes, "write_bytes": write_bytes,
        "demand": flops, "limitations": limitations}


def _target_cache_input(kernel: KernelSpec, hardware):
    work = target_cache_logical_work(kernel)
    read_bytes, write_bytes = work["read_bytes"], work["write_bytes"]
    if kernel.execution_support == "supported":
        from src.kernel.memory_access_builder import target_cache_memory_access_program
        from src.kernel.memory_cache_update import target_cache_source_demands
        source = target_cache_source_demands(
            target_cache_memory_access_program(kernel, hardware), expected_kernel=kernel)
        read_bytes = float(source["legacy_unique_slot_read_price_bytes"] if kernel.variant in (
            "mla_kv_cache", "dsa_fp8_index_cache") else source["read_bytes"])
        write_bytes = float(source["write_bytes"])
    total_bytes = read_bytes + write_bytes
    flops, limitations = work["demand"], work["limitations"]
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


def _model(kernel: KernelSpec, hardware):
    if kernel.variant == "paged_kv_cache":
        return paged_kv_cache_resource_input(kernel, hardware)
    return _target_cache_input(kernel, hardware)


def _dsv4_c128_stage_fields(kernel: KernelSpec):
    if (kernel.variant != "dsv4_compressed_state_cache"
            or kernel.shape["compress_ratio"] != 128
            or kernel.attributes["finish_stage"] is not None):
        return None
    states = sum(new // 128 - old // 128 for old, new in zip(
        kernel.attributes["state_old_lens"], kernel.attributes["state_new_lens"], strict=True))
    fields = []
    for stage in ("compress", "store"):
        shape = ({**kernel.shape, "element_count": states * 512, "dtype_bytes": 4}
            if stage == "compress" else dict(kernel.shape))
        dtype = "fp32" if stage == "compress" else kernel.dtype
        attributes = {**kernel.attributes, "finish_stage": stage}
        fields.append((stage, shape, dtype, attributes))
    return tuple(fields)


def _dsv4_c128_stage_kernels(kernel: KernelSpec):
    fields = _dsv4_c128_stage_fields(kernel)
    if fields is None:
        return None
    launches = dict(dsv4_c128_stage_launches(kernel.launch))
    return tuple((stage, replace(kernel, shape=shape, dtype=dtype,
        attributes=attributes, launch=launches[stage]))
        for stage, shape, dtype, attributes in fields)


def _dsv4_c128_without_launch_stage_kernels(kernel: KernelSpec):
    fields = _dsv4_c128_stage_fields(kernel)
    if fields is None:
        return None
    return tuple((stage, replace(kernel, shape=shape, dtype=dtype,
        attributes=attributes, launch=None, execution_support="unsupported",
        unsupported_reason="launch_selector_unavailable"))
        for stage, shape, dtype, attributes in fields)


class KVCacheUpdatePredictor(KernelPredictor):
    kind = KernelKind.KV_CACHE

    def analytical_without_launch(self, kernel: KernelSpec, hardware):
        if (kernel.variant == "dsv4_qnorm_rope_kv_insert" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "vllm_0_30_dspark_partial_context_insert_v1"):
            return replace(self.roofline_without_launch(kernel, hardware),
                bottleneck_group="analytical_partial_context_insert")
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "dense_bf16_source_flops_bytes_v1"):
            return self.roofline_without_launch(kernel, hardware)
        return super().analytical_without_launch(kernel, hardware)

    def resource_input(self, kernel: KernelSpec, hardware):
        if (kernel.variant == "dsv4_qnorm_rope_kv_insert" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "vllm_0_30_dspark_partial_context_insert_v1"):
            raise NotImplementedError("DSpark partial context insert native provider and physical slot services are unbound")
        return _model(kernel, hardware)

    def midlayer(self, kernel: KernelSpec, hardware):
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in ("dense_fp16_source_flops_bytes_v1", "dense_bf16_source_flops_bytes_v1")):
            raise NotImplementedError("dense FP16 mathematical work has no native launch")
        stage_kernels = _dsv4_c128_stage_kernels(kernel)
        if stage_kernels is not None:
            formula = PerformanceFormula(hardware)
            return sum_serial_kernel_predictions((stage, formula.predict_kernel(
                self.resource_input(stage_kernel, hardware)[0])) for stage, stage_kernel in stage_kernels)
        if kernel.variant == "paged_kv_cache":
            inp, _ = forward_analytical_resource_input(kernel, hardware)
        else:
            inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(inp)

    def roofline_without_launch(self, kernel: KernelSpec, hardware):
        if (kernel.variant == "dsv4_qnorm_rope_kv_insert" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "vllm_0_30_dspark_partial_context_insert_v1"):
            work = target_cache_logical_work(kernel)
            demand = {role: 0 for role in ("compute_tensor", "compute_vector", "compute_reduction",
                "compute_fma", "compute_sfu", "compute_alu")}
            demand.update(work["demand"])
            return prediction_from_ns(roofline_ns(demand, hardware,
                work["read_bytes"] + work["write_bytes"]), "roofline")
        stage_kernels = _dsv4_c128_without_launch_stage_kernels(kernel)
        if stage_kernels is not None:
            return prediction_from_ns(1e9 * sum(self.roofline_without_launch(
                stage_kernel, hardware).t_total_s for _, stage_kernel in stage_kernels), "roofline")
        if HWSpec.source_contract_supported("aggregate_gpu_source_roofline", hardware.hardware_id) and kernel.variant in (
                "dsv4_qnorm_rope_kv_insert", "dsv4_compressor_state_cache", "dsv4_compressed_state_cache"):
            work = target_cache_logical_work(kernel)
            if hardware.gpu_vector_memory_only:
                return prediction_from_ns(1e9 * (work["read_bytes"] + work["write_bytes"])
                    / hardware.resource_by_role("dram").capacity, "roofline")
            missing = tuple(role for role, amount in work["demand"].items()
                if amount > 0 and role not in hardware._role_map)
            if missing:
                raise NotImplementedError(f"BW1100 {kernel.variant}: source Roofline requires {missing}; native launch is not required")
            demand = {role: 0 for role in ("compute_tensor", "compute_vector", "compute_reduction",
                "compute_fma", "compute_sfu", "compute_alu")}
            demand.update(work["demand"])
            return prediction_from_ns(roofline_ns(demand, hardware,
                work["read_bytes"] + work["write_bytes"]), "roofline")
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in ("dense_fp16_source_flops_bytes_v1", "dense_bf16_source_flops_bytes_v1")):
            work = {"fp16": dense_fp16_source_logical_work, "bf16": dense_bf16_source_logical_work}[kernel.dtype](kernel)
            return prediction_from_ns(roofline_ns(work["demand"], hardware, work["logical_bytes"]), "roofline")
        return super().roofline_without_launch(kernel, hardware)

    def roofline(self, kernel: KernelSpec, hardware):
        stage_kernels = _dsv4_c128_stage_kernels(kernel)
        if stage_kernels is not None:
            return prediction_from_ns(1e9 * sum(self.roofline(
                stage_kernel, hardware).t_total_s for _, stage_kernel in stage_kernels), "roofline")
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory),
            "roofline",
        )
