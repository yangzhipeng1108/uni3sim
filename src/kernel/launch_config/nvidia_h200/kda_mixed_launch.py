"""Complete prepared K3 speculative plus non-speculative source bodies.

vLLM0.30 kda.py::_forward owns the stable subgroup partition, six
IndexSelect producers, two recurrent branches, two IndexCopy restores and
one outer gated norm. Contiguous subgroups use views/direct output instead.
Prepared metadata is a caller producer; no GPU sort/H2D setup is invented.
RecoverSSM callers bind their distinct verify/record metadata explicitly;
default native norm remains a separate source program. Compiler resources,
typed prices and deployment identity are unbound.
"""

from dataclasses import replace
from collections.abc import Mapping

from src.kernel.launch_config.nvidia_h200.kda_launch import select_kda_decode_launch
from src.kernel.launch_config.nvidia_h200.kda_spec_launch import _select_kda_spec_program
from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_indexed_rows_launch
from src.kernel.launch_config.selection_primitives import require_shape


def select_kda_mixed_decode_recipe(shape, dtype, attributes, hardware):
    """Return every device submission after projection and partition setup.

    The source has spec requests, so its native ordinary-decode early
    return is unreachable even if that operator is installed. The ordinary
    branch uses packed Triton convolution/recurrence with the actual extended
    speculative cache strides, not ordinary-only three-slot layout facts.
    Neither subgroup normalizes or copies into final output independently.
    """
    return _select_kda_mixed_program(shape, dtype, attributes, hardware, non_spec_prefill=None, recoverssm_spec=None)


def select_kda_mixed_prefill_recipe(shape, dtype, attributes, hardware, non_spec_attributes):
    """Complete mixed spec plus prepared Triton/FlashKDA prefill subgroup.

    The non-spec subgroup can itself contain ordinary decode requests;its
    actual query segments,phase counts,initial-state flags,configs/workspace
    remain caller facts. It writes either a final-output slice or a distinct
    group buffer restored by IndexCopy. No per-subgroup copy/norm is added.
    """
    if not isinstance(non_spec_attributes, Mapping):
        raise TypeError("mixed prefill requires complete actual non-spec subgroup attributes")
    return _select_kda_mixed_program(shape, dtype, attributes, hardware, non_spec_prefill=non_spec_attributes, recoverssm_spec=None)


def select_kda_recoverssm_mixed_decode_recipe(shape, dtype, attributes, hardware, spec_attributes):
    """Complete prepared RecoverSSM verify plus ordinary decode caller.

    Verification reads one checkpoint per sequence and writes records, not
    one state per query token. Ordinary rows still use packed recurrence.
    Partition producers, indexed restores and one outer norm are retained.
    Context creation and later accepted-state commit remain separate calls.
    """
    if not isinstance(spec_attributes, Mapping):
        raise TypeError("mixed RecoverSSM requires complete actual verify subgroup attributes")
    return _select_kda_mixed_program(shape, dtype, attributes, hardware,
        non_spec_prefill=None, recoverssm_spec=spec_attributes)


def select_kda_recoverssm_mixed_prefill_recipe(shape, dtype, attributes, hardware, spec_attributes, non_spec_attributes):
    """Complete RecoverSSM plus prepared Triton/FlashKDA prefill subgroup."""
    if not isinstance(spec_attributes, Mapping) or not isinstance(non_spec_attributes, Mapping):
        raise TypeError("mixed RecoverSSM prefill requires both complete subgroup attribute mappings")
    return _select_kda_mixed_program(shape, dtype, attributes, hardware,
        non_spec_prefill=non_spec_attributes, recoverssm_spec=spec_attributes)


def _recoverssm_spec_core(shape, dtype, attributes, hardware, spec_attributes, *, prefill):
    from src.kernel.launch_config.nvidia_h200.kda_recoverssm_launch import _select_kda_recoverssm_decode_program

    phase = "prefill" if prefill else "decode"
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_recoverssm_mixed_" + phase
            or attributes["source_batch_program"] != ("mixed_recoverssm_prefill" if prefill else "mixed_recoverssm_ordinary_triton")
            or attributes["source_compiler"] != "triton_3_7_1_cuda_sm90"
            or attributes["source_recoverssm"] is not True
            or attributes["source_checkpoint_state"] != "disabled"
            or attributes["source_graph_padding"] != "none"
            or attributes["source_output_norm_implementation"] != "forward_cuda"
            or attributes["source_output_norm_alias"] != "same_view"
            or attributes["source_recurrent_state_dtype"] != "fp32"
            or attributes["source_conv_state_dtype"] != "bf16"
            or attributes["source_state_inner_layout"] != "heads_value_key_contiguous"
            or attributes["source_state_indices_dtype"] != "int32"
            or attributes["source_accepted_tokens_dtype"] != "int32"
            or attributes["source_query_start_dtype"] != "int32"
            or attributes["source_input_layout"] != "packed_projection_and_dense_conv_output"
            or attributes["source_raw_gate_layout"] != "contiguous_bf16"
            or attributes["source_conv_weight_layout"] != "channels_taps_fp32"):
        raise NotImplementedError("mixed RecoverSSM requires its exact0.30 cache/metadata/CUDA-norm source")
    heads, dim, window = shape["heads"], shape["head_dim"], shape["max_query_len"]
    if (shape["conv_kernel"] != 4 or type(attributes["source_configured_num_spec"]) is not int
            or attributes["source_configured_num_spec"] != window - 1):
        raise ValueError("mixed RecoverSSM must preserve configured four-tap speculative history")
    cache_fields = ("source_conv_state_length", "source_conv_channel_stride_elements",
        "source_conv_tap_stride_elements", "source_conv_slot_stride_elements", "source_state_slot_stride_elements")
    if any(type(attributes[key]) is not int or attributes[key] <= 0 for key in cache_fields):
        raise ValueError("mixed RecoverSSM needs actual positive integer cache strides")
    layout = attributes["source_conv_state_layout"]
    if layout not in ("SD", "DS"):
        raise NotImplementedError("mixed RecoverSSM needs actual SD/DS convolution layout")
    if (attributes["source_conv_channel_stride_elements"], attributes["source_conv_tap_stride_elements"]) != (
            (1, 3 * heads * dim) if layout == "SD" else (attributes["source_conv_state_length"], 1)):
        raise ValueError("mixed RecoverSSM convolution strides disagree with the declared SD/DS layout")
    starts, ids = attributes["query_start_loc"], attributes["state_indices"]
    stride = attributes["source_state_indices_row_stride_elements"]
    if (not isinstance(starts, (tuple, list)) or len(starts) < 2
            or not isinstance(ids, (tuple, list)) or len(ids) != len(starts) - 1
            or any(not isinstance(row, (tuple, list)) or not row
                   or len(row) != len(ids[0]) or any(type(value) is not int or not -2**31 <= value < 2**31 for value in row) for row in ids)
            or type(stride) is not int or stride < len(ids[0])
            or attributes["source_state_indices_column_stride_elements"] != 1
            or attributes["source_accepted_tokens_stride_elements"] != 1
            or attributes["source_query_start_stride_elements"] != 1):
        raise ValueError("mixed RecoverSSM needs actual first-column checkpoint views and dense query/acceptance vectors")
    mode = attributes["source_token_partition"]
    if mode not in ("indexed", "contiguous_slices"):
        raise NotImplementedError("mixed RecoverSSM partition is not source-bound")
    beta_stride = heads if mode == "indexed" else attributes["source_input_token_strides"][2]
    expected_conv = (attributes["source_conv_state_length"], attributes["source_conv_channel_stride_elements"],
        attributes["source_conv_tap_stride_elements"], attributes["source_conv_slot_stride_elements"])
    if (spec_attributes["query_start_loc"] != starts
            or tuple(spec_attributes["state_indices"]) != tuple(row[0] for row in ids)
            or spec_attributes["previous_num_accepted_tokens"] != attributes["num_accepted_tokens"]
            or tuple(spec_attributes["source_metadata_strides"]) != (stride, 1)
            or tuple(spec_attributes["source_conv_state_strides"]) != expected_conv
            or spec_attributes["source_cache_slot_strides"][0] != attributes["source_state_slot_stride_elements"]
            or spec_attributes["source_configured_num_spec"] != attributes["source_configured_num_spec"]
            or spec_attributes["gate_lower_bound"] != attributes["gate_lower_bound"]
            or tuple(spec_attributes["source_activation_token_strides"]) != ((3 * heads * dim,) * 3 + (heads * dim, beta_stride, heads * dim))
            or spec_attributes["source_mixed_output_placement"] != (
                "core_output_slice" if mode == "contiguous_slices" else "distinct_group_buffer")):
        raise ValueError("RecoverSSM verify subgroup must bind actual shared pools, metadata, packed/sliced strides and output placement")
    spec_shape = {**shape, "rows": shape["spec_rows"], "sequences": len(starts) - 1,
        "spec_query_len": window, "state_pool_slots": attributes["state_pool_slots"]}
    return _select_kda_recoverssm_decode_program(spec_shape, dtype, spec_attributes, hardware, caller="mixed_" + phase)


def _select_kda_mixed_program(shape, dtype, attributes, hardware, *, non_spec_prefill, recoverssm_spec):
    prefill = non_spec_prefill is not None
    rows, spec_rows, heads, dim, width, max_query = require_shape(shape,
        ("rows", "spec_rows", "heads", "head_dim", "conv_kernel", "max_query_len"))
    ordinary, = require_shape(shape, ("non_spec_rows" if prefill else "ordinary_rows",))
    group_name = "non_spec" if prefill else "ordinary"
    spec_shape = {**shape, "rows": spec_rows}
    if recoverssm_spec is None:
        spec_core = _select_kda_spec_program(spec_shape, dtype, attributes, hardware,
            caller="mixed_prefill" if prefill else "mixed_decode")
    else:
        spec_core = _recoverssm_spec_core(shape, dtype, attributes, hardware, recoverssm_spec, prefill=prefill)
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_mixed_program"] != ("spec_and_prepared_prefill" if prefill else "spec_and_plain_decode_no_prefill")
            or attributes["source_output_gate_layout"] != "row_strided_bf16_dense_heads"
            or attributes["source_final_output_layout"] != "fresh_contiguous_bf16"
            or attributes["source_norm_eps"] != 1e-5):
        raise NotImplementedError("K3 mixed decode requires its exact prepared0.30 no-prefill/Torch2.13/direct-output source")
    if rows != spec_rows + ordinary or rows >= 2**31 or (not prefill and ordinary * heads > 65535):
        raise ValueError("K3 mixed physical rows must conserve both subgroup extents and CUDA axis capacity")
    indices = (attributes["spec_token_indices"], attributes[group_name + "_token_indices"])
    for index, count in zip(indices, (spec_rows, ordinary)):
        if (not isinstance(index, (tuple, list)) or len(index) != count
                or any(type(value) is not int or not 0 <= value < rows for value in index)
                or any(a >= b for a, b in zip(index, index[1:]))):
            raise ValueError("K3 mixed token indices must be actual stable ascending subgroup rows")
    if set(indices[0]) & set(indices[1]) or len(set((*indices[0], *indices[1]))) != rows:
        raise ValueError("K3 mixed token partition must cover every physical row exactly once")
    mode = attributes["source_token_partition"]
    starts = attributes["spec_token_start"], attributes[group_name + "_token_start"]
    if mode == "contiguous_slices":
        for start, count, index in zip(starts, (spec_rows, ordinary), indices):
            if type(start) is not int or tuple(index) != tuple(range(start, start + count)):
                raise ValueError("K3 contiguous subgroup starts must match their actual source row indices")
    elif mode == "indexed":
        if starts != (None, None):
            raise ValueError("K3 indexed partition requires actual absent contiguous subgroup starts")
    else:
        raise NotImplementedError("K3 mixed subgroup partition is not source-bound")
    ordinary_indices = non_spec_prefill["state_indices"] if prefill else attributes["ordinary_state_indices"]
    slots = attributes["state_pool_slots"]
    if prefill:
        ns_starts = non_spec_prefill["query_start_loc"]
        if (not isinstance(ns_starts, (tuple, list)) or len(ns_starts) < 2
                or any(type(value) is not int for value in ns_starts)
                or ns_starts[0] != 0 or ns_starts[-1] != ordinary
                or any(a >= b for a, b in zip(ns_starts, ns_starts[1:]))):
            raise ValueError("K3 mixed prefill requires every actual non-spec query segment")
        sequences = len(ns_starts) - 1
        phases = attributes["non_spec_phase_counts"]
        if (not isinstance(phases, (tuple, list)) or len(phases) != 2
                or any(type(value) is not int or value < 0 for value in phases)
                or phases[0] == 0 or sum(phases) != sequences):
            raise ValueError("K3 mixed prefill requires actual nonzero prefill/decode phase counts")
        if (non_spec_prefill["state_pool_slots"] != slots
                or non_spec_prefill["source_state_pool_row_stride_elements"] != attributes["source_state_slot_stride_elements"]
                or non_spec_prefill["gate_lower_bound"] != attributes["gate_lower_bound"]
                or non_spec_prefill["source_speculation_state"] != "mixed_prefill_branch"
                or non_spec_prefill["source_mixed_output_placement"] != (
                    "core_output_slice" if mode == "contiguous_slices" else "distinct_group_buffer")):
            raise ValueError("K3 mixed prefill subgroup must bind the same state pool/gate and actual output placement")
    else:
        sequences = ordinary
    if (not isinstance(ordinary_indices, (tuple, list)) or len(ordinary_indices) != sequences
            or any(type(index) is not int or not 0 < index < min(slots, 2**31) for index in ordinary_indices)
            or len(set(ordinary_indices)) != sequences):
        raise ValueError("K3 mixed ordinary rows require distinct live checkpoint slots")
    if recoverssm_spec is None:
        occupied = set()
        for seq, (begin, end) in enumerate(zip(attributes["query_start_loc"], attributes["query_start_loc"][1:])):
            state_ids = attributes["state_indices"][seq]
            occupied.update(state_ids[:end - begin])
            occupied.add(state_ids[attributes["num_accepted_tokens"][seq] - 1])
    else:
        # RecoverSSM reads the first checkpoint column only; query records
        # do not become per-token state-pool writes or accepted-column reads.
        occupied = {index for index in recoverssm_spec["state_indices"] if index > 0}
    if occupied & set(ordinary_indices):
        raise ValueError("K3 mixed subgroups must not update/read overlapping state slots")
    if not prefill and (type(attributes["ordinary_state_index_stride"]) is not int or attributes["ordinary_state_index_stride"] <= 0):
        raise ValueError("K3 ordinary state indices require their actual positive row stride")
    strides = attributes["source_input_token_strides"]
    columns = (3 * heads * dim, heads * dim, heads)
    if (not isinstance(strides, (tuple, list)) or len(strides) != 3
            or any(type(value) is not int or value < lower for value, lower in zip(strides, columns))
            or strides[1] != heads * dim or max(strides) * rows * 2 >= 2**63):
        raise ValueError("K3 mixed inputs must retain packed QKV/beta and dense raw gate token strides")
    gate_stride = attributes["source_output_gate_token_stride"]
    if type(gate_stride) is not int or gate_stride < heads * dim or 2 * rows * gate_stride >= 2**63:
        raise ValueError("K3 mixed outer norm needs the actual row-strided output gate")
    stages = []
    if mode == "indexed":
        alignments = attributes["source_input_address_alignments"]
        if (not isinstance(alignments, (tuple, list)) or len(alignments) != 3
                or any(type(value) is not int or value not in (1, 2, 4, 8, 16) for value in alignments)):
            raise ValueError("K3 mixed packing requires actual source address alignments capped at16 bytes")
        for group, count, index in zip(("spec", group_name), (spec_rows, ordinary), indices):
            for name, column, stride, alignment in zip(("qkv", "raw_gate", "raw_beta"), columns, strides, alignments):
                launch = select_torch213_indexed_rows_launch({"rows": count, "source_rows": rows, "columns": column}, dtype,
                    {"tensor_iterator_source": attributes["tensor_iterator_source"], "source_row_layout": "dense_columns_strided_rows",
                     "source_index_dtype": "int64", "source_row_indices": index, "source_row_stride_elements": stride,
                     "source_address_alignment": alignment, "source_operation": "index_select_rows"}, hardware)
                stages.append((f"{group}_{name}_index_select", launch))
    stages.extend(spec_core)
    ordinary_shape = {**shape, "rows": ordinary}
    if prefill:
        from src.kernel.launch_config.nvidia_h200.kda_launch import _select_kda_prepared_prefill_program

        if non_spec_prefill["source_prefill_backend"] == "flashkda":
            beta_stride = heads if mode == "indexed" else strides[2]
            if non_spec_prefill["source_beta_row_stride_elements"] != beta_stride:
                raise ValueError("mixed FlashKDA beta stride must reflect actual gathered or sliced input")
        core = _select_kda_prepared_prefill_program(ordinary_shape, dtype, non_spec_prefill, hardware, mixed=True)
        stages.extend(("non_spec_" + role, launch) for role, launch in core)
    else:
        for stage in ("convolution", "recurrent"):
            launch = select_kda_decode_launch(ordinary_shape, dtype, "kimi_k3_kda_decode",
                {"source_component": "vllm_0_30_kimi_k3_triton_decode", "stage": stage}, hardware)
            stages.append(("ordinary_" + stage, replace(launch, execution_variant=None)))
    if mode == "indexed":
        for group, count, index in zip(("spec", group_name), (spec_rows, ordinary), indices):
            launch = select_torch213_indexed_rows_launch({"rows": count, "source_rows": rows, "columns": heads * dim}, dtype,
                {"tensor_iterator_source": attributes["tensor_iterator_source"], "source_row_layout": "dense_columns_strided_rows",
                 "source_index_dtype": "int64", "source_row_indices": index, "source_row_stride_elements": heads * dim,
                 "source_operation": "index_copy_rows"}, hardware)
            stages.append((group + "_output_index_copy", launch))
    norm = select_kda_decode_launch(shape, dtype, "kimi_k3_kda_decode",
        {"source_component": "vllm_0_30_kimi_k3_triton_decode", "stage": "output_norm"}, hardware)
    return (*stages, ("output_norm", replace(norm, execution_variant=None)))
