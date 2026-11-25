"""Complete prepared K3 RecoverSSM verify/accepted-state commit host rules.

vLLM0.30 kimi_k3/nvidia/ops/recoverssm.py owns all geometry and flags.
Verification never stores the checkpoint: it emits output, FP32 corrections
and BF16 raw key/gate records. Commit folds accepted records into all layers
after preparing its device plan and compacting convolution history. Context
creation/pointer-table H2D setup is a separate once-per-context recipe.
Compiler registers/shared resources, deployment identity and prices remain
unbound. No ordinary per-token-state or foreign hardware facts are reused.
"""

import math
from dataclasses import replace

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _source(shape, dtype, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 RecoverSSM requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_recoverssm"
            or attributes["source_compiler"] != "triton_3_7_1_cuda_sm90"
            or attributes["source_recurrent_state_dtype"] != "fp32"
            or attributes["source_correction_dtype"] != "fp32"
            or attributes["source_kg_dtype"] != "bf16"
            or attributes["source_cache_inner_layout"] != "dense_heads_position_dimension"
            or attributes["source_metadata_dtype"] != "int32"):
        raise NotImplementedError("RecoverSSM requires exact0.30 BF16/FP32 prepared source")
    heads, dim, window, slots = require_shape(shape, ("heads", "head_dim", "spec_query_len", "state_pool_slots"))
    if heads not in (6, 12, 24, 48, 96) or dim != 128 or not 2 <= window < 2**31 or slots >= 2**31:
        raise NotImplementedError("RecoverSSM binds K3's128-wide heads and configured speculative state window")
    bound = attributes["gate_lower_bound"]
    if bound is not None and (type(bound) not in (int, float) or not math.isfinite(bound) or not -5 <= bound < 0):
        raise ValueError("RecoverSSM gate bound must be absent or finite in[-5,0)")
    return heads, dim, window, slots


def _metadata(shape, attributes, slots):
    rows, batch = shape["rows"], shape["sequences"]
    if any(type(value) is not int or not 0 <= value < 2**31 for value in (rows, batch)) or batch > 65535:
        raise ValueError("RecoverSSM rows/sequences must fit the source CUDA grid and INT32 query fields")
    starts, indices = attributes["query_start_loc"], attributes["state_indices"]
    if (not isinstance(starts, (tuple, list)) or len(starts) != batch + 1
            or any(type(value) is not int for value in starts)
            or starts[0] != 0 or starts[-1] != rows
            or any(not 0 <= b - a <= shape["spec_query_len"] for a, b in zip(starts, starts[1:]))):
        raise ValueError("RecoverSSM needs actual nondecreasing query segments within its speculative window")
    if (not isinstance(indices, (tuple, list)) or len(indices) != batch
            or any(type(index) is not int or not -2**31 <= index < slots for index in indices)):
        raise ValueError("RecoverSSM needs one actual INT32 checkpoint index per sequence")
    live = [index for index in indices if index > 0]
    if len(set(live)) != len(live):
        raise ValueError("RecoverSSM live sequence checkpoint slots must be distinct")
    strides = attributes["source_metadata_strides"]
    if (not isinstance(strides, (tuple, list)) or len(strides) != 2
            or any(type(value) is not int or value <= 0 for value in strides)
            or max(batch * strides[0], (batch + 1) * strides[1]) * 4 >= 2**63):
        raise ValueError("RecoverSSM needs actual valid state-index/query-start strides")
    return rows, batch, starts, indices


def _slot_strides(strides, heads, dim, window, slots):
    minimum = (heads * dim * dim, heads * window * dim, heads * window * 2 * dim)
    if (not isinstance(strides, (tuple, list)) or len(strides) != 3
            or any(type(value) is not int or value < lower for value, lower in zip(strides, minimum))
            or max(strides) * slots * 4 >= 2**63):
        raise ValueError("RecoverSSM state/correction/key-gate slots must have dense nonoverlapping inner dimensions")


def _context_layouts(attributes, heads, dim, window, slots, layers, history):
    layouts = attributes["source_layer_slot_layouts"]
    if not isinstance(layouts, (tuple, list)) or len(layouts) != layers:
        raise ValueError("RecoverSSM context needs actual per-layer cache slot layouts")
    for layout in layouts:
        _slot_strides(layout["state_correction_kg_strides"], heads, dim, window, slots)
        conv_stride, channel, token = layout["conv_strides"]
        length, channels = history + window - 1, 3 * heads * dim
        if (any(type(value) is not int or value <= 0 for value in (conv_stride, channel, token))
                or (channel, token) not in ((length, 1), (1, channels))
                or conv_stride < channels * length or slots * conv_stride * 2 >= 2**63):
            raise ValueError("RecoverSSM conv layout must preserve the actual SD/DS extended history")


def select_kda_recoverssm_context_create_recipe(shape, dtype, attributes, hardware):
    """Whole KDARecoverSSMCommitContext.create with prepared layer caches.

    Ten torch.tensor(list, int64, device) calls upload distinct host tables;
    their default non_blocking=False host completion must be preserved by
    a stream/host-aware consumer. Four empty INT32 plan buffers allocate
    storage without fills. A_log and dt_bias stacks produce fresh storage;
    their following contiguous() calls alias that already contiguous result.
    Convolution transpose/view and cache/parameter pointer lookups are host
    operations. This recipe belongs to context creation, never every commit.
    It does not claim host synchronization prices or compiler resources.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig
    from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_float_stack0_recipe

    heads, dim, window, slots = _source(shape, dtype, attributes, hardware)
    layers, history = require_shape(shape, ("layers", "conv_history_len"))
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_context_storage"] != "distinct_layer_pools_dense_inner"
            or attributes["source_context_create_inputs"] != "prepared_same_device_layer_caches_and_parameters"
            or attributes["source_context_host_tables"] != "pageable_int64_list_nonblocking_false"
            or attributes["source_host_device_link"] != "pcie"
            or attributes["source_parameter_layout"] != "contiguous_fp32_A_log_and_flat_dt_bias"
            or history != 3):
        raise NotImplementedError("RecoverSSM context creation needs exact prepared K3 caches/parameters and host upload source")
    capacity = attributes["source_plan_capacity"]
    if type(capacity) is not int or not 0 <= capacity < 2**31:
        raise ValueError("RecoverSSM context needs actual nonnegative INT32 plan capacity")
    _context_layouts(attributes, heads, dim, window, slots, layers, history)
    alignments = attributes["source_parameter_address_alignments"]
    if not isinstance(alignments, (tuple, list)) or len(alignments) != 2:
        raise ValueError("RecoverSSM context needs separate per-layer A_log/dt_bias address alignments")
    stages = [(role + "_h2d", PhysicalTransferConfig(
        algorithm="stream_copy", bytes=layers * 8, direction="h2d", link="pcie")) for role in (
            "conv_state_base_addrs", "conv_state_block_strides", "conv_state_dim_strides", "conv_state_token_strides",
            "state_base_addrs", "state_block_strides", "correction_cache_base_addrs", "correction_cache_block_strides",
            "kg_cache_base_addrs", "kg_cache_block_strides")]
    for index, (role, sizes) in enumerate((("A_log", (heads,)), ("dt_bias", (heads, dim)))):
        stack = select_torch213_float_stack0_recipe({"inputs": layers}, "fp32", {
            "tensor_iterator_source": "pytorch_2_13_sm90", "source_stack_layout": "contiguous_same_shape_dim0_fresh_output",
            "source_stack_input_shapes": (sizes,) * layers, "source_stack_input_dtypes": ("fp32",) * layers,
            "source_stack_input_address_alignments": alignments[index]}, hardware)
        stages.extend((role + "_" + name, launch) for name, launch in stack)
    return tuple(stages)


def select_kda_recoverssm_verify_launch(shape, dtype, attributes, hardware):
    """One complete verify kernel after convolution, preserving checkpoint.

    q/k/v can be strided split views of one convolved QKV allocation. Only
    token strides may have gaps; heads/features remain dense. Null state
    rows write output zeros inside this kernel and leave records untouched;
    they do not create separate fills or alter the launch grid.
    """
    heads, dim, window, slots = _source(shape, dtype, attributes, hardware)
    rows, batch, _, _ = _metadata(shape, attributes, slots)
    if attributes["source_verify_inputs"] != "prepared_qkv_raw_gate_beta_dense_inner":
        raise NotImplementedError("RecoverSSM verify requires actual prepared projection/convolution views")
    _slot_strides(attributes["source_cache_slot_strides"], heads, dim, window, slots)
    strides = attributes["source_activation_token_strides"]
    minimum = (heads * dim,) * 4 + (heads, heads * dim)
    if (not isinstance(strides, (tuple, list)) or len(strides) != 6
            or any(type(value) is not int or value < lower for value, lower in zip(strides, minimum))
            or max(strides) * rows * 2 >= 2**63):
        raise ValueError("RecoverSSM q/k/v/gate/beta/output require actual dense inner token strides")
    if rows == 0:
        return None
    return LaunchConfig("vllm_0_30_sm90_kda_recoverssm_verify_bv32",
        (ceil_div(dim, 32), batch, heads), (128, 1, 1), parameters={
            "source_BK": dim, "source_BV": 32, "source_spec_query_len": window,
            "source_num_stages": 2, "source_use_lower_bound": float(attributes["gate_lower_bound"] is not None)})


def _commit_plan(starts, indices, window, slots, attributes):
    """Evaluate source integer plan only to validate accesses/ownership.

    The actual plan still submits its own GPU kernel. Its results are not
    supplied back as a host-produced fake device tensor or timing lookup.
    Accepted lengths are clamped to0..min(query_len,configured window),
    unlike the previous-step acceptance used by ordinary speculative state.
    """
    batch = len(indices)
    accepted = attributes["num_accepted_tokens"]
    if (not isinstance(accepted, (tuple, list))
            or any(type(value) is not int or not -2**31 <= value < 2**31 for value in accepted)):
        raise ValueError("RecoverSSM commit requires actual signed INT32 accepted counts")
    mapping = attributes["request_indices"]
    if mapping is None:
        requests = tuple(range(batch))
    elif (isinstance(mapping, (tuple, list)) and len(mapping) >= batch
            and all(type(value) is int and 0 <= value < len(accepted) for value in mapping[:batch])):
        requests = tuple(mapping[:batch])
    else:
        raise ValueError("RecoverSSM commit needs valid optional request mapping")
    if any(request >= len(accepted) for request in requests):
        raise ValueError("RecoverSSM accepted-count tensor is shorter than its request indices")
    align = attributes["source_align_metadata"]
    strides = attributes["source_commit_metadata_strides"]
    if (not isinstance(strides, (tuple, list)) or len(strides) != 5
            or any(type(value) is not int or value < 0 for value in strides)
            or strides[0] <= 0 or (strides[1] <= 0 if mapping is not None else strides[1] != 0)):
        raise ValueError("RecoverSSM commit needs actual accepted/request/table/computed metadata strides")
    if align is not None:
        size, table, computed = align["block_size"], align["block_table"], align["num_computed_tokens"]
        if (type(size) is not int or not window <= size < 2**31
                or not isinstance(table, (tuple, list)) or not table
                or not isinstance(computed, (tuple, list))
                or any(type(value) is not int or not 0 <= value < 2**31 for value in computed)
                or any(not isinstance(row, (tuple, list)) or not row for row in table)):
            raise ValueError("RecoverSSM align requires complete block table/computed tokens/block size")
        width = len(table[0])
        if (any(len(row) != width or any(type(value) is not int or not -2**31 <= value < slots for value in row) for row in table)
                or any(request >= min(len(table), len(computed)) for request in requests)):
            raise ValueError("RecoverSSM align request/table slots must match actual pool extents")
        if min(strides[2:]) <= 0 or strides[2] < width * strides[3]:
            raise ValueError("RecoverSSM align metadata strides do not preserve the actual table rows")
    elif tuple(strides[2:]) != (0, 0, 0):
        raise ValueError("RecoverSSM absent align metadata must have source zero strides")
    extents = (len(accepted), len(mapping) if mapping is not None else 0,
               len(table) if align is not None else 0, width if align is not None else 0,
               len(computed) if align is not None else 0)
    if any(extent * stride * 4 >= 2**63 for extent, stride in zip(extents, strides)):
        raise ValueError("RecoverSSM commit metadata addressing exceeds signed INT64")
    plan = []
    for seq, (source, request) in enumerate(zip(indices, requests)):
        count = min(max(accepted[request], 0), starts[seq + 1] - starts[seq], window)
        final, boundary, recovery = source, 0, 0
        if align is not None:
            position = computed[request]
            if position + count >= 2**31:
                raise ValueError("RecoverSSM final computed-token count overflows source INT32")
            final = table[request][min((position + count) // size, width - 1)]
            next_boundary = (position // size + 1) * size
            if next_boundary >= 2**31:
                raise ValueError("RecoverSSM next boundary overflows source INT32")
            recovery = next_boundary - position
            if position + count >= next_boundary:
                column = next_boundary // size - 1
                if column >= width:
                    raise ValueError("RecoverSSM boundary table lookup exceeds actual width")
                boundary = table[request][column]
        plan.append((source, count, final, boundary, recovery) if source > 0 and count > 0 else (source, 0, 0, 0, 0))
    # Different requests must not overwrite each other's live source or
    # destination slots while independent GPU programs are still reading.
    owners = {}
    live_sources = {source: seq for seq, (source, *_rest) in enumerate(plan) if source > 0}
    for seq, (_source, count, final, boundary, recovery) in enumerate(plan):
        if not count or final <= 0:
            continue
        if boundary == final and boundary > 0 and recovery != count:
            raise ValueError("RecoverSSM final/boundary writes alias with different recovery lengths")
        for destination in (final, boundary):
            if destination <= 0:
                continue
            if ((destination in live_sources and live_sources[destination] != seq)
                    or (destination in owners and owners[destination] != seq)):
                raise ValueError("RecoverSSM cross-request state writes overlap another source/destination")
            owners[destination] = seq
    return tuple(plan)


def select_kda_recoverssm_commit_recipe(shape, dtype, attributes, hardware):
    """All three commit kernels for an already created current context.

    No per-layer Python loop: compact and state commit cover all actual
    layers on grid.z. Even zero accepts/null rows still submit these three
    kernels for a nonempty batch. H2D pointer tables and parameter stack
    are once-per-context producers, not silently recreated on each commit.
    """
    heads, dim, window, slots = _source(shape, dtype, attributes, hardware)
    _, batch, starts, indices = _metadata(shape, attributes, slots)
    if batch == 0:
        return ()
    layers, history = require_shape(shape, ("layers", "conv_history_len"))
    if (attributes["source_commit_context"] != "prepared_current_layer_pointer_tables"
            or attributes["source_context_storage"] != "distinct_layer_pools_dense_inner"
            or type(attributes["source_plan_capacity"]) is not int or attributes["source_plan_capacity"] < batch
            or history != 3 or layers * heads > 65535):
        raise NotImplementedError("RecoverSSM commit requires its prepared all-layer K3 context/capacity/four-tap history")
    _context_layouts(attributes, heads, dim, window, slots, layers, history)
    _commit_plan(starts, indices, window, slots, attributes)
    flags = {"source_spec_query_len": window,
             "source_align_mode": float(attributes["source_align_metadata"] is not None)}
    plan = LaunchConfig("vllm_0_30_sm90_kda_recoverssm_prepare_commit_plan",
        (batch, 1, 1), (32, 1, 1), parameters={**flags, "source_num_stages": 3,
            "source_has_request_indices": float(attributes["request_indices"] is not None)})
    conv = LaunchConfig("vllm_0_30_sm90_kda_recoverssm_compact_conv",
        (ceil_div(3 * heads * dim, 256), batch, layers), (128, 1, 1), parameters={
            **flags, "source_BLOCK_D": 256, "source_BLOCK_HISTORY": 4, "source_num_stages": 3})
    state = LaunchConfig("vllm_0_30_sm90_kda_recoverssm_commit_state_bv32",
        (ceil_div(dim, 32), batch, layers * heads), (128, 1, 1), parameters={
            **flags, "source_BK": dim, "source_BV": 32, "source_num_stages": 2,
            "source_use_lower_bound": float(attributes["gate_lower_bound"] is not None)})
    return (("prepare_commit_plan", plan), ("compact_conv_state", conv), ("commit_recurrent_state", state))


def select_kda_recoverssm_decode_recipe(shape, dtype, attributes, hardware):
    """Complete pure prepared RecoverSSM decode body before later commit."""
    return _select_kda_recoverssm_decode_program(shape, dtype, attributes, hardware, caller="pure")


def _select_kda_recoverssm_decode_program(shape, dtype, attributes, hardware, *, caller):
    """Pure call includes norm; mixed callers receive only conv and verify."""
    from src.kernel.launch_config.nvidia_h200.kda_launch import select_kda_decode_launch

    verify = select_kda_recoverssm_verify_launch(shape, dtype, attributes, hardware)
    program = {"pure": "pure_spec_recoverssm_direct_output",
        "mixed_decode": "mixed_spec_recoverssm_ordinary_decode",
        "mixed_prefill": "mixed_spec_recoverssm_prefill"}[caller]
    if (attributes["source_batch_program"] != program
            or attributes["source_conv_weight_layout"] != "channels_taps_fp32"
            or attributes["source_output_norm_implementation"] != "forward_cuda"
            or attributes["source_output_norm_alias"] != "same_view"
            or attributes["source_graph_padding"] != "none"
            or type(attributes["source_norm_eps"]) not in (int, float) or attributes["source_norm_eps"] != 1e-5):
        raise NotImplementedError("RecoverSSM full decode requires its pure direct-output/CUDA-norm source")
    if verify is None:
        raise NotImplementedError("empty outer KDA decode needs its separate caller handling")
    heads, dim, window = shape["heads"], shape["head_dim"], shape["spec_query_len"]
    gate_stride = attributes["source_output_gate_token_stride"]
    if (attributes["source_activation_token_strides"][5] != heads * dim
            or attributes["source_output_gate_layout"] != "row_strided_bf16_dense_heads"
            or type(gate_stride) is not int or gate_stride < heads * dim
            or gate_stride * shape["rows"] * 2 >= 2**63):
        raise NotImplementedError("RecoverSSM in-place norm needs contiguous output and actual row-strided dense-head gate")
    if (type(attributes["source_configured_num_spec"]) is not int
            or attributes["source_configured_num_spec"] != window - 1):
        raise ValueError("RecoverSSM convolution must retain configured speculative history")
    accepts = attributes["previous_num_accepted_tokens"]
    if (not isinstance(accepts, (tuple, list)) or len(accepts) != shape["sequences"]
            or any(type(value) is not int or not 1 <= value <= window for value in accepts)):
        raise ValueError("RecoverSSM convolution needs previous accepted counts within its history window")
    length, channel, token, slot = attributes["source_conv_state_strides"]
    channels = 3 * heads * dim
    if (any(type(value) is not int or value <= 0 for value in (length, channel, token, slot))
            or length != window + 2 or (channel, token) not in ((length, 1), (1, channels))
            or slot < length * channels or slot * shape["state_pool_slots"] * 2 >= 2**63):
        raise ValueError("RecoverSSM convolution needs actual extended SD/DS history strides")
    # No detached tensor is materialized for the first state-ID column.
    history = window + 2
    conv = LaunchConfig("vllm_0_30_sm90_kda_recoverssm_causal_conv1d_update",
        (shape["sequences"], ceil_div(channels, 256), 1), (128, 1, 1), parameters={
            "source_np2_state_length": 1 << (history - 1).bit_length(),
            "source_programmatic_stream_serialization": 1})
    if caller != "pure":
        return (("spec_convolution", conv), ("recoverssm_verify", verify))
    norm = select_kda_decode_launch(shape, dtype, "kimi_k3_kda_decode",
        {"source_component": "vllm_0_30_kimi_k3_triton_decode", "stage": "output_norm"}, hardware)
    return (("spec_convolution", conv), ("recoverssm_verify", verify),
            ("output_norm", replace(norm, execution_variant=None)))
