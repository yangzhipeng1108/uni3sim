"""MoE operand projection from caller-owned trace ports.

Role-local views rename actual storage; this module never allocates metadata,
expert temporaries, or a second serving arena. Opaque full-expert boundaries
cannot authorize the intermediate ports of a decomposed native recipe.
"""
from dataclasses import replace

from src.kernel.kernel_spec import KernelKind, linear_expert_partition
from src.kernel.memory_access_builder import ElementwiseBindings
from src.kernel.memory_fused_moe import FusedMoEBindings
from src.serving.workload.workload_graph import WorkloadKind


def _binding(call, source, reads, writes, producers):
    from .memory_bindings import source_memory_events
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(reads, producers, strict=True)))
    views = {}
    for view in (*reads, *writes):
        if view.view_id in views and views[view.view_id] != view:
            raise ValueError("MoE in-place role changed its actual storage view")
        views[view.view_id] = view
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id,
        allocations, tuple(views.values()), events,
        tuple(view.view_id for view in reads), tuple(view.view_id for view in writes),
        ready, call.rank, "source-forward", visibility), None


def _roles(ports, names):
    if len(ports) != len(names):
        raise ValueError("MoE source port arity differs from its declared ABI")
    return tuple(replace(view, view_id=name) for view, name in zip(ports, names, strict=True))


def _layout(view, shape, dtype):
    if (view.shape, view.dtype) != (shape, dtype):
        raise ValueError("MoE source operand geometry or dtype differs from its kernel")
    stride = view.element_bytes
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    if view.strides_bytes != tuple(reversed(strides)):
        raise ValueError("MoE source operand requires its contiguous public layout")


def _topk(call, source):
    spec = call.kernel_spec
    if source.kind is not WorkloadKind.VECTOR or source.variant != "moe_topk":
        return None, "moe_topk_source_operation_mapping_unavailable"
    # Masked-owner routing is a separate input/output operation. It must not
    # be mistaken for the producer of top-k weights and native source rows.
    if "stage" in spec.attributes and spec.attributes["stage"] != "route":
        return None, "moe_owner_mask_operand_mapping_unavailable"
    mathematical = spec.launch is None
    if mathematical:
        if ("source_component" not in source.attributes
                or source.attributes["source_component"] != "symbolic_softmax_topk"):
            return None, "moe_topk_mathematical_source_operation_unavailable"
        for name in ("source_program", "normalize_topk_probability", "mathematical_program"):
            if name not in spec.attributes or name not in source.attributes:
                raise ValueError("MoE mathematical routing omitted its source contract")
            if spec.attributes[name] != source.attributes[name]:
                raise ValueError("MoE mathematical routing changed its actual source version or semantics")
        source_stage = source.attributes["stage"] if "stage" in source.attributes else None
        kernel_stage = spec.attributes["stage"] if "stage" in spec.attributes else None
        if source_stage != kernel_stage:
            raise ValueError("MoE mathematical routing changed its actual source stage")
        if "source_program" not in spec.attributes or spec.attributes["source_program"] not in (
                "vllm_0_11_fp32_softmax_topk", "vllm_0_18_bf16_softmax_topk",
                "vllm_0_24_bf16_softmax_topk", "vllm_0_30_bf16_softmax_topk"):
            return None, "moe_topk_mathematical_source_program_unavailable"
        ascend = spec.attributes["source_program"] == "vllm_0_18_bf16_softmax_topk"
        normalized011 = (spec.attributes["source_program"] == "vllm_0_11_fp32_softmax_topk"
            and spec.attributes["normalize_topk_probability"])
        expected_outputs = 2 if ascend else 5 if normalized011 else 3
        if len(source.outputs) != expected_outputs:
            raise ValueError("MoE mathematical source output arity differs from its version")
    else:
        ascend = False
        if len(source.outputs) not in (3, 5):
            return None, "moe_topk_native_third_output_unavailable"
    if len(source.inputs) != 1:
        raise ValueError("MoE top-k source requires exactly its gating logits")
    rows, experts, topk = (spec.shape[name] for name in ("tokens", "experts", "topk"))
    logits, = _roles(source.inputs, ("logits",))
    # Native top-k writes raw weights; the mathematical wrapper also
    # includes Python normalization and its distinct final result.
    weight = source.outputs[3] if len(source.outputs) == 5 and not mathematical else source.outputs[0]
    writes = _roles((weight, source.outputs[1]) if ascend else (weight, source.outputs[1], source.outputs[2]),
        ("weights", "expert_ids") if ascend else ("weights", "expert_ids", "source_rows"))
    _layout(logits, (rows, experts), spec.dtype if mathematical else logits.dtype)
    output_dtypes = ("bf16", "int32") if ascend else ("fp32", "int32", "int32")
    for view, dtype in zip(writes, output_dtypes, strict=True):
        _layout(view, (rows, topk), dtype)
    if mathematical:
        if len(source.outputs) == 5:
            extra = _roles(source.outputs[3:], ("raw_weights", "selected_weight_sum"))
            _layout(extra[0], (rows, topk), "fp32")
            _layout(extra[1], (rows, 1), "fp32")
            writes += extra
        return _binding(call, source, (logits,), writes, source.input_producers)
    from src.kernel.memory_moe_routing import has_gpu_topk_source_abi
    if not has_gpu_topk_source_abi(spec):
        return None, "moe_topk_native_source_abi_unavailable"
    return _binding(call, source, (logits,), writes, source.input_producers)


def _native_grouped(call, source):
    spec = call.kernel_spec
    if source.kind is not WorkloadKind.FUSED_MOE or source.variant != spec.variant:
        return None, "moe_grouped_intermediate_and_alignment_ports_unavailable"
    # Explicit primitive source ports: query, weight, sorted tokens, expert
    # blocks, padded count, then W2 router weights when its ABI consumes them.
    from src.kernel.memory_fused_moe import native_grouped_moe_interface
    from src.kernel.memory_fused_moe_naive import has_h200_bf16_controls, h200_bf16_naive
    h200 = has_h200_bf16_controls(spec)
    if not native_grouped_moe_interface(spec) and not h200:
        return None, "moe_grouped_native_source_abi_unavailable"
    names = (("query", "weight", "expert_ids", "num_tokens_post_padded")
        if h200 and h200_bf16_naive(spec) else
        ("query", "weight", "sorted_token_ids", "expert_ids", "num_tokens_post_padded"))
    if spec.variant == "w2":
        names += ("router_weight",)
    reads = _roles(source.inputs, names)
    writes = _roles(source.outputs, ("output",))
    from .memory_bindings import source_memory_events
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(reads, source.input_producers, strict=True)))
    return FusedMoEBindings(call.call_id + ":source-memory", call.call_id, allocations,
        (*reads, *writes), events, "query", "weight", "output", reads[2].allocation_id,
        ready[0], ready[1], ready[2], call.rank, "source-forward", visibility), None


def _validate_public_expert_ports(call, source):
    """Check checkpoint parameters without claiming a backend-formatted ABI."""
    if ("source_component" not in source.attributes
            or source.attributes["source_component"] != "symbolic_unquantized_experts"):
        return
    names = ("hidden_states", "routing_weights", "expert_ids", "w13_weight", "w2_weight")
    if (source.attributes["memory_input_roles"] != names
            or source.attributes["memory_output_roles"] != ("output",)
            or source.attributes["expert_parameter_layout"] != "unquantized_checkpoint_expert_major_nk"):
        raise ValueError("MoE expert source parameter roles or layout differ")
    if (len(source.inputs), len(source.outputs)) != (5, 1):
        raise ValueError("MoE expert source parameter port arity differs")
    spec = call.kernel_spec
    tokens, experts, topk, hidden, intermediate = (spec.shape[name]
        for name in ("tokens", "experts", "topk", "hidden", "intermediate"))
    local_experts = len(linear_expert_partition(experts, source.attributes["expert_parallel_size"],
        source.attributes["expert_rank"] if source.attributes["expert_parallel_size"] > 1 else 0))
    w13, w2 = source.inputs[3:]
    _layout(w13, (local_experts, 2 * intermediate, hidden), "bf16")
    _layout(w2, (local_experts, hidden, intermediate), "bf16")
    if (source.input_producers[3:] != (None, None)
            or len({view.allocation_id for view in source.inputs}) != len(source.inputs)):
        raise ValueError("MoE expert parameters must remain distinct readonly source parents")
    if (source.inputs[0].shape != (tokens, hidden) or source.inputs[1].shape != (tokens, topk)
            or source.inputs[2].shape != (tokens, topk) or source.outputs[0].shape != (tokens, hidden)):
        raise ValueError("MoE expert source public geometry differs from the current call")


def bind_moe_memory(call, source):
    """Bind only source-declared public MoE operands; explain absent mappings."""
    if source.rank != call.rank or source.pool_name != call.pool_name or call.source_node_ids != (source.node_id,):
        raise ValueError("MoE binding crossed its actual source call")
    spec = call.kernel_spec
    if "symbolic_zero_extent_parameter_roles" in source.attributes:
        if (spec.kind is not KernelKind.VECTOR or spec.variant != "elementwise" or spec.attributes["operation"] != "fill"
                or spec.attributes["reads"] != 0 or spec.attributes["writes"] != 1
                or source.attributes["owned_experts"] or len(source.inputs) != 3 or len(source.outputs) != 1):
            raise ValueError("empty expert output must be one actual zero write with no parameters")
        return _binding(call, source, (), _roles(source.outputs, ("output",)), ())
    if spec.kind is KernelKind.FUSED_MOE:
        _validate_public_expert_ports(call, source)
        if spec.variant in ("w1", "w2"):
            return _native_grouped(call, source)
        return None, "moe_expert_callee_storage_ports_unavailable"
    if spec.kind is not KernelKind.VECTOR:
        return None, "moe_source_family_operand_mapping_unavailable"
    if spec.variant == "moe_topk":
        return _topk(call, source)
    if spec.variant in ("moe_routing_weight_norm", "moe_routing_weight_scale"):
        # A full router boundary records these as its outputs. It does not
        # supply the internal kernel producer ordering; do not invent it.
        if source.kind is not WorkloadKind.VECTOR or source.variant != spec.variant:
            return None, "moe_routing_normalization_producer_ports_unavailable"
        names = ("weights",) if spec.variant == "moe_routing_weight_norm" else ("weights", "row_sum")
        reads, writes = _roles(source.inputs, names), _roles(source.outputs, ("output",))
        return _binding(call, source, reads, writes, source.input_producers)
    primitive_roles = {
        "moe_router_logit_cast": (("input",), ("output",)),
        "moe_routed_id_widen": (("input",), ("output",)),
        "moe_routed_id_extent": (("input",), ("output",)),
        "moe_expert_map_lookup": (("expert_ids", "expert_map"), ("output",)),
        "moe_count_and_sort": (("topk_ids", "cumsum"), ("cumsum", "sorted_token_ids")),
        "moe_silu": (("gate",), ("output",)),
        "moe_mul": (("silu", "up"), ("output",)),
        "moe_sum": (("routed",), ("output",)),
        "moe_reduce": (("routed",), ("output",)),
    }
    if spec.variant in primitive_roles:
        if source.kind is not WorkloadKind.VECTOR or source.variant != spec.variant:
            return None, "moe_recipe_intermediate_ports_unavailable"
        reads, writes = primitive_roles[spec.variant]
        return _binding(call, source, _roles(source.inputs, reads), _roles(source.outputs, writes),
            source.input_producers)
    return None, "moe_recipe_intermediate_ports_unavailable"
