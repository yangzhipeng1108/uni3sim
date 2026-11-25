"""Shared lowering of one logical FusedMoE invocation into physical kernels.

Serving and Kernel Dataset Forward evaluate the same public logical FusedMoE
call.  This module is deliberately below both owners: it consumes already
validated routing/selection facts and creates only production ``KernelSpec``
objects.  It must never inspect profiler names, counters, or durations.
"""

from collections.abc import Mapping

from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    MOE_FILL_ROLES,
)
from src.kernel.launch_config.select_launch_config import (
    select_fused_moe_cast_roles,
    select_fused_moe_reduce_work_fractions,
    select_fused_moe_recipe,
    select_launch_config,
)


def build_fused_moe_physical_specs(
    *,
    hardware_id: str,
    shape: Mapping[str, int],
    selection_shape: Mapping[str, int] | None = None,
    dtype: str,
    prediction_attributes: Mapping[str, object],
    selection_attributes: Mapping[str, object],
    hardware,
) -> tuple[KernelSpec, ...]:
    """Lower one source-defined logical FusedMoE call into KernelSpecs.

    ``prediction_attributes`` carries its routing topology, while
    ``selection_attributes`` carries only source-supported launch inputs such
    as the explicit launch policy.  Keeping both inputs separate prevents an
    evaluation observation from becoming a production launch decision.
    """

    if not isinstance(shape, Mapping):
        raise TypeError("FusedMoE physical-spec shape must be a mapping")
    if selection_shape is None:
        selection_shape = shape
    if not isinstance(selection_shape, Mapping):
        raise TypeError(
            "FusedMoE physical-spec selection_shape must be a mapping")
    if not isinstance(prediction_attributes, Mapping):
        raise TypeError(
            "FusedMoE physical-spec prediction attributes must be a mapping")
    if not isinstance(selection_attributes, Mapping):
        raise TypeError(
            "FusedMoE physical-spec selection attributes must be a mapping")

    required_prediction_attributes = {
        "tokens_per_expert",
        "local_tokens_per_expert",
        "expert_parallel_size",
        "expert_rank",
        "owned_experts",
        "expert_weight_tile_rows",
    }
    missing = required_prediction_attributes - set(prediction_attributes)
    if missing:
        raise ValueError(
            "FusedMoE physical-spec prediction attributes are incomplete: "
            f"missing={sorted(missing)}")

    ascend018 = ("source_component" in selection_attributes
        and selection_attributes["source_component"] == "symbolic_unquantized_experts"
        and "routing_weight_dtype" in selection_attributes
        and selection_attributes["routing_weight_dtype"] == "bf16")
    if ascend018 and (hardware_id != "ascend_910b" or dtype != "bf16"
            or prediction_attributes["expert_parallel_size"] != 2):
        raise ValueError("BF16 expert weights require the explicit Ascend0.18 EP2 source")
    # The custom0.18 dispatcher already returns INT64 local-expert counts.
    # It is not the later full-range V2 ABI selected by E/H alone; its
    # public .to(int64) is an alias. Keep W1/activation/W2 shared, while
    # exposing dispatch/finalization as unbound mathematical source ports.
    recipe = (("moe_init_routing", "w1", "moe_swiglu", "w2", "moe_abs", "moe_token_unpermute")
        if ascend018 else select_fused_moe_recipe(
            hardware_id, selection_shape, hardware, selection_attributes))
    if hardware_id == "nvidia_h200" and "source_router_state" in selection_attributes:
        router_state = selection_attributes["source_router_state"]
        if router_state == "prepared_topk_weights_and_ids":
            # The model's external router already owns casts and normalization.
            # Keep expert-map conversion, which belongs to aligned assignment.
            router_roles = {"logits_cast", "moe_topk", "normalize_sum", "normalize_div"}
            recipe = tuple(stage for stage in recipe if stage not in router_roles)
        elif router_state != "softmax_logits_ready":
            raise NotImplementedError("H200 MoE requires explicit router producer ownership")
    reduce_work_fractions = select_fused_moe_reduce_work_fractions(
        hardware_id, selection_shape, hardware, selection_attributes)
    cast_roles = (() if ascend018 else select_fused_moe_cast_roles(
        hardware_id, selection_shape, hardware, selection_attributes))
    reduce_invocations = recipe.count("moe_reduce")
    sum_invocations = recipe.count("moe_sum")
    silu_invocations = recipe.count("moe_silu")
    mul_invocations = recipe.count("moe_mul")
    fill_invocations = recipe.count("moe_fill")
    cast_invocations = recipe.count("moe_cast")
    if reduce_invocations != len(reduce_work_fractions):
        raise ValueError(
            "fused-MoE recipe and source-derived moe_reduce work fractions "
            "disagree")
    source_naive_fill = (hardware_id == "nvidia_h200"
        and selection_attributes["model_execution_version"] == "vllm_0_30_0"
        and fill_invocations == 1)
    if fill_invocations not in (0, len(MOE_FILL_ROLES)) and not source_naive_fill:
        raise ValueError(
            "fused-MoE recipe must contain either zero or exactly three "
            "typed moe_fill invocations")
    if cast_invocations != len(cast_roles):
        raise ValueError(
            "fused-MoE recipe cast count disagrees with source-typed Cast "
            "roles")

    local_counts = prediction_attributes["local_tokens_per_expert"]
    if not isinstance(local_counts, tuple):
        raise TypeError(
            "FusedMoE physical-spec local_tokens_per_expert must be a tuple")
    if any(isinstance(count, bool) or not isinstance(count, int)
           or count < 0 for count in local_counts):
        raise ValueError(
            "FusedMoE physical-spec local routing counts must be "
            "non-negative integers")
    expert_parallel_size = prediction_attributes["expert_parallel_size"]
    if (isinstance(expert_parallel_size, bool)
            or not isinstance(expert_parallel_size, int)
            or expert_parallel_size <= 0):
        raise ValueError(
            "FusedMoE physical-spec expert_parallel_size must be positive")
    local_routed_rows = sum(local_counts)
    # An EP rank with no routed rows executes routing/finalization but has no
    # mathematical W1/W2/SwiGLU work.  Do not invent a zero-M launch.
    skip_empty_npu_projection = (
        hardware.kernel_memory_model == "npu_tile"
        and expert_parallel_size == 2
        and local_routed_rows == 0)
    graph_token_domain_attributes = {
        name: prediction_attributes[name]
        for name in (
            "graph_logical_tokens", "graph_model_input_tokens",
            "graph_runtime_mode", "model_input_shape",
        )
        if name in prediction_attributes
    }

    fill_ordinal = 0
    cast_ordinal = 0
    reduce_ordinal = 0
    specs = []
    for variant in recipe:
        physical_shape = shape
        launch_shape = selection_shape
        stage_dtype = dtype
        stage_selection_attributes = {}
        if variant in ("w1", "w2"):
            if skip_empty_npu_projection:
                continue
            kind = KernelKind.FUSED_MOE
            attributes = dict(prediction_attributes)
        elif variant == "moe_sum":
            if sum_invocations <= 0:
                raise ValueError(
                    "fused-MoE recipe contains moe_sum without a positive "
                    "invocation count")
            kind = KernelKind.VECTOR
            attributes = {"work_fraction": 1.0 / sum_invocations}
        elif variant == "moe_reduce":
            if reduce_ordinal >= len(reduce_work_fractions):
                raise ValueError(
                    "fused-MoE recipe contains more moe_reduce invocations "
                    "than its source-derived work partition")
            kind = KernelKind.VECTOR
            attributes = {
                "work_fraction": reduce_work_fractions[reduce_ordinal],
            }
            reduce_ordinal += 1
        elif variant in ("moe_silu", "moe_mul"):
            invocation_count = (
                silu_invocations if variant == "moe_silu" else mul_invocations)
            if invocation_count <= 0:
                raise ValueError(
                    f"fused-MoE recipe contains {variant} without a positive "
                    "invocation count")
            kind = KernelKind.VECTOR
            attributes = {"work_fraction": 1.0 / invocation_count}
        elif variant == "moe_fill":
            kind = KernelKind.VECTOR
            attributes = {"fill_role": ("num_tokens_post_padded" if source_naive_fill
                else MOE_FILL_ROLES[fill_ordinal])}
            fill_ordinal += 1
        elif variant == "moe_cast":
            kind = KernelKind.VECTOR
            attributes = {"cast_role": cast_roles[cast_ordinal]}
            cast_ordinal += 1
        elif variant == "moe_swiglu":
            if (hardware.kernel_memory_model == "npu_tile"
                    and local_routed_rows == 0):
                continue
            kind = KernelKind.VECTOR
            attributes = {"routed_rows": local_routed_rows}
        elif variant == "moe_compute_expert_tokens":
            kind = KernelKind.VECTOR
            attributes = {
                "tokens_per_expert": prediction_attributes[
                    "tokens_per_expert"],
            }
        elif variant in ("logits_cast", "normalize_sum", "normalize_div",
                         "expert_ids_long", "expert_map_lookup"):
            # These are real0.11 Torch submissions with their own tensor ABI,
            # not generic MoE vector work on [T,E,K,H,I].
            role = variant
            kind = KernelKind.VECTOR
            tokens, experts, topk = (selection_shape[name]
                for name in ("tokens", "experts", "topk"))
            stage_dtype = "fp32"
            if role == "logits_cast":
                elements = tokens * experts
                variant = "elementwise"
                attributes = {"operation": "cast_bf16_float32", "reads": 1, "writes": 1}
            elif role == "normalize_sum":
                elements = tokens
                variant = "reduce"
                attributes = {"reduction_factor": topk}
            elif role == "normalize_div":
                elements = tokens * topk
                variant = "elementwise"
                attributes = {"operation": "div", "reads": 2, "writes": 1}
            else:
                from src.kernel.launch_config.nvidia_h200.moe_legacy_launch import legacy_expert_map_elements
                elements = legacy_expert_map_elements(selection_shape, selection_attributes)
                stage_dtype = "int32"
                if role == "expert_ids_long":
                    variant = "elementwise"
                    attributes = {"operation": "cast_int32_int64", "reads": 1, "writes": 1,
                        "tensor_iterator_source": "pytorch_2_8_sm90"}
                else:
                    variant = "gather"
                    attributes = {"gather_factor": 1, "index_count": elements,
                        "index_bytes": 8, "scattered": True, "materialized_output": True}
            physical_shape = {"element_count": elements}
            launch_shape = {**selection_shape, "element_count": elements}
            # The division contract revalidates the exact broadcast geometry
            # using T/E/K; preserve these dimensions on the physical spec.
            if role == "normalize_div":
                physical_shape = launch_shape
            stage_selection_attributes["source_moe_router_stage"] = role
        elif variant == "moe_zero_output":
            # The0.30 EP-map body zeroes every routed W2 output slot,
            # including local slots subsequently overwritten by W2.
            if ("tensor_iterator_source" not in selection_attributes
                    or selection_attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"):
                raise NotImplementedError("H2000.30 output zero requires explicit Torch2.13 Fill source")
            elements = selection_shape["tokens"] * selection_shape["topk"] * selection_shape["hidden"]
            kind = KernelKind.VECTOR
            variant = "elementwise"
            physical_shape = {"element_count": elements}
            launch_shape = physical_shape
            attributes = {"operation": "fill", "reads": 0, "writes": 1}
        elif variant.startswith("moe_"):
            kind = KernelKind.VECTOR
            attributes = {}
        else:
            raise ValueError(f"unknown fused-MoE recipe variant: {variant!r}")

        if hardware_id == "nvidia_h200" and variant == "moe_topk":
            # fused_topk consumes the FP32 producer, even when the logical
            # public call received BF16 gate logits before logits_cast.
            stage_dtype = ("fp32" if selection_attributes["model_execution_version"] == "vllm_0_11_0"
                else selection_attributes["routing_input_dtype"])
            stage_selection_attributes["routing_input_dtype"] = stage_dtype

        attributes.update(graph_token_domain_attributes)
        if graph_token_domain_attributes and physical_shape is not shape:
            attributes["model_input_shape"] = dict(physical_shape)

        if ascend018 and variant in ("moe_init_routing", "moe_abs", "moe_token_unpermute"):
            specs.append(KernelSpec(kind, hardware_id, shape, dtype=dtype, variant=variant,
                attributes={"mathematical_program": "ascend_0_18_moe_dispatch_math_v1",
                    "local_experts": len(prediction_attributes["owned_experts"]),
                    "routed_rows": local_routed_rows}, launch=None,
                execution_support="unsupported", unsupported_reason="launch_selector_unavailable"))
            continue

        launch_attributes = dict(selection_attributes)
        launch_attributes.update(stage_selection_attributes)
        for name, value in attributes.items():
            if (name in launch_attributes
                    and launch_attributes[name] != value):
                raise ValueError(
                    f"fused-MoE launch attribute {name!r} conflicts with "
                    "its source-derived physical invocation")
            launch_attributes[name] = value
        launch = select_launch_config(
            hardware_id, kind, launch_shape, stage_dtype, variant,
            launch_attributes,
            hardware=hardware)
        support = ({"execution_support": "unsupported", "unsupported_reason": "typed_service_unavailable"}
            if variant == "elementwise" and attributes["operation"] == "div" else {})
        specs.append(KernelSpec(
            kind, hardware_id, physical_shape, dtype=stage_dtype,
            variant=launch.execution_variant,
            launch=launch, attributes=attributes, **support))

    if fill_ordinal != fill_invocations:
        raise AssertionError("typed moe_fill invocation count did not conserve")
    if cast_ordinal != cast_invocations:
        raise AssertionError("typed moe_cast invocation count did not conserve")
    if reduce_ordinal != reduce_invocations:
        raise AssertionError(
            "typed moe_reduce invocation count did not conserve")
    return tuple(specs)


def build_bound_fp8_expert_stage_specs(kernel, source_launches):
    """Materialize the already selected physical FP8 recipe for prediction."""
    from src.kernel.kernel_spec import KernelKind

    if (kernel.kind is not KernelKind.FUSED_MOE or kernel.variant != "routed_experts"
            or kernel.attributes["routed_weight_dtype"] != "fp8"
            or kernel.attributes["mathematical_program"] != "block_fp8_swiglu_v1"
            or kernel.attributes["expert_parallel_size"] != 1):
        raise ValueError("bound FP8 stages require the EP-off block-FP8 logical expert")
    tokens, topk = kernel.shape["tokens"], kernel.shape["topk"]
    hidden, intermediate = kernel.shape["hidden"], kernel.shape["intermediate"]
    main_attributes = {name: value for name, value in kernel.attributes.items()
        if name not in ("mathematical_program", "swiglu_limit")}
    specs = []
    for role, launch in source_launches:
        if role in ("w1", "w2"):
            spec = KernelSpec(KernelKind.FUSED_MOE, kernel.hardware_id,
                shape=kernel.shape, dtype=kernel.dtype, variant=role,
                launch=launch, attributes=main_attributes)
        elif role in ("input_fp8_quant", "intermediate_fp8_quant", "moe_silu_fp8_quant"):
            rows, columns = ((tokens, hidden) if role == "input_fp8_quant"
                else (tokens * topk, intermediate))
            spec = KernelSpec(KernelKind.VECTOR, kernel.hardware_id,
                shape={"element_count": rows * columns, "dtype_bytes": 2,
                    "rows": rows, "columns": columns},
                dtype=kernel.dtype, variant="fp8_group_quant", launch=launch,
                attributes={"group_size": 128})
        elif role in ("moe_align", "moe_align_small_batch", "moe_count_and_sort",
                "moe_act_and_mul", "moe_sum"):
            attributes = ({"work_fraction": 1.0} if role == "moe_sum" else {})
            spec = KernelSpec(KernelKind.VECTOR, kernel.hardware_id,
                shape=kernel.shape, dtype=kernel.dtype, variant=role,
                launch=launch, attributes=attributes)
        elif role == "moe_fill":
            spec = KernelSpec(KernelKind.VECTOR, kernel.hardware_id,
                shape=kernel.shape, dtype=kernel.dtype, variant=role,
                launch=launch, attributes={"fill_role": "num_tokens_post_padded"})
        else:
            raise NotImplementedError(f"bound FP8 expert stage is not priced: {role}")
        specs.append((role, spec))
    return tuple(specs)
