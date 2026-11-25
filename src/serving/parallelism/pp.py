"""Pipeline-parallel plan rules matching pinned vLLM placement."""

from dataclasses import replace


def pipeline_layer_ranges(
        num_layers, pipeline_parallel_size, layer_partition=None):
    """Return vLLM's default contiguous decoder-layer partitions."""

    for name, value in (
            ("num_layers", num_layers),
            ("pipeline_parallel_size", pipeline_parallel_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if pipeline_parallel_size > num_layers:
        raise ValueError(
            "pipeline_parallel_size cannot exceed num_layers")
    if layer_partition is None:
        layers_per_partition = num_layers // pipeline_parallel_size
        partitions = [layers_per_partition] * pipeline_parallel_size
        remaining_layers = num_layers % pipeline_parallel_size
        for offset in range(2, remaining_layers + 2):
            partitions[-offset] += 1
    else:
        if (not isinstance(layer_partition, (tuple, list))
                or len(layer_partition) != pipeline_parallel_size
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in layer_partition)
                or sum(layer_partition) != num_layers):
            raise ValueError(
                "pipeline layer partition must contain one positive layer "
                "count per stage and conserve num_layers")
        partitions = list(layer_partition)
    ranges = []
    layer_start = 0
    for layer_count in partitions:
        layer_end = layer_start + layer_count
        ranges.append((layer_start, layer_end))
        layer_start = layer_end
    if layer_start != num_layers:
        raise RuntimeError("pipeline layer placement does not conserve layers")
    return tuple(ranges)


def pipeline_layer_ranges_for_model(model, pipeline_parallel_size):
    """Return the source-backed PP partition for a model/runtime pair."""

    if model.model_type == "glm_moe_dsa" and pipeline_parallel_size == 2:
        # Explicit source candidate VLLM_PP_LAYER_PARTITION=38,40. Default
        # 39,39 starts stage1 at a shared-indexer layer without local top-k.
        # This is a declared partition, not an observed runtime placement.
        layer_partition = (38, 40)
    elif (model.model_type == "glm_moe_dsa"
            and pipeline_parallel_size == 4):
        layer_partition = (18, 20, 20, 20)
    elif model.model_type == "glm_moe_dsa" and pipeline_parallel_size > 1:
        default = pipeline_layer_ranges(model.num_hidden_layers, pipeline_parallel_size)
        if all(model.indexer_types[first] == "full" for first, _ in default):
            return default
        # The two-tensor PP protocol cannot transport an earlier stage's
        # top-k. Declare a balanced partition at actual full-indexer layers;
        # the native bridge must receive this explicit partition as well.
        full = tuple(index for index, kind in enumerate(model.indexer_types) if kind == "full")
        if not full or full[0] != 0 or len(full) < pipeline_parallel_size:
            raise NotImplementedError("ordinary GLM PP needs one local full indexer per nonempty stage")
        cuts = [0]
        for stage in range(1, pipeline_parallel_size):
            remaining = pipeline_parallel_size - stage - 1
            candidates = full[1:len(full) - remaining]
            cut = min((index for index in candidates if index > cuts[-1]),
                key=lambda index: (abs(index * pipeline_parallel_size - model.num_hidden_layers * stage), index))
            cuts.append(cut)
        cuts.append(model.num_hidden_layers)
        layer_partition = tuple(end - first for first, end in zip(cuts, cuts[1:]))
    else:
        layer_partition = None
    return pipeline_layer_ranges(
        model.num_hidden_layers,
        pipeline_parallel_size,
        layer_partition,
    )


def pipeline_layer_ranges_for_plan(plan, *, num_layers):
    """Recover the deployment's actual nonempty contiguous stage ownership.

    Defaults belong to plan construction. A source replica must preserve
    that chosen partition when projecting a distributed plan to local ranks.
    """
    from .parallel_plan import ParallelPlan
    if not isinstance(plan, ParallelPlan):
        raise TypeError("pipeline source ownership requires ParallelPlan")
    if type(num_layers) is not int or num_layers < 1 or plan.num_layers != num_layers:
        raise ValueError("pipeline plan layer count must match the complete source model")
    mapping = plan.layer_to_pipeline_stage
    size = plan.config.pipeline_parallel_size
    if (not isinstance(mapping, tuple) or len(mapping) != plan.num_layers
            or any(type(stage) is not int or not 0 <= stage < size for stage in mapping)
            or set(mapping) != set(range(size)) or mapping != tuple(sorted(mapping))):
        raise ValueError("pipeline plan must assign all layers to nonempty contiguous stages")
    return pipeline_layer_ranges(plan.num_layers, size, tuple(mapping.count(stage) for stage in range(size)))


def apply(plan, config, layer_partition=None):
    ranges = pipeline_layer_ranges(
        plan.num_layers, config.pipeline_parallel_size, layer_partition)
    if config.pipeline_parallel_size == 1:
        return plan
    stages = tuple(
        stage
        for stage, (layer_start, layer_end) in enumerate(ranges)
        for unused_layer in range(layer_start, layer_end)
    )
    return replace(
        plan,
        layer_to_pipeline_stage=stages,
        enabled_strategies=plan.enabled_strategies + ("pipeline_parallel",),
        communication_rules=plan.communication_rules + ("pipeline_activation_send",),
    )
