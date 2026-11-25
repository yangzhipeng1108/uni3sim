"""GLM vLLM0.30 PP source projection using the shared virtual layers.

The upstream TP-partial hidden is transmitted without an added reduction.
Its default flattened TP lanes are gathered by the receiver, exactly as
the original transport requests. This does not prove numerical TP/PP
equivalence, provider selection, async retirement or normal capacity.
"""

from .traced_forward import ForwardCollectiveGroup, ForwardPipelineGroup, ForwardTrace
from .traced_models import (DeepseekV32Model, _glm_source_weight_dtype, _plain_greedy_target,
    _run_vllm030_pipeline_stages, _glm_fresh_dense_prefill_without_topk)
from .traced_operators import _dsa_source_phase


def trace_glm_pipeline_forward(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_max_model_len, max_batched_tokens, workspace_initial_capacity_bytes,
        selection_source, source_profile, target_sampling_source="vllm_0_30_plain_greedy_bf16",
        prefill_attention_source=None, source_prompt_lengths=None, indexer_max_logits_mb=512, source_program_cache=None):
    """Execute source TP/PP layouts with their actual contiguous stages."""
    from .runtime_inputs import ExpertRoutingInput, vllm_0_30_source_model_inputs
    from src.serving.schedule.schedule_step import schedule_step_sha256
    from src.serving.deployment.validated_scope import (is_glm_fp8_rank_local_source_scope,
        is_glm_fp8_data_parallel_source_scope)
    config = pool.parallel_plan.config
    tp = config.tensor_parallel_size
    if model.weight_dtype == "fp8" and config.data_parallel_size > 1 and any(request.mtp_proposer_mode != "disabled" or request.mtp_proposed_tokens
            or request.mtp_proposer_tokens or request.mtp_verified_tokens or request.mtp_next_proposed_tokens
            for request in step.requests):
        raise NotImplementedError("ordinary GLM FP8 DP source requires MTP-off requests")
    h200_pp2 = (pool.hardware_id == "nvidia_h200" and config.pipeline_parallel_size == 2
        and not config.enable_expert_parallel)
    a800_pp4 = (pool.hardware_id == "nvidia_a800" and config.pipeline_parallel_size == 4
        and config.enable_expert_parallel and config.expert_parallel_size == 8 and model.weight_dtype == "bf16")
    ordinary_parallel = (model.weight_dtype == "bf16" and pool.hardware_id in ("nvidia_a800", "nvidia_h200")
        and config.pipeline_parallel_size > 1 and model.num_attention_heads % tp == 0
        and model.hidden_size % tp == 0 and model.intermediate_size % tp == 0
        and (config.enable_expert_parallel or model.moe_intermediate_size % tp == 0)
        or config.pipeline_parallel_size > 1 and (is_glm_fp8_rank_local_source_scope(pool, model)
            or is_glm_fp8_data_parallel_source_scope(pool, model)))
    sequence_size = tp if config.enable_expert_parallel and config.data_parallel_size > 1 and tp > 1 else 1
    if (model.model_type != "glm_moe_dsa" or (model.num_hidden_layers != 78 and not ordinary_parallel)
            or model.compute_dtype != "bf16" or model.weight_dtype != _glm_source_weight_dtype(source_profile)
            or not (ordinary_parallel or (h200_pp2 or a800_pp4) and tp == 8)
            or config.context_parallel_size != 1 or config.sequence_parallel_size not in (1, sequence_size)
            or (not ordinary_parallel and not config.is_rank_local)
            or len(pool.ranks) != tp * config.pipeline_parallel_size * config.data_parallel_size
            or (not config.enable_expert_parallel and model.moe_intermediate_size % (tp * config.data_parallel_size))):
        raise NotImplementedError("GLM pipeline source requires legal BF16 TP/PP/DP or whole-block FP8 rank-local TP/PP shapes with CP1")
    stage_ids = tuple(range(config.pipeline_parallel_size))
    assignments = pool.parallel_plan.layer_to_pipeline_stage
    if len(assignments) != model.num_hidden_layers or assignments != tuple(sorted(assignments)) or set(assignments) != set(stage_ids):
        raise ValueError("GLM pipeline requires the deployment's complete ordered layer assignments")
    ranges = tuple(tuple(i for i, owner in enumerate(assignments) if owner == stage) for stage in stage_ids)
    rows = sum(request.scheduled_tokens for request in step.requests)
    if (step.pool_name != pool.name or not step.requests or any(request.phase not in ("prefill", "decode")
            or request.scheduled_tokens <= 0 or request.mtp_proposer_tokens or request.mtp_proposed_tokens
            or request.mtp_verified_tokens for request in step.requests)):
        raise NotImplementedError("GLM pipeline projection requires ordinary MTP-off source requests")
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"GLM pipeline {name} requires explicit integer metadata for every row")
    if (any(not 0 <= token < model.vocab_size for token in input_ids)
            or positions != tuple(p for request in step.requests for p in range(request.computed_tokens,
                request.computed_tokens + request.scheduled_tokens))):
        raise ValueError("GLM pipeline IDs/positions must match the checkpoint and scheduled contexts")
    step, input_ids, order = vllm_0_30_source_model_inputs(step, embedding_token_ids=input_ids, prompt_lengths=source_prompt_lengths)
    positions = tuple(positions[row] for row in order["token_row_permutation"])
    slot_mapping = tuple(slot_mapping[row] for row in order["token_row_permutation"])
    q_lens = tuple(request.scheduled_tokens for request in step.requests)
    kv_lens = tuple(request.computed_tokens + request.scheduled_tokens for request in step.requests)
    phases = tuple(order["model_attention_phases"])
    if prefill_attention_source is None and "prefill" in phases and all(q == k and k <= model.index_topk
            for q, k, phase in zip(q_lens, kv_lens, phases) if phase == "prefill"):
        prefill_attention_source = "dense_mha_fresh"
    _dsa_source_phase(q_lens, kv_lens, phases, prefill_attention_source)
    if (any(model.indexer_types[indexes[0]] != "full" for indexes in ranges)
            and not (ordinary_parallel and _glm_fresh_dense_prefill_without_topk(
                model, q_lens, kv_lens, phases, prefill_attention_source))):
        raise NotImplementedError("GLM PP source requires a local full indexer: no incoming top-k port exists for a stage starting at a shared indexer; only pure fresh dense-MHA prefill avoids that dependency")
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slot_mapping)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)
            or type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
            or type(max_batched_tokens) is not int or max_batched_tokens < rows):
        raise ValueError("GLM pipeline requires complete bounded cache/context/batch metadata")
    if (not isinstance(workspace_initial_capacity_bytes, tuple) or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("GLM pipeline requires explicit workspace entry capacities for every physical rank")
    if selection_source != "declared_causal_topk" or not isinstance(expert_routing, ExpertRoutingInput):
        raise ValueError("GLM pipeline requires explicit causal selection and canonical routing input")
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("GLM pipeline only adapts the existing plain target sampler")
    if config.data_parallel_size > 1:
        return _trace_glm_data_pipeline(model, step, pool, expert_routing, input_ids=input_ids,
            phases=phases, ranges=ranges, source_max_model_len=source_max_model_len,
            max_batched_tokens=max_batched_tokens, workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
            selection_source=selection_source, source_profile=source_profile, target_sampling_source=target_sampling_source,
            prefill_attention_source=prefill_attention_source, model_input_order=order, indexer_max_logits_mb=indexer_max_logits_mb,
            source_program_cache=source_program_cache)
    stages = tuple(tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}")
        for rank in pool.physical_ranks_for_pipeline_stage(stage)) for stage in stage_ids)
    groups = tuple(ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:stage-{stage}")
        for stage, traces in enumerate(stages)) if tp > 1 else ()
    pipeline = ForwardPipelineGroup(tuple(trace for traces in stages for trace in traces),
        namespace=step.step_id, collective_groups=groups)
    pipeline.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "pipeline_layer_partition": tuple(len(indexes) for indexes in ranges),
        "tp_partial_hidden_sent_without_added_reduction": tp > 1, "numerical_tp_pp_equivalence_bound": False,
        "runner_intermediate_copy_source": "vllm_0_30_gpu_model_runner_sync_and_gather_intermediate_tensors",
        "normal_capacity_accepted": False, "actual_provider_bound": False}
    reduce, gather = (pool.collective_for("all_reduce"), pool.collective_for("all_gather")) if tp > 1 else (None, None)
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    digest = schedule_step_sha256(step)
    resolutions = tuple(expert_routing.resolve(step_id=step.step_id, layer_index=layer,
        moe_invocation_id=invocations[layer], schedule_step_sha256=digest, tokens=rows,
        num_experts=model.num_experts, topk=model.num_experts_per_tok, model_config_sha256=model.config_sha256,
        pool_name=pool.name, tensor_parallel_size=tp, expert_parallel_size=config.expert_parallel_size,
        expert_rank_group=(groups[assignments[layer]].participants if config.enable_expert_parallel
            else (stages[assignments[layer]][0].rank,)), expert_placement_strategy="linear",
        moe_layer_ordinal=model.moe_layer_ordinal(layer), assumption_model_semantics_sha256=(
            model.expert_routing_assumption_semantics_sha256 if expert_routing.source_kind == "explicit_assumption" else None))
        if model.layer_uses_moe(layer) else None for layer in range(model.num_hidden_layers))

    def run_stage(trace, stage, incoming):
        network = DeepseekV32Model(trace, model, pool.name, groups[stage] if groups else None, reduce, token_ids=input_ids,
            q_lens=q_lens, kv_lens=kv_lens, request_phases=phases, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_max_model_len=source_max_model_len,
            max_batched_tokens=max_batched_tokens, resolutions=resolutions, invocation_ids=invocations,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes[pool.ranks.index(trace.rank)],
            selection_source=selection_source, source_profile=source_profile,
            prefill_attention_source=prefill_attention_source, model_input_order=order,
            expert_parallel_size=config.expert_parallel_size,
            layer_range=(ranges[stage][0], ranges[stage][-1] + 1))
        pos = trace.tensor("positions", (rows,), "int64")
        ids = trace.tensor("input_ids", (rows,), "int32") if network.first_stage else None
        hidden = network(ids, pos, intermediate_tensors=incoming)
        if not network.last_stage:
            return hidden
        indices = trace.tensor("logits_indices", (len(q_lens),), "int32")
        _plain_greedy_target(trace, model, pool.name, groups[stage] if groups else None, gather, hidden, indices, q_lens,
            sample_logits=target_sampling_source == "vllm_0_30_plain_greedy_bf16",
            cast_logits=target_sampling_source == "vllm_0_30_plain_sampler_bf16")
        return None

    _run_vllm030_pipeline_stages(pipeline, pool, rows=rows, hidden_size=model.hidden_size,
        max_batched_tokens=max_batched_tokens, run_stage=run_stage, mark_source_projection=not ordinary_parallel)
    if ordinary_parallel:
        partition = tuple(len(indexes) for indexes in ranges)
        for stage, traces in enumerate(stages):
            for trace in traces:
                for identifier, node in tuple(trace._nodes.items()):
                    trace._nodes[identifier] = node.with_attributes({**node.attributes,
                        "glm_parallel_source": True, "source_projection_only": True,
                        "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
                        "source_pipeline_stage": stage, "pipeline_layer_partition": partition})
            if groups:
                for identifier, node in tuple(groups[stage]._contracts.items()):
                    groups[stage]._contracts[identifier] = node.with_attributes({**node.attributes,
                        "glm_parallel_source": True, "source_projection_only": True,
                        "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
                        "source_pipeline_stage": stage, "pipeline_layer_partition": partition})
    return pipeline


def _trace_glm_data_pipeline(model, step, pool, expert_routing, *, input_ids, phases, ranges,
        source_max_model_len, max_batched_tokens, workspace_initial_capacity_bytes, selection_source,
        source_profile, target_sampling_source, prefill_attention_source, model_input_order, indexer_max_logits_mb,
        source_program_cache=None):
    """Shared DP experts with one genuine PP transport per replica.

    The PP decoder keeps full local rows and late TP reduction. When the
    source MoE alone uses SP, its internal shard/EP/restore is distinct from
    model SP; dense layers remain ordinary TP modules.
    """
    from dataclasses import replace
    from .traced_glm_pcp import GLMPCPContext, pcp_model_rows
    from .traced_glm_mtp import GLMLegacyDummyAttentionFactory
    from .traced_operators import SymbolicGather
    from .traced_models import _source_expert_resolutions
    config = pool.parallel_plan.config
    tp, pp, dp = config.tensor_parallel_size, config.pipeline_parallel_size, config.data_parallel_size
    moe_sp = config.enable_expert_parallel and tp > 1
    requests_by_dp, ids_by_dp, phases_by_dp = [[] for _ in range(dp)], [[] for _ in range(dp)], [[] for _ in range(dp)]
    offset = 0
    for request, phase in zip(step.requests, phases, strict=True):
        rank = request.data_parallel_rank
        if rank >= dp:
            raise ValueError("GLM PP source request DP owner is outside its deployment")
        requests_by_dp[rank].append(request)
        ids_by_dp[rank].extend(input_ids[offset:offset + request.scheduled_tokens])
        phases_by_dp[rank].append(phase)
        offset += request.scheduled_tokens
    steps = tuple(replace(step, requests=tuple(requests), finished_request_ids=(), preempted_request_ids=())
        if requests else None for requests in requests_by_dp)
    from .traced_glm_pcp import _glm_legacy_input_state
    input_state = _glm_legacy_input_state(model, step, pool, ids_by_dp, source_program_cache,
        runner_limits=(source_max_model_len, max_batched_tokens))
    rows = tuple(pcp_model_rows(local, tuple(ids), 1) if local is not None else None
        for local, ids in zip(steps, ids_by_dp, strict=True))
    request_counts = tuple(0 if row is None else row.physical_rows for row in rows)
    counts = tuple(count if count else 1 for count in request_counts)
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    tensor = {(replica, stage): ForwardCollectiveGroup(tuple(traces[replica * pp * tp + stage * tp + lane]
        for lane in range(tp)), namespace=f"{step.step_id}:dp{replica}:stage{stage}:tp")
        for replica in range(dp) for stage in range(pp)} if tp > 1 else {}
    data = {(stage, lane): ForwardCollectiveGroup(tuple(traces[replica * pp * tp + stage * tp + lane]
        for replica in range(dp)), namespace=f"{step.step_id}:stage{stage}:lane{lane}:dp")
        for stage in range(pp) for lane in range(tp)}
    expert = {stage: ForwardCollectiveGroup(tuple(traces[replica * pp * tp + stage * tp + lane]
        for replica in range(dp) for lane in range(tp)), namespace=f"{step.step_id}:stage{stage}:ep")
        for stage in range(pp)} if config.enable_expert_parallel else {}
    root = ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:joined",
        subgroups=(*tensor.values(), *data.values(), *expert.values()))
    root.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "pipeline_layer_partition": tuple(len(indexes) for indexes in ranges),
        "tp_partial_hidden_sent_without_added_reduction": tp > 1,
        "numerical_tp_pp_equivalence_bound": False, "normal_capacity_accepted": False,
        "actual_provider_bound": False, "model_sequence_parallel": False, "moe_sequence_parallel": moe_sp,
        "global_model_input_order": model_input_order, "source_data_parallel_rows": request_counts,
        "source_model_rows_including_dummy": counts, "indexer_max_logits_mb": indexer_max_logits_mb,
        "idle_source_scope": "full legacy metadata=None target; no request cache, PP transfer, target logits or sampler"}
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    expert_rows = sum(-(-count // tp) * tp if moe_sp else count for count in counts)
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, expert_rows)
    reduce = pool.collective_for("all_reduce") if tp > 1 else None
    gather = pool.collective_for("all_gather") if tp > 1 else None
    dispatch, combine = (pool.collective_for(operation) for operation in ("all_gatherv", "reduce_scatterv"))
    def context_for(replica, stage, lane):
        return GLMPCPContext(rows[replica], 0, None, gather, dispatch=dispatch, combine=combine,
            tensor_group=tensor[(replica, stage)] if tp > 1 else None, tensor_collective=reduce,
            data_group=data[(stage, lane)], data_parallel_rank=replica, physical_rows_by_data_rank=counts,
            sequence_parallel=False, expert_group=expert[stage] if config.enable_expert_parallel else None,
            moe_sequence_parallel=moe_sp, dummy_run=steps[replica] is None, request_rows_by_data_rank=request_counts)
    for replica in range(dp):
        dummy = steps[replica] is None
        stage_traces = tuple(tuple(traces[replica * pp * tp + stage * tp + lane] for lane in range(tp)) for stage in range(pp))
        stage_groups = tuple(tensor[(replica, stage)] for stage in range(pp)) if tp > 1 else ()
        pipeline = ForwardPipelineGroup(tuple(trace for group in stage_traces for trace in group),
            namespace=f"{step.step_id}:dp{replica}:pipeline", collective_groups=stage_groups)
        local_step, local_rows = steps[replica], rows[replica]
        def run_stage(trace, stage, incoming):
            lane = stage_traces[stage].index(trace)
            context = context_for(replica, stage, lane)
            group = stage_groups[stage] if tp > 1 else None
            network = DeepseekV32Model(trace, model, pool.name, group, reduce,
                token_ids=((None if input_state.first_token_ids[replica] is None else
                    (input_state.first_token_ids[replica],)) if dummy else tuple(ids_by_dp[replica])),
                q_lens=(1,) if dummy else tuple(request.scheduled_tokens for request in local_step.requests),
                kv_lens=(1,) if dummy else tuple(request.computed_tokens + request.scheduled_tokens for request in local_step.requests),
                request_phases=("decode",) if dummy else tuple(phases_by_dp[replica]), slot_mapping=None if dummy else local_rows.slots_by_rank[0],
                cache_capacity_tokens=None if dummy else local_rows.cache_capacity_tokens, source_max_model_len=source_max_model_len,
                max_batched_tokens=max_batched_tokens, resolutions=resolutions, invocation_ids=invocations,
                workspace_initial_capacity_bytes=workspace_initial_capacity_bytes[pool.ranks.index(trace.rank)],
                selection_source=None if dummy else selection_source, source_profile=source_profile,
                prefill_attention_source=None if dummy else prefill_attention_source,
                model_input_order=None, expert_parallel_size=config.expert_parallel_size,
                layer_range=(ranges[stage][0], ranges[stage][-1] + 1), pcp_context=context,
                startup_attention_factory=GLMLegacyDummyAttentionFactory(tokens=1,source_max_model_len=source_max_model_len,
                    max_batched_tokens=max_batched_tokens,indexer_max_logits_mb=indexer_max_logits_mb) if dummy else None)
            positions = trace.tensor("positions", (counts[replica],), "int64")
            ids = trace.tensor("input_ids", (counts[replica],), "int32") if network.first_stage else None
            output = network(ids, positions, intermediate_tensors=incoming)
            if network.last_stage:
                if dummy:
                    SymbolicGather(trace,"dummy:selected-hidden",pool.name)(output,1,indices=(0,))
                    return None
                indices = trace.tensor("logits_indices", (len(local_step.requests),), "int32")
                _plain_greedy_target(trace, model, pool.name, group, gather, output, indices,
                    tuple(request.scheduled_tokens for request in local_step.requests),
                    sample_logits=target_sampling_source == "vllm_0_30_plain_greedy_bf16",
                    cast_logits=target_sampling_source == "vllm_0_30_plain_sampler_bf16")
                return None
            return output
        _run_vllm030_pipeline_stages(pipeline, pool, rows=counts[replica], hidden_size=model.hidden_size,
            max_batched_tokens=max_batched_tokens, run_stage=run_stage, mark_source_projection=False,dummy_run=dummy,
            declare_external_dummy_inputs=dummy)
    partition = tuple(len(indexes) for indexes in ranges)
    def metadata(node):
        from .workload_graph import WorkloadKind
        logical = pool.logical_rank_for_physical_rank(node.rank)
        return {"glm_parallel_source": True, "source_projection_only": True,
            "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
            "source_pipeline_stage": (logical // tp) % pp, "pipeline_layer_partition": partition,
            "source_data_parallel_rank": logical // (pp * tp),
            **({"source_dummy_run":True,"source_actual_request_count":0} if steps[logical//(pp*tp)] is None
                and node.kind not in (WorkloadKind.COLLECTIVE,WorkloadKind.P2P) else {}),
            **({"source_glm_fp8_target_only": True} if model.weight_dtype == "fp8" else {}),
            "source_glm_model_sequence_parallel": False, "source_glm_moe_sequence_parallel": moe_sp,
            **({"source_pcp_sequence_parallel": False, "source_glm_context_parallel_size": 1}
                if node.kind is not WorkloadKind.P2P else {})}
    for trace in traces:
        for identifier, node in tuple(trace._nodes.items()):
            trace._nodes[identifier] = node.with_attributes({**node.attributes, **metadata(node)})
    for group in root.subgroups:
        for identifier, node in tuple(group._contracts.items()):
            group._contracts[identifier] = node.with_attributes({**node.attributes, **metadata(node)})
    root.source_scope.update({"dummy_input_history_bound":source_program_cache is not None,
        "dummy_input_randomization":False,"retained_first_token_ids":input_state.first_token_ids,
        "dummy_input_randomization_source":"pinned_default_environment_false",
        "dummy_input_runtime_observation_bound":False,"retained_input_value_origins":input_state.value_origins})
    if source_program_cache is not None:
        root.graph  # Validate joined source before committing run history.
        source_program_cache.glm_legacy_inputs[id(pool)] = input_state
    return root
