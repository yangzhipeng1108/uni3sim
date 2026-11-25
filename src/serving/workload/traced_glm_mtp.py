"""V32 eager startup and legacy padded MTP on shared virtual forwards.

Adapt v0.30 gpu_model_runner, llm_base_proposer and deepseek_v32/nvidia/mtp.
Input IDs and acceptance are explicit scenarios. Logical page coordinates,
source metadata math and complete useful operator work do not establish a
native allocator, provider, launch or observed runtime execution.
"""

from dataclasses import dataclass, replace

from src.serving.schedule.schedule_step import schedule_step_sha256

from .runtime_inputs import ExpertRoutingInput, V32PaddedMTPInputs, vllm_0_30_source_model_inputs
from .traced_forward import ForwardCollectiveGroup, ForwardPipelineGroup, ForwardTrace, ForwardWorkspace
from .traced_models import (DeepseekV32Model, DeepseekV32MTPPredictorLayer, _GLMDecodeSourceProgram, _glm_source_weight_dtype,
    _source_expert_resolutions, _run_vllm030_pipeline_stages)
from .traced_operators import SymbolicGreedy, SymbolicLinear, SymbolicTPLogitsProcessor
from .traced_sampling import MTPSourceOps, _mtp_retained_last_rows, prepare_padded_drafter, sample_padded_raw_logits
from .workload_graph import WorkloadKind
from .model_config import is_glm_mtp_tensor_model_scope, is_glm_mtp_data_parallel_model_scope


def glm_fp8_rank_local_expert_source_metadata():
    """Declared0.30 NoDPEP prepare with deferred Cutlass block128 experts."""
    from .traced_glm_pcp import glm_fp8_deferred_expert_source_metadata
    return {**glm_fp8_deferred_expert_source_metadata(),
        "source_fp8_prepare_implementation": "vllm_0_30_no_dp_ep_modular",
        "source_fp8_dispatch_scope": "replicated_tp_rows_without_cross_rank_dispatch",
        "source_fp8_output_reduction": "decoder_late_tp_sum_or_partial_pp_hidden_frontier"}


@dataclass(frozen=True)
class _GLMMTPReplicaContext:
    """One real DP replica's traces and globally coordinated expert rows."""
    traces: tuple
    target_contexts: tuple
    draft_contexts: tuple
    target_resolutions: tuple
    draft_resolutions: tuple
    expert_parallel_size: int


@dataclass(frozen=True)
class _GLMMTPLegacyState:
    pool: object
    model_sha256: str
    num_drafts: int
    share_indices: bool
    skip_topk: tuple
    step_ids: frozenset


def _mtp_dp_rows(step, token_ids, *, block_size, num_drafts, draft_pass):
    from .traced_glm_pcp import PCPModelRows, PCPQueryChunk
    chunks, positions, slots = [], [], []
    base = offset = 0
    for request in step.requests:
        first = request.computed_tokens
        count = request.scheduled_tokens
        if draft_pass is not None and draft_pass > 0:
            first += count - 1 + draft_pass
            count = 1
        end = first + count
        chunks.append(PCPQueryChunk(request.request_id, request.phase if draft_pass in (None, 0) else "decode",
            0, count, offset, offset, first, base, end))
        positions.extend(range(first, end))
        slots.extend(range(base + first, base + end))
        offset += count
        extent = request.computed_tokens + request.scheduled_tokens + num_drafts - 1
        base += -(-extent // block_size) * block_size
    if len(token_ids) != offset:
        raise ValueError("MTP DP rows require every actual target/draft input ID")
    return PCPModelRows((tuple(chunks),), (token_ids,), (tuple(positions),), (tuple(slots),),
        ((False,) * offset,), tuple(range(offset)), offset, base)


def _trace_glm_mtp_data_parallel_forward(model, step, pool, expert_routing, *, input_ids, source_profile,
        source_max_model_len, max_batched_tokens, max_requests, workspace_initial_capacity_bytes,
        selection_source, prefill_attention_source, source_inputs, indexer_max_logits_mb,
        legacy_skip_topk_by_data_rank, source_program_cache):
    """TP/DP experts share routed shards for target and each draft pass."""
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from .traced_glm_pcp import GLMPCPContext
    from .traced_forward import SourceProgramCache
    config = pool.parallel_plan.config
    dp, pp, tp = config.data_parallel_size, config.pipeline_parallel_size, config.tensor_parallel_size
    ep = config.expert_parallel_size
    moe_sp = config.enable_expert_parallel and tp > 1
    model_sp = moe_sp and pp == 1
    if (pool.hardware_id != "nvidia_h200" or not is_glm_mtp_data_parallel_model_scope(model, config)
            or model.weight_dtype != _glm_source_weight_dtype(source_profile)
            or len(pool.ranks) != config.worker_count):
        raise NotImplementedError("V32 source requires SP/CP/DP1 or explicit BF16 whole-shard TP CP1 MTP DP; FP8 DP binds whole128 expert shards or uniform EP with deferred Cutlass input quantization")
    if (not isinstance(source_inputs, V32PaddedMTPInputs) or source_inputs.request_ids != tuple(r.request_id for r in step.requests)
            or not isinstance(expert_routing, ExpertRoutingInput) or step.pool_name != pool.name):
        raise ValueError("MTP DP requires the complete original schedule and independent typed draft inputs")
    if any(source_inputs.discarded_rows):
        raise NotImplementedError("MTP DP discarded replay/backup rows are outside its source support")
    if (not isinstance(workspace_initial_capacity_bytes, tuple) or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(value) is not int or value < 0 for value in workspace_initial_capacity_bytes)):
        raise ValueError("MTP DP requires actual per-rank workspace entry history")
    num_drafts = len(source_inputs.draft_input_token_ids)
    if (source_program_cache is not None and type(source_program_cache) is not SourceProgramCache
            or type(indexer_max_logits_mb) is not int or indexer_max_logits_mb <= 0):
        raise ValueError("GLM MTP requires its typed run history and positive indexer reserve")
    share = model.index_share_for_mtp_iteration
    prior = None
    if source_program_cache is not None and id(pool) in source_program_cache.glm_mtp_legacy_states:
        prior = source_program_cache.glm_mtp_legacy_states[id(pool)]
        if (type(prior) is not _GLMMTPLegacyState or prior.pool is not pool or prior.model_sha256 != model.config_sha256
                or prior.num_drafts != num_drafts or prior.share_indices != share or len(prior.skip_topk) != dp
                or step.step_id in prior.step_ids):
            raise ValueError("GLM MTP dummy requires its own unreplayed model/pool/proposer history")
    previous_skip = prior.skip_topk if prior is not None else ((False,) * dp if source_program_cache is not None or not share else None)
    if legacy_skip_topk_by_data_rank is not None:
        if (type(legacy_skip_topk_by_data_rank) is not tuple or len(legacy_skip_topk_by_data_rank) != dp
                or any(type(value) is not bool for value in legacy_skip_topk_by_data_rank)
                or previous_skip is not None and previous_skip != legacy_skip_topk_by_data_rank
                or not share and any(legacy_skip_topk_by_data_rank)):
            raise ValueError("GLM MTP declared skip-topk history conflicts with actual run-owned state")
        previous_skip = legacy_skip_topk_by_data_rank
    packed, packed_ids, packed_order = vllm_0_30_source_model_inputs(step, embedding_token_ids=input_ids,
        prompt_lengths=None, num_speculative_tokens=num_drafts)
    if (tuple(len(ids) for ids in source_inputs.draft_input_token_ids)
            != (step.token_count, *((len(step.requests),) * (num_drafts - 1)))):
        raise ValueError("MTP DP scenario must retain every physical target row and every recurrent request")
    row_indices, request_indices = {}, {}
    offset = 0
    for index, request in enumerate(packed.requests):
        row_indices[request.request_id] = tuple(range(offset, offset + request.scheduled_tokens))
        request_indices[request.request_id] = index
        offset += request.scheduled_tokens
    if any(request.data_parallel_rank >= dp for request in step.requests):
        raise ValueError("MTP DP request owner is outside the physical deployment")
    local_steps, local_ids, local_inputs, layouts = [], [], [], []
    for replica in range(dp):
        requests = tuple(request for request in packed.requests if request.data_parallel_rank == replica)
        if not requests:
            local_steps.append(None)
            local_ids.append(())
            local_inputs.append(None)
            layouts.append((None,) * (num_drafts + 1))
            continue
        local = replace(step, requests=requests, finished_request_ids=(), preempted_request_ids=())
        indices = tuple(index for request in requests for index in row_indices[request.request_id])
        ids = tuple(packed_ids[index] for index in indices)
        drafts = (tuple(source_inputs.draft_input_token_ids[0][index] for index in indices),
            *(tuple(values[request_indices[request.request_id]] for request in requests)
                for values in source_inputs.draft_input_token_ids[1:]))
        scenario = replace(source_inputs, request_ids=tuple(request.request_id for request in requests),
            discarded_rows=(False,) * len(requests), draft_input_token_ids=drafts)
        local_steps.append(local)
        local_ids.append(ids)
        local_inputs.append(scenario)
        layouts.append((_mtp_dp_rows(local, ids, block_size=scenario.block_size_tokens,
            num_drafts=num_drafts, draft_pass=None), *(_mtp_dp_rows(local, values,
                block_size=scenario.block_size_tokens, num_drafts=num_drafts, draft_pass=index)
                for index, values in enumerate(drafts))))
    request_counts = tuple(tuple(0 if layout[index] is None else layout[index].physical_rows for layout in layouts)
        for index in range(num_drafts + 1))
    if any(local is None for local in local_steps) and previous_skip is None:
        raise NotImplementedError("GLM MTP warm dummy requires explicit retained skip-topk history or its run-owned source cache")
    dummy_rows = num_drafts + 1
    counts = tuple(tuple(dummy_rows if count == 0 else count for count in phase) for phase in request_counts)
    if max(max(phase) for phase in counts) > max_batched_tokens:
        raise ValueError("GLM MTP source batch capacity must contain the full N+1 legacy dummy batch")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    tensor = {(replica, stage): ForwardCollectiveGroup(tuple(traces[replica * pp * tp + stage * tp + lane]
        for lane in range(tp)), namespace=f"{step.step_id}:dp{replica}:stage{stage}:tp")
        for replica in range(dp) for stage in range(pp)} if tp > 1 else {}
    data = {(stage, lane): ForwardCollectiveGroup(tuple(traces[replica * pp * tp + stage * tp + lane]
        for replica in range(dp)), namespace=(f"{step.step_id}:stage{stage}:dp" if tp == 1 else
            f"{step.step_id}:stage{stage}:lane{lane}:dp")) for stage in range(pp) for lane in range(tp)}
    expert = {stage: ForwardCollectiveGroup(tuple(traces[replica * pp * tp + stage * tp + lane]
        for replica in range(dp) for lane in range(tp)), namespace=f"{step.step_id}:stage{stage}:ep")
        for stage in range(pp)} if config.enable_expert_parallel else {}
    root = ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:joined", subgroups=(*tensor.values(), *data.values(), *expert.values()))
    dispatch, combine = (pool.collective_for(operation) for operation in ("all_gatherv", "reduce_scatterv"))
    reduce = pool.collective_for("all_reduce") if tp > 1 else None
    gather = pool.collective_for("all_gather") if tp > 1 else None
    def context(replica, stage, lane, phase):
        return GLMPCPContext(layouts[replica][phase], 0, None, gather, dispatch=dispatch, combine=combine,
            tensor_group=tensor[(replica, stage)] if tp > 1 else None, tensor_collective=reduce,
            data_group=data[(stage, lane)], data_parallel_rank=replica, physical_rows_by_data_rank=counts[phase],
            expert_group=expert[stage] if config.enable_expert_parallel else None,
            sequence_parallel=model_sp, moe_sequence_parallel=moe_sp, dummy_run=local_steps[replica] is None,
            request_rows_by_data_rank=request_counts[phase])
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    expert_counts = tuple(sum(-(-count // tp) * tp if moe_sp else count for count in phase) for phase in counts)
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, expert_counts[0])
    drafts = tuple(expert_routing.resolve(step_id=step.step_id, layer_index=None,
        moe_invocation_id=f"{step.step_id}:mtp-pass-{index}:moe", schedule_step_sha256=schedule_step_sha256(step),
        tokens=expert_counts[index + 1], num_experts=model.num_experts, topk=model.num_experts_per_tok,
        model_config_sha256=model.config_sha256, pool_name=pool.name, tensor_parallel_size=tp, expert_parallel_size=ep,
        expert_rank_group=(pool.physical_ranks_for_pipeline_stage(pp - 1) if config.enable_expert_parallel
            else (pool.physical_ranks_for_pipeline_stage(pp - 1)[0],)), expert_placement_strategy="linear",
        moe_layer_ordinal=sum(model.layer_uses_moe(layer) for layer in range(model.num_hidden_layers)),
        assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
            if expert_routing.source_kind == "explicit_assumption" else None)) for index in range(num_drafts))
    assignments = pool.parallel_plan.layer_to_pipeline_stage
    partition = tuple(assignments.count(stage) for stage in range(pp))
    local_plan = build_parallel_plan(replace(config, data_parallel_size=1, sequence_parallel_size=1,
        enable_expert_parallel=config.enable_expert_parallel and tp > 1), num_layers=model.num_hidden_layers,
        num_experts=model.num_experts, pipeline_layer_partition=partition)
    for replica, local in enumerate(local_steps):
        target_contexts = tuple(context(replica, stage, lane, 0) for stage in range(pp) for lane in range(tp))
        draft_contexts = tuple(tuple(context(replica, pp - 1, lane, index + 1) for lane in range(tp)) for index in range(num_drafts))
        local_traces = traces[replica * pp * tp:(replica + 1) * pp * tp]
        if local is None:
            _trace_glm_mtp_legacy_dummy(model, step, pool, local_traces,
                tuple(tensor[(replica,stage)] for stage in range(pp)) if tp>1 else (),
                target_contexts, draft_contexts, resolutions, drafts, invocations, partition,
                source_profile=source_profile, source_max_model_len=source_max_model_len,
                max_batched_tokens=max_batched_tokens, workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
                indexer_max_logits_mb=indexer_max_logits_mb, skip_topk=previous_skip[replica],
                reduce=reduce, gather=gather)
            continue
        view = replace(pool, ranks=tuple(trace.rank for trace in local_traces), parallel_plan=local_plan)
        trace_glm_dsa_mtp_forward(model, local, view, expert_routing, input_ids=local_ids[replica],
            source_profile=source_profile, source_max_model_len=source_max_model_len,
            max_batched_tokens=max_batched_tokens, max_requests=max_requests,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes[replica * pp * tp:(replica + 1) * pp * tp],
            selection_source=selection_source, prefill_attention_source=prefill_attention_source,
            source_inputs=local_inputs[replica], indexer_max_logits_mb=indexer_max_logits_mb,
            _replica_context=_GLMMTPReplicaContext(local_traces,
                target_contexts, draft_contexts, resolutions, drafts, ep))
    def metadata(node):
        logical = pool.logical_rank_for_physical_rank(node.rank)
        return {"glm_parallel_source": True, "source_projection_only": True,
            "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
            "source_pipeline_stage": logical // tp % pp, "pipeline_layer_partition": partition,
            "source_data_parallel_rank": logical // (pp * tp), "source_glm_model_sequence_parallel": model_sp,
            "source_glm_moe_sequence_parallel": moe_sp,
            **({"source_dummy_run":True,"source_actual_request_count":0} if local_steps[logical//(pp*tp)] is None
                and node.kind not in (WorkloadKind.COLLECTIVE,WorkloadKind.P2P) else {}),
            **({"source_glm_context_parallel_size": 1, "source_pcp_sequence_parallel": model_sp}
                if node.kind is not WorkloadKind.P2P else {})}
    for trace in traces:
        trace._nodes = {name: node.with_attributes({**node.attributes, **metadata(node)}) for name, node in trace._nodes.items()}
    for group in (*tensor.values(), *data.values(), *expert.values()):
        group._contracts = {name: node.with_attributes({**node.attributes, **metadata(node)})
            for name, node in group._contracts.items()}
    root.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "pipeline_layer_partition": partition, "MTP_data_parallel_target_rows": request_counts[0],
        "MTP_data_parallel_draft_rows": request_counts[1:], "global_model_input_order": packed_order,
        "MTP_physical_target_rows_including_dummy": counts[0], "MTP_physical_draft_rows_including_dummy": counts[1:],
        "MTP_legacy_dummy_skip_topk": previous_skip, "indexer_max_logits_mb": indexer_max_logits_mb,
        "idle_source_scope": "full N+1-row target plus N independent proposer-buffer forwards on last PP; no logits, cache writes or PP transfer",
        "dummy_draft_PP_provider_equivalence": pp == 1,
        "legacy_runtime_scope": "eager/noCG/noDBO/compilerSPoff; model/MoE SP padding represented separately",
        "target_draft_topk_backing": "shared last-stage target TopK; proposer load_model rebinds every draft module",
        "model_sequence_parallel": model_sp, "moe_sequence_parallel": moe_sp,
        "normal_capacity_accepted": False, "actual_provider_bound": False}
    if source_program_cache is not None:
        root.graph
        source_program_cache.glm_mtp_legacy_states[id(pool)] = _GLMMTPLegacyState(pool,model.config_sha256,
            num_drafts,share,tuple(share if local is not None else previous_skip[replica]
                for replica,local in enumerate(local_steps)),
            frozenset((step.step_id,)) if prior is None else prior.step_ids | {step.step_id})
    return root


def _trace_glm_mtp_legacy_dummy(model, step, pool, traces, groups, target_contexts, draft_contexts,
        resolutions, draft_resolutions, invocations, partition, *, source_profile, source_max_model_len,
        max_batched_tokens, workspace_initial_capacity_bytes, indexer_max_logits_mb, skip_topk, reduce, gather):
    """Eager idle target and proposer read independent retained runner inputs."""
    from .traced_operators import SymbolicGather
    from .workload_graph import WorkloadEdge
    tp = pool.parallel_plan.config.tensor_parallel_size
    pp, num_drafts = len(partition), len(draft_resolutions)
    rows = num_drafts + 1
    factory = GLMLegacyDummyAttentionFactory(tokens=rows,source_max_model_len=source_max_model_len,
        max_batched_tokens=max_batched_tokens,indexer_max_logits_mb=indexer_max_logits_mb)
    def run_stage(trace,stage,incoming):
        lane = traces.index(trace) % tp
        group = groups[stage] if groups else None
        start = sum(partition[:stage])
        network = DeepseekV32Model(trace,model,pool.name,group,reduce,token_ids=None,
            q_lens=(rows,),kv_lens=(rows,),request_phases=("decode",),slot_mapping=None,cache_capacity_tokens=None,
            source_max_model_len=source_max_model_len,max_batched_tokens=max_batched_tokens,resolutions=resolutions,
            invocation_ids=invocations,workspace_initial_capacity_bytes=workspace_initial_capacity_bytes[pool.ranks.index(trace.rank)],
            selection_source=None,source_profile=source_profile,prefill_attention_source=None,
            model_input_order=None,
            num_speculative_tokens=num_drafts,
            expert_parallel_size=pool.parallel_plan.config.expert_parallel_size,layer_range=(start,start+partition[stage]),
            pcp_context=target_contexts[stage*tp+lane],startup_attention_factory=factory)
        positions = trace.tensor("positions",(rows,),"int64")
        ids = trace.tensor("input_ids",(rows,),"int32") if stage==0 else None
        output = network(ids,positions,intermediate_tensors=incoming)
        if stage != pp-1:
            return output
        # The target result remains live through proposer.dummy_run. Its
        # selected-hidden gather follows all draft calls in the runner.
        completion = trace._writers[output.view.allocation_id]
        def order_call(before):
            added = set(trace._nodes)-before
            roots = added-{edge.destination for edge in trace._edges if edge.source in added and edge.destination in added}
            trace._edges.extend(WorkloadEdge(completion,root,"barrier") for root in sorted(roots))
        draft_ids = trace.tensor("mtp.input_ids",(max_batched_tokens,),"int32",persistent=True).narrow(0,0,rows)
        draft_positions = trace.tensor("mtp.positions",(max_batched_tokens,),"int64",persistent=True).narrow(0,0,rows)
        previous = trace.tensor("mtp.hidden_states",(max_batched_tokens,model.hidden_size),"bf16",persistent=True).narrow(0,0,rows)
        # load_model rebinds every draft module to the target's top-k arena.
        topk = network.topk_buffer
        input_norm = None
        for index in range(num_drafts):
            draft = DeepseekV32MTPPredictorLayer(trace,model,pool.name,f"mtp-pass-{index}",group=group,
                collective=reduce,gather=gather,input_token_ids=None,q_lens=(rows,),kv_lens=(rows,),request_phases=("decode",),
                slots=None,cache_capacity_tokens=None,resolution=draft_resolutions[index],
                invocation_id=f"{step.step_id}:mtp-pass-{index}:moe",source_profile=source_profile,
                source_max_model_len=source_max_model_len,max_batched_tokens=max_batched_tokens,workspace=network.workspace,
                topk_buffer=topk,cache=None,index_cache=None,rotary_cache=network.rotary_cache,
                index_rotary_cache=network.index_rotary_cache,slot_tensor=None,input_norm=input_norm,
                indexer_type="shared" if skip_topk else "full",prefill_attention_source=None,selection_source=None,
                startup_attention_factory=factory,expert_parallel_size=pool.parallel_plan.config.expert_parallel_size,
                pcp_context=draft_contexts[index][lane])
            before = set(trace._nodes)
            result = draft(draft_ids,draft_positions,previous,position_values=None)
            order_call(before)
            completion = trace._writers[result[0].view.allocation_id]
            del result
            input_norm = draft.input_norm
        before = set(trace._nodes)
        SymbolicGather(trace,"dummy:selected-hidden",pool.name)(output,1,indices=(rows-1,))
        order_call(before)
        return None
    if pp==1:
        for trace in traces:run_stage(trace,0,None)
    else:
        pipeline = ForwardPipelineGroup(traces,namespace=f"{step.step_id}:dummy-dp{target_contexts[0].data_parallel_rank}",
            collective_groups=groups)
        _run_vllm030_pipeline_stages(pipeline,pool,rows=rows,hidden_size=model.hidden_size,
            max_batched_tokens=max_batched_tokens,run_stage=run_stage,mark_source_projection=False,
            dummy_run=True,declare_external_dummy_inputs=True)


def _reserve_glm_no_metadata_indexer(trace, model, prefix, workspace, max_model_len, logits_mb):
    """Pinned sparse_attn_indexer's metadata=None allocation-only branch."""
    capacity = 40 * max_model_len
    workspace.get_simultaneous(((capacity, model.index_head_dim), "fp8"),
        ((capacity, 4), "uint8"), ((1024 * 1024,), "uint8"))
    dummy_logits = trace.tensor(trace.unique_name(prefix + ":profile-logits"),
        (logits_mb * 1024 * 1024,), "uint8")


class GLMLegacyDummyAttentionFactory:
    """Actual eager idle work, with full configured constructor capacities."""

    def __init__(self, *, tokens, source_max_model_len, max_batched_tokens, indexer_max_logits_mb):
        if (any(type(value) is not int or value <= 0 for value in
                (tokens, source_max_model_len, max_batched_tokens, indexer_max_logits_mb)) or tokens > max_batched_tokens):
            raise ValueError("GLM dummy requires positive source limits containing every physical row")
        self.tokens, self.max_model_len = tokens, source_max_model_len
        self.max_batched_tokens, self.logits_mb = max_batched_tokens, indexer_max_logits_mb

    def validate(self, *, rows, max_batched_tokens, source_max_model_len,
                 cache_capacity_tokens, slot_mapping, selection_source, prefill_attention_source):
        if (rows != self.tokens or max_batched_tokens != self.max_batched_tokens or source_max_model_len != self.max_model_len
                or any(value is not None for value in (cache_capacity_tokens, slot_mapping, selection_source, prefill_attention_source))):
            raise ValueError("GLM dummy requires its complete source capacities and no request/cache metadata")

    def attention(self, trace, model, pool, prefix, *, layer_index, tp_size,
                  workspace, rotary_cache, topk_buffer, dummy_index_q, dummy_index_weights,
                  mtp_indexer_type=None):
        return _GLMLegacyDummyAttention(trace, model, pool, prefix, layer_index=layer_index, tp_size=tp_size,
            workspace=workspace, rotary_cache=rotary_cache, topk_buffer=topk_buffer,
            dummy_index_q=dummy_index_q, dummy_index_weights=dummy_index_weights, factory=self,
            mtp_indexer_type=mtp_indexer_type)


class _GLMLegacyDummyAttention:
    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, workspace,
                 rotary_cache, topk_buffer, dummy_index_q, dummy_index_weights, factory, mtp_indexer_type):
        from .traced_models import DeepseekV32AttentionInput, _source_linear
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.workspace, self.factory = workspace, factory
        self.inputs = DeepseekV32AttentionInput(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, cache=None, index_cache=None, topk_buffer=topk_buffer,
            rotary_cache=rotary_cache, index_rotary_cache=rotary_cache, slot_mapping=None,
            dummy_index_q=dummy_index_q, dummy_index_weights=dummy_index_weights, no_metadata=True,
            mtp_indexer_type=mtp_indexer_type)
        self.o_proj = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.inputs.local_heads * model.v_head_dim, model.hidden_size, f"model.layers.{layer_index}.self_attn.o_proj")

    def __call__(self, positions, hidden_states):
        from .traced_operators import SymbolicFillZero
        if hidden_states.shape[0] != self.factory.tokens:
            raise ValueError("GLM dummy hidden rows differ from its actual legacy dummy batch")
        with self.inputs(positions, hidden_states) as prefix:
            q_c,q_nope,q_pe,ql_nope,index_q,weights,mqa_q,kv_out,rope_out,output = prefix
            if self.inputs.has_indexer:
                _reserve_glm_no_metadata_indexer(self.trace, self.model, self.prefix, self.workspace,
                    self.factory.max_model_len, self.factory.logits_mb)
            SymbolicFillZero(self.trace, self.prefix + ":dummy-output-zero", self.pool)(
                output, controls=(q_c,ql_nope,index_q,weights,mqa_q))
            return self.o_proj(output)[0]


class GLMStartupAttentionFactory:
    """Explicit eager/no-metadata storage program, never request attention."""

    def __init__(self, *, tokens, source_max_model_len, indexer_max_logits_mb):
        if any(type(value) is not int or value <= 0 for value in
                (tokens, source_max_model_len, indexer_max_logits_mb)):
            raise ValueError("GLM startup requires explicit positive constructor limits and indexer logits MiB")
        self.tokens, self.max_model_len, self.logits_mb = tokens, source_max_model_len, indexer_max_logits_mb

    def validate(self, *, rows, max_batched_tokens, source_max_model_len,
                 cache_capacity_tokens, slot_mapping, selection_source, prefill_attention_source):
        if (rows != self.tokens or max_batched_tokens != self.tokens
                or source_max_model_len != self.max_model_len
                or any(value is not None for value in (cache_capacity_tokens, slot_mapping,
                    selection_source, prefill_attention_source))):
            raise ValueError("GLM startup factory requires the full constructor batch and no request KV/slot/attention metadata")

    def attention(self, trace, model, pool, prefix, *, layer_index, tp_size,
                  workspace, rotary_cache, topk_buffer):
        return _GLMStartupAttention(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, workspace=workspace, rotary_cache=rotary_cache,
            topk_buffer=topk_buffer, factory=self)


class _GLMStartupAttention:
    """attention.forward prefix plus sparse indexer profile reservation.

    Same source linear operators and Q absorption as requests. The no-metadata
    branch passes no cache or slots to fused_norm_rope, reserves the indexer
    workspace, zeroes its existing output, and returns through o_proj.
    These storage-only nodes carry no timing/provider claim.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size,
                 workspace, rotary_cache, topk_buffer, factory):
        from .traced_models import _source_linear
        from .traced_operators import SymbolicBatchedLinear, SymbolicDSAQuery
        from .traced_sampling import StorageOnlyOps
        self.trace, self.model, self.prefix = trace, model, prefix
        self.heads, self.factory, self.workspace = model.num_attention_heads // tp_size, factory, workspace
        self.topk_buffer, self.rope = topk_buffer, rotary_cache
        self.has_indexer = layer_index == model.num_hidden_layers or model.indexer_types[layer_index] == "full"
        self.ops = StorageOnlyOps(trace, prefix + ":startup", pool)
        path = f"model.layers.{layer_index}.self_attn"
        self.qkv = _source_linear(trace, model, pool, prefix + ":qkv-a", model.hidden_size,
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, path + ".fused_qkv_a_proj")
        self.q_b = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), path + ".q_b_proj")
        self.q_absorb = SymbolicBatchedLinear(trace, prefix + ":q-absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.o_proj = _source_linear(trace, model, pool, prefix + ":o-proj", self.heads * model.v_head_dim,
            model.hidden_size, path + ".o_proj")
        self.wk, self.index_q = None, None
        if self.has_indexer:
            self.wk = _source_linear(trace, model, pool, prefix + ":wk-weights", model.hidden_size,
                model.index_head_dim + model.index_n_heads, path + ".indexer.wk_weights_proj")
            self.index_q = _source_linear(trace, model, pool, prefix + ":index-q", model.q_lora_rank,
                model.index_n_heads * model.index_head_dim, path + ".indexer.wq_b")
        dummy_q = trace.tensor(trace.unique_name("startup.dummy-index-q"), (1, 1, 1), "bf16", persistent=True)
        dummy_weights = trace.tensor(trace.unique_name("startup.dummy-index-weights"), (1, 1), "fp32", persistent=True)
        self.fused_q = SymbolicDSAQuery(trace, prefix + ":fused-q", pool, model,
            has_indexer=self.has_indexer, rotary_cache=rotary_cache, index_rotary_cache=rotary_cache,
            dummy_index_q=dummy_q, dummy_index_weights=dummy_weights)

    def _reserve_indexer(self):
        # indexer.get_max_prefill_buffer_size returns 40*runtime max len.
        # No divisor for GLM's uncompressed indexer. UINT8 dummy logits are
        # local to sparse_attn_indexer, while the workspace manager retains
        # its arena after the temporary views have been discarded.
        _reserve_glm_no_metadata_indexer(self.trace, self.model, self.prefix, self.workspace,
            self.factory.max_model_len, self.factory.logits_mb)

    def __call__(self, positions, hidden_states):
        trace, model, rows = self.trace, self.model, self.factory.tokens
        if hidden_states.shape != (rows, model.hidden_size) or positions.shape != (rows,):
            raise ValueError("GLM no-metadata startup attention requires its full constructor batch")
        qkv = self.qkv(hidden_states)[0]
        q_c, kv_c, k_pe = qkv.split((model.q_lora_rank, model.kv_lora_rank, model.qk_rope_head_dim))
        kw, index_k, index_weights = None, None, None
        if self.has_indexer:
            kw = self.wk(hidden_states)[0]
            index_k, index_weights = kw.split((model.index_head_dim, model.index_n_heads))
        output = trace.tensor(trace.unique_name(self.prefix + ":output"), (rows, self.heads * model.v_head_dim), "bf16")
        kv_out = trace.tensor(trace.unique_name(self.prefix + ":kv-out"), kv_c.shape, "bf16")
        rope_out = trace.tensor(trace.unique_name(self.prefix + ":rope-out"), k_pe.shape, "bf16")
        inputs = (positions, q_c, kv_c, k_pe, self.rope)
        if index_k is not None:
            inputs += (index_k,)
        q_c = self.ops.operation("no_metadata_fused_norm_rope", inputs, shape=q_c.shape,
            dtype="bf16", writes=(kv_out, rope_out, self.topk_buffer) if self.has_indexer else (kv_out, rope_out))
        q = self.q_b(q_c)[0].view_as_shape(rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
        q_nope, q_pe = q.split((model.qk_nope_head_dim, model.qk_rope_head_dim))
        ql_nope = self.q_absorb(q_nope.transpose(0, 1)).transpose(0, 1)
        index_q = None if self.index_q is None else self.index_q(q_c)[0].view_as_shape(
            rows, model.index_n_heads, model.index_head_dim)
        index_fp8, weights_out, mqa_q = self.fused_q(positions, q_pe, ql_nope, index_q, index_weights)
        if self.has_indexer:
            self._reserve_indexer()
        self.ops.operation("no_metadata_attention_zero_output", (), shape=output.shape, dtype=output.dtype, out=output)
        return self.o_proj(output)[0]


def trace_glm_startup_profile(model, pool, expert_routing, *, tokens, max_requests,
        source_max_model_len, workspace_initial_capacity_bytes, indexer_max_logits_mb,
        source_profile, include_mtp, num_speculative_tokens):
    """Full target, optional sequential NextN dummy forwards, then sampler.

    v0.30 eager CP/PP1 EP-off startup only. Draft forwards all consume the
    constructor's full token buffers; they neither compact to request count
    nor sample/reject tokens, and do not recycle the previous pass's result.
    Routing remains an explicit conditional input, not a proven global bound.
    """
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    from .traced_sampling import VLLM030WorkerStorage, worker_profile_storage_forward
    config = pool.parallel_plan.config
    weight_dtype = _glm_source_weight_dtype(source_profile)
    if (model.model_type != "glm_moe_dsa" or model.num_hidden_layers != 78
            or model.compute_dtype != "bf16" or model.weight_dtype != weight_dtype
            or pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size not in (1, 8)
            or len(pool.ranks) != config.tensor_parallel_size or config.enable_expert_parallel
            or any(value != 1 for value in (config.pipeline_parallel_size, config.sequence_parallel_size,
                config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("GLM startup requires accurate BF16-compute H200 TP1/8 PP/SP/CP/DP1 EP-off source")
    if (type(include_mtp) is not bool or type(num_speculative_tokens) is not int
            or (include_mtp and num_speculative_tokens < 1) or (not include_mtp and num_speculative_tokens != 0)):
        raise NotImplementedError("GLM startup requires target-only depth0 or a positive sequential eager MTP depth")
    factory = GLMStartupAttentionFactory(tokens=tokens, source_max_model_len=source_max_model_len,
        indexer_max_logits_mb=indexer_max_logits_mb)
    if (type(max_requests) is not int or max_requests <= 0 or source_max_model_len > model.max_position_embeddings
            or not isinstance(expert_routing, ExpertRoutingInput)
            or not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("GLM startup requires explicit request/context limits, routing and per-rank workspace history")
    batch = min(tokens, max_requests)
    lengths = (tokens // batch,) * (batch - 1) + (tokens // batch + tokens % batch,)
    if max(lengths) > source_max_model_len:
        raise ValueError("GLM startup context must cover every dummy query")
    step = ScheduleStep("glm-startup-profile", pool.name, tuple(ScheduledRequest(str(i), "prefill", q, 0)
        for i, q in enumerate(lengths)))
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, tokens)
    draft_resolution = None
    if include_mtp:
        draft_resolution = expert_routing.resolve(step_id=step.step_id, layer_index=None,
            moe_invocation_id=f"{step.step_id}:mtp-pass-0:moe", schedule_step_sha256=schedule_step_sha256(step),
            tokens=tokens, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name=pool.name, tensor_parallel_size=len(pool.ranks),
            expert_parallel_size=1, expert_rank_group=(pool.ranks[0],), expert_placement_strategy="linear",
            moe_layer_ordinal=sum(model.layer_uses_moe(layer) for layer in range(model.num_hidden_layers)),
            assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
                if expert_routing.source_kind == "explicit_assumption" else None))
    traces = tuple(ForwardTrace(rank=rank, namespace=f"glm-startup:rank-{rank}") for rank in pool.ranks)
    group = None if len(traces) == 1 else ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective = None if group is None else pool.collective_for("all_reduce")
    gather = None if group is None else pool.collective_for("all_gather")

    def rank_profile(trace, capacity):
        network = DeepseekV32Model(trace, model, pool.name, group, collective, token_ids=(0,) * tokens,
            q_lens=lengths, kv_lens=lengths, request_phases=("prefill",) * batch, slot_mapping=None,
            cache_capacity_tokens=None, source_max_model_len=source_max_model_len, max_batched_tokens=tokens,
            resolutions=resolutions, invocation_ids=invocations, workspace_initial_capacity_bytes=capacity,
            selection_source=None, source_profile=source_profile, prefill_attention_source=None,
            model_input_order=None, startup_attention_factory=factory)
        worker = VLLM030WorkerStorage(trace, tokens=tokens, max_requests=max_requests,
            inputs_embeds_size=model.hidden_size, dtype=model.compute_dtype)
        head = (SymbolicLinear(trace, "profile.lm-head", pool.name, model.hidden_size, model.vocab_size,
            kind=WorkloadKind.LM_HEAD) if group is None else SymbolicTPLogitsProcessor(trace, "profile.lm-head", pool.name,
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather))
        draft_forward = None
        if include_mtp:
            draft_ids = trace.tensor("profile.mtp.input_ids", (tokens,), "int32", persistent=True)
            draft_pos = trace.tensor("profile.mtp.positions", (tokens,), "int64", persistent=True)
            draft_hidden = trace.tensor("profile.mtp.hidden_states", (tokens, model.hidden_size), "bf16", persistent=True)
            draft_embeds = trace.tensor("profile.mtp.inputs_embeds", (tokens, model.hidden_size), "bf16", persistent=True)
            draft_arange = trace.tensor("profile.mtp.arange", (max(max_requests + 1, tokens),), "int32", persistent=True)
            draft_backup = trace.tensor("profile.mtp.backup_next_token_ids", (max_requests,), "int32", persistent=True)
            draft_slots = trace.tensor("profile.mtp.slot_mapping", (tokens,), "int64", persistent=True)
            draft_topk = trace.tensor("profile.mtp.topk", (tokens, model.index_topk), "int32", persistent=True)
            predictor = DeepseekV32MTPPredictorLayer(trace, model, pool.name, "profile.mtp", group=group,
                collective=collective, gather=gather, input_token_ids=(0,) * tokens, q_lens=lengths, kv_lens=lengths,
                request_phases=("prefill",) * batch, slots=None, cache_capacity_tokens=None,
                resolution=draft_resolution, invocation_id=f"{step.step_id}:mtp-pass-0:moe", source_profile=source_profile,
                source_max_model_len=source_max_model_len, max_batched_tokens=tokens, workspace=network.workspace,
                topk_buffer=draft_topk, cache=None, index_cache=None, rotary_cache=network.rotary_cache,
                index_rotary_cache=network.index_rotary_cache, slot_tensor=None, indexer_type="full",
                prefill_attention_source=None, selection_source=None, startup_attention_factory=factory)

            def draft_forward():
                trace.checkpoint("worker.drafter.begin")
                for index in range(num_speculative_tokens):
                    trace.checkpoint(f"worker.drafter.pass-{index}.begin")
                    # The return value is discarded before the next call.
                    predictor(draft_ids, draft_pos, draft_hidden, position_values=(0,) * tokens)
                    trace.checkpoint(f"worker.drafter.pass-{index}.end")
                trace.checkpoint("worker.drafter.end")

        compute_logits = (lambda x: head(x)[0]) if group is None else head
        trace.checkpoint("worker.profile.begin")
        worker_profile_storage_forward(trace, network, compute_logits, input_ids=worker.input_ids,
            positions=worker.positions, query_lengths=lengths,
            sampler_profile="vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise",
            pool_name=pool.name, drafter_dummy_run=draft_forward)
        trace.checkpoint("worker.profile.end")

    for trace, capacity in zip(traces, workspace_initial_capacity_bytes, strict=True):
        rank_profile(trace, capacity)
    return traces[0] if group is None else group


class _GLMMTP5SourceProgram(_GLMDecodeSourceProgram):
    """Private fixed-branch issuer, built from one complete source recording.

    Ordinary GLM's storage/graph binding stays the owner. A current source
    prefix executes four target layers plus all five proposer passes and
    rejection. It is metadata evidence only, never a predicted partial model.
    Every target layer's actual routing is resolved independently.
    """

    def source_unchanged(self):
        return self.guard.unchanged()

    @staticmethod
    def create(source, prefix, model, pool):
        from .traced_models import _DenseMetadataSourceGuard
        from . import traced_sampling
        program = _GLMMTP5SourceProgram(source, model, pool)
        program.guard = _DenseMetadataSourceGuard((__name__, traced_sampling.__name__))
        program.mtp_prefix = prefix
        def layout(view):
            return (view.allocation_id,view.offset_bytes,view.shape,view.strides_bytes,view.dtype,view.element_bytes)
        bindings=[]
        for full,current in zip(source.traces,prefix.traces,strict=True):
            by_layout={}
            for name,view in full._views.items():
                key=layout(view)
                if key not in by_layout:
                    by_layout[key]=[]
                by_layout[key].append(name)
            origins={}
            for name,view in current._views.items():
                key=layout(view)
                if key in by_layout:
                    if key not in origins:
                        origins[key]=[]
                    origins[key].append(name)
            # Record alias groups rather than their Cartesian product. All
            # current aliases must still agree before any target is rebound.
            bindings.append(tuple((key,tuple(names),tuple(by_layout[key]))
                for key,names in origins.items()))
        program.mtp_view_bindings=tuple(bindings)
        parameters=[]
        for trace in prefix.traces:
            nodes=[]
            for name,node in trace._nodes.items():
                attrs=node.attributes
                if 'embedding_token_ids' in attrs:
                    role=name[len(trace.namespace)+1:]
                    if role=='embedding':
                        nodes.append((name,'embedding',-1))
                    elif role in tuple(f'mtp-pass-{i}:embedding' for i in range(5)):
                        nodes.append((name,'embedding',int(role.split(':')[0][-1])))
                    else:
                        return program
                if 'kv_lens' in attrs:
                    role=name[len(trace.namespace)+1:]
                    if (node.kind is not WorkloadKind.EAGER_ATTN
                            or not role.endswith((':attention:indexer',':attention:sparse'))
                            or not role.startswith(('layer-','mtp-pass-'))):
                        return program
                    nodes.append((name,'attention',int(role.split(':')[0][-1]) if role.startswith('mtp-pass-') else 0))
                if 'operation' in attrs:
                    operation=attrs['operation']
                    if operation in ('rejection','prepare_next_padded'):
                        nodes.append((name,operation,0))
                    if operation in ('gather','scatter'):
                        index=trace._views[attrs['symbolic_tensor_inputs'][1]]
                        if index.allocation_id.endswith(':token_indices_to_sample'):
                            nodes.append((name,'retained',0))
            caches=tuple(name for name in trace._allocations
                if name.endswith((':cache',':index-cache',':mtp.kv-cache',':mtp.index-cache')))
            if (sum(role=='embedding' for _,role,_ in nodes)!=6
                    or len(caches)!=model.num_hidden_layers+model.indexer_types.count('full')+2
                    or any(trace._views[name].allocation_id!=name or trace._views[name].offset_bytes
                        or trace._views[name].strides_bytes!=(trace._views[name].shape[1]*trace._views[name].element_bytes,
                            trace._views[name].element_bytes) for name in caches)):
                return program
            parameters.append((tuple(nodes),caches))
        program.mtp_parameters=tuple(parameters)
        return program

    def parameter_prefix(self, *, input_ids, drafts, ordering, kv_lens, positions,
                         retained, arena_size, verified, accepted):
        """Rebind the recorded prefix from the guarded q6 source semantics.

        This is metadata, not a partial predicted model. The full recorded
        target and five drafts still bind and lower below, including fresh
        routing for every MoE invocation. Workspace capacity is in the key.
        """
        from copy import copy
        if 'mtp_parameters' not in self.__dict__:
            return None
        result=copy(self.mtp_prefix)
        traces=[]
        selected=tuple(positions[index]+1 for index in retained)
        for prior,(nodes,caches) in zip(self.mtp_prefix.traces,self.mtp_parameters,strict=True):
            trace=copy(prior)
            trace._nodes,trace._allocations,trace._views=dict(prior._nodes),dict(prior._allocations),dict(prior._views)
            for name,role,stage in nodes:
                node=prior._nodes[name]
                if role=='embedding':
                    attrs={'embedding_token_ids':input_ids if stage==-1 else drafts[stage]}
                    if 'symbolic_model_input_order' in node.attributes:
                        attrs['symbolic_model_input_order']=ordering
                    shape={}
                elif role=='attention':
                    lengths=kv_lens if stage==0 else tuple(length+stage for length in selected)
                    shape={'kv_len':max(lengths)}
                    attrs={'kv_lens':lengths}
                    if 'selected_kv_lens' in node.attributes:
                        chosen=tuple(min(self.model.index_topk,length) for length in (
                            selected if 'reused_selection_kv_lens' in node.attributes else lengths))
                        shape['topk']=max(chosen)
                        attrs['selected_kv_lens']=chosen
                    if 'reused_selection_kv_lens' in node.attributes:
                        attrs['reused_selection_kv_lens']=selected
                elif role=='rejection':
                    shape,attrs={},dict(verified_tokens=verified,accepted_tokens=accepted)
                elif role=='prepare_next_padded':
                    shape,attrs={},dict(valid_sampled_counts=tuple(count+1 for count in accepted))
                elif role=='retained':
                    shape,attrs={},dict(selected_indices=retained)
                else:
                    raise RuntimeError('unknown compiled MTP parameter role')
                trace._nodes[name]=node.with_updates(shape,attrs)
            for name in caches:
                view=prior._views[name]
                shape=(arena_size,view.shape[1])
                trace._allocations[name]=replace(prior._allocations[name],size_bytes=arena_size*shape[1]*view.element_bytes)
                trace._views[name]=ForwardTrace._dense_view(name,shape,view.dtype)
            trace._recorded_graph=None
            traces.append(trace)
        result.traces=tuple(traces)
        result._recorded_graph=None
        return result

    @staticmethod
    def bind_views(current, prior, view_bindings):
        from src.kernel.memory_access import TensorView
        updates={}
        for old_layout,origins,targets in view_bindings:
            view=current._views[origins[0]]
            layout=(view.allocation_id,view.offset_bytes,view.shape,
                view.strides_bytes,view.dtype,view.element_bytes)
            for origin in origins:
                alias=current._views[origin]
                if (type(alias) is not TensorView or "_region" not in alias.__dict__
                        or alias.allocation_id != prior._views[origin].allocation_id
                        or (alias.allocation_id,alias.offset_bytes,alias.shape,
                            alias.strides_bytes,alias.dtype,alias.element_bytes) != layout):
                    return None
            if layout != old_layout:
                for target in targets:
                    updates[target]=view.with_identity(target,view.allocation_id)
        return updates

    @staticmethod
    def bind(program, prefix, resolutions, invocations, draft_resolutions, step):
        from copy import copy
        from types import MappingProxyType
        import weakref
        from .traced_models import _GLMDecodeSourceBinding
        from .traced_operators import SymbolicRoutedExperts
        if not program.source_unchanged():
            return None
        rows, batch = step.token_count, len(step.requests)
        routes = {}
        for base, layer, resolution, invocation, count in (
                *((f'layer-{i}:mlp:routed', i, resolutions[i], invocations[i], rows)
                    for i in range(program.model.num_hidden_layers) if program.model.layer_uses_moe(i)),
                *((f'mtp-pass-{i}:decoder:mlp:routed', program.model.num_hidden_layers,
                    draft_resolutions[i], f'{step.step_id}:mtp-pass-{i}:moe', rows if i == 0 else batch)
                    for i in range(5))):
            operator = SymbolicRoutedExperts(prefix.traces[0], 'binding', program.pool.name,
                program.model,tp_size=8,layer_index=layer,resolution=resolution,
                invocation_id=invocation,routing_source_stage='dsv32_moe_route')
            metadata = operator.routing_metadata(count)
            routes[base] = {**metadata,'routing_provenance':MappingProxyType(metadata['routing_provenance'])}
        result, traces, changed = copy(program.source), [], {}
        # Tensor ports/graph edges are the recorded full program. The short
        # metadata prefix has a different final target producer; never copy it.
        ports = {'symbolic_tensor_inputs','symbolic_tensor_outputs','symbolic_parameter_inputs'}
        for template, current, prior, graph, view_bindings in zip(program.source.traces,prefix.traces,
                program.mtp_prefix.traces,program.local_graphs,program.mtp_view_bindings,strict=True):
            if (set(current._nodes) != set(prior._nodes) or set(current._allocations) != set(prior._allocations)
                    or set(current._views) != set(prior._views)):
                return None
            trace = copy(template)
            trace._nodes,trace._allocations,trace._views = dict(template._nodes),dict(template._allocations),dict(template._views)
            for name,node in current._nodes.items():
                old = prior._nodes[name]
                if ((node.kind,node.pool_name,node.rank,node.dtype,node.variant,set(node.shape),set(node.attributes))
                        != (old.kind,old.pool_name,old.rank,old.dtype,old.variant,set(old.shape),set(old.attributes))
                        or any(node.attributes[k] != old.attributes[k] for k in ports if k in node.attributes)):
                    return None
                shape = {k:v for k,v in node.shape.items() if v != old.shape[k]}
                attrs = {k:v for k,v in node.attributes.items() if k not in ports and v != old.attributes[k]}
                if node.kind is WorkloadKind.FUSED_MOE:
                    attrs = {}
                if shape or attrs:
                    trace._nodes[name] = template._nodes[name].with_updates(shape,attrs)
                    changed[name] = trace._nodes[name]
            for layer in range(program.model.num_hidden_layers):
                base = trace.qualify(f'layer-{layer}')
                for role in (('attention:sparse','attention:indexer') if program.model.indexer_types[layer]=='full' else ('attention:sparse',)):
                    origin = current._nodes[current.qualify('layer-0:'+role)]
                    prior_node = prior._nodes[prior.qualify('layer-0:'+role)]
                    shape = {k:v for k,v in origin.shape.items() if v != prior_node.shape[k]}
                    attrs = {k:v for k,v in origin.attributes.items() if k not in ports and v != prior_node.attributes[k]}
                    name = base+':'+role
                    if shape or attrs:
                        trace._nodes[name] = template._nodes[name].with_updates(shape,attrs)
                        changed[name] = trace._nodes[name]
            for base,metadata in routes.items():
                name = trace.qualify(base)
                if name not in trace._nodes:
                    return None
                trace._nodes[name] = trace._nodes[name].with_updates({},metadata)
                changed[name] = trace._nodes[name]
            for name,current_allocation in current._allocations.items():
                if name not in template._allocations:
                    return None
                old = template._allocations[name]
                trace._allocations[name] = current_allocation.with_lifetime(name,old.allocated_event,old.freed_event)
            view_updates=program.bind_views(current,prior,view_bindings)
            if view_updates is None:
                return None
            trace._views.update(view_updates)
            trace._recorded_graph = (trace._nodes,graph.with_nodes(trace._nodes.values()))
            traces.append(trace)
        result.traces = tuple(traces)
        result._recorded_graph = program.graph.with_nodes(changed[n.node_id] if n.node_id in changed else n for n in program.graph.nodes)
        graph = result._recorded_graph
        object.__setattr__(graph,'_glm_decode_binding',_GLMDecodeSourceBinding(
            weakref.ref(program),weakref.ref(graph),graph.nodes,tuple(changed.values())))
        return result


def trace_glm_dsa_mtp_forward(model, step, pool, expert_routing, *, input_ids, source_profile,
                            source_max_model_len, max_batched_tokens, max_requests,
                            workspace_initial_capacity_bytes, selection_source,
                            prefill_attention_source, source_inputs, _replica_context=None,
                            indexer_max_logits_mb=512, legacy_skip_topk_by_data_rank=None, source_program_cache=None,
                            _reuse_prefix=False):
    """Full target → greedy rejection → declared sequential padded drafts.

    Both target and drafter retain speculative_config and its decode threshold.
    Draft IDs explicitly declare the pass count, including the source's
    one-draft early return. No numerical draft or acceptance is generated.
    A mixed step may have fresh or preemption-replay requests whose proposals
    are not consumed. The source still forwards their complete first-pass
    rows, then one row per incoming request on each remaining pass. It does
    not compact the physical drafter batch to the proposal consumers.
    The target uses its actual PP ranges and tensor transport; rejection and
    the complete sequential drafter execute only on the last PP stage.
    """
    config = pool.parallel_plan.config
    if config.data_parallel_size > 1:
        if _replica_context is not None:
            raise ValueError("MTP DP cannot nest replica projections")
        return _trace_glm_mtp_data_parallel_forward(model, step, pool, expert_routing,
            input_ids=input_ids, source_profile=source_profile, source_max_model_len=source_max_model_len,
            max_batched_tokens=max_batched_tokens, max_requests=max_requests,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes, selection_source=selection_source,
            prefill_attention_source=prefill_attention_source, source_inputs=source_inputs,
            indexer_max_logits_mb=indexer_max_logits_mb, legacy_skip_topk_by_data_rank=legacy_skip_topk_by_data_rank,
            source_program_cache=source_program_cache)
    if _replica_context is not None and (type(_replica_context) is not _GLMMTPReplicaContext
            or (_replica_context.target_contexts[0].expert_group is not None) != (_replica_context.expert_parallel_size > 1)
            or tuple(trace.rank for trace in _replica_context.traces) != pool.ranks):
        raise ValueError("MTP replica projection requires its actual whole-shard TP and global expert physical traces")
    expert_size = config.expert_parallel_size if _replica_context is None else _replica_context.expert_parallel_size
    if (model.model_type != "glm_moe_dsa" or model.mtp_architecture != "glm_moe_dsa_mtp"
            or model.num_hidden_layers != 78 or model.num_mtp_layers != 1
            or model.compute_dtype != "bf16" or model.weight_dtype != _glm_source_weight_dtype(source_profile)
            or pool.hardware_id != "nvidia_h200" or not is_glm_mtp_tensor_model_scope(model, config.tensor_parallel_size,
                expert_parallel=config.enable_expert_parallel and _replica_context is None)
            or len(pool.ranks) != config.worker_count or config.pipeline_parallel_size > model.num_hidden_layers
            or any(value != 1 for value in (config.sequence_parallel_size,
                config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("V32 padded source requires its H200 BF16-compute whole-shard TP/nonempty PP SP/CP/DP1 checkpoint, with EP-off or BF16 stage-local EP")
    if not isinstance(source_inputs, V32PaddedMTPInputs) or not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("V32 padded source requires typed independent draft and routing scenarios")
    if source_inputs.request_ids != tuple(request.request_id for request in step.requests) or step.pool_name != pool.name:
        raise ValueError("V32 padded scenario must identify the complete incoming request order")
    if (any(type(value) is not int or value < 1 for value in (source_max_model_len, max_batched_tokens, max_requests))
            or source_max_model_len > model.max_position_embeddings or len(step.requests) > max_requests
            or sum(request.scheduled_tokens for request in step.requests) > max_batched_tokens):
        raise ValueError("V32 padded source requires explicit constructor limits covering all rows and requests")
    if (not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("V32 padded source requires explicit per-rank workspace entry history")
    drafts = source_inputs.draft_input_token_ids
    num_drafts = len(drafts)
    consumers = tuple(request for request in step.requests if request.mtp_proposer_mode == "padded")
    proposer_depths = {request.mtp_proposer_tokens for request in consumers}
    if proposer_depths not in ({num_drafts}, {0}):
        raise NotImplementedError("V32 source requires declared passes matching its step-global drafter execution decision")
    drafter_executes = proposer_depths != {0}
    disabled = tuple(request for request in step.requests if request.mtp_proposer_mode == "disabled")
    # gpu_model_runner.propose_draft_token_ids keeps the complete input
    # batch in prepare_inputs_padded; llm_base_proposer.set_inputs_first_pass
    # globally shifts all target rows and replaces one retained row per
    # request. K=1 returns only after that full forward and full-batch sample;
    # K>1 keeps the same batch size for rollback and every one-row recurrence.
    # Disabled rows are therefore an output-consumer distinction, not
    # permission to drop their physical work. A non-discarded sampled
    # candidate is distinct from a consumed output/proposal: raw output 0
    # stays 0 and no candidate IDs are exported as consumed for that request.
    # The independent typed scenario supplies IDs, never a generated value.
    if any(not ((request.phase == "prefill" and request.computed_tokens == 0)
                or (request.phase == "decode" and request.resumed_from_preemption
                    and request.recomputed_tokens == request.scheduled_tokens))
            or request.raw_model_output_tokens not in (0, 1)
            or any((request.mtp_proposed_tokens, request.mtp_verified_tokens, request.mtp_accepted_tokens,
                request.mtp_proposer_tokens, request.mtp_next_proposed_tokens))
            for request in disabled):
        raise NotImplementedError("V32 mixed disabled requests require fresh or explicit full preemption-replay rows")
    guard = "V32 source requires complete padded draft rows matching its declared passes, valid anchors and target IDs"
    if not isinstance(input_ids, tuple) or any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_ids):
        raise NotImplementedError(guard + ": target IDs must be an in-vocabulary integer tuple")
    if (not consumers or len(consumers) + len(disabled) != len(step.requests)
            or any(request.mtp_proposer_tokens not in (0, num_drafts) for request in consumers)):
        raise NotImplementedError(guard + ": incomplete padded/disabled request partition or proposer depth")
    discarded_by_id = dict(zip(source_inputs.request_ids, source_inputs.discarded_rows, strict=True))
    # GPUModelRunner discards a sample while optimistic seq_len is still
    # behind Request.num_tokens. A full preemption replay chunk has no
    # consumed target sample, but the step-global padded drafter still runs.
    # llm_base_proposer.prepare_next_token_ids_padded takes its explicit
    # backup ID and publishes valid_count=0; no acceptance draw is added.
    for request in step.requests:
        if discarded_by_id[request.request_id]:
            if request.raw_model_output_tokens or any((request.mtp_verified_tokens,
                    request.mtp_padding_tokens, request.mtp_accepted_tokens)):
                raise ValueError("discarded GLM replay requires zero consumed sample and no verified/padding suffix")
            if (request.recomputed_tokens != request.scheduled_tokens
                    or request.mtp_proposer_mode != "padded" or request.mtp_proposer_tokens != num_drafts):
                raise NotImplementedError("discarded GLM source supports complete physical replay chunks with the full padded drafter")
    invalid_raw = tuple(request.request_id for request in consumers
        if not discarded_by_id[request.request_id]
        and request.raw_model_output_tokens != request.mtp_accepted_tokens + 1)
    if invalid_raw:
        raise NotImplementedError(guard + f": raw output differs from accepted draft prefix plus target for {invalid_raw!r}")
    invalid_anchors = tuple(request.request_id for request in step.requests
        if request.scheduled_tokens < request.mtp_verified_tokens + 1)
    if invalid_anchors:
        raise NotImplementedError(guard + f": scheduled rows lack a target anchor after verified drafts for {invalid_anchors!r}")
    # This synchronous source has one pending sampled token before the
    # scheduled draft suffix (scheduler.py:656-660,813-823). Extra target
    # rows need an explicit recomputation domain; they are not additional
    # speculative slots with no declared draft token.
    if any(request.phase == "decode" and request.recomputed_tokens == 0
            and request.scheduled_tokens != request.mtp_verified_tokens + 1 for request in consumers):
        raise ValueError("MTP target verification must schedule one terminal token after the verified draft tokens")
    block_size = source_inputs.block_size_tokens
    if pool.kv_cache is not None:
        actual_block_size = next(tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")
        if block_size != actual_block_size:
            raise ValueError("V32 logical source block size disagrees with its pool cache")
    step, input_ids, ordering = vllm_0_30_source_model_inputs(step, embedding_token_ids=input_ids,
        prompt_lengths=None, num_speculative_tokens=num_drafts)
    requests = step.requests
    discarded_rows = tuple(discarded_by_id[request.request_id] for request in requests)
    q_lens = tuple(request.scheduled_tokens for request in requests)
    kv_lens = tuple(request.computed_tokens + request.scheduled_tokens for request in requests)
    phases = tuple(ordering["model_attention_phases"])
    rows, batch = sum(q_lens), len(q_lens)
    verified = tuple(request.mtp_verified_tokens for request in requests)
    accepted = tuple(request.mtp_accepted_tokens for request in requests)
    retained = _mtp_retained_last_rows(requests)
    positions = tuple(position for request in requests for position in range(request.computed_tokens,
        request.computed_tokens + request.scheduled_tokens))
    last_positions = tuple(positions[index] + num_drafts - 1 for index in retained)
    if max((*kv_lens, *(position + 1 for position in last_positions if drafter_executes))) > source_max_model_len:
        raise NotImplementedError("V32 cap-exceeded padded positions are outside the complete source domain")
    if (tuple(len(ids) for ids in drafts) != (rows, *((batch,) * (num_drafts - 1)))
            or any(token >= model.vocab_size for ids in drafts for token in ids)):
        raise ValueError("V32 scenario must supply independent in-vocabulary IDs for every row of every draft pass")
    if drafter_executes and any(drafts[0][index] != input_ids[index + 1] for index in range(rows - 1) if index not in retained):
        raise ValueError("V32 first-pass draft IDs disagree with the source global shift at non-sampled rows")
    # Disjoint logical per-request page arenas; no native block IDs or
    # KVAllocation are invented. A final recurrent write may extend a page.
    bases, arena_size = [], 0
    for length in kv_lens:
        bases.append(arena_size)
        arena_size += -(-(length + num_drafts - 1) // block_size) * block_size
    slots = tuple(bases[index] + request.computed_tokens + offset for index, request in enumerate(requests)
        for offset in range(request.scheduled_tokens))
    old_selected_contexts = tuple(positions[index] + 1 for index in retained)
    assignments = pool.parallel_plan.layer_to_pipeline_stage
    pp = config.pipeline_parallel_size
    if (len(assignments) != model.num_hidden_layers or assignments != tuple(sorted(assignments))
            or set(assignments) != set(range(pp))):
        raise ValueError("V32 padded pipeline requires complete ordered nonempty target layer stages")
    ranges = tuple((assignments.index(stage), assignments.index(stage) + assignments.count(stage)) for stage in range(pp))
    from .traced_models import _DENSE_METADATA_SOURCE_CONTRACT
    reusable = (not _reuse_prefix and source_program_cache is not None
        and source_program_cache.glm_mtp_structural_reuse and _replica_context is None
        and pool.role == 'decode' and config.tensor_parallel_size == 8 and pp == 1 and config.expert_parallel_size == 1
        and model.model_id == 'zai-org/GLM-5.3' and model.weight_dtype == 'fp8' and num_drafts == 5
        and all(r.phase == 'decode' and r.scheduled_tokens == 6 and r.mtp_verified_tokens == 5
            and r.mtp_proposer_tokens == 5 and r.mtp_next_proposed_tokens == 5 and not r.mtp_padding_tokens
            # Rejected drafts leave overlap with the native high watermark;
            # an ordinary q6 round still executes the same source branch.
            # Preemption and full-row replay retain the complete-source path.
            and not r.resumed_from_preemption and r.recomputed_tokens <= r.mtp_verified_tokens
            and not r.prefix_replay_tokens for r in requests)
        and min(kv_lens) > model.index_topk + 5 and _DENSE_METADATA_SOURCE_CONTRACT.unchanged())
    key = ('glm-mtp5-fixed-source-v1',model,id(pool),q_lens,source_profile,source_max_model_len,
        max_batched_tokens,max_requests,workspace_initial_capacity_bytes,selection_source,
        prefill_attention_source,indexer_max_logits_mb,source_inputs.parallel_drafting,
        source_inputs.use_local_argmax_reduction,source_inputs.probabilistic,
        source_inputs.block_size_tokens,source_inputs.discarded_rows) if reusable else None
    program = source_program_cache.lookup(key) if reusable else None
    if program is not None and not program.source_unchanged():
        program = None
        reusable = False
    namespace = step.step_id if program is None else program.source.namespace
    prefix_only = _reuse_prefix or program is not None
    tp = config.tensor_parallel_size
    stages = (tuple(_replica_context.traces[stage * tp:(stage + 1) * tp] for stage in range(pp)) if _replica_context is not None else
        tuple(tuple(ForwardTrace(rank=rank, namespace=f"{namespace}:rank-{rank}")
            for rank in pool.physical_ranks_for_pipeline_stage(stage)) for stage in range(pp)))
    groups = (tuple(_replica_context.target_contexts[stage * tp].tensor_group for stage in range(pp))
        if _replica_context is not None else tuple(None if len(traces) == 1 else ForwardCollectiveGroup(traces,
            namespace=namespace if pp == 1 else f"{namespace}:stage-{stage}")
            for stage, traces in enumerate(stages)))
    collective = None if config.tensor_parallel_size == 1 else pool.collective_for("all_reduce")
    gather = None if config.tensor_parallel_size == 1 else pool.collective_for("all_gather")
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = (_source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
        if _replica_context is None else _replica_context.target_resolutions)
    draft_resolutions = (tuple(expert_routing.resolve(step_id=step.step_id, layer_index=None,
        moe_invocation_id=f"{step.step_id}:mtp-pass-{index}:moe", schedule_step_sha256=schedule_step_sha256(step),
        tokens=count, num_experts=model.num_experts, topk=model.num_experts_per_tok,
        model_config_sha256=model.config_sha256, pool_name=pool.name, tensor_parallel_size=config.tensor_parallel_size,
        expert_parallel_size=config.expert_parallel_size,
        expert_rank_group=tuple(trace.rank for trace in stages[-1]) if config.enable_expert_parallel else (stages[-1][0].rank,),
        expert_placement_strategy="linear",
        moe_layer_ordinal=sum(model.layer_uses_moe(layer) for layer in range(model.num_hidden_layers)),
        assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
            if expert_routing.source_kind == "explicit_assumption" else None))
        for index, count in enumerate((rows, *((batch,) * (num_drafts - 1)))))
        if _replica_context is None else _replica_context.draft_resolutions) if drafter_executes else ()

    if program is not None:
        prefix=program.parameter_prefix(input_ids=input_ids,drafts=drafts,ordering=ordering,
            kv_lens=kv_lens,positions=positions,retained=retained,arena_size=arena_size,
            verified=verified,accepted=accepted)
        if prefix is not None:
            bound=_GLMMTP5SourceProgram.bind(program,prefix,resolutions,invocations,draft_resolutions,step)
            if bound is not None:
                return bound

    def target_network(trace, capacity, stage):
        return DeepseekV32Model(trace, model, pool.name, groups[stage], collective, token_ids=input_ids,
            q_lens=q_lens, kv_lens=kv_lens, request_phases=phases, slot_mapping=slots,
            cache_capacity_tokens=arena_size, source_max_model_len=source_max_model_len,
            max_batched_tokens=max_batched_tokens, resolutions=resolutions, invocation_ids=invocations,
            workspace_initial_capacity_bytes=capacity, selection_source=selection_source,
            source_profile=source_profile, prefill_attention_source=prefill_attention_source,
            model_input_order=ordering, num_speculative_tokens=num_drafts, expert_parallel_size=expert_size,
            layer_range=None if pp == 1 else ranges[stage],
            pcp_context=None if _replica_context is None else _replica_context.target_contexts[pool.ranks.index(trace.rank)])

    def rank_forward(trace, capacity, stage, incoming):
        group = groups[stage]
        if stage != pp - 1:
            network = target_network(trace, capacity, stage)
            ids = trace.tensor("input_ids", (rows,), "int32") if network.first_stage else None
            pos = trace.tensor("positions", (rows,), "int64")
            return network(ids, pos, intermediate_tensors=incoming)
        ops = MTPSourceOps(trace, "mtp-source", pool.name, model_execution_version="vllm_0_30_0",
            source_geometry_program="vllm_0_30_torch_2_13_cuda_source")
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        selected_rows, target_values, bonus_values = [], [], []
        offset, logits_offset = 0, 0
        for query, count in zip(q_lens, verified, strict=True):
            selected_rows.extend(range(offset + query - count - 1, offset + query))
            target_values.extend(range(logits_offset, logits_offset + count))
            bonus_values.append(logits_offset + count)
            offset += query
            logits_offset += count + 1
        indices = ops.h2d(trace.tensor("target.logits-indices", (len(selected_rows),), "int32"))
        prefix = None
        if any(verified):
            prefix = ops.h2d(trace.tensor("target.cu-draft-tokens", (batch,), "int32"))
            sampled_prefix = ops.h2d(trace.tensor("target.cu-sampled-tokens", (batch,), "int32"))
            target_indices = ops.h2d(trace.tensor("target.target-indices", (sum(verified),), "int32"))
            bonus_indices = ops.h2d(trace.tensor("target.bonus-indices", (batch,), "int32"))
            selected_ids = ops.gather(ids, indices, selected_indices=tuple(selected_rows))
            shifted = ops.unary(target_indices, expression="add", scalar=1)
            previous_draft = ops.gather(selected_ids, shifted,
                selected_indices=tuple(index + 1 for index in target_values))
        if any(request.mtp_padding_tokens for request in requests):
            # _calc_spec_decode_metadata has already copied raw -1 drafts;
            # _preprocess clamps the shared target input buffer before lookup.
            ops.unary(ids, inplace=True, expression="clamp_min", scalar=0)
        network = target_network(trace, capacity, stage)
        if prefix_only:
            network.layers = network.layers[:4]
        hidden = network(ids if network.first_stage else None, pos, intermediate_tensors=incoming)
        selected_hidden = ops.gather(hidden, indices, selected_indices=tuple(selected_rows))
        head = (SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size, model.vocab_size,
            kind=WorkloadKind.LM_HEAD) if group is None else SymbolicTPLogitsProcessor(trace, "lm-head", pool.name,
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather))
        logits = head(selected_hidden)[0] if group is None else head(selected_hidden)
        if prefix is None:
            sampled = SymbolicGreedy(trace, "target-sampler", pool.name)(logits)
        else:
            sampled = sample_padded_raw_logits(ops, logits, prefix=prefix, previous_draft=previous_draft,
                target_indices=target_indices, bonus_indices=bonus_indices, verified=verified, accepted=accepted,
                target_values=tuple(target_values), bonus_values=tuple(bonus_values))
        if not drafter_executes:
            # GPUModelRunner checks max scheduled sequence + K before
            # propose_draft_token_ids; target/rejection still ran in full.
            return
        backup = ops.h2d(trace.tensor("mtp.backup-next-token", (max_requests,), "int32", persistent=True).narrow(0, 0, batch))
        discard = trace.tensor("parent.discard-mask", (max_requests,), "bool", persistent=True).narrow(0, 0, batch)
        query_start = trace.tensor("parent.query-start", (max_requests + 1,), "int32", persistent=True).narrow(0, 0, batch + 1)
        next_ids, sample_indices, rejected = prepare_padded_drafter(ops, sampled, discard, backup, query_start,
            vocab_size=model.vocab_size, valid_sampled_counts=tuple(0 if discarded else count + 1
                for count, discarded in zip(accepted, discarded_rows, strict=True)),
            discarded_rows=discarded_rows, prefix=prefix)
        draft_ids = trace.tensor("mtp.input-ids", (max_batched_tokens,), "int32", persistent=True)
        draft_pos = trace.tensor("mtp.positions", (max_batched_tokens,), "int64", persistent=True)
        draft_hidden = trace.tensor("mtp.hidden-states", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
        unused_embeds = trace.tensor("mtp.inputs-embeds", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
        arange = trace.tensor("mtp.arange", (max(max_requests + 1, max_batched_tokens),), "int32", persistent=True)
        first_ids = draft_ids.narrow(0, 0, rows)
        first_pos = draft_pos.narrow(0, 0, rows)
        first_hidden = draft_hidden.narrow(0, 0, rows)
        if rows > 1:
            ops.copy(ids.narrow(0, 1, rows - 1), first_ids.narrow(0, 0, rows - 1))
        ops.scatter(next_ids, sample_indices, first_ids, selected_indices=retained)
        ops.copy(pos, first_pos)
        ops.copy(hidden, first_hidden)
        mtp_cache = trace.tensor("mtp.kv-cache", (arena_size, model.kv_lora_rank + model.qk_rope_head_dim), "bf16", persistent=True)
        mtp_index_cache = trace.tensor("mtp.index-cache", (arena_size, 132), "uint8", persistent=True)
        topk = network.topk_buffer
        arenas = dict(cache_capacity_tokens=arena_size, source_profile=source_profile,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            workspace=network.workspace, topk_buffer=topk, cache=mtp_cache, index_cache=mtp_index_cache,
            rotary_cache=network.rotary_cache, index_rotary_cache=network.index_rotary_cache,
            selection_source=selection_source, num_speculative_tokens=num_drafts,
            group=group, collective=collective, gather=gather, expert_parallel_size=expert_size)
        predictor = DeepseekV32MTPPredictorLayer(trace, model, pool.name, "mtp-pass-0",
            input_token_ids=drafts[0], q_lens=q_lens, kv_lens=kv_lens, request_phases=phases,
            slots=slots, slot_tensor=network.slot_mapping, indexer_type="full",
            resolution=draft_resolutions[0], invocation_id=f"{step.step_id}:mtp-pass-0:moe",
            prefill_attention_source=prefill_attention_source,
            pcp_context=None if _replica_context is None else _replica_context.draft_contexts[0][stages[stage].index(trace)], **arenas)
        post_norm, recycle = predictor(first_ids, first_pos, first_hidden, position_values=positions)
        # llm_base_proposer only toggles skip_topk and compacts the first
        # pass's selection when the draft checkpoint enables index sharing.
        if model.index_share_for_mtp_iteration:
            compacted = ops.gather(topk, sample_indices, selected_indices=retained)
            ops.copy(compacted, topk.narrow(0, 0, batch))
        sample_hidden = ops.gather(post_norm, sample_indices, selected_indices=retained)
        if num_drafts == 1:
            # llm_base_proposer.py returns before position/hidden gathers,
            # seq-len rollback, metadata updates, casts or stack allocation.
            first_draft = ops.unary(predictor.compute_logits(sample_hidden), "argmax")
            output = first_draft.view_as_shape(batch, 1)
            return
        selected_positions = ops.gather(first_pos, sample_indices, selected_indices=retained)
        recurrent_hidden = ops.gather(recycle, sample_indices, selected_indices=retained)
        first_draft = ops.unary(predictor.compute_logits(sample_hidden), "argmax")
        seq_lens = trace.tensor("parent.seq-lens", (max_requests,), "int32", persistent=True).narrow(0, 0, batch)
        if rejected is not None:
            ops.binary(seq_lens, rejected, "subtract", inplace=True)
        blocks = trace.tensor("parent.block-table", (max_requests, -(-source_max_model_len // block_size)),
            "int32", persistent=True).narrow(0, 0, batch)
        slots_buffer = trace.tensor("mtp.slot-mapping-buffer", (max_batched_tokens,), "int64", persistent=True)
        out_pos, out_slots = draft_pos.narrow(0, 0, batch), slots_buffer.narrow(0, 0, batch)
        draft_outputs = [first_draft]
        for draft_index in range(1, num_drafts):
            next_id_values = ops.unary(draft_outputs[-1], "cast_ids")
            ops.step_metadata(selected_positions, blocks, seq_lens, out_positions=out_pos, out_slots=out_slots,
                block_size=block_size, max_model_len=source_max_model_len)
            ops.copy(next_id_values, draft_ids.narrow(0, 0, batch))
            ops.copy(recurrent_hidden, draft_hidden.narrow(0, 0, batch))
            current_positions = tuple(positions[index] + draft_index for index in retained)
            current_slots = tuple(base + position for base, position in zip(bases, current_positions, strict=True))
            recurrent = DeepseekV32MTPPredictorLayer(trace, model, pool.name, f"mtp-pass-{draft_index}",
                input_token_ids=drafts[draft_index], q_lens=(1,) * batch,
                kv_lens=tuple(position + 1 for position in current_positions),
                request_phases=("decode",) * batch, slots=current_slots, slot_tensor=out_slots,
                indexer_type="shared" if model.index_share_for_mtp_iteration else "full",
                resolution=draft_resolutions[draft_index], invocation_id=f"{step.step_id}:mtp-pass-{draft_index}:moe",
                prefill_attention_source=None, input_norm=predictor.input_norm,
                reused_selection_kv_lens=old_selected_contexts if model.index_share_for_mtp_iteration else None,
                pcp_context=None if _replica_context is None else _replica_context.draft_contexts[draft_index][stages[stage].index(trace)], **arenas)
            final_hidden, recurrent_hidden = recurrent(draft_ids.narrow(0, 0, batch), out_pos,
                draft_hidden.narrow(0, 0, batch), position_values=current_positions)
            draft_outputs.append(ops.unary(recurrent.compute_logits(final_hidden), "argmax"))
            selected_positions = out_pos
        output = ops.stack(tuple(draft_outputs))

    if pp == 1:
        for trace, capacity in zip(stages[0], workspace_initial_capacity_bytes, strict=True):
            rank_forward(trace, capacity, 0, None)
        source = stages[0][0] if groups[0] is None else groups[0]
    else:
        namespace = (step.step_id if _replica_context is None else
            f"{step.step_id}:dp{_replica_context.target_contexts[0].data_parallel_rank}:pipeline")
        source = ForwardPipelineGroup(tuple(trace for traces in stages for trace in traces), namespace=namespace,
            collective_groups=tuple(group for group in groups if group is not None))
        source.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
            "pipeline_layer_partition": tuple(end - first for first, end in ranges),
            "normal_capacity_accepted": False, "actual_provider_bound": False}
        _run_vllm030_pipeline_stages(source, pool, rows=rows, hidden_size=model.hidden_size,
            max_batched_tokens=max_batched_tokens, run_stage=lambda trace, stage, incoming:
                rank_forward(trace, workspace_initial_capacity_bytes[pool.ranks.index(trace.rank)], stage, incoming))
    for stage, traces in enumerate(stages):
        attrs = {"glm_parallel_source": True, "source_projection_only": True,
            "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile}
        if pp > 1:
            attrs |= {"source_pipeline_stage": stage, "pipeline_layer_partition": tuple(end - first for first, end in ranges)}
        rank_local_fp8_ep = model.weight_dtype == "fp8" and config.enable_expert_parallel and _replica_context is None
        if rank_local_fp8_ep:
            attrs |= {"source_glm_rank_local_fp8_mtp_ep": True}
        for trace in traces:
            trace._nodes = {name: node.with_attributes({**node.attributes, **attrs,
                **(glm_fp8_rank_local_expert_source_metadata() if rank_local_fp8_ep and node.kind is WorkloadKind.FUSED_MOE else {})})
                for name, node in trace._nodes.items()}
        if groups[stage] is not None:
            groups[stage]._contracts = {name: node.with_attributes({**node.attributes, **attrs})
                for name, node in groups[stage]._contracts.items()}
    if model.weight_dtype == "fp8" and config.enable_expert_parallel and _replica_context is None:
        source.source_scope = {"source_projection_only": True, "model_execution_version": "vllm_0_30_0",
            "model_source_profile": source_profile, "source_glm_rank_local_fp8_mtp_ep": True,
            "pipeline_layer_partition": tuple(end - first for first, end in ranges),
            "normal_capacity_accepted": False, "actual_provider_bound": False}
    arguments = dict(input_ids=input_ids,source_profile=source_profile,source_max_model_len=source_max_model_len,
        max_batched_tokens=max_batched_tokens,max_requests=max_requests,
        workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,selection_source=selection_source,
        prefill_attention_source=prefill_attention_source,source_inputs=source_inputs,
        indexer_max_logits_mb=indexer_max_logits_mb,legacy_skip_topk_by_data_rank=legacy_skip_topk_by_data_rank)
    if program is not None:
        bound = _GLMMTP5SourceProgram.bind(program,source,resolutions,invocations,draft_resolutions,step)
        if bound is not None:
            return bound
        return trace_glm_dsa_mtp_forward(model,step,pool,expert_routing,**arguments)
    if reusable:
        prefix = trace_glm_dsa_mtp_forward(model,step,pool,expert_routing,_reuse_prefix=True,**arguments)
        program = _GLMMTP5SourceProgram.create(source,prefix,model,pool)
        source_program_cache.store(key,program)
        return program.snapshot()
    return source
