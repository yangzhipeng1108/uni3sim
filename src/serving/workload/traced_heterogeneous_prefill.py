"""Source-only Flash TP8 prompt forward for the declared BW1100 scenario.

This isolated prefill graph declares mathematical tensors and RoCE payloads.
Native scratch, provider execution, PD transport and request timing remain
unbound. Its greedy output is symbolic metadata, not request completion.
"""
from dataclasses import replace

from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep, schedule_step_sha256
from .runtime_inputs import ExpertRoutingInput
from .traced_forward import ForwardCollectiveGroup, ForwardTrace
from .traced_heterogeneous_moe import COMPONENT, _tag
from .traced_models import DeepseekV4DecoderLayer, DeepseekV4MoE
from .traced_operators import (SymbolicGreedy, SymbolicLogitsRowSelect, SymbolicMHC,
    SymbolicRMSNorm, SymbolicTPLogitsProcessor, SymbolicVocabParallelEmbedding)
from .workload_graph import WorkloadGraph


def trace_flash_heterogeneous_prefill(model, *, input_ids, root_seed=0, layer_count=None, state_observer=None,
        rank_start=0, max_context_tokens=None, request_q_lens=None, namespace=None,
        gpu_hardware_id="hygon_bw1100"):
    """Trace fresh explicit prompts on all eight TP ranks.

    The unchanged checkpoint supplies the complete layer stack by default;
    ``layer_count`` selects a prefix solely for bounded smoke runs. One seeded
    canonical routing resolution per layer is reused on every TP rank. TP
    shards matrix widths, while every rank retains all prompt token rows.
    Returns ``(graph, traces, deployment, symbolic_greedy_outputs)``.
    ``request_q_lens`` partitions the flat IDs into independent fresh prompts;
    each request owns a context-sized, disjoint cache region. An optional
    namespace distinguishes separate prefill groups in a combined graph.
    An optional ``state_observer(trace, layers)`` can retain the real layer
    owners for cache-port inspection; it does not perform PD transfer.
    """
    if state_observer is not None and not callable(state_observer):
        raise TypeError("prefill state observer must be callable")
    if (model.model_type != "deepseek_v4" or model.hidden_size != 4096
            or model.num_hidden_layers != 43 or model.num_experts != 256
            or model.num_experts_per_tok != 6 or model.expert_dtype != "fp4"
            or model.weight_dtype != "fp8" or model.compute_dtype != "bf16"):
        raise NotImplementedError("prefill requires the unchanged DeepSeek V4 Flash checkpoint geometry")
    if (not isinstance(input_ids, tuple) or not input_ids
            or any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_ids)):
        raise ValueError("prefill requires explicit valid prompt IDs within the checkpoint context")
    q_lens = (len(input_ids),) if request_q_lens is None else request_q_lens
    if (not isinstance(q_lens, tuple) or not q_lens
            or any(type(length) is not int or not 1 <= length <= model.max_position_embeddings for length in q_lens)
            or sum(q_lens) != len(input_ids)):
        raise ValueError("prefill request lengths must partition prompt IDs within the checkpoint context")
    if namespace is not None and (not isinstance(namespace, str) or not namespace.strip()):
        raise ValueError("prefill namespace must be a nonempty string")
    layers = model.num_hidden_layers if layer_count is None else layer_count
    if type(layers) is not int or not 1 <= layers <= model.num_hidden_layers:
        raise ValueError("prefill layer count must select a nonempty unchanged checkpoint prefix")
    if type(rank_start) is not int or rank_start < 0:
        raise ValueError("prefill placement requires a nonnegative physical rank start")
    context = max(128, max(q_lens)) if max_context_tokens is None else max_context_tokens
    if type(context) is not int or not max(q_lens) <= context <= model.max_position_embeddings:
        raise ValueError("prefill context capacity must cover the prompt within the checkpoint limit")
    routing = ExpertRoutingInput.seeded_assumption("user_uniform_flash_gpu_fpga", root_seed)
    collectives = tuple(CollectiveConfig(operation, "ring", "roce", "ring",
        "heterogeneous_roce_switch0", "analytical_roofline_v1")
        for operation in ("all_reduce", "all_gather"))
    deployment = build_deployment_plan(DeploymentConfig("colocated", (
        WorkerPoolConfig("prefill_gpu", "prefill", gpu_hardware_id,
            parallelism=ParallelConfig(tensor_parallel_size=8), ranks=tuple(range(rank_start, rank_start + 8)),
            collectives=collectives),)), num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    pool = deployment.worker_pools["prefill_gpu"]
    rows = len(input_ids)
    step_id = "heterogeneous-prefill" if namespace is None else namespace
    requests = ((ScheduledRequest("prompt", "prefill", rows, 0),) if request_q_lens is None else
        tuple(ScheduledRequest(f"prompt-{index}", "prefill", length, 0) for index, length in enumerate(q_lens)))
    step = ScheduleStep(step_id, pool.name, requests, attention_kv_layout="mla")
    step_digest = schedule_step_sha256(step)
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(layers))
    resolutions = tuple(routing.resolve(step_id=step.step_id, layer_index=layer,
        moe_invocation_id=invocations[layer], schedule_step_sha256=step_digest,
        tokens=rows, num_experts=model.num_experts, topk=model.num_experts_per_tok,
        model_config_sha256=model.config_sha256, pool_name=pool.name,
        tensor_parallel_size=8, expert_parallel_size=1, expert_rank_group=(pool.ranks[0],),
        expert_placement_strategy="linear", moe_layer_ordinal=model.moe_layer_ordinal(layer))
        for layer in range(layers))
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id,
        participant_nic_ids=tuple("prefill-gpu-nic:" + str(rank // 2) for rank in pool.ranks),
        # An explicit logical simulation route over the common switch:
        # every neighbor uses a different NIC. No undocumented intra-NIC
        # GPU path, library ring selection or hardware measurement is used.
        ring_order=pool.ranks[::2] + pool.ranks[1::2])
    collective, gather = pool.collective_for("all_reduce"), pool.collective_for("all_gather")
    request_capacity = max(128, -(-context // 128) * 128)
    capacity = request_capacity * len(q_lens)
    slots = tuple(index * request_capacity + offset
        for index, length in enumerate(q_lens) for offset in range(length))
    source_max_model_len = max(128, context)
    linear_source_profile = "vllm_0_30_logical_group128_fp32"
    outputs, models = [], []
    for trace in traces:
        ids = trace.tensor("input-ids", (rows,), "int32")
        positions = trace.tensor("positions", (rows,), "int64")
        hidden = SymbolicVocabParallelEmbedding(trace, "embedding", pool.name, model.hidden_size,
            model.vocab_size, group, collective, token_ids=input_ids)(ids)
        topk = trace.tensor("model.topk-buffer", (rows, model.index_topk), "int32", persistent=True)
        rope_kinds = tuple(dict.fromkeys("swa" if ratio <= 1 else "compressed"
            for ratio in model.compress_ratios[:layers]))
        ropes = {kind: trace.tensor("model.rope-" + kind,
            (source_max_model_len, model.qk_rope_head_dim), "fp32", persistent=True) for kind in rope_kinds}
        residual = post_mix = res_mix = None
        decoder_layers = []
        models.append(decoder_layers)
        for layer in range(layers):
            prefix = f"layer-{layer}"
            ffn = DeepseekV4MoE(trace, model, pool.name, prefix + ":ffn", group, collective,
                layer_index=layer, resolution=resolutions[layer], invocation_id=invocations[layer],
                linear_source_profile=linear_source_profile, source_multiprocessors=None, workspace=None)
            decoder_layers.append(DeepseekV4DecoderLayer(trace, model, pool.name, prefix, group, collective,
                ropes["swa" if model.compress_ratios[layer] <= 1 else "compressed"], topk,
                layer_index=layer, q_lens=q_lens, kv_lens=q_lens, request_phases=("prefill",) * len(q_lens),
                slot_mapping=slots, cache_capacity_tokens=capacity,
                source_max_model_len=source_max_model_len, linear_source_profile=linear_source_profile,
                source_use_deep_gemm=None, hardware=None, resolution=resolutions[layer],
                invocation_id=invocations[layer], indexer_selection_source="declared_causal_topk",
                source_max_batched_tokens=rows, ffn=ffn, mhc_workspace_scope="logical_math_only"))
        for decoder in decoder_layers:
            hidden, residual, post_mix, res_mix = decoder(hidden, positions, ids, post_mix, res_mix, residual)
        hidden = SymbolicMHC(trace, "final-mhc-post", pool.name, model, None,
            variant="mhc_post", source_use_deep_gemm=None, workspace_scope="logical_math_only")(
                hidden, residual, post_mix, res_mix)
        hidden = SymbolicMHC(trace, "hc-head", pool.name, model, None,
            variant="hc_head", source_use_deep_gemm=None, workspace_scope="logical_math_only")(hidden)
        hidden = SymbolicRMSNorm(trace, "final-norm", pool.name, model.hidden_size,
            model.target_semantics.rms_norm_eps)(hidden)
        indices = trace.tensor("logits-indices", (len(q_lens),), "int32")
        endpoint = 0
        selected_indices = []
        for length in q_lens:
            endpoint += length
            selected_indices.append(endpoint - 1)
        selected = SymbolicLogitsRowSelect(trace, "logits-select", pool.name)(hidden,
            indices, selected_indices=tuple(selected_indices))
        logits = SymbolicTPLogitsProcessor(trace, "lm-head", pool.name, hidden_size=model.hidden_size,
            vocab_size=model.vocab_size, group=group, config=gather)(selected)
        outputs.append(SymbolicGreedy(trace, "target-sampler", pool.name)(logits))
        if state_observer is not None:
            state_observer(trace, tuple(decoder_layers))
    joined = group.graph
    for trace in traces:
        _tag(trace)
    # Collective contracts are materialized at join time and therefore need
    # the same explicit scenario tag as the rank-local operator nodes.
    graph = WorkloadGraph(tuple(replace(node, attributes={**node.attributes,
        "heterogeneous_operator_component": (node.attributes["source_component"]
            if "source_component" in node.attributes else None),
        "source_component": COMPONENT, "heterogeneous_forward_scope": "isolated_prefill",
        "symbolic_workspace_complete": False}) for node in joined.nodes), joined.edges)
    graph.topological_order()
    return graph, traces, deployment, tuple(outputs)
