"""Pure BW1100 D-only virtual source with declared, already-prefilled KV.

No prompt forward, PD transfer, hardware launch or numerical token value is
inferred. Each call advances the same cache owners and symbolic sampled token.
The caller must price a step before retiring its transient graph.
"""
from dataclasses import replace
from collections import Counter
from hashlib import sha256
import json
from math import prod

from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep, schedule_step_sha256
from .runtime_inputs import ExpertRoutingInput, _shared_routing_group_id, seeded_expert_routes
from .traced_forward import ForwardCollectiveGroup, ForwardLocalGroup, ForwardTrace
from .traced_heterogeneous_moe import COMPONENT, _tag, _mlp
from .traced_models import DeepseekV4DecoderLayer, DeepseekV4MoE
from .traced_operators import (SymbolicGreedy, SymbolicLogitsRowSelect, SymbolicMHC,
    SymbolicRMSNorm, SymbolicTPLogitsProcessor, SymbolicEmbedding, SymbolicLinear, SymbolicV4Routing, SymbolicAdd)
from .workload_graph import WorkloadGraph, WorkloadKind, WorkloadNode, WorkloadEdge


class FlashGPUDecode:
    """One unsplit fixed batch, eight real BW identities, MTP disabled."""

    def __init__(self, model, *, concurrency, scheme="tp8", initial_context=131072,
                 max_steps=64, root_seed=0, layer_count=None):
        if (model.model_type != "deepseek_v4" or model.hidden_size != 4096
                or model.num_hidden_layers != 43 or model.num_experts != 256
                or model.num_experts_per_tok != 6 or model.moe_intermediate_size != 2048
                or model.num_shared_experts != 1 or model.num_attention_heads != 64 or model.expert_dtype != "fp4"
                or model.weight_dtype != "fp8" or model.compute_dtype != "bf16"):
            raise ValueError("GPU decode requires the unchanged Flash checkpoint")
        if scheme not in ("tp8", "dp8_ep8"):
            raise NotImplementedError("GPU decode scheme is not implemented: " + scheme)
        if type(concurrency) is not int or concurrency not in (1, 128, 192):
            raise ValueError("only C1 internal smoke and formal C128/C192 are supported")
        if scheme == "dp8_ep8" and concurrency not in (128, 192):
            raise ValueError("EP8 source requires complete C128/C192 lanes")
        if (type(initial_context) is not int or initial_context != 131072
                or type(max_steps) is not int or not 1 <= max_steps <= 64
                or initial_context + max_steps > model.max_position_embeddings
                or type(root_seed) is not int or root_seed < 0):
            raise ValueError("decode requires declared 128K context, 1..64 steps and nonnegative seed")
        layers = model.num_hidden_layers if layer_count is None else layer_count
        if type(layers) is not int or not 1 <= layers <= model.num_hidden_layers:
            raise ValueError("layer_count must select an unchanged checkpoint prefix")
        self.model, self.concurrency, self.scheme = model, concurrency, scheme
        self.initial_context, self.max_steps, self.root_seed = initial_context, max_steps, root_seed
        self.layer_count, self.step_index, self.pending = layers, 0, False
        self.stride = -(-(initial_context + max_steps) // 128) * 128
        self.local_request_ids = tuple(tuple(range(concurrency)) if scheme == "tp8" else
            tuple(range(rank, concurrency, 8)) for rank in range(8))
        local_rows = concurrency if scheme == "tp8" else concurrency // 8
        configs = tuple(CollectiveConfig(op, "ring", "roce", "ring", "flash-d-roce-switch0",
            "analytical_roofline_v1") for op in ("all_reduce", "all_gather"))
        self.deployment = build_deployment_plan(DeploymentConfig("colocated", (
            WorkerPoolConfig("decode_gpu", "colocated", "hygon_bw1100",
                parallelism=(ParallelConfig(tensor_parallel_size=8) if scheme == "tp8" else
                    ParallelConfig(data_parallel_size=8, enable_expert_parallel=True)), ranks=tuple(range(8)),
                collectives=configs if scheme == "tp8" else ()),)), num_layers=model.num_hidden_layers, num_experts=model.num_experts)
        self.pool = self.deployment.worker_pools["decode_gpu"]
        self.traces = tuple(ForwardTrace(rank=r, namespace=f"flash-d:rank-{r}") for r in self.pool.ranks)
        self.tokens = tuple(t.tensor("declared-first-input-token", (local_rows,), "int32") for t in self.traces)
        self.topks = tuple(t.tensor("topk", (local_rows, model.index_topk), "int32", persistent=True)
            for t in self.traces)
        self.ropes = tuple({kind: t.tensor("rope-" + kind, (self.stride, model.qk_rope_head_dim),
            "fp32", persistent=True) for kind in ("swa", "compressed")} for t in self.traces)
        self.caches = None
        self.layers = ()
        self.routing = ExpertRoutingInput.seeded_assumption("declared_flash_gpu_d_only", root_seed)

    def step(self):
        if self.pending or self.step_index >= self.max_steps:
            raise ValueError("price and retire the current step before advancing an unfinished decode")
        model, rows, index = self.model, self.concurrency, self.step_index
        kv = self.initial_context + index + 1
        step_id = f"flash-d-step-{index}"
        requests = tuple(ScheduledRequest(f"request-{r}", "decode", 1, kv - 1) for r in range(rows))
        schedule = ScheduleStep(step_id, self.pool.name, requests, attention_kv_layout="mla")
        digest = schedule_step_sha256(schedule)
        invocations = tuple(f"{step_id}:layer-{layer}:moe" for layer in range(self.layer_count))
        # The routing key is identical across parallel scenarios. Placement is
        # metadata, not an extra stochastic input to the token routes.
        resolutions = tuple(self.routing.resolve(step_id=step_id, layer_index=layer,
            moe_invocation_id=invocations[layer], schedule_step_sha256=digest, tokens=rows,
            num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name=self.pool.name,
            tensor_parallel_size=8, expert_parallel_size=1, expert_rank_group=(0,),
            expert_placement_strategy="linear", moe_layer_ordinal=model.moe_layer_ordinal(layer))
            for layer in range(self.layer_count))
        routes = self._global_routes(step_id, resolutions)
        if self.scheme == "dp8_ep8":
            return self._step_ep(resolutions, invocations, routes)
        group = ForwardCollectiveGroup(self.traces, namespace=step_id,
            participant_nic_ids=tuple("flash-d-nic:" + str(r // 2) for r in self.pool.ranks),
            ring_order=(0, 2, 4, 6, 1, 3, 5, 7))
        collective, gather = self.pool.collective_for("all_reduce"), self.pool.collective_for("all_gather")
        outputs, rank_layers = [], []
        for rank, trace in enumerate(self.traces):
            prefix = f"step-{index}"
            ids = self.tokens[rank].view_as_shape(rows)
            positions = trace.tensor(prefix + ":positions", (rows,), "int64")
            # Numerical sampled IDs are unknown. The partial table lookup
            # therefore charges the public all-row operand envelope on each
            # shard, including all partial-output stores. This is an explicit
            # conservative byte assumption, not fabricated token ownership.
            node = WorkloadNode(trace.unique_name(prefix + ":embedding"), WorkloadKind.GATHER,
                self.pool.name, {"elements": rows * model.hidden_size, "rows": rows,
                    "hidden": model.hidden_size}, "bf16", "embedding", rank=trace.rank,
                attributes={"gather_factor": 1, "index_count": rows, "index_bytes": 4,
                    "scattered": True, "materialized_output": True,
                    "sampled_token_values": "unknown", "embedding_payload_scope": "all_row_upper_bound_per_tp_shard"})
            partial = trace.call(node, (ids,), shape=(rows, model.hidden_size), dtype="bf16")
            hidden = group.call(prefix + ":embedding-all-reduce", partial,
                pool_name=self.pool.name, config=collective, inplace=False)
            layers = []
            for layer in range(self.layer_count):
                name = prefix + f":layer-{layer}"
                ffn = DeepseekV4MoE(trace, model, self.pool.name, name + ":ffn", group, collective,
                    layer_index=layer, resolution=resolutions[layer], invocation_id=invocations[layer],
                    linear_source_profile="vllm_0_30_logical_group128_fp32", source_multiprocessors=None)
                layers.append(DeepseekV4DecoderLayer(trace, model, self.pool.name, name, group, collective,
                    self.ropes[rank]["swa" if model.compress_ratios[layer] <= 1 else "compressed"],
                    self.topks[rank], layer_index=layer, q_lens=(1,) * rows, kv_lens=(kv,) * rows,
                    request_phases=("decode",) * rows, slot_mapping=tuple(r * self.stride + kv - 1 for r in range(rows)),
                    cache_capacity_tokens=self.stride * rows, source_max_model_len=self.stride,
                    linear_source_profile="vllm_0_30_logical_group128_fp32", source_use_deep_gemm=None,
                    hardware=None, resolution=resolutions[layer], invocation_id=invocations[layer],
                    indexer_selection_source="declared_causal_topk", source_max_batched_tokens=rows,
                    ffn=ffn, mhc_workspace_scope="logical_math_only",
                    cache_tensors=None if self.caches is None else self.caches[rank][layer]))
            residual = post_mix = res_mix = None
            for decoder in layers:
                hidden, residual, post_mix, res_mix = decoder(hidden, positions, ids, post_mix, res_mix, residual)
            hidden = SymbolicMHC(trace, prefix + ":final-mhc-post", self.pool.name, model, None,
                variant="mhc_post", source_use_deep_gemm=None, workspace_scope="logical_math_only")(
                    hidden, residual, post_mix, res_mix)
            hidden = SymbolicMHC(trace, prefix + ":hc-head", self.pool.name, model, None,
                variant="hc_head", source_use_deep_gemm=None, workspace_scope="logical_math_only")(hidden)
            hidden = SymbolicRMSNorm(trace, prefix + ":final-norm", self.pool.name, model.hidden_size,
                model.target_semantics.rms_norm_eps)(hidden)
            indices = trace.tensor(prefix + ":logits-indices", (rows,), "int32")
            selected = SymbolicLogitsRowSelect(trace, prefix + ":logits-select", self.pool.name)(
                hidden, indices, selected_indices=tuple(range(rows)))
            logits = SymbolicTPLogitsProcessor(trace, prefix + ":lm-head", self.pool.name,
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather)(selected)
            outputs.append(SymbolicGreedy(trace, prefix + ":sampler", self.pool.name)(logits))
            rank_layers.append(tuple(layers))
        self.layers = tuple(rank_layers)
        self.caches = tuple(tuple(d.attn.cache_tensors for d in layers) for layers in self.layers)
        self.tokens = tuple(outputs)
        joined = group.graph
        return self._complete_step(joined, resolutions, routes)

    def _global_routes(self, step_id, resolutions):
        # Reuse the routing owner's canonical key and draws. It is the
        # common global request-order assumption, not actual TP placement.
        model, cache, result = self.model, {}, []
        for layer, resolution in enumerate(resolutions):
            key = _shared_routing_group_id(step_id=step_id, layer_index=layer,
                tokens=self.concurrency, num_experts=model.num_experts, topk=model.num_experts_per_tok,
                model_config_sha256=model.config_sha256, pool_name=self.pool.name,
                tensor_parallel_size=8, expert_parallel_size=1, expert_rank_group=(0,),
                expert_placement_strategy="linear")
            if key not in cache:
                cache[key] = seeded_expert_routes(self.concurrency, model.num_experts, model.num_experts_per_tok,
                    source_id=self.routing.source_id, root_seed=self.root_seed, moe_invocation_id=key)
            routes = cache[key]
            counts = Counter(expert for row in routes for expert in row)
            if tuple(counts[e] for e in range(model.num_experts)) != resolution.tokens_per_expert:
                raise ValueError("global route IDs disagree with their canonical histogram")
            result.append(routes)
        return tuple(result)

    def _complete_step(self, joined, resolutions, routes):
        index = self.step_index
        kv = self.initial_context + index + 1
        graph = _gpu_decode_graph(self.traces, scheme=self.scheme, joined=joined)
        graph.topological_order()
        self.pending = True
        self.step_index += 1
        return graph, {"step_index": index, "position": kv - 1, "kv_length": kv,
            "q_len": 1, "microbatch": 1, "global_histograms": tuple(r.tokens_per_expert for r in resolutions),
            "global_route_sha256": tuple(sha256(json.dumps(r, separators=(",", ":")).encode()).hexdigest()
                for r in routes), "local_request_ids": self.local_request_ids,
            "input_token_source": "declared_first_token" if index == 0 else "previous_symbolic_sampler",
            "cache_input_source": "declared_already_prefilled_KV" if index == 0 else "retained_previous_step_KV"}

    def _step_ep(self, resolutions, invocations, routes):
        """TP1 attention lanes and integer EP8 experts; no FPGA substitution.

        Gathered model rows stay rank-major, as in the all-gather/scatter
        process-group contract. Their natural request IDs remain explicit.
        The shared KernelSpec still rejects EP8 until its owner freezes the
        proposed minimal extension; this source does not bypass that gate.
        """
        model, rows, index = self.model, self.concurrency // 8, self.step_index
        kv, contexts = self.initial_context + index + 1, []
        for rank, trace in enumerate(self.traces):
            prefix, group = f"step-{index}", ForwardLocalGroup(trace)
            ids = self.tokens[rank].view_as_shape(rows)
            positions = trace.tensor(prefix + ":positions", (rows,), "int64")
            hidden = SymbolicEmbedding(trace, prefix + ":embedding", self.pool.name, model.hidden_size, "bf16",
                vocab_size=model.vocab_size)(ids)
            layers = []
            for layer in range(self.layer_count):
                layers.append(DeepseekV4DecoderLayer(trace, model, self.pool.name, prefix + f":layer-{layer}",
                    group, None, self.ropes[rank]["swa" if model.compress_ratios[layer] <= 1 else "compressed"],
                    self.topks[rank], layer_index=layer, q_lens=(1,) * rows, kv_lens=(kv,) * rows,
                    request_phases=("decode",) * rows, slot_mapping=tuple(r * self.stride + kv - 1 for r in range(rows)),
                    cache_capacity_tokens=self.stride * rows, source_max_model_len=self.stride,
                    linear_source_profile="vllm_0_30_logical_group128_fp32", source_use_deep_gemm=None,
                    hardware=None, resolution=None, invocation_id=None,
                    indexer_selection_source="declared_causal_topk", source_max_batched_tokens=rows,
                    ffn=_requires_ep_coordinator, mhc_workspace_scope="logical_math_only",
                    cache_tensors=None if self.caches is None else self.caches[rank][layer]))
            contexts.append({"trace": trace, "ids": ids, "positions": positions, "hidden": hidden,
                "layers": tuple(layers), "residual": None, "post_mix": None, "res_mix": None})
        row_requests = tuple(request for lane in self.local_request_ids for request in lane)
        for layer in range(self.layer_count):
            hidden, weights, selected, shared = [], [], [], []
            for ctx in contexts:
                trace, rank = ctx["trace"], ctx["trace"].rank
                prefix = f"step-{index}:layer-{layer}"
                x, ctx["residual"], ctx["post_mix"], ctx["res_mix"] = ctx["layers"][layer].prepare_ffn(
                    ctx["hidden"], ctx["positions"], ctx["post_mix"], ctx["res_mix"], ctx["residual"])
                logits, _ = SymbolicLinear(trace, prefix + ":router", self.pool.name,
                    model.hidden_size, model.num_experts, kind=WorkloadKind.ROUTER, output_dtype="fp32")(x)
                w, ids = SymbolicV4Routing(trace, prefix + ":route", self.pool.name,
                    model, layer_index=layer)(x, logits, ctx["ids"])
                writer = trace._writers[ids.view.allocation_id]
                trace._nodes[writer] = trace._nodes[writer].with_attributes({**trace._nodes[writer].attributes,
                    "assumed_topk_ids": tuple(routes[layer][request] for request in self.local_request_ids[rank]),
                    "gpu_decode_global_row_request_ids": self.local_request_ids[rank],
                    "routing_source_kind": "declared_seeded_assumption_not_checkpoint_values"})
                hidden.append(x)
                weights.append(w)
                selected.append(ids)
                shared.append(_mlp(trace, prefix + ":shared", x, model.moe_intermediate_size,
                    self.pool.name, weight_dtype="fp8", limit=model.swiglu_limit))
            global_x = _ring_all_gatherv(self.traces, tuple(hidden), name=prefix + ":dispatch-hidden", stage="hidden")
            global_w = _ring_all_gatherv(self.traces, tuple(weights), name=prefix + ":dispatch-weights", stage="topk_weights", after=global_x)
            global_ids = _ring_all_gatherv(self.traces, tuple(selected), name=prefix + ":dispatch-ids", stage="topk_ids", after=global_w)
            partials = []
            for rank, trace in enumerate(self.traces):
                owned = tuple(range(rank * (model.num_experts // 8), (rank + 1) * (model.num_experts // 8)))
                counts = resolutions[layer].tokens_per_expert
                local_counts = tuple(counts[e] if e in owned else 0 for e in range(model.num_experts))
                node = _gpu_node(trace, prefix + ":routed", WorkloadKind.FUSED_MOE,
                    {"tokens": self.concurrency, "experts": model.num_experts, "topk": model.num_experts_per_tok,
                        "hidden": model.hidden_size, "intermediate": model.moe_intermediate_size}, "bf16", "full",
                    {"tokens_per_expert": counts, "local_tokens_per_expert": local_counts,
                        "expert_parallel_size": 8, "expert_rank": rank, "owned_experts": owned,
                        "expert_placement_strategy": "linear", "moe_invocation_id": invocations[layer],
                        "routing_provenance": dict(resolutions[layer].routing_provenance),
                        "routed_scaling_factor": model.routed_scaling_factor, "routed_weight_dtype": "mxfp4",
                        "weight_group_size": 32, "swiglu_limit": model.swiglu_limit,
                        "gpu_decode_global_row_request_ids": row_requests,
                        "output_is_reduced": False, "output_includes_zero_local_contributions": True,
                        "symbolic_modular_arena_bound": False, "symbolic_marlin_multiprocessors": None})
                partials.append(trace.call(node, (global_x[rank], global_w[rank], global_ids[rank]),
                    shape=(self.concurrency, model.hidden_size), dtype="bf16"))
            reduced = _ring_reduce_scatterv(self.traces, tuple(partials), name=prefix + ":combine")
            for rank, ctx in enumerate(contexts):
                ctx["hidden"] = SymbolicAdd(ctx["trace"], prefix + ":shared-add", self.pool.name)(
                    reduced[rank], shared[rank])
        outputs = []
        for ctx in contexts:
            trace, hidden, prefix = ctx["trace"], ctx["hidden"], f"step-{index}"
            hidden = SymbolicMHC(trace, prefix + ":final-mhc-post", self.pool.name, model, None,
                variant="mhc_post", source_use_deep_gemm=None, workspace_scope="logical_math_only")(
                    hidden, ctx["residual"], ctx["post_mix"], ctx["res_mix"])
            hidden = SymbolicMHC(trace, prefix + ":hc-head", self.pool.name, model, None,
                variant="hc_head", source_use_deep_gemm=None, workspace_scope="logical_math_only")(hidden)
            hidden = SymbolicRMSNorm(trace, prefix + ":final-norm", self.pool.name, model.hidden_size,
                model.target_semantics.rms_norm_eps)(hidden)
            indices = trace.tensor(prefix + ":logits-indices", (rows,), "int32")
            selected = SymbolicLogitsRowSelect(trace, prefix + ":logits-select", self.pool.name)(
                hidden, indices, selected_indices=tuple(range(rows)))
            logits, _ = SymbolicLinear(trace, prefix + ":lm-head", self.pool.name, model.hidden_size,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)
            outputs.append(SymbolicGreedy(trace, prefix + ":sampler", self.pool.name)(logits))
        self.layers = tuple(ctx["layers"] for ctx in contexts)
        self.caches = tuple(tuple(d.attn.cache_tensors for d in layers) for layers in self.layers)
        self.tokens = tuple(outputs)
        return self._complete_step(None, resolutions, routes)

    def retire(self):
        """A whole-step completion barrier preserves cache/token identity.

        Only transient metadata is dropped. The evaluation retains sampler
        completion times and starts the next complete batch at this barrier.
        """
        if not self.pending:
            raise ValueError("no priced decode step is awaiting retirement")
        self.layers = ()
        retained = (*self.tokens, *self.topks, *(t for ropes in self.ropes for t in ropes.values()),
            *(t for rank in self.caches for layer in rank for _, t in layer))
        for trace in self.traces:
            live_views = {t.view.view_id for t in retained if t.trace is trace}
            storage = {name for name, count in trace._references.items() if count > 0}
            trace._nodes.clear()
            trace._edges.clear()
            trace._events.clear()
            trace._views = {n: v for n, v in trace._views.items() if n in live_views}
            trace._references = {n: c for n, c in trace._references.items() if n in storage}
            trace._allocations = {n: replace(a, allocated_event=trace._event("retain:" + n), freed_event=None)
                for n, a in trace._allocations.items() if n in storage}
            trace._writers = {n: None for n in storage}
            trace._readers = {n: set() for n in storage}
            trace._persistent_ids.intersection_update(storage)
            trace._names = {n: c for n, c in trace._names.items() if n == trace.qualify("view")}
            trace._checkpoints.clear()
            trace._collective_ids.clear()
        self.pending = False


RING = (0, 2, 4, 6, 1, 3, 5, 7)


def _requires_ep_coordinator(*args):
    raise ValueError("EP8 FFN must run through the layer-wise cross-rank coordinator")


def _gpu_node(trace, name, kind, shape, dtype, variant, attributes):
    return WorkloadNode(trace.unique_name(name), kind, "decode_gpu", shape, dtype, variant, rank=trace.rank,
        attributes={"source_component": COMPONENT, "heterogeneous_forward_scope": "gpu_decode_only",
            "gpu_decode_scheme": "dp8_ep8", "gpu_decode_microbatch": 1,
            "symbolic_workspace_complete": False, **attributes})


def _gpu_decode_graph(traces, *, scheme, joined=None):
    if scheme not in ("tp8", "dp8_ep8"):
        raise ValueError("decode graph scheme must be explicit")
    if joined is None:
        joined = WorkloadGraph(tuple(n for t in traces for n in t._nodes.values()),
            tuple(e for t in traces for e in t._capture_program_edges()))
    for trace in traces:
        _tag(trace)
    return WorkloadGraph(tuple(replace(n, attributes={**n.attributes,
        "source_component": COMPONENT, "heterogeneous_forward_scope": "gpu_decode_only",
        "gpu_decode_scheme": scheme, "gpu_decode_microbatch": 1,
        "symbolic_workspace_complete": False}) for n in joined.nodes), joined.edges)


def _ring_inputs(traces, tensors):
    if (not isinstance(traces, tuple) or len(traces) != 8 or tuple(t.rank for t in traces) != tuple(range(8))
            or not isinstance(tensors, tuple) or len(tensors) != 8):
        raise ValueError("EP ring requires exactly the eight ordered source traces and ports")
    first = tensors[0]
    for trace, tensor in zip(traces, tensors, strict=True):
        trace._check_tensor(tensor)
        if (len(tensor.shape) != 2 or not tensor.is_contiguous or tensor.shape != first.shape
                or tensor.dtype != first.dtype or tensor.dtype not in ("bf16", "fp32", "int32")):
            raise ValueError("EP ring ports require equal positive typed contiguous row segments")
    return first.shape


def _ring_send(x, destination, *, name, operation, phase, owner, stage, after=None):
    source = x.trace
    if (operation not in ("all_gatherv", "reduce_scatterv") or type(phase) is not int or not 0 <= phase < 7
            or type(owner) is not int or owner not in RING
            or stage not in ("hidden", "topk_weights", "topk_ids", "partial_output")
            or RING[(RING.index(source.rank) + 1) % 8] != destination.rank
            or source.rank // 2 == destination.rank // 2):
        raise ValueError("EP ring transfer disagrees with its explicit switch-only route")
    writer = source._writers[x.view.allocation_id]
    if writer is None:
        raise ValueError("EP ring payload must depend on a produced source port")
    predecessor = None
    if after is not None:
        source._check_tensor(after)
        predecessor = source._writers[after.view.allocation_id]
        if predecessor is None:
            raise ValueError("EP collective predecessor must have a produced completion port")
    received = destination.tensor(name + ":receive", x.shape, x.dtype)
    size = prod(x.shape) * x.view.element_bytes
    node = _gpu_node(source, name, WorkloadKind.P2P, {}, x.dtype, None,
        {"algorithm": "send_recv", "bytes": size, "direction": "point_to_point", "link": "roce",
            "topology": "point_to_point", "participants": (source.rank, destination.rank),
            "operation": "send_recv", "bytes_semantics": "payload", "source_rank": source.rank,
            "destination_rank": destination.rank, "backend": "analytical_roofline_v1",
            "resource_id": "flash-d-roce-switch0",
            "participant_nic_ids": ("flash-d-nic:" + str(source.rank // 2), "flash-d-nic:" + str(destination.rank // 2)),
            "source_view": x.view.view_id, "destination_view": received.view.view_id,
            "symbolic_tensor_inputs": (x.view.view_id,), "symbolic_tensor_outputs": (received.view.view_id,),
            "payload_shape": x.shape, "payload_dtype": x.dtype, "collective_operation": operation,
            "collective_phase": phase, "collective_segment_owner": owner, "collective_stage": stage,
            "collective_predecessor_node_id": predecessor,
            "nic_hbm_role_bytes": (size, size), "nic_hbm_service_bound": False,
            "native_provider_bound": False, "collective_software_startup_bound": False})
    identifier = source.qualify(node.node_id)
    source._nodes[identifier] = node.with_attributes(node.attributes, node_id=identifier)
    source._edges.append(WorkloadEdge(writer, identifier))
    if predecessor is not None:
        source._edges.append(WorkloadEdge(predecessor, identifier, "barrier"))
    source._readers[x.view.allocation_id].add(identifier)
    destination._writers[received.view.allocation_id] = identifier
    source._event("send:" + identifier)
    destination._event("receive:" + identifier)
    return received


def _ring_all_gatherv(traces, tensors, *, name, stage, after=()):
    """Seven source-executed ring rounds; variable operation, equal lanes here.

    The high-level operation remains all_gatherv. Each concrete segment is
    forwarded once per hop on the already declared ring; no same-NIC path
    or alternative all-to-all algorithm is inferred. Output stays rank-major.
    """
    rows, width = _ring_inputs(traces, tensors)
    expected_dtype = {"hidden": "bf16", "topk_weights": "fp32", "topk_ids": "int32"}
    if stage not in expected_dtype or tensors[0].dtype != expected_dtype[stage]:
        raise ValueError("EP gather stage has the wrong typed public payload")
    if not isinstance(after, tuple) or len(after) not in (0, 8):
        raise ValueError("EP gather predecessor must be absent or one completion port per rank")
    for rank, tensor in enumerate(after):
        traces[rank]._check_tensor(tensor)
        if traces[rank]._writers[tensor.view.allocation_id] is None:
            raise ValueError("EP gather predecessor requires an actual produced local port")
    chunks = [{rank: tensors[rank]} for rank in range(8)]
    outgoing = [[] for _ in range(8)]
    for phase in range(7):
        received = []
        for position, rank in enumerate(RING):
            owner, target = RING[(position - phase) % 8], RING[(position + 1) % 8]
            received.append((target, owner, _ring_send(chunks[rank][owner], traces[target],
                name=name + f":phase-{phase}:rank-{rank}", operation="all_gatherv", phase=phase, owner=owner, stage=stage,
                after=after[rank] if after else None)))
            outgoing[rank].append(traces[target]._writers[received[-1][2].view.allocation_id])
        for target, owner, tensor in received:
            chunks[target][owner] = tensor
    outputs = []
    request_ids = tuple(request for rank in range(8) for request in range(rank, 8 * rows, 8))
    for rank, trace in enumerate(traces):
        node = _gpu_node(trace, name + ":materialize", WorkloadKind.GATHER,
            {"elements": 8 * rows * width, "rows": 8 * rows, "hidden": width}, tensors[rank].dtype, "gather",
            {"gather_factor": 1, "index_count": 0, "index_bytes": 4, "scattered": False,
                "materialized_output": True, "gpu_decode_global_row_request_ids": request_ids,
                "collective_operation": "all_gatherv", "collective_stage": stage,
                "packing_program": "rank_order_concatenate_received_segments"})
        output = trace.call(node, tuple(chunks[rank][owner] for owner in range(8)),
            shape=(8 * rows, width), dtype=tensors[rank].dtype)
        writer = trace._writers[output.view.allocation_id]
        trace._edges.extend(WorkloadEdge(send, writer, "barrier") for send in outgoing[rank])
        trace._nodes[writer] = trace._nodes[writer].with_attributes({**trace._nodes[writer].attributes,
            "collective_outgoing_completion_ids": tuple(outgoing[rank])})
        outputs.append(output)
    return tuple(outputs)


def _ring_reduce_scatterv(traces, tensors, *, name):
    total_rows, width = _ring_inputs(traces, tensors)
    if total_rows % 8 or tensors[0].dtype != "bf16":
        raise ValueError("EP reduce-scatter requires full BF16 partials split into eight complete row segments")
    rows = total_rows // 8
    chunks = [{owner: tensors[rank].narrow(0, owner * rows, rows) for owner in range(8)} for rank in range(8)]
    outgoing = [[] for _ in range(8)]
    for phase in range(7):
        received = []
        for position, rank in enumerate(RING):
            owner, target = RING[(position - phase - 1) % 8], RING[(position + 1) % 8]
            received.append((target, owner, _ring_send(chunks[rank][owner], traces[target],
                name=name + f":phase-{phase}:rank-{rank}", operation="reduce_scatterv", phase=phase,
                owner=owner, stage="partial_output")))
            outgoing[rank].append(traces[target]._writers[received[-1][2].view.allocation_id])
        for target, owner, tensor in received:
            trace = traces[target]
            result = SymbolicAdd(trace, name + f":phase-{phase}:owner-{owner}:sum", "decode_gpu")(
                chunks[target][owner], tensor)
            writer = trace._writers[result.view.allocation_id]
            trace._nodes[writer] = trace._nodes[writer].with_attributes({**trace._nodes[writer].attributes,
                "collective_operation": "reduce_scatterv", "collective_phase": phase,
                "collective_segment_owner": owner, "collective_stage": "partial_output"})
            chunks[target][owner] = result
    outputs = tuple(chunks[rank][rank] for rank in range(8))
    for rank, trace in enumerate(traces):
        writer = trace._writers[outputs[rank].view.allocation_id]
        trace._edges.extend(WorkloadEdge(send, writer, "barrier") for send in outgoing[rank])
        trace._nodes[writer] = trace._nodes[writer].with_attributes({**trace._nodes[writer].attributes,
            "collective_outgoing_completion_ids": tuple(outgoing[rank])})
    return outputs
