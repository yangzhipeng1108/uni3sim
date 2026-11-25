"""Metadata-only Flash MoE projection for the user-declared GPU/FPGA design.

This is a heterogeneous design, not a binding to an installed vLLM provider.
The existing Flash checkpoint supplies all mathematical widths and formats.
The standalone MoE boundary and declared-KV decode model share the same
expert body; a complete Prefill/PD/Decode request remains outside this API.
"""
from dataclasses import dataclass, replace
from math import prod

from src.serving.deployment.deployment_config import DeploymentConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.parallelism.parallel_config import ParallelConfig
from .runtime_inputs import seeded_expert_routes
from .traced_forward import ForwardTrace
from .traced_operators import SymbolicLinear, SymbolicV4Routing, SymbolicSiluAndMul, SymbolicAdd
from .workload_graph import WorkloadEdge, WorkloadGraph, WorkloadKind, WorkloadNode

COMPONENT = "flash_heterogeneous_moe_mathematical_v1"
MICROBATCH_SCHEDULE = "fixed_gpu_fpga_token_microbatches_v2"


@dataclass(frozen=True)
class HeterogeneousMoETrace:
    graph: WorkloadGraph
    traces: tuple
    deployment: object
    routes: tuple
    outputs: tuple
    microbatch_token_ids: tuple


@dataclass
class FlashHeterogeneousDecodeState:
    """Live virtual cache/token owners, never tensor payloads or durations."""
    deployment: object
    traces: tuple
    cache_tensors: tuple
    token: object
    topk: object
    rope: object
    capacity: int
    max_context_tokens: int
    kv_length: int
    input_graph: object = None
    step_index: int = 0
    layer_owners: tuple = ()
    local_token_ids: tuple = ()
    gpu_rank: int = 0
    microbatch_index: int = 0
    routing_concurrency: int = 1
    cache_stride_tokens: int = 0


def _node(trace, name, kind, pool, shape, dtype, variant, attributes=None):
    return WorkloadNode(trace.unique_name(name), kind, pool, shape, dtype, variant,
        rank=trace.rank, attributes={"source_component": COMPONENT,
            **({} if attributes is None else attributes)})


def _tag(trace):
    for name, node in tuple(trace._nodes.items()):
        operator_component = (node.attributes["source_component"] if "source_component" in node.attributes else None)
        if operator_component == COMPONENT:
            continue
        updates = {
            **({"heterogeneous_operator_component": operator_component}
                if operator_component != COMPONENT else {}), "source_component": COMPONENT}
        trace._nodes[name] = (node.with_updates({}, updates, compose=True) if "_module_source_frame" in node.__dict__
            else node.with_attributes({**node.attributes, **updates}))


def _staging(trace, name, x, direction, *, output_resource):
    node = _node(trace, name, WorkloadKind.MEMCPY, "experts", {}, x.dtype, None,
        {"algorithm": "fpga_staging", "bytes": prod(x.shape) * x.view.element_bytes,
            "direction": direction, "link": "ddr"})
    out = trace.tensor(name + ":out", x.shape, x.dtype, memory_domain=output_resource)
    return trace.call(node, (x,), shape=x.shape, dtype=x.dtype, out=out)


def _transfer(x, destination, *, name, chip_trace):
    """A network-only service owned by the FPGA endpoint's declared link.

    Source/destination and physical NIC identities remain explicit. Separate
    source/destination memory services account for DDR and GPU HBM; this call
    is neither a DMA controller measurement nor GPU compute on an FPGA.
    """
    source = x.trace
    source._check_tensor(x)
    received = destination.tensor(name + ":recv", x.shape, x.dtype,
        memory_domain="buffer" if destination.rank >= 8 else "dram")
    node = _node(chip_trace, name, WorkloadKind.P2P, "experts", {}, x.dtype, None,
        {"algorithm": "send_recv", "bytes": prod(x.shape) * x.view.element_bytes,
            "direction": "point_to_point", "link": "roce", "topology": "point_to_point",
            "participants": (source.rank, destination.rank), "operation": "send_recv",
            "bytes_semantics": "payload", "source_rank": source.rank,
            "destination_rank": destination.rank, "backend": "analytical_roofline_v1",
            "resource_id": "heterogeneous_roce_switch0",
            "participant_nic_ids": tuple("gpu-nic:" + str(t.rank // 2) if t.rank < 8
                else "fpga-nic:" + str(t.rank - 8) for t in (source, destination)),
            "source_view": x.view.view_id, "destination_view": received.view.view_id})
    identifier = chip_trace.qualify(node.node_id)
    chip_trace._nodes[identifier] = node.with_attributes(node.attributes, node_id=identifier)
    writer = source._writers[x.view.allocation_id]
    if writer is None:
        raise ValueError("heterogeneous transfers require a produced source")
    chip_trace._edges.append(WorkloadEdge(writer, identifier))
    source._readers[x.view.allocation_id].add(identifier)
    destination._writers[received.view.allocation_id] = identifier
    source._event("send:" + identifier)
    destination._event("receive:" + identifier)
    return received


def _quant(trace, name, x, pool):
    rows, width = x.shape
    node = _node(trace, name, WorkloadKind.VECTOR, pool,
        {"element_count": rows * width, "rows": rows, "columns": width}, "bf16", "fp8_group_quant",
        {"group_size": 128, "input_resource": trace._allocations[x.view.allocation_id].memory_domain})
    return trace.call_outputs(node, (x,), outputs=(("values", x.shape, "fp8"),
        ("scales", (rows, width // 128), "fp32")))


def _fp8_linear(trace, name, x, scales, n, pool, *, weight_dtype):
    rows, k = x.shape
    node = _node(trace, name, WorkloadKind.LINEAR, pool,
        {"b": 1, "m": rows, "n": n, "k": k}, "fp8", "gemm",
        {"weight_dtype": weight_dtype, "weight_group_size": 32 if weight_dtype == "mxfp4" else 128,
            "has_bias": False})
    return trace.call(node, (x, scales), shape=(rows, n), dtype="bf16")


class _HeterogeneousMLPBody:
    """The unchanged source body, recorded by the common module-frame owner."""

    def __init__(self, trace, prefix, intermediate, pool, weight_dtype, limit):
        self.trace, self.prefix, self.intermediate = trace, prefix, intermediate
        self.pool, self.weight_dtype, self.limit = pool, weight_dtype, limit

    def __call__(self, x):
        trace, prefix, pool = self.trace, self.prefix, self.pool
        q, scales = _quant(trace, prefix + ":input-quant", x, pool)
        up = _fp8_linear(trace, prefix + ":w13", q, scales, 2 * self.intermediate, pool,
            weight_dtype=self.weight_dtype)
        activated = SymbolicSiluAndMul(trace, prefix + ":clamped-silu", pool, swiglu_limit=self.limit)(up)
        q2, scales2 = _quant(trace, prefix + ":activation-quant", activated, pool)
        return (_fp8_linear(trace, prefix + ":w2", q2, scales2, x.shape[-1], pool,
            weight_dtype=self.weight_dtype),)


def _mlp(trace, prefix, x, intermediate, pool, *, weight_dtype, limit):
    if "_heterogeneous_mlp_programs" not in trace.__dict__:
        trace._heterogeneous_mlp_programs = {}
    key = (intermediate, pool, weight_dtype, limit,
        _node, _node.__code__, _quant, _quant.__code__, _fp8_linear, _fp8_linear.__code__,
        SymbolicSiluAndMul, SymbolicSiluAndMul.__init__, SymbolicSiluAndMul.__init__.__code__,
        SymbolicSiluAndMul.__call__, SymbolicSiluAndMul.__call__.__code__)
    if key not in trace._heterogeneous_mlp_programs:
        trace._heterogeneous_mlp_programs[key] = trace._module_program(_HeterogeneousMLPBody)
    module = _HeterogeneousMLPBody(trace, prefix, intermediate, pool, weight_dtype, limit)
    output, = trace._call_module_frame(trace._heterogeneous_mlp_programs[key], prefix, module, (x,),
        readonly_arguments=(0,))
    return output


def _trace_flash_moe(model, traces, inputs, routes, batches, *, prefix, layer_index, root_seed):
    """Keep GPU model batches fixed; pipeline only routed FPGA token groups.

    Each chip waits for all its source-GPU contributions to one microbatch,
    packs expert-major input, computes each expert once, then returns rows.
    Distinct microbatches have no artificial barrier: NIC/DDR/HBM/MM resource
    reservations decide whether receive/compute/send can overlap.
    """
    chips = len(traces) - 8
    prepared = []
    for x, ids, local in inputs:
        trace = x.trace
        name = prefix + f":gpu-{trace.rank}"
        logits, _ = SymbolicLinear(trace, name + ":router", "decode_gpu", model.hidden_size,
            model.num_experts, kind=WorkloadKind.ROUTER, output_dtype="fp32")(x)
        weights, selected = SymbolicV4Routing(trace, name + ":top6", "decode_gpu",
            model, layer_index=layer_index)(x, logits, ids)
        route_writer = trace._writers[selected.view.allocation_id]
        route_node = trace._nodes[route_writer]
        trace._nodes[route_writer] = route_node.with_attributes({**route_node.attributes,
            "assumed_topk_ids": tuple(routes[t] for t in local),
            "routing_distribution": "uniform_without_replacement", "routing_seed": root_seed})
        shared_dtype = model.weight_dtype_for_module(
            f"model.layers.{layer_index}.mlp.shared_experts.gate_up_proj", "Linear")
        if shared_dtype != "fp8":
            raise NotImplementedError("heterogeneous shared MLP requires the checkpoint block-FP8 source")
        shared = _mlp(trace, name + ":shared", x, model.moe_intermediate_size * model.num_shared_experts,
            "decode_gpu", weight_dtype=shared_dtype, limit=model.swiglu_limit)
        prepared.append((x, local, weights, selected, shared))
    returned = [[] for _ in prepared]
    returned_assignments = [[] for _ in prepared]
    for mb, tokens in enumerate(batches):
        token_set = set(tokens)
        for chip in range(chips):
            fpga = traces[8 + chip]
            name = prefix + f":mb-{mb}:chip-{chip}"
            chunks, assignments_by_gpu = [], []
            for index, (x, local, _, selected, _) in enumerate(prepared):
                assignments = tuple(sorted(((row, expert) for row, token in enumerate(local)
                    if token in token_set for expert in routes[token]
                    if expert // (model.num_experts // chips) == chip), key=lambda pair: pair[1]))
                if not assignments:
                    continue
                trace, count = x.trace, len(assignments)
                source_name = name + f":gpu-{trace.rank}"
                pack_node = _node(trace, source_name + ":pack", WorkloadKind.GATHER, "decode_gpu",
                    {"elements": count * model.hidden_size, "rows": count, "hidden": model.hidden_size}, "bf16", "gather",
                    {"gather_factor": 1, "index_count": count, "index_bytes": 4,
                        "scattered": True, "materialized_output": True,
                        "assignment_rows_experts": assignments})
                packed = trace.call(pack_node, (x, selected), shape=(count, model.hidden_size), dtype="bf16")
                wire = _transfer(packed, fpga, name=source_name + ":dispatch", chip_trace=fpga)
                ddr = _staging(fpga, source_name + ":ddr-receive", wire, "network_to_ddr", output_resource="ddr")
                chunks.append(_staging(fpga, source_name + ":ddr-read", ddr, "ddr_to_buffer", output_resource="buffer"))
                assignments_by_gpu.append((index, assignments))
            if not chunks:
                continue
            # Actual materialized packing, including bytes, precedes the
            # expert views. Never pretend received disjoint buffers alias.
            records = tuple(sorted((expert, index, row, chunk_index, chunk_row)
                for chunk_index, (index, assignments) in enumerate(assignments_by_gpu)
                for chunk_row, (row, expert) in enumerate(assignments)))
            count = len(records)
            gather = _node(fpga, name + ":pack-experts", WorkloadKind.GATHER, "experts",
                {"elements": count * model.hidden_size, "rows": count, "hidden": model.hidden_size}, "bf16", "gather",
                {"gather_factor": 1, "index_count": 0, "index_bytes": 4,
                    "scattered": False, "materialized_output": True,
                    "expert_gpu_row_order": tuple(record[:3] for record in records), "microbatch_index": mb,
                    "packing_program": "concatenate_explicit_row_views"})
            # Received packets already have known expert-major row ranges.
            # Copy explicit views; no hidden index array or free concat.
            spans = tuple(chunks[chunk].narrow(0, row, 1) for _, _, _, chunk, row in records)
            buffered = fpga.call(gather, spans, shape=(count, model.hidden_size), dtype="bf16")
            expert_outputs, expert_ids, locations = [], sorted({record[0] for record in records}), {}
            offset = 0
            for expert in expert_ids:
                expert_records = tuple(record for record in records if record[0] == expert)
                n = len(expert_records)
                for expert_row, (_, index, row, _, _) in enumerate(expert_records):
                    locations[(index, row, expert)] = (len(expert_outputs), expert_row)
                inp = buffered.narrow(0, offset, n)
                expert_outputs.append(_mlp(fpga, name + f":expert-{expert}", inp,
                    model.moe_intermediate_size, "experts", weight_dtype="mxfp4", limit=model.swiglu_limit))
                offset += n
            # Return only this GPU's explicit row views. Controls enforce
            # the full chip/microbatch compute frontier before send.
            for index, assignments in assignments_by_gpu:
                trace = prepared[index][0].trace
                source_name, rows = name + f":gpu-{trace.rank}", len(assignments)
                pack_back = _node(fpga, source_name + ":pack-return", WorkloadKind.GATHER, "experts",
                    {"elements": rows * model.hidden_size, "rows": rows, "hidden": model.hidden_size}, "bf16", "gather",
                    {"gather_factor": 1, "index_count": 0, "index_bytes": 4,
                        "scattered": False, "materialized_output": True,
                        "packing_program": "concatenate_explicit_row_views",
                        "expert_output_ids": tuple(expert_ids), "assignment_rows_experts": assignments})
                sources = tuple(expert_outputs[locations[(index, row, expert)][0]].narrow(
                    0, locations[(index, row, expert)][1], 1) for row, expert in assignments)
                output = fpga.call(pack_back, sources, shape=(rows, model.hidden_size), dtype="bf16",
                    controls=tuple(expert_outputs))
                ddr_out = _staging(fpga, source_name + ":ddr-store", output, "hbm_to_ddr", output_resource="ddr")
                send = _staging(fpga, source_name + ":ddr-send", ddr_out, "ddr_to_network", output_resource="buffer")
                returned[index].append(_transfer(send, trace, name=source_name + ":return", chip_trace=fpga))
                returned_assignments[index].append(assignments)
    outputs = []
    for index, (x, local, weights, _, shared) in enumerate(prepared):
        trace, rows = x.trace, x.shape[0]
        name = prefix + f":gpu-{trace.rank}"
        merge = _node(trace, name + ":weighted-merge", WorkloadKind.VECTOR, "decode_gpu",
            {"elements": rows * model.hidden_size, "tokens": rows, "hidden": model.hidden_size,
                "experts": model.num_experts, "intermediate": model.moe_intermediate_size, "topk": 6},
            "bf16", "moe_sum", {"work_fraction": 1.0,
                "return_assignment_rows_experts": tuple(returned_assignments[index]), "global_token_ids": local,
                "weighted_merge_implementation_bound": False})
        routed = trace.call(merge, (*returned[index], weights), shape=x.shape, dtype="bf16")
        outputs.append(SymbolicAdd(trace, name + ":shared-add", "decode_gpu")(routed, shared))
    return tuple(outputs)


def _flash_heterogeneous_placement(model, concurrency, microbatches, fpga_cards, root_seed, layer_index,
        gpu_hardware_id="hygon_bw1100"):
    if (model.model_type != "deepseek_v4" or model.hidden_size != 4096
            or model.num_experts != 256 or model.num_experts_per_tok != 6
            or model.expert_dtype != "fp4" or model.weight_dtype != "fp8"
            or model.compute_dtype != "bf16" or model.num_hidden_layers != 43):
        raise NotImplementedError("heterogeneous boundary requires the original Flash geometry and MXFP4 weights")
    if (type(concurrency) is not int or concurrency < 1 or type(microbatches) is not int
            or microbatches not in (1, 2, 3) or type(fpga_cards) is not int or fpga_cards not in (8, 16)
            or type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers):
        raise ValueError("invalid heterogeneous Flash concurrency/microbatch/card/layer configuration")
    chips = 2 * fpga_cards
    deployment = build_deployment_plan(DeploymentConfig("colocated", (
        WorkerPoolConfig("decode_gpu", "colocated", gpu_hardware_id,
            parallelism=ParallelConfig(data_parallel_size=8), ranks=tuple(range(8))),
        WorkerPoolConfig("experts", "colocated", "intel_agmf039r47",
            parallelism=ParallelConfig(data_parallel_size=chips, enable_expert_parallel=True),
            ranks=tuple(range(8, 8 + chips))))), num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    traces = tuple(ForwardTrace(rank=r, namespace=f"heterogeneous:rank-{r}") for r in range(8 + chips))
    routes = seeded_expert_routes(concurrency, model.num_experts, 6,
        source_id="user_uniform_flash_gpu_fpga", root_seed=root_seed,
        moe_invocation_id=f"layer-{layer_index}:decode")
    batches = tuple(tuple(range(i * concurrency // min(concurrency, microbatches),
        (i + 1) * concurrency // min(concurrency, microbatches)))
        for i in range(min(concurrency, microbatches)))
    return deployment, traces, routes, batches


def trace_flash_heterogeneous_moe(model, *, concurrency=1, microbatches=1, fpga_cards=8,
        root_seed=0, layer_index=3, gpu_hardware_id="hygon_bw1100"):
    """One decode MoE boundary, DP8 GPU and chip-level FPGA EP16/32.

    Routes are uniform distinct top6 draws, *not* evaluated router scores.
    Each assignment sends one BF16 hidden row and returns one BF16 expert
    row. IDs/packet framing and controller latency are explicitly unbound;
    this initial graph represents the payload channel, not a full protocol.
    """
    deployment, traces, routes, batches = _flash_heterogeneous_placement(
        model, concurrency, microbatches, fpga_cards, root_seed, layer_index, gpu_hardware_id)
    inputs = []
    for gpu in range(8):
        local = tuple(t for t in range(concurrency) if t % 8 == gpu)
        if not local:
            continue
        trace, rows = traces[gpu], len(local)
        prefix = f"gpu-{gpu}:layer-{layer_index}"
        x = trace.tensor(prefix + ":hidden", (rows, model.hidden_size), "bf16")
        ids = trace.tensor(prefix + ":token-ids", (rows,), "int32")
        inputs.append((x, ids, local))
    outputs = _trace_flash_moe(model, traces, tuple(inputs), routes, batches,
        prefix=f"layer-{layer_index}", layer_index=layer_index, root_seed=root_seed)
    for trace in traces:
        _tag(trace)
    graph = WorkloadGraph(tuple(n for t in traces for n in t._nodes.values()),
        tuple(e for t in traces for e in t._capture_program_edges()))
    graph.topological_order()
    return HeterogeneousMoETrace(graph, traces, deployment, routes, tuple(outputs), batches)


def _requires_joint_expert_forward(hidden, input_ids):
    raise ValueError("heterogeneous decoder FFN requires the layer-wise cross-GPU expert coordinator")


def trace_flash_heterogeneous_decode_layers(model, *, concurrency=1, microbatches=1, fpga_cards=8,
        root_seed=0, layer_count=1, kv_length=1, include_model_boundary=False, state_observer=None,
        execution_state=None, execution_states=None, defer_graph=False, gpu_hardware_id="hygon_bw1100"):
    """Trace each layer across fixed DP batches before coordinating FPGA MoE.

    Only routed experts are partitioned by microbatch. Attention, router,
    shared experts, caches and sampler retain the same GPU-local shapes.
    Native attention scratch/provider remain explicitly unbound.
    """
    from .traced_forward import ForwardLocalGroup
    from .traced_models import DeepseekV4DecoderLayer
    if state_observer is not None and not callable(state_observer):
        raise TypeError("decode state observer must be callable")
    if execution_state is not None and execution_states is not None:
        raise ValueError("provide one live state or a complete tuple of DP states")
    states = ((execution_state,) if execution_state is not None else execution_states)
    if type(defer_graph) is not bool or (defer_graph and states is None):
        raise ValueError("deferred graph joining requires live execution state")
    if type(include_model_boundary) is not bool or (include_model_boundary and layer_count != model.num_hidden_layers):
        raise ValueError("full model boundary requires the complete unchanged checkpoint layer stack")
    if type(layer_count) is not int or not 1 <= layer_count <= model.num_hidden_layers:
        raise ValueError("decode layer count must belong to the unchanged checkpoint")
    if type(kv_length) is not int or not 1 <= kv_length <= model.max_position_embeddings:
        raise ValueError("decode requires an explicit positive checkpoint context length")
    # Also validates the unchanged checkpoint, EP geometry and MB domain.
    deployment, traces, _, batches = _flash_heterogeneous_placement(
        model, concurrency, microbatches, fpga_cards, root_seed, 0, gpu_hardware_id)
    if states is not None:
        if not isinstance(states, tuple) or not states:
            raise ValueError("live decode states must be a nonempty tuple")
        for state in states:
            if (not isinstance(state, FlashHeterogeneousDecodeState)
                    or concurrency != state.routing_concurrency or not include_model_boundary
                    or layer_count != model.num_hidden_layers or kv_length != state.kv_length + 1
                    or kv_length > state.max_context_tokens or len(state.traces) != 8 + 2 * fpga_cards
                    or len(state.cache_tensors) != layer_count
                    or state.deployment.worker_pools["decode_gpu"].hardware_id != gpu_hardware_id
                    or (not isinstance(state.input_graph, WorkloadGraph)
                        and not (defer_graph and state.input_graph is None))):
                raise ValueError("live heterogeneous decode requires consecutive retained-state steps and all checkpoint layers")
        deployment, traces = states[0].deployment, states[0].traces
        if any(state.traces is not traces or state.step_index != states[0].step_index for state in states):
            raise ValueError("DP decode states must share physical traces and one decode round")
        active = tuple((state.gpu_rank, state.local_token_ids if state.local_token_ids else (0,), state)
            for state in states)
        if len({gpu for gpu, _, _ in active}) != len(active):
            raise ValueError("GPU batches must not be partitioned by FPGA microbatch")
        if (sorted(token for _, local, _ in active for token in local) != list(range(concurrency))
                or any(not local or any(type(token) is not int or token % 8 != gpu for token in local)
                    for gpu, local, _ in active)):
            raise ValueError("live decode requires the complete unchanged DP-local request batches")
        step_index = states[0].step_index
    else:
        active = tuple((gpu, tuple(t for t in range(concurrency) if t % 8 == gpu), None)
            for gpu in range(min(8, concurrency)))
        step_index = None
    all_routes = tuple(seeded_expert_routes(concurrency, model.num_experts, 6,
        source_id="user_uniform_flash_gpu_fpga", root_seed=root_seed,
        moe_invocation_id=(f"layer-{layer}:decode" if step_index is None else
            f"step-{step_index}:layer-{layer}:decode")) for layer in range(layer_count))
    contexts = []
    for gpu, local, state in active:
        trace, rows = traces[gpu], len(local)
        group = ForwardLocalGroup(trace)
        prefix = ("" if state is None else f"decode-step-{state.step_index}:") + f"gpu-{gpu}"
        ids = (trace.tensor(prefix + ":input-ids", (rows,), "int32") if state is None
            else state.token.view_as_shape(rows))
        positions = trace.tensor(prefix + ":positions", (rows,), "int64")
        if include_model_boundary:
            from .traced_operators import SymbolicEmbedding
            x = SymbolicEmbedding(trace, prefix + ":embedding", "decode_gpu", model.hidden_size,
                "bf16", vocab_size=model.vocab_size)(ids)
        else:
            x = trace.tensor(prefix + ":hidden-input", (rows, model.hidden_size), "bf16")
        topk = (trace.tensor(prefix + ":topk", (rows, model.index_topk), "int32", persistent=True)
            if state is None else state.topk)
        rope = (trace.tensor(prefix + ":rope", (max(128, kv_length), model.qk_rope_head_dim), "fp32", persistent=True)
            if state is None else state.rope)
        capacity = -(-(rows * kv_length) // 128) * 128 if state is None else state.capacity
        stride = kv_length if state is None or state.cache_stride_tokens == 0 else state.cache_stride_tokens
        slots = tuple(row * stride + kv_length - 1 for row in range(rows))
        layers = []
        for layer in range(layer_count):
            # The coordinator owns the routed-expert join across all GPUs.
            layers.append(DeepseekV4DecoderLayer(trace, model, "decode_gpu", prefix + f":layer-{layer}",
                group, None, rope, topk, layer_index=layer, q_lens=(1,) * rows,
                kv_lens=(kv_length,) * rows, request_phases=("decode",) * rows,
                slot_mapping=slots, cache_capacity_tokens=capacity,
                source_max_model_len=max(128, kv_length) if state is None else max(128, state.max_context_tokens),
                linear_source_profile="vllm_0_30_logical_group128_fp32", source_use_deep_gemm=None,
                hardware=None, resolution=None, invocation_id=None,
                indexer_selection_source="declared_causal_topk", source_max_batched_tokens=rows,
                ffn=_requires_joint_expert_forward, mhc_workspace_scope="logical_math_only",
                cache_tensors=None if state is None else state.cache_tensors[layer]))
        contexts.append({"trace": trace, "local": local, "state": state, "prefix": prefix,
            "ids": ids, "positions": positions, "x": x, "layers": layers,
            "residual": None, "post_mix": None, "res_mix": None})
    for layer in range(layer_count):
        inputs = []
        for ctx in contexts:
            x, residual, post_mix, res_mix = ctx["layers"][layer].prepare_ffn(ctx["x"], ctx["positions"],
                ctx["post_mix"], ctx["res_mix"], ctx["residual"])
            ctx["residual"], ctx["post_mix"], ctx["res_mix"] = residual, post_mix, res_mix
            inputs.append((x, ctx["ids"], ctx["local"]))
        prefix = ("" if step_index is None else f"decode-step-{step_index}:") + f"layer-{layer}"
        outputs = _trace_flash_moe(model, traces, tuple(inputs), all_routes[layer], batches,
            prefix=prefix, layer_index=layer, root_seed=root_seed)
        for ctx, output in zip(contexts, outputs, strict=True):
            ctx["x"] = output
    outputs = []
    for ctx in contexts:
        trace, prefix, x = ctx["trace"], ctx["prefix"], ctx["x"]
        rows = len(ctx["local"])
        if include_model_boundary:
            from .traced_operators import SymbolicMHC, SymbolicRMSNorm, SymbolicLogitsRowSelect, SymbolicGreedy
            x = SymbolicMHC(trace, prefix + ":final-mhc-post", "decode_gpu", model, None,
                variant="mhc_post", source_use_deep_gemm=None, workspace_scope="logical_math_only")(
                    x, ctx["residual"], ctx["post_mix"], ctx["res_mix"])
            x = SymbolicMHC(trace, prefix + ":hc-head", "decode_gpu", model, None,
                variant="hc_head", source_use_deep_gemm=None, workspace_scope="logical_math_only")(x)
            x = SymbolicRMSNorm(trace, prefix + ":final-norm", "decode_gpu", model.hidden_size,
                model.target_semantics.rms_norm_eps)(x)
            indices = trace.tensor(prefix + ":logits-indices", (rows,), "int32")
            selected = SymbolicLogitsRowSelect(trace, prefix + ":logits-select", "decode_gpu")(
                x, indices, selected_indices=tuple(range(rows)))
            logits, _ = SymbolicLinear(trace, prefix + ":lm-head", "decode_gpu", model.hidden_size,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)
            x = SymbolicGreedy(trace, prefix + ":sampler", "decode_gpu")(logits)
        outputs.append(x)
        if state_observer is not None:
            state_observer(trace, 0, tuple(ctx["layers"]))
        state = ctx["state"]
        if state is not None:
            state.token, state.kv_length = x, kv_length
            state.step_index += 1
            state.layer_owners = tuple(ctx["layers"])
    if defer_graph:
        return None
    for trace in traces:
        _tag(trace)
    prior_graphs = () if states is None else tuple(state.input_graph for state in states)
    graph = WorkloadGraph((*(n for prior in prior_graphs for n in prior.nodes),
        *(n for t in traces for n in t._nodes.values())),
        (*(e for prior in prior_graphs for e in prior.edges),
            *(e for t in traces for e in t._capture_program_edges())))
    graph.topological_order()
    return HeterogeneousMoETrace(graph, traces, deployment, all_routes, tuple(outputs), batches)
