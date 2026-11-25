"""Kimi-K3 source candidates; complete runtime/provider support remains separate."""

from collections.abc import Mapping

from .model_config import ModelConfig
from .runtime_inputs import ExpertRoutingResolution, validate_expert_routing_provenance
from .shape_inference import infer_gemm_dimensions
from .workload_graph import WorkloadEdge, WorkloadGraph, WorkloadKind, WorkloadNode


def vllm030_kimi_k3_attention_residual_node(
    model: ModelConfig, *, rows: int, stage: str, layer_index: int | None,
    pool_name: str, rank: int,
) -> WorkloadNode:
    """Emit one source call with externally supplied prefix/delta/block tensors.

    Scope is eager, BF16, SP/PP1 without auxiliary hidden-state capture. The
    full workload builder still rejects K3 until its other operators exist.
    Callers must not chain these components across omitted attention/MLP work.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 attention residual requires exact hybrid model semantics")
    if type(rows) is not int or rows <= 0:
        raise ValueError("K3 attention residual rows must be positive integers")
    if stage not in ("pre_attention", "post_attention", "output"):
        raise ValueError("unknown K3 attention residual source stage")
    block_size = source.attention_residual_block_size
    capacity = -(-model.num_hidden_layers // block_size)
    if stage == "output":
        if layer_index is not None:
            raise ValueError("output attention residual has no decoder layer index")
        blocks, write, has_delta, output_norm = capacity, -1, True, False
        suffix = "output"
    else:
        if type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers:
            raise ValueError("decoder attention residual requires an in-range layer index")
        blocks = -(-layer_index // block_size)
        boundary = layer_index % block_size == 0
        write = layer_index // block_size if stage == "pre_attention" and boundary else -1
        has_delta = layer_index != 0 if stage == "pre_attention" else not boundary
        if stage == "post_attention":
            blocks += int(boundary)
        output_norm = True
        suffix = f"layer{layer_index}:{stage}"
    return WorkloadNode(
        f"{pool_name}:r{rank}:kimi-k3-attn-res:{suffix}", WorkloadKind.VECTOR,
        pool_name, {"rows": rows, "hidden": model.hidden_size,
                    "elements": rows * model.hidden_size},
        dtype="bf16", variant="kimi_k3_attn_res", rank=rank,
        attributes={"num_blocks": blocks, "block_write_idx": write,
                    "block_capacity": capacity, "has_delta": has_delta,
                    "apply_output_norm": output_norm,
                    "source_component": "vllm_0_30_kimi_k3_attention_residual",
                    "launch_policy": "analytical_launch"})


def vllm030_kimi_k3_ordinary_mlp_workload(
    model: ModelConfig, *, rows: int, layer_index: int, role: str,
    pool_name: str, rank: int,
) -> WorkloadGraph:
    """TP1 ordinary KimiMLP component with its full gate/SiTU/down chain.

    This source branch excludes GEMM-RS/AR, shared/routed fusion and MXFP4
    expert activation fusion. It does not bind their actual dispatch. The
    source tanhf implementation remains unpriced; no complete model latency
    may be inferred from this component. Input is post-attention normalized.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 ordinary MLP requires exact hybrid source semantics")
    if type(rows) is not int or rows <= 0:
        raise ValueError("K3 ordinary MLP rows must be positive integers")
    if type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers:
        raise ValueError("K3 ordinary MLP requires an in-range layer index")
    is_moe = layer_index >= model.num_dense_layers and layer_index % source.moe_layer_frequency == 0
    if role not in ("dense", "shared") or (role == "dense") == is_moe:
        raise ValueError("K3 ordinary MLP role disagrees with decoder source branch")
    if role == "shared" and model.num_shared_experts <= 0:
        raise ValueError("K3 shared MLP requires shared experts")
    base = f"language_model.model.layers.{layer_index}"
    base += ".mlp" if role == "dense" else ".block_sparse_moe.shared_experts"
    positions = ("gate_up_proj", "down_proj") if role == "dense" else (
        "shared_experts.gate_up_proj", "shared_experts.down_proj")
    nodes = []
    prefix = f"{pool_name}:r{rank}:kimi-k3-mlp:layer{layer_index}:{role}"
    for suffix, position in zip(("gate_up_proj", "down_proj"), positions, strict=True):
        m, n, k = infer_gemm_dimensions(model, position, rows, tp_size=1)
        module = f"{base}.{suffix}"
        weight_dtype = model.weight_dtype_for_module(module, "Linear")
        if weight_dtype != "bf16":
            raise NotImplementedError("K3 ordinary MLP requires the unquantized LinearBase source branch")
        nodes.append(WorkloadNode(
            f"{prefix}:{suffix}", WorkloadKind.LINEAR, pool_name,
            {"b": 1, "m": m, "n": n, "k": k}, dtype="bf16", variant="gemm", rank=rank,
            attributes={"has_bias": False, "launch_policy": "analytical_launch", "position": position,
                        "module_path": module, "weight_dtype": weight_dtype, "weight_group_size": 0,
                        "source_component": "vllm_0_30_kimi_k3_ordinary_mlp"}))
    intermediate = nodes[1].shape["k"]
    activation = WorkloadNode(
        f"{prefix}:situ", WorkloadKind.ACTIVATION, pool_name,
        {"rows": rows, "intermediate": intermediate, "elements": rows * intermediate},
        dtype="bf16", variant="kimi_k3_situ", rank=rank,
        attributes={"activation_beta": source.activation_situ_beta,
                    "activation_linear_beta": source.activation_situ_linear_beta,
                    "source_component": "vllm_0_30_kimi_k3_ordinary_mlp",
                    "launch_policy": "analytical_launch"})
    return WorkloadGraph((nodes[0], activation, nodes[1]), (
        WorkloadEdge(nodes[0].node_id, activation.node_id),
        WorkloadEdge(activation.node_id, nodes[1].node_id)))


def vllm030_kimi_k3_triton_decode_workload(
    model: ModelConfig, *, layer_index: int, state_indices: tuple[int, ...],
    state_pool_slots: int, conv_state_dtype: str, recurrent_state_dtype: str,
    pool_name: str, rank: int,
) -> WorkloadGraph:
    """Complete ordinary TP1 KDA decode source chain, not the whole decoder.

    Explicitly declares the Triton backend, MTP/prefix/RecoverSSM off, eager
    unpadded one-token rows, default BF16/FP32 caches and different active
    state slots. No backend dispatch or Scheduler input is fabricated. Other
    cache policies/backends need their own source program. Ordinary metadata
    has non_spec_token_start=None, so packed recurrent output is copied into
    core_attn_out; gated RMSNorm then aliases core_attn_out in place.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 Triton decode requires exact hybrid source semantics")
    if (type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers
            or source.layer_attention_types[layer_index] != "kda"):
        raise ValueError("K3 Triton decode requires an actual KDA layer")
    if (not isinstance(state_indices, tuple) or not state_indices
            or type(state_pool_slots) is not int or state_pool_slots <= 1
            or any(type(index) is not int or not 0 < index < state_pool_slots for index in state_indices)
            or len(set(state_indices)) != len(state_indices)):
        raise ValueError("K3 Triton decode requires distinct positive declared state indices in a valid pool")
    if conv_state_dtype != "bf16" or recurrent_state_dtype != "fp32":
        raise NotImplementedError("K3 Triton decode source chain requires BF16 conv/FP32 recurrent cache")
    rows = len(state_indices)
    base = f"language_model.model.layers.{layer_index}.self_attn"
    prefix = f"{pool_name}:r{rank}:kimi-k3-kda-decode:layer{layer_index}"
    common = {"source_component": "vllm_0_30_kimi_k3_triton_decode", "launch_policy": "analytical_launch"}

    def projection(position):
        m, n, k = infer_gemm_dimensions(model, position, rows, tp_size=1)
        module = f"{base}.{position}"
        if model.weight_dtype_for_module(module, "Linear") != "bf16":
            raise NotImplementedError("K3 ordinary KDA projection requires BF16 LinearBase")
        return WorkloadNode(
            f"{prefix}:{position}", WorkloadKind.LINEAR, pool_name,
            {"b": 1, "m": m, "n": n, "k": k}, dtype="bf16", variant="gemm", rank=rank,
            attributes=common | {"position": position, "module_path": module, "has_bias": False,
                                 "weight_dtype": "bf16", "weight_group_size": 0})

    def stage(name):
        return WorkloadNode(
            f"{prefix}:{name}", WorkloadKind.VECTOR, pool_name,
            {"rows": rows, "heads": source.kda_num_heads, "head_dim": source.kda_head_dim,
             "elements": rows * source.kda_num_heads * source.kda_head_dim},
            dtype="bf16", variant="kimi_k3_kda_decode", rank=rank,
            attributes=common | {"stage": name, "state_indices": state_indices,
                                 "state_pool_slots": state_pool_slots, "gate_lower_bound": source.gate_lower_bound})

    packed, fb, out = (projection(position) for position in ("in_proj_qkvgfab", "f_b_proj", "o_proj"))
    conv, recurrent, copy, norm = (stage(name) for name in ("convolution", "recurrent", "output_copy", "output_norm"))
    nodes = (packed, fb, conv, recurrent, copy, norm, out)
    edges = ((packed, fb), (packed, conv), (fb, recurrent), (conv, recurrent),
             (recurrent, copy), (copy, norm), (norm, out))
    # The eager branch enqueues f_b before convolution on the same stream.
    # Their data inputs are independent, but source ordering is not overlap.
    return WorkloadGraph(nodes, (
        *(WorkloadEdge(a.node_id, b.node_id) for a, b in edges),
        WorkloadEdge(fb.node_id, conv.node_id, "barrier")))


def vllm030_kimi_k3_triton_fresh_prefill_workload(
    model: ModelConfig, *, layer_index: int, query_start_loc: tuple[int, int],
    state_index: int, state_pool_slots: int, has_initial_state: bool,
    conv_state_dtype: str, recurrent_state_dtype: str, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Ordinary TP1 fresh one-sequence Triton KDA source chain.

    Complete64-token chunks, MTP/RecoverSSM/prefix checkpoints off. Conv Q/K/V
    are separate producers; dense FP32 initial zeros, chunk final state,
    pool scatter and the distinct BF16 output copy remain explicit. The
    multi-kernel chunk program remains unpriced, not a decode recurrence.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 fresh prefill requires exact hybrid source semantics")
    if (type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers
            or source.layer_attention_types[layer_index] != "kda"):
        raise ValueError("K3 fresh prefill requires an actual KDA layer")
    if (not isinstance(query_start_loc, tuple) or len(query_start_loc) != 2
            or any(type(value) is not int for value in query_start_loc)
            or query_start_loc[0] != 0 or not 64 <= query_start_loc[1] < 2**31
            or query_start_loc[1] % 64 or has_initial_state is not False):
        raise ValueError("K3 fresh prefill requires a complete INT32 sequence of full64-token chunks without initial state")
    if (type(state_pool_slots) is not int or not 1 < state_pool_slots <= 2**31
            or type(state_index) is not int or not 0 < state_index < state_pool_slots):
        raise ValueError("K3 fresh prefill requires one positive INT32 state index, not null/padding")
    if conv_state_dtype != "bf16" or recurrent_state_dtype != "fp32":
        raise NotImplementedError("K3 fresh prefill requires BF16 conv/FP32 recurrent cache")
    return _kimi_k3_triton_chunk_workload(
        model, layer_index=layer_index, query_start_loc=query_start_loc, state_indices=(state_index,),
        state_pool_slots=state_pool_slots, has_initial_state=False, phase="fresh_prefill",
        variant="kimi_k3_kda_prefill", pool_name=pool_name, rank=rank)


def vllm030_kimi_k3_triton_mixed_workload(
    model: ModelConfig, *, layer_index: int, query_start_loc: tuple[int, int, int],
    state_indices: tuple[int, int], state_pool_slots: int, has_initial_state: tuple[bool, bool],
    conv_state_dtype: str, recurrent_state_dtype: str, pool_name: str, rank: int,
) -> WorkloadGraph:
    """One ordinary decode plus one fresh full64-chunk prefill through chunk KDA.

    With any prefill, upstream runs all non-speculative rows through three
    causal_conv1d_fn calls, gather, chunk and two-state scatter, including
    the one-token decode segment. It does not call the pure-decode packed
    recurrence. Cache indices/initial flags are source inputs, not observed
    native metadata. Chunk internals remain explicitly unpriced.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 mixed KDA requires exact hybrid source semantics")
    if (type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers
            or source.layer_attention_types[layer_index] != "kda"):
        raise ValueError("K3 mixed KDA requires an actual KDA layer")
    if (not isinstance(query_start_loc, tuple) or len(query_start_loc) != 3
            or any(type(value) is not int for value in query_start_loc)
            or query_start_loc[:2] != (0, 1) or not 65 <= query_start_loc[2] < 2**31
            or (query_start_loc[2] - 1) % 64 or has_initial_state != (True, False)
            or not isinstance(has_initial_state, tuple) or any(type(flag) is not bool for flag in has_initial_state)):
        raise ValueError("K3 mixed KDA requires one initialized decode and one fresh full64-chunk INT32 segment")
    if (type(state_pool_slots) is not int or not 2 < state_pool_slots <= 2**31
            or not isinstance(state_indices, tuple) or len(state_indices) != 2
            or any(type(index) is not int or not 0 < index < state_pool_slots for index in state_indices)
            or len(set(state_indices)) != 2):
        raise ValueError("K3 mixed KDA requires two distinct positive INT32 state indices")
    if conv_state_dtype != "bf16" or recurrent_state_dtype != "fp32":
        raise NotImplementedError("K3 mixed KDA requires BF16 conv/FP32 recurrent cache")
    return _kimi_k3_triton_chunk_workload(
        model, layer_index=layer_index, query_start_loc=query_start_loc, state_indices=state_indices,
        state_pool_slots=state_pool_slots, has_initial_state=has_initial_state, phase="mixed",
        variant="kimi_k3_kda_mixed", pool_name=pool_name, rank=rank)


def vllm030_kimi_k3_triton_context_prefill_workload(
    model: ModelConfig, *, layer_index: int, query_start_loc: tuple[int, int],
    state_index: int, state_pool_slots: int, has_initial_state: bool,
    conv_state_dtype: str, recurrent_state_dtype: str, pool_name: str, rank: int,
) -> WorkloadGraph:
    """One full64-chunk context extension with initialized recurrent/conv state.

    Query tokens are new; their prior length/cache values remain external.
    This uses the same source chunk program as fresh prefill, with one true
    gather mask and prior convolution history reads. Not a Scheduler trace.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 context prefill requires exact hybrid source semantics")
    if (type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers
            or source.layer_attention_types[layer_index] != "kda"):
        raise ValueError("K3 context prefill requires an actual KDA layer")
    if (not isinstance(query_start_loc, tuple) or len(query_start_loc) != 2
            or any(type(value) is not int for value in query_start_loc)
            or query_start_loc[0] != 0 or not 64 <= query_start_loc[1] < 2**31
            or query_start_loc[1] % 64 or has_initial_state is not True):
        raise ValueError("K3 context prefill requires one full64-chunk INT32 segment with initial state")
    if (type(state_pool_slots) is not int or not 1 < state_pool_slots <= 2**31
            or type(state_index) is not int or not 0 < state_index < state_pool_slots):
        raise ValueError("K3 context prefill requires one positive INT32 state index")
    if conv_state_dtype != "bf16" or recurrent_state_dtype != "fp32":
        raise NotImplementedError("K3 context prefill requires BF16 conv/FP32 recurrent cache")
    return _kimi_k3_triton_chunk_workload(
        model, layer_index=layer_index, query_start_loc=query_start_loc, state_indices=(state_index,),
        state_pool_slots=state_pool_slots, has_initial_state=True, phase="context_prefill",
        variant="kimi_k3_kda_context_prefill", pool_name=pool_name, rank=rank)


def _kimi_k3_triton_chunk_workload(model, *, layer_index, query_start_loc, state_indices,
                                  state_pool_slots, has_initial_state, phase, variant, pool_name, rank):
    source = model.kimi_k3_semantics
    rows = query_start_loc[-1]
    base = f"language_model.model.layers.{layer_index}.self_attn"
    label = {"fresh_prefill": "prefill", "mixed": "mixed", "context_prefill": "context-prefill"}[phase]
    prefix = f"{pool_name}:r{rank}:kimi-k3-kda-{label}:layer{layer_index}"
    common = {"source_component": f"vllm_0_30_kimi_k3_triton_{phase}", "launch_policy": "analytical_launch"}

    def projection(position):
        m, n, k = infer_gemm_dimensions(model, position, rows, tp_size=1)
        module = f"{base}.{position}"
        if model.weight_dtype_for_module(module, "Linear") != "bf16":
            raise NotImplementedError("K3 ordinary KDA prefill projection requires BF16 LinearBase")
        return WorkloadNode(
            f"{prefix}:{position}", WorkloadKind.LINEAR, pool_name,
            {"b": 1, "m": m, "n": n, "k": k}, dtype="bf16", variant="gemm", rank=rank,
            attributes=common | {"position": position, "module_path": module, "has_bias": False,
                                 "weight_dtype": "bf16", "weight_group_size": 0})

    def stage(name):
        elements = (len(state_indices) * source.kda_num_heads * source.kda_head_dim**2
                    if name in ("initial_state", "final_state_scatter") else rows * source.kda_num_heads * source.kda_head_dim)
        return WorkloadNode(
            f"{prefix}:{name}", WorkloadKind.VECTOR, pool_name,
            {"rows": rows, "heads": source.kda_num_heads, "head_dim": source.kda_head_dim, "elements": elements},
            dtype="bf16", variant=variant, rank=rank,
            attributes=common | {"stage": name, "state_indices": state_indices, "state_pool_slots": state_pool_slots,
                                 "query_start_loc": query_start_loc, "has_initial_state": has_initial_state,
                                 "gate_lower_bound": source.gate_lower_bound})

    packed, fb, out = (projection(position) for position in ("in_proj_qkvgfab", "f_b_proj", "o_proj"))
    q, k, v, initial, chunk, scatter, copy, norm = (stage(name) for name in (
        "q_convolution", "k_convolution", "v_convolution", "initial_state", "chunk",
        "final_state_scatter", "output_copy", "output_norm"))
    nodes = (packed, fb, q, k, v, initial, chunk, scatter, copy, norm, out)
    data = ((packed, fb), (packed, q), (packed, k), (packed, v),
            (q, chunk), (k, chunk), (v, chunk), (fb, chunk), (initial, chunk),
            (chunk, scatter), (chunk, copy), (copy, norm), (norm, out))
    barriers = ((fb, q), (q, k), (k, v), (v, initial), (scatter, copy))
    return WorkloadGraph(nodes, (
        *(WorkloadEdge(a.node_id, b.node_id) for a, b in data),
        *(WorkloadEdge(a.node_id, b.node_id, "barrier") for a, b in barriers)))


def vllm030_kimi_k3_dense_fresh_prefill_layer_workload(
    model: ModelConfig, *, query_start_loc: tuple[int, int], state_index: int, state_pool_slots: int,
    has_initial_state: bool, conv_state_dtype: str, recurrent_state_dtype: str, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Join the first dense decoder-layer source body, retaining chunk/SiTU gaps."""
    if model.num_dense_layers != 1:
        raise NotImplementedError("K3 first prefill layer requires its exact one dense layer")
    attention = vllm030_kimi_k3_triton_fresh_prefill_workload(
        model, layer_index=0, query_start_loc=query_start_loc, state_index=state_index,
        state_pool_slots=state_pool_slots, has_initial_state=has_initial_state,
        conv_state_dtype=conv_state_dtype, recurrent_state_dtype=recurrent_state_dtype,
        pool_name=pool_name, rank=rank)
    rows = query_start_loc[1]
    pre = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="pre_attention", layer_index=0, pool_name=pool_name, rank=rank)
    post = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="post_attention", layer_index=0, pool_name=pool_name, rank=rank)
    mlp = vllm030_kimi_k3_ordinary_mlp_workload(
        model, rows=rows, layer_index=0, role="dense", pool_name=pool_name, rank=rank)
    return WorkloadGraph((pre, *attention.nodes, post, *mlp.nodes), (
        *attention.edges, *mlp.edges, WorkloadEdge(pre.node_id, attention.nodes[0].node_id),
        WorkloadEdge(attention.nodes[-1].node_id, post.node_id), WorkloadEdge(post.node_id, mlp.nodes[0].node_id)))


def vllm030_kimi_k3_dense_decode_layer_workload(
    model: ModelConfig, *, state_indices: tuple[int, ...], state_pool_slots: int,
    conv_state_dtype: str, recurrent_state_dtype: str, pool_name: str, rank: int,
) -> WorkloadGraph:
    """First real decoder-layer source body, with all its component edges.

    Source inputs are embedding hidden rows and declared KDA cache slots.
    This joins both AttnRes calls, the eager Triton KDA decode chain and
    ordinary dense SiTU MLP. SiTU is explicitly unpriced. It is one layer,
    not a substitute for the complete 93-layer model or runtime execution.
    """
    if model.num_dense_layers != 1:
        raise NotImplementedError("K3 first-layer source join requires its exact one dense layer")
    attention = vllm030_kimi_k3_triton_decode_workload(
        model, layer_index=0, state_indices=state_indices, state_pool_slots=state_pool_slots,
        conv_state_dtype=conv_state_dtype, recurrent_state_dtype=recurrent_state_dtype,
        pool_name=pool_name, rank=rank)
    rows = len(state_indices)
    pre = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="pre_attention", layer_index=0, pool_name=pool_name, rank=rank)
    post = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="post_attention", layer_index=0, pool_name=pool_name, rank=rank)
    mlp = vllm030_kimi_k3_ordinary_mlp_workload(
        model, rows=rows, layer_index=0, role="dense", pool_name=pool_name, rank=rank)
    return WorkloadGraph((pre, *attention.nodes, post, *mlp.nodes), (
        *attention.edges, *mlp.edges,
        WorkloadEdge(pre.node_id, attention.nodes[0].node_id),
        WorkloadEdge(attention.nodes[-1].node_id, post.node_id),
        WorkloadEdge(post.node_id, mlp.nodes[0].node_id)))


def vllm030_kimi_k3_nope_fresh_prefill_workload(
    model: ModelConfig, *, layer_index: int, slot_mapping: tuple[int, ...],
    cache_capacity_tokens: int, cache_dtype: str, pool_name: str, rank: int,
) -> WorkloadGraph:
    """TP1 BF16 NoPE MLA, one fresh sequence and merged QKV/gate frontend.

    T>=512 selects the merged projection independently of auxiliary-stream
    allocation. No context, DCP, MTP, graph padding or GEMM-RS/AR is enabled.
    FA3 reuses the canonical analytical prefill selector; this source graph
    does not establish actual backend/provider dispatch. The compiled gate
    retains its BF16 sigmoid boundary and is explicitly unpriced.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 NoPE prefill requires exact hybrid source semantics")
    if (type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers
            or source.layer_attention_types[layer_index] != "mla_nope"):
        raise ValueError("K3 NoPE prefill requires an actual NoPE MLA layer")
    if not source.mla_use_output_gate:
        raise NotImplementedError("K3 NoPE prefill requires its configured output gate")
    if (not isinstance(slot_mapping, tuple) or len(slot_mapping) < 512
            or type(cache_capacity_tokens) is not int or cache_capacity_tokens <= 0
            or any(type(slot) is not int or not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)
            or len(set(slot_mapping)) != len(slot_mapping)):
        raise ValueError("K3 merged NoPE fresh prefill requires T>=512 distinct valid nonnegative INT64 slots")
    if cache_dtype != "bf16" or model.compute_dtype != "bf16":
        raise NotImplementedError("K3 NoPE prefill source branch requires BF16 activation/cache storage")
    rows = len(slot_mapping)
    base = f"language_model.model.layers.{layer_index}.self_attn"
    prefix = f"{pool_name}:r{rank}:kimi-k3-nope-prefill:layer{layer_index}"
    common = {"source_component": "vllm_0_30_kimi_k3_nope_fresh_prefill", "launch_policy": "analytical_launch"}

    def projection(position):
        m, n, k = infer_gemm_dimensions(model, position, rows, tp_size=1)
        module = f"{base}.{position}"
        if model.weight_dtype_for_module(module, "Linear") != "bf16":
            raise NotImplementedError("K3 ordinary MLA projection requires BF16 LinearBase")
        return WorkloadNode(
            f"{prefix}:{position}", WorkloadKind.LINEAR, pool_name,
            {"b": 1, "m": m, "n": n, "k": k}, dtype="bf16", variant="gemm", rank=rank,
            attributes=common | {"position": position, "module_path": module, "has_bias": False,
                                 "weight_dtype": "bf16", "weight_group_size": 0})

    def stage(name, width):
        return WorkloadNode(
            f"{prefix}:{name}", WorkloadKind.VECTOR, pool_name,
            {"rows": rows, "heads": model.num_attention_heads,
             "q_lora_rank": model.q_lora_rank, "kv_lora_rank": model.kv_lora_rank,
             "elements": rows * width}, dtype="bf16", variant="kimi_k3_nope_prefill", rank=rank,
            attributes=common | {"stage": name, "slot_mapping": slot_mapping,
                                 "cache_capacity_tokens": cache_capacity_tokens,
                                 "rms_norm_eps": source.rms_norm_eps})

    packed, qb, kvb, out = (projection(position) for position in (
        "fused_qkv_a_g_proj", "q_b_proj", "kv_b_proj", "o_proj"))
    norm = stage("q_kv_norm", model.q_lora_rank + model.kv_lora_rank)
    concat = stage("key_cache_concat", model.num_attention_heads * model.head_dim + 576)
    gate = stage("output_gate", model.num_attention_heads * model.v_head_dim)
    attention = WorkloadNode(
        f"{prefix}:attention", WorkloadKind.FLASH_ATTN, pool_name,
        {"batch": 1, "q_len": rows, "kv_len": rows,
         "qk_head_dim": model.head_dim, "v_head_dim": model.v_head_dim,
         "num_query_heads": model.num_attention_heads, "kv_lora_rank": model.kv_lora_rank},
        dtype="bf16", variant="mla_prefill", rank=rank,
        attributes=common | {"causal": True, "kv_dist": "uniform", "kv_layout": "mla",
                             "q_lens": (rows,), "kv_lens": (rows,),
                             "full_cuda_graphs": False, "max_cudagraph_capture_size": 0})
    nodes = (packed, norm, qb, kvb, concat, attention, gate, out)
    edges = ((packed, norm), (norm, qb), (norm, kvb), (qb, concat), (kvb, concat),
             (concat, attention), (attention, gate), (packed, gate), (gate, out))
    return WorkloadGraph(nodes, (
        *(WorkloadEdge(a.node_id, b.node_id) for a, b in edges),
        WorkloadEdge(qb.node_id, kvb.node_id, "barrier")))


def vllm030_kimi_k3_tp1_native_latent_output_workload(
    model: ModelConfig, *, rows: int, layer_index: int, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Native TP1 MoERunner tail on externally supplied BF16 branch outputs.

    Inputs are the weighted/combined routed latent and the shared MLP result,
    not the original hidden states. No expert dispatch/routing is omitted
    inside a claimed full MoE call: this is explicitly just its output tail.
    TP1 makes LatentMoERunner._use_fused_path false. Base MoERunner applies
    the nonlinear RMSNorm after expert combination, uses an ordinary beta=0
    up-projection, then adds shared output in a distinct tensor operation.
    MegaMoE's residual.addmm_ and the TP>1 tail tiers are separate programs.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 native latent output requires exact hybrid source semantics")
    if type(rows) is not int or rows <= 0:
        raise ValueError("K3 native latent output requires positive integer rows")
    if (type(layer_index) is not int or not model.num_dense_layers <= layer_index < model.num_hidden_layers
            or layer_index % source.moe_layer_frequency != 0):
        raise ValueError("K3 native latent output requires an actual MoE layer")
    if not source.latent_moe_use_norm or model.num_shared_experts <= 0 or model.compute_dtype != "bf16":
        raise NotImplementedError("K3 native latent output requires BF16 shared output and configured latent RMSNorm")
    common = {"source_component": "vllm_0_30_kimi_k3_tp1_native_latent_output",
              "launch_policy": "analytical_launch"}
    prefix = f"{pool_name}:r{rank}:kimi-k3-native-latent-output:layer{layer_index}"
    norm = WorkloadNode(
        f"{prefix}:latent_norm", WorkloadKind.NORM, pool_name,
        {"rows": rows, "hidden": source.routed_expert_hidden_size,
         "elements": rows * source.routed_expert_hidden_size},
        dtype="bf16", variant="rmsnorm", rank=rank,
        attributes=common | {"rms_norm_eps": source.rms_norm_eps,
                             "source_symbol": "KimiRoutedOutputTransform.forward:self.norm(hidden_states)"})
    m, n, k = infer_gemm_dimensions(model, "routed_expert_up_proj", rows, tp_size=1)
    up = WorkloadNode(
        f"{prefix}:up_proj", WorkloadKind.LINEAR, pool_name,
        {"b": 1, "m": m, "n": n, "k": k}, dtype="bf16", variant="gemm", rank=rank,
        attributes=common | {"position": "routed_expert_up_proj", "has_bias": False,
                             "module_path": f"language_model.model.layers.{layer_index}.block_sparse_moe.routed_expert_up_proj",
                             "weight_dtype": "bf16", "weight_group_size": 0,
                             "source_symbol": "KimiRoutedOutputTransform.forward:self.up_proj(hidden_states)"})
    add = WorkloadNode(
        f"{prefix}:shared_add", WorkloadKind.ELEMENTWISE, pool_name,
        {"elements": rows * model.hidden_size}, dtype="bf16", variant="add", rank=rank,
        attributes=common | {"operation": "add", "reads": 2, "writes": 1,
                             "source_symbol": "MoERunner.forward:shared_output+fused_output"})
    return WorkloadGraph((norm, up, add), (
        WorkloadEdge(norm.node_id, up.node_id), WorkloadEdge(up.node_id, add.node_id)))


def vllm030_kimi_k3_marlin_expert_node(
    model: ModelConfig, *, rows: int, layer_index: int,
    routing: ExpertRoutingResolution, pool_name: str, rank: int,
) -> WorkloadNode:
    """TP1 standard-format Marlin expert source aggregate, not dispatch.

    Inputs are down-projected BF16 latent rows and already selected FP32
    weights/INT32 IDs. The supplied histogram remains an explicit routing
    input with its original provenance, not logits-derived checkpoint truth.
    Top-k selection, latent down projection, shared experts and output tail
    are outside this aggregate. It retains SiTU instead of generic SwiGLU.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 Marlin experts require exact hybrid source semantics")
    if type(rows) is not int or rows <= 0:
        raise ValueError("K3 Marlin experts require positive integer rows")
    if (type(layer_index) is not int or not model.num_dense_layers <= layer_index < model.num_hidden_layers
            or layer_index % source.moe_layer_frequency != 0):
        raise ValueError("K3 Marlin experts require an actual MoE layer")
    if not isinstance(routing, ExpertRoutingResolution):
        raise TypeError("K3 Marlin experts require a typed routing resolution")
    if routing.routing_provenance["layer_index"] != layer_index:
        raise ValueError("K3 Marlin expert routing belongs to a different layer")
    invocation = routing.routing_provenance["moe_invocation_id"]
    validate_expert_routing_provenance(
        routing.routing_provenance, tokens_per_expert=routing.tokens_per_expert,
        tokens=rows, num_experts=model.num_experts, topk=model.num_experts_per_tok,
        moe_invocation_id=invocation)
    module = f"language_model.model.layers.{layer_index}.block_sparse_moe.experts"
    if model.weight_dtype_for_module(module, "RoutedExperts") != "mxfp4":
        raise NotImplementedError("K3 Marlin experts require checkpoint MXFP4 RoutedExperts")
    return WorkloadNode(
        f"{pool_name}:r{rank}:kimi-k3-marlin-experts:layer{layer_index}",
        WorkloadKind.FUSED_MOE, pool_name,
        {"tokens": rows, "experts": model.num_experts, "topk": model.num_experts_per_tok,
         "hidden": source.routed_expert_hidden_size, "intermediate": model.moe_intermediate_size},
        dtype="bf16", variant="full", rank=rank,
        attributes={
            "source_component": "vllm_0_30_kimi_k3_marlin_experts", "launch_policy": "analytical_launch",
            "module_path": module, "moe_invocation_id": invocation,
            "routing_provenance": dict(routing.routing_provenance),
            "tokens_per_expert": routing.tokens_per_expert,
            "local_tokens_per_expert": routing.tokens_per_expert,
            "expert_parallel_size": 1, "expert_rank": 0,
            "owned_experts": tuple(range(model.num_experts)), "expert_weight_tile_rows": None,
            "activation_beta": source.activation_situ_beta,
            "activation_linear_beta": source.activation_situ_linear_beta,
            "routed_scaling_factor": model.routed_scaling_factor})


def vllm030_kimi_k3_marlin_routed_output_workload(
    model: ModelConfig, *, rows: int, layer_index: int,
    routing: ExpertRoutingResolution, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Join the routed expert source body to its native TP1 output tail.

    External inputs remain the down-projected latent, selected IDs/weights,
    and shared output. This closes the routed latent -> hidden branch; it
    does not invent original-input projections, routing or a shared MLP.
    """
    experts = vllm030_kimi_k3_marlin_expert_node(
        model, rows=rows, layer_index=layer_index, routing=routing, pool_name=pool_name, rank=rank)
    tail = vllm030_kimi_k3_tp1_native_latent_output_workload(
        model, rows=rows, layer_index=layer_index, pool_name=pool_name, rank=rank)
    return WorkloadGraph((experts, *tail.nodes), (
        *tail.edges, WorkloadEdge(experts.node_id, tail.nodes[0].node_id)))


def vllm030_kimi_k3_native_marlin_moe_workload(
    model: ModelConfig, *, rows: int, layer_index: int, routing: ExpertRoutingResolution,
    shared_experts_stream_token_threshold: int, pool_name: str, rank: int,
    aux_streams_available: bool | None = None,
) -> WorkloadGraph:
    """Native TP1 Marlin MoE source body, from original hidden to output.

    This explicitly selects the standard modular Marlin source program,
    without claiming auto backend dispatch. Rows>256 make the model's
    router/down calls sequential. For smaller rows the caller must explicitly
    bind source auxiliary-stream availability. The shared-stream threshold
    controls external shared overlap even when no-DP/EP prepare/finalize
    cannot overlap it internally. Shared completion joins before the latent
    output transform; there is no added empirical overlap discount.
    The routing histogram is still a declared input, not derived logits.
    """
    if (type(rows) is not int or rows <= 0
            or type(shared_experts_stream_token_threshold) is not int
            or shared_experts_stream_token_threshold < 0):
        raise ValueError("K3 native MoE requires positive rows and a nonnegative shared-stream threshold")
    if aux_streams_available is not None and type(aux_streams_available) is not bool:
        raise TypeError("K3 native MoE auxiliary-stream availability must be boolean or unbound")
    if ((rows <= 256 or rows <= shared_experts_stream_token_threshold)
            and aux_streams_available is not True):
        raise ValueError("K3 serial native MoE requires rows>256 and rows above the shared-stream threshold; small rows require explicit auxiliary streams")
    overlap_router = rows <= 256
    overlap_shared = rows <= shared_experts_stream_token_threshold
    # This validates the actual MoE layer, exact MXFP4 source scope and the
    # supplied canonical routing provenance before any projection is emitted.
    routed_output = vllm030_kimi_k3_marlin_routed_output_workload(
        model, rows=rows, layer_index=layer_index, routing=routing, pool_name=pool_name, rank=rank)
    base = f"language_model.model.layers.{layer_index}.block_sparse_moe"
    prefix = f"{pool_name}:r{rank}:kimi-k3-native-marlin-moe:layer{layer_index}"
    common = {"source_component": "vllm_0_30_kimi_k3_native_marlin_moe",
              "launch_policy": "analytical_launch",
              "shared_experts_stream_token_threshold": shared_experts_stream_token_threshold,
              "aux_streams_available": aux_streams_available}

    def projection(position, suffix, kind, output_dtype):
        m, n, k = infer_gemm_dimensions(model, position, rows, tp_size=1)
        return WorkloadNode(
            f"{prefix}:{suffix}", kind, pool_name, {"b": 1, "m": m, "n": n, "k": k},
            dtype="bf16", variant="gemm", rank=rank,
            attributes=common | {"position": position, "module_path": f"{base}.{suffix}",
                                 "has_bias": False, "weight_dtype": "bf16", "weight_group_size": 0,
                                 "output_dtype": output_dtype})

    gate = projection("router", "gate", WorkloadKind.ROUTER, "fp32")
    down = projection("routed_expert_down_proj", "routed_expert_down_proj", WorkloadKind.LINEAR, "bf16")
    shared = vllm030_kimi_k3_ordinary_mlp_workload(
        model, rows=rows, layer_index=layer_index, role="shared", pool_name=pool_name, rank=rank)
    route = WorkloadNode(
        f"{prefix}:topk", WorkloadKind.VECTOR, pool_name,
        {"rows": rows, "experts": model.num_experts, "topk": model.num_experts_per_tok,
         "elements": rows * model.num_experts}, dtype="fp32", variant="kimi_k3_moe_route", rank=rank,
        attributes=common | {"expert_id_selection": "score_topk", "score_function": "sigmoid",
                             "expert_group_count": 1, "topk_group_count": 1,
                             "normalize_topk_probability": True,
                             "routed_scaling_factor": model.routed_scaling_factor})
    experts, norm, _, add = routed_output.nodes
    nodes = (gate, down, *shared.nodes, route, *routed_output.nodes)
    barriers = [(down, shared.nodes[0])]
    if overlap_router:
        barriers.append((gate, shared.nodes[0]))
    else:
        barriers.append((gate, down))
    if overlap_shared:
        barriers.extend(((down, route), (shared.nodes[-1], norm)))
    else:
        barriers.append((shared.nodes[-1], route))
    return WorkloadGraph(nodes, (
        *shared.edges, *routed_output.edges,
        WorkloadEdge(gate.node_id, route.node_id),
        WorkloadEdge(down.node_id, experts.node_id),
        WorkloadEdge(route.node_id, experts.node_id),
        WorkloadEdge(shared.nodes[-1].node_id, add.node_id),
        *(WorkloadEdge(a.node_id, b.node_id, "barrier") for a, b in barriers)))


def vllm030_kimi_k3_nope_prefill_layer_workload(
    model: ModelConfig, *, layer_index: int, slot_mapping: tuple[int, ...], cache_capacity_tokens: int,
    cache_dtype: str, routing: ExpertRoutingResolution, shared_experts_stream_token_threshold: int,
    pool_name: str, rank: int,
) -> WorkloadGraph:
    """One complete NoPE decoder-layer source body on declared fresh inputs.

    AttnRes prefix/delta/block tensors and this layer's input hidden states
    are external. Previous layers, the whole hybrid stack and runtime are
    not executed or inferred by this join. Expert/top-k and gate services
    retain their explicit unpriced boundaries.
    """
    attention = vllm030_kimi_k3_nope_fresh_prefill_workload(
        model, layer_index=layer_index, slot_mapping=slot_mapping,
        cache_capacity_tokens=cache_capacity_tokens, cache_dtype=cache_dtype, pool_name=pool_name, rank=rank)
    rows = len(slot_mapping)
    moe = vllm030_kimi_k3_native_marlin_moe_workload(
        model, rows=rows, layer_index=layer_index, routing=routing,
        shared_experts_stream_token_threshold=shared_experts_stream_token_threshold, pool_name=pool_name, rank=rank)
    pre = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="pre_attention", layer_index=layer_index, pool_name=pool_name, rank=rank)
    post = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="post_attention", layer_index=layer_index, pool_name=pool_name, rank=rank)
    return WorkloadGraph((pre, *attention.nodes, post, *moe.nodes), (
        *attention.edges, *moe.edges,
        WorkloadEdge(pre.node_id, attention.nodes[0].node_id),
        WorkloadEdge(attention.nodes[-1].node_id, post.node_id),
        *(WorkloadEdge(post.node_id, moe.nodes[index].node_id) for index in (0, 1, 2))))


def vllm030_kimi_k3_kda_moe_fresh_prefill_layer_workload(
    model: ModelConfig, *, layer_index: int, query_start_loc: tuple[int, int],
    state_index: int, state_pool_slots: int, has_initial_state: bool,
    conv_state_dtype: str, recurrent_state_dtype: str, routing: ExpertRoutingResolution,
    shared_experts_stream_token_threshold: int, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Join a fresh full-chunk KDA MoE layer on explicit source inputs.

    The normalized original hidden rows feed gate/down/shared directly;
    the expert aggregate receives the down-projected latent. No native
    allocation, checkpoint-derived route or automatic backend is inferred.
    """
    attention = vllm030_kimi_k3_triton_fresh_prefill_workload(
        model, layer_index=layer_index, query_start_loc=query_start_loc,
        state_index=state_index, state_pool_slots=state_pool_slots, has_initial_state=has_initial_state,
        conv_state_dtype=conv_state_dtype, recurrent_state_dtype=recurrent_state_dtype,
        pool_name=pool_name, rank=rank)
    rows = query_start_loc[1]
    moe = vllm030_kimi_k3_native_marlin_moe_workload(
        model, rows=rows, layer_index=layer_index, routing=routing,
        shared_experts_stream_token_threshold=shared_experts_stream_token_threshold, pool_name=pool_name, rank=rank)
    pre = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="pre_attention", layer_index=layer_index, pool_name=pool_name, rank=rank)
    post = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="post_attention", layer_index=layer_index, pool_name=pool_name, rank=rank)
    return WorkloadGraph((pre, *attention.nodes, post, *moe.nodes), (
        *attention.edges, *moe.edges,
        WorkloadEdge(pre.node_id, attention.nodes[0].node_id),
        WorkloadEdge(attention.nodes[-1].node_id, post.node_id),
        *(WorkloadEdge(post.node_id, moe.nodes[index].node_id) for index in (0, 1, 2))))


def vllm030_kimi_k3_fresh_prefill_decoder_stack_workload(
    model: ModelConfig, *, query_start_loc: tuple[int, int],
    state_indices_by_layer: Mapping[int, int], state_pool_slots: int,
    slot_mapping_by_layer: Mapping[int, tuple[int, ...]], cache_capacity_tokens: int,
    routing_by_layer: Mapping[int, ExpertRoutingResolution],
    shared_experts_stream_token_threshold: int, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Complete decoder-stack source candidate, excluding embedding/logits/sampler.

    One fresh T>=512 sequence of full64 chunks, BF16 attention/conv cache,
    FP32 recurrent cache, TP/PP/SP/CP/DP1 EP-off, explicit Triton KDA and
    standard Marlin/no-DP-EP MoE. Each layer has a separate logical cache
    namespace: these indices do not establish physical shared-pool native
    allocations. Routes retain their declared provenance; source joins do
    not calculate logits-derived histograms. Chunk/SiTU/route/expert/gate
    services remain explicitly unpriced. Final AttnRes has no output norm;
    compute_logits applies norm later, after runner row selection.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 decoder stack requires exact hybrid source semantics")
    if (model.num_dense_layers != 1 or source.moe_layer_frequency != 1
            or source.layer_attention_types[0] != "kda"):
        raise NotImplementedError("K3 decoder stack requires first dense KDA and remaining MoE layers")
    if (not isinstance(query_start_loc, tuple) or len(query_start_loc) != 2
            or any(type(value) is not int for value in query_start_loc)
            or query_start_loc[0] != 0 or not 512 <= query_start_loc[1] < 2**31
            or query_start_loc[1] % 64):
        raise ValueError("K3 decoder stack requires fresh T>=512 full64-token chunks")
    kda = {i for i, kind in enumerate(source.layer_attention_types) if kind == "kda"}
    mla = {i for i, kind in enumerate(source.layer_attention_types) if kind == "mla_nope"}
    if kda | mla != set(range(model.num_hidden_layers)):
        raise NotImplementedError("K3 decoder stack contains an unsupported attention type")
    expected_routes = set(range(1, model.num_hidden_layers))
    for name, values, expected in (("state_indices_by_layer", state_indices_by_layer, kda),
                                    ("slot_mapping_by_layer", slot_mapping_by_layer, mla),
                                    ("routing_by_layer", routing_by_layer, expected_routes)):
        if (not isinstance(values, Mapping) or any(type(key) is not int for key in values)
                or set(values) != expected):
            raise ValueError(f"K3 decoder stack requires exactly the actual layer keys in {name}")
    if any(not isinstance(slots, tuple) or len(slots) != query_start_loc[1]
           for slots in slot_mapping_by_layer.values()):
        raise ValueError("K3 decoder stack MLA slots must match the fresh sequence rows")
    nodes, edges = [], []
    previous = None
    for layer_index, kind in enumerate(source.layer_attention_types):
        common = {"pool_name": pool_name, "rank": rank}
        if kind == "kda":
            inputs = {"query_start_loc": query_start_loc, "state_index": state_indices_by_layer[layer_index],
                      "state_pool_slots": state_pool_slots, "has_initial_state": False,
                      "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32", **common}
            if layer_index == 0:
                layer = vllm030_kimi_k3_dense_fresh_prefill_layer_workload(model, **inputs)
            else:
                layer = vllm030_kimi_k3_kda_moe_fresh_prefill_layer_workload(
                    model, layer_index=layer_index, routing=routing_by_layer[layer_index],
                    shared_experts_stream_token_threshold=shared_experts_stream_token_threshold, **inputs)
        elif kind == "mla_nope":
            layer = vllm030_kimi_k3_nope_prefill_layer_workload(
                model, layer_index=layer_index, slot_mapping=slot_mapping_by_layer[layer_index],
                cache_capacity_tokens=cache_capacity_tokens, cache_dtype="bf16",
                routing=routing_by_layer[layer_index],
                shared_experts_stream_token_threshold=shared_experts_stream_token_threshold, **common)
        else:
            raise NotImplementedError("K3 decoder stack contains an unsupported attention type")
        nodes.extend(layer.nodes)
        edges.extend(layer.edges)
        if previous is not None:
            edges.append(WorkloadEdge(previous.node_id, layer.nodes[0].node_id))
        previous = layer.nodes[-1]
    final = vllm030_kimi_k3_attention_residual_node(
        model, rows=query_start_loc[1], stage="output", layer_index=None, pool_name=pool_name, rank=rank)
    nodes.append(final)
    edges.append(WorkloadEdge(previous.node_id, final.node_id))
    return WorkloadGraph(nodes, edges)


def vllm030_kimi_k3_fresh_prefill_logits_workload(
    model: ModelConfig, *, query_start_loc: tuple[int, int], pool_name: str, rank: int,
) -> WorkloadGraph:
    """Ordinary one-sequence runner selection -> final norm -> BF16 logits.

    Input is the full stack's pre-norm hidden output. No prompt logprobs,
    speculation, pooling, auxiliary capture or logits processor overrides.
    The INT32 query boundary selects only its last hidden row, before the
    K3 compute_logits RMSNorm. Sampler and index construction are outside
    this component; the generic row-copy is a logical Roofline projection.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 logits require exact hybrid source semantics")
    if (not isinstance(query_start_loc, tuple) or len(query_start_loc) != 2
            or any(type(value) is not int for value in query_start_loc)
            or query_start_loc[0] != 0 or not 0 < query_start_loc[1] < 2**31):
        raise ValueError("K3 logits require one nonempty INT32 query sequence")
    return _kimi_k3_selected_logits_workload(model, query_start_loc=query_start_loc,
                                             label="prefill", pool_name=pool_name, rank=rank)


def _kimi_k3_selected_logits_workload(model, *, query_start_loc, label, pool_name, rank):
    source = model.kimi_k3_semantics
    module = "language_model.lm_head"
    if model.weight_dtype_for_module(module, "Linear") != "bf16":
        raise NotImplementedError("K3 logits require the unquantized BF16 head")
    common = {"source_component": "vllm_0_30_kimi_k3_fresh_prefill_logits", "launch_policy": "analytical_launch"}
    prefix = f"{pool_name}:r{rank}:kimi-k3-{label}-logits"
    sequences = len(query_start_loc) - 1
    select = WorkloadNode(
        f"{prefix}:select", WorkloadKind.VECTOR, pool_name,
        {"elements": sequences * model.hidden_size, "rows": sequences, "hidden": model.hidden_size,
         "source_rows": query_start_loc[-1]}, dtype="bf16", variant="vllm_logits_row_select", rank=rank,
        attributes=common | {"model_execution_version": "vllm_0_30_0", "index_source": "query_start_loc_int32",
                             "selected_indices": tuple(end - 1 for end in query_start_loc[1:])})
    norm = WorkloadNode(
        f"{prefix}:norm", WorkloadKind.NORM, pool_name,
        {"rows": sequences, "hidden": model.hidden_size}, dtype="bf16", variant="rmsnorm", rank=rank,
        attributes=common | {"rms_norm_eps": source.rms_norm_eps, "module_path": "language_model.model.norm"})
    m, n, k = infer_gemm_dimensions(model, "lm_head", sequences, tp_size=1)
    head = WorkloadNode(
        f"{prefix}:lm_head", WorkloadKind.LM_HEAD, pool_name,
        {"b": 1, "m": m, "n": n, "k": k}, dtype="bf16", variant="gemv" if sequences == 1 else "gemm", rank=rank,
        attributes=common | {"has_bias": False, "position": "lm_head", "module_path": module,
                             "weight_dtype": "bf16", "weight_group_size": 0})
    return WorkloadGraph((select, norm, head), (
        WorkloadEdge(select.node_id, norm.node_id), WorkloadEdge(norm.node_id, head.node_id)))


def vllm030_kimi_k3_tp1_embedding_workload(
    model: ModelConfig, *, input_ids: tuple[int, ...], pool_name: str, rank: int,
) -> WorkloadGraph:
    """INT32 runner IDs -> materialized INT64 IDs -> ordinary BF16 embedding.

    Valid IDs are explicit source inputs. The gather retains both the
    weight-row read and distinct hidden-output write, without cache-hit or
    actual library launch claims. This is TP1, not the fused TP embedding.
    """
    if model.attention_layout != "hybrid_kda_mla" or model.kimi_k3_semantics is None:
        raise ValueError("K3 embedding requires exact hybrid source semantics")
    if (not isinstance(input_ids, tuple) or not input_ids
            or any(type(token) is not int or not 0 <= token < min(model.vocab_size, 2**31) for token in input_ids)):
        raise ValueError("K3 embedding requires nonempty valid INT32 token IDs")
    module = "language_model.model.embed_tokens"
    if model.weight_dtype_for_module(module, "Embedding") != "bf16":
        raise NotImplementedError("K3 embedding requires BF16 unquantized weights")
    rows = len(input_ids)
    prefix = f"{pool_name}:r{rank}:kimi-k3-embedding"
    common = {"source_component": "vllm_0_30_kimi_k3_tp1_embedding", "launch_policy": "analytical_launch"}
    widen = WorkloadNode(
        f"{prefix}:id_widen", WorkloadKind.VECTOR, pool_name, {"rows": rows, "elements": rows},
        dtype="int32", variant="vllm_embedding_id_widen", rank=rank,
        attributes=common | {"embedding_source": "vllm_0_30_unquantized_tp1"})
    gather = WorkloadNode(
        f"{prefix}:gather", WorkloadKind.GATHER, pool_name,
        {"rows": rows, "hidden": model.hidden_size, "elements": rows * model.hidden_size},
        dtype="bf16", variant="gather", rank=rank,
        attributes=common | {"gather_factor": 1, "index_count": rows, "index_bytes": 8,
                             "scattered": True, "materialized_output": True, "module_path": module})
    return WorkloadGraph((widen, gather), (WorkloadEdge(widen.node_id, gather.node_id),))


def vllm030_kimi_k3_fresh_prefill_forward_logits_workload(
    model: ModelConfig, *, input_ids: tuple[int, ...],
    state_indices_by_layer: Mapping[int, int], state_pool_slots: int,
    slot_mapping_by_layer: Mapping[int, tuple[int, ...]], cache_capacity_tokens: int,
    routing_by_layer: Mapping[int, ExpertRoutingResolution],
    shared_experts_stream_token_threshold: int, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Embedding -> complete decoder source stack -> selected-row logits.

    Inherits the explicit TP1/T>=512/fresh/source-backend/cache/route scope
    of the constituent APIs. Ordinary sampling, its output IDs, metadata
    producers, normal capacity and provider/runtime acceptance are outside
    this candidate. Unpriced chunk/SiTU/route/expert/gate calls stay present;
    generic build_workload must still reject incomplete K3 support.
    """
    embedding = vllm030_kimi_k3_tp1_embedding_workload(model, input_ids=input_ids, pool_name=pool_name, rank=rank)
    query = (0, len(input_ids))
    stack = vllm030_kimi_k3_fresh_prefill_decoder_stack_workload(
        model, query_start_loc=query, state_indices_by_layer=state_indices_by_layer,
        state_pool_slots=state_pool_slots, slot_mapping_by_layer=slot_mapping_by_layer,
        cache_capacity_tokens=cache_capacity_tokens, routing_by_layer=routing_by_layer,
        shared_experts_stream_token_threshold=shared_experts_stream_token_threshold, pool_name=pool_name, rank=rank)
    logits = vllm030_kimi_k3_fresh_prefill_logits_workload(model, query_start_loc=query, pool_name=pool_name, rank=rank)
    return WorkloadGraph((*embedding.nodes, *stack.nodes, *logits.nodes), (
        *embedding.edges, *stack.edges, *logits.edges,
        WorkloadEdge(embedding.nodes[-1].node_id, stack.nodes[0].node_id),
        WorkloadEdge(stack.nodes[-1].node_id, logits.nodes[0].node_id)))


def vllm030_kimi_k3_nope_decode_workload(
    model: ModelConfig, *, layer_index: int, slot_mapping: tuple[int, ...],
    kv_lens: tuple[int, ...], cache_capacity_tokens: int, cache_dtype: str,
    aux_stream_available: bool, pool_name: str, rank: int,
) -> WorkloadGraph:
    """One decode token per request, BF16 NoPE MLA TP1 source candidate.

    Source cache contents, slots and sequence lengths are caller inputs.
    The exact unquantized model passes a shared aux stream, so T<512 splits
    the input weight into QKV and gate slices. The two source branches join
    at the output gate; actual CUDA streams/events and the low-latency-vs-
    F.linear backend remain unbound. T>=512 has one merged projection.
    Absorption and V up-projection are BMMs; concat+cache is one source
    call, with no RoPE or extra output copy. Analytical MLA selector reuse
    does not establish actual vLLM0.30 backend/provider dispatch.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 NoPE decode requires exact hybrid source semantics")
    if (type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers
            or source.layer_attention_types[layer_index] != "mla_nope"):
        raise ValueError("K3 NoPE decode requires an actual NoPE MLA layer")
    if (not isinstance(slot_mapping, tuple) or not slot_mapping
            or type(cache_capacity_tokens) is not int or cache_capacity_tokens <= 0
            or any(type(slot) is not int or not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)
            or len(set(slot_mapping)) != len(slot_mapping)):
        raise ValueError("K3 NoPE decode requires distinct valid nonnegative INT64 slots")
    rows = len(slot_mapping)
    if (not isinstance(kv_lens, tuple) or len(kv_lens) != rows
            or any(type(length) is not int or not 0 < length < 2**31
                   or length > cache_capacity_tokens for length in kv_lens)):
        raise ValueError("K3 NoPE decode requires one positive INT32 sequence length within cache capacity per token/request")
    if cache_dtype != "bf16" or model.compute_dtype != "bf16":
        raise NotImplementedError("K3 NoPE decode source requires BF16 activation/cache storage")
    if aux_stream_available is not True or not source.mla_use_output_gate:
        raise NotImplementedError("K3 NoPE decode requires its unquantized shared auxiliary-stream source branch")
    heads, qrank, kvrank, pe, nope, value = (
        model.num_attention_heads, model.q_lora_rank, model.kv_lora_rank,
        model.qk_rope_head_dim, model.qk_nope_head_dim, model.v_head_dim)
    base = f"language_model.model.layers.{layer_index}.self_attn"
    prefix = f"{pool_name}:r{rank}:kimi-k3-nope-decode:layer{layer_index}"
    common = {"source_component": "vllm_0_30_kimi_k3_nope_decode", "launch_policy": "analytical_launch"}
    if model.weight_dtype_for_module(f"{base}.fused_qkv_a_g_proj", "Linear") != "bf16":
        raise NotImplementedError("K3 NoPE decode requires unquantized BF16 projections")

    def projection(position, n, k, b=1):
        return WorkloadNode(
            f"{prefix}:{position}", WorkloadKind.LINEAR, pool_name,
            {"b": b, "m": rows, "n": n, "k": k}, dtype="bf16",
            variant="kimi_k3_nope_bmm" if b != 1 else "gemm", rank=rank,
            attributes=common | {"position": position, "module_path": f"{base}.{position}",
                                 "has_bias": False, "weight_dtype": "bf16", "weight_group_size": 0})

    def stage(name, width):
        return WorkloadNode(
            f"{prefix}:{name}", WorkloadKind.VECTOR, pool_name,
            {"rows": rows, "heads": heads, "q_lora_rank": qrank, "kv_lora_rank": kvrank,
             "elements": rows * width}, dtype="bf16", variant="kimi_k3_nope_decode", rank=rank,
            attributes=common | {"stage": name, "slot_mapping": slot_mapping,
                                 "cache_capacity_tokens": cache_capacity_tokens, "rms_norm_eps": source.rms_norm_eps})

    split_frontend = rows < 512
    if split_frontend:
        packed = projection("qkv_a_weight_slice", qrank + kvrank + pe, model.hidden_size)
        gate_projection = projection("gate_weight_slice", heads * value, model.hidden_size)
        front = (packed, gate_projection)
    else:
        packed = projection("fused_qkv_a_g_proj", qrank + kvrank + pe + heads * value, model.hidden_size)
        gate_projection = packed
        front = (packed,)
    norm = stage("q_kv_norm", qrank + kvrank)
    qb = projection("q_b_proj", heads * model.head_dim, qrank)
    absorb = projection("W_UK_T", kvrank, nope, b=heads)
    concat = stage("query_cache_concat", heads * (kvrank + pe))
    attention = WorkloadNode(
        f"{prefix}:attention", WorkloadKind.FLASH_ATTN, pool_name,
        {"batch": rows, "q_len": 1, "kv_len": max(kv_lens), "qk_head_dim": kvrank + pe,
         "v_head_dim": kvrank, "num_query_heads": heads, "kv_lora_rank": kvrank},
        dtype="bf16", variant="mla_decode", rank=rank,
        attributes=common | {"causal": True, "kv_dist": "uniform", "kv_layout": "mla",
                             "q_lens": (1,) * rows, "kv_lens": kv_lens,
                             "full_cuda_graphs": False, "max_cudagraph_capture_size": 0})
    up = projection("W_UV", value, kvrank, b=heads)
    gate = stage("output_gate", heads * value)
    out = projection("o_proj", model.hidden_size, heads * value)
    nodes = (*front, norm, qb, absorb, concat, attention, up, gate, out)
    pairs = ((packed, norm), (norm, qb), (qb, absorb), (absorb, concat), (qb, concat), (norm, concat),
             (packed, concat), (concat, attention), (attention, up), (up, gate),
             (gate_projection, gate), (gate, out))
    return WorkloadGraph(nodes, (WorkloadEdge(a.node_id, b.node_id) for a, b in pairs))


def vllm030_kimi_k3_moe_decode_layer_workload(
    model: ModelConfig, *, layer_index: int, state_indices: tuple[int, ...] | None,
    state_pool_slots: int, slot_mapping: tuple[int, ...] | None, kv_lens: tuple[int, ...],
    cache_capacity_tokens: int, routing: ExpertRoutingResolution,
    shared_experts_stream_token_threshold: int, aux_streams_available: bool, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Actual configured KDA/NoPE MoE decoder-layer source, both forks retained.

    This source scenario declares available ordinary auxiliary streams.
    KDA uses state indices with no token slots; NoPE uses token slots with
    no recurrent-state index. Native allocator/metadata/provider execution
    remains a separate binding from these explicit source inputs.
    """
    source = model.kimi_k3_semantics
    if source is None or model.attention_layout != "hybrid_kda_mla":
        raise ValueError("K3 decode layer requires exact hybrid source semantics")
    if (type(layer_index) is not int or not model.num_dense_layers <= layer_index < model.num_hidden_layers
            or layer_index % source.moe_layer_frequency != 0):
        raise ValueError("K3 decode layer requires an actual MoE layer")
    if aux_streams_available is not True:
        raise NotImplementedError("K3 decode layer requires explicit ordinary auxiliary streams")
    kind = source.layer_attention_types[layer_index]
    if kind == "kda":
        if state_indices is None or slot_mapping is not None:
            raise ValueError("KDA decode consumes recurrent-state indices, not token slots")
        if (not isinstance(kv_lens, tuple) or len(kv_lens) != len(state_indices)
                or any(type(length) is not int or not 0 < length < 2**31 for length in kv_lens)):
            raise ValueError("KDA decode requires one positive INT32 sequence length per state row")
        attention = vllm030_kimi_k3_triton_decode_workload(
            model, layer_index=layer_index, state_indices=state_indices, state_pool_slots=state_pool_slots,
            conv_state_dtype="bf16", recurrent_state_dtype="fp32", pool_name=pool_name, rank=rank)
        rows = len(state_indices)
    elif kind == "mla_nope":
        if slot_mapping is None or state_indices is not None:
            raise ValueError("NoPE decode consumes token slots, not recurrent-state indices")
        attention = vllm030_kimi_k3_nope_decode_workload(
            model, layer_index=layer_index, slot_mapping=slot_mapping, kv_lens=kv_lens,
            cache_capacity_tokens=cache_capacity_tokens, cache_dtype="bf16",
            aux_stream_available=aux_streams_available, pool_name=pool_name, rank=rank)
        rows = len(slot_mapping)
    else:
        raise NotImplementedError("K3 decode layer contains an unsupported attention type")
    moe = vllm030_kimi_k3_native_marlin_moe_workload(
        model, rows=rows, layer_index=layer_index, routing=routing,
        shared_experts_stream_token_threshold=shared_experts_stream_token_threshold,
        aux_streams_available=aux_streams_available, pool_name=pool_name, rank=rank)
    pre = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="pre_attention", layer_index=layer_index, pool_name=pool_name, rank=rank)
    post = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="post_attention", layer_index=layer_index, pool_name=pool_name, rank=rank)
    attention_roots = {node.node_id for node in attention.nodes} - {edge.destination for edge in attention.edges}
    return WorkloadGraph((pre, *attention.nodes, post, *moe.nodes), (
        *attention.edges, *moe.edges,
        *(WorkloadEdge(pre.node_id, node_id) for node_id in sorted(attention_roots)),
        WorkloadEdge(attention.nodes[-1].node_id, post.node_id),
        *(WorkloadEdge(post.node_id, moe.nodes[index].node_id) for index in (0, 1, 2))))


def vllm030_kimi_k3_single_decode_forward_logits_workload(
    model: ModelConfig, *, input_ids: tuple[int], sequence_length: int,
    state_indices_by_layer: Mapping[int, int], state_pool_slots: int,
    slot_mapping_by_layer: Mapping[int, tuple[int, ...]], cache_capacity_tokens: int,
    routing_by_layer: Mapping[int, ExpertRoutingResolution],
    shared_experts_stream_token_threshold: int, aux_streams_available: bool, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Single-request continuation: embedding ->93 layers -> one-row logits.

    No numerical cache or model output is generated. Histograms, input ID,
    context length and per-layer cache addresses are explicit source inputs.
    Includes ordinary eager stream-event dependencies and final AttnRes;
    the existing single-sequence logits boundary selects row0 before norm.
    Sampler, context/decode backend dispatch and normal Serving are separate.
    """
    source = model.kimi_k3_semantics
    if source is None or model.attention_layout != "hybrid_kda_mla":
        raise ValueError("K3 single decode requires exact hybrid source semantics")
    if (model.num_dense_layers != 1 or source.moe_layer_frequency != 1
            or source.layer_attention_types[0] != "kda"):
        raise NotImplementedError("K3 single decode requires first dense KDA and remaining MoE layers")
    if (not isinstance(input_ids, tuple) or len(input_ids) != 1
            or type(sequence_length) is not int or not 1 < sequence_length < 2**31):
        raise ValueError("K3 single decode requires one continuation token and an INT32 context length")
    if aux_streams_available is not True:
        raise NotImplementedError("K3 single decode requires explicit ordinary auxiliary streams")
    kda = {i for i, kind in enumerate(source.layer_attention_types) if kind == "kda"}
    mla = {i for i, kind in enumerate(source.layer_attention_types) if kind == "mla_nope"}
    if kda | mla != set(range(model.num_hidden_layers)):
        raise NotImplementedError("K3 single decode contains an unsupported attention type")
    for name, values, expected in (("state_indices_by_layer", state_indices_by_layer, kda),
                                    ("slot_mapping_by_layer", slot_mapping_by_layer, mla),
                                    ("routing_by_layer", routing_by_layer, set(range(1, model.num_hidden_layers)))):
        if (not isinstance(values, Mapping) or any(type(key) is not int for key in values) or set(values) != expected):
            raise ValueError(f"K3 single decode requires exactly the actual layer keys in {name}")
    if any(not isinstance(slots, tuple) or len(slots) != 1 for slots in slot_mapping_by_layer.values()):
        raise ValueError("K3 single decode requires one token slot per NoPE layer")
    embedding = vllm030_kimi_k3_tp1_embedding_workload(model, input_ids=input_ids, pool_name=pool_name, rank=rank)
    nodes, edges, previous = list(embedding.nodes), list(embedding.edges), embedding.nodes[-1]
    for layer_index, kind in enumerate(source.layer_attention_types):
        if layer_index == 0:
            layer = vllm030_kimi_k3_dense_decode_layer_workload(
                model, state_indices=(state_indices_by_layer[0],), state_pool_slots=state_pool_slots,
                conv_state_dtype="bf16", recurrent_state_dtype="fp32", pool_name=pool_name, rank=rank)
        else:
            layer = vllm030_kimi_k3_moe_decode_layer_workload(
                model, layer_index=layer_index,
                state_indices=(state_indices_by_layer[layer_index],) if kind == "kda" else None,
                state_pool_slots=state_pool_slots,
                slot_mapping=slot_mapping_by_layer[layer_index] if kind == "mla_nope" else None,
                kv_lens=(sequence_length,), cache_capacity_tokens=cache_capacity_tokens,
                routing=routing_by_layer[layer_index],
                shared_experts_stream_token_threshold=shared_experts_stream_token_threshold,
                aux_streams_available=aux_streams_available, pool_name=pool_name, rank=rank)
        nodes.extend(layer.nodes)
        edges.extend(layer.edges)
        edges.append(WorkloadEdge(previous.node_id, layer.nodes[0].node_id))
        previous = layer.nodes[-1]
    final = vllm030_kimi_k3_attention_residual_node(
        model, rows=1, stage="output", layer_index=None, pool_name=pool_name, rank=rank)
    logits = vllm030_kimi_k3_fresh_prefill_logits_workload(model, query_start_loc=(0, 1), pool_name=pool_name, rank=rank)
    nodes.extend((final, *logits.nodes))
    edges.extend((WorkloadEdge(previous.node_id, final.node_id), WorkloadEdge(final.node_id, logits.nodes[0].node_id),
                  *logits.edges))
    return WorkloadGraph(nodes, edges)


def vllm030_kimi_k3_append_plain_greedy_sampler(
    model: ModelConfig, graph: WorkloadGraph, *, sampling_source: str,
) -> WorkloadGraph:
    """Append an explicit plain greedy source branch to a BF16 K3 logits graph.

    This source profile has all_greedy=True, no penalties, allowed-token
    mask, bad words, active logits processing, thinking-budget holder or requested
    logprobs. No temperature/top-k/top-p/random stage executes. These are
    declared scenario facts, not inferred native SamplingMetadata. Argmax
    returns INT64, so sampled.long() aliases it; the final INT32 conversion
    materializes once and unsqueeze is a view. Argmax services stay unpriced.
    Loaded source-known builtin processors may be inactive or skipped by
    greedy early return; their constructor/state work is not a logits kernel.
    """
    if model.attention_layout != "hybrid_kda_mla" or model.kimi_k3_semantics is None:
        raise ValueError("K3 greedy source requires exact hybrid semantics")
    if sampling_source != "vllm_0_30_plain_greedy_bf16":
        raise NotImplementedError("K3 sampler source supports explicit plain greedy without processors/logprobs only")
    head = graph.nodes[-1]
    if (head.kind is not WorkloadKind.LM_HEAD or head.dtype != "bf16"
            or ("output_dtype" in head.attributes and head.attributes["output_dtype"] not in (None, "bf16"))):
        raise ValueError("K3 sampler source requires complete unquantized BF16 TP1 logits")
    expected = infer_gemm_dimensions(model, "lm_head", head.shape["m"], tp_size=1)
    if (head.shape["b"] != 1 or (head.shape["m"], head.shape["n"], head.shape["k"]) != expected
            or model.vocab_size >= 2**31):
        raise ValueError("K3 sampler source requires complete unquantized BF16 TP1 logits")
    rows, pool, rank = head.shape["m"], head.pool_name, head.rank
    prefix = f"{pool}:r{rank}:kimi-k3-plain-greedy"
    common = {"source_component": "vllm_0_30_kimi_k3_plain_greedy", "launch_policy": "analytical_launch"}
    cast = WorkloadNode(
        f"{prefix}:logits_fp32", WorkloadKind.VECTOR, pool,
        {"elements": rows * model.vocab_size, "rows": rows, "vocab": model.vocab_size},
        dtype="bf16", variant="vllm_logits_fp32_cast", rank=rank,
        attributes=common | {"sampling_source": "vllm_0_30_plain_sampler_bf16"})
    argmax = WorkloadNode(
        f"{prefix}:argmax", WorkloadKind.VECTOR, pool,
        {"elements": rows * model.vocab_size, "rows": rows, "vocab": model.vocab_size},
        dtype="fp32", variant="vllm_plain_greedy_sample", rank=rank,
        attributes=common | {"sampling_source": sampling_source, "stage": "argmax"})
    narrow = WorkloadNode(
        f"{prefix}:output_int32", WorkloadKind.VECTOR, pool,
        {"elements": rows, "rows": rows, "vocab": model.vocab_size},
        dtype="int64", variant="vllm_plain_greedy_sample", rank=rank,
        attributes=common | {"sampling_source": sampling_source, "stage": "output_int32"})
    return WorkloadGraph((*graph.nodes, cast, argmax, narrow), (
        *graph.edges, WorkloadEdge(head.node_id, cast.node_id), WorkloadEdge(cast.node_id, argmax.node_id),
        WorkloadEdge(argmax.node_id, narrow.node_id)), graph.block_repeats)


def vllm030_kimi_k3_nope_mixed_workload(
    model: ModelConfig, *, layer_index: int, query_start_loc: tuple[int, int, int],
    decode_kv_len: int, slot_mapping: tuple[int, ...], cache_capacity_tokens: int,
    cache_dtype: str, pool_name: str, rank: int,
) -> WorkloadGraph:
    """One decode then one fresh prefill, shared BF16 NoPE source frontend.

    T>=512 selects merged QKV/gate without an aux-stream capability guess.
    Prefill has no context. Slots, cached decode data and length are declared
    source inputs. The caller runs prefill first despite decode-first token
    order. Common norm/Q-B/gate/O-proj run once on all rows; KV-B is prefill
    only and Q absorption/value BMM are decode only. Attention writes into
    disjoint slices of one output buffer; no concatenation/copy is emitted.
    This is not a native metadata, backend, provider or runtime assertion.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 NoPE mixed requires exact hybrid source semantics")
    if (type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers
            or source.layer_attention_types[layer_index] != "mla_nope"):
        raise ValueError("K3 NoPE mixed requires an actual NoPE MLA layer")
    if (not isinstance(query_start_loc, tuple) or len(query_start_loc) != 3
            or any(type(value) is not int for value in query_start_loc)
            or query_start_loc[:2] != (0, 1) or not 512 <= query_start_loc[2] < 2**31):
        raise ValueError("K3 NoPE mixed requires one decode-first row and one fresh prefill with total T>=512")
    rows, prefill_rows = query_start_loc[2], query_start_loc[2] - 1
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens <= 0
            or type(decode_kv_len) is not int or not 1 < decode_kv_len < 2**31
            or decode_kv_len + prefill_rows > cache_capacity_tokens
            or not isinstance(slot_mapping, tuple) or len(slot_mapping) != rows
            or any(type(slot) is not int or not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)
            or len(set(slot_mapping)) != rows):
        raise ValueError("K3 NoPE mixed requires distinct INT64 writes and capacity for both fresh/past token sets")
    if cache_dtype != "bf16" or model.compute_dtype != "bf16" or not source.mla_use_output_gate:
        raise NotImplementedError("K3 NoPE mixed source requires BF16 storage and its configured output gate")
    heads, qrank, kvrank = model.num_attention_heads, model.q_lora_rank, model.kv_lora_rank
    base = f"language_model.model.layers.{layer_index}.self_attn"
    prefix = f"{pool_name}:r{rank}:kimi-k3-nope-mixed:layer{layer_index}"
    common = {"source_component": "vllm_0_30_kimi_k3_nope_mixed", "launch_policy": "analytical_launch"}

    def projection(position, count):
        m, n, k = infer_gemm_dimensions(model, position, count, tp_size=1)
        if model.weight_dtype_for_module(f"{base}.{position}", "Linear") != "bf16":
            raise NotImplementedError("K3 NoPE mixed requires ordinary BF16 LinearBase projections")
        return WorkloadNode(
            f"{prefix}:{position}", WorkloadKind.LINEAR, pool_name,
            {"b": 1, "m": m, "n": n, "k": k}, dtype="bf16", variant="gemm", rank=rank,
            attributes=common | {"position": position, "module_path": f"{base}.{position}",
                                 "weight_dtype": "bf16", "weight_group_size": 0, "has_bias": False})

    def stage(name, count, width, slots):
        return WorkloadNode(
            f"{prefix}:{name}", WorkloadKind.VECTOR, pool_name,
            {"rows": count, "heads": heads, "q_lora_rank": qrank, "kv_lora_rank": kvrank,
             "elements": count * width}, dtype="bf16", variant="kimi_k3_nope_mixed", rank=rank,
            attributes=common | {"stage": name, "slot_mapping": slots,
                                 "cache_capacity_tokens": cache_capacity_tokens, "rms_norm_eps": source.rms_norm_eps})

    def bmm(position, n, k):
        return WorkloadNode(
            f"{prefix}:{position}", WorkloadKind.LINEAR, pool_name,
            {"b": heads, "m": 1, "n": n, "k": k}, dtype="bf16", variant="kimi_k3_nope_bmm", rank=rank,
            attributes=common | {"position": position, "module_path": f"{base}.{position}"})

    def attention(suffix, count, length, qk, value, variant):
        return WorkloadNode(
            f"{prefix}:{suffix}", WorkloadKind.FLASH_ATTN, pool_name,
            {"batch": 1, "q_len": count, "kv_len": length, "qk_head_dim": qk, "v_head_dim": value,
             "num_query_heads": heads, "kv_lora_rank": kvrank}, dtype="bf16", variant=variant, rank=rank,
            attributes=common | {"causal": True, "kv_dist": "uniform", "kv_layout": "mla",
                                 "q_lens": (count,), "kv_lens": (length,),
                                 "full_cuda_graphs": False, "max_cudagraph_capture_size": 0})

    packed, qb, out = (projection(position, rows) for position in ("fused_qkv_a_g_proj", "q_b_proj", "o_proj"))
    norm = stage("q_kv_norm", rows, qrank + kvrank, slot_mapping)
    kvb = projection("kv_b_proj", prefill_rows)
    key = stage("key_cache_concat", prefill_rows, heads * model.head_dim + 576, slot_mapping[1:])
    prefill = attention("prefill_attention", prefill_rows, prefill_rows, model.head_dim, model.v_head_dim, "mla_prefill")
    absorb = bmm("W_UK_T", kvrank, model.qk_nope_head_dim)
    query = stage("query_cache_concat", 1, heads * 576, slot_mapping[:1])
    decode = attention("decode_attention", 1, decode_kv_len, 576, kvrank, "mla_decode")
    up = bmm("W_UV", model.v_head_dim, kvrank)
    gate = stage("output_gate", rows, heads * model.v_head_dim, slot_mapping)
    nodes = (packed, norm, qb, kvb, key, prefill, absorb, query, decode, up, gate, out)
    pairs = ((packed, norm), (norm, qb), (norm, kvb), (kvb, key), (norm, key), (packed, key),
             (qb, prefill), (kvb, prefill), (key, prefill), (qb, absorb),
             (absorb, query), (qb, query), (norm, query), (packed, query), (query, decode),
             (decode, up), (prefill, gate), (up, gate), (packed, gate), (gate, out))
    return WorkloadGraph(nodes, (
        *(WorkloadEdge(a.node_id, b.node_id) for a, b in pairs),
        WorkloadEdge(qb.node_id, kvb.node_id, "barrier"),
        WorkloadEdge(prefill.node_id, absorb.node_id, "barrier")))


def vllm030_kimi_k3_mixed_forward_logits_workload(
    model: ModelConfig, *, input_ids: tuple[int, ...], decode_kv_len: int,
    state_indices_by_layer: Mapping[int, tuple[int, int]], state_pool_slots: int,
    slot_mapping_by_layer: Mapping[int, tuple[int, ...]], cache_capacity_tokens: int,
    routing_by_layer: Mapping[int, ExpertRoutingResolution],
    shared_experts_stream_token_threshold: int, aux_streams_available: bool, pool_name: str, rank: int,
) -> WorkloadGraph:
    """Decode-first one-token continuation plus fresh512+ prompt to two logits.

    All93 layers share the packed batch. KDA sends both segments to chunk;
    NoPE runs fresh prefill before decode-only absorption. Final last-row
    selection picks rows0 andT-1 before normalization and the two-row head.
    Explicit IDs, cache addresses, flags and routes are scenario inputs,
    with no native metadata/provider or numerical output assertion.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 mixed stack requires exact hybrid source semantics")
    if (model.num_dense_layers != 1 or source.moe_layer_frequency != 1
            or source.layer_attention_types[0] != "kda"):
        raise NotImplementedError("K3 mixed stack requires first dense KDA and remaining MoE layers")
    if (not isinstance(input_ids, tuple) or not 513 <= len(input_ids) < 2**31 or (len(input_ids) - 1) % 64
            or type(decode_kv_len) is not int or not 1 < decode_kv_len < 2**31):
        raise ValueError("K3 mixed stack requires one decode-first ID and a fresh512+ full64-chunk prompt")
    if aux_streams_available is not True:
        raise NotImplementedError("K3 mixed stack requires explicit ordinary auxiliary streams")
    kda = {i for i, kind in enumerate(source.layer_attention_types) if kind == "kda"}
    mla = {i for i, kind in enumerate(source.layer_attention_types) if kind == "mla_nope"}
    if kda | mla != set(range(model.num_hidden_layers)):
        raise NotImplementedError("K3 mixed stack contains an unsupported attention type")
    for name, values, expected in (("state_indices_by_layer", state_indices_by_layer, kda),
                                    ("slot_mapping_by_layer", slot_mapping_by_layer, mla),
                                    ("routing_by_layer", routing_by_layer, set(range(1, model.num_hidden_layers)))):
        if not isinstance(values, Mapping) or any(type(key) is not int for key in values) or set(values) != expected:
            raise ValueError(f"K3 mixed stack requires exactly the actual layer keys in {name}")
    rows, query = len(input_ids), (0, 1, len(input_ids))
    embedding = vllm030_kimi_k3_tp1_embedding_workload(model, input_ids=input_ids, pool_name=pool_name, rank=rank)
    nodes, edges, previous = list(embedding.nodes), list(embedding.edges), embedding.nodes[-1]
    for layer_index, kind in enumerate(source.layer_attention_types):
        if kind == "kda":
            attention = vllm030_kimi_k3_triton_mixed_workload(
                model, layer_index=layer_index, query_start_loc=query,
                state_indices=state_indices_by_layer[layer_index], state_pool_slots=state_pool_slots,
                has_initial_state=(True, False), conv_state_dtype="bf16", recurrent_state_dtype="fp32",
                pool_name=pool_name, rank=rank)
        elif kind == "mla_nope":
            attention = vllm030_kimi_k3_nope_mixed_workload(
                model, layer_index=layer_index, query_start_loc=query, decode_kv_len=decode_kv_len,
                slot_mapping=slot_mapping_by_layer[layer_index], cache_capacity_tokens=cache_capacity_tokens,
                cache_dtype="bf16", pool_name=pool_name, rank=rank)
        else:
            raise AssertionError("validated K3 attention type became unsupported")
        pre = vllm030_kimi_k3_attention_residual_node(
            model, rows=rows, stage="pre_attention", layer_index=layer_index, pool_name=pool_name, rank=rank)
        post = vllm030_kimi_k3_attention_residual_node(
            model, rows=rows, stage="post_attention", layer_index=layer_index, pool_name=pool_name, rank=rank)
        if layer_index == 0:
            mlp = vllm030_kimi_k3_ordinary_mlp_workload(
                model, rows=rows, layer_index=layer_index, role="dense", pool_name=pool_name, rank=rank)
        else:
            mlp = vllm030_kimi_k3_native_marlin_moe_workload(
                model, rows=rows, layer_index=layer_index, routing=routing_by_layer[layer_index],
                shared_experts_stream_token_threshold=shared_experts_stream_token_threshold,
                aux_streams_available=aux_streams_available, pool_name=pool_name, rank=rank)
        attention_roots = {node.node_id for node in attention.nodes} - {edge.destination for edge in attention.edges}
        mlp_roots = {node.node_id for node in mlp.nodes} - {edge.destination for edge in mlp.edges}
        nodes.extend((pre, *attention.nodes, post, *mlp.nodes))
        edges.extend((*attention.edges, *mlp.edges, WorkloadEdge(previous.node_id, pre.node_id),
                      *(WorkloadEdge(pre.node_id, node_id) for node_id in sorted(attention_roots)),
                      WorkloadEdge(attention.nodes[-1].node_id, post.node_id),
                      *(WorkloadEdge(post.node_id, node_id) for node_id in sorted(mlp_roots))))
        previous = mlp.nodes[-1]
    final = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="output", layer_index=None, pool_name=pool_name, rank=rank)
    logits = _kimi_k3_selected_logits_workload(model, query_start_loc=query, label="mixed", pool_name=pool_name, rank=rank)
    nodes.extend((final, *logits.nodes))
    edges.extend((WorkloadEdge(previous.node_id, final.node_id), WorkloadEdge(final.node_id, logits.nodes[0].node_id),
                  *logits.edges))
    return WorkloadGraph(nodes, edges)


def vllm030_kimi_k3_nope_context_prefill_workload(
    model: ModelConfig, *, layer_index: int, slot_mapping: tuple[int, ...], context_length: int,
    context_block_table: tuple[int, ...], block_size: int, cache_capacity_tokens: int,
    cache_dtype: str, prefill_source: str, pool_name: str, rank: int,
    context_chunk_lengths: tuple[int, ...] | None = None, backend_supports_out: bool | None = None,
) -> WorkloadGraph:
    """One paged prefix partitioned into complete/continuation context chunks.

    This is an explicit source scenario with a declared FA3 new-token
    projection. Noncausal context attention exports logical work without
    binding a launch. No native metadata, prefix hit or provider is inferred.
    The history starts at0, covers one request and all query rows, has no
    empty slices and cannot alias the new writes. Multiple chunks require
    explicit supports_out capability; only their first partial may write
    directly into the accumulator. Continuations merge output and FP32 LSE.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 NoPE context requires exact hybrid source semantics")
    if (type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers
            or source.layer_attention_types[layer_index] != "mla_nope"):
        raise ValueError("K3 NoPE context requires an actual NoPE MLA layer")
    if (not isinstance(slot_mapping, tuple) or not 512 <= len(slot_mapping) < 2**31
            or type(context_length) is not int or not 0 < context_length < 2**31
            or type(block_size) is not int or not 0 < block_size < 2**31 or block_size & (block_size - 1)
            or type(cache_capacity_tokens) is not int or cache_capacity_tokens <= 0
            or cache_capacity_tokens % block_size or context_length + len(slot_mapping) > cache_capacity_tokens
            or any(type(slot) is not int or not 0 <= slot < min(cache_capacity_tokens, 2**63) for slot in slot_mapping)
            or len(set(slot_mapping)) != len(slot_mapping)
            or not isinstance(context_block_table, tuple) or len(context_block_table) != -(-context_length // block_size)
            or any(type(index) is not int or not 0 <= index < min(cache_capacity_tokens // block_size, 2**31)
                   for index in context_block_table)
            or len(set(context_block_table)) != len(context_block_table)):
        raise ValueError("K3 NoPE context requires complete INT32 history geometry and distinct valid INT64 new slots")
    history = {context_block_table[position // block_size] * block_size + position % block_size
               for position in range(context_length)}
    if history.intersection(slot_mapping):
        raise ValueError("K3 NoPE context new writes must not overwrite gathered history")
    if (cache_dtype != "bf16" or model.compute_dtype != "bf16" or not source.mla_use_output_gate
            or prefill_source != "declared_fa3_bf16"):
        raise NotImplementedError("K3 NoPE context requires BF16 cache/output gate and explicit declared_fa3_bf16 source")
    chunks = (context_length,) if context_chunk_lengths is None else context_chunk_lengths
    if (not isinstance(chunks, tuple) or not chunks
            or any(type(length) is not int or length <= 0 for length in chunks)
            or sum(chunks) != context_length or any(length % block_size for length in chunks[:-1])):
        raise ValueError("K3 context chunks must partition the complete prefix with aligned continuation starts")
    if backend_supports_out is not None and type(backend_supports_out) is not bool:
        raise ValueError("K3 context backend supports_out must be an explicit boolean or unspecified for a single chunk")
    chunked = len(chunks) > 1
    if chunked and backend_supports_out is None:
        raise NotImplementedError("K3 multiple context chunks require explicit backend supports_out capability")
    rows, heads = len(slot_mapping), model.num_attention_heads
    base = f"language_model.model.layers.{layer_index}.self_attn"
    prefix = f"{pool_name}:r{rank}:kimi-k3-nope-context:layer{layer_index}"
    common = {"source_component": "vllm_0_30_kimi_k3_nope_context_prefill", "launch_policy": "analytical_launch"}

    def projection(label, position, count):
        m, n, k = infer_gemm_dimensions(model, position, count, tp_size=1)
        module = f"{base}.{position}"
        if model.weight_dtype_for_module(module, "Linear") != "bf16":
            raise NotImplementedError("K3 context projections require ordinary BF16 LinearBase")
        return WorkloadNode(
            f"{prefix}:{label}", WorkloadKind.LINEAR, pool_name,
            {"b": 1, "m": m, "n": n, "k": k}, dtype="bf16", variant="gemm", rank=rank,
            attributes=common | {"position": position, "module_path": module, "weight_dtype": "bf16",
                                 "weight_group_size": 0, "has_bias": False})

    def stage(name, elements, *, label=None, chunk_rows=None, start=0):
        attributes = common | {"stage": name, "slot_mapping": slot_mapping,
                               "context_block_table": context_block_table,
                               "cache_capacity_tokens": cache_capacity_tokens, "rms_norm_eps": source.rms_norm_eps}
        if chunked:
            attributes |= {"context_start": start, "context_length": context_length}
        return WorkloadNode(
            f"{prefix}:{name if label is None else label}", WorkloadKind.VECTOR, pool_name,
            {"rows": rows, "context_rows": context_length if chunk_rows is None else chunk_rows,
             "heads": heads, "block_size": block_size,
             "q_lora_rank": model.q_lora_rank, "kv_lora_rank": model.kv_lora_rank, "elements": elements},
            dtype="fp32" if name == "context_lse_copy" else "bf16", variant="kimi_k3_nope_context", rank=rank,
            attributes=attributes)

    def attention(label, length, causal, variant, *, start=None, output_target=None):
        attributes = common | {"causal": causal, "kv_dist": "uniform", "kv_layout": "mla",
                               "q_lens": (rows,), "kv_lens": (length,), "return_softmax_lse": True,
                               "full_cuda_graphs": False, "max_cudagraph_capture_size": 0}
        if start is not None:
            attributes["context_start"] = start
        if output_target is not None:
            attributes["output_target"] = output_target
        return WorkloadNode(
            f"{prefix}:{label}", WorkloadKind.FLASH_ATTN, pool_name,
            {"batch": 1, "q_len": rows, "kv_len": length, "qk_head_dim": 192, "v_head_dim": 128,
             "num_query_heads": heads, "kv_lora_rank": 512}, dtype="bf16", variant=variant, rank=rank,
            attributes=attributes)

    packed = projection("fused_qkv_a_g_proj", "fused_qkv_a_g_proj", rows)
    norm = stage("q_kv_norm", rows * 2048)
    qb = projection("q_b_proj", "q_b_proj", rows)
    kvb = projection("new_kv_b_proj", "kv_b_proj", rows)
    key = stage("key_cache_concat", rows * (heads * 192 + 576))
    suffix = attention("new_token_attention", rows, True, "mla_prefill")
    nodes = [packed, norm, qb, kvb, key, suffix]
    pairs = [(packed, norm), (norm, qb), (norm, kvb), (qb, key), (kvb, key), (norm, key), (packed, key),
             (key, suffix), (qb, suffix), (kvb, suffix)]
    barriers = [(qb, kvb)]
    start, previous = 0, suffix
    for index, length in enumerate(chunks):
        label = f"chunk{index}:" if chunked else ""
        gather = stage("context_gather", length * 576, label=f"{label}context_gather", chunk_rows=length, start=start)
        context_kvb = projection(f"{label}context_kv_b_proj", "kv_b_proj", length)
        context_key = stage("context_key_concat", length * heads * 192,
                            label=f"{label}context_key_concat", chunk_rows=length, start=start)
        context_attention = attention(
            f"{label}context_attention", length, False, "kimi_k3_nope_context_attention",
            start=start if chunked else None,
            output_target=("context_accumulator" if index == 0 and backend_supports_out else "chunk_partial")
                          if chunked else None)
        nodes.extend((gather, context_kvb, context_key, context_attention))
        pairs.extend(((gather, context_kvb), (context_kvb, context_key), (gather, context_key),
                      (qb, context_attention), (context_key, context_attention), (context_kvb, context_attention)))
        barriers.append((previous, gather))
        if not chunked:
            current = context_attention
        elif index == 0:
            lse_copy = stage("context_lse_copy", rows * heads,
                             label=f"{label}context_lse_copy", chunk_rows=length, start=start)
            if backend_supports_out:
                nodes.append(lse_copy)
                pairs.append((context_attention, lse_copy))
            else:
                output_copy = stage("context_output_copy", rows * heads * 128,
                                    label=f"{label}context_output_copy", chunk_rows=length, start=start)
                nodes.extend((output_copy, lse_copy))
                pairs.extend(((context_attention, output_copy), (context_attention, lse_copy)))
                barriers.append((output_copy, lse_copy))
            current = lse_copy
        else:
            current = stage("context_partial_merge", rows * heads * 128,
                            label=f"{label}context_partial_merge", chunk_rows=length, start=start)
            nodes.append(current)
            pairs.extend(((previous, current), (context_attention, current)))
        previous, start = current, start + length
    merge = stage("merge", rows * heads * 128)
    gate = stage("output_gate", rows * heads * 128)
    out = projection("o_proj", "o_proj", rows)
    nodes.extend((merge, gate, out))
    pairs.extend(((suffix, merge), (previous, merge), (merge, gate), (packed, gate), (gate, out)))
    return WorkloadGraph(nodes, (
        *(WorkloadEdge(a.node_id, b.node_id) for a, b in pairs),
        *(WorkloadEdge(a.node_id, b.node_id, "barrier") for a, b in barriers)))


def vllm030_kimi_k3_context_prefill_forward_logits_workload(
    model: ModelConfig, *, input_ids: tuple[int, ...], context_length: int,
    state_indices_by_layer: Mapping[int, int], state_pool_slots: int,
    slot_mapping_by_layer: Mapping[int, tuple[int, ...]],
    context_block_tables_by_layer: Mapping[int, tuple[int, ...]], block_size: int,
    cache_capacity_tokens: int, routing_by_layer: Mapping[int, ExpertRoutingResolution],
    shared_experts_stream_token_threshold: int, aux_streams_available: bool,
    prefill_source: str, pool_name: str, rank: int,
    context_chunk_lengths: tuple[int, ...] | None = None, backend_supports_out: bool | None = None,
) -> WorkloadGraph:
    """Initialized single-request context ->93 layers -> last-row BF16 logits.

    Prefix chunks cover all new query rows of this request. KDA consumes its
    existing per-layer conv/recurrent state; NoPE gathers each declared
    historical paged latent and merges two attention partials. New input
    IDs, source routes and addresses are declarations, not a native trace.
    """
    source = model.kimi_k3_semantics
    if model.attention_layout != "hybrid_kda_mla" or source is None:
        raise ValueError("K3 context stack requires exact hybrid source semantics")
    if (model.num_dense_layers != 1 or source.moe_layer_frequency != 1
            or source.layer_attention_types[0] != "kda"):
        raise NotImplementedError("K3 context stack requires first dense KDA and remaining MoE layers")
    if (not isinstance(input_ids, tuple) or not 512 <= len(input_ids) < 2**31 or len(input_ids) % 64
            or type(context_length) is not int or not 0 < context_length < 2**31
            or context_length + len(input_ids) > model.max_position_embeddings):
        raise ValueError("K3 context stack requires a full64-chunk query512+ and positive prefix within the configured length")
    if aux_streams_available is not True:
        raise NotImplementedError("K3 context stack requires explicit ordinary auxiliary streams")
    kda = {i for i, kind in enumerate(source.layer_attention_types) if kind == "kda"}
    mla = {i for i, kind in enumerate(source.layer_attention_types) if kind == "mla_nope"}
    if kda | mla != set(range(model.num_hidden_layers)):
        raise NotImplementedError("K3 context stack contains an unsupported attention type")
    for name, values, expected in (("state_indices_by_layer", state_indices_by_layer, kda),
                                    ("slot_mapping_by_layer", slot_mapping_by_layer, mla),
                                    ("context_block_tables_by_layer", context_block_tables_by_layer, mla),
                                    ("routing_by_layer", routing_by_layer, set(range(1, model.num_hidden_layers)))):
        if not isinstance(values, Mapping) or any(type(key) is not int for key in values) or set(values) != expected:
            raise ValueError(f"K3 context stack requires exactly the actual layer keys in {name}")
    if any(not isinstance(slots, tuple) or len(slots) != len(input_ids) for slots in slot_mapping_by_layer.values()):
        raise ValueError("K3 context stack requires one new slot per query token on every NoPE layer")
    rows = len(input_ids)
    embedding = vllm030_kimi_k3_tp1_embedding_workload(model, input_ids=input_ids, pool_name=pool_name, rank=rank)
    nodes, edges, previous = list(embedding.nodes), list(embedding.edges), embedding.nodes[-1]
    for layer, kind in enumerate(source.layer_attention_types):
        if kind == "kda":
            attention = vllm030_kimi_k3_triton_context_prefill_workload(
                model, layer_index=layer, query_start_loc=(0, rows), state_index=state_indices_by_layer[layer],
                state_pool_slots=state_pool_slots, has_initial_state=True,
                conv_state_dtype="bf16", recurrent_state_dtype="fp32", pool_name=pool_name, rank=rank)
        elif kind == "mla_nope":
            attention = vllm030_kimi_k3_nope_context_prefill_workload(
                model, layer_index=layer, slot_mapping=slot_mapping_by_layer[layer], context_length=context_length,
                context_block_table=context_block_tables_by_layer[layer], block_size=block_size,
                cache_capacity_tokens=cache_capacity_tokens, cache_dtype="bf16", prefill_source=prefill_source,
                pool_name=pool_name, rank=rank, context_chunk_lengths=context_chunk_lengths,
                backend_supports_out=backend_supports_out)
        else:
            raise AssertionError("validated K3 attention type became unsupported")
        pre = vllm030_kimi_k3_attention_residual_node(
            model, rows=rows, stage="pre_attention", layer_index=layer, pool_name=pool_name, rank=rank)
        post = vllm030_kimi_k3_attention_residual_node(
            model, rows=rows, stage="post_attention", layer_index=layer, pool_name=pool_name, rank=rank)
        mlp = (vllm030_kimi_k3_ordinary_mlp_workload(
            model, rows=rows, layer_index=layer, role="dense", pool_name=pool_name, rank=rank) if layer == 0 else
            vllm030_kimi_k3_native_marlin_moe_workload(
                model, rows=rows, layer_index=layer, routing=routing_by_layer[layer],
                shared_experts_stream_token_threshold=shared_experts_stream_token_threshold,
                aux_streams_available=aux_streams_available, pool_name=pool_name, rank=rank))
        attention_roots = {node.node_id for node in attention.nodes} - {edge.destination for edge in attention.edges}
        mlp_roots = {node.node_id for node in mlp.nodes} - {edge.destination for edge in mlp.edges}
        nodes.extend((pre, *attention.nodes, post, *mlp.nodes))
        edges.extend((*attention.edges, *mlp.edges, WorkloadEdge(previous.node_id, pre.node_id),
                      *(WorkloadEdge(pre.node_id, node_id) for node_id in sorted(attention_roots)),
                      WorkloadEdge(attention.nodes[-1].node_id, post.node_id),
                      *(WorkloadEdge(post.node_id, node_id) for node_id in sorted(mlp_roots))))
        previous = mlp.nodes[-1]
    final = vllm030_kimi_k3_attention_residual_node(
        model, rows=rows, stage="output", layer_index=None, pool_name=pool_name, rank=rank)
    logits = _kimi_k3_selected_logits_workload(
        model, query_start_loc=(0, rows), label="context-prefill", pool_name=pool_name, rank=rank)
    nodes.extend((final, *logits.nodes))
    edges.extend((WorkloadEdge(previous.node_id, final.node_id), WorkloadEdge(final.node_id, logits.nodes[0].node_id),
                  *logits.edges))
    return WorkloadGraph(nodes, edges)
