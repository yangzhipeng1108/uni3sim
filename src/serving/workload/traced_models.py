"""Adapted model forwards executed on lightweight symbolic operators only.

Qwen3 source: vLLM 0.30 model_executor/models/qwen3.py (attention/decoder)
and qwen2.py (Qwen2MLP, Qwen2Model). PP1, no auxiliary hidden states,
unquantized BF16, source vllm_c CUDA in-place Norm/RoPE profile. Constructors replace
weight loading/runtime modules; tensor .view is named .view_as_shape here.
V4 FFN/input preparation adapt vllm/models/deepseek_v4/nvidia/model.py and
attention.py for a declared synchronous modular, pure-text TP/DP/EP/PP source.
GLM BF16/official FP8 adapts deepseek_v32/nvidia/model.py and attention.py, reusing the
DeepseekV2 MLP body and modular Runner frames for an explicit all-decode
scope, ordinary all-reduce/norm and EP1/EP8 with SP/MTP off; its optional target
chain shares the typed TP logits/greedy primitives.
K2.6 reuses the ordinary MLA wrapper and BF16 dense/shared modules; its
explicit CP1 all-decode FlashMLA source caller does not bind a GPU provider.
No torch import, real tensor allocation, CUDA access or numerical forward.
"""

from contextlib import contextmanager
import weakref

from src.kernel.memory_access import MemoryAllocation

from .model_config import ModelConfig, is_validated_qwen3_pipeline_model_scope, is_validated_qwen3_moe_ep2_model_scope, is_qwen3_moe_pipeline_model_scope, is_glm_fp8_tensor_model_scope, is_deepseek_v4_parallel_model_scope, is_glm_mtp_tensor_model_scope
from dataclasses import dataclass, replace

from .traced_forward import ForwardActivationGroup, ForwardCollectiveGroup, ForwardKVCacheGroup, ForwardPipelineGroup, ForwardTrace, ForwardWorkspace, ForwardSourceProgram, SourceProgramCache, ForwardCollectiveSourceProgram, ForwardLocalGroup
from .traced_operators import (
    gqa_cache_shape, gqa_full_source_metadata,
    SymbolicAttention, SymbolicEmbedding, SymbolicGather, SymbolicGreedy,
    SymbolicLinear, SymbolicRowParallelLinear, SymbolicBatchedLinear, SymbolicDSANormRope, SymbolicDSAQuery,
    SymbolicConcat, SymbolicCopy, SymbolicDSAIndexer, SymbolicDSASparseAttention,
    SymbolicFloat32Cast, SymbolicBiasedSigmoidTopK, SymbolicPositionZeroMask,
    SymbolicMLACacheUpdate, SymbolicMLALatentDecode, SymbolicAttentionStateMerge,
    SymbolicGroupedRouting, SymbolicRoutedExperts,
    SymbolicSoftmaxTopK, SymbolicUnquantizedExperts, SymbolicExpertOwnerMask, unquantized_expert_source_metadata,
    VLLM030_GPU_MODEL_RUNNER_TOPK_CONTEXT,
    SymbolicK3AttentionResidual, SymbolicK3SiTU, SymbolicK3Routing, SymbolicK3Experts, SymbolicK3Stage,
    SymbolicAllReduceRMSNorm,
    SymbolicLogitsRowSelect, SymbolicTPLogitsProcessor,
    SymbolicRMSNorm, SymbolicRotary, SymbolicSiluAndMul,
    SymbolicFillZero, SymbolicAdd, SymbolicScalarMultiply, SymbolicBlockFP8Linear, SymbolicV4GroupedOutput, SymbolicV4InverseRope,
    SymbolicV4QKVNorm, SymbolicV4RoutedExperts, SymbolicV4Routing,
    SymbolicV4Compressor, SymbolicV4KVInsert, SymbolicV4ShortTopK, SymbolicV4SparseAttention,
    SymbolicV4IndexerQuery, SymbolicV4ScoredTopK,
    SymbolicMHC, SymbolicMHCPipelineInput, SymbolicVocabParallelEmbedding, SymbolicV4AuxHiddenMean,
    _dsa_source_phase, _dsa_sm90_k5_flattening,
)
from .workload_graph import WorkloadKind, WorkloadEdge
from .rope_dispatch import ROPE_PROGRAM_FUSED, vllm_rope_program


class QwenMLP:
    def __init__(self, trace, model, pool, prefix, *, group=None, collective=None, collective_inplace=False,
                 model_execution_version=None):
        tp_size = 1 if group is None else len(group.traces)
        if model.intermediate_size % tp_size:
            raise ValueError("dense intermediate width must divide the TP group")
        intermediate = model.intermediate_size // tp_size
        self.gate_up_proj = SymbolicLinear(trace, prefix + ":gate-up", pool,
            model.hidden_size, 2 * intermediate)
        self.act_fn = SymbolicSiluAndMul(trace, prefix + ":silu-mul", pool,
            model_execution_version=model_execution_version)
        self.down_proj = SymbolicLinear(trace, prefix + ":down", pool,
            intermediate, model.hidden_size)
        if group is not None:
            self.down_proj = SymbolicRowParallelLinear(self.down_proj, group, collective, inplace=collective_inplace)

    def __call__(self, x):
        # Qwen2MLP.forward, reused by Qwen3: preserve gate_up's local lifetime.
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def maybe_fused_act_quant(act_fn, x, linear):
    """Unquantized branch of vLLM's helper; never infer a quant producer.

    SymbolicLinear declares an unquantized weight/input contract. The source
    helper's get_input_quant_key is therefore None and returns act_fn(x).
    Quantized consumers require their own explicit fused/source program.
    """
    consumer = linear.projection if type(linear) is SymbolicRowParallelLinear else linear
    if type(consumer) is not SymbolicLinear:
        raise NotImplementedError("symbolic activation/quant fusion requires an explicit consumer quantization contract")
    return act_fn(x)


class LlamaMLP(QwenMLP):
    """Keep LlamaMLP.forward's overwritten-local lifetime, not Qwen's alias."""

    def __call__(self, x):
        x, _ = self.gate_up_proj(x)
        x = maybe_fused_act_quant(self.act_fn, x, self.down_proj)
        x, _ = self.down_proj(x)
        return x


class DeepseekV2MLP(QwenMLP):
    """The same source forward body, with TP-sharded dense/shared widths.

    The constructor declares whether down_proj owns TP reduction or the
    caller keeps its partial. Ordinary SiluAndMul has no V4 clamp.
    """

    def __init__(self, trace, model, pool, prefix, *, intermediate_size, tp_size, module_path,
                 reduce_results, reduction_group=None, reduction_config=None):
        if (model.compute_dtype != "bf16"
                or type(intermediate_size) is not int or intermediate_size <= 0
                or type(tp_size) is not int or tp_size <= 0 or intermediate_size % tp_size):
            raise NotImplementedError("symbolic DeepSeek dense/shared MLP requires whole BF16 TP shards")
        if type(reduce_results) is not bool:
            raise ValueError("MLP reduce_results must explicitly declare its source branch")
        if reduce_results:
            if (not isinstance(reduction_group, ForwardCollectiveGroup)
                    or len(reduction_group.traces) != tp_size):
                raise ValueError("MLP reduction must cover its complete TP group")
        elif reduction_group is not None or reduction_config is not None:
            raise ValueError("partial MLP output must not carry reduction metadata")
        local = intermediate_size // tp_size
        self.gate_up_proj = _source_linear(trace, model, pool, prefix + ":gate-up", model.hidden_size, 2 * local,
            module_path + ".gate_up_proj")
        self.act_fn = SymbolicSiluAndMul(trace, prefix + ":silu-mul", pool)
        self.down_proj = _source_linear(trace, model, pool, prefix + ":down", local, model.hidden_size,
            module_path + ".down_proj")
        if reduce_results:
            self.down_proj = SymbolicRowParallelLinear(self.down_proj, reduction_group, reduction_config)


class DeepseekV32MoERunner:
    """Declared BF16-compute modular Runner frames; output un-reduced.

    Fused/shared, monolithic, latent transforms, DP/SP, DBO and additional
    output scaling are outside this source profile. V32 passes
    apply_routed_scale_to_output=False, so the router applies its factor and
    the Runner receives1.0; no duplicate output scaling is inserted.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, resolution, invocation_id,
                 expert_parallel_size=1, expert_rank=0):
        if (model.model_type != "glm_moe_dsa" or type(tp_size) is not int
                or tp_size < 1 or model.num_attention_heads % tp_size
                or (tp_size not in (1, 8) and model.weight_dtype != "bf16" and not is_glm_fp8_tensor_model_scope(model, tp_size))
                or model.num_shared_experts != 1):
            raise NotImplementedError("symbolic V32 Runner requires legal GLM TP shards with one shared expert")
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size, model.num_experts,
            kind=WorkloadKind.ROUTER, output_dtype="fp32")
        self.shared_experts = DeepseekV2MLP(trace, model, pool, prefix + ":shared", tp_size=tp_size,
            intermediate_size=model.moe_intermediate_size * model.num_shared_experts,
            module_path=f"model.layers.{layer_index}.mlp.shared_experts", reduce_results=False)
        self.router = SymbolicGroupedRouting(trace, prefix + ":route", pool, model)
        self.routed_experts = SymbolicRoutedExperts(trace, prefix + ":routed", pool, model, tp_size=tp_size,
            layer_index=layer_index,
            resolution=resolution, invocation_id=invocation_id, routing_source_stage="dsv32_moe_route",
            expert_parallel_size=expert_parallel_size, expert_rank=expert_rank)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)

    def _apply_quant_method(self, hidden_states, router_logits, shared_experts_input):
        shared_output = self.shared_experts(shared_experts_input)
        topk_weights, topk_ids = self.router(hidden_states, router_logits)
        fused_output = self.routed_experts(hidden_states, topk_weights, topk_ids)
        return shared_output, fused_output

    def _forward_impl(self, hidden_states, router_logits, shared_experts_input):
        router_logits, _ = self.gate(hidden_states)
        return self._apply_quant_method(hidden_states, router_logits, shared_experts_input)

    def __call__(self, *, hidden_states, router_logits):
        if router_logits is not hidden_states:
            raise ValueError("V32 Runner must own its source gate, given the hidden-state placeholder")
        shared_output, fused_output = self._forward_impl(hidden_states, router_logits, hidden_states)
        return self.add(shared_output, fused_output)


class DeepseekV32MoE:
    """DeepseekV2MoE.forward with sequence parallelism disabled."""

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, resolution, invocation_id,
                 expert_parallel_size=1, expert_rank=0):
        self.experts = DeepseekV32MoERunner(trace, model, pool, prefix,
            layer_index=layer_index, tp_size=tp_size, resolution=resolution, invocation_id=invocation_id,
            expert_parallel_size=expert_parallel_size, expert_rank=expert_rank)

    def __call__(self, hidden_states):
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(num_tokens, hidden_dim)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        return final_hidden_states.view_as_shape(num_tokens, hidden_dim)


def _is_kimi_k2_text_source(model):
    from .model_config import is_kimi_k2_parallel_model_scope
    return is_kimi_k2_parallel_model_scope(model, 1, 1)


@dataclass(frozen=True)
class _KimiMoEContext:
    tensor_size: int
    expert_size: int = 1
    expert_rank: int = 0
    data_size: int = 1
    group: object = None
    rows: tuple = ()
    dispatch: object = None
    combine: object = None
    moe_sequence_parallel: bool = False
    gather: object = None
    logical_rows: int = 0
    tensor_lane: int = 0
    model_sequence_parallel: bool = False

    def shard(self, trace, hidden, *, pool, name):
        chunk = -(-self.logical_rows // self.tensor_size)
        padded = chunk * self.tensor_size
        if hidden.shape[0] != self.logical_rows:
            raise ValueError("Kimi internal MoE SP must receive its actual full request rows")
        if padded != self.logical_rows:
            output = trace.tensor(trace.unique_name(name + ":padded"), (padded, hidden.shape[1]), hidden.dtype)
            SymbolicCopy(trace, name + ":pad-copy", pool)(hidden, output.narrow(0, 0, self.logical_rows))
            SymbolicFillZero(trace, name + ":pad-zero", pool)(output.narrow(0, self.logical_rows, padded - self.logical_rows))
            return output.narrow(0, self.tensor_lane * chunk, chunk)
        shard = hidden.narrow(0, self.tensor_lane * chunk, chunk)
        cloned = trace.tensor(trace.unique_name(name + ":clone"), shard.shape, shard.dtype)
        return SymbolicCopy(trace, name + ":clone-copy", pool)(shard, cloned)


@dataclass(frozen=True)
class _KimiReplicaContext:
    traces: tuple
    tensor_groups: tuple
    distribution_groups: tuple
    rows: tuple
    rank: int
    resolutions: tuple
    invocations: tuple
    expert_size: int
    moe_sequence_parallel: bool


class DeepseekV2KimiMoERunner(DeepseekV32MoERunner):
    """Modular K2 Runner with BF16 DP prepare and exact output scaling.

    EP-off shards routed weights over TP x DP; global EP owns complete
    experts. DP+EP uses model SP in PP1 or internal MoE SP with PP>1,
    replicating shared weights. Other branches retain late TP summation.
    """

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index, resolution, invocation_id,
                 expert_parallel_size=1, expert_context=None):
        tp_size = 1 if group is None else len(group.traces)
        from .model_config import is_kimi_k2_parallel_model_scope
        if not is_kimi_k2_parallel_model_scope(model, tp_size, 1, expert_parallel=expert_parallel_size > 1):
            raise NotImplementedError("Kimi Runner requires legal BF16 K2 text TP and whole INT4 expert groups")
        if expert_context is not None and (not isinstance(expert_context, _KimiMoEContext)
                or expert_context.tensor_size != tp_size or expert_context.data_size <= 1
                or expert_context.expert_size != expert_parallel_size):
            raise ValueError("Kimi DP requires its normalized physical expert context")
        context = (_KimiMoEContext(tp_size, expert_parallel_size,
            group.participants.index(trace.rank) if expert_parallel_size > 1 else 0)
            if expert_context is None else expert_context)
        if ((expert_context is None and expert_parallel_size not in (1, tp_size))
                or (group is None and config is not None)):
            raise NotImplementedError("Kimi Runner requires its actual tensor group and expert ownership")
        self.trace, self.context = trace, context
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size, model.num_experts,
            kind=WorkloadKind.ROUTER)
        self.shared_experts = DeepseekV2MLP(trace, model, pool, prefix + ":shared", tp_size=1 if context.moe_sequence_parallel else tp_size,
            intermediate_size=model.moe_intermediate_size,
            module_path=f"language_model.model.layers.{layer_index}.mlp.shared_experts", reduce_results=False)
        self.router = SymbolicGroupedRouting(trace, prefix + ":route", pool, model, apply_routed_scale_to_output=True)
        self.routed_experts = SymbolicRoutedExperts(trace, prefix + ":routed", pool, model, tp_size=tp_size,
            layer_index=layer_index, resolution=resolution, invocation_id=invocation_id, routing_source_stage="kimi_moe_route",
            expert_parallel_size=expert_parallel_size,
            expert_rank=context.expert_rank,
            expert_tp_size=1 if expert_parallel_size > 1 else tp_size * context.data_size)
        self.output_scale = SymbolicScalarMultiply(trace, prefix + ":output-scale", pool, model.routed_scaling_factor)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)
        self.group, self.config, self.name, self.pool = group, config, prefix, pool
        self.expert_parallel_size, self.invocation_id = expert_parallel_size, invocation_id

    def __call__(self, *, hidden_states, router_logits):
        if router_logits is not hidden_states:
            raise ValueError("Kimi Runner owns its BF16 gate and requires the hidden-state placeholder")
        context = self.context
        if context.moe_sequence_parallel and not context.model_sequence_parallel and hidden_states is not None:
            hidden_states = context.shard(self.trace, hidden_states, pool=self.pool, name=self.name + ":sequence")
        hidden = hidden_states
        logits = None if hidden is None else self.gate(hidden)[0]
        shared = None if hidden is None else self.shared_experts(hidden)
        weights, ids = (None, None) if hidden is None else self.router(hidden, logits)
        if context.group is not None:
            dimension = "expert" if context.moe_sequence_parallel else "data"
            hidden, weights, ids = tuple(context.group.call_variable(self.name + ":dispatch-" + role, self.trace, value,
                pool_name=self.pool, config=context.dispatch, rows_by_rank=context.rows,
                row_shape=(width,), dtype=dtype, group_dimension=dimension) for role, value, width, dtype in (
                    ("input", hidden, self.routed_experts.model.hidden_size, "bf16"),
                    ("weights", weights, self.routed_experts.model.num_experts_per_tok, "fp32"),
                    ("ids", ids, self.routed_experts.model.num_experts_per_tok, "int32")))
        fused = self.routed_experts(hidden, weights, ids)
        if context.group is not None:
            fused = context.group.call_variable(self.name + ":combine", self.trace, fused, pool_name=self.pool,
                config=context.combine, rows_by_rank=context.rows, row_shape=(self.routed_experts.model.hidden_size,),
                dtype="bf16", group_dimension=dimension, moe_invocation_id=self.invocation_id)
            if hidden_states is None:
                return None
            output = self.trace.tensor(self.trace.unique_name(self.name + ":finalize-output"), hidden_states.shape, "bf16")
            fused = SymbolicCopy(self.trace, self.name + ":finalize-copy", self.pool)(fused, output)
        result = self.add(shared, self.output_scale(fused))
        if context.model_sequence_parallel:
            return result
        if context.moe_sequence_parallel:
            gathered = self.group.call(self.name + ":sequence-restore", result, pool_name=self.pool,
                config=context.gather, inplace=False, dim=0)
            return gathered if gathered.shape[0] == context.logical_rows else gathered.narrow(0, 0, context.logical_rows)
        if self.group is None:
            return result
        return self.group.call(self.name + ":all-reduce", result, pool_name=self.pool, config=self.config, inplace=False,
            moe_invocation_id=self.invocation_id if self.expert_parallel_size > 1 else None)


class DeepseekV2KimiMoE(DeepseekV32MoE):
    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index, resolution, invocation_id,
                 expert_parallel_size=1, expert_context=None):
        self.experts = DeepseekV2KimiMoERunner(trace, model, pool, prefix, group, config,
            layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
            expert_parallel_size=expert_parallel_size, expert_context=expert_context)


def _source_linear(trace, model, pool, name, in_features, out_features, module_path):
    """Bind exact checkpoint exclusions to the shared linear primitives."""
    storage = model.weight_dtype_for_module(module_path, "Linear")
    if storage == "bf16":
        return SymbolicLinear(trace, name, pool, in_features, out_features)
    if storage == "fp8":
        return SymbolicBlockFP8Linear(trace, name, pool, in_features, out_features,
            source_profile="vllm_0_30_logical_group128_fp32")
    raise NotImplementedError("symbolic source linear requires BF16 or block-FP8 module weights")


class DeepseekV2MLAInput:
    """K2.6/K2.7 ordinary MLA wrapper up to its attention call.

    Source: layers/mla.py MultiHeadLatentAttentionWrapper.forward, with the
    constructor's default fuse_qkv_rmsnorm=False, CP1 and no gates/indexer.
    The context retains the genuine caller locals while the backend runs.
    Attention/cache effects and the output projection are separate consumers.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, rotary_cache):
        if (not _is_kimi_k2_text_source(model)
                or model.target_semantics.rope_layout != "interleaved"
                or type(tp_size) is not int or tp_size < 1 or model.num_attention_heads % tp_size):
            raise NotImplementedError("ordinary symbolic MLA input requires exact K2 text and whole BF16 query heads")
        if type(layer_index) is not int or not 0 <= layer_index < model.num_hidden_layers:
            raise ValueError("MLA input layer must belong to the checkpoint")
        trace._check_tensor(rotary_cache)
        if (rotary_cache.dtype != "bf16" or len(rotary_cache.shape) != 2
                or rotary_cache.shape[-1] != model.qk_rope_head_dim):
            raise ValueError("MLA input requires its BF16 source rotary cache")
        self.trace, self.model, self.heads = trace, model, model.num_attention_heads // tp_size
        path = f"language_model.model.layers.{layer_index}.self_attn"
        self.fused_qkv_a_proj = _source_linear(trace, model, pool, prefix + ":qkv-a", model.hidden_size,
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, path + ".fused_qkv_a_proj")
        self.q_a_layernorm = SymbolicRMSNorm(trace, prefix + ":q-a-norm", pool,
            model.q_lora_rank, model.target_semantics.rms_norm_eps)
        self.kv_a_layernorm = SymbolicRMSNorm(trace, prefix + ":kv-a-norm", pool,
            model.kv_lora_rank, model.target_semantics.rms_norm_eps)
        self.q_b_proj = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), path + ".q_b_proj")
        self.rotary_emb = SymbolicRotary(trace, prefix + ":rope", pool,
            self.heads, 1, model.qk_rope_head_dim, rotary_cache,
            model_execution_version="vllm_0_30_0")

    @contextmanager
    def __call__(self, positions, hidden_states):
        self.trace._check_tensor(positions)
        self.trace._check_tensor(hidden_states)
        model, rows = self.model, hidden_states.shape[0]
        if (hidden_states.shape != (rows, model.hidden_size) or hidden_states.dtype != "bf16"
                or positions.shape != (rows,) or positions.dtype != "int64"):
            raise ValueError("MLA source inputs require BF16 hidden rows and matching INT64 positions")
        q_c = None
        kv_lora = None
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split((model.q_lora_rank, model.kv_lora_rank + model.qk_rope_head_dim))
        q_proj_layer = self.q_b_proj
        q_c = self.q_a_layernorm(q_c)
        q_proj_input = q_c
        kv_c, k_pe = kv_lora.split((model.kv_lora_rank, model.qk_rope_head_dim))
        kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)
        q = q_proj_layer(q_proj_input)[0]
        q = q.view_as_shape(rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
        q_pe = q.narrow(-1, model.qk_nope_head_dim, model.qk_rope_head_dim)
        _, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        yield q, kv_c_normed, k_pe


class DeepseekV2MLADecodeAttention:
    """K2.6 TP8 CP1 all-decode wrapper, declared BF16 FlashMLA source.

    Library internals/compiled LSE program and physical page allocation remain
    unbound. The source operator program is never an installed provider.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, kv_lens, cache,
                 slot_mapping, slots, rotary_cache, reduction_group, reduction_config, backend_source, q_lens=None, reduce_results=True):
        if backend_source != "vllm_0_30_flashmla_bf16":
            raise NotImplementedError("MLA decode requires an explicit BF16 FlashMLA source declaration")
        if ((tp_size == 1 and (reduction_group is not None or reduction_config is not None))
                or (tp_size != 1 and (not isinstance(reduction_group, ForwardCollectiveGroup)
                    or len(reduction_group.traces) != tp_size))):
            raise ValueError("K2.6 attention requires its complete TP reduction group")
        self.trace, self.model, self.name = trace, model, prefix
        self.graph_token_shape = None
        self.inputs = DeepseekV2MLAInput(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, rotary_cache=rotary_cache)
        self.heads = self.inputs.heads
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_mapping,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        self.q_absorb = SymbolicBatchedLinear(trace, prefix + ":q-absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.concat = SymbolicConcat(trace, prefix + ":concat-query", pool)
        self.latent = SymbolicMLALatentDecode(trace, prefix + ":latent", pool, heads=self.heads,
            kv_lens=kv_lens, cache=cache, q_lens=q_lens,
            source_profile="vllm_0_30_causal_flashmla_bf16_source")
        if sum(self.latent.q_lens) != len(slots):
            raise ValueError("MLA decode needs one distinct update slot per query row")
        self.v_up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool,
            self.heads, model.kv_lora_rank, model.v_head_dim)
        path = f"language_model.model.layers.{layer_index}.self_attn.o_proj"
        projection = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, path)
        self.o_proj = (projection if reduction_group is None or not reduce_results
            else SymbolicRowParallelLinear(projection, reduction_group, reduction_config))

    def _forward_mqa(self, query, *, public_output_rows=None):
        query = self.concat(*query)
        return self.latent(query, public_output_rows=public_output_rows)

    def _mla(self, query, latent, rope):
        rows, model = query.shape[0], self.model
        active = sum(self.latent.q_lens)
        if rows != active:
            latent, rope = latent.narrow(0, 0, active), rope.narrow(0, 0, active)
        self.update(latent, rope)
        output = self.trace.tensor(self.trace.unique_name(self.name + ":attention-output"),
            (rows, self.heads * model.v_head_dim), "bf16")
        # FULL metadata's num_actual_tokens is itself padded. PIECEWISE
        # metadata keeps actual rows. The public backend output retains the
        # FULL extent, including seq-len-zero anonymous rows; useful cache
        # attention math still covers only the real request windows.
        if self.graph_token_shape is not None and self.graph_token_shape.runtime_mode == "FULL":
            return self._decode_into(query, output, public_output_rows=rows)
        if rows != active:
            self._decode_into(query.narrow(0, 0, active), output.narrow(0, 0, active))
            return output
        return self._decode_into(query, output)

    def _decode_into(self, query, output, *, public_output_rows=None):
        rows, model = query.shape[0], self.model
        q_nope, q_pe = query.split((model.qk_nope_head_dim, model.qk_rope_head_dim))
        absorbed = self.trace.tensor(self.trace.unique_name(self.name + ":absorbed-query"),
            (rows, self.heads, model.kv_lora_rank), "bf16")
        self.q_absorb(q_nope.transpose(0, 1), out=absorbed.transpose(0, 1))
        mqa_q = (absorbed, q_pe)
        attn_out, lse = self._forward_mqa(mqa_q, public_output_rows=public_output_rows)
        self.v_up(attn_out.transpose(0, 1),
            out=output.view_as_shape(rows, self.heads, model.v_head_dim).transpose(0, 1))
        return output

    def __call__(self, positions, hidden_states):
        self.trace._check_tensor(positions)
        self.trace._check_tensor(hidden_states)
        rows = sum(self.latent.q_lens)
        physical = rows if self.graph_token_shape is None else self.graph_token_shape.model_input_tokens
        if (hidden_states.shape != (physical, self.model.hidden_size) or hidden_states.dtype != "bf16"
                or positions.shape != (physical,) or positions.dtype != "int64"):
            raise ValueError("MLA decode requires exactly one BF16 hidden row and INT64 position per declared context")
        with self.inputs(positions, hidden_states) as (query, latent, rope):
            attn_out = self._mla(query, latent, rope)
            return self.o_proj(attn_out)[0]


class MLADummyAttention:
    """MLA's metadata=None branch: wrapper projections, zero output, O.

    Both pinned0.27.1 and0.30.0 return before cache append, absorption and
    attention. Their temporary profile allocation is retained as an explicit
    lower bound: max_num_seqs/block_size are not supplied by this source API.
    It must not be mistaken for a complete native workspace/capacity receipt.
    """

    def __init__(self, trace, model, pool, prefix, *, inputs, projection, workspace_min_rows):
        if type(workspace_min_rows) is not int or workspace_min_rows < 1:
            raise ValueError("MLA dummy requires its explicit profile workspace row lower bound")
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.inputs, self.o_proj = inputs, projection
        self.workspace_min_rows = workspace_min_rows

    def __call__(self, positions, hidden_states):
        with self.inputs(positions, hidden_states) as (query, latent, rope):
            output = self.trace.tensor(self.trace.unique_name(self.prefix + ":dummy-output"),
                (query.shape[0], self.inputs.heads * self.model.v_head_dim), "bf16")
            workspace = self.trace.tensor(self.trace.unique_name(self.prefix + ":dummy-profile-workspace-lower-bound"),
                (self.workspace_min_rows, self.inputs.heads,
                    self.model.qk_nope_head_dim + self.model.v_head_dim), "bf16")
            SymbolicFillZero(self.trace, self.prefix + ":dummy-output-zero", self.pool)(output, controls=(query, latent, rope))
            del workspace
            return self.o_proj(output)[0]


class MLAFreshPrefillBackend:
    """Shared MLACommonImpl.forward_mha fresh BF16 source body.

    The prefill backend returns a fresh output (out=None); the caller copies
    it into its preallocated output. This does not bind a numerical provider,
    internal score/LSE workspace, physical traffic or a compiled launch.
    """

    def __init__(self, trace, model, pool, prefix, *, heads, q_lens, cache, module_path, fa3_hardware=None,
                 fa3_length_inputs=None, fa3_model_version="vllm_0_30_0"):
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.heads, self.q_lens, self.cache = heads, q_lens, cache
        self.fa3_hardware = fa3_hardware
        self.fa3_model_version = fa3_model_version
        self.fa3_length_inputs = {} if fa3_length_inputs is None else fa3_length_inputs
        if fa3_hardware is not None:
            fa3_hardware.fa3_mla_source_revision_for_model(fa3_model_version)
        self.kv_b_proj = _source_linear(trace, model, pool, prefix + ":kv-b", model.kv_lora_rank,
            heads * (model.qk_nope_head_dim + model.v_head_dim), module_path + ".kv_b_proj")
        self.copy_nope = SymbolicCopy(trace, prefix + ":key-nope-copy", pool)
        self.copy_rope = SymbolicCopy(trace, prefix + ":key-rope-copy", pool)
        self.copy_output = SymbolicCopy(trace, prefix + ":output-copy", pool)

    def __call__(self, query, latent, rope, output):
        model, rows = self.model, sum(self.q_lens)
        for value in (query, latent, rope, output):
            self.trace._check_tensor(value)
        if (any(value.dtype != "bf16" for value in (query, latent, rope, output))
                or query.shape != (rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
                or latent.shape != (rows, model.kv_lora_rank) or rope.shape != (rows, 1, model.qk_rope_head_dim)
                or output.shape != (rows, self.heads * model.v_head_dim)):
            raise ValueError("fresh MLA backend requires matching BF16 query, latent, RoPE and output rows")
        key, value = self._expand_kv(latent, rope)
        output_prefill = self._attention(query, key, value, kv_lens=self.q_lens,
            causal=True, return_lse=False)
        self.copy_output(output_prefill.view_as_shape(rows, self.heads * model.v_head_dim), output)

    def _expand_kv(self, latent, rope):
        model, rows = self.model, latent.shape[0]
        kv_nope = self.kv_b_proj(latent)[0].view_as_shape(
            rows, self.heads, model.qk_nope_head_dim + model.v_head_dim)
        k_nope, value = kv_nope.split((model.qk_nope_head_dim, model.v_head_dim))
        key = self.trace.tensor(self.trace.unique_name(self.prefix + ":key"),
            (rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim), "bf16")
        self.copy_nope(k_nope, key.narrow(-1, 0, model.qk_nope_head_dim))
        self.copy_rope(rope.expand(rows, self.heads, model.qk_rope_head_dim),
            key.narrow(-1, model.qk_nope_head_dim, model.qk_rope_head_dim))
        return key, value

    def _attention(self, query, key, value, *, kv_lens, causal, return_lse, lse_layout="heads_rows", allow_empty_kv=False,
                   fa3_metadata_key=("new_tokens",)):
        from .workload_graph import WorkloadNode
        model, rows = self.model, sum(self.q_lens)
        if lse_layout not in ("heads_rows", "rows_heads"):
            raise ValueError("MLA prefill LSE requires an explicit head/row source layout")
        if type(allow_empty_kv) is not bool or (allow_empty_kv and (causal or not return_lse)):
            raise ValueError("empty KV source requires noncausal context with explicit LSE")
        empty_context = allow_empty_kv and any(length == 0 for length in kv_lens)
        node = WorkloadNode(self.trace.unique_name(self.prefix + ":prefill-attention"), WorkloadKind.FLASH_ATTN,
            self.pool, {"batch": len(self.q_lens), "q_len": max(self.q_lens), "kv_len": max(kv_lens),
                "num_query_heads": self.heads, "qk_head_dim": model.qk_nope_head_dim + model.qk_rope_head_dim,
                "v_head_dim": model.v_head_dim, "kv_lora_rank": model.kv_lora_rank},
            dtype="bf16", variant="mla_prefill", rank=self.trace.rank, attributes={
                "source_component": ("symbolic_mla_context_prefill_mathematical" if return_lse
                    else "symbolic_mla_fresh_prefill_mathematical"), "launch_policy": "analytical_launch",
                "causal": causal, "kv_dist": "uniform", "kv_layout": "mla", "q_lens": self.q_lens,
                "kv_lens": kv_lens, "full_cuda_graphs": False, "max_cudagraph_capture_size": 0,
                "return_softmax_lse": return_lse, "mathematical_program": (
                    "bf16_mla_prefill_softmax_v1" if causal else
                        "bf16_mla_context_empty_rows_softmax_v1" if empty_context else "bf16_mla_context_softmax_v1"),
                "symbolic_workspace_complete": False})
        if (self.fa3_hardware is not None
                and not empty_context and lse_layout == "heads_rows"
                and (node.shape["qk_head_dim"], node.shape["v_head_dim"]) == (192, 128)):
            result = self._fa3_attention(node, query, key, value, fa3_metadata_key)
            return result if return_lse else result[0]
        if return_lse:
            return self.trace.call_outputs(node, (query, key, value), outputs=(
                ("output", (rows, self.heads, model.v_head_dim), "bf16"),
                ("lse", (self.heads, rows) if lse_layout == "heads_rows" else (rows, self.heads), "fp32")),
                controls=(self.cache,))
        return self.trace.call(node, (query, key, value),
            shape=(rows, self.heads, model.v_head_dim), dtype="bf16", controls=(self.cache,))

    def _fa3_attention(self, node, query, key, value, metadata_key):
        return _trace_fa3_prefill_call(self.trace, node, query, key, value,
            cache=self.cache, hardware=self.fa3_hardware, version=self.fa3_model_version,
            length_inputs=self.fa3_length_inputs, metadata_key=metadata_key)


def _trace_fa3_prefill_call(trace, node, query, key, value, *, cache, hardware,
                          version, length_inputs, metadata_key, output=None):
    """Interpret the declared versioned default Hopper library source.

    These are actual allocations in this model trace, not standalone
    readiness fixtures. Device/binary identity and transport stay partial.
    """
    from src.kernel.memory_flash_attention import (
        fa3_prefill_storage_layouts, fa3_split_prefill_call_storage_layouts, fa3_split_prefill_tensor_parent_plan,
    )
    revision = hardware.fa3_mla_source_revision_for_model(version)
    attributes = {name: node.attributes[name] for name in (
        "causal", "kv_dist", "kv_layout", "q_lens", "kv_lens", "full_cuda_graphs",
        "max_cudagraph_capture_size", "return_softmax_lse")}
    parts = hardware.fa3_mla_forward_specs(node.shape, node.dtype, node.variant, attributes,
        scheduler_metadata_precomputed=False)
    split = parts[-1].variant == "combine"
    records = (fa3_split_prefill_call_storage_layouts(*parts[-2:], hardware)
        if split else fa3_prefill_storage_layouts(parts[-1]))
    layouts = {role: (shape, dtype) for role, shape, dtype, _ in records}
    metadata_identity = (metadata_key, node.attributes["q_lens"], node.attributes["kv_lens"], node.attributes["causal"])
    if metadata_identity not in length_inputs:
        cu_query = trace.tensor(node.node_id + ":cu_query", *layouts["cu_query"])
        cu_key = (cu_query if node.attributes["causal"] and node.attributes["q_lens"] == node.attributes["kv_lens"]
            else trace.tensor(node.node_id + ":cu_key", *layouts["cu_key"]))
        length_inputs[metadata_identity] = (cu_query, cu_key)
    cu_query, cu_key = length_inputs[metadata_identity]
    output_roles = (("output", "lse", "scheduler_metadata", "partial_output", "partial_lse")
        if split else ("output", "lse", "scheduler_metadata"))
    source_node = replace(node, attributes={**attributes,
        "source_component": "fa3_mla_split_prefill_call" if split else "fa3_mla_unsplit_prefill_call",
        "launch_policy": "analytical_launch",
        "source_model_execution_version": version, "fa3_source_revision": revision,
        "scheduler_metadata_precomputed": False,
        **({"fa3_dynamic_splits": fa3_split_prefill_tensor_parent_plan(*parts[-2:], hardware,
            binding_id=node.node_id)["dynamic_splits"]} if split else {}),
        "fa3_source_output_roles": output_roles,
        "fa3_backend_source": hardware.fa3_mla_model_backend_source(version),
        "fa3_runtime_binary_bound": False,
        "fa3_length_array_transport_bound": False})
    if output is None:
        results = trace.call_outputs(source_node, (query, key, value, cu_query, cu_key),
            outputs=tuple((role, *layouts[role]) for role in output_roles), controls=(cache,))
    else:
        trace._check_tensor(output)
        if (output.shape, output.dtype) != layouts["output"]:
            raise ValueError("FA3 caller output must preserve its actual source extent and dtype")
        results = (output, *(trace.tensor(node.node_id + ":" + role, *layouts[role]) for role in output_roles[1:]))
        trace.call(source_node, (query, key, value, cu_query, cu_key), shape=output.shape,
            dtype=output.dtype, out=output, writes=results[1:], controls=(cache,))
    # Workspace roots remain visible to source/lifetime accounting, but
    # are not returned to the model and never declared valid partial rows.
    return results[:2]



def _plan_mla_logical_context_chunks(lengths, offsets, row_budget, alignment):
    """Pack request contexts in source order, splitting continuations on pages."""
    chunks = []
    request, start = 0, 0
    while request < len(lengths):
        chunk = []
        available = row_budget
        while request < len(lengths):
            remaining = lengths[request] - start
            if remaining == 0:
                request += 1
                start = 0
                continue
            if remaining <= available:
                chunk.append((request, offsets[request] + start, remaining))
                available -= remaining
                request += 1
                start = 0
                continue
            take = min((available // alignment) * alignment,
                ((remaining - 1) // alignment) * alignment)
            if take:
                chunk.append((request, offsets[request] + start, take))
                start += take
            break
        if not chunk:
            raise ValueError("MLA context chunk cannot fit one source-aligned page")
        chunks.append(tuple(chunk))
    return tuple(chunks)


class MLAContextPrefillBackend(MLAFreshPrefillBackend):
    """Request contexts and source-aligned continuations in a logical arena.

    Mirrors forward_mha: causal suffix O/LSE, BF16 cached-latent gather and
    expansion, noncausal prefix O/LSE, then the final state merge. This is
    not a native block-table, compiler workspace or provider binding.
    """

    def __init__(self, *args, context_lengths, context_offsets, context_chunk_rows=None,
                 context_chunk_alignment=None, **kwargs):
        super().__init__(*args, **kwargs)
        if (not isinstance(context_lengths, tuple) or not isinstance(context_offsets, tuple)
                or len(context_lengths) != len(self.q_lens) or len(context_offsets) != len(context_lengths)
                or any(type(length) is not int or length < 0 for length in context_lengths)
                or not any(length > 0 for length in context_lengths)
                or any(type(offset) is not int or offset < 0 for offset in context_offsets)
                or any(offset + length > self.cache.shape[0]
                    for offset, length in zip(context_offsets, context_lengths, strict=True))):
            raise ValueError("MLA context requires nonnegative request contexts and disjoint logical cache slices")
        intervals = sorted((offset, offset + length) for offset, length in
            zip(context_offsets, context_lengths, strict=True) if length)
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("MLA context requires disjoint logical cache slices")
        self.context_lengths, self.context_offsets = context_lengths, context_offsets
        if context_chunk_rows is not None and (type(context_chunk_rows) is not int or context_chunk_rows <= 0):
            raise ValueError("MLA context chunk row budget must be a positive declared source value")
        self.chunk_rows = sum(context_lengths) if context_chunk_rows is None else context_chunk_rows
        if context_chunk_alignment is not None and (
                type(context_chunk_alignment) is not int or context_chunk_alignment <= 0):
            raise ValueError("MLA context chunk alignment must be a positive source block size")
        if sum(context_lengths) > self.chunk_rows and (
                context_chunk_alignment is None or self.chunk_rows % context_chunk_alignment):
            raise ValueError("multi-chunk MLA requires a page-aligned source row budget")
        self.chunk_alignment = 1 if context_chunk_alignment is None else context_chunk_alignment
        self.gather = SymbolicCopy(self.trace, self.prefix + ":context-cache-gather", self.pool)
        self.merge = SymbolicAttentionStateMerge(self.trace, self.prefix + ":context-merge", self.pool)
        self.context_accumulate = SymbolicAttentionStateMerge(self.trace, self.prefix + ":context-accumulate", self.pool)
        self.init_output = SymbolicCopy(self.trace, self.prefix + ":context-init-output", self.pool)
        self.init_lse = SymbolicCopy(self.trace, self.prefix + ":context-init-lse", self.pool)

    def __call__(self, query, latent, rope, output, *, expanded_kv=None):
        rows, model = sum(self.q_lens), self.model
        if (query.shape != (rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
                or latent.shape != (rows, model.kv_lora_rank) or rope.shape != (rows, 1, model.qk_rope_head_dim)
                or output.shape != (rows, self.heads * model.v_head_dim)
                or any(value.dtype != "bf16" for value in (query, latent, rope, output))):
            raise ValueError("MLA context ports require matching BF16 rows")
        if expanded_kv is None:
            key, value = self._expand_kv(latent, rope)
        else:
            if not isinstance(expanded_kv,tuple) or len(expanded_kv)!=2:
                raise ValueError("MLA context requires the actual expanded key/value pair")
            key,value=expanded_kv
            self.trace._check_tensor(key)
            self.trace._check_tensor(value)
            if (key.shape!=(rows,self.heads,model.qk_nope_head_dim+model.qk_rope_head_dim)
                    or value.shape!=(rows,self.heads,model.v_head_dim) or key.dtype!="bf16" or value.dtype!="bf16"):
                raise ValueError("MLA context expanded operands must match the actual new-token source rows")
        suffix, suffix_lse = self._attention(query, key, value, kv_lens=self.q_lens, causal=True, return_lse=True)
        # Source gather's typed BF16 payload in the declared logical arena.
        # Native page-table/starts metadata is unbound, not invented INT64
        # per-token row-index traffic.
        context_rows = sum(self.context_lengths)
        # vLLM packs page-aligned request contexts into bounded workspace chunks.
        # The symbolic call keeps the full query batch and marks requests
        # absent from a chunk with zero KV rows; native query-slice launches
        # and their exact workspace metadata remain outside this abstraction.
        chunks = _plan_mla_logical_context_chunks(self.context_lengths, self.context_offsets,
            self.chunk_rows, self.chunk_alignment)
        workspace_rows = min(context_rows, self.chunk_rows)
        gathered = self.trace.tensor(self.trace.unique_name(self.prefix + ":context-workspace"),
            (workspace_rows, model.kv_lora_rank + model.qk_rope_head_dim), "bf16")
        # Keep one source gather and one expansion per planned chunk, with
        # explicit logical views instead of native block tables or row IDs.
        prefix, prefix_lse = None, None
        seen_context = [False] * len(self.context_lengths)
        for chunk_index, chunk in enumerate(chunks):
            chunk_lengths = [0] * len(self.context_lengths)
            for request_index, _, length in chunk:
                chunk_lengths[request_index] = length
            empty_prefix_rows = sum(length for length, seen in
                zip(self.q_lens, seen_context, strict=True) if not seen)
            for request_index, _, _ in chunk:
                seen_context[request_index] = True
            chunk_rows = sum(chunk_lengths)
            active = gathered if chunk_rows == workspace_rows else gathered.narrow(0, 0, chunk_rows)
            sources = tuple(self.cache.narrow(0, offset, length)
                for _, offset, length in chunk)
            node = self.gather.node(WorkloadKind.GATHER,
                {"elements": chunk_rows * gathered.shape[1], "rows": chunk_rows, "hidden": gathered.shape[1]},
                "bf16", "gather", {"gather_factor": 1, "index_count": 0,
                    "index_bytes": 8, "scattered": False, "materialized_output": True})
            self.trace.call(node, sources, shape=active.shape, dtype="bf16", out=active)
            context_latent, context_rope = active.split((model.kv_lora_rank, model.qk_rope_head_dim))
            key, value = self._expand_kv(context_latent,
                context_rope.view_as_shape(chunk_rows, 1, model.qk_rope_head_dim))
            partial, partial_lse = self._attention(query, key, value,
                kv_lens=tuple(chunk_lengths), causal=False, return_lse=True,
                allow_empty_kv=any(length == 0 for length in chunk_lengths),
                fa3_metadata_key=("context", chunk_index))
            if context_rows <= self.chunk_rows:
                prefix, prefix_lse = partial, partial_lse
            elif chunk_index == 0:
                # _compute_prefill_context initializes independent running
                # O/LSE after its first partial; only continuations merge.
                prefix = self.trace.tensor(self.trace.unique_name(self.prefix + ":context-output"),
                    partial.shape, "bf16")
                prefix_lse = self.trace.tensor(self.trace.unique_name(self.prefix + ":context-lse"),
                    partial_lse.shape, "fp32")
                self.init_output(partial, prefix)
                lse_copy = self.init_lse.node(WorkloadKind.GATHER,
                    {"elements": self.heads * rows, "rows": self.heads, "hidden": rows}, "fp32", "gather",
                    {"gather_factor": 1, "index_count": 0, "index_bytes": 8, "scattered": False,
                        "materialized_output": True, "mathematical_program": "materialized_fp32_copy_bytes_v1"})
                self.trace.call(lse_copy, (partial_lse,), shape=prefix_lse.shape, dtype="fp32", out=prefix_lse)
            else:
                self.context_accumulate(prefix, prefix_lse, partial, partial_lse,
                    out=prefix, out_lse=prefix_lse, inplace=True,
                    empty_prefix_rows=empty_prefix_rows)
        self.merge(prefix, prefix_lse, suffix, suffix_lse,
            out=output.view_as_shape(rows, self.heads, model.v_head_dim),
            empty_prefix_rows=sum(length for length, context in
                zip(self.q_lens, self.context_lengths, strict=True) if context == 0))

class DeepseekV2MLAFreshPrefillAttention:
    """Ordinary BF16 MLA prefill, with an optional complete logical context.

    MLACommonPrefill.forward_mha passes out=None with unquantized output,
    then copies the backend's fresh output into the wrapper's output slice.
    This mathematical source domain does not bind a prefill provider.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, q_lens, cache,
                 slot_mapping, slots, rotary_cache, reduction_group, reduction_config,
                 context_length=0, context_slots=(), context_lengths=(), context_offsets=(), context_chunk_rows=None,
                 context_chunk_alignment=None, reduce_results=True, fa3_hardware=None, fa3_length_inputs=None):
        if (not isinstance(q_lens, tuple) or not q_lens
                or any(type(length) is not int or length < 1 for length in q_lens)
                or sum(q_lens) != len(slots)):
            raise ValueError("fresh MLA requires one declared length per sequence and all update slots")
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.q_lens, self.cache = q_lens, cache
        self.inputs = DeepseekV2MLAInput(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, rotary_cache=rotary_cache)
        self.heads = self.inputs.heads
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_mapping,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        path = f"language_model.model.layers.{layer_index}.self_attn"
        if context_length:
            if (context_lengths or context_offsets or context_slots != tuple(range(context_length))):
                raise ValueError("single-context compatibility ports require the explicit zero-based logical slice")
            context_lengths, context_offsets = (context_length,), (0,)
        elif context_slots:
            raise ValueError("context slots require a declared context length")
        backend = MLAContextPrefillBackend if context_lengths else MLAFreshPrefillBackend
        context = {"context_lengths": context_lengths, "context_offsets": context_offsets,
            "context_chunk_rows": context_chunk_rows,
            "context_chunk_alignment": context_chunk_alignment} if context_lengths else {}
        if not context_lengths and context_offsets:
            raise ValueError("context offsets require matching positive context lengths")
        self.prefill_backend = backend(trace, model, pool, prefix,
            heads=self.heads, q_lens=q_lens, cache=cache, module_path=path, fa3_hardware=fa3_hardware,
            fa3_length_inputs=fa3_length_inputs, **context)
        projection = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, path + ".o_proj")
        self.o_proj = (projection if reduction_group is None or not reduce_results
            else SymbolicRowParallelLinear(projection, reduction_group, reduction_config))

    def _forward_mha(self, query, latent, rope, output):
        self.prefill_backend(query, latent, rope, output)

    def __call__(self, positions, hidden_states):
        rows = sum(self.q_lens)
        if hidden_states.shape != (rows, self.model.hidden_size) or positions.shape != (rows,):
            raise ValueError("fresh MLA inputs must cover all declared sequence rows")
        with self.inputs(positions, hidden_states) as (query, latent, rope):
            self.update(latent, rope)
            output = self.trace.tensor(self.trace.unique_name(self.prefix + ":attention-output"),
                (rows, self.heads * self.model.v_head_dim), "bf16")
            self._forward_mha(query, latent, rope, output)
            return self.o_proj(output)[0]


class DeepseekV2MLAMixedAttention:
    """Decode-prefix plus fresh or cached-context prefill suffix."""

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, q_lens, kv_lens,
                 num_decodes, cache, slot_mapping, slots, rotary_cache, reduction_group, reduction_config,
                 context_chunk_rows=None, context_chunk_alignment=None, reduce_results=True, fa3_hardware=None, fa3_length_inputs=None):
        if (type(num_decodes) is not int or not 0 <= num_decodes < len(q_lens)
                or len(kv_lens) != len(q_lens)
                or (num_decodes and len(set(q_lens[:num_decodes])) != 1)
                or any(kv < q for q, kv in zip(q_lens, kv_lens))):
            raise ValueError("MLA mixed source requires decode-first with fresh or nonempty-context prefill suffix")
        self.trace, self.model, self.prefix = trace, model, prefix
        self.rows, self.num_decodes = sum(q_lens), num_decodes
        self.decode_rows = sum(q_lens[:num_decodes])
        self.graph_token_shape = None
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_mapping,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        prefill_rows = self.rows - self.decode_rows
        context_lengths = tuple(kv - q for q, kv in zip(q_lens[num_decodes:], kv_lens[num_decodes:]))
        context_lengths = context_lengths if any(context_lengths) else ()
        context_offsets = tuple(slots[sum(q_lens[:index])] - (kv_lens[index] - q_lens[index])
            for index in range(num_decodes, len(kv_lens))) if context_lengths else ()
        self.prefill = DeepseekV2MLAFreshPrefillAttention(trace, model, pool, prefix + ":prefill",
            layer_index=layer_index, tp_size=tp_size, q_lens=q_lens[num_decodes:], cache=cache,
            slot_mapping=slot_mapping.narrow(0, self.decode_rows, prefill_rows), slots=slots[self.decode_rows:],
            rotary_cache=rotary_cache, reduction_group=reduction_group, reduction_config=reduction_config,
            context_lengths=context_lengths, context_offsets=context_offsets, context_chunk_rows=context_chunk_rows,
            context_chunk_alignment=context_chunk_alignment, reduce_results=reduce_results, fa3_hardware=fa3_hardware, fa3_length_inputs=fa3_length_inputs)
        self.decode = None
        if num_decodes:
            self.decode = DeepseekV2MLADecodeAttention(trace, model, pool, prefix + ":decode",
                layer_index=layer_index, tp_size=tp_size, kv_lens=kv_lens[:num_decodes], cache=cache,
                slot_mapping=slot_mapping.narrow(0, 0, self.decode_rows), slots=slots[:self.decode_rows],
                rotary_cache=rotary_cache, reduction_group=reduction_group, reduction_config=reduction_config,
                backend_source="vllm_0_30_flashmla_bf16", q_lens=q_lens[:num_decodes], reduce_results=reduce_results)

    def __call__(self, positions, hidden_states):
        physical = self.rows if self.graph_token_shape is None else self.graph_token_shape.model_input_tokens
        if hidden_states.shape != (physical, self.model.hidden_size) or positions.shape != (physical,):
            raise ValueError("mixed MLA inputs must cover decode and fresh prefill rows")
        with self.prefill.inputs(positions, hidden_states) as (query, latent, rope):
            if physical != self.rows:
                query, latent, rope = (value.narrow(0, 0, self.rows) for value in (query, latent, rope))
            self.update(latent, rope)
            output = self.trace.tensor(self.trace.unique_name(self.prefix + ":attention-output"),
                (physical, self.prefill.heads * self.model.v_head_dim), "bf16")
            count = self.rows - self.decode_rows
            self.prefill._forward_mha(query.narrow(0, self.decode_rows, count),
                latent.narrow(0, self.decode_rows, count), rope.narrow(0, self.decode_rows, count),
                output.narrow(0, self.decode_rows, count))
            if self.decode is not None:
                self.decode._decode_into(query.narrow(0, 0, self.decode_rows),
                    output.narrow(0, 0, self.decode_rows))
            return self.prefill.o_proj(output)[0]


def _v32_indexer_type(model, layer_index, mtp_indexer_type):
    if type(layer_index) is not int:
        raise ValueError("V32 layer requires an integer physical checkpoint index")
    if 0 <= layer_index < model.num_hidden_layers:
        if mtp_indexer_type is not None:
            raise ValueError("ordinary V32 layers cannot declare an MTP indexer branch")
        return model.indexer_types[layer_index]
    if (layer_index == model.num_hidden_layers and model.mtp_architecture == "glm_moe_dsa_mtp"
            and model.num_mtp_layers == 1 and mtp_indexer_type in ("full", "shared")):
        return mtp_indexer_type
    raise ValueError("V32 attention requires a target layer or its explicitly declared appended MTP indexer")


class DeepseekV32AttentionInput:
    """V32 attention.forward prefix before _sparse_indexer_and_attn.

    This is a source component, not a complete attention/model adapter. It
    keeps the source prefix frame live while the caller executes the rest
    of attention inside its context. Q absorption covers every model row
    before either prefill/decode backend is selected. CP/SP1, no HiSparse.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size,
                 cache, index_cache, topk_buffer, rotary_cache, index_rotary_cache, slot_mapping,
                 dummy_index_q, dummy_index_weights, mtp_indexer_type=None, pcp_outputs_only=False, no_metadata=False):
        if (model.model_type != "glm_moe_dsa" or model.weight_dtype not in ("bf16", "fp8")
                or model.compute_dtype != "bf16" or type(tp_size) is not int or tp_size < 1
                or model.num_attention_heads % tp_size
                or (tp_size not in (1, 8) and not pcp_outputs_only and model.weight_dtype != "bf16"
                    and not is_glm_fp8_tensor_model_scope(model, tp_size))):
            raise NotImplementedError("V32 attention prefix requires BF16-compute GLM TP1/TP8")
        indexer_type = _v32_indexer_type(model, layer_index, mtp_indexer_type)
        self.trace, self.model, self.prefix = trace, model, prefix
        if type(no_metadata) is not bool or no_metadata and pcp_outputs_only:
            raise ValueError("V32 metadata=None source is distinct from PCP gathered outputs")
        self.pcp_outputs_only, self.no_metadata = pcp_outputs_only, no_metadata
        self.local_heads = model.num_attention_heads // tp_size
        self.has_indexer = indexer_type == "full"
        module = f"model.layers.{layer_index}.self_attn."
        self.fused_qkv_a_proj = _source_linear(trace, model, pool, prefix + ":qkv-a",
            model.hidden_size, model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim,
            module + "fused_qkv_a_proj")
        self.wk_weights_proj, self.indexer_wq_b = None, None
        if self.has_indexer:
            self.wk_weights_proj = _source_linear(trace, model, pool, prefix + ":indexer-wk-weights",
                model.hidden_size, model.index_head_dim + model.index_n_heads, module + "indexer.wk_weights_proj")
            self.indexer_wq_b = _source_linear(trace, model, pool, prefix + ":indexer-wq-b",
                model.q_lora_rank, model.index_n_heads * model.index_head_dim, module + "indexer.wq_b")
        self.norm_rope = SymbolicDSANormRope(trace, prefix + ":norm-rope", pool, model,
            has_indexer=self.has_indexer, cache=None if pcp_outputs_only or no_metadata else cache,
            index_cache=None if pcp_outputs_only or no_metadata else index_cache, topk_buffer=topk_buffer,
            rotary_cache=rotary_cache, index_rotary_cache=index_rotary_cache,
            slot_mapping=None if pcp_outputs_only or no_metadata else slot_mapping,
            outputs_only=pcp_outputs_only or no_metadata, no_metadata=no_metadata)
        self.q_b_proj = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.local_heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), module + "q_b_proj")
        self.q_absorb = SymbolicBatchedLinear(trace, prefix + ":q-absorb", pool,
            self.local_heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.fused_q = SymbolicDSAQuery(trace, prefix + ":fused-q", pool, model,
            has_indexer=self.has_indexer, rotary_cache=rotary_cache, index_rotary_cache=index_rotary_cache,
            dummy_index_q=dummy_index_q, dummy_index_weights=dummy_index_weights)

    @contextmanager
    def __call__(self, positions, hidden_states):
        model, trace = self.model, self.trace
        trace._check_tensor(positions)
        trace._check_tensor(hidden_states)
        if (len(hidden_states.shape) != 2 or hidden_states.dtype != "bf16"
                or hidden_states.shape[-1] != model.hidden_size
                or positions.shape != (hidden_states.shape[0],) or positions.dtype != "int64"
                or (not (self.pcp_outputs_only or self.no_metadata) and self.norm_rope.slot_mapping.shape != positions.shape)
                or self.norm_rope.topk_buffer.shape[0] < hidden_states.shape[0]):
            raise ValueError("V32 attention prefix requires BF16 hidden rows and aligned INT64 positions")
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split((model.q_lora_rank, model.kv_lora_rank,
            model.qk_rope_head_dim), dim=-1)
        kw, index_k, index_weights = None, None, None
        if self.has_indexer:
            kw = self.wk_weights_proj(hidden_states)[0]
            index_k, index_weights = kw.split((model.index_head_dim, model.index_n_heads), dim=-1)
        rows = hidden_states.shape[0]
        output = trace.tensor(trace.unique_name(self.prefix + ":attention-output"),
            (rows, self.local_heads * model.v_head_dim), "bf16")
        kv_c_out = trace.tensor(trace.unique_name(self.prefix + ":kv-c-out"), kv_c.shape, "bf16")
        k_pe_out = trace.tensor(trace.unique_name(self.prefix + ":k-pe-out"), k_pe.shape, "bf16")
        index_k_out = (trace.tensor(trace.unique_name(self.prefix + ":index-k-out"), index_k.shape, "bf16")
            if self.pcp_outputs_only and self.has_indexer else None)
        q_c = self.norm_rope(positions, q_c, kv_c, k_pe, index_k, kv_c_out=kv_c_out, k_pe_out=k_pe_out,
            index_k_out=index_k_out)
        q = self.q_b_proj(q_c)[0].view_as_shape(rows, self.local_heads,
            model.qk_nope_head_dim + model.qk_rope_head_dim)
        q_nope, q_pe = q.split((model.qk_nope_head_dim, model.qk_rope_head_dim), dim=-1)
        ql_nope = self.q_absorb(q_nope.transpose(0, 1)).transpose(0, 1)
        index_q = None
        if self.has_indexer:
            index_q = self.indexer_wq_b(q_c)[0].view_as_shape(rows, model.index_n_heads, model.index_head_dim)
        index_q_fp8, index_weights_out, mqa_q = self.fused_q(positions,
            q_pe, ql_nope, index_q, index_weights)
        # The suspended source frame owns qkv_lora/kw/index_q automatically,
        # including locals not passed to the backend. No manual keep/free
        # table is needed when the caller enters the backend/output suffix.
        yield (q_c, q_nope, q_pe, ql_nope, index_q_fp8, index_weights_out, mqa_q,
            kv_c_out, k_pe_out, output, *((index_k_out,) if self.pcp_outputs_only else ()))


class DeepseekV32Attention:
    """V32 BF16-compute latent attention, CP/SP1 and no HiSparse.

    A declared valid causal top-k domain is required. The backend's useful
    math/public output storage is represented, while its page remapping,
    ignored return and callee scratch stay explicit gaps. The module returns
    the un-reduced O projection, as the source decoder expects.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, tp_size, q_lens, kv_lens, request_phases,
                 cache, index_cache, topk_buffer, rotary_cache, index_rotary_cache, slot_mapping,
                 dummy_index_q, dummy_index_weights, workspace, max_batched_tokens, selection_source,
                 prefill_attention_source=None, source_max_model_len=None, mtp_indexer_type=None,
                 num_speculative_tokens=0, reused_selection_kv_lens=None):
        indexer_type = _v32_indexer_type(model, layer_index, mtp_indexer_type)
        if mtp_indexer_type is not None and any(phase == "prefill" for phase in request_phases) and prefill_attention_source != "force_sparse_mqa":
            raise NotImplementedError("V32 MTP prefill requires explicit force_sparse_mqa; no fresh-dense inference")
        prefill_lengths = tuple((q, kv) for q, kv, phase in zip(q_lens, kv_lens, request_phases, strict=True)
            if phase == "prefill")
        if (prefill_attention_source is None and prefill_lengths
                and all(q == kv and kv <= model.index_topk for q, kv in prefill_lengths)):
            prefill_attention_source = "dense_mha_fresh"
        _dsa_source_phase(q_lens, kv_lens, request_phases, prefill_attention_source, num_speculative_tokens,
            flattened_decode=_dsa_sm90_k5_flattening(model, num_speculative_tokens))
        self.dense_mha = prefill_attention_source == "dense_mha_fresh"
        if self.dense_mha and any(kv > model.index_topk for _, kv in prefill_lengths):
            raise NotImplementedError("fresh dense MHA requires every prefill context within top-k capacity")
        self.num_decodes = request_phases.count("decode")
        if not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace:
            raise ValueError("V32 attention requires this rank's explicitly initialized source workspace")
        if type(max_batched_tokens) is not int or max_batched_tokens < sum(q_lens):
            raise ValueError("V32 source batch capacity must cover all declared rows")
        if selection_source != "declared_causal_topk":
            raise NotImplementedError("V32 attention requires explicit valid top-k cardinality applicability")
        self.inputs = DeepseekV32AttentionInput(trace, model, pool, prefix, layer_index=layer_index,
            tp_size=tp_size, cache=cache, index_cache=index_cache, topk_buffer=topk_buffer,
            rotary_cache=rotary_cache, index_rotary_cache=index_rotary_cache, slot_mapping=slot_mapping,
            dummy_index_q=dummy_index_q, dummy_index_weights=dummy_index_weights,
            mtp_indexer_type=mtp_indexer_type)
        self.model, self.heads = model, self.inputs.local_heads
        self.indexer = (SymbolicDSAIndexer(trace, prefix + ":indexer", pool, model,
            q_lens=q_lens, kv_lens=kv_lens, request_phases=request_phases, cache=index_cache,
            topk_buffer=topk_buffer, selection_source=selection_source, prefill_attention_source=prefill_attention_source,
            workspace=workspace if source_max_model_len is not None else None, source_max_model_len=source_max_model_len,
            num_speculative_tokens=num_speculative_tokens)
            if self.inputs.has_indexer else None)
        self.concat = SymbolicConcat(trace, prefix + ":concat-query", pool)
        self.q_concat_buffer, = workspace.get_simultaneous(
            ((max_batched_tokens, 64, model.kv_lora_rank + model.qk_rope_head_dim), "bf16"))
        self.sparse = None if self.dense_mha and not self.num_decodes else SymbolicDSASparseAttention(
            trace, prefix + ":sparse", pool, model, tp_size=tp_size,
            q_lens=q_lens[:self.num_decodes] if self.dense_mha else q_lens,
            kv_lens=kv_lens[:self.num_decodes] if self.dense_mha else kv_lens,
            request_phases=request_phases[:self.num_decodes] if self.dense_mha else request_phases, cache=cache,
            topk_buffer=topk_buffer, indexer_type=indexer_type,
            selection_source=selection_source, prefill_attention_source=None if self.dense_mha else prefill_attention_source,
            num_speculative_tokens=num_speculative_tokens, reused_selection_kv_lens=reused_selection_kv_lens)
        self.prefill_backend = (MLAFreshPrefillBackend(trace, model, pool, prefix,
            heads=self.heads, q_lens=q_lens[self.num_decodes:], cache=cache,
            module_path=f"model.layers.{layer_index}.self_attn")
            if self.dense_mha else None)
        self.decode_q_absorb = SymbolicBatchedLinear(trace, prefix + ":decode-q-absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.v_up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool, self.heads, model.kv_lora_rank, model.v_head_dim)
        self.o_proj = _source_linear(trace, model, pool, prefix + ":o-proj",
            self.heads * model.v_head_dim, model.hidden_size, f"model.layers.{layer_index}.self_attn.o_proj")

    def _sparse_indexer_and_attn(self, index_q_fp8, index_weights_out, ql_nope, mqa_q, output):
        if self.indexer is not None:
            self.indexer(index_q_fp8, index_weights_out)
        rows = output.shape[0]
        # The source arena has 64 heads; concat_mla_q writes only the actual
        # head prefix, with the original padded token stride. Padding is not
        # initialized or charged as an extra copy by the mathematical model.
        q = self.q_concat_buffer.narrow(0, 0, rows).narrow(1, 0, self.heads)
        q = self.concat(ql_nope, mqa_q, out=q)
        attn_out = self.sparse(q)
        x = attn_out.view_as_shape(rows, self.heads, self.model.kv_lora_rank).transpose(0, 1)
        out = output.view_as_shape(rows, self.heads, self.model.v_head_dim).transpose(0, 1)
        self.v_up(x, out=out)

    def __call__(self, positions, hidden_states):
        with self.inputs(positions, hidden_states) as prepared:
            q_c, q_nope, q_pe, ql_nope, index_q_fp8, index_weights_out, mqa_q, kv_c, k_pe, output = prepared
            if self.dense_mha:
                if self.indexer is not None:
                    self.indexer(index_q_fp8, index_weights_out)
                # V32 calls forward_impl with a fresh full-head Q concat.
                # The earlier all-T Q absorption/fused_q remain source work.
                mha_q = self.concat(q_nope, mqa_q)
                self._forward_dense_mha(mha_q, kv_c, k_pe, output)
            else:
                self._sparse_indexer_and_attn(index_q_fp8, index_weights_out, ql_nope, mqa_q, output)
            return self.o_proj(output)[0]

    def _forward_dense_mha(self, query, latent, rope, output):
        """MLACommon.forward_impl executes prefill first, then its decode prefix."""
        trace, model, decodes = self.inputs.trace, self.model, self.num_decodes
        rows = query.shape[0] - decodes
        self.prefill_backend(query.narrow(0, decodes, rows), latent.narrow(0, decodes, rows),
            rope.narrow(0, decodes, rows).unsqueeze(1), output.narrow(0, decodes, rows))
        if decodes:
            decode_query = query.narrow(0, 0, decodes)
            q_nope, q_pe = decode_query.split((model.qk_nope_head_dim, model.qk_rope_head_dim))
            # The common wrapper absorbs these rows again; the V32 prefix's
            # earlier all-T absorption is still live source work, not its input.
            absorbed = trace.tensor(trace.unique_name(self.decode_q_absorb.name + ":output"),
                (decodes, self.heads, model.kv_lora_rank), "bf16")
            self.decode_q_absorb(q_nope.transpose(0, 1), out=absorbed.transpose(0, 1))
            packed = self.q_concat_buffer.narrow(0, 0, decodes).narrow(1, 0, self.heads)
            packed = self.concat(absorbed, q_pe, out=packed)
            attn_out = self.sparse(packed)
            self.v_up(attn_out.transpose(0, 1), out=output.narrow(0, 0, decodes).view_as_shape(
                decodes, self.heads, model.v_head_dim).transpose(0, 1))


class _V32SharedAttentionFrame:
    """One actual shared-indexer attention frame, with live storage ports.

    Routing remains outside this frame. The constructor scope groups only
    equal checkpoint linear storage, and the existing frame binder validates
    aliases, writers, readers and workspace geometry before replaying.
    """

    def __init__(self, attention):
        self.attention = attention

    def __call__(self, positions, hidden_states, cache, topk, concat):
        return (self.attention(positions, hidden_states),)


class DeepseekV32DecoderLayer:
    """PP/SP1 source layer with explicit all-reduce → residual RMSNorm.

    This declares the ordinary branch of fused_allreduce_rms_norm, rather
    than claiming selection of its optional FlashInfer fusion/provider.
    """

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index, self_attn, resolution, invocation_id,
                 expert_parallel_size=1, input_moe_invocation_id=None, pcp_context=None,
                 source_attention_program=None):
        self.group, self.config, self.pool, self.prefix = group, config, pool, prefix
        self.trace, self.pcp_context = trace, pcp_context
        self.sequence_parallel = pcp_context is not None and pcp_context.sequence_parallel
        self.self_attn = self_attn
        self.source_attention_program = source_attention_program
        self.attention_frame = _V32SharedAttentionFrame(self_attn) if source_attention_program is not None else None
        self.input_layernorm = SymbolicRMSNorm(trace, prefix + ":input-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        self.post_attention_layernorm = SymbolicRMSNorm(trace, prefix + ":post-attention-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        self.input_reduce_norm = (self.input_layernorm if group is None or self.sequence_parallel else
            SymbolicAllReduceRMSNorm(self.input_layernorm, group, config,
                name=prefix + ":input-all-reduce", pool_name=pool, moe_invocation_id=input_moe_invocation_id))
        self.post_attention_reduce_norm = (self.post_attention_layernorm if group is None or self.sequence_parallel else
            SymbolicAllReduceRMSNorm(self.post_attention_layernorm, group, config,
                name=prefix + ":attention-all-reduce", pool_name=pool))
        tp_size = 1 if group is None else len(group.traces)
        uses_moe = (model.layer_uses_moe(layer_index) if layer_index < model.num_hidden_layers
            else model.mtp_layer_uses_moe(layer_index - model.num_hidden_layers))
        if uses_moe:
            if pcp_context is None:
                moe_factory = DeepseekV32MoE
            else:
                from .traced_glm_pcp import GLMPCPMoE
                moe_factory = lambda *args, **kwargs: GLMPCPMoE(*args, context=pcp_context, **kwargs)
            self.mlp = moe_factory(trace, model, pool, prefix + ":mlp", tp_size=tp_size,
                layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
                expert_parallel_size=expert_parallel_size,
                expert_rank=(pcp_context.expert_rank
                    if pcp_context is not None and expert_parallel_size > 1 else
                    group.participants.index(trace.rank) if expert_parallel_size > 1 else 0))
        else:
            self.mlp = DeepseekV2MLP(trace, model, pool, prefix + ":mlp", tp_size=1 if self.sequence_parallel else tp_size,
                intermediate_size=model.intermediate_size, module_path=f"model.layers.{layer_index}.mlp", reduce_results=False)

    def __call__(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_reduce_norm(hidden_states, residual)
        if self.sequence_parallel:
            hidden_states = self.pcp_context.gather_hidden(self.trace, hidden_states,
                pool=self.pool, name=self.prefix + ":sp-attention-all-gather")
        if self.source_attention_program is None:
            hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        else:
            attention = self.self_attn
            hidden_states, = attention.inputs.trace._call_module_frame(
                self.source_attention_program, self.prefix, self.attention_frame,
                (positions, hidden_states, attention.sparse.cache,
                    attention.sparse.topk_buffer, attention.q_concat_buffer),
                dependencies=tuple(sorted(attention.inputs.trace._readers[
                    attention.q_concat_buffer.view.allocation_id])))
        if self.sequence_parallel:
            hidden_states = self.pcp_context.scatter_attention(self.trace, hidden_states,
                pool=self.pool, name=self.prefix + ":sp-attention")
        hidden_states, residual = self.post_attention_reduce_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV32MTPPredictorLayer:
    """The appended V32 draft layer with explicit arenas and source branches.

    The caller supplies each pass's draft IDs, routing and full/shared indexer
    declaration. This component does not prepare proposer inputs or decide
    acceptance, sample tokens, bind a provider or assert normal capacity.
    """

    def __init__(self, trace, model, pool, prefix, *, group, collective, gather,
                 input_token_ids, q_lens, kv_lens, request_phases, slots, cache_capacity_tokens,
                 resolution, invocation_id, source_profile, source_max_model_len, max_batched_tokens,
                 workspace, topk_buffer, cache, index_cache, rotary_cache, index_rotary_cache,
                 slot_tensor, indexer_type, prefill_attention_source, selection_source="declared_causal_topk",
                 input_mode="token_ids", input_norm=None, num_speculative_tokens=0,
                 reused_selection_kv_lens=None, startup_attention_factory=None, expert_parallel_size=1,
                 pcp_context=None):
        from .traced_operators import SymbolicV32EHNorm, SymbolicV32EmbedEHNorm
        from .traced_glm_mtp import GLMLegacyDummyAttentionFactory, GLMStartupAttentionFactory
        dummy_attention = type(startup_attention_factory) is GLMLegacyDummyAttentionFactory
        if (model.model_type != "glm_moe_dsa" or model.mtp_architecture != "glm_moe_dsa_mtp"
                or model.num_mtp_layers != 1 or model.num_hidden_layers != 78
                or model.compute_dtype != "bf16" or model.weight_dtype != _glm_source_weight_dtype(source_profile)
                or not model.mtp_layer_uses_moe(0)):
            raise NotImplementedError("V32 draft requires its accurate BF16-compute appended single MoE layer and weight source")
        if group is None:
            if collective is not None or gather is not None:
                raise ValueError("TP1 V32 draft cannot declare collective configurations")
            tp_size = 1
        elif (isinstance(group, ForwardCollectiveGroup) and is_glm_mtp_tensor_model_scope(model, len(group.traces))
                and any(trace is item for item in group.traces)):
            tp_size = len(group.traces)
        else:
            raise NotImplementedError("V32 draft requires TP1 or this rank's complete legal TP group")
        if (type(expert_parallel_size) is not int or expert_parallel_size < 1
                or pcp_context is None and (expert_parallel_size not in (1, tp_size)
                    or not is_glm_mtp_tensor_model_scope(model, tp_size, expert_parallel=expert_parallel_size > 1))):
            raise NotImplementedError("V32 draft requires EP-off or its complete declared BF16 expert group")
        if pcp_context is not None:
            from .traced_glm_pcp import GLMPCPContext
            if (not isinstance(pcp_context, GLMPCPContext) or pcp_context.context_parallel_size != 1
                    or pcp_context.tensor_parallel_size != tp_size
                    or model.weight_dtype == "fp8" and expert_parallel_size == 1 and (pcp_context.sequence_parallel
                        or pcp_context.moe_sequence_parallel
                        or model.moe_intermediate_size % (tp_size * len(pcp_context.physical_rows_by_data_rank) * 128))
                    or expert_parallel_size != (1 if pcp_context.expert_group is None else len(pcp_context.expert_group.traces))
                    or model.num_experts % expert_parallel_size
                    or startup_attention_factory is not None and not (dummy_attention and pcp_context.dummy_run)):
                raise NotImplementedError("V32 distributed draft requires its BF16-compute whole-shard TP CP1 DP/expert context")
        if (not isinstance(q_lens, tuple) or not q_lens or any(type(q) is not int or q < 1 for q in q_lens)
                or not isinstance(kv_lens, tuple) or len(kv_lens) != len(q_lens)
                or any(type(k) is not int or not q <= k <= model.max_position_embeddings
                    for q, k in zip(q_lens, kv_lens, strict=True))
                or not isinstance(request_phases, tuple) or len(request_phases) != len(q_lens)
                or any(phase not in ("decode", "prefill")
                    for phase, q in zip(request_phases, q_lens, strict=True))):
            raise ValueError("V32 draft requires complete positive query/context lengths and explicit backend phases")
        if startup_attention_factory is not None:
            if type(startup_attention_factory) not in (GLMStartupAttentionFactory, GLMLegacyDummyAttentionFactory):
                raise TypeError("V32 startup requires its typed no-metadata attention factory")
            startup_attention_factory.validate(rows=sum(q_lens), max_batched_tokens=max_batched_tokens,
                source_max_model_len=source_max_model_len, cache_capacity_tokens=cache_capacity_tokens,
                slot_mapping=slots, selection_source=selection_source, prefill_attention_source=prefill_attention_source)
            if (any(value is not None for value in (cache, index_cache, slot_tensor, reused_selection_kv_lens))
                    or indexer_type != "full" and not dummy_attention or input_mode != "token_ids"):
                raise ValueError("V32 startup draft has no request caches/slots or reused selection")
        if (startup_attention_factory is None and any(phase == "prefill" for phase in request_phases)
                and prefill_attention_source != "force_sparse_mqa"):
            raise NotImplementedError("V32 multiquery/prefill draft requires explicit force_sparse_mqa metadata")
        if startup_attention_factory is None:
            _dsa_source_phase(q_lens, kv_lens, request_phases, prefill_attention_source, num_speculative_tokens,
                flattened_decode=_dsa_sm90_k5_flattening(model, num_speculative_tokens))
        rows = sum(q_lens)
        if startup_attention_factory is None and (type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
                or type(max_batched_tokens) is not int or max_batched_tokens < rows
                or type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
                or not isinstance(slots, tuple) or len(slots) != rows or len(set(slots)) != rows
                or any(type(slot) is not int or not 0 <= slot < cache_capacity_tokens for slot in slots)):
            raise ValueError("V32 draft requires explicit source limits and distinct logical cache slots")
        if ((input_token_ids is not None if dummy_attention else
                not isinstance(input_token_ids, tuple) or len(input_token_ids) != rows
                or any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_token_ids))
                or indexer_type not in ("full", "shared") or input_mode not in ("token_ids", "inputs_embeds")):
            raise ValueError("V32 draft requires its own valid IDs, full/shared indexer and explicit input mode")
        if not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace:
            raise ValueError("V32 draft workspace must belong to the current rank")
        cache_arenas = (() if startup_attention_factory is not None else (
            (cache, (cache_capacity_tokens, model.kv_lora_rank + model.qk_rope_head_dim), "bf16"),
            (index_cache, (cache_capacity_tokens, 132), "uint8"), (slot_tensor, (rows,), "int64")))
        for tensor, shape, dtype in (*cache_arenas,
                (topk_buffer, (max_batched_tokens, model.index_topk), "int32"),
                (rotary_cache, (model.max_position_embeddings, model.qk_rope_head_dim), "bf16"),
                (index_rotary_cache, (model.max_position_embeddings, model.qk_rope_head_dim), "bf16")):
            trace._check_tensor(tensor)
            if tensor.shape != shape or tensor.dtype != dtype:
                raise ValueError("V32 draft requires exact typed cache/top-k/rotary/slot arenas")
        self.trace, self.model, self.rows, self.input_mode = trace, model, rows, input_mode
        self.pool, self.prefix = pool, prefix
        self.legacy_dummy_run = dummy_attention
        self.position_values = (None if dummy_attention else (0,) * rows if startup_attention_factory is not None else
            tuple(p for q, k in zip(q_lens, kv_lens, strict=True) for p in range(k - q, k)))
        epsilon = model.target_semantics.rms_norm_eps
        self.embed_tokens, self.eh_norm, self.embed_eh_norm = None, None, None
        norm_type = SymbolicV32EmbedEHNorm if input_mode == "token_ids" and group is None else SymbolicV32EHNorm
        if input_norm is not None and (type(input_norm) is not norm_type or input_norm.trace is not trace
                or input_norm.width != model.hidden_size or input_norm.epsilon != epsilon
                or input_norm.legacy_dummy_retained_inputs != dummy_attention
                or (norm_type is SymbolicV32EmbedEHNorm and input_norm.vocab != model.vocab_size)):
            raise ValueError("V32 recurrent draft input norm must share this rank's exact typed weight metadata")
        if input_mode == "token_ids" and group is None:
            self.embed_eh_norm = (input_norm if input_norm is not None else
                SymbolicV32EmbedEHNorm(trace, prefix + ":embed-eh-norm", pool, model.hidden_size, model.vocab_size, epsilon,
                    legacy_dummy_retained_inputs=dummy_attention))
        else:
            if input_mode == "token_ids":
                self.embed_tokens = SymbolicVocabParallelEmbedding(trace, prefix + ":embedding", pool,
                    model.hidden_size, model.vocab_size, group, collective, token_ids=input_token_ids,
                    input_values_source="legacy_dummy_retained_buffer" if dummy_attention else None)
            self.eh_norm = (input_norm if input_norm is not None else
                SymbolicV32EHNorm(trace, prefix + ":eh-norm", pool, model.hidden_size, epsilon,
                    legacy_dummy_retained_inputs=dummy_attention))
        self.input_norm = self.embed_eh_norm if self.embed_eh_norm is not None else self.eh_norm
        self.input_token_ids = input_token_ids
        self.eh_proj = SymbolicLinear(trace, prefix + ":eh-proj", pool, 2 * model.hidden_size, model.hidden_size)
        dummy_q = trace.tensor(prefix + ":dummy-index-q", (1, 1, 1), "bf16", persistent=True)
        dummy_weights = trace.tensor(prefix + ":dummy-index-weights", (1, 1), "fp32", persistent=True)
        attention = (startup_attention_factory.attention(trace, model, pool, prefix + ":attention",
            layer_index=model.num_hidden_layers, tp_size=tp_size, workspace=workspace,
            rotary_cache=rotary_cache, topk_buffer=topk_buffer,
            **({"dummy_index_q": dummy_q, "dummy_index_weights": dummy_weights,
                "mtp_indexer_type": indexer_type} if dummy_attention else {})) if startup_attention_factory is not None else
            DeepseekV32Attention(trace, model, pool, prefix + ":attention",
            layer_index=model.num_hidden_layers, tp_size=tp_size, q_lens=q_lens, kv_lens=kv_lens,
            request_phases=request_phases, cache=cache, index_cache=index_cache if indexer_type == "full" else None,
            topk_buffer=topk_buffer, rotary_cache=rotary_cache, index_rotary_cache=index_rotary_cache,
            slot_mapping=slot_tensor, dummy_index_q=dummy_q, dummy_index_weights=dummy_weights,
            workspace=workspace, max_batched_tokens=max_batched_tokens, selection_source=selection_source,
            prefill_attention_source=prefill_attention_source, source_max_model_len=source_max_model_len,
            mtp_indexer_type=indexer_type, num_speculative_tokens=num_speculative_tokens,
            reused_selection_kv_lens=reused_selection_kv_lens))
        self.mtp_block = DeepseekV32DecoderLayer(trace, model, pool, prefix + ":decoder", group, collective,
            layer_index=model.num_hidden_layers, self_attn=attention, resolution=resolution, invocation_id=invocation_id,
            expert_parallel_size=expert_parallel_size, pcp_context=pcp_context)
        norm = SymbolicRMSNorm(trace, prefix + ":final-norm", pool, model.hidden_size, epsilon)
        self.pcp_context = pcp_context
        self.sequence_parallel = pcp_context is not None and pcp_context.sequence_parallel
        self.final_norm = (norm if group is None or self.sequence_parallel else SymbolicAllReduceRMSNorm(norm, group, collective,
            name=prefix + ":final-all-reduce", pool_name=pool,
            moe_invocation_id=invocation_id if expert_parallel_size > 1 or pcp_context is not None else None))
        self.head = (SymbolicLinear(trace, prefix + ":head", pool, model.hidden_size, model.vocab_size,
            kind=WorkloadKind.LM_HEAD) if group is None else SymbolicTPLogitsProcessor(trace, prefix + ":head", pool,
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather))

    def __call__(self, ids, positions, previous_hidden_states, *, position_values, inputs_embeds=None):
        for tensor in (ids, positions, previous_hidden_states):
            self.trace._check_tensor(tensor)
        if (ids.shape != (self.rows,) or ids.dtype != "int32"
                or positions.shape != (self.rows,) or positions.dtype != "int64"
                or previous_hidden_states.shape != (self.rows, self.model.hidden_size) or previous_hidden_states.dtype != "bf16"
                or (position_values is not None if self.legacy_dummy_run else
                    not isinstance(position_values, tuple) or any(type(value) is not int for value in position_values))
                or position_values != self.position_values):
            raise ValueError("V32 draft inputs must match its own typed rows and exact declared context positions")
        if self.input_mode == "token_ids":
            if inputs_embeds is not None:
                raise ValueError("V32 token-ID draft cannot silently switch to external embeddings")
            if self.embed_eh_norm is not None:
                eh = self.embed_eh_norm(ids, positions, previous_hidden_states,
                    position_values=position_values, input_token_ids=self.input_token_ids)
            else:
                eh = self.eh_norm(positions, self.embed_tokens(ids), previous_hidden_states, position_values=position_values)
        else:
            if inputs_embeds is None:
                raise ValueError("V32 inputs_embeds draft requires its explicit tensor port")
            eh = self.eh_norm(positions, inputs_embeds, previous_hidden_states, position_values=position_values)
        if self.sequence_parallel:
            # mtp.py shards the complete 2H EH input before its replicated
            # projection, then runs the same SP decoder as the target.
            context = self.pcp_context
            padded = context._pad_hidden(self.trace, eh, pool=self.pool, name=self.prefix + ":sp-eh")
            eh = padded.narrow(0, context.tensor_rank * context.shard_rows, context.shard_rows)
        hidden_states = self.eh_proj(eh)[0]
        hidden_states, residual = self.mtp_block(positions, hidden_states, None)
        hidden_states, _ = self.final_norm(hidden_states, residual)
        if self.sequence_parallel:
            hidden_states = self.pcp_context.gather_hidden(self.trace, hidden_states,
                pool=self.pool, name=self.prefix + ":sp-final-all-gather")
        return hidden_states, hidden_states

    def compute_logits(self, hidden_states):
        # V32 mtp.py consumes the post-final-norm hidden without another norm.
        return self.head(hidden_states)[0] if isinstance(self.head, SymbolicLinear) else self.head(hidden_states)


def _glm_source_weight_dtype(source_profile):
    if source_profile == "vllm_0_30_v32_bf16_sync_explicit_allreduce":
        return "bf16"
    if source_profile == "vllm_0_30_v32_fp8_sync_explicit_allreduce":
        return "fp8"
    raise NotImplementedError("GLM decoder requires an explicit modular BF16/FP8 all-reduce source profile")


def _glm_fresh_dense_prefill_without_topk(model, q_lens, kv_lens, phases, prefill_source, num_speculative_tokens=0):
    """The CUDA GLM/DSV32 dense-MHA branch has no sparse top-k read.

    GlmMoeDsaForCausalLM resolves directly to the NVIDIA DSV32 model in
    vLLM0.30. Sparse MLA's short fresh-prefill MHA path bypasses the shared
    top-k buffer; decode, mixed batches and force-MQA do not.
    """
    return (num_speculative_tokens == 0 and prefill_source in (None, "dense_mha_fresh")
        and isinstance(q_lens, tuple) and isinstance(kv_lens, tuple) and isinstance(phases, tuple)
        and bool(q_lens) and len(q_lens) == len(kv_lens) == len(phases)
        and all(phase == "prefill" and type(q) is int and type(kv) is int
            and 0 < q == kv <= model.index_topk for q, kv, phase in zip(q_lens, kv_lens, phases, strict=True)))


class DeepseekV32Model:
    """Pure-text GLM decoder or contiguous PP stage, without aux/MTP."""

    def __init__(self, trace, model, pool, group, config, *, token_ids, q_lens, kv_lens, request_phases,
                 slot_mapping, cache_capacity_tokens, source_max_model_len, max_batched_tokens,
                 resolutions, invocation_ids, workspace_initial_capacity_bytes, selection_source, source_profile,
                 prefill_attention_source, model_input_order, num_speculative_tokens=0, startup_attention_factory=None,
                 expert_parallel_size=1, layer_range=None, pcp_context=None):
        weight_dtype = _glm_source_weight_dtype(source_profile)
        if (model.model_type != "glm_moe_dsa" or model.weight_dtype != weight_dtype or model.compute_dtype != "bf16"
                or model.target_semantics.rope_layout != "interleaved"
                or model.target_semantics.indexer_rope_layout != "interleaved"):
            raise NotImplementedError("GLM decoder requires checkpoint/source weight dtype agreement and matching interleaved RoPE")
        self.group, self.config, self.pool = group, config, pool
        self.trace, self.pcp_context = trace, pcp_context
        self.sequence_parallel = pcp_context is not None and pcp_context.sequence_parallel
        start, end = (0, model.num_hidden_layers) if layer_range is None else layer_range
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= model.num_hidden_layers):
            raise ValueError("GLM stage requires a nonempty contiguous checkpoint layer range")
        self.first_stage, self.last_stage = start == 0, end == model.num_hidden_layers
        from .traced_glm_mtp import GLMStartupAttentionFactory, GLMLegacyDummyAttentionFactory
        dummy_attention = type(startup_attention_factory) is GLMLegacyDummyAttentionFactory
        if (not self.first_stage and model.indexer_types[start] != "full"
                and not dummy_attention
                and not _glm_fresh_dense_prefill_without_topk(model, q_lens, kv_lens, request_phases,
                    prefill_attention_source, num_speculative_tokens)):
            raise NotImplementedError("each GLM stage must start with its source full indexer")
        if startup_attention_factory is not None:
            if layer_range is not None and not dummy_attention:
                raise NotImplementedError("GLM startup factory does not bind pipeline stages")
            if type(startup_attention_factory) not in (GLMStartupAttentionFactory, GLMLegacyDummyAttentionFactory):
                raise TypeError("GLM startup requires its typed no-metadata attention factory")
            startup_attention_factory.validate(rows=sum(q_lens), max_batched_tokens=max_batched_tokens,
                source_max_model_len=source_max_model_len, cache_capacity_tokens=cache_capacity_tokens,
                slot_mapping=slot_mapping, selection_source=selection_source, prefill_attention_source=prefill_attention_source)
        self.embed_tokens = (SymbolicEmbedding(trace, "embedding", pool, model.hidden_size, "bf16",
            model_input_order=model_input_order) if group is None else
            SymbolicVocabParallelEmbedding(trace, "embedding", pool, model.hidden_size,
                model.vocab_size, group, config, token_ids=token_ids,
                input_values_source="legacy_dummy_retained_buffer" if dummy_attention and token_ids is None else None,
                model_input_order=model_input_order if trace.rank == group.participants[0] else None)) if self.first_stage else None
        self.workspace = ForwardWorkspace(trace, "worker.workspace", alignment_bytes=256,
            initial_capacity_bytes=workspace_initial_capacity_bytes, locked=False)
        topk = trace.tensor("topk", (max_batched_tokens, model.index_topk), "int32", persistent=True)
        rope = trace.tensor("rotary_cache", (model.max_position_embeddings, model.qk_rope_head_dim), "bf16", persistent=True)
        # Both source get_rope calls have equal arguments in this declared
        # GLM branch, and the factory cache returns the same object.
        index_rope = rope
        slots = (trace.tensor("slot_mapping", (sum(q_lens),), "int64", persistent=True)
            if startup_attention_factory is None else None)
        self.slot_mapping = slots
        dummy_q = trace.tensor("dummy:index-q", (1, 1, 1), "bf16", persistent=True)
        dummy_weights = trace.tensor("dummy:index-weights", (1, 1), "fp32", persistent=True)
        self.layers = []
        attention_programs = {}
        # Reuse the existing import-time source contract once per constructor
        # scope. Wrapped or replaced source must execute every actual layer.
        attention_frames = (startup_attention_factory is None and pcp_context is None
            and _DENSE_METADATA_SOURCE_CONTRACT.unchanged())
        for layer in range(start, end):
            prefix = f"layer-{layer}"
            cache = (trace.tensor(prefix + ":cache", (cache_capacity_tokens,
                model.kv_lora_rank + model.qk_rope_head_dim), "bf16", persistent=True)
                if startup_attention_factory is None else None)
            index_cache = (trace.tensor(prefix + ":index-cache", (cache_capacity_tokens, 132), "uint8", persistent=True)
                if startup_attention_factory is None and model.indexer_types[layer] == "full" else None)
            attention = (startup_attention_factory.attention(trace, model, pool, prefix + ":attention",
                layer_index=layer, tp_size=1 if group is None else len(group.traces), workspace=self.workspace,
                rotary_cache=rope, topk_buffer=topk,
                **({"dummy_index_q": dummy_q, "dummy_index_weights": dummy_weights} if dummy_attention else {}))
                if startup_attention_factory is not None else
                (DeepseekV32Attention if pcp_context is None else pcp_context.attention)(trace, model, pool, prefix + ":attention", layer_index=layer,
                tp_size=1 if group is None else len(group.traces), q_lens=q_lens, kv_lens=kv_lens, request_phases=request_phases,
                cache=cache, index_cache=index_cache, topk_buffer=topk, rotary_cache=rope, index_rotary_cache=index_rope,
                slot_mapping=slots, dummy_index_q=dummy_q, dummy_index_weights=dummy_weights, workspace=self.workspace,
                max_batched_tokens=max_batched_tokens, selection_source=selection_source,
                prefill_attention_source=prefill_attention_source, source_max_model_len=source_max_model_len,
                num_speculative_tokens=num_speculative_tokens))
            attention_program = None
            if attention_frames and model.indexer_types[layer] == "shared" and not attention.dense_mha:
                storage = tuple(model.weight_dtype_for_module(f"model.layers.{layer}.self_attn.{name}", "Linear")
                    for name in ("fused_qkv_a_proj", "q_b_proj", "o_proj"))
                if storage not in attention_programs:
                    attention_programs[storage] = trace._module_program(_V32SharedAttentionFrame)
                attention_program = attention_programs[storage]
            self.layers.append(DeepseekV32DecoderLayer(trace, model, pool, prefix, group, config, layer_index=layer,
                self_attn=attention, resolution=resolutions[layer], invocation_id=invocation_ids[layer],
                source_attention_program=attention_program,
                expert_parallel_size=expert_parallel_size, input_moe_invocation_id=(invocation_ids[layer - 1]
                    if (expert_parallel_size > 1 or num_speculative_tokens and pcp_context is not None)
                        and layer > 0 and model.layer_uses_moe(layer - 1) else None),
                pcp_context=pcp_context))
        self.norm = (SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size, model.target_semantics.rms_norm_eps)
            if self.last_stage else None)
        self.rotary_cache, self.index_rotary_cache = rope, index_rope
        self.topk_buffer = topk
        self.final_reduce_norm = (self.norm if group is None or not self.last_stage or self.sequence_parallel else
            SymbolicAllReduceRMSNorm(self.norm, group, config, name="final-all-reduce", pool_name=pool,
                moe_invocation_id=invocation_ids[-1] if expert_parallel_size > 1 or num_speculative_tokens and pcp_context is not None else None))

    def __call__(self, input_ids, positions, *, intermediate_tensors=None):
        if self.first_stage:
            if intermediate_tensors is not None:
                raise ValueError("first GLM stage cannot consume pipeline intermediates")
            hidden_states = self.embed_tokens(input_ids)
            if self.sequence_parallel:
                hidden_states = self.pcp_context.shard_embedding(self.trace, hidden_states, pool=self.pool)
            residual = None
        else:
            if not isinstance(intermediate_tensors, dict) or set(intermediate_tensors) != {"hidden_states", "residual"}:
                raise ValueError("non-first GLM stage requires both explicit intermediate ports")
            hidden_states, residual = intermediate_tensors["hidden_states"], intermediate_tensors["residual"]
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not self.last_stage:
            return {"hidden_states": hidden_states, "residual": residual}
        hidden_states, _ = self.final_reduce_norm(hidden_states, residual)
        if self.sequence_parallel:
            hidden_states = self.pcp_context.gather_hidden(self.trace, hidden_states,
                pool=self.pool, name="sp-final-all-gather")
        return hidden_states


def _source_expert_resolutions(model, step, pool, expert_routing, invocation_ids, rows):
    """Common source routing identity; no model-specific histograms or IDs."""
    from src.serving.schedule.schedule_step import schedule_step_sha256
    from .runtime_inputs import ExpertRoutingInput
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("virtual MoE source requires canonical explicit routing input")
    digest = schedule_step_sha256(step)
    config = pool.parallel_plan.config
    active = tuple(layer for layer in range(model.num_hidden_layers) if model.layer_uses_moe(layer))
    if not active:
        return (None,) * model.num_hidden_layers
    layers = tuple((layer, invocation_ids[layer], ordinal,
        (pool.physical_ranks_for_pipeline_stage(pool.parallel_plan.layer_to_pipeline_stage[layer])
            if config.enable_expert_parallel else
            (pool.physical_ranks_for_pipeline_stage(pool.parallel_plan.layer_to_pipeline_stage[layer])[0],)))
        for ordinal, layer in enumerate(active))
    resolved = expert_routing.resolve_layers(layers, step_id=step.step_id,
        schedule_step_sha256=digest, tokens=rows, num_experts=model.num_experts,
        topk=model.num_experts_per_tok, model_config_sha256=model.config_sha256,
        pool_name=pool.name, tensor_parallel_size=config.tensor_parallel_size,
        expert_parallel_size=config.expert_parallel_size, expert_placement_strategy="linear",
        assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
            if expert_routing.source_kind == "explicit_assumption" else None))
    by_layer = dict(zip(active, resolved, strict=True))
    return tuple(by_layer[layer] if layer in by_layer else None for layer in range(model.num_hidden_layers))



def _plain_greedy_target(trace, model, pool, group, gather, hidden_states, indices, q_lens,
                         *, model_execution_version="vllm_0_30_0", sample_logits=True, cast_logits=False):
    """Shared typed target boundary, keeping indices in the caller's frame."""
    if type(sample_logits) is not bool or type(cast_logits) is not bool or sample_logits and cast_logits:
        raise TypeError("target output flags must be exclusive booleans")
    cumulative, selected = 0, []
    for count in q_lens:
        cumulative += count
        selected.append(cumulative - 1)
    sample_hidden_states = SymbolicLogitsRowSelect(trace, "logits-row-select", pool)(
        hidden_states, indices, selected_indices=tuple(selected), model_execution_version=model_execution_version)
    if group is None:
        logits, _ = SymbolicLinear(trace, "lm-head", pool, model.hidden_size,
            model.vocab_size, kind=WorkloadKind.LM_HEAD)(sample_hidden_states)
    else:
        processor = SymbolicTPLogitsProcessor(trace, "lm-head", pool, hidden_size=model.hidden_size,
            vocab_size=model.vocab_size, group=group, config=gather, model_execution_version=model_execution_version)
        logits = processor(sample_hidden_states)
    return (SymbolicGreedy(trace, "target-sampler", pool)(logits, model_execution_version=model_execution_version,
        sample_logits=sample_logits) if sample_logits or cast_logits else logits)


@dataclass(frozen=True)
class _GLMDecodeSourceBinding:
    program: object
    graph: object
    nodes: tuple
    dynamic_nodes: tuple

    def owner(self, graph):
        from .workload_graph import _node_update_source_unchanged, _workload_node_field_updates_read
        program = self.program()
        if (program is None or self.graph() is not graph or graph.nodes is not self.nodes
                or graph.edges is not program.graph.edges or graph.block_repeats is not program.graph.block_repeats
                or not _DENSE_METADATA_SOURCE_CONTRACT.unchanged()
                or isinstance(program,_GLMDecodeSourceProgram) and not program.source_unchanged()
                or not _node_update_source_unchanged()):
            return None
        for node, shape, attrs in program.static_contract:
            if node.shape is not shape or node.attributes is not attrs:
                return None
        for node in self.dynamic_nodes:
            binding = _workload_node_field_updates_read(node)
            if binding is None or binding[0] is not program.graph.node(node.node_id):
                return None
        return program


def _decode_source_execution_projection(graph):
    """Expose the existing full structural contract and actual current routes."""
    if "_glm_decode_binding" not in graph.__dict__:
        return None
    binding = graph.__dict__["_glm_decode_binding"]
    if type(binding) is not _GLMDecodeSourceBinding:
        return None
    program = binding.owner(graph)
    if program is None:
        return None
    contract = program.execution_contract
    if (graph.logical_node_count != contract[3] or graph.logical_edge_count != contract[4]):
        return None
    routing = []
    for name in program.execution_routing_ids:
        attributes = graph.node(name).attributes
        routing.append((name, attributes["moe_invocation_id"],
            attributes["tokens_per_expert"], attributes["routing_provenance"]))
    return contract, tuple(routing)


class _GLMDecodeSourceProgram(ForwardSourceProgram):
    """Run-local TP8/EP1 q1 source, with a real one-layer parameter prefix.

    This ordinary group binding preserves the complete recorded source DAG
    and every rank's storage. It does not enter the dual-rank native recipe
    domain or extrapolate communication/state from a representative rank.
    """

    def source_unchanged(self):
        return _DENSE_METADATA_SOURCE_CONTRACT.unchanged()

    def __init__(self, source, model, pool):
        from copy import copy
        from types import MappingProxyType
        from .workload_graph import WorkloadGraph
        self.model, self.pool, self.graph = model, pool, source.graph
        # The report's existing structural contract excludes dynamic fields.
        # Retain its complete, ordered DAG once in this experiment's program.
        self.execution_contract = (
            tuple((node.node_id, node.kind.value, node.pool_name, node.rank) for node in self.graph.nodes),
            self.graph.edges, self.graph.block_repeats,
            self.graph.logical_node_count, self.graph.logical_edge_count)
        self.execution_routing_ids = tuple(node.node_id for node in self.graph.nodes
            if node.kind is WorkloadKind.FUSED_MOE)
        self.static_contract = tuple((node, node.shape, node.attributes) for node in self.graph.nodes)
        self.local_graphs = tuple(WorkloadGraph(tuple(trace._nodes.values()), tuple(trace._edges))
            for trace in source.traces)
        self.source = copy(source)
        ranks = []
        for trace, graph in zip(source.traces, self.local_graphs, strict=True):
            rank = copy(trace)
            for name in ("_nodes", "_allocations", "_views", "_references", "_writers", "_names",
                    "_checkpoints", "_workspace_capacities", "_module_program_types", "_module_frames"):
                setattr(rank, name, MappingProxyType(dict(getattr(trace, name))))
            rank._readers = MappingProxyType({name: frozenset(readers) for name, readers in trace._readers.items()})
            rank._events, rank._edges = tuple(trace._events), tuple(trace._edges)
            rank._persistent_ids, rank._collective_ids = frozenset(trace._persistent_ids), frozenset(trace._collective_ids)
            rank._captures = tuple(trace._captures)
            rank._recorded_graph = (rank._nodes, graph)
            ranks.append(rank)
        self.source.traces = tuple(ranks)
        self.source._recorded_graph = self.graph

    def snapshot(self):
        from copy import copy
        result = copy(self.source)
        result.traces = tuple(copy(trace) for trace in self.source.traces)
        object.__setattr__(self.graph, "_glm_decode_binding", _GLMDecodeSourceBinding(
            weakref.ref(self), weakref.ref(self.graph), self.graph.nodes, ()))
        return result

    def bind(self, prefix, *, resolutions, invocation_ids, rows):
        from copy import copy
        from types import MappingProxyType
        result = copy(self.source)
        routes = {}
        for layer in range(self.model.num_hidden_layers):
            if self.model.layer_uses_moe(layer):
                operator = SymbolicRoutedExperts(prefix.traces[0], "binding", "binding", self.model,
                    tp_size=8, layer_index=layer, resolution=resolutions[layer],
                    invocation_id=invocation_ids[layer], routing_source_stage="dsv32_moe_route")
                metadata = operator.routing_metadata(rows)
                routes[layer] = {**metadata, "routing_provenance": MappingProxyType(metadata["routing_provenance"])}
        traces, changed = [], {}
        for template, current, graph in zip(self.source.traces, prefix.traces, self.local_graphs, strict=True):
            trace = copy(template)
            trace._nodes, trace._allocations, trace._views = (
                dict(template._nodes), dict(template._allocations), dict(template._views))
            # These current values are emitted by the actual source primitives.
            attention = current._nodes[current.qualify("layer-0:attention:sparse")]
            indexer = current._nodes[current.qualify("layer-0:attention:indexer")]
            embedding = current._nodes[current.qualify("embedding")]
            embedding_fields = {name: embedding.attributes[name] for name in (
                "embedding_token_ids", "symbolic_model_input_order") if name in embedding.attributes}
            name = trace.qualify("embedding")
            trace._nodes[name] = trace._nodes[name].with_updates({}, embedding_fields)
            changed[name] = trace._nodes[name]
            for layer in range(self.model.num_hidden_layers):
                base = trace.qualify(f"layer-{layer}")
                name = base + ":attention:sparse"
                trace._nodes[name] = trace._nodes[name].with_updates(
                    {field: attention.shape[field] for field in ("kv_len", "topk")},
                    {field: attention.attributes[field] for field in ("kv_lens", "selected_kv_lens")})
                changed[name] = trace._nodes[name]
                if self.model.indexer_types[layer] == "full":
                    name = base + ":attention:indexer"
                    trace._nodes[name] = trace._nodes[name].with_updates(
                        {field: indexer.shape[field] for field in ("kv_len", "topk")},
                        {"kv_lens": indexer.attributes["kv_lens"]})
                    changed[name] = trace._nodes[name]
                if layer in routes:
                    name = base + ":mlp:routed"
                    trace._nodes[name] = trace._nodes[name].with_updates({}, routes[layer])
                    changed[name] = trace._nodes[name]
                for role in (("cache", "index-cache") if self.model.indexer_types[layer] == "full" else ("cache",)):
                    name, origin = base + ":" + role, current.qualify("layer-0:" + role)
                    old = trace._allocations[name]
                    trace._allocations[name] = current._allocations[origin].with_lifetime(
                        name, old.allocated_event, old.freed_event)
                    trace._views[name] = current._views[origin].with_identity(name, name)
            trace._recorded_graph = (trace._nodes, graph.with_nodes(trace._nodes.values()))
            traces.append(trace)
        result.traces = tuple(traces)
        result._recorded_graph = self.graph.with_nodes(
            changed[node.node_id] if node.node_id in changed else node for node in self.graph.nodes)
        graph = result._recorded_graph
        object.__setattr__(graph, "_glm_decode_binding", _GLMDecodeSourceBinding(
            weakref.ref(self), weakref.ref(graph), graph.nodes, tuple(changed.values())))
        return result


def _trace_glm_dsa_source(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
                        cache_capacity_tokens, source_max_model_len, max_batched_tokens,
                        workspace_initial_capacity_bytes, selection_source, source_profile, target_sampling_source,
                        prefill_attention_source=None, source_prompt_lengths=None, include_lm_head=False,
                        source_program_cache=None, num_speculative_tokens=0):
    """One GLM source program, optionally including its plain target chain."""
    from .runtime_inputs import ExpertRoutingInput
    from .runtime_inputs import vllm_0_30_source_model_inputs
    from src.serving.deployment.validated_scope import is_glm_bf16_tensor_source_scope, is_glm_fp8_rank_local_source_scope
    if source_program_cache is not None and type(source_program_cache) is not SourceProgramCache:
        raise TypeError("GLM source reuse requires its run-local SourceProgramCache")
    weight_dtype = _glm_source_weight_dtype(source_profile)
    if type(include_lm_head) is not bool:
        raise TypeError("GLM include_lm_head must be boolean")
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("symbolic GLM target supports only its declared BF16 plain greedy chain")
    config = pool.parallel_plan.config
    ordinary_parallel = (weight_dtype == "bf16" and is_glm_bf16_tensor_source_scope(pool, model)
        or weight_dtype == "fp8" and is_glm_fp8_rank_local_source_scope(pool, model))
    if (model.model_type != "glm_moe_dsa" or model.compute_dtype != "bf16" or model.weight_dtype != weight_dtype
            or (not ordinary_parallel and (pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size not in (1, 8)))
            or len(pool.ranks) != config.tensor_parallel_size
            or (config.enable_expert_parallel and config.expert_parallel_size != 8 and not ordinary_parallel)
            or any(value != 1 for value in (config.pipeline_parallel_size,
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("symbolic GLM decoder requires H200 checkpoint/source agreement, TP1/TP8 PP/SP/CP/DP1, EP-off or EP8")
    if (not step.requests or step.pool_name != pool.name
            or any(r.phase not in ("decode", "prefill") or r.scheduled_tokens <= 0 or r.mtp_proposed_tokens
                or r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests)):
        raise NotImplementedError("symbolic GLM decoder requires all-decode single-token requests with MTP off")
    fresh_dense = (prefill_attention_source is None and any(r.phase == "prefill" for r in step.requests)
        and all((r.phase == "prefill" and r.computed_tokens == 0 and 0 < r.scheduled_tokens <= model.index_topk)
            or (r.phase == "decode" and r.scheduled_tokens == 1) for r in step.requests))
    if any(r.scheduled_tokens > 1 for r in step.requests) and prefill_attention_source != "force_sparse_mqa" and not fresh_dense:
        raise NotImplementedError("GLM all-decode source needs explicit force_sparse_mqa for prefill/mixed")
    if type(num_speculative_tokens) is not int or num_speculative_tokens < 0:
        raise ValueError("GLM target requires a nonnegative declared speculative depth")
    if num_speculative_tokens and any(r.phase != "decode" or r.scheduled_tokens != 1 for r in step.requests):
        raise NotImplementedError("GLM target-only speculative config requires explicit single-row decode")
    rows = sum(r.scheduled_tokens for r in step.requests)
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"symbolic GLM {name} requires one explicit integer per model row")
    if any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("GLM decoder token IDs must belong to the accurate vocabulary")
    if positions != tuple(p for r in step.requests for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens)):
        raise ValueError("GLM decoder positions must match request contexts")
    step, input_ids, model_input_order = vllm_0_30_source_model_inputs(step,
        embedding_token_ids=input_ids, prompt_lengths=source_prompt_lengths, num_speculative_tokens=num_speculative_tokens)
    positions = tuple(positions[row] for row in model_input_order["token_row_permutation"])
    slot_mapping = tuple(slot_mapping[row] for row in model_input_order["token_row_permutation"])
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    phases = tuple(model_input_order["model_attention_phases"])
    if fresh_dense and "prefill" in phases:
        prefill_attention_source = "dense_mha_fresh"
    _dsa_source_phase(q_lens, kv_lens, phases, prefill_attention_source)
    if (type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
            or type(max_batched_tokens) is not int or max_batched_tokens < rows):
        raise ValueError("GLM source context and batch limits must contain every declared row: "
            f"context_limit={source_max_model_len}, native_limit={model.max_position_embeddings}, "
            f"max_kv={max(kv_lens)}, batch_limit={max_batched_tokens}, rows={rows}")
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slot_mapping)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping)):
        raise ValueError("GLM logical cache must cover contexts and distinct valid write slots")
    if (not isinstance(workspace_initial_capacity_bytes, tuple) or len(workspace_initial_capacity_bytes) != len(pool.ranks)
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("GLM source workspace requires explicit entry capacities for every physical rank")
    if selection_source != "declared_causal_topk":
        raise NotImplementedError("GLM decoder requires declared valid causal top-k cardinality, not inferred IDs")
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("GLM decoder requires canonical explicit routing input")
    reusable = (source_program_cache is not None and source_program_cache.glm_ordinary_structural_reuse
        and pool.hardware_id == "nvidia_h200" and len(pool.ranks) == 8
        and config.expert_parallel_size == 1 and all(q == 1 for q in q_lens)
        and phases == ("decode",) * len(phases) and _DENSE_METADATA_SOURCE_CONTRACT.unchanged())
    key = ("glm-q1-source-v1", model, id(pool), q_lens, source_profile, selection_source,
        prefill_attention_source, source_max_model_len, max_batched_tokens,
        workspace_initial_capacity_bytes, include_lm_head, target_sampling_source, num_speculative_tokens) if reusable else None
    program = source_program_cache.lookup(key) if reusable else None
    namespace = step.step_id if program is None else program.source.namespace
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{namespace}:rank-{rank}") for rank in pool.ranks)
    group = None if len(traces) == 1 else ForwardCollectiveGroup(traces, namespace=namespace)
    collective = None if group is None else pool.collective_for("all_reduce")
    target_head = include_lm_head or target_sampling_source is not None
    gather = None if not target_head or group is None else pool.collective_for("all_gather")
    invocation_ids = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocation_ids, rows)

    def rank_forward(trace, capacity, *, prefix=False):
        network = DeepseekV32Model(trace, model, pool.name, group, collective, token_ids=input_ids,
            q_lens=q_lens, kv_lens=kv_lens, request_phases=phases,
            slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            resolutions=resolutions, invocation_ids=invocation_ids, workspace_initial_capacity_bytes=capacity,
            selection_source=selection_source, source_profile=source_profile,
            prefill_attention_source=prefill_attention_source, model_input_order=model_input_order,
            expert_parallel_size=config.expert_parallel_size, layer_range=(0, 1) if prefix else None,
            num_speculative_tokens=num_speculative_tokens)
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        indices = trace.tensor("logits_indices", (len(q_lens),), "int32") if target_head and not prefix else None
        hidden_states = network(ids, pos)
        if target_head and not prefix:
            # _prepare_inputs supplies INT32 query_start_loc[1:]-1. Even
            # identity indices materialize sample_hidden_states in execute_model.
            sampled = _plain_greedy_target(trace, model, pool.name, group, gather, hidden_states, indices, q_lens,
                sample_logits=target_sampling_source == "vllm_0_30_plain_greedy_bf16",
                cast_logits=target_sampling_source == "vllm_0_30_plain_sampler_bf16")

    for trace, capacity in zip(traces, workspace_initial_capacity_bytes, strict=True):
        rank_forward(trace, capacity, prefix=program is not None)
        if ordinary_parallel:
            for identifier, node in tuple(trace._nodes.items()):
                trace._nodes[identifier] = node.with_attributes({**node.attributes,
                    "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
                    "source_projection_only": True, "glm_parallel_source": True})
    if ordinary_parallel and group is not None:
        for identifier, node in tuple(group._contracts.items()):
            group._contracts[identifier] = node.with_attributes({**node.attributes,
                "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile,
                "source_projection_only": True, "glm_parallel_source": True})
    if program is not None:
        return program.bind(group, resolutions=resolutions, invocation_ids=invocation_ids, rows=rows)
    source = traces[0] if group is None else group
    if reusable:
        program = _GLMDecodeSourceProgram(source, model, pool)
        source_program_cache.store(key, program)
        source = program.snapshot()
    return source


def trace_glm_dsa_decoder(model, step, pool, expert_routing, **source_inputs):
    """GLM model.forward scope only; no logits/sampler or Serving dispatch."""
    return _trace_glm_dsa_source(model, step, pool, expert_routing, target_sampling_source=None, **source_inputs)


def trace_glm_dsa_forward(model, step, pool, expert_routing, *, target_sampling_source, **source_inputs):
    """GLM decoder → selected rows → BF16 TP head → plain INT32 greedy.

    Decode or an explicitly forced sparse-MQA prefill/mixed source domain.
    Metadata producer/library/compiler services, full worker memory and
    actual provider remain separate; routing pricing refusal is unchanged.
    """
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("complete GLM target requires an explicit plain BF16 greedy source")
    return _trace_glm_dsa_source(model, step, pool, expert_routing,
        target_sampling_source=target_sampling_source, include_lm_head=True, **source_inputs)


class DeepseekV4MLP:
    """DeepseekV4MLP.forward with TP shards or SP replication, no reduction."""

    def __init__(self, trace, model, pool, prefix, *, tp_size, linear_source_profile):
        from .traced_operators import _v4_source_block_geometry
        if not _v4_source_block_geometry(model, tp_size, expert_parallel=True):
            raise NotImplementedError("symbolic V4 shared MLP requires whole-block output-group TP geometry")
        intermediate = model.moe_intermediate_size * model.num_shared_experts
        # n_local = ceil((intermediate // block_size) / TP), as in source.
        if intermediate % 128:
            raise NotImplementedError("V4 shared block-FP8 source requires complete checkpoint128 blocks")
        local = -(-(intermediate // 128) // tp_size) * 128
        self.gate_up_proj = SymbolicBlockFP8Linear(trace, prefix + ":gate-up", pool,
            model.hidden_size, 2 * local, source_profile=linear_source_profile)
        self.act_fn = SymbolicSiluAndMul(trace, prefix + ":silu-mul", pool, swiglu_limit=model.swiglu_limit)
        self.down_proj = SymbolicBlockFP8Linear(trace, prefix + ":down", pool,
            local, model.hidden_size, source_profile=linear_source_profile)

    def __call__(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV4AttentionInput:
    """V4 attention input projections and fused Q/KV norm, all ratios.

    Adapted from attention.py _run_parallel_input_projections and
    _split_qkv_and_norm. The explicit synchronous source profile preserves
    value dependencies and frame references; it does not model aux-stream
    retirement, CUDA capture or an installed linear provider.
    """

    def __init__(self, trace, model, pool, prefix, *, layer_index, linear_source_profile):
        layers = model.dspark_layer_indices if model.is_dspark_draft else range(model.num_hidden_layers)
        if (model.model_type != "deepseek_v4" or type(layer_index) is not int or layer_index not in layers):
            raise ValueError("V4 attention input requires its actual checkpoint target or draft layer")
        self.compress_ratio = max(1, model.compress_ratios[layer_index])
        self.fused_wqa_wkv = SymbolicBlockFP8Linear(trace, prefix + ":fused-wqa-wkv", pool,
            model.hidden_size, model.q_lora_rank + model.head_dim, source_profile=linear_source_profile)
        self.compressor_projection = None
        self.indexer_weights_proj = None
        self.indexer_compressor_projection = None
        if self.compress_ratio > 1:
            coefficients = 2 if self.compress_ratio == 4 else 1
            self.compressor_projection = SymbolicLinear(trace, prefix + ":compressor-kv-score", pool,
                model.hidden_size, 2 * coefficients * model.head_dim, output_dtype="fp32")
        if self.compress_ratio == 4:
            self.indexer_weights_proj = SymbolicLinear(trace, prefix + ":indexer-weights", pool,
                model.hidden_size, model.index_n_heads)
            self.indexer_compressor_projection = SymbolicLinear(trace, prefix + ":indexer-kv-score", pool,
                model.hidden_size, 4 * model.index_head_dim, output_dtype="fp32")
        self.norm = SymbolicV4QKVNorm(trace, prefix + ":q-kv-rmsnorm", pool, model)

    def _run_parallel_input_projections(self, hidden_states):
        # Source execute_in_parallel returns every auxiliary output, including
        # C4 indexer projections later unused by its short-context return.
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        kv_score = None if self.compressor_projection is None else self.compressor_projection(hidden_states)[0]
        indexer_weights = None if self.indexer_weights_proj is None else self.indexer_weights_proj(hidden_states)[0]
        indexer_kv_score = (None if self.indexer_compressor_projection is None
            else self.indexer_compressor_projection(hidden_states)[0])
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def __call__(self, hidden_states):
        qr_kv, kv_score, indexer_kv_score, indexer_weights = self._run_parallel_input_projections(hidden_states)
        qr, qr_scale, kv = self.norm(qr_kv)
        if qr_scale is not None:
            raise NotImplementedError("V4 SM90 source does not prequantize the q-lora norm result")
        return qr, kv, kv_score, indexer_kv_score, indexer_weights


@dataclass(frozen=True)
class _V4ExpertContext:
    tensor_size: int
    data_size: int = 1
    expert_size: int = 1
    expert_rank: int = 0
    data_group: object = None
    data_rank: int = 0
    rows: tuple = ()
    dispatch: object = None
    combine: object = None
    sequence_parallel: bool = False
    valid_tokens: int | None = None
    local_valid_rows: int | None = None


@dataclass(frozen=True)
class _V4ReplicaContext:
    traces: tuple
    tensor_groups: tuple
    data_groups: tuple
    rows: tuple
    rank: int
    resolutions: tuple
    invocations: tuple
    expert_size: int
    sequence_parallel: bool = False
    expert_groups: tuple = ()
    valid_rows: tuple = ()


@dataclass(frozen=True)
class _V4SequenceContext:
    group: ForwardCollectiveGroup
    gather_config: object
    combine_config: object
    rows: int
    lane: int

    @property
    def shard_rows(self):
        return -(-self.rows // len(self.group.traces))

    def pad(self, trace, value, *, pool, name):
        from .traced_operators import SymbolicFillZero
        if value.shape[0] != self.rows:
            raise ValueError("V4 SP input must contain exactly the actual request rows")
        padded = self.shard_rows * len(self.group.traces)
        if padded == self.rows:
            return value
        output = trace.tensor(trace.unique_name(name + ":padded"), (padded, *value.shape[1:]), value.dtype)
        SymbolicCopy(trace, name + ":copy", pool)(value, output.narrow(0, 0, self.rows))
        SymbolicFillZero(trace, name + ":pad-zero", pool)(output.narrow(0, self.rows, padded - self.rows))
        return output

    def shard(self, trace, value, *, pool, name):
        return self.pad(trace, value, pool=pool, name=name).narrow(0, self.lane * self.shard_rows, self.shard_rows)

    def gather(self, value, *, pool, name):
        gathered = self.group.call(name, value, pool_name=pool, config=self.gather_config, inplace=False, dim=0)
        return gathered if gathered.shape[0] == self.rows else gathered.narrow(0, 0, self.rows)

    def reduce(self, trace, value, *, pool, name):
        padded = self.pad(trace, value, pool=pool, name=name)
        return self.group.call_variable(name, trace, padded, pool_name=pool, config=self.combine_config,
            rows_by_rank=(self.shard_rows,) * len(self.group.traces), row_shape=value.shape[1:],
            dtype=value.dtype, group_dimension="tensor")


class DeepseekV4MoERunner:
    """Synchronous modular MoERunner source frames, no LoRA/overlap.

    Separate Python frames preserve the source release of logits and route
    outputs before the final shared-add/all-reduce. Unknown expert workspace
    remains declared on its node; this is not complete memory acceptance.
    """

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index,
                 resolution, invocation_id, linear_source_profile, workspace=None, source_multiprocessors=None,
                 expert_context=None):
        self.gate = SymbolicLinear(trace, prefix + ":router", pool,
            model.hidden_size, model.num_experts, kind=WorkloadKind.ROUTER, output_dtype="fp32")
        tp = 1 if group is None else len(group.traces)
        if (group is not None and trace not in group.traces
                or expert_context is not None and (not isinstance(expert_context, _V4ExpertContext)
                    or expert_context.tensor_size != tp)):
            raise ValueError("V4 MoE requires its actual physical TP group and normalized expert context")
        context = _V4ExpertContext(tp) if expert_context is None else expert_context
        self.trace, self.model, self.context, self.invocation_id = trace, model, context, invocation_id
        self.shared_experts = DeepseekV4MLP(trace, model, pool, prefix + ":shared", tp_size=1 if context.sequence_parallel else tp,
            linear_source_profile=linear_source_profile)
        self.router = SymbolicV4Routing(trace, prefix + ":route", pool, model, layer_index=layer_index,
            valid_rows=context.local_valid_rows if layer_index < model.num_hash_layers else None)
        self.routed_experts = SymbolicV4RoutedExperts(trace, prefix + ":routed", pool, model,
            tp_size=tp, resolution=resolution, invocation_id=invocation_id, workspace=workspace,
            source_multiprocessors=source_multiprocessors, expert_parallel_size=context.expert_size,
            expert_rank=context.expert_rank, expert_tp_size=1 if context.expert_size > 1 else tp * context.data_size,
            valid_tokens=context.valid_tokens if layer_index < model.num_hash_layers else None)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)
        self.group, self.config, self.pool, self.prefix = group, config, pool, prefix

    def _apply_quant_method(self, hidden_states, router_logits, input_ids):
        shared_output = None if hidden_states is None else self.shared_experts(hidden_states)
        topk_weights, topk_ids = (None, None) if hidden_states is None else self.router(hidden_states, router_logits, input_ids)
        context = self.context
        dispatched = hidden_states
        if context.data_group is not None:
            gathered = tuple(context.data_group.call_variable(self.prefix + ":dispatch-" + role, self.trace, value,
                pool_name=self.pool, config=context.dispatch, rows_by_rank=context.rows,
                row_shape=(width,), dtype=dtype, group_dimension="expert" if context.sequence_parallel else "data")
                for role, value, width, dtype in (("hidden", hidden_states, self.model.hidden_size, "bf16"),
                    ("weights", topk_weights, self.model.num_experts_per_tok, "fp32"),
                    ("ids", topk_ids, self.model.num_experts_per_tok, "int32")))
            dispatched, topk_weights, topk_ids = gathered
        fused_output = self.routed_experts(dispatched, topk_weights, topk_ids)
        if context.data_group is not None:
            combined = context.data_group.call_variable(self.prefix + ":combine", self.trace, fused_output,
                pool_name=self.pool, config=context.combine, rows_by_rank=context.rows,
                row_shape=(self.model.hidden_size,), dtype="bf16", group_dimension="expert" if context.sequence_parallel else "data",
                moe_invocation_id=self.invocation_id)
            if hidden_states is None:
                return None, None
            target = self.trace.tensor(self.trace.unique_name(self.prefix + ":finalize-output"), hidden_states.shape, "bf16")
            fused_output = SymbolicCopy(self.trace, self.prefix + ":finalize-copy", self.pool)(combined, target)
        return shared_output, fused_output

    def _forward_impl(self, hidden_states, input_ids):
        router_logits = None if hidden_states is None else self.gate(hidden_states)[0]
        return self._apply_quant_method(hidden_states, router_logits, input_ids)

    def __call__(self, *, hidden_states, router_logits, input_ids):
        if router_logits is not hidden_states:
            raise ValueError("V4 Runner must own gate execution from its source hidden-state placeholder")
        shared_output, fused_output = self._forward_impl(hidden_states, input_ids)
        if hidden_states is None:
            return None
        result = self.add(shared_output, fused_output)
        if self.group is None or self.context.sequence_parallel:
            return result
        return self.group.call(self.prefix + ":all-reduce", result,
            pool_name=self.pool, config=self.config, inplace=False)


class DeepseekV4OutputProjection:
    """deep_gemm_fp8_o_proj source: inverse RoPE, grouped wo_a, row wo_b."""

    def __init__(self, trace, model, pool, prefix, group, config, rotary_cache, *, linear_source_profile, reduce_results=True):
        tp_size = 1 if group is None else len(group.traces)
        self.inverse_rope = SymbolicV4InverseRope(trace, prefix + ":inverse-rope", pool, model, tp_size=tp_size)
        self.wo_a = SymbolicV4GroupedOutput(trace, prefix + ":wo-a", pool, model, tp_size=tp_size)
        self.wo_b = SymbolicBlockFP8Linear(trace, prefix + ":wo-b", pool,
            model.o_groups // tp_size * model.o_lora_rank, model.hidden_size, source_profile=linear_source_profile)
        self.rotary_cache = rotary_cache
        self.group, self.config, self.pool, self.prefix = group, config, pool, prefix
        if type(reduce_results) is not bool or group is not None and trace not in group.traces:
            raise ValueError("V4 attention output requires its actual TP group and explicit reduction branch")
        self.reduce_results = reduce_results

    def __call__(self, o, positions):
        o_proj_input, o_scale = self.inverse_rope(o, positions, self.rotary_cache)
        z = self.wo_a(o_proj_input, o_scale)
        output, _ = self.wo_b(z.view_as_shape(z.shape[0], z.shape[1] * z.shape[2]))
        if self.group is None or not self.reduce_results:
            return output
        return self.group.call(self.prefix + ":all-reduce", output,
            pool_name=self.pool, config=self.config, inplace=False)


class DeepseekV4Attention:
    """Synchronous V4 FlashMLA forward with explicit scored-C4 applicability.

    C4's short-context path omits query/scoring, retaining both input
    projections and its compressor. C128 metadata tensors are explicit
    external forward inputs. Cache arenas are logical declarations, not a
    binding to the normal/native physical page allocator.
    """

    def __init__(self, trace, model, pool, prefix, group, config, rotary_cache, topk_buffer, *,
                 layer_index, q_lens, kv_lens, request_phases, slot_mapping,
                 cache_capacity_tokens, source_max_model_len, linear_source_profile, indexer_selection_source=None,
                 workspace=None, source_max_batched_tokens=None, cache_tensors=None, slot_tensor=None, reduce_results=True, swa_request_phases=None,dummy_run=False):
        from .runtime_inputs import dsv4_c128_metadata_width
        if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < 128
                or cache_capacity_tokens % 8 or cache_capacity_tokens < sum(kv_lens)):
            raise ValueError("V4 source logical arenas require explicit8-row capacity covering active contexts")
        self.trace, self.model, self.prefix = trace, model, prefix
        self.rotary_cache = rotary_cache
        tp_size = 1 if group is None else len(group.traces)
        self.local_heads = model.num_attention_heads // tp_size
        self.padded_heads = 64 if self.local_heads <= 64 else 128
        ratio = max(1, model.compress_ratios[layer_index])
        roles = {"swa-cache"}
        if ratio > 1:
            roles.update(("partial-state", "compressed-cache"))
        if ratio == 4:
            roles.update(("indexer-partial-state", "indexer-cache"))
        if cache_tensors is not None and (not isinstance(cache_tensors, tuple)
                or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in cache_tensors)
                or len(cache_tensors) != len(roles) or {pair[0] for pair in cache_tensors} != roles):
            raise ValueError("V4 cache binding must provide exactly this layer's named persistent tensors")
        supplied = None if cache_tensors is None else dict(cache_tensors)
        caches = {}

        def cache(role, shape, dtype):
            tensor = (trace.tensor(prefix + ":" + role, shape, dtype, persistent=True)
                if supplied is None else supplied[role])
            trace._check_tensor(tensor)
            if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous:
                raise ValueError("V4 cache binding shape/dtype/layout differs from the declared arena")
            caches[role] = tensor
            return tensor

        long_c4 = ratio == 4 and max(kv_lens) // 4 > model.index_topk
        if long_c4 and indexer_selection_source != "declared_causal_topk":
            raise NotImplementedError("long C4 requires an explicit selected-cardinality declaration")
        self.inputs = DeepseekV4AttentionInput(trace, model, pool, prefix, layer_index=layer_index,
            linear_source_profile=linear_source_profile)
        self.wq_b = SymbolicBlockFP8Linear(trace, prefix + ":wq-b", pool,
            model.q_lora_rank, self.local_heads * model.head_dim, source_profile=linear_source_profile)
        self.swa_cache = cache("swa-cache", (cache_capacity_tokens, 584), "uint8")
        self.insert = SymbolicV4KVInsert(trace, prefix + ":swa-insert", pool, model,
            tp_size=tp_size, cache=self.swa_cache, slot_mapping=slot_mapping, slot_tensor=slot_tensor,dummy_run=dummy_run)
        self.compressor, self.indexer_compressor, self.short_topk = None, None, None
        self.indexer_wq_b, self.indexer_query, self.scored_topk = None, None, None
        compressed_cache = None
        if ratio > 1:
            coff, block = (2, 4) if ratio == 4 else (1, 8)
            partial = cache("partial-state", (cache_capacity_tokens // block, block,
                2 * coff * model.head_dim), "fp32")
            compressed_cache = cache("compressed-cache", (cache_capacity_tokens // ratio, 584), "uint8")
            self.compressor = SymbolicV4Compressor(trace, prefix + ":compressor", pool,
                head_dim=model.head_dim, compress_ratio=ratio, q_lens=q_lens, kv_lens=kv_lens,
                state_cache=partial, kv_cache=compressed_cache, graph_runtime_mode="NONE", slot_mapping=slot_mapping,dummy_run=dummy_run)
        if ratio == 4:
            index_partial = cache("indexer-partial-state",
                (cache_capacity_tokens // 4, 4, 4 * model.index_head_dim), "fp32")
            index_cache = cache("indexer-cache", (cache_capacity_tokens // 4, 132), "uint8")
            self.indexer_compressor = SymbolicV4Compressor(trace, prefix + ":indexer-compressor", pool,
                head_dim=model.index_head_dim, compress_ratio=4, q_lens=q_lens, kv_lens=kv_lens,
                state_cache=index_partial, kv_cache=index_cache, graph_runtime_mode="NONE", slot_mapping=slot_mapping,dummy_run=dummy_run)
            if long_c4:
                self.indexer_wq_b = SymbolicBlockFP8Linear(trace, prefix + ":indexer-wq-b", pool,
                    model.q_lora_rank, model.index_n_heads * model.index_head_dim, source_profile=linear_source_profile)
                self.indexer_query = SymbolicV4IndexerQuery(trace, prefix + ":indexer-query", pool, model)
                self.scored_topk = SymbolicV4ScoredTopK(trace, prefix + ":indexer-score-topk", pool, model,
                    topk_buffer, index_cache, q_lens=q_lens, kv_lens=kv_lens,
                    request_phases=request_phases, selection_source=indexer_selection_source,
                    workspace=workspace, source_max_model_len=source_max_model_len)
            else:
                self.short_topk = SymbolicV4ShortTopK(trace, prefix + ":short-topk", pool, topk_buffer)
        c128_indices = (None, None)
        if ratio == 128:
            capacity = dsv4_c128_metadata_width(max(kv_lens), source_max_model_len)
            # The metadata builder provides two distinct external input buffers.
            # Creating their descriptors does not price/build metadata on GPU.
            counts = tuple(sum(q for q, p in zip(q_lens,
                request_phases if swa_request_phases is None else swa_request_phases, strict=True) if p == phase)
                for phase in ("prefill", "decode"))
            c128_indices = tuple(None if count == 0 else trace.tensor(trace.unique_name(prefix + ":c128-" + phase + "-indices"),
                (count, capacity), "int32", persistent=True)
                for phase, count in zip(("prefill", "decode"), counts, strict=True))
        self.attn = SymbolicV4SparseAttention(trace, prefix + ":sparse", pool, model,
            compress_ratio=ratio, q_lens=q_lens, kv_lens=kv_lens,
            request_phases=request_phases if swa_request_phases is None else swa_request_phases,
            swa_cache=self.swa_cache, compressed_cache=compressed_cache, topk_buffer=topk_buffer,
            c128_indices=c128_indices, source_max_model_len=source_max_model_len,
            indexer_selection_source=indexer_selection_source, workspace=workspace,
            source_max_batched_tokens=source_max_batched_tokens,dummy_run=dummy_run)
        self.o_proj = DeepseekV4OutputProjection(trace, model, pool, prefix + ":out", group, config,
            rotary_cache, linear_source_profile=linear_source_profile, reduce_results=reduce_results)
        self.cache_tensors = tuple(caches.items())

    def __call__(self, positions, hidden_states, llama_4_scaling=None):
        if llama_4_scaling is not None:
            raise NotImplementedError("V4 source profile does not consume llama4 scaling")
        rows = hidden_states.shape[0]
        o_padded = self.trace.tensor(self.trace.unique_name(self.prefix + ":o-padded"),
            (rows, self.padded_heads, self.model.head_dim), "bf16")
        qr_kv, kv_score, indexer_kv_score, indexer_weights = self.inputs._run_parallel_input_projections(hidden_states)
        qr, qr_scale, kv = self.inputs.norm(qr_kv)
        if qr_scale is not None:
            raise NotImplementedError("V4 SM90 source does not prequantize the q-lora norm result")
        self._prepare_and_attn(hidden_states, qr, kv, qr_scale, kv_score,
            indexer_kv_score, indexer_weights, positions, o_padded)
        o = o_padded.narrow(1, 0, self.local_heads)
        return self.o_proj(o, positions)

    def _prepare_and_attn(self, hidden_states, qr, kv, qr_scale, kv_score,
                          indexer_kv_score, indexer_weights, positions, o_padded):
        # The upstream helper owns Q/indexer outputs. Its return releases
        # those locals before forward enters the inverse-RoPE/output path.
        # The caller still owns qr/kv and the input projection results.
        rows = hidden_states.shape[0]
        q, _ = self.wq_b(qr)
        q = q.view_as_shape(rows, self.local_heads, self.model.head_dim)
        q = self.insert(q, kv, positions, self.rotary_cache)
        index_q, index_weights_out = None, None
        if self.indexer_wq_b is not None:
            index_q, _ = self.indexer_wq_b(qr)
            index_q = index_q.view_as_shape(rows, self.model.index_n_heads, self.model.index_head_dim)
            index_q, index_weights_out = self.indexer_query(index_q, indexer_weights, positions, self.rotary_cache)
        controls = []
        if self.indexer_compressor is not None:
            self.indexer_compressor(indexer_kv_score, positions, self.rotary_cache)
            if self.short_topk is not None:
                self.short_topk(positions, controls=(self.indexer_compressor.kv_cache,))
            controls.append(self.indexer_compressor.state_cache)
        if self.compressor is not None:
            self.compressor(kv_score, positions, self.rotary_cache)
            controls.append(self.compressor.state_cache)
        if self.scored_topk is not None:
            self.scored_topk(index_q, index_weights_out,
                controls=(self.swa_cache, self.compressor.kv_cache, *controls))
        self.attn(q, positions, o_padded, controls=tuple(controls))


class DeepseekV4ProfileAttention(DeepseekV4Attention):
    """The same forward with vLLM's explicit no-metadata profile helpers.

    This is a storage program: no real cache/selector/attention execution,
    no timing claim. Query/output projections still execute symbolically.
    Workspace and dummy logits are distinct source allocations, never an
    inferred correction to an observed activation peak.
    """

    def __init__(self, trace, model, pool, prefix, group, config, rotary_cache, topk_buffer, *,
                 layer_index, workspace, source_max_model_len, source_max_batched_tokens,
                 linear_source_profile, indexer_max_logits_mb):
        from .traced_sampling import StorageOnlyOps
        if (not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace
                or any(type(value) is not int or value <= 0 for value in (
                    source_max_model_len, source_max_batched_tokens, indexer_max_logits_mb))):
            raise ValueError("V4 profile requires explicit rank workspace, context/batch limits and indexer logits MiB")
        self.trace, self.model, self.prefix = trace, model, prefix
        self.local_heads = model.num_attention_heads // len(group.traces)
        self.padded_heads = 64 if self.local_heads <= 64 else 128
        self.ratio = max(1, model.compress_ratios[layer_index])
        self.workspace, self.max_model_len = workspace, source_max_model_len
        self.max_batched_tokens, self.indexer_max_logits_mb = source_max_batched_tokens, indexer_max_logits_mb
        self.rotary_cache, self.topk_buffer = rotary_cache, topk_buffer
        self.ops = StorageOnlyOps(trace, prefix + ":profile", pool)
        self.inputs = DeepseekV4AttentionInput(trace, model, pool, prefix, layer_index=layer_index,
            linear_source_profile=linear_source_profile)
        self.wq_b = SymbolicBlockFP8Linear(trace, prefix + ":wq-b", pool,
            model.q_lora_rank, self.local_heads * model.head_dim, source_profile=linear_source_profile)
        self.indexer_wq_b, self.indexer_query = None, None
        if self.ratio == 4:
            self.indexer_wq_b = SymbolicBlockFP8Linear(trace, prefix + ":indexer-wq-b", pool,
                model.q_lora_rank, model.index_n_heads * model.index_head_dim, source_profile=linear_source_profile)
            self.indexer_query = SymbolicV4IndexerQuery(trace, prefix + ":indexer-query", pool, model)
        self.o_proj = DeepseekV4OutputProjection(trace, model, pool, prefix + ":out", group, config,
            rotary_cache, linear_source_profile=linear_source_profile)

    def _profile_compressor(self, kv_score, head_dim):
        # compressor.py splits views, then returns before APE/save/finish.
        coff = 2 if self.ratio == 4 else 1
        kv, score = kv_score.split((coff * head_dim, coff * head_dim), dim=-1)

    def _profile_indexer_inputs(self, qr, indexer_weights, indexer_kv_score, positions):
        q, _ = self.indexer_wq_b(qr)
        q = q.view_as_shape(q.shape[0], self.model.index_n_heads, self.model.index_head_dim)
        result = self.indexer_query(q, indexer_weights, positions, self.rotary_cache)
        self._profile_compressor(indexer_kv_score, self.model.index_head_dim)
        return result

    def _profile_indexer(self, hidden_states, q_quant, weights):
        # sparse_attn_indexer.py: no-metadata get_simultaneous result is
        # discarded; its arena persists via the manager. The UINT8 dummy
        # allocation is local to this call, not the whole model forward.
        capacity = self.max_model_len * 40 // 4
        self.workspace.get_simultaneous(((capacity, self.model.index_head_dim), "fp8"),
            ((capacity, 4), "uint8"), ((1024 * 1024,), "uint8"))
        dummy_logits = self.trace.tensor(self.trace.unique_name(self.prefix + ":profile-logits"),
            (self.indexer_max_logits_mb * 1024 * 1024,), "uint8")
        return self.topk_buffer

    def _profile_sparse_attention(self, q, o_padded):
        # nvidia/flashmla.py reserves the maximum dummy-prefill ports,
        # independently of the active request lengths, then zeroes output.
        if q.shape != o_padded.shape or q.dtype != o_padded.dtype:
            raise ValueError("V4 profile Q and zeroed output require matching padded shape/dtype")
        n = 0 if self.ratio == 1 else -(-self.max_model_len // self.ratio)
        width = n + self.model.window_size + self.max_batched_tokens
        topk = 0 if self.ratio == 1 else self.topk_buffer.shape[-1]
        combined = -(-(topk + self.model.window_size) // 128) * 128
        self.workspace.get_simultaneous(((4, width, self.model.head_dim), "bf16"),
            ((self.max_batched_tokens, combined), "int32"), ((self.max_batched_tokens,), "int32"))
        self.ops.operation("no_metadata_flashmla_zero_output", (),
            shape=o_padded.shape, dtype=o_padded.dtype, out=o_padded)

    def _prepare_and_attn(self, hidden_states, qr, kv, qr_scale, kv_score,
                          indexer_kv_score, indexer_weights, positions, o_padded):
        q, _ = self.wq_b(qr)
        q = q.view_as_shape(q.shape[0], self.local_heads, self.model.head_dim)
        if self.local_heads < self.padded_heads:
            q = self.ops.unary("no_metadata_pad_query_heads", q,
                shape=(q.shape[0], self.padded_heads, self.model.head_dim))
        index_q, index_weights_out = None, None
        if self.indexer_wq_b is not None:
            index_q, index_weights_out = self._profile_indexer_inputs(qr, indexer_weights,
                indexer_kv_score, positions)
        if self.ratio > 1:
            self._profile_compressor(kv_score, self.model.head_dim)
        if index_q is not None:
            self._profile_indexer(hidden_states, index_q, index_weights_out)
        self._profile_sparse_attention(q, o_padded)


class DeepseekV4DecoderLayer:
    """Non-MegaMoE DeepseekV4DecoderLayer.forward on virtual modules."""

    def __init__(self, trace, model, pool, prefix, group, config, rotary_cache, topk_buffer, *,
                 layer_index, q_lens, kv_lens, request_phases, slot_mapping, cache_capacity_tokens,
                 source_max_model_len, linear_source_profile, source_use_deep_gemm, hardware,
                 resolution, invocation_id, indexer_selection_source=None, workspace=None, source_max_batched_tokens=None,
                 profile_indexer_max_logits_mb=None, ffn=None, mhc_workspace_scope="source_launch", cache_tensors=None, slot_tensor=None,
                 expert_context=None, sequence_context=None, swa_request_phases=None,dummy_run=False):
        self.pre = SymbolicMHC(trace, prefix + ":pre", pool, model, hardware,
            variant="mhc_pre", source_use_deep_gemm=source_use_deep_gemm, workspace_scope=mhc_workspace_scope)
        self.sequence, self.trace, self.pool, self.prefix = sequence_context, trace, pool, prefix
        self.attn_pre = SymbolicMHC(trace, prefix + ":attn-post-pre", pool, model, hardware,
            variant="mhc_post_pre", source_use_deep_gemm=source_use_deep_gemm, workspace_scope=mhc_workspace_scope)
        self.ffn_pre = SymbolicMHC(trace, prefix + ":ffn-post-pre", pool, model, hardware,
            variant="mhc_post_pre", source_use_deep_gemm=source_use_deep_gemm, workspace_scope=mhc_workspace_scope)
        if profile_indexer_max_logits_mb is None:
            self.attn = DeepseekV4Attention(trace, model, pool, prefix + ":attn", group, config, rotary_cache, topk_buffer,
                layer_index=layer_index, q_lens=q_lens, kv_lens=kv_lens, request_phases=request_phases,
                slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
                source_max_model_len=source_max_model_len, linear_source_profile=linear_source_profile,
                indexer_selection_source=indexer_selection_source, workspace=workspace,
                source_max_batched_tokens=source_max_batched_tokens, cache_tensors=cache_tensors, slot_tensor=slot_tensor,
                reduce_results=sequence_context is None, swa_request_phases=swa_request_phases,dummy_run=dummy_run)
        else:
            if cache_tensors is not None:
                raise ValueError("no-metadata profile does not consume an existing request cache")
            self.attn = DeepseekV4ProfileAttention(trace, model, pool, prefix + ":attn", group, config,
                rotary_cache, topk_buffer, layer_index=layer_index, workspace=workspace,
                source_max_model_len=source_max_model_len, source_max_batched_tokens=source_max_batched_tokens,
                linear_source_profile=linear_source_profile, indexer_max_logits_mb=profile_indexer_max_logits_mb)
        self.ffn = ffn if ffn is not None else DeepseekV4MoE(trace, model, pool, prefix + ":ffn", group, config,
            layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
            linear_source_profile=linear_source_profile, workspace=workspace,
            source_multiprocessors=int(hardware.num_compute_units), expert_context=expert_context)

    def prepare_ffn(self, x, positions, post_mix=None, res_mix=None, residual=None):
        """Trace attention and return the exact FFN inputs for staged execution."""
        if residual is None:
            residual, post_mix, res_mix, x = self.pre(x)
        else:
            residual, post_mix, res_mix, x = self.attn_pre(x, residual, post_mix, res_mix)
        if self.sequence is not None:
            x = self.sequence.gather(x, pool=self.pool, name=self.prefix + ":sp-attention-gather")
        x = self.attn(positions, x, None)
        if self.sequence is not None:
            x = self.sequence.reduce(self.trace, x, pool=self.pool, name=self.prefix + ":sp-attention-scatter")
        residual, post_mix, res_mix, x = self.ffn_pre(x, residual, post_mix, res_mix)
        return x, residual, post_mix, res_mix

    def __call__(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
        x, residual, post_mix, res_mix = self.prepare_ffn(
            x, positions, post_mix, res_mix, residual)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix


@dataclass(frozen=True)
class _V4TargetRankState:
    trace: object
    layer_range: tuple
    caches: tuple
    topk: object
    ropes: tuple
    broadcast: object
    workspace: object


@dataclass(frozen=True)
class _V4TargetSourceState:
    owner: object
    model_sha256: str
    config: object
    ranks: tuple
    pool_name: str
    step: object
    step_ids: tuple
    bounds: tuple
    source_branch: tuple
    aux_layers: tuple
    stages: tuple
    groups: tuple
    rank_states: tuple
    intermediate_buffers: dict
    slot_histories: tuple
    capture_producers: tuple
    sampler_feedback: tuple = ()
    v2_input_ports: tuple = ()
    dummy_run: bool = False
    v2_state_indices: tuple = ()
    retained_sampling_ports: tuple = ()
    admitted_request_ids: tuple = ()


@dataclass(frozen=True)
class _V4DistributedTargetState:
    owner: object
    config: object
    ranks: tuple
    states: tuple
    tensor_groups: tuple
    data_groups: tuple
    expert_groups: tuple


def _v4_target_continuation(prior, model, step, pool, *, bounds, source_branch,
        aux_layers, workspace_entries,dummy_run=False,admitted_request_ids=()):
    """Validate the whole previous source revision before any mutation."""
    if (type(prior) is not _V4TargetSourceState or '_v4_target_source_state' not in vars(prior.owner)
            or prior.owner._v4_target_source_state is not prior):
        raise ValueError("V4 continuation requires its current source-owned target revision")
    old_requests={} if prior.dummy_run else {request.request_id:request for request in prior.step.requests}
    current_requests=() if dummy_run else step.requests
    branch_matches=(prior.source_branch==source_branch or dummy_run and not prior.dummy_run
        and prior.source_branch[:3]==source_branch[:3] and prior.source_branch[-1] is not None
        and prior.source_branch[-1]==source_branch[-1]
        and prior.source_branch[3:-1]==('v2_plain_greedy',) and source_branch[3:-1]==('dummy_uniform_decode',)
        or not dummy_run and prior.dummy_run and bool(admitted_request_ids)
        and prior.source_branch[:3]==source_branch[:3] and prior.source_branch[-1] is not None
        and prior.source_branch[-1]==source_branch[-1]
        and prior.source_branch[3:-1]==('dummy_uniform_decode',) and source_branch[3:-1]==('v2_plain_greedy',))
    if (prior.model_sha256 != model.config_sha256 or prior.config != pool.parallel_plan.config
            or prior.ranks != pool.ranks or prior.pool_name != pool.name or prior.bounds != bounds
            or not branch_matches or prior.aux_layers != aux_layers
            or step.step_id in prior.step_ids
            or set(admitted_request_ids).intersection(set(old_requests)-set(prior.step.finished_request_ids))
            or {r.request_id for r in current_requests} != (set(old_requests)-set(prior.step.finished_request_ids))|set(admitted_request_ids)
            or any(new.computed_tokens < old.computed_tokens
                or new.computed_tokens > old.computed_tokens + old.scheduled_tokens
                or new.data_parallel_rank != old.data_parallel_rank
                for new in current_requests if new.request_id not in admitted_request_ids for old in (old_requests[new.request_id],))):
        raise ValueError("V4 continuation changed model, topology, limits, branch, request owners or accepted prefix")
    mapping=pool.parallel_plan.layer_to_pipeline_stage
    pp=prior.config.pipeline_parallel_size
    if (pool.parallel_plan.num_layers!=model.num_hidden_layers or pool.parallel_plan.num_experts!=model.num_experts
            or len(mapping)!=model.num_hidden_layers or set(mapping)!=set(range(pp))):
        raise ValueError("V4 continuation plan differs from its complete target layer/expert source")
    ranges=tuple((mapping.index(stage),mapping.index(stage)+mapping.count(stage)) for stage in range(pp))
    if (tuple(stage for stage,(first,end) in enumerate(ranges) for _ in range(first,end))!=mapping
            or any(state.layer_range!=ranges[ordinal//prior.config.tensor_parallel_size]
                for ordinal,state in enumerate(prior.rank_states))
            or tuple(tuple(trace.rank for trace in stage) for stage in prior.stages)
                !=tuple(pool.physical_ranks_for_pipeline_stage(stage) for stage in range(pp))):
        raise ValueError("V4 continuation PP layer cuts or physical stage grouping changed")
    if not prior.dummy_run and 'dspark_cache_state' in vars(prior.owner):
        cache=prior.owner.dspark_cache_state
        if (type(cache) is not _DSparkCacheState or cache.owner is not prior.owner
                or cache.model_sha256!=prior.source_branch[-1]
                or {r.request_id:(r.data_parallel_rank,r.computed_tokens) for r in cache.step.requests
                    if r.request_id not in prior.step.finished_request_ids}
                    !={r.request_id:(r.data_parallel_rank,r.computed_tokens) for r in current_requests
                        if r.request_id not in admitted_request_ids}):
            raise ValueError("V4 next target must start at its current draft revision's exact accepted context")
    if prior.sampler_feedback and not dummy_run:
        rejected=prior.sampler_feedback[0][2]
        rejected_by_id=dict(zip((r.request_id for r in prior.step.requests),rejected,strict=True))
        if (any(record[2]!=rejected for record in prior.sampler_feedback)
                or any(new.computed_tokens!=old.computed_tokens+old.scheduled_tokens-rejected_by_id[new.request_id]
                    for new in step.requests if new.request_id not in admitted_request_ids for old in (old_requests[new.request_id],))):
            raise ValueError('V4 next target context must match its prior V2 declared rejection scenario')
    if tuple(state.workspace.capacity_bytes for state in prior.rank_states) != workspace_entries:
        raise ValueError("V4 continuation requires the exact current workspace entry on every rank")
    for rank,state in zip(pool.ranks,prior.rank_states,strict=True):
        if (state.trace.rank != rank or state.trace._recorded_graph is not None
                or state.workspace.trace is not state.trace):
            raise ValueError("V4 continuation requires the open original rank and workspace owner")
        for pairs in state.caches:
            for role,tensor in pairs:
                state.trace._check_tensor(tensor)
                if (tensor.view.allocation_id not in state.trace._persistent_ids
                        or state.trace._allocations[tensor.view.allocation_id].freed_event is not None):
                    raise ValueError("V4 continuation cache lost its original live persistent backing")
        for name,tensor in state.ropes:
            state.trace._check_tensor(tensor)
        state.trace._check_tensor(state.topk)
    tp=prior.config.tensor_parallel_size
    expected_ranks={trace.rank for stage in prior.stages[1:] for trace in stage}
    if set(prior.intermediate_buffers)!=expected_ranks:
        raise ValueError("V4 continuation requires every original PP intermediate buffer owner")
    for stage,traces in enumerate(prior.stages[1:],start=1):
        first=prior.rank_states[stage*tp].layer_range[0]
        roles={'hidden_states',*(f'aux_hidden_states_{index}' for index,layer in enumerate(prior.aux_layers) if layer<first)}
        if any(set(prior.intermediate_buffers[trace.rank])!=roles for trace in traces):
            raise ValueError("V4 continuation PP intermediate roles differ from their source constructor")
    for rank,buffers in prior.intermediate_buffers.items():
        state=next(state for state in prior.rank_states if state.trace.rank==rank)
        for name,tensor in buffers.items():
            state.trace._check_tensor(tensor)
            expected=((bounds[2],model.hidden_size) if name.startswith('aux_hidden_states_')
                else (bounds[2],model.hc_mult,model.hidden_size))
            if tensor.dtype!='bf16' or tensor.shape!=expected or tensor.view.allocation_id not in state.trace._persistent_ids:
                raise ValueError("V4 continuation intermediate buffer differs from its source constructor")
    return prior.stages,prior.groups


class DeepseekV4Model:
    """V4 target stage loop; source metadata/provider are explicit."""

    def __init__(self, trace, model, pool, group, config, *, token_ids, q_lens, kv_lens, request_phases,
                 slot_mapping, cache_capacity_tokens, source_max_model_len, max_batched_tokens,
                 linear_source_profile, source_use_deep_gemm, hardware, resolutions, invocation_ids, model_input_order,
                 indexer_selection_source=None, workspace_initial_capacity_bytes=None, profile_indexer_max_logits_mb=None,
                 slot_tensor=None, layer_range=None, expert_context=None, sequence_context=None,
                 aux_hidden_state_layers=(), _source_state=None, _retain_source_state=False, swa_request_phases=None,dummy_run=False):
        start, end = (0, model.num_hidden_layers) if layer_range is None else layer_range
        if not 0 <= start < end <= model.num_hidden_layers:
            raise ValueError("V4 stage requires a nonempty contiguous checkpoint layer range")
        self.trace = trace
        if (type(aux_hidden_state_layers) is not tuple or any(type(layer) is not int
                or not 0 <= layer < model.num_hidden_layers for layer in aux_hidden_state_layers)
                or tuple(sorted(set(aux_hidden_state_layers))) != aux_hidden_state_layers
                or len(aux_hidden_state_layers) not in (0, 3)):
            raise ValueError("V4 captures require three ordered distinct zero-based target layer IDs")
        if aux_hidden_state_layers and sequence_context is not None and not isinstance(sequence_context, _V4SequenceContext):
            raise TypeError("V4 auxiliary capture model-SP requires its normalized source sequence context")
        self.aux_layers, self.start_layer, self.end_layer = aux_hidden_state_layers, start, end
        self.hidden_size = model.hidden_size
        self.aux_hidden_states = ()
        self.aux_post = {layer: SymbolicMHC(trace, f"aux-layer-{layer}:mhc-post", pool, model, hardware,
            variant="mhc_post", source_use_deep_gemm=source_use_deep_gemm)
            for layer in aux_hidden_state_layers if start <= layer < end}
        self.aux_mean = {layer: SymbolicV4AuxHiddenMean(trace, f"aux-layer-{layer}:mean", pool)
            for layer in self.aux_post}
        self.sequence, self.pool = sequence_context, pool
        self.first_stage, self.last_stage = start == 0, end == model.num_hidden_layers
        self.embed_tokens = ((SymbolicEmbedding(trace, "embedding", pool, model.hidden_size, "bf16",
            model_input_order=model_input_order, vocab_size=model.vocab_size) if group is None else
            SymbolicVocabParallelEmbedding(trace, "embedding", pool, model.hidden_size,
            model.vocab_size, group, config, token_ids=token_ids,
            model_input_order=model_input_order if trace.rank == group.participants[0] else None)) if self.first_stage else None)
        if _source_state is not None and (type(_source_state) is not _V4TargetRankState
                or _source_state.trace is not trace or _source_state.layer_range != (start,end)):
            raise ValueError("V4 continued constructor requires its own exact source rank/layers")
        topk_buffer = (_source_state.topk if _source_state is not None else
            trace.tensor("model.topk-buffer", (max_batched_tokens, model.index_topk), "int32", persistent=True))
        # finalize_mhc_broadcast_weights retains the first layer's reduced
        # FP32 weight buffer. Ordinary parameters are accounted separately;
        # this derived model-owned buffer belongs to the source residency.
        self.hc_attn_fn_broadcast = (_source_state.broadcast if _source_state is not None else trace.tensor("model.first-layer-mhc-broadcast",
            (model.hc_mult * (model.hc_mult + 2), model.hidden_size), "fp32", persistent=True) if self.first_stage else None)
        # build_deepseek_v4_rope/get_rope shares equal constructor keys. C4
        # and C128 use the same compressed-table key; the indexer borrows it.
        rope_kinds = tuple(dict.fromkeys("swa" if ratio <= 1 else "compressed"
            for ratio in model.compress_ratios[start:end]))
        ropes = (dict(_source_state.ropes) if _source_state is not None else {kind: trace.tensor("model.rope-" + kind,
            (model.max_position_embeddings, model.qk_rope_head_dim), "fp32", persistent=True) for kind in rope_kinds})
        self.layers = []
        workspace = (_source_state.workspace if _source_state is not None else None if workspace_initial_capacity_bytes is None else
            ForwardWorkspace(trace, "worker.workspace", alignment_bytes=256,
                initial_capacity_bytes=workspace_initial_capacity_bytes, locked=False))
        self.workspace = workspace
        self.rope_tables = ropes if aux_hidden_state_layers else None
        for layer in range(start, end):
            rope = ropes["swa" if model.compress_ratios[layer] <= 1 else "compressed"]
            self.layers.append(DeepseekV4DecoderLayer(trace, model, pool, f"layer-{layer}", group, config,
                rope, topk_buffer, layer_index=layer, q_lens=q_lens, kv_lens=kv_lens,
                request_phases=request_phases, slot_mapping=slot_mapping, slot_tensor=slot_tensor, cache_capacity_tokens=cache_capacity_tokens,
                source_max_model_len=source_max_model_len, linear_source_profile=linear_source_profile,
                source_use_deep_gemm=source_use_deep_gemm, hardware=hardware,
                resolution=resolutions[layer], invocation_id=invocation_ids[layer],
                indexer_selection_source=indexer_selection_source, workspace=workspace,
                source_max_batched_tokens=max_batched_tokens, profile_indexer_max_logits_mb=profile_indexer_max_logits_mb,
                expert_context=expert_context, sequence_context=sequence_context,
                cache_tensors=None if _source_state is None else _source_state.caches[layer-start],
                swa_request_phases=swa_request_phases,dummy_run=dummy_run))
        self._source_state = (_V4TargetRankState(trace,(start,end),
            tuple(layer.attn.cache_tensors for layer in self.layers),topk_buffer,tuple(ropes.items()),
            self.hc_attn_fn_broadcast,workspace) if _retain_source_state else None)
        if not self.first_stage:
            self.layers[0].pre = SymbolicMHCPipelineInput(trace, f"layer-{start}:pre", pool, model,
                source_use_deep_gemm=source_use_deep_gemm)
        self.post = SymbolicMHC(trace, "final-mhc-post", pool, model, hardware,
            variant="mhc_post", source_use_deep_gemm=source_use_deep_gemm)
        self.head = (SymbolicMHC(trace, "hc-head", pool, model, hardware,
            variant="hc_head", source_use_deep_gemm=source_use_deep_gemm) if self.last_stage else None)
        self.norm = (SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size,
            model.target_semantics.rms_norm_eps) if self.last_stage else None)

    def __call__(self, input_ids, positions, intermediate_tensors=None, *, aux_layer_ids=()):
        if aux_layer_ids:
            if not self.first_stage or not self.last_stage or self.sequence is not None or self.aux_layers or intermediate_tensors is not None:
                raise ValueError("legacy DSpark auxiliary call requires its original PP1 non-SP target")
            hidden_states = self.embed_tokens(input_ids)
            residual, post_mix, res_mix = None, None, None
            auxiliary = []
            final_recon = None
            for index, layer in enumerate(self.layers):
                hidden_states, residual, post_mix, res_mix = layer(hidden_states, positions, input_ids,
                    post_mix, res_mix, residual)
                if index in aux_layer_ids:
                    final_recon = self.post(hidden_states, residual, post_mix, res_mix)
                    auxiliary.append(_dspark_source_op(self.post.trace, "dspark.aux-mean-"+str(index), self.post.pool_name,
                        "dspark_aux_mean", (final_recon,), rows=hidden_states.shape[0], width=self.norm.width,
                        source_rows=hidden_states.shape[0]*4, shape=(hidden_states.shape[0],self.norm.width)))
            hidden_states = (final_recon if len(self.layers)-1 in aux_layer_ids else
                self.post(hidden_states, residual, post_mix, res_mix))
            hidden_states = self.head(hidden_states)
            hidden_states = self.norm(hidden_states)
            return (hidden_states, tuple(auxiliary)) if aux_layer_ids else hidden_states
        if self.first_stage:
            if intermediate_tensors is not None:
                raise ValueError("V4 first stage cannot receive an intermediate tensor")
            hidden_states = self.embed_tokens(input_ids)
        else:
            expected = {"hidden_states", *(f"aux_hidden_states_{index}" for index, layer in enumerate(self.aux_layers)
                if layer < self.start_layer)}
            if intermediate_tensors is None or set(intermediate_tensors) != expected:
                raise ValueError("V4 later stage requires its expanded hidden_states and exact upstream capture slots")
            hidden_states = intermediate_tensors["hidden_states"]
            for key in expected - {"hidden_states"}:
                value = intermediate_tensors[key]
                self.trace._check_tensor(value)
                if value.shape != (positions.shape[0], self.hidden_size) or value.dtype != "bf16":
                    raise ValueError("V4 upstream capture slot must be BF16 [actual rows,hidden]")
        if self.sequence is not None:
            hidden_states = self.sequence.shard(self.trace, hidden_states, pool=self.pool, name="sp-embedding")
            input_ids = self.sequence.shard(self.trace, input_ids, pool=self.pool, name="sp-input-ids")
        residual, post_mix, res_mix = None, None, None
        local_aux, final_aux_recon = {}, None
        for layer_index, layer in enumerate(self.layers, start=self.start_layer):
            hidden_states, residual, post_mix, res_mix = layer(hidden_states, positions, input_ids,
                post_mix, res_mix, residual)
            if layer_index in self.aux_layers:
                final_aux_recon = self.aux_post[layer_index](hidden_states, residual, post_mix, res_mix)
                auxiliary = self.aux_mean[layer_index](final_aux_recon)
                if self.sequence is not None:
                    auxiliary = self.sequence.gather(auxiliary, pool=self.pool,
                        name=f"aux-layer-{layer_index}:sp-capture-gather")
                local_aux[f"aux_hidden_states_{self.aux_layers.index(layer_index)}"] = auxiliary
        hidden_states = (final_aux_recon if self.end_layer - 1 in self.aux_layers
            else self.post(hidden_states, residual, post_mix, res_mix))
        if not self.last_stage:
            # The source stage runs on its synchronous stream. C4's short
            # branch still executes an unused index-weight projection; such
            # calls must complete before this stage's send even without a
            # downstream tensor consumer. Join the actual local terminals.
            from .workload_graph import WorkloadEdge
            frontier = self.trace._writers[hidden_states.view.allocation_id]
            nodes = set(self.trace._nodes)
            consumed = {edge.source for edge in self.trace._edges
                if edge.source in nodes and edge.destination in nodes}
            if not self.aux_layers:
                for terminal in sorted(nodes - consumed - {frontier}):
                    self.trace._edges.append(WorkloadEdge(terminal, frontier, "barrier"))
            return {"hidden_states": hidden_states, **local_aux}
        if self.sequence is not None:
            hidden_states = self.sequence.gather(hidden_states, pool=self.pool, name="sp-final-hidden-gather")
        hidden_states = self.head(hidden_states)
        hidden_states = self.norm(hidden_states)
        self.aux_hidden_states = tuple(local_aux[f"aux_hidden_states_{index}"] if layer >= self.start_layer
            else intermediate_tensors[f"aux_hidden_states_{index}"] for index, layer in enumerate(self.aux_layers))
        return hidden_states


def trace_deepseek_v4_forward(model, step, pool, hardware, expert_routing, *, input_ids, positions,
                             slot_mapping, cache_capacity_tokens, source_max_model_len, max_batched_tokens,
                             source_use_deep_gemm, linear_source_profile, source_prompt_lengths=None,
                             indexer_selection_source=None, workspace_initial_capacity_bytes=None, sample_logits=True,
                             cast_logits=False, _replica_context=None, aux_hidden_state_layers=(),
                             _retain_source_state=False, _prior_source=None, _dspark_target_model=None,
                             _preflight_only=False, _v2_accepted_tokens=None, _v2_prompt_lengths=None,
                             _pp_feedback_inputs=None,_dummy_run=False,
                             _v2_state_indices=None,_v2_state_capacity=None,_admitted_request_ids=(),_sampling_state_ports=None):
    """Complete metadata-only V4 target forward and greedy chain, local TP/EP/PP.

    Requests and token vectors are packed by the shared source metadata program.
    Routing is an explicit normalized dynamic input; no checkpoint assignment
    or token values are computed. Native allocator/metadata services and
    workspace gaps remain separate from this declared model-forward scope.
    """
    from .runtime_inputs import vllm_0_30_source_model_inputs
    from .runtime_inputs import ExpertRoutingInput
    if (type(_dummy_run) is not bool or _dummy_run and (_replica_context is None
            or _v2_accepted_tokens is not None or sample_logits or cast_logits
            or len(step.requests)!=1 or step.requests[0].computed_tokens!=0
            or step.requests[0].scheduled_tokens not in (1,6))):
        raise ValueError('V4 dummy source requires its explicit coordinated no-sampler uniform decode profile')
    if type(_retain_source_state) is not bool or _prior_source is not None and not _retain_source_state:
        raise ValueError("V4 continuation requires explicit source-state retention")
    current_by_id={r.request_id:r for r in step.requests}
    if (type(_admitted_request_ids) is not tuple or len(set(_admitted_request_ids))!=len(_admitted_request_ids)
            or any(type(key) is not str or key not in current_by_id or current_by_id[key].computed_tokens!=0
                for key in _admitted_request_ids) or _admitted_request_ids and _dspark_target_model is None):
        raise ValueError('V4 admission requires distinct current fresh requests and explicit DSpark identity')
    if _pp_feedback_inputs is not None:
        _validate_dspark_pp_inputs(_pp_feedback_inputs,step,pool,_prior_source,_dspark_target_model)
    source_owner=0 if _replica_context is None else _replica_context.rank
    if _retain_source_state and (pool.parallel_plan.num_layers!=model.num_hidden_layers
            or pool.parallel_plan.num_experts!=model.num_experts or any(r.data_parallel_rank!=source_owner for r in step.requests)):
        raise ValueError("V4 retained source requires its full target plan and actual standalone request owner")
    if _retain_source_state and workspace_initial_capacity_bytes is None:
        raise NotImplementedError("V4 retained source requires explicit rank workspace entries")
    if type(_preflight_only) is not bool or _preflight_only and (_replica_context is None or not _retain_source_state):
        raise ValueError("V4 target preflight belongs to its coordinated retained source entry")
    if _dspark_target_model is not None:
        if (not isinstance(_dspark_target_model,ModelConfig) or not _dspark_target_model.is_dspark_draft
                or _dspark_target_model.model_id != model.model_id+'-DSpark'
                or any(getattr(_dspark_target_model,name)!=getattr(model,name) for name in
                    ('hidden_size','num_hidden_layers','num_attention_heads','num_experts','vocab_size','head_dim','hc_mult'))):
            raise ValueError("V4 speculative target source requires its registered DSpark draft identity")
        if aux_hidden_state_layers != _dspark_target_model.dspark_target_layer_ids:
            raise ValueError("V4 DSpark target source requires its exact three configured capture layers")
        if any(request.computed_tokens and (request.phase!='decode'
                or request.scheduled_tokens!=_dspark_target_model.dspark_block_size+1) for request in step.requests):
            raise NotImplementedError("V4 DSpark verification source requires each full N+1 target query")
        if sample_logits or cast_logits:
            raise NotImplementedError("V4 DSpark target logits require their separate complete V2 sampler source")
    if type(sample_logits) is not bool or type(cast_logits) is not bool or sample_logits and cast_logits:
        raise TypeError("V4 output flags must be exclusive booleans")
    v2_sampling=_v2_accepted_tokens is not None
    if not v2_sampling and (_v2_state_indices is not None or _v2_state_capacity is not None):
        raise ValueError('V4 resident request rows require the explicit V2 sampler source')
    if not v2_sampling and _v2_prompt_lengths is not None:
        raise ValueError('V4 V2 prompt declarations require their explicit accepted-prefix scenario')
    if v2_sampling:
        if _v2_state_indices is None:_v2_state_indices=tuple(range(len(step.requests)))
        if _v2_state_capacity is None:_v2_state_capacity=len(step.requests)
        sampling_ports={} if _sampling_state_ports is None else _sampling_state_ports
        if _prior_source is not None and _prior_source.retained_sampling_ports:
            if sampling_ports:raise ValueError('V4 cannot replace its live retained SamplingStates with constructor aliases')
            sampling_ports=dict(_prior_source.retained_sampling_ports)
        if type(sampling_ports) is not dict:
            raise ValueError('V4 SamplingStates aliases require explicit physical-rank ownership')
        if sampling_ports and _prior_source is None:
            raise ValueError('V4 SamplingStates aliases require their original retained trace owner')
        original_traces={} if _prior_source is None else {trace.rank:trace for stage in _prior_source.stages for trace in stage}
        for rank,ports in sampling_ports.items():
            if rank not in pool.physical_ranks_for_pipeline_stage(pool.parallel_plan.config.pipeline_parallel_size-1):
                raise ValueError('V4 SamplingStates aliases must belong to actual last-PP workers')
            values=dict(ports)
            if set(values)!={'temperature','seeds'}:raise ValueError('V4 SamplingStates aliases require both original ports')
            for role,dtype in (('temperature','fp32'),('seeds','int64')):
                port=values[role]
                if (port.trace is not original_traces[rank] or port.shape!=(_v2_state_capacity,) or port.dtype!=dtype
                        or port.trace._allocations[port.view.allocation_id].freed_event is not None):
                    raise ValueError('V4 SamplingStates aliases changed their worker, capacity or dtype')
        if (type(_v2_state_indices) is not tuple or len(_v2_state_indices)!=len(step.requests)
                or len(set(_v2_state_indices))!=len(step.requests) or type(_v2_state_capacity) is not int
                or _v2_state_capacity<len(step.requests)
                or any(type(index) is not int or not 0<=index<_v2_state_capacity for index in _v2_state_indices)):
            raise ValueError('V4 V2 sampler requires distinct request indices within its resident capacity')
        if _prior_source is not None and (
                ({} if _prior_source.dummy_run else {r.request_id:index for r,index in zip(_prior_source.step.requests,_prior_source.v2_state_indices,strict=True)
                    if r.request_id not in _prior_source.step.finished_request_ids}
                )!={r.request_id:index for r,index in zip(step.requests,_v2_state_indices,strict=True)
                    if r.request_id not in _admitted_request_ids}
                or any(dict(ports)['temperature'].shape!=(_v2_state_capacity,) for _,ports in _prior_source.v2_input_ports)):
            raise ValueError('V4 V2 continuation changed its resident request-state indices or capacity')
        if not _retain_source_state or _dspark_target_model is None:
            raise ValueError('V4 V2 sampling requires its retained DSpark target source profile')
        if (type(_v2_accepted_tokens) is not tuple or type(_v2_prompt_lengths) is not tuple
                or len(_v2_accepted_tokens)!=len(step.requests) or len(_v2_prompt_lengths)!=len(step.requests)
                or any(type(a) is not int or not 0<=a<(request.scheduled_tokens if request.computed_tokens else 1)
                    for request,a in zip(step.requests,_v2_accepted_tokens,strict=True))
                or any(type(n) is not int or not 1<=n<=source_max_model_len for n in _v2_prompt_lengths)
                or _prior_source is not None and not _prior_source.dummy_run and ('v2_sampler_prompt_lengths' not in vars(_prior_source.owner)
                    or {r.request_id:prompt for r,prompt in zip(_prior_source.step.requests,_prior_source.owner.v2_sampler_prompt_lengths,strict=True)
                        if r.request_id not in _prior_source.step.finished_request_ids}
                        !={r.request_id:prompt for r,prompt in zip(step.requests,_v2_prompt_lengths,strict=True)
                            if r.request_id not in _admitted_request_ids})):
            raise ValueError('V4 V2 source requires valid acceptance and unchanged explicit prompt lengths')
        logits_rows=sum(request.scheduled_tokens if request.computed_tokens else 1 for request in step.requests)
        if logits_rows>max(1,2**30//(model.vocab_size*4)):
            raise NotImplementedError('V4 V2 multi-chunk sampling remains separate from this short-request source profile')
    if indexer_selection_source not in (None, "declared_causal_topk"):
        raise ValueError("unsupported symbolic V4 indexer selection source")
    config = pool.parallel_plan.config
    if (type(aux_hidden_state_layers) is not tuple or len(aux_hidden_state_layers) not in (0, 3)
            or any(type(layer) is not int or not 0 <= layer < model.num_hidden_layers for layer in aux_hidden_state_layers)
            or tuple(sorted(set(aux_hidden_state_layers))) != aux_hidden_state_layers):
        raise ValueError("V4 captures require three ordered distinct zero-based target layer IDs")
    if _replica_context is not None and not isinstance(_replica_context, _V4ReplicaContext):
        raise TypeError("V4 DP source requires its normalized replica context")
    if _replica_context is not None and _replica_context.valid_rows and (
            type(_replica_context.valid_rows) is not tuple or len(_replica_context.valid_rows)!=len(_replica_context.rows)
            or any(type(valid) is not int or not 0<=valid<=physical
                for valid,physical in zip(_replica_context.valid_rows,_replica_context.rows,strict=True))):
        raise ValueError('V4 source valid rows must describe each physical DP batch')
    if workspace_initial_capacity_bytes is not None and (
            not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != config.worker_count
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("V4 source workspace history needs one explicit nonnegative entry capacity per physical rank")
    if (model.model_type != "deepseek_v4" or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
            or pool.hardware_id != "nvidia_h200" or hardware.hardware_id != pool.hardware_id
            or not is_deepseek_v4_parallel_model_scope(
                model, config.tensor_parallel_size, config.pipeline_parallel_size,
                expert_parallel=config.enable_expert_parallel or _replica_context is not None and _replica_context.expert_size > 1)
            or len(pool.ranks) != config.worker_count or not config.is_rank_local):
        raise NotImplementedError("symbolic V4 target source requires H200 block-FP8/BF16 whole-block TP/EP and nonempty PP, SP/CP1")
    if config.pipeline_parallel_size > 1 and source_use_deep_gemm is not False:
        raise NotImplementedError("V4 pipeline entry currently requires its declared optional TileLang prenorm branch")
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("symbolic V4 forward requires explicit canonical routing input")
    if any(r.mtp_proposed_tokens or r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests):
        raise NotImplementedError("symbolic V4 source currently requires MTP-off")
    source_step = step
    source_slots = slot_mapping
    rows = sum(r.scheduled_tokens for r in step.requests)
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(v) is not int for v in values):
            raise ValueError(f"symbolic V4 {name} requires one explicit integer per model row")
    if any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("symbolic V4 input IDs must belong to the accurate checkpoint vocabulary")
    if (type(source_max_model_len) is not int
            or not max(r.computed_tokens + r.scheduled_tokens for r in step.requests)
                <= source_max_model_len <= model.max_position_embeddings):
        raise ValueError("symbolic V4 source maximum length must contain every declared context")
    expected_positions = tuple(p for r in step.requests for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens))
    if positions != ((0,)*rows if _dummy_run else expected_positions) or any(value >= source_max_model_len for value in positions):
        raise ValueError("symbolic V4 positions must agree with explicit request contexts and maximum length")
    if type(max_batched_tokens) is not int or max_batched_tokens < rows:
        raise ValueError("symbolic V4 batch capacity must cover all scheduled rows")
    if type(source_use_deep_gemm) is not bool:
        raise TypeError("symbolic V4 mHC needs an explicit source DeepGEMM branch")
    if _dummy_run and (input_ids!=(0,)*rows or slot_mapping!=(-1,)*rows):
        raise ValueError('V4 dummy source must explicitly zero IDs/positions and mask every KV slot')
    if not _dummy_run and len(set(slot_mapping)) != rows:
        raise ValueError("symbolic V4 source requires distinct writable logical slots")
    spec_depth = 0 if _dspark_target_model is None else _dspark_target_model.dspark_block_size
    step, input_ids, model_input_order = vllm_0_30_source_model_inputs(step,
        embedding_token_ids=input_ids, prompt_lengths=source_prompt_lengths,
        num_speculative_tokens=spec_depth)
    if spec_depth and model_input_order['request_permutation'] != list(range(len(step.requests))):
        raise NotImplementedError("V4 DSpark target source preserves actual request order; mixed reorder remains unbound")
    positions = tuple(positions[row] for row in model_input_order["token_row_permutation"])
    slot_mapping = tuple(slot_mapping[row] for row in model_input_order["token_row_permutation"])
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    phases = tuple(model_input_order["model_attention_phases"])
    if _dummy_run:phases=('decode',)
    swa_phases=phases
    swa_threshold=1+spec_depth
    if spec_depth:
        _,_,swa_order=vllm_0_30_source_model_inputs(source_step,
            embedding_token_ids=None,prompt_lengths=source_prompt_lengths,
            num_speculative_tokens=spec_depth,decode_threshold=1+2*spec_depth)
        if swa_order['request_permutation']!=list(range(len(source_step.requests))):
            raise NotImplementedError("V4 DSpark SWA source needs its original identity request order")
        swa_phases=tuple(swa_order['model_attention_phases']);swa_threshold=1+2*spec_depth
    bounds=(cache_capacity_tokens,source_max_model_len,max_batched_tokens)
    branch=(source_use_deep_gemm,linear_source_profile,indexer_selection_source,
        *(('v2_plain_greedy',) if v2_sampling else ()),
        *(('dummy_uniform_decode',) if _dummy_run else ()),
        None if _dspark_target_model is None else _dspark_target_model.config_sha256)
    if _prior_source is not None:
        stages,groups=_v4_target_continuation(_prior_source,model,source_step,pool,bounds=bounds,
            source_branch=branch,aux_layers=aux_hidden_state_layers,workspace_entries=workspace_initial_capacity_bytes,dummy_run=_dummy_run,
            admitted_request_ids=_admitted_request_ids)
    else:
        stages = tuple(tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}")
            for rank in pool.physical_ranks_for_pipeline_stage(stage)) for stage in range(config.pipeline_parallel_size))
        if _replica_context is not None:
            stages = tuple(_replica_context.traces[stage * config.tensor_parallel_size:(stage + 1) * config.tensor_parallel_size]
                for stage in range(config.pipeline_parallel_size))
        groups = tuple(None if len(traces) == 1 else ForwardCollectiveGroup(traces,
            namespace=step.step_id if len(stages) == 1 else f"{step.step_id}:stage-{stage}")
            for stage, traces in enumerate(stages))
        if _replica_context is not None:
            groups = _replica_context.tensor_groups if config.tensor_parallel_size > 1 else (None,) * len(stages)
    histories=()
    if _retain_source_state and _dummy_run:
        if type(cache_capacity_tokens) is not int or cache_capacity_tokens<128 or cache_capacity_tokens%8:
            raise ValueError('V4 retained dummy source requires its actual aligned cache arena capacity')
        histories=((),)
    elif _retain_source_state:
        if (type(cache_capacity_tokens) is not int or cache_capacity_tokens<128 or cache_capacity_tokens%8
                or any(type(slot) is not int or not 0<=slot<cache_capacity_tokens for slot in source_slots)):
            raise ValueError("V4 retained source needs valid explicit logical cache bounds and slots")
        prior_histories={} if _prior_source is None else dict(zip(
            (r.request_id for r in _prior_source.step.requests),_prior_source.slot_histories,strict=True))
        maps=[{} if _prior_source is None or r.request_id in _admitted_request_ids else dict(prior_histories[r.request_id])
            for r in source_step.requests]
        occupied={slot:(request.request_id,pos) for request,history in zip(source_step.requests,maps,strict=True)
            for pos,slot in history.items()}
        cursor=0
        for request,history in zip(source_step.requests,maps,strict=True):
            for pos,slot in zip(range(request.computed_tokens,request.computed_tokens+request.scheduled_tokens),
                    source_slots[cursor:cursor+request.scheduled_tokens],strict=True):
                if (pos in history and history[pos]!=slot
                        or slot in occupied and occupied[slot]!=(request.request_id,pos)):
                    raise ValueError("V4 retained source cannot remap positions, alias request slots or silently evict history")
                history[pos]=slot;occupied[slot]=(request.request_id,pos)
            cursor+=request.scheduled_tokens
        histories=tuple(tuple(sorted(history.items())) for history in maps)
    if (any(ratio==4 for ratio in model.compress_ratios[:model.num_hidden_layers])
            and max(kv_lens)//4>model.index_topk and indexer_selection_source!='declared_causal_topk'):
        raise NotImplementedError("V4 continued long C4 source requires its explicit selected-cardinality declaration")
    mapping = pool.parallel_plan.layer_to_pipeline_stage
    ranges = tuple((mapping.index(stage), mapping.index(stage) + mapping.count(stage))
        for stage in range(config.pipeline_parallel_size))
    if tuple(stage for stage, (first, end) in enumerate(ranges) for _ in range(first, end)) != mapping:
        raise ValueError("V4 source pipeline requires contiguous stage layer ranges")
    collective = pool.collective_for("all_reduce") if config.tensor_parallel_size > 1 else None
    gather = pool.collective_for("all_gather") if config.tensor_parallel_size > 1 else None
    if _retain_source_state:
        if any(item is not None and item.link=='roce' for item in (collective,gather)):
            raise NotImplementedError("V4 retained source TP groups require an explicit NIC mapping for RoCE")
        if _replica_context is not None:
            pool.collective_for('all_gatherv')
            pool.collective_for('reduce_scatterv')
        for stage in range(config.pipeline_parallel_size-1):
            pool.pipeline_p2p_for(stage,stage+1)
    invocations = (tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
        if _replica_context is None else _replica_context.invocations)
    resolutions = (_source_expert_resolutions(model, source_step, pool, expert_routing, invocations, rows)
        if _replica_context is None else _replica_context.resolutions)
    if _preflight_only:
        return None
    if _prior_source is not None:
        if 'v2_sampler_outputs_by_rank' in vars(_prior_source.owner):
            del _prior_source.owner.v2_sampler_outputs_by_rank
        if 'aux_hidden_states_by_rank' in vars(_prior_source.owner):
            del _prior_source.owner.aux_hidden_states_by_rank
        if 'dspark_cache_state' in vars(_prior_source.owner):
            # A live cache revision remains; the prior large debug outputs do
            # not form state. Keep an empty retirement marker for next draft.
            _prior_source.owner.dspark_outputs=()
    selected_rows = (tuple(row for index,request in enumerate(step.requests)
        for row in (range(sum(q_lens[:index]),sum(q_lens[:index+1])) if request.computed_tokens
            else (sum(q_lens[:index+1])-1,))) if spec_depth else
        tuple(sum(q_lens[:i + 1]) - 1 for i in range(len(q_lens))))
    previous_nodes={trace.rank:set(trace._nodes) for stage in stages for trace in stage}
    invocation_frontier=()
    if _prior_source is not None:
        # Raw local IDs keep DP collectives in the joined owner's namespace.
        # A local TP/PP graph cannot join another replica's group by itself.
        old_ids={node for stage in stages for trace in stage for node in trace._nodes}
        invocation_frontier=tuple(sorted(old_ids-{edge.source for stage in stages for trace in stage
            for edge in trace._edges if edge.source in old_ids and edge.destination in old_ids}))
    rank_states=[]
    intermediate_buffers={} if _prior_source is None else _prior_source.intermediate_buffers
    auxiliary_by_rank = {}
    v2_outputs={};v2_batches={};dummy_batches={}
    auxiliary_workspaces, auxiliary_ropes = {}, {}
    def rank_forward(trace, stage, incoming):
        group = groups[stage]
        ordinal = pool.ranks.index(trace.rank)
        lane = stages[stage].index(trace)
        model_sp = _replica_context is not None and _replica_context.sequence_parallel
        sequence = (_V4SequenceContext(group, gather, pool.collective_for("reduce_scatterv"), rows, lane)
            if model_sp else None)
        expert_context = (_V4ExpertContext(config.tensor_parallel_size, expert_size=config.expert_parallel_size,
            expert_rank=lane if config.enable_expert_parallel else 0) if _replica_context is None else
            _V4ExpertContext(config.tensor_parallel_size, len(_replica_context.rows), _replica_context.expert_size,
                _replica_context.rank * config.tensor_parallel_size + lane if _replica_context.expert_size > 1 else 0,
                _replica_context.data_groups[stage][lane], _replica_context.rank, _replica_context.rows,
                pool.collective_for("all_gatherv"), pool.collective_for("reduce_scatterv")))
        valid_rows=(_replica_context.rows if _replica_context is not None and not _replica_context.valid_rows else
            _replica_context.valid_rows if _replica_context is not None else (rows,))
        if _replica_context is not None and _replica_context.valid_rows:
            expert_context=replace(expert_context,valid_tokens=sum(valid_rows),local_valid_rows=0 if _dummy_run else None)
        if model_sp:
            expert_context = replace(expert_context, data_group=_replica_context.expert_groups[stage],
                rows=tuple(-(-count // config.tensor_parallel_size) for count in _replica_context.rows
                    for _ in range(config.tensor_parallel_size)), sequence_parallel=True,
                valid_tokens=sum(valid_rows), local_valid_rows=max(0, min(sequence.shard_rows,
                    valid_rows[_replica_context.rank] - lane * sequence.shard_rows)))
        workspace_capacity = None if workspace_initial_capacity_bytes is None else workspace_initial_capacity_bytes[ordinal]
        old_terminals=previous_nodes[trace.rank]-{edge.source for edge in trace._edges
            if edge.source in previous_nodes[trace.rank] and edge.destination in previous_nodes[trace.rank]}
        slots = trace.tensor(trace.unique_name("slot_mapping"), (rows,), "int64")
        network = DeepseekV4Model(trace, model, pool.name, group, collective,
            token_ids=input_ids, q_lens=q_lens, kv_lens=kv_lens, request_phases=phases,
            slot_mapping=slot_mapping, slot_tensor=slots, cache_capacity_tokens=cache_capacity_tokens,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            linear_source_profile=linear_source_profile, source_use_deep_gemm=source_use_deep_gemm,
            hardware=hardware, resolutions=resolutions, invocation_ids=invocations, model_input_order=model_input_order,
            indexer_selection_source=indexer_selection_source, workspace_initial_capacity_bytes=workspace_capacity,
            layer_range=ranges[stage], expert_context=expert_context, sequence_context=sequence,
            aux_hidden_state_layers=aux_hidden_state_layers,
            _source_state=None if _prior_source is None else _prior_source.rank_states[ordinal],
            _retain_source_state=_retain_source_state,swa_request_phases=swa_phases if spec_depth else None,dummy_run=_dummy_run)
        if _retain_source_state:
            rank_states.append(network._source_state)
        ids = trace.tensor(trace.unique_name("input_ids"), (rows,), "int32")
        pos = trace.tensor(trace.unique_name("positions"), (rows,), "int64")
        if _dummy_run:
            SymbolicFillZero(trace,'dummy:zero-IDs',pool.name)(ids)
            SymbolicFillZero(trace,'dummy:zero-positions',pool.name)(pos)
            from .traced_operators import SymbolicOperator
            op=SymbolicOperator(trace,'dummy:PAD-slots',pool.name)
            node=op.node(WorkloadKind.ELEMENTWISE,{'elements':rows},'int64','elementwise',
                {'operation':'fill','reads':0,'writes':1,'source_constant':-1,'source_dummy_run':True})
            trace.call(node,(),shape=slots.shape,dtype='int64',out=slots)
            if _dspark_target_model is not None:
                starts=trace.tensor(trace.unique_name('dummy-query-start'),(2,),'int32')
                mapping=trace.tensor(trace.unique_name('dummy-idx-mapping'),(1,),'int64')
                node=op.node(WorkloadKind.VECTOR,{'elements':3,'rows':1},'int32','vllm_dummy_input_batch',{
                    'source_component':'vllm_0_30_InputBatch_make_dummy','source_dummy_run':True,
                    'query_start_values':(0,rows),'idx_mapping_values':(0,),'position_values':(0,)*rows,
                    'is_padding_values':(True,)*rows,'internal_launch_bound':False})
                trace.call_effect(node,(),writes=(starts,mapping))
                dummy_batches[trace.rank]=(('positions',pos),('query_start_loc',starts),('idx_mapping',mapping))
        if _pp_feedback_inputs is not None and trace.rank in _pp_feedback_inputs:
            from .traced_operators import SymbolicOperator
            feedback=_pp_feedback_inputs[trace.rank]
            op=SymbolicOperator(trace,step.step_id+':pp-feedback-prepare',pool.name)
            node=op.node(WorkloadKind.VECTOR,{'elements':rows,'rows':len(step.requests)},'int32','pp_feedback_prepare_inputs',{
                'source_component':'vllm_0_30_PPHandler_feedback','source_operation':'serialized_update_requests_and_prepare_inputs',
                'main_stream_waits_receive_event':True,'source_projection_only':True,'values_computed':False,
                'pipeline_overlap_bound':False,'request_ids':tuple(request.request_id for request in step.requests),
                'feedback_request_ids':tuple(request.request_id for request in _prior_source.step.requests),
                'feedback_request_state_indices':_prior_source.v2_state_indices,
                'current_request_state_indices':() if not v2_sampling else _v2_state_indices,
                'completed_previous_request_ids':_prior_source.step.finished_request_ids,
                'admitted_request_ids':_admitted_request_ids,
                'feedback_order_scope':'update prior PP decode states; finish old requests; add/stage new requests; prepare current order'})
            trace.call(node,(feedback['sampled'],feedback['counts'],feedback['drafts']),shape=ids.shape,dtype='int32',out=ids)
        old=set(trace._nodes)
        hidden_states = network(ids, pos, intermediate_tensors=incoming)
        roots=set(trace._nodes)-old-{edge.destination for edge in trace._edges if edge.source not in old}
        if _prior_source is not None:
            for previous in (invocation_frontier if stage==0 else old_terminals):
                for root in roots:
                    if previous!=root:
                        trace._edges.append(WorkloadEdge(previous,root,'barrier'))
        if not network.last_stage:
            return hidden_states
        if aux_hidden_state_layers:
            auxiliary_by_rank[trace.rank] = network.aux_hidden_states
            auxiliary_workspaces[trace.rank] = network.workspace
            auxiliary_ropes[trace.rank] = network.rope_tables
        if _dummy_run:
            SymbolicGather(trace,'dummy:selected-hidden',pool.name)(hidden_states,1,indices=(rows-1,))
            return None
        indices = trace.tensor(trace.unique_name("logits_indices"), (len(selected_rows),), "int64" if spec_depth else "int32")
        if spec_depth:
            from .traced_operators import SymbolicOperator
            op=SymbolicOperator(trace,'logits-row-select',pool.name)
            node=op.node(WorkloadKind.GATHER,{'elements':len(selected_rows)*model.hidden_size,
                'rows':len(selected_rows),'hidden':model.hidden_size},'bf16','gather',
                {'gather_factor':1,'index_count':len(selected_rows),'index_bytes':8,'scattered':True,
                 'materialized_output':True,'selected_indices':selected_rows,
                 'source_component':'vllm_0_30_V2_INT64_logits_indices'})
            selected=trace.call(node,(hidden_states,indices),shape=(len(selected_rows),model.hidden_size),dtype='bf16')
        else:
            selected = SymbolicLogitsRowSelect(trace, "logits-row-select", pool.name)(hidden_states,
                indices, selected_indices=selected_rows)
        if group is None:
            logits = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)[0]
        else:
            logits = SymbolicTPLogitsProcessor(trace, "lm-head", pool.name, hidden_size=model.hidden_size,
                vocab_size=model.vocab_size, group=group, config=gather)(selected)
        if sample_logits or cast_logits:
            sampled = SymbolicGreedy(trace, "target-sampler", pool.name)(logits, sample_logits=sample_logits)
        if v2_sampling:
            from .traced_sampling import vllm030_v2_greedy_rejection_forward, vllm030_v2_plain_greedy_forward
            batch=len(step.requests)
            sampler_queries=tuple(request.scheduled_tokens if request.computed_tokens else 1 for request in step.requests)
            sampler=(vllm030_v2_plain_greedy_forward if sum(q-1 for q in sampler_queries)==0 else
                vllm030_v2_greedy_rejection_forward)
            def metadata(role,shape,dtype):
                if role in ('temperature','seeds') and trace.rank in sampling_ports:
                    return dict(sampling_ports[trace.rank])[role]
                return trace.tensor(trace.unique_name('V2-input.'+role),shape,dtype,persistent=role in ('temperature','seeds'))
            sampler_ports={role:metadata(role,shape,dtype) for role,shape,dtype in (
                ('cu_num_logits',(batch+1,),'int32'),('idx_mapping',(batch,),'int64'),
                ('expanded_idx_mapping',(len(selected_rows),),'int64'),('expanded_local_pos',(len(selected_rows),),'int32'),
                ('temperature',(_v2_state_capacity,),'fp32'),('seeds',(_v2_state_capacity,),'int64'),
                ('seq_lens',(batch,),'int32'),('prefill_len',(_v2_state_capacity,),'int32'))}
            v2_outputs[trace.rank]=sampler(trace,logits,input_ids=ids,positions=pos,
                logits_indices=indices,**sampler_ports,
                logits_per_request=sampler_queries,
                selected_indices=selected_rows,request_state_indices=_v2_state_indices,accepted_tokens=_v2_accepted_tokens,
                sequence_lengths=tuple(request.computed_tokens+request.scheduled_tokens for request in step.requests),
                prompt_lengths=_v2_prompt_lengths,num_speculative_tokens=spec_depth,
                source_profile='vllm_0_30_V2_plain_allgreedy_no_sampling_params',pool_name=pool.name,name='target-V2-sampler')
            v2_batches[trace.rank]=(('positions',pos),('query_start_loc',metadata('target-query-start',(batch+1,),'int32')),
                ('idx_mapping',sampler_ports['idx_mapping']),('temperature',sampler_ports['temperature']),('seeds',sampler_ports['seeds']),
                ('input_ids',ids))
        return None

    def retain(result):
        if _dummy_run and _dspark_target_model is not None:result.dummy_target_input_ports_by_rank=dummy_batches
        if v2_sampling:
            result.v2_sampler_outputs_by_rank=v2_outputs
            result.v2_sampler_prompt_lengths=_v2_prompt_lengths
        if _retain_source_state:
            if '_dspark_capture_consumed' in vars(result):
                del result._dspark_capture_consumed
            result._v4_capture_producers={rank:tuple(next(trace for stage in stages for trace in stage if trace.rank==rank)._writers[port.view.allocation_id]
                for port in ports) for rank,ports in auxiliary_by_rank.items()}
            result._v4_target_source_state=_V4TargetSourceState(result,model.config_sha256,config,pool.ranks,
                pool.name,source_step,(source_step.step_id,) if _prior_source is None else (*_prior_source.step_ids,source_step.step_id),
                bounds,branch,aux_hidden_state_layers,stages,groups,tuple(rank_states),intermediate_buffers,histories,
                tuple((rank,tuple((layer,writer,port.view.view_id) for layer,writer,port in zip(aux_hidden_state_layers,
                    result._v4_capture_producers[rank],ports,strict=True))) for rank,ports in auxiliary_by_rank.items()),
                tuple((rank,output['declared_num_sampled'],output['declared_num_rejected']) for rank,output in v2_outputs.items()),
                tuple(v2_batches.items()),_dummy_run,() if not v2_sampling else _v2_state_indices,
                tuple((rank,tuple((role,port) for role,port in ports if role in ('temperature','seeds')))
                    for rank,ports in v2_batches.items()) if v2_sampling else
                    () if _prior_source is None else _prior_source.retained_sampling_ports,_admitted_request_ids)
            if _prior_source is not None and _prior_source.owner is not result:
                del _prior_source.owner._v4_target_source_state
        result.target_source_step_scope={'retained_source_state':_retain_source_state,
            'source_dummy_run':_dummy_run,'real_request_outputs':0 if _dummy_run else len(step.requests),
            'source_continuation':_prior_source is not None,'target_selected_logit_rows':selected_rows,
            'source_attention_decode_threshold':swa_threshold,
            'source_indexer_decode_threshold':model_input_order['decode_threshold'],
            'source_swa_attention_phases':swa_phases,'source_indexer_attention_phases':phases,
            'target_rejection_sampler_bound':v2_sampling if spec_depth else sample_logits,
            'V2_request_state_indices':() if not v2_sampling else _v2_state_indices,
            'target_rejection_scope':'symbolic fixed launches +caller-declared acceptance; no numerical values',
            'V2_input_metadata_scope':'declared per-step typed ports; native persistent inputBatch storage and preparation unbound',
            'native_Scheduler_bound':False,'complete_priced_request':False,
            'persistent_slot_scope':'declared logical/no-eviction; native pages unbound',
            'invocation_order_scope':'sequential source calls; sampler/PP sample transport cost remains unbound'}
        return result

    if len(stages) > 1:
        pipeline = ForwardPipelineGroup(tuple(trace for traces in stages for trace in traces),
            namespace=step.step_id if _replica_context is None else f"{step.step_id}:dp{_replica_context.rank}:pipeline",
            collective_groups=tuple(group for group in groups if group is not None),
            tensor_roles=(("hidden_states", *(f"aux_hidden_states_{index}" for index in range(3)))
                if aux_hidden_state_layers else ("hidden_states",)),
            stage_tensor_roles=(tuple(("hidden_states",
                *(f"aux_hidden_states_{index}" for index, layer in enumerate(aux_hidden_state_layers) if first <= layer < end),
                *(f"aux_hidden_states_{index}" for index, layer in enumerate(aux_hidden_state_layers) if layer < first))
                for first, end in ranges) if aux_hidden_state_layers else None))
        pipeline.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
            "pipeline_layer_partition": tuple(end - first for first, end in ranges),
            "normal_capacity_accepted": False, "actual_provider_bound": False}
        _run_vllm030_pipeline_stages(pipeline, pool, rows=rows, hidden_size=model.hidden_size,
            hidden_mult=model.hc_mult, max_batched_tokens=max_batched_tokens, run_stage=rank_forward,
            intermediate_buffers=intermediate_buffers if _retain_source_state else None,
            dummy_run=_dummy_run, initialize_intermediate_zero=bool(aux_hidden_state_layers))
        for trace in pipeline.traces:
            trace._nodes = {name: replace(node, attributes={**node.attributes,
                "model_source_profile": "vllm_0_30_v4_sync_fp8_ds_mla",
                **({'source_dummy_run':True,'source_actual_request_count':0}
                    if _dummy_run and node.kind not in (WorkloadKind.P2P,WorkloadKind.COLLECTIVE) else {}),
                **({"source_v4_aux_capture": True} if aux_hidden_state_layers else {})}) if name not in previous_nodes[trace.rank] else node for name, node in trace._nodes.items()}
        if aux_hidden_state_layers:
            pipeline.aux_hidden_states_by_rank = auxiliary_by_rank
            pipeline.aux_hidden_state_layers = aux_hidden_state_layers
            pipeline.aux_model_input_order = model_input_order
            pipeline.aux_target_model = model
            pipeline.aux_target_step = source_step
            pipeline.aux_target_config = config
            pipeline.aux_target_pool_ranks = pool.ranks
            pipeline.aux_workspaces_by_rank = auxiliary_workspaces
            pipeline.aux_rope_tables_by_rank = auxiliary_ropes
        return retain(pipeline)
    for trace in stages[0]:
        rank_forward(trace, 0, None)
        trace._nodes = {name: replace(node, attributes={**node.attributes,
            "model_execution_version": "vllm_0_30_0", "source_projection_only": True,
            "model_source_profile": "vllm_0_30_v4_sync_fp8_ds_mla",
            **({'source_dummy_run':True,'source_actual_request_count':0}
                if _dummy_run and node.kind not in (WorkloadKind.P2P,WorkloadKind.COLLECTIVE) else {}),
            **({"source_v4_aux_capture": True} if aux_hidden_state_layers else {})}) if name not in previous_nodes[trace.rank] else node for name, node in trace._nodes.items()}
    result = stages[0][0] if groups[0] is None else groups[0]
    if aux_hidden_state_layers:
        result.aux_hidden_states_by_rank = auxiliary_by_rank
        result.aux_hidden_state_layers = aux_hidden_state_layers
        result.aux_model_input_order = model_input_order
        result.aux_target_model = model
        result.aux_target_step = source_step
        result.aux_target_config = config
        result.aux_target_pool_ranks = pool.ranks
        result.aux_workspaces_by_rank = auxiliary_workspaces
        result.aux_rope_tables_by_rank = auxiliary_ropes
    return retain(result)


def trace_deepseek_v4_data_parallel_forward(model, step, pool, hardware, expert_routing, *, input_ids,
        slot_mapping_by_replica, cache_capacity_tokens_by_replica, source_max_model_len, max_batched_tokens,
        source_use_deep_gemm, linear_source_profile, source_prompt_lengths_by_replica=None,
        indexer_selection_source=None, sample_logits=True, cast_logits=False, workspace_initial_capacity_bytes=None,
        aux_hidden_state_layers=(), _retain_source_state=False, _prior_source=None, _dspark_target_model=None,
        _v2_accepted_tokens_by_replica=None, _v2_prompt_lengths_by_replica=None,_pp_feedback_inputs=None,
        _v2_state_indices_by_replica=None,_v2_state_capacities=None,_admitted_request_ids_by_replica=None,
        _sampling_state_ports_by_rank=None):
    """Coordinated V4 modular DP requests over the complete TP/PP forward.

    Each replica owns its original requests and declared cache coordinates.
    Prepare gathers hidden/weights/IDs after local routing; finalize sums DP
    expert partials before shared-add and the native final TP all-reduce.
    This source API does not extend the shared normal-runtime input schema.
    """
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_plan
    from .model_config import is_deepseek_v4_data_parallel_model_scope
    from .runtime_inputs import ExpertRoutingInput
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("V4 DP source requires canonical explicit routing input")
    config = pool.parallel_plan.config
    tp, dp, pp = config.tensor_parallel_size, config.data_parallel_size, config.pipeline_parallel_size
    if _admitted_request_ids_by_replica is None:_admitted_request_ids_by_replica=((),)*dp
    if type(_admitted_request_ids_by_replica) is not tuple or len(_admitted_request_ids_by_replica)!=dp:
        raise ValueError('V4 DP admission requires every replica in actual worker order')
    if (_v2_state_indices_by_replica is None)!=(_v2_state_capacities is None):
        raise ValueError('V4 DP resident request rows require both indices and capacities')
    if _v2_state_indices_by_replica is not None and (
            type(_v2_state_indices_by_replica) is not tuple or len(_v2_state_indices_by_replica)!=dp
            or type(_v2_state_capacities) is not tuple or len(_v2_state_capacities)!=dp):
        raise ValueError('V4 DP resident request rows require every replica in physical order')
    if _v2_accepted_tokens_by_replica is None:
        if _v2_prompt_lengths_by_replica is not None:
            raise ValueError('V4 DP V2 prompt declarations require their explicit acceptance scenario')
        v2_acceptance=v2_prompts=(None,)*dp
    else:
        if (not _retain_source_state or type(_v2_accepted_tokens_by_replica) is not tuple
                or len(_v2_accepted_tokens_by_replica)!=dp or type(_v2_prompt_lengths_by_replica) is not tuple
                or len(_v2_prompt_lengths_by_replica)!=dp
                or any(values is not None and type(values) is not tuple for values in
                    (*_v2_accepted_tokens_by_replica,*_v2_prompt_lengths_by_replica))):
            raise ValueError('V4 DP V2 sampler requires complete per-replica declared scenarios')
        v2_acceptance,v2_prompts=_v2_accepted_tokens_by_replica,_v2_prompt_lengths_by_replica
    if type(_retain_source_state) is not bool or _prior_source is not None and not _retain_source_state:
        raise ValueError("V4 DP continuation requires explicit source state retention")
    if _retain_source_state and (_dspark_target_model is None or workspace_initial_capacity_bytes is None):
        raise ValueError("V4 DP retained speculative source requires its draft identity and all rank workspace entries")
    if _prior_source is not None and (type(_prior_source) is not _V4DistributedTargetState
            or '_v4_distributed_target_state' not in vars(_prior_source.owner)
            or _prior_source.owner._v4_distributed_target_state is not _prior_source
            or _prior_source.config!=config or _prior_source.ranks!=pool.ranks or len(_prior_source.states)!=dp):
        raise ValueError("V4 DP continuation needs its current original physical topology and source revision")
    if workspace_initial_capacity_bytes is not None and (
            not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != config.worker_count
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("V4 DP workspace history needs one explicit nonnegative entry capacity per physical rank")
    if (pool.hardware_id != "nvidia_h200" or hardware.hardware_id != pool.hardware_id
            or not is_deepseek_v4_data_parallel_model_scope(model, config) or len(pool.ranks) != config.worker_count):
        raise NotImplementedError("V4 DP source requires whole-block TP/DP and uniform global EP, nonempty PP, CP1")
    model_sp = config.enable_expert_parallel and tp > 1 and pp == 1
    if (not isinstance(input_ids, tuple) or len(input_ids) != step.token_count or not step.requests
            or step.pool_name != pool.name or len({r.request_id for r in step.requests}) != len(step.requests)
            or not isinstance(slot_mapping_by_replica, tuple) or len(slot_mapping_by_replica) != dp
            or not isinstance(cache_capacity_tokens_by_replica, tuple) or len(cache_capacity_tokens_by_replica) != dp):
        raise ValueError("V4 DP source requires distinct original requests, IDs and per-replica cache coordinates")
    if source_prompt_lengths_by_replica is None:
        source_prompt_lengths_by_replica = (None,) * dp
    if not isinstance(source_prompt_lengths_by_replica, tuple) or len(source_prompt_lengths_by_replica) != dp:
        raise ValueError("V4 DP prompt lengths must have one declaration per replica")
    requests, ids = [[] for _ in range(dp)], [[] for _ in range(dp)]
    offset = 0
    for request in step.requests:
        owner = request.data_parallel_rank
        if not 0 <= owner < dp:
            raise ValueError("V4 DP request owner is outside its physical deployment")
        requests[owner].append(request)
        ids[owner].extend(input_ids[offset:offset + request.scheduled_tokens])
        offset += request.scheduled_tokens
    rows = tuple(sum(r.scheduled_tokens for r in replica) for replica in requests)
    dummy_rows=6 if _dspark_target_model is not None else 1
    work_rows=tuple(count if count else dummy_rows for count in rows)
    if (aux_hidden_state_layers or _retain_source_state) and _dspark_target_model is None and any(count == 0 for count in rows):
        raise NotImplementedError("V4 distributed capture source requires active target replicas; idle/dummy capture remains unbound")
    for replica, count in enumerate(rows):
        slots, capacity = slot_mapping_by_replica[replica], cache_capacity_tokens_by_replica[replica]
        if count:
            if not isinstance(slots, tuple) or len(slots) != count or type(capacity) is not int or capacity < 128:
                raise ValueError("V4 active DP replica requires explicit cache capacity and one slot per actual row")
        elif (slots is not None or capacity is not None and (not _retain_source_state
                or type(capacity) is not int or capacity<128 or capacity%8)
                or source_prompt_lengths_by_replica[replica] is not None):
            raise ValueError("idle V4 DP replica requires no request slots and an explicit native capacity when retained")
        if _v2_accepted_tokens_by_replica is not None and (
                count==0 and (v2_acceptance[replica] is not None or v2_prompts[replica] is not None)
                or count>0 and (v2_acceptance[replica] is None or v2_prompts[replica] is None)):
            raise ValueError('V4 target V2 feedback must cover exactly the actual active replicas')
    traces = (tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
        if _prior_source is None else _prior_source.owner.traces)
    tensor = tuple(tuple(ForwardCollectiveGroup(traces[(replica * pp + stage) * tp:(replica * pp + stage + 1) * tp],
        namespace=f"{step.step_id}:dp{replica}:stage{stage}:tp") for stage in range(pp)) for replica in range(dp)) if tp > 1 else ()
    data = tuple(tuple(ForwardCollectiveGroup(tuple(traces[(replica * pp + stage) * tp + lane] for replica in range(dp)),
        namespace=f"{step.step_id}:stage{stage}:lane{lane}:dp") for lane in range(tp)) for stage in range(pp))
    expert_groups = (tuple(ForwardCollectiveGroup(tuple(traces[(replica * pp + stage) * tp + lane]
        for replica in range(dp) for lane in range(tp)), namespace=f"{step.step_id}:stage{stage}:ep")
        for stage in range(pp)) if model_sp else ())
    joined = ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:joined", subgroups=(
        *(group for replica in tensor for group in replica), *(group for stage in data for group in stage), *expert_groups))
    if _prior_source is not None:
        # Reuse group sequence/submission owners, including the draft-only EP
        # group registered at the final PP stage in the previous round.
        joined=_prior_source.owner
        tensor,data,expert_groups=_prior_source.tensor_groups,_prior_source.data_groups,_prior_source.expert_groups
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    physical_rows = sum(-(-count // tp) * tp for count in work_rows) if model_sp else sum(work_rows)
    if model_sp or any(count==0 for count in rows):
        from src.serving.schedule.schedule_step import schedule_step_sha256
        digest = schedule_step_sha256(step)
        resolutions = tuple(expert_routing.resolve(step_id=step.step_id, layer_index=layer,
            moe_invocation_id=invocations[layer], schedule_step_sha256=digest,
            tokens=sum(rows) if layer < model.num_hash_layers else physical_rows,
            num_experts=model.num_experts, topk=model.num_experts_per_tok, model_config_sha256=model.config_sha256,
            pool_name=pool.name, tensor_parallel_size=tp, expert_parallel_size=config.expert_parallel_size,
            expert_rank_group=(pool.physical_ranks_for_pipeline_stage(pool.parallel_plan.layer_to_pipeline_stage[layer])
                if config.enable_expert_parallel else (pool.physical_ranks_for_pipeline_stage(pool.parallel_plan.layer_to_pipeline_stage[layer])[0],)),
            expert_placement_strategy="linear", moe_layer_ordinal=model.moe_layer_ordinal(layer))
            for layer in range(model.num_hidden_layers))
    else:
        resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, sum(rows))
    ranges = pipeline_layer_ranges_for_plan(pool.parallel_plan, num_layers=model.num_hidden_layers)
    local_config = replace(config, data_parallel_size=1, enable_expert_parallel=False)
    plan = build_parallel_plan(local_config, num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=tuple(end - first for first, end in ranges))
    if _prior_source is not None and 'dspark_distributed_cache_state' in vars(joined):
        cache=joined.dspark_distributed_cache_state
        if (type(cache) is not _DSparkDistributedCacheState or cache.owner is not joined or len(cache.states)!=dp
                or any(not state.dummy_run and {r.request_id:(r.data_parallel_rank,r.computed_tokens) for r in state.step.requests
                    if r.request_id not in state.step.finished_request_ids}
                    !={r.request_id:(r.data_parallel_rank,r.computed_tokens) for r in requests[replica]
                        if r.request_id not in _admitted_request_ids_by_replica[replica]}
                    for replica,state in enumerate(cache.states))):
            raise ValueError("V4 DP target must start at each current draft accepted context")
    prepared=[]
    from src.serving.schedule.schedule_step import ScheduledRequest
    model_requests=tuple(tuple(values) if values else (ScheduledRequest('_dummy_req_0','decode',0,dummy_rows,data_parallel_rank=owner),)
        for owner,values in enumerate(requests))
    if _retain_source_state:
        for replica in range(dp):
            local_traces=traces[replica*pp*tp:(replica+1)*pp*tp]
            context=_V4ReplicaContext(local_traces,tensor[replica] if tp>1 else (),data,work_rows,replica,
                resolutions,invocations,config.expert_parallel_size,model_sp,expert_groups,rows)
            local_pool=replace(pool,ranks=tuple(trace.rank for trace in local_traces),parallel_plan=plan)
            local_step=replace(step,requests=model_requests[replica],finished_request_ids=tuple(
                key for key in step.finished_request_ids if key in {r.request_id for r in requests[replica]}),preempted_request_ids=())
            arguments=dict(input_ids=tuple(ids[replica]) if rows[replica] else (0,)*dummy_rows,
                positions=tuple(pos for request in requests[replica]
                for pos in range(request.computed_tokens,request.computed_tokens+request.scheduled_tokens)),
                slot_mapping=slot_mapping_by_replica[replica] if rows[replica] else (-1,)*dummy_rows,
                cache_capacity_tokens=cache_capacity_tokens_by_replica[replica],
                source_max_model_len=source_max_model_len,max_batched_tokens=max_batched_tokens,
                source_use_deep_gemm=source_use_deep_gemm,linear_source_profile=linear_source_profile,
                source_prompt_lengths=source_prompt_lengths_by_replica[replica],indexer_selection_source=indexer_selection_source,
                sample_logits=sample_logits,cast_logits=cast_logits,_replica_context=context,
                aux_hidden_state_layers=aux_hidden_state_layers,workspace_initial_capacity_bytes=
                    workspace_initial_capacity_bytes[replica*pp*tp:(replica+1)*pp*tp],
                _retain_source_state=True,_prior_source=None if _prior_source is None else _prior_source.states[replica],
                _dspark_target_model=_dspark_target_model,_v2_accepted_tokens=v2_acceptance[replica],
                _v2_prompt_lengths=v2_prompts[replica],
                _admitted_request_ids=_admitted_request_ids_by_replica[replica],
                _sampling_state_ports=None if _sampling_state_ports_by_rank is None else
                    {rank:ports for rank,ports in _sampling_state_ports_by_rank.items() if rank in local_pool.ranks},
                _v2_state_indices=None if _v2_state_indices_by_replica is None or not rows[replica] else _v2_state_indices_by_replica[replica],
                _v2_state_capacity=None if _v2_state_capacities is None or not rows[replica] else _v2_state_capacities[replica],
                _dummy_run=not rows[replica],
                _pp_feedback_inputs=None if _pp_feedback_inputs is None or not rows[replica]
                    or _prior_source is not None and _prior_source.states[replica].dummy_run else
                    {rank:ports for rank,ports in _pp_feedback_inputs.items() if rank in local_pool.ranks})
            if not rows[replica]:arguments['positions']=(0,)*dummy_rows
            trace_deepseek_v4_forward(model,local_step,local_pool,hardware,expert_routing,**arguments,_preflight_only=True)
            prepared.append((local_pool,local_step,arguments))
        # Every replica, including the last, has passed before any retirement.
        if _prior_source is not None:
            if 'aux_hidden_states_by_rank' in vars(joined):del joined.aux_hidden_states_by_rank
            if 'dspark_outputs_by_replica' in vars(joined):del joined.dspark_outputs_by_replica
            if 'v2_sampler_outputs_by_rank' in vars(joined):del joined.v2_sampler_outputs_by_rank
            if 'dspark_distributed_cache_state' in vars(joined):
                for cache in joined.dspark_distributed_cache_state.states:cache.owner.dspark_outputs=()
    capture_ports, capture_workspaces, capture_ropes, capture_orders = {}, {}, {}, []
    retained=[]
    v2_outputs={};dummy_inputs={}
    for replica, count in enumerate(rows):
        local_traces = traces[replica * pp * tp:(replica + 1) * pp * tp]
        groups = tensor[replica] if tp > 1 else ()
        context = _V4ReplicaContext(local_traces, groups, data, work_rows, replica, resolutions, invocations,
            config.expert_parallel_size, model_sp, expert_groups,rows)
        local_pool = replace(pool, ranks=tuple(trace.rank for trace in local_traces), parallel_plan=plan)
        if not count:
            if _retain_source_state:
                local_pool,local_step,arguments=prepared[replica]
                local_source=trace_deepseek_v4_forward(model,local_step,local_pool,hardware,expert_routing,**arguments)
                retained.append(local_source._v4_target_source_state)
            else:
                dummy_step=replace(step,requests=model_requests[replica],finished_request_ids=(),preempted_request_ids=())
                local_source=trace_deepseek_v4_forward(model,dummy_step,local_pool,hardware,expert_routing,input_ids=(0,)*dummy_rows,
                    positions=(0,)*dummy_rows,slot_mapping=(-1,)*dummy_rows,cache_capacity_tokens=128,
                    source_max_model_len=source_max_model_len,max_batched_tokens=max_batched_tokens,
                    source_use_deep_gemm=source_use_deep_gemm,linear_source_profile=linear_source_profile,
                    indexer_selection_source=indexer_selection_source,sample_logits=False,cast_logits=False,
                    _replica_context=context,_dummy_run=True,aux_hidden_state_layers=aux_hidden_state_layers,
                    _dspark_target_model=_dspark_target_model,workspace_initial_capacity_bytes=None if workspace_initial_capacity_bytes is None else
                        workspace_initial_capacity_bytes[replica*pp*tp:(replica+1)*pp*tp])
            if aux_hidden_state_layers:
                dummy_inputs.update(local_source.dummy_target_input_ports_by_rank)
                capture_ports.update(local_source.aux_hidden_states_by_rank);capture_workspaces.update(local_source.aux_workspaces_by_rank)
                capture_ropes.update(local_source.aux_rope_tables_by_rank);capture_orders.append(local_source.aux_model_input_order)
            continue
        local_step = replace(step, requests=tuple(requests[replica]), finished_request_ids=(), preempted_request_ids=())
        if _retain_source_state:
            local_pool,local_step,arguments=prepared[replica]
            local_source=trace_deepseek_v4_forward(model,local_step,local_pool,hardware,expert_routing,**arguments)
            retained.append(local_source._v4_target_source_state)
            if v2_acceptance[replica] is not None:v2_outputs.update(local_source.v2_sampler_outputs_by_rank)
        else:
            local_source = trace_deepseek_v4_forward(model, local_step, local_pool, hardware, expert_routing, input_ids=tuple(ids[replica]),
            positions=tuple(position for request in requests[replica]
                for position in range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens)),
            slot_mapping=slot_mapping_by_replica[replica], cache_capacity_tokens=cache_capacity_tokens_by_replica[replica],
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            source_use_deep_gemm=source_use_deep_gemm, linear_source_profile=linear_source_profile,
            source_prompt_lengths=source_prompt_lengths_by_replica[replica], indexer_selection_source=indexer_selection_source,
            sample_logits=sample_logits, cast_logits=cast_logits, _replica_context=context,
            aux_hidden_state_layers=aux_hidden_state_layers,
            _dspark_target_model=_dspark_target_model,
            workspace_initial_capacity_bytes=(None if workspace_initial_capacity_bytes is None else
                workspace_initial_capacity_bytes[replica * pp * tp:(replica + 1) * pp * tp]))
        if aux_hidden_state_layers:
            if set(capture_ports).intersection(local_source.aux_hidden_states_by_rank):
                raise ValueError("V4 distributed target captures cannot duplicate physical-rank owners")
            capture_ports.update(local_source.aux_hidden_states_by_rank)
            capture_workspaces.update(local_source.aux_workspaces_by_rank)
            capture_ropes.update(local_source.aux_rope_tables_by_rank)
            capture_orders.append(local_source.aux_model_input_order)
    joined.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "source_data_parallel_rows": rows, "model_sequence_parallel": model_sp,
        "source_model_rows_including_dummy":work_rows,
        "source_expert_physical_rows": physical_rows, "source_hash_expert_valid_rows": sum(rows),
        "source_padding_routing": "hash_cuda_masks_padding; score_dsv4_triton_does_not_consume_mask",
        "actual_provider_bound": False, "normal_capacity_accepted": False,
        "idle_source_scope": "complete target dummy backbone, all-PAD attention, no target logits/sampler; DSpark propose validated separately",
        "dummy_cache_capacity_scope":"128-row empty logical arenas; native worker cache residency external",
        "dummy_pipeline_source":"rank-local intermediate views; no PP transfer, receive gather or runner copy",
        "dummy_intermediate_initialization_scope":"constructor-zero for bound fresh auxiliary source; unbound ordinary history remains unknown",
        "idle_target_source_bound":any(count==0 for count in rows),"real_request_ids":tuple(r.request_id for r in step.requests)}
    if aux_hidden_state_layers:
        joined.aux_dummy_replicas=tuple(count==0 for count in rows)
        joined.dummy_target_input_ports_by_rank=dummy_inputs
        joined.aux_hidden_states_by_rank = capture_ports
        joined.aux_hidden_state_layers = aux_hidden_state_layers
        joined.aux_model_input_order_by_replica = tuple(capture_orders)
        joined.aux_target_model = model
        joined.aux_target_step = step
        joined.aux_target_config = config
        joined.aux_target_pool_ranks = pool.ranks
        joined.aux_target_plan = pool.parallel_plan
        joined.aux_workspaces_by_rank = capture_workspaces
        joined.aux_rope_tables_by_rank = capture_ropes
        joined.aux_tensor_groups = tensor
        joined.aux_data_groups = data
        joined.aux_expert_groups = expert_groups
    if _retain_source_state:
        if '_dspark_capture_consumed' in vars(joined):del joined._dspark_capture_consumed
        joined._v4_distributed_target_state=_V4DistributedTargetState(joined,config,pool.ranks,
            tuple(retained),tensor,data,expert_groups)
        if _v2_accepted_tokens_by_replica is not None:
            joined.v2_sampler_outputs_by_rank=v2_outputs
    return joined


def trace_deepseek_v4_startup_profile(model, pool, hardware, expert_routing, *, tokens, max_requests,
        source_max_model_len, source_use_deep_gemm, workspace_initial_capacity_bytes,
        indexer_max_logits_mb, source_profile):
    """Complete V4 layer loop plus shared worker/sampler storage scopes.

    Explicit eager/no-attention-metadata, synchronous modular TP, pure text,
    EP/SP/CP/MTP-off and native PyTorch-sort/FP32-noise sampler. This is not
    a normal request graph or a timed program. No device tensor is created.
    """
    from .runtime_inputs import ExpertRoutingInput
    from .traced_sampling import VLLM030WorkerStorage, worker_profile_storage_forward
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    config = pool.parallel_plan.config
    if (source_profile != "vllm_0_30_v4_sync_no_metadata_pytorch_sort_fp32_noise"
            or model.model_type != "deepseek_v4" or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
            or pool.hardware_id != "nvidia_h200" or hardware.hardware_id != pool.hardware_id
            or (model.model_id, config.tensor_parallel_size) not in (("deepseek-ai/DeepSeek-V4-Flash", 4),
                ("deepseek-ai/DeepSeek-V4-Pro", 8)) or config.enable_expert_parallel
            or any(value != 1 for value in (config.pipeline_parallel_size, config.sequence_parallel_size,
                config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("V4 startup storage requires explicit H200 TP4/8 synchronous no-metadata source")
    if (any(type(value) is not int or value <= 0 for value in (tokens, max_requests, source_max_model_len,
                indexer_max_logits_mb)) or source_max_model_len > model.max_position_embeddings
            or type(source_use_deep_gemm) is not bool):
        raise ValueError("V4 startup requires positive explicit limits/logits MiB and a declared mHC source branch")
    if (not isinstance(workspace_initial_capacity_bytes, tuple)
            or len(workspace_initial_capacity_bytes) != config.tensor_parallel_size
            or any(type(value) is not int or value < 0 for value in workspace_initial_capacity_bytes)):
        raise ValueError("V4 startup requires explicit workspace entry history per physical rank")
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("V4 startup routing must be a canonical explicit dynamic input")
    rows = min(tokens, max_requests)
    lengths = (tokens // rows,) * (rows - 1) + (tokens // rows + tokens % rows,)
    if max(lengths) > source_max_model_len:
        raise ValueError("V4 startup maximum length must cover each dummy request, not the sum of batch tokens")
    step = ScheduleStep("startup-profile", pool.name, tuple(
        ScheduledRequest(str(index), "prefill", length, 0) for index, length in enumerate(lengths)))
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, tokens)
    traces = tuple(ForwardTrace(rank=rank, namespace=f"profile:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    collective, gather = pool.collective_for("all_reduce"), pool.collective_for("all_gather")

    def rank_profile(trace, capacity):
        network = DeepseekV4Model(trace, model, pool.name, group, collective, token_ids=(0,) * tokens,
            q_lens=lengths, kv_lens=lengths, request_phases=("prefill",) * rows,
            slot_mapping=None, cache_capacity_tokens=None, source_max_model_len=source_max_model_len,
            max_batched_tokens=tokens, linear_source_profile="vllm_0_30_row_major_group128_fp32",
            source_use_deep_gemm=source_use_deep_gemm, hardware=hardware,
            resolutions=resolutions, invocation_ids=invocations, model_input_order=None,
            workspace_initial_capacity_bytes=capacity, profile_indexer_max_logits_mb=indexer_max_logits_mb)
        worker_storage = VLLM030WorkerStorage(trace, tokens=tokens, max_requests=max_requests,
            inputs_embeds_size=model.hidden_size, dtype=model.compute_dtype)
        head = SymbolicTPLogitsProcessor(trace, "profile.lm-head", pool.name, hidden_size=model.hidden_size,
            vocab_size=model.vocab_size, group=group, config=gather)
        trace.checkpoint("worker.profile.begin")
        worker_profile_storage_forward(trace, network, head,
            input_ids=worker_storage.input_ids, positions=worker_storage.positions,
            query_lengths=lengths, sampler_profile="vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise",
            pool_name=pool.name)
        trace.checkpoint("worker.profile.end")

    for trace, capacity in zip(traces, workspace_initial_capacity_bytes, strict=True):
        rank_profile(trace, capacity)
    return group


class DeepseekV4MoE:
    """V4's non-MegaMoE forward specialization; its Runner owns GateLinear."""

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index,
                 resolution, invocation_id, linear_source_profile, workspace=None, source_multiprocessors=None,
                 expert_context=None):
        if model.model_type != "deepseek_v4" or model.num_shared_experts != 1:
            raise NotImplementedError("symbolic V4 modular source currently requires its one shared expert")
        self.hash_routing = layer_index < model.num_hash_layers
        self.experts = DeepseekV4MoERunner(trace, model, pool, prefix, group, config,
            layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
            linear_source_profile=linear_source_profile, workspace=workspace,
            source_multiprocessors=source_multiprocessors, expert_context=expert_context)

    def __call__(self, hidden_states, input_ids=None):
        if self.hash_routing and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids")
        return self._forward_fused_moe(hidden_states, input_ids)

    def _forward_fused_moe(self, hidden_states, input_ids=None):
        org_shape = hidden_states.shape
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            input_ids=input_ids,
        )
        return final_hidden_states.view_as_shape(org_shape)


def _v4_validate_target_V2_feedback(state, rejected):
    """Match the immutable declared source branch, never infer numeric values."""
    if not state.sampler_feedback:
        return
    queries=tuple(request.scheduled_tokens if request.computed_tokens else 1 for request in state.step.requests)
    counts=tuple(q-r for q,r in zip(queries,rejected,strict=True))
    if any(sampled!=counts or num_rejected!=rejected for _,sampled,num_rejected in state.sampler_feedback):
        raise ValueError('DSpark capture acceptance disagrees with its actual target V2 declared sampling feedback')


def _v4_dspark_produced_captures(source, model, step, pool, workspace_entries, *,
        prior_context=None, target_query_lens=None, num_rejected=None):
    """Bind actual same-TP final-stage source ports, without a new schema."""
    if type(source) not in (ForwardTrace, ForwardCollectiveGroup, ForwardPipelineGroup):
        raise TypeError("DSpark produced captures require their actual V4 source owner")
    required = ("aux_hidden_states_by_rank", "aux_hidden_state_layers", "aux_model_input_order", "aux_target_model",
        "aux_target_step", "aux_target_config", "aux_target_pool_ranks", "aux_workspaces_by_rank", "aux_rope_tables_by_rank")
    if any(name not in vars(source) for name in required) or "_dspark_capture_consumed" in vars(source):
        raise ValueError("DSpark target capture owner is incomplete or already consumed")
    target, previous, target_config = source.aux_target_model, source.aux_target_step, source.aux_target_config
    config = pool.parallel_plan.config
    if (not isinstance(target, ModelConfig) or target.is_dspark_draft or model.model_id != target.model_id + "-DSpark"
            or any(getattr(model, field) != getattr(target, field) for field in ("hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_experts", "moe_intermediate_size", "vocab_size", "hc_mult", "q_lora_rank",
                "head_dim", "o_groups", "o_lora_rank", "max_position_embeddings"))
            or source.aux_hidden_state_layers != model.dspark_target_layer_ids):
        raise ValueError("DSpark produced captures must match the registered target identity, shapes and layer IDs")
    if (target_config.tensor_parallel_size != config.tensor_parallel_size
            or target_config.enable_expert_parallel != config.enable_expert_parallel
            or target_config.data_parallel_size != 1 or target_config.context_parallel_size != 1
            or target_config.sequence_parallel_size != 1 or previous.pool_name != pool.name
            or tuple(source.aux_target_pool_ranks[-config.tensor_parallel_size:]) != pool.ranks
            or set(source.aux_hidden_states_by_rank) != set(pool.ranks)):
        raise ValueError("DSpark capture binding requires same-TP/EP final-target-stage ranks and standalone DP1")
    if ('_v4_target_source_state' in vars(source)
            and source._v4_target_source_state.source_branch[-1]!=model.config_sha256):
        raise ValueError("DSpark retained targets require their explicit V2 logits source profile")
    if tuple(request.request_id for request in previous.requests) != tuple(request.request_id for request in step.requests):
        raise ValueError("DSpark produced captures require exactly the original request order")
    if prior_context is None:
        if any(old.phase!='prefill' or old.computed_tokens!=0 or old.data_parallel_rank!=0
                or old.scheduled_tokens!=new.computed_tokens for old,new in zip(previous.requests,step.requests,strict=True)):
            raise ValueError("DSpark fresh capture binding requires the same original requests and complete new target contexts")
        context_rows=sum(request.computed_tokens for request in step.requests)
    else:
        old_requests={} if prior_context.dummy_run else {r.request_id:r for r in prior_context.step.requests}
        admitted=source._v4_target_source_state.admitted_request_ids
        if ('_v4_target_source_state' not in vars(source)
                or type(source._v4_target_source_state) is not _V4TargetSourceState
                or source._v4_target_source_state.owner is not source
                or source._v4_target_source_state.source_branch[-1]!=model.config_sha256
                or tuple(request.scheduled_tokens for request in previous.requests)!=target_query_lens
                or {r.request_id for r in previous.requests}!=(set(old_requests)-set(prior_context.step.finished_request_ids))|set(admitted)
                or any(target.computed_tokens!=(0 if target.request_id in admitted else old_requests[target.request_id].computed_tokens)
                    or target.data_parallel_rank!=new.data_parallel_rank
                    or new.computed_tokens!=target.computed_tokens+target.scheduled_tokens-rejected
                    for target,new,rejected in zip(previous.requests,step.requests,num_rejected,strict=True))):
            raise ValueError("DSpark incremental captures require the actual matching target verification and accepted source prefix")
        context_rows=sum(target_query_lens)
    if '_v4_target_source_state' in vars(source):
        _v4_validate_target_V2_feedback(source._v4_target_source_state,
            (0,)*len(step.requests) if prior_context is None else num_rejected)
    if tuple(source.aux_model_input_order["token_row_permutation"]) != tuple(range(context_rows)):
        raise NotImplementedError("DSpark produced capture reordering needs its explicit incremental context-state contract")
    all_traces = (source,) if type(source) is ForwardTrace else source.traces
    traces = tuple(next(trace for trace in all_traces if trace.rank == rank) for rank in pool.ranks)
    if type(source) is ForwardPipelineGroup:
        group = source.collective_groups[-1] if config.tensor_parallel_size > 1 else None
    else:
        group = source if type(source) is ForwardCollectiveGroup else None
    if group is not None and tuple(group.traces) != traces:
        raise ValueError("DSpark produced capture TP group differs from the last-stage target group")
    if prior_context is not None and (traces!=prior_context.traces or group is not prior_context.group):
        raise ValueError("DSpark incremental target captures must use the original physical traces and TP group")
    for lane, trace in enumerate(traces):
        captures = source.aux_hidden_states_by_rank[trace.rank]
        if type(captures) is not tuple or len(captures) != 3:
            raise ValueError("DSpark target owner requires three actual produced capture ports")
        for index,capture in enumerate(captures):
            trace._check_tensor(capture)
            if capture.shape != (context_rows, model.hidden_size) or capture.dtype != "bf16":
                raise ValueError("DSpark produced capture port shape/dtype differs from actual context rows")
            producer = trace._writers[capture.view.allocation_id]
            if producer is None or producer not in trace._nodes or capture.view.view_id not in trace._nodes[producer].attributes["symbolic_tensor_outputs"]:
                raise ValueError("DSpark capture port requires its actual target producer")
            if '_v4_target_source_state' in vars(source):
                layer,expected_writer,expected_view=dict(source._v4_target_source_state.capture_producers)[trace.rank][index]
                if (layer!=model.dspark_target_layer_ids[index] or producer!=expected_writer
                        or capture.view.view_id!=expected_view):
                    raise ValueError("DSpark retained target port differs from its actual layer/round producer")
        workspace = source.aux_workspaces_by_rank[trace.rank]
        if (not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace
                or workspace.capacity_bytes != workspace_entries[lane]
                or prior_context is not None and workspace is not prior_context.workspaces[lane]):
            raise ValueError("DSpark binding must reuse the actual target workspace and exact current entry capacity")
    return traces, group


def _v4_dspark_distributed_capture_owner(source, model, *, tensor_size, data_size, expert_size, pool_name):
    from src.serving.parallelism.parallel_config import ParallelConfig
    required = ("aux_hidden_states_by_rank", "aux_hidden_state_layers", "aux_model_input_order_by_replica", "aux_target_model",
        "aux_target_step", "aux_target_config", "aux_target_pool_ranks", "aux_target_plan", "aux_workspaces_by_rank",
        "aux_rope_tables_by_rank", "aux_tensor_groups", "aux_data_groups", "aux_expert_groups", "source_scope")
    if (type(source) is not ForwardCollectiveGroup or any(name not in vars(source) for name in required)
            or "_dspark_capture_consumed" in vars(source)):
        raise ValueError("DSpark distributed capture owner is incomplete, foreign or already consumed")
    target, config = source.aux_target_model, source.aux_target_config
    if (not isinstance(target, ModelConfig) or not isinstance(config, ParallelConfig) or target.is_dspark_draft
            or model.model_id != target.model_id + "-DSpark"
            or any(getattr(model, name) != getattr(target, name) for name in ("hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_experts", "moe_intermediate_size", "vocab_size", "hc_mult", "q_lora_rank",
                "head_dim", "o_groups", "o_lora_rank", "max_position_embeddings"))
            or source.aux_hidden_state_layers != model.dspark_target_layer_ids):
        raise ValueError("DSpark distributed captures must match the actual target identity, shapes and layer IDs")
    if (config.tensor_parallel_size != tensor_size or config.data_parallel_size != data_size
            or config.expert_parallel_size != expert_size or config.context_parallel_size != 1
            or config.sequence_parallel_size != 1 or source.aux_target_step.pool_name != pool_name
            or tuple(trace.rank for trace in source.traces) != source.aux_target_pool_ranks
            or source.aux_target_plan.config != config or source.aux_target_plan.num_layers != target.num_hidden_layers
            or source.aux_target_plan.num_experts != target.num_experts
            or type(source.aux_model_input_order_by_replica) is not tuple or len(source.aux_model_input_order_by_replica) != data_size):
        raise ValueError("DSpark distributed capture topology differs from its actual target plan")
    pp, tp = config.pipeline_parallel_size, tensor_size
    final_ranks = tuple(source.aux_target_pool_ranks[(replica * pp + pp - 1) * tp + lane]
        for replica in range(data_size) for lane in range(tp))
    if (set(source.aux_hidden_states_by_rank) != set(final_ranks)
            or set(source.aux_workspaces_by_rank) != set(final_ranks) or set(source.aux_rope_tables_by_rank) != set(final_ranks)
            or len(source.aux_data_groups) != pp or any(len(groups) != tp for groups in source.aux_data_groups)
            or tp > 1 and (len(source.aux_tensor_groups) != data_size
                or any(len(groups) != pp for groups in source.aux_tensor_groups))):
        raise ValueError("DSpark distributed captures require every actual final-stage owner and group")
    return config, final_ranks


def _v4_dspark_distributed_captures(source, model, step, pool, workspace_entries, context, *,
        prior_context=None, target_query_lens=None, num_rejected=None,dummy_run=False):
    """Actual DP ports, including the one explicitly source-proven SP trim."""
    tp, dp = pool.parallel_plan.config.tensor_parallel_size, len(context.rows)
    config, final_ranks = _v4_dspark_distributed_capture_owner(source, model, tensor_size=tp, data_size=dp,
        expert_size=context.expert_size, pool_name=pool.name)
    if pool.ranks != final_ranks[context.rank * tp:(context.rank + 1) * tp]:
        raise ValueError("DSpark distributed draft ranks differ from their own final target stage")
    target_requests = tuple(request for request in source.aux_target_step.requests if request.data_parallel_rank == context.rank)
    if not dummy_run and tuple(request.request_id for request in target_requests) != tuple(request.request_id for request in step.requests):
        raise ValueError("DSpark distributed captures require the same original request owners and order")
    retained=None
    if '_v4_distributed_target_state' in vars(source):
        state=source._v4_distributed_target_state
        if (type(state) is not _V4DistributedTargetState or state.owner is not source or len(state.states)!=dp):
            raise ValueError("DSpark distributed retained target lacks its actual per-replica revisions")
        retained=state.states[context.rank]
        if (type(retained) is not _V4TargetSourceState or '_v4_target_source_state' not in vars(retained.owner)
                or retained.owner._v4_target_source_state is not retained
                or retained.source_branch[-1]!=model.config_sha256):
            raise ValueError("DSpark distributed retained target requires its explicit V2 logits profile")
    if dummy_run:
        if (target_requests or 'aux_dummy_replicas' not in vars(source) or not source.aux_dummy_replicas[context.rank]
                or retained is not None and not retained.dummy_run
                or len(step.requests)!=1 or step.requests[0].computed_tokens!=1 or step.requests[0].scheduled_tokens!=5):
            raise ValueError('DSpark dummy capture requires its actual six-row zero-position idle target')
        rows=6
    elif prior_context is None:
        if any(old.phase!='prefill' or old.computed_tokens!=0 or old.scheduled_tokens!=new.computed_tokens
                for old,new in zip(target_requests,step.requests,strict=True)):
            raise ValueError("DSpark distributed fresh captures require the same original request owners and complete contexts")
        rows=sum(request.computed_tokens for request in step.requests)
    else:
        old_requests={} if prior_context.dummy_run else {r.request_id:r for r in prior_context.step.requests}
        admitted=() if retained is None else retained.admitted_request_ids
        if (retained is None or tuple(request.scheduled_tokens for request in target_requests)!=target_query_lens
                or {r.request_id for r in target_requests}!=(set(old_requests)-set(prior_context.step.finished_request_ids))|set(admitted)
                or any(target.computed_tokens!=(0 if target.request_id in admitted else old_requests[target.request_id].computed_tokens)
                    or target.data_parallel_rank!=new.data_parallel_rank
                    or new.computed_tokens!=target.computed_tokens+target.scheduled_tokens-rejected
                    for target,new,rejected in zip(target_requests,step.requests,num_rejected,strict=True))):
            raise ValueError("DSpark distributed incremental captures require the actual matching verification and accepted prefix")
        rows=sum(target_query_lens)
    if retained is not None and not dummy_run:
        _v4_validate_target_V2_feedback(retained,(0,)*len(step.requests) if prior_context is None else num_rejected)
    order = source.aux_model_input_order_by_replica[context.rank]
    if order is None or tuple(order["token_row_permutation"]) != tuple(range(rows)):
        raise NotImplementedError("DSpark distributed capture row reordering requires its explicit context-state contract")
    traces = tuple(next(trace for trace in source.traces if trace.rank == rank) for rank in pool.ranks)
    group = source.aux_tensor_groups[context.rank][-1] if tp > 1 else None
    if (tuple(context.traces) != traces or tp > 1 and context.tensor_groups != (group,)
            or context.data_groups != (source.aux_data_groups[-1],)):
        raise ValueError("DSpark distributed binding must reuse the actual last-stage TP and DP groups")
    if prior_context is not None and (traces!=prior_context.traces or group is not prior_context.group):
        raise ValueError("DSpark distributed incremental captures must reuse the original physical traces and TP group")
    target_sp = config.enable_expert_parallel and tp > 1 and config.pipeline_parallel_size == 1
    for lane, trace in enumerate(traces):
        ports = source.aux_hidden_states_by_rank[trace.rank]
        if type(ports) is not tuple or len(ports) != 3:
            raise ValueError("DSpark distributed target requires three actual capture ports per rank")
        workspace = source.aux_workspaces_by_rank[trace.rank]
        if (not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace
                or workspace.capacity_bytes != workspace_entries[lane]
                or prior_context is not None and workspace is not prior_context.workspaces[lane]):
            raise ValueError("DSpark distributed binding requires the actual workspace and exact current entry capacity")
        for index, (layer, capture) in enumerate(zip(model.dspark_target_layer_ids, ports, strict=True)):
            trace._check_tensor(capture)
            if (capture.dtype != "bf16" or capture.shape != (rows, model.hidden_size) or not capture.is_contiguous
                    or capture.view.offset_bytes != 0):
                raise ValueError("DSpark distributed capture must be the exact BF16 context prefix")
            writer = trace._writers[capture.view.allocation_id]
            if writer is None or writer not in trace._nodes:
                raise ValueError("DSpark distributed capture requires its actual target producer")
            producer = trace._nodes[writer]
            if retained is not None:
                actual_layer,actual_writer,actual_view=dict(retained.capture_producers)[trace.rank][index]
                if (actual_layer!=layer or actual_writer!=writer or actual_view!=capture.view.view_id):
                    raise ValueError("DSpark distributed capture differs from its immutable layer/round producer")
            if target_sp:
                contracts=tuple(name for name,submissions in group._submissions.items()
                    if trace.rank in submissions and submissions[trace.rank]==writer)
                if (producer.kind is not WorkloadKind.COLLECTIVE or producer.attributes["operation"] != "all_gather"
                        or len(contracts)!=1
                        or retained is None and contracts[0] != group.namespace + f":aux-layer-{layer}:sp-capture-gather"
                        or len(producer.attributes["symbolic_tensor_outputs"]) != 1):
                    raise ValueError("DSpark SP capture requires its own actual layer/rank TP gather")
                full = trace._views[producer.attributes["symbolic_tensor_outputs"][0]]
                if (full.allocation_id != capture.view.allocation_id or full.shape != (-(-rows // tp) * tp, model.hidden_size)
                        or full.dtype != capture.dtype or full.offset_bytes != 0 or full.strides_bytes != capture.view.strides_bytes):
                    raise ValueError("DSpark SP capture trim differs from its actual full gather backing")
            elif retained is None:
                last = source.aux_target_plan.layer_to_pipeline_stage[layer] == config.pipeline_parallel_size - 1
                suffix = f":aux-layer-{layer}:mean" if last else f":receive:runner-copy:aux_hidden_states_{index}"
                constructor = dummy_run and not last and (
                    capture.view.allocation_id == trace.qualify(f"worker.intermediate:aux_hidden_states_{index}")
                    and capture.view.allocation_id in trace._persistent_ids
                    and "source_component" in producer.attributes
                    and producer.attributes["source_component"] == "vllm_0_30_V4_PP_intermediate_constructor"
                    and producer.attributes["source_intermediate_role"] == f"aux_hidden_states_{index}"
                    and producer.attributes["constructor_once"] is True
                    and producer.attributes["source_constant"] == 0)
                outputs = tuple(trace._views[view] for view in producer.attributes["symbolic_tensor_outputs"])
                if not constructor and (not producer.node_id.endswith(suffix)
                        or capture.view.view_id not in producer.attributes["symbolic_tensor_outputs"]):
                    raise ValueError("DSpark capture port is not its actual local mean or own PP slot copy")
                if constructor and (producer.kind is not WorkloadKind.ELEMENTWISE
                        or producer.attributes["operation"] != "fill"
                        or producer.shape["elements"] * 2 != trace._allocations[capture.view.allocation_id].size_bytes
                        or not any(view.view_id == capture.view.allocation_id and view.dtype == "bf16"
                            and view.offset_bytes == 0 and len(view.shape) == 2 and view.shape[0] >= rows
                            and view.shape[1] == model.hidden_size and view.strides_bytes == (model.hidden_size * 2, 2)
                            and view.shape[0] * model.hidden_size * 2 == trace._allocations[capture.view.allocation_id].size_bytes
                            for view in outputs)):
                    raise ValueError("DSpark dummy capture must use its actual local constructor backing")
    return traces, group


@dataclass(frozen=True)
class _DSparkCacheState:
    """One source-owned logical cache revision, never a native page receipt."""
    owner: object
    model_sha256: str
    config: object
    pool_name: str
    ranks: tuple
    step: object
    step_ids: tuple
    context_slots: tuple
    query_slots: tuple
    capacity: int
    max_batched_tokens: int
    traces: tuple
    group: object
    caches: tuple
    workspaces: tuple
    ropes: tuple
    topk: tuple
    replica_signature: tuple
    request_states: tuple = ()
    input_buffers: tuple = ()
    dummy_run: bool = False


def _dspark_admission_ids(inputs):
    if inputs is None or type(inputs) is dict and 'admitted_requests' not in inputs:return ()
    if (type(inputs) is not dict or type(inputs['admitted_requests']) is not tuple
            or any(type(row) is not tuple or len(row)!=3 or type(row[0]) is not str
                or type(row[1]) is not int or type(row[2]) is not tuple for row in inputs['admitted_requests'])
            or len({row[0] for row in inputs['admitted_requests']})!=len(inputs['admitted_requests'])):
        raise ValueError('DSpark admission requires explicit distinct request/row/prompt descriptors')
    return tuple(row[0] for row in inputs['admitted_requests'])


def _dspark_incremental_context(prior, model, step, pool, workspace_entries,
        context_slots, query_slots, capacity, max_batched_tokens, target_query_lens, num_rejected, replica_context,
        admitted_request_ids=()):
    """Preflight fresh target prompts and continuing verification before mutation.

    DFlash's input kernel retains every target row, masks rejected context
    slots with -1, then starts the next query after the last accepted row.
    Previous query KV is provisional: accepted rows are reprojected from
    TARGET captures; an old draft KV row never becomes context by assertion.
    A newly admitted request projects its entire fresh prompt and has no old
    logical context, even when it reuses a finished request's physical pages.
    """
    if (type(prior) is not _DSparkCacheState or 'dspark_cache_state' not in vars(prior.owner)
            or 'dspark_outputs' not in vars(prior.owner)
            or prior.owner.dspark_cache_state is not prior):
        raise ValueError("DSpark cache continuation needs its current unconsumed source revision")
    signature = (() if replica_context is None else
        (replica_context.rank, replica_context.expert_size, replica_context.sequence_parallel))
    old_requests={} if prior.dummy_run else {request.request_id:request for request in prior.step.requests}
    if (prior.model_sha256 != model.config_sha256 or prior.config != pool.parallel_plan.config
            or prior.pool_name != pool.name or prior.ranks != pool.ranks or prior.replica_signature != signature
            or prior.capacity != capacity or prior.max_batched_tokens != max_batched_tokens
            or prior.step.step_id == step.step_id or step.step_id in prior.step_ids
            or set(admitted_request_ids).intersection(set(old_requests)-set(prior.step.finished_request_ids))
            or (set(old_requests)-set(prior.step.finished_request_ids))|set(admitted_request_ids) != {r.request_id for r in step.requests}
            or any(old_requests[r.request_id].data_parallel_rank != r.data_parallel_rank for r in step.requests
                if r.request_id not in admitted_request_ids)
            or tuple(workspace.capacity_bytes for workspace in prior.workspaces) != workspace_entries):
        raise ValueError("DSpark cache continuation changed its model, topology, requests, limits or workspace owner")
    if (any(request.phase != "decode" or request.scheduled_tokens != model.dspark_block_size
                or request.mtp_proposer_mode != "disabled" for request in step.requests)
            or any(type(maps) is not tuple or len(maps) != 3 for maps in (context_slots, query_slots))):
        raise ValueError("DSpark incremental source requires unchanged full query-block ownership and three layer maps")
    for cslots, qslots in zip(context_slots, query_slots, strict=True):
        if (type(cslots) is not tuple or type(qslots) is not tuple
                or len(cslots) != sum(r.computed_tokens for r in step.requests)
                or len(qslots) != sum(r.scheduled_tokens for r in step.requests)
                or any(type(slot) is not int or not 0 <= slot < capacity for slot in cslots + qslots)
                or len(set(cslots + qslots)) != len(cslots) + len(qslots)):
            raise ValueError("DSpark incremental cache requires complete distinct accepted-context and query slots")
    count = len(step.requests)
    if (type(target_query_lens) is not tuple or len(target_query_lens) != count
            or type(num_rejected) is not tuple or len(num_rejected) != count
            or any(type(q) is not int or q<1 or (q!=request.computed_tokens or rejected!=0
                if request.request_id in admitted_request_ids else q!=model.dspark_block_size+1)
                for request,q,rejected in zip(step.requests,target_query_lens,num_rejected,strict=True))
            or any(type(r) is not int or not 0 <= r <= model.dspark_block_size for r in num_rejected)):
        raise ValueError("DSpark incremental source requires full target verification rows and explicit rejected counts")
    valid = tuple(q - r for q, r in zip(target_query_lens, num_rejected, strict=True))
    if any(new.computed_tokens != (0 if new.request_id in admitted_request_ids else old_requests[new.request_id].computed_tokens) + accepted
            for new, accepted in zip(step.requests, valid, strict=True)):
        raise ValueError("DSpark incremental accepted target rows must exactly extend the old context")
    expected_positions = tuple(position for new, physical, accepted in zip(step.requests, target_query_lens, valid, strict=True)
        for old_count in (0 if new.request_id in admitted_request_ids else old_requests[new.request_id].computed_tokens,)
        for position in (*range(old_count, old_count + accepted), *((0,) * (physical - accepted))))
    insert_slots = []
    for old_slots, old_query, new_slots in zip(prior.context_slots, prior.query_slots, context_slots, strict=True):
        old_cursor = query_cursor = new_cursor = 0
        old_maps={}
        for request in (() if prior.dummy_run else prior.step.requests):
            old_maps[request.request_id]=(old_slots[old_cursor:old_cursor+request.computed_tokens],
                old_query[query_cursor:query_cursor+request.scheduled_tokens])
            old_cursor+=request.computed_tokens;query_cursor+=request.scheduled_tokens
        writes = []
        for new, accepted, physical in zip(step.requests, valid, target_query_lens, strict=True):
            fresh=new.request_id in admitted_request_ids
            old_count=0 if fresh else old_requests[new.request_id].computed_tokens
            old_query_count=0 if fresh else old_requests[new.request_id].scheduled_tokens
            before,previous_query=((),()) if fresh else old_maps[new.request_id]
            after = new_slots[new_cursor:new_cursor + old_count]
            appended = new_slots[new_cursor + old_count:new_cursor + old_count + accepted]
            if (before != after or appended[:min(accepted, old_query_count)] != previous_query[:min(accepted, old_query_count)]):
                raise ValueError("DSpark incremental cache must preserve old context and actual same-request query slots")
            writes.extend((*appended, *((-1,) * (physical - accepted))))
            new_cursor += old_count + accepted
        insert_slots.append(tuple(writes))
    for lane, trace in enumerate(prior.traces):
        if trace.rank != pool.ranks[lane] or trace._recorded_graph is not None:
            raise ValueError("DSpark incremental cache needs its open original physical-rank trace")
        if prior.input_buffers:
            from .traced_sampling import dspark_input_buffer_values
            if len(prior.input_buffers)!=len(prior.traces):
                raise ValueError('DSpark continuation lost its original prepared input buffers')
            dspark_input_buffer_values(prior.input_buffers[lane],physical_rows=sum(r.scheduled_tokens for r in step.requests)
                if replica_context is None else replica_context.rows[replica_context.rank],vocab_size=model.vocab_size)
        for cache in prior.caches[lane]:
            trace._check_tensor(cache)
            if (cache.shape != (capacity, 584) or cache.dtype != 'uint8'
                    or cache.view.allocation_id not in trace._persistent_ids
                    or trace._writers[cache.view.allocation_id] is None and not prior.dummy_run):
                raise ValueError("DSpark incremental cache lacks its persistent preceding query producer")
    return expected_positions, tuple(insert_slots)



def _dspark_dummy_anchor(prior,model):
    """Dummy propose reads resident state row0, never the last batch's row0."""
    if prior is None:return 0
    if not prior.request_states:
        if not prior.dummy_run:
            raise ValueError('DSpark admitted idle source lost its retained request states')
        return 0
    anchors=[]
    for trace,state in zip(prior.traces,prior.request_states,strict=True):
        if (state.trace is not trace or '_v2_post_update_state' not in vars(trace) or trace._v2_post_update_state is not state
                or set(state.request_ids)!=set(state.finished_request_ids)
                or len(state.last_sampled_values)!=state.limits[0]
                or type(state.last_sampled_values[0]) is not int or not 0<=state.last_sampled_values[0]<model.vocab_size):
            raise ValueError('DSpark admitted idle requires all requests finished and an explicit retained state0 sampled value')
        trace._check_tensor(state.last_sampled)
        if (state.last_sampled.view.allocation_id not in trace._persistent_ids
                or trace._allocations[state.last_sampled.view.allocation_id].freed_event is not None
                or trace._writers[state.last_sampled.view.allocation_id] is None):
            raise ValueError('DSpark admitted idle lost its live last-sampled state backing')
        anchors.append(state.last_sampled_values[0])
    if len(set(anchors))!=1:raise ValueError('DSpark idle TP lanes disagree on retained state0 sampled value')
    return anchors[0]


def _dspark_dummy_continuation(prior,model,step,pool,workspace_entries,capacity,max_batched_tokens,context):
    from .traced_sampling import dspark_input_buffer_values
    if (type(prior) is not _DSparkCacheState
            or not prior.dummy_run and {r.request_id for r in prior.step.requests}!=set(prior.step.finished_request_ids)
            or 'dspark_cache_state' not in vars(prior.owner) or prior.owner.dspark_cache_state is not prior
            or prior.model_sha256!=model.config_sha256 or prior.config!=pool.parallel_plan.config
            or prior.ranks!=pool.ranks or prior.pool_name!=pool.name
            or prior.replica_signature!=(context.rank,context.expert_size,context.sequence_parallel)
            or prior.capacity!=capacity or prior.max_batched_tokens!=max_batched_tokens
            or step.step_id in prior.step_ids or len(prior.input_buffers)!=len(pool.ranks)
            or tuple(w.capacity_bytes for w in prior.workspaces)!=workspace_entries):
        raise ValueError('DSpark dummy continuation requires its finished or never-admitted original worker/cache revision')
    _dspark_dummy_anchor(prior,model)
    for lane,trace in enumerate(prior.traces):
        if trace.rank!=pool.ranks[lane] or trace._recorded_graph is not None:
            raise ValueError('DSpark dummy continuation lost its original open rank')
        dspark_input_buffer_values(prior.input_buffers[lane],physical_rows=context.rows[context.rank],vocab_size=model.vocab_size)
        for cache in prior.caches[lane]:
            trace._check_tensor(cache)
            if (cache.view.allocation_id not in trace._persistent_ids
                    or trace._allocations[cache.view.allocation_id].freed_event is not None):
                raise ValueError('DSpark idle cache lost its original live persistent backing')
            if not prior.request_states and trace._writers[cache.view.allocation_id] is not None:
                raise ValueError('DSpark never-admitted dummy cache acquired a real writer')


def _v4_dspark_dummy_preflight(model,step,pool,source,prior,inputs,context,workspace_entries):
    if (type(inputs) is not dict or set(inputs)!={'dummy_plan'} or type(inputs['dummy_plan']) is not dict
            or 'dummy_run' not in inputs['dummy_plan'] or inputs['dummy_plan']['dummy_run'] is not True
            or context is None or type(source) is not ForwardCollectiveGroup or '_v4_distributed_target_state' not in vars(source)):
        raise ValueError('DSpark dummy source requires its explicit actual target-bound preparation plan')
    _v4_dspark_distributed_captures(source,model,step,pool,workspace_entries,context,prior_context=prior,dummy_run=True)
    target=source._v4_distributed_target_state.states[context.rank]
    if not target.dummy_run or 'dummy_target_input_ports_by_rank' not in vars(source):
        raise ValueError('DSpark dummy source requires the actual retained dummy target input owner')
    arguments=[]
    for trace in context.traces:
        metadata=dict(source.dummy_target_input_ports_by_rank[trace.rank])
        for role in ('positions','query_start_loc','idx_mapping'):
            trace._check_tensor(metadata[role]);writer=trace._writers[metadata[role].view.allocation_id]
            if writer is None or writer not in trace._nodes or not trace._nodes[writer].attributes['source_dummy_run']:
                raise ValueError('DSpark dummy metadata lost its actual zero/PAD target producer')
        arguments.append(dict(target_positions=metadata['positions'],target_query_start_loc=metadata['query_start_loc'],
            idx_mapping=metadata['idx_mapping'],hidden_size=model.hidden_size,noise_token_id=model.dspark_noise_token_id,
            step_id=source.aux_target_step.step_id,pool_name=pool.name,
            prior_buffers=None if prior is None else prior.input_buffers[len(arguments)]))
        if prior is not None and prior.request_states:
            lane=len(arguments)-1
            retained=dict(dict(target.retained_sampling_ports)[trace.rank])
            arguments[-1].update(request_state=prior.request_states[lane],temperature=retained['temperature'],seeds=retained['seeds'])
    return inputs['dummy_plan'],tuple(arguments)


def _v4_dspark_dflash_preflight(model, step, pool, source, prior_context, inputs, *, input_ids,
        context_positions, context_slot_mappings, query_slot_mappings, cache_capacity_tokens,
        source_max_model_len, max_batched_tokens, compute_confidence, replica_context):
    """Preflight complete post-update/prepare plans before touching ANY rank."""
    from .traced_sampling import dspark_prepare_input_plan, validate_dspark_prepare_ports, vllm030_v2_post_update_forward
    if inputs is None:
        if prior_context is not None and (prior_context.request_states or prior_context.input_buffers):
            raise ValueError('DSpark cannot drop its bound persistent native input-source program')
        return None
    fields={'sampled_token_ids','group_ids','layer_group_indices','kernel_block_sizes','block_table_ids','max_requests','all_token_ids_storage'}
    if (type(inputs) is not dict or not fields.issubset(inputs)
            or not set(inputs).issubset(fields|{'initialize_last_sampled','admitted_requests'}) or source is None):
        raise ValueError('DSpark DFlash input source requires all explicit group/table/state declarations and an actual target')
    if compute_confidence:
        raise NotImplementedError('DSpark V2 no-adaptive input source excludes the optional adaptive confidence computation')
    state=(source._v4_target_source_state if replica_context is None else
        source._v4_distributed_target_state.states[replica_context.rank])
    if (type(state) is not _V4TargetSourceState or state.owner._v4_target_source_state is not state
            or not state.v2_input_ports or 'v2_sampler_outputs_by_rank' not in vars(source)):
        raise ValueError('DSpark input source requires its current actual target V2 input/sampler ports')
    previous=state.step
    target_positions=tuple(pos for request in previous.requests for pos in range(request.computed_tokens,request.computed_tokens+request.scheduled_tokens))
    meta_by_rank=dict(state.v2_input_ports);feedback=dict((rank,rejected) for rank,_,rejected in state.sampler_feedback)
    first_rejected=feedback[pool.ranks[0]]
    plan=dspark_prepare_input_plan(target_query_lengths=tuple(r.scheduled_tokens for r in previous.requests),
        target_positions=target_positions,rejected_counts=first_rejected,group_ids=inputs['group_ids'],
        layer_group_indices=inputs['layer_group_indices'],kernel_block_sizes=inputs['kernel_block_sizes'],block_table_ids=inputs['block_table_ids'],
        max_requests=inputs['max_requests'],max_tokens=max_batched_tokens,max_model_len=source_max_model_len,
        cache_capacity_tokens=cache_capacity_tokens,num_query_per_request=model.dspark_block_size)
    if (plan['contexts']!=tuple(r.computed_tokens for r in step.requests) or plan['context_positions']!=context_positions
            or tuple(plan['full_context_slots_by_group'][g] for g in plan['layer_group_indices'])!=context_slot_mappings
            or tuple(plan['query_slots_by_group'][g] for g in plan['layer_group_indices'])!=query_slot_mappings
            or type(inputs['sampled_token_ids']) is not tuple or len(inputs['sampled_token_ids'])!=len(step.requests)
            or any(not ids or ids[-1]!=input_ids[index*model.dspark_block_size] for index,ids in enumerate(inputs['sampled_token_ids']))):
        raise ValueError('DSpark actual prepared metadata must match its complete declared context/query slots and sampled anchors')
    if prior_context is not None and (len(prior_context.request_states) not in (0,len(pool.ranks))
            or not prior_context.request_states and (not prior_context.dummy_run or not _dspark_admission_ids(inputs))
            or len(prior_context.input_buffers)!=len(pool.ranks)):
        raise ValueError('DSpark input source continuation needs its original request and preparation buffers')
    prepared=[]
    traces=(source,) if type(source) is ForwardTrace else source.traces
    for lane,rank in enumerate(pool.ranks):
        trace=next(trace for trace in traces if trace.rank==rank);meta=dict(meta_by_rank[rank])
        sampler=source.v2_sampler_outputs_by_rank[rank]
        post_args=dict(request_ids=tuple(r.request_id for r in previous.requests),state_indices=state.v2_state_indices,
            query_lengths=tuple(r.scheduled_tokens for r in previous.requests),computed_before=tuple(r.computed_tokens for r in previous.requests),
            sampled_token_ids=inputs['sampled_token_ids'],initial_total_lengths=tuple(r.scheduled_tokens for r in previous.requests)
                if prior_context is None or not prior_context.request_states else None,
            max_requests=inputs['max_requests'],max_model_len=source_max_model_len,vocab_size=model.vocab_size,
            all_token_ids_storage=inputs['all_token_ids_storage'],idx_mapping=meta['idx_mapping'],query_start_loc=meta['query_start_loc'],
            step_id=previous.step_id,pool_name=pool.name,prior_state=None if prior_context is None or not prior_context.request_states else prior_context.request_states[lane])
        post_args['finished_request_ids']=previous.finished_request_ids
        post_args['initialize_last_sampled']=False if 'initialize_last_sampled' not in inputs else inputs['initialize_last_sampled']
        post_args['admitted_requests']=() if 'admitted_requests' not in inputs else inputs['admitted_requests']
        post_args['admission_input_ids']=None if not post_args['admitted_requests'] else meta['input_ids']
        if prior_context is not None and not prior_context.request_states:
            post_args['initial_last_sampled']=dict(prior_context.input_buffers[lane].shared)['dummy-last-sampled']
        _,computed=vllm030_v2_post_update_forward(trace,sampler,**post_args,_preflight_only=True)
        if computed!=plan['contexts']:raise ValueError('DSpark post_update cannot diverge from prepared accepted context')
        prep_args=dict(target_positions=meta['positions'],target_query_start_loc=meta['query_start_loc'],idx_mapping=meta['idx_mapping'],
            temperature=meta['temperature'],seeds=meta['seeds'],hidden_size=model.hidden_size,noise_token_id=model.dspark_noise_token_id,
            step_id=previous.step_id,pool_name=pool.name,prior_buffers=None if prior_context is None else prior_context.input_buffers[lane])
        validate_dspark_prepare_ports(trace,plan,state_indices=post_args['state_indices'],**prep_args)
        prepared.append((sampler,post_args,prep_args))
    return plan,tuple(prepared)


def _dspark_routing_inputs(model, step, pool, expert_routing, tokens):
    from src.serving.schedule.schedule_step import schedule_step_sha256
    config=pool.parallel_plan.config
    invocations=tuple(f'{step.step_id}:layer-{layer}:moe' for layer in model.dspark_layer_indices)
    ranks=pool.ranks if config.enable_expert_parallel else (pool.ranks[0],)
    resolutions=expert_routing.resolve_layers(tuple((layer,invocation,ordinal,ranks)
        for ordinal,(layer,invocation) in enumerate(zip(model.dspark_layer_indices,invocations,strict=True))),
        step_id=step.step_id,schedule_step_sha256=schedule_step_sha256(step),tokens=tokens,
        num_experts=model.num_experts,topk=model.num_experts_per_tok,model_config_sha256=model.config_sha256,
        pool_name=pool.name,tensor_parallel_size=config.tensor_parallel_size,expert_parallel_size=config.expert_parallel_size,
        expert_placement_strategy='linear')
    return invocations,resolutions


def trace_deepseek_v4_dspark_forward(model, step, pool, hardware, expert_routing, *,
        input_ids, positions, context_positions, context_slot_mappings, query_slot_mappings,
        captured_target_layers, cache_capacity_tokens, source_max_model_len, max_batched_tokens,
        workspace_initial_capacity_bytes, source_use_deep_gemm, compute_confidence, target_capture_source=None, _replica_context=None,
        prior_context=None, target_query_lens=None, num_rejected=None, _dflash_inputs=None, _physical_query_rows=None,_dummy_run=False):
    """Complete three-layer draft, full-vocab greedy Markov source branch.

    Positive context and all-valid full query blocks are explicit inputs.
    An optional current source-owned cache revision retains history and projects
    only this target verification's physical rows, masking its rejected suffix.
    Captures are either declared external ports or actual producer ports from
    a complete same-TP target source; cache slots remain declared/logical.
    A bound target is symbolic source execution, not numerical model output.
    Serving proposer, native pages, normal memory, numerical sampling, provider
    and pricing acceptance remain separate.
    """
    from .runtime_inputs import ExpertRoutingInput
    from src.serving.schedule.schedule_step import ScheduleStep, schedule_step_sha256
    from src.serving.parallelism.tp import validate_dspark_parallelism, dspark_local_shape
    from .traced_operators import SymbolicOperator, SymbolicDSparkNoncausalSWA
    if not isinstance(model, ModelConfig) or not model.is_dspark_draft:
        raise ValueError("standalone DSpark source requires its registered draft ModelConfig")
    if not isinstance(step, ScheduleStep) or step.pool_name != pool.name or not step.requests:
        raise ValueError("DSpark requires its complete declared query-block schedule")
    config = pool.parallel_plan.config
    if (type(_dummy_run) is not bool or _dummy_run and (_replica_context is None or target_capture_source is None
            or len(step.requests)!=1 or step.requests[0].computed_tokens!=1)):
        raise ValueError('DSpark dummy forward requires one non-request query block and its real idle target')
    if prior_context is None and (target_query_lens is not None or num_rejected is not None):
        raise ValueError("DSpark acceptance declarations require their preceding cache revision")
    if _replica_context is not None and not isinstance(_replica_context, _V4ReplicaContext):
        raise TypeError("DSpark DP source requires its normalized replica context")
    validate_dspark_parallelism(model, config)
    if pool.hardware_id != "nvidia_h200" or hardware.hardware_id != pool.hardware_id or pool.role != "colocated":
        raise NotImplementedError("standalone DSpark source requires its declared H200 colocated draft pool")
    if not isinstance(expert_routing, ExpertRoutingInput) or expert_routing.source_kind == "not_applicable":
        raise ValueError("DSpark requires explicit routing input")
    if source_use_deep_gemm is not False or type(compute_confidence) is not bool:
        raise NotImplementedError("DSpark source binds optional TileLang prenorm and explicit confidence choice")
    tp = config.tensor_parallel_size
    local_shapes = tuple(dspark_local_shape(model, pool.parallel_plan, rank) for rank in range(tp))
    if (len(pool.ranks) != tp or type(captured_target_layers) is not tuple
            or any(type(layer) is not int for layer in captured_target_layers)
            or captured_target_layers != model.dspark_target_layer_ids):
        raise ValueError("DSpark source requires all draft ranks and exact ordered target capture IDs")
    q_lens = tuple(request.scheduled_tokens for request in step.requests)
    contexts = tuple(request.computed_tokens for request in step.requests)
    owner = 0 if _replica_context is None else _replica_context.rank
    if any(request.data_parallel_rank != owner for request in step.requests):
        raise ValueError("DSpark request owner is outside its standalone DP1 pool")
    if (any(request.phase != "decode" or request.scheduled_tokens != model.dspark_block_size
            or request.computed_tokens < 1 or request.mtp_proposer_mode != "disabled"
            for request in step.requests)):
        raise NotImplementedError("DSpark standalone source needs positive fresh contexts and complete all-valid query blocks")
    rows, context_rows = sum(q_lens), sum(contexts)
    work_rows = rows if _physical_query_rows is None else _physical_query_rows
    if (type(work_rows) is not int or type(max_batched_tokens) is not int
            or work_rows < rows or work_rows % model.dspark_block_size
            or work_rows > max_batched_tokens
            or work_rows > rows and (_dflash_inputs is None or _replica_context is None)):
        raise NotImplementedError("DSpark unequal DP physical rows require actual prepared buffers")
    if _replica_context is not None and (tuple(trace.rank for trace in _replica_context.traces) != pool.ranks
            or len(_replica_context.rows) < 2 or not 0 <= owner < len(_replica_context.rows)
            or _replica_context.rows[owner] != work_rows or len(_replica_context.data_groups) != 1
            or len(_replica_context.data_groups[0]) != tp
            or _replica_context.expert_size not in (1, tp * len(_replica_context.rows))):
        raise ValueError("DSpark DP source context disagrees with actual rank groups and query rows")
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < max(128, context_rows + rows)
            or cache_capacity_tokens % 8 or type(source_max_model_len) is not int
            or not max(c + q for c, q in zip(contexts, q_lens, strict=True)) <= source_max_model_len <= model.max_position_embeddings
            or type(max_batched_tokens) is not int or max_batched_tokens < rows
            or prior_context is None and max_batched_tokens < context_rows):
        raise ValueError("DSpark source bounds must contain its actual context projection and query rows")
    expected_positions = tuple(p for context, query in zip(contexts, q_lens, strict=True) for p in range(context, context + query))
    expected_context_positions = ((0,)*6 if _dummy_run else tuple(p for context in contexts for p in range(context))) if prior_context is None else context_positions
    if (type(input_ids) is not tuple or len(input_ids) != rows
            or any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_ids)
            or type(positions) is not tuple or positions != expected_positions
            or type(context_positions) is not tuple or context_positions != expected_context_positions
            or any(type(pos) is not int for pos in positions + context_positions)):
        raise ValueError("DSpark source needs explicit IDs and exact context/query positions")
    for request_index, query in enumerate(q_lens):
        start = sum(q_lens[:request_index])
        if input_ids[start + 1:start + query] != (model.dspark_noise_token_id,) * (query - 1):
            raise ValueError("DSpark full-block source requires anchor then declared noise IDs")
    for maps, length in ((context_slot_mappings, context_rows), (query_slot_mappings, rows)):
        if (type(maps) is not tuple or len(maps) != 3 or any(type(slots) is not tuple or len(slots) != length
                or any(type(slot) is not int or (slot!=-1 if _dummy_run else not 0<=slot<cache_capacity_tokens) for slot in slots) for slots in maps)):
            raise ValueError("DSpark needs three complete explicit per-layer logical slot mappings")
    for context_slots, query_slots in zip(context_slot_mappings, query_slot_mappings, strict=True):
        if not _dummy_run and len(set(context_slots + query_slots)) != context_rows + rows:
            raise ValueError("DSpark fresh context and query slots must be disjoint for every layer")
    if (type(workspace_initial_capacity_bytes) is not tuple or len(workspace_initial_capacity_bytes) != tp
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("DSpark needs actual per-rank workspace entry capacities")
    insert_slots = context_slot_mappings
    projection_rows = 6 if _dummy_run else context_rows
    if prior_context is not None and _dummy_run:
        _dspark_dummy_continuation(prior_context,model,step,pool,workspace_initial_capacity_bytes,
            cache_capacity_tokens,max_batched_tokens,_replica_context)
        traces,group=_v4_dspark_distributed_captures(target_capture_source,model,step,pool,workspace_initial_capacity_bytes,
            _replica_context,prior_context=prior_context,dummy_run=True)
    elif prior_context is not None:
        expected, insert_slots = _dspark_incremental_context(prior_context, model, step, pool,
            workspace_initial_capacity_bytes, context_slot_mappings, query_slot_mappings, cache_capacity_tokens,
            max_batched_tokens, target_query_lens, num_rejected, _replica_context,_dspark_admission_ids(_dflash_inputs))
        if context_positions != expected:
            raise ValueError("DSpark incremental context positions must mask the rejected target suffix")
        projection_rows = sum(target_query_lens)
        if projection_rows > max_batched_tokens:
            raise ValueError("DSpark incremental target rows exceed the actual worker bound")
        traces, group = prior_context.traces, prior_context.group
        if target_capture_source is not None:
            if _replica_context is None:
                traces,group=_v4_dspark_produced_captures(target_capture_source,model,step,pool,
                    workspace_initial_capacity_bytes,prior_context=prior_context,target_query_lens=target_query_lens,
                    num_rejected=num_rejected)
            else:
                traces,group=_v4_dspark_distributed_captures(target_capture_source,model,step,pool,
                    workspace_initial_capacity_bytes,_replica_context,prior_context=prior_context,
                    target_query_lens=target_query_lens,num_rejected=num_rejected)
    elif target_capture_source is not None:
        if _replica_context is None:
            traces, group = _v4_dspark_produced_captures(target_capture_source, model, step, pool,
                workspace_initial_capacity_bytes)
        else:
            traces, group = _v4_dspark_distributed_captures(target_capture_source, model, step, pool,
                workspace_initial_capacity_bytes, _replica_context,dummy_run=_dummy_run)
    else:
        traces = (tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
            if _replica_context is None else _replica_context.traces)
        group = ((ForwardCollectiveGroup(traces, namespace=step.step_id + ":tp") if tp > 1 else None)
            if _replica_context is None else _replica_context.tensor_groups[0] if tp > 1 else None)
    prior_nodes = tuple(set(trace._nodes) for trace in traces)
    prior_contracts = set(group._contracts) if group is not None else set()
    reduce = pool.collective_for("all_reduce") if group is not None else None
    gather = pool.collective_for("all_gather") if group is not None else None
    invocations,resolutions=(_dspark_routing_inputs(model,step,pool,expert_routing,rows)
        if _replica_context is None else (_replica_context.invocations,_replica_context.resolutions))
    dflash=(_v4_dspark_dummy_preflight(model,step,pool,target_capture_source,prior_context,_dflash_inputs,_replica_context,
        workspace_initial_capacity_bytes) if _dummy_run else _v4_dspark_dflash_preflight(model,step,pool,target_capture_source,prior_context,_dflash_inputs,input_ids=input_ids,
        context_positions=context_positions,context_slot_mappings=context_slot_mappings,query_slot_mappings=query_slot_mappings,
        cache_capacity_tokens=cache_capacity_tokens,source_max_model_len=source_max_model_len,max_batched_tokens=max_batched_tokens,
        compute_confidence=compute_confidence,replica_context=_replica_context))
    if prior_context is not None:
        # Routing and all local validation succeeded. Previous outputs are not
        # cache state and must not keep full-vocabulary logits live this round.
        del prior_context.owner.dspark_outputs
    outputs, cache_state, workspace_state, rope_state, topk_state = [], [], [], [], []
    request_states,input_buffers=[],[]
    for lane, trace in enumerate(traces):
        layout = local_shapes[lane]
        model_sp = _replica_context is not None and _replica_context.sequence_parallel
        sequence = (_V4SequenceContext(group, gather, pool.collective_for("reduce_scatterv"), work_rows, lane)
            if model_sp else None)
        workspace = (prior_context.workspaces[lane] if prior_context is not None else ForwardWorkspace(trace, "worker.workspace", alignment_bytes=256,
            initial_capacity_bytes=workspace_initial_capacity_bytes[lane], locked=False) if target_capture_source is None else
            target_capture_source.aux_workspaces_by_rank[trace.rank])
        target_ropes = {} if target_capture_source is None else target_capture_source.aux_rope_tables_by_rank[trace.rank]
        rope = (prior_context.ropes[lane] if prior_context is not None else target_ropes["swa"] if "swa" in target_ropes else trace.tensor("draft.rope-swa",
            (model.max_position_embeddings, model.qk_rope_head_dim), "fp32", persistent=True))
        topk = (prior_context.topk[lane] if prior_context is not None else
            trace.tensor("draft.unused-topk-buffer", (max_batched_tokens, model.index_topk), "int32", persistent=True))
        prepared_ports=None
        embedding_ids=input_ids
        if dflash is not None:
            from .traced_sampling import vllm030_v2_post_update_forward,dspark_prepare_inputs_forward,dspark_dummy_prepare_inputs_forward
            plan,arguments=dflash
            if _dummy_run:
                prep_args=arguments[lane]
                buffers=dspark_dummy_prepare_inputs_forward(trace,plan,**prep_args,_allocate_only=True)
                if prior_context is not None and prior_context.request_states:
                    request_states.append(prior_context.request_states[lane])
            else:
                sampler,post_args,prep_args=arguments[lane]
                request_state=vllm030_v2_post_update_forward(trace,sampler,**post_args)
                buffers=dspark_prepare_inputs_forward(trace,plan,request_state=request_state,sampler_output=sampler,**prep_args,_allocate_only=True)
                request_states.append(request_state)
            prepared_ports=dict(buffers.shared)
        captures = (tuple(trace.tensor(trace.unique_name(f"target.capture-{layer}"), (projection_rows, model.hidden_size), "bf16")
            for layer in captured_target_layers) if target_capture_source is None else
            target_capture_source.aux_hidden_states_by_rank[trace.rank])
        combined = SymbolicConcat(trace, "draft:capture-concat", pool.name)(*captures)
        if prepared_ports is not None and not _dummy_run:
            trace._edges.append(WorkloadEdge(trace._writers[request_state.num_computed.view.allocation_id],
                trace._writers[combined.view.allocation_id],'barrier'))
        if target_capture_source is not None or prior_context is not None:
            consumed = {edge.source for edge in trace._edges
                if edge.source in prior_nodes[lane] and edge.destination in prior_nodes[lane]}
            for terminal in sorted(prior_nodes[lane] - consumed):
                edge = WorkloadEdge(terminal, trace._writers[combined.view.allocation_id], "barrier")
                if edge not in trace._edges:
                    trace._edges.append(edge)
        main = SymbolicBlockFP8Linear(trace, "draft:main-proj", pool.name, 3 * model.hidden_size, model.hidden_size,
            source_profile="vllm_0_30_row_major_group128_fp32")(combined)[0]
        main = SymbolicRMSNorm(trace, "draft:main-norm", pool.name, model.hidden_size, model.rms_norm_eps)(main)
        if prepared_ports is not None:
            main=SymbolicCopy(trace,'draft:context-hidden-copy',pool.name)(main,prepared_ports['hidden_states'].narrow(0,0,projection_rows))
            before_prepare=set(trace._nodes)
            buffers=(dspark_dummy_prepare_inputs_forward(trace,plan,**{**prep_args,'prior_buffers':buffers}) if _dummy_run else
                dspark_prepare_inputs_forward(trace,plan,request_state=request_state,sampler_output=sampler,
                    **{**prep_args,'prior_buffers':buffers}))
            input_buffers.append(buffers)
            prepared_calls=tuple(node for key,node in trace._nodes.items() if key not in before_prepare)
            trace._edges.append(WorkloadEdge(trace._writers[main.view.allocation_id],prepared_calls[0].node_id,'barrier'))
            from .traced_sampling import dspark_attention_metadata_forward
            attention_ports=dspark_attention_metadata_forward(trace,buffers,plan,step_id=prep_args['step_id'],pool_name=pool.name)
            from .traced_sampling import dspark_input_buffer_values
            embedding_ids,query_position_values=dspark_input_buffer_values(buffers,physical_rows=work_rows,vocab_size=model.vocab_size)
            if embedding_ids[:rows]!=input_ids or query_position_values[:rows]!=positions:
                raise ValueError('DSpark prepared values differ from their current declared active query')
        context_pos = (trace.tensor(trace.unique_name("draft.context-positions"), (projection_rows,), "int64") if prepared_ports is None else
            prepared_ports['context_positions'].narrow(0,0,projection_rows))
        query_pos = (trace.tensor(trace.unique_name("draft.query-positions"), (rows,), "int64") if prepared_ports is None else
            prepared_ports['positions'].narrow(0,0,work_rows))
        networks, context_frontier = [], []
        for ordinal, layer in enumerate(model.dspark_layer_indices):
            prefix = f"draft-layer-{layer}"
            context = (_V4ExpertContext(tp, expert_size=config.expert_parallel_size,
                expert_rank=lane if config.enable_expert_parallel else 0) if _replica_context is None else
                _V4ExpertContext(tp, len(_replica_context.rows), _replica_context.expert_size,
                    _replica_context.rank * tp + lane if _replica_context.expert_size > 1 else 0,
                    _replica_context.data_groups[0][lane], _replica_context.rank, _replica_context.rows,
                    pool.collective_for("all_gatherv"), pool.collective_for("reduce_scatterv")))
            if model_sp:
                context = replace(context, data_group=_replica_context.expert_groups[0],
                    rows=tuple(-(-count // tp) for count in _replica_context.rows for _ in range(tp)),
                    sequence_parallel=True, valid_tokens=sum(_replica_context.rows),
                    local_valid_rows=max(0, min(sequence.shard_rows, work_rows - lane * sequence.shard_rows)))
            network = DeepseekV4DecoderLayer(trace, model, pool.name, prefix, group, reduce, rope, topk,
                layer_index=layer, q_lens=q_lens, kv_lens=tuple(c + q for c, q in zip(contexts, q_lens, strict=True)),
                request_phases=("decode",) * len(q_lens), slot_mapping=query_slot_mappings[ordinal],
                cache_capacity_tokens=cache_capacity_tokens, source_max_model_len=source_max_model_len,
                linear_source_profile="vllm_0_30_row_major_group128_fp32", source_use_deep_gemm=False,
                hardware=hardware, resolution=resolutions[ordinal], invocation_id=invocations[ordinal],
                workspace=workspace, source_max_batched_tokens=max_batched_tokens, expert_context=context,
                sequence_context=sequence,slot_tensor=None if prepared_ports is None else buffers.query_slots[plan['layer_group_indices'][ordinal]].narrow(0,0,rows),
                cache_tensors=(None if prior_context is None else
                    (("swa-cache", prior_context.caches[lane][ordinal]),)),dummy_run=_dummy_run)
            if ordinal == 0:
                network.pre = SymbolicMHCPipelineInput(trace, prefix + ":first-expanded-pre", pool.name, model,
                    source_use_deep_gemm=False, source_context="dspark")
            network.attn.attn = SymbolicDSparkNoncausalSWA(trace, prefix + ":noncausal-swa", pool.name, model,
                q_lens=q_lens, context_lens=contexts, cache=network.attn.swa_cache,
                context_slots=context_slot_mappings[ordinal], query_slots=query_slot_mappings[ordinal],
                prepared_metadata=None if prepared_ports is None else (
                    prepared_ports['query_start'].narrow(0,0,len(q_lens)+1),
                    prepared_ports['seq_lens'].narrow(0,0,len(q_lens)),
                    buffers.table_ports[plan['layer_group_indices'][ordinal]].narrow(0,0,len(q_lens)),
                    *attention_ports[plan['layer_group_indices'][ordinal]]),dummy_run=_dummy_run)
            if work_rows != rows:
                network.attn.insert = SymbolicV4KVInsert(trace, prefix + ":swa-insert", pool.name, model,
                    tp_size=tp, cache=network.attn.swa_cache, slot_mapping=query_slot_mappings[ordinal],
                    slot_tensor=buffers.query_slots[plan['layer_group_indices'][ordinal]].narrow(0,0,rows),
                    physical_query_rows=work_rows,dummy_run=_dummy_run)
            # SAME main_x for all three context projections. Keep the discarded
            # Q-lora projection work but normalize only the selected KV tail.
            full, _ = network.attn.inputs.fused_wqa_wkv(main)
            if prepared_ports is not None:
                trace._edges.append(WorkloadEdge(prepared_calls[-1].node_id,trace._writers[full.view.allocation_id],'barrier'))
            kv = full.narrow(1, model.q_lora_rank, model.head_dim)
            kv = SymbolicRMSNorm(trace, prefix + ":context-kv-norm", pool.name, model.head_dim, model.rms_norm_eps)(kv)
            if _dummy_run:
                context_frontier.append(trace._writers[kv.view.allocation_id]);networks.append(network)
                continue
            dummy = trace.tensor(trace.unique_name(prefix + ":context-dummy-q"), (projection_rows, layout["query_heads"], model.head_dim), "bf16")
            SymbolicFillZero(trace, prefix + ":context-dummy-zero", pool.name)(dummy)
            SymbolicV4KVInsert(trace, prefix + ":context-insert", pool.name, model, tp_size=tp,
                cache=network.attn.swa_cache, slot_mapping=insert_slots[ordinal],
                slot_tensor=None if prepared_ports is None else buffers.context_slots[plan['layer_group_indices'][ordinal]].narrow(0,0,projection_rows),
                masked_dspark_context=prior_context is not None)(dummy, kv, context_pos, rope)
            context_frontier.append(trace._writers[network.attn.swa_cache.view.allocation_id])
            networks.append(network)
        cache_state.append(tuple(network.attn.swa_cache for network in networks))
        workspace_state.append(workspace); rope_state.append(rope); topk_state.append(topk)
        ids = (trace.tensor(trace.unique_name("draft.query-input-ids"), (rows,), "int32") if prepared_ports is None else
            prepared_ports['input_ids'].narrow(0,0,work_rows))
        before_embedding = set(trace._nodes)
        embedding_name = "target-aliased:embedding" if target_capture_source is None else "embedding"
        if group is None:
            hidden = SymbolicEmbedding(trace, embedding_name, pool.name, model.hidden_size, "bf16", vocab_size=model.vocab_size)(ids)
        else:
            hidden = SymbolicVocabParallelEmbedding(trace, embedding_name, pool.name, model.hidden_size,
                model.vocab_size, group, reduce,token_ids=embedding_ids)(ids)
        for node_id in set(trace._nodes) - before_embedding:
            for producer in context_frontier:
                trace._edges.append(WorkloadEdge(producer, node_id, "barrier"))
        layer_ids = ids
        if sequence is not None:
            hidden = sequence.shard(trace, hidden, pool=pool.name, name="draft:sp-embedding")
            layer_ids = sequence.shard(trace, ids, pool=pool.name, name="draft:sp-input-ids")
        expanded = trace.tensor(trace.unique_name("draft.expanded-hidden"), (hidden.shape[0], model.hc_mult, model.hidden_size), "bf16")
        hidden = SymbolicCopy(trace, "draft:embedding-repeat", pool.name)(hidden.unsqueeze(1).expand(
            hidden.shape[0], model.hc_mult, model.hidden_size), expanded)
        residual = post_mix = res_mix = None
        for network in networks:
            hidden, residual, post_mix, res_mix = network(hidden, query_pos, layer_ids, post_mix, res_mix, residual)
        hidden = SymbolicMHC(trace, "draft:final-post", pool.name, model, hardware,
            variant="mhc_post", source_use_deep_gemm=False)(hidden, residual, post_mix, res_mix)
        if sequence is not None:
            hidden = sequence.gather(hidden, pool=pool.name, name="draft:sp-final-hidden-gather")
        head_hidden = SymbolicMHC(trace, "draft:hc-head", pool.name, model, hardware,
            variant="hc_head", source_use_deep_gemm=False)(hidden)
        # sample_from_anchor=True selects all query rows, but the source's
        # advanced indexing still materializes a fresh INT64-indexed gather.
        sample_hidden = SymbolicGather(trace, "draft:sample-hidden-select", pool.name)(head_hidden, rows,
            indices=tuple(range(rows)), indices_tensor=None if prepared_ports is None else
                prepared_ports['sample_indices'].narrow(0,0,rows))
        normed = SymbolicRMSNorm(trace, "draft:final-norm", pool.name, model.hidden_size, model.rms_norm_eps)(sample_hidden)
        head_name = "target-aliased:lm-head" if target_capture_source is None else "lm-head"
        if group is None:
            logits = SymbolicLinear(trace, head_name, pool.name, model.hidden_size,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)(normed)[0]
        else:
            logits = SymbolicTPLogitsProcessor(trace, head_name, pool.name, hidden_size=model.hidden_size,
                vocab_size=model.vocab_size, group=group, config=gather)(normed)
        batch, count = len(q_lens), model.dspark_block_size
        base = logits.view_as_shape(batch, count, model.vocab_size)
        anchor_indices = tuple(index * count for index in range(batch))
        prev = SymbolicGather(trace, "draft:anchor-select", pool.name)(ids, batch, indices=anchor_indices,
            indices_tensor=None if prepared_ports is None else prepared_ports['anchor_indices'].narrow(0,0,batch))
        tokens = trace.tensor(trace.unique_name("speculator.draft-tokens"), (batch, count), "int64")
        embeddings = []
        for index in range(count):
            op = SymbolicOperator(trace, f"draft:markov-{index}:embed", pool.name)
            node = op.node(WorkloadKind.GATHER, {"elements": batch * model.dspark_markov_rank,
                "rows": batch, "hidden": model.dspark_markov_rank}, "bf16", "dspark_markov_embedding",
                {"source_component": "vllm_0_30_dspark_torch_embedding", "index_bytes": prev.view.element_bytes,
                 "gather_factor": 1, "index_count": batch, "scattered": True, "materialized_output": True})
            weight = trace.parameter_view("draft:markov-embedding:weight", (model.vocab_size, model.dspark_markov_rank), "bf16")
            embedding = trace.call(node, (prev,), shape=(batch, model.dspark_markov_rank), dtype="bf16", parameters=(weight,))
            embeddings.append(embedding)
            bias = SymbolicLinear(trace, "draft:markov-bias", pool.name, model.dspark_markov_rank,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)(embedding)[0]
            current = base.narrow(1, index, 1).view_as_shape(batch, model.vocab_size)
            biased = SymbolicAdd(trace, f"draft:markov-{index}:bias-add", pool.name)(current, bias)
            op = SymbolicOperator(trace, f"draft:markov-{index}:argmax", pool.name)
            node = op.node(WorkloadKind.VECTOR, {"elements": batch * model.vocab_size, "rows": batch,
                "vocab": model.vocab_size}, "bf16", "dspark_direct_argmax",
                {"source_component": "vllm_0_30_dspark_greedy_full_vocab", "argmax_output_dtype": "int64",
                 "probabilistic": False, "topk_truncation": False, "values_computed": False})
            prev = trace.call(node, (biased,), shape=(batch,), dtype="int64")
            SymbolicCopy(trace, f"draft:markov-{index}:store", pool.name)(prev.unsqueeze(1), tokens.narrow(1, index, 1))
        confidence = None
        if compute_confidence:
            packed = SymbolicConcat(trace, "draft:confidence-markov-stack", pool.name)(*embeddings)
            trace._edges.append(WorkloadEdge(trace._writers[prev.view.allocation_id], trace._writers[packed.view.allocation_id], "barrier"))
            packed = packed.view_as_shape(rows, model.dspark_markov_rank)
            joined = SymbolicConcat(trace, "draft:confidence-concat", pool.name)(sample_hidden, packed)
            fp32 = SymbolicFloat32Cast(trace, "draft:confidence-float", pool.name)(joined)
            prediction = SymbolicLinear(trace, "draft:confidence-proj", pool.name,
                model.hidden_size + model.dspark_markov_rank, 1)(fp32)[0]
            op = SymbolicOperator(trace, "draft:confidence-sigmoid", pool.name)
            node = op.node(WorkloadKind.ELEMENTWISE, {"elements": rows}, "fp32", "elementwise",
                {"operation": "sigmoid", "reads": 1, "writes": 1, "source_component": "vllm_0_30_dspark_confidence"})
            confidence = trace.call(node, (prediction,), shape=(batch, count), dtype="fp32")
        outputs.append({"head_hidden": head_hidden, "base_logits": logits, "draft_tokens": tokens, "confidence": confidence})
        for node_id, node in tuple(trace._nodes.items()):
            if node_id in prior_nodes[lane]:
                continue
            trace._nodes[node_id] = node.with_attributes({**node.attributes, "source_dspark_standalone": target_capture_source is None and _replica_context is None,
                "source_dspark_data_parallel": _replica_context is not None,
                "source_dspark_bound_target": target_capture_source is not None,
                "model_execution_version": "vllm_0_30_0", "model_source_profile": ("vllm_0_30_v4_dspark_distributed" if _replica_context is not None else
                    "vllm_0_30_v4_dspark_standalone" if target_capture_source is None else "vllm_0_30_v4_dspark_same_tp_target_bound"),
                "source_projection_only": True, "target_capture_layers": captured_target_layers,
                "draft_layer_indices": model.dspark_layer_indices,"source_dummy_run":_dummy_run})
    if group is not None:
        for node_id, node in tuple(group._contracts.items()):
            if node_id in prior_contracts:
                continue
            group._contracts[node_id] = node.with_attributes({**node.attributes, "source_dspark_standalone": target_capture_source is None and _replica_context is None,
                "source_dspark_data_parallel": _replica_context is not None,
                "source_dspark_bound_target": target_capture_source is not None,
                "model_execution_version": "vllm_0_30_0", "model_source_profile": ("vllm_0_30_v4_dspark_distributed" if _replica_context is not None else
                    "vllm_0_30_v4_dspark_standalone" if target_capture_source is None else "vllm_0_30_v4_dspark_same_tp_target_bound"),
                "source_projection_only": True})
    result = ((traces[0] if group is None else group)
        if target_capture_source is None or _replica_context is not None else target_capture_source)
    if (target_capture_source is None or '_v4_target_source_state' in vars(target_capture_source)
            or '_v4_distributed_target_state' in vars(target_capture_source)):
        result.dspark_cache_state = _DSparkCacheState(result, model.config_sha256, config, pool.name, pool.ranks,
            step, (step.step_id,) if prior_context is None else (*prior_context.step_ids, step.step_id),
            context_slot_mappings, query_slot_mappings, cache_capacity_tokens, max_batched_tokens,
            traces, group, tuple(cache_state), tuple(workspace_state), tuple(rope_state), tuple(topk_state),
            () if _replica_context is None else (_replica_context.rank,
                _replica_context.expert_size, _replica_context.sequence_parallel),tuple(request_states),tuple(input_buffers),_dummy_run)
        if prior_context is not None and prior_context.owner is not result:
            del prior_context.owner.dspark_cache_state
    result.dspark_outputs = tuple(outputs)
    result.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "standalone_dspark_three_layer_source": target_capture_source is None and _replica_context is None, "target_forward_executed": target_capture_source is not None,
        "input_capture_scope": ("declared target capture ports; no numerical/observed values" if target_capture_source is None else
            "produced complete symbolic target ports with actual PP relay; no numerical/observed values"),
        "target_weight_alias_scope": ("declared external target embedding/head; combined residency unbound"
            if target_capture_source is None else
            "same readonly target LM-head parameter reused; target embedding/module shapes declared; combined residency unbound"),
        "sample_from_anchor": True, "query_rows_per_request": model.dspark_block_size,
        "persistent_cache_continuation": prior_context is not None, "target_context_projection_rows": projection_rows,
        "accepted_context_lengths": contexts, "context_acceptance_scope": "explicit caller declarations; no numerical target acceptance",
        "compute_confidence": compute_confidence, "source_model_sequence_parallel": model_sp,
        "post_update_and_DFlash_input_source_bound":dflash is not None and not _dummy_run,
        "dummy_prepare_source_bound":dflash is not None and _dummy_run,
        "source_dummy_run":_dummy_run,
        "active_query_rows": rows, "physical_query_rows": work_rows,
        "padding_input_values": "InputBuffers constructor zeros plus retained prepare writes" if dflash is not None else "no DP padding",
        "source_moe_sequence_parallel": model_sp, "normal_capacity_accepted": False,
        "actual_provider_bound": False, "complete_priced_request": False, "Serving_proposer_bound": False}
    if target_capture_source is not None:
        if _replica_context is None:
            result._dspark_capture_consumed = True
        result.source_scope["same_TP_target_draft_composed"] = True
        result.source_scope["target_source_PP"] = target_capture_source.aux_target_config.pipeline_parallel_size
    return result


@dataclass(frozen=True)
class _DSparkDistributedCacheState:
    owner: object
    config: object
    states: tuple
    tensor_groups: tuple
    data_groups: tuple
    expert_groups: tuple



def trace_deepseek_v4_dspark_data_parallel_forward(model, step, pool, hardware, expert_routing, *, input_ids,
        context_slot_mappings_by_replica, query_slot_mappings_by_replica, cache_capacity_tokens_by_replica,
        source_uniform_num_requests, source_max_model_len, max_batched_tokens, workspace_initial_capacity_bytes,
        source_use_deep_gemm, compute_confidence, target_capture_source=None, prior_context=None,
        target_query_lens_by_replica=None, num_rejected_by_replica=None, _dflash_inputs_by_replica=None):
    """Complete modular draft TP/DP/global-EP source, all-active full blocks.

    The real DFlash DP synchronizer declares a uniform physical query budget
    across replicas. Unequal active batches read their persistent prepared
    input tail; only actual rows enter attention/KV insert and the final head.
    Idle/dummy and CG remain unsupported.
    Context work remains per-DP full and
    replicated across TP. DP EP-off shards each expert's I by TP*DP; global
    EP uses complete owned experts and model-SP at TP>1. No target/captured
    numerical values, proposer, native Scheduler, normal memory or pricing.
    Optional source cache revisions retain each replica's exact TP/DP/EP groups;
    declared target rejection feedback is validated for ALL replicas first.
    """
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.parallelism.tp import validate_dspark_data_parallelism
    from src.serving.schedule.schedule_step import ScheduleStep, schedule_step_sha256
    from .runtime_inputs import ExpertRoutingInput
    if not isinstance(model, ModelConfig) or not isinstance(step, ScheduleStep) or not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("DSpark DP requires typed model, schedule and explicit routing input")
    config = pool.parallel_plan.config
    if _dflash_inputs_by_replica is not None and (
            type(_dflash_inputs_by_replica) is not tuple
            or len(_dflash_inputs_by_replica) != config.data_parallel_size
            or any(type(inputs) is not dict for inputs in _dflash_inputs_by_replica)):
        raise ValueError("DSpark DP input source requires one complete declaration per physical replica")
    if prior_context is None:
        if target_query_lens_by_replica is not None or num_rejected_by_replica is not None:
            raise ValueError("DSpark DP acceptance declarations require their preceding distributed cache revision")
    elif (type(prior_context) is not _DSparkDistributedCacheState
            or 'dspark_distributed_cache_state' not in vars(prior_context.owner)
            or prior_context.owner.dspark_distributed_cache_state is not prior_context
            or prior_context.config != config
            or target_capture_source is None and tuple(trace.rank for trace in prior_context.owner.traces)!=pool.ranks
            or target_capture_source is not None and target_capture_source is not prior_context.owner):
        raise ValueError("DSpark DP continuation requires its current source-owned topology and cache revision")
    validate_dspark_data_parallelism(model, config)
    tp, dp = config.tensor_parallel_size, config.data_parallel_size
    # These source groups have no implicit NIC routing. Resolve every local
    # requirement before registering a group or retiring any prior output.
    pool.collective_for('all_gatherv')
    pool.collective_for('reduce_scatterv')
    if tp>1:
        for operation in ('all_reduce','all_gather'):
            if pool.collective_for(operation).link=='roce':
                raise NotImplementedError("DSpark DP source TP groups require an explicit NIC mapping for RoCE")
    if (pool.hardware_id != "nvidia_h200" or hardware.hardware_id != pool.hardware_id or pool.role != "colocated"
            or len(pool.ranks) != config.worker_count or pool.parallel_plan.num_layers != 3
            or pool.parallel_plan.num_experts != model.num_experts or step.pool_name != pool.name):
        raise ValueError("DSpark DP needs its actual H200 draft ranks and three-layer plan")
    if (type(source_uniform_num_requests) is not int or source_uniform_num_requests < 1
            or type(input_ids) is not tuple or len(input_ids) != step.token_count
            or len({request.request_id for request in step.requests}) != len(step.requests)
            or type(workspace_initial_capacity_bytes) is not tuple or len(workspace_initial_capacity_bytes) != tp * dp
            or any(type(size) is not int or size < 0 for size in workspace_initial_capacity_bytes)):
        raise ValueError("DSpark DP needs distinct requests, explicit uniform budget, original IDs and workspace entries")
    for values in (context_slot_mappings_by_replica, query_slot_mappings_by_replica, cache_capacity_tokens_by_replica):
        if type(values) is not tuple or len(values) != dp:
            raise ValueError("DSpark DP needs exactly one complete context/query/cache declaration per replica")
    requests, token_ids = [[] for _ in range(dp)], [[] for _ in range(dp)]
    offset = 0
    for request in step.requests:
        replica = request.data_parallel_rank
        if not 0 <= replica < dp:
            raise ValueError("DSpark DP request owner is outside the actual pool")
        requests[replica].append(request)
        token_ids[replica].extend(input_ids[offset:offset + request.scheduled_tokens])
        offset += request.scheduled_tokens
    idle=tuple(not values for values in requests)
    actual_rows=tuple(sum(request.scheduled_tokens for request in values) for values in requests)
    if any(idle):
        if (_dflash_inputs_by_replica is None or target_capture_source is None
                or 'aux_dummy_replicas' not in vars(target_capture_source) or target_capture_source.aux_dummy_replicas!=idle):
            raise NotImplementedError('DSpark idle source requires each actual retained dummy target and its null-table plan')
        from src.serving.schedule.schedule_step import ScheduledRequest
        for owner in range(dp):
            if idle[owner]:
                requests[owner]=[ScheduledRequest('_dummy_req_0','decode',0,5,computed_tokens=1,data_parallel_rank=owner)]
                anchor=_dspark_dummy_anchor(None if prior_context is None else prior_context.states[owner],model)
                token_ids[owner]=[anchor,*((model.dspark_noise_token_id,)*4)]
    if max(map(len, requests)) != source_uniform_num_requests:
        raise NotImplementedError("DSpark DP uniform full-block source rejects a stale DP budget")
    if any(request.scheduled_tokens != model.dspark_block_size for request in step.requests):
        raise ValueError("DSpark DP uniform budget requires complete declared query blocks")
    rows = tuple(sum(request.scheduled_tokens for request in values) for values in requests)
    unequal = len(set(rows)) > 1
    if unequal and (_dflash_inputs_by_replica is None or target_capture_source is None):
        raise NotImplementedError("DSpark unequal DP rows require actual prepared target-bound input buffers")
    work_rows = (source_uniform_num_requests * model.dspark_block_size,) * dp if unequal else rows
    if type(max_batched_tokens) is not int or max(work_rows) > max_batched_tokens:
        raise ValueError("DSpark uniform physical DP rows exceed the worker token bound")
    if prior_context is not None and (len(prior_context.states) != dp
            or type(target_query_lens_by_replica) is not tuple or len(target_query_lens_by_replica) != dp
            or type(num_rejected_by_replica) is not tuple or len(num_rejected_by_replica) != dp):
        raise ValueError("DSpark DP continuation needs explicit target verification feedback for every original replica")
    model_sp = config.enable_expert_parallel and tp > 1
    # Validate complete query declarations before mutating an existing target.
    if source_use_deep_gemm is not False or type(compute_confidence) is not bool:
        raise NotImplementedError("DSpark DP needs its explicit TileLang branch and boolean confidence choice")
    if any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_ids):
        raise ValueError("DSpark DP requires exact known vocabulary IDs")
    for replica in range(dp):
        values = requests[replica]
        context_rows = sum(request.computed_tokens for request in values)
        capacity = cache_capacity_tokens_by_replica[replica]
        if (any(request.phase != "decode" or request.computed_tokens < 1 or request.mtp_proposer_mode != "disabled" for request in values)
                or type(capacity) is not int or capacity < max(128, context_rows + rows[replica]) or capacity % 8
                or type(source_max_model_len) is not int or not max(request.computed_tokens + request.scheduled_tokens
                    for request in values) <= source_max_model_len <= model.max_position_embeddings
                or type(max_batched_tokens) is not int or max_batched_tokens < rows[replica]
                or prior_context is None and max_batched_tokens < context_rows):
            raise ValueError("DSpark DP complete context/query bounds must be valid before source composition")
        for index in range(len(values)):
            block = model.dspark_block_size
            if tuple(token_ids[replica][index * block + 1:(index + 1) * block]) != (model.dspark_noise_token_id,) * (block - 1):
                raise ValueError("DSpark DP requires each complete anchor-plus-noise query block")
        context_maps, query_maps = context_slot_mappings_by_replica[replica], query_slot_mappings_by_replica[replica]
        for maps, size in ((context_maps, context_rows), (query_maps, rows[replica])):
            if (type(maps) is not tuple or len(maps) != 3 or any(type(slots) is not tuple or len(slots) != size
                    or any(type(slot) is not int or (slot!=-1 if idle[replica] else not 0<=slot<capacity) for slot in slots) for slots in maps)):
                raise ValueError("DSpark DP needs all three explicit valid per-layer cache maps")
        if not idle[replica] and any(len(set(old + new)) != context_rows + rows[replica] for old, new in zip(context_maps, query_maps, strict=True)):
            raise ValueError("DSpark DP fresh context/query slots must be distinct")
    if target_capture_source is None:
        if prior_context is None:
            traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
            tensor = tuple(ForwardCollectiveGroup(traces[replica * tp:(replica + 1) * tp],
                namespace=f"{step.step_id}:dp{replica}:tp") for replica in range(dp)) if tp > 1 else ()
            data = tuple(ForwardCollectiveGroup(tuple(traces[replica * tp + lane] for replica in range(dp)),
                namespace=f"{step.step_id}:lane{lane}:dp") for lane in range(tp))
            expert = (ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:ep"),) if model_sp else ()
            joined = ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:joined", subgroups=(*tensor, *data, *expert))
        else:
            joined = prior_context.owner
            traces = joined.traces
            tensor, data, expert = prior_context.tensor_groups, prior_context.data_groups, prior_context.expert_groups
        target_scope = None
    else:
        target_config, final_ranks = _v4_dspark_distributed_capture_owner(target_capture_source, model, tensor_size=tp,
            data_size=dp, expert_size=config.expert_parallel_size, pool_name=pool.name)
        if pool.ranks != final_ranks:
            raise ValueError("DSpark DP pool must match every actual final-stage target rank in original order")
        joined = target_capture_source
        traces = tuple(next(trace for trace in joined.traces if trace.rank == rank) for rank in pool.ranks)
        tensor = tuple(joined.aux_tensor_groups[replica][-1] for replica in range(dp)) if tp > 1 else ()
        data = joined.aux_data_groups[-1]
        referenced = (*tensor, *data)
        if any(type(group) is not ForwardCollectiveGroup or not any(group is existing for existing in joined.subgroups)
                for group in referenced):
            raise ValueError("DSpark DP binding requires the actual target's registered TP and DP source groups")
        if (any(group.participants != pool.ranks[replica * tp:(replica + 1) * tp] for replica, group in enumerate(tensor))
                or any(group.participants != pool.ranks[lane::tp] for lane, group in enumerate(data))):
            raise ValueError("DSpark DP target group participants differ from actual final-stage placement")
        if model_sp and target_config.pipeline_parallel_size == 1:
            expert = joined.aux_expert_groups
            if (len(expert) != 1 or type(expert[0]) is not ForwardCollectiveGroup or expert[0].participants != pool.ranks
                    or not any(expert[0] is existing for existing in joined.subgroups)):
                raise ValueError("DSpark DP PP1 binding requires its actual target global EP group")
        elif prior_context is not None:
            expert=prior_context.expert_groups
            if (model_sp and (len(expert)!=1 or expert[0].participants!=pool.ranks
                    or not any(expert[0] is existing for existing in joined.subgroups))):
                raise ValueError("DSpark DP incremental binding lost the original final-stage EP group")
        else:
            expert = (ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:dspark-last-stage:ep"),) if model_sp else ()
        target_scope = dict(joined.source_scope)
    old_group_contracts = tuple(set(group._contracts) for group in (*data, *expert))
    physical_rows = sum(-(-count // tp) * tp for count in work_rows) if model_sp else sum(work_rows)
    invocations,resolutions=_dspark_routing_inputs(model,step,pool,expert_routing,physical_rows)
    local_config = replace(config, data_parallel_size=1, enable_expert_parallel=False)
    local_plan = build_parallel_plan(local_config, num_layers=3, num_experts=model.num_experts)
    outputs, states, prepared = [], [], []
    for replica in range(dp):
        local_traces = traces[replica * tp:(replica + 1) * tp]
        context = _V4ReplicaContext(local_traces, (tensor[replica],) if tp > 1 else (), (data,), work_rows, replica,
            resolutions, invocations, config.expert_parallel_size, model_sp, expert)
        local_pool = replace(pool, ranks=tuple(trace.rank for trace in local_traces), parallel_plan=local_plan)
        local_step = replace(step, requests=tuple(requests[replica]), finished_request_ids=tuple(
            key for key in step.finished_request_ids if key in {r.request_id for r in requests[replica]}), preempted_request_ids=())
        if idle[replica]:
            context_positions=(0,)*6
            if prior_context is not None:
                _dspark_dummy_continuation(prior_context.states[replica],model,local_step,local_pool,
                    workspace_initial_capacity_bytes[replica*tp:(replica+1)*tp],cache_capacity_tokens_by_replica[replica],max_batched_tokens,context)
            _v4_dspark_dummy_preflight(model,local_step,local_pool,target_capture_source,
                None if prior_context is None else prior_context.states[replica],_dflash_inputs_by_replica[replica],context,
                workspace_initial_capacity_bytes[replica*tp:(replica+1)*tp])
            prepared.append((local_pool,local_step,context,context_positions))
            continue
        if prior_context is None:
            context_positions = tuple(position for request in requests[replica] for position in range(request.computed_tokens))
        else:
            context_positions, _ = _dspark_incremental_context(prior_context.states[replica], model, local_step, local_pool,
                workspace_initial_capacity_bytes[replica * tp:(replica + 1) * tp],
                context_slot_mappings_by_replica[replica], query_slot_mappings_by_replica[replica],
                cache_capacity_tokens_by_replica[replica], max_batched_tokens,
                target_query_lens_by_replica[replica], num_rejected_by_replica[replica], context,
                _dspark_admission_ids(None if _dflash_inputs_by_replica is None else _dflash_inputs_by_replica[replica]))
            if sum(target_query_lens_by_replica[replica]) > max_batched_tokens:
                raise ValueError("DSpark DP target verification rows exceed the worker bound")
        if target_capture_source is not None:
            _v4_dspark_distributed_captures(target_capture_source,model,local_step,local_pool,
                workspace_initial_capacity_bytes[replica*tp:(replica+1)*tp],context,
                prior_context=None if prior_context is None else prior_context.states[replica],
                target_query_lens=None if prior_context is None else target_query_lens_by_replica[replica],
                num_rejected=None if prior_context is None else num_rejected_by_replica[replica])
        _v4_dspark_dflash_preflight(model,local_step,local_pool,target_capture_source,
            None if prior_context is None else prior_context.states[replica],
            None if _dflash_inputs_by_replica is None else _dflash_inputs_by_replica[replica],
            input_ids=tuple(token_ids[replica]),context_positions=context_positions,
            context_slot_mappings=context_slot_mappings_by_replica[replica],
            query_slot_mappings=query_slot_mappings_by_replica[replica],
            cache_capacity_tokens=cache_capacity_tokens_by_replica[replica],
            source_max_model_len=source_max_model_len,max_batched_tokens=max_batched_tokens,
            compute_confidence=compute_confidence,replica_context=context)
        prepared.append((local_pool, local_step, context, context_positions))
    if target_capture_source is not None:
        for group in expert:
            if not any(group is existing for existing in joined.subgroups):
                if any(group.namespace == existing.namespace for existing in joined.subgroups):
                    raise ValueError("DSpark last-stage EP source group namespace collides")
                joined.subgroups = (*joined.subgroups, group)
    if prior_context is not None:
        # All replicas have passed preflight. Their local output owners are
        # released at their own forward entry; discard the joined alias now.
        if 'dspark_outputs_by_replica' in vars(joined):del joined.dspark_outputs_by_replica
    for replica, (local_pool, local_step, context, context_positions) in enumerate(prepared):
        source = trace_deepseek_v4_dspark_forward(model, local_step, local_pool, hardware, expert_routing,
            input_ids=tuple(token_ids[replica]), positions=tuple(position for request in requests[replica]
                for position in range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens)),
            context_positions=context_positions,
            context_slot_mappings=context_slot_mappings_by_replica[replica], query_slot_mappings=query_slot_mappings_by_replica[replica],
            captured_target_layers=model.dspark_target_layer_ids, cache_capacity_tokens=cache_capacity_tokens_by_replica[replica],
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes[replica * tp:(replica + 1) * tp],
            source_use_deep_gemm=source_use_deep_gemm, compute_confidence=compute_confidence, _replica_context=context,
            target_capture_source=target_capture_source,
            prior_context=None if prior_context is None else prior_context.states[replica],
            target_query_lens=None if prior_context is None else target_query_lens_by_replica[replica],
            num_rejected=None if prior_context is None else num_rejected_by_replica[replica],
            _dflash_inputs=None if _dflash_inputs_by_replica is None else _dflash_inputs_by_replica[replica],
            _physical_query_rows=work_rows[replica],_dummy_run=idle[replica])
        outputs.append(source.dspark_outputs)
        if target_capture_source is None or '_v4_distributed_target_state' in vars(target_capture_source):
            states.append(source.dspark_cache_state)
    for group, old_contracts in zip((*data, *expert), old_group_contracts, strict=True):
        for name, node in tuple(group._contracts.items()):
            if name in old_contracts:
                continue
            group._contracts[name] = node.with_attributes({**node.attributes,
                "source_dspark_standalone": False, "source_dspark_data_parallel": True,
                "source_projection_only": True, "model_execution_version": "vllm_0_30_0"})
    if target_capture_source is None or '_v4_distributed_target_state' in vars(target_capture_source):
        joined.dspark_distributed_cache_state = _DSparkDistributedCacheState(joined, config, tuple(states), tensor, data, expert)
    joined.dspark_outputs_by_replica = tuple(outputs)
    joined.source_scope = {"source_projection_only": True, "model_execution_version": "vllm_0_30_0",
        "source_uniform_num_requests": source_uniform_num_requests, "source_query_rows_by_replica": actual_rows,
        "source_dummy_replicas":idle,"source_query_rows_including_dummy":rows,
        "source_physical_query_rows_by_replica": work_rows,
        "source_expert_physical_rows": physical_rows, "source_model_sequence_parallel": model_sp,
        "source_moe_sequence_parallel": model_sp, "context_KV_scope": "active own DP context; dummy projections only, no KV stores",
        "persistent_cache_continuation": prior_context is not None,
        "post_update_and_DFlash_input_source_bound": _dflash_inputs_by_replica is not None,
        "target_context_projection_rows_by_replica": tuple(sum(target_query_lens_by_replica[index])
            if prior_context is not None else 6 if idle[index] else sum(r.computed_tokens for r in requests[index]) for index in range(dp)),
        "context_acceptance_scope": "explicit caller declarations; no numerical target rejection sampler",
        "query_budget_scope": "full blocks and native dummy replicas with constructor or retained request state; uniform physical DP budget; no CG",
        "post_update_bound_by_replica":tuple(not value for value in idle),
        "padding_input_values": "InputBuffers constructor zeros plus retained prepare writes" if _dflash_inputs_by_replica is not None else "no DP padding",
        "captured_target_scope": ("external declared ports; DP target capture composition unbound" if target_capture_source is None else
            "actual target per-replica capture producers with explicit SP-prefix or PP-worker-copy provenance"),
        "complete_priced_request": False, "normal_capacity_accepted": False, "actual_provider_bound": False,
        "Serving_proposer_bound": False, "native_Scheduler_executed": False}
    if target_capture_source is not None:
        joined._dspark_capture_consumed = True
        joined.source_scope.update({"target_forward_executed": True, "same_TP_target_draft_composed": True,
            "target_source_PP": target_capture_source.aux_target_config.pipeline_parallel_size,
            "target_model_sequence_parallel": target_scope["model_sequence_parallel"],
            "target_query_rows_by_replica": target_scope["source_data_parallel_rows"],
            "target_expert_physical_rows": target_scope["source_expert_physical_rows"],
            "draft_expert_physical_rows": physical_rows, "draft_model_sequence_parallel": model_sp})
    return joined


class Qwen3Attention:
    qk_norm = True

    def __init__(self, trace, model, pool, prefix, rope_cache, kv_cache, q_lens, kv_lens, slot_mapping,
                 *, profile_no_metadata=False, group=None, collective=None, rope_program=ROPE_PROGRAM_FUSED,
                 hardware_id="nvidia_h200", collective_inplace=False,
                 model_execution_version="vllm_0_30_0", legacy_dummy_run=False):
        tp_size = 1 if group is None else len(group.traces)
        if model.num_attention_heads % tp_size:
            raise ValueError("query head count must divide the TP group")
        heads = model.num_attention_heads // tp_size
        kv_heads = model.kv_heads_per_tensor_parallel_rank(tp_size)
        self.head_dim = model.head_dim
        self.q_size = heads * model.head_dim
        self.kv_size = kv_heads * model.head_dim
        self.qkv_proj = SymbolicLinear(trace, prefix + ":qkv", pool,
            model.hidden_size, self.q_size + 2 * self.kv_size, bias=model.attention_bias)
        if self.qk_norm:
            norm_program = ("vllm_ascend_cann" if hardware_id == "ascend_910b" else
                "vllm_c_cuda_0_11" if model_execution_version == "vllm_0_11_0" else "vllm_c_cuda")
            self.q_norm = SymbolicRMSNorm(trace, prefix + ":q-norm", pool, model.head_dim, model.rms_norm_eps,
                storage_program=norm_program, model_execution_version=model_execution_version)
            self.k_norm = SymbolicRMSNorm(trace, prefix + ":k-norm", pool, model.head_dim, model.rms_norm_eps,
                storage_program=norm_program, model_execution_version=model_execution_version)
        self.rotary_emb = SymbolicRotary(trace, prefix + ":rotary", pool,
            heads, kv_heads, model.head_dim, rope_cache, program=rope_program,
            rotary_dim=model.rotary_dim, model_execution_version=model_execution_version,
            operand_semantics="aclnn_materialized_operands" if hardware_id == "ascend_910b" else "aten_tensor_iterator")
        self.attn = SymbolicAttention(trace, prefix + ":attention", pool,
            heads, kv_heads, model.head_dim, kv_cache, q_lens, kv_lens, slot_mapping,
            profile_no_metadata=profile_no_metadata, model_execution_version=model_execution_version,
            legacy_dummy_run=legacy_dummy_run)
        self.o_proj = SymbolicLinear(trace, prefix + ":o-proj", pool, self.q_size, model.hidden_size)
        if group is not None:
            self.o_proj = SymbolicRowParallelLinear(self.o_proj, group, collective, inplace=collective_inplace)

    def __call__(self, positions, hidden_states):
        # Qwen3Attention.forward; only Tensor.view's spelling is adapted.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view_as_shape(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view_as_shape(q.shape)
        k_by_head = k.view_as_shape(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view_as_shape(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaAttention(Qwen3Attention):
    """vLLM LlamaAttention.forward, sharing QKV/RoPE/cache primitives."""

    qk_norm = False

    def __call__(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer:
    attention_type = Qwen3Attention
    mlp_type = QwenMLP

    def __init__(self, trace, model, pool, prefix, rope_cache, kv_cache, q_lens, kv_lens, slot_mapping,
                 *, profile_no_metadata=False, group=None, collective=None, mlp_factory=None,
                 rope_program=ROPE_PROGRAM_FUSED, hardware_id="nvidia_h200", ffn_factory=None, collective_inplace=False,
                 model_execution_version="vllm_0_30_0", legacy_dummy_run=False):
        self._source_prefix = prefix
        norm_program = ("vllm_ascend_cann" if hardware_id == "ascend_910b" else
            "vllm_c_cuda_0_11" if model_execution_version == "vllm_0_11_0" else "vllm_c_cuda")
        self.input_layernorm = SymbolicRMSNorm(trace, prefix + ":input-norm", pool, model.hidden_size, model.rms_norm_eps,
            storage_program=norm_program, model_execution_version=model_execution_version)
        self.self_attn = self.attention_type(trace, model, pool, prefix, rope_cache, kv_cache, q_lens, kv_lens, slot_mapping,
            profile_no_metadata=profile_no_metadata, group=group, collective=collective, rope_program=rope_program,
            hardware_id=hardware_id, collective_inplace=collective_inplace,
            model_execution_version=model_execution_version, legacy_dummy_run=legacy_dummy_run)
        self.remote_ffn = None
        if ffn_factory is None:
            self.post_attention_layernorm = SymbolicRMSNorm(trace, prefix + ":post-attn-norm", pool, model.hidden_size, model.rms_norm_eps,
                storage_program=norm_program, model_execution_version=model_execution_version)
            factory = self.mlp_type if mlp_factory is None else mlp_factory
            self.mlp = factory(trace, model, pool, prefix, group=group, collective=collective,
                **({"model_execution_version": model_execution_version} if factory in (QwenMLP, LlamaMLP) else {}),
                **({"collective_inplace": True} if collective_inplace else {}))
        else:
            self.remote_ffn = ffn_factory(trace, model, pool, prefix,
                mlp_type=self.mlp_type if mlp_factory is None else mlp_factory)

    def __call__(self, positions, hidden_states, residual):
        # Qwen3DecoderLayer.forward, retaining source alias/caller lifetimes.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        if self.remote_ffn is not None:
            return self.remote_ffn(hidden_states, residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaDecoderLayer(Qwen3DecoderLayer):
    attention_type = LlamaAttention
    mlp_type = LlamaMLP


class RemoteFFN:
    """Existing AF placement around the shared source norm/MLP modules."""

    def __init__(self, trace, model, pool, prefix, *, mlp_type, attention_pool, ffn_pool, transport, group):
        lane = attention_pool.ranks.index(trace.rank)
        destination = transport.pool_traces[ffn_pool.name][lane]
        self.transport, self.attention_pool, self.ffn_pool, self.prefix = transport, pool, ffn_pool.name, prefix
        self.invocations = 0
        self.norm = SymbolicRMSNorm(destination, prefix + ":post-attn-norm", ffn_pool.name,
            model.hidden_size, model.rms_norm_eps, storage_program="vllm_c_cuda")
        self.mlp = mlp_type(destination, model, ffn_pool.name, prefix, group=group,
            collective=None if group is None else ffn_pool.collective_for("all_reduce"))

    def __call__(self, hidden_states, residual):
        boundary = self.prefix if self.invocations == 0 else f"{self.prefix}:repeat-{self.invocations}"
        self.invocations += 1
        self.transport.end_local_capture(hidden_states.trace,
            {"hidden_states": hidden_states, "residual": residual})
        hidden_states, residual = self.transport.exchange(hidden_states, residual,
            source_pool=self.attention_pool, destination_pool=self.ffn_pool, boundary=boundary + ":to-ffn")
        self.transport.begin_local_capture(hidden_states.trace,
            {"hidden_states": hidden_states, "residual": residual}, boundary=boundary + ":ffn")
        hidden_states, residual = self.norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        self.transport.end_local_capture(hidden_states.trace,
            {"hidden_states": hidden_states, "residual": residual})
        hidden_states, residual = self.transport.exchange(hidden_states, residual, source_pool=self.ffn_pool,
            destination_pool=self.attention_pool, boundary=boundary + ":to-attention")
        self.transport.begin_local_capture(hidden_states.trace,
            {"hidden_states": hidden_states, "residual": residual}, boundary=boundary + ":attention")
        return hidden_states, residual


def _dense_final_norm(trace, model, pool, hardware_id, *, model_execution_version=None):
    return SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size, model.rms_norm_eps,
        storage_program=("vllm_ascend_cann" if hardware_id == "ascend_910b" else
            "vllm_c_cuda_0_11" if model_execution_version == "vllm_0_11_0" else "vllm_c_cuda"),
        model_execution_version=model_execution_version)


def _ascend018_interleaved_rope_constructor_views(trace, cache):
    """Retain the plugin's one expanded backing, not two extra RoPE inputs.

    AscendRotaryEmbedding.__init__ records module-global cos/sin views via
    _record_cos_and_sin_cache_interleaved. Its repeat creates one allocation;
    chunk/squeeze retain aliases. Ordinary GQA rope_forward_oot still consumes
    the original cos_sin_cache, so these constructor buffers add no read ports.
    They are persistent model buffers, not parameter or activation payloads.
    """
    trace._check_tensor(cache)
    if (cache.dtype != "bf16" or len(cache.shape) != 2 or not cache.is_contiguous
            or cache.shape[1] % 2):
        raise ValueError("Ascend0.18 interleaved constructor requires dense BF16 even-width RoPE cache")
    positions, width = cache.shape
    backing = trace.tensor("model.rope-interleaved-cache", (positions, 2, width), cache.dtype, persistent=True)
    strides = (backing.view.strides_bytes[0], backing.view.strides_bytes[2])
    cos = trace.alias(backing, "model.rope-cos-cache", shape=cache.shape, strides_bytes=strides)
    sin = trace.alias(backing, "model.rope-sin-cache", shape=cache.shape, strides_bytes=strides,
        offset_bytes=width * backing.view.element_bytes)
    return cos, sin


class Qwen3Model:
    decoder_type = Qwen3DecoderLayer
    model_execution_version = "vllm_0_30_0"
    # Derived models may build their own source layers while inheriting forward.
    # Only the homogeneous constructor below registers a replay program.
    _source_module_program = None

    def __init__(self, trace, model, pool, q_lens, kv_lens, cache_capacity_tokens, slot_mapping,
                 *, profile_no_metadata=False, group=None, collective=None, token_ids=None, mlp_factories=None,
                 fused_embedding=True, rope_program=ROPE_PROGRAM_FUSED, hardware_id="nvidia_h200", layer_range=None,
                 ffn_factory=None, input_mode="token_ids", model_execution_version=None,
                 maca_copy_runtime_source=None, legacy_dummy_run=False):
        self.trace = trace
        self.group = group
        self.model_execution_version = (type(self).model_execution_version if model_execution_version is None else model_execution_version)
        if self.model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
            raise ValueError("dense forward requires its explicit common model source version")
        if maca_copy_runtime_source is not None and (
                hardware_id != "metax_c500" or self.model_execution_version != "vllm_0_24_0"
                or group is not None or model.num_experts):
            raise NotImplementedError("MACA Copy runtime source requires ordinary C5000.24 dense TP1")
        if input_mode not in ("token_ids", "inputs_embeds"):
            raise ValueError("model input_mode requires token_ids or inputs_embeds")
        if type(profile_no_metadata) is not bool:
            raise TypeError("model attention profile must be explicit boolean metadata")
        if type(legacy_dummy_run) is not bool or legacy_dummy_run and (not profile_no_metadata
                or self.model_execution_version != "vllm_0_30_0" or hardware_id not in ("nvidia_a800", "nvidia_h200")):
            raise ValueError("legacy dummy model requires explicit CUDA0.30 metadata=None")
        tp_size = 1 if group is None else len(group.traces)
        if layer_range is None:
            layer_range = (0, model.num_hidden_layers)
        if (not isinstance(layer_range, tuple) or len(layer_range) != 2
                or any(type(index) is not int for index in layer_range)
                or not 0 <= layer_range[0] < layer_range[1] <= model.num_hidden_layers):
            raise ValueError("model stage requires an explicit nonempty decoder layer range")
        self.first_stage, self.last_stage = layer_range[0] == 0, layer_range[1] == model.num_hidden_layers
        self._source_module_program = (trace._module_program(self.decoder_type)
            if mlp_factories is None and ffn_factory is None and not model.num_experts
                and model.compute_dtype == model.weight_dtype and model.compute_dtype in ("bf16", "fp16")
                and not profile_no_metadata and self.decoder_type in (Qwen3DecoderLayer, LlamaDecoderLayer)
            else None)
        self.inputs_embeds_shape = (sum(q_lens), model.hidden_size)
        self.inputs_embeds_dtype = model.compute_dtype
        ascend018 = self.model_execution_version == "vllm_0_18_0"
        if ascend018 and (hardware_id != "ascend_910b" or model.compute_dtype != "bf16"
                or model.weight_dtype != "bf16" or ffn_factory is not None
                or (model.num_experts and (model.model_type != "qwen3_moe" or mlp_factories is None or tp_size != 2))):
            raise NotImplementedError("Ascend0.18 ordinary model requires unquantized BF16 dense source")
        self.embed_tokens = None
        if self.first_stage and input_mode == "token_ids":
            self.embed_tokens = (SymbolicEmbedding(trace, "embedding", pool, model.hidden_size, model.compute_dtype,
                model_execution_version=self.model_execution_version, vocab_size=model.vocab_size,
                maca_copy_runtime_source=maca_copy_runtime_source)
            if group is None else SymbolicVocabParallelEmbedding(trace, "embedding", pool,
                model.hidden_size, model.vocab_size, group, collective, token_ids=token_ids, fused=fused_embedding,
                input_values_source="legacy_dummy_retained_buffer" if legacy_dummy_run and token_ids is None else None,
                dtype=model.compute_dtype, model_execution_version=self.model_execution_version,
                reduction_inplace=ascend018,
                mask_program=("vllm_0_11" if self.model_execution_version == "vllm_0_11_0" else
                    "vllm_0_24" if self.model_execution_version == "vllm_0_24_0" else
                    "vllm_ascend_no_added_vocab" if hardware_id == "ascend_910b" else "vllm_0_30")))
        # get_rope caches one module shared by identical layer constructors.
        self.rope_cache = trace.tensor("model.rope-cache",
            (model.max_position_embeddings, model.head_dim if model.rotary_dim is None else model.rotary_dim),
            model.compute_dtype, persistent=True)
        # Exact-class OOT registration selects this constructor only for the
        # default RotaryEmbedding. Llama3/native and scaled constructors have
        # different persistent programs; do not infer their storage here.
        self._rope_constructor_cache_views = (_ascend018_interleaved_rope_constructor_views(trace, self.rope_cache)
            if ascend018 and model.rope_scaling_type == "default" else ())
        self.layers = []
        for layer in range(*layer_range):
            prefix = f"layer-{layer}"
            cache = None if profile_no_metadata else trace.tensor(prefix + ":kv-cache",
                gqa_cache_shape(cache_capacity_tokens, model.kv_heads_per_tensor_parallel_rank(tp_size), model.head_dim),
                model.compute_dtype, persistent=True)
            self.layers.append(self.decoder_type(trace, model, pool, prefix,
                self.rope_cache, cache, q_lens, kv_lens, slot_mapping,
                profile_no_metadata=profile_no_metadata, group=group, collective=collective,
                mlp_factory=None if mlp_factories is None else mlp_factories[layer], rope_program=rope_program,
                hardware_id=hardware_id, ffn_factory=ffn_factory, collective_inplace=ascend018,
                model_execution_version=self.model_execution_version, legacy_dummy_run=legacy_dummy_run))
        self.norm = (_dense_final_norm(trace, model, pool, hardware_id,
            model_execution_version=self.model_execution_version) if self.last_stage else None)

    def __call__(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        # Qwen2Model.forward's ordinary PP/no-aux path, inherited by Qwen3.
        if self.first_stage:
            if intermediate_tensors is not None:
                raise ValueError("first pipeline stage cannot consume intermediate tensors")
            if inputs_embeds is not None:
                self.trace._check_tensor(inputs_embeds)
                if (inputs_embeds.shape != self.inputs_embeds_shape
                        or inputs_embeds.dtype != self.inputs_embeds_dtype):
                    raise ValueError("model inputs_embeds requires the declared typed hidden-state extent")
                hidden_states = inputs_embeds
            else:
                if self.embed_tokens is None:
                    raise ValueError("model constructed for inputs_embeds cannot consume token IDs")
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            if inputs_embeds is not None:
                raise ValueError("inputs_embeds belongs to the first model stage")
            if not isinstance(intermediate_tensors, dict) or set(intermediate_tensors) != {"hidden_states", "residual"}:
                raise ValueError("later pipeline stages require both intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers:
            if self._source_module_program is None:
                hidden_states, residual = layer(positions, hidden_states, residual)
            else:
                hidden_states, residual = self.trace._call_module_frame(self._source_module_program,
                    layer._source_prefix, layer, (positions, hidden_states, residual), group=self.group)
        if not self.last_stage:
            return {"hidden_states": hidden_states, "residual": residual}
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaModel(Qwen3Model):
    decoder_type = LlamaDecoderLayer


class Glm4MoeAttention(Qwen3Attention):
    """GLM4 standard GQA body, including optional QK norm and partial RoPE."""

    def __init__(self, trace, model, *args, **kwargs):
        self.qk_norm = model.qk_norm
        super().__init__(trace, model, *args, **kwargs)

    def __call__(self, positions, hidden_states):
        if self.qk_norm:
            return Qwen3Attention.__call__(self, positions, hidden_states)
        return LlamaAttention.__call__(self, positions, hidden_states)


class Glm4MoE:
    """vLLM0.11 Glm4MoE.forward; separate shared branch, then late TP sum."""

    def __init__(self, trace, model, pool, prefix, *, resolution, invocation_id, group=None, collective=None):
        if group is not None or collective is not None:
            raise NotImplementedError("GLM4 common forward currently requires its maintained TP1 domain")
        self.cast = SymbolicFloat32Cast(trace, prefix + ":router-cast", pool)
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size,
            model.num_experts, kind=WorkloadKind.ROUTER)
        self.router = SymbolicBiasedSigmoidTopK(trace, prefix + ":route", pool, model)
        self.experts = SymbolicUnquantizedExperts(trace, prefix + ":experts", pool, model,
            tp_size=1, resolution=resolution, invocation_id=invocation_id, inplace=True, model_execution_version="vllm_0_11_0")
        self.shared = (DeepseekV2MLP(trace, model, pool, prefix + ":shared", tp_size=1,
            intermediate_size=model.moe_intermediate_size * model.num_shared_experts,
            module_path=prefix + ".shared_experts", reduce_results=False)
            if model.num_shared_experts else None)
        self.scale = SymbolicScalarMultiply(trace, prefix + ":output-scale", pool,
            model.routed_scaling_factor, inplace=False)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)

    def __call__(self, hidden_states):
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(-1, hidden_dim)
        router_logits = self.gate(self.cast(hidden_states))[0]
        # FusedMoE.forward_impl's non-modular branch computes shared first;
        # UnquantizedFusedMoEMethod then calls fused_experts(inplace=True).
        shared_output = None if self.shared is None else self.shared(hidden_states)
        weights, ids = self.router(hidden_states, router_logits)
        output = self.experts(hidden_states, weights, ids)
        output = self.scale(output)
        if shared_output is not None:
            output = self.add(output, shared_output)
        return output.view_as_shape(num_tokens, hidden_dim)


class Glm4MoeDecoderLayer(Qwen3DecoderLayer):
    attention_type = Glm4MoeAttention


class Glm4MoeModel(Qwen3Model):
    """Glm4MoeModel uses the same embedding/residual loop and final norm."""
    decoder_type = Glm4MoeDecoderLayer
    model_execution_version = "vllm_0_11_0"


class Glm4MoeMultiTokenPredictorLayer:
    """vLLM0.11 draft-config layer0 body; weights/KV reused across passes.

    speculative.py resets num_hidden_layers=0 before constructing GLM4's
    draft. The maintained first_k_dense_replace=1 source therefore has a
    dense decoder, distinct from the0.30 appended-layer MoE constructor.
    The caller supplies genuine proposer metadata; this class neither
    invents next-token values nor performs sampling/rejection decisions.
    """

    def __init__(self, trace, model, pool, prefix, rope_cache, cache, q_lens, kv_lens, slots, *, ffn_factory=None):
        _validate_glm4_source(model)
        if (model.mtp_architecture != "glm4_moe_mtp" or model.num_mtp_layers != 1
                or model.mtp_layer_uses_moe(0)):
            raise NotImplementedError("GLM4 draft source requires its single dense0.11 predictor layer")
        self.trace, self.model = trace, model
        self.embed_tokens = SymbolicEmbedding(trace, prefix + ":embedding", pool, model.hidden_size, "bf16",
            model_execution_version="vllm_0_11_0")
        self.mask = SymbolicPositionZeroMask(trace, prefix + ":position-mask", pool)
        self.enorm = SymbolicRMSNorm(trace, prefix + ":embedding-norm", pool, model.hidden_size, model.rms_norm_eps)
        self.hnorm = SymbolicRMSNorm(trace, prefix + ":hidden-norm", pool, model.hidden_size, model.rms_norm_eps)
        self.cat = SymbolicConcat(trace, prefix + ":concat", pool)
        self.eh_proj = SymbolicLinear(trace, prefix + ":merge", pool, 2 * model.hidden_size, model.hidden_size)
        self.mtp_block = Glm4MoeDecoderLayer(trace, model, pool, prefix + ":decoder", rope_cache,
            cache, q_lens, kv_lens, slots, hardware_id="nvidia_a800", ffn_factory=ffn_factory)
        self.add = SymbolicAdd(trace, prefix + ":residual-add", pool)
        self.head_norm = SymbolicRMSNorm(trace, prefix + ":head-norm", pool, model.hidden_size, model.rms_norm_eps)
        self.head = SymbolicLinear(trace, prefix + ":head", pool, model.hidden_size, model.vocab_size,
            kind=WorkloadKind.LM_HEAD)

    def __call__(self, input_ids, positions, previous_hidden_states, *, position_values,
                 q_lens, kv_lens, slots, inputs_embeds=None, slot_tensor=None):
        rows = sum(q_lens)
        if (not isinstance(q_lens, tuple) or not q_lens or not isinstance(kv_lens, tuple) or len(q_lens) != len(kv_lens)
                or any(type(q) is not int or type(k) is not int or not 0 < q <= k
                    for q, k in zip(q_lens, kv_lens))
                or input_ids.shape != (rows,) or input_ids.dtype not in ("int32", "int64")
                or previous_hidden_states.shape != (rows, self.model.hidden_size)
                or previous_hidden_states.dtype != "bf16" or len(slots) != rows
                or len(set(slots)) != rows
                or not isinstance(position_values, tuple) or len(position_values) != rows
                or any(type(p) is not int or not 0 <= p < self.model.max_position_embeddings for p in position_values)):
            raise ValueError("GLM4 MTP requires its declared per-pass query/KV/input/hidden metadata")
        attention = self.mtp_block.self_attn.attn
        if (sum(kv_lens) > attention.cache.shape[1]
                or any(type(slot) is not int or not 0 <= slot < attention.cache.shape[1] for slot in slots)):
            raise ValueError("GLM4 MTP slots must be unique valid positions in its cache")
        attention.q_lens, attention.kv_lens, attention.slot_mapping = q_lens, kv_lens, slots
        attention.slot_tensor = slot_tensor
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.mask(positions, inputs_embeds, position_values=position_values)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(self.cat(inputs_embeds, previous_hidden_states))[0]
        hidden_states, residual = self.mtp_block(positions, hidden_states, None)
        hidden_states = self.add(residual, hidden_states)
        return hidden_states

    def compute_logits(self, hidden_states):
        return self.head(self.head_norm(hidden_states))[0]


def _validate_glm4_source(model):
    if (model.model_type != "glm4_moe" or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
            or model.hidden_act != "silu" or model.rms_norm_eps is None or model.rotary_dim is None
            or model.rope_scaling_type != "default" or model.attention_bias or model.mlp_bias
            or model.moe_group_count != 1 or model.moe_topk_group_count != 1):
        raise NotImplementedError("GLM4 common source requires BF16, explicit norms/partial RoPE and single-group routing")


def _glm4_model_inputs(model, step, pool, expert_routing, *, input_ids, positions,
        slot_mapping, cache_capacity_tokens, source_profile, mtp_source=False,
        trace=None, ffn_factory=None, expert_pool=None):
    """Shared target model construction and typed inputs, before forward."""
    from functools import partial
    _validate_glm4_source(model)
    config = pool.parallel_plan.config
    if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200")
            or mtp_source and pool.hardware_id != "nvidia_a800" or config.enable_expert_parallel
            or any(value != 1 for value in (config.tensor_parallel_size, config.pipeline_parallel_size,
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("GLM4 common decoder requires A800/H200 TP1; MTP retains its A800 source")
    q_lens, kv_lens, rows = _dense_source_metadata(model, step, input_ids=input_ids,
        positions=positions, slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, hardware_id=pool.hardware_id, mtp_source=mtp_source)
    trace = ForwardTrace(rank=pool.ranks[0], namespace=step.step_id) if trace is None else trace
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool if expert_pool is None else expert_pool,
        expert_routing, invocations, rows)
    factories = tuple(partial(Glm4MoE, resolution=resolutions[layer], invocation_id=invocations[layer])
        if model.layer_uses_moe(layer) else None for layer in range(model.num_hidden_layers))
    network = Glm4MoeModel(trace, model, pool.name, q_lens, kv_lens, cache_capacity_tokens, slot_mapping,
        mlp_factories=factories, hardware_id=pool.hardware_id, ffn_factory=ffn_factory)
    ids = trace.tensor("input_ids", (rows,), "int32")
    pos = trace.tensor("positions", (rows,), "int64")
    return trace, network, ids, pos, q_lens, kv_lens


def trace_glm4_decoder(model, step, pool, expert_routing, **source_inputs):
    """Complete target hidden-state forward; caller owns sampling/proposing."""
    trace, network, ids, pos, _, _ = _glm4_model_inputs(model, step, pool, expert_routing,
        mtp_source=False, **source_inputs)
    hidden_states = network(ids, pos)
    return trace, hidden_states


def trace_glm4_forward(model, step, pool, expert_routing, *,
        target_sampling_source="vllm_0_11_plain_greedy_bf16", **source_inputs):
    """Target-off GLM4 with explicit logits, FP32 cast or greedy output."""
    if target_sampling_source not in (None, "vllm_0_11_plain_sampler_bf16", "vllm_0_11_plain_greedy_bf16"):
        raise ValueError("GLM4 target output requires its matching0.11 sampler source")
    trace, hidden_states = trace_glm4_decoder(model, step, pool, expert_routing, **source_inputs)
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    indices = tuple(sum(q_lens[:i + 1]) - 1 for i in range(len(q_lens)))
    logits_indices = trace.tensor("logits_indices", (len(indices),), "int32")
    selected = SymbolicLogitsRowSelect(trace, "logits-select", pool.name)(hidden_states, logits_indices,
        selected_indices=indices, model_execution_version="vllm_0_11_0", hardware_id=pool.hardware_id)
    logits, _ = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
        model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)
    if target_sampling_source is not None:
        SymbolicGreedy(trace, "target-sampler", pool.name)(logits, model_execution_version="vllm_0_11_0",
            sample_logits=target_sampling_source == "vllm_0_11_plain_greedy_bf16")
    return trace


def trace_glm4_mtp_forward(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_profile, max_model_len, max_batched_tokens, max_requests,
        trace=None, ffn_factory=None, expert_pool=None):
    """Complete0.11 eager all-greedy padded target/rejection/Eagle-MTP graph.

    Acceptance and positions are explicit schedule scenarios; no token is
    computed. Logical KV coordinates are disjoint request arenas, not native
    physical pages. This initial source domain rejects context-cap overflow,
    discarded/incomplete rows, trees, auxiliary hidden states and parallelism.
    """
    from .traced_sampling import MTPSourceOps
    requests = step.requests
    if (any(type(cap) is not int or cap <= 0 for cap in (max_model_len, max_batched_tokens, max_requests))
            or max_model_len > model.max_position_embeddings or len(requests) > max_requests
            or sum(r.scheduled_tokens for r in requests) > max_batched_tokens):
        raise ValueError("GLM4 MTP requires explicit source capacities covering every active row/request")
    depths = {r.mtp_proposer_tokens for r in requests}
    if (len(depths) != 1 or not depths or min(depths) <= 0
            or any(r.mtp_proposer_mode != "padded" or r.raw_model_output_tokens != r.mtp_accepted_tokens + 1
                or r.scheduled_tokens < r.mtp_verified_tokens + 1 for r in requests)):
        raise NotImplementedError("GLM4 MTP source requires one padded depth and a valid sampled anchor for every request")
    depth = next(iter(depths))
    if pool.kv_cache is None:
        raise ValueError("GLM4 MTP requires an explicit cache block-size contract")
    # Eagle samples the first draft at the retained target row, then only
    # advances that position for the remaining passes. Rejected target-tail
    # rows stay in the padded first pass but do not advance that anchor.
    if any(r.computed_tokens + r.scheduled_tokens > max_model_len
            or r.computed_tokens + r.scheduled_tokens - r.mtp_verified_tokens
                + r.mtp_accepted_tokens + depth - 2 >= max_model_len for r in requests):
        raise NotImplementedError("GLM4 MTP cap-exceeded padding/cache slots are not migrated")
    trace, network, ids, pos, q_lens, kv_lens = _glm4_model_inputs(model, step, pool, expert_routing,
        input_ids=input_ids, positions=positions, slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, mtp_source=True, trace=trace, ffn_factory=ffn_factory, expert_pool=expert_pool)
    ops = MTPSourceOps(trace, "mtp-source", pool.name)
    rows, batch = sum(q_lens), len(q_lens)
    verified, accepted = tuple(r.mtp_verified_tokens for r in requests), tuple(r.mtp_accepted_tokens for r in requests)
    source_has_spec = any(verified)

    # EagleProposer constructor buffers, including the unused pure-text
    # inputs_embeds arena. Capacities are explicit; active slices are views.
    draft_ids = trace.tensor("mtp.input-ids", (max_batched_tokens,), "int32", persistent=True)
    draft_pos = trace.tensor("mtp.positions", (max_batched_tokens,), "int64", persistent=True)
    draft_hidden = trace.tensor("mtp.hidden-states", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
    unused_embeds = trace.tensor("mtp.inputs-embeds", (max_batched_tokens, model.hidden_size), "bf16", persistent=True)
    arange = trace.tensor("mtp.arange", (max(max_requests + 1, max_batched_tokens),), "int32", persistent=True)
    backup = trace.tensor("mtp.backup-next-token-ids", (max_requests,), "int32", persistent=True)
    query_start = trace.tensor("parent.query-start", (max_requests + 1,), "int32", persistent=True)
    seq_lens = trace.tensor("parent.seq-lens", (max_requests,), "int32", persistent=True).narrow(0, 0, batch)
    # This is only the source table's logical active-read descriptor. Its
    # physical backing/IDs continue to belong to the native cache owner.
    block_size = next(tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")
    block_table = trace.tensor("parent.block-table", (max_requests, -(-max_model_len // block_size)),
        "int32", persistent=True).narrow(0, 0, batch)
    parent_slots = trace.tensor("parent.slots", (rows,), "int64", persistent=True)
    arena_bases, arena_size = [], 0
    for length in kv_lens:
        arena_bases.append(arena_size)
        arena_size += -(-(length + depth - 1) // block_size) * block_size
    draft_slots = tuple(arena_bases[i] + r.computed_tokens + j for i, r in enumerate(requests)
        for j in range(r.scheduled_tokens))
    if slot_mapping != draft_slots:
        raise ValueError("GLM4 MTP target and draft must share the declared logical page coordinates")
    draft_cache = trace.tensor("mtp.kv-cache", (2, arena_size, model.num_key_value_heads, model.head_dim),
        "bf16", persistent=True)
    predictor = Glm4MoeMultiTokenPredictorLayer(trace, model, pool.name, "mtp", network.rope_cache,
        draft_cache, q_lens, kv_lens, draft_slots, ffn_factory=ffn_factory)

    # GPUModelRunner._calc_spec_decode_metadata: all four index/prefix
    # arrays are INT32 (explicit cumsum_dtype=np.int32 in0.11).
    logits_indices_values, bonus_values, target_values = [], [], []
    offset, sampled_offset = 0, 0
    for query, count in zip(q_lens, verified):
        logits_indices_values.extend(range(offset + query - count - 1, offset + query))
        bonus_values.append(sampled_offset + count)
        target_values.extend(range(sampled_offset, sampled_offset + count))
        offset += query
        sampled_offset += count + 1
    logits_indices_values, bonus_values, target_values = map(tuple,
        (logits_indices_values, bonus_values, target_values))
    logits_indices = trace.tensor("target.logits-indices", (len(logits_indices_values),), "int32")
    if source_has_spec:
        ops.h2d(logits_indices)
        prefix = ops.h2d(trace.tensor("target.cu-draft-tokens", (batch,), "int32"))
        bonus_indices = ops.h2d(trace.tensor("target.bonus-indices", (batch,), "int32"))
        target_indices = ops.h2d(trace.tensor("target.draft-indices", (sum(verified),), "int32"))
        selected_ids = ops.gather(ids, logits_indices, selected_indices=logits_indices_values)
        shifted = ops.unary(target_indices, expression="add", scalar=1)
        previous_draft_ids = ops.gather(selected_ids, shifted, selected_indices=tuple(i + 1 for i in target_values))
    hidden_states = network(ids, pos)
    selected_hidden = ops.gather(hidden_states, logits_indices, selected_indices=logits_indices_values)
    logits = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
        model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected_hidden)[0]
    def sample_target():
        if source_has_spec:
            bonus_logits = ops.gather(logits, bonus_indices, selected_indices=bonus_values)
            bonus_ids = SymbolicGreedy(trace, "target-bonus-sampler", pool.name)(bonus_logits,
                model_execution_version="vllm_0_11_0").view_as_shape(batch)
            target_logits = ops.gather(logits, target_indices, selected_indices=target_values)
            #0.11 compute_probs(all_greedy=True) returns the original BF16 logits.
            target_argmax = ops.unary(target_logits, "argmax")
            sampled = ops.fill(trace.tensor("target.rejection-output", (batch, max(verified) + 1), "int32"), value=-1)
            return ops.rejection(sampled, prefix, previous_draft_ids, target_argmax, bonus_ids,
                verified=verified, accepted=accepted)
        else:
            return SymbolicGreedy(trace, "target-sampler", pool.name)(logits, model_execution_version="vllm_0_11_0")
    sampled = sample_target()

    # prepare_next_token_ids_padded: actual Tensor expressions, with INT64
    # bool.sum output. No discarded requests in this declared source domain.
    def prepare_next():
        backup_front = ops.h2d(backup.narrow(0, 0, batch))
        valid_sampled = ops.copy(sampled)
        if sampled.shape[1] == 1:
            valid_mask = ops.fill(trace.tensor("mtp.valid-mask", sampled.shape, "bool"), value=True)
        else:
            valid_mask = ops.binary(ops.unary(valid_sampled, "compare", expression="not_equal", scalar=-1),
                ops.unary(valid_sampled, "compare", expression="less", scalar=model.vocab_size), "logical_and")
        valid_count = ops.unary(valid_mask, "sum")
        last_valid = ops.unary(valid_count, expression="subtract", scalar=1)
        safe_last = ops.unary(last_valid, expression="clamp_min", scalar=0)
        selected_tokens = ops.row_gather(valid_sampled, safe_last)
        next_tokens = ops.where(ops.unary(last_valid, "compare", expression="not_equal", scalar=-1), selected_tokens, backup_front)
        return next_tokens, valid_count
    next_tokens, valid_count = prepare_next()
    def prepare_source_inputs():
        if source_has_spec:
            # torch.cat([cu[0:1], cu[1:]-cu[:-1]]) also materializes for B=1.
            first = prefix.narrow(0, 0, 1)
            if batch > 1:
                difference = ops.binary(prefix.narrow(0, 1, batch - 1), prefix.narrow(0, 0, batch - 1))
                num_draft = ops.concat((first, difference))
            else:
                num_draft = ops.concat((first,))
            positive = ops.unary(num_draft, "compare", expression="greater", scalar=0)
            plus_one = ops.unary(num_draft, expression="add", scalar=1)
            rejected_candidate = ops.binary(plus_one, valid_count, "promote_subtract")
            zero_draft = ops.fill(trace.tensor("mtp.zero-draft-count", (batch,), "int32"))
            rejected = ops.where(positive, rejected_candidate, zero_draft)
            source_metadata = {"slot_mapping": ops.gather(parent_slots, arange.narrow(0, 0, rows),
                selected_indices=tuple(range(rows)))}
            query_end = ops.unary(query_start.narrow(0, 1, batch), expression="subtract", scalar=1)
            last_indices = ops.binary(query_end, rejected, "promote_subtract")
            retained = tuple(sum(q_lens[:i]) + q_lens[i] - (verified[i] - accepted[i]) - 1 for i in range(batch))
            # The three advanced index expressions copy complete padded inputs,
            # even though arange selects every row in order.
            target_ids = ops.gather(ids, arange.narrow(0, 0, rows), selected_indices=tuple(range(rows)))
            target_pos = ops.gather(pos, arange.narrow(0, 0, rows), selected_indices=tuple(range(rows)))
            target_hidden = ops.gather(hidden_states, arange.narrow(0, 0, rows), selected_indices=tuple(range(rows)))
        else:
            retained = tuple(sum(q_lens[:i + 1]) - 1 for i in range(batch))
            last_indices = ops.unary(query_start.narrow(0, 1, batch), expression="subtract", scalar=1)
            target_ids, target_pos, target_hidden = ids, pos, hidden_states
            source_metadata = {"slot_mapping": parent_slots}
        return target_ids, target_pos, target_hidden, last_indices, retained, source_metadata
    target_ids, target_pos, target_hidden, last_indices, retained, source_metadata = prepare_source_inputs()
    id_front, pos_front = draft_ids.narrow(0, 0, rows), draft_pos.narrow(0, 0, rows)
    hidden_front = draft_hidden.narrow(0, 0, rows)
    if rows > 1:
        ops.copy(target_ids.narrow(0, 1, rows - 1), id_front.narrow(0, 0, rows - 1))
    ops.scatter(next_tokens, last_indices, id_front, selected_indices=retained)
    ops.copy(target_pos, pos_front)
    ops.copy(target_hidden, hidden_front)
    ret_hidden = predictor(id_front, pos_front, hidden_front, position_values=positions,
        q_lens=q_lens, kv_lens=kv_lens, slots=draft_slots, slot_tensor=source_metadata["slot_mapping"])
    sample_hidden = ops.gather(ret_hidden, last_indices, selected_indices=retained)
    draft_logits = predictor.compute_logits(sample_hidden)
    if depth == 1:
        output = ops.unary(draft_logits, "argmax").view_as_shape(batch, 1)
        return trace
    selected_positions = ops.gather(target_pos, last_indices, selected_indices=retained)
    recurrent_hidden = ops.gather(ret_hidden, last_indices, selected_indices=retained)
    generated = [ops.unary(draft_logits, "argmax")]
    current_positions = tuple(positions[index] for index in retained)
    for draft_index in range(1, depth):
        next_ids = ops.unary(generated[-1], "cast_ids")
        ops.unary(selected_positions, inplace=True, expression="add", scalar=1)
        exceeds = ops.unary(selected_positions, "compare", expression="greater_equal", scalar=max_model_len)
        clamped_positions = ops.where(exceeds, selected_positions)
        ops.unary(seq_lens, inplace=True, expression="add", scalar=1)
        ops.masked_fill(seq_lens, exceeds, masked_rows=0)
        # Selected block-ID gather/address arithmetic; table reads are
        # represented by the selected INT32 port, not invented page values.
        block_numbers = ops.unary(clamped_positions, expression="floor_divide", scalar=block_size)
        gathered_blocks = ops.row_gather(block_table, block_numbers)
        scaled_blocks = ops.unary(gathered_blocks, expression="multiply", scalar=block_size)
        remainders = ops.unary(clamped_positions, expression="remainder", scalar=block_size)
        slot_values = ops.binary(scaled_blocks, remainders, "promote_add")
        source_metadata["slot_mapping"] = ops.masked_fill(slot_values, exceeds, masked_rows=0)
        ops.copy(next_ids, draft_ids.narrow(0, 0, batch))
        ops.copy(clamped_positions, draft_pos.narrow(0, 0, batch))
        ops.copy(recurrent_hidden, draft_hidden.narrow(0, 0, batch))
        current_positions = tuple(p + 1 for p in current_positions)
        current_slots = tuple(arena_bases[i] + p for i, p in enumerate(current_positions))
        ret_hidden = predictor(draft_ids.narrow(0, 0, batch), draft_pos.narrow(0, 0, batch),
            draft_hidden.narrow(0, 0, batch), position_values=current_positions,
            q_lens=(1,) * batch, kv_lens=tuple(k + draft_index for k in kv_lens), slots=current_slots,
            slot_tensor=source_metadata["slot_mapping"])
        recurrent_hidden = ret_hidden.narrow(0, 0, batch)
        draft_logits = predictor.compute_logits(ret_hidden.narrow(0, 0, batch))
        generated.append(ops.unary(draft_logits, "argmax"))
    output = ops.stack(tuple(generated))
    return trace


class Qwen3MoeRunner:
    """Ordinary unquantized DP/SP1 Runner with a separately owned gate."""

    def __init__(self, trace, model, pool, prefix, *, group, collective, resolution, invocation_id, ep_size=1,
                 model_execution_version="vllm_0_30_0", topk_forward_context=None):
        self.trace, self.group, self.collective, self.pool, self.prefix = trace, group, collective, pool, prefix
        self.ep_size, self.invocation_id = ep_size, invocation_id
        self.collective_inplace = model_execution_version == "vllm_0_18_0"
        self.owner_mask = (SymbolicExpertOwnerMask(trace, prefix + ":owner-mask", pool, model,
            expert_rank=group.participants.index(trace.rank), expert_parallel_size=ep_size)
            if self.collective_inplace else None)
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size,
            model.num_experts, kind=WorkloadKind.ROUTER)
        self.router = SymbolicSoftmaxTopK(trace, prefix + ":route", pool, model,
            model_execution_version=model_execution_version, source_forward_context=topk_forward_context)
        self.experts = SymbolicUnquantizedExperts(trace, prefix + ":experts", pool, model,
            tp_size=1 if group is None else len(group.traces), resolution=resolution, invocation_id=invocation_id,
            expert_parallel_size=ep_size, expert_rank=group.participants.index(trace.rank) if ep_size > 1 else 0,
            inplace=model_execution_version == "vllm_0_11_0",
            routing_weight_dtype="bf16" if self.collective_inplace else "fp32", model_execution_version=model_execution_version)

    def __call__(self, *, hidden_states, router_logits):
        if hidden_states is not router_logits:
            raise ValueError("Qwen3 Runner must own the gate over the source hidden-state placeholder")
        logits, _ = self.gate(hidden_states)
        weights, ids = self.router(hidden_states, logits)
        if self.owner_mask is not None:
            weights = self.owner_mask(weights, ids)
        output = self.experts(hidden_states, weights, ids)
        if self.group is not None:
            return self.group.call(self.prefix + ":all-reduce", output,
                pool_name=self.pool, config=self.collective, inplace=self.collective_inplace,
                moe_invocation_id=self.invocation_id if self.ep_size > 1 else None)
        return output


class Qwen3MoeSparseMoeBlock:
    def __init__(self, trace, model, pool, prefix, *, group, collective, resolution, invocation_id, ep_size=1,
                 model_execution_version="vllm_0_30_0", collective_inplace=False, topk_forward_context=None):
        if collective_inplace != (model_execution_version == "vllm_0_18_0"):
            raise ValueError("MoE collective storage must match its model source")
        self.is_sequence_parallel = False
        self.experts = Qwen3MoeRunner(trace, model, pool, prefix,
            group=group, collective=collective, resolution=resolution, invocation_id=invocation_id, ep_size=ep_size,
            model_execution_version=model_execution_version, topk_forward_context=topk_forward_context)

    def __call__(self, hidden_states):
        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(-1, hidden_dim)
        if self.is_sequence_parallel:
            raise NotImplementedError("Qwen3 virtual MoE sequence parallelism is not migrated")
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states



class Qwen3MoeDistributedBlock:
    """0.30 DP MoE: shared expert service and optional TP row chunks.

    All real and padded row counts are declared before rank execution. The
    collective join later binds every actual producer, including idle replicas.
    Attention/KV and the request/token transcript keep only local real rows.
    """
    def __init__(self, trace, model, pool, prefix, *, group, collective, resolution, invocation_id,
                 expert_group, expert_collectives, restore_collective, tensor_group, enable_expert_parallel,
                 logical_rows_by_replica, rows_by_expert_rank, data_parallel_rank, model_execution_version="vllm_0_30_0", collective_inplace=False,
                 request_rows_by_replica=None):
        from .traced_operators import SymbolicFillZero
        if (model_execution_version != "vllm_0_30_0" or collective_inplace
                or type(enable_expert_parallel) is not bool or model.num_shared_experts
                or not isinstance(tensor_group, (ForwardCollectiveGroup, ForwardLocalGroup))
                or len(rows_by_expert_rank) != len(expert_group.traces)
                or len(logical_rows_by_replica) * (len(tensor_group.traces) if enable_expert_parallel else 1) != len(expert_group.traces)):
            raise NotImplementedError("distributed Qwen3MoE requires explicit0.30 TP/SP over DP with complete EP ownership")
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        self.group, self.expert_group, self.expert_collectives = tensor_group, expert_group, expert_collectives
        self.sequence_parallel = enable_expert_parallel and len(tensor_group.traces) > 1
        self.group_dimension = "expert" if enable_expert_parallel else "data"
        self.output_reduce = collective
        self.local_rows = logical_rows_by_replica[data_parallel_rank]
        self.rows_by_expert_rank = rows_by_expert_rank
        self.lane = tensor_group.participants.index(trace.rank)
        self.chunk_rows = rows_by_expert_rank[expert_group.participants.index(trace.rank)]
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size, model.num_experts, kind=WorkloadKind.ROUTER)
        self.router = SymbolicSoftmaxTopK(trace, prefix + ":route", pool, model, model_execution_version=model_execution_version)
        self.fill = SymbolicFillZero(trace, prefix + ":sequence-pad-zero", pool)
        self.experts = SymbolicUnquantizedExperts(trace, prefix + ":experts", pool, model,
            tp_size=len(logical_rows_by_replica) * len(tensor_group.traces),
            expert_parallel_size=len(expert_group.traces) if enable_expert_parallel else 1,
            expert_rank=expert_group.participants.index(trace.rank) if enable_expert_parallel else 0, resolution=resolution, invocation_id=invocation_id,
            model_execution_version=model_execution_version)
        self.invocation_id = invocation_id
        self.restore_collective = restore_collective
        self.logical_rows_by_replica = logical_rows_by_replica
        self.request_rows_by_replica = logical_rows_by_replica if request_rows_by_replica is None else request_rows_by_replica
        if (not isinstance(self.request_rows_by_replica, tuple)
                or len(self.request_rows_by_replica) != len(logical_rows_by_replica)
                or any(type(real) is not int or real < 0 or (real != work and (real != 0 or work != 1))
                    for real, work in zip(self.request_rows_by_replica, logical_rows_by_replica, strict=True))):
            raise ValueError("Qwen DP request rows must match real work rows or an explicit one-row dummy")

    def __call__(self, hidden_states):
        tp = len(self.group.traces)
        local = logits = None
        if self.local_rows:
            if hidden_states is None or hidden_states.shape != (self.local_rows, self.model.hidden_size):
                raise ValueError("MoE input must contain exactly this DP replica's actual rows")
            padded_rows = self.chunk_rows * tp
            if not self.sequence_parallel:
                local = hidden_states
            elif padded_rows != self.local_rows:
                padded = self.trace.tensor(self.trace.unique_name(self.prefix + ":sequence-padded"),
                    (padded_rows, self.model.hidden_size), hidden_states.dtype)
                SymbolicCopy(self.trace, self.prefix + ":sequence-pad-copy", self.pool)(hidden_states,
                    padded.narrow(0, 0, self.local_rows))
                self.fill(padded.narrow(0, self.local_rows, padded_rows - self.local_rows))
                local = padded.narrow(0, self.lane * self.chunk_rows, self.chunk_rows)
            else:
                selected = hidden_states.narrow(0, self.lane * self.chunk_rows, self.chunk_rows)
                clone = self.trace.tensor(self.trace.unique_name(self.prefix + ":sequence-clone"), selected.shape, selected.dtype)
                local = SymbolicCopy(self.trace, self.prefix + ":sequence-clone-copy", self.pool)(selected, clone)
            logits, _ = self.gate(local)
        elif hidden_states is not None:
            raise ValueError("idle DP rank must not invent local hidden rows")
        dispatched = self.expert_group.call_variable(self.prefix + ":dispatch-hidden", self.trace, local,
            pool_name=self.pool, config=self.expert_collectives[0], rows_by_rank=self.rows_by_expert_rank,
            row_shape=(self.model.hidden_size,), dtype="bf16", group_dimension=self.group_dimension)
        global_logits = self.expert_group.call_variable(self.prefix + ":dispatch-logits", self.trace, logits,
            pool_name=self.pool, config=self.expert_collectives[0], rows_by_rank=self.rows_by_expert_rank,
            row_shape=(self.model.num_experts,), dtype="bf16", group_dimension=self.group_dimension)
        weights, ids = self.router(dispatched, global_logits)
        partial = self.experts(dispatched, weights, ids)
        node_id = self.trace._writers[partial.view.allocation_id]
        node = self.trace._nodes[node_id]
        self.trace._nodes[node_id] = node.with_attributes({**node.attributes,
            "source_dp_logical_rows": self.logical_rows_by_replica,
            "source_dp_request_rows": self.request_rows_by_replica,
            "source_dp_lane_rows": self.rows_by_expert_rank,
            "source_dp_row_order": "dp_then_tp_lane_then_row" if self.sequence_parallel else "dp_then_row",
            "source_tensor_parallel_size": len(self.group.traces),
            "source_padding_routing": "declared_routing_for_physical_rows_including_padding_not_observed_topk"})
        reduced = self.expert_group.call_variable(self.prefix + ":combine", self.trace, partial,
            pool_name=self.pool, config=self.expert_collectives[1], rows_by_rank=self.rows_by_expert_rank,
            row_shape=(self.model.hidden_size,), dtype="bf16", moe_invocation_id=self.invocation_id, group_dimension=self.group_dimension)
        if not self.local_rows:
            return None
        if self.sequence_parallel:
            restored = self.group.call(self.prefix + ":sequence-restore", reduced, pool_name=self.pool,
                config=self.restore_collective, inplace=False, dim=0)
            return restored.narrow(0, 0, self.local_rows)
        return self.group.call(self.prefix + ":tensor-output-reduce", reduced, pool_name=self.pool,
            config=self.output_reduce, inplace=False)


@dataclass(frozen=True)
class _QwenLegacyInputState:
    pool: object
    model_sha256: str
    step_ids: tuple
    first_token_ids: tuple


def trace_qwen3_moe_data_parallel_forward(model, step, pool, expert_routing, *, input_ids,
        source_profile="cuda_eager_inplace", sample_logits=True, source_program_cache=None):
    """Complete ordinary0.30 DP source with EP-on/off and optional TP/SP."""
    from functools import partial
    config = pool.parallel_plan.config
    tp, dp = config.tensor_parallel_size, config.data_parallel_size
    from src.serving.deployment.validated_scope import is_qwen3_moe_dp_sp_source_scope
    if not is_qwen3_moe_dp_sp_source_scope(pool, model):
        raise NotImplementedError("coordinated Qwen3MoE source requires legal0.30 BF16 DP expert geometry")
    if (source_profile != "cuda_eager_inplace" or type(sample_logits) is not bool
            or source_program_cache is not None and type(source_program_cache) is not SourceProgramCache):
        raise ValueError("Qwen DP source requires its explicit eager program and run-local source cache")
    if (not isinstance(input_ids, tuple) or len(input_ids) != step.token_count
            or any(type(token) is not int or not 0 <= token < model.vocab_size for token in input_ids)
            or step.pool_name != pool.name or not step.requests or len({r.request_id for r in step.requests}) != len(step.requests)
            or any(request.data_parallel_rank >= dp for request in step.requests)):
        raise ValueError("coordinated source requires explicit global token rows and valid DP identities")
    by_replica = tuple(tuple(request for request in step.requests if request.data_parallel_rank == index) for index in range(dp))
    logical_rows = tuple(sum(request.scheduled_tokens for request in requests) for requests in by_replica)
    work_rows = tuple(count if count else 1 for count in logical_rows)
    sequence_parallel = config.enable_expert_parallel and tp > 1
    lane_rows = (tuple(-(-count // tp) for count in work_rows for _ in range(tp))
        if sequence_parallel else work_rows)
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, sum(lane_rows))
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    pp = config.pipeline_parallel_size
    tensor_groups = tuple(ForwardCollectiveGroup(traces[index * tp:(index + 1) * tp],
        namespace=f"{step.step_id}:dp{index // pp}:stage{index % pp}:tp") if tp > 1 else ForwardLocalGroup(traces[index])
        for index in range(dp * pp))
    distribution_groups = tuple(ForwardCollectiveGroup(tuple(traces[replica * pp * tp + stage * tp + lane]
        for replica in range(dp) for lane in (range(tp) if config.enable_expert_parallel else (data_lane,))),
        namespace=f"{step.step_id}:stage{stage}:distribution{data_lane}")
        for stage in range(pp) for data_lane in (range(1) if config.enable_expert_parallel else range(tp)))
    joined = ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:joined", subgroups=(
        *(group for group in tensor_groups if isinstance(group, ForwardCollectiveGroup)), *distribution_groups))
    collectives = tuple(pool.collective_for(operation) for operation in ("all_gatherv", "reduce_scatterv"))
    row_ids, offset = {}, 0
    for request in step.requests:
        row_ids[request.request_id] = input_ids[offset:offset + request.scheduled_tokens]
        offset += request.scheduled_tokens
    state_key = id(pool)
    prior_inputs = None
    if source_program_cache is not None and state_key in source_program_cache.qwen_legacy_inputs:
        prior_inputs = source_program_cache.qwen_legacy_inputs[state_key]
        if (type(prior_inputs) is not _QwenLegacyInputState or prior_inputs.pool is not pool
                or prior_inputs.model_sha256 != model.config_sha256 or len(prior_inputs.first_token_ids) != dp
                or step.step_id in prior_inputs.step_ids):
            raise ValueError("Qwen dummy inputs require their own unreplayed model/pool source history")
    # CpuGpuBuffer constructs the input-ID buffers with torch.zeros. A fresh
    # run-owned cache binds that constructor; a standalone warm source lacks
    # the history and therefore keeps unknown IDs rather than assuming zero.
    previous_ids = ((None,) * dp if source_program_cache is None else (0,) * dp) if prior_inputs is None else prior_inputs.first_token_ids
    current_ids = tuple(row_ids[requests[0].request_id][0] if requests else previous_ids[replica]
        for replica, requests in enumerate(by_replica))
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_plan
    ranges = pipeline_layer_ranges_for_plan(pool.parallel_plan, num_layers=model.num_hidden_layers)
    for replica, requests in enumerate(by_replica):
        def stage_factories(trace, stage):
            group = tensor_groups[replica * pp + stage]
            distribution = distribution_groups[stage * (1 if config.enable_expert_parallel else tp)
                + (0 if config.enable_expert_parallel else group.participants.index(trace.rank))]
            factories = tuple(partial(Qwen3MoeDistributedBlock, resolution=resolution, invocation_id=invocation,
                expert_group=distribution, expert_collectives=collectives,
                restore_collective=pool.collective_for("all_gather") if sequence_parallel else None,
                tensor_group=group, enable_expert_parallel=config.enable_expert_parallel, logical_rows_by_replica=work_rows,
                rows_by_expert_rank=lane_rows, data_parallel_rank=replica, request_rows_by_replica=logical_rows)
                for resolution, invocation in zip(resolutions, invocations, strict=True))
            return factories
        if not requests:
            stage_groups = tensor_groups[replica * pp:(replica + 1) * pp]
            def run_dummy(trace, stage, incoming):
                group = stage_groups[stage] if tp > 1 else None
                network = Qwen3Model(trace, model, pool.name, (1,), (1,), None, (-1,), profile_no_metadata=True,
                    legacy_dummy_run=True, layer_range=ranges[stage], mlp_factories=stage_factories(trace, stage),
                    token_ids=None if current_ids[replica] is None else (current_ids[replica],),
                    group=group, collective=pool.collective_for("all_reduce") if tp > 1 else None,
                    rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type), hardware_id=pool.hardware_id,
                    model_execution_version="vllm_0_30_0", fused_embedding=True)
                ids = trace.tensor("input_ids", (1,), "int32", persistent=True) if network.first_stage else None
                positions = trace.tensor("positions", (1,), "int64", persistent=True)
                output = network(ids, positions, intermediate_tensors=incoming)
                if network.last_stage:
                    SymbolicGather(trace, "dummy:selected-hidden", pool.name)(output, 1, indices=(0,))
                trace._nodes = {name: node.with_attributes({**node.attributes, "source_dummy_run": True,
                    "source_actual_request_count": 0, "source_projection_only": True,
                    "model_execution_version": "vllm_0_30_0"}) for name, node in trace._nodes.items()}
                return output
            if pp > 1:
                pipeline = ForwardPipelineGroup(tuple(trace for group in stage_groups for trace in group.traces),
                    namespace=f"{step.step_id}:dp{replica}:dummy", collective_groups=stage_groups if tp > 1 else ())
                _run_vllm_legacy_dummy_stages(pipeline, rows=1, hidden_size=model.hidden_size,
                    max_batched_tokens=1, run_stage=run_dummy)
            else:
                for trace in stage_groups[0].traces:
                    run_dummy(trace, 0, None)
            continue
        if pp > 1 and requests:
            local_ids = tuple(token for request in requests for token in row_ids[request.request_id])
            q_lens = tuple(request.scheduled_tokens for request in requests)
            kv_lens = tuple(request.computed_tokens + request.scheduled_tokens for request in requests)
            slots, capacity = [], 0
            for request in requests:
                slots.extend(range(capacity + request.computed_tokens, capacity + request.computed_tokens + request.scheduled_tokens))
                capacity += request.computed_tokens + request.scheduled_tokens
            stage_groups = tensor_groups[replica * pp:(replica + 1) * pp]
            pipeline = ForwardPipelineGroup(tuple(trace for group in stage_groups for trace in group.traces),
                namespace=f"{step.step_id}:dp{replica}:pipeline", collective_groups=stage_groups if tp > 1 else ())
            def run_stage(trace, stage, incoming):
                group = stage_groups[stage] if tp > 1 else None
                network = Qwen3Model(trace, model, pool.name, q_lens, kv_lens, capacity, tuple(slots),
                    layer_range=ranges[stage], mlp_factories=stage_factories(trace, stage), token_ids=local_ids,
                    group=group, collective=pool.collective_for("all_reduce") if tp > 1 else None,
                    rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type), hardware_id=pool.hardware_id,
                    model_execution_version="vllm_0_30_0", fused_embedding=True)
                ids = trace.tensor("input_ids", (logical_rows[replica],), "int32") if stage == 0 else None
                positions = trace.tensor("positions", (logical_rows[replica],), "int64")
                outputs = network(ids, positions, intermediate_tensors=incoming)
                if network.last_stage:
                    indices = trace.tensor("logits_indices", (len(requests),), "int32")
                    _plain_greedy_target(trace, model, pool.name, group,
                        pool.collective_for("all_gather") if tp > 1 else None, outputs, indices, q_lens,
                        model_execution_version="vllm_0_30_0", sample_logits=sample_logits)
                    return None
                return outputs
            _run_vllm030_pipeline_stages(pipeline, pool, rows=logical_rows[replica], hidden_size=model.hidden_size,
                max_batched_tokens=logical_rows[replica], run_stage=run_stage, mark_source_projection=False)
            continue
        for stage in range(pp):
            group = tensor_groups[replica * pp + stage]
            for trace in group.traces:
                factories = stage_factories(trace, stage)
                local_step = replace(step, requests=requests,
                    finished_request_ids=tuple(request.request_id for request in requests if request.request_id in step.finished_request_ids),
                    preempted_request_ids=())
                positions, slots, capacity = [], [], 0
                for request in requests:
                    positions.extend(range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens))
                    slots.extend(range(capacity + request.computed_tokens, capacity + request.computed_tokens + request.scheduled_tokens))
                    capacity += request.computed_tokens + request.scheduled_tokens
                _trace_dense_forward(model, local_step, input_ids=tuple(token for request in requests for token in row_ids[request.request_id]),
                    positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=capacity,
                    source_profile=source_profile, rank=trace.rank, namespace=trace.namespace, model_type=Qwen3Model,
                    trace=trace, group=group if tp > 1 else None,
                    collective=pool.collective_for("all_reduce") if tp > 1 else None,
                    gather=pool.collective_for("all_gather") if tp > 1 else None,
                    mlp_factories=factories, sample_logits=sample_logits, cast_logits=False, hardware_id=pool.hardware_id,
                    fused_embedding=True, rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
                    model_execution_version="vllm_0_30_0")
    joined.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "source_data_parallel_rows": logical_rows, "source_model_rows_including_dummy": work_rows,
        "source_expert_physical_rows": sum(lane_rows), "actual_provider_bound": False, "normal_capacity_accepted": False,
        "idle_source_scope": "legacy eager one-row full backbone; metadata=None FlashAttention zero output; no KV writes, PP transfer or sampler",
        "dummy_input_values_source": "retained legacy input/position buffers; values unknown, no reset",
        "dummy_intermediate_capacity_scope": "one-row logical views; configured native resident capacity remains unbound"}
    joined.source_scope["dummy_first_input_ids"] = tuple(current_ids[index] if not rows else None for index, rows in enumerate(logical_rows))
    joined.source_scope["dummy_input_history_bound"] = source_program_cache is not None
    if source_program_cache is not None:
        joined.source_scope["dummy_input_values_source"] = "run-owned constructor or last prepared first ID; retained positions remain unknown"
    if source_program_cache is not None:
        source_program_cache.qwen_legacy_inputs[state_key] = _QwenLegacyInputState(pool, model.config_sha256,
            (step.step_id,) if prior_inputs is None else (*prior_inputs.step_ids, step.step_id), current_ids)
    return joined

def _validate_qwen3_moe_source(model, config, *, pipeline=False):
    if model.model_type == "qwen3_moe" and config.tensor_parallel_size > model.num_experts:
        raise NotImplementedError("Qwen3MoE source requires TP <= the number of experts")
    if (model.model_type != "qwen3_moe" or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
            or model.hidden_act != "silu" or model.rope_scaling_type != "default"
            or model.num_shared_experts != 0 or type(model.moe_normalize_topk_probability) is not bool
            or (config.enable_expert_parallel and (config.expert_parallel_size != config.tensor_parallel_size))
            or any(value != 1 for value in (
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
            or (config.pipeline_parallel_size != 1 and not (pipeline and is_qwen3_moe_pipeline_model_scope(
                model, config.tensor_parallel_size, config.pipeline_parallel_size)))):
        raise NotImplementedError("virtual Qwen3 MoE requires BF16, default RoPE, PP/SP/CP/DP1 and EP1 or TP=EP=2")


def _vllm011_qwen3_moe_ep2_source(model, pool):
    return _versioned_qwen3_moe_ep2_source(model, pool, "vllm_0_11_0")


def _versioned_qwen3_moe_ep2_source(model, pool, model_execution_version):
    hardware = {"vllm_0_11_0": ("nvidia_a800", "nvidia_h200"), "vllm_0_24_0": ("metax_c500",), "vllm_0_18_0": ("ascend_910b",)}
    config = pool.parallel_plan.config
    return (model_execution_version in hardware and pool.hardware_id in hardware[model_execution_version]
        and is_validated_qwen3_moe_ep2_model_scope(model)
        and model.num_experts in (2, 128) and config.tensor_parallel_size == 2
        and config.enable_expert_parallel and config.expert_parallel_size == 2
        and len(pool.ranks) == 2 and config.is_rank_local
        and all(value == 1 for value in (config.pipeline_parallel_size, config.sequence_parallel_size,
            config.context_parallel_size, config.data_parallel_size)))


def _vllm030_qwen3_moe_full_source(model, pool, version):
    """Qwen3-MoE BF16 FULL caller, DP1; EP partitions experts, never rows."""
    config = pool.parallel_plan.config
    return (version == "vllm_0_30_0" and pool.hardware_id == "nvidia_h200"
        and model.model_type == "qwen3_moe" and model.compute_dtype == model.weight_dtype == "bf16"
        and model.head_dim == 128 and is_validated_qwen3_moe_ep2_model_scope(model)
        and config.tensor_parallel_size in (1, 2) and len(pool.ranks) == config.tensor_parallel_size
        and all(value == 1 for value in (config.pipeline_parallel_size, config.data_parallel_size,
            config.sequence_parallel_size, config.context_parallel_size))
        and ((config.tensor_parallel_size == 1 and not config.enable_expert_parallel)
            or (config.tensor_parallel_size == config.expert_parallel_size == 2 and config.enable_expert_parallel)))


def trace_qwen3_moe_forward(model, step, pool, expert_routing, *, input_ids, positions,
        slot_mapping, cache_capacity_tokens, source_profile, model_execution_version="vllm_0_30_0", sample_logits=True,
        cast_logits=False, graph_token_shape=None, graph_eager_kinds=(), source_namespace=None,
        graph_inputs=None, fa3_full_aot_source=None):
    """Full Qwen3-MoE stack on the same GQA/residual/target/storage primitives."""
    from functools import partial
    config = pool.parallel_plan.config
    _validate_qwen3_moe_source(model, config)
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
        raise NotImplementedError("Qwen3 MoE requires its version-bound source")
    if model_execution_version == "vllm_0_11_0" and (
            pool.hardware_id not in ("nvidia_a800", "nvidia_h200")
            or (pool.hardware_id == "nvidia_h200" and not _vllm011_qwen3_moe_ep2_source(model, pool))
            or (not _vllm011_qwen3_moe_ep2_source(model, pool) and (config.tensor_parallel_size != 1
                or len(pool.ranks) != 1 or config.enable_expert_parallel))
            or source_profile != "cuda_eager_inplace"
            or expert_routing.source_kind not in ("explicit_assumption", "profile_observation")
            #0.11 uses a positive-byte softmax workspace outside this fused
            # power-of-two domain; that storage is not silently omitted.
            or model.num_experts > 256 or model.num_experts & (model.num_experts - 1)):
        raise NotImplementedError("vLLM0.11 Qwen3 MoE requires A800 TP1/EP-off or A800/H200 validated TP=EP2, explicit seeded or profiled routing and power-of-two experts <=256")
    if model_execution_version in ("vllm_0_18_0", "vllm_0_24_0") and not _versioned_qwen3_moe_ep2_source(model, pool, model_execution_version):
        raise NotImplementedError("C5000.24 MoE requires its validated TP=EP2 ordinary source")
    full_source = _vllm030_qwen3_moe_full_source(model, pool, model_execution_version)
    if graph_inputs is not None or fa3_full_aot_source is not None:
        from .graph_padding import VLLMDenseGraphInputs
        if (not full_source or not isinstance(graph_inputs, VLLMDenseGraphInputs)
                or graph_token_shape is None or graph_token_shape.runtime_mode not in ("FULL", "PIECEWISE")
                or graph_inputs.request_ids != tuple(request.request_id for request in step.requests)
                or graph_token_shape.logical_tokens != step.token_count
                or source_profile != "cuda_eager_inplace"
                or sample_logits is not True or cast_logits):
            raise ValueError("MoE FULL/PIECEWISE requires paired retained caller state and exact H200 BF16 source")
        if (len(graph_inputs.retained_padding_token_ids) != graph_token_shape.padding_tokens
                or any(token >= model.vocab_size for token in graph_inputs.retained_padding_token_ids)):
            raise ValueError("MoE FULL/PIECEWISE retained input tail must cover the actual physical rows")
        if graph_token_shape.runtime_mode == "FULL":
            if (graph_inputs.fa3_full_capture_id is None or fa3_full_aot_source is None
                    or graph_eager_kinds or any(request.phase != "decode" or request.scheduled_tokens != 1
                        for request in step.requests)):
                raise ValueError("MoE FULL requires paired Q1 retained caller state and AOT capture bounds")
            state, max_capture = fa3_full_aot_source
            if (state != graph_inputs or type(max_capture) is not int
                    or max_capture < graph_token_shape.model_input_tokens):
                raise ValueError("MoE FULL retained caller and global AOT capture bound must agree")
        elif (graph_inputs.fa3_full_capture_id is not None or fa3_full_aot_source is not None
                or set(graph_eager_kinds) != {"flash_attn", "kv_update"}):
            raise ValueError("MoE PIECEWISE requires eager Attention/KV and cannot borrow FULL AOT state")
    if graph_token_shape is not None and not (full_source and graph_inputs is not None) and (
            model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0")
            or not _versioned_qwen3_moe_ep2_source(model, pool, model_execution_version) or sample_logits is not True):
        raise NotImplementedError("MoE capture requires its declared versioned target source and retained inputs")
    rows = (sum(request.scheduled_tokens for request in step.requests) if graph_token_shape is None
        else graph_token_shape.model_input_tokens)
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
    # Routing always resolves against the actual ScheduleStep and invocation
    # identity above. Only source object names may use a stable program key.
    if source_namespace is not None and (type(source_namespace) is not str or not source_namespace):
        raise ValueError("source namespace must be an explicit nonempty string")
    source_step = step if source_namespace is None else replace(step, step_id=source_namespace)
    # The ordinary eager source follows v1/worker/gpu_model_runner.py,
    # whose forward context passes no padding mask. GPU-v2, capture and
    # distributed expert paths must retain their separate input contracts.
    topk_context = (VLLM030_GPU_MODEL_RUNNER_TOPK_CONTEXT
        if pool.hardware_id == "nvidia_h200" and model_execution_version == "vllm_0_30_0"
        and source_profile == "cuda_eager_inplace" and graph_token_shape is None
        and config.tensor_parallel_size == config.expert_parallel_size == 1 else None)
    factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
        ep_size=config.expert_parallel_size, model_execution_version=model_execution_version,
        topk_forward_context=topk_context)
        for resolution, invocation in zip(resolutions, invocations, strict=True))
    source_options = dict(hardware_id=pool.hardware_id,
        fused_embedding=(pool.hardware_id not in ("metax_c500", "ascend_910b")
            and not (model_execution_version == "vllm_0_11_0" and config.tensor_parallel_size > 1)),
        rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
        model_execution_version=model_execution_version, graph_token_shape=graph_token_shape,
        graph_eager_kinds=graph_eager_kinds, fa3_full_aot_source=fa3_full_aot_source)
    if config.tensor_parallel_size == 1:
        return _trace_dense_forward(model, source_step, input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_profile=source_profile, rank=pool.ranks[0],
            namespace=source_step.step_id, model_type=Qwen3Model, mlp_factories=factories,
            sample_logits=sample_logits, cast_logits=cast_logits, **source_options)
    if sample_logits is not True and not (pool.hardware_id == "nvidia_h200"
            and model_execution_version == "vllm_0_30_0" and config.tensor_parallel_size == 2
            and (not config.enable_expert_parallel or config.expert_parallel_size == 2
                and is_validated_qwen3_moe_ep2_model_scope(model))):
        raise NotImplementedError("Qwen3 MoE static LM-head-only source requires TP1 or existing H200 TP2/validated TP=EP2 source")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{source_step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=source_step.step_id)
    for trace in traces:
        _trace_dense_forward(model, source_step, input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_profile=source_profile, rank=trace.rank,
            namespace=trace.namespace, model_type=Qwen3Model, trace=trace, group=group,
            collective=pool.collective_for("all_reduce"), gather=pool.collective_for("all_gather"),
            mlp_factories=factories, sample_logits=sample_logits, cast_logits=cast_logits,
            retained_padding_token_ids=None if graph_inputs is None else graph_inputs.retained_padding_token_ids,
            **source_options)
        if graph_inputs is not None:
            capture = trace.source_capture
            capture["padding_ids"] = "caller_declared_retained_source_values"
            capture["padding_positions"] = 0
            capture["retained_padding_inputs"] = {
                "request_ids": graph_inputs.request_ids, "source_kind": graph_inputs.source_kind,
                "source_id": graph_inputs.source_id, "token_ids": graph_inputs.retained_padding_token_ids}
    if graph_token_shape is not None:
        _remap_collective_capture_nodes(group)
    return group


class KimiK3MLP:
    """Ordinary SP-off KimiMLP, with an explicit dense or shared reduction."""

    def __init__(self, trace, model, pool, prefix, *, shared=False, tp_size=1,
                 reduce_results=True, reduction_group=None, reduction_config=None):
        width = model.moe_intermediate_size * model.num_shared_experts if shared else model.intermediate_size
        if (type(tp_size) is not int or tp_size < 1 or width % tp_size or type(reduce_results) is not bool
                or reduce_results and tp_size > 1 and (not isinstance(reduction_group, ForwardCollectiveGroup)
                    or len(reduction_group.traces) != tp_size)
                or (tp_size == 1 or not reduce_results) and (reduction_group is not None or reduction_config is not None)):
            raise ValueError("K3 MLP requires whole TP intermediate shards and its explicit output reduction")
        width //= tp_size
        self.gate_up_proj = SymbolicLinear(trace, prefix + ":gate-up", pool, model.hidden_size, 2 * width)
        self.act_fn = SymbolicK3SiTU(trace, prefix + ":situ", pool, model)
        projection = SymbolicLinear(trace, prefix + ":down", pool, width, model.hidden_size)
        self.down_proj = (projection if reduction_group is None else
            SymbolicRowParallelLinear(projection, reduction_group, reduction_config))

    def __call__(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


@dataclass(frozen=True)
class _K3MoEDataContext:
    group: ForwardCollectiveGroup
    rank: int
    rows: tuple
    dispatch: object
    combine: object
    sequence_parallel: bool = False
    tensor_size: int = 1
    tensor_rank: int = 0

    @property
    def data_size(self):
        return len(self.rows) // self.tensor_size if self.sequence_parallel else len(self.rows)

    @property
    def local_rows(self):
        return self.rows[self.rank * self.tensor_size + self.tensor_rank if self.sequence_parallel else self.rank]


@dataclass(frozen=True)
class _K3SequenceContext:
    group: ForwardCollectiveGroup
    gather_config: object
    combine_config: object
    rows: int
    lane: int

    @property
    def shard_rows(self):
        return -(-self.rows // len(self.group.traces))

    def pad(self, trace, hidden, *, pool, name):
        from .traced_operators import SymbolicFillZero
        padded_rows = self.shard_rows * len(self.group.traces)
        if hidden.shape[0] != self.rows:
            raise ValueError("K3 SP requires the actual unpadded request rows")
        if padded_rows == self.rows:
            return hidden
        padded = trace.tensor(trace.unique_name(name + ":padded"), (padded_rows, hidden.shape[1]), hidden.dtype)
        SymbolicCopy(trace, name + ":copy", pool)(hidden, padded.narrow(0, 0, self.rows))
        SymbolicFillZero(trace, name + ":pad-zero", pool)(padded.narrow(0, self.rows, padded_rows - self.rows))
        return padded

    def shard(self, trace, hidden, *, pool):
        return self.pad(trace, hidden, pool=pool, name="sp-embedding").narrow(0, self.lane * self.shard_rows, self.shard_rows)

    def gather(self, hidden, *, pool, name):
        if hidden.shape[0] != self.shard_rows:
            raise ValueError("K3 SP gather requires this lane's complete row shard")
        full = self.group.call(name, hidden, pool_name=pool, config=self.gather_config, inplace=False, dim=0)
        return full if full.shape[0] == self.rows else full.narrow(0, 0, self.rows)

    def scatter(self, trace, hidden, *, pool, name):
        padded = self.pad(trace, hidden, pool=pool, name=name)
        return self.group.call_variable(name + ":reduce-scatter", trace, padded, pool_name=pool,
            config=self.combine_config, rows_by_rank=(self.shard_rows,) * len(self.group.traces),
            row_shape=(hidden.shape[1],), dtype=hidden.dtype, group_dimension="tensor")


@dataclass(frozen=True)
class _K3ReplicaContext:
    traces: tuple
    tensor_groups: tuple
    data_groups: tuple
    rows: tuple
    rank: int
    resolutions: tuple
    invocations: tuple
    expert_parallel_size: int
    model_sequence_parallel: bool = False
    expert_groups: tuple = ()
    request_rows: tuple = ()


class KimiK3MoE:
    """Modular Marlin with native EP or an explicit expert-TP tail."""

    def __init__(self, trace, model, pool, prefix, *, resolution, invocation_id, layer_index,
                 tp_size=1, reduction_group=None, reduction_config=None, latent_tail_tier=None,
                 expert_parallel_size=1, data_context=None):
        if data_context is not None and not isinstance(data_context, _K3MoEDataContext):
            raise TypeError("K3 DP experts require their normalized source context")
        data_size = 1 if data_context is None else data_context.data_size
        fused_tail = expert_parallel_size == 1 and tp_size * data_size > 1
        if (type(tp_size) is not int or tp_size < 1
                or tp_size == 1 and (reduction_group is not None or reduction_config is not None)
                or not fused_tail and latent_tail_tier is not None
                or fused_tail and latent_tail_tier not in ("ALLREDUCE_OVERLAP", "COLUMN_PARALLEL")):
            raise ValueError("K3 MoE requires an explicit expert-TP tail or native EP/TP1 tail")
        if data_size > 1 and fused_tail and latent_tail_tier != "ALLREDUCE_OVERLAP":
            raise NotImplementedError("K3 DP expert-TP source currently binds the full replicated ALLREDUCE_OVERLAP tail")
        if tp_size > 1:
            from src.serving.deployment.deployment_config import CollectiveConfig
            if (not isinstance(reduction_group, ForwardCollectiveGroup)
                    or len(reduction_group.traces) != tp_size
                    or not any(trace is item for item in reduction_group.traces)
                    or not isinstance(reduction_config, CollectiveConfig) or reduction_config.operation != "all_reduce"):
                raise ValueError("K3 MoE requires this rank's complete TP all-reduce group")
        self.trace, self.pool, self.prefix = trace, pool, prefix
        self.group, self.collective, self.tail = reduction_group, reduction_config, latent_tail_tier
        self.invocation_id, self.tp_size = invocation_id, tp_size
        self.native_ep_tail = expert_parallel_size > 1
        self.sequence_parallel = data_context is not None and data_context.sequence_parallel
        self.tensor_rank = 0 if reduction_group is None else reduction_group.participants.index(trace.rank)
        if data_context is not None:
            if (not isinstance(data_context, _K3MoEDataContext)
                    or not isinstance(data_context.group, ForwardCollectiveGroup)
                    or type(data_context.rank) is not int or not 0 <= data_context.rank < len(data_context.group.traces)
                    or data_context.group.participants[data_context.rank * tp_size + self.tensor_rank
                        if data_context.sequence_parallel else data_context.rank] != trace.rank
                    or len(data_context.rows) != len(data_context.group.traces)
                    or any(type(rows) is not int or rows < 0 for rows in data_context.rows)
                    or not sum(data_context.rows) or data_context.dispatch.operation != "all_gatherv"
                    or data_context.combine.operation != "reduce_scatterv"):
                raise ValueError("K3 DP experts require this rank's complete row owners and dispatch/combine")
        self.data_context = data_context
        latent = model.kimi_k3_semantics.routed_expert_hidden_size
        self.model_latent = latent
        self.gate = SymbolicLinear(trace, prefix + ":router", pool, model.hidden_size,
            model.num_experts, kind=WorkloadKind.ROUTER, output_dtype="fp32")
        self.down = SymbolicLinear(trace, prefix + ":latent-down", pool, model.hidden_size, latent)
        self.router = SymbolicK3Routing(trace, prefix + ":route", pool, model)
        self.shared = KimiK3MLP(trace, model, pool, prefix + ":shared", shared=True,
            tp_size=1 if self.sequence_parallel else tp_size, reduce_results=False)
        self.experts = SymbolicK3Experts(trace, prefix + ":experts", pool, model,
            resolution=resolution, invocation_id=invocation_id, layer_index=layer_index,
            tp_size=tp_size, tensor_rank=self.tensor_rank, expert_parallel_size=expert_parallel_size,
            data_parallel_size=1 if data_context is None else data_context.data_size,
            data_rank=0 if data_context is None else data_context.rank)
        self.norm = SymbolicRMSNorm(trace, prefix + ":latent-norm", pool, latent, model.kimi_k3_semantics.rms_norm_eps)
        self.up = SymbolicLinear(trace, prefix + ":latent-up", pool, latent, model.hidden_size)
        self.add = SymbolicAdd(trace, prefix + ":shared-add", pool)

    def _reduce(self, name, value):
        return self.group.call(self.prefix + ":" + name, value, pool_name=self.pool,
            config=self.collective, inplace=False, moe_invocation_id=self.invocation_id)

    def _column_tail(self, latent, shared):
        # The checkpoint up weight stays replicated. Only its output-row slice
        # is read by addmm_, which mutates this rank's slice of shared partial.
        width = shared.shape[-1] // self.tp_size
        shard = shared.narrow(1, self.tensor_rank * width, width)
        weight = self.trace.parameter_view(self.up.name + ":weight",
            (shared.shape[-1], latent.shape[-1]), latent.dtype)
        weight_shard = self.trace._handle(weight).narrow(0, self.tensor_rank * width, width)
        node = self.up.node(WorkloadKind.LINEAR,
            {"b": 1, "m": latent.shape[0], "n": width, "k": latent.shape[-1]}, latent.dtype, "gemm", {
                "source_component": "vllm_0_30_kimi_k3_latent_tp_tail",
                "source_operation": "addmm_", "latent_tail_tier": self.tail,
                "has_bias": False, "alpha": 1.0, "beta": 1.0,
                "tensor_parallel_size": self.tp_size, "tensor_rank": self.tensor_rank,
                "replicated_weight_shape": weight.shape, "symbolic_workspace_complete": False})
        self.trace.call(node, (latent, shard), shape=shard.shape, dtype=shard.dtype,
            out=shard, writes=(shared,), parameters=(weight_shard.view,))
        return self._reduce("shared-all-reduce", shared)

    def _runner(self, hidden_states, router_logits, shared_experts_input):
        shared_output = self.shared(shared_experts_input) if shared_experts_input is not None else None
        weights, ids = self.router(router_logits) if router_logits is not None else (None, None)
        if self.data_context is not None:
            context = self.data_context
            def dispatch(name, tensor, width, dtype):
                return context.group.call_variable(self.prefix + ":dp-dispatch-" + name,
                    self.trace, tensor, pool_name=self.pool, config=context.dispatch,
                    rows_by_rank=context.rows, row_shape=(width,), dtype=dtype,
                    group_dimension="expert" if self.sequence_parallel else "data")
            hidden_states = dispatch("latent", hidden_states, self.model_latent, "bf16")
            weights = dispatch("weights", weights, self.experts.model.num_experts_per_tok, "fp32")
            ids = dispatch("ids", ids, self.experts.model.num_experts_per_tok, "int32")
        fused_output = self.experts(hidden_states, weights, ids)
        if self.data_context is not None:
            context = self.data_context
            fused_output = context.group.call_variable(self.prefix + ":dp-combine", self.trace, fused_output,
                pool_name=self.pool, config=context.combine, rows_by_rank=context.rows,
                row_shape=(self.model_latent,), dtype="bf16", moe_invocation_id=self.invocation_id,
                group_dimension="expert" if self.sequence_parallel else "data")
            if context.local_rows == 0:
                return None
            copied = self.trace.tensor(self.trace.unique_name(self.prefix + ":dp-finalize-output"),
                fused_output.shape, fused_output.dtype)
            fused_output = SymbolicCopy(self.trace, self.prefix + ":dp-finalize-copy", self.pool)(fused_output, copied)
        if self.group is not None and not self.sequence_parallel:
            fused_output = self._reduce("latent-all-reduce", fused_output)
            if self.native_ep_tail:
                shared_output = self._reduce("shared-all-reduce", shared_output)
        fused_output = self.norm(fused_output)
        if self.native_ep_tail and self.group is not None and not self.sequence_parallel:
            # Base MoERunner reduces shared output synchronously before the
            # nonlinear output transform; this is not the aux-stream tier.
            self.trace._edges.append(WorkloadEdge(self.trace._writers[shared_output.view.allocation_id],
                self.trace._writers[fused_output.view.allocation_id], "barrier"))
        if self.tail == "COLUMN_PARALLEL":
            return self._column_tail(fused_output, shared_output)
        fused_output, _ = self.up(fused_output)
        if self.group is not None and not self.native_ep_tail and not self.sequence_parallel:
            # Independent data branches permit the source's aux-stream
            # shared all-reduce to overlap the replicated up-projection.
            shared_output = self._reduce("shared-all-reduce", shared_output)
        return self.add(shared_output, fused_output)

    def __call__(self, hidden_states):
        if hidden_states is None:
            if self.data_context is None or self.data_context.local_rows != 0:
                raise ValueError("K3 idle MoE requires an actual empty DP row owner")
            return self._runner(None, None, None)
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view_as_shape(-1, hidden_size)
        router_output, _ = self.gate(hidden_states)
        routed_hidden_states, _ = self.down(hidden_states)
        final_hidden_states = self._runner(routed_hidden_states, router_output, hidden_states)
        return final_hidden_states.view_as_shape(num_tokens, hidden_size)


class KimiK3KDADecode:
    """Explicit eager Triton/native KDA source without tensor execution."""

    def __init__(self, trace, model, pool, prefix, *, rows, state_indices, state_pool_slots,
                 tp_size=1, reduction_group=None, reduction_config=None, reduce_results=True,
                 dummy_run=False, decode_backend="triton"):
        from .shape_inference import infer_gemm_dimensions
        from src.serving.deployment.deployment_config import CollectiveConfig
        self.trace, self.prefix, self.pool = trace, prefix, pool
        source = model.kimi_k3_semantics
        if (type(tp_size) is not int or tp_size < 1 or type(reduce_results) is not bool or source.kda_num_heads % tp_size
                or tp_size == 1 and (reduction_group is not None or reduction_config is not None)
                or tp_size > 1 and (not isinstance(reduction_group, ForwardCollectiveGroup)
                    or len(reduction_group.traces) != tp_size
                    or not any(trace is item for item in reduction_group.traces)
                    or not isinstance(reduction_config, CollectiveConfig) or reduction_config.operation != "all_reduce")):
            raise ValueError("K3 KDA requires whole local heads and its complete ordinary TP reduction group")
        self.heads, self.dim = source.kda_num_heads // tp_size, source.kda_head_dim
        if (type(dummy_run) is not bool or decode_backend not in ("triton","native")
                or dummy_run and (rows!=1 or state_indices!=(0,))
                or not dummy_run and (not isinstance(state_indices,tuple) or not state_indices
                    or any(type(value) is not int or not 0<value<state_pool_slots for value in state_indices))):
            raise ValueError("K3 KDA dummy requires exactly one reserved null-state row and an explicit backend")
        if decode_backend=="native" and self.heads not in (12,24,48,96):
            raise NotImplementedError("K3 native KDA requires a source-supported local head count; no automatic backend substitution")
        self.dummy_run,self.decode_backend=dummy_run,decode_backend
        self.width, self.rows = self.heads * self.dim, rows
        _, packed, hidden = infer_gemm_dimensions(model, "in_proj_qkvgfab", rows, tp_size=tp_size)
        self.padding = packed - (4 * self.width + self.dim + self.heads)
        self.in_proj_qkvgfab = SymbolicLinear(trace, prefix + ":packed", pool, hidden, packed)
        self.f_b_proj = SymbolicLinear(trace, prefix + ":f-b", pool, self.dim, self.width)
        projection = SymbolicLinear(trace, prefix + ":out", pool, self.width, hidden)
        self.o_proj = (projection if reduction_group is None or not reduce_results else
            SymbolicRowParallelLinear(projection, reduction_group, reduction_config))
        self.conv_state = trace.tensor(prefix + ":conv-state",
            (state_pool_slots, 3 * self.width, source.short_conv_kernel_size - 1), "bf16", persistent=True)
        self.recurrent_state = trace.tensor(prefix + ":recurrent-state",
            (state_pool_slots, self.heads, self.dim, self.dim), "fp32", persistent=True)
        self.indices = trace.tensor(prefix + ":state-indices", (len(state_indices),), "int32")
        self.stages = {stage: SymbolicK3Stage(trace, prefix + ":" + stage, pool,
            variant="kimi_k3_kda_decode", shape={"rows": rows, "heads": self.heads,
                "head_dim": self.dim, "elements": rows * self.width}, attributes={
                    "source_component": "vllm_0_30_kimi_k3_triton_decode", "stage": stage,
                    "state_indices": state_indices, "state_pool_slots": state_pool_slots,
                    "gate_lower_bound": source.gate_lower_bound})
            for stage in ("convolution", "recurrent", "output_copy", "output_norm")}

    def _forward(self, *, mixed_qkv, g1, g2, beta, core_attn_out):
        if self.decode_backend=="native":
            # fused_kda_decode has no null-slot guard: dummy slot0 retains
            # real conv/recurrent reads and writes, disjoint from requests.
            conv=self.conv_state.narrow(0,0,1) if self.dummy_run else self.conv_state
            recurrent=self.recurrent_state.narrow(0,0,1) if self.dummy_run else self.recurrent_state
            stage=SymbolicK3Stage(self.trace,self.prefix+":native-fused-decode",self.pool,
                variant="kimi_k3_kda_decode",shape=self.stages['recurrent'].shape,
                attributes={**self.stages['recurrent'].attributes,
                    "source_component":"vllm_0_30_kimi_k3_native_decode","stage":"native_fused_decode",
                    "reserved_null_state_access":self.dummy_run})
            stage(mixed_qkv,g1,g2,beta,conv,recurrent,self.indices,
                shape=core_attn_out.shape,out=core_attn_out,writes=(conv,recurrent))
            return
        if self.dummy_run:
            from .workload_graph import WorkloadEdge
            # causal_conv1d's state0 branch reads only indices and leaves
            # its torch.empty output untouched. Recurrence then writes zero
            # without reading q/k/v, gates or either state allocation.
            packed_conv_out=self.trace.tensor(self.trace.unique_name(self.prefix+":null-conv-out"),mixed_qkv.shape,'bf16')
            conv=self.stages['convolution']
            node=conv.node(WorkloadKind.VECTOR,conv.shape,'bf16',conv.variant,
                {**conv.attributes,'source_null_state_skip':True,'symbolic_workspace_complete':False})
            self.trace.call_effect(node,(self.indices,),controls=(mixed_qkv,))
            stage=self.stages['recurrent']
            zero=stage.node(WorkloadKind.VECTOR,stage.shape,'bf16',stage.variant,
                {**stage.attributes,'source_null_state_skip':True,'source_output_constant':0,'symbolic_workspace_complete':False})
            recurrent_output=self.trace.call(zero,(self.indices,),shape=core_attn_out.shape,dtype='bf16',
                controls=(packed_conv_out,g1,beta))
            self.trace._edges.append(WorkloadEdge(node.node_id,zero.node_id,'barrier'))
            self.stages['output_copy'](recurrent_output,shape=core_attn_out.shape,out=core_attn_out)
            self.stages['output_norm'](core_attn_out,g2,shape=core_attn_out.shape,out=core_attn_out,
                workspace=(('rstd',(self.rows*self.heads,),'fp32'),))
            return
        # source non_spec_token_start=None: packed recurrence returns fresh
        # output, assignment copies into core, then gated norm aliases core.
        packed_conv_out = self.stages["convolution"](mixed_qkv, self.conv_state, self.indices,
            shape=mixed_qkv.shape, writes=(self.conv_state,))
        recurrent_output = self.stages["recurrent"](packed_conv_out, g1, beta, self.recurrent_state, self.indices,
            shape=core_attn_out.shape, writes=(self.recurrent_state,))
        self.stages["output_copy"](recurrent_output, shape=core_attn_out.shape, out=core_attn_out)
        self.stages["output_norm"](core_attn_out, g2, shape=core_attn_out.shape, out=core_attn_out,
            workspace=(("rstd", (self.rows * self.heads,), "fp32"),))

    def __call__(self, hidden_states, positions):
        num_tokens = hidden_states.shape[0]
        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        split_sizes = [3 * self.width, self.width, self.dim, self.heads]
        if self.padding:
            split_sizes.append(self.padding)
        projected = projected_qkvgfab.split(split_sizes, dim=-1)
        mixed_qkv, g_proj_states, f_a, beta = projected[:4]
        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = g1.view_as_shape(1, num_tokens, self.heads, self.dim)
        g2 = g_proj_states.view_as_shape(num_tokens, self.heads, self.dim)
        core_attn_out = self.trace.tensor(self.prefix + ":core-out", (1, num_tokens, self.heads, self.dim), "bf16")
        self._forward(mixed_qkv=mixed_qkv, g1=g1, g2=g2, beta=beta, core_attn_out=core_attn_out)
        core_attn_out = core_attn_out.view_as_shape(num_tokens, self.width)
        return self.o_proj(core_attn_out)[0]


class KimiK3KDAPrefill(KimiK3KDADecode):
    """Fresh or mixed chunk program, using the source's three conv calls."""

    def __init__(self, trace, model, pool, prefix, *, query_lengths, initial_flags, **inputs):
        super().__init__(trace, model, pool, prefix, **inputs)
        self.sequences = len(query_lengths)
        starts = [0]
        for length in query_lengths:
            starts.append(starts[-1] + length)
        self.query_start = trace.tensor(prefix + ":query-start", (self.sequences + 1,), "int32")
        self.initial_mask = trace.tensor(prefix + ":has-initial-state", (self.sequences,), "bool")
        phase = ("context_prefill" if initial_flags is True else "fresh_prefill") if self.sequences == 1 else "mixed"
        attributes = self.stages["convolution"].attributes
        self.stages = {stage: SymbolicK3Stage(trace, prefix + ":" + stage, pool,
            variant="kimi_k3_kda_"+phase if self.sequences == 1 and initial_flags is True
                else "kimi_k3_kda_prefill" if self.sequences == 1 else "kimi_k3_kda_mixed",
            shape={"rows": self.rows, "heads": self.heads,
                "head_dim": self.dim, "elements": self.sequences * self.heads * self.dim**2
                    if stage in ("initial_state", "final_state_scatter") else self.rows * self.width},
            attributes={**attributes, "source_component": "vllm_0_30_kimi_k3_triton_" + phase, "stage": stage,
                "query_start_loc": tuple(starts), "has_initial_state": initial_flags})
            for stage in ("q_convolution", "k_convolution", "v_convolution", "initial_state", "chunk",
                "final_state_scatter", "output_copy", "output_norm")}

    def _prefill_conv(self, x, state, stage):
        # empty_like compacts the non-dense [channel,token] input into its
        # column-major layout. The wrapper's transpose is dense token-major;
        # use one backing and expose the kernel's transposed output view.
        output = self.trace.tensor(self.trace.unique_name(self.prefix + ":" + stage + ":out"),
            (self.rows, self.width), "bf16")
        self.stages[stage](x.transpose(0, 1), state, self.indices, self.query_start, self.initial_mask,
            shape=(self.width, self.rows), out=output.transpose(0, 1), writes=(state,))
        return output

    def _forward(self, *, mixed_qkv, g1, g2, beta, core_attn_out):
        q_ns, k_ns, v_ns = mixed_qkv.split((self.width,) * 3)
        q_state = self.conv_state.narrow(1, 0, self.width)
        k_state = self.conv_state.narrow(1, self.width, self.width)
        v_state = self.conv_state.narrow(1, 2 * self.width, self.width)
        q_ns = self._prefill_conv(q_ns, q_state, "q_convolution")
        k_ns = self._prefill_conv(k_ns, k_state, "k_convolution")
        v_ns = self._prefill_conv(v_ns, v_state, "v_convolution")
        q_ns, k_ns, v_ns = (x.view_as_shape(1, self.rows, self.heads, self.dim) for x in (q_ns, k_ns, v_ns))
        initial = self.stages["initial_state"](self.recurrent_state, self.indices, self.initial_mask,
            shape=(self.sequences, self.heads, self.dim, self.dim), dtype="fp32")
        stage = self.stages["chunk"]
        node = stage.node(WorkloadKind.VECTOR, stage.shape, "bf16", stage.variant,
            {**stage.attributes, "symbolic_workspace_complete": False})
        final_state = self.trace.tensor(self.trace.unique_name(self.prefix + ":chunk-final-state"), initial.shape, "fp32")
        # chunk_gla_fwd_o_gk uses o=v when out is None. The wrapper's
        # contiguous V already has dense storage; its output is an alias.
        output = self.trace.call(node, (q_ns, k_ns, v_ns, g1, beta, initial, self.query_start),
            shape=v_ns.shape, dtype="bf16", out=v_ns, writes=(final_state,))
        self.stages["final_state_scatter"](final_state, self.indices, shape=self.recurrent_state.shape,
            dtype="fp32", out=self.recurrent_state)
        self.stages["output_copy"](output, shape=core_attn_out.shape, out=core_attn_out, controls=(self.recurrent_state,))
        self.stages["output_norm"](core_attn_out, g2, shape=core_attn_out.shape, out=core_attn_out,
            workspace=(("rstd", (self.rows * self.heads,), "fp32"),))


class KimiK3NoPEDecode:
    """NoPE MLA: fused norm, absorbed Q, fused cache/concat and V-up out=."""

    def __init__(self, trace, model, pool, prefix, *, rows, kv_lens, slots, cache_capacity_tokens,
                 tp_size=1, reduction_group=None, reduction_config=None, reduce_results=True,dummy_run=False):
        from src.serving.deployment.deployment_config import CollectiveConfig
        self.trace, self.model, self.pool, self.prefix = trace, model, pool, prefix
        if (type(tp_size) is not int or tp_size < 1 or type(reduce_results) is not bool or model.num_attention_heads % tp_size
                or tp_size == 1 and (reduction_group is not None or reduction_config is not None)
                or tp_size > 1 and (not isinstance(reduction_group, ForwardCollectiveGroup)
                    or len(reduction_group.traces) != tp_size
                    or not any(trace is item for item in reduction_group.traces)
                    or not isinstance(reduction_config, CollectiveConfig) or reduction_config.operation != "all_reduce")):
            raise ValueError("K3 NoPE requires whole local heads and its complete ordinary TP reduction group")
        self.heads, self.rows = model.num_attention_heads // tp_size, rows
        if type(dummy_run) is not bool or dummy_run and (rows!=1 or kv_lens!=(1,) or slots!=(-1,)):
            raise ValueError("K3 NoPE dummy requires its one-token null-block metadata and PAD write slot")
        self.dummy_run=dummy_run
        self.split_frontend = rows < 512
        qkv = model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim
        gate = self.heads * model.v_head_dim
        self.packed = SymbolicLinear(trace, prefix + ":qkv-a", pool, model.hidden_size,
            qkv if self.split_frontend else qkv + gate)
        self.gate_proj = SymbolicLinear(trace, prefix + ":gate-proj", pool, model.hidden_size, gate)
        self.q_b_proj = SymbolicLinear(trace, prefix + ":q-b", pool, model.q_lora_rank, self.heads * model.head_dim)
        self.absorb = SymbolicBatchedLinear(trace, prefix + ":absorb", pool,
            self.heads, model.qk_nope_head_dim, model.kv_lora_rank)
        self.up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool,
            self.heads, model.kv_lora_rank, model.v_head_dim)
        projection = SymbolicLinear(trace, prefix + ":out", pool, gate, model.hidden_size)
        self.o_proj = (projection if reduction_group is None or not reduce_results else
            SymbolicRowParallelLinear(projection, reduction_group, reduction_config))
        self.cache = trace.tensor(prefix + ":cache", (cache_capacity_tokens, 576), "bf16", persistent=True)
        self.slots = trace.tensor(prefix + ":slots", (rows,), "int64")
        self.attention = SymbolicMLALatentDecode(trace, prefix + ":attention", pool,
            heads=self.heads, kv_lens=kv_lens, cache=self.cache.narrow(0,0,1) if dummy_run else self.cache)
        self.shape = {"rows": rows, "heads": self.heads, "q_lora_rank": model.q_lora_rank,
            "kv_lora_rank": model.kv_lora_rank, "elements": rows * (model.q_lora_rank + model.kv_lora_rank)}
        self.attrs = {"source_component": "vllm_0_30_kimi_k3_nope_decode", "slot_mapping": slots,
            "cache_capacity_tokens": cache_capacity_tokens, "rms_norm_eps": model.kimi_k3_semantics.rms_norm_eps}
        self.variant = "kimi_k3_nope_decode"

    def _stage(self, name, width, *, rows=None, slot_mapping=None):
        rows = self.rows if rows is None else rows
        attributes = {**self.attrs, "stage": name}
        if slot_mapping is not None:
            attributes["slot_mapping"] = slot_mapping
        return SymbolicK3Stage(self.trace, self.prefix + ":" + name, self.pool,
            variant=self.variant, shape={**self.shape, "rows": rows, "elements": rows * width}, attributes=attributes)

    def _apply_q_lora_attention(self, positions, hidden_states, qkv_lora):
        model, rows = self.model, hidden_states.shape[0]
        q_c, kv_c, k_pe = qkv_lora.split([model.q_lora_rank, model.kv_lora_rank, model.qk_rope_head_dim], dim=-1)
        norm = self._stage("q_kv_norm", model.q_lora_rank + model.kv_lora_rank)
        node = norm.node(WorkloadKind.VECTOR, norm.shape, "bf16", norm.variant,
            {**norm.attributes, "symbolic_workspace_complete": False})
        q_c, kv_c_normed = self.trace.call_outputs(node, (q_c, kv_c), outputs=(
            ("q", q_c.shape, "bf16"), ("kv", kv_c.shape, "bf16")))
        q = self.q_b_proj(q_c)[0].view_as_shape(rows, self.heads, model.head_dim)
        attn_out = self.trace.tensor(self.prefix + ":attn-out", (rows, self.heads * model.v_head_dim), "bf16")
        self._attention(positions, q, kv_c_normed, k_pe.unsqueeze(1), attn_out)
        return attn_out

    def _attention(self, positions, q, kv_c_normed, k_pe, attn_out, *, controls=()):
        model, rows = self.model, q.shape[0]
        q_nope, q_pe = q.split([model.qk_nope_head_dim, model.qk_rope_head_dim], dim=-1)
        ql_nope = self.absorb(q_nope.transpose(0, 1), controls=controls).transpose(0, 1)
        slot_tensor = self.slots if rows == self.rows else self.slots.narrow(0, 0, rows)
        mqa_q = self._stage("query_cache_concat", self.heads * 576, rows=rows,
            slot_mapping=self.attrs["slot_mapping"][:rows])(
            *((ql_nope,q_pe,slot_tensor) if self.dummy_run else (ql_nope,q_pe,kv_c_normed,k_pe,slot_tensor)),
            shape=(rows, self.heads, 576), writes=() if self.dummy_run else (self.cache,),
            controls=(kv_c_normed,k_pe) if self.dummy_run else ())
        latent_out, lse = self.attention(mqa_q)
        self.up(latent_out.transpose(0, 1),
            out=attn_out.view_as_shape(rows, self.heads, model.v_head_dim).transpose(0, 1))

    def __call__(self, positions, hidden_states):
        model = self.model
        if self.split_frontend:
            qkv_lora = self.packed(hidden_states)[0]
            attn_out = self._apply_q_lora_attention(positions, hidden_states, qkv_lora)
            gate = self.gate_proj(hidden_states)[0]
        else:
            qkv_lora, gate = self.packed(hidden_states)[0].split([
                model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim,
                self.heads * model.v_head_dim], dim=-1)
            attn_out = self._apply_q_lora_attention(positions, hidden_states, qkv_lora)
        attn_out = self._stage("output_gate", self.heads * model.v_head_dim)(
            attn_out, gate, shape=attn_out.shape)
        return self.o_proj(attn_out)[0]


class KimiK3NoPEFreshPrefill(KimiK3NoPEDecode):
    """Fresh BF16 NoPE wrapper with the declared supports_out prefill branch."""

    def __init__(self, trace, model, pool, prefix, *, fa3_hardware=None, fa3_length_inputs=None, **inputs):
        super().__init__(trace, model, pool, prefix, **inputs)
        self.fa3_hardware = fa3_hardware
        self.fa3_length_inputs = {} if fa3_length_inputs is None else fa3_length_inputs
        self.kv_b_proj = SymbolicLinear(trace, prefix + ":kv-b", pool, model.kv_lora_rank,
            self.heads * (model.qk_nope_head_dim + model.v_head_dim))
        self.attrs = {**self.attrs, "source_component": "vllm_0_30_kimi_k3_nope_fresh_prefill"}
        self.variant = "kimi_k3_nope_prefill"

    def _attention(self, positions, q, kv_c_normed, k_pe, attn_out):
        self._prefill(q, kv_c_normed, k_pe, attn_out, offset=0)

    def _expand_prefill(self, q, kv_c_normed, k_pe, *, offset):
        model, rows = self.model, q.shape[0]
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view_as_shape(
            rows, self.heads, model.qk_nope_head_dim + model.v_head_dim)
        k_nope, v = kv_nope.split((model.qk_nope_head_dim, model.v_head_dim))
        slot_tensor = self.slots if rows == self.rows else self.slots.narrow(0, offset, rows)
        k = self._stage("key_cache_concat", self.heads * model.head_dim + 576, rows=rows,
            slot_mapping=self.attrs["slot_mapping"][offset:offset + rows])(
            q, k_nope, k_pe, kv_c_normed, slot_tensor, shape=q.shape, writes=(self.cache,))
        return k,v

    def _prefill(self, q, kv_c_normed, k_pe, attn_out, *, offset):
        model,rows=self.model,q.shape[0]
        k,v=self._expand_prefill(q,kv_c_normed,k_pe,offset=offset)
        from .workload_graph import WorkloadNode
        node = WorkloadNode(self.trace.unique_name(self.prefix + ":prefill-attention"), WorkloadKind.FLASH_ATTN,
            self.pool, {"batch": 1, "q_len": rows, "kv_len": rows, "num_query_heads": self.heads,
                "qk_head_dim": model.head_dim, "v_head_dim": model.v_head_dim, "kv_lora_rank": model.kv_lora_rank},
            dtype="bf16", variant="mla_prefill", rank=self.trace.rank, attributes={
                "source_component": "vllm_0_30_symbolic_k3_prefill_attention", "launch_policy": "analytical_launch",
                "causal": True, "kv_dist": "uniform", "kv_layout": "mla", "q_lens": (rows,), "kv_lens": (rows,),
                "full_cuda_graphs": False, "max_cudagraph_capture_size": 0,
                "prefill_output_source": "supports_out", "return_softmax_lse": False,
                "mathematical_program": "bf16_mla_prefill_softmax_v1", "symbolic_workspace_complete": False})
        if self.fa3_hardware is not None:
            return _trace_fa3_prefill_call(self.trace, node, q, k, v, cache=self.cache,
                hardware=self.fa3_hardware, version="vllm_0_30_0", length_inputs=self.fa3_length_inputs,
                metadata_key=("k3_new_tokens",), output=attn_out.view_as_shape(rows,self.heads,model.v_head_dim))
        self.trace.call(node, (q, k, v), shape=(rows, self.heads, model.v_head_dim), dtype="bf16",
            out=attn_out.view_as_shape(rows, self.heads, model.v_head_dim), controls=(self.cache,))


class KimiK3NoPEContextPrefill(KimiK3NoPEFreshPrefill):
    """Actual NoPE key/cache concat plus existing context FA3 and merge."""

    def __init__(self,trace,model,pool,prefix,*,context_length,context_chunk_rows,module_path,**inputs):
        super().__init__(trace,model,pool,prefix,**inputs)
        self.context_backend=MLAContextPrefillBackend(trace,model,pool,prefix+":context",
            heads=self.heads,q_lens=(self.rows,),cache=self.cache,module_path=module_path,
            context_lengths=(context_length,),context_offsets=(0,),context_chunk_rows=context_chunk_rows,
            fa3_hardware=self.fa3_hardware,
            fa3_length_inputs=self.fa3_length_inputs)

    def _attention(self,positions,q,kv_c_normed,k_pe,attn_out):
        key,value=self._expand_prefill(q,kv_c_normed,k_pe,offset=0)
        self.context_backend(q,kv_c_normed,k_pe,attn_out,expanded_kv=(key,value))


class KimiK3NoPEMixed(KimiK3NoPEFreshPrefill):
    def __init__(self, trace, model, pool, prefix, **inputs):
        super().__init__(trace, model, pool, prefix, **inputs)
        self.attrs = {**self.attrs, "source_component": "vllm_0_30_kimi_k3_nope_mixed"}
        self.variant = "kimi_k3_nope_mixed"
        self.attention = SymbolicMLALatentDecode(trace, prefix + ":attention", pool,
            heads=self.heads, kv_lens=inputs["kv_lens"][:1], cache=self.cache)

    def _attention(self, positions, q, kv_c_normed, k_pe, attn_out):
        prefill_out = attn_out.narrow(0, 1, self.rows - 1)
        self._prefill(q.narrow(0, 1, self.rows - 1), kv_c_normed.narrow(0, 1, self.rows - 1),
            k_pe.narrow(0, 1, self.rows - 1), prefill_out, offset=1)
        KimiK3NoPEDecode._attention(self, positions.narrow(0, 0, 1), q.narrow(0, 0, 1),
            kv_c_normed.narrow(0, 0, 1), k_pe.narrow(0, 0, 1), attn_out.narrow(0, 0, 1), controls=(prefill_out,))


class KimiK3DecoderLayer:
    def __init__(self, trace, model, pool, prefix, *, layer, attention, resolution, invocation_id, residual_capacity,
                 tp_size=1, reduction_group=None, reduction_config=None, latent_tail_tier=None,
                 expert_parallel_size=1, data_context=None, sequence_context=None):
        size = model.kimi_k3_semantics.attention_residual_block_size
        self.is_block_write_layer = layer % size == 0
        blocks = -(-layer // size)
        self.pre = SymbolicK3AttentionResidual(trace, prefix + ":pre-attn", pool, model,
            blocks=blocks, write=layer // size if self.is_block_write_layer else -1, capacity=residual_capacity)
        self.post = SymbolicK3AttentionResidual(trace, prefix + ":post-attn", pool, model,
            blocks=blocks + self.is_block_write_layer, capacity=residual_capacity)
        self.self_attn = attention
        self.sequence_context, self.trace, self.pool, self.prefix = sequence_context, trace, pool, prefix
        self.mlp = (KimiK3MoE(trace, model, pool, prefix + ":moe", resolution=resolution,
            invocation_id=invocation_id, layer_index=layer, tp_size=tp_size, reduction_group=reduction_group,
            reduction_config=reduction_config, latent_tail_tier=latent_tail_tier,
            expert_parallel_size=expert_parallel_size, data_context=data_context) if model.layer_uses_moe(layer)
            else KimiK3MLP(trace, model, pool, prefix + ":mlp", tp_size=tp_size if sequence_context is None else 1,
                reduction_group=reduction_group if sequence_context is None else None,
                reduction_config=reduction_config if sequence_context is None else None))

    def __call__(self, positions, hidden_states, residual, prefix_sum):
        hidden_states = self.pre(prefix_sum, hidden_states, residual)
        if self.sequence_context is not None:
            hidden_states = self.sequence_context.gather(hidden_states, pool=self.pool, name=self.prefix + ":sp-attention-gather")
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        if self.sequence_context is not None:
            hidden_states = self.sequence_context.scatter(self.trace, hidden_states, pool=self.pool,
                name=self.prefix + ":sp-attention")
        if self.is_block_write_layer:
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states
        hidden_states = self.post(prefix_sum, prefix_delta, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, prefix_sum, residual


def trace_kimi_k3_forward(model, step, pool, expert_routing, *, input_ids, positions,
        slot_mapping, cache_capacity_tokens, state_indices_by_layer, state_pool_slots,
        source_max_model_len, max_batched_tokens, prefill_output_source=None, latent_tail_tier=None,
        _replica_context=None,_dummy_run=False,kda_decode_backend="triton",fa3_hardware=None):
    """Full hybrid TP/PP source graph, with an explicit TP latent-tail program.

    An expert-TP tier is a caller declaration, never a token-count provider
    inference. EP sets internal expert TP to one and uses the native tail.
    This entry does not select a GPU implementation or extend kernel pricing.
    """
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_plan
    from .model_config import is_kimi_k3_parallel_model_scope

    config, source = pool.parallel_plan.config, model.kimi_k3_semantics
    if fa3_hardware is not None:
        if fa3_hardware.hardware_id != pool.hardware_id or prefill_output_source not in (None,"supports_out"):
            raise ValueError("K3 FA3 requires its actual eager hardware pool and output-source contract")
        fa3_hardware.fa3_mla_source_revision_for_model("vllm_0_30_0")
    if _replica_context is not None and not isinstance(_replica_context, _K3ReplicaContext):
        raise TypeError("K3 replica context must be constructed by the complete DP source")
    tp, pp = config.tensor_parallel_size, config.pipeline_parallel_size
    if (type(_dummy_run) is not bool or _dummy_run and (_replica_context is None
            or not _replica_context.request_rows or _replica_context.request_rows[_replica_context.rank]!=0)):
        raise ValueError("K3 dummy forward requires an actual idle replica without request coordinates")
    if kda_decode_backend not in ('triton','native') or _dummy_run and input_ids!=(0,):
        raise ValueError("K3 source requires its explicit KDA backend and V2 dummy input zeros")
    if (model.model_type != "kimi_linear" or source is None or pool.hardware_id != "nvidia_h200"
            or model.compute_dtype != "bf16" or not source.latent_moe_use_norm or not source.mla_use_output_gate
            or (model.qk_nope_head_dim, model.qk_rope_head_dim, model.kv_lora_rank, model.v_head_dim) != (128, 64, 512, 128)
            or source.kda_head_dim != 128 or source.short_conv_kernel_size != 4
            or not is_kimi_k3_parallel_model_scope(model, tp, pp,
                expert_parallel=config.enable_expert_parallel) or len(pool.ranks) != tp * pp
            or any(value != 1 for value in (
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("virtual K3 requires BF16 hybrid whole TP or stage-local EP shards/nonempty PP, SP/CP/DP1 source")
    ep = config.expert_parallel_size if _replica_context is None else _replica_context.expert_parallel_size
    data_size = 1 if _replica_context is None else len(_replica_context.rows)
    fused_tail = ep == 1 and tp * data_size > 1
    if (not fused_tail and latent_tail_tier is not None
            or fused_tail and latent_tail_tier not in ("ALLREDUCE_OVERLAP", "COLUMN_PARALLEL")):
        raise ValueError("K3 expert-TP requires an explicit ordinary tail; EP and TP1 require the native tail")
    if (step.pool_name != pool.name or not step.requests or any(r.mtp_proposed_tokens or r.mtp_proposer_tokens or r.mtp_verified_tokens
            for r in step.requests)):
        raise NotImplementedError("virtual K3 requires ordinary nonempty target-off source requests")
    decode = _dummy_run or all(r.phase == "decode" and r.scheduled_tokens == 1 and r.computed_tokens > 0 for r in step.requests)
    fresh = (not _dummy_run and len(step.requests) == 1 and step.requests[0].phase == "prefill" and step.requests[0].computed_tokens == 0
        and step.requests[0].scheduled_tokens >= 512)
    context = (len(step.requests)==1 and step.requests[0].phase=="prefill" and step.requests[0].computed_tokens>0
        and step.requests[0].scheduled_tokens>=512 and step.requests[0].scheduled_tokens%64==0)
    mixed = (not _dummy_run and len(step.requests) == 2 and step.requests[0].phase == "decode"
        and step.requests[0].scheduled_tokens == 1 and step.requests[0].computed_tokens > 0
        and step.requests[1].phase == "prefill" and step.requests[1].computed_tokens == 0
        and step.requests[1].scheduled_tokens >= 512)
    if not decode and not fresh and not mixed and not context:
        raise NotImplementedError("virtual K3 supports single-token decode, fresh prefill with full/tail64 chunks or decode-first/fresh mixed")
    if ((fresh or mixed or context) and prefill_output_source != "supports_out") or (decode and prefill_output_source is not None):
        raise ValueError("K3 fresh prefill requires an explicit supports_out backend source declaration")
    rows = 1 if _dummy_run else sum(r.scheduled_tokens for r in step.requests)
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"K3 {name} requires one explicit integer per source row")
    if any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("K3 input ID is outside the accurate vocabulary")
    if positions != ((0,) if _dummy_run else tuple(p for r in step.requests for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens))) or any(value >= model.max_position_embeddings for value in positions):
        raise ValueError("K3 positions must match the declared source contexts")
    kv_lens = (1,) if _dummy_run else tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    if (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slot_mapping)) != rows or (slot_mapping!=(-1,) if _dummy_run else
                any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping))):
        raise ValueError("K3 logical KV pool requires disjoint valid write slots and all contexts")
    if (type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
            or type(max_batched_tokens) is not int or max_batched_tokens < rows):
        raise ValueError("K3 source constructor limits must contain its scheduled contexts and rows")
    if context and step.requests[0].computed_tokens>min(8*source_max_model_len,65536):
        raise NotImplementedError("K3 multi-chunk context requires the caller's physical page/alignment contract")
    if (type(state_pool_slots) is not int or state_pool_slots <= 1
            or not isinstance(state_indices_by_layer, tuple) or len(state_indices_by_layer) != model.num_hidden_layers
            or any(not isinstance(indices, tuple) for indices in state_indices_by_layer)
            or any((kind == "kda") != bool(indices)
                for kind, indices in zip(source.layer_attention_types, state_indices_by_layer, strict=True))
            or any(indices and len(indices) != (1 if _dummy_run else len(step.requests)) for indices in state_indices_by_layer)
            or any(type(index) is not int or (index!=0 if _dummy_run else not 0 < index < state_pool_slots)
                for indices in state_indices_by_layer for index in indices)
            or any(len(set(indices)) != len(indices) for indices in state_indices_by_layer)):
        raise ValueError("K3 requires layer-specific distinct positive KDA state slots")
    ranges = pipeline_layer_ranges_for_plan(pool.parallel_plan, num_layers=model.num_hidden_layers)
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = (_source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
        if _replica_context is None else _replica_context.resolutions)
    if _replica_context is not None:
        invocations = _replica_context.invocations
    traces = (tuple(ForwardTrace(rank=rank, namespace=step.step_id if tp * pp == 1 else f"{step.step_id}:rank-{rank}")
        for rank in pool.ranks) if _replica_context is None else _replica_context.traces)
    groups = ((() if tp == 1 else tuple(ForwardCollectiveGroup(traces[stage * tp:(stage + 1) * tp],
        namespace=step.step_id if pp == 1 else f"{step.step_id}:stage-{stage}") for stage in range(pp)))
        if _replica_context is None else _replica_context.tensor_groups)
    collective = None if tp == 1 else pool.collective_for("all_reduce")
    gather = None if tp == 1 else pool.collective_for("all_gather")
    pipeline = (ForwardPipelineGroup(traces,
        namespace=step.step_id if _replica_context is None else f"{step.step_id}:dp{_replica_context.rank}:pipeline",
        collective_groups=groups)
        if pp > 1 else None)
    if (model.weight_dtype_for_module("language_model.model.embed_tokens", "Embedding") != "bf16"
            or model.weight_dtype_for_module("language_model.lm_head", "Linear") != "bf16"):
        raise NotImplementedError("K3 source embedding and head require BF16 checkpoint storage")

    def forward(trace, stage, incoming):
        group = None if tp == 1 else groups[stage]
        lane = traces[stage * tp:(stage + 1) * tp].index(trace)
        model_sp = _replica_context is not None and _replica_context.model_sequence_parallel
        sequence = (_K3SequenceContext(group, gather, pool.collective_for("reduce_scatterv"), rows, lane)
            if model_sp else None)
        data_context = (None if _replica_context is None else _K3MoEDataContext(
            _replica_context.expert_groups[stage] if model_sp else _replica_context.data_groups[stage][lane],
            _replica_context.rank,
            tuple(-(-count // tp) for count in _replica_context.rows for _ in range(tp)) if model_sp else _replica_context.rows,
            pool.collective_for("all_gatherv"), pool.collective_for("reduce_scatterv"), model_sp, tp, lane))
        start, end = ranges[stage]
        capacity = -(-end // source.attention_residual_block_size)
        layers = []
        fa3_length_inputs = {} if fa3_hardware is not None else None
        for layer in range(start, end):
            kind = source.layer_attention_types[layer]
            prefix = f"layer-{layer}"
            base = f"language_model.model.layers.{layer}"
            projections = (("self_attn.in_proj_qkvgfab", "self_attn.f_b_proj", "self_attn.o_proj")
                if kind == "kda" else ("self_attn.fused_qkv_a_g_proj", "self_attn.q_b_proj", "self_attn.o_proj"))
            projections += (("block_sparse_moe.gate", "block_sparse_moe.routed_expert_down_proj",
                "block_sparse_moe.routed_expert_up_proj", "block_sparse_moe.shared_experts.gate_up_proj",
                "block_sparse_moe.shared_experts.down_proj") if model.layer_uses_moe(layer)
                else ("mlp.gate_up_proj", "mlp.down_proj"))
            if any(model.weight_dtype_for_module(base + "." + name, "Linear") != "bf16" for name in projections):
                raise NotImplementedError("K3 source projections require checkpoint BF16 module weights")
            if kind == "kda":
                attention_type = KimiK3KDADecode if decode else KimiK3KDAPrefill
                attention = attention_type(trace, model, pool.name, prefix + ":attention", rows=rows,
                    state_indices=state_indices_by_layer[layer], state_pool_slots=state_pool_slots,
                    tp_size=tp, reduction_group=group, reduction_config=collective,
                    reduce_results=not model_sp,
                    **({"dummy_run":_dummy_run,"decode_backend":kda_decode_backend} if decode else {"query_lengths": tuple(r.scheduled_tokens for r in step.requests),
                        "initial_flags": False if fresh else True if context else (True, False)}))
            elif kind == "mla_nope":
                if not decode and model.weight_dtype_for_module(base + ".self_attn.kv_b_proj", "Linear") != "bf16":
                    raise NotImplementedError("K3 prefill KV expansion requires BF16 checkpoint weights")
                attention_type = (KimiK3NoPEDecode if decode else KimiK3NoPEFreshPrefill if fresh
                    else KimiK3NoPEContextPrefill if context else KimiK3NoPEMixed)
                attention = attention_type(trace, model, pool.name, prefix + ":attention", rows=rows,
                    kv_lens=kv_lens, slots=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
                    tp_size=tp, reduction_group=group, reduction_config=collective, reduce_results=not model_sp,
                    dummy_run=_dummy_run,
                    **({"fa3_hardware":fa3_hardware,"fa3_length_inputs":fa3_length_inputs} if not decode else {}),
                    **({"context_length":step.requests[0].computed_tokens,
                        "context_chunk_rows":min(8*source_max_model_len,65536),"module_path":base+".self_attn"} if context else {}))
            else:
                raise NotImplementedError("K3 source has an unadapted attention type")
            layers.append(KimiK3DecoderLayer(trace, model, pool.name, prefix, layer=layer, attention=attention,
                resolution=resolutions[layer], invocation_id=invocations[layer], residual_capacity=capacity,
                tp_size=tp, reduction_group=group, reduction_config=collective, latent_tail_tier=latent_tail_tier,
                expert_parallel_size=config.expert_parallel_size if _replica_context is None else _replica_context.expert_parallel_size,
                data_context=data_context, sequence_context=sequence))
        pos = trace.tensor("positions", (rows,), "int64")
        if _dummy_run:
            SymbolicFillZero(trace,"dummy:positions-zero",pool.name)(pos)
        if stage == 0:
            ids = trace.tensor("input_ids", (rows,), "int32")
            if _dummy_run:
                SymbolicFillZero(trace,"dummy:input-IDs-zero",pool.name)(ids)
            hidden_states = (SymbolicEmbedding(trace, "embedding", pool.name, model.hidden_size, "bf16")(ids)
                if group is None else SymbolicVocabParallelEmbedding(trace, "embedding", pool.name,
                    model.hidden_size, model.vocab_size, group, collective, token_ids=input_ids)(ids))
            if sequence is not None:
                hidden_states = sequence.shard(trace, hidden_states, pool=pool.name)
            residual = trace.tensor("block-residual", (rows if sequence is None else sequence.shard_rows,
                capacity, model.hidden_size), "bf16")
            prefix_sum = hidden_states
        else:
            if incoming is None:
                raise ValueError("K3 pipeline stage requires its received hidden and residual tensors")
            hidden_states = None
            prefix_sum = incoming["hidden_states"]
            received_residual = incoming["residual"]
            residual = trace.tensor("block-residual", (rows, capacity, model.hidden_size), "bf16")
            SymbolicCopy(trace, "attn-res:receive-copy", pool.name)(
                received_residual, residual.narrow(1, 0, received_residual.shape[1]))
        for layer in layers:
            hidden_states, prefix_sum, residual = layer(pos, hidden_states, residual, prefix_sum)
        if stage < pp - 1:
            combined = SymbolicAdd(trace, "pipeline:prefix-add", pool.name)(hidden_states, prefix_sum)
            return {"hidden_states": combined, "residual": residual}
        hidden_states = SymbolicK3AttentionResidual(trace, "output-attn-res", pool.name, model,
            blocks=capacity, output_norm=False, capacity=capacity)(prefix_sum, hidden_states, residual)
        if sequence is not None:
            hidden_states = sequence.gather(hidden_states, pool=pool.name, name="sp-final-hidden-gather")
        if _dummy_run:
            SymbolicGather(trace,"dummy:selected-hidden",pool.name)(hidden_states,1,indices=(0,))
            return None
        selected_rows = tuple(sum(r.scheduled_tokens for r in step.requests[:i + 1]) - 1 for i in range(len(step.requests)))
        indices = trace.tensor("logits-indices", (len(selected_rows),), "int32")
        sample_hidden = SymbolicLogitsRowSelect(trace, "logits-row-select", pool.name)(
            hidden_states, indices, selected_indices=selected_rows)
        sample_hidden = SymbolicRMSNorm(trace, "final-norm", pool.name, model.hidden_size, source.rms_norm_eps)(sample_hidden)
        logits = (SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
            model.vocab_size, kind=WorkloadKind.LM_HEAD)(sample_hidden)[0] if group is None else
            SymbolicTPLogitsProcessor(trace, "lm-head", pool.name, hidden_size=model.hidden_size,
                vocab_size=model.vocab_size, group=group, config=gather)(sample_hidden))
        sampled = SymbolicGreedy(trace, "target-sampler", pool.name)(logits)
        return None

    if pipeline is None:
        for trace in traces:
            forward(trace, 0, None)
        result = traces[0] if tp == 1 else groups[0]
        if tp > 1:
            result.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
                "latent_tail_tier": latent_tail_tier, "normal_capacity_accepted": False,
                "actual_provider_bound": False}
        return result
    pipeline.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "pipeline_layer_partition": tuple(end - start for start, end in ranges),
        "latent_tail_tier": latent_tail_tier,
        "normal_capacity_accepted": False, "actual_provider_bound": False}
    _run_vllm030_pipeline_stages(pipeline, pool, rows=rows, hidden_size=model.hidden_size,
        max_batched_tokens=max_batched_tokens, run_stage=forward,
        residual_blocks=tuple(-(-end // source.attention_residual_block_size) for _, end in ranges),dummy_run=_dummy_run)
    if not _dummy_run:_bind_k3_pipeline_view_receipts(pipeline)
    return pipeline


def _bind_k3_pipeline_view_receipts(pipeline):
    """Bind real K3 receive reshapes and block-buffer copies to PP metadata.

    Residual all-gather returns a flattened two-axis view, while the runner
    and Attention Residual consume three-axis aliases. Keep their actual
    view/allocation identities rather than inferring aliases from names.
    """
    from dataclasses import asdict
    nodes = {name: node for trace in pipeline.traces for name, node in trace._nodes.items()}
    traces = {trace.rank: trace for trace in pipeline.traces}
    parents = {}
    for trace in pipeline.traces:
        for edge in trace._edges:
            if edge.kind == "data":
                if edge.destination not in parents:
                    parents[edge.destination] = []
                parents[edge.destination].append(edge.source)
    for trace in pipeline.traces:
        for name, message in tuple(trace._nodes.items()):
            if message.kind is not WorkloadKind.P2P:
                continue
            role = message.attributes["payload_tensor"]["name"]
            target = traces[message.attributes["destination_rank"]]
            copies = tuple(node for node in target._nodes.values()
                if node.node_id.endswith(":receive:runner-copy:" + role))
            if len(copies) != 1:
                raise ValueError("K3 PP receipt requires its actual destination runner copy")
            copy, = copies
            predecessor, = parents[copy.node_id]
            # Collective graph assembly joins local submissions later. Use
            # this worker's own completed call, not another lane's record.
            received = target._nodes[predecessor] if predecessor in target._nodes else nodes[predecessor]
            port = (message.attributes["symbolic_tensor_outputs"][0] if received.kind is WorkloadKind.P2P else
                received.attributes["symbolic_tensor_outputs"][0])
            metadata = {"received_or_gathered": asdict(target._views[port]),
                "runner_input": asdict(target._views[copy.attributes["symbolic_tensor_inputs"][0]])}
            residual = None
            if role == "residual":
                block_copy = next(node for node in target._nodes.values() if node.node_id.endswith(":attn-res:receive-copy"))
                first_pre = next(node for node in target._nodes.values()
                    if node.variant == "kimi_k3_attn_res" and node.node_id.endswith(":pre-attn"))
                residual = {"runner_output": asdict(target._views[copy.attributes["symbolic_tensor_outputs"][0]]),
                    "block_copy_input": asdict(target._views[block_copy.attributes["symbolic_tensor_inputs"][0]]),
                    "block_copy_output": asdict(target._views[block_copy.attributes["symbolic_tensor_outputs"][0]]),
                    "attention_input": asdict(target._views[first_pre.attributes["symbolic_tensor_inputs"][-1]])}
                residual["block_residual_backing_bytes"] = target._allocations[
                    target._views[block_copy.attributes["symbolic_tensor_outputs"][0]].allocation_id].size_bytes
            trace._nodes[name] = message.with_attributes({**message.attributes,
                "model_source_profile": "vllm_0_30_k3_mxfp4_sync",
                "symbolic_pipeline_views": {**message.attributes["symbolic_pipeline_views"],
                    "destination_postprocess_views": metadata, "destination_residual_views": residual}})


def trace_kimi_k3_data_parallel_forward(model, step, pool, expert_routing, *, input_ids,
        source_inputs_by_replica, source_max_model_len, max_batched_tokens, latent_tail_tier=None,
        cache_capacity_tokens_by_replica=None,state_pool_slots_by_replica=None,kda_decode_backend="triton",fa3_hardware=None):
    """Full original DP requests over K3's existing TP/PP source modules.

    Modular MXFP4 prepare gathers latent/weights/IDs after local routing;
    finalize sum-scatters the latent before its TP tail/normalization.
    Real request coordinates and retained idle pool extents stay separate.
    The decode backend is an explicit source declaration, not GPU selection.
    """
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_plan
    from .model_config import is_kimi_k3_data_parallel_model_scope
    from .runtime_inputs import K3SourceInputs
    config = pool.parallel_plan.config
    tp, pp, dp = config.tensor_parallel_size, config.pipeline_parallel_size, config.data_parallel_size
    model_sp = config.enable_expert_parallel and tp > 1 and pp == 1
    if (pool.hardware_id != "nvidia_h200" or not is_kimi_k3_data_parallel_model_scope(model, config)
            or len(pool.ranks) != config.worker_count):
        raise NotImplementedError("K3 DP source requires whole-shard TP/nonempty PP, uniform global EP or EP-off and CP/SP1")
    if config.enable_expert_parallel and latent_tail_tier is not None:
        raise ValueError("K3 global EP uses the native latent tail")
    if not config.enable_expert_parallel and latent_tail_tier != "ALLREDUCE_OVERLAP":
        raise NotImplementedError("K3 DP expert-TP source requires its explicitly declared full up-projection tier")
    if (not isinstance(source_inputs_by_replica, tuple) or len(source_inputs_by_replica) != dp
            or not isinstance(input_ids, tuple) or len(input_ids) != step.token_count
            or step.pool_name != pool.name or not step.requests
            or len({request.request_id for request in step.requests}) != len(step.requests)):
        raise ValueError("K3 DP source requires original distinct requests, explicit IDs and per-replica cache coordinates")
    requests, ids = [[] for _ in range(dp)], [[] for _ in range(dp)]
    offset = 0
    for request in step.requests:
        owner = request.data_parallel_rank
        if not 0 <= owner < dp:
            raise ValueError("K3 DP request owner is outside its actual deployment")
        requests[owner].append(request)
        ids[owner].extend(input_ids[offset:offset + request.scheduled_tokens])
        offset += request.scheduled_tokens
    request_rows = tuple(sum(request.scheduled_tokens for request in replica) for replica in requests)
    rows=tuple(count if count else 1 for count in request_rows)
    if ((cache_capacity_tokens_by_replica is None)!=(state_pool_slots_by_replica is None)
            or not all(request_rows) and cache_capacity_tokens_by_replica is None):
        raise ValueError("K3 idle source requires each worker's actual retained MLA and recurrent capacities")
    if cache_capacity_tokens_by_replica is not None and (
            type(cache_capacity_tokens_by_replica) is not tuple or type(state_pool_slots_by_replica) is not tuple
            or len(cache_capacity_tokens_by_replica)!=dp or len(state_pool_slots_by_replica)!=dp
            or any(type(value) is not int or value<1 for value in cache_capacity_tokens_by_replica)
            or any(type(value) is not int or value<=1 for value in state_pool_slots_by_replica)):
        raise ValueError("K3 retained hybrid capacities must cover every actual DP worker")
    for replica, state in enumerate(source_inputs_by_replica):
        if request_rows[replica]:
            if not isinstance(state, K3SourceInputs) or state.request_ids != tuple(r.request_id for r in requests[replica]):
                raise ValueError("K3 DP cache/state coordinates must match their original request owner")
            if cache_capacity_tokens_by_replica is not None and (state.cache_capacity_tokens!=cache_capacity_tokens_by_replica[replica]
                    or state.state_pool_slots!=state_pool_slots_by_replica[replica]):
                raise ValueError("K3 retained pool capacities disagree with their active request receipt")
        elif state is not None:
            raise ValueError("idle K3 DP replicas cannot invent cache/state coordinates")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    tensor = tuple(tuple(ForwardCollectiveGroup(traces[(replica * pp + stage) * tp:(replica * pp + stage + 1) * tp],
        namespace=f"{step.step_id}:dp{replica}:stage{stage}:tp") for stage in range(pp)) for replica in range(dp)) if tp > 1 else ()
    data = tuple(tuple(ForwardCollectiveGroup(tuple(traces[(replica * pp + stage) * tp + lane] for replica in range(dp)),
        namespace=f"{step.step_id}:stage{stage}:lane{lane}:dp") for lane in range(tp)) for stage in range(pp))
    expert = (tuple(ForwardCollectiveGroup(tuple(traces[(replica * pp + stage) * tp + lane]
        for replica in range(dp) for lane in range(tp)), namespace=f"{step.step_id}:stage{stage}:ep")
        for stage in range(pp)) if model_sp else ())
    joined = ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:joined",
        subgroups=(*(group for replica in tensor for group in replica), *(group for stage in data for group in stage), *expert))
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    expert_tokens = sum(-(-count // tp) * tp for count in rows) if model_sp else sum(rows)
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, expert_tokens)
    local_config = replace(config, data_parallel_size=1, enable_expert_parallel=False)
    plan = build_parallel_plan(local_config, num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=tuple(end - start for start, end in pipeline_layer_ranges_for_plan(pool.parallel_plan, num_layers=model.num_hidden_layers)))
    ranges = pipeline_layer_ranges_for_plan(pool.parallel_plan, num_layers=model.num_hidden_layers)
    for replica in range(dp):
        local_traces = traces[replica * pp * tp:(replica + 1) * pp * tp]
        groups = tensor[replica] if tp > 1 else ()
        context = _K3ReplicaContext(local_traces, groups, data, rows, replica, resolutions, invocations,
            config.expert_parallel_size, model_sp, expert,request_rows)
        local_pool = replace(pool, ranks=tuple(trace.rank for trace in local_traces), parallel_plan=plan)
        if request_rows[replica] == 0:
            trace_kimi_k3_forward(model,step,local_pool,expert_routing,input_ids=(0,),positions=(0,),slot_mapping=(-1,),
                cache_capacity_tokens=cache_capacity_tokens_by_replica[replica],state_pool_slots=state_pool_slots_by_replica[replica],
                state_indices_by_layer=tuple((0,) if kind=='kda' else () for kind in model.kimi_k3_semantics.layer_attention_types),
                source_max_model_len=source_max_model_len,max_batched_tokens=max_batched_tokens,
                latent_tail_tier=latent_tail_tier,_replica_context=context,_dummy_run=True,kda_decode_backend=kda_decode_backend,fa3_hardware=fa3_hardware)
            continue
        state = source_inputs_by_replica[replica]
        local_step = replace(step, requests=tuple(requests[replica]), finished_request_ids=(), preempted_request_ids=())
        trace_kimi_k3_forward(model, local_step, local_pool, expert_routing,
            input_ids=tuple(ids[replica]), positions=tuple(position for r in requests[replica]
                for position in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens)),
            slot_mapping=state.slot_mapping, cache_capacity_tokens=state.cache_capacity_tokens,
            state_indices_by_layer=state.state_indices_by_layer, state_pool_slots=state.state_pool_slots,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            prefill_output_source=state.prefill_output_source, latent_tail_tier=latent_tail_tier, _replica_context=context,
            kda_decode_backend=kda_decode_backend, fa3_hardware=fa3_hardware)
    for trace in traces:
        owner=pool.logical_rank_for_physical_rank(trace.rank)//(pp*tp)
        if not request_rows[owner]:
            trace._nodes={key:node.with_attributes({**node.attributes,'source_dummy_run':True,'source_actual_request_count':0,
                'source_k3_request_rows':request_rows,'source_k3_model_rows':rows,'source_k3_kda_decode_backend':kda_decode_backend})
                if node.kind not in (WorkloadKind.COLLECTIVE,WorkloadKind.P2P) else node for key,node in trace._nodes.items()}
    joined.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
        "source_data_parallel_rows": request_rows,"source_model_rows_including_dummy":rows,
        "model_sequence_parallel": model_sp, "moe_sequence_parallel": model_sp,
        "source_shard_sp_mlp": False,
        "source_moe_skip_padding": True, "source_expert_physical_rows": expert_tokens,
        "source_padding_routing": "valid_grouped_topk_does_not_consume_padding_mask",
        "source_padding_route_histogram": "explicit_assumption_not_observed_expert_ids",
        "latent_tail_tier": latent_tail_tier, "tail_selection": "explicit_caller_declaration_no_provider_binding",
        "actual_provider_bound": False, "normal_capacity_accepted": False,
        "kda_decode_backend":kda_decode_backend,"kda_backend_selection":"explicit source declaration, never native-auto provider binding",
        "idle_source_scope": "complete V2 one-token target; reserved state0/backend-specific access, PAD MLA writes, null-block read, local PP intermediate; no target head or sampler",
        "cache_capacity_tokens_by_replica":cache_capacity_tokens_by_replica,"state_pool_slots_by_replica":state_pool_slots_by_replica}
    return joined


class DeepseekV2KimiDecoderLayer(Qwen3DecoderLayer):
    """BF16 residual decoder, with the pinned0.30 model-SP branch."""

    def __init__(self, trace, model, pool, prefix, group, config, *, layer_index, attention, resolution, invocation_id,
                 expert_parallel_size=1, expert_context=None):
        self.trace, self.pool, self.prefix, self.tensor_group = trace, pool, prefix, group
        self.sequence_context = (expert_context if expert_context is not None
            and expert_context.model_sequence_parallel and model.layer_uses_moe(layer_index) else None)
        self.remote_ffn = None
        self.input_layernorm = SymbolicRMSNorm(trace, prefix + ":input-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        self.self_attn = attention
        self.post_attention_layernorm = SymbolicRMSNorm(trace, prefix + ":post-attn-norm", pool,
            model.hidden_size, model.target_semantics.rms_norm_eps)
        if model.layer_uses_moe(layer_index):
            self.mlp = DeepseekV2KimiMoE(trace, model, pool, prefix + ":mlp", group, config,
                layer_index=layer_index, resolution=resolution, invocation_id=invocation_id,
                expert_parallel_size=expert_parallel_size, expert_context=expert_context)
        else:
            self.mlp = DeepseekV2MLP(trace, model, pool, prefix + ":mlp", intermediate_size=model.intermediate_size,
                tp_size=1 if group is None else len(group.traces), module_path=f"language_model.model.layers.{layer_index}.mlp",
                reduce_results=group is not None, reduction_group=group, reduction_config=config)


    def __call__(self, positions, hidden, residual=None):
        if self.sequence_context is None:
            return super().__call__(positions, hidden, residual)
        context = self.sequence_context
        input_sp = residual is not None and hidden.shape[0] != positions.shape[0]
        if residual is None:
            residual, hidden = hidden, self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        if input_sp:
            gathered = self.tensor_group.call(self.prefix + ":sp-attention-input", hidden, pool_name=self.pool,
                config=context.gather, inplace=False, dim=0)
            hidden = gathered.narrow(0, 0, positions.shape[0]) if gathered.shape[0] != positions.shape[0] else gathered
        hidden = self.self_attn(positions, hidden)
        chunk = -(-context.logical_rows // context.tensor_size)
        padded_rows = chunk * context.tensor_size
        if hidden.shape[0] != padded_rows:
            padded = self.trace.tensor(self.trace.unique_name(self.prefix + ":attention-padded"), (padded_rows, hidden.shape[1]), "bf16")
            SymbolicCopy(self.trace, self.prefix + ":attention-pad-copy", self.pool)(hidden, padded.narrow(0, 0, hidden.shape[0]))
            SymbolicFillZero(self.trace, self.prefix + ":attention-pad-zero", self.pool)(padded.narrow(0, hidden.shape[0], padded_rows - hidden.shape[0]))
            hidden = padded
        hidden = self.tensor_group.call_variable(self.prefix + ":sp-attention-output", self.trace, hidden,
            pool_name=self.pool, config=context.combine, rows_by_rank=(chunk,) * context.tensor_size,
            row_shape=(hidden.shape[1],), dtype="bf16", group_dimension="tensor")
        if not input_sp:
            residual = context.shard(self.trace, residual, pool=self.pool, name=self.prefix + ":residual-chunk")
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


class DeepseekV2KimiModel(Qwen3Model):
    """K2.6 decoder or PP stage, sharing the same source forward."""

    def __init__(self, trace, model, pool, group, config, *, token_ids, q_lens, kv_lens, slots, cache_capacity_tokens,
                 resolutions, invocation_ids, model_input_order, num_decodes, expert_parallel_size=1,
                 context_chunk_rows=None, context_chunk_alignment=None, layer_range=None, physical_rows=None, expert_context=None,
                 dummy_run=False, dummy_workspace_min_rows=None, fa3_hardware=None):
        self._source_module_program = None
        self.trace, self.pool, self.hidden_size, self.tensor_group = trace, pool, model.hidden_size, group
        self.sequence_context = expert_context if expert_context is not None and expert_context.model_sequence_parallel else None
        start, end = (0, model.num_hidden_layers) if layer_range is None else layer_range
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= model.num_hidden_layers):
            raise ValueError("Kimi stage requires a nonempty contiguous checkpoint layer range")
        self.first_stage, self.last_stage = start == 0, end == model.num_hidden_layers
        tp_size = 1 if group is None else len(group.traces)
        self.embed_tokens = None
        if self.first_stage:
            self.embed_tokens = (SymbolicEmbedding(trace, "embedding", pool, model.hidden_size, "bf16",
                model_input_order=model_input_order, vocab_size=model.vocab_size) if group is None else
                SymbolicVocabParallelEmbedding(trace, "embedding", pool, model.hidden_size,
                    model.vocab_size, group, config, token_ids=None if dummy_run else token_ids,
                    input_values_source="legacy_dummy_retained_buffer" if dummy_run else None,
                    model_input_order=model_input_order if trace.rank == group.participants[0] else None))
        self.rope_cache = trace.tensor("model.rope-cache", (model.max_position_embeddings,
            model.qk_rope_head_dim), "bf16", persistent=True)
        rows = sum(q_lens)
        physical_rows = rows if physical_rows is None else physical_rows
        self.slot_mapping = None if dummy_run else trace.tensor("slot_mapping", (physical_rows,), "int64", persistent=True)
        slot_mapping = None if dummy_run else (self.slot_mapping if physical_rows == rows else self.slot_mapping.narrow(0, 0, rows))
        fa3_length_inputs = {} if fa3_hardware is not None else None
        self.layers = []
        for layer in range(start, end):
            prefix = f"layer-{layer}"
            cache = None if dummy_run else trace.tensor(prefix + ":kv-cache", (cache_capacity_tokens,
                model.kv_lora_rank + model.qk_rope_head_dim), "bf16", persistent=True)
            if dummy_run:
                inputs = DeepseekV2MLAInput(trace, model, pool, prefix + ":attention", layer_index=layer,
                    tp_size=tp_size, rotary_cache=self.rope_cache)
                projection = _source_linear(trace, model, pool, prefix + ":attention:o-proj",
                    inputs.heads * model.v_head_dim, model.hidden_size,
                    f"language_model.model.layers.{layer}.self_attn.o_proj")
                if group is not None and not (self.sequence_context is not None and model.layer_uses_moe(layer)):
                    projection = SymbolicRowParallelLinear(projection, group, config)
                attention = MLADummyAttention(trace, model, pool, prefix + ":attention", inputs=inputs,
                    projection=projection, workspace_min_rows=dummy_workspace_min_rows)
            elif num_decodes < len(q_lens):
                attention = DeepseekV2MLAMixedAttention(trace, model, pool, prefix + ":attention",
                    layer_index=layer, tp_size=tp_size, q_lens=q_lens, kv_lens=kv_lens,
                    num_decodes=num_decodes, cache=cache,
                    slot_mapping=slot_mapping, slots=slots, rotary_cache=self.rope_cache,
                    reduction_group=group, reduction_config=config, context_chunk_rows=context_chunk_rows,
                    context_chunk_alignment=context_chunk_alignment, fa3_hardware=fa3_hardware, fa3_length_inputs=fa3_length_inputs,
                    reduce_results=not (self.sequence_context is not None and model.layer_uses_moe(layer)))
            else:
                attention = DeepseekV2MLADecodeAttention(trace, model, pool, prefix + ":attention", layer_index=layer,
                    tp_size=tp_size, kv_lens=kv_lens, cache=cache, slot_mapping=slot_mapping, slots=slots,
                    rotary_cache=self.rope_cache, reduction_group=group, reduction_config=config,
                    backend_source="vllm_0_30_flashmla_bf16", q_lens=q_lens,
                    reduce_results=not (self.sequence_context is not None and model.layer_uses_moe(layer)))
            self.layers.append(DeepseekV2KimiDecoderLayer(trace, model, pool, prefix, group, config,
                layer_index=layer, attention=attention, resolution=resolutions[layer], invocation_id=invocation_ids[layer],
                expert_parallel_size=expert_parallel_size, expert_context=expert_context))
        self.norm = (SymbolicRMSNorm(trace, "final-norm", pool, model.hidden_size, model.target_semantics.rms_norm_eps)
            if self.last_stage else None)


    def __call__(self, input_ids, positions, intermediate_tensors=None):
        if self.sequence_context is None:
            return super().__call__(input_ids, positions, intermediate_tensors)
        if not self.first_stage or not self.last_stage or intermediate_tensors is not None:
            raise ValueError("Kimi across-layer SP is confined to the actual PP1 source branch")
        hidden = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        # Preserve the actual0.30.0 shape condition, including its one-row
        # branch, rather than inventing a GPU-runner padding/reorder receipt.
        if hidden.shape[0] != positions.shape[0]:
            combined = SymbolicConcat(self.trace, "sp-final-pair", self.pool)(hidden, residual)
            combined = self.tensor_group.call("sp-final-pair-gather", combined, pool_name=self.pool,
                config=self.sequence_context.gather, inplace=False, dim=0)
            if combined.shape[0] != positions.shape[0]:
                combined = combined.narrow(0, 0, positions.shape[0])
            hidden, residual = combined.split((self.hidden_size, self.hidden_size))
        hidden, _ = self.norm(hidden, residual)
        return hidden


def _kimi_single_rank_source_scope(model, pool):
    config = pool.parallel_plan.config
    return (_is_kimi_k2_text_source(model)
        and pool.hardware_id in ("nvidia_a800", "nvidia_h200") and len(pool.ranks) == 1
        and config.is_rank_local and not config.enable_expert_parallel
        and all(value == 1 for value in (config.tensor_parallel_size, config.pipeline_parallel_size,
            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size)))


def _kimi_pipeline_source_scope(model, pool):
    config = pool.parallel_plan.config
    from src.serving.deployment.validated_scope import is_kimi_k2_rank_local_source_scope
    return (config.pipeline_parallel_size > 1 and is_kimi_k2_rank_local_source_scope(pool, model)
        or _is_kimi_k2_text_source(model)
        and pool.hardware_id == "nvidia_a800" and config.pipeline_parallel_size == 2
        and config.tensor_parallel_size == config.expert_parallel_size == 8 and config.enable_expert_parallel
        and config.is_rank_local and len(pool.ranks) == 16
        and all(value == 1 for value in (config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
        and pool.parallel_plan.layer_to_pipeline_stage == (0,) * 31 + (1,) * 30)


class _KimiDecodeSourceProgram(_GLMDecodeSourceProgram):
    """Bind TP8/EP1 q1 Kimi source facts onto its complete recorded DAG.

    The actual current first layer supplies attention, slots and cache extent;
    every MoE layer resolves its own current routing. Every rank retains its
    original storage lifetimes, collective edges and complete target tail.
    """

    def bind(self, prefix, *, resolutions, invocation_ids, rows):
        from copy import copy
        from types import MappingProxyType
        result = copy(self.source)
        routes = {}
        for layer in range(self.model.num_hidden_layers):
            if self.model.layer_uses_moe(layer):
                operator = SymbolicRoutedExperts(prefix.traces[0], "binding", "binding", self.model,
                    tp_size=8, layer_index=layer, resolution=resolutions[layer],
                    invocation_id=invocation_ids[layer], routing_source_stage="kimi_moe_route")
                metadata = operator.routing_metadata(rows)
                routes[layer] = {**metadata, "routing_provenance": MappingProxyType(metadata["routing_provenance"])}
        traces, changed = [], {}
        for template, current, graph in zip(self.source.traces, prefix.traces, self.local_graphs, strict=True):
            trace = copy(template)
            trace._nodes, trace._allocations, trace._views = (
                dict(template._nodes), dict(template._allocations), dict(template._views))
            attention = current._nodes[current.qualify("layer-0:attention:latent")]
            update = current._nodes[current.qualify("layer-0:attention:kv-update")]
            embedding = current._nodes[current.qualify("embedding")]
            fields = {name: embedding.attributes[name] for name in (
                "embedding_token_ids", "symbolic_model_input_order") if name in embedding.attributes}
            name = trace.qualify("embedding")
            trace._nodes[name] = trace._nodes[name].with_updates({}, fields)
            changed[name] = trace._nodes[name]
            for layer in range(self.model.num_hidden_layers):
                base = trace.qualify(f"layer-{layer}")
                name = base + ":attention:latent"
                trace._nodes[name] = trace._nodes[name].with_updates(
                    {"kv_len": attention.shape["kv_len"]}, {
                        "kv_lens": attention.attributes["kv_lens"],
                        "source_cache_shape": attention.attributes["source_cache_shape"]})
                changed[name] = trace._nodes[name]
                name = base + ":attention:kv-update"
                trace._nodes[name] = trace._nodes[name].with_updates({},
                    {"symbolic_slot_mapping": update.attributes["symbolic_slot_mapping"]})
                changed[name] = trace._nodes[name]
                if layer in routes:
                    name = base + ":mlp:routed"
                    trace._nodes[name] = trace._nodes[name].with_updates({}, routes[layer])
                    changed[name] = trace._nodes[name]
                name, origin = base + ":kv-cache", current.qualify("layer-0:kv-cache")
                old = trace._allocations[name]
                trace._allocations[name] = current._allocations[origin].with_lifetime(
                    name, old.allocated_event, old.freed_event)
                trace._views[name] = current._views[origin].with_identity(name, name)
            trace._recorded_graph = (trace._nodes, graph.with_nodes(trace._nodes.values()))
            traces.append(trace)
        result.traces = tuple(traces)
        result._recorded_graph = self.graph.with_nodes(
            changed[node.node_id] if node.node_id in changed else node for node in self.graph.nodes)
        graph = result._recorded_graph
        object.__setattr__(graph, "_glm_decode_binding", _GLMDecodeSourceBinding(
            weakref.ref(self), weakref.ref(graph), graph.nodes, tuple(changed.values())))
        return result


def trace_kimi_k2_forward(model, step, pool, expert_routing, *, input_ids, positions, slot_mapping,
        cache_capacity_tokens, source_max_model_len, max_batched_tokens, source_profile, target_sampling_source=None,
        source_block_size=None, graph_token_shape=None, graph_eager_kinds=(), include_lm_head=False,
        source_prompt_lengths=None, source_program_cache=None, _replica_context=None, _dummy_run=False, fa3_hardware=None):
    """Full K2.6/K2.7 ordinary target graph, optionally including greedy.

    Explicit modular/no-overlap, BF16 latent KV, MTP-off; actual provider,
    full workspace and normal capacity remain separate gates. Context-prefill
    covers one shared chunk, or single-request block-aligned continuations.
    The historical profile name covers the common v0.30 source program; each
    model still retains its independent identity and target-semantics contract.
    """
    from .runtime_inputs import vllm_0_30_source_model_inputs
    if fa3_hardware is not None:
        if fa3_hardware.hardware_id != pool.hardware_id or graph_token_shape is not None:
            raise ValueError("FA3 context source binding requires its own eager hardware pool")
        fa3_hardware.fa3_mla_source_revision_for_model("vllm_0_30_0")
    if source_program_cache is not None and type(source_program_cache) is not SourceProgramCache:
        raise TypeError("Kimi source reuse requires its run-local SourceProgramCache")
    if _replica_context is not None and not isinstance(_replica_context, _KimiReplicaContext):
        raise TypeError("Kimi source requires its private normalized replica context")
    if (type(_dummy_run) is not bool or _dummy_run and (_replica_context is None
            or len(step.requests) != 1 or step.requests[0].request_id != "_dummy_req_0"
            or step.token_count != 1 or input_ids != (0,) or positions != (0,)
            or slot_mapping != (-1,) or cache_capacity_tokens is not None
            or graph_token_shape is not None or source_program_cache is not None)):
        raise ValueError("Kimi dummy source requires its private one-row DP dummy without request/cache coordinates")
    config = pool.parallel_plan.config
    pipeline_source = _kimi_pipeline_source_scope(model, pool)
    single_rank = _kimi_single_rank_source_scope(model, pool)
    from src.serving.deployment.validated_scope import is_kimi_k2_rank_local_source_scope
    if type(include_lm_head) is not bool:
        raise TypeError("K2.6 include_lm_head must be boolean")
    if (source_profile != "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
            or not _is_kimi_k2_text_source(model)
            or not (single_rank or pipeline_source or is_kimi_k2_rank_local_source_scope(pool, model))
            or any(value != 1 for value in (config.sequence_parallel_size,
                config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("virtual K2 text requires legal eager H200 TP/PP with EP-off or stage-local EP, or its existing A800 source; SP/CP/DP1")
    ordinary = bool(step.requests) and all(
        (r.phase == "prefill" and r.computed_tokens >= 0 and r.scheduled_tokens > 0)
        or (r.phase == "decode" and r.scheduled_tokens > 0 and r.computed_tokens > 0)
        for r in step.requests)
    if (step.pool_name != pool.name or not ordinary
            or any(r.mtp_proposed_tokens or r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests)):
        raise NotImplementedError("virtual K2.6 requires explicit positive decode/prefill queries with MTP execution off")
    if target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16"):
        raise NotImplementedError("virtual K2.6 target requires the declared BF16 plain greedy chain")
    rows = sum(r.scheduled_tokens for r in step.requests)
    physical_rows = rows
    if graph_token_shape is not None:
        from .graph_padding import GraphTokenShape
        if (not isinstance(graph_token_shape, GraphTokenShape) or not single_rank
                or graph_token_shape.logical_tokens != rows
                or graph_token_shape.runtime_mode not in ("FULL", "PIECEWISE")
                or (graph_token_shape.runtime_mode == "FULL" and any(
                    request.phase != "decode" or request.scheduled_tokens != 1 for request in step.requests))):
            raise NotImplementedError("Kimi source capture requires TP1/PP1 and FULL single-token decode or PIECEWISE")
        physical_rows = graph_token_shape.model_input_tokens
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(value) is not int for value in values):
            raise ValueError(f"K2.6 {name} requires one explicit integer per source row")
    if any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("K2.6 token IDs must belong to the accurate checkpoint vocabulary")
    if not _dummy_run and positions != tuple(position for r in step.requests
            for position in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens)):
        raise ValueError("K2.6 positions must match the declared source contexts")
    step, input_ids, model_input_order = vllm_0_30_source_model_inputs(step, embedding_token_ids=input_ids, prompt_lengths=source_prompt_lengths,
        decode_threshold=128, require_uniform=True)
    order = model_input_order["token_row_permutation"]
    positions = tuple(positions[row] for row in order)
    slots = tuple(slot_mapping[row] for row in order)
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    num_decodes = model_input_order["model_attention_phases"].count("decode")
    context_requests = tuple(request for request in step.requests[num_decodes:] if request.computed_tokens > 0)
    context_chunk_rows = None
    if context_requests:
        if sum(request.computed_tokens for request in context_requests) > min(8 * source_max_model_len, 65536):
            if type(source_block_size) is not int or source_block_size <= 0 or 65536 % source_block_size:
                raise NotImplementedError("virtual K2.6 multi-chunk context requires an explicit source block size dividing 65536")
            context_chunk_rows = 65536
        # Arena IDs stay attached to requests through source batch reordering.
        # Each request's context starts immediately before its first new slot.
        cursor = 0
        context_ranges = []
        for request in step.requests:
            request_slots = slots[cursor:cursor + request.scheduled_tokens]
            start = request_slots[0] - request.computed_tokens
            if start < 0 or request_slots != tuple(range(start + request.computed_tokens,
                    start + request.computed_tokens + request.scheduled_tokens)):
                raise ValueError("K2.6 context requires the declared disjoint logical request arena, not native page IDs")
            context_ranges.append((start, start + request.computed_tokens + request.scheduled_tokens))
            cursor += request.scheduled_tokens
        ordered_ranges = sorted(context_ranges)
        if any(left[1] > right[0] for left, right in zip(ordered_ranges, ordered_ranges[1:])):
            raise ValueError("K2.6 context requires disjoint logical request arenas")
    if (type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings
            or type(max_batched_tokens) is not int or max_batched_tokens < physical_rows):
        raise ValueError("K2.6 source context and batch limits must contain every row")
    if not _dummy_run and (type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slots)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slots)):
        raise ValueError("K2.6 logical cache must contain contexts and distinct valid update slots")
    reusable = (source_program_cache is not None and _replica_context is None and pool.hardware_id == "nvidia_h200"
        and config.tensor_parallel_size == 8 and config.pipeline_parallel_size == 1
        and config.expert_parallel_size == 1 and len(pool.ranks) == 8
        and graph_token_shape is None and num_decodes == len(q_lens)
        and all(q == 1 for q in q_lens) and _DENSE_METADATA_SOURCE_CONTRACT.unchanged())
    key = ("kimi-q1-source-v1", model, id(pool), q_lens, source_profile,
        source_max_model_len, max_batched_tokens, source_block_size,
        include_lm_head, target_sampling_source) if reusable else None
    program = source_program_cache.lookup(key) if reusable else None
    namespace = step.step_id if program is None else program.source.namespace
    stages = tuple(tuple(ForwardTrace(rank=rank, namespace=f"{namespace}:rank-{rank}")
        for rank in pool.physical_ranks_for_pipeline_stage(stage)) for stage in range(config.pipeline_parallel_size))
    groups = tuple(None if config.tensor_parallel_size == 1 else ForwardCollectiveGroup(traces,
        namespace=f"{step.step_id}:stage-{stage}" if pipeline_source else namespace)
        for stage, traces in enumerate(stages))
    if _replica_context is not None:
        stages = tuple(_replica_context.traces[stage * config.tensor_parallel_size:(stage + 1) * config.tensor_parallel_size]
            for stage in range(config.pipeline_parallel_size))
        groups = _replica_context.tensor_groups if config.tensor_parallel_size > 1 else (None,) * len(stages)
    collective = None if config.tensor_parallel_size == 1 else pool.collective_for("all_reduce")
    target_head = not _dummy_run and (include_lm_head or target_sampling_source is not None)
    gather = None if config.tensor_parallel_size == 1 or not target_head else pool.collective_for("all_gather")
    invocation_ids = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocation_ids, physical_rows) if _replica_context is None else _replica_context.resolutions
    if _replica_context is not None:
        invocation_ids = _replica_context.invocations

    def rank_forward(trace, stage, incoming):
        group = groups[stage]
        expert = None
        if _replica_context is not None:
            replica, tp = _replica_context, config.tensor_parallel_size
            lane, sp = stages[stage].index(trace), replica.moe_sequence_parallel
            expert = _KimiMoEContext(tp, replica.expert_size, replica.rank * tp + lane if replica.expert_size > 1 else 0,
                len(replica.rows), replica.distribution_groups[stage][0 if sp else lane],
                tuple(-(-count // tp) for count in replica.rows for _ in range(tp)) if sp else replica.rows,
                pool.collective_for("all_gatherv"), pool.collective_for("reduce_scatterv"), sp,
                pool.collective_for("all_gather") if tp > 1 else None, rows, lane, sp and config.pipeline_parallel_size == 1)
        own_layers = tuple(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage) if owner == stage)
        network = DeepseekV2KimiModel(trace, model, pool.name, group, collective, token_ids=input_ids,
            q_lens=q_lens, kv_lens=kv_lens, slots=slots, cache_capacity_tokens=cache_capacity_tokens,
            resolutions=resolutions, invocation_ids=invocation_ids, model_input_order=model_input_order,
            num_decodes=num_decodes, expert_parallel_size=config.expert_parallel_size if expert is None else expert.expert_size,
            context_chunk_rows=context_chunk_rows, context_chunk_alignment=source_block_size,
            layer_range=(0, 1) if program is not None else (own_layers[0], own_layers[-1] + 1),
            physical_rows=physical_rows, fa3_hardware=fa3_hardware, expert_context=expert, dummy_run=_dummy_run,
            dummy_workspace_min_rows=min(8 * source_max_model_len, 65536) if _dummy_run else None)
        ids = trace.tensor("input_ids", (physical_rows,), "int32") if network.first_stage else None
        pos = trace.tensor("positions", (physical_rows,), "int64")
        indices = (trace.tensor("logits_indices", (len(q_lens),), "int32")
            if network.last_stage and target_head else None)
        if graph_token_shape is not None:
            for layer in network.layers:
                layer.self_attn.graph_token_shape = graph_token_shape
            trace.begin_capture(key=f"{source_profile}:{graph_token_shape.runtime_mode}:{physical_rows}",
                logical_tokens=rows, model_input_tokens=physical_rows,
                inputs={"input_ids": ids, "positions": pos, "slot_mapping": network.slot_mapping},
                runtime_mode=graph_token_shape.runtime_mode, eager_kinds=graph_eager_kinds)
        hidden_states = network(ids, pos, intermediate_tensors=incoming)
        if graph_token_shape is not None:
            trace.end_capture(hidden_states)
        if not network.last_stage:
            mark_source_scope(trace)
            return hidden_states
        if _dummy_run:
            SymbolicGather(trace, "dummy:selected-hidden", pool.name)(hidden_states, 1, indices=(rows - 1,))
        elif target_head:
            sampled = _plain_greedy_target(trace, model, pool.name, group, gather, hidden_states, indices, q_lens,
                sample_logits=target_sampling_source == "vllm_0_30_plain_greedy_bf16",
                cast_logits=target_sampling_source == "vllm_0_30_plain_sampler_bf16")
        mark_source_scope(trace)
        return None

    def mark_source_scope(trace):
        trace._nodes = {name: replace(node, attributes={**node.attributes,
            "source_projection_only": True, "model_execution_version": "vllm_0_30_0",
            "model_source_profile": source_profile,
            **({"source_dummy_run": True, "source_actual_request_count": 0} if _dummy_run else {})})
            for name, node in trace._nodes.items()}

    if pipeline_source:
        pipeline = ForwardPipelineGroup(tuple(trace for traces in stages for trace in traces),
            namespace=step.step_id if _replica_context is None else f"{step.step_id}:dp{_replica_context.rank}:pipeline", collective_groups=groups if config.tensor_parallel_size > 1 else ())
        pipeline.source_scope = {"model_execution_version": "vllm_0_30_0", "source_projection_only": True,
            "model_source_profile": source_profile,
            "pipeline_layer_partition": tuple(pool.parallel_plan.layer_to_pipeline_stage.count(stage)
                for stage in range(config.pipeline_parallel_size)), "tp_partial_hidden_sent_without_added_reduction": False,
            "numerical_tp_pp_equivalence_bound": False, "normal_capacity_accepted": False, "actual_provider_bound": False}
        if _dummy_run:
            _run_vllm_legacy_dummy_stages(pipeline, rows=rows, hidden_size=model.hidden_size,
                max_batched_tokens=max_batched_tokens, run_stage=rank_forward)
        else:
            _run_vllm030_pipeline_stages(pipeline, pool, rows=rows, hidden_size=model.hidden_size,
                max_batched_tokens=max_batched_tokens, run_stage=rank_forward)
        for trace in pipeline.traces:
            mark_source_scope(trace)
        return pipeline
    for trace in stages[0]:
        rank_forward(trace, 0, None)
    if program is not None:
        return program.bind(groups[0], resolutions=resolutions, invocation_ids=invocation_ids, rows=rows)
    source = stages[0][0] if single_rank else groups[0]
    if reusable:
        program = _KimiDecodeSourceProgram(source, model, pool)
        source_program_cache.store(key, program)
        source = program.snapshot()
    return source


def trace_kimi_k2_data_parallel_forward(model, step, pool, expert_routing, *, input_ids,
        slot_mapping_by_replica, cache_capacity_tokens_by_replica, source_max_model_len, max_batched_tokens,
        source_profile="vllm_0_30_k2_6_int4_sync_explicit_allreduce", target_sampling_source="vllm_0_30_plain_greedy_bf16", source_prompt_lengths_by_replica=None, source_block_size=None, fa3_hardware=None):
    """Complete modular K2 TP/DP/global-EP/PP source, with BF16 prepare.

    PP1 keeps MoE sequence shards across layers; PP>1 restores actual
    request rows before transmitting hidden/residual. Idle replicas run the
    legacy GPUModelRunner one-row metadata=None dummy backbone. Native V2
    metadata/provider selection and complete normal capacity remain unbound.
    """
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from .model_config import is_kimi_k2_data_parallel_model_scope
    from .runtime_inputs import ExpertRoutingInput
    if not isinstance(expert_routing, ExpertRoutingInput):
        raise TypeError("Kimi DP source requires explicit canonical routing input")
    config = pool.parallel_plan.config
    tp, dp, pp = config.tensor_parallel_size, config.data_parallel_size, config.pipeline_parallel_size
    if (source_profile != "vllm_0_30_k2_6_int4_sync_explicit_allreduce" or pool.hardware_id != "nvidia_h200"
            or not is_kimi_k2_data_parallel_model_scope(model, config) or len(pool.ranks) != config.worker_count):
        raise NotImplementedError("Kimi DP source requires0.30.0 H200 whole-group INT4 TP/DP or uniform global EP, CP1")
    moe_sp = config.enable_expert_parallel and tp > 1
    if (not isinstance(input_ids, tuple) or len(input_ids) != step.token_count or not step.requests
            or step.pool_name != pool.name or len({request.request_id for request in step.requests}) != len(step.requests)
            or not isinstance(slot_mapping_by_replica, tuple) or len(slot_mapping_by_replica) != dp
            or not isinstance(cache_capacity_tokens_by_replica, tuple) or len(cache_capacity_tokens_by_replica) != dp):
        raise ValueError("Kimi DP source requires distinct original requests, IDs and per-replica cache coordinates")
    requests, ids = [[] for _ in range(dp)], [[] for _ in range(dp)]
    offset = 0
    for request in step.requests:
        owner = request.data_parallel_rank
        if not 0 <= owner < dp:
            raise ValueError("Kimi DP request owner is outside the physical deployment")
        requests[owner].append(request)
        ids[owner].extend(input_ids[offset:offset + request.scheduled_tokens])
        offset += request.scheduled_tokens
    if source_prompt_lengths_by_replica is not None and (
            not isinstance(source_prompt_lengths_by_replica, tuple) or len(source_prompt_lengths_by_replica) != dp
            or any((not isinstance(lengths, tuple) or len(lengths) != len(own)) if own else lengths is not None
                for own, lengths in zip(requests, source_prompt_lengths_by_replica, strict=True))):
        raise ValueError("Kimi DP prompt lengths must preserve each active replica's original request order; idle is None")
    rows = tuple(sum(request.scheduled_tokens for request in replica) for replica in requests)
    work_rows = tuple(count if count else 1 for count in rows)
    for replica, count in enumerate(rows):
        slots, capacity = slot_mapping_by_replica[replica], cache_capacity_tokens_by_replica[replica]
        if count:
            if not isinstance(slots, tuple) or len(slots) != count or type(capacity) is not int or capacity <= 0:
                raise ValueError("Kimi active DP source needs one explicit slot per actual row and its cache capacity")
        elif slots is not None or capacity is not None:
            raise ValueError("idle Kimi DP source cannot invent request/cache coordinates")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    tensor = tuple(tuple(ForwardCollectiveGroup(traces[(replica * pp + stage) * tp:(replica * pp + stage + 1) * tp],
        namespace=f"{step.step_id}:dp{replica}:stage{stage}:tp") for stage in range(pp)) for replica in range(dp)) if tp > 1 else ()
    distribution = tuple(tuple(ForwardCollectiveGroup(tuple(traces[(replica * pp + stage) * tp + lane]
        for replica in range(dp) for lane in (range(tp) if moe_sp else (data_lane,))),
        namespace=f"{step.step_id}:stage{stage}:distribution{data_lane}")
        for data_lane in range(1 if moe_sp else tp)) for stage in range(pp))
    joined = ForwardCollectiveGroup(traces, namespace=f"{step.step_id}:joined", subgroups=(
        *(group for replica in tensor for group in replica), *(group for stage in distribution for group in stage)))
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    physical_rows = sum(-(-count // tp) * tp for count in work_rows) if moe_sp else sum(work_rows)
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, physical_rows)
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_plan
    ranges = pipeline_layer_ranges_for_plan(pool.parallel_plan, num_layers=model.num_hidden_layers)
    local_config = replace(config, data_parallel_size=1, enable_expert_parallel=False)
    plan = build_parallel_plan(local_config, num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=tuple(end - first for first, end in ranges))
    for replica, count in enumerate(rows):
        local_traces = traces[replica * pp * tp:(replica + 1) * pp * tp]
        groups = tensor[replica] if tp > 1 else ()
        local_pool = replace(pool, ranks=tuple(trace.rank for trace in local_traces), parallel_plan=plan)
        context = _KimiReplicaContext(local_traces, groups, distribution, work_rows, replica, resolutions, invocations,
            config.expert_parallel_size, moe_sp)
        from src.serving.schedule.schedule_step import ScheduledRequest
        local_requests = tuple(requests[replica]) if count else (ScheduledRequest("_dummy_req_0", "decode", 0, 1,
            computed_tokens=1, data_parallel_rank=replica),)
        local_step = replace(step, requests=local_requests, finished_request_ids=(), preempted_request_ids=())
        trace_kimi_k2_forward(model, local_step, local_pool, expert_routing, input_ids=tuple(ids[replica]) if count else (0,),
            positions=tuple(position for request in requests[replica]
                for position in range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens)) if count else (0,),
            slot_mapping=slot_mapping_by_replica[replica] if count else (-1,), cache_capacity_tokens=cache_capacity_tokens_by_replica[replica],
            source_profile=source_profile, target_sampling_source=target_sampling_source,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens, _replica_context=context, source_block_size=source_block_size,
            source_prompt_lengths=None if source_prompt_lengths_by_replica is None else source_prompt_lengths_by_replica[replica],
            _dummy_run=not count, fa3_hardware=fa3_hardware)
    for trace in traces:
        for key, node in tuple(trace._nodes.items()):
            trace._nodes[key] = node.with_attributes({**node.attributes, "source_projection_only": True,
                "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile})
    for group in (*(group for replica in tensor for group in replica), *(group for stage in distribution for group in stage)):
        group._contracts = {name: node.with_attributes({**node.attributes, "source_projection_only": True,
            "model_execution_version": "vllm_0_30_0", "model_source_profile": source_profile})
            for name, node in group._contracts.items()}
    joined.source_scope = {"source_projection_only": True, "model_execution_version": "vllm_0_30_0",
        "source_data_parallel_rows": rows, "source_expert_physical_rows": physical_rows,
        "source_model_rows_including_dummy": work_rows,
        "model_sequence_parallel": moe_sp and pp == 1, "moe_internal_sequence_parallel": moe_sp and pp > 1,
        "source_sp_shape_branch": "v030_literal_hidden_rows_vs_position_rows; GPU runner padding not inferred",
        "source_dispatch_input": "bf16_hidden_fp32_weights_int32_ids_after_local_routing",
        "normal_capacity_accepted": False, "actual_provider_bound": False,
        "idle_source_scope": "legacy GPUModelRunner complete one-row dummy backbone; metadata=None MLA zero output; no cache writes/logits/sampler",
        "dummy_profile_workspace_scope": "source maximum-length lower bound; max_num_seqs/block_size contribution unbound",
        "dummy_input_values_source": "retained legacy input/position buffers; values unknown, no reset",
        "dummy_pipeline_source": "local persistent intermediates; no PP send/recv or TP receive gather",
        "native_v2_dummy_metadata_bound": False}
    return joined


def _validate_qwen3_source(model):
    if (model.model_type != "qwen3" or model.compute_dtype not in ("bf16", "fp16")
            or model.weight_dtype != model.compute_dtype or model.num_experts
            or model.hidden_act != "silu"
            or model.rope_scaling_type != "default" or model.rms_norm_eps is None):
        raise NotImplementedError("symbolic complete model requires unquantized 16-bit default-RoPE Qwen3")


def trace_qwen3_forward(model: ModelConfig, step, *, input_ids, positions,
                        slot_mapping, cache_capacity_tokens, source_profile, rank=0, namespace=None,
                        hardware_id="nvidia_h200", model_execution_version="vllm_0_30_0", sample_logits=True, cast_logits=False,
                        maca_copy_runtime_source=None, ascend_piecewise_source=False):
    _validate_qwen3_source(model)
    if hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b", "nvidia_h20", "nvidia_b200", "nvidia_b300", "ascend_950pr", "ascend_950dt"):
        raise NotImplementedError("dense CUDA-alike forward requires its supported source platform")
    return _trace_dense_forward(model, step, input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, rank=rank, namespace=namespace, model_type=Qwen3Model,
        rope_program=vllm_rope_program(hardware_id, model.rope_scaling_type), hardware_id=hardware_id,
        model_execution_version=model_execution_version, sample_logits=sample_logits, cast_logits=cast_logits,
        maca_copy_runtime_source=maca_copy_runtime_source, ascend_piecewise_source=ascend_piecewise_source)


def trace_llama_forward(model: ModelConfig, step, *, input_ids, positions,
                        slot_mapping, cache_capacity_tokens, source_profile, rank=0, namespace=None,
                        hardware_id="nvidia_h200", model_execution_version="vllm_0_30_0", sample_logits=True, cast_logits=False,
                        maca_copy_runtime_source=None):
    _validate_llama_source(model)
    if hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b", "nvidia_h20", "nvidia_b200", "nvidia_b300", "ascend_950pr", "ascend_950dt"):
        raise NotImplementedError("dense CUDA-alike forward requires its supported source platform")
    return _trace_dense_forward(model, step, input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, rank=rank, namespace=namespace, model_type=LlamaModel,
        rope_program=vllm_rope_program(hardware_id, model.rope_scaling_type), hardware_id=hardware_id,
        model_execution_version=model_execution_version, sample_logits=sample_logits, cast_logits=cast_logits,
        maca_copy_runtime_source=maca_copy_runtime_source)


def _validate_llama_source(model):
    if (model.model_type != "llama" or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16" or model.num_experts
            or model.hidden_act != "silu" or model.attention_bias
            or model.rope_scaling_type not in ("default", "llama3") or model.rms_norm_eps is None):
        raise NotImplementedError("symbolic Llama requires dense BF16 SiLU with default/llama3 cached RoPE")


def trace_target_model_structural_forward(model, step, pool, expert_routing, *,
        hidden_shape, hidden_dtype, representative_layers):
    """A800 TP1 representative GLM/Kimi layers from a typed hidden port.

    Reuses the vLLM0.30 source modules without embedding, head or sampler.
    Selected layers form a representative source composition, not the omitted
    full decoder. GLM sparse decode shares the first layer's top-k buffer;
    fresh dense-MHA prefill does not consume it in the attention backend.
    Fresh short GLM prefill follows dense MHA, rather than forcing sparse MQA
    to match an older artifact. This is no0.27.1/runtime equivalence claim.
    """
    config = pool.parallel_plan.config
    targets = {"zai-org/GLM-5.2": (0, 3), "moonshotai/Kimi-K2.6": (0, 1)}
    if (model.model_id not in targets or representative_layers != targets[model.model_id]
            or model.compute_dtype != "bf16" or pool.hardware_id != "nvidia_a800"
            or config.tensor_parallel_size != 1 or len(pool.ranks) != 1
            or not config.is_rank_local or config.enable_expert_parallel):
        raise NotImplementedError("target structural source requires exact GLM5.2/K2.6 A800 TP1 representative layers")
    if (model.model_type not in ("glm_moe_dsa", "kimi_k2") or step.pool_name != pool.name
            or len(step.requests) != 1):
        raise NotImplementedError("target structural source requires one ordinary representative request")
    request = step.requests[0]
    if (not ((request.phase == "prefill" and request.computed_tokens == 0 and request.scheduled_tokens > 0)
            or (request.phase == "decode" and request.scheduled_tokens == 1 and request.computed_tokens > 0))
            or request.mtp_proposer_mode != "disabled" or request.mtp_proposed_tokens
            or request.mtp_proposer_tokens or request.mtp_verified_tokens or request.mtp_next_proposed_tokens):
        raise NotImplementedError("target structural source requires fresh prefill or single-token decode with MTP off")
    rows, capacity = request.scheduled_tokens, request.computed_tokens + request.scheduled_tokens
    if (hidden_shape != (rows, model.hidden_size) or hidden_dtype != "bf16"
            or not isinstance(hidden_shape, tuple) or any(type(size) is not int for size in hidden_shape)
            or capacity > model.max_position_embeddings):
        raise ValueError("target structural hidden port must match source rows, checkpoint width and BF16 dtype")
    if model.model_type == "glm_moe_dsa" and (model.weight_dtype != "bf16"
            or rows > model.index_topk or tuple(model.indexer_types[layer] for layer in representative_layers) != ("full", "shared")):
        raise NotImplementedError("GLM representative source requires BF16 full/shared indexer and short fresh prefill")
    trace = ForwardTrace(rank=pool.ranks[0], namespace=step.step_id)
    trace.source_scope = {"execution_scope": "representative_decoder_layers_only",
        "input_source": "captured_hidden_shape_dtype", "source_projection_only": True,
        "model_execution_version": "vllm_0_30_0", "representative_layers": representative_layers,
        "embedding_executed": False, "sampler_executed": False,
        "actual_provider_bound": False, "full_decoder_executed": False,
        "source_version_equivalence_claimed": False,
        "prefill_attention_source": "dense_mha_fresh" if request.phase == "prefill" else None}
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)

    def forward():
        slots = tuple(range(request.computed_tokens, capacity))
        slot_mapping = trace.tensor("slot_mapping", (rows,), "int64", persistent=True)
        rope = trace.tensor("rotary_cache", (model.max_position_embeddings, model.qk_rope_head_dim), "bf16", persistent=True)
        layers = []
        if model.model_type == "glm_moe_dsa":
            topk = trace.tensor("topk", (rows, model.index_topk), "int32", persistent=True)
            dummy_q = trace.tensor("dummy:index-q", (1, 1, 1), "bf16", persistent=True)
            dummy_weights = trace.tensor("dummy:index-weights", (1, 1), "fp32", persistent=True)
            workspace = ForwardWorkspace(trace, "worker.workspace", alignment_bytes=256,
                initial_capacity_bytes=0, locked=False)
        for layer in representative_layers:
            prefix = f"layer-{layer}"
            cache = trace.tensor(prefix + ":cache", (capacity, model.kv_lora_rank + model.qk_rope_head_dim),
                "bf16", persistent=True)
            if model.model_type == "glm_moe_dsa":
                index_cache = (trace.tensor(prefix + ":index-cache", (capacity, 132), "uint8", persistent=True)
                    if model.indexer_types[layer] == "full" else None)
                attention = DeepseekV32Attention(trace, model, pool.name, prefix + ":attention",
                    layer_index=layer, tp_size=1, q_lens=(rows,), kv_lens=(capacity,), request_phases=(request.phase,),
                    cache=cache, index_cache=index_cache, topk_buffer=topk, rotary_cache=rope, index_rotary_cache=rope,
                    slot_mapping=slot_mapping, dummy_index_q=dummy_q, dummy_index_weights=dummy_weights,
                    workspace=workspace, max_batched_tokens=rows, selection_source="declared_causal_topk",
                    source_max_model_len=capacity)
                decoder = DeepseekV32DecoderLayer(trace, model, pool.name, prefix, None, None,
                    layer_index=layer, self_attn=attention, resolution=resolutions[layer], invocation_id=invocations[layer])
            else:
                arguments = dict(layer_index=layer, tp_size=1, cache=cache, slot_mapping=slot_mapping,
                    slots=slots, rotary_cache=rope, reduction_group=None, reduction_config=None)
                attention = (DeepseekV2MLAFreshPrefillAttention(trace, model, pool.name, prefix + ":attention",
                    q_lens=(rows,), **arguments) if request.phase == "prefill" else
                    DeepseekV2MLADecodeAttention(trace, model, pool.name, prefix + ":attention",
                        kv_lens=(capacity,), backend_source="vllm_0_30_flashmla_bf16", **arguments))
                decoder = DeepseekV2KimiDecoderLayer(trace, model, pool.name, prefix, None, None,
                    layer_index=layer, attention=attention, resolution=resolutions[layer], invocation_id=invocations[layer])
            layers.append(decoder)
        hidden = trace.tensor("captured-hidden-input", hidden_shape, hidden_dtype)
        positions = trace.tensor("positions", (rows,), "int64")
        residual = None
        for layer in layers:
            hidden, residual = layer(positions, hidden, residual)

    forward()
    return trace


def trace_dense_structural_forward(model, step, pool, *, hidden_shape, hidden_dtype,
                                   model_execution_version, graph_token_shape=None, input_all_reduce=False,
                                   expert_routing=None):
    """Existing GQA decoder plus head from a captured typed hidden port.

    Qwen2Model.forward(inputs_embeds) and LlamaModel.forward(inputs_embeds)
    share this ordinary no-auxiliary body in local vLLM0.11 and0.30.
    Preserve the caller's source identity; this projection does not execute
    embedding, sampler, tensor values, a runtime or a native provider.
    Logical cache slots follow the observed ScheduleStep's disjoint requests.
    """
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_30_0"):
        raise NotImplementedError("structural dense forward requires its declared common source version")
    from .graph_padding import GraphTokenShape
    if graph_token_shape is not None and not isinstance(graph_token_shape, GraphTokenShape):
        raise NotImplementedError("structural dense forward requires explicit typed graph padding")
    parallel = pool.parallel_plan.config
    if model.model_type == "qwen3":
        _validate_qwen3_source(model)
        model_type = Qwen3Model
    elif model.model_type == "llama":
        _validate_llama_source(model)
        model_type = LlamaModel
    elif model.model_type == "qwen3_moe":
        _validate_qwen3_moe_source(model, parallel)
        model_type = Qwen3Model
    else:
        raise NotImplementedError("structural hidden-input forward requires its existing BF16 Qwen3/MoE or Llama source")
    if model.model_type != "qwen3_moe" and expert_routing is not None:
        raise ValueError("structural dense forward cannot consume expert routing")
    tp = parallel.tensor_parallel_size
    if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
            or tp not in (1, 2) or len(pool.ranks) != tp
            or parallel.pipeline_parallel_size != 1 or not parallel.is_rank_local
            or (parallel.enable_expert_parallel and model.model_type != "qwen3_moe")):
        raise NotImplementedError("structural hidden-input forward requires its existing TP1/TP2 PP/SP/CP/DP1 and Qwen3MoE EP2 scope")
    if type(input_all_reduce) is not bool or input_all_reduce != (tp == 2):
        raise ValueError("structural TP2 requires an explicit captured input all-reduce; TP1 does not")
    if (step.pool_name != pool.name or not step.requests
            or any(request.phase not in ("prefill", "decode") or request.scheduled_tokens <= 0
                or (request.phase == "decode" and request.scheduled_tokens != 1)
                or request.mtp_proposer_mode != "disabled" or request.mtp_proposed_tokens
                or request.mtp_proposer_tokens or request.mtp_verified_tokens or request.mtp_next_proposed_tokens
                for request in step.requests)):
        raise NotImplementedError("structural dense forward requires ordinary prefill/decode/mixed with MTP off")
    q_lens = tuple(request.scheduled_tokens for request in step.requests)
    kv_lens = tuple(request.computed_tokens + request.scheduled_tokens for request in step.requests)
    rows = sum(q_lens)
    physical_rows = rows
    if graph_token_shape is not None:
        if (model.model_type not in ("qwen3", "llama")
                or graph_token_shape.logical_tokens != rows
                or graph_token_shape.runtime_mode not in ("FULL", "PIECEWISE")):
            raise NotImplementedError("structural graph padding requires dense TP1/TP2 and matching logical rows")
        physical_rows = graph_token_shape.model_input_tokens
    if (not isinstance(hidden_shape, tuple) or any(type(dimension) is not int for dimension in hidden_shape)
            or hidden_shape != (physical_rows, model.hidden_size) or hidden_dtype != model.compute_dtype):
        raise ValueError("structural hidden input must match captured rows, model width and BF16 dtype")
    if max(kv_lens) > model.max_position_embeddings:
        raise ValueError("structural source contexts exceed the checkpoint rotary extent")
    slots, offset = [], 0
    for request in step.requests:
        slots.extend(range(offset + request.computed_tokens,
            offset + request.computed_tokens + request.scheduled_tokens))
        offset += request.computed_tokens + request.scheduled_tokens
    scope = {"execution_scope": "decoder_and_logits_only", "input_source": "captured_hidden_shape_dtype",
        "model_execution_version": model_execution_version, "source_projection_only": True,
        "embedding_executed": False, "sampler_executed": False, "actual_provider_bound": False,
        "captured_input_all_reduce": input_all_reduce, "logical_tokens": rows,
        "model_input_tokens": physical_rows, "graph_capture_executed": False}
    factories = None
    if model.model_type == "qwen3_moe":
        from functools import partial
        invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
        resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
        factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
            ep_size=parallel.expert_parallel_size) for resolution, invocation in zip(resolutions, invocations, strict=True))
        scope["moe_routing_math_source"] = "vllm_0_30_bf16_softmax_topk"
        scope["moe_router_native_source_equivalence_claimed"] = False
    traces = tuple(ForwardTrace(rank=rank, namespace=step.step_id if tp == 1 else f"{step.step_id}:rank-{rank}")
        for rank in pool.ranks)
    group = None if tp == 1 else ForwardCollectiveGroup(traces, namespace=step.step_id)
    selected_indices = tuple(sum(q_lens[:index + 1]) - 1 for index in range(len(q_lens)))

    def rank_forward(trace):
        trace.source_scope = dict(scope)
        network = model_type(trace, model, pool.name, q_lens, kv_lens, offset, tuple(slots),
            input_mode="inputs_embeds", group=group, mlp_factories=factories,
            collective=None if group is None else pool.collective_for("all_reduce"),
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type), hardware_id=pool.hardware_id)
        if graph_token_shape is not None:
            network.inputs_embeds_shape = hidden_shape
            slot_port = trace.tensor("captured-slot-mapping", (physical_rows,), "int64")
            for layer in network.layers:
                layer.self_attn.attn.graph_token_shape = graph_token_shape
                layer.self_attn.attn.slot_tensor = slot_port
        hidden = trace.tensor("captured-hidden-input", hidden_shape, hidden_dtype)
        if group is not None:
            hidden = group.call("embedding:all-reduce", hidden, pool_name=pool.name,
                config=pool.collective_for("all_reduce"), inplace=False)
        positions = trace.tensor("positions", (physical_rows,), "int64")
        hidden = network(None, positions, inputs_embeds=hidden)
        indices = trace.tensor("logits_indices", (len(q_lens),), "int32")
        selected = SymbolicLogitsRowSelect(trace, "logits-boundary-gather", pool.name)(
            hidden, indices, selected_indices=selected_indices, model_execution_version=model_execution_version)
        if group is None:
            logits, _ = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)(selected)
        else:
            logits = SymbolicTPLogitsProcessor(trace, "lm-head", pool.name, hidden_size=model.hidden_size,
                vocab_size=model.vocab_size, group=group, config=pool.collective_for("all_gather"),
                model_execution_version=model_execution_version)(selected)

    for trace in traces:
        rank_forward(trace)
    if group is not None:
        group.source_scope = dict(scope)
        return group
    return traces[0]


def trace_af_forward(model, step, deployment, expert_routing, *, input_ids, positions, slot_mapping,
                     cache_capacity_tokens, source_profile,
                     max_model_len=None, max_batched_tokens=None, max_requests=None,
                     model_execution_version="vllm_0_30_0", graph_token_shape=None, graph_eager_kinds=()):
    """Project one configured AF pair while executing the common model forward.

    Vanilla upstream vLLM supplies the local modules, not an AF runtime. The
    two full-tensor exchanges are the repository's existing AF service contract.
    """
    from functools import partial
    from src.serving.deployment.deployment_config import worker_role_operator_group, worker_role_phase
    if deployment.mode not in ("af_disagg", "pd_af_disagg"):
        raise NotImplementedError("virtual AF requires the existing standalone attention and FFN pools")
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_30_0"):
        raise ValueError("virtual AF requires its declared supported model version")
    if model_execution_version == "vllm_0_11_0" and model.model_type not in ("qwen3", "llama", "qwen3_moe", "glm4_moe"):
        raise NotImplementedError("virtual AF0.11 requires its existing Qwen/Llama or GLM4 source")
    if graph_token_shape is not None and (model_execution_version != "vllm_0_11_0"
            or model.model_type not in ("qwen3", "llama") or model.num_experts
            or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"):
        raise NotImplementedError("segmented AF capture requires its A8000.11 BF16 dense Qwen/Llama source")
    attention_pool = deployment.worker_pools[step.pool_name]
    if worker_role_operator_group(attention_pool.role) != "attention":
        raise ValueError("virtual AF steps must belong to the attention Scheduler")
    phase = worker_role_phase(attention_pool.role)
    routes = tuple(route for route in deployment.routes if route.payload == "activation"
        and route.source_pool == attention_pool.name)
    if len(routes) != 1:
        raise ValueError("virtual AF requires exactly one outgoing activation route")
    ffn_pool = deployment.worker_pools[routes[0].destination_pool]
    if (worker_role_operator_group(ffn_pool.role) != "ffn" or worker_role_phase(ffn_pool.role) != phase
            or sum(route.payload == "activation" and route.source_pool == ffn_pool.name
                and route.destination_pool == attention_pool.name for route in deployment.routes) != 1):
        raise ValueError("virtual AF requires a reciprocal same-phase FFN partner")
    pools = (attention_pool, ffn_pool)
    if deployment.mode == "af_disagg" and (len(deployment.worker_pools) != 2 or phase != "colocated"):
        raise NotImplementedError("virtual standalone AF requires its two-pool deployment")
    if deployment.mode == "pd_af_disagg":
        if (phase not in ("prefill", "decode")
                or any(pool.parallel_plan.config.tensor_parallel_size != 1
                    or pool.parallel_plan.config.enable_expert_parallel for pool in pools)):
            raise NotImplementedError("virtual PD+AF preserves its existing TP1/EP-off source scope")
        if any(request.phase != phase and not (phase == "decode" and request.phase == "prefill"
                and (request.resumed_from_preemption or request.recomputed_tokens > 0)) for request in step.requests):
            raise ValueError("PD+AF request phase must match its attention owner or explicit decode recompute")
    tp = attention_pool.parallel_plan.config.tensor_parallel_size
    if tp not in (1, 2) or any(pool.hardware_id != "nvidia_a800"
            or pool.parallel_plan.config.tensor_parallel_size != tp
            or pool.parallel_plan.config.pipeline_parallel_size != 1
            or not pool.parallel_plan.config.is_rank_local or len(pool.ranks) != tp for pool in pools):
        raise NotImplementedError("virtual AF requires A800, matching TP1/2 and PP/SP/CP/DP1")
    if attention_pool.parallel_plan.config.enable_expert_parallel:
        raise NotImplementedError("AF expert parallelism belongs to the FFN pool")
    ep = ffn_pool.parallel_plan.config.expert_parallel_size
    if ep > 1:
        from src.serving.deployment.validated_scope import is_validated_standalone_af_ep2_scope
        if not is_validated_standalone_af_ep2_scope(deployment, model):
            raise NotImplementedError("virtual AF EP2 requires the existing source-backed two-layer scope")
    factories = None
    if model.model_type == "qwen3":
        _validate_qwen3_source(model)
        model_type = Qwen3Model
    elif model.model_type == "llama":
        _validate_llama_source(model)
        model_type = LlamaModel
    elif model.model_type == "qwen3_moe":
        config = ffn_pool.parallel_plan.config
        _validate_qwen3_moe_source(model, config)
        if ep == 1 and tp != 1:
            raise NotImplementedError("ordinary virtual AF MoE preserves the existing TP1 domain")
        rows = sum(request.scheduled_tokens for request in step.requests)
        invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
        resolutions = _source_expert_resolutions(model, step, ffn_pool, expert_routing, invocations, rows)
        factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
            ep_size=ep, model_execution_version=model_execution_version)
            for resolution, invocation in zip(resolutions, invocations, strict=True))
        model_type = Qwen3Model
    elif model.model_type == "glm4_moe":
        _validate_glm4_source(model)
        if tp != 1 or ep != 1:
            raise NotImplementedError("virtual GLM4 AF preserves its A800 TP1/EP-off source domain")
        model_type = Glm4MoeModel
    else:
        raise NotImplementedError("virtual AF requires an adapted Qwen3, Qwen3MoE or Llama source model")
    traces = {pool.name: tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}")
        for rank in pool.ranks) for pool in pools}
    groups = {pool.name: ForwardCollectiveGroup(traces[pool.name], namespace=f"{step.step_id}:{pool.name}")
        for pool in pools} if tp > 1 else {}
    transport = ForwardActivationGroup(deployment, traces,
        collective_groups=tuple(groups.values()), graph_token_shape=graph_token_shape,
        graph_eager_kinds=graph_eager_kinds)
    factory = partial(RemoteFFN, attention_pool=attention_pool, ffn_pool=ffn_pool,
        transport=transport, group=None if tp == 1 else groups[ffn_pool.name])
    if model.model_type == "glm4_moe":
        source_inputs = dict(input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_profile=source_profile,
            trace=traces[attention_pool.name][0], ffn_factory=factory, expert_pool=ffn_pool)
        if any(r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests):
            trace_glm4_mtp_forward(model, step, attention_pool, expert_routing,
                max_model_len=max_model_len, max_batched_tokens=max_batched_tokens, max_requests=max_requests,
                **source_inputs)
        else:
            trace_glm4_forward(model, step, attention_pool, expert_routing, **source_inputs)
        return transport
    for trace in traces[attention_pool.name]:
        _trace_dense_forward(model, step, input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cache_capacity_tokens=cache_capacity_tokens, source_profile=source_profile,
            rank=trace.rank, namespace=trace.namespace, model_type=model_type, trace=trace,
            group=None if tp == 1 else groups[attention_pool.name],
            collective=None if tp == 1 else attention_pool.collective_for("all_reduce"),
            gather=None if tp == 1 else attention_pool.collective_for("all_gather"),
            hardware_id=attention_pool.hardware_id, ffn_factory=factory, mlp_factories=factories,
            model_execution_version=model_execution_version,
            graph_token_shape=graph_token_shape, graph_eager_kinds=graph_eager_kinds,
            af_transport=transport,
            fused_embedding=model_execution_version != "vllm_0_11_0")
    if graph_token_shape is not None:
        for group in groups.values():
            _remap_collective_capture_nodes(group)
    return transport


def _run_vllm_legacy_dummy_stages(pipeline, *, rows, hidden_size, max_batched_tokens, run_stage):
    """Legacy _dummy_run uses local intermediate views with sync_self=False."""
    stages = (tuple(group.traces for group in pipeline.collective_groups) if pipeline.collective_groups
        else tuple((trace,) for trace in pipeline.traces))
    for stage, traces in enumerate(stages):
        for trace in traces:
            incoming = None if stage == 0 else {role: trace.tensor("worker.intermediate:" + role,
                (max_batched_tokens, hidden_size), "bf16", persistent=True).narrow(0, 0, rows)
                for role in pipeline.tensor_roles}
            run_stage(trace, stage, incoming)


def _run_vllm030_pipeline_stages(pipeline, pool, *, rows, hidden_size, max_batched_tokens, run_stage,
                                  mark_source_projection=True, residual_blocks=None, hidden_mult=1, intermediate_buffers=None,
                                  dummy_run=False, initialize_intermediate_zero=False, declare_external_dummy_inputs=False):
    """Shared sync tensor-dict receive/gather/copy and original TP lane send.

    Model callbacks own decoder execution and final logits; this routine only
    expresses the pinned0.30 runner/transport's lightweight tensor lifetimes.
    """
    from dataclasses import replace
    groups = pipeline.collective_groups
    stages = (tuple(group.traces for group in groups) if groups
        else tuple((trace,) for trace in pipeline.traces))
    tp = len(stages[0])
    roles = pipeline.tensor_roles
    auxiliary_roles = pipeline.stage_tensor_roles
    if type(hidden_mult) is not int or hidden_mult < 1 or (hidden_mult != 1
            and roles != ("hidden_states",) and auxiliary_roles is None):
        raise ValueError("expanded PP hidden rows require the source's single hidden tensor")
    hidden_shape = (hidden_size,) if hidden_mult == 1 else (hidden_mult, hidden_size)
    if type(dummy_run) is not bool or type(initialize_intermediate_zero) is not bool:
        raise TypeError("pipeline dummy mode and constructor binding must be explicit booleans")
    if type(declare_external_dummy_inputs) is not bool or declare_external_dummy_inputs and (
            not dummy_run or initialize_intermediate_zero or hidden_mult != 1 or auxiliary_roles is not None
            or residual_blocks is not None):
        raise ValueError("external legacy PP inputs require the explicit ordinary dummy buffer path")
    if residual_blocks is not None and (not isinstance(residual_blocks, tuple)
            or len(residual_blocks) != len(stages) or any(type(value) is not int or value < 0 for value in residual_blocks)):
        raise ValueError("pipeline residual blocks require one nonnegative extent per stage")
    def mark_projection():
        if not mark_source_projection:
            return
        for trace in pipeline.traces:
            for identifier, node in tuple(trace._nodes.items()):
                if "model_execution_version" in node.attributes and node.attributes["model_execution_version"] != "vllm_0_30_0":
                    raise ValueError("pipeline node source version conflicts with its vLLM0.30 projection")
                trace._nodes[identifier] = replace(node, attributes={**node.attributes,
                    "source_projection_only": True, "model_execution_version": "vllm_0_30_0"})
    def worker_buffer(trace, name, shape):
        created = False
        if intermediate_buffers is None:
            buffer = trace.tensor("worker.intermediate:" + name, shape, "bf16", persistent=True)
            created = True
        else:
            if trace.rank not in intermediate_buffers:
                intermediate_buffers[trace.rank] = {}
            owned = intermediate_buffers[trace.rank]
            if name not in owned:
                owned[name] = trace.tensor("worker.intermediate:" + name, shape, "bf16", persistent=True)
                created = True
            buffer = owned[name]
            trace._check_tensor(buffer)
            if buffer.shape != shape or buffer.dtype != "bf16":
                raise ValueError("continued PP buffer differs from its original source constructor")
        if created and initialize_intermediate_zero:
            SymbolicFillZero(trace, trace.unique_name("worker.intermediate-constructor-zero:" + name), pool.name)(buffer)
            writer = trace._writers[buffer.view.allocation_id]
            node = trace._nodes[writer]
            trace._nodes[writer] = node.with_attributes({**node.attributes,
                "source_component": "vllm_0_30_V4_PP_intermediate_constructor",
                "source_intermediate_role": name, "constructor_once": True})
        return buffer
    if dummy_run:
        for stage, traces in enumerate(stages):
            for trace in traces:
                prepared = None
                if stage > 0:
                    incoming_roles = roles if auxiliary_roles is None else auxiliary_roles[stage - 1]
                    prepared = {name: worker_buffer(trace, name,
                        (max_batched_tokens,residual_blocks[stage-1],hidden_size) if name=="residual" and residual_blocks is not None else
                        (max_batched_tokens, hidden_size) if name.startswith("aux_hidden_states_")
                        else (max_batched_tokens, *hidden_shape)).narrow(0, 0, rows) for name in incoming_roles}
                    if declare_external_dummy_inputs:
                        for role, port in prepared.items():
                            trace.declare_legacy_dummy_PP_input(port, role=role)
                run_stage(trace, stage, prepared)
        mark_projection()
        return
    gather = pool.collective_for("all_gather") if tp > 1 else None
    incoming = [None] * tp
    for stage, traces in enumerate(stages):
        next_incoming = []
        for lane, trace in enumerate(traces):
            prepared = None
            if stage > 0:
                # recv_tensor_dict retains both full backings through its
                # postprocess gathers, then the runner copies into its buffers.
                wire = incoming[lane]
                incoming_roles = roles if auxiliary_roles is None else auxiliary_roles[stage - 1]
                gathered = ({name: groups[stage].call("receive:" + name, wire[name], pool_name=pool.name,
                    config=gather, inplace=False, dim=0,
                    output_shape=((rows * residual_blocks[stage - 1], hidden_size)
                        if name == "residual" and residual_blocks is not None else (rows, hidden_size)
                        if name.startswith("aux_hidden_states_") else (rows, *hidden_shape)))
                    for name in incoming_roles} if tp > 1 else dict(wire))
                if residual_blocks is not None:
                    gathered["residual"] = gathered["residual"].view_as_shape(
                        rows, residual_blocks[stage - 1], hidden_size)
                incoming[lane] = None
                del wire
                prepared = {}
                # recv gathers follow the transmitted dict; runner copies
                # follow its persistent allocation dict, whose aux slots
                # were reserved in ascending slot order.
                copy_roles = (incoming_roles if auxiliary_roles is None else
                    ("hidden_states", *sorted(name for name in incoming_roles if name != "hidden_states")))
                for name in copy_roles:
                    blocks = residual_blocks[stage - 1] if name == "residual" and residual_blocks is not None else None
                    shape=((max_batched_tokens,hidden_size) if name.startswith('aux_hidden_states_') else
                        (max_batched_tokens,*hidden_shape) if blocks is None else (max_batched_tokens,blocks,hidden_size))
                    buffer=worker_buffer(trace,name,shape)
                    active = buffer if max_batched_tokens == rows else buffer.narrow(0, 0, rows)
                    received = gathered[name]
                    if active.shape != received.shape:
                        active = active.narrow(1, 0, received.shape[1])
                    copied = SymbolicCopy(trace, "receive:runner-copy:" + name, pool.name)(received, active)
                    prepared[name] = copied
                del gathered
            outputs = run_stage(trace, stage, prepared)
            if stage == len(stages) - 1:
                del prepared
                continue
            outgoing_roles = roles if auxiliary_roles is None else auxiliary_roles[stage]
            if auxiliary_roles is not None:
                # PPHandler appends upstream slots after local slots, using
                # the copied worker input dictionary as the relay backing.
                for name in outgoing_roles:
                    if name not in outputs:
                        if prepared is None or name not in prepared:
                            raise ValueError("V4 auxiliary relay is missing its actual upstream slot")
                        outputs[name] = prepared[name]
                if tuple(outputs) != outgoing_roles:
                    raise ValueError("V4 auxiliary relay order differs from the source tensor dictionary")
                nodes = set(trace._nodes)
                consumed = {edge.source for edge in trace._edges if edge.source in nodes and edge.destination in nodes}
                terminals = nodes - consumed
            del prepared
            destination = stages[stage + 1][lane]
            wire = {}
            for name in outgoing_roles:
                tensor = outputs[name]
                length = tensor.shape[0]
                for dimension in tensor.shape[1:]:
                    length *= dimension
                length //= tp
                sent = (tensor.view_as_shape(-1, hidden_size) if tp == 1 and len(tensor.shape) == 3 and hidden_mult == 1
                    else tensor if tp == 1 else tensor.view_as_shape(-1).narrow(0, lane * length, length))
                if tp == 1:
                    wire[name] = pipeline.send(sent, config=pool.pipeline_p2p_for(stage, stage + 1), pool_name=pool.name,
                        tensor_name=name, frontier=outputs["hidden_states"])
                else:
                    full = destination.tensor(destination.unique_name("receive:" + name), tensor.shape, tensor.dtype)
                    received = full.view_as_shape(-1).narrow(0, lane * length, length)
                    wire[name] = pipeline.send(sent, config=pool.pipeline_p2p_for(stage, stage + 1), pool_name=pool.name,
                        tensor_name=name, frontier=outputs["hidden_states"], out=received)
                if auxiliary_roles is not None and name == "hidden_states":
                    # Join the synchronous stage at the send, rather than
                    # adding aux_mean -> reconstructed-hidden and forming a
                    # cycle when the last layer's reconstruction is reused.
                    message = pipeline._previous[trace.rank]
                    for terminal in sorted(terminals):
                        edge = WorkloadEdge(terminal, message, "barrier")
                        if edge not in trace._edges:
                            trace._edges.append(edge)
            next_incoming.append(wire)
            del outputs, tensor, sent, wire
        incoming = next_incoming
    mark_projection()


def trace_dense_pipeline_forward(model, step, pool, *, input_ids, positions, slot_mapping,
                                 cache_capacity_tokens, source_profile, model_execution_version="vllm_0_30_0",
                                 graph_token_shape=None, graph_eager_kinds=(), expert_routing=None):
    """Run the adapted Qwen/Llama forward over the plan's actual PP stages."""
    from src.serving.parallelism.tp import validate_gqa_parallelism
    config = pool.parallel_plan.config
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_30_0"):
        raise ValueError("virtual PP requires its supported declared model version")
    moe = model.model_type == "qwen3_moe" and model_execution_version == "vllm_0_30_0"
    if moe:
        _validate_qwen3_moe_source(model, config, pipeline=True)
        model_scope = is_qwen3_moe_pipeline_model_scope(model, config.tensor_parallel_size, config.pipeline_parallel_size)
    else:
        if model.model_type == "qwen3":
            _validate_qwen3_source(model)
        elif model.model_type == "llama":
            _validate_llama_source(model)
        else:
            raise NotImplementedError("dense pipeline requires an adapted Qwen3 or Llama source")
        validate_gqa_parallelism(model, config)
        model_scope = config.pipeline_parallel_size > 1
    if (not model_scope or pool.hardware_id not in ("nvidia_a800", "nvidia_h200")
            or (config.enable_expert_parallel and not moe) or not config.is_rank_local
            or len(pool.ranks) != config.worker_count):
        raise NotImplementedError("virtual PP requires ordinary BF16 Qwen3 GQA with legal TP/stage-local EP geometry")
    factories = None
    if moe:
        from functools import partial
        invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
        resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, step.token_count)
        factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
            ep_size=config.expert_parallel_size, model_execution_version=model_execution_version)
            for resolution, invocation in zip(resolutions, invocations, strict=True))
    q_lens, kv_lens, rows = _dense_source_metadata(model, step, input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, hardware_id=pool.hardware_id)
    physical_rows = rows if graph_token_shape is None else graph_token_shape.model_input_tokens
    if graph_token_shape is not None and (model_execution_version != "vllm_0_11_0"
            or graph_token_shape.logical_tokens != rows):
        raise NotImplementedError("pipeline capture requires the declared0.11 stage model program")
    tp = config.tensor_parallel_size
    stages = tuple(tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}")
        for rank in pool.physical_ranks_for_pipeline_stage(stage)) for stage in range(config.pipeline_parallel_size))
    traces = tuple(trace for stage in stages for trace in stage)
    groups = (() if tp == 1 else tuple(ForwardCollectiveGroup(stage,
        namespace=f"{step.step_id}:stage-{index}") for index, stage in enumerate(stages)))
    pipeline = ForwardPipelineGroup(traces, namespace=step.step_id, collective_groups=groups)
    mapping = pool.parallel_plan.layer_to_pipeline_stage
    ranges = tuple((mapping.index(stage), mapping.index(stage) + mapping.count(stage))
        for stage in range(config.pipeline_parallel_size))
    if tuple(stage for stage, (start, end) in enumerate(ranges) for _ in range(start, end)) != mapping:
        raise ValueError("dense pipeline requires contiguous layer ownership")
    model_type = LlamaModel if model.model_type == "llama" else Qwen3Model

    def run_stage(trace, stage, lane, incoming):
        group = None if tp == 1 else groups[stage]
        if stage > 0:
            # GPUModelRunner.sync_and_slice_intermediate_tensors copies both
            # received roles into worker-owned buffers before model capture.
            # Only active slices are known here; max_num_tokens backing and
            # allocator reservation are not inferred from a scheduled batch.
            prepared = {}
            for name in ("hidden_states", "residual"):
                buffer = trace.tensor("worker.intermediate:" + name,
                    (physical_rows, model.hidden_size), model.compute_dtype, persistent=True)
                prepared[name] = SymbolicCopy(trace, "receive:runner-copy:" + name, step.pool_name)(incoming[name], buffer)
            incoming = prepared
            trace.source_scope = {"worker_intermediate_active_window_only": True,
                "worker_intermediate_backing_capacity_bound": False, "actual_provider_bound": False}
        network = model_type(trace, model, step.pool_name, q_lens, kv_lens, cache_capacity_tokens,
            slot_mapping, layer_range=ranges[stage], mlp_factories=factories,
            token_ids=None if physical_rows != rows else input_ids,
            group=group, collective=None if tp == 1 else pool.collective_for("all_reduce"),
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type), hardware_id=pool.hardware_id,
            model_execution_version=model_execution_version, fused_embedding=model_execution_version != "vllm_0_11_0")
        ids = trace.tensor("input_ids", (physical_rows,), "int32") if stage == 0 else None
        pos = trace.tensor("positions", (physical_rows,), "int64")
        if graph_token_shape is not None:
            slots = trace.tensor("capture.slot-mapping", (physical_rows,), "int64")
            for layer in network.layers:
                layer.self_attn.attn.graph_token_shape = graph_token_shape
                layer.self_attn.attn.slot_tensor = slots
            inputs = ({"input_ids": ids} if stage == 0 else incoming)
            trace.begin_capture(key=f"{source_profile}:stage-{stage}:{graph_token_shape.runtime_mode}:{physical_rows}",
                logical_tokens=rows, model_input_tokens=physical_rows,
                inputs={**inputs, "positions": pos, "slot_mapping": slots},
                runtime_mode=graph_token_shape.runtime_mode, eager_kinds=graph_eager_kinds,
                source_stage="vllm_pipeline")
        hidden_states = network(ids, pos, intermediate_tensors=incoming)
        if graph_token_shape is not None:
            trace.end_capture(outputs={"hidden_states": hidden_states} if network.last_stage else hidden_states)
        if network.last_stage:
            indices = trace.tensor("logits_indices", (len(q_lens),), "int32")
            sampled = _plain_greedy_target(trace, model, step.pool_name, group,
                None if tp == 1 else pool.collective_for("all_gather"), hidden_states, indices, q_lens,
                model_execution_version=model_execution_version)
            return None
        boundary = pool.pipeline_p2p_for(stage, stage + 1)
        if tp == 1:
            return {name: pipeline.send(hidden_states[name], config=boundary, pool_name=step.pool_name,
                tensor_name=name, frontier=hidden_states["hidden_states"]) for name in ("hidden_states", "residual")}
        # parallel_state.send_tensor_dict/irecv_tensor_dict: each lane sends
        # reshape(TP,-1)[lane]. Receiver allocates a FULL backing for each
        # role, receives its slice, then replaces the dict value by a fresh
        # gathered tensor. The postprocess closures retain both old backings
        # until recv_tensor_dict has completed both gathers.
        destination = stages[stage + 1][lane]
        result = {}
        for name in ("hidden_states", "residual"):
            tensor = hidden_states[name]
            length = physical_rows * model.hidden_size // tp
            sent = tensor.view_as_shape(-1).narrow(0, lane * length, length)
            full = destination.tensor(f"receive:{stage}:{name}", tensor.shape, tensor.dtype)
            received = full.view_as_shape(-1).narrow(0, lane * length, length)
            wire = pipeline.send(sent, config=boundary, pool_name=step.pool_name,
                tensor_name=name, frontier=hidden_states["hidden_states"], out=received)
            if model_execution_version == "vllm_0_11_0":
                # 0.11 recv_tensor_dict receives/gathers one role before
                # allocating the next. Its slice's old full backing dies
                # after this fresh gather, unlike0.30 postprocess closures.
                result[name] = groups[stage + 1].call("receive:" + name, wire,
                    pool_name=step.pool_name, config=pool.collective_for("all_gather"), inplace=False, dim=0,
                    output_shape=(physical_rows, model.hidden_size))
                del wire, received, full
            else:
                result[name] = wire
        return result

    def gather_incoming(trace, stage, wire):
        return {name: groups[stage].call("receive:" + name, wire[name], pool_name=step.pool_name,
            config=pool.collective_for("all_gather"), inplace=False, dim=0,
            output_shape=(physical_rows, model.hidden_size)) for name in ("hidden_states", "residual")}

    incoming = [None] * tp
    for stage, stage_traces in enumerate(stages):
        next_incoming = []
        for lane, trace in enumerate(stage_traces):
            if tp > 1 and stage > 0 and model_execution_version == "vllm_0_30_0":
                gathered = gather_incoming(trace, stage, incoming[lane])
                incoming[lane] = None
            else:
                gathered = incoming[lane]
                if tp > 1 and stage > 0:
                    incoming[lane] = None
            next_incoming.append(run_stage(trace, stage, lane, gathered))
            del gathered
        incoming = next_incoming
    if graph_token_shape is not None:
        for group in groups:
            _remap_collective_capture_nodes(group)
    return pipeline


def trace_dense_parallel_forward(model, step, pool, *, input_ids, positions,
                                 slot_mapping, cache_capacity_tokens, source_profile, graph_token_shape=None,
                                 graph_eager_kinds=(), model_execution_version="vllm_0_30_0", sample_logits=True,
                                 cast_logits=False, graph_inputs=None, fa3_full_aot_source=None):
    from .graph_padding import VLLMDenseGraphInputs
    config = pool.parallel_plan.config
    if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
            or len(pool.ranks) < 2 or len(pool.ranks) != config.tensor_parallel_size
            or config.enable_expert_parallel
            or any(value != 1 for value in (config.pipeline_parallel_size,
                config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
        raise NotImplementedError("parallel dense source requires TP>=2, PP/SP/CP/DP1 and EP-off")
    if model.model_type == "qwen3":
        _validate_qwen3_source(model)
        model_type = Qwen3Model
    elif model.model_type == "llama":
        _validate_llama_source(model)
        model_type = LlamaModel
    else:
        raise NotImplementedError("parallel dense source requires an adapted Qwen3 or Llama model")
    if graph_inputs is not None:
        if not isinstance(graph_inputs, VLLMDenseGraphInputs):
            raise TypeError("dense capture requires its typed retained input state")
        if (graph_token_shape is None or model_execution_version != "vllm_0_30_0"
                or graph_inputs.request_ids != tuple(request.request_id for request in step.requests)):
            raise ValueError("dense retained input state must match this captured vLLM0.30 request order")
        if (len(graph_inputs.retained_padding_token_ids) != graph_token_shape.padding_tokens
                or any(token >= model.vocab_size for token in graph_inputs.retained_padding_token_ids)):
            raise ValueError("dense retained padding token IDs must cover exactly the capture tail within the vocabulary")
    if graph_token_shape is not None:
        if pool.hardware_id == "ascend_910b" and model_execution_version != "vllm_0_18_0":
            raise NotImplementedError("Ascend captured model requires its explicit0.18 source")
        if len(pool.ranks) != 2 or pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"):
            raise NotImplementedError("parallel source capture requires its adapted dense TP2 source")
        masked_capture = (model_execution_version == "vllm_0_11_0"
            or (model_execution_version == "vllm_0_24_0" and pool.hardware_id == "metax_c500")
            or (model_execution_version == "vllm_0_18_0" and pool.hardware_id == "ascend_910b"))
        if (graph_token_shape.model_input_tokens != graph_token_shape.logical_tokens
                and not masked_capture and graph_inputs is None):
            raise NotImplementedError("TP2 captured embedding requires retained padding token IDs; only full buckets are bound")
    declared_full = graph_inputs is not None and graph_inputs.fa3_full_capture_id is not None
    if declared_full != (fa3_full_aot_source is not None):
        raise ValueError("FA3 FULL parallel caller requires paired retained state and AOT capture bounds")
    if fa3_full_aot_source is not None:
        state, max_capture = fa3_full_aot_source
        if (state != graph_inputs or type(max_capture) is not int or max_capture <= 0
                or graph_token_shape is None or max_capture < graph_token_shape.model_input_tokens):
            raise ValueError("FA3 FULL parallel AOT facts must match the actual retained caller and global capture bound")
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    for trace in traces:
        _trace_dense_forward(model, step, input_ids=input_ids, positions=positions,
            slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
            source_profile=source_profile, rank=trace.rank, namespace=trace.namespace,
            model_type=model_type, trace=trace, group=group,
            collective=pool.collective_for("all_reduce"), gather=pool.collective_for("all_gather"),
            fused_embedding=(model_execution_version != "vllm_0_11_0" and pool.hardware_id not in ("metax_c500", "ascend_910b")),
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type), hardware_id=pool.hardware_id,
            graph_token_shape=graph_token_shape, graph_eager_kinds=graph_eager_kinds,
            model_execution_version=model_execution_version, sample_logits=sample_logits, cast_logits=cast_logits,
            retained_padding_token_ids=None if graph_inputs is None else graph_inputs.retained_padding_token_ids,
            fa3_full_aot_source=fa3_full_aot_source)
        if graph_inputs is not None:
            capture = trace.source_capture
            capture["padding_ids"] = "caller_declared_retained_source_values"
            capture["padding_positions"] = 0
            capture["retained_padding_inputs"] = {
                "request_ids": graph_inputs.request_ids, "source_kind": graph_inputs.source_kind,
                "source_id": graph_inputs.source_id, "token_ids": graph_inputs.retained_padding_token_ids}
    if graph_token_shape is not None:
        _remap_collective_capture_nodes(group)
    return group



def _dense_parallel_source_prefix(model, step, pool, arguments, *, mlp_factories=None, tail_forward=None,
                                  binding_target=False, binding_tail_nodes=None, layer_count=1):
    """Execute the actual first model pipeline range on both source ranks."""
    q_lens, kv_lens, rows = _dense_source_metadata(model, step,
        input_ids=arguments["input_ids"], positions=arguments["positions"],
        slot_mapping=arguments["slot_mapping"], cache_capacity_tokens=arguments["cache_capacity_tokens"],
        source_profile=arguments["source_profile"], hardware_id=pool.hardware_id, allow_unknown_input_ids=True)
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    shape = arguments["graph_token_shape"]
    for trace in traces:
        network = Qwen3Model(trace, model, step.pool_name, q_lens, kv_lens,
            arguments["cache_capacity_tokens"], arguments["slot_mapping"],
            group=group, collective=pool.collective_for("all_reduce"),
            token_ids=None if shape.padding_tokens else arguments["input_ids"],
            fused_embedding=False, layer_range=(0, layer_count), hardware_id=pool.hardware_id, mlp_factories=mlp_factories,
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
            model_execution_version=arguments["model_execution_version"])
        ids = trace.tensor("input_ids", (shape.model_input_tokens,), "int32")
        positions = trace.tensor("positions", (shape.model_input_tokens,), "int64")
        slots = trace.tensor("capture.slot-mapping", (shape.model_input_tokens,), "int64")
        for layer in network.layers:
            layer.self_attn.attn.graph_token_shape = shape
            layer.self_attn.attn.slot_tensor = slots
            if arguments["model_execution_version"] == "vllm_0_18_0":
                layer.self_attn.attn.graph_attention_source = "ascend_0_18_full_fia"
        if binding_target:
            indices = tuple(sum(q_lens[:index + 1]) - 1 for index in range(len(q_lens)))
            logits_indices = trace.tensor("logits_indices", (len(indices),), "int32")
        trace.begin_capture(key=f"{arguments['source_profile']}:{shape.runtime_mode}:{shape.model_input_tokens}",
            logical_tokens=rows, model_input_tokens=shape.model_input_tokens,
            inputs={"input_ids": ids, "positions": positions, "slot_mapping": slots},
            runtime_mode=shape.runtime_mode, eager_kinds=arguments["graph_eager_kinds"])
        output = network(ids, positions)
        if network.last_stage:
            # A one-layer source prefix has already executed the model's
            # final norm. Its output is the final hidden tensor, not PP
            # intermediate tensors; no later layer remains for tail_forward.
            capture_output = output
        elif binding_target:
            hidden = (output["hidden_states"] if tail_forward is None else
                tail_forward(trace, group, output["hidden_states"]))
            capture_output, _ = _dense_final_norm(trace, model, pool.name, pool.hardware_id,
                model_execution_version=arguments["model_execution_version"])(
                hidden, output["residual"])
        else:
            capture_output = (output["hidden_states"] if tail_forward is None else
                tail_forward(trace, group, output["hidden_states"]))
        trace.end_capture(capture_output)
        if binding_target:
            before = len(trace._nodes)
            selected, logits, sampled = _dense_forward_target(trace, model, pool.name, capture_output, logits_indices, indices,
                group=group, gather=pool.collective_for("all_gather"),
                model_execution_version=arguments["model_execution_version"], hardware_id=pool.hardware_id,
                sample_logits=arguments["sample_logits"], cast_logits=arguments["cast_logits"], maca_copy_runtime_source=None)
            binding_tail_nodes[trace.rank] = tuple(trace._nodes)[before:]
            del selected, logits, sampled, logits_indices
        # Release source caller references before tracing the next rank. Rank
        # execution order is not a cross-rank storage lifetime order.
        del output, capture_output, network, ids, positions, slots
    _remap_collective_capture_nodes(group)
    return group


def _dense_parallel_metadata_prefix(model, step, pool, arguments, program):
    """Bind the recorded actual two-rank prefix with shared GQA metadata."""
    from copy import copy
    result = copy(program.prefix)
    result.traces = tuple(_dense_metadata_prefix(model, step,
        {**arguments, "hardware_id": pool.hardware_id}, rank_program,
        tp_size=pool.parallel_plan.config.tensor_parallel_size) for rank_program in program.rank_programs)
    return result


def _dense_dynamic_bindings(source, prefix, tail_nodes, model, *, node_bindings=(), collective_attribute_bindings=(), bucket_rebind=False):
    if bucket_rebind:
        return _dense_bucket_bindings(source, prefix, tail_nodes, model,
            collective_attribute_bindings=collective_attribute_bindings)
    nodes, persistent, transient, views = list(node_bindings), {}, {}, {}
    collective_geometry = []
    dynamic_shapes = {"rows", "elements", "m"}
    for full, witness in zip(source.traces, prefix.traces, strict=True):
        pairs = []
        for layer in range(model.num_hidden_layers):
            target, origin = full.qualify(f"layer-{layer}"), witness.qualify("layer-0")
            attention = witness._nodes[origin + ":attention"]
            piecewise = witness.source_captures[0]["runtime_mode"] == "PIECEWISE"
            attention_shapes = (("batch", "q_len", "kv_len")
                if piecewise or ("mathematical_program" in attention.attributes and attention.attributes["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1")
                else ("kv_len",))
            attention_attributes = (("q_lens", "kv_lens", "source_output_rows") if piecewise
                else ("q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens"))
            nodes.append((target + ":attention", origin + ":attention", attention_shapes,
                attention_attributes))
            nodes.append((target + ":attention:kv-update", origin + ":attention:kv-update", ("elements", "seq_len"),
                ("symbolic_slot_mapping",)))
            pairs.extend(((target + ":attention", origin + ":attention"), (target + ":attention:kv-update", origin + ":attention:kv-update")))
        for name in tail_nodes[witness.rank]:
            if name in witness._collective_ids:
                continue
            node = witness._nodes[name]
            nodes.append((name, name, tuple(key for key in node.shape if key in dynamic_shapes),
                ("selected_indices",) if "selected_indices" in node.attributes else ()))
            pairs.append((name, name))
        for identifier, contract in prefix._contracts.items():
            if contract.attributes["operation"] != "all_gather":
                continue
            if full.rank == source.participants[0]:
                collective_geometry.append((identifier, identifier))
            pairs.append((source._submissions[identifier][full.rank], prefix._submissions[identifier][witness.rank]))
        def view_index(trace):
            index = {}
            for view in trace._views.values():
                if view.allocation_id not in index:
                    index[view.allocation_id] = []
                index[view.allocation_id].append(view)
            return index

        full_views = view_index(full)
        witness_views = view_index(witness)
        roots = {}
        for target, origin in pairs:
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs"):
                left, right = full._nodes[target].attributes[field], witness._nodes[origin].attributes[field]
                if len(left) != len(right):
                    return None
                for target_view, origin_view in zip(left, right, strict=True):
                    target_root, origin_root = full._views[target_view].allocation_id, witness._views[origin_view].allocation_id
                    if target_root in roots and roots[target_root] != origin_root:
                        return None
                    roots[target_root] = origin_root
                    views[target_view] = origin_view
        for target_root, origin_root in roots.items():
            if target_root not in full_views or origin_root not in witness_views:
                return None
            left = tuple(full_views[target_root])
            right = tuple(witness_views[origin_root])
            target = persistent if target_root in full._persistent_ids else transient
            target[target_root] = origin_root
            if len(left) != len(right):
                # Shared metadata roots may feed one narrow view per layer;
                # their actual ports above give every occurrence explicitly.
                views[target_root] = origin_root
                continue
            for old, current in zip(left, right, strict=True):
                if (old.shape, old.strides_bytes, old.offset_bytes, old.dtype) != (
                        current.shape, current.strides_bytes, current.offset_bytes, current.dtype):
                    return None
                views[old.view_id] = current.view_id
    return ForwardCollectiveSourceProgram(source, prefix, node_bindings=nodes,
        allocation_bindings=tuple(persistent.items()), transient_allocation_bindings=tuple(transient.items()),
        view_bindings=tuple(views.items()),
        capture_bindings=((0, 0, ("logical_tokens",)),),
        collective_geometry_bindings=tuple(collective_geometry),
        collective_attribute_bindings=collective_attribute_bindings)


def _dense_bucket_bindings(source, prefix, tail_nodes, model, *, collective_attribute_bindings=()):
    nodes, persistent, transient, views = [], {}, {}, {}
    collective_geometry = []
    dynamic_shapes = {"rows", "elements", "m", "tok", "source_rows", "seq_len", "batch", "q_len", "kv_len"}
    dynamic_attributes = {"q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens",
        "symbolic_slot_mapping", "selected_indices", "source_output_rows", "input_outer_groups",
        "index_count", "symbolic_physical_token_extent"}
    for full, witness in zip(source.traces, prefix.traces, strict=True):
        pairs = []
        layer_prefix = witness.qualify("layer-0")
        piecewise = witness.source_captures[0]["runtime_mode"] == "PIECEWISE"
        full_q1 = (not piecewise and witness._nodes[layer_prefix + ":attention"].attributes[
            "source_model_execution_version"] == "vllm_0_11_0")

        def bind_node(target, origin, attention=False):
            if target not in full._nodes or origin not in witness._nodes:
                return False
            node, current = full._nodes[target], witness._nodes[origin]
            if (node.kind, node.dtype, node.variant, node.rank) != (
                    current.kind, current.dtype, current.variant, current.rank):
                return False
            if attention and node.kind is WorkloadKind.KV_UPDATE:
                shapes, attributes = ("elements", "seq_len"), ("symbolic_slot_mapping", "symbolic_physical_token_extent")
            elif attention:
                shapes = (("batch", "q_len", "kv_len") if piecewise or full_q1
                    or ("mathematical_program" in current.attributes
                        and current.attributes["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1")
                    else ("kv_len",))
                attributes = (("q_lens", "kv_lens", "source_output_rows") if piecewise
                    else ("q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens", "source_output_rows") if full_q1
                    else ("q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens"))
            else:
                shapes = tuple(key for key in node.shape if key in dynamic_shapes)
                attributes = tuple(key for key in node.attributes if key in dynamic_attributes)
            nodes.append((target, origin, shapes, attributes))
            pairs.append((target, origin))
            return True

        layer_nodes = {index: tuple(name for name in witness._nodes
            if name.startswith(witness.qualify(f"layer-{index}") + ":")) for index in (0, 1)}
        layer_names = {name for names in layer_nodes.values() for name in names}
        tail = set(tail_nodes[witness.rank])
        for layer in range(model.num_hidden_layers):
            origin_layer = 0 if layer == 0 else 1
            origin_prefix = witness.qualify(f"layer-{origin_layer}")
            target_prefix = full.qualify(f"layer-{layer}")
            for origin in layer_nodes[origin_layer]:
                target = target_prefix + origin[len(origin_prefix):]
                if not bind_node(target, origin, attention=origin in (
                        origin_prefix + ":attention", origin_prefix + ":attention:kv-update")):
                    return None
        for origin in witness._nodes:
            if origin in layer_names or origin in witness._collective_ids or origin in tail:
                continue
            if not bind_node(origin, origin):
                return None
        for origin in tail_nodes[witness.rank]:
            if origin not in witness._collective_ids and not bind_node(origin, origin):
                return None
        for layer in range(model.num_hidden_layers):
            origin_layer = 0 if layer == 0 else 1
            for origin, contract in prefix._contracts.items():
                if ":layer-0:" in origin or ":layer-1:" in origin:
                    if f":layer-{1 - origin_layer}:" in origin:
                        continue
                    target = origin.replace(f":layer-{origin_layer}:", f":layer-{layer}:")
                elif layer:
                    continue
                else:
                    target = origin
                if target not in source._contracts or target not in source._submissions:
                    return None
                if full.rank == source.participants[0]:
                    collective_geometry.append((target, origin))
                pairs.append((source._submissions[target][full.rank],
                    prefix._submissions[origin][witness.rank]))
        full_views = {}
        witness_views = {}
        for trace, index in ((full, full_views), (witness, witness_views)):
            for view in trace._views.values():
                if view.allocation_id not in index:
                    index[view.allocation_id] = []
                index[view.allocation_id].append(view)
        roots = {}
        for target, origin in pairs:
            target_attributes = full._nodes[target].attributes
            origin_attributes = witness._nodes[origin].attributes
            parameters = (set(target_attributes["symbolic_parameter_inputs"])
                if "symbolic_parameter_inputs" in target_attributes else set())
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_tensor_controls"):
                if field not in target_attributes or field not in origin_attributes:
                    continue
                left, right = target_attributes[field], origin_attributes[field]
                if len(left) != len(right):
                    return None
                for target_view, origin_view in zip(left, right, strict=True):
                    if target_view in parameters:
                        continue
                    target_root = full._views[target_view].allocation_id
                    origin_root = witness._views[origin_view].allocation_id
                    if full._writers[target_root] == target or target_root not in roots:
                        roots[target_root] = origin_root
                        views[target_root] = origin_root
                    if target_view in views and views[target_view] != origin_view:
                        old = witness._views[views[target_view]]
                        current = witness._views[origin_view]
                        if (old.shape, old.strides_bytes, old.offset_bytes, old.dtype) != (
                                current.shape, current.strides_bytes, current.offset_bytes, current.dtype):
                            return None
                    views[target_view] = origin_view
        port_views = dict(views)
        for target_root, origin_root in roots.items():
            if target_root not in full_views or origin_root not in witness_views:
                return None
            left, right = tuple(full_views[target_root]), tuple(witness_views[origin_root])
            target = persistent if target_root in full._persistent_ids else transient
            target[target_root] = origin_root
            if len(left) != len(right):
                for old in left:
                    if old.view_id in views:
                        continue
                    geometry = (old.shape, old.strides_bytes, old.offset_bytes, old.dtype)
                    candidates = [origin for target_view, origin in port_views.items()
                        if (full._views[target_view].shape, full._views[target_view].strides_bytes,
                            full._views[target_view].offset_bytes, full._views[target_view].dtype) == geometry]
                    if not candidates:
                        return None
                    views[old.view_id] = candidates[0]
                continue
            for old, current in zip(left, right, strict=True):
                if (old.dtype, old.element_bytes, len(old.shape)) != (
                        current.dtype, current.element_bytes, len(current.shape)):
                    return None
                if old.view_id in views and views[old.view_id] != current.view_id:
                    previous = witness._views[views[old.view_id]]
                    if (previous.shape, previous.strides_bytes, previous.offset_bytes, previous.dtype) != (
                            current.shape, current.strides_bytes, current.offset_bytes, current.dtype):
                        return None
                views[old.view_id] = current.view_id
    return ForwardCollectiveSourceProgram(source, prefix, node_bindings=nodes,
        allocation_bindings=tuple(persistent.items()), transient_allocation_bindings=tuple(transient.items()),
        view_bindings=tuple(views.items()),
        capture_bindings=((0, 0, ("logical_tokens", "model_input_tokens", "key", "inputs", "output")),),
        collective_geometry_bindings=tuple(collective_geometry),
        collective_attribute_bindings=collective_attribute_bindings, geometry_rebind=True)


def _dense_parallel_dynamic_program(model, step, pool, arguments, *, cache, launch_policy):
    if not _DENSE_METADATA_SOURCE_CONTRACT.unchanged():
        source = trace_dense_parallel_forward(model, step, pool, **arguments)
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
        return source
    shape = arguments["graph_token_shape"]
    # Runtime control flow and static source facts stay in the key; the actual
    # prefix supplies physical-bucket and logical-row geometry through bindings.
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens", "graph_token_shape"}
    piecewise = shape.runtime_mode == "PIECEWISE"
    domain = "dense_dynamic_piecewise" if piecewise else "dense_dynamic_q1"
    key = (domain, model, pool.name, pool.hardware_id, pool.ranks, pool.parallel_plan.config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        shape.runtime_mode, tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    if piecewise:
        key += (("source_control_flow", shape.logical_tokens < shape.model_input_tokens,
            all(request.scheduled_tokens == 1 for request in step.requests)),)
    elif shape.logical_tokens == shape.model_input_tokens:
        key += (("source_control_flow", "full_exact"),)
    if arguments["model_execution_version"] == "vllm_0_18_0":
        # FIA's direct K/V and persistent cache reads are different programs.
        key += (("ascend_fia_prefill_no_cache", all(request.computed_tokens == 0 for request in step.requests)),)
    program = cache.lookup(key)
    if piecewise and program is not None and ("_dense_parallel_metadata_guard" not in program.__dict__
            or program._dense_parallel_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        raise ValueError("cached dense piecewise source lost its source metadata contract")
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"], cache_key=key)
        + ":tp2-geometry" if program is None else program.source.namespace)
    current = replace(step, step_id=namespace)
    tail_nodes = {}
    prefix = _dense_parallel_source_prefix(model, current, pool, arguments,
        binding_target=True, binding_tail_nodes=tail_nodes, layer_count=2)
    for trace in prefix.traces:
        trace.bind_launch_policy(launch_policy)
    if program is not None:
        result = program.bind(prefix)
        if result is not None:
            return result
        if piecewise:
            raise ValueError("cached dense piecewise source rejected its declared control-flow and geometry key")
    source = trace_dense_parallel_forward(model, current, pool, **arguments)
    for trace in source.traces:
        trace.bind_launch_policy(launch_policy)
    program = _dense_dynamic_bindings(source, prefix, tail_nodes, model, bucket_rebind=True)
    if program is None:
        return source
    program._dense_parallel_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:
        raise RuntimeError("recorded typed geometry rejected its own actual prefix")
    return result


def _dense_parallel_eager_domain(model, step, pool, arguments):
    config = pool.parallel_plan.config
    return (model.model_type == "qwen3" and model.num_hidden_layers == 64 and model.hidden_size == 5120
        and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
        and (pool.hardware_id, arguments["model_execution_version"]) in (
            ("nvidia_a800", "vllm_0_11_0"), ("nvidia_h200", "vllm_0_11_0"),
            ("metax_c500", "vllm_0_24_0"), ("ascend_910b", "vllm_0_18_0"))
        and (pool.hardware_id != "ascend_910b"
            or any(request.scheduled_tokens != 1 for request in step.requests)
            or all(request.computed_tokens == 0 for request in step.requests))
        and len(pool.ranks) == config.tensor_parallel_size == 2
        and not config.enable_expert_parallel and config.pipeline_parallel_size == 1 and config.is_rank_local
        and arguments["graph_token_shape"] is None
        and not arguments["graph_eager_kinds"] and arguments["graph_inputs"] is None
        and arguments["input_ids"] is None and arguments["sample_logits"] and not arguments["cast_logits"]
        and all(request.mtp_proposer_mode == "disabled" for request in step.requests))


def _dense_parallel_eager_prefix(model, step, pool, arguments):
    q_lens, kv_lens, rows = _dense_source_metadata(model, step,
        input_ids=arguments["input_ids"], positions=arguments["positions"], slot_mapping=arguments["slot_mapping"],
        cache_capacity_tokens=arguments["cache_capacity_tokens"], source_profile=arguments["source_profile"],
        hardware_id=pool.hardware_id, allow_unknown_input_ids=True)
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    for trace in traces:
        network = Qwen3Model(trace, model, pool.name, q_lens, kv_lens,
            arguments["cache_capacity_tokens"], arguments["slot_mapping"], group=group,
            collective=pool.collective_for("all_reduce"), token_ids=arguments["input_ids"],
            fused_embedding=False, layer_range=(0, 2), hardware_id=pool.hardware_id,
            rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
            model_execution_version=arguments["model_execution_version"])
        ids = trace.tensor("input_ids", (rows,), "int32")
        pos = trace.tensor("positions", (rows,), "int64")
        indices = tuple(sum(q_lens[:i + 1]) - 1 for i in range(len(q_lens)))
        logits_indices = trace.tensor("logits_indices", (len(indices),), "int32")
        intermediate = network(ids, pos)
        hidden, residual = _dense_final_norm(trace, model, pool.name, pool.hardware_id,
            model_execution_version=arguments["model_execution_version"])(
            intermediate["hidden_states"], intermediate["residual"])
        selected, logits, output = _dense_forward_target(trace, model, pool.name, hidden, logits_indices, indices,
            group=group, gather=pool.collective_for("all_gather"), model_execution_version=arguments["model_execution_version"],
            hardware_id=pool.hardware_id, sample_logits=arguments["sample_logits"], cast_logits=arguments["cast_logits"],
            maca_copy_runtime_source=None)
        del selected, logits, output, hidden, residual, intermediate, logits_indices, ids, pos, network
    return group


def _dense_eager_origin_suffix(suffix):
    pieces = suffix.split(":")
    return ":".join("layer-0" if piece == "layer-0" else "layer-1"
        if piece.startswith("layer-") and piece[6:].isdigit() else piece for piece in pieces)


def _dense_eager_bindings(source, prefix, model):
    nodes, persistent, transient, views, collectives = [], {}, {}, {}, []
    dynamic_shape_fields = {"rows", "elements", "m", "q_len", "kv_len", "seq_len", "batch", "tok"}
    dynamic_attributes = {"q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens",
        "symbolic_slot_mapping", "selected_indices", "index_count"}
    # Eager attention returns the actual row extent. Grouped Q/K norm
    # materialization likewise takes its outer groups from current T.
    dynamic_attributes.update(("source_output_rows", "input_outer_groups"))
    for full, witness in zip(source.traces, prefix.traces, strict=True):
        pairs = []
        for name, node in full._nodes.items():
            suffix = _dense_eager_origin_suffix(name[len(full.namespace) + 1:])
            origin = witness.qualify(suffix)
            if origin not in witness._nodes:
                return None
            current = witness._nodes[origin]
            if (node.kind, node.dtype, node.variant, node.rank) != (current.kind, current.dtype, current.variant, current.rank):
                return None
            if name not in full._collective_ids:
                fields = dynamic_shape_fields | ({"source_rows"} if "selected_indices" in node.attributes else set())
                nodes.append((name, origin, tuple(key for key in node.shape if key in fields),
                    tuple(key for key in node.attributes if key in dynamic_attributes)))
            pairs.append((name, origin))
        for target, origin in pairs:
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs"):
                left, right = full._nodes[target].attributes[field], witness._nodes[origin].attributes[field]
                if len(left) != len(right):
                    return None
                for target_view, origin_view in zip(left, right, strict=True):
                    old, current = full._views[target_view], witness._views[origin_view]
                    if (old.shape, old.strides_bytes, old.offset_bytes, old.dtype) != (
                            current.shape, current.strides_bytes, current.offset_bytes, current.dtype):
                        return None
                    if target_view not in views:
                        views[target_view] = origin_view
                    mapping = persistent if old.allocation_id in full._persistent_ids else transient
                    if old.allocation_id not in mapping:
                        mapping[old.allocation_id] = current.allocation_id
        for mapping in (persistent, transient):
            for target, origin in tuple(mapping.items()):
                if target not in full._allocations:
                    continue
                views[target] = origin
                left = tuple(view for view in full._views.values() if view.allocation_id == target)
                right = tuple(view for view in witness._views.values() if view.allocation_id == origin)
                if len(left) == len(right):
                    for old, current in zip(left, right, strict=True):
                        if old.view_id not in views:
                            views[old.view_id] = current.view_id
    for target, contract in source._contracts.items():
        suffix = _dense_eager_origin_suffix(target[len(source.namespace) + 1:])
        collectives.append((target, prefix.namespace + ":" + suffix))
    return ForwardCollectiveSourceProgram(source, prefix, node_bindings=nodes,
        allocation_bindings=tuple(persistent.items()), transient_allocation_bindings=tuple(transient.items()),
        view_bindings=tuple(views.items()), collective_geometry_bindings=tuple(collectives))


def _dense_parallel_eager_program(model, step, pool, arguments, *, cache, launch_policy):
    if not _DENSE_METADATA_SOURCE_CONTRACT.unchanged():
        source = trace_dense_parallel_forward(model, step, pool, **arguments)
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
        return source
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens"}
    key = ("dense_eager_geometry", model, pool.name, pool.hardware_id, pool.ranks, pool.parallel_plan.config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    # Attention's all-Q1 program and TP logits' one-row view/reorder are
    # distinct source branches, independent of variable eager geometry.
    key += (("source_control_flow", all(request.scheduled_tokens == 1 for request in step.requests),
        len(step.requests) == 1),)
    if arguments["model_execution_version"] == "vllm_0_18_0":
        # Fresh eager FIA reads contiguous K/V; mixed cached FIA reads pages.
        # Pure cached Q1 may select PA and is outside this reuse domain.
        key += (("ascend_fia_prefill_no_cache", all(request.computed_tokens == 0 for request in step.requests)),)
    program = cache.lookup(key)
    if program is not None and (
            "_dense_eager_metadata_guard" not in program.__dict__
            or program._dense_eager_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        raise ValueError("cached dense eager source lost its source metadata contract")
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"], cache_key=key)
        + ":tp2-eager-geometry" if program is None else program.source.namespace)
    current = replace(step, step_id=namespace)
    prefix = _dense_parallel_eager_prefix(model, current, pool, arguments)
    for trace in prefix.traces:trace.bind_launch_policy(launch_policy)
    if program is not None:
        result = program.bind(prefix)
        if result is not None:return result
        raise ValueError("cached dense eager source rejected its declared control-flow and geometry key")
    source = trace_dense_parallel_forward(model, current, pool, **arguments)
    for trace in source.traces:trace.bind_launch_policy(launch_policy)
    program = _dense_eager_bindings(source, prefix, model)
    if program is None:return source
    program._dense_eager_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:raise RuntimeError("eager geometry rejected its actual prefix")
    return result


def _dense_direct_parameters(model, step, pool, arguments, program):
    """Issue the existing native frame from a proved dense FULL recipe."""
    from copy import deepcopy
    from math import prod
    from types import MappingProxyType
    from .traced_forward import (_native_source_scope, _native_recipe_signature,
        _native_recipe_matches, _bind_native_source_parameters, _SourceParameterSlots)
    shape = arguments["graph_token_shape"]
    if (pool.hardware_id not in ("nvidia_a800", "nvidia_h200")
            or arguments["model_execution_version"] != "vllm_0_11_0"
            or model.compute_dtype != model.weight_dtype or model.compute_dtype != "bf16"
            or shape is None or shape.runtime_mode != "FULL"
            or len(step.requests) != shape.logical_tokens
            or any(request.phase != "decode" or request.scheduled_tokens != 1
                or request.mtp_proposer_mode != "disabled" for request in step.requests)
            or type(program) is not ForwardCollectiveSourceProgram
            or type(program.source) is not ForwardCollectiveGroup
            or type(program.prefix) is not ForwardCollectiveGroup
            or not _native_source_scope(program, program.prefix)
            or program.collective_attribute_bindings or program.collective_geometry_bindings
            or any(type(rank) is not ForwardSourceProgram
                or type(rank.source) is not ForwardTrace or type(rank.prefix) is not ForwardTrace
                or rank.view_bindings or rank.capture_bindings or rank.transient_origins
                for rank in program.rank_programs)
            or any(len(trace.source_captures) != 1
                or trace.source_captures[0]["runtime_mode"] != "FULL"
                or (trace.source_captures[0]["logical_tokens"], trace.source_captures[0]["model_input_tokens"])
                    != (shape.logical_tokens, shape.model_input_tokens)
                for trace in (*program.source.traces, *program.prefix.traces))):
        return None
    if "_dense_parameter_recipe" not in program.__dict__:
        # The ordinary binder owns the first complete source/port/storage proof.
        if program.bind(program.prefix) is None:
            return None
        ranks = []
        for rank in program.rank_programs:
            entries = []
            for target, origin, shapes, attrs in rank.node_bindings:
                node = rank.prefix._nodes[origin]
                if node.kind is WorkloadKind.FLASH_ATTN:
                    if shapes != ("kv_len",) or attrs != ("kv_lens", "model_input_kv_lens"):
                        return None
                    kind = "attention"
                elif node.kind is WorkloadKind.KV_UPDATE:
                    if shapes or attrs != ("symbolic_slot_mapping",):
                        return None
                    kind = "slots"
                else:
                    return None
                entries.append((target, kind))
            origins = tuple(dict.fromkeys(origin for _, origin in rank.allocation_bindings))
            if len(origins) != 1:
                return None
            prefix = rank.prefix
            state = deepcopy((prefix.namespace, prefix.rank, prefix.event_order,
                prefix._persistent_ids, prefix._references, prefix._writers, prefix._readers,
                rank._capture_contract(prefix)))
            layout = MappingProxyType({target: 0 if kind == "attention" else 1 for target, kind in entries})
            ranks.append((layout, origins[0], tuple(prefix._nodes.items()),
                tuple(prefix._allocations.items()), tuple(prefix._views.items()), state))
        group = program.prefix
        collective_state = deepcopy((group.namespace, group.participants, group.participant_nic_ids,
            group.ring_order, group._explicit_ring_order, group._sequences, group._submissions, group._layouts))
        program._dense_parameter_recipe = (_native_recipe_signature(program), tuple(ranks),
            tuple(group._contracts.items()), collective_state)
    recipe = program._dense_parameter_recipe
    if not _native_recipe_matches(program, recipe[0]):
        return None
    for rank, (_, _, nodes, allocations, views, state) in zip(program.rank_programs, recipe[1], strict=True):
        prefix = rank.prefix
        for actual, expected in ((prefix._nodes, nodes), (prefix._allocations, allocations), (prefix._views, views)):
            if len(actual) != len(expected) or any(name not in actual or actual[name] is not value for name, value in expected):
                return None
        if (prefix.namespace, prefix.rank, prefix.event_order, prefix._persistent_ids,
                prefix._references, prefix._writers, prefix._readers, rank._capture_contract(prefix)) != state:
            return None
    group = program.prefix
    if (len(group._contracts) != len(recipe[2]) or any(name not in group._contracts
            or group._contracts[name] is not node for name, node in recipe[2])
            or (group.namespace, group.participants, group.participant_nic_ids, group.ring_order,
                group._explicit_ring_order, group._sequences, group._submissions, group._layouts) != recipe[3]):
        return None
    metadata = _dense_parameter_metadata(model, step, {**arguments, "hardware_id": pool.hardware_id})
    if type(metadata["kv_len"]) is not int or metadata["kv_len"] < 0:
        raise ValueError("source frame requires canonical bound shape dimensions")
    cache_shape = gqa_cache_shape(arguments["cache_capacity_tokens"],
        model.kv_heads_per_tensor_parallel_rank(pool.parallel_plan.config.tensor_parallel_size), model.head_dim)
    rank_values = []
    parameters = (({"kv_len": metadata["kv_len"]}, {"kv_lens": metadata["kv_lens"],
        "model_input_kv_lens": metadata["attention_shape"].model_input_kv_lens}),
        ({}, {"symbolic_slot_mapping": metadata["slot_mapping"]}))
    for rank, (layout, cache, *_) in zip(program.rank_programs, recipe[1], strict=True):
        updates = _SourceParameterSlots(layout, parameters)
        view = rank.prefix._views[cache]
        allocations = {cache: replace(rank.prefix._allocations[cache], size_bytes=prod(cache_shape) * view.element_bytes)}
        views = {cache: ForwardTrace._dense_view(cache, cache_shape, view.dtype)}
        rank_values.append((updates, allocations, views))
    return _bind_native_source_parameters(program, tuple(rank_values), {})


def _dense_parallel_source_program(model, step, pool, arguments, *, cache, launch_policy, execution_context=None):
    """Reuse the recorded Qwen32 BF16 TP2 vendor/version source."""
    config = pool.parallel_plan.config
    shape = arguments["graph_token_shape"]
    if (model.model_type in ("qwen3", "llama") and not model.num_experts
            and model.compute_dtype == model.weight_dtype == "bf16"
            and pool.hardware_id == "nvidia_h200"
            and arguments["model_execution_version"] == "vllm_0_30_0"
            and shape is not None and shape.runtime_mode == "FULL"
            and arguments["graph_inputs"] is not None
            and "fa3_full_aot_source" in arguments
            and not arguments["graph_eager_kinds"]
            and all(request.phase == "decode" and request.scheduled_tokens == 1
                and request.mtp_proposer_mode == "disabled" for request in step.requests)):
        return _dense_retained_full_source_program(model, step, pool, arguments,
            cache=cache, launch_policy=launch_policy, execution_context=execution_context)
    mixed = (model.model_type == "qwen3" and model.num_hidden_layers == 64 and model.hidden_size == 5120
        and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
        and (pool.hardware_id, arguments["model_execution_version"]) in (
            ("nvidia_a800", "vllm_0_11_0"), ("nvidia_h200", "vllm_0_11_0"))
        and len(pool.ranks) == config.tensor_parallel_size == 2
        and not config.enable_expert_parallel and config.pipeline_parallel_size == 1 and config.is_rank_local
        and shape is not None and shape.runtime_mode == "PIECEWISE"
        and arguments["graph_eager_kinds"] == ("flash_attn", "kv_update")
        and arguments["graph_inputs"] is None and arguments["sample_logits"] and not arguments["cast_logits"]
        and {request.phase for request in step.requests} == {"prefill", "decode"}
        and all(request.mtp_proposer_mode == "disabled" for request in step.requests))
    if mixed:
        return _dense_parallel_dynamic_program(model, step, pool, arguments,
            cache=cache, launch_policy=launch_policy)
    supported = (model.model_type == "qwen3" and model.num_hidden_layers == 64 and model.hidden_size == 5120
        and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
        and (pool.hardware_id, arguments["model_execution_version"]) in (
            ("nvidia_a800", "vllm_0_11_0"), ("nvidia_h200", "vllm_0_11_0"),
            ("metax_c500", "vllm_0_24_0"), ("ascend_910b", "vllm_0_18_0"))
        and len(pool.ranks) == config.tensor_parallel_size == 2
        and not config.enable_expert_parallel and config.pipeline_parallel_size == 1 and config.is_rank_local
        and shape is not None and shape.runtime_mode == "FULL" and not arguments["graph_eager_kinds"]
        and arguments["graph_inputs"] is None and arguments["sample_logits"] and not arguments["cast_logits"]
        and all(request.phase == "decode" and request.scheduled_tokens == 1
            and request.mtp_proposer_mode == "disabled" for request in step.requests))
    if not supported:
        return (_dense_parallel_eager_program(model, step, pool, arguments, cache=cache, launch_policy=launch_policy)
            if _dense_parallel_eager_domain(model, step, pool, arguments) else
            trace_dense_parallel_forward(model, step, pool, **arguments))
    if (pool.hardware_id in ("nvidia_a800", "nvidia_h200")
            and arguments["input_ids"] is None and 2 <= shape.logical_tokens == shape.model_input_tokens):
        return _dense_parallel_dynamic_program(model, step, pool, arguments,
            cache=cache, launch_policy=launch_policy)
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens"}
    key = (model, pool.name, pool.hardware_id, pool.ranks, config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    if arguments["model_execution_version"] == "vllm_0_18_0":
        # FIA reads direct K/V for a fresh invocation and persistent cache for
        # the other branch. These complete source programs cannot be shared.
        key += (("ascend_fia_prefill_no_cache", all(request.computed_tokens == 0 for request in step.requests)),)
    program = cache.lookup(key)
    metadata_scope = _DENSE_METADATA_SOURCE_CONTRACT.unchanged()
    if not metadata_scope:
        source = trace_dense_parallel_forward(model, step, pool, **arguments)
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
        return source
    if program is not None and (not metadata_scope
            or "_dense_parallel_metadata_guard" not in program.__dict__
            or program._dense_parallel_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        program = None
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"], cache_key=key)
        + ":tp2" if program is None else program.source.namespace)
    local_step = replace(step, step_id=namespace)
    if program is not None and metadata_scope:
        bound = _dense_direct_parameters(model, step, pool, arguments, program)
        if bound is not None:
            return bound
    prefix = (_dense_parallel_metadata_prefix(model, step, pool, arguments, program)
        if program is not None and metadata_scope else _dense_parallel_source_prefix(model, local_step, pool, arguments))
    if program is not None:
        bound = program.bind(prefix)
        if bound is not None:
            return bound
    if (metadata_scope and pool.hardware_id in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
            and arguments["input_ids"] is None and 2 <= shape.logical_tokens < shape.model_input_tokens):
        # A previously unseen exact geometry can be proved by the shared
        # physical-bucket program. The unchanged constructor below specializes
        # that complete proof into the ordinary exact-key source program.
        source = _dense_parallel_dynamic_program(model, step, pool, arguments, cache=cache, launch_policy=launch_policy)
        if source.namespace != local_step.step_id:
            local_step = replace(step, step_id=source.namespace)
            prefix = _dense_parallel_source_prefix(model, local_step, pool, arguments)
    else:
        source = trace_dense_parallel_forward(model, local_step, pool, **arguments)
    if launch_policy == "injected_launch":
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
    bindings, allocations = [], []
    for trace, current in zip(source.traces, prefix.traces, strict=True):
        for name, node in current._nodes.items():
            if name not in trace._nodes:
                return source
            actual = trace._nodes[name]
            if ((node.kind, node.pool_name, node.shape, node.dtype, node.variant, node.rank) !=
                    (actual.kind, actual.pool_name, actual.shape, actual.dtype, actual.variant, actual.rank)
                    or {key: value for key, value in node.attributes.items() if key != "launch_policy"} !=
                    {key: value for key, value in actual.attributes.items() if key != "launch_policy"}):
                return source
        for layer in range(model.num_hidden_layers):
            target, origin = trace.qualify(f"layer-{layer}"), current.qualify("layer-0")
            bindings.extend(((target + ":attention", origin + ":attention", ("kv_len",),
                ("kv_lens", "model_input_kv_lens")),
                (target + ":attention:kv-update", origin + ":attention:kv-update", (), ("symbolic_slot_mapping",))))
            allocations.append((target + ":kv-cache", origin + ":kv-cache"))
    program = ForwardCollectiveSourceProgram(source, prefix, node_bindings=bindings, allocation_bindings=allocations)
    if metadata_scope:
        program._dense_parallel_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:
        raise RuntimeError("new collective program rejected its own actual source prefix")
    return result


def _dense_retained_full_source_program(model, step, pool, arguments, *, cache, launch_policy, routing=None, execution_context=None):
    """Reuse a complete recorded FULL body with current attention metadata.

    The retained IDs and capture bounds remain static key facts. Request
    identity changes alone do not change this homogeneous dense body; current
    positions, KV lengths, slots and caller identity are rebound every time.
    No native prefix/ABI reconstruction or cached prediction is involved.
    """
    from .traced_forward import _source_frame_traces
    single = pool.parallel_plan.config.tensor_parallel_size == 1 and routing is None
    def record(namespace=None):
        if single:
            inputs = {name: value for name, value in arguments.items() if name != "graph_inputs"}
            if namespace is not None:
                inputs["namespace"] = namespace
            return _trace_dense_forward(model, step, **inputs)
        if routing is None:
            source_step = step if namespace is None else replace(step, step_id=namespace)
            return trace_dense_parallel_forward(model, source_step, pool, **arguments)
        # Routing still resolves against the actual scheduled step. Only
        # source object names use the immutable program's opaque identity.
        return trace_qwen3_moe_forward(model, step, pool, routing, source_namespace=namespace, **arguments)
    if not _DENSE_METADATA_SOURCE_CONTRACT.unchanged():
        return record()
    state = arguments["graph_inputs"]
    shape = arguments["graph_token_shape"]
    if (state.request_ids != tuple(request.request_id for request in step.requests)
            or len(state.retained_padding_token_ids) != shape.padding_tokens):
        raise ValueError("retained FULL source must match the current request rows")
    _dense_source_metadata(model, step, input_ids=arguments["input_ids"],
        positions=arguments["positions"], slot_mapping=arguments["slot_mapping"],
        cache_capacity_tokens=arguments["cache_capacity_tokens"],
        source_profile=arguments["source_profile"], hardware_id=pool.hardware_id)
    static_state = replace(state, request_ids=tuple(f"row-{index}" for index in range(len(state.request_ids))))
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens", "graph_inputs", "fa3_full_aot_source", "namespace"}
    routing_key = None
    if routing is not None:
        invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
        resolutions = _source_expert_resolutions(model, step, pool, routing, invocations, shape.model_input_tokens)
        branches, by_histogram = [], {}
        for resolution, invocation in zip(resolutions, invocations, strict=True):
            counts = resolution.tokens_per_expert
            identity = id(counts)
            if identity in by_histogram:
                retained, branch = by_histogram[identity]
                if retained is not counts:
                    raise RuntimeError("retained branch histogram identity collision")
            else:
                branch = tuple(any(unquantized_expert_source_metadata(model, resolution, invocation,
                    rows=shape.model_input_tokens, expert_parallel_size=pool.parallel_plan.config.expert_parallel_size,
                    expert_rank=rank)["local_tokens_per_expert"]) for rank in range(len(pool.ranks)))
                if type(counts) is tuple and all(type(count) is int for count in counts):
                    by_histogram[identity] = counts, branch
            branches.append(branch)
        branches = tuple(branches)
        routing_key = routing.policy_descriptor_sha256, branches
    communication = (() if single else (pool.collective_for("all_reduce"), pool.collective_for("all_gather")))
    key = ("retained_full", model, pool.name, pool.hardware_id, pool.ranks,
        pool.parallel_plan.config, communication,
        launch_policy, static_state, arguments["fa3_full_aot_source"][1], routing_key,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    namespace_key = key
    context = None
    if execution_context is not None:
        hardware, runtime = execution_context
        context = (hardware, runtime.graph_execution, runtime.norm_source_batch_invariant, runtime.bf16_gemm_model, runtime.bf16_expert_model)
        key += ((id(hardware), *context[1:]),)
    program = cache.lookup(key)
    if program is None:
        source = record(cache.namespace(model.model_type, pool.name, pool.ranks[0],
            arguments["model_execution_version"], cache_key=namespace_key))
        if context is not None:
            source = _bind_full_fa3_gqa_source(source, pool, hardware, arguments["model_execution_version"],
                runtime.graph_execution, state, model=model)
            _annotate_dense_source_program(source, runtime)
        nodes, allocations = [], []
        for trace in _source_frame_traces(source):
            origin = trace.qualify("layer-0")
            for layer in range(model.num_hidden_layers):
                target = trace.qualify(f"layer-{layer}")
                nodes.extend(((target + ":attention", origin + ":attention", ("kv_len",),
                    ("kv_lens", "model_input_kv_lens")),
                    (target + ":attention:kv-update", origin + ":attention:kv-update", (),
                        ("symbolic_slot_mapping",))))
                allocations.append((target + ":kv-cache", origin + ":kv-cache"))
            for component in ("prepare", "copy", "clear"):
                name = trace.qualify("fa3_aot." + component)
                nodes.append((name, name, ("kv_len",) if component == "prepare" else (),
                    ("kv_lens", "model_input_kv_lens", "fa3_full_aot_used_key")))
            if routing is not None:
                nodes.extend((name, name, (), ("tokens_per_expert", "local_tokens_per_expert",
                    "moe_invocation_id", "routing_provenance")) for name, node in trace._nodes.items()
                    if node.kind is WorkloadKind.FUSED_MOE)
        collectives = (() if routing is None else tuple((name, name, ("moe_invocation_id",))
            for name, node in source._contracts.items() if "moe_invocation_id" in node.attributes))
        program = (ForwardSourceProgram(source, source, node_bindings=nodes, allocation_bindings=allocations)
            if single else ForwardCollectiveSourceProgram(source, source,
                node_bindings=nodes, allocation_bindings=allocations, collective_attribute_bindings=collectives))
        if context is not None:
            program._retained_full_execution_context = context
        cache.store(key, program)
    if context is not None:
        direct = _retained_full_parameters(model, step, pool, routing, arguments, program,
            resolutions=resolutions if routing is not None else None)
        if direct is not None:
            for trace in _source_frame_traces(direct):
                trace.source_capture["retained_padding_inputs"] = {
                    "request_ids": state.request_ids, "source_kind": state.source_kind,
                    "source_id": state.source_id, "token_ids": state.retained_padding_token_ids}
            return direct
    prefix = (_dense_metadata_prefix(model, step, arguments, program) if single else
        _dense_parallel_metadata_prefix(model, step, pool, arguments, program) if routing is None else
        _moe_metadata_prefix(model, step, pool, routing, arguments, program))
    if prefix is None:
        raise ValueError("retained FULL expert state changed its recorded source branch")
    bound = program.bind(prefix)
    if bound is None:
        raise RuntimeError("retained FULL source rejected its complete recorded body")
    for trace in _source_frame_traces(bound):
        capture = trace.source_capture
        capture["retained_padding_inputs"] = {
            "request_ids": state.request_ids, "source_kind": state.source_kind,
            "source_id": state.source_id, "token_ids": state.retained_padding_token_ids}
    return bound


def _annotate_dense_source_program(source, runtime):
    """Retain the existing execution annotations on a newly recorded body."""
    from .traced_forward import _source_frame_traces
    for trace in _source_frame_traces(source):
        annotated = {}
        for name, node in trace._nodes.items():
            updates = {}
            if (runtime.norm_source_batch_invariant is False and node.kind is WorkloadKind.NORM
                    and node.variant in ("rmsnorm", "add_rmsnorm")):
                updates["source_batch_invariant"] = False
            if (runtime.bf16_gemm_model is not None and node.kind in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD)
                    and node.dtype == "bf16" and node.attributes["weight_dtype"] == "bf16"
                    and node.attributes["weight_group_size"] == 0 and node.attributes["has_bias"] is False
                    and "weight_quantization" not in node.attributes
                    and ("output_dtype" not in node.attributes or node.attributes["output_dtype"] in (None, "bf16"))):
                updates["bf16_gemm_model"] = runtime.bf16_gemm_model
            annotated[name] = node.with_attributes({**node.attributes, **updates}) if updates else node
        trace._nodes = annotated
        trace._recorded_graph = None
    source._recorded_graph = None
    source.graph.bind_model_execution_version(runtime.model_execution_version)


_RETAINED_FULL_PARAMETERS = weakref.WeakKeyDictionary()


def _retained_full_parameters(model, step, pool, routing, arguments, program, resolutions=None):
    """Feed current Q1/KV/AOT/routing packets to the existing source frame."""
    from math import prod
    from types import MappingProxyType
    from .traced_forward import (_native_source_scope, _native_recipe_signature,
        _native_recipe_matches, _bind_native_source_parameters, _SourceParameterSlots)
    if not _native_source_scope(program, program.prefix):
        return None
    if program not in _RETAINED_FULL_PARAMETERS:
        if program.bind(program.prefix) is None:
            return None
        ranks, layers = [], {}
        for rank in program.rank_programs:
            layout, entries, shared = {}, [], {}
            for target, origin, shapes, attrs in rank.node_bindings:
                base = rank.prefix._nodes[origin]
                if base.kind is WorkloadKind.FLASH_ATTN and shapes == ("kv_len",) and attrs == ("kv_lens", "model_input_kv_lens"):
                    tag = ("attention", base.shape["batch"])
                elif base.kind is WorkloadKind.KV_UPDATE and not shapes and attrs == ("symbolic_slot_mapping",):
                    tag = ("slots", None)
                elif ("fa3_full_aot_component" in base.attributes
                        and attrs == ("kv_lens", "model_input_kv_lens", "fa3_full_aot_used_key")
                        and shapes in ((), ("kv_len",))):
                    tag = ("aot", base.attributes["fa3_full_aot_layout"][0])
                elif base.kind is WorkloadKind.FUSED_MOE and not shapes and attrs == (
                        "tokens_per_expert", "local_tokens_per_expert", "moe_invocation_id", "routing_provenance"):
                    layer = base.attributes["routing_provenance"]["layer_index"]
                    layers[base.attributes["moe_invocation_id"]] = layer
                    tag = ("expert", layer)
                elif base.kind is WorkloadKind.COLLECTIVE and not shapes and attrs == ("moe_invocation_id",):
                    invocation = base.attributes["moe_invocation_id"]
                    candidates = tuple(node.attributes["routing_provenance"]["layer_index"]
                        for node in rank.prefix._nodes.values() if node.kind is WorkloadKind.FUSED_MOE
                        and node.attributes["moe_invocation_id"] == invocation)
                    if len(candidates) != 1:
                        return None
                    tag = ("collective", candidates[0])
                else:
                    return None
                key = (tag, shapes, attrs)
                if tag[0] == "expert" or key not in shared:
                    slot = len(entries)
                    entries.append((base, shapes, tag))
                    shared[key] = slot
                layout[target] = shared[key]
            origins = tuple(dict.fromkeys(origin for target, origin in rank.allocation_bindings))
            if len(origins) != 1:
                return None
            ranks.append((MappingProxyType(layout), tuple(entries), origins[0]))
        joined = tuple((target, layers[program.prefix._contracts[origin].attributes["moe_invocation_id"]])
            for target, origin, attrs in program.collective_attribute_bindings if attrs == ("moe_invocation_id",))
        if len(joined) != len(program.collective_attribute_bindings):
            return None
        _RETAINED_FULL_PARAMETERS[program] = (_native_recipe_signature(program), tuple(ranks), joined)
    signature, ranks, joined = _RETAINED_FULL_PARAMETERS[program]
    if not _native_recipe_matches(program, signature):
        del _RETAINED_FULL_PARAMETERS[program]
        return None
    metadata = _dense_parameter_metadata(model, step, {**arguments, "hardware_id": pool.hardware_id})
    shape = arguments["graph_token_shape"]
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers)) if routing is not None else ()
    if resolutions is None:
        resolutions = (_source_expert_resolutions(model, step, pool, routing, invocations, shape.model_input_tokens)
            if routing is not None else ())
    cache_shape = gqa_cache_shape(arguments["cache_capacity_tokens"],
        model.kv_heads_per_tensor_parallel_rank(pool.parallel_plan.config.tensor_parallel_size), model.head_dim)
    rank_values, placement = [], {}
    for rank, (layout, entries, cache) in zip(program.rank_programs, ranks, strict=True):
        packets = []
        for base, shapes, (kind, value) in entries:
            if kind in ("attention", "aot"):
                padded = metadata["kv_lens"] + (0,) * (value - len(metadata["kv_lens"]))
                attrs = {"kv_lens": metadata["kv_lens"], "model_input_kv_lens": padded}
                if kind == "aot":
                    attrs["fa3_full_aot_used_key"] = padded
                packets.append(({name: metadata[name] for name in shapes}, attrs))
            elif kind == "slots":
                packets.append(({}, {"symbolic_slot_mapping": metadata["slot_mapping"]}))
            elif kind == "collective":
                packets.append(({}, {"moe_invocation_id": invocations[value]}))
            elif kind == "expert":
                resolution = resolutions[value]
                counts = resolution.tokens_per_expert
                key = (id(counts), base.shape["tokens"], base.attributes["expert_parallel_size"], base.attributes["expert_rank"])
                if key not in placement:
                    placement[key] = (counts, unquantized_expert_source_metadata(model, resolution, invocations[value],
                        rows=base.shape["tokens"], expert_parallel_size=base.attributes["expert_parallel_size"],
                        expert_rank=base.attributes["expert_rank"]))
                retained, current = placement[key]
                if retained is not counts:
                    raise RuntimeError("retained source histogram identity collision")
                fields = ("tokens_per_expert", "local_tokens_per_expert", "moe_invocation_id", "routing_provenance")
                if any(current[name] != base.attributes[name] for name in current if name not in fields):
                    return None
                packets.append(({}, {"tokens_per_expert": counts,
                    "local_tokens_per_expert": current["local_tokens_per_expert"],
                    "moe_invocation_id": invocations[value], "routing_provenance": resolution.routing_provenance}))
            else:
                raise AssertionError("uncompiled retained source parameter")
        view = rank.prefix._views[cache]
        allocations = {cache: replace(rank.prefix._allocations[cache], size_bytes=prod(cache_shape) * view.element_bytes)}
        views = {cache: ForwardTrace._dense_view(cache, cache_shape, view.dtype)}
        rank_values.append((_SourceParameterSlots(layout, tuple(packets)), allocations, views))
    return _bind_native_source_parameters(program, tuple(rank_values), {
        target: ({}, {"moe_invocation_id": invocations[layer]}) for target, layer in joined})


def _moe_parallel_source_prefix(model, step, pool, routing, arguments, template, *, binding_target=False, binding_tail_nodes=None):
    """Actual first layer plus current expert source leaves for all later layers."""
    from functools import partial
    shape = arguments["graph_token_shape"]
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, routing, invocations, shape.model_input_tokens)
    factories = tuple(partial(Qwen3MoeSparseMoeBlock, resolution=resolution, invocation_id=invocation,
        ep_size=pool.parallel_plan.config.expert_parallel_size,
        model_execution_version=arguments["model_execution_version"])
        for resolution, invocation in zip(resolutions, invocations, strict=True))

    def tail(trace, group, output):
        original = next(current for current in template.traces if current.rank == trace.rank)
        roots, ports = {}, {}

        def local(name):
            if not name.startswith(trace.namespace + ":"):
                raise ValueError("expert source port belongs to another source program")
            return name[len(trace.namespace) + 1:]

        def port(name):
            if name in ports:
                return ports[name]
            view = original._views[name]
            storage = view.allocation_id
            if storage not in roots:
                root = original._views[storage]
                roots[storage] = trace.tensor(local(storage), root.shape, root.dtype,
                    persistent=storage in original._persistent_ids,
                    memory_domain=original._allocations[storage].memory_domain)
            result = roots[storage] if name == storage else trace.alias(roots[storage], local(name),
                shape=view.shape, strides_bytes=view.strides_bytes, offset_bytes=view.offset_bytes, dtype=view.dtype)
            if (result.view.shape, result.view.strides_bytes, result.view.offset_bytes, result.view.dtype) != (
                    view.shape, view.strides_bytes, view.offset_bytes, view.dtype):
                raise ValueError("expert source input geometry changed from the recorded source")
            ports[name] = result
            return result

        for node in original._nodes.values():
            if node.kind is not WorkloadKind.FUSED_MOE:
                continue
            layer = node.attributes["routing_provenance"]["layer_index"]
            if layer == 0:
                continue  # Already executed by the actual first-layer model.
            names = node.attributes["symbolic_tensor_inputs"]
            expected_roles = ("hidden_states", "routing_weights", "expert_ids", "w13_weight", "w2_weight")
            if (node.attributes["memory_input_roles"] != expected_roles or len(names) != len(expected_roles)
                    or len(set(names)) != len(names)
                    or node.attributes["memory_output_roles"] != ("output",)
                    or node.attributes["expert_parameter_layout"] != "unquantized_checkpoint_expert_major_nk"):
                raise ValueError("expert replay requires the exact declared public input/parameter roles")
            x, weights, ids = (port(names[index]) for index in range(3))
            inplace = original._views[node.attributes["symbolic_tensor_outputs"][0]].allocation_id == x.view.allocation_id
            expert = SymbolicUnquantizedExperts(trace, local(node.node_id), pool.name, model,
                tp_size=pool.parallel_plan.config.tensor_parallel_size, resolution=resolutions[layer],
                invocation_id=invocations[layer], expert_parallel_size=pool.parallel_plan.config.expert_parallel_size,
                expert_rank=node.attributes["expert_rank"], inplace=inplace,
                routing_weight_dtype=node.attributes["routing_weight_dtype"],
                model_execution_version=node.attributes["model_execution_version"])
            output = expert(x, weights, ids)
            name = f"layer-{layer}:all-reduce"
            contract = template._contracts[template.namespace + ":" + name]
            output = group.call(name, output, pool_name=pool.name, config=pool.collective_for("all_reduce"),
                inplace=contract.attributes["symbolic_collective_storage"] == "inplace",
                moe_invocation_id=invocations[layer])
        return output

    return _dense_parallel_source_prefix(model, replace(step, step_id=template.namespace), pool, arguments,
        mlp_factories=factories, tail_forward=tail,
        binding_target=binding_target, binding_tail_nodes=binding_tail_nodes)


def _moe_exact_source_bindings(source, prefix, model):
    bindings, allocations, collectives = [], [], []
    expert_fields = ("tokens_per_expert", "local_tokens_per_expert", "moe_invocation_id", "routing_provenance")
    for trace, current in zip(source.traces, prefix.traces, strict=True):
        for layer in range(model.num_hidden_layers):
            target, origin = trace.qualify(f"layer-{layer}"), current.qualify("layer-0")
            attention_attributes = ("kv_lens", "model_input_kv_lens")
            bindings.extend(((target + ":attention", origin + ":attention", ("kv_len",), attention_attributes),
                (target + ":attention:kv-update", origin + ":attention:kv-update", (), ("symbolic_slot_mapping",))))
            allocations.append((target + ":kv-cache", origin + ":kv-cache"))
        for name, node in current._nodes.items():
            if node.kind is WorkloadKind.FUSED_MOE:
                bindings.append((name, name, (), expert_fields))
        if trace.rank == source.participants[0]:
            collectives.extend((identifier, identifier, ("moe_invocation_id",))
                for identifier, node in prefix._contracts.items() if "moe_invocation_id" in node.attributes)
    return ForwardCollectiveSourceProgram(source, prefix, node_bindings=bindings,
        allocation_bindings=allocations, collective_attribute_bindings=collectives)


def _moe_dynamic_bindings(source, prefix, tail_nodes, model):
    expert_fields = ("tokens_per_expert", "local_tokens_per_expert", "moe_invocation_id", "routing_provenance")
    nodes = tuple((name, name, (), expert_fields)
        for trace in prefix.traces for name, node in trace._nodes.items()
        if node.kind is WorkloadKind.FUSED_MOE)
    collectives = tuple((name, name, ("moe_invocation_id",))
        for name, node in prefix._contracts.items() if "moe_invocation_id" in node.attributes)
    return _dense_dynamic_bindings(source, prefix, tail_nodes, model,
        node_bindings=nodes, collective_attribute_bindings=collectives)


def _moe_parallel_dynamic_program(model, step, pool, routing, arguments, *, cache, launch_policy, exact_key):
    """Prove each new logical extent with current source operations and ports."""
    shape = arguments["graph_token_shape"]
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, routing, invocations, shape.model_input_tokens)
    # Preserve the actual per-layer/per-rank source branch, even when a
    # later routing realization leaves a rank with no local assignments.
    branch = tuple(tuple(any(unquantized_expert_source_metadata(model, resolution, invocation,
        rows=shape.model_input_tokens, expert_parallel_size=pool.parallel_plan.config.expert_parallel_size,
        expert_rank=rank)["local_tokens_per_expert"]) for rank in range(len(pool.ranks)))
        for resolution, invocation in zip(resolutions, invocations, strict=True))
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens", "graph_token_shape"}
    domain = "moe_dynamic_q1" if shape.runtime_mode == "FULL" else "moe_dynamic_piecewise"
    key = (domain, model, pool.name, pool.hardware_id, pool.ranks, pool.parallel_plan.config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        routing.policy_descriptor_sha256, shape.runtime_mode, shape.model_input_tokens, branch,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    if shape.runtime_mode == "PIECEWISE":
        key += (("source_control_flow", shape.logical_tokens < shape.model_input_tokens,
            all(request.scheduled_tokens == 1 for request in step.requests)),)
    program = cache.lookup(key)
    if program is not None and ("_moe_metadata_guard" not in program.__dict__
            or program._moe_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        if exact_key is None:
            raise ValueError("cached piecewise source lost its source metadata contract")
        program = None
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"], cache_key=key)
        + ":ep2-geometry" if program is None else program.source.namespace)
    if program is None:
        source = trace_qwen3_moe_forward(model, step, pool, routing, source_namespace=namespace, **arguments)
        if launch_policy == "injected_launch":
            for trace in source.traces:
                trace.bind_launch_policy(launch_policy)
    else:
        source = program.source
    tail_nodes = {}
    prefix = _moe_parallel_source_prefix(model, step, pool, routing, arguments, source,
        binding_target=True, binding_tail_nodes=tail_nodes)
    if launch_policy == "injected_launch":
        for trace in prefix.traces:
            trace.bind_launch_policy(launch_policy)
    def specialize(result):
        if exact_key is None:
            return result
        exact = _moe_exact_source_bindings(result, prefix, model)
        exact._moe_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
        cache.store(exact_key, exact)
        return result

    if program is not None:
        result = program.bind(prefix)
        if result is not None:
            return specialize(result)
        if exact_key is None:
            raise ValueError("cached piecewise source rejected its declared control-flow and geometry key")
        # A changed current source contract resumes the ordinary complete
        # source. No cross-domain prediction or cached duration is supplied.
        source = trace_qwen3_moe_forward(model, step, pool, routing, source_namespace=namespace, **arguments)
        if launch_policy == "injected_launch":
            for trace in source.traces:
                trace.bind_launch_policy(launch_policy)
        tail_nodes = {}
        prefix = _moe_parallel_source_prefix(model, step, pool, routing, arguments, source,
            binding_target=True, binding_tail_nodes=tail_nodes)
        if launch_policy == "injected_launch":
            for trace in prefix.traces:
                trace.bind_launch_policy(launch_policy)
    program = _moe_dynamic_bindings(source, prefix, tail_nodes, model)
    if program is None:
        return source
    program._moe_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:
        raise RuntimeError("new MoE geometry rejected its own actual source prefix")
    return specialize(result)



_MOE_PARAMETER_RECIPES = weakref.WeakKeyDictionary()


def _moe_direct_parameters(model, step, pool, routing, arguments, program):
    """Bind actual dynamic parameters directly to the recorded native recipe."""
    from math import prod
    from types import MappingProxyType
    from .traced_forward import (_native_source_scope, _native_recipe_signature,
        _native_recipe_matches, _bind_native_source_parameters, _SourceParameterSlots)
    shape = arguments["graph_token_shape"]
    if (shape is None or type(shape.logical_tokens) is not int or shape.logical_tokens <= 0
            or type(shape.model_input_tokens) is not int or shape.model_input_tokens < shape.logical_tokens
            or arguments["model_execution_version"] not in ("vllm_0_11_0", "vllm_0_24_0", "vllm_0_18_0")
            or not _native_source_scope(program, program.prefix)):
        return None
    if (shape.runtime_mode != "FULL" or any(request.phase != "decode" or request.scheduled_tokens != 1 for request in step.requests)
            or len(step.requests) != shape.logical_tokens
            or any(len(trace.source_captures) != 1
                or trace.source_captures[0]["runtime_mode"] != "FULL"
                or (trace.source_captures[0]["logical_tokens"],trace.source_captures[0]["model_input_tokens"]) != (shape.logical_tokens,shape.model_input_tokens)
                for trace in (*program.source.traces,*program.prefix.traces))):
        return None
    if program in _MOE_PARAMETER_RECIPES:
        recipe = _MOE_PARAMETER_RECIPES[program]
        if not _native_recipe_matches(program,recipe[0]):
            del _MOE_PARAMETER_RECIPES[program]
    if program not in _MOE_PARAMETER_RECIPES:
        # Compile from the same complete current prefix contracts, once.
        # Unknown or late static edits keep the ordinary source path.
        if program.bind(program.prefix) is None:
            return None
        ranks = []
        for index,rank_program in enumerate(program.rank_programs):
            if rank_program.prefix is not program.prefix.traces[index]:return None
            if rank_program.view_bindings or rank_program.capture_bindings or rank_program.transient_origins:
                return None
            entries = []
            for target, origin, shapes, attrs in rank_program.node_bindings:
                base = rank_program.prefix._nodes[origin]
                if base.kind is WorkloadKind.FLASH_ATTN:
                    if any(name != "kv_len" for name in shapes) or attrs != ("kv_lens","model_input_kv_lens"):
                        return None
                    tag = ("attention",None)
                elif base.kind is WorkloadKind.KV_UPDATE:
                    if shapes or attrs != ("symbolic_slot_mapping",): return None
                    tag = ("slots",None)
                elif base.kind is WorkloadKind.FUSED_MOE:
                    if shapes or attrs != ("tokens_per_expert","local_tokens_per_expert","moe_invocation_id","routing_provenance"):
                        return None
                    tag = ("expert",base.attributes["routing_provenance"]["layer_index"])
                elif base.kind is WorkloadKind.COLLECTIVE:
                    if shapes or attrs != ("moe_invocation_id",):return None
                    invocation = base.attributes["moe_invocation_id"]
                    layers = tuple(node.attributes["routing_provenance"]["layer_index"]
                        for node in rank_program.prefix._nodes.values() if node.kind is WorkloadKind.FUSED_MOE
                        and node.attributes["moe_invocation_id"] == invocation)
                    if len(layers) != 1:return None
                    tag = ("collective",layers[0])
                else:return None
                entries.append((target,origin,shapes,attrs,base,tag))
            allocation_origins = tuple(dict.fromkeys(origin for target,origin in rank_program.allocation_bindings))
            if len(allocation_origins) != 1:return None
            cache, = allocation_origins
            # The cold source recipe names every target. Only pure identical
            # attention/slot/invocation packets share a slot; every expert entry
            # retains its independent placement and provenance validation.
            layout, shared_slots, packet_entries = {}, {}, []
            for entry in entries:
                target, _, shapes, _, _, (kind, layer) = entry
                if kind == "expert":
                    slot = len(packet_entries)
                    packet_entries.append(entry)
                else:
                    key = (kind, shapes) if kind == "attention" else (kind, layer)
                    if key not in shared_slots:
                        shared_slots[key] = len(packet_entries)
                        packet_entries.append(entry)
                    slot = shared_slots[key]
                layout[target] = slot
            ranks.append((tuple(rank_program.prefix._nodes.items()),
                tuple(packet_entries),cache,MappingProxyType(layout)))
        joined = []
        for target,origin,attrs in program.collective_attribute_bindings:
            if attrs != ("moe_invocation_id",):return None
            invocation = program.prefix._contracts[origin].attributes["moe_invocation_id"]
            layers = tuple(node.attributes["routing_provenance"]["layer_index"]
                for node in program.prefix.traces[0]._nodes.values() if node.kind is WorkloadKind.FUSED_MOE
                and node.attributes["moe_invocation_id"] == invocation)
            if len(layers) != 1:return None
            joined.append((target,layers[0]))
        _MOE_PARAMETER_RECIPES[program] = (_native_recipe_signature(program),tuple(ranks),
            tuple(program.prefix._contracts.items()),tuple(joined))
    recipe = _MOE_PARAMETER_RECIPES[program]
    for rank_program,(expected,entries,cache,layout) in zip(program.rank_programs,recipe[1],strict=True):
        if (len(rank_program.prefix._nodes) != len(expected)
                or any(name not in rank_program.prefix._nodes or rank_program.prefix._nodes[name] is not node
                    for name,node in expected)):
            return None
    if (len(program.prefix._contracts) != len(recipe[2]) or any(name not in program.prefix._contracts
            or program.prefix._contracts[name] is not node for name,node in recipe[2])):
        return None
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, routing, invocations, shape.model_input_tokens)
    metadata = _dense_parameter_metadata(model,step,{**arguments,"hardware_id":pool.hardware_id})
    if type(metadata["kv_len"]) is not int or metadata["kv_len"] < 0:
        raise ValueError("source frame requires canonical bound shape dimensions")
    cache_shape = gqa_cache_shape(arguments["cache_capacity_tokens"],
        model.kv_heads_per_tensor_parallel_rank(pool.parallel_plan.config.tensor_parallel_size),model.head_dim)
    placement = {};rank_values = []
    for rank_program,(expected,entries,cache,layout) in zip(program.rank_programs,recipe[1],strict=True):
        packets = []
        for target,origin,shapes,attrs,base,(kind,layer) in entries:
            if kind == "attention":
                packets.append(({name:metadata[name] for name in shapes},
                    {"kv_lens":metadata["kv_lens"],"model_input_kv_lens":metadata["attention_shape"].model_input_kv_lens}))
            elif kind == "slots":
                packets.append(({},{"symbolic_slot_mapping":metadata["slot_mapping"]}))
            elif kind == "collective":
                packets.append(({},{"moe_invocation_id":invocations[layer]}))
            elif kind == "expert":
                resolution = resolutions[layer]
                counts = resolution.tokens_per_expert
                key = (id(counts),base.shape["tokens"],base.attributes["expert_parallel_size"],base.attributes["expert_rank"])
                if key not in placement:
                    placement[key] = (counts,unquantized_expert_source_metadata(model,resolution,invocations[layer],
                        rows=base.shape["tokens"],expert_parallel_size=base.attributes["expert_parallel_size"],
                        expert_rank=base.attributes["expert_rank"]))
                retained,current = placement[key]
                if retained is not counts:raise RuntimeError("source placement count identity collision")
                if any(current[name] != base.attributes[name] for name in current if name not in attrs):return None
                packets.append(({},{"tokens_per_expert":counts,
                    "local_tokens_per_expert":current["local_tokens_per_expert"],
                    "moe_invocation_id":invocations[layer],"routing_provenance":resolution.routing_provenance}))
            else:raise AssertionError("uncompiled source parameter kind")
        view = rank_program.prefix._views[cache]
        allocations = {cache:replace(rank_program.prefix._allocations[cache],size_bytes=prod(cache_shape)*view.element_bytes)}
        views = {cache:ForwardTrace._dense_view(cache,cache_shape,view.dtype)}
        rank_values.append((_SourceParameterSlots(layout,tuple(packets)),allocations,views))
    collective_values = {target:({},{"moe_invocation_id":invocations[layer]})for target,layer in recipe[3]}
    return _bind_native_source_parameters(program,tuple(rank_values),collective_values)

def _moe_metadata_prefix(model, step, pool, routing, arguments, program):
    """Bind an actual recorded prefix; no source operator/port is synthesized."""
    from copy import copy
    from .runtime_inputs import ExpertRoutingResolution
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, routing, invocations,
        arguments["graph_token_shape"].model_input_tokens)
    result = copy(program.prefix)
    result.traces = tuple(_dense_metadata_prefix(model, step,
        {**arguments, "hardware_id": pool.hardware_id}, rank_program,
        tp_size=pool.parallel_plan.config.tensor_parallel_size) for rank_program in program.rank_programs)
    result._contracts = dict(program.prefix._contracts)
    layers_by_invocation = {}
    dynamic_fields = ("tokens_per_expert", "local_tokens_per_expert", "moe_invocation_id", "routing_provenance")
    placement_by_identity = {}
    native_counts_by_identity = {}
    for trace in result.traces:
        for name, node in tuple(trace._nodes.items()):
            if node.kind is not WorkloadKind.FUSED_MOE:
                continue
            layer = node.attributes["routing_provenance"]["layer_index"]
            layers_by_invocation[node.attributes["moe_invocation_id"]] = layer
            resolution = resolutions[layer]
            native = type(model) is ModelConfig and type(resolution) is ExpertRoutingResolution
            counts = resolution.tokens_per_expert if native else None
            identity = id(counts)
            native = native and type(counts) is tuple
            if native:
                if identity not in native_counts_by_identity:
                    native_counts_by_identity[identity] = (counts, all(type(value) is int for value in counts))
                retained, native = native_counts_by_identity[identity]
                if retained is not counts:
                    raise RuntimeError("source placement count identity collision")
            rows, ep_size, ep_rank = node.shape["tokens"], node.attributes["expert_parallel_size"], node.attributes["expert_rank"]
            native = native and all(type(value) is int for value in (rows, ep_size, ep_rank))
            key = (identity, rows, ep_size, ep_rank)
            if native and key in placement_by_identity:
                # The pure helper's placement depends only on this actual
                # immutable histogram and physical expert partition. Every
                # layer keeps its current invocation and routing ledger.
                current = {**placement_by_identity[key], "moe_invocation_id": invocations[layer],
                    "routing_provenance": resolution.routing_provenance}
            else:
                current = unquantized_expert_source_metadata(model, resolution, invocations[layer],
                    rows=rows, expert_parallel_size=ep_size, expert_rank=ep_rank)
                if native:
                    placement_by_identity[key] = current
            if any(current[key] != node.attributes[key] for key in current if key not in dynamic_fields):
                return None
            trace._nodes[name] = node.with_attributes({**node.attributes,
                **{key: current[key] for key in dynamic_fields}})
    for identifier, contract in tuple(result._contracts.items()):
        if "moe_invocation_id" not in contract.attributes:
            continue
        invocation = invocations[layers_by_invocation[contract.attributes["moe_invocation_id"]]]
        result._contracts[identifier] = contract.with_attributes({**contract.attributes, "moe_invocation_id": invocation})
        for trace in result.traces:
            local = result._submissions[identifier][trace.rank]
            node = trace._nodes[local]
            trace._nodes[local] = node.with_attributes({**node.attributes, "moe_invocation_id": invocation})
    return result


def _moe_parallel_source_program(model, step, pool, routing, arguments, *, cache, launch_policy, execution_context=None):
    """Bind current MoE metadata without changing its real-step routing identity."""
    shape = arguments["graph_token_shape"]
    if (_vllm030_qwen3_moe_full_source(model, pool, arguments["model_execution_version"])
            and shape is not None and shape.runtime_mode == "FULL"
            and arguments["graph_inputs"] is not None and "fa3_full_aot_source" in arguments
            and not arguments["graph_eager_kinds"]
            and all(request.phase == "decode" and request.scheduled_tokens == 1
                and request.mtp_proposer_mode == "disabled" for request in step.requests)):
        return _dense_retained_full_source_program(model, step, pool, arguments,
            cache=cache, launch_policy=launch_policy, routing=routing, execution_context=execution_context)
    mixed = (_vllm011_qwen3_moe_ep2_source(model, pool)
        and arguments["model_execution_version"] == "vllm_0_11_0"
        and shape is not None and shape.runtime_mode == "PIECEWISE"
        and arguments["graph_eager_kinds"] == ("flash_attn", "kv_update")
        and arguments["sample_logits"] and not arguments["cast_logits"]
        and routing.source_kind in ("explicit_assumption", "profile_observation")
        and {request.phase for request in step.requests} == {"prefill", "decode"}
        and all(request.mtp_proposer_mode == "disabled" for request in step.requests))
    if mixed and _DENSE_METADATA_SOURCE_CONTRACT.unchanged():
        return _moe_parallel_dynamic_program(model, step, pool, routing, arguments,
            cache=cache, launch_policy=launch_policy, exact_key=None)
    supported = (_versioned_qwen3_moe_ep2_source(model, pool, arguments["model_execution_version"])
        and shape is not None and shape.runtime_mode == "FULL" and not arguments["graph_eager_kinds"]
        and arguments["sample_logits"] and not arguments["cast_logits"]
        and routing.source_kind in ("explicit_assumption", "profile_observation")
        and all(request.phase == "decode" and request.scheduled_tokens == 1
            and request.mtp_proposer_mode == "disabled" for request in step.requests))
    if not supported:
        return trace_qwen3_moe_forward(model, step, pool, routing, **arguments)
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens"}
    key = ("moe", model, pool.name, pool.hardware_id, pool.ranks, pool.parallel_plan.config,
        pool.collective_for("all_reduce"), pool.collective_for("all_gather"), launch_policy,
        routing.policy_descriptor_sha256,
        tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    if arguments["model_execution_version"] == "vllm_0_18_0":
        # FIA direct K/V and persistent-cache readers are distinct source programs.
        key += (("ascend_fia_prefill_no_cache", all(request.computed_tokens == 0 for request in step.requests)),)
    program = cache.lookup(key)
    metadata_domain = (pool.hardware_id, arguments["model_execution_version"]) in (
        ("nvidia_h200", "vllm_0_11_0"), ("nvidia_a800", "vllm_0_11_0"),
        ("metax_c500", "vllm_0_24_0"), ("ascend_910b", "vllm_0_18_0"))
    metadata_scope = metadata_domain and _DENSE_METADATA_SOURCE_CONTRACT.unchanged()
    if program is not None and metadata_domain and (not metadata_scope
            or "_moe_metadata_guard" not in program.__dict__
            or program._moe_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
        program = None
    if program is not None:
        if metadata_scope:
            direct = _moe_direct_parameters(model, step, pool, routing, arguments, program)
            if direct is not None:
                return direct
        prefix = (_moe_metadata_prefix(model, step, pool, routing, arguments, program) if metadata_scope
            else _moe_parallel_source_prefix(model, step, pool, routing, arguments, program.source))
        if prefix is None:
            prefix = _moe_parallel_source_prefix(model, step, pool, routing, arguments, program.source)
        if launch_policy == "injected_launch":
            for trace in prefix.traces:
                trace.bind_launch_policy(launch_policy)
        bound = program.bind(prefix)
        if bound is not None:
            return bound
    namespace = (cache.namespace(model.model_type, pool.name, pool.ranks[0], arguments["model_execution_version"], cache_key=key)
        + ":ep2" if program is None else program.source.namespace)
    geometry_domain = (metadata_scope and pool.hardware_id in ("nvidia_h200", "nvidia_a800")
        and arguments["model_execution_version"] == "vllm_0_11_0" and model.num_experts == 128
        and model.num_hidden_layers == 48 and model.hidden_size == 2048 and model.moe_intermediate_size == 768
        and arguments["source_profile"] == "cuda_eager_inplace" and arguments["input_ids"] is None
        and routing.source_kind == "profile_observation" and routing.distribution == "model_profile_layer_mean_v1"
        and 2 <= shape.logical_tokens < shape.model_input_tokens)
    if geometry_domain:
        return _moe_parallel_dynamic_program(model, step, pool, routing, arguments,
            cache=cache, launch_policy=launch_policy, exact_key=key)
    source = trace_qwen3_moe_forward(model, step, pool, routing, source_namespace=namespace, **arguments)
    if launch_policy == "injected_launch":
        for trace in source.traces:
            trace.bind_launch_policy(launch_policy)
    prefix = _moe_parallel_source_prefix(model, step, pool, routing, arguments, source,
        binding_target=arguments["model_execution_version"] == "vllm_0_18_0", binding_tail_nodes={})
    if launch_policy == "injected_launch":
        for trace in prefix.traces:
            trace.bind_launch_policy(launch_policy)
    program = _moe_exact_source_bindings(source, prefix, model)
    if metadata_scope:
        program._moe_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    cache.store(key, program)
    result = program.bind(prefix)
    if result is None:
        raise RuntimeError("new MoE source program rejected its own actual source prefix")
    return result


def _remap_collective_capture_nodes(group):
    """Use joined collective IDs in every rank's captured source segments."""
    mapping = {local: identifier for identifier, submissions in group._submissions.items()
               for local in submissions.values()}
    for trace in group.traces:
        for capture in trace.source_captures:
            for field in ("node_ids", "model_node_ids", "eager_node_ids"):
                capture[field] = tuple(mapping[node] if node in mapping else node for node in capture[field])
            capture["source_segments"] = tuple(tuple(mapping[node] if node in mapping else node for node in segment)
                for segment in capture["source_segments"])


def _dense_source_metadata(model, step, *, input_ids, positions, slot_mapping,
                           cache_capacity_tokens, source_profile, hardware_id, mtp_source=False,
                           allow_unknown_input_ids=False):
    expected_profile = ("vllm_0_11_glm4_sync" if model.model_type == "glm4_moe"
        else "vllm_ascend_eager" if hardware_id == "ascend_910b" else "cuda_eager_inplace")
    if source_profile != expected_profile:
        raise NotImplementedError("symbolic dense forward requires the declared hardware's eager source profile")
    if not mtp_source and any(r.mtp_proposer_mode != "disabled" or r.mtp_proposed_tokens or r.mtp_proposer_tokens
            or r.mtp_verified_tokens or r.mtp_next_proposed_tokens for r in step.requests):
        raise NotImplementedError("initial symbolic Qwen3 source requires MTP-off")
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    rows = sum(q_lens)
    inputs = (("positions", positions), ("slot_mapping", slot_mapping))
    if input_ids is not None or not allow_unknown_input_ids:
        inputs = (("input_ids", input_ids), *inputs)
    for name, values in inputs:
        if not isinstance(values, tuple) or len(values) != rows or any(type(v) is not int for v in values):
            raise ValueError(f"symbolic {name} must provide one explicit integer per model row")
    if input_ids is not None and any(not 0 <= value < model.vocab_size for value in input_ids):
        raise ValueError("symbolic embedding ID outside the checkpoint vocabulary")
    if any(not 0 <= value < model.max_position_embeddings for value in positions):
        raise ValueError("symbolic position outside the checkpoint RoPE cache")
    # gpu_model_runner._prepare_inputs: ordinary-text positions are each
    # request's computed-token count plus its within-query offset.
    expected_positions = tuple(position for request in step.requests
        for position in range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens))
    if positions != expected_positions:
        raise ValueError("symbolic Qwen3 positions disagree with the declared ordinary-text context")
    if type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens):
        raise ValueError("symbolic logical cache capacity cannot hold active sequences")
    if len(set(slot_mapping)) != rows or any(not 0 <= slot < cache_capacity_tokens for slot in slot_mapping):
        raise ValueError("initial symbolic cache source requires unique valid writable slots")
    return q_lens, kv_lens, rows


def _dense_forward_target(trace, model, pool_name, hidden_states, logits_indices, indices, *,
        group, gather, model_execution_version, hardware_id, sample_logits, cast_logits, maca_copy_runtime_source):
    selected = SymbolicLogitsRowSelect(trace, "logits-boundary-gather", pool_name)(
        hidden_states, logits_indices, selected_indices=indices, model_execution_version=model_execution_version,
        hardware_id=hardware_id)
    if group is None:
        head = SymbolicLinear(trace, "lm-head", pool_name, model.hidden_size,
            model.vocab_size, kind=WorkloadKind.LM_HEAD)
        logits, _ = head(selected)
    else:
        processor = SymbolicTPLogitsProcessor(trace, "lm-head", pool_name,
            hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather, model_execution_version=model_execution_version)
        logits = processor(selected)
    if sample_logits or cast_logits:
        output = SymbolicGreedy(trace, "target-sampler", pool_name)(logits,
            model_execution_version=model_execution_version, sample_logits=sample_logits,
            maca_copy_runtime_source=maca_copy_runtime_source)
    return selected, logits, output if sample_logits or cast_logits else None


def _trace_dense_forward(model, step, *, input_ids, positions, slot_mapping,
                         cache_capacity_tokens, source_profile, rank, namespace, model_type,
                         trace=None, group=None, collective=None, gather=None, mlp_factories=None,
                         fused_embedding=True, rope_program=ROPE_PROGRAM_FUSED, hardware_id="nvidia_h200", ffn_factory=None,
                         graph_token_shape=None, graph_eager_kinds=(), model_execution_version=None, sample_logits=True,
                         cast_logits=False, retained_padding_token_ids=None, af_transport=None,
                         maca_copy_runtime_source=None, ascend_piecewise_source=False, fa3_full_aot_source=None):
    """Trace complete Qwen3 target-off/all-greedy source for one scheduled step.

    Positions/slots are explicit caller metadata, never generated token values.
    Ordinary eager TP1 or0.11/A800 dense TP1/TP2 IDs may remain unknown:
    only their INT32 shape port is consumed by the public logical program.
    The0.11 TP1 gather and TP2 masked program also cover captured-buffer tails.
    Persistent KV objects describe a declared logical pool, not a physical
    allocator's normal capacity. Profile assumptions do not bind real runtime.
    """
    model_execution_version = model_type.model_execution_version if model_execution_version is None else model_execution_version
    if type(ascend_piecewise_source) is not bool:
        raise TypeError("Ascend PIECEWISE source selection must be explicit boolean")
    if ascend_piecewise_source and (model_execution_version != "vllm_0_18_0" or hardware_id != "ascend_910b"
            or model_type is not Qwen3Model or model.model_type != "qwen3" or model.compute_dtype != "bf16"
            or group is not None or graph_token_shape is not None and graph_token_shape.runtime_mode != "PIECEWISE"):
        raise NotImplementedError("Ascend PIECEWISE source requires ordinary0.18 Qwen3 BF16 TP1")
    if maca_copy_runtime_source is not None and (
            hardware_id != "metax_c500" or model_execution_version != "vllm_0_24_0"
            or group is not None or model.num_experts or model_type not in (Qwen3Model, LlamaModel)):
        raise NotImplementedError("MACA Copy runtime source requires ordinary C5000.24 dense TP1")
    if type(sample_logits) is not bool or type(cast_logits) is not bool or sample_logits and cast_logits:
        raise TypeError("dense source output scope requires exclusive sampling/cast booleans")
    if cast_logits and model.compute_dtype != "bf16":
        raise NotImplementedError("cast-only sampler source requires BF16 input logits")
    masked_moe = (model_type is Qwen3Model and model.model_type == "qwen3_moe"
        and ((model_execution_version == "vllm_0_11_0" and hardware_id in ("nvidia_a800", "nvidia_h200"))
            or (model_execution_version == "vllm_0_24_0" and hardware_id == "metax_c500")
            or (model_execution_version == "vllm_0_18_0" and hardware_id == "ascend_910b"))
        and mlp_factories is not None and isinstance(group, ForwardCollectiveGroup)
        and len(group.traces) == 2 and not fused_embedding)
    if model_execution_version == "vllm_0_24_0" and (
            hardware_id != "metax_c500" or (model.num_experts and not masked_moe) or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16" or model_type not in (Qwen3Model, LlamaModel)):
        raise NotImplementedError("vLLM0.24 dense model source requires C500 homogeneous BF16 Qwen3 or Llama")
    masked_source = ((model_execution_version == "vllm_0_11_0" and hardware_id in ("nvidia_a800", "nvidia_h200"))
        or (model_execution_version == "vllm_0_24_0" and hardware_id == "metax_c500")
        or (model_execution_version == "vllm_0_18_0" and hardware_id == "ascend_910b"))
    masked_tp2 = (masked_source
        and not fused_embedding and group is not None and len(group.traces) == 2)
    # vLLM0.30 retains the captured input-ID tail. TP1 embedding owns the
    # whole vocabulary, so its public work depends on the physical row count.
    full_moe = (model_type is Qwen3Model and model.model_type == "qwen3_moe"
        and model.compute_dtype == model.weight_dtype == "bf16"
        and is_validated_qwen3_moe_ep2_model_scope(model) and mlp_factories is not None
        and len(mlp_factories) == model.num_hidden_layers and fa3_full_aot_source is not None
        and hardware_id == "nvidia_h200" and model_execution_version == "vllm_0_30_0")
    piecewise_moe = (model_type is Qwen3Model and model.model_type == "qwen3_moe"
        and model.compute_dtype == model.weight_dtype == "bf16"
        and is_validated_qwen3_moe_ep2_model_scope(model) and mlp_factories is not None
        and len(mlp_factories) == model.num_hidden_layers and hardware_id == "nvidia_h200"
        and model_execution_version == "vllm_0_30_0" and graph_token_shape is not None
        and graph_token_shape.runtime_mode == "PIECEWISE"
        and set(graph_eager_kinds) == {"flash_attn", "kv_update"})
    plain_tp1_capture = ((masked_source or (model_execution_version == "vllm_0_30_0"
        and hardware_id == "nvidia_h200"))
        and group is None and model_type in (Qwen3Model, LlamaModel)
        and (not model.num_experts or full_moe or piecewise_moe))
    static_moe = (model_type is Qwen3Model and model.model_type == "qwen3_moe"
        and mlp_factories is not None and hardware_id == "nvidia_h200")
    static_tp2 = (model_type in (Qwen3Model, LlamaModel) and (not model.num_experts or static_moe)
        and hardware_id == "nvidia_h200" and isinstance(group, ForwardCollectiveGroup)
        and len(group.traces) == 2 and input_ids is not None)
    if not sample_logits and (model_type not in (Qwen3Model, LlamaModel)
            or (not static_moe and (model.num_experts or mlp_factories is not None))
            or hardware_id not in ("nvidia_a800", "nvidia_h200") or source_profile != "cuda_eager_inplace"
            or model_execution_version != "vllm_0_30_0" or (group is not None and not static_tp2) or graph_token_shape is not None
            or ffn_factory is not None):
        raise NotImplementedError("static LM-head-only source requires0.30 CUDA dense TP1/H200 dense TP2 or H200 Qwen3-MoE TP1/validated TP=EP2 eager without MTP/graph")
    q_lens, kv_lens, rows = _dense_source_metadata(model, step, input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, cache_capacity_tokens=cache_capacity_tokens,
        source_profile=source_profile, hardware_id=hardware_id,
        allow_unknown_input_ids=(model_type in (Qwen3Model, LlamaModel)
            and (not model.num_experts or masked_moe or (static_moe and model_execution_version == "vllm_0_30_0" and group is None))
            and (group is None or (not fused_embedding and len(group.traces) == 2
                and ((model_execution_version == "vllm_0_11_0" and hardware_id in ("nvidia_a800", "nvidia_h200"))
                    or (model_execution_version in ("vllm_0_24_0", "vllm_0_30_0") and hardware_id == "metax_c500")
                    or (model_execution_version == "vllm_0_18_0" and hardware_id == "ascend_910b"))))
            and (graph_token_shape is None or masked_tp2 or plain_tp1_capture)
            and (mlp_factories is None or static_moe or masked_moe) and ffn_factory is None))
    trace = ForwardTrace(rank=rank, namespace=namespace) if trace is None else trace
    # 0.11 forward_native runs TP1 gather or TP2 mask/gather/fill over ALL
    # captured rows. Logical payload is independent of IDs, including the retained
    # input-buffer tail. Validate scheduled IDs above, but do not invent tail
    # values or pass a logical-length tuple as the physical embedding input.
    embedding_ids = (None if masked_tp2 and graph_token_shape is not None
        and graph_token_shape.padding_tokens else input_ids)
    if retained_padding_token_ids is not None:
        if (graph_token_shape is None or not isinstance(retained_padding_token_ids, tuple)
                or input_ids is None or model_execution_version != "vllm_0_30_0"
                or not fused_embedding or group is None or len(group.traces) != 2
                or hardware_id not in ("nvidia_a800", "nvidia_h200")
                or len(retained_padding_token_ids) != graph_token_shape.padding_tokens):
            raise ValueError("retained padding IDs require a matching fused CUDA TP2 capture and scheduled IDs")
        embedding_ids = input_ids + retained_padding_token_ids
    network = model_type(trace, model, step.pool_name, q_lens, kv_lens, cache_capacity_tokens, slot_mapping,
        group=group, collective=collective, token_ids=embedding_ids, mlp_factories=mlp_factories,
        fused_embedding=fused_embedding, rope_program=rope_program, hardware_id=hardware_id, ffn_factory=ffn_factory,
        model_execution_version=model_execution_version,
        **({"maca_copy_runtime_source": maca_copy_runtime_source} if maca_copy_runtime_source is not None else {}))
    if ascend_piecewise_source:
        # The configured PIECEWISE source also governs overflow NONE calls:
        # using_paged_attention is false unless configured FULL_DECODE_ONLY.
        for layer in network.layers:
            layer.self_attn.attn.graph_attention_source = "ascend_0_18_piecewise_fia"
    aot_frame = None
    if fa3_full_aot_source is not None:
        state,max_capture = fa3_full_aot_source
        if (hardware_id != "nvidia_h200" or model_execution_version != "vllm_0_30_0"
                or graph_token_shape is None or graph_token_shape.runtime_mode != "FULL"
                or (group is not None and (type(group) is not ForwardCollectiveGroup or len(group.traces) != 2))
                or model.compute_dtype not in ("bf16", "fp16") or model.compute_dtype != model.weight_dtype
                or (((group is not None and model.model_type not in ("qwen3", "llama"))
                    or (model.compute_dtype == "fp16" and model.model_type != "qwen3")
                    or model.num_experts) and not full_moe)
                or model.head_dim != 128):
            raise ValueError("FA3 AOT source requires its exact ordinary H200 FULL caller")
        attention = network.layers[0].self_attn.attn
        aot_frame = _trace_full_fa3_aot(trace, model, step.pool_name, q_lens, kv_lens,
            graph_token_shape, state, max_capture, heads=attention.heads, kv_heads=attention.kv_heads)
        for layer in network.layers:
            layer.self_attn.attn.fa3_full_aot = aot_frame
    # gpu_model_runner allocates INT32 IDs and passes its slice unchanged
    # through _preprocess/_model_forward for text-only models.
    physical_rows = rows if graph_token_shape is None else graph_token_shape.model_input_tokens
    ids = trace.tensor("input_ids", (physical_rows,), "int32")
    pos = trace.tensor("positions", (physical_rows,), "int64")
    if graph_token_shape is not None:
        if model_type not in (Qwen3Model, LlamaModel):
            raise NotImplementedError("source capture requires the ordinary dense Qwen/Llama forward")
        if hardware_id == "ascend_910b" and model_execution_version != "vllm_0_18_0":
            raise NotImplementedError("Ascend captured model requires its explicit0.18 source")
        slots = trace.tensor("capture.slot-mapping", (physical_rows,), "int64")
        for layer in network.layers:
            layer.self_attn.attn.graph_token_shape = graph_token_shape
            layer.self_attn.attn.slot_tensor = slots
            if model_execution_version == "vllm_0_18_0":
                if ascend_piecewise_source:
                    if set(graph_eager_kinds) != {"flash_attn", "kv_update"}:
                        raise ValueError("Ascend PIECEWISE source requires eager attention and KV update")
                elif graph_token_shape.runtime_mode != "FULL" or hardware_id != "ascend_910b":
                    raise NotImplementedError("Ascend0.18 captured attention requires configured FULL")
                else:
                    layer.self_attn.attn.graph_attention_source = "ascend_0_18_full_fia"
        if af_transport is None:
            capture_inputs={"input_ids": ids, "positions": pos, "slot_mapping": slots}
            aot_inputs=({"fa3_cu_query":aot_frame[2],"fa3_used_key":aot_frame[3],
                "fa3_scheduler_metadata":aot_frame[0]} if aot_frame is not None else None)
            trace.begin_capture(key=f"{source_profile}:{graph_token_shape.runtime_mode}:{physical_rows}", logical_tokens=rows,
                model_input_tokens=physical_rows, inputs=capture_inputs, extra_inputs=aot_inputs,
                runtime_mode=graph_token_shape.runtime_mode, eager_kinds=graph_eager_kinds)
        else:
            af_transport.attention_capture_inputs[trace.rank] = {"positions": pos, "slot_mapping": slots}
            af_transport.begin_local_capture(trace, {"input_ids": ids}, boundary="entry")
    indices = tuple(sum(q_lens[:i + 1]) - 1 for i in range(len(q_lens)))
    logits_indices = trace.tensor("logits_indices", (len(indices),), "int32")
    hidden_states = network(ids, pos)
    if graph_token_shape is not None:
        if af_transport is None:
            trace.end_capture(hidden_states)
        else:
            af_transport.end_local_capture(trace, {"hidden_states": hidden_states})
    selected, logits, output = _dense_forward_target(trace, model, step.pool_name, hidden_states, logits_indices, indices,
        group=group, gather=gather, model_execution_version=model_execution_version, hardware_id=hardware_id,
        sample_logits=sample_logits, cast_logits=cast_logits, maca_copy_runtime_source=maca_copy_runtime_source)
    return trace


def ascend018_segment_source_scope(trace, *, model, source_profile, model_execution_version,
        tensor_parallel_size, source_lifetime_policy, source_compilation_options):
    """Bind only the declared 0.18 dense TP1 attention-partition source policy.

    The optional options tuple carries upstream compile fields as immutable
    key/value pairs, not a new production configuration/schema. Every field
    is mandatory for this explicit policy; defaults never select a backend.
    """
    if (source_lifetime_policy != "vllm_0_18_npugraph_ex_piecewise"
            or model_execution_version != "vllm_0_18_0"
            or source_profile != "vllm_ascend_0_18_no_metadata_head_only"
            or model.model_type != "qwen3" or model.compute_dtype != "bf16"
            or tensor_parallel_size != 1):
        raise NotImplementedError("segment lifetime policy requires exact Qwen3 BF16 TP1 Ascend0.18 scope")
    required = {"mode": "VLLM_COMPILE", "backend": "vllm_ascend.compilation.compiler_interface.AscendCompiler",
        "cudagraph_mode": "PIECEWISE", "cudagraph_copy_inputs": False,
        "use_inductor_graph_partition": False, "enable_npugraph_ex": True}
    if (not isinstance(source_compilation_options, tuple)
            or any(not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str)
                for pair in source_compilation_options)):
        raise ValueError("source policy requires explicit immutable upstream compilation options")
    options = dict(source_compilation_options)
    if len(options) != len(source_compilation_options) or set(options) != {*required, "splitting_ops"}:
        raise ValueError("source compilation option fields must be complete and unique")
    for name, expected in required.items():
        if type(options[name]) is not type(expected) or options[name] != expected:
            raise ValueError("source compilation option is outside the declared segment policy: " + name)
    splitting = options["splitting_ops"]
    if (not isinstance(splitting, tuple) or not splitting
            or any(not isinstance(name, str) or not name for name in splitting)
            or len(set(splitting)) != len(splitting)
            or "vllm::unified_attention_with_output" not in splitting):
        raise ValueError("source compilation options must explicitly split the attention-with-output op")
    # v0.18 CompilationConfig attention/cache split names. Other members do
    # not occur in this exact Qwen3 source; arbitrary new cuts are rejected.
    allowed_splits = {"vllm::unified_attention", "vllm::unified_attention_with_output",
        "vllm::unified_mla_attention", "vllm::unified_mla_attention_with_output",
        "vllm::mamba_mixer2", "vllm::mamba_mixer", "vllm::short_conv", "vllm::linear_attention",
        "vllm::plamo2_mamba_mixer", "vllm::gdn_attention_core", "vllm::olmo_hybrid_gdn_full_forward",
        "vllm::kda_attention", "vllm::sparse_attn_indexer", "vllm::rocm_aiter_sparse_attn_indexer",
        "vllm::unified_kv_cache_update", "vllm::unified_mla_kv_cache_update", "vllm::mla_forward"}
    if not set(splitting) <= allowed_splits:
        raise ValueError("unrepresented split operators cannot borrow the attention partition")
    splits = tuple(node.node_id for node in trace.graph.nodes if "source_operation" in node.attributes
        and node.attributes["source_operation"] == "flash_attention_profile_fill")
    if not splits:
        raise ValueError("declared source graph contains no bound attention split operations")
    return {"split_node_ids": splits, "escaped_allocation_ids": (),
        "start_event": trace.checkpoints["worker.profile.begin"], "end_event": trace.checkpoints["worker.profile.end"],
        "segment_start_event": trace.checkpoints["model.source.begin"],
        "segment_end_event": trace.checkpoints["model.source.end"]}


def trace_qwen3_startup_profile(model, *, tokens, max_requests, source_profile, num_compute_units=None,
        tensor_parallel_size=1, collective_configs=(), model_execution_version="vllm_0_30_0",
        record_model_scope=False):
    """Version-bound dense startup caller storage for Qwen3 and Llama.

    Conditions select FA's no-metadata output and PyTorch-sort, cold-cache
    native Triton sampling, or the explicit0.11/FlashInfer0.4.1 top-k-first
    storage path. Source versions do not identify a historical library version
    or bind a compiled configuration merely from its backend name.
    Preallocated input buffers and RoPE cache are persistent, not transient
    headroom. Library workspace and other worker state are not inferred.
    The explicit profiles use an unquantized model-dtype head, without a
    head_dtype override. CUDA Sampler.forward casts its logits to FP32.
    Ascend0.18's dummy sampler only gathers hidden rows and calls the head,
    retaining its BF16 logits and never invoking a sampling algorithm.
    FP16 Qwen3 is bound only for the PyTorch-sort branch.
    """
    from .traced_sampling import (TritonSamplingStorageCache, VLLM011WorkerStorage, VLLM024WorkerStorage, VLLM030WorkerStorage,
        worker_profile_storage_forward, ascend018_profile_storage_forward)
    from src.serving.deployment.deployment_config import CollectiveConfig
    from .model_config import is_dense_parallel_model_scope
    if model.compute_dtype != "bf16" and not (
            model.compute_dtype == "fp16" and model.weight_dtype == "fp16"
            and model.model_type == "qwen3"
            and source_profile == "cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise"):
        raise NotImplementedError("FP16 startup sampler storage source requires homogeneous Qwen3 and explicit PyTorch-sort")
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
        raise NotImplementedError("dense startup requires its version-bound0.11,0.18,0.24 or0.30 source")
    if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
        raise ValueError("dense startup requires a positive integer tensor parallel size")
    if model_execution_version != "vllm_0_30_0" and tensor_parallel_size not in (1, 2):
        raise NotImplementedError("legacy dense startup source requires TP1/TP2")
    if not is_dense_parallel_model_scope(model, tensor_parallel_size, 1):
        raise NotImplementedError("dense startup requires complete unquantized Qwen3/Llama TP geometry")
    if not isinstance(collective_configs, tuple):
        raise TypeError("startup collectives require canonical configuration tuple")
    if tensor_parallel_size == 1:
        if collective_configs:
            raise ValueError("TP1 startup does not consume collective configuration")
        collective = gather = None
    else:
        if (len(collective_configs) != 2 or any(not isinstance(item, CollectiveConfig) for item in collective_configs)
                or tuple(item.operation for item in collective_configs) != ("all_reduce", "all_gather")):
            raise ValueError("TP startup requires explicit all-reduce/all-gather configuration")
        collective, gather = collective_configs
    ascend018 = model_execution_version == "vllm_0_18_0"
    metax024 = model_execution_version == "vllm_0_24_0"
    if metax024 and (source_profile != "vllm_metax_0_24_no_metadata_native_triton_fp32_noise_cold_cache"
            or tensor_parallel_size != 1):
        raise NotImplementedError("MetaX0.24 startup requires its explicit BF16 TP1 native-sampler source")
    if ascend018 and source_profile != "vllm_ascend_0_18_no_metadata_head_only":
        raise NotImplementedError("Ascend0.18 startup requires its head-only source profile")
    if source_profile == "vllm_ascend_0_18_no_metadata_head_only":
        if not ascend018 or tensor_parallel_size != 1 or num_compute_units is not None:
            raise NotImplementedError("Ascend head-only startup requires BF16 TP1 and explicit0.18 source")
        sampler_profile = None
    elif source_profile == "vllm_metax_0_24_no_metadata_native_triton_fp32_noise_cold_cache":
        if not metax024 or type(num_compute_units) is not int or num_compute_units <= 0:
            raise ValueError("MetaX0.24 native sampler requires its version and positive compute-unit count")
        sampler_profile = "vllm_0_24_native_triton_or_sort_mixed_no_logprobs_fp32_noise"
    elif source_profile == "cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise":
        if num_compute_units is not None:
            raise ValueError("PyTorch-sort source profile does not consume an SM count")
        sampler_profile = ("vllm_0_11_pytorch_sort_mixed_no_logprobs_fp32_noise"
            if model_execution_version == "vllm_0_11_0" else "vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise")
    elif source_profile == "cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first":
        if model_execution_version != "vllm_0_11_0":
            raise NotImplementedError("FlashInfer0.4.1 startup storage requires the explicit0.11 worker source")
        if num_compute_units is not None:
            raise ValueError("FlashInfer top-k-first storage does not consume an SM count")
        sampler_profile = "vllm_0_11_flashinfer_0_4_1_top_k_first_mixed_no_logprobs"
    elif source_profile == "cuda_eager_inplace_fa_no_metadata_native_triton_fp32_noise_cold_cache":
        if model_execution_version != "vllm_0_30_0":
            raise NotImplementedError("dense0.11 startup requires the explicit PyTorch-sort sampler source")
        if type(num_compute_units) is not int or num_compute_units <= 0:
            raise ValueError("Triton startup source profile requires an explicit positive SM count")
        sampler_profile = "vllm_0_30_triton_mixed_no_logprobs_fp32_noise"
    else:
        raise NotImplementedError("startup profile requires explicit supported source/backend choices")
    if model.model_type == "qwen3":
        _validate_qwen3_source(model)
        model_type = Qwen3Model
    elif model.model_type == "llama":
        _validate_llama_source(model)
        model_type = LlamaModel
    else:
        raise NotImplementedError("dense startup source requires supported Qwen3 or BF16 Llama")
    if any(type(value) is not int or value <= 0 for value in (tokens, max_requests)):
        raise ValueError("startup profile token and request capacities must be positive integers")
    if tokens > model.max_position_embeddings:
        raise NotImplementedError("initial startup source requires token capacity within the RoPE cache")
    rows = min(tokens, max_requests)
    lengths = (tokens // rows,) * (rows - 1) + (tokens // rows + tokens % rows,)
    traces = tuple(ForwardTrace(rank=rank, namespace=None if tensor_parallel_size == 1 else f"startup:rank-{rank}")
        for rank in range(tensor_parallel_size))
    group = None if tensor_parallel_size == 1 else ForwardCollectiveGroup(traces, namespace="startup")
    # Reuse the existing backend-specific norm/RoPE source modules. These
    # caller scopes do not claim compiler buffer-reuse or library workspace.
    def rank_forward(trace):
        hardware_id = "ascend_910b" if ascend018 else "metax_c500" if metax024 else "nvidia_h200"
        network = model_type(trace, model, "main", lengths, lengths, tokens, None,
            profile_no_metadata=True, rope_program=vllm_rope_program(hardware_id, model.rope_scaling_type),
            hardware_id=hardware_id, group=group, collective=collective,
            model_execution_version=model_execution_version,
            fused_embedding=not (model_execution_version == "vllm_0_11_0" and group is not None),
            # CpuGpuBuffer initializes input_ids to zero. This dense DP1
            # profile has no MoE-driven randomize_inputs branch; the tuple
            # records those source metadata values without numerical work.
            token_ids=None if group is None else (0,) * tokens)
        if ascend018:
            # Only the consumed source ports are needed to measure transient
            # headroom. This does not infer the full NPU worker residency.
            input_ids = trace.tensor("profile.input_ids", (tokens,), "int32", persistent=True)
            positions = trace.tensor("profile.positions", (tokens,), "int64", persistent=True)
        else:
            storage_type = (VLLM011WorkerStorage if model_execution_version == "vllm_0_11_0" else
                VLLM024WorkerStorage if metax024 else VLLM030WorkerStorage)
            worker_storage = storage_type(trace, tokens=tokens, max_requests=max_requests,
                inputs_embeds_size=model.hidden_size, dtype=model.compute_dtype)
            input_ids, positions = worker_storage.input_ids, worker_storage.positions
        if group is None:
            head = SymbolicLinear(trace, "profile.lm-head", "main", model.hidden_size,
                model.vocab_size, kind=WorkloadKind.LM_HEAD)
            compute_logits = lambda x: head(x)[0]
        else:
            # Platform.use_all_gather is true for CUDA; every TP rank's
            # dummy sampler consumes full-vocabulary logits after gathering.
            compute_logits = SymbolicTPLogitsProcessor(trace, "profile.lm-head", "main",
                hidden_size=model.hidden_size, vocab_size=model.vocab_size, group=group, config=gather,
                model_execution_version=model_execution_version)
        cache = None if num_compute_units is None else TritonSamplingStorageCache(trace,
            num_compute_units=num_compute_units, model_execution_version="vllm_0_24_0" if metax024 else "vllm_0_30_0")
        trace.checkpoint("worker.profile.begin")
        if type(record_model_scope) is not bool:
            raise TypeError("model scope recording must be explicit boolean")
        model_forward = network
        if record_model_scope:
            def model_forward(*args, **kwargs):
                trace.checkpoint("model.source.begin")
                result = network(*args, **kwargs)
                trace.checkpoint("model.source.end")
                return result
        if ascend018:
            ascend018_profile_storage_forward(trace, model_forward, compute_logits,
                input_ids=input_ids, positions=positions, query_lengths=lengths)
        else:
            worker_profile_storage_forward(trace, network, compute_logits,
                input_ids=input_ids, positions=positions, query_lengths=lengths,
                sampler_profile=sampler_profile, triton_cache=cache, model_execution_version=model_execution_version)
        trace.checkpoint("worker.profile.end")

    for trace in traces:
        rank_forward(trace)
    return traces[0] if group is None else group


def _ordinary_model_only_source_scope(model, deployment, runtime_inputs):
    """Existing static LM-head scope, shared by binding and graph execution."""
    if (runtime_inputs.model_execution_version != "vllm_0_30_0"
            or runtime_inputs.symbolic_forward_source not in (None, "cuda_eager_inplace")
            or runtime_inputs.target_sampling_source is not None
            or deployment.mode != "colocated" or len(deployment.worker_pools) != 1):
        return False
    pool, = deployment.worker_pools.values()
    parallel = pool.parallel_plan.config
    moe = (model.model_type == "qwen3_moe" and pool.hardware_id == "nvidia_h200"
        and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation"))
    dense = (model.model_type in ("qwen3", "llama") and not model.num_experts
        and runtime_inputs.expert_routing.source_kind == "not_applicable")
    ep2 = (moe and parallel.tensor_parallel_size == parallel.expert_parallel_size == 2
        and parallel.enable_expert_parallel and is_validated_qwen3_moe_ep2_model_scope(model))
    tp2 = (pool.hardware_id == "nvidia_h200" and parallel.tensor_parallel_size == 2
        and (dense or ep2 or moe and not parallel.enable_expert_parallel))
    return ((dense or moe) and model.compute_dtype == model.weight_dtype
        and model.compute_dtype in ("bf16", "fp16")
        and (model.compute_dtype != "fp16" or dense and model.model_type == "qwen3")
        and pool.hardware_id in ("nvidia_a800", "nvidia_h200")
        and len(pool.ranks) == (2 if tp2 else 1) and parallel.is_rank_local
        and (not parallel.enable_expert_parallel or ep2) and parallel.tensor_parallel_size == (2 if tp2 else 1)
        and all(value == 1 for value in (parallel.pipeline_parallel_size,
            parallel.sequence_parallel_size, parallel.context_parallel_size, parallel.data_parallel_size))
        and not runtime_inputs.graph_execution.enabled and runtime_inputs.model_source_inputs is None
        and runtime_inputs.mtp_draft_sampling_source is None and runtime_inputs.mtp_proposer_execution_source is None)


def _ordinary_mla_pd_source_scope(model, deployment):
    """Existing H200 MLA PD pools whose per-pool source is already traced.

    This concerns model execution only. Request handoff still belongs to
    the existing PD/KV owner and does not become a native-provider receipt.
    """
    from src.serving.deployment.deployment_config import worker_role_phase
    if (deployment.mode != "pd_disagg" or model.compute_dtype != "bf16"
            or model.attention_layout not in ("mla", "dsa_mla")
            or not ((model.model_type == "glm_moe_dsa" and model.weight_dtype in ("bf16", "fp8"))
                or (_is_kimi_k2_text_source(model)))):
        return False
    pools = tuple(deployment.worker_pools.values())
    if {worker_role_phase(pool.role) for pool in pools} != {"prefill", "decode"}:
        return False
    from src.serving.deployment.validated_scope import (is_glm_distributed_source_scope,
        is_glm_bf16_tensor_source_scope, is_glm_bf16_pipeline_source_scope)
    for pool in pools:
        if (is_glm_distributed_source_scope(pool, model) or is_glm_bf16_tensor_source_scope(pool, model)
                or is_glm_bf16_pipeline_source_scope(pool, model)):
            continue
        config = pool.parallel_plan.config
        if (pool.hardware_id != "nvidia_h200" or len(pool.ranks) != 8
                or config.tensor_parallel_size != 8 or not config.is_rank_local
                or (model.model_type == "kimi_k2" and not config.enable_expert_parallel)
                or any(value != 1 for value in (config.pipeline_parallel_size,
                    config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
                or (config.enable_expert_parallel and config.expert_parallel_size != 8)):
            return False
    return True


def _ordinary_af_capture_scope(model, deployment, model_execution_version):
    pools = tuple(deployment.worker_pools.values())
    return (model_execution_version == "vllm_0_11_0"
        and model.model_type in ("qwen3", "llama") and not model.num_experts
        and model.compute_dtype == model.weight_dtype == "bf16"
        and deployment.mode in ("af_disagg", "pd_af_disagg")
        and all(pool.hardware_id == "nvidia_a800" and pool.parallel_plan.config.tensor_parallel_size in (1, 2)
            and pool.parallel_plan.config.pipeline_parallel_size == 1
            and pool.parallel_plan.config.is_rank_local and not pool.parallel_plan.config.enable_expert_parallel
            and len(pool.ranks) == pool.parallel_plan.config.tensor_parallel_size for pool in pools)
        and (deployment.mode != "pd_af_disagg" or all(pool.parallel_plan.config.tensor_parallel_size == 1 for pool in pools)))


def _validate_maca_copy_runtime_scope(model, deployment, runtime_inputs):
    if runtime_inputs.maca_copy_runtime_source is None:
        return
    if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
            or model.model_type not in ("qwen3", "llama") or model.num_experts):
        raise NotImplementedError("MACA Copy runtime source requires ordinary C5000.24 dense TP1")
    pool, = deployment.worker_pools.values()
    config = pool.parallel_plan.config
    if (pool.hardware_id != "metax_c500" or any(value != 1 for value in (
            config.tensor_parallel_size, config.pipeline_parallel_size,
            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
            or config.enable_expert_parallel):
        raise NotImplementedError("MACA Copy runtime source requires ordinary C5000.24 dense TP1")


def bind_ordinary_forward_source(model, deployment, runtime_inputs):
    """Resolve complete ordinary API facts without retrying a legacy graph."""
    _validate_maca_copy_runtime_scope(model, deployment, runtime_inputs)
    bound = _bind_ordinary_forward_source(model, deployment, runtime_inputs)
    if (bound.symbolic_forward_source is None
            and (bound.symbolic_source_max_model_len is not None
                or bound.symbolic_source_max_batched_tokens is not None)):
        raise NotImplementedError("explicit MLA constructor inputs have no supported common forward; legacy fallback is forbidden")
    if (bound.symbolic_forward_source is None
            and bound.model_execution_version in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0")):
        raise NotImplementedError("declared vLLM source API inputs have no supported common forward; legacy fallback is forbidden")
    return bound


def _bind_ordinary_forward_source(model, deployment, runtime_inputs):
    """Bind an existing ordinary source from complete explicit API settings.

    This selects no sampler, token value, provider or constructor state. An
    already selected source is immutable; unsupported source inputs still
    fail in the shared forward rather than retrying the old builder.
    """
    from dataclasses import replace
    from src.serving.deployment.validated_scope import (is_dense_parallel_source_scope,
        is_qwen3_moe_rank_local_source_scope)
    if (runtime_inputs.symbolic_forward_source is None
            and model.model_id == "moonshotai/Kimi-K3" and model.compute_dtype == "bf16"
            and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.attention_kv_layout == "mla" and runtime_inputs.attention_kv_distribution == "uniform"
            and runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation", "replay_observation")
            and runtime_inputs.symbolic_source_max_model_len is not None
            and runtime_inputs.symbolic_source_max_batched_tokens is not None
            and not runtime_inputs.graph_execution.enabled and runtime_inputs.model_source_inputs is None
            and runtime_inputs.mtp_draft_sampling_source is None and runtime_inputs.mtp_proposer_execution_source is None
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1):
        pool, = deployment.worker_pools.values()
        parallel = pool.parallel_plan.config
        if (pool.hardware_id == "nvidia_h200" and pool.role == "colocated"
                and parallel.tensor_parallel_size == 1 and 1 <= parallel.pipeline_parallel_size <= model.num_hidden_layers
                and len(pool.ranks) == parallel.pipeline_parallel_size
                and parallel.is_rank_local and not parallel.enable_expert_parallel):
            return replace(runtime_inputs, symbolic_forward_source="vllm_0_30_k3_mxfp4_sync")
    if (runtime_inputs.symbolic_forward_source is None
            and model.model_type in ("glm_moe_dsa", "kimi_k2") and model.compute_dtype == "bf16"
            and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.attention_kv_layout == "mla" and runtime_inputs.attention_kv_distribution == "uniform"
            and runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            and runtime_inputs.symbolic_source_max_model_len is not None
            and runtime_inputs.symbolic_source_max_batched_tokens is not None
            and (not runtime_inputs.graph_execution.enabled or (deployment.mode == "colocated"
                and len(deployment.worker_pools) == 1 and all(_kimi_single_rank_source_scope(model, pool)
                    for pool in deployment.worker_pools.values())))
            and runtime_inputs.model_source_inputs is None
            and runtime_inputs.mtp_draft_sampling_source is None and runtime_inputs.mtp_proposer_execution_source is None
            and ((deployment.mode == "colocated" and len(deployment.worker_pools) == 1)
                or _ordinary_mla_pd_source_scope(model, deployment))):
        a800_glm_pipeline = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and model.model_type == "glm_moe_dsa" and model.weight_dtype == "bf16"
            and all(pool.hardware_id == "nvidia_a800"
                and pool.parallel_plan.config.tensor_parallel_size == 8
                and pool.parallel_plan.config.pipeline_parallel_size == 4
                and pool.parallel_plan.config.enable_expert_parallel
                and pool.parallel_plan.config.expert_parallel_size == 8
                for pool in deployment.worker_pools.values()))
        a800_kimi_pipeline = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and all(_kimi_pipeline_source_scope(model, pool) for pool in deployment.worker_pools.values()))
        from src.serving.deployment.validated_scope import (is_glm_distributed_source_scope,
            is_glm_bf16_tensor_source_scope, is_glm_bf16_pipeline_source_scope)
        glm_bf16_source = all(is_glm_bf16_tensor_source_scope(pool, model) or is_glm_distributed_source_scope(pool, model)
            or is_glm_bf16_pipeline_source_scope(pool, model)
            for pool in deployment.worker_pools.values())
        if (a800_glm_pipeline or a800_kimi_pipeline
                or glm_bf16_source
                or all(_kimi_single_rank_source_scope(model, pool) for pool in deployment.worker_pools.values())
                or all(pool.hardware_id == "nvidia_h200" for pool in deployment.worker_pools.values())):
            if model.model_type == "glm_moe_dsa" and model.weight_dtype in ("bf16", "fp8"):
                profile = ("vllm_0_30_v32_bf16_sync_explicit_allreduce" if model.weight_dtype == "bf16"
                    else "vllm_0_30_v32_fp8_sync_explicit_allreduce")
                return replace(runtime_inputs, symbolic_forward_source=profile)
            if (_is_kimi_k2_text_source(model)):
                return replace(runtime_inputs, symbolic_forward_source="vllm_0_30_k2_6_int4_sync_explicit_allreduce")
    if (runtime_inputs.symbolic_forward_source is None and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.attention_kv_layout == "paged_varlen"
            and runtime_inputs.attention_kv_distribution == "uniform"
            and _ordinary_model_only_source_scope(model, deployment, runtime_inputs)):
        # Missing sampling facts retain the static output boundary. The
        # complete Serving entry still rejects a model-only graph.
        return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
    if (runtime_inputs.symbolic_forward_source is None and model.model_type == "deepseek_v4"
            and model.compute_dtype == "bf16" and model.weight_dtype == "fp8"
            and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation")
            and runtime_inputs.attention_kv_layout == "mla" and runtime_inputs.attention_kv_distribution == "uniform"
            and runtime_inputs.v4_eager_metadata_source == "vllm_0_30_eager_dict_all_valid"
            and runtime_inputs.v4_routing_source in ("vllm_0_30_modular_cuda_top6_pure_text", "vllm_0_30_triton_top6_pure_text")
            and runtime_inputs.v4_source_max_model_len is not None
            and runtime_inputs.v4_source_max_batched_tokens is not None
            and runtime_inputs.v4_source_use_deep_gemm is not None
            and runtime_inputs.embedding_token_ids is not None
            and not runtime_inputs.graph_execution.enabled and runtime_inputs.model_source_inputs is None
            and runtime_inputs.mtp_draft_sampling_source is None
            and runtime_inputs.mtp_proposer_execution_source is None
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1):
        pool, = deployment.worker_pools.values()
        parallel = pool.parallel_plan.config
        from src.serving.deployment.validated_scope import is_deepseek_parallel_source_scope
        if (is_deepseek_parallel_source_scope(pool, model, model_execution_version=runtime_inputs.model_execution_version,
                symbolic_forward_source="vllm_0_30_v4_sync_fp8_ds_mla", graph_enabled=runtime_inputs.graph_execution.enabled)
                and (parallel.pipeline_parallel_size == 1 or runtime_inputs.v4_source_use_deep_gemm is False)):
            return replace(runtime_inputs, symbolic_forward_source="vllm_0_30_v4_sync_fp8_ds_mla")
    dense = (model.model_type in ("qwen3", "llama") and not model.num_experts
        and runtime_inputs.expert_routing.source_kind == "not_applicable")
    moe = (model.model_type == "qwen3_moe" and model.num_experts in (2, 128)
        and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation"))
    ordinary_versions = {"vllm_0_11_0": "vllm_0_11", "vllm_0_18_0": "vllm_0_18", "vllm_0_24_0": "vllm_0_24", "vllm_0_30_0": "vllm_0_30"}
    if runtime_inputs.model_execution_version not in ordinary_versions:
        return runtime_inputs
    sampler_prefix = ordinary_versions[runtime_inputs.model_execution_version]
    source011 = runtime_inputs.model_execution_version == "vllm_0_11_0"
    source024 = runtime_inputs.model_execution_version == "vllm_0_24_0"
    source018 = runtime_inputs.model_execution_version == "vllm_0_18_0"
    ordinary_moe_capture = ((source011 or source024 or source018) and moe and deployment.mode == "colocated"
        and len(deployment.worker_pools) == 1
        and all(_versioned_qwen3_moe_ep2_source(model, item, runtime_inputs.model_execution_version) for item in deployment.worker_pools.values()))
    if (runtime_inputs.symbolic_forward_source is not None
            or runtime_inputs.serving_framework != "vllm"
            or runtime_inputs.target_sampling_source != sampler_prefix + "_plain_greedy_" + model.compute_dtype
            or not (dense or moe) or (model.compute_dtype == "fp16" and model.model_type != "qwen3")
            or model.compute_dtype not in ("bf16", "fp16") or model.weight_dtype != model.compute_dtype
            or runtime_inputs.attention_kv_layout != "paged_varlen"
            or runtime_inputs.attention_kv_distribution != "uniform"
            or (runtime_inputs.graph_execution.enabled and not (_ordinary_af_capture_scope(model, deployment, runtime_inputs.model_execution_version)
                or (source011 or source024 or source018) and (dense or ordinary_moe_capture)
                and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                and all(pool.hardware_id in (("metax_c500",) if source024 else ("ascend_910b",) if source018 else ("nvidia_a800", "nvidia_h200"))
                    for pool in deployment.worker_pools.values())))
            or runtime_inputs.model_source_inputs is not None
            or runtime_inputs.mtp_draft_sampling_source is not None
            or runtime_inputs.mtp_proposer_execution_source is not None
            or runtime_inputs.v4_eager_metadata_source is not None):
        return runtime_inputs
    if (source024 or source018) and (not (dense or ordinary_moe_capture) or deployment.mode != "colocated"
            or len(deployment.worker_pools) != 1
            or any(pool.hardware_id != ("metax_c500" if source024 else "ascend_910b") for pool in deployment.worker_pools.values())):
        return runtime_inputs
    if source011 and moe and deployment.mode in ("pd_disagg", "af_disagg"):
        # These existing CLI modes already execute this version-bound MoE
        # forward. Bind the same complete facts for programmatic callers;
        # the forward still validates roles, routes and per-step inputs.
        pools = tuple(deployment.worker_pools.values())
        if (model.compute_dtype == "bf16" and len(pools) == 2
                and all(item.hardware_id == "nvidia_a800" and len(item.ranks) == 1
                    and item.parallel_plan.config.tensor_parallel_size == 1
                    and item.parallel_plan.config.pipeline_parallel_size == 1
                    and item.parallel_plan.config.is_rank_local
                    and not item.parallel_plan.config.enable_expert_parallel for item in pools)):
            return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
        return runtime_inputs
    if dense and deployment.mode in ("pd_disagg", "af_disagg", "pd_af_disagg"):
        # The already adapted per-pool model consumes the same declared
        # sampler in these existing A800 modes. IDs still belong to each
        # step; the forward validates transfers, ownership and pool roles.
        pools = tuple(deployment.worker_pools.values())
        configs = tuple(item.parallel_plan.config for item in pools)
        if (model.compute_dtype == "bf16"
                and all(item.hardware_id == "nvidia_a800" for item in pools)
                and len({config.tensor_parallel_size for config in configs}) == 1
                and all(config.tensor_parallel_size in (1, 2) and config.pipeline_parallel_size == 1
                    and config.is_rank_local and not config.enable_expert_parallel for config in configs)):
            return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
        return runtime_inputs
    if dense and model.compute_dtype in ("bf16", "fp16") and deployment.mode == "colocated" and len(deployment.worker_pools) > 1:
        pools = tuple(deployment.worker_pools.values())
        # Automatic binding has one common runtime/backend declaration.
        # Mixed hardware requires each pool's explicit source selection.
        if len({pool.hardware_id for pool in pools}) != 1:
            return runtime_inputs
        configs = tuple(item.parallel_plan.config for item in pools)
        if all(is_dense_parallel_source_scope(pool, model,
                model_execution_version=runtime_inputs.model_execution_version,
                symbolic_forward_source="cuda_eager_inplace", graph_enabled=runtime_inputs.graph_execution.enabled)
                for pool in pools):
            return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
        if (model.compute_dtype == "bf16"
                and all(item.hardware_id == "nvidia_a800" and item.role == "colocated" for item in pools)
                and len({config.tensor_parallel_size for config in configs}) == 1
                and all(config.tensor_parallel_size in (1, 2) and config.pipeline_parallel_size == 1
                    and config.is_rank_local and not config.enable_expert_parallel for config in configs)):
            return replace(runtime_inputs, symbolic_forward_source="cuda_eager_inplace")
        return runtime_inputs
    if deployment.mode != "colocated" or len(deployment.worker_pools) != 1:
        return runtime_inputs
    pool, = deployment.worker_pools.values()
    parallel = pool.parallel_plan.config
    general_dense = is_dense_parallel_source_scope(pool, model,
        model_execution_version=runtime_inputs.model_execution_version,
        symbolic_forward_source="cuda_eager_inplace", graph_enabled=runtime_inputs.graph_execution.enabled)
    general_moe = is_qwen3_moe_rank_local_source_scope(pool, model,
        model_execution_version=runtime_inputs.model_execution_version,
        symbolic_forward_source="cuda_eager_inplace", graph_enabled=runtime_inputs.graph_execution.enabled)
    pipeline = (dense and pool.hardware_id == "nvidia_a800" and not parallel.enable_expert_parallel
        and is_validated_qwen3_pipeline_model_scope(model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size))
    hardware_ids = (("nvidia_a800", "nvidia_h200") if source011
        else ("metax_c500",) if source024 else ("ascend_910b",) if source018
        else ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"))
    if (pool.hardware_id not in hardware_ids or pool.role != "colocated"
            or (source011 and pool.hardware_id == "nvidia_h200" and model.compute_dtype != "bf16")
            or ((source024 or source018) and (not (dense or ordinary_moe_capture) or model.compute_dtype != "bf16"))
            or (not (general_dense or general_moe) and (parallel.tensor_parallel_size not in (1, 2)
                or (parallel.pipeline_parallel_size != 1 and not pipeline)))
            or not parallel.is_rank_local or dense and parallel.enable_expert_parallel):
        return runtime_inputs
    if moe and not general_moe and (pool.hardware_id not in (("nvidia_a800", "nvidia_h200") if source011 else ("metax_c500",) if source024 else ("ascend_910b",) if source018 else ("nvidia_h200",))
            or (source011 and pool.hardware_id == "nvidia_h200" and not _vllm011_qwen3_moe_ep2_source(model, pool)) or not (
            (parallel.tensor_parallel_size == 1 or not source011 and parallel.tensor_parallel_size == 2)
                and not parallel.enable_expert_parallel
            or parallel.tensor_parallel_size == parallel.expert_parallel_size == 2
                and parallel.enable_expert_parallel and is_validated_qwen3_moe_ep2_model_scope(model))):
        return runtime_inputs
    if moe and model.compute_dtype != "bf16":
        return runtime_inputs
    return replace(runtime_inputs, symbolic_forward_source=(
        "vllm_ascend_eager" if pool.hardware_id == "ascend_910b" else "cuda_eager_inplace"))


def _dense_source_prefix(model, step, arguments):
    """Execute the real first pipeline range of the same dense model source."""
    q_lens, kv_lens, rows = _dense_source_metadata(model, step,
        input_ids=arguments["input_ids"], positions=arguments["positions"],
        slot_mapping=arguments["slot_mapping"], cache_capacity_tokens=arguments["cache_capacity_tokens"],
        source_profile=arguments["source_profile"], hardware_id=arguments["hardware_id"],
        allow_unknown_input_ids=True)
    trace = ForwardTrace(rank=arguments["rank"], namespace=arguments["namespace"])
    network = arguments["model_type"](trace, model, step.pool_name, q_lens, kv_lens,
        arguments["cache_capacity_tokens"], arguments["slot_mapping"],
        token_ids=arguments["input_ids"], layer_range=(0, 1),
        rope_program=arguments["rope_program"], hardware_id=arguments["hardware_id"],
        model_execution_version=arguments["model_execution_version"],
        maca_copy_runtime_source=arguments["maca_copy_runtime_source"])
    shape = arguments["graph_token_shape"]
    ids = trace.tensor("input_ids", (shape.model_input_tokens,), "int32")
    positions = trace.tensor("positions", (shape.model_input_tokens,), "int64")
    slots = trace.tensor("capture.slot-mapping", (shape.model_input_tokens,), "int64")
    for layer in network.layers:
        layer.self_attn.attn.graph_token_shape = shape
        layer.self_attn.attn.slot_tensor = slots
        if arguments["model_execution_version"] == "vllm_0_18_0":
            layer.self_attn.attn.graph_attention_source = "ascend_0_18_full_fia"
    trace.begin_capture(key=f"{arguments['source_profile']}:FULL:{shape.model_input_tokens}",
        logical_tokens=rows, model_input_tokens=shape.model_input_tokens,
        inputs={"input_ids": ids, "positions": positions, "slot_mapping": slots})
    output = network(ids, positions)
    trace.end_capture(output["hidden_states"])
    return trace


def _dense_source_program(model, step, arguments, *, cache, launch_policy, pool=None, execution_context=None):
    """Reuse only the proven ordinary captured dense source program domain."""
    shape = arguments["graph_token_shape"]
    if (pool is not None and execution_context is not None
            and model.model_type in ("qwen3", "llama") and not model.num_experts
            and model.compute_dtype == model.weight_dtype == "bf16"
            and (arguments["hardware_id"], arguments["model_execution_version"]) == ("nvidia_h200", "vllm_0_30_0")
            and shape.runtime_mode == "FULL" and not arguments["graph_eager_kinds"]
            and all(request.phase == "decode" and request.scheduled_tokens == 1
                and request.mtp_proposer_mode == "disabled" for request in step.requests)):
        if "fa3_full_aot_source" in arguments:
            return _dense_retained_full_source_program(model, step, pool,
                {**arguments, "graph_inputs": arguments["fa3_full_aot_source"][0]},
                cache=cache, launch_policy=launch_policy, execution_context=execution_context)
        return _record_dense_source_program(model, step, arguments, cache=cache,
            launch_policy=launch_policy, execution_context=execution_context)
    supported = (model.model_type in ("qwen3", "llama") and not model.num_experts
        and model.num_hidden_layers > 1 and model.compute_dtype == model.weight_dtype == "bf16"
        and (arguments["hardware_id"], arguments["model_execution_version"]) in (
            ("metax_c500", "vllm_0_24_0"), ("nvidia_a800", "vllm_0_11_0"),
            ("nvidia_h200", "vllm_0_11_0"),
            ("ascend_910b", "vllm_0_18_0"))
        and shape.runtime_mode == "FULL" and not arguments["graph_eager_kinds"]
        and all(request.phase == "decode" and request.scheduled_tokens == 1
            and request.mtp_proposer_mode == "disabled" for request in step.requests))
    if not supported:
        return _trace_dense_forward(model, step, **arguments)
    return _record_dense_source_program(model, step, arguments, cache=cache, launch_policy=launch_policy)


class _DenseMetadataSourceGuard:
    """Run-local source identity contract, recorded once per source program."""

    def __init__(self, extra_modules=()):
        import sys
        from types import FunctionType
        from . import graph_padding
        modules = tuple(sys.modules[name] for name in (
            __name__, SymbolicAttention.__module__, ForwardTrace.__module__,
            graph_padding.__name__, *extra_modules))
        self.modules = tuple((module, dict(vars(module))) for module in modules)
        classes = {value for module in modules for value in vars(module).values() if isinstance(value, type)}
        self.classes = tuple((cls, self._class_contract(cls)) for cls in classes)
        functions = {value for module in modules for value in vars(module).values()
            if isinstance(value, FunctionType)}
        for cls in classes:
            for member in vars(cls).values():
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                if isinstance(member, FunctionType):
                    functions.add(member)
        self.functions = tuple((function, function.__code__, function.__defaults__,
            None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)) for function in functions)

    @staticmethod
    def _class_contract(cls):
        values = dict(vars(cls))
        if cls in (ModelConfig, ForwardTrace, ForwardCollectiveGroup, MemoryAllocation) and "__slotnames__" in values:
            # copy.copy and pickle memoize CPython's empty slots lookup.
            # No source semantics change; nonempty or non-list values reject.
            if type(values["__slotnames__"]) is not list or values["__slotnames__"]:
                return None
            del values["__slotnames__"]
        return values

    def seal(self):
        # Include this very contract in the model-owner namespace snapshot.
        self.modules = tuple((module, dict(vars(module))) for module, _ in self.modules)

    def unchanged(self):
        return (all(vars(module) == snapshot for module, snapshot in self.modules)
            and all(self._class_contract(cls) == snapshot for cls, snapshot in self.classes)
            and all(function.__code__ is code and function.__defaults__ == defaults
                and function.__kwdefaults__ == kwdefaults
                for function, code, defaults, kwdefaults in self.functions))


def _dense_parameter_metadata(model, step, arguments):
    q_lens, kv_lens, _ = _dense_source_metadata(model, step,
        input_ids=arguments["input_ids"], positions=arguments["positions"],
        slot_mapping=arguments["slot_mapping"], cache_capacity_tokens=arguments["cache_capacity_tokens"],
        source_profile=arguments["source_profile"], hardware_id=arguments["hardware_id"],
        allow_unknown_input_ids=True)
    if arguments["model_execution_version"] == "vllm_0_18_0":
        from .graph_padding import GraphAttentionShape
        attention_shape = GraphAttentionShape.from_ascend_full(
            arguments["graph_token_shape"], q_lens, kv_lens, prefill_no_cache=q_lens == kv_lens)
        metadata = {"attention_shape": attention_shape,
            "kv_len": max(attention_shape.model_input_kv_lens),
            "kv_lens": kv_lens, "slot_mapping": arguments["slot_mapping"]}
    else:
        metadata = gqa_full_source_metadata(q_lens, kv_lens, arguments["slot_mapping"], arguments["graph_token_shape"])
    return metadata


def _dense_metadata_prefix(model, step, arguments, program, *, tp_size=1):
    """Bind the recorded actual prefix from the same pure primitive metadata.

    No model operations are synthesized: all nodes, ports, aliases, events and
    lifetimes are retained from its actual source recording. Only the declared
    attention values and persistent KV layout change.
    """
    from copy import copy
    from math import prod
    metadata = _dense_parameter_metadata(model, step, arguments)
    result = copy(program.prefix)
    result._nodes = dict(result._nodes)
    result._allocations, result._views = dict(result._allocations), dict(result._views)
    attention = result.qualify("layer-0:attention")
    node = result._nodes[attention]
    physical_kv = (metadata["kv_lens"] + (0,) * (node.shape["batch"] - len(metadata["kv_lens"]))
        if "source_program" in node.attributes
        and node.attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode"
        else metadata["attention_shape"].model_input_kv_lens)
    result._nodes[attention] = replace(node, shape={**node.shape, "kv_len": metadata["kv_len"]},
        attributes={**node.attributes, "kv_lens": metadata["kv_lens"],
            "model_input_kv_lens": physical_kv})
    if "fa3_full_aot_source" in arguments:
        prepare=result.qualify("fa3_aot.prepare")
        old_prepare=result._nodes[prepare]
        padded=metadata["kv_lens"]+(0,)*(old_prepare.shape["batch"]-len(metadata["kv_lens"]))
        for component in ("prepare", "copy", "clear"):
            name=result.qualify("fa3_aot."+component)
            node=result._nodes[name]
            result._nodes[name]=replace(node,
                shape={**node.shape,"kv_len":metadata["kv_len"]} if component=="prepare" else node.shape,
                attributes={**node.attributes,"kv_lens":metadata["kv_lens"],"model_input_kv_lens":padded,
                    "fa3_full_aot_used_key":padded})
    update = attention + ":kv-update"
    result._nodes[update] = result._nodes[update].with_attributes({
        **result._nodes[update].attributes, "symbolic_slot_mapping": metadata["slot_mapping"]})
    cache = result.qualify("layer-0:kv-cache")
    shape = gqa_cache_shape(arguments["cache_capacity_tokens"], model.kv_heads_per_tensor_parallel_rank(tp_size), model.head_dim)
    result._allocations[cache] = replace(result._allocations[cache], size_bytes=prod(shape) * result._views[cache].element_bytes)
    result._views[cache] = ForwardTrace._dense_view(cache, shape, result._views[cache].dtype)
    return result


def _record_dense_source_program(model, step, arguments, *, cache, launch_policy, execution_context=None):
    """Record/bind after the caller has established the source program domain."""
    dynamic = {"positions", "slot_mapping", "cache_capacity_tokens", "namespace"}
    # All actual static source arguments, including known/unknown token IDs,
    # remain in this run-local key. ModelConfig is an immutable full contract.
    key = (model, step.pool_name, tuple(request.scheduled_tokens for request in step.requests),
        launch_policy, tuple((name, value) for name, value in arguments.items() if name not in dynamic))
    context = None
    if execution_context is not None:
        hardware, runtime = execution_context
        context = (hardware, runtime.graph_execution, runtime.norm_source_batch_invariant, runtime.bf16_gemm_model, runtime.bf16_expert_model)
        key += ((id(hardware), *context[1:]),)
    program = cache.lookup(key)
    arguments = dict(arguments)
    namespace_key = key[:5]
    if (arguments["graph_token_shape"] is not None
            and arguments["graph_token_shape"].runtime_mode == "FULL"
            and all(request.phase == "decode" and request.scheduled_tokens == 1
                for request in step.requests)):
        namespace_key = (*key[:4], tuple(
            (name, None if name == "input_ids" else value)
            for name, value in key[4]))
    arguments["namespace"] = (cache.namespace(model.model_type, step.pool_name,
        arguments["rank"], arguments["model_execution_version"], cache_key=namespace_key) if program is None else program.source.namespace)
    # Ascend 0.18 FULL decode uses the same per-layer source recurrence. Its
    # prefix helper binds the current FIA q/kv lengths and slot mapping before
    # the recorded program is reused, including padded graph rows.
    metadata_domain = (arguments["model_type"] in (Qwen3Model, LlamaModel)
        and (arguments["hardware_id"], arguments["model_execution_version"]) in (
            ("nvidia_a800", "vllm_0_11_0"), ("nvidia_h200", "vllm_0_11_0"),
            ("nvidia_h200", "vllm_0_30_0"),
            ("metax_c500", "vllm_0_24_0"),
            ("ascend_910b", "vllm_0_18_0"))
        and arguments["graph_token_shape"].runtime_mode == "FULL")
    metadata_scope = metadata_domain and _DENSE_METADATA_SOURCE_CONTRACT.unchanged()
    seal_storage = (metadata_scope
        and model.compute_dtype == model.weight_dtype == "bf16"
        and ((arguments["hardware_id"] in ("nvidia_a800", "nvidia_h200")
              and arguments["source_profile"] == "cuda_eager_inplace")
             or (arguments["hardware_id"] == "ascend_910b"
                 and arguments["source_profile"] == "vllm_ascend_eager")))
    if program is not None and metadata_domain:
        if (not metadata_scope or "_dense_metadata_guard" not in program.__dict__
                or program._dense_metadata_guard is not _DENSE_METADATA_SOURCE_CONTRACT):
            program = None
    prefix = (_dense_metadata_prefix(model, step, arguments, program)
        if program is not None and metadata_scope else _dense_source_prefix(model, step, arguments))
    if program is not None:
        bound = program.bind(prefix, native_frame=seal_storage)
        if bound is not None:
            if seal_storage:
                from .traced_forward import _seal_native_bound_storage
                _seal_native_bound_storage(program, bound)
            return bound
    source = _trace_dense_forward(model, step, **arguments)
    if launch_policy == "injected_launch":
        source.bind_launch_policy(launch_policy)
    for name, node in prefix._nodes.items():
        if name not in source._nodes:
            return source
        complete = source._nodes[name]
        if ((node.kind, node.pool_name, node.shape, node.dtype, node.variant, node.rank) !=
                (complete.kind, complete.pool_name, complete.shape, complete.dtype, complete.variant, complete.rank)
                or {key: value for key, value in node.attributes.items() if key != "launch_policy"} !=
                {key: value for key, value in complete.attributes.items() if key != "launch_policy"}):
            # A changed/opaque prefix is still executed by the complete real
            # source. It cannot establish a reusable parameter-binding program.
            return source
    if context is not None:
        _annotate_dense_source_program(source, runtime)
    node_bindings, allocation_bindings = [], []
    # The model constructor uses the same homogeneous attention class for
    # every layer. All dynamic values below come from that class's real prefix
    # invocation; static nodes and the complete source DAG are never inferred.
    for layer in range(model.num_hidden_layers):
        target, origin = source.qualify(f"layer-{layer}"), prefix.qualify("layer-0")
        node_bindings.extend(((target + ":attention", origin + ":attention", ("kv_len",),
            ("kv_lens", "model_input_kv_lens")),
            (target + ":attention:kv-update", origin + ":attention:kv-update", (), ("symbolic_slot_mapping",))))
        allocation_bindings.append((target + ":kv-cache", origin + ":kv-cache"))
    if "fa3_full_aot_source" in arguments:
        node_bindings.append((source.qualify("fa3_aot.prepare"),prefix.qualify("fa3_aot.prepare"),("kv_len",),
            ("kv_lens","model_input_kv_lens","fa3_full_aot_used_key")))
    program = ForwardSourceProgram(source, prefix,
        node_bindings=node_bindings, allocation_bindings=allocation_bindings)
    if metadata_scope:
        program._dense_metadata_guard = _DENSE_METADATA_SOURCE_CONTRACT
    if context is not None:
        program._retained_full_execution_context = context
    cache.store(key, program)
    # Return an independently bound source even on the first invocation;
    # downstream launch binding must never mutate the stored template.
    bound = program.bind(prefix, native_frame=seal_storage)
    if bound is None:
        raise RuntimeError("newly recorded dense source program rejected its own actual prefix")
    if seal_storage:
        from .traced_forward import _seal_native_bound_storage
        _seal_native_bound_storage(program, bound)
    return bound


def _bind_ordinary_fa3_gqa_source(source, pool, hardware, version, *, graph_execution=None, full_capture_source=None, full_capture_model=None):
    """Bind default eager caller facts and one shared metadata prepare.

    Eager preparation is aggregated at the first attention boundary. For
    PIECEWISE, its caller preparation stays outside captured regions; only
    attention nodes actually recorded as eager are admitted. Native page
    IDs, metadata transports and workspace parents remain explicitly unbound.
    """
    from .workload_graph import WorkloadNode, WorkloadEdge
    if full_capture_source is not None and full_capture_source.fa3_full_capture_id is not None:
        return _bind_full_fa3_gqa_source(source, pool, hardware, version, graph_execution, full_capture_source, model=full_capture_model)
    page_size = (None if pool.kv_cache is None else next(
        tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm"))
    traces = (source,) if isinstance(source, ForwardTrace) else source.traces
    changed = False
    for trace in traces:
        captures = trace.source_captures
        if captures and any(capture["runtime_mode"] != "PIECEWISE" for capture in captures):
            continue
        eager_nodes = {name for capture in captures for name in capture["eager_node_ids"]}
        bound, preparations, setup, edges = {}, {}, {}, []
        for name, node in trace._nodes.items():
            eligible = (node.kind is WorkloadKind.FLASH_ATTN
                and "source_component" in node.attributes
                and node.attributes["source_component"] == "symbolic_gqa_mathematical"
                and node.attributes["source_model_execution_version"] == version
                and node.attributes["graph_runtime_mode"] == "NONE"
                and (not captures or name in eager_nodes)
                and node.dtype in ("bf16", "fp16") and node.shape["head_dim"] in (64, 128, 256))
            if not eligible:
                bound[name] = node
                continue
            graph_facts=({"graph_configured_mode":graph_execution.configured_mode,
                "max_capture_tokens":max(graph_execution.capture_sizes) if graph_execution.capture_sizes else 0,
                "query_tokens":sum(node.attributes["q_lens"])} if graph_execution is not None
                    and graph_execution.configured_mode!="NONE" else {})
            facts=hardware.fa3_gqa_model_call_facts(version,page_size=page_size,**graph_facts)
            attributes = {**node.attributes, **facts,
                "fa3_gqa_operand_dtype": node.dtype,
                "fa3_gqa_execution_origin": "piecewise_eager" if captures else "ordinary_eager",
                "fa3_gqa_native_workspace_bound": False}
            key = (tuple(node.shape.items()), node.dtype, node.attributes["q_lens"], node.attributes["kv_lens"])
            if key not in preparations:
                prepare_id = name + ":shared-fa3-metadata"
                preparations[key] = prepare_id
                prepare_attributes = {field: value for field, value in attributes.items()
                    if field not in ("symbolic_tensor_inputs", "symbolic_tensor_outputs",
                        "symbolic_parameter_inputs", "symbolic_tensor_controls")}
                prepare = WorkloadNode(prepare_id, WorkloadKind.VECTOR, node.pool_name,
                    node.shape, "int32", "fa3_prepare_varlen", node.rank,
                    {**prepare_attributes, "source_component": "fa3_gqa_shared_metadata_prepare"})
                if captures:
                    # The caller owns lengths before model capture/replay.
                    # No QKV access or native metadata allocation is invented.
                    setup[prepare_id] = prepare
                else:
                    bound[prepare_id] = prepare
                    edges.extend(WorkloadEdge(edge.source, prepare_id)
                        for edge in trace._edges if edge.destination == name)
            prepare_id = preparations[key]
            attributes["fa3_gqa_metadata_prepare_node"] = prepare_id
            bound[name] = node.with_attributes(attributes)
            edges.append(WorkloadEdge(prepare_id, name))
        if preparations:
            trace._nodes = {**setup, **bound}
            trace._edges = [*trace._edges, *edges]
            if captures:
                # Existing captured/eager members, segments and ports retain
                # their identity; only the outside-capture prefix grew.
                trace._captures = [{**capture, "node_start": capture["node_start"] + len(setup)}
                    for capture in captures]
            trace._recorded_graph = None
            changed = True
    if changed and not isinstance(source, ForwardTrace):
        source._recorded_graph = None
    return source


def _bind_expanded_fa3_prefill_source(source, hardware, version):
    """Bind the default FA3 caller to GLM's actual expanded D256 operands.

    The trace already owns Q/K/V/O. Native scratch and length-array transport
    remain unbound; unlike ordinary GQA, this MLA wrapper prepares each call.
    """
    revision = hardware.fa3_mla_source_revision_for_model(version)
    traces = (source,) if isinstance(source, ForwardTrace) else source.traces
    for trace in traces:
        if trace.source_captures:
            continue
        for name, node in tuple(trace._nodes.items()):
            if (node.kind is WorkloadKind.FLASH_ATTN and node.variant == "mla_prefill"
                    and node.dtype == "bf16" and "source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_mla_fresh_prefill_mathematical"
                    and (node.shape["qk_head_dim"], node.shape["v_head_dim"], node.shape["kv_lora_rank"]) == (256, 256, 512)
                    and node.attributes["causal"] is True and node.attributes["return_softmax_lse"] is False
                    and node.attributes["q_lens"] == node.attributes["kv_lens"]):
                trace._nodes[name] = node.with_attributes({**node.attributes,
                    "source_component": "fa3_expanded_mla_prefill_call",
                    "source_model_execution_version": version, "fa3_expanded_source_revision": revision,
                    "fa3_backend_source": "vllm_0_30_default_hopper_flash_attn3",
                    "scheduler_metadata_precomputed": False, "fa3_runtime_binary_bound": False,
                    "fa3_length_array_transport_bound": False, "fa3_native_workspace_bound": False})
    return source


def _trace_deepseek_distributed_runtime_source(model, step, pool, runtime_inputs, hardware_by_id):
    """Project the existing declared logical arenas independently per DP owner.

    Coordinates are derived from actual request contexts, just as the
    ordinary DP1 source entry; native page IDs remain a separate receipt.
    No hybrid state, workspace history or provider facts are invented.
    """
    dp = pool.parallel_plan.config.data_parallel_size
    if (runtime_inputs.embedding_token_ids is None or runtime_inputs.model_source_inputs is not None
            or runtime_inputs.mtp_draft_sampling_source is not None or runtime_inputs.mtp_proposer_execution_source is not None):
        raise NotImplementedError("DeepSeek DP target requires explicit token IDs and no MTP/foreign source state")
    slots, capacities, prompt_lengths = [[] for _ in range(dp)], [0] * dp, [[] for _ in range(dp)]
    if model.model_type == "deepseek_v4" and runtime_inputs.v4_source_prompt_lengths is not None:
        if len(runtime_inputs.v4_source_prompt_lengths) != len(step.requests):
            raise ValueError("V4 DP prompt lengths must match the original scheduled request order")
    kimi_prompts = runtime_inputs.symbolic_source_prompt_lengths
    if model.model_type == "kimi_k2":
        by_request = runtime_inputs.symbolic_source_prompt_lengths_by_request
        if by_request is not None:
            missing = tuple(request.request_id for request in step.requests if request.request_id not in by_request)
            if missing:
                raise ValueError(f"source prompt lengths are missing scheduled requests: {missing!r}")
            kimi_prompts = tuple(by_request[request.request_id] for request in step.requests)
        if kimi_prompts is not None and len(kimi_prompts) != len(step.requests):
            raise ValueError("Kimi DP prompt lengths must match original request order")
    for index, request in enumerate(step.requests):
        owner = request.data_parallel_rank
        if not 0 <= owner < dp:
            raise ValueError("DeepSeek DP request owner is outside the physical deployment")
        context, query = request.computed_tokens, request.scheduled_tokens
        slots[owner].extend(range(capacities[owner] + context, capacities[owner] + context + query))
        capacities[owner] += context + query
        if model.model_type == "kimi_k2" and kimi_prompts is not None:
            prompt_lengths[owner].append(kimi_prompts[index])
        if model.model_type == "deepseek_v4" and runtime_inputs.v4_source_prompt_lengths is not None:
            prompt_lengths[owner].append(runtime_inputs.v4_source_prompt_lengths[index])
    slot_mapping = tuple(tuple(values) if values else None for values in slots)
    if model.model_type == "deepseek_v3":
        from .traced_deepseek_v3 import trace_deepseek_v3_data_parallel_forward
        return trace_deepseek_v3_data_parallel_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, slot_mapping_by_replica=slot_mapping,
            cache_capacity_tokens_by_replica=tuple(value if value else None for value in capacities),
            source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            source_profile=runtime_inputs.symbolic_forward_source, target_sampling_source=runtime_inputs.target_sampling_source,
            fa3_hardware=None if hardware_by_id is None else hardware_by_id[pool.hardware_id])
    if model.model_type == "kimi_k2":
        return trace_kimi_k2_data_parallel_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, slot_mapping_by_replica=slot_mapping,
            cache_capacity_tokens_by_replica=tuple(value if value else None for value in capacities),
            source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            source_profile=runtime_inputs.symbolic_forward_source, target_sampling_source=runtime_inputs.target_sampling_source,
            source_block_size=None if pool.kv_cache is None else next(
                tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm"),
            source_prompt_lengths_by_replica=None if kimi_prompts is None else tuple(
                tuple(values) if values else None for values in prompt_lengths),
            fa3_hardware=None if hardware_by_id is None else hardware_by_id[pool.hardware_id])
    if model.model_type != "deepseek_v4":
        raise NotImplementedError("distributed logical source entry requires V3 or V4")
    if hardware_by_id is None or pool.hardware_id not in hardware_by_id:
        raise ValueError("V4 DP mHC source requires the explicit deployment HWSpec")
    alignment = max(model.compress_ratios)
    return trace_deepseek_v4_data_parallel_forward(model, step, pool, hardware_by_id[pool.hardware_id], runtime_inputs.expert_routing,
        input_ids=runtime_inputs.embedding_token_ids, slot_mapping_by_replica=slot_mapping,
        cache_capacity_tokens_by_replica=tuple(-(-max(value, alignment) // alignment) * alignment if value else None
            for value in capacities),
        source_max_model_len=runtime_inputs.v4_source_max_model_len, max_batched_tokens=runtime_inputs.v4_source_max_batched_tokens,
        source_prompt_lengths_by_replica=tuple(tuple(values) if values else None for values in prompt_lengths),
        source_use_deep_gemm=runtime_inputs.v4_source_use_deep_gemm, linear_source_profile="vllm_0_30_row_major_group128_fp32",
        indexer_selection_source=runtime_inputs.v4_indexer_selection_source,
        sample_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16",
        cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
        workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes)


@dataclass
class _DSparkPPFeedback:
    received: dict
    send_buffers: list
    last_transfers: dict


def _validate_dspark_pp_inputs(inputs,step,pool,prior,draft):
    if (type(inputs) is not dict or type(prior) is not _V4TargetSourceState or draft is None
            or pool.parallel_plan.config.pipeline_parallel_size<2
            or prior.dummy_run
            or any(request.computed_tokens and request.scheduled_tokens!=6 for request in step.requests)):
        raise ValueError('PP feedback requires its retained full DSpark verification source')
    traces={trace.rank:trace for stage in prior.stages for trace in stage}
    ranks=tuple(rank for stage in range(pool.parallel_plan.config.pipeline_parallel_size-1)
        for rank in pool.physical_ranks_for_pipeline_stage(stage))
    if set(inputs)!=set(ranks):raise ValueError('PP feedback must cover exactly every original non-last pipeline rank')
    batch=len(prior.step.requests)
    for rank in ranks:
        ports=inputs[rank];trace=traces[rank]
        stage=next(index for index,rows in enumerate(prior.stages) if trace in rows)
        lane=prior.stages[stage].index(trace)
        participants=tuple(rows[lane].rank for rows in prior.stages)
        if type(ports) is not dict or set(ports)!={'sampled','counts','drafts'}:
            raise ValueError('PP feedback requires all sampled/count/draft receive ports')
        for role,shape,dtype in (('sampled',(batch,6),'int64'),('counts',(2,batch),'int32'),('drafts',(batch,5),'int64')):
            port=ports[role];trace._check_tensor(port)
            producer=trace._writers[port.view.allocation_id]
            # The producer belongs to the sending last-stage trace.
            nodes=tuple(owner._nodes[producer] for owner in traces.values() if producer in owner._nodes)
            if (port.shape!=shape or port.dtype!=dtype or len(nodes)!=1
                    or 'source_dspark_pp_feedback' not in nodes[0].attributes
                    or nodes[0].attributes['source_dspark_pp_feedback'] is not True
                    or nodes[0].attributes['destination_rank']!=rank
                    or nodes[0].attributes['source_step_id']!=prior.step.step_id
                    or nodes[0].attributes['source_rank']!=participants[-1]
                    or nodes[0].attributes['broadcast_participants']!=participants
                    or nodes[0].attributes['data_parallel_rank']!=step.requests[0].data_parallel_rank
                    or nodes[0].attributes['payload_tensor']['name']!=role):
                raise ValueError('PP feedback input differs from its actual source transfer or request extent')


def _dspark_pp_send(trace, destination, value, *, pool, config, step_id, role, owner, lane, stage,
        participants, controls):
    """One leg of an explicitly configured direct broadcast representation.

    The source API is torch.distributed.broadcast. Its logical payload is
    represented as a direct fanout on the declared fabric, independently of
    the unbound NCCL algorithm, streams and GPU allocator retirement.
    """
    from .workload_graph import WorkloadNode,WorkloadKind,WorkloadEdge
    trace._check_tensor(value)
    producer=trace._writers[value.view.allocation_id]
    if producer is None:
        raise ValueError('PP feedback broadcast requires its actual produced tensor')
    name=trace.unique_name(f'{step_id}:pp-feedback:{role}:to-{destination.rank}')
    output=destination.tensor(destination.unique_name(f'pp-receive:{step_id}:{role}'),value.shape,value.dtype)
    size=value.view.element_bytes
    for extent in value.shape: size*=extent
    trace._nodes[name]=WorkloadNode(name,WorkloadKind.P2P,pool.name,{},dtype=value.dtype,rank=trace.rank,attributes={
        'algorithm':'send_recv','backend':config.backend,'backend_config_sha256':config.backend_config_sha256,
        'bytes':size,'bytes_semantics':'payload','direction':'point_to_point','link':config.link,
        'topology':'point_to_point','participants':(trace.rank,destination.rank),'operation':'send_recv',
        'source_rank':trace.rank,'destination_rank':destination.rank,'ring_order':(),'reduction':None,
        'resource_id':config.resource_id,'source_stage':pool.parallel_plan.config.pipeline_parallel_size-1,
        'destination_stage':stage,'payload_tensor':{'name':role,'shape':value.shape,'dtype':value.dtype,'bytes':size},
        'source_component':'vllm_0_30_PPHandler_feedback','source_dspark_pp_feedback':True,
        'source_collective_operation':'broadcast','physical_representation':'declared_direct_fanout',
        'broadcast_participants':participants,'data_parallel_rank':owner,'tensor_parallel_rank':lane,
        'source_step_id':step_id,
        'source_projection_only':True,'native_collective_algorithm_bound':False,'side_stream_retirement_bound':False,
        'symbolic_tensor_inputs':(value.view.view_id,),'symbolic_tensor_outputs':(output.view.view_id,),
        'symbolic_tensor_output_allocations':(output.view.allocation_id,)})
    trace._edges.append(WorkloadEdge(producer,name))
    trace._edges.extend(WorkloadEdge(before,name,'barrier') for before in controls)
    trace._readers[value.view.allocation_id].add(name)
    destination._writers[output.view.allocation_id]=name
    trace._event('send:'+name);destination._event('receive:'+name)
    return output,name


def _dspark_pp_feedback(source, pool, step, *, prior=None, draft=False):
    from .traced_operators import SymbolicOperator
    from .workload_graph import WorkloadKind
    config=pool.parallel_plan.config
    if config.pipeline_parallel_size==1:return None
    physical=pool.collective_for('broadcast')
    traces={trace.rank:trace for trace in source.traces}
    state=_DSparkPPFeedback({},[],{}) if prior is None else prior
    for owner in range(config.data_parallel_size):
        count=sum(request.data_parallel_rank==owner for request in step.requests)
        if count==0:continue
        for lane in range(config.tensor_parallel_size):
            participants=tuple(pool.ranks[(owner*config.pipeline_parallel_size+stage)*config.tensor_parallel_size+lane]
                for stage in range(config.pipeline_parallel_size))
            trace=traces[participants[-1]];key=(owner,lane)
            if not draft:
                sampler=source.v2_sampler_outputs_by_rank[trace.rank]
                op=SymbolicOperator(trace,step.step_id+':pp-sample-pad',pool.name)
                node=op.node(WorkloadKind.VECTOR,{'elements':count*6},'int64','pp_sample_pad',{
                    'source_component':'vllm_0_30_PPHandler_feedback','source_operation':'torch_constant_pad',
                    'input_columns':sampler['sampled_token_ids'].shape[1],'output_columns':6,'padding_value':0,
                    'source_projection_only':True,'internal_launch_bound':False})
                padded=trace.call(node,(sampler['sampled_token_ids'],),shape=(count,6),dtype='int64')
                node=op.node(WorkloadKind.VECTOR,{'elements':count*2},'int32','pp_stack_counts',{
                    'source_component':'vllm_0_30_PPHandler_feedback','source_operation':'torch_stack',
                    'source_projection_only':True,'internal_launch_bound':False})
                combined=trace.call(node,(sampler['num_sampled'],sampler['num_rejected']),shape=(2,count),dtype='int32')
                payloads=(('sampled',padded),('counts',combined))
            else:
                values=(source.dspark_outputs[lane] if config.data_parallel_size==1
                    else source.dspark_outputs_by_replica[owner][lane])
                target=source._v4_target_source_state if config.data_parallel_size==1 else source._v4_distributed_target_state.states[owner]
                metadata=dict(dict(target.v2_input_ports)[trace.rank])
                op=SymbolicOperator(trace,step.step_id+':pp-pack-drafts',pool.name)
                node=op.node(WorkloadKind.GATHER,{'elements':count*5,'rows':count,'hidden':5},'int64','gather',{
                    'gather_factor':1,'index_count':count,'index_bytes':8,'scattered':True,'materialized_output':True,
                    'source_component':'vllm_0_30_PPHandler_feedback','source_operation':'pack_draft_broadcast',
                    'source_projection_only':True,'request_state_layout':'resident source RequestState rows',
                    'request_state_indices':target.v2_state_indices})
                packed=trace.call(node,(values['resident_draft_tokens'],metadata['idx_mapping']),shape=(count,5),dtype='int64')
                payloads=(('drafts',packed),)
            for role,value in payloads:
                controls=() if key not in state.last_transfers else state.last_transfers[key]
                transfers=[]
                for stage,rank in enumerate(participants[:-1]):
                    output,name=_dspark_pp_send(trace,traces[rank],value,pool=pool,config=physical,step_id=step.step_id,
                        role=role,owner=owner,lane=lane,stage=stage,participants=participants,controls=controls)
                    if rank not in state.received:state.received[rank]={}
                    state.received[rank][role]=output;transfers.append(name)
                state.last_transfers[key]=tuple(transfers);state.send_buffers.append(value)
    return state


def _trace_dspark_runtime_source(step, model, pool, runtime, hardware, cache):
    """Adapt preserved native speculation facts to the actual target/draft source.

    The native schedule remains attached to the source owner. Only the model
    caller view drops generic MTP flags; DSpark has its own V2/Markov program.
    Live recurrence is never reconstructed from allocation events.
    """
    from .runtime_inputs import DSparkSourceInputs
    from .traced_sampling import dspark_prepare_input_plan
    from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep, schedule_step_sha256
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.workload.workload_graph import WorkloadGraph
    frame = runtime.model_source_inputs
    config = pool.parallel_plan.config
    if config.pipeline_parallel_size>1:pool.collective_for('broadcast')
    if (not isinstance(frame, DSparkSourceInputs) or len(frame.replicas) != config.data_parallel_size
            or runtime.embedding_token_ids is None or cache is None
            or frame.draft_model.model_id != model.model_id + '-DSpark'):
        raise ValueError('DSpark standard source requires its current native frame, target IDs and run-local state owner')
    if any(local is None for local in frame.replicas):
        raise NotImplementedError('DSpark idle source requires its actual native cache metadata, not an absent frame')
    if any(request.data_parallel_rank >= len(frame.replicas) or request.mtp_proposer_mode != 'padded'
            or request.mtp_verified_tokens not in (0,5) or request.mtp_proposer_tokens != 5
            or request.resumed_from_preemption or request.scheduler_prefix_hit_tokens
            for request in step.requests):
        raise NotImplementedError('DSpark native source requires full depth5, no preemption/prefix and original DP owners')
    key = (model.config_sha256,frame.draft_model.config_sha256,pool, runtime.v4_source_max_model_len,
        runtime.v4_source_max_batched_tokens)
    previous = None
    if pool.name in cache.dspark_sources:
        old_key,previous = cache.dspark_sources[pool.name]
        if key != old_key:
            raise ValueError('DSpark retained source cannot change its model, placement or constructor limits')
        prior_schedule=previous._dspark_native_schedule
        expected_ids={request.request_id for request in prior_schedule.requests}-set(prior_schedule.finished_request_ids)
        admitted_ids={key for local in frame.replicas for key in local.admitted_request_ids}
        if expected_ids.intersection(admitted_ids) or expected_ids|admitted_ids != {request.request_id for request in step.requests}:
            raise NotImplementedError('DSpark retained source requires exact prior survivors plus explicit new admissions; only removes prior finished requests')
    elif any(request.computed_tokens for request in step.requests):
        raise ValueError('DSpark source cannot restart a continued request without its retained native state')
    prior_requests=() if previous is None else previous._dspark_native_schedule.requests
    prior_by_id={request.request_id:request for request in prior_requests}
    for request in step.requests:
        expected_recomputed=0 if previous is None or request.request_id in admitted_ids else max(0,min(request.computed_tokens+request.scheduled_tokens,
            prior_by_id[request.request_id].computed_tokens+prior_by_id[request.request_id].scheduled_tokens)-request.computed_tokens)
        if request.recomputed_tokens!=expected_recomputed:
            raise NotImplementedError('DSpark source only admits recomputation from its actual speculative rollback history')
    plans=[];declarations=[];slots=[];capacities=[];prompts=[];accepted=[]
    request_token_ids_by_key={request.request_id:runtime.embedding_token_ids[start:start+request.scheduled_tokens]
        for index,request in enumerate(step.requests) for start in (sum(r.scheduled_tokens for r in step.requests[:index]),)}
    for owner,local in enumerate(frame.replicas):
        requests=tuple(request for request in step.requests if request.data_parallel_rank==owner)
        dummy=not requests
        if dummy:
            old_cache=None if previous is None else (previous.dspark_cache_state if config.data_parallel_size==1
                else previous.dspark_distributed_cache_state.states[owner])
            retained=old_cache is not None and bool(old_cache.request_states)
            expected_idle_source='retained_request_state' if retained else 'never_admitted_constructor'
            if local.idle_request_state_source!=expected_idle_source:
                raise ValueError('DSpark idle declaration differs from its actual retained request-state history')
        if (local.request_ids != tuple(request.request_id for request in requests)
                or frame.draft_model.vocab_size != model.vocab_size
                or any(token >= model.vocab_size for ids in local.sampled_token_ids for token in ids)
                or any(len(ids)!=request.mtp_accepted_tokens+1 for ids,request in zip(local.sampled_token_ids,requests,strict=True))):
            raise ValueError('DSpark typed source frame differs from original Scheduler request order or feedback')
        requests_by_id={request.request_id:request for request in requests}
        if any(requests_by_id[key].computed_tokens!=0 or requests_by_id[key].phase!='prefill'
                or requests_by_id[key].scheduled_tokens!=local.prompt_lengths[local.request_ids.index(key)]
                for key in local.admitted_request_ids):
            raise NotImplementedError('DSpark native admission requires each complete fresh prompt, without chunking or resumption')
        groups=tuple(sorted(set(local.draft_layer_groups)))
        capacity=local.num_kv_blocks*min(local.block_sizes[group] for group in (local.target_swa_group,*groups))
        capacity=capacity//128*128
        if capacity<128:
            raise NotImplementedError('DSpark source requires an aligned resident subset of the native cache pool')
        tables=[]
        for group in groups:
            columns=-(-runtime.v4_source_max_model_len//local.block_sizes[group])
            if any(len(row)>columns for row in local.block_tables[group]):
                raise ValueError('DSpark native block table exceeds declared request extent')
            tables.append(((0,)*columns,) if dummy else tuple((*row,*((0,)*(columns-len(row)))) for row in local.block_tables[group]))
        rejected=tuple(request.mtp_verified_tokens-request.mtp_accepted_tokens for request in requests)
        positions=tuple(pos for request in requests for pos in range(request.computed_tokens,request.computed_tokens+request.scheduled_tokens))
        plan=dspark_prepare_input_plan(target_query_lengths=(6,) if dummy else tuple(request.scheduled_tokens for request in requests),
            target_positions=(0,)*6 if dummy else positions,rejected_counts=(0,) if dummy else rejected,group_ids=groups,
            layer_group_indices=tuple(groups.index(group) for group in local.draft_layer_groups),
            kernel_block_sizes=tuple(local.block_sizes[group] for group in groups),block_table_ids=tuple(tables),
            max_requests=local.request_state_capacity,max_tokens=runtime.v4_source_max_batched_tokens,
            max_model_len=runtime.v4_source_max_model_len,cache_capacity_tokens=capacity,num_query_per_request=5,dummy_run=dummy)
        target_slots=[]
        size=local.block_sizes[local.target_swa_group]
        for request,row in zip(requests,local.block_tables[local.target_swa_group],strict=True):
            for pos in range(request.computed_tokens,request.computed_tokens+request.scheduled_tokens):
                if pos//size>=len(row) or row[pos//size]==0:
                    raise NotImplementedError('DSpark target source requires actual resident native SWA pages')
                target_slots.append(row[pos//size]*size+pos%size)
        declaration=dict(sampled_token_ids=local.sampled_token_ids,group_ids=groups,
            layer_group_indices=plan['layer_group_indices'],kernel_block_sizes=plan['kernel_block_sizes'],
            block_table_ids=plan['block_table_ids'],max_requests=local.request_state_capacity,all_token_ids_storage=runtime.dspark_all_token_ids_storage,
            initialize_last_sampled=previous is None,
            admitted_requests=tuple((key,local.request_state_indices[local.request_ids.index(key)],request_token_ids_by_key[key])
                for key in local.admitted_request_ids))
        plans.append(plan);declarations.append({'dummy_plan':plan} if dummy else declaration);slots.append(None if dummy else tuple(target_slots));capacities.append(capacity)
        prompts.append(None if dummy else local.prompt_lengths);accepted.append(None if dummy else tuple(request.mtp_accepted_tokens for request in requests))
    target_step=ScheduleStep(step.step_id,pool.name,tuple(ScheduledRequest(request.request_id,request.phase,
        request.input_tokens,request.output_tokens,computed_tokens=request.computed_tokens,data_parallel_rank=request.data_parallel_rank)
        for request in step.requests),finished_request_ids=step.finished_request_ids,attention_kv_layout='mla')
    old_ids=set() if previous is None else {node.node_id for node in previous.graph.nodes}
    prior_target=None if previous is None else (previous._v4_target_source_state if config.data_parallel_size==1
        else previous._v4_distributed_target_state)
    prior_draft=None if previous is None else (previous.dspark_cache_state if config.data_parallel_size==1
        else previous.dspark_distributed_cache_state)
    if previous is not None:
        states=(prior_draft,) if config.data_parallel_size==1 else prior_draft.states
        for owner,state in enumerate(states):
            if state.dummy_run:continue
            for trace,request_state in zip(state.traces,state.request_states,strict=True):
                if '_v2_resident_draft_tokens' not in vars(trace):
                    raise ValueError('DSpark continuation lost its resident request draft-token source')
                resident,steps=trace._v2_resident_draft_tokens
                trace._check_tensor(resident)
                writer=trace._writers[resident.view.allocation_id]
                if (resident.shape!=(frame.replicas[owner].request_state_capacity,5)
                        or resident.view.allocation_id not in trace._persistent_ids
                        or trace._allocations[resident.view.allocation_id].freed_event is not None
                        or steps[-1]!=previous._dspark_native_schedule.step_id
                        or writer is None or writer not in trace._nodes
                        or trace._nodes[writer].variant!='vllm_gpu_v2_store_drafts'
                        or trace._nodes[writer].attributes['request_state_indices']!=request_state.state_indices):
                    raise ValueError('DSpark continuation changed its live resident draft-token backing or state rows')
    pp_inputs=None if previous is None or config.pipeline_parallel_size==1 else previous._dspark_pp_feedback.received
    ranks=tuple(pool.ranks[(owner*config.pipeline_parallel_size+config.pipeline_parallel_size-1)*config.tensor_parallel_size+lane]
        for owner in range(config.data_parallel_size) for lane in range(config.tensor_parallel_size))
    draft_pool=replace(pool,ranks=ranks,pipeline_p2p=(),parallel_plan=build_parallel_plan(replace(config,pipeline_parallel_size=1),
        num_layers=3,num_experts=model.num_experts))
    draft_requests=[];query_ids=[];context_maps=[];query_maps=[]
    for owner,(plan,local) in enumerate(zip(plans,frame.replicas,strict=True)):
        original=tuple(request for request in step.requests if request.data_parallel_rank==owner)
        if original:
            draft_requests.extend(ScheduledRequest(request.request_id,'decode',0,5,computed_tokens=context,data_parallel_rank=owner)
                for request,context in zip(original,plan['contexts'],strict=True))
        query_ids.extend(token for ids in local.sampled_token_ids for token in (ids[-1],*((frame.draft_model.dspark_noise_token_id,)*4)))
        context_maps.append(tuple(plan['full_context_slots_by_group'][group] for group in plan['layer_group_indices']))
        query_maps.append(tuple(plan['query_slots_by_group'][group] for group in plan['layer_group_indices']))
    draft_step=ScheduleStep(step.step_id+':dspark',pool.name,tuple(draft_requests),
        finished_request_ids=step.finished_request_ids,attention_kv_layout='mla')
    draft_rows=tuple(max(1,len(local.request_ids))*5 for local in frame.replicas)
    draft_work_rows=(max(draft_rows),)*config.data_parallel_size
    physical_rows=(sum(-(-count//config.tensor_parallel_size)*config.tensor_parallel_size for count in draft_work_rows)
        if config.data_parallel_size>1 and config.enable_expert_parallel and config.tensor_parallel_size>1 else sum(draft_work_rows))
    if max(draft_work_rows)>runtime.v4_source_max_batched_tokens:
        raise ValueError('DSpark uniform physical DP rows exceed the worker token bound')
    _dspark_routing_inputs(frame.draft_model,draft_step,draft_pool,runtime.expert_routing,physical_rows)
    if previous is not None:
        if previous._dspark_native_frame.draft_model != frame.draft_model:
            raise ValueError('DSpark continuation cannot change its original draft configuration')
        for owner,(old,new,plan) in enumerate(zip(previous._dspark_native_frame.replicas,frame.replicas,plans,strict=True)):
            if (old.block_sizes,old.draft_layer_groups,old.target_swa_group,old.num_kv_blocks,
                    {key:prompt for key,prompt in zip(old.request_ids,old.prompt_lengths,strict=True)
                        if key not in previous._dspark_native_schedule.finished_request_ids},old.request_state_capacity,
                    {key:index for key,index in zip(old.request_ids,old.request_state_indices,strict=True)
                        if key not in previous._dspark_native_schedule.finished_request_ids}) != (
                    new.block_sizes,new.draft_layer_groups,new.target_swa_group,new.num_kv_blocks,
                    {key:prompt for key,prompt in zip(new.request_ids,new.prompt_lengths,strict=True)
                        if key not in new.admitted_request_ids},new.request_state_capacity,
                    {key:index for key,index in zip(new.request_ids,new.request_state_indices,strict=True)
                        if key not in new.admitted_request_ids}):
                raise ValueError('DSpark continuation cannot change native group/capacity/prompt/request-state bindings')
            local=prior_draft if config.data_parallel_size==1 else prior_draft.states[owner]
            local_requests=(tuple(request for request in draft_step.requests if request.data_parallel_rank==owner)
                if new.request_ids else (ScheduledRequest('_dummy_req_0','decode',0,5,computed_tokens=1,data_parallel_rank=owner),))
            local_step=replace(draft_step,requests=local_requests,finished_request_ids=tuple(
                key for key in draft_step.finished_request_ids if key in {r.request_id for r in local_requests}))
            local_pool=draft_pool
            replica_context=None
            if config.data_parallel_size>1:
                local_pool=replace(draft_pool,ranks=tuple(trace.rank for trace in local.traces),
                    parallel_plan=build_parallel_plan(replace(draft_pool.parallel_plan.config,data_parallel_size=1,enable_expert_parallel=False),
                        num_layers=3,num_experts=model.num_experts))
                replica_context=_V4ReplicaContext(local.traces,(),(),draft_work_rows,
                    owner,(),(),config.expert_parallel_size,config.enable_expert_parallel and config.tensor_parallel_size>1,())
            if not new.request_ids:
                _dspark_dummy_continuation(local,frame.draft_model,local_step,local_pool,
                    tuple(workspace.capacity_bytes for workspace in local.workspaces),capacities[owner],
                    runtime.v4_source_max_batched_tokens,replica_context)
                if local.request_states:
                    from .traced_sampling import validate_dspark_idle_request_state
                    target=prior_target if config.data_parallel_size==1 else prior_target.states[owner]
                    if len(target.retained_sampling_ports)!=len(local.traces):
                        raise ValueError('DSpark admitted idle lost its retained SamplingStates ports')
                    sampling_ports=dict(target.retained_sampling_ports)
                    for trace,state in zip(local.traces,local.request_states,strict=True):
                        if trace.rank not in sampling_ports:
                            raise ValueError('DSpark admitted idle lost its own-rank SamplingStates ports')
                        ports=dict(sampling_ports[trace.rank])
                        if set(ports)!={'temperature','seeds'}:
                            raise ValueError('DSpark admitted idle requires its complete retained SamplingStates ports')
                        validate_dspark_idle_request_state(trace,state,ports['temperature'],ports['seeds'])
                continue
            expected,_=_dspark_incremental_context(local,frame.draft_model,local_step,local_pool,
                tuple(workspace.capacity_bytes for workspace in local.workspaces),context_maps[owner],query_maps[owner],
                capacities[owner],runtime.v4_source_max_batched_tokens,plan['target_query_lengths'],plan['rejected_counts'],replica_context,
                new.admitted_request_ids)
            if expected!=plan['context_positions']:
                raise ValueError('DSpark continuation changed its accepted context projection')
    sampling_aliases={}
    if previous is not None:
        states=(prior_draft,) if config.data_parallel_size==1 else prior_draft.states
        for owner,state in enumerate(states):
            if not state.request_states and frame.replicas[owner].admitted_request_ids:
                for trace,buffers in zip(state.traces,state.input_buffers,strict=True):
                    ports=dict(buffers.shared)
                    if any(role not in ports for role in ('dummy-temperature','dummy-seeds','dummy-last-sampled')):
                        raise ValueError('DSpark first admission lost its actual idle constructor state')
                    from .traced_sampling import validate_vllm030_last_sampled_constructor
                    validate_vllm030_last_sampled_constructor(trace,ports['dummy-last-sampled'],frame.replicas[owner].request_state_capacity)
                    sampling_aliases[trace.rank]=(('temperature',ports['dummy-temperature']),('seeds',ports['dummy-seeds']))
    target_args=dict(input_ids=runtime.embedding_token_ids,source_max_model_len=runtime.v4_source_max_model_len,
        max_batched_tokens=runtime.v4_source_max_batched_tokens,source_use_deep_gemm=False,
        linear_source_profile='vllm_0_30_row_major_group128_fp32',sample_logits=False,
        workspace_initial_capacity_bytes=runtime.symbolic_workspace_initial_capacity_bytes if previous is None else
            tuple(rank.workspace.capacity_bytes for state in ((prior_target,) if config.data_parallel_size==1 else prior_target.states) for rank in state.rank_states),
        aux_hidden_state_layers=frame.draft_model.dspark_target_layer_ids,_retain_source_state=True,
        _prior_source=prior_target,_dspark_target_model=frame.draft_model,indexer_selection_source=runtime.v4_indexer_selection_source,
        _pp_feedback_inputs=pp_inputs)
    if config.data_parallel_size==1:
        source=trace_deepseek_v4_forward(model,target_step,pool,hardware,runtime.expert_routing,**target_args,
            positions=tuple(pos for request in target_step.requests for pos in range(request.computed_tokens,request.computed_tokens+request.scheduled_tokens)),
            slot_mapping=slots[0],cache_capacity_tokens=capacities[0],_v2_prompt_lengths=prompts[0],_v2_accepted_tokens=accepted[0],
            _v2_state_indices=frame.replicas[0].request_state_indices,_v2_state_capacity=frame.replicas[0].request_state_capacity,
            _admitted_request_ids=frame.replicas[0].admitted_request_ids,_sampling_state_ports=sampling_aliases)
    else:
        source=trace_deepseek_v4_data_parallel_forward(model,target_step,pool,hardware,runtime.expert_routing,**target_args,
            slot_mapping_by_replica=tuple(slots),cache_capacity_tokens_by_replica=tuple(capacities),
            _v2_prompt_lengths_by_replica=tuple(prompts),_v2_accepted_tokens_by_replica=tuple(accepted),
            _v2_state_indices_by_replica=tuple(local.request_state_indices for local in frame.replicas),
            _v2_state_capacities=tuple(local.request_state_capacity for local in frame.replicas),
            _admitted_request_ids_by_replica=tuple(local.admitted_request_ids for local in frame.replicas),
            _sampling_state_ports_by_rank=sampling_aliases)
    pp_feedback=_dspark_pp_feedback(source,pool,step)
    draft_args=dict(input_ids=tuple(query_ids),source_max_model_len=runtime.v4_source_max_model_len,
        max_batched_tokens=runtime.v4_source_max_batched_tokens,
        workspace_initial_capacity_bytes=tuple(source.aux_workspaces_by_rank[rank].capacity_bytes for rank in ranks),
        source_use_deep_gemm=False,compute_confidence=False,target_capture_source=source,prior_context=prior_draft)
    if config.data_parallel_size==1:
        source=trace_deepseek_v4_dspark_forward(frame.draft_model,draft_step,draft_pool,hardware,runtime.expert_routing,**draft_args,
            positions=plans[0]['query_positions'],context_positions=plans[0]['context_positions'],
            context_slot_mappings=context_maps[0],query_slot_mappings=query_maps[0],cache_capacity_tokens=capacities[0],
            captured_target_layers=frame.draft_model.dspark_target_layer_ids,_dflash_inputs=declarations[0],
            target_query_lens=None if previous is None else plans[0]['target_query_lengths'],
            num_rejected=None if previous is None else plans[0]['rejected_counts'])
    else:
        source=trace_deepseek_v4_dspark_data_parallel_forward(frame.draft_model,draft_step,draft_pool,hardware,runtime.expert_routing,**draft_args,
            context_slot_mappings_by_replica=tuple(context_maps),query_slot_mappings_by_replica=tuple(query_maps),
            cache_capacity_tokens_by_replica=tuple(capacities),source_uniform_num_requests=max(max(1,len(local.request_ids)) for local in frame.replicas),
            _dflash_inputs_by_replica=tuple(declarations),
            target_query_lens_by_replica=None if previous is None else tuple(plan['target_query_lengths'] for plan in plans),
            num_rejected_by_replica=None if previous is None else tuple(plan['rejected_counts'] for plan in plans))
    from .traced_sampling import vllm030_store_draft_tokens_forward
    draft_states=(source.dspark_cache_state,) if config.data_parallel_size==1 else source.dspark_distributed_cache_state.states
    for owner,state in enumerate(draft_states):
        if not frame.replicas[owner].request_ids:continue
        values=source.dspark_outputs if config.data_parallel_size==1 else source.dspark_outputs_by_replica[owner]
        for trace,request_state,output in zip(state.traces,state.request_states,values,strict=True):
            output['resident_draft_tokens']=vllm030_store_draft_tokens_forward(trace,output['draft_tokens'],
                request_state=request_state,step_id=step.step_id,pool_name=pool.name)
    source._dspark_native_schedule=step
    source._dspark_pp_feedback=_dspark_pp_feedback(source,pool,step,prior=pp_feedback,draft=True)
    source._dspark_native_frame=frame
    source.source_scope.update(native_scheduler_schedule_sha256=schedule_step_sha256(step),
        native_speculation_preserved=True,full_speculative_runner_bound=False,
        PP_feedback_broadcast_source_bound=config.pipeline_parallel_size>1,
        PP_feedback_topology='declared direct fanout; NCCL algorithm/stream retirement unbound',
        draft_request_state_staging_bound=True,
        request_state_indices_by_replica=tuple(local.request_state_indices for local in frame.replicas),
        request_state_index_source='explicit frame; native bridge projects sorted release/LIFO admission from SchedulerOutput',
        native_tables_source='actual per-request Scheduler group IDs; zero padded unused table tail',
        storage_report_scope='retained complete source history; peak is an envelope, not current-step/normal capacity')
    cache.dspark_sources[pool.name]=(key,source)
    nodes=tuple(node for node in source.graph.nodes if node.node_id not in old_ids)
    ids={node.node_id for node in nodes}
    graph=WorkloadGraph(nodes,tuple(edge for edge in source.graph.edges if edge.source in ids and edge.destination in ids))
    return source,graph


def build_symbolic_workload(step, model, deployment, *, launch_policy, runtime_inputs, kv_cache_events=(), hardware_by_id=None,
                            trace_observer=None, source_program_cache=None):
    """Production dispatch from normalized schedule to the shared symbolic graph.

    Slot coordinates describe disjoint request ranges in a declared logical
    cache arena. Physical page allocation remains the Scheduler/KV manager's
    separate contract; this source projection never invents native page IDs.
    """
    from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent, host_kv_transfer_attributes
    from .model_config import validate_named_target_identity
    validate_named_target_identity(model)
    _validate_maca_copy_runtime_scope(model, deployment, runtime_inputs)
    if step.attention_kv_layout is not None and step.attention_kv_layout != runtime_inputs.attention_kv_layout:
        raise ValueError("scheduler attention KV layout disagrees with RuntimeInputs")
    kv_cache_events = tuple(kv_cache_events)
    if trace_observer is not None and not callable(trace_observer):
        raise TypeError("symbolic trace observer must be callable")
    if source_program_cache is not None and not isinstance(source_program_cache, SourceProgramCache):
        raise TypeError("symbolic source programs require their run-local cache owner")

    def finish(source, graph=None):
        context = _retained_full_execution_context(source.graph)
        bound_full = (context is not None and context == (hardware_by_id[pool.hardware_id],
            runtime_inputs.graph_execution, runtime_inputs.norm_source_batch_invariant, runtime_inputs.bf16_gemm_model, runtime_inputs.bf16_expert_model))
        if (launch_policy == "analytical_launch" and hardware_by_id is not None
                and runtime_inputs.model_execution_version == "vllm_0_30_0"
                and model.model_type == "glm_moe_dsa" and not runtime_inputs.graph_execution.enabled
                and runtime_inputs.symbolic_forward_source in (
                    "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
                and hardware_by_id[pool.hardware_id].supports_source_contract("vllm_0_30_fa3_gqa_fresh_prefill")):
            source = _bind_expanded_fa3_prefill_source(source, hardware_by_id[pool.hardware_id],
                runtime_inputs.model_execution_version)
        if (not bound_full and (launch_policy == "analytical_launch" or (launch_policy == "injected_launch"
                and ((model.model_type == "qwen3" and model.compute_dtype == model.weight_dtype == "fp16")
                    or (model.model_type in ("qwen3", "llama") and not model.num_experts
                        and model.compute_dtype == model.weight_dtype == "bf16"))
                and not runtime_inputs.graph_execution.enabled and parallel.tensor_parallel_size == 1))
                and hardware_by_id is not None and runtime_inputs.model_execution_version == "vllm_0_30_0"
                and runtime_inputs.graph_execution.configured_mode in ("NONE", "PIECEWISE", "FULL_AND_PIECEWISE", "FULL_DECODE_ONLY")
                and model.attention_layout == "standard_gqa"
                and not any(request.mtp_proposer_mode != "disabled" or request.mtp_proposed_tokens
                    or request.mtp_verified_tokens or request.mtp_proposer_tokens for request in step.requests)
                and hardware_by_id[pool.hardware_id].supports_source_contract("vllm_0_30_fa3_gqa_fresh_prefill")):
            source = _bind_ordinary_fa3_gqa_source(source, pool, hardware_by_id[pool.hardware_id],
                runtime_inputs.model_execution_version,graph_execution=runtime_inputs.graph_execution,
                full_capture_model=model, full_capture_source=(runtime_inputs.model_source_inputs
                    if isinstance(runtime_inputs.model_source_inputs, VLLMDenseGraphInputs) else None))
        if launch_policy == "injected_launch":
            traces = (source,) if isinstance(source, ForwardTrace) else source.traces
            for trace in traces:
                trace.bind_launch_policy(launch_policy)
        if host_transfers:
            source = ForwardKVCacheGroup(source, pool=pool, step_id=step.step_id, transfer_attributes=host_transfers)
        graph = source.graph if graph is None else graph
        if pool.hardware_id == "nvidia_h200" and not bound_full:
            graph.bind_model_execution_version(runtime_inputs.model_execution_version)
        if trace_observer is not None:
            trace_observer(source, graph)
        return graph
    for event in kv_cache_events:
        if not isinstance(event, KVCacheEvent):
            raise TypeError("symbolic workload requires normalized cache events")
    pool = deployment.worker_pools[step.pool_name]
    parallel = pool.parallel_plan.config
    versioned_moe_source = (runtime_inputs.model_execution_version in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0")
        and runtime_inputs.symbolic_forward_source == ("vllm_ascend_eager" if runtime_inputs.model_execution_version == "vllm_0_18_0" else "cuda_eager_inplace") and model.model_type == "qwen3_moe"
        and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
        and _versioned_qwen3_moe_ep2_source(model, pool, runtime_inputs.model_execution_version))
    moe_full_source = (_vllm030_qwen3_moe_full_source(model, pool, runtime_inputs.model_execution_version)
        and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
        and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
        and runtime_inputs.graph_execution.configured_mode in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "PIECEWISE")
        and runtime_inputs.graph_execution.uniform_decode_query_len == 1
        and runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16")
    moe_capture = runtime_inputs.graph_execution.enabled and (versioned_moe_source or moe_full_source)
    if (model.model_type in ("qwen3", "llama") and runtime_inputs.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager")
            and runtime_inputs.target_sampling_source is not None):
        expected_sampler = ("vllm_0_11" if runtime_inputs.model_execution_version == "vllm_0_11_0"
            else "vllm_0_18" if runtime_inputs.model_execution_version == "vllm_0_18_0"
            else "vllm_0_24" if runtime_inputs.model_execution_version == "vllm_0_24_0"
            else "vllm_0_30") + "_plain_greedy_" + model.compute_dtype
        cast_only = (runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16"
            and runtime_inputs.model_execution_version == "vllm_0_30_0" and model.compute_dtype == "bf16")
        if runtime_inputs.target_sampling_source != expected_sampler and not cast_only:
            raise ValueError("ordinary dense sampler source must match the model compute dtype")
    from .graph_padding import VLLMDenseGraphInputs
    dense_graph_inputs = (runtime_inputs.model_source_inputs
        if isinstance(runtime_inputs.model_source_inputs, VLLMDenseGraphInputs) else None)
    if dense_graph_inputs is not None and (
            ((model.model_type not in ("qwen3", "llama") or model.num_experts) and not moe_full_source)
            or (parallel.tensor_parallel_size != 2 and not (parallel.tensor_parallel_size == 1
                and pool.hardware_id == "nvidia_h200"
                and (dense_graph_inputs.fa3_full_capture_id is not None or moe_full_source)))
            or parallel.pipeline_parallel_size != 1
            or pool.hardware_id not in ("nvidia_a800", "nvidia_h200")):
        raise NotImplementedError("dense retained input state requires NVIDIA dense TP2/PP1 capture")
    if dense_graph_inputs is not None and dense_graph_inputs.fa3_full_capture_id is not None:
        if (pool.hardware_id != "nvidia_h200" or parallel.tensor_parallel_size not in (1, 2)
                or model.compute_dtype not in ("bf16", "fp16") or model.compute_dtype != model.weight_dtype
                or (((parallel.tensor_parallel_size == 2 and model.model_type not in ("qwen3", "llama"))
                    or (model.compute_dtype == "fp16" and model.model_type != "qwen3")
                    or model.num_experts) and not moe_full_source)
                or model.head_dim != 128
                or runtime_inputs.graph_execution.configured_mode not in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
                or launch_policy != "analytical_launch"):
            raise ValueError("FA3 FULL declaration requires its exact H200 TP1/2 source route")
        if dense_graph_inputs.request_ids != tuple(request.request_id for request in step.requests):
            raise ValueError("FA3 FULL capture input order must match the actual requests")
    if (runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and runtime_inputs.model_execution_version == "vllm_0_30_0" and runtime_inputs.target_sampling_source is None):
        if (not _ordinary_model_only_source_scope(model, deployment, runtime_inputs)
                or kv_cache_events or any(request.mtp_proposer_mode != "disabled" or any((request.mtp_proposed_tokens,
                    request.mtp_verified_tokens, request.mtp_proposer_tokens, request.mtp_next_proposed_tokens)) for request in step.requests)):
            raise NotImplementedError("static LM-head-only source requires0.30 CUDA dense TP1/H200 dense TP2 or H200 Qwen3-MoE TP1/validated TP=EP2 eager without MTP/graph")
    if runtime_inputs.model_execution_version == "vllm_0_11_0" and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace":
        moe_ep2 = _vllm011_qwen3_moe_ep2_source(model, pool)
        from src.serving.deployment.validated_scope import is_validated_standalone_af_ep2_scope
        moe_disagg = (model.model_type == "qwen3_moe" and not runtime_inputs.graph_execution.enabled
            and ((deployment.mode == "af_disagg" and ((parallel.tensor_parallel_size == 1 and not parallel.enable_expert_parallel)
                or is_validated_standalone_af_ep2_scope(deployment, model)))
                or (deployment.mode == "pd_disagg" and (moe_ep2 or (parallel.tensor_parallel_size == 1
                    and not parallel.enable_expert_parallel)))))
        dense_pipeline = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and is_validated_qwen3_pipeline_model_scope(model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
            and not model.tie_word_embeddings
            and not parallel.enable_expert_parallel)
        dense_disagg = (deployment.mode in ("af_disagg", "pd_disagg", "pd_af_disagg") and model.model_type in ("qwen3", "llama")
            and not model.num_experts and (not runtime_inputs.graph_execution.enabled
                or _ordinary_af_capture_scope(model, deployment, runtime_inputs.model_execution_version))
            and not parallel.enable_expert_parallel and parallel.tensor_parallel_size in (1, 2))
        h200_dense = (pool.hardware_id == "nvidia_h200" and model.model_type in ("qwen3", "llama")
            and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and parallel.pipeline_parallel_size == 1 and parallel.tensor_parallel_size in (1, 2)
            and not parallel.enable_expert_parallel)
        h200_moe = (pool.hardware_id == "nvidia_h200" and model.model_type == "qwen3_moe"
            and moe_ep2 and deployment.mode == "colocated" and len(deployment.worker_pools) == 1)
        roofline_dense = (pool.hardware_id in ("nvidia_h20", "nvidia_b200", "nvidia_b300", "ascend_950pr", "ascend_950dt")
            and model.model_type in ("qwen3", "llama")
            and not model.num_experts and model.compute_dtype == model.weight_dtype == "bf16"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and parallel.pipeline_parallel_size == 1 and parallel.tensor_parallel_size in (1, 2)
            and not parallel.enable_expert_parallel)
        if ((pool.hardware_id != "nvidia_a800" and not h200_dense and not h200_moe and not roofline_dense) or model.model_type not in ("qwen3", "llama", "qwen3_moe")
                or (deployment.mode != "colocated" and not (dense_disagg or moe_disagg)) or (parallel.enable_expert_parallel and not moe_ep2)
                or (parallel.pipeline_parallel_size > 1 and not dense_pipeline)
                or (model.model_type == "qwen3_moe" and ((runtime_inputs.graph_execution.enabled and not moe_capture)
                    or (not moe_ep2 and not moe_disagg and (parallel.tensor_parallel_size != 1 or len(pool.ranks) != 1))))):
            raise NotImplementedError("vLLM0.11 source requires H200 BF16 dense colocated TP1/TP2 PP1 or existing A800 dense/MoE deployment scope")
    if runtime_inputs.model_execution_version in ("vllm_0_18_0", "vllm_0_24_0"):
        if (pool.hardware_id != ("ascend_910b" if runtime_inputs.model_execution_version == "vllm_0_18_0" else "metax_c500") or (model.model_type not in ("qwen3", "llama") and not versioned_moe_source)
                or (model.num_experts and not versioned_moe_source) or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
                or deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                or parallel.tensor_parallel_size not in (1, 2) or parallel.pipeline_parallel_size != 1
                or not parallel.is_rank_local or (parallel.enable_expert_parallel and not versioned_moe_source)):
            raise NotImplementedError("versioned device model source requires matching C5000.24 or Ascend0.18 BF16 dense colocated TP1/TP2 PP1 EP-off")
    intents = tuple(event for event in kv_cache_events if event.action in (
        KVCacheAction.LOAD, KVCacheAction.EVICT, KVCacheAction.STORE))
    host_transfers = ()
    if intents:
        from src.serving.deployment.validated_scope import is_validated_colocated_tp2_host_kv_scope
        glm4_af_host = (model.model_type == "glm4_moe" and deployment.mode == "af_disagg"
            and pool.hardware_id == "nvidia_a800" and len(pool.ranks) == 1
            and runtime_inputs.symbolic_forward_source == "vllm_0_11_glm4_sync")
        v030_mla_host = (model.model_type in ("glm_moe_dsa", "kimi_k2")
            and model.attention_layout in ("mla", "dsa_mla")
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.symbolic_forward_source in (
                "vllm_0_30_v32_bf16_sync_explicit_allreduce",
                "vllm_0_30_v32_fp8_sync_explicit_allreduce",
                "vllm_0_30_k2_6_int4_sync_explicit_allreduce")
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and pool.hardware_id == "nvidia_h200" and len(pool.ranks) == 8
            and parallel.tensor_parallel_size == 8
            and parallel.pipeline_parallel_size == 1
            and not parallel.enable_expert_parallel and parallel.is_rank_local)
        ordinary_host = (model.attention_layout == "standard_gqa"
            and (model.model_type in ("qwen3", "qwen3_moe", "llama") or glm4_af_host)
            and parallel.pipeline_parallel_size == 1
            and not parallel.enable_expert_parallel and parallel.is_rank_local)
        if not ordinary_host and not v030_mla_host:
            raise NotImplementedError("virtual Host KV requires a validated GQA or vLLM0.30 MLA domain")
        if (len(pool.ranks) != 1
                and not is_validated_colocated_tp2_host_kv_scope(deployment, model)
                and not v030_mla_host):
            raise NotImplementedError("virtual Host KV requires one owner, validated Qwen3 TP2 or vLLM0.30 MLA TP8")
        host_transfers = tuple(host_kv_transfer_attributes(event, cache_config=pool.kv_cache,
            model_bytes_per_token=model.kv_bytes_per_token_per_parallel_rank(parallel.tensor_parallel_size,
                parallel.pipeline_parallel_size), scheduled_request_ids={request.request_id for request in step.requests})
            for event in intents)
    from src.serving.deployment.validated_scope import is_qwen3_moe_dp_sp_source_scope, is_deepseek_parallel_source_scope, is_kimi_k2_data_parallel_source_scope
    if runtime_inputs.symbolic_forward_source == "vllm_0_30_v4_dspark":
        from src.serving.deployment.validated_scope import is_dspark_runtime_source_scope
        if (deployment.mode != 'colocated' or len(deployment.worker_pools)!=1 or launch_policy!='analytical_launch'
                or not is_dspark_runtime_source_scope(pool,model,runtime_inputs) or intents):
            raise NotImplementedError('DSpark standard source requires its declared eager colocated geometry without Host KV')
        if hardware_by_id is None or pool.hardware_id not in hardware_by_id:
            raise ValueError('DSpark standard source requires the actual deployment HWSpec')
        source,graph=_trace_dspark_runtime_source(step,model,pool,runtime_inputs,hardware_by_id[pool.hardware_id],source_program_cache)
        return finish(source,graph)
    if parallel.data_parallel_size > 1 and model.model_type in ("deepseek_v3", "deepseek_v4", "kimi_k2"):
        if (not (is_deepseek_parallel_source_scope if model.model_type != "kimi_k2" else is_kimi_k2_data_parallel_source_scope)(pool, model, model_execution_version=runtime_inputs.model_execution_version,
                symbolic_forward_source=runtime_inputs.symbolic_forward_source, graph_enabled=runtime_inputs.graph_execution.enabled)
                or deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                or launch_policy != "analytical_launch" or host_transfers):
            raise NotImplementedError("DeepSeek DP runtime requires the versioned H200 colocated eager analytical target source")
        return finish(_trace_deepseek_distributed_runtime_source(model, step, pool, runtime_inputs, hardware_by_id))
    if is_qwen3_moe_dp_sp_source_scope(pool, model):
        if (runtime_inputs.model_execution_version != "vllm_0_30_0"
                or runtime_inputs.symbolic_forward_source != "cuda_eager_inplace"
                or deployment.mode not in ("colocated", "pd_disagg")
                or (deployment.mode == "colocated" and len(deployment.worker_pools) != 1)
                or launch_policy != "analytical_launch" or runtime_inputs.graph_execution.enabled or host_transfers
                or runtime_inputs.target_sampling_source != "vllm_0_30_plain_greedy_bf16"):
            raise NotImplementedError("coordinated MoE DP/SP currently requires ordinary eager0.30 colocated analytical source")
        return finish(trace_qwen3_moe_data_parallel_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, source_profile=runtime_inputs.symbolic_forward_source,
            source_program_cache=source_program_cache))
    if launch_policy == "injected_launch":
        injected_h200_fp16_eager_tp1 = (pool.hardware_id == "nvidia_h200"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and model.model_type == "qwen3" and not model.num_experts
            and model.compute_dtype == model.weight_dtype == "fp16"
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_fp16"
            and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
            and parallel.is_rank_local and not parallel.enable_expert_parallel
            and not runtime_inputs.graph_execution.enabled and not intents
            and runtime_inputs.mtp_draft_sampling_source is None
            and runtime_inputs.mtp_proposer_execution_source is None)
        injected_h200_bf16_eager_tp1 = (pool.hardware_id == "nvidia_h200"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and model.model_type in ("qwen3", "llama") and not model.num_experts
            and model.compute_dtype == model.weight_dtype == "bf16"
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
            and parallel.is_rank_local and not parallel.enable_expert_parallel
            and not runtime_inputs.graph_execution.enabled and not intents
            and runtime_inputs.mtp_draft_sampling_source is None
            and runtime_inputs.mtp_proposer_execution_source is None)
        injected_pipeline = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and is_validated_qwen3_pipeline_model_scope(model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size))
        if (runtime_inputs.symbolic_forward_source != ("vllm_ascend_eager" if runtime_inputs.model_execution_version == "vllm_0_18_0" else "cuda_eager_inplace")
                or model.model_type not in ("qwen3", "llama") or model.num_experts
                or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
                or (pool.hardware_id != "nvidia_a800" and not (((pool.hardware_id == "nvidia_h200"
                    and runtime_inputs.model_execution_version == "vllm_0_11_0")
                    or (pool.hardware_id == "metax_c500" and runtime_inputs.model_execution_version == "vllm_0_24_0")
                    or (pool.hardware_id == "ascend_910b" and runtime_inputs.model_execution_version == "vllm_0_18_0"))
                    and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                    and parallel.pipeline_parallel_size == 1))
                or deployment.mode not in ("colocated", "af_disagg", "pd_disagg", "pd_af_disagg")
                or parallel.tensor_parallel_size not in (1, 2) or (parallel.pipeline_parallel_size != 1 and not injected_pipeline)
                or parallel.enable_expert_parallel or intents) and not versioned_moe_source and not injected_h200_fp16_eager_tp1 and not injected_h200_bf16_eager_tp1:
            raise NotImplementedError("virtual injected selection requires ordinary A800 BF16 dense deployment or H200 vLLM0.11 dense colocated TP1/TP2 PP1 or declared H2000.30 FP16 eagerTP1 source")
    elif launch_policy != "analytical_launch":
        raise ValueError("virtual source requires an explicit analytical or injected launch policy")
    if runtime_inputs.embedding_token_ids is None:
        # TP1 owns the whole vocabulary; the0.11 TP2 masked program instead
        # gathers all rows before masking/AR. Their public logical payloads
        # depend only on shape. Keep IDs unknown, never dummy or sequential;
        # the fused TP branch still needs actual IDs for shard ownership.
        shape_only_embedding = ((parallel.tensor_parallel_size == 1 and len(pool.ranks) == 1)
            or (parallel.tensor_parallel_size == 2 and len(pool.ranks) == 2
                and ((runtime_inputs.model_execution_version == "vllm_0_11_0" and pool.hardware_id in ("nvidia_a800", "nvidia_h200"))
                    or (runtime_inputs.model_execution_version in ("vllm_0_24_0", "vllm_0_30_0") and pool.hardware_id == "metax_c500")
                    or (runtime_inputs.model_execution_version == "vllm_0_18_0" and pool.hardware_id == "ascend_910b"))))
        shape_only_moe = (model.model_type == "qwen3_moe" and parallel.tensor_parallel_size == 1
            and pool.hardware_id == "nvidia_h200" and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.expert_routing.source_kind in ("explicit_assumption", "profile_observation"))
        shape_only_replicas = (runtime_inputs.model_execution_version == "vllm_0_11_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and all(item.role == "colocated" and item.hardware_id == "nvidia_a800"
                and item.parallel_plan.config.tensor_parallel_size in (1, 2)
                and item.parallel_plan.config.pipeline_parallel_size == 1 and item.parallel_plan.config.is_rank_local
                and not item.parallel_plan.config.enable_expert_parallel
                for item in deployment.worker_pools.values()))
        shape_only_dense = (((model.model_type in ("qwen3", "llama") and not model.num_experts
                and runtime_inputs.expert_routing.source_kind == "not_applicable") or shape_only_moe or versioned_moe_source)
            and model.compute_dtype == model.weight_dtype and model.compute_dtype in ("bf16", "fp16")
            and deployment.mode == "colocated" and (len(deployment.worker_pools) == 1 or shape_only_replicas) and shape_only_embedding
            and parallel.pipeline_parallel_size == 1
            and parallel.is_rank_local and (not parallel.enable_expert_parallel or versioned_moe_source)
            and runtime_inputs.serving_framework == "vllm"
            and runtime_inputs.model_execution_version in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0")
            and runtime_inputs.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager")
            and runtime_inputs.mtp_draft_sampling_source is None and runtime_inputs.mtp_proposer_execution_source is None
            and runtime_inputs.model_source_inputs is None
            and (not runtime_inputs.graph_execution.enabled or (parallel.tensor_parallel_size == 1
                and pool.hardware_id == "nvidia_h200"
                and runtime_inputs.model_execution_version == "vllm_0_30_0")
                or (parallel.tensor_parallel_size in (1, 2)
                and ((runtime_inputs.model_execution_version == "vllm_0_11_0" and pool.hardware_id in ("nvidia_a800", "nvidia_h200"))
                    or (runtime_inputs.model_execution_version == "vllm_0_24_0" and pool.hardware_id == "metax_c500")
                    or (runtime_inputs.model_execution_version == "vllm_0_18_0" and pool.hardware_id == "ascend_910b"))))
            # The Scheduler already resolved prefix hits/misses and supplies
            # the resulting token/KV lengths. Performance tracing needs those
            # shapes, not embedding values or a second token-level cache test.
            # Host transfers retain their separate payload contract.
            and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events)
            and all(request.mtp_proposer_mode == "disabled" and not any((request.mtp_proposed_tokens,
                request.mtp_verified_tokens, request.mtp_proposer_tokens, request.mtp_next_proposed_tokens))
                for request in step.requests))
        if not shape_only_dense:
            raise ValueError("symbolic forward requires explicit scheduled token IDs outside ordinary shape-only source; no value fallback")
    if runtime_inputs.graph_execution.enabled:
        config = pool.parallel_plan.config
        pipeline_capture = (runtime_inputs.model_execution_version == "vllm_0_11_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and pool.hardware_id == "nvidia_a800" and not config.enable_expert_parallel
            and is_validated_qwen3_pipeline_model_scope(model, config.tensor_parallel_size, config.pipeline_parallel_size))
        sglang_capture = (runtime_inputs.serving_framework == "sglang" and model.model_type == "deepseek_v3"
            and pool.hardware_id == "nvidia_h200" and config.tensor_parallel_size == 8
            and len(pool.ranks) == 8 and config.is_rank_local and config.pipeline_parallel_size == 1
            and not config.enable_expert_parallel and deployment.mode == "colocated" and not kv_cache_events)
        host_capture = (runtime_inputs.model_execution_version == "vllm_0_11_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace" and pool.hardware_id == "nvidia_a800"
            and model.model_type in ("qwen3", "llama") and not model.num_experts)
        af_capture = (_ordinary_af_capture_scope(model, deployment, runtime_inputs.model_execution_version)
            and (not intents or host_capture))
        kimi_capture = (runtime_inputs.symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and _kimi_single_rank_source_scope(model, pool)
            and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE) for event in kv_cache_events))
        if not kimi_capture and not af_capture and not sglang_capture and not moe_capture and not pipeline_capture and (model.model_type not in ("qwen3", "llama") or model.num_experts or deployment.mode != "colocated"
                or pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
                or config.tensor_parallel_size not in (1, 2) or len(pool.ranks) != config.tensor_parallel_size
                or any(value != 1 for value in (config.pipeline_parallel_size,
                    config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
                or config.enable_expert_parallel or (intents and not host_capture)):
            raise NotImplementedError("source capture requires NVIDIA dense Qwen/Llama TP1/TP2 PP1 EP-off without Host KV")
    if runtime_inputs.symbolic_forward_source == "vllm_0_30_k3_mxfp4_sync":
        from .runtime_inputs import K3SourceInputs, K3DataParallelSourceInputs
        from src.serving.deployment.validated_scope import is_kimi_k3_runtime_source_scope
        state = runtime_inputs.model_source_inputs
        if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                or launch_policy != "analytical_launch" or not is_kimi_k3_runtime_source_scope(pool, model, runtime_inputs)):
            raise NotImplementedError("K3 standard source requires its declared eager H200 parallel geometry and latent tier")
        if not isinstance(state, (K3SourceInputs, K3DataParallelSourceInputs)) or runtime_inputs.embedding_token_ids is None:
            raise ValueError("ordinary K3 execution requires Scheduler token IDs and hybrid cache coordinates")
        if any(event.action not in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events):
            raise NotImplementedError("K3 source does not support Host KV transfer events")
        if parallel.data_parallel_size > 1:
            if not isinstance(state, K3DataParallelSourceInputs):
                raise ValueError("K3 DP source requires the actual per-replica hybrid state bundle")
            return finish(trace_kimi_k3_data_parallel_forward(model, step, pool, runtime_inputs.expert_routing,
                input_ids=runtime_inputs.embedding_token_ids, source_inputs_by_replica=state.source_inputs_by_replica,
                source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
                max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
                latent_tail_tier=runtime_inputs.k3_latent_tail_tier,
                cache_capacity_tokens_by_replica=state.cache_capacity_tokens_by_replica,
                state_pool_slots_by_replica=state.state_pool_slots_by_replica,
                kda_decode_backend=runtime_inputs.k3_kda_decode_backend,
                fa3_hardware=None if hardware_by_id is None else hardware_by_id[pool.hardware_id]))
        if not isinstance(state, K3SourceInputs) or state.request_ids != tuple(request.request_id for request in step.requests):
            raise ValueError("K3 cache/state coordinates must match the original scheduled requests")
        positions = tuple(position for request in step.requests
            for position in range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens))
        return finish(trace_kimi_k3_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=positions,
            slot_mapping=state.slot_mapping, cache_capacity_tokens=state.cache_capacity_tokens,
            state_indices_by_layer=state.state_indices_by_layer, state_pool_slots=state.state_pool_slots,
            source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            prefill_output_source=state.prefill_output_source, latent_tail_tier=runtime_inputs.k3_latent_tail_tier,
            kda_decode_backend=runtime_inputs.k3_kda_decode_backend,
            fa3_hardware=(hardware_by_id[pool.hardware_id] if hardware_by_id is not None
                and launch_policy=="analytical_launch" and not runtime_inputs.graph_execution.enabled else None)))
    positions, slots, offset = [], [], 0
    source_prompt_lengths = runtime_inputs.symbolic_source_prompt_lengths
    prompt_lengths_by_request = runtime_inputs.symbolic_source_prompt_lengths_by_request
    if prompt_lengths_by_request is not None:
        missing = [request.request_id for request in step.requests
            if request.request_id not in prompt_lengths_by_request]
        if missing:
            raise ValueError(f"source prompt lengths are missing scheduled requests: {missing!r}")
        source_prompt_lengths = tuple(prompt_lengths_by_request[request.request_id]
            for request in step.requests)
    for request in step.requests:
        context, query = request.computed_tokens, request.scheduled_tokens
        positions.extend(range(context, context + query))
        slots.extend(range(offset + context, offset + context + query))
        offset += context + query
    if runtime_inputs.symbolic_forward_source in ("sglang_0_5_19_deepseek_v3_decode_fp8",
            "sglang_0_5_19_deepseek_v3_decode_fp8_flashinfer_ar_norm",
            "sglang_0_5_19_deepseek_v3_target_fp8", "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm"):
        from .traced_sglang import (SOURCE_FUSED_PROFILE, SOURCE_TARGET_PROFILE, SOURCE_TARGET_FUSED_PROFILE, PREFILL_PREFIX_SOURCE,
            SGLangPrefillInputs, SGLangGraphInputs, SGLangEagleChainState,
            trace_sglang_deepseek_v3_decode, trace_sglang_deepseek_v3_forward)
        if deployment.mode != "colocated":
            raise NotImplementedError("SGLang source requires colocated deployment")
        # These named profiles declare the Cutlass/FlashMLA source branches
        # and whether FlashInfer AR/residual/RMSNorm fusion is enabled.
        # They do not infer installed providers or runtime flags.
        arguments = dict(input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions),
            slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_profile=runtime_inputs.symbolic_forward_source,
            linear_source_profile="logical_group128_fp32", mla_attention_source="flashmla_bf16",
            use_deep_gemm_bmm=False,
            allreduce_fusion=(runtime_inputs.symbolic_forward_source in (SOURCE_FUSED_PROFILE, SOURCE_TARGET_FUSED_PROFILE)),
            fused_shared_experts=1, scale_routed_in_topk=True,
            grouped_topk_source="triton_grouped", moe_runner_source="cutlass_fp8")
        if runtime_inputs.graph_execution.enabled:
            if isinstance(runtime_inputs.model_source_inputs, SGLangEagleChainState):
                state = runtime_inputs.model_source_inputs
                if state.phase == "prefill" and any(request.computed_tokens for request in step.requests):
                    raise NotImplementedError("SGLang EAGLE prefix prefill requires graph-off source inputs")
                if state.phase == "verify":
                    batch = len(step.requests)
                    if state.proposal_depth != 2:
                        raise NotImplementedError("EAGLE graph source requires its topk1/steps2 source")
                    from .traced_sglang import resolve_eagle_capture_batch
                    captured_batch = resolve_eagle_capture_batch(state, runtime_inputs.graph_execution.capture_sizes)
                    arguments["eagle_graph_capture_batch_size"] = captured_batch
                    if captured_batch > batch:
                        # Same reserved FlashMLA page0 as ordinary SG graphs;
                        # translate declared logical arenas, never native pages.
                        arguments["slot_mapping"] = tuple(slot + 64 for slot in slots)
                        arguments["cache_capacity_tokens"] = offset + 64
                        state = replace(state, draft_slot_mapping=tuple(slot + 64 for slot in state.draft_slot_mapping),
                            draft_decode_slot_mapping=tuple(slot + 64 for slot in state.draft_decode_slot_mapping),
                            draft_cache_capacity_tokens=state.draft_cache_capacity_tokens + 64)
                        arguments["eagle_state"] = state
                # Source prefill remains eager; its first draft-extend uses
                # different preparation from the verified three-stage replay.
            else:
                token_shape = runtime_inputs.graph_execution.resolve(step.token_count,
                    uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
                if token_shape.runtime_mode == "FULL":
                    arguments["graph_token_shape"] = token_shape
                    if token_shape.padding_tokens:
                        # SGLang's FlashMLA allocator reserves page0 (64 rows)
                        # for graph dummy slot0. Keep active logical ranges apart;
                        # this is not a native allocator/capacity claim.
                        arguments["slot_mapping"] = tuple(slot + 64 for slot in slots)
                        arguments["cache_capacity_tokens"] = offset + 64
        graph_inputs = runtime_inputs.model_source_inputs if isinstance(runtime_inputs.model_source_inputs, SGLangGraphInputs) else None
        if graph_inputs is not None:
            arguments["graph_inputs"] = graph_inputs
        if runtime_inputs.symbolic_forward_source in (SOURCE_TARGET_PROFILE, SOURCE_TARGET_FUSED_PROFILE):
            source_inputs = (arguments.pop("eagle_state") if "eagle_state" in arguments else
                (None if graph_inputs is not None else runtime_inputs.model_source_inputs))
            if isinstance(source_inputs, SGLangPrefillInputs):
                if source_inputs.request_ids != tuple(request.request_id for request in step.requests):
                    raise ValueError("SGLang prefill metadata must match the scheduled request order")
                arguments.update(one_shot_kv_indices=(source_inputs.one_shot_kv_indices or None),
                    max_kv_chunk_capacity=source_inputs.max_kv_chunk_capacity,
                    prefix_chunk_kv_indices=source_inputs.prefix_chunk_kv_indices)
                source_inputs = None
            return finish(trace_sglang_deepseek_v3_forward(model, step, pool, runtime_inputs.expert_routing,
                **arguments, prefill_source=PREFILL_PREFIX_SOURCE,
                sampling_source=runtime_inputs.target_sampling_source, eagle_state=source_inputs))
        return finish(trace_sglang_deepseek_v3_decode(model, step, pool, runtime_inputs.expert_routing, **arguments))
    if runtime_inputs.symbolic_forward_source in ("vllm_0_27_1_deepseek_v3_sync_fp8", "vllm_0_11_deepseek_v3_sync_fp8"):
        if runtime_inputs.mtp_draft_sampling_source == "vllm_0_27_1_eager_full_logits_greedy":
            from .traced_deepseek_v3 import build_deepseek_v3_mtp_workload
            return build_deepseek_v3_mtp_workload(step, model, deployment,
                launch_policy=launch_policy, runtime_inputs=runtime_inputs,
                kv_cache_events=kv_cache_events, trace_observer=trace_observer,hardware_by_id=hardware_by_id)
        from .traced_deepseek_v3 import trace_deepseek_v3_decode
        if deployment.mode != "colocated":
            raise NotImplementedError("DeepSeek-V3 initial virtual source requires colocated deployment")
        if (offset == 0 or max(r.computed_tokens + r.scheduled_tokens for r in step.requests)
                > runtime_inputs.symbolic_source_max_model_len
                or runtime_inputs.symbolic_source_max_model_len > model.max_position_embeddings
                or step.token_count > runtime_inputs.symbolic_source_max_batched_tokens):
            raise ValueError("DeepSeek-V3 declared source constructor limits must contain its scheduled contexts and rows")
        return finish(trace_deepseek_v3_decode(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
            target_sampling_source=runtime_inputs.target_sampling_source,
            source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            fa3_hardware=(hardware_by_id[pool.hardware_id] if launch_policy == "analytical_launch"
                and runtime_inputs.model_execution_version == "vllm_0_27_1"
                and hardware_by_id is not None and pool.hardware_id == "nvidia_h200" else None)))
    if runtime_inputs.symbolic_forward_source == "vllm_0_11_glm4_sync":
        if deployment.mode not in ("colocated", "pd_disagg", "af_disagg", "pd_af_disagg"):
            raise NotImplementedError("GLM4 common forward requires colocated, PD or its configured AF pair")
        has_mtp = any(r.mtp_proposer_tokens or r.mtp_verified_tokens for r in step.requests)
        if has_mtp:
            if runtime_inputs.mtp_draft_sampling_source != "vllm_0_11_eager_full_logits_greedy":
                raise NotImplementedError("GLM4 MTP-off source cannot omit its proposer/rejection work")
            if pool.kv_cache is None:
                raise ValueError("GLM4 MTP requires an explicit cache block-size contract")
            block_size = next(tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")
            slots, offset = [], 0
            for request in step.requests:
                slots.extend(range(offset + request.computed_tokens,
                    offset + request.computed_tokens + request.scheduled_tokens))
                extent = request.computed_tokens + request.scheduled_tokens + max(request.mtp_proposer_tokens - 1, 0)
                offset += -(-extent // block_size) * block_size
        if deployment.mode in ("af_disagg", "pd_af_disagg"):
            return finish(trace_af_forward(model, step, deployment, runtime_inputs.expert_routing,
                input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
                cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
                max_model_len=runtime_inputs.symbolic_source_max_model_len,
                max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
                max_requests=runtime_inputs.symbolic_source_max_requests))
        if has_mtp:
            return finish(trace_glm4_mtp_forward(model, step, pool, runtime_inputs.expert_routing,
                input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
                cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
                max_model_len=runtime_inputs.symbolic_source_max_model_len,
                max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
                max_requests=runtime_inputs.symbolic_source_max_requests))
        return finish(trace_glm4_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
            # GPUModelRunner._sample selects the ordinary sampler when this
            # pool has no speculative metadata. PD prefill keeps
            # the global rejection declaration but does not run a proposer.
            target_sampling_source=("vllm_0_11_plain_greedy_bf16"
                if runtime_inputs.target_sampling_source == "vllm_0_11_greedy_rejection_bf16"
                else runtime_inputs.target_sampling_source)))
    from .runtime_inputs import DSparkFirstProposalInputs
    if isinstance(runtime_inputs.model_source_inputs,DSparkFirstProposalInputs):
        state=runtime_inputs.model_source_inputs
        if (runtime_inputs.symbolic_forward_source!="vllm_0_30_v4_sync_fp8_ds_mla"
                or deployment.mode!="colocated" or len(deployment.worker_pools)!=1
                or any(event.action not in (KVCacheAction.ALLOCATE,KVCacheAction.RELEASE,KVCacheAction.MISS) for event in kv_cache_events)
                or hardware_by_id is None or state.request_ids!=tuple(r.request_id for r in step.requests)):
            raise ValueError("DSpark first proposal requires its explicit paired colocated source and current state")
        return finish(trace_deepseek_v4_dspark_first_proposal(model,state.draft_model,step,pool,
            hardware_by_id[pool.hardware_id],runtime_inputs.expert_routing,input_ids=runtime_inputs.embedding_token_ids,
            target_positions=state.target_positions,target_slot_mapping=state.target_slot_mapping,
            anchor_token_ids=state.anchor_token_ids,draft_token_ids=state.draft_token_ids,
            source_kind=state.source_kind,source_id=state.source_id,
            max_model_len=runtime_inputs.v4_source_max_model_len,max_batched_tokens=runtime_inputs.v4_source_max_batched_tokens,
            verification=state if state.is_verification else None))
    if runtime_inputs.symbolic_forward_source == "vllm_0_30_v4_sync_fp8_ds_mla":
        if model.model_type != "deepseek_v4":
            raise NotImplementedError("symbolic V4 source requires its accurate DeepSeek-V4 model config")
        if deployment.mode != "colocated":
            raise NotImplementedError("symbolic V4 source currently requires colocated deployment")
        if hardware_by_id is None or pool.hardware_id not in hardware_by_id:
            raise ValueError("symbolic V4 source needs the explicit deployment HWSpec for mHC split storage")
        # This is only a declared logical cache arena containing current
        # request ranges and one descriptor row for the largest compression
        # ratio. It never replaces the physical/native KV capacity ledger.
        alignment = max(model.compress_ratios)
        logical_capacity = -(-max(offset, alignment) // alignment) * alignment
        return finish(trace_deepseek_v4_forward(model, step, pool, hardware_by_id[pool.hardware_id],
            runtime_inputs.expert_routing, input_ids=runtime_inputs.embedding_token_ids,
            positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=logical_capacity,
            source_max_model_len=runtime_inputs.v4_source_max_model_len,
            max_batched_tokens=runtime_inputs.v4_source_max_batched_tokens,
            source_use_deep_gemm=runtime_inputs.v4_source_use_deep_gemm,
            source_prompt_lengths=runtime_inputs.v4_source_prompt_lengths,
            indexer_selection_source=runtime_inputs.v4_indexer_selection_source,
            workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes,
            sample_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16",
            cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
            linear_source_profile="vllm_0_30_row_major_group128_fp32"))
    if (runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
            and runtime_inputs.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy"):
        from .traced_glm_mtp import trace_glm_dsa_mtp_forward
        pd_decode = (deployment.mode == "pd_disagg" and pool.role == "decode"
            and parallel.pipeline_parallel_size == 1)
        if (not (deployment.mode == "colocated" or pd_decode)
                or parallel.pipeline_parallel_size > 1 and len(deployment.worker_pools) != 1):
            raise NotImplementedError("V32 padded MTP source requires a complete colocated target pool or pool-local PD decode/PP1")
        return finish(trace_glm_dsa_mtp_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, source_profile=runtime_inputs.symbolic_forward_source,
            source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            max_requests=runtime_inputs.symbolic_source_max_requests,
            workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes,
            selection_source=runtime_inputs.glm_indexer_selection_source,
            prefill_attention_source=runtime_inputs.glm_prefill_attention_source,
            source_inputs=runtime_inputs.model_source_inputs,
            indexer_max_logits_mb=runtime_inputs.glm_indexer_max_logits_mb,
            legacy_skip_topk_by_data_rank=runtime_inputs.glm_mtp_legacy_skip_topk_by_data_rank,
            source_program_cache=source_program_cache))
    if (runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
            and parallel.pipeline_parallel_size > 1):
        from .traced_glm_pipeline import trace_glm_pipeline_forward
        if (deployment.mode not in ("colocated", "pd_disagg")
                or deployment.mode == "colocated" and len(deployment.worker_pools) != 1):
            raise NotImplementedError("GLM PP source projection requires its configured colocated or PD pool")
        return finish(trace_glm_pipeline_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes,
            selection_source=runtime_inputs.glm_indexer_selection_source,
            source_profile=runtime_inputs.symbolic_forward_source,
            target_sampling_source=runtime_inputs.target_sampling_source,
            prefill_attention_source=runtime_inputs.glm_prefill_attention_source,
            source_prompt_lengths=source_prompt_lengths,indexer_max_logits_mb=runtime_inputs.glm_indexer_max_logits_mb,
            source_program_cache=source_program_cache))
    if runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce",
            "vllm_0_30_k2_6_int4_sync_explicit_allreduce"):
        from src.serving.deployment.validated_scope import is_glm_distributed_source_scope, is_glm_fp8_data_parallel_source_scope
        single_stage = (len(pool.ranks) == parallel.tensor_parallel_size
            and all(value == 1 for value in (parallel.pipeline_parallel_size, parallel.sequence_parallel_size,
                parallel.context_parallel_size, parallel.data_parallel_size)))
        if (deployment.mode not in ("colocated", "pd_disagg")
                or not (single_stage or _kimi_pipeline_source_scope(model, pool) or is_glm_distributed_source_scope(pool, model)
                    or is_glm_fp8_data_parallel_source_scope(pool, model))):
            raise NotImplementedError("virtual MLA production dispatch requires colocated/PD, one rank per TP lane and PP/SP/CP/DP1")
    if deployment.mode == "pd_disagg":
        # The same per-pool forward runs on either side of the existing
        # request-scoped KV handoff. No model graph is copied into a PD path.
        from src.serving.deployment.validated_scope import is_validated_tp1_tp2_multi_pool_scope, is_gqa_pd_source_scope
        from src.serving.deployment.deployment_config import worker_role_phase
        single_rank_source = (model.model_type in ("qwen3", "qwen3_moe", "llama")
            and model.attention_layout == "standard_gqa"
            and {worker_role_phase(item.role) for item in deployment.worker_pools.values()} == {"prefill", "decode"}
            and all(item.hardware_id == "nvidia_a800" and len(item.ranks) == 1
                and item.parallel_plan.config.worker_count == 1 and item.parallel_plan.config.is_rank_local
                and not item.parallel_plan.config.enable_expert_parallel for item in deployment.worker_pools.values()))
        mla_source = (runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce",
            "vllm_0_30_k2_6_int4_sync_explicit_allreduce") and _ordinary_mla_pd_source_scope(model, deployment))
        if not (single_rank_source or is_validated_tp1_tp2_multi_pool_scope(deployment, model) or mla_source
                or runtime_inputs.model_execution_version == "vllm_0_30_0" and is_gqa_pd_source_scope(deployment, model)):
            raise NotImplementedError("virtual PD forward requires one-rank GQA, validated TP1/TP2 ownership or an existing H200 MLA per-pool source")
    if deployment.mode in ("af_disagg", "pd_af_disagg"):
        graph_token_shape = None
        if runtime_inputs.graph_execution.enabled:
            uniform_decode = all(request.phase == "decode" and request.scheduled_tokens == 1 for request in step.requests)
            token_shape = runtime_inputs.graph_execution.resolve(step.token_count, uniform_decode=uniform_decode)
            if token_shape.runtime_mode == "FULL" and not uniform_decode:
                raise ValueError("AF FULL source capture requires uniform decode; use FULL_AND_PIECEWISE for prefill or mixed work")
            if token_shape.runtime_mode in ("FULL", "PIECEWISE"):
                graph_token_shape = token_shape
        return finish(trace_af_forward(model, step, deployment, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids,
            positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_profile=runtime_inputs.symbolic_forward_source,
            model_execution_version=runtime_inputs.model_execution_version,
            graph_token_shape=graph_token_shape,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if graph_token_shape is not None and graph_token_shape.runtime_mode == "PIECEWISE" else ())))
    if runtime_inputs.symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce":
        if parallel.pipeline_parallel_size > 1 and (deployment.mode != "colocated" or len(deployment.worker_pools) != 1):
            raise NotImplementedError("Kimi pipeline source requires one colocated pool")
        token_shape = (runtime_inputs.graph_execution.resolve(step.token_count,
            uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
            if runtime_inputs.graph_execution.enabled else None)
        captured = token_shape is not None and token_shape.runtime_mode in ("FULL", "PIECEWISE")
        return finish(trace_kimi_k2_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            source_profile=runtime_inputs.symbolic_forward_source, target_sampling_source=runtime_inputs.target_sampling_source,
            include_lm_head=True, source_prompt_lengths=source_prompt_lengths,
            graph_token_shape=token_shape if captured else None,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if captured and token_shape.runtime_mode == "PIECEWISE" else ()),
            source_program_cache=(source_program_cache if launch_policy == "analytical_launch" else None),
            fa3_hardware=(hardware_by_id[pool.hardware_id] if pool.hardware_id == "nvidia_h200"
                and hardware_by_id is not None and not captured
                and launch_policy == "analytical_launch" else None),
            source_block_size=None if pool.kv_cache is None else next(
                tier.block_size_tokens for tier in pool.kv_cache.tiers if tier.name.lower() == "hbm")))
    if parallel.pipeline_parallel_size > 1:
        from src.serving.deployment.validated_scope import is_dense_parallel_source_scope
        colocated_replicas = (deployment.mode == "colocated"
            and all(is_dense_parallel_source_scope(replica, model,
                model_execution_version=runtime_inputs.model_execution_version,
                symbolic_forward_source=runtime_inputs.symbolic_forward_source,
                graph_enabled=runtime_inputs.graph_execution.enabled)
                for replica in deployment.worker_pools.values()))
        if ((deployment.mode != "colocated" or len(deployment.worker_pools) != 1)
                and not colocated_replicas
                and not (deployment.mode == "pd_disagg" and runtime_inputs.model_execution_version == "vllm_0_30_0")
                or runtime_inputs.symbolic_forward_source not in ("cuda_eager_inplace", "vllm_ascend_eager")):
            raise NotImplementedError("virtual PP requires one colocated dense source pool")
        token_shape = (runtime_inputs.graph_execution.resolve(step.token_count,
            uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
            if runtime_inputs.graph_execution.enabled else None)
        captured = token_shape is not None and token_shape.runtime_mode in ("FULL", "PIECEWISE")
        return finish(trace_dense_pipeline_forward(model, step, pool, expert_routing=runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
            model_execution_version=runtime_inputs.model_execution_version,
            graph_token_shape=token_shape if captured else None,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if captured and token_shape.runtime_mode == "PIECEWISE" else ())))
    # MLA callers validate their own precise checkpoint, TP/EP and source
    # metadata domain before the generic dense/Qwen-only EP gate below.
    if runtime_inputs.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce"):
        if parallel.context_parallel_size > 1 or parallel.data_parallel_size > 1:
            from .traced_glm_pcp import trace_glm_pcp_forward
            glm_forward = trace_glm_pcp_forward
            source_program_arguments = {"indexer_max_logits_mb":runtime_inputs.glm_indexer_max_logits_mb,
                "source_program_cache":source_program_cache}
        else:
            glm_forward = trace_glm_dsa_forward
            source_program_arguments = {
                "source_program_cache": source_program_cache if launch_policy == "analytical_launch" else None,
                "num_speculative_tokens": runtime_inputs.glm_target_num_speculative_tokens}
        return finish(glm_forward(model, step, pool, runtime_inputs.expert_routing,
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_max_model_len=runtime_inputs.symbolic_source_max_model_len,
            max_batched_tokens=runtime_inputs.symbolic_source_max_batched_tokens,
            workspace_initial_capacity_bytes=runtime_inputs.symbolic_workspace_initial_capacity_bytes,
            selection_source=runtime_inputs.glm_indexer_selection_source,
            prefill_attention_source=runtime_inputs.glm_prefill_attention_source,
            source_prompt_lengths=source_prompt_lengths,
            source_profile=runtime_inputs.symbolic_forward_source, target_sampling_source=runtime_inputs.target_sampling_source,
            **source_program_arguments))
    if (deployment.mode not in ("colocated", "pd_disagg") or pool.hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b", "nvidia_h20", "nvidia_b200", "nvidia_b300", "ascend_950pr", "ascend_950dt")
            or len(pool.ranks) != parallel.tensor_parallel_size
            or (parallel.enable_expert_parallel and model.model_type != "qwen3_moe")
            or any(value != 1 for value in (parallel.pipeline_parallel_size, parallel.sequence_parallel_size,
                parallel.context_parallel_size, parallel.data_parallel_size))):
        raise NotImplementedError("symbolic dense production path requires its CUDA-alike colocated source, PP/SP/CP/DP1 EP-off")
    if model.model_type == "qwen3_moe":
        graph_token_shape = None
        if runtime_inputs.graph_execution.enabled:
            token_shape = runtime_inputs.graph_execution.resolve(step.token_count,
                uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
            if token_shape.runtime_mode in ("FULL", "PIECEWISE"):
                graph_token_shape = token_shape
        arguments = dict(
            input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions), slot_mapping=tuple(slots),
            cache_capacity_tokens=offset, source_profile=runtime_inputs.symbolic_forward_source,
            model_execution_version=runtime_inputs.model_execution_version,
            sample_logits=runtime_inputs.target_sampling_source not in (None, "vllm_0_30_plain_sampler_bf16"),
            cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
            graph_token_shape=graph_token_shape,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if graph_token_shape is not None and graph_token_shape.runtime_mode == "PIECEWISE" else ()))
        if dense_graph_inputs is not None:
            arguments["graph_inputs"] = dense_graph_inputs
            if dense_graph_inputs.fa3_full_capture_id is not None:
                arguments["fa3_full_aot_source"] = (dense_graph_inputs, max(runtime_inputs.graph_execution.capture_sizes))
        ordinary_program = (source_program_cache is not None and deployment.mode == "colocated" and not host_transfers
            and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events))
        return finish(_moe_parallel_source_program(model, step, pool, runtime_inputs.expert_routing, arguments,
            cache=source_program_cache, launch_policy=launch_policy,
            execution_context=(hardware_by_id[pool.hardware_id], runtime_inputs)
                if hardware_by_id is not None and pool.hardware_id == "nvidia_h200" else None) if ordinary_program else
            trace_qwen3_moe_forward(model, step, pool, runtime_inputs.expert_routing, **arguments))
    if parallel.tensor_parallel_size > 1:
        graph_token_shape = None
        if runtime_inputs.graph_execution.enabled:
            token_shape = runtime_inputs.graph_execution.resolve(step.token_count,
                uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
            if token_shape.runtime_mode in ("FULL", "PIECEWISE"):
                graph_token_shape = token_shape
        arguments = dict(input_ids=runtime_inputs.embedding_token_ids, positions=tuple(positions),
            slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_profile=runtime_inputs.symbolic_forward_source, graph_token_shape=graph_token_shape,
            model_execution_version=runtime_inputs.model_execution_version,
            sample_logits=runtime_inputs.target_sampling_source not in (None, "vllm_0_30_plain_sampler_bf16"),
            cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
            graph_inputs=dense_graph_inputs,
            graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                if graph_token_shape is not None and graph_token_shape.runtime_mode == "PIECEWISE" else ()))
        ordinary_program = (source_program_cache is not None and deployment.mode == "colocated"
            and pool.role == "colocated" and not host_transfers
            and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events))
        if dense_graph_inputs is not None and dense_graph_inputs.fa3_full_capture_id is not None:
            arguments["fa3_full_aot_source"] = (dense_graph_inputs, max(runtime_inputs.graph_execution.capture_sizes))
        return finish(_dense_parallel_source_program(model, step, pool, arguments,
            cache=source_program_cache, launch_policy=launch_policy,
            execution_context=(hardware_by_id[pool.hardware_id], runtime_inputs)
                if hardware_by_id is not None and pool.hardware_id == "nvidia_h200" else None) if ordinary_program else
            trace_dense_parallel_forward(model, step, pool, **arguments))
    if model.model_type == "qwen3":
        forward = trace_qwen3_forward
    elif model.model_type == "llama":
        forward = trace_llama_forward
    else:
        raise NotImplementedError("symbolic dense source requires an adapted Qwen3 or Llama forward")
    if runtime_inputs.graph_execution.enabled:
        token_shape = runtime_inputs.graph_execution.resolve(step.token_count,
            uniform_decode=all(r.phase == "decode" and r.scheduled_tokens == 1 for r in step.requests))
        if token_shape.runtime_mode in ("FULL", "PIECEWISE"):
            (_validate_qwen3_source if model.model_type == "qwen3" else _validate_llama_source)(model)
            arguments = dict(input_ids=runtime_inputs.embedding_token_ids,
                positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=offset,
                source_profile=runtime_inputs.symbolic_forward_source, rank=pool.ranks[0], namespace=step.step_id,
                model_type=Qwen3Model if model.model_type == "qwen3" else LlamaModel,
                hardware_id=pool.hardware_id, rope_program=vllm_rope_program(pool.hardware_id, model.rope_scaling_type),
                graph_token_shape=token_shape,
                model_execution_version=runtime_inputs.model_execution_version,
                maca_copy_runtime_source=runtime_inputs.maca_copy_runtime_source,
                graph_eager_kinds=(runtime_inputs.graph_execution.piecewise_eager_workload_kinds
                    if token_shape.runtime_mode == "PIECEWISE" else ()))
            if (pool.hardware_id == "ascend_910b" and runtime_inputs.model_execution_version == "vllm_0_18_0"
                    and runtime_inputs.graph_execution.configured_mode == "PIECEWISE"):
                arguments["ascend_piecewise_source"] = True
            if dense_graph_inputs is not None and dense_graph_inputs.fa3_full_capture_id is not None:
                arguments["fa3_full_aot_source"]=(dense_graph_inputs,max(runtime_inputs.graph_execution.capture_sizes))
            ordinary_program = (source_program_cache is not None and deployment.mode == "colocated"
                and parallel.is_rank_local and not parallel.enable_expert_parallel and not host_transfers
                and all(event.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE,
                    KVCacheAction.HIT, KVCacheAction.MISS) for event in kv_cache_events))
            return finish(_dense_source_program(model, step, arguments, cache=source_program_cache,
                launch_policy=launch_policy, pool=pool,
                execution_context=(hardware_by_id[pool.hardware_id], runtime_inputs)
                    if hardware_by_id is not None and pool.hardware_id == "nvidia_h200" else None)
                if ordinary_program else _trace_dense_forward(model, step, **arguments))
    return finish(forward(model, step, input_ids=runtime_inputs.embedding_token_ids,
        positions=tuple(positions), slot_mapping=tuple(slots), cache_capacity_tokens=offset,
        source_profile=runtime_inputs.symbolic_forward_source, rank=pool.ranks[0],
        namespace=step.step_id, hardware_id=pool.hardware_id,
        model_execution_version=runtime_inputs.model_execution_version,
        maca_copy_runtime_source=runtime_inputs.maca_copy_runtime_source,
        sample_logits=runtime_inputs.target_sampling_source not in (None, "vllm_0_30_plain_sampler_bf16"),
        cast_logits=runtime_inputs.target_sampling_source == "vllm_0_30_plain_sampler_bf16",
        **({"ascend_piecewise_source": True} if pool.hardware_id == "ascend_910b"
            and runtime_inputs.model_execution_version == "vllm_0_18_0"
            and runtime_inputs.graph_execution.configured_mode == "PIECEWISE" else {})))


def _retained_full_execution_context(graph):
    from .traced_forward import _NativeSourceFrame
    frame = graph.__dict__["_native_source_frame"] if "_native_source_frame" in graph.__dict__ else None
    owner = frame.owner() if type(frame) is _NativeSourceFrame else None
    return (owner.__dict__["_retained_full_execution_context"]
        if owner is not None and "_retained_full_execution_context" in owner.__dict__ else None)


def _bind_full_fa3_gqa_source(source, pool, hardware, version, graph_execution, state, *, model=None):
    """Narrow declared Q1 capture; native metadata/page transport stays partial."""
    from .workload_graph import WorkloadNode, WorkloadEdge
    from .traced_forward import _source_view_layouts
    from math import prod
    config = pool.parallel_plan.config
    if (version != "vllm_0_30_0" or graph_execution.configured_mode not in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
            or graph_execution.uniform_decode_query_len != 1 or config.tensor_parallel_size not in (1, 2)
            or any(value != 1 for value in (config.pipeline_parallel_size, config.data_parallel_size,
                config.context_parallel_size, config.sequence_parallel_size))
            or (config.enable_expert_parallel and (model is None
                or not _vllm030_qwen3_moe_full_source(model, pool, version)))):
        raise ValueError("FA3 FULL requires its declared ordinary TP1/2 Q1 source scenario")
    page = (None if pool.kv_cache is None else next(t.block_size_tokens for t in pool.kv_cache.tiers if t.name.lower() == "hbm"))
    traces = (source,) if isinstance(source, ForwardTrace) else source.traces
    if len(traces) != config.tensor_parallel_size or {trace.rank for trace in traces} != set(pool.ranks):
        raise ValueError("FA3 FULL source must preserve every actual rank exactly once")
    for trace in traces:
        captures = trace.source_captures
        if len(captures) != 1 or captures[0]["runtime_mode"] != "FULL":
            raise ValueError("FA3 FULL source requires its actual complete captured model")
        physical = captures[0]["model_input_tokens"]
        bpad = min(physical, state.fa3_full_max_num_seqs)
        facts = hardware.fa3_gqa_model_call_facts(version, page_size=page,
            graph_configured_mode=graph_execution.configured_mode,
            max_capture_tokens=max(graph_execution.capture_sizes), query_tokens=physical)
        # Index this invocation's immutable layouts once, retaining the first
        # exact alias in source order. Avoid cloning every static view merely
        # to read layout, and scanning all views for each layer's Q and O.
        physical_views_by_layout = {}
        for view in _source_view_layouts(trace):
            key = (view.allocation_id, view.offset_bytes, view.shape, view.strides_bytes)
            if key not in physical_views_by_layout:
                physical_views_by_layout[key] = view
        bound, prepare = {}, None
        native_aot_edges = {(edge.source, edge.destination) for edge in trace._edges if edge.kind == "data"}
        native_aot_chains = {}

        def native_aot_prepare(node, attributes):
            required = {"fa3_full_aot_prepare_node", "fa3_full_aot_copy_node", "fa3_full_aot_clear_node",
                "fa3_full_aot_group", "fa3_full_aot_input_producers", "symbolic_tensor_inputs"}
            if not required.issubset(attributes):
                raise ValueError("FA3 native AOT source requires its complete producer chain")
            names = tuple(attributes[key] for key in (
                "fa3_full_aot_prepare_node", "fa3_full_aot_copy_node", "fa3_full_aot_clear_node"))
            key = (*names, attributes["fa3_full_aot_group"])
            if key not in native_aot_chains:
                for name, component, kind in zip(names, ("prepare", "copy", "clear"),
                        (WorkloadKind.VECTOR, WorkloadKind.MEMCPY, WorkloadKind.VECTOR), strict=True):
                    if name not in trace._nodes:
                        raise ValueError("FA3 native AOT source omitted a producer")
                    helper = trace._nodes[name]
                    helper_fields = {"fa3_full_aot_component", "fa3_full_aot_group", "source_model_execution_version"}
                    if (helper.kind is not kind or helper.rank != node.rank or helper.pool_name != node.pool_name
                            or not helper_fields.issubset(helper.attributes)
                            or helper.attributes["fa3_full_aot_component"] != component
                            or helper.attributes["fa3_full_aot_group"] != attributes["fa3_full_aot_group"]
                            or helper.attributes["source_model_execution_version"] != version):
                        raise ValueError("FA3 native AOT source changed its producer geometry or placement")
                if (names[0], names[1]) not in native_aot_edges or (names[1], names[2]) not in native_aot_edges:
                    raise ValueError("FA3 native AOT source lost prepare/copy/clear dependencies")
                native_aot_chains[key] = {names[2]}
            inputs, producers = attributes["symbolic_tensor_inputs"], attributes["fa3_full_aot_input_producers"]
            if (type(inputs) is not tuple or type(producers) is not tuple or not inputs or len(inputs) != len(producers)
                    or inputs[-1] not in trace._views
                    or trace._views[inputs[-1]].allocation_id != attributes["fa3_full_aot_group"]
                    or producers[-1] not in native_aot_chains[key]
                    or (producers[-1], node.node_id) not in native_aot_edges):
                raise ValueError("FA3 native AOT source lost its actual semaphore writer dependency")
            native_aot_chains[key].add(node.node_id)
            return trace._nodes[names[0]]

        edges = []
        for name, node in trace._nodes.items():
            if not (node.kind is WorkloadKind.FLASH_ATTN and "source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_gqa_mathematical"):
                bound[name] = node
                continue
            attrs = node.attributes
            q, kv = attrs["q_lens"], attrs["kv_lens"]
            if (node.dtype not in ("bf16", "fp16") or node.shape["head_dim"] != 128 or attrs["graph_runtime_mode"] != "FULL"
                    or any(value != 1 for value in q) or len(q) > bpad
                    or max(kv) > state.fa3_full_capture_max_seq_len or name not in captures[0]["node_ids"]):
                raise ValueError("FA3 FULL source lost its exact captured decode domain")
            qport = trace._views[attrs["symbolic_tensor_inputs"][0]]
            oport = trace._views[attrs["symbolic_tensor_outputs"][0]]
            physical_views = []
            for port in (qport, oport):
                key = (port.allocation_id, port.offset_bytes, (physical, *port.shape[1:]), port.strides_bytes)
                if key not in physical_views_by_layout:
                    raise ValueError("FA3 FULL requires actual full Q/O parent views")
                parent_view = physical_views_by_layout[key]
                parent = trace._allocations[port.allocation_id]
                physical_views.append((port.allocation_id, port.offset_bytes,
                    parent_view.shape, parent_view.strides_bytes, parent.allocated_event, parent.freed_event))
            attrs = {**attrs, **facts, "source_program": "vllm_0_30_fa3_gqa_full_decode",
                "fa3_gqa_caller_scenario":"vllm030_full_decode_zero_request_tail_default_aot",
                "model_input_q_lens": tuple(q) + (0,) * (bpad-len(q)),
                "model_input_kv_lens": tuple(kv) + (0,) * (bpad-len(q)),
                "graph_model_input_tokens": physical, "fa3_gqa_operand_dtype": node.dtype,
                "fa3_gqa_execution_origin": "full_captured", "fa3_gqa_native_workspace_bound": False,
                "fa3_full_capture_id": state.fa3_full_capture_id,
                "fa3_full_max_num_seqs": state.fa3_full_max_num_seqs,
                "fa3_full_capture_max_seq_len": state.fa3_full_capture_max_seq_len,
                "fa3_full_physical_parents": tuple(physical_views),
                "fa3_full_input_producers": (attrs["fa3_full_aot_input_producers"] if "fa3_full_aot_input_producers" in attrs else
                    tuple(trace._writers[trace._views[view].allocation_id] for view in attrs["symbolic_tensor_inputs"]))}
            shape = {**node.shape, "batch": bpad}
            native_aot = "fa3_full_aot_prepare_node" in attrs
            if native_aot:
                native_prepare = native_aot_prepare(node, attrs)
                if prepare is not None and prepare is not native_prepare:
                    raise ValueError("FA3 FULL source requires one shared metadata producer per rank")
                prepare = native_prepare
            if prepare is None:
                prepare_id = name + ":shared-fa3-metadata"
                prepare_attrs = {field:value for field,value in attrs.items() if field not in (
                    "symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_parameter_inputs", "symbolic_tensor_controls")}
                prepare = WorkloadNode(prepare_id, WorkloadKind.VECTOR, node.pool_name, shape, "int32",
                    "fa3_prepare_varlen", node.rank, {**prepare_attrs, "source_component":"fa3_gqa_shared_metadata_prepare"})
            attrs["fa3_gqa_metadata_prepare_node"] = prepare.node_id
            bound[name] = WorkloadNode(name, node.kind, node.pool_name, shape, node.dtype, node.variant, node.rank, attrs)
            if not native_aot:
                # Native AOT already records prepare -> copy -> clear -> the
                # semaphore writer chain through every attention layer.
                # Adding a second prepare fanout invents redundant edges and
                # prevents the exact layer-repeat representation from folding.
                # Synthetic metadata has no such source proof and still needs
                # its explicit prepare dependency on every consumer.
                edges.append(WorkloadEdge(prepare.node_id, name))
        if prepare is None:
            raise ValueError("FA3 FULL source has no actual captured attention")
        trace._nodes = {prepare.node_id: prepare, **bound}
        trace._edges = [*trace._edges, *edges]
        trace._captures = [{**captures[0], "node_start": captures[0]["node_start"]+(0 if prepare.node_id in bound else 1)}]
        trace._recorded_graph = None
    if not isinstance(source, ForwardTrace):
        source._recorded_graph = None
    return source


def _trace_full_fa3_aot(trace, model, pool, q_lens, kv_lens, token_shape, state, max_capture, *, heads, kv_heads):
    """Real AOT parents and field writers; copied rounded holes stay undefined."""
    from .workload_graph import WorkloadNode
    from itertools import accumulate
    b=min(token_shape.model_input_tokens,state.fa3_full_max_num_seqs)
    rounded=(b+3)//4*4
    dynamic=b<=992
    vectors=2+int(dynamic)
    n=vectors*rounded+1
    capacity=1+4*((max(state.fa3_full_max_num_seqs,max_capture)+3)//4*4)
    if b<len(q_lens) or any(q!=1 for q in q_lens):
        raise ValueError("FA3 AOT requires real Q1 requests inside its padded descriptor")
    physical_q=tuple(q_lens)+(0,)*(b-len(q_lens))
    physical_k=tuple(kv_lens)+(0,)*(b-len(q_lens))
    prefix=(0,*accumulate(physical_q))
    offsets=(0,rounded,2*rounded,3*rounded) if dynamic else (0,rounded,2*rounded)
    holes=tuple(offset+i for offset in offsets[:-1] for i in range(b,rounded))
    group=trace.qualify("fa3_aot.scheduler_storage")
    layout=(b,rounded,n,capacity,dynamic,offsets,holes)
    attrs={"source_model_execution_version":"vllm_0_30_0","fa3_full_aot_group":group,
        "fa3_full_aot_layout":layout,"fa3_full_aot_cu_query":prefix,"fa3_full_aot_used_key":physical_k,
        "q_lens":q_lens,"kv_lens":kv_lens,"model_input_q_lens":physical_q,"model_input_kv_lens":physical_k,
        "graph_runtime_mode":"FULL","graph_model_input_tokens":token_shape.model_input_tokens,
        "fa3_full_aot_hole_validity":"uninitialized_copied_not_zero",
        "fa3_full_aot_head_shape":(heads, kv_heads, model.head_dim), "fa3_gqa_operand_dtype":model.compute_dtype}
    cu=trace.tensor("fa3_aot.cu_query",(b+1,),"int32",persistent=True)
    seq=trace.tensor("fa3_aot.used_key",(b,),"int32",persistent=True)
    packet=trace.tensor("fa3_aot.generated_packet",(n,),"int32")
    fields=tuple(trace.alias(packet,"fa3_aot.generated_field_"+str(i),shape=(b,) if i<len(offsets)-1 else (1,),
        strides_bytes=(4,),offset_bytes=offset*4) for i,offset in enumerate(offsets))
    shape={"batch":b,"q_len":1,"kv_len":max(kv_lens),"num_query_heads":heads,
        "num_kv_heads":kv_heads,"head_dim":model.head_dim}
    prepare=WorkloadNode("fa3_aot.prepare",WorkloadKind.VECTOR,pool,shape,"int32","fa3_prepare_varlen",trace.rank,
        {**attrs,"source_component":"fa3_full_aot_prepare","fa3_full_aot_component":"prepare"})
    trace.call(prepare,(cu,seq),shape=fields[0].shape,dtype="int32",out=fields[0],writes=fields[1:])
    storage=trace.tensor("fa3_aot.scheduler_storage",(capacity,),"int32",persistent=True)
    returned=storage.narrow(0,0,n)
    copied=WorkloadNode("fa3_aot.copy",WorkloadKind.MEMCPY,pool,{},None,None,trace.rank,
        {**attrs,"source_component":"fa3_full_aot_copy","fa3_full_aot_component":"copy",
            "algorithm":"stream_copy","direction":"d2d","link":"dram","bytes":n*4})
    trace.call(copied,(packet,),shape=returned.shape,dtype="int32",out=returned)
    tail=storage.narrow(0,n,capacity-n)
    clear=WorkloadNode("fa3_aot.clear",WorkloadKind.VECTOR,pool,{"elements":capacity-n},"int32","elementwise",trace.rank,
        {**attrs,"source_component":"fa3_full_aot_clear","fa3_full_aot_component":"clear", "operation":"fill", "reads":0,"writes":1,
            "mathematical_program":"vllm_0_30_fa3_aot_tail_clear"})
    trace.call(clear,(),shape=tail.shape,dtype="int32",out=tail)
    semaphore=trace.alias(storage,"fa3_aot.semaphore",shape=(1,),strides_bytes=(4,),offset_bytes=offsets[-1]*4)
    return (returned,semaphore,cu,seq,layout,trace.qualify(prepare.node_id),trace.qualify(copied.node_id),trace.qualify(clear.node_id))


def _dspark_source_op(trace, name, pool, operation, inputs, *, rows, width, source_rows, shape, dtype="bf16", attributes=None):
    from .traced_sampling import MTPSourceOps
    return MTPSourceOps(trace,name,pool,model_execution_version="vllm_0_30_0")._call(operation,inputs,
        rows=rows,width=width,source_rows=source_rows,dtype=dtype,shape=shape,output_dtype=dtype,attributes=attributes,
        out=trace.tensor(name+":out",shape,dtype) if not inputs else None)


def _dspark_regular_first_pre(trace, model, pool, hidden):
    """The draft's real 3D first input uses regular, not broadcast, mHC."""
    from .traced_operators import SymbolicOperator
    rows, streams, width = hidden.shape
    if streams!=4 or width!=model.hidden_size or hidden.dtype!="bf16":
        raise ValueError("DSpark regular first pre requires its repeated BF16 four-stream input")
    shape={"elements":rows*streams*width,"hidden":width,"hc_mult":streams,"n_splits":1,"dtype_bytes":2}
    op=SymbolicOperator(trace,"dspark.first-pre",pool)
    attrs={"source_component":"vllm_0_30_dspark_regular_mhc_pre"}
    projection=op.node(WorkloadKind.VECTOR,shape,"bf16","mhc_prenorm",attrs)
    mixes,sqrsum=trace.call_outputs(projection,(hidden,),outputs=(
        ("mixes",(1,rows,streams*(streams+2)),"fp32"),("sqrsum",(1,rows),"fp32")))
    final=op.node(WorkloadKind.VECTOR,{**shape,"sinkhorn_iters":model.hc_sinkhorn_iters},"bf16","mhc_pre_norm",attrs)
    post,res,x=trace.call_outputs(final,(hidden,mixes,sqrsum),outputs=(
        ("post_mix",(rows,streams,1),"fp32"),("res_mix",(rows,streams,streams),"fp32"),
        ("layer_input",(rows,width),"bf16")))
    return hidden,post,res,x


class DSparkNoncausalAttention:
    """Exact all-valid logical block window; native page/index ABI stays unbound."""
    def __init__(self, original, indices, lengths, *, q_lens, kv_lens):
        self.original,self.indices,self.lengths=original,indices,lengths
        self.q_lens,self.kv_lens=q_lens,kv_lens

    def __call__(self,q,positions,out,*,controls=()):
        from .traced_operators import SymbolicOperator
        old=self.original
        if old.ratio!=1 or any(q!=5 for q in self.q_lens):
            raise ValueError("DSpark draft attention only admits its noncompressed five-query source block")
        model=old.model
        op=SymbolicOperator(old.trace,old.name+":dspark",old.pool_name)
        node=op.node(WorkloadKind.EAGER_ATTN,{"batch":len(self.q_lens),"q_len":5,"kv_len":max(self.kv_lens),
            "num_query_heads":q.shape[1],"qk_head_dim":512,"v_head_dim":512,"topk":0,"window_size":128,
            "compress_ratio":1,"compressed_state_bytes":584,"swa_token_bytes":584,"dtype_bytes":2},"bf16","dsv4_sparse_decode",{
                "source_component":"vllm_0_30_symbolic_v4_sparse","causal":False,"indexer_type":"shared",
                "kv_layout":"mla","kv_dist":"uniform","q_lens":self.q_lens,"kv_lens":self.kv_lens,
                "selected_kv_lens":(0,)*len(self.q_lens),"topk_capacity":model.index_topk,"topk_index_dtype":"int32",
                "swa_kv_lens":tuple(5+min(128,kv-5) for kv in self.kv_lens),"compressed_selection":"candidate_upper_bound",
                "mathematical_program":"vllm_0_30_dspark_swa_noncausal_softmax_v1","num_speculative_tokens":5,
                "physical_indices_bound":False,"logical_indices_source":"declared_all_valid_logical_arena"})
        return old.trace.call(node,(q,old.swa_cache,self.indices,self.lengths),shape=out.shape,dtype="bf16",out=out,
            controls=(*controls,positions))


def trace_deepseek_v4_dspark_first_proposal(target,draft,step,pool,hardware,expert_routing,*,input_ids,
        target_positions,target_slot_mapping,anchor_token_ids,draft_token_ids,source_kind,source_id,
        max_model_len=128,max_batched_tokens=128,verification=None):
    """Full target and three-layer draft first proposal, explicit source scenario.

    This first request has no rejected prefix yet. Proposal values are caller
    assumptions for source ownership, never numerically generated checkpoint
    outputs. Later target verification/cross-step state remain separate work.
    """
    from src.serving.schedule.schedule_step import schedule_step_sha256
    from .model_config import validate_named_target_identity
    from .runtime_inputs import vllm_0_30_source_model_inputs
    from .traced_operators import SymbolicOperator
    from .traced_sampling import MTPSourceOps
    validate_named_target_identity(target)
    config=pool.parallel_plan.config
    if (target.model_type!="deepseek_v4" or draft.dspark_block_size!=5
            or draft.model_id!=target.model_id+"-DSpark" or draft.num_hidden_layers!=target.num_hidden_layers
            or any(getattr(target,key)!=getattr(draft,key) for key in ("hidden_size","num_attention_heads","head_dim",
                "vocab_size","hc_mult","window_size","compute_dtype","weight_dtype"))
            or draft.compress_ratios[draft.num_hidden_layers:]!=(0,0,0)
            or pool.hardware_id!="nvidia_h200" or config.tensor_parallel_size!=8 or not config.is_rank_local
            or any(value!=1 for value in (config.pipeline_parallel_size,config.sequence_parallel_size,
                config.context_parallel_size,config.data_parallel_size))
            or config.enable_expert_parallel or source_kind not in ("declared_synthetic","explicit_assumption")
            or not isinstance(source_id,str) or not source_id):
        raise ValueError("DSpark first proposal requires exact paired official V4 target/draft and declared H200 TP8 source")
    requests=step.requests
    if verification is None:
        if (len(requests)!=1 or requests[0].phase!="prefill" or requests[0].computed_tokens!=0
                or requests[0].raw_model_output_tokens!=1 or requests[0].mtp_verified_tokens!=0
                or requests[0].mtp_proposer_tokens!=5 or requests[0].mtp_proposer_mode!="padded"):
            raise NotImplementedError("DSpark first source request requires one fresh target prefill plus five-token proposal")
        retained = step.token_count
        context_valid = step.token_count
    else:
        from .runtime_inputs import DSparkFirstProposalInputs
        from src.serving.schedule.schedule_step import BackendStep
        if not isinstance(verification, DSparkFirstProposalInputs) or not verification.is_verification:
            raise ValueError("DSpark verification requires typed complete source facts")
        BackendStep(step, generated_tokens={request.request_id: (request.raw_model_output_tokens if verification.delivered_token_count is None else verification.delivered_token_count) for request in step.requests},
            embedding_token_ids=input_ids, embedding_token_source="vllm_explicit_request_token_ids",
            model_source_inputs=verification)
        if (verification.draft_model!=draft or verification.target_positions!=target_positions
                or verification.target_slot_mapping!=target_slot_mapping or verification.anchor_token_ids!=anchor_token_ids
                or verification.draft_token_ids!=draft_token_ids or verification.source_id!=source_id):
            raise ValueError("DSpark verification trace arguments disagree with typed source state")
        retained = verification.retained_kv_tokens
        context_valid = verification.accepted_drafts + 1
    rows=step.token_count
    if (not isinstance(input_ids,tuple) or len(input_ids)!=rows or any(type(v)is not int or not 0<=v<target.vocab_size for v in input_ids)
            or type(target_positions)is not tuple or target_positions!=tuple(range(requests[0].computed_tokens,requests[0].computed_tokens+rows))
            or type(target_slot_mapping)is not tuple or target_slot_mapping!=target_positions
            or any(type(value)is not int for values in (target_positions,target_slot_mapping) for value in values)
            or not isinstance(anchor_token_ids,tuple) or len(anchor_token_ids)!=1
            or not isinstance(draft_token_ids,tuple) or len(draft_token_ids)!=5
            or any(not isinstance(ids,tuple) or len(ids)!=1 for ids in draft_token_ids)
            or any(type(v)is not int or not 0<=v<target.vocab_size for ids in (anchor_token_ids,*draft_token_ids) for v in ids)
            or type(max_model_len)is not int or max(requests[0].computed_tokens+rows,retained+5)>max_model_len or max_model_len>target.max_position_embeddings
            or type(max_batched_tokens)is not int or max(rows,5)>max_batched_tokens):
        raise ValueError("DSpark source requires explicit token transcripts and constructor limits")
    q_lens,kv_lens=(rows,),(requests[0].computed_tokens+rows,)
    draft_q,draft_k=(5,),(retained+5,)
    step, input_ids, model_input_order = vllm_0_30_source_model_inputs(step,
        embedding_token_ids=input_ids,prompt_lengths=None,num_speculative_tokens=5,decode_threshold=11)
    capacity=-(-max(kv_lens[0],draft_k[0],128)//128)*128
    traces=tuple(ForwardTrace(rank=rank,namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group=ForwardCollectiveGroup(traces,namespace=step.step_id)
    collective,gather=pool.collective_for("all_reduce"),pool.collective_for("all_gather")
    digest=schedule_step_sha256(step)
    def resolution(model,layer,tokens,invocation):
        return expert_routing.resolve(step_id=step.step_id,layer_index=layer,moe_invocation_id=invocation,
            schedule_step_sha256=digest,tokens=tokens,num_experts=model.num_experts,topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256,pool_name=pool.name,tensor_parallel_size=8,expert_parallel_size=1,
            expert_rank_group=(pool.ranks[0],),expert_placement_strategy="linear",moe_layer_ordinal=model.moe_layer_ordinal(layer))
    target_invocations=tuple(f"{step.step_id}:layer-{i}:moe" for i in range(target.num_hidden_layers))
    target_resolutions=tuple(resolution(target,i,rows,target_invocations[i]) for i in range(target.num_hidden_layers))
    draft_invocations=tuple(f"{step.step_id}:layer-{draft.num_hidden_layers+i}:moe" for i in range(3))
    draft_resolutions=tuple(resolution(draft,draft.num_hidden_layers+i,5,draft_invocations[i]) for i in range(3))
    # All positions are declared valid logical slots. Native page tables and
    # cache addresses are not inferred from this source-owned logical arena.
    for trace in traces:
        slots=trace.tensor("target.slots",(rows,),"int64")
        network=DeepseekV4Model(trace,target,pool.name,group,collective,token_ids=input_ids,
            q_lens=q_lens,kv_lens=kv_lens,request_phases=tuple(model_input_order["model_attention_phases"]),slot_mapping=target_slot_mapping,
            slot_tensor=slots,cache_capacity_tokens=capacity,source_max_model_len=max_model_len,
            max_batched_tokens=max_batched_tokens,linear_source_profile="vllm_0_30_row_major_group128_fp32",
            source_use_deep_gemm=False,hardware=hardware,resolutions=target_resolutions,invocation_ids=target_invocations,
            model_input_order=model_input_order,indexer_selection_source="declared_causal_topk")
        shared_embedding=trace.parameter_view("shared.embedding.weight",(-(-target.vocab_size//64)*64//8,target.hidden_size),"bf16")
        network.embed_tokens.shared_weight=shared_embedding
        ids=trace.tensor("target.input_ids",(rows,),"int32")
        pos=trace.tensor("target.positions",(rows,),"int64")
        hidden,aux=network(ids,pos,aux_layer_ids=draft.dspark_target_layer_ids)
        if len(aux)!=3:
            raise ValueError("DSpark requires all three actual target auxiliary outputs")
        ops=MTPSourceOps(trace,"dspark.ops",pool.name,model_execution_version="vllm_0_30_0")
        head=SymbolicTPLogitsProcessor(trace,"lm-head",pool.name,hidden_size=target.hidden_size,
            vocab_size=target.vocab_size,group=group,config=gather)
        if verification is None:
            chosen=SymbolicLogitsRowSelect(trace,"target.logits-select",pool.name)(hidden,
                trace.tensor("target.logits_indices",(1,),"int32"),selected_indices=(rows-1,))
            # V2 all-greedy keeps BF16 logits, just like verification.
            bonus=ops.unary(ops.unary(head(chosen),"argmax"),"cast_ids")
        else:
            # Pure greedy apply_sampling_params returns BF16 logits unchanged.
            # All five local-stat rows are examined; row six is consumed only
            # by the all-accepted resample path. No global FP32 cast buffer.
            logits=head(hidden)
            target_ids=ops.unary(logits.narrow(0,0,5),"argmax")
            target_bonus=(ops.unary(logits.narrow(0,5,1),"argmax")
                if verification.accepted_drafts==5 else None)
            previous_ids=ids.narrow(0,1,5)
            active_ids=target_ids.narrow(0,0,min(context_valid,5))
            verify_inputs=(previous_ids,active_ids) if target_bonus is None else (previous_ids,active_ids,target_bonus)
            sampled=trace.tensor("dspark.verification.sampled",(1,6),"int64")
            verify_node=ops._node("dspark_greedy_verify",rows=1,width=6,source_rows=6,dtype="int32",
                attributes={"num_speculative_tokens":5,"accepted_drafts":verification.accepted_drafts})
            # Only the valid prefix view is written. Tail storage remains
            # undefined; no consumer may read it or assume a sentinel.
            count=trace.tensor("dspark.verification.num_sampled",(1,),"int32")
            rejected=trace.tensor("dspark.verification.num_rejected",(1,),"int32")
            active=trace.call(verify_node,verify_inputs,shape=(1,context_valid),dtype="int64",
                out=sampled.narrow(1,0,context_valid),writes=(count,rejected))
            bonus=ops.unary(active.narrow(1,context_valid-1,1).view_as_shape(1),"cast_ids")
        aux_concat=SymbolicConcat(trace,"dspark.aux-concat",pool.name)(*aux)
        projected,_=SymbolicBlockFP8Linear(trace,"dspark.main-proj",pool.name,3*target.hidden_size,target.hidden_size,
            source_profile="vllm_0_30_row_major_group128_fp32")(aux_concat)
        main_x=SymbolicRMSNorm(trace,"dspark.main-norm",pool.name,target.hidden_size,target.target_semantics.rms_norm_eps)(projected)
        ops=MTPSourceOps(trace,"dspark.ops",pool.name,model_execution_version="vllm_0_30_0")
        persistent=trace.tensor("dspark.hidden_buffer",(max_batched_tokens,target.hidden_size),"bf16",persistent=True)
        ops.copy(main_x,out=persistent.narrow(0,0,rows))
        main_x=persistent.narrow(0,0,rows)
        query_ids=trace.tensor("dspark.query_ids",(5,),"int32")
        # Anchor is the actual target sampler output; noise is source constant.
        ops.copy(bonus.view_as_shape(1),out=query_ids.narrow(0,0,1))
        ops.fill(query_ids.narrow(0,1,4),value=draft.dspark_noise_token_id)
        if verification is None:
            query_pos=trace.tensor("dspark.query_positions",(5,),"int64")
            context_pos=ops.copy(pos)
            context_slots=ops.copy(slots)
            for i in range(5):
                value=ops.unary(pos.narrow(0,rows-1,1),expression="add",scalar=i+1)
                ops.copy(value,out=query_pos.narrow(0,i,1))
            query_slots=trace.tensor("dspark.query_slots",(5,),"int64")
            ops.copy(query_pos,out=query_slots)
        else:
            # The actual verify counters feed the next prepare writer.
            # Candidates describe the caller's explicit contiguous logical
            # arena. Native block-table translation/null blocks stay unbound.
            logical_query_slots=trace.tensor("dspark.logical_query_slot_candidates",(5,),"int64")
            prepare=ops._node("dspark_roundtrip_prepare",rows=1,width=6,source_rows=6,dtype="int64",
                attributes={"num_speculative_tokens":5,"accepted_drafts":verification.accepted_drafts,
                    "declared_context_slot_candidates":target_slot_mapping,
                    "declared_query_slot_candidates":tuple(range(retained,retained+5)),
                    "physical_slot_candidates_bound":False})
            context_pos,context_slots,query_pos,query_slots,seq=trace.call_outputs(prepare,
                (pos,slots,logical_query_slots,count,rejected),outputs=(
                    ("context_positions",(6,),"int64"),("context_slots",(6,),"int64"),
                    ("query_positions",(5,),"int64"),("query_slots",(5,),"int64"),("seq_lens",(1,),"int32")))
        rope=trace.tensor("dspark.rope",(draft.max_position_embeddings,draft.qk_rope_head_dim),"fp32",persistent=True)
        topk=trace.tensor("dspark.topk-buffer",(max_batched_tokens,draft.index_topk),"int32",persistent=True)
        indices_op=SymbolicOperator(trace,"dspark.noncausal_indices",pool.name)
        index_node=indices_op.node(WorkloadKind.VECTOR,{"elements":5*192,"rows":5,"width":192,"source_rows":1},
            "int32","vllm_mtp_eager",{"source_component":"vllm_0_30_symbolic_mtp_eager","operation":"dspark_swa_indices",
                "model_execution_version":"vllm_0_30_0","num_speculative_tokens":5,"kv_lens":draft_k,
                "physical_indices_bound":False,"all_query_rows_valid":True})
        cuq=trace.tensor("dspark.cuQ",(2,),"int32")
        ops.fill(cuq.narrow(0,0,1),value=0)
        ops.fill(cuq.narrow(0,1,1),value=5)
        if verification is None:
            seq=ops.fill(trace.tensor("dspark.seq_lens",(1,),"int32"),value=rows+5)
        mapping=ops.fill(trace.tensor("dspark.token_to_request",(5,),"int32"),value=0)
        valid=ops.unary(query_slots,"compare",expression="greater_equal",scalar=0)
        inputs=(cuq,seq,mapping,valid,trace.tensor("dspark.logical_block_table",(1,capacity//128),"int32"))
        indices,lengths=trace.call_outputs(index_node,inputs,outputs=(("indices",(5,1,192),"int32"),("lengths",(5,),"int32")))
        layers=[]
        for i in range(3):
            prefix="dspark.layer-"+str(i)
            layer=DeepseekV4DecoderLayer(trace,draft,pool.name,prefix,group,collective,rope,topk,
                layer_index=draft.num_hidden_layers+i,q_lens=draft_q,kv_lens=draft_k,request_phases=("decode",),
                slot_mapping=tuple(range(retained,retained+5)),slot_tensor=query_slots,cache_capacity_tokens=capacity,
                source_max_model_len=max_model_len,linear_source_profile="vllm_0_30_row_major_group128_fp32",
                source_use_deep_gemm=False,hardware=hardware,resolution=draft_resolutions[i],invocation_id=draft_invocations[i])
            layer.attn.attn=DSparkNoncausalAttention(layer.attn.attn,indices,lengths,q_lens=draft_q,kv_lens=draft_k)
            # This SAME main_x drives every layer's full fused projection.
            fused,_=layer.attn.inputs.fused_wqa_wkv(main_x)
            kv=fused.narrow(1,draft.q_lora_rank,draft.head_dim)
            kv=SymbolicRMSNorm(trace,prefix+":attn:kv-norm",pool.name,draft.head_dim,draft.target_semantics.rms_norm_eps)(kv)
            dummy=_dspark_source_op(trace,prefix+":context_dummy_q",pool.name,"dspark_zero_query",(),
                rows=rows,width=draft.num_attention_heads//8*512,source_rows=rows,
                shape=(rows,draft.num_attention_heads//8,512))
            if verification is None:
                insert=SymbolicV4KVInsert(trace,prefix+":context_insert",pool.name,draft,tp_size=8,
                    cache=layer.attn.swa_cache,slot_mapping=target_slot_mapping,slot_tensor=context_slots)
            else:
                from .traced_operators import SymbolicDSparkPartialContextInsert
                insert=SymbolicDSparkPartialContextInsert(trace,prefix+":context_insert",pool.name,draft,tp_size=8,
                    cache=layer.attn.swa_cache,slot_mapping=target_slot_mapping[:context_valid]+(-1,)*(rows-context_valid),
                    slot_tensor=context_slots,valid_token_count=context_valid)
            insert(dummy,kv,context_pos,rope)
            layers.append(layer)
        embedding=SymbolicVocabParallelEmbedding(trace,"embedding",pool.name,draft.hidden_size,draft.vocab_size,
            group,collective,token_ids=(anchor_token_ids[0],)+(draft.dspark_noise_token_id,)*4)
        embedding.shared_weight=shared_embedding
        query_hidden=embedding(query_ids)
        expanded=_dspark_source_op(trace,"dspark.repeat",pool.name,"dspark_repeat_hc",(query_hidden,),
            rows=20,width=draft.hidden_size,source_rows=5,shape=(5,4,draft.hidden_size))
        residual,post_mix,res_mix,x=_dspark_regular_first_pre(trace,draft,pool.name,expanded)
        x=layers[0].attn(query_pos,x)
        residual,post_mix,res_mix,x=layers[0].ffn_pre(x,residual,post_mix,res_mix)
        x=layers[0].ffn(x,query_ids)
        for layer in layers[1:]:
            x,residual,post_mix,res_mix=layer(x,query_pos,query_ids,post_mix,res_mix,residual)
        x=SymbolicMHC(trace,"dspark.final-post",pool.name,draft,hardware,variant="mhc_post",source_use_deep_gemm=False)(x,residual,post_mix,res_mix)
        head_hidden=SymbolicMHC(trace,"dspark.hc-head",pool.name,draft,hardware,variant="hc_head",source_use_deep_gemm=False)(x)
        logits=head(SymbolicRMSNorm(trace,"dspark.final-norm",pool.name,draft.hidden_size,draft.target_semantics.rms_norm_eps)(head_hidden))
        base=logits.view_as_shape(1,5,draft.vocab_size)
        # DSparkMarkovHead.disable_tp=True: every rank owns the full W2;
        # there is no vocabulary gather in any of the five sequential steps.
        markov_head=SymbolicLinear(trace,"dspark.markov-w2",pool.name,draft.dspark_markov_rank,
            draft.vocab_size,kind=WorkloadKind.LM_HEAD)
        previous=query_ids.narrow(0,0,1)
        sampled=[]
        weight=trace.parameter_view("dspark.markov-w1.weight",(draft.vocab_size,draft.dspark_markov_rank),"bf16")
        for i in range(5):
            op=SymbolicOperator(trace,"dspark.markov-w1",pool.name)
            tokens=anchor_token_ids if i==0 else draft_token_ids[i-1]
            node=op.node(WorkloadKind.VECTOR,{"elements":draft.dspark_markov_rank,"rows":1,
                "width":draft.dspark_markov_rank,"source_rows":draft.vocab_size},"bf16","vllm_mtp_eager",{
                    "source_component":"vllm_0_30_symbolic_mtp_eager","operation":"gather",
                    "model_execution_version":"vllm_0_30_0","index_bytes":4,"selected_indices":tokens})
            markov=trace.call(node,(previous,),shape=(1,draft.dspark_markov_rank),dtype="bf16",parameters=(weight,))
            bias,_=markov_head(markov)
            corrected=SymbolicAdd(trace,"dspark.markov-add",pool.name)(base.narrow(1,i,1).view_as_shape(1,draft.vocab_size),bias)
            previous=ops.unary(ops.unary(corrected,"argmax"),"cast_ids")
            sampled.append(previous)
        ops.copy(ops.concat(tuple(sampled)),out=trace.tensor("dspark.proposals",(5,),"int32",persistent=True))
    return group


# Record the complete import-time source after every helper is defined. Both
# construction and sealing belong here: sealing alone would omit later helper
# code/default identities from the function contract.
_DENSE_METADATA_SOURCE_CONTRACT = _DenseMetadataSourceGuard()
_DENSE_METADATA_SOURCE_CONTRACT.seal()
