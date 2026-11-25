"""SGLang0.5.19 eager DSV3 target frames on the shared virtual operators.

The unfused and FlashInfer-fused communication profiles are separate source
programs. Explicit topk1/steps1 or steps2 state covers declared eager frames,
without establishing an actual runtime. No weights, tensors or GPU execute.
Linear/MLA/expert arithmetic reuses the existing mathematical Kernel owner;
DeepGEMM installation, native traffic and normal capacity remain unbound.
"""

from dataclasses import dataclass, replace
from math import prod

from .traced_forward import ForwardCollectiveGroup, ForwardTrace
from .traced_models import DeepseekV2MLP, MLAFreshPrefillBackend, _source_expert_resolutions, _source_linear
from .traced_deepseek_v3 import DeepseekV3Rotary
from .traced_operators import (
    SymbolicBatchedLinear, SymbolicConcat, SymbolicLinear, SymbolicMLACacheUpdate,
    SymbolicMLALatentDecode, SymbolicOperator, SymbolicRMSNorm, SymbolicAttentionStateMerge,
)
from .workload_graph import WorkloadKind

SOURCE_PROFILE = "sglang_0_5_19_deepseek_v3_decode_fp8"
SOURCE_FUSED_PROFILE = "sglang_0_5_19_deepseek_v3_decode_fp8_flashinfer_ar_norm"
SOURCE_TARGET_PROFILE = "sglang_0_5_19_deepseek_v3_target_fp8"
SOURCE_TARGET_FUSED_PROFILE = "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm"
PREFILL_SOURCE = "fresh_mha_bf16_ragged_eager_cp1_no_hook"
PREFILL_PREFIX_SOURCE = "ragged_eager_cp1_no_hook_prefix_threshold8192"
SAMPLING_SOURCE = "sglang_0_5_19_plain_greedy_fp32"
ROUTING_PROGRAM = "sglang_0_5_19_fp32_sigmoid_bias_grouped_topk"
ROUTING_MATH = "fp32_sigmoid_bias_grouped_topk_flops_bytes_v1"


@dataclass(frozen=True)
class SGLangGraphInputs:
    """Declared retained graph tail and dummy-cache applicability, not observation.

    The registry copies only active input IDs, while positions/slots reset
    their tails to zero and FlashMLA's padded sequence length is one. Dummy
    slot0 writes are allowed only when it does not alias real request history.
    """
    request_ids: tuple[str, ...]
    source_kind: str
    source_id: str
    retained_padding_token_ids: tuple[int, ...]
    dummy_cache_slot_zero_disjoint_from_active_history: bool

    def __post_init__(self):
        if (not isinstance(self.request_ids, tuple) or not self.request_ids
                or any(not isinstance(value, str) or not value for value in self.request_ids)
                or len(set(self.request_ids)) != len(self.request_ids)
                or self.source_kind not in ("declared_synthetic", "explicit_assumption")
                or not isinstance(self.source_id, str) or not self.source_id):
            raise ValueError("SGLang graph inputs require ordered request identities and explicit source declaration")
        if (not isinstance(self.retained_padding_token_ids, tuple)
                or any(type(value) is not int or not 0 <= value < 2**31 for value in self.retained_padding_token_ids)):
            raise ValueError("SGLang graph retained padding requires explicit token-ID metadata")
        if self.dummy_cache_slot_zero_disjoint_from_active_history is not True:
            raise NotImplementedError("SGLang graph dummy cache slot0 must be declared disjoint from active request history")


@dataclass(frozen=True)
class SGLangPrefillInputs:
    """Declared one-shot or chunked index metadata, never a provider receipt."""
    request_ids: tuple[str, ...]
    source_kind: str
    source_id: str
    one_shot_kv_indices: tuple[int, ...]
    # sglang/srt/environ.py SGLANG_MAX_KV_CHUNK_CAPACITY default.
    max_kv_chunk_capacity: int = 131072
    prefix_chunk_kv_indices: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        if (not isinstance(self.request_ids, tuple) or not self.request_ids
                or any(not isinstance(identifier, str) or not identifier.strip() for identifier in self.request_ids)
                or len(set(self.request_ids)) != len(self.request_ids)
                or self.source_kind not in ("declared_synthetic", "explicit_assumption")
                or not isinstance(self.source_id, str) or not self.source_id.strip()):
            raise ValueError("SGLang prefill requires named requests and an explicit declared metadata source")
        if (not isinstance(self.one_shot_kv_indices, tuple)
                or any(type(index) is not int or not 0 <= index < 2**31 for index in self.one_shot_kv_indices)
                or type(self.max_kv_chunk_capacity) is not int or self.max_kv_chunk_capacity <= 0
                or len(self.one_shot_kv_indices) > self.max_kv_chunk_capacity):
            raise ValueError("SGLang one-shot metadata requires INT32 cache indices within the declared chunk capacity")
        if (not isinstance(self.prefix_chunk_kv_indices, tuple)
                or any(not isinstance(chunk, tuple) or not chunk or len(chunk) > self.max_kv_chunk_capacity
                    or any(type(index) is not int or not 0 <= index < 2**31 for index in chunk)
                    for chunk in self.prefix_chunk_kv_indices)
                or bool(self.one_shot_kv_indices) == bool(self.prefix_chunk_kv_indices)):
            raise ValueError("SGLang prefill requires exactly one explicit one-shot or chunked index program")


@dataclass(frozen=True)
class SGLangEagleChainState:
    """Explicit batched EAGLE topk1/steps1 or steps2, never observed output.

    NextN uses the same official config/revision and its FP8 recipe. No tree,
    draft IDs or acceptance is inferred from a schedule length. bonus_token_ids
    is the target sample (discarded for non-final prefill chunks) or verify
    bonus; draft_top1_ids is its next draft candidate. Prefix prefill declares
    either a final suffix or one non-final request's next prompt token.
    """
    request_ids: tuple[str, ...]
    phase: str
    source_kind: str
    source_id: str
    bonus_token_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]
    target_argmax_ids: tuple[int, ...]
    draft_top1_ids: tuple[int, ...]
    retrieve_index: tuple[tuple[int, int], ...]
    retrieve_next_token: tuple[tuple[int, int], ...]
    retrieve_next_sibling: tuple[tuple[int, int], ...]
    accept_index: tuple[tuple[int, int], ...]
    accept_lens: tuple[int, ...]
    draft_slot_mapping: tuple[int, ...]
    draft_cache_capacity_tokens: int
    input_bonus_dtype: str
    proposal_depth: int = 1
    draft_decode_slot_mapping: tuple[int, ...] = ()
    prefill_tail_source: str | None = None
    prefill_next_prompt_token_ids: tuple[int | None, ...] | None = None
    target_graph_inputs: SGLangGraphInputs | None = None
    extend_graph_inputs: SGLangGraphInputs | None = None

    def __post_init__(self):
        if (self.target_graph_inputs is None) != (self.extend_graph_inputs is None):
            raise ValueError("EAGLE padded graphs require both target and extend retained input states")
        if self.target_graph_inputs is not None:
            if (self.phase != "verify" or self.proposal_depth != 2
                    or any(not isinstance(value, SGLangGraphInputs) or value.request_ids != self.request_ids
                        for value in (self.target_graph_inputs, self.extend_graph_inputs))
                    or len(self.target_graph_inputs.retained_padding_token_ids) != len(self.extend_graph_inputs.retained_padding_token_ids)
                    or not self.target_graph_inputs.retained_padding_token_ids
                    or len(self.target_graph_inputs.retained_padding_token_ids) % 3):
                raise ValueError("EAGLE padded source states require matching verify requests and complete three-token dummy windows")
        if self.prefill_tail_source not in (None, "final_suffix_no_prompt_override", "nonfinal_chunk_next_prompt_token"):
            raise ValueError("SGLang EAGLE prefill tail requires an explicit supported source or None")
        if self.phase == "verify" and self.prefill_tail_source is not None:
            raise ValueError("SGLang EAGLE verify cannot declare a prefill tail source")
        if self.input_bonus_dtype not in ("int32", "int64"):
            raise ValueError("SGLang EAGLE bonus carry requires its explicit INT32/INT64 source dtype")
        if (not isinstance(self.request_ids, tuple) or not self.request_ids
                or any(not isinstance(value, str) or not value for value in self.request_ids)
                or len(set(self.request_ids)) != len(self.request_ids)
                or self.phase not in ("prefill", "verify")
                or self.source_kind not in ("declared_synthetic", "explicit_assumption")
                or not isinstance(self.source_id, str) or not self.source_id.strip()):
            raise ValueError("SGLang EAGLE requires named requests and an explicit declared source scenario")
        if type(self.proposal_depth) is not int or self.proposal_depth not in (1, 2):
            raise NotImplementedError("SGLang chain source covers only the existing one/two-step proposal modes")
        batch, width = len(self.request_ids), self.proposal_depth + 1
        if self.prefill_tail_source == "nonfinal_chunk_next_prompt_token":
            values = self.prefill_next_prompt_token_ids
            if (not isinstance(values, tuple) or len(values) != batch
                    or sum(value is not None for value in values) != 1
                    or any(value is not None and (type(value) is not int or value < 0) for value in values)):
                raise ValueError("SGLang non-final prefill requires exactly one declared next prompt token")
        elif self.prefill_next_prompt_token_ids is not None:
            raise ValueError("SGLang next prompt tokens require their non-final prefill tail source")
        for name in ("bonus_token_ids", "draft_token_ids", "target_argmax_ids", "draft_top1_ids", "draft_slot_mapping", "draft_decode_slot_mapping"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(type(value) is not int or value < 0 for value in values):
                raise ValueError("SGLang EAGLE token/slot scenarios require explicit nonnegative integer tuples")
        if (len(self.bonus_token_ids) != batch or len(self.draft_top1_ids) != batch
                or type(self.draft_cache_capacity_tokens) is not int or self.draft_cache_capacity_tokens <= 0
                or not self.draft_slot_mapping or len(set(self.draft_slot_mapping)) != len(self.draft_slot_mapping)
                or any(slot >= self.draft_cache_capacity_tokens for slot in self.draft_slot_mapping)):
            raise ValueError("SGLang EAGLE requires complete bonus/proposal IDs and distinct bounded draft slots")
        for name in ("retrieve_index", "retrieve_next_token", "retrieve_next_sibling", "accept_index"):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or any(not isinstance(row, tuple) or len(row) != width
                    or any(type(value) is not int for value in row) for row in values)):
                raise ValueError("SGLang EAGLE tree rows require exact integers and declared chain width")
        if not isinstance(self.accept_lens, tuple) or any(type(value) is not int for value in self.accept_lens):
            raise ValueError("SGLang EAGLE accepted lengths require exact integers")
        if self.phase == "prefill":
            if (self.input_bonus_dtype != "int64" or len(self.target_argmax_ids) != batch or self.bonus_token_ids != self.target_argmax_ids
                    or any(getattr(self, name) != () for name in ("draft_token_ids", "retrieve_index",
                        "retrieve_next_token", "retrieve_next_sibling", "accept_index", "accept_lens", "draft_decode_slot_mapping"))):
                raise ValueError("SGLang EAGLE prefill has one bonus per request and no verify tree or acceptance path")
        else:
            if (len(self.target_argmax_ids) != batch * width or len(self.draft_token_ids) != batch * self.proposal_depth
                    or self.retrieve_index != tuple(tuple(range(row * width, (row + 1) * width)) for row in range(batch))
                    or self.retrieve_next_token != (tuple(range(1, width)) + (-1,),) * batch
                    or self.retrieve_next_sibling != ((-1,) * width,) * batch
                    or len(self.accept_lens) != batch or len(self.accept_index) != batch
                    or len(self.draft_decode_slot_mapping) != batch * (self.proposal_depth - 1)
                    or len(set(self.draft_decode_slot_mapping)) != len(self.draft_decode_slot_mapping)
                    or any(slot >= self.draft_cache_capacity_tokens for slot in self.draft_decode_slot_mapping)):
                raise ValueError("SGLang EAGLE verify requires complete explicit per-request chains and draft-decode slots")
            for row in range(batch):
                accepted = 0
                for column in range(self.proposal_depth):
                    if self.target_argmax_ids[row * width + column] != self.draft_token_ids[row * self.proposal_depth + column]:
                        break
                    accepted += 1
                path = tuple(row * width + column if column <= accepted else -1 for column in range(width))
                if (self.accept_lens[row] != 1 + accepted or self.accept_index[row] != path
                        or self.bonus_token_ids[row] != self.target_argmax_ids[row * width + accepted]):
                    raise ValueError("SGLang EAGLE acceptance/bonus contradicts the explicitly declared target/draft IDs")


def resolve_eagle_capture_batch(state, capture_sizes):
    """Keep public token-row units and select a shared paired B/3B bucket."""
    batch = len(state.request_ids)
    candidates = tuple(size for size in capture_sizes if size >= batch and 3 * size in capture_sizes)
    if not candidates:
        raise NotImplementedError("SGLang EAGLE graph requires paired token-row buckets for draft B and verify/extend3B")
    selected = min(candidates)
    if selected > batch:
        if state.target_graph_inputs is None or state.extend_graph_inputs is None:
            raise NotImplementedError("SGLang EAGLE padding requires explicit target and extend retained input states")
        if len(state.target_graph_inputs.retained_padding_token_ids) != 3 * (selected - batch):
            raise ValueError("EAGLE retained graph tails disagree with the selected paired token-row buckets")
    elif state.target_graph_inputs is not None:
        raise ValueError("EAGLE exact bucket cannot carry retained padding metadata")
    return selected


class SGLangRMSNorm(SymbolicRMSNorm):
    """sgl_kernel RMSNorm/add-RMSNorm have the same public in-place ports."""
    def node(self, *args, **kwargs):
        node = super().node(*args, **kwargs)
        return replace(node, attributes={**node.attributes,
            "source_storage_profile": "sglang_0_5_19_sgl_kernel"})


class SGLangAllReduceRMSNorm:
    """Existing FlashInfer source wrapper: two fresh outputs, no input writes."""
    def __init__(self, trace, name, pool, width, epsilon, group, collective):
        self.name, self.pool, self.epsilon = name, pool, epsilon
        self.group, self.collective = group, collective
        self.weight = trace.tensor(name + ":weight", (width,), "bf16", persistent=True)

    def __call__(self, hidden, residual):
        return self.group.call(self.name, hidden, pool_name=self.pool, config=self.collective, inplace=False,
            implementation="sglang_flashinfer_allreduce_residual_rmsnorm", residual=residual,
            rmsnorm_weight=self.weight, rmsnorm_epsilon=self.epsilon)


class SGLangEmbedding(SymbolicOperator):
    """The CUDA fused Triton embedding branch, with actual INT64 IDs."""
    def __init__(self, trace, name, pool, model, group, collective, token_ids):
        super().__init__(trace, name, pool)
        self.model, self.group, self.collective, self.ids = model, group, collective, token_ids

    def __call__(self, ids):
        rows = len(self.ids)
        node = self.node(WorkloadKind.GATHER, {"elements": rows * self.model.hidden_size,
            "rows": rows, "hidden": self.model.hidden_size}, "bf16", "embedding", {
                "source_component": "sglang_0_5_19_symbolic_vocab_embedding",
                "embedding_source": "sglang_0_5_19_triton_unquantized_tp",
                "tensor_parallel_size": len(self.group.traces), "vocab_size": self.model.vocab_size,
                "embedding_token_ids": self.ids, "index_bytes": 8, "index_count": rows,
                "gather_factor": 1, "scattered": True, "materialized_output": True})
        partial = self.trace.call(node, (ids,), shape=(rows, self.model.hidden_size), dtype="bf16")
        return self.group.call(self.name + ":all-reduce", partial, pool_name=self.pool_name,
            config=self.collective, inplace=False, implementation="sglang_custom_all_reduce_v2")


class SGLangAbsorbedPrefill(SymbolicOperator):
    """FlashInferMLA parent EXTEND returns one fresh latent O, no public LSE.

    Source method dispatch uses the whole batch's declared prefix sum below
    the8192 default threshold. It is not FlashMLA's uniform fixed-q decode;
    query lengths can differ and exceed128. Physical pages/provider unbound.
    """
    def __init__(self, trace, name, pool, *, heads, q_lens, kv_lens, cache):
        super().__init__(trace, name, pool)
        self.heads, self.q_lens, self.kv_lens, self.cache = heads, q_lens, kv_lens, cache
        if (len(q_lens) != len(kv_lens) or any(q <= 0 or k < q for q, k in zip(q_lens, kv_lens, strict=True))
                or not 0 < sum(k - q for q, k in zip(q_lens, kv_lens, strict=True)) < 8192):
            raise ValueError("SGLang absorbed prefill requires a nonempty batch prefix below its declared8192 threshold")

    def __call__(self, query):
        rows = sum(self.q_lens)
        if query.dtype != "bf16" or query.shape != (rows, self.heads, 576):
            raise ValueError("SGLang absorbed prefill requires every BF16 latent/RoPE query row")
        node = self.node(WorkloadKind.FLASH_ATTN, {"batch": len(self.q_lens), "q_len": max(self.q_lens),
            "kv_len": max(self.kv_lens), "num_query_heads": self.heads,
            "qk_head_dim": 576, "v_head_dim": 512, "kv_lora_rank": 512}, "bf16", "mla_absorbed_prefill", {
                "source_component": "sglang_0_5_19_symbolic_absorbed_prefill", "causal": True,
                "kv_layout": "mla", "kv_dist": "uniform", "q_lens": self.q_lens, "kv_lens": self.kv_lens,
                "full_cuda_graphs": False, "max_cudagraph_capture_size": 0, "return_softmax_lse": False,
                "mathematical_program": "bf16_mla_absorbed_prefill_softmax_v1", "symbolic_workspace_complete": False})
        return self.trace.call(node, (query, self.cache), shape=(rows, self.heads, 512), dtype="bf16"), None


class SGLangDecodeAttention:
    """forward_absorb_prepare/core: BF16 absorption, separate RoPE/cache."""
    def __init__(self, trace, model, pool, layer, group, *, cache, slots, slot_tensor, rope, kv_lens, q_lens=None,
            absorbed_prefill=False, graph_dummy_rows=0, name_prefix=None):
        self.trace, self.model, self.pool = trace, model, pool
        self.heads = model.num_attention_heads // len(group.traces)
        prefix, path = f"layer-{layer}:attention", f"model.layers.{layer}.self_attn"
        if name_prefix is not None:
            prefix = name_prefix
        self.qkv = _source_linear(trace, model, pool, prefix + ":qkv-a", model.hidden_size,
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, path + ".fused_qkv_a_proj_with_mqa")
        self.q_norm = SGLangRMSNorm(trace, prefix + ":q-a-norm", pool, model.q_lora_rank, model.rms_norm_eps)
        self.kv_norm = SGLangRMSNorm(trace, prefix + ":kv-a-norm", pool, model.kv_lora_rank, model.rms_norm_eps)
        self.q_b = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), path + ".q_b_proj")
        self.absorb = SymbolicBatchedLinear(trace, prefix + ":q-absorb", pool, self.heads,
            model.qk_nope_head_dim, model.kv_lora_rank)
        self.rotary = DeepseekV3Rotary(trace, prefix + ":rope", pool, self.heads, 1, model.qk_rope_head_dim, rope)
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_tensor,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots,
            sglang_dummy_rows=graph_dummy_rows)
        self.concat = SymbolicConcat(trace, prefix + ":query", pool)
        if absorbed_prefill:
            self.attn = SGLangAbsorbedPrefill(trace, prefix + ":latent", pool, heads=self.heads,
                q_lens=q_lens, kv_lens=kv_lens, cache=cache)
        else:
            options = {} if q_lens is None else {"q_lens": q_lens,
                "source_profile": "sglang_0_5_19_causal_flashmla_bf16_source", "source_batch_invariant": False}
            if graph_dummy_rows:
                options.update(sglang_dummy_rows=graph_dummy_rows // (q_lens[0] if q_lens is not None else 1),
                    source_profile="sglang_0_5_19_causal_flashmla_bf16_source", source_batch_invariant=False)
            self.attn = SymbolicMLALatentDecode(trace, prefix + ":latent", pool, heads=self.heads, kv_lens=kv_lens, cache=cache, **options)
        self.v_up = SymbolicBatchedLinear(trace, prefix + ":v-up", pool, self.heads, model.kv_lora_rank, model.v_head_dim)
        self.o = _source_linear(trace, model, pool, prefix + ":o-proj", self.heads * model.v_head_dim,
            model.hidden_size, path + ".o_proj")

    def __call__(self, hidden, positions):
        model, rows = self.model, hidden.shape[0]
        combined, _ = self.qkv(hidden)
        q_latent, kv_latent = combined.split((model.q_lora_rank, model.kv_lora_rank + model.qk_rope_head_dim))
        k_nope = self.kv_norm(kv_latent.narrow(-1, 0, model.kv_lora_rank))
        k_pe = kv_latent.narrow(-1, model.kv_lora_rank, model.qk_rope_head_dim).view_as_shape(rows, 1, model.qk_rope_head_dim)
        q, _ = self.q_b(self.q_norm(q_latent))
        q = q.view_as_shape(rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
        q_nope, q_pe = q.split((model.qk_nope_head_dim, model.qk_rope_head_dim))
        absorbed = self.absorb(q_nope.transpose(0, 1)).transpose(0, 1)
        q_pe, k_pe = self.rotary(positions, q_pe, k_pe)
        self.update(k_nope, k_pe)
        query = self.concat(absorbed, q_pe)
        latent, lse = self.attn(query)
        output = self.v_up(latent.transpose(0, 1)).transpose(0, 1)
        # Source bmm's [heads,T,V].transpose is non-dense for T>1;
        # reshape to [T,heads*V] materializes before o_proj consumes it.
        if rows > 1:
            node = SymbolicOperator(self.trace, self.o.name + ":reshape-copy", self.pool).node(
                WorkloadKind.GATHER, {"elements": rows * self.heads * model.v_head_dim,
                    "rows": rows, "hidden": self.heads * model.v_head_dim}, "bf16", "gather", {
                        "gather_factor": 1, "index_count": 0, "index_bytes": 8,
                        "scattered": False, "materialized_output": True})
            output = self.trace.call(node, (output,), shape=(rows, self.heads * model.v_head_dim), dtype="bf16")
        else:
            output = output.view_as_shape(rows, self.heads * model.v_head_dim)
        return self.o(output)[0]


class SGLangFreshMHAAttention:
    """Fresh prefix-free forward_normal_prepare/core, no backend FP8 hook.

    Unlike vLLM's wrapper, SGLang concatenates K in one kernel and directly
    consumes the fresh MHA result; no caller output or copy is allocated.
    """
    def __init__(self, trace, model, pool, layer, group, *, cache, slots, slot_tensor, rope, q_lens):
        self.trace, self.model = trace, model
        self.heads = model.num_attention_heads // len(group.traces)
        prefix, path = f"layer-{layer}:attention", f"model.layers.{layer}.self_attn"
        self.qkv = _source_linear(trace, model, pool, prefix + ":qkv-a", model.hidden_size,
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, path + ".fused_qkv_a_proj_with_mqa")
        self.q_norm = SGLangRMSNorm(trace, prefix + ":q-a-norm", pool, model.q_lora_rank, model.rms_norm_eps)
        self.kv_norm = SGLangRMSNorm(trace, prefix + ":kv-a-norm", pool, model.kv_lora_rank, model.rms_norm_eps)
        self.q_b = _source_linear(trace, model, pool, prefix + ":q-b", model.q_lora_rank,
            self.heads * (model.qk_nope_head_dim + model.qk_rope_head_dim), path + ".q_b_proj")
        self.rotary = DeepseekV3Rotary(trace, prefix + ":rope", pool, self.heads, 1, model.qk_rope_head_dim, rope)
        self.update = SymbolicMLACacheUpdate(trace, prefix + ":kv-update", pool, cache, slot_tensor,
            latent_dim=model.kv_lora_rank, rope_dim=model.qk_rope_head_dim, slots=slots)
        self.backend = MLAFreshPrefillBackend(trace, model, pool, prefix, heads=self.heads,
            q_lens=q_lens, cache=cache, module_path=path)
        self.concat = SymbolicConcat(trace, prefix + ":key-concat", pool)
        self.o = _source_linear(trace, model, pool, prefix + ":o-proj", self.heads * model.v_head_dim,
            model.hidden_size, path + ".o_proj")

    def _prepare(self, hidden, positions):
        model, rows = self.model, hidden.shape[0]
        combined, _ = self.qkv(hidden)
        q_latent, kv_latent = combined.split((model.q_lora_rank, model.kv_lora_rank + model.qk_rope_head_dim))
        latent = self.kv_norm(kv_latent.narrow(-1, 0, model.kv_lora_rank))
        rope = kv_latent.narrow(-1, model.kv_lora_rank, model.qk_rope_head_dim).view_as_shape(rows, 1, model.qk_rope_head_dim)
        query, _ = self.q_b(self.q_norm(q_latent))
        query = query.view_as_shape(rows, self.heads, model.qk_nope_head_dim + model.qk_rope_head_dim)
        _, q_pe = query.split((model.qk_nope_head_dim, model.qk_rope_head_dim))
        self.rotary(positions, q_pe, rope)
        self.update(latent, rope)
        return query, latent, rope

    def _expand_fresh_kv(self, latent, rope):
        model, rows = self.model, latent.shape[0]
        expanded = self.backend.kv_b_proj(latent)[0].view_as_shape(rows, self.heads,
            model.qk_nope_head_dim + model.v_head_dim)
        k_nope, value = expanded.split((model.qk_nope_head_dim, model.v_head_dim))
        key = self.concat(k_nope, rope.expand(rows, self.heads, model.qk_rope_head_dim))
        return key, value

    def __call__(self, hidden, positions):
        model, rows = self.model, hidden.shape[0]
        query, latent, rope = self._prepare(hidden, positions)
        key, value = self._expand_fresh_kv(latent, rope)
        output = self.backend._attention(query, key, value, kv_lens=self.backend.q_lens,
            causal=True, return_lse=False)
        return self.o(output.view_as_shape(rows, self.heads * model.v_head_dim))[0]


class SGLangChunkedMHAAttention(SGLangFreshMHAAttention):
    """Causal suffix, then ordered noncausal chunks and fresh O/LSE merges.

    FlashInfer returns [rows,heads] LSE. Transposed virtual views adapt the
    same mathematical merge owner; they do not materialize a layout copy or
    claim a native merge/provider. Empty per-request chunks retain the
    initialized zero-O/negative-infinity-LSE state and both merge input reads.
    """
    def __init__(self, *args, chunk_lengths, chunk_indices, chunk_index_tensors, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_lengths, self.chunk_indices = chunk_lengths, chunk_indices
        self.chunk_index_tensors = chunk_index_tensors
        self.merge = SymbolicAttentionStateMerge(self.trace, self.backend.prefix + ":chunk-merge", self.backend.pool)

    def __call__(self, hidden, positions):
        model, rows = self.model, hidden.shape[0]
        query, latent, rope = self._prepare(hidden, positions)
        key, value = self._expand_fresh_kv(latent, rope)
        accum, accum_lse = self.backend._attention(query, key, value, kv_lens=self.backend.q_lens,
            causal=True, return_lse=True, lse_layout="rows_heads")
        del latent, rope, key, value
        for lengths, indices, index_tensor in zip(self.chunk_lengths, self.chunk_indices, self.chunk_index_tensors, strict=True):
            chunk_rows = sum(lengths)
            node = self.concat.node(WorkloadKind.GATHER, {"elements": chunk_rows * 576,
                "rows": chunk_rows, "hidden": 576}, "bf16", "gather", {
                    "gather_factor": 1, "index_count": chunk_rows, "index_bytes": 4,
                    "scattered": True, "materialized_output": True,
                    "selected_indices": indices, "index_producer_bound": False})
            latent, rope = self.trace.call_outputs(node, (self.backend.cache, index_tensor), outputs=(
                ("latent", (chunk_rows, model.kv_lora_rank), "bf16"),
                ("rope", (chunk_rows, 1, model.qk_rope_head_dim), "bf16")))
            key, value = self.backend._expand_kv(latent, rope)
            partial, partial_lse = self.backend._attention(query, key, value,
                kv_lens=lengths, causal=False, return_lse=True, lse_layout="rows_heads", allow_empty_kv=True)
            output = self.trace.tensor(self.trace.unique_name(self.backend.prefix + ":chunk-output"), accum.shape, "bf16")
            output_lse = self.trace.tensor(self.trace.unique_name(self.backend.prefix + ":chunk-lse"), accum_lse.shape, "fp32")
            self.merge(partial, partial_lse.transpose(0, 1), accum, accum_lse.transpose(0, 1),
                out=output, out_lse=output_lse.transpose(0, 1),
                empty_prefix_rows=sum(q for q, k in zip(self.backend.q_lens, lengths, strict=True) if k == 0))
            accum, accum_lse = output, output_lse
            del latent, rope, key, value, partial, partial_lse, output, output_lse
        return self.o(accum.view_as_shape(rows, self.heads * model.v_head_dim))[0]


class SGLangOneShotMHAAttention(SGLangFreshMHAAttention):
    """Source one-shot: gather updated prefix+suffix cache, then expand once.

    INT32 cache indices are explicit caller metadata. Their req-to-token
    producer and actual backend planning remain outside this source scope.
    """
    def __init__(self, *args, kv_lens, kv_indices, kv_index_tensor, max_kv_chunk_capacity, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_lens, self.kv_indices = kv_lens, kv_indices
        self.trace._check_tensor(kv_index_tensor)
        if kv_index_tensor.shape != (sum(kv_lens),) or kv_index_tensor.dtype != "int32":
            raise ValueError("SGLang one-shot requires its shared INT32 cache-index tensor")
        self.kv_index_tensor = kv_index_tensor
        self.capacity = max_kv_chunk_capacity

    def __call__(self, hidden, positions):
        model, rows = self.model, hidden.shape[0]
        query, suffix_latent, suffix_rope = self._prepare(hidden, positions)
        cache_rows = sum(self.kv_lens)
        node = self.concat.node(WorkloadKind.GATHER, {"elements": cache_rows * 576,
            "rows": cache_rows, "hidden": 576}, "bf16", "gather", {
                "gather_factor": 1, "index_count": cache_rows, "index_bytes": 4,
                "scattered": True, "materialized_output": True,
                "selected_indices": self.kv_indices, "index_producer_bound": False})
        latent, rope = self.trace.call_outputs(node, (self.backend.cache, self.kv_index_tensor), outputs=(
            ("latent", (cache_rows, model.kv_lora_rank), "bf16"),
            ("rope", (cache_rows, 1, model.qk_rope_head_dim), "bf16")))
        del suffix_latent, suffix_rope
        expanded = self.backend.kv_b_proj(latent)[0].view_as_shape(cache_rows, self.heads,
            model.qk_nope_head_dim + model.v_head_dim)
        k_nope, value = expanded.split((model.qk_nope_head_dim, model.v_head_dim))
        key = self.concat(k_nope, rope.expand(cache_rows, self.heads, model.qk_rope_head_dim))
        node = self.concat.node(WorkloadKind.FLASH_ATTN, {"batch": len(self.kv_lens),
            "q_len": max(self.backend.q_lens), "kv_len": max(self.kv_lens), "num_query_heads": self.heads,
            "qk_head_dim": model.qk_nope_head_dim + model.qk_rope_head_dim,
            "v_head_dim": model.v_head_dim, "kv_lora_rank": model.kv_lora_rank}, "bf16", "mla_prefill", {
                "source_component": "sglang_0_5_19_symbolic_one_shot_prefill", "causal": True,
                "kv_layout": "mla", "kv_dist": "uniform", "q_lens": self.backend.q_lens, "kv_lens": self.kv_lens,
                "full_cuda_graphs": False, "max_cudagraph_capture_size": 0, "return_softmax_lse": False,
                "mathematical_program": "bf16_mla_one_shot_prefill_softmax_v1",
                "prefix_cache_threshold": 8192, "max_kv_chunk_capacity": self.capacity,
                "model_execution_version": "sglang_0_5_19", "symbolic_workspace_complete": False})
        output = self.trace.call(node, (query, key, value), shape=(rows, self.heads, model.v_head_dim), dtype="bf16")
        return self.o(output.view_as_shape(rows, self.heads * model.v_head_dim))[0]


def _prefill_logits_hidden(trace, pool, hidden, q_lens):
    """LogitsProcessor's INT32 cumsum -> INT64 sub -> hidden[last_index]."""
    count, source_rows = len(q_lens), hidden.shape[0]
    lengths = trace.tensor(trace.unique_name("extend-seq-lens"), (count,), "int32")
    attributes = {"source_component": "sglang_0_5_19_symbolic_logits_boundary",
        "model_execution_version": "sglang_0_5_19"}
    def node(name, operation, dtype, width, **extra):
        return SymbolicOperator(trace, name, pool).node(WorkloadKind.VECTOR,
            {"elements": count * width, "rows": count, "width": width,
                "source_rows": source_rows if operation == "gather" else count,
                "dtype_bytes": {"int32": 4, "int64": 8, "bf16": 2}[dtype]}, dtype, "vllm_mtp_eager",
            {**attributes, "operation": operation, **extra})
    cumulative = trace.call(node("logits-cumsum", "cumsum", "int32", 1), (lengths,), shape=(count,), dtype="int64")
    indices = trace.call(node("logits-last-index", "unary", "int64", 1), (cumulative,), shape=(count,), dtype="int64")
    total, selected = 0, []
    for length in q_lens:
        total += length
        selected.append(total - 1)
    return trace.call(node("logits-hidden-select", "gather", "bf16", hidden.shape[1],
        index_bytes=8, selected_indices=tuple(selected)), (hidden, indices),
        shape=(count, hidden.shape[1]), dtype="bf16")


class SGLangGreedy(SymbolicOperator):
    """CUDA torch.argmax branch: fresh INT64 IDs, no vLLM output cast.

    This explicit source declares all_greedy, no processors/penalties,
    logprobs, grammar or return mask, async assert and NaN sanitize off,
    and token sync off. It does not observe or execute those runtime flags.
    """
    def __call__(self, logits):
        self.trace._check_tensor(logits)
        if len(logits.shape) != 2 or logits.dtype != "fp32":
            raise ValueError("SGLang ordinary greedy requires FP32 vocabulary rows")
        rows, vocab = logits.shape
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * vocab, "rows": rows, "vocab": vocab},
            "fp32", "vllm_plain_greedy_sample", {
                "source_component": "sglang_0_5_19_symbolic_logits_boundary",
                "sampling_source": SAMPLING_SOURCE, "stage": "argmax"})
        return self.trace.call(node, (logits,), shape=(rows,), dtype="int64")


class SGLangEagleOps:
    """Typed source materializations; no integer/selection issue-rate claim."""
    def __init__(self, trace, pool, *, proposal_depth=1, namespace="eagle"):
        self.trace, self.pool = trace, pool
        self.proposal_depth, self.namespace = proposal_depth, namespace

    def node(self, name, operation, dtype, rows, width, *, source_rows=None, **attributes):
        return SymbolicOperator(self.trace, self.namespace + ":" + name, self.pool).node(WorkloadKind.VECTOR,
            {"elements": rows * width, "rows": rows, "width": width,
                "source_rows": rows if source_rows is None else source_rows}, dtype, "vllm_mtp_eager", {
                "source_component": "sglang_0_5_19_symbolic_logits_boundary", "operation": operation,
                "model_execution_version": "sglang_0_5_19", "eagle_source": f"topk1_steps{self.proposal_depth}_eager", **attributes})

    def copy(self, source, name, destination=None):
        width = prod(source.shape[1:])
        return self.trace.call(self.node(name, "copy", source.dtype, source.shape[0], width), (source,),
            shape=source.shape, dtype=source.dtype, out=destination)

    def fill(self, destination, name):
        width = prod(destination.shape[1:])
        return self.trace.call(self.node(name, "fill", destination.dtype, destination.shape[0], width), (),
            shape=destination.shape, dtype=destination.dtype, out=destination)

    def gather(self, source, indices, selected, name):
        width = 1 if len(source.shape) == 1 else source.shape[1]
        shape = (len(selected), *source.shape[1:])
        return self.trace.call(self.node(name, "gather", source.dtype, len(selected), width,
            source_rows=source.shape[0], index_bytes=indices.view.element_bytes, selected_indices=selected),
            (source, indices), shape=shape, dtype=source.dtype)

    def unary(self, source, name):
        return self.trace.call(self.node(name, "unary", source.dtype, source.shape[0], 1),
            (source,), shape=source.shape, dtype=source.dtype)


def _eagle_prepare_verify(trace, pool, state, context, input_ids, draft_tokens=None, draft_tree_ports=None):
    """Connected draft columns, token cat and eager FULL_MASK tree build."""
    batch, width = len(state.request_ids), state.proposal_depth + 1
    contexts = (context,) if type(context) is int else context
    if len(contexts) != batch:
        raise ValueError("SGLang verify tree needs one declared context per request")
    ops = SGLangEagleOps(trace, pool, proposal_depth=state.proposal_depth)
    bonus = trace.tensor("eagle:carry-bonus", (batch,), state.input_bonus_dtype)
    if draft_tokens is None:
        if state.proposal_depth != 1:
            raise ValueError("SGLang steps2 verify requires the connected draft-decode token producer")
        proposal = trace.tensor("eagle:carry-top1", (batch, 1), "int64")
        draft = ops.copy(proposal, "draft-column-copy")
    else:
        trace._check_tensor(draft_tokens)
        if draft_tokens.shape != (batch, state.proposal_depth) or draft_tokens.dtype != "int64":
            raise ValueError("SGLang draft token port must match every declared request and proposal column")
        draft = draft_tokens
    operation = "concat" if bonus.dtype == "int64" else "promote_concat"
    ids = trace.call(ops.node("tree-token-cat", operation, bonus.dtype, batch, width),
        (bonus, draft), shape=(batch * width,), dtype="int64")
    # Eager FlashMLA has verify_mask=None. FULL_MASK's prefix is initialized
    # even though this backend's causal fixed-q path never consumes the mask.
    mask = ops.fill(trace.tensor("eagle:tree-mask", (width * (sum(contexts) + batch * width),), "bool"), "tree-mask-fill")
    retrieve_shape = (3, width) if batch == 1 else (3, batch, width)
    retrieve = ops.fill(trace.tensor("eagle:retrieve-buf", retrieve_shape, "int64"), "retrieve-fill")
    if draft_tree_ports is None:
        parent = trace.tensor("eagle:parent-prealloc", (batch, state.proposal_depth), "int64", persistent=True)
        scores = trace.tensor("eagle:score-index-prealloc", (batch, state.proposal_depth), "int64", persistent=True)
    else:
        parent, scores = draft_tree_ports
        for tensor in (parent, scores):
            trace._check_tensor(tensor)
            if tensor.shape != (batch, state.proposal_depth) or tensor.dtype != "int64":
                raise ValueError("EAGLE tree graph ports must match the declared topk1 chain")
    seq = trace.tensor("eagle:seq-len", (batch,), "int64")
    positions = trace.call(ops.node("tree-build", "eagle_tree_build", "int64", batch, width),
        ((scores, seq, retrieve) if state.proposal_depth == 1 else (scores, parent, seq, retrieve)),
        shape=(batch * width,), dtype="int64", writes=(retrieve, mask), controls=(ids,))
    return ids, positions, retrieve, mask


def _eagle_verify_outputs(trace, pool, state, ids, retrieve, argmax):
    batch, width = len(state.request_ids), state.proposal_depth + 1
    ops = SGLangEagleOps(trace, pool, proposal_depth=state.proposal_depth)
    predicted = ops.fill(trace.tensor("eagle:predict", (batch * width,), "int32"), "predict-zero")
    accepted_indices = ops.fill(trace.tensor("eagle:accept-index", (batch, width), "int32"), "accept-index-fill")
    correct = trace.tensor("eagle:correct-drafts", (batch,), "int32")
    accepted = tuple(length - 1 for length in state.accept_lens)
    predicted = trace.call(ops.node("tree-verify", "eagle_tree_verify", "int64", batch, width,
        accepted_drafts=accepted[0] if batch == 1 and width == 2 else accepted), (ids, retrieve, argmax),
        shape=predicted.shape, dtype=predicted.dtype, out=predicted, writes=(accepted_indices, correct))
    accepted_lens = ops.unary(correct, "accept-plus-bonus")
    seq = trace.tensor("eagle:old-seq-len", (batch,), "int64")
    # The source performs seq_lens + INT32 accept_lens, yielding INT64.
    updated_seq = trace.call(ops.node("new-seq-len", "promote_add", "int32", batch, 1),
        (seq, accepted_lens), shape=(batch,), dtype="int64")
    selected = tuple(index if index >= 0 else batch * width + index for row in state.accept_index for index in row)
    accepted_tokens = ops.gather(predicted, accepted_indices.view_as_shape(batch * width), selected, "accepted-token-gather")
    bonus = trace.call(ops.node("bonus-token", "eagle_bonus", "int32", batch, width),
        (accepted_tokens, accepted_lens), shape=(batch,), dtype="int32")
    cast_ids = trace.call(ops.node("draft-input-int64", "cast_int64", "int32", batch * width, 1),
        (predicted,), shape=(batch * width,), dtype="int64", controls=(bonus, updated_seq))
    declared = tuple(state.target_argmax_ids[row * width + column] if column < state.accept_lens[row] else 0
        for row in range(batch) for column in range(width))
    return cast_ids, declared, accepted_lens


def _eagle_topk1_postprocess(trace, pool, state, logits, positions, *, draft_tokens=None):
    """Local topk1.py's split8192 public buffers, without launching Triton."""
    batch, vocab = logits.shape
    splits = (vocab + 8191) // 8192
    ops = SGLangEagleOps(trace, pool, proposal_depth=state.proposal_depth, namespace="eagle-topk1")
    values, indices = trace.call_outputs(ops.node("partial", "eagle_topk1_partial", "fp32", batch, vocab),
        (logits,), outputs=(("values", (batch, splits), "fp32"), ("indices", (batch, splits), "int32")))
    probability = trace.tensor(trace.unique_name("eagle-topk1:probability"), (batch, 1), "fp32")
    writes = (probability, positions) if draft_tokens is None else (probability, positions, draft_tokens)
    top1 = trace.call(ops.node("finalize", "eagle_topk1_finalize", "fp32", batch, splits,
        write_draft_token=draft_tokens is not None), (values, indices, positions),
        shape=(batch, 1), dtype="int64", writes=writes)
    return probability, top1



def _graph_input_buffer(ops, source, rows, name, *, fill_value=None):
    """Prepare one runner's static port; unknown retained tails stay unknown."""
    shape = (rows, *source.shape[1:])
    destination = ops.trace.tensor(name, shape, source.dtype, persistent=True)
    if fill_value is not None:
        node = ops.node(name + ":reset", "fill", source.dtype, rows, prod(shape[1:]), source_fill_value=fill_value)
        ops.trace.call(node, (), shape=shape, dtype=source.dtype, out=destination)
    ops.copy(source, name + ":active-copy", destination.narrow(0, 0, source.shape[0]))
    return destination

def _eagle_draft_decode(trace, model, step, pool, group, collective, gather, state, *, rope,
        allreduce_fusion, resolution, invocation, cache, capture=False, capture_batch_size=None):
    """The existing steps2 loop performs exactly one model before verify."""
    logical_batch = len(state.request_ids)
    batch = logical_batch if capture_batch_size is None else capture_batch_size
    dummy = batch - logical_batch
    ops = SGLangEagleOps(trace, pool.name, proposal_depth=state.proposal_depth, namespace="eagle-draft")
    proposal = trace.tensor("eagle-draft:carry-top1", (logical_batch, 1), "int64")
    hidden = trace.tensor("eagle-draft:carry-hidden", (logical_batch, model.hidden_size), "bf16")
    sequence = trace.tensor("eagle-draft:sequence-lengths", (logical_batch,), "int64")
    positions = ops.copy(sequence, "positions-repeat-topk1")
    slots = trace.tensor("nextn-draft:slot-mapping", (logical_batch,), "int64")
    if dummy:
        proposal = _graph_input_buffer(ops, proposal, batch, "eagle-draft:graph-top1", fill_value=0)
        hidden = _graph_input_buffer(ops, hidden, batch, "eagle-draft:graph-hidden", fill_value=0)
        positions = _graph_input_buffer(ops, positions, batch, "eagle-draft:graph-positions", fill_value=0)
        slots = _graph_input_buffer(ops, slots, batch, "eagle-draft:graph-slots", fill_value=0)
    parent = trace.tensor("eagle:parent-prealloc", (batch, 2), "int64", persistent=True)
    scores = trace.tensor("eagle:score-index-prealloc", (batch, 2), "int64", persistent=True)
    if capture:
        probability = trace.tensor("eagle-draft:carry-probability", (logical_batch, 1), "fp32")
        if dummy:
            probability = _graph_input_buffer(ops, probability, batch, "eagle-draft:graph-probability", fill_value=0)
        trace.begin_capture(key=f"{group.source_profile}:eagle-draft:{batch}", logical_tokens=logical_batch,
            model_input_tokens=batch, inputs={"input_ids": proposal.view_as_shape(batch), "positions": positions,
                "slot_mapping": slots}, source_stage="sglang_eagle_draft_decode",
            padding_program="sglang_zero_slot0_seq1" if dummy else "vllm_masked_minus1", extra_inputs={"hidden_states": hidden, "topk_p": probability, "topk_index": proposal}, source_graph_batch_size=batch)
    tokens = trace.tensor("eagle-draft:tokens", (batch, 2), "int64")
    ops.copy(proposal, "first-column", tokens.narrow(1, 0, 1))
    logits, hidden, positions = _trace_nextn_extend(trace, model, step, pool, group, collective, gather, state,
        ids=proposal.view_as_shape(batch), token_ids=state.draft_token_ids[::2], positions=positions, rope=rope,
        target_hidden=hidden, allreduce_fusion=allreduce_fusion, resolution=resolution, invocation=invocation,
        accepted_lens=None, decode_carry=True, cache=cache, slot_tensor=slots, graph_dummy_requests=dummy)
    _eagle_topk1_postprocess(trace, pool.name, state, logits, positions, draft_tokens=tokens.narrow(1, 1, 1))
    if capture:
        trace.call(ops.node("positions-reset", "unary", "int64", batch, 1), (positions,),
            shape=positions.shape, dtype="int64", out=positions)
        trace.end_capture(outputs={"parent_list": parent, "top_scores_index": scores, "draft_tokens": tokens})
    return (tokens.narrow(0, 0, logical_batch), parent.narrow(0, 0, logical_batch), scores.narrow(0, 0, logical_batch)) if dummy else (tokens, parent, scores)


def _trace_nextn_extend(trace, model, step, pool, group, collective, gather, state, *, ids, token_ids,
        positions, rope, target_hidden, allreduce_fusion, resolution, invocation, accepted_lens,
        decode_carry=False, cache=None, slot_tensor=None, capture=False, capture_batch_size=None, graph_dummy_requests=0):
    """Actual SG NextN fused EH norm, one MoE decoder, shared head and carry."""
    batch, rows = len(state.request_ids), len(token_ids)
    logical_rows = rows
    dummy = graph_dummy_requests if decode_carry else (0 if capture_batch_size is None else capture_batch_size - batch)
    if capture and (decode_carry or state.phase != "verify" or state.proposal_depth != 2):
        raise NotImplementedError("NextN staged capture requires the existing exact steps2 draft-extend source")
    prefix = "nextn-draft" if decode_carry else "nextn"
    ops = SGLangEagleOps(trace, pool.name, proposal_depth=state.proposal_depth,
        namespace="eagle-draft" if decode_carry else "eagle")
    if state.phase == "verify" and not decode_carry:
        correct_drafts = ops.unary(accepted_lens, "draft-correct-sub-one")
        start = ops.fill(trace.tensor("eagle:select-arange", (batch,), "int64"), "select-arange-fill")
        index = trace.call(ops.node("select-add-accept", "promote_add", "int32", batch, 1),
            (start, accepted_lens), shape=(batch,), dtype="int64")
        index = ops.unary(index, "select-sub-one")
    if cache is None:
        cache = trace.tensor("nextn:kv-cache", (state.draft_cache_capacity_tokens, 576), "bf16", persistent=True)
    slots = trace.tensor(prefix + ":slot-mapping", (rows,), "int64") if slot_tensor is None else slot_tensor
    q_lens = (1,) * batch if decode_carry else tuple(request.scheduled_tokens for request in step.requests)
    if decode_carry:
        draft_positions = positions
    else:
        context = trace.tensor(prefix + ":extend-prefix", (batch,), "int32")
        lengths = trace.tensor(prefix + ":extend-lens", (batch,), "int32")
        starts = trace.tensor(prefix + ":extend-start", (batch,), "int32")
        draft_positions = trace.call(ops.node("draft-positions", "eagle_positions", "int32", batch, max(q_lens),
            query_lens=q_lens), (context, lengths), shape=(rows,), dtype="int64", writes=(starts,))
    if dummy:
        if decode_carry:
            token_ids += (0,) * dummy
            q_lens += (1,) * dummy
        else:
            padded_rows = rows + 3 * dummy
            ids = _graph_input_buffer(ops, ids, padded_rows, "eagle:extend-graph-ids")
            draft_positions = _graph_input_buffer(ops, draft_positions, padded_rows, "eagle:extend-graph-positions", fill_value=0)
            slots = _graph_input_buffer(ops, slots, padded_rows, "eagle:extend-graph-slots", fill_value=0)
            target_hidden = _graph_input_buffer(ops, target_hidden, padded_rows, "eagle:extend-graph-hidden")
            correct_drafts = _graph_input_buffer(ops, correct_drafts, batch + dummy, "eagle:extend-graph-correct", fill_value=3)
            accepted_lens = _graph_input_buffer(ops, accepted_lens, batch + dummy, "eagle:extend-graph-accepted", fill_value=3)
            token_ids += state.extend_graph_inputs.retained_padding_token_ids
            q_lens += (3,) * dummy
        rows = len(token_ids)
    if capture:
        trace.begin_capture(key=f"{group.source_profile}:eagle-extend:{batch + dummy}", logical_tokens=logical_rows,
            model_input_tokens=rows, inputs={"input_ids": ids, "positions": draft_positions, "slot_mapping": slots},
            source_stage="sglang_eagle_draft_extend",
            padding_program="sglang_zero_slot0_seq1" if dummy else "vllm_masked_minus1", extra_inputs={"hidden_states": target_hidden,
                "num_correct_drafts": correct_drafts, "num_accept_tokens": accepted_lens}, source_graph_batch_size=batch + dummy)
    embedded = SGLangEmbedding(trace, prefix + ":embedding", pool.name, model, group, collective, token_ids)(ids)
    ew = trace.tensor(prefix + ":enorm-weight", (model.hidden_size,), "bf16", persistent=True)
    hw = trace.tensor(prefix + ":hnorm-weight", (model.hidden_size,), "bf16", persistent=True)
    combined = trace.call(ops.node("fused-eh-norm", "eagle_eh_norm", "bf16", rows, model.hidden_size,
        epsilon=model.rms_norm_eps), (embedded, target_hidden, ew, hw), shape=(rows, 2 * model.hidden_size), dtype="bf16")
    hidden = SymbolicLinear(trace, prefix + ":eh-proj", pool.name, 2 * model.hidden_size, model.hidden_size)(combined)[0]
    residual, hidden = hidden, SGLangRMSNorm(trace, prefix + ":input-norm", pool.name, model.hidden_size, model.rms_norm_eps)(hidden)
    kwargs = dict(cache=cache, slots=(state.draft_decode_slot_mapping if decode_carry else state.draft_slot_mapping) + (0,) * (rows - logical_rows),
        slot_tensor=slots, rope=rope)
    prefix_tokens = sum(request.computed_tokens for request in step.requests)
    if state.phase == "prefill" and prefix_tokens == 0:
        attention = SGLangFreshMHAAttention(trace, model, pool.name, 61, group, **kwargs, q_lens=q_lens)
    else:
        attention = SGLangDecodeAttention(trace, model, pool.name, 61, group, **kwargs,
            kv_lens=tuple(request.computed_tokens + query for request, query in zip(step.requests, q_lens[:batch], strict=True)) + (1,) * dummy,
            q_lens=q_lens, graph_dummy_rows=rows - logical_rows, absorbed_prefill=state.phase == "prefill",
            name_prefix=prefix + ":attention" if decode_carry else None)
    hidden = attention(hidden, draft_positions)
    if allreduce_fusion:
        hidden, residual = SGLangAllReduceRMSNorm(trace, prefix + ":post-attn-fused-all-reduce", pool.name,
            model.hidden_size, model.rms_norm_eps, group, collective)(hidden, residual)
    else:
        hidden = group.call(prefix + ":attention-all-reduce", hidden, pool_name=pool.name,
            config=collective, inplace=False, implementation="sglang_custom_all_reduce_v2")
        hidden, residual = SGLangRMSNorm(trace, prefix + ":post-attn-norm", pool.name,
            model.hidden_size, model.rms_norm_eps)(hidden, residual)
    hidden = SGLangFusedSharedMoE(trace, pool.name, model, 61, resolution, invocation,
        name=prefix + ":mlp" if decode_carry else None)(hidden)
    hidden = group.call(prefix + ":mlp-all-reduce", hidden, pool_name=pool.name, config=collective,
        inplace=False, implementation="sglang_custom_all_reduce_v2")
    hidden, _ = SGLangRMSNorm(trace, prefix + ":shared-head-norm", pool.name,
        model.hidden_size, model.rms_norm_eps)(hidden, residual)
    full_hidden = hidden
    if state.phase == "prefill":
        hidden = _prefill_logits_hidden(trace, pool.name, hidden, q_lens)
    local = SymbolicLinear(trace, prefix + ":lm-head", pool.name, model.hidden_size, model.vocab_size // 8,
        kind=WorkloadKind.LM_HEAD)(hidden)[0]
    gathered = group.call(prefix + ":logits-all-gather", local, pool_name=pool.name, config=gather, inplace=False, dim=-1)
    logits = trace.call(SymbolicOperator(trace, prefix + ":logits-fp32", pool.name).node(WorkloadKind.VECTOR,
        {"elements": hidden.shape[0] * model.vocab_size, "rows": hidden.shape[0], "vocab": model.vocab_size},
        "bf16", "vllm_logits_fp32_cast", {"source_component": "sglang_0_5_19_symbolic_logits_boundary",
            "sampling_source": "sglang_0_5_19_logits_processor_bf16"}), (gathered,),
        shape=(hidden.shape[0], model.vocab_size), dtype="fp32")
    if capture:
        trace.end_capture(outputs={"next_token_logits": logits, "hidden_states": full_hidden})
        if dummy:
            logits, full_hidden = logits.narrow(0, 0, logical_rows), full_hidden.narrow(0, 0, logical_rows)
            accepted_lens = accepted_lens.narrow(0, 0, batch)
    if decode_carry:
        return logits, full_hidden, draft_positions
    if state.phase == "prefill":
        probabilities = trace.call(ops.node("draft-softmax", "eagle_softmax", "fp32", batch, model.vocab_size),
            (logits,), shape=logits.shape, dtype="fp32")
        probability, top1 = trace.call_outputs(ops.node("draft-max", "eagle_max", "fp32", batch, model.vocab_size),
            (probabilities,), outputs=(("probability", (batch, 1), "fp32"), ("id", (batch, 1), "int64")))
        return probability, top1, hidden
    # The worker selects source rows only after full draft-extend logits.

    selected = tuple(row * (state.proposal_depth + 1) + length - 1 for row, length in enumerate(state.accept_lens))
    selected_logits = ops.gather(logits, index, selected, "next-logits-gather")
    selected_hidden = ops.gather(full_hidden, index, selected, "next-hidden-gather")
    next_id = SGLangGreedy(trace, "nextn:next-argmax", pool.name)(selected_logits).view_as_shape(batch, 1)
    probability = ops.fill(trace.tensor("nextn:top1-probability", (batch, 1), "fp32"), "top1-ones")
    return probability, next_id, selected_hidden


class SGLangFusedSharedMoE(SymbolicOperator):
    """FP32 gate; standard grouped outputs include shared ID256/weight1."""
    def __init__(self, trace, pool, model, layer, resolution, invocation, *, name=None):
        super().__init__(trace, f"layer-{layer}:mlp" if name is None else name, pool)
        self.model, self.resolution, self.invocation = model, resolution, invocation
        self.gate = SymbolicLinear(trace, self.name + ":gate", pool, model.hidden_size,
            model.num_experts, output_dtype="fp32")
        self.bias = trace.tensor(self.name + ":bias", (model.num_experts,), "fp32", persistent=True)

    def __call__(self, hidden):
        model, rows = self.model, hidden.shape[0]
        logits, _ = self.gate(hidden)
        node = self.node(WorkloadKind.VECTOR, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
            "intermediate": model.moe_intermediate_size}, "fp32", "moe_topk", {
                "source_component": "sglang_0_5_19_symbolic_grouped_routing", "source_program": ROUTING_PROGRAM,
                "mathematical_program": ROUTING_MATH, "normalize_topk_probability": True,
                "expert_group_count": 8, "topk_group_count": 4, "fused_shared_experts": 1,
                "routed_scaling_factor": model.routed_scaling_factor})
        weights, ids = self.trace.call_outputs(node, (logits, self.bias), outputs=(
            ("weights", (rows, model.num_experts_per_tok + 1), "fp32"),
            ("ids", (rows, model.num_experts_per_tok + 1), "int32")))
        counts = self.resolution.tokens_per_expert
        node = self.node(WorkloadKind.FUSED_MOE, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
            "intermediate": model.moe_intermediate_size // 8}, "bf16", "full", {
                "source_component": "sglang_0_5_19_symbolic_fused_shared_moe", "expert_parallel_size": 1,
                "expert_rank": 0, "owned_experts": tuple(range(model.num_experts)),
                "tokens_per_expert": counts, "local_tokens_per_expert": counts, "expert_placement_strategy": "linear",
                "moe_invocation_id": self.invocation, "output_is_reduced": False,
                "routed_scaling_factor": 1.0, "routed_weight_dtype": "fp8", "weight_group_size": 0,
                "fused_shared_experts": 1, "routing_provenance": dict(self.resolution.routing_provenance),
                "symbolic_workspace_complete": False})
        return self.trace.call(node, (hidden, weights, ids), shape=hidden.shape, dtype="bf16")


def trace_sglang_deepseek_v3_forward(model, step, pool, expert_routing, *, input_ids, positions,
        slot_mapping, cache_capacity_tokens, source_profile, linear_source_profile, mla_attention_source,
        use_deep_gemm_bmm, allreduce_fusion, fused_shared_experts, scale_routed_in_topk, grouped_topk_source, moe_runner_source, prefill_source=None, sampling_source=None, eagle_state=None,
        one_shot_kv_indices=None, max_kv_chunk_capacity=131072, graph_token_shape=None, graph_inputs=None,
        prefix_chunk_kv_indices=(), eagle_graph_capture_batch_size=None):
    """Full target model/logits in an explicit existing eager source branch.

    Optional plain greedy/EAGLE is traced, never numerically executed.
    IDs/positions/slots are actual caller metadata, while logical
    arenas are never native page/allocation observations.
    """
    config = pool.parallel_plan.config
    if (source_profile not in (SOURCE_PROFILE, SOURCE_FUSED_PROFILE, SOURCE_TARGET_PROFILE, SOURCE_TARGET_FUSED_PROFILE) or model.model_id != "deepseek-ai/DeepSeek-V3"
            or model.model_type != "deepseek_v3" or model.num_hidden_layers != 61
            or (model.compute_dtype, model.weight_dtype) != ("bf16", "fp8")
            or pool.hardware_id != "nvidia_h200" or config.tensor_parallel_size != 8
            or config.enable_expert_parallel or not config.is_rank_local
            or len(pool.ranks) != 8 or config.pipeline_parallel_size != 1):
        raise NotImplementedError("SGLang V3 source requires official FP8 H200 TP8 PP/SP/CP/DP1 EP-off")
    if (linear_source_profile != "logical_group128_fp32" or mla_attention_source != "flashmla_bf16"
            or use_deep_gemm_bmm is not False or type(allreduce_fusion) is not bool
            or allreduce_fusion != (source_profile in (SOURCE_FUSED_PROFILE, SOURCE_TARGET_FUSED_PROFILE))
            or type(fused_shared_experts) is not int or fused_shared_experts != 1
            or scale_routed_in_topk is not True or grouped_topk_source != "triton_grouped"
            or moe_runner_source != "cutlass_fp8"):
        raise NotImplementedError("SGLang source requires declared block128/FP32-scale linear, BF16 MLA BMM, unfused AR and top-k-scaled fused shared expert")
    target = source_profile in (SOURCE_TARGET_PROFILE, SOURCE_TARGET_FUSED_PROFILE)
    if sampling_source is not None and (not target or sampling_source != SAMPLING_SOURCE):
        raise NotImplementedError("SGLang sampling requires a target profile and its explicit plain FP32 greedy source")
    if (prefill_source not in (PREFILL_SOURCE, PREFILL_PREFIX_SOURCE) if target else prefill_source is not None):
        raise NotImplementedError("SGLang target requires its explicit BF16 ragged eager CP1 no-hook prefill source")
    eagle = eagle_state is not None
    eagle_capture = eagle_graph_capture_batch_size is not None
    if eagle_capture and (not isinstance(eagle_state, SGLangEagleChainState)
            or eagle_state.phase != "verify" or eagle_state.proposal_depth != 2
            or type(eagle_graph_capture_batch_size) is not int
            or eagle_graph_capture_batch_size < len(step.requests)
            or graph_token_shape is not None or graph_inputs is not None):
        raise NotImplementedError("EAGLE graph source requires exact real batch or declared padding, topk1/steps2 verify and separate stage extents")
    eagle_dummy_requests = 0 if not eagle_capture else eagle_graph_capture_batch_size - len(step.requests)
    if isinstance(eagle_state, SGLangEagleChainState) and eagle_state.target_graph_inputs is not None and not eagle_dummy_requests:
        raise ValueError("EAGLE retained graph metadata requires a padded staged capture")
    if eagle_dummy_requests:
        if (eagle_state.target_graph_inputs is None or eagle_state.extend_graph_inputs is None):
            raise NotImplementedError("EAGLE graph source requires exact real batch or explicit retained target/extend tails")
        if (len(eagle_state.target_graph_inputs.retained_padding_token_ids) != 3 * eagle_dummy_requests
                or any(token >= model.vocab_size for port in (eagle_state.target_graph_inputs, eagle_state.extend_graph_inputs)
                    for token in port.retained_padding_token_ids)):
            raise ValueError("EAGLE graph tails must cover the selected bucket within checkpoint vocabulary")
    if eagle:
        if not isinstance(eagle_state, SGLangEagleChainState):
            raise TypeError("SGLang EAGLE requires its typed explicit chain state")
        if (not target or sampling_source != SAMPLING_SOURCE
                or eagle_state.request_ids != tuple(r.request_id for r in step.requests)
                or model.mtp_architecture != "deepseek_mtp" or model.num_mtp_layers != 1):
            raise NotImplementedError("SGLang EAGLE requires official target/NextN requests with plain greedy")
        phases = {request.phase for request in step.requests}
        if phases not in ({"prefill"}, {"decode"}):
            raise NotImplementedError("SGLang EAGLE requires an unmixed prefill or decode batch")
        phase = "prefill" if phases == {"prefill"} else "verify"
        if (eagle_state.phase != phase or any(request.mtp_proposer_tokens != eagle_state.proposal_depth
                    or request.mtp_proposer_mode != "padded" for request in step.requests)
                or len(eagle_state.draft_slot_mapping) != sum(request.scheduled_tokens for request in step.requests)
                or eagle_state.draft_cache_capacity_tokens < sum(request.computed_tokens + request.scheduled_tokens for request in step.requests)
                or any(token >= model.vocab_size for values in (eagle_state.bonus_token_ids,
                    eagle_state.draft_token_ids, eagle_state.target_argmax_ids, eagle_state.draft_top1_ids) for token in values)):
            raise ValueError("SGLang EAGLE scenario disagrees with the scheduled request or model bounds")
        if phase == "prefill":
            if any(request.mtp_verified_tokens or request.mtp_proposed_tokens for request in step.requests):
                raise NotImplementedError("SGLang EAGLE prefill cannot contain a verify tree")
            prefix_tokens = sum(request.computed_tokens for request in step.requests)
            if prefix_tokens and (prefill_source != PREFILL_PREFIX_SOURCE or not 0 < prefix_tokens < 8192):
                raise NotImplementedError("SGLang EAGLE prefix prefill requires its explicit below8192 absorbed-MLA source")
            if prefix_tokens and eagle_state.prefill_tail_source not in (
                    "final_suffix_no_prompt_override", "nonfinal_chunk_next_prompt_token"):
                raise NotImplementedError("SGLang EAGLE prefix prefill requires an explicit final suffix or next-prompt-token override")
            if (eagle_state.prefill_next_prompt_token_ids is not None
                    and any(token is not None and token >= model.vocab_size
                        for token in eagle_state.prefill_next_prompt_token_ids)):
                raise ValueError("SGLang next prompt token must be within the checkpoint vocabulary")
            if (eagle_state.prefill_next_prompt_token_ids is not None
                    and any(token is not None and request.raw_model_output_tokens != 0
                        for request, token in zip(step.requests,
                            eagle_state.prefill_next_prompt_token_ids, strict=True))):
                raise ValueError("SGLang non-final chunk override requires an un-emitted target sample")
        else:
            width, depth = eagle_state.proposal_depth + 1, eagle_state.proposal_depth
            for row, request in enumerate(step.requests):
                if (request.scheduled_tokens != width or request.computed_tokens <= 0
                        or request.mtp_verified_tokens != depth or request.mtp_proposed_tokens != depth
                        or request.mtp_accepted_tokens + 1 != eagle_state.accept_lens[row]
                        or input_ids[row * width + 1:(row + 1) * width] != eagle_state.draft_token_ids[row * depth:(row + 1) * depth]):
                    raise ValueError("SGLang EAGLE verify must consume every explicitly declared chain and acceptance prefix")
    fresh = bool(step.requests) and all(r.phase == "prefill" and r.computed_tokens == 0
        and r.scheduled_tokens > 0 for r in step.requests)
    has_prefix = (prefill_source == PREFILL_PREFIX_SOURCE and bool(step.requests)
        and all(r.phase == "prefill" and r.scheduled_tokens > 0 for r in step.requests)
        and 0 < sum(r.computed_tokens for r in step.requests))
    long_prefix = has_prefix and sum(r.computed_tokens for r in step.requests) >= 8192
    if type(max_kv_chunk_capacity) is not int or max_kv_chunk_capacity <= 0:
        raise ValueError("SGLang maximum KV chunk capacity must be an explicit positive source value")
    chunked = long_prefix and sum(r.computed_tokens + r.scheduled_tokens for r in step.requests) > max_kv_chunk_capacity
    one_shot = long_prefix and not chunked
    chunk_lengths = ()
    if chunked:
        if one_shot_kv_indices is not None:
            raise NotImplementedError("SGLang over-capacity prefix requires explicit chunked source indices, not one-shot indices")
        width = max_kv_chunk_capacity // len(step.requests)
        if width <= 0:
            raise NotImplementedError("SGLang prefix capacity must provide at least one chunk row per request")
        contexts = tuple(request.computed_tokens for request in step.requests)
        chunk_lengths = tuple(tuple(max(0, min(width, context - start)) for context in contexts)
            for start in range(0, max(contexts), width))
        if (not isinstance(prefix_chunk_kv_indices, tuple) or len(prefix_chunk_kv_indices) != len(chunk_lengths)
                or any(not isinstance(indices, tuple) or len(indices) != sum(lengths)
                    or any(type(index) is not int or not 0 <= index < min(cache_capacity_tokens, 2**31) for index in indices)
                    for indices, lengths in zip(prefix_chunk_kv_indices, chunk_lengths, strict=True))):
            raise ValueError("SGLang chunked prefix requires every caller-declared INT32 chunk index within cache bounds")
    elif prefix_chunk_kv_indices:
        raise ValueError("SGLang chunked prefix indices require the over-capacity long-prefix source branch")
    if one_shot:
        lengths = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
        if (not isinstance(one_shot_kv_indices, tuple) or len(one_shot_kv_indices) != sum(lengths)
                or any(type(index) is not int or not 0 <= index < min(cache_capacity_tokens, 2**31)
                    for index in one_shot_kv_indices)):
            raise ValueError("SGLang one-shot needs explicit in-bounds INT32 prefix+suffix cache indices")
        cursor, offset = 0, 0
        for request, length in zip(step.requests, lengths):
            if one_shot_kv_indices[cursor + request.computed_tokens:cursor + length] != slot_mapping[offset:offset + request.scheduled_tokens]:
                raise ValueError("SGLang one-shot indices must end each request at its freshly updated suffix cache slots")
            cursor, offset = cursor + length, offset + request.scheduled_tokens
    elif one_shot_kv_indices is not None:
        raise ValueError("SGLang one-shot cache indices require the explicit long-prefix source branch")
    decode = bool(step.requests) and all(r.phase == "decode" and r.scheduled_tokens == 1
        and r.computed_tokens > 0 for r in step.requests)
    if (step.pool_name != pool.name or not (decode or target and (fresh or has_prefix) or eagle and eagle_state.phase == "verify")
            or (not eagle and any(r.mtp_proposed_tokens or r.mtp_verified_tokens or r.mtp_proposer_tokens
                or r.mtp_proposer_mode != "disabled" for r in step.requests))):
        raise NotImplementedError("SGLang target requires declared ordinary prefill/decode or an explicit EAGLE chain")
    q_lens = tuple(r.scheduled_tokens for r in step.requests)
    rows = sum(q_lens)
    if graph_token_shape is not None:
        from .graph_padding import GraphTokenShape
        if (not isinstance(graph_token_shape, GraphTokenShape) or graph_token_shape.runtime_mode != "FULL"
                or graph_token_shape.logical_tokens != rows or not decode or eagle):
            raise NotImplementedError("SGLang FULL capture currently requires ordinary single-token decode")
        if graph_token_shape.padding_tokens and not isinstance(graph_inputs, SGLangGraphInputs):
            raise NotImplementedError("SGLang padded capture requires retained input IDs and dummy slot0 cache-state semantics")
    if graph_inputs is not None:
        if (not isinstance(graph_inputs, SGLangGraphInputs) or graph_token_shape is None
                or graph_inputs.request_ids != tuple(request.request_id for request in step.requests)
                or len(graph_inputs.retained_padding_token_ids) != graph_token_shape.padding_tokens
                or any(token >= model.vocab_size for token in graph_inputs.retained_padding_token_ids)):
            raise ValueError("SGLang graph source inputs must match active requests, checkpoint vocabulary and exact padded tail")
    # communicator.apply_flashinfer_allreduce_fusion and
    # should_fuse_mlp_allreduce_with_next_layer both test the token-row
    # domain. Above2048 the same decoder executes ordinary AR plus norm.
    fusion_used = allreduce_fusion and rows <= 2048
    for name, values in (("input_ids", input_ids), ("positions", positions), ("slot_mapping", slot_mapping)):
        if not isinstance(values, tuple) or len(values) != rows or any(type(v) is not int for v in values):
            raise ValueError(f"SGLang {name} must contain one explicit integer per token row")
    kv_lens = tuple(r.computed_tokens + r.scheduled_tokens for r in step.requests)
    if (any(not 0 <= v < model.vocab_size for v in input_ids)
            or positions != tuple(p for r in step.requests for p in range(r.computed_tokens, r.computed_tokens + r.scheduled_tokens))
            or any(not 0 <= p < model.max_position_embeddings for p in positions)
            or type(cache_capacity_tokens) is not int or cache_capacity_tokens < sum(kv_lens)
            or len(set(slot_mapping)) != rows or any(not 0 <= s < cache_capacity_tokens for s in slot_mapping)):
        raise ValueError("SGLang IDs/positions/logical slots must cover valid explicit source contexts")
    logical_rows = rows
    if eagle_dummy_requests:
        if (cache_capacity_tokens < sum(kv_lens) + 64 or any(slot < 64 for slot in slot_mapping)
                or eagle_state.draft_cache_capacity_tokens < sum(kv_lens) + 64
                or any(slot < 64 for slot in (*eagle_state.draft_slot_mapping, *eagle_state.draft_decode_slot_mapping))):
            raise ValueError("EAGLE padded graphs require reserved64 dummy pages for target and draft history")
        input_ids += eagle_state.target_graph_inputs.retained_padding_token_ids
        positions += (0,) * (3 * eagle_dummy_requests)
        slot_mapping += (0,) * (3 * eagle_dummy_requests)
        q_lens += (3,) * eagle_dummy_requests
        kv_lens += (4,) * eagle_dummy_requests
        rows += 3 * eagle_dummy_requests
    padding = 0 if graph_token_shape is None else graph_token_shape.padding_tokens
    if padding:
        # FlashMLA PAGE_SIZE=64; PagedTokenToKVPoolAllocator starts at
        # page1, and MLATokenToKVPool allocates size+page_size. These are
        # source logical ranges, not a native page table/capacity receipt.
        if cache_capacity_tokens < sum(kv_lens) + 64 or any(slot < 64 for slot in slot_mapping):
            raise ValueError("SGLang padded graph requires a reserved64 dummy page disjoint from active logical history")
        input_ids = input_ids + graph_inputs.retained_padding_token_ids
        positions = positions + (0,) * padding
        slot_mapping = slot_mapping + (0,) * padding
        q_lens = q_lens + (1,) * padding
        kv_lens = kv_lens + (1,) * padding
        rows += padding
        # The graph backend reads dummy seq_lens=1 at reserved cache slot0.
        # This is a declared anonymous cache owner, never a real request row.
        fusion_used = allreduce_fusion and rows <= 2048
    traces = tuple(ForwardTrace(rank=rank, namespace=f"{step.step_id}:rank-{rank}") for rank in pool.ranks)
    group = ForwardCollectiveGroup(traces, namespace=step.step_id)
    group.source_profile = source_profile
    group.source_parameters = {"allreduce_fusion": allreduce_fusion, "allreduce_fusion_used": fusion_used,
        "use_deep_gemm_bmm": False, "max_kv_chunk_capacity": max_kv_chunk_capacity,
        "one_shot_index_producer_bound": False if one_shot else None,
        "prefix_chunk_index_producer_bound": False if chunked else None,
        "prefix_chunk_lengths": chunk_lengths,
        "linear_source_profile": linear_source_profile, "mla_attention_source": mla_attention_source,
        "fused_shared_experts": 1, "scale_routed_in_topk": True, "grouped_topk_source": grouped_topk_source,
        "moe_runner_source": moe_runner_source, "prefill_source": prefill_source, "sampling_source": sampling_source,
        "eagle_source": f"topk1_steps{eagle_state.proposal_depth}_eager" if eagle else None,
        "eagle_scope": "source_logical_roles_not_native_dynamic_traffic" if eagle else None,
        "eagle_prefill_prefix_tokens": sum(request.computed_tokens for request in step.requests)
            if eagle and eagle_state.phase == "prefill" else None,
        "eagle_prefill_tail_source": eagle_state.prefill_tail_source if eagle else None,
        "eagle_prefill_next_prompt_token_ids": eagle_state.prefill_next_prompt_token_ids if eagle else None,
        "eagle_graph_capture_batch_size": eagle_graph_capture_batch_size,
        "fuse_max_batch_size": 2048, "normal_capacity_accepted": False}
    group.source_parameters["eagle_dummy_requests"] = eagle_dummy_requests
    group.source_parameters["graph_padding_source"] = None if not padding else {
        "source_kind": graph_inputs.source_kind, "source_id": graph_inputs.source_id,
        "logical_rows": logical_rows, "model_input_rows": rows,
        "retained_padding_token_ids": graph_inputs.retained_padding_token_ids,
        "dummy_positions": 0, "dummy_cache_slot": 0, "dummy_seq_len": 1,
        "reserved_dummy_cache_rows": 64, "cache_layout_scope": "logical_arena_not_native_pages",
        "dummy_cache_slot_disjointness": "caller_declared_not_observed",
        "actual_runtime_bound": False}
    invocations = tuple(f"{step.step_id}:layer-{layer}:moe" for layer in range(model.num_hidden_layers))
    resolutions = _source_expert_resolutions(model, step, pool, expert_routing, invocations, rows)
    if eagle:
        from src.serving.schedule.schedule_step import schedule_step_sha256
        extend_pass = int(eagle_state.phase == "verify" and eagle_state.proposal_depth == 2)
        invocation = f"{step.step_id}:mtp-pass-{extend_pass}:moe"
        draft_resolution = expert_routing.resolve(step_id=step.step_id, layer_index=None,
            moe_invocation_id=invocation, schedule_step_sha256=schedule_step_sha256(step), tokens=rows,
            num_experts=model.num_experts, topk=model.num_experts_per_tok, model_config_sha256=model.config_sha256,
            pool_name=pool.name, tensor_parallel_size=8, expert_parallel_size=1, expert_rank_group=(pool.ranks[0],),
            expert_placement_strategy="linear", moe_layer_ordinal=sum(model.layer_uses_moe(i) for i in range(61)),
            assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
                if expert_routing.source_kind == "explicit_assumption" else None))
        if eagle_state.phase == "verify" and eagle_state.proposal_depth == 2:
            draft_decode_invocation = f"{step.step_id}:mtp-pass-0:moe"
            draft_decode_resolution = expert_routing.resolve(step_id=step.step_id, layer_index=None,
                moe_invocation_id=draft_decode_invocation, schedule_step_sha256=schedule_step_sha256(step), tokens=len(step.requests) + eagle_dummy_requests,
                num_experts=model.num_experts, topk=model.num_experts_per_tok, model_config_sha256=model.config_sha256,
                pool_name=pool.name, tensor_parallel_size=8, expert_parallel_size=1, expert_rank_group=(pool.ranks[0],),
                expert_placement_strategy="linear", moe_layer_ordinal=sum(model.layer_uses_moe(i) for i in range(61)),
                assumption_model_semantics_sha256=(model.expert_routing_assumption_semantics_sha256
                    if expert_routing.source_kind == "explicit_assumption" else None))
        group.eagle_source_state = eagle_state
    collective, gather = pool.collective_for("all_reduce"), pool.collective_for("all_gather")
    for trace in traces:
        rope = trace.tensor("rope-cache", (model.max_position_embeddings, model.qk_rope_head_dim), "bf16", persistent=True)
        draft_cache = None
        if eagle and eagle_state.phase == "verify":
            draft_tokens = None
            draft_tree_ports = None
            if eagle_state.proposal_depth == 2:
                draft_cache = trace.tensor("nextn:kv-cache", (eagle_state.draft_cache_capacity_tokens, 576), "bf16", persistent=True)
                draft_tokens, parent, scores = _eagle_draft_decode(trace, model, step, pool, group, collective, gather, eagle_state,
                    rope=rope, allreduce_fusion=allreduce_fusion and len(step.requests) <= 2048,
                    resolution=draft_decode_resolution, invocation=draft_decode_invocation, cache=draft_cache, capture=eagle_capture, capture_batch_size=eagle_graph_capture_batch_size)
                draft_tree_ports = (parent, scores)
            ids, pos, retrieve, tree_mask = _eagle_prepare_verify(trace, pool.name, eagle_state,
                tuple(request.computed_tokens for request in step.requests), input_ids[:logical_rows], draft_tokens=draft_tokens,
                draft_tree_ports=draft_tree_ports)
        else:
            ids = trace.tensor("input_ids", (rows,), "int64")
            pos = trace.tensor("positions", (rows,), "int64")
        slots = trace.tensor("slot_mapping", (logical_rows if eagle_dummy_requests else rows,), "int64", persistent=True)
        if eagle_dummy_requests:
            prep = SGLangEagleOps(trace, pool.name, proposal_depth=2, namespace="eagle-target-inputs")
            ids = _graph_input_buffer(prep, ids, rows, "eagle:target-graph-ids")
            pos = _graph_input_buffer(prep, pos, rows, "eagle:target-graph-positions", fill_value=0)
            slots = _graph_input_buffer(prep, slots, rows, "eagle:target-graph-slots", fill_value=0)
        if graph_token_shape is not None:
            trace.begin_capture(key=f"{source_profile}:FULL:{rows}", logical_tokens=logical_rows,
                model_input_tokens=rows, inputs={"input_ids": ids, "positions": pos, "slot_mapping": slots},
                padding_program="sglang_zero_slot0_seq1" if padding else "vllm_masked_minus1")
        elif eagle_capture:
            trace.begin_capture(key=f"{source_profile}:eagle-verify:{eagle_graph_capture_batch_size}", logical_tokens=logical_rows,
                model_input_tokens=rows, inputs={"input_ids": ids, "positions": pos, "slot_mapping": slots},
                source_stage="sglang_eagle_target_verify", source_graph_batch_size=eagle_graph_capture_batch_size,
                padding_program="sglang_zero_slot0_seq1" if eagle_dummy_requests else "vllm_masked_minus1")
        hidden = SGLangEmbedding(trace, "embedding", pool.name, model, group, collective, input_ids)(ids)
        one_shot_indices = trace.tensor("one-shot-kv-indices", (sum(kv_lens),), "int32") if one_shot else None
        chunk_index_tensors = tuple(trace.tensor(f"prefix-chunk-{index}-kv-indices", (len(indices),), "int32")
            for index, indices in enumerate(prefix_chunk_kv_indices))
        residual = None
        for layer in range(model.num_hidden_layers):
            prefix = f"layer-{layer}"
            input_norm = SGLangRMSNorm(trace, prefix + ":input-norm", pool.name, model.hidden_size, model.rms_norm_eps)
            if residual is None:
                residual, hidden = hidden, input_norm(hidden)
            elif fusion_used:
                hidden, residual = SGLangAllReduceRMSNorm(trace, prefix + ":input-fused-all-reduce", pool.name,
                    model.hidden_size, model.rms_norm_eps, group, collective)(hidden, residual)
            else:
                hidden, residual = input_norm(hidden, residual)
            cache = trace.tensor(prefix + ":kv-cache", (cache_capacity_tokens, 576), "bf16", persistent=True)
            if chunked:
                attention = SGLangChunkedMHAAttention(trace, model, pool.name, layer, group,
                    cache=cache, slots=slot_mapping, slot_tensor=slots, rope=rope, q_lens=q_lens,
                    chunk_lengths=chunk_lengths, chunk_indices=prefix_chunk_kv_indices, chunk_index_tensors=chunk_index_tensors)
            elif one_shot:
                attention = SGLangOneShotMHAAttention(trace, model, pool.name, layer, group,
                    cache=cache, slots=slot_mapping, slot_tensor=slots, rope=rope, q_lens=q_lens,
                    kv_lens=kv_lens, kv_indices=one_shot_kv_indices, kv_index_tensor=one_shot_indices,
                    max_kv_chunk_capacity=max_kv_chunk_capacity)
            elif fresh:
                attention = SGLangFreshMHAAttention(trace, model, pool.name, layer, group,
                    cache=cache, slots=slot_mapping, slot_tensor=slots, rope=rope, q_lens=q_lens)
            else:
                attention = SGLangDecodeAttention(trace, model, pool.name, layer, group,
                    cache=cache, slots=slot_mapping, slot_tensor=slots, rope=rope, kv_lens=kv_lens,
                    q_lens=q_lens if eagle or has_prefix else None, absorbed_prefill=has_prefix,
                    graph_dummy_rows=padding + 3 * eagle_dummy_requests)
            partial = attention(hidden, pos)
            if fusion_used:
                hidden, residual = SGLangAllReduceRMSNorm(trace, prefix + ":post-attn-fused-all-reduce", pool.name,
                    model.hidden_size, model.rms_norm_eps, group, collective)(partial, residual)
            else:
                reduced = group.call(prefix + ":attention-all-reduce", partial, pool_name=pool.name, config=collective,
                    inplace=False, implementation="sglang_custom_all_reduce_v2")
                hidden, residual = SGLangRMSNorm(trace, prefix + ":post-attn-norm", pool.name,
                    model.hidden_size, model.rms_norm_eps)(reduced, residual)
            if model.layer_uses_moe(layer):
                hidden = SGLangFusedSharedMoE(trace, pool.name, model, layer, resolutions[layer], invocations[layer])(hidden)
            else:
                mlp = DeepseekV2MLP(trace, model, pool.name, prefix + ":mlp", intermediate_size=model.intermediate_size,
                    tp_size=8, module_path=f"model.layers.{layer}.mlp", reduce_results=False)
                hidden = mlp(hidden)
            if not fusion_used or layer == model.num_hidden_layers - 1:
                hidden = group.call(prefix + ":mlp-all-reduce", hidden, pool_name=pool.name, config=collective,
                    inplace=False, implementation="sglang_custom_all_reduce_v2")
        hidden, residual = SGLangRMSNorm(trace, "final-norm", pool.name, model.hidden_size, model.rms_norm_eps)(hidden, residual)
        if eagle:
            target_hidden = hidden
        if fresh or has_prefix:
            hidden = _prefill_logits_hidden(trace, pool.name, hidden, q_lens)
        logits_rows = hidden.shape[0]
        local, _ = SymbolicLinear(trace, "lm-head", pool.name, model.hidden_size, model.vocab_size // 8,
            kind=WorkloadKind.LM_HEAD)(hidden)
        logits = group.call("logits-all-gather", local, pool_name=pool.name, config=gather, inplace=False, dim=-1)
        node = SymbolicOperator(trace, "logits-fp32", pool.name).node(WorkloadKind.VECTOR,
            {"elements": logits_rows * model.vocab_size, "rows": logits_rows, "vocab": model.vocab_size}, "bf16", "vllm_logits_fp32_cast", {
                "source_component": "sglang_0_5_19_symbolic_logits_boundary",
                "sampling_source": "sglang_0_5_19_logits_processor_bf16"})
        output = trace.call(node, (logits,), shape=(logits_rows, model.vocab_size), dtype="fp32")
        if graph_token_shape is not None:
            # DecodeCudaGraphRunner captures LogitsProcessor, then slices
            # next_token_logits[:raw_num_token] before the eager sampler.
            trace.end_capture(output)
            if padding:
                output = output.narrow(0, 0, logical_rows)
        elif eagle_capture:
            trace.end_capture(outputs={"next_token_logits": output, "hidden_states": target_hidden})
            if eagle_dummy_requests:
                output, target_hidden = output.narrow(0, 0, logical_rows), target_hidden.narrow(0, 0, logical_rows)
                ids, pos = ids.narrow(0, 0, logical_rows), pos.narrow(0, 0, logical_rows)
        if sampling_source is not None:
            output = SGLangGreedy(trace, "sampler-argmax", pool.name)(output)
        if eagle:
            ops = SGLangEagleOps(trace, pool.name, proposal_depth=eagle_state.proposal_depth)
            accepted_lens = None
            if eagle_state.phase == "prefill":
                # eagle_utils._eagle_prefill_tail_tokens clones the complete
                # tail vector for a non-final chunk, then fills that request's
                # next prompt token; the target sample stays separate.
                tail_tokens = output
                overrides = eagle_state.prefill_next_prompt_token_ids
                if overrides is not None:
                    tail_tokens = ops.copy(output, "prefill-tail-clone")
                    for row, token in enumerate(overrides):
                        if token is not None:
                            tail = tail_tokens.narrow(0, row, 1)
                            trace.call(ops.node("prefill-next-prompt-fill", "fill", "int64", 1, 1,
                                source_fill_value=token), (), shape=(1,), dtype="int64", out=tail)
                draft_ids = trace.tensor("eagle:prefill-new-input", (rows,), "int64")
                offset, declared_tokens = 0, []
                for row, length in enumerate(q_lens):
                    pieces = ((ids.narrow(0, offset + 1, length - 1), tail_tokens.narrow(0, row, 1))
                        if length > 1 else (tail_tokens.narrow(0, row, 1),))
                    shifted = trace.call(ops.node("prefill-shift-cat", "concat", "int64", length, 1),
                        pieces, shape=(length,), dtype="int64")
                    ops.copy(shifted, "prefill-input-copy", draft_ids.narrow(0, offset, length))
                    tail_id = (overrides[row] if overrides is not None and overrides[row] is not None
                        else eagle_state.bonus_token_ids[row])
                    declared_tokens.extend((*input_ids[offset + 1:offset + length], tail_id))
                    offset += length
                draft_tokens = tuple(declared_tokens)
            else:
                draft_ids, draft_tokens, accepted_lens = _eagle_verify_outputs(trace, pool.name, eagle_state,
                    ids, retrieve, output)
            carry = _trace_nextn_extend(trace, model, step, pool, group, collective, gather, eagle_state,
                ids=draft_ids, token_ids=draft_tokens, positions=pos, rope=rope, target_hidden=target_hidden,
                allreduce_fusion=fusion_used, resolution=draft_resolution, invocation=invocation,
                accepted_lens=accepted_lens, cache=draft_cache, capture=eagle_capture, capture_batch_size=eagle_graph_capture_batch_size)
    return group


def trace_sglang_deepseek_v3_decode(*args, **kwargs):
    """Compatibility entry for the original explicit decode-only profiles."""
    return trace_sglang_deepseek_v3_forward(*args, **kwargs)
