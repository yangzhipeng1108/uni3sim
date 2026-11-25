"""Shape inference for transformer GEMM layers.

This module converts model semantics into GEMM dimensions before workload
lowering. It has no dependency on profiler traces or predictor implementations.

Layer positions:
  qkv_proj    : Q+K+V fused projection with sharded Q and sharded/replicated KV
  fused_qkv_a_proj: vLLM MLA/DSA fused Q-A and KV-A projection (replicated);
                DeepSeek-V4 fused wqa_wkv projection (replicated)
  q_a_proj    : Transformers GLM DSA latent-Q compression (replicated)
  q_b_proj    : MLA/DSA latent-Q expansion, column-parallel; DeepSeek-V4 wq_b
  kv_a_proj_with_mqa: Transformers GLM DSA latent-KV compression (replicated)
  kv_b_proj   : MLA/DSA latent-KV expansion, column-parallel
  indexer_wq_b: GLM DSA indexer query expansion (replicated)
  indexer_wk  : GLM DSA indexer key projection (replicated)
  indexer_weights_proj: GLM DSA indexer head weights (replicated)
  indexer_wk_weights_proj: vLLM GLM DSA fused indexer key/weight projection
  o_proj      : output projection, row-parallel; DeepSeek-V4 wo_b
                (K = o_groups * o_lora_rank sharded over TP)
  gate_up_proj: fused gate+up (SwiGLU), column-parallel
  down_proj   : down projection, row-parallel
  lm_head     : vLLM padded vocabulary projection, sharded over TP ranks
"""

from .model_config import ModelConfig


VLLM_VOCAB_PADDING_SIZE = 64


def padded_vocab_size(vocab_size: int) -> int:
    """Apply the pinned vLLM 0.11.0 vocabulary padding contract."""

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return (
        (vocab_size + VLLM_VOCAB_PADDING_SIZE - 1)
        // VLLM_VOCAB_PADDING_SIZE
        * VLLM_VOCAB_PADDING_SIZE
    )


def _kimi_k3_gemm_dimensions(model, position, rows, tp_size, tail_tier):
    """Ordinary SM90, SP-off K3 source constructor/projection dimensions.

    Shapes are logical matrices before backend-specific expert packing. The
    KDA input's own 16-element row padding is part of its actual constructor;
    MXFP4 ordinary linears use UnquantizedLinearMethod, not ModelOpt FP8.
    A TP>1 latent up-projection needs an explicit LatentMoERunner tail tier;
    token count alone cannot bind environment/stream/backend selection. Those
    tiers require the unreduced-MoE/shared-expert fused tail predicate. Tier2
    uses a view of the replicated weight and addmm_(beta=1) into the shared
    output; its smaller N does not imply sharded parameter residency or a
    beta=0 GEMM. Shape inference does not price that epilogue or select a tier.
    """
    source = model.kimi_k3_semantics
    if model.model_type != "kimi_linear" or source is None:
        raise ValueError("K3 projection dimensions require exact hybrid source semantics")
    if tail_tier not in (None, "ALLREDUCE_OVERLAP", "COLUMN_PARALLEL"):
        raise NotImplementedError("K3 source dimensions support explicit ordinary latent tail tiers only")
    if tail_tier is not None and position != "routed_expert_up_proj":
        raise ValueError("K3 latent tail tier applies only to the routed up-projection")

    def shard(value, role):
        if value % tp_size:
            raise ValueError(f"K3 {role} must divide evenly over TP")
        return value // tp_size

    h = model.hidden_size
    projection = source.kda_num_heads * source.kda_head_dim
    if position == "in_proj_qkvgfab":
        local_heads = shard(source.kda_num_heads, "KDA heads")
        local_projection = local_heads * source.kda_head_dim
        raw = 4 * local_projection + source.kda_head_dim + local_heads
        return rows, -(-raw // 16) * 16, h
    if position == "f_b_proj":
        return rows, shard(projection, "KDA projection"), source.kda_head_dim
    if position == "fused_qkv_a_g_proj":
        if not source.mla_use_output_gate:
            raise NotImplementedError("K3 merged gated projection requires its output gate")
        local_heads = shard(model.num_attention_heads, "MLA heads")
        # KimiK3MergedQKVGateLinear inflates the first two constructor
        # shards by TP so that MergedColumnParallelLinear retains them.
        n = model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim
        return rows, n + local_heads * model.v_head_dim, h
    if position == "q_b_proj":
        return rows, shard(model.num_attention_heads, "MLA heads") * model.head_dim, model.q_lora_rank
    if position == "kv_b_proj":
        return (rows, shard(model.num_attention_heads, "MLA heads")
                * (model.qk_nope_head_dim + model.v_head_dim), model.kv_lora_rank)
    if position == "o_proj":
        # The exact checkpoint has equal KDA head width and MLA value width.
        # Keep this checked rather than using one branch as a fallback.
        if projection != model.num_attention_heads * model.v_head_dim:
            raise NotImplementedError("K3 O projection requires an explicit attention-layer role for unequal widths")
        return rows, h, shard(projection, "attention output")
    if position in ("gate_up_proj", "down_proj"):
        intermediate = shard(model.intermediate_size, "dense intermediate")
        return ((rows, 2 * intermediate, h) if position == "gate_up_proj"
                else (rows, h, intermediate))
    if position in ("shared_experts.gate_up_proj", "shared_experts.down_proj"):
        intermediate = shard(model.num_shared_experts * model.moe_intermediate_size, "shared intermediate")
        return ((rows, 2 * intermediate, h) if position.endswith("gate_up_proj")
                else (rows, h, intermediate))
    if position in ("experts.w13_weight", "experts.w2_weight"):
        # KimiMoE uses an explicit source floor of 256 intermediate values
        # per TP rank (EP-off). Later backend padding/packing is separate.
        intermediate = max(shard(model.moe_intermediate_size, "routed intermediate"), 256)
        latent = source.routed_expert_hidden_size
        return ((rows, 2 * intermediate, latent) if position == "experts.w13_weight"
                else (rows, latent, intermediate))
    if position == "routed_expert_down_proj":
        return rows, source.routed_expert_hidden_size, h
    if position == "routed_expert_up_proj":
        if tp_size == 1:
            if tail_tier is not None:
                raise ValueError("K3 TP1 uses the native replicated up-projection, not a fused TP tail tier")
            return rows, h, source.routed_expert_hidden_size
        if tail_tier is None:
            raise NotImplementedError("K3 TP>1 latent up-projection requires an explicit source tail tier")
        n = h if tail_tier == "ALLREDUCE_OVERLAP" else shard(h, "latent tail output")
        return rows, n, source.routed_expert_hidden_size
    if position == "router":
        return rows, model.num_experts, h
    if position == "lm_head":
        return rows, shard(padded_vocab_size(model.vocab_size), "padded vocabulary"), h
    raise NotImplementedError(f"K3 source does not use an unqualified {position!r} projection")


def infer_gemm_dimensions(model: ModelConfig, layer_position: str,
                           seq_len: int, batch_size: int = 1,
                           tp_size: int = 1, *, kimi_k3_tail_tier: str | None = None):
    """Return (M, N, K) for a named GEMM position given model config.

    TP=N sharding:
      qkv_proj: Q heads are sharded; KV heads are sharded or replicated
      Column-parallel (gate_up_proj): N divided by tp_size
      Row-parallel    (o_proj, down_proj):      K divided by tp_size
      lm_head: padded vocabulary N divided by tp_size

    Args:
        model: ModelConfig from load_model_config()
        layer_position: one of qkv_proj / fused_qkv_a_proj / q_a_proj /
                        q_b_proj / kv_a_proj_with_mqa / kv_b_proj /
                        indexer_wq_b / indexer_wk / indexer_weights_proj /
                        o_proj / gate_up_proj / down_proj / lm_head
        seq_len: sequence length
        batch_size: batch size
        tp_size: tensor parallelism factor
        kimi_k3_tail_tier: explicitly declared ordinary latent-MoE source tier;
                           required for K3 TP>1 up-projection dimensions only

    Returns:
        (M, N, K) tuple of integers

    Raises:
        ValueError: if tp_size does not divide the sharded dimension, or layer
                    position is unknown.
    """
    if isinstance(tp_size, bool) or not isinstance(tp_size, int):
        raise TypeError("tp_size must be an integer")
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if model.attention_layout == "hybrid_kda_mla":
        if any(type(value) is not int or value <= 0 for value in (seq_len, batch_size)):
            raise ValueError("K3 projection token and batch counts must be positive integers")
        return _kimi_k3_gemm_dimensions(
            model, layer_position, seq_len * batch_size, tp_size, kimi_k3_tail_tier)
    if kimi_k3_tail_tier is not None:
        raise ValueError("K3 latent tail tier cannot be applied to another model")

    H = model.hidden_size
    I = model.intermediate_size
    HQ = model.num_attention_heads
    D = model.head_dim
    M = batch_size * seq_len

    if layer_position == "qkv_proj":
        if HQ % tp_size:
            raise ValueError(
                "qkv_proj query heads must be divisible by tp_size: "
                f"{HQ} vs {tp_size}")
        local_query_heads = HQ // tp_size
        local_kv_heads = model.kv_heads_per_tensor_parallel_rank(tp_size)
        N = (local_query_heads + 2 * local_kv_heads) * D
        K = H
    elif layer_position == "fused_qkv_a_proj":
        if model.attention_layout not in ("mla", "dsa_mla", "dsv4_sparse_mla"):
            raise ValueError(
                f"{layer_position} requires the pinned MLA/DSA vLLM "
                "execution contract")
        # Both pinned MLA vLLM implementations use disable_tp=True for the
        # merged Q-A and KV-A projection, so every TP rank sees the full
        # output width.  DeepSeek-V4 fuses wq_a and wkv into the replicated
        # fused_wqa_wkv projection (N = q_lora_rank + head_dim).
        if model.attention_layout == "dsv4_sparse_mla":
            N = model.q_lora_rank + model.head_dim
        else:
            N = (
                model.q_lora_rank
                + model.kv_lora_rank
                + model.qk_rope_head_dim
            )
        K = H
    elif layer_position in (
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
            "indexer_wq_b", "indexer_wk", "indexer_weights_proj",
            "indexer_wk_weights_proj"):
        if layer_position in (
                "q_a_proj", "kv_a_proj_with_mqa", "indexer_wk",
                "indexer_wk_weights_proj"):
            allowed_layouts = ("dsa_mla",)
        elif layer_position == "kv_b_proj":
            allowed_layouts = ("mla", "dsa_mla")
        elif layer_position in ("indexer_wq_b", "indexer_weights_proj"):
            allowed_layouts = ("dsa_mla", "dsv4_sparse_mla")
        elif layer_position == "q_b_proj":
            allowed_layouts = ("mla", "dsa_mla", "dsv4_sparse_mla")
        else:
            raise AssertionError(
                f"unhandled MLA/DSA projection: {layer_position!r}")
        if model.attention_layout not in allowed_layouts:
            raise ValueError(
                f"{layer_position} does not support attention layout "
                f"{model.attention_layout!r}")
        if layer_position == "q_a_proj":
            N = model.q_lora_rank
            K = H
        elif layer_position == "q_b_proj":
            if model.attention_layout == "dsv4_sparse_mla":
                total_n = HQ * model.head_dim
            else:
                total_n = HQ * (
                    model.qk_nope_head_dim + model.qk_rope_head_dim)
            if total_n % tp_size:
                raise ValueError(
                    "MLA q_b_proj output must be divisible by tp_size: "
                    f"{total_n} vs {tp_size}")
            N = total_n // tp_size
            K = model.q_lora_rank
        elif layer_position == "kv_a_proj_with_mqa":
            N = model.kv_lora_rank + model.qk_rope_head_dim
            K = H
        elif layer_position == "kv_b_proj":
            total_n = HQ * (
                model.qk_nope_head_dim + model.v_head_dim)
            if total_n % tp_size:
                raise ValueError(
                    "MLA kv_b_proj output must be divisible by tp_size: "
                    f"{total_n} vs {tp_size}")
            N = total_n // tp_size
            K = model.kv_lora_rank
        elif layer_position == "indexer_wq_b":
            N = model.index_n_heads * model.index_head_dim
            K = model.q_lora_rank
        elif layer_position == "indexer_wk":
            N = model.index_head_dim
            K = H
        elif layer_position == "indexer_weights_proj":
            N = model.index_n_heads
            K = H
        elif layer_position == "indexer_wk_weights_proj":
            N = model.index_head_dim + model.index_n_heads
            K = H
        else:
            raise AssertionError(
                f"unhandled MLA/DSA projection: {layer_position!r}")
    elif layer_position == "o_proj":
        if model.attention_layout == "standard_gqa":
            total_k = HQ * D
        elif model.attention_layout in ("mla", "dsa_mla"):
            total_k = HQ * model.v_head_dim
        elif model.attention_layout == "dsv4_sparse_mla":
            # DeepSeek-V4 wo_b is row-parallel over the grouped LoRA output
            # (K = o_groups * o_lora_rank), after the batched wo_a expansion.
            total_k = model.o_groups * model.o_lora_rank
        else:
            raise AssertionError(
                f"unhandled attention layout: {model.attention_layout!r}")
        if total_k % tp_size != 0:
            raise ValueError(
                f"o_proj K={total_k} not divisible by tp_size={tp_size}")
        N = H
        K = total_k // tp_size
    elif layer_position == "gate_up_proj":
        total_n = 2 * I
        if total_n % tp_size != 0:
            raise ValueError(
                f"gate_up_proj N={total_n} not divisible by tp_size={tp_size}")
        N = total_n // tp_size
        K = H
    elif layer_position == "down_proj":
        if I % tp_size != 0:
            raise ValueError(
                f"down_proj K={I} not divisible by tp_size={tp_size}")
        N = H
        K = I // tp_size
    elif layer_position == "lm_head":
        padded_vocab = padded_vocab_size(model.vocab_size)
        if padded_vocab % tp_size != 0:
            raise ValueError(
                "padded lm_head vocabulary must be divisible by tp_size: "
                f"{padded_vocab} vs {tp_size}")
        N = padded_vocab // tp_size
        K = H
    elif layer_position == "router":
        if model.num_experts <= 0:
            raise ValueError("router position requires a MoE model")
        N = model.num_experts
        K = H
    else:
        raise ValueError(f"Unknown layer_position: '{layer_position}'")

    return M, N, K
