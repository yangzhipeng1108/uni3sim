"""Tensor-parallel plan rules."""

from dataclasses import replace


def validate_gqa_parallelism(model, config):
    """Validate physical GQA/MLP geometry, independently of provider support.

    QKVParallelLinear partitions Q heads and either partitions or replicates
    KV heads. EP reuses the stage's TP/DP ranks; it does not add an axis.
    Backend/source support and available memory are separate gates.
    """
    if model.attention_layout != "standard_gqa":
        raise ValueError("GQA parallel geometry requires standard_gqa attention")
    if config.context_parallel_size != 1:
        raise ValueError("GQA source geometry requires PCP1")
    tp = config.tensor_parallel_size
    for name, extent in (("query heads", model.num_attention_heads),
                         ("hidden size", model.hidden_size)):
        if extent % tp:
            raise ValueError(f"{name} must be divisible by TP")
    model.kv_heads_per_tensor_parallel_rank(tp)
    if (-(-model.vocab_size // 64) * 64) % tp:
        raise ValueError("64-row padded vocabulary must divide the TP group")
    if config.pipeline_parallel_size > model.num_hidden_layers:
        raise ValueError("pipeline stages must each own a decoder layer")
    if config.enable_expert_parallel and not model.num_experts:
        raise ValueError("expert parallelism requires a MoE model")
    if model.num_experts:
        divisor = 1 if config.enable_expert_parallel else tp * config.data_parallel_size
        if model.moe_intermediate_size % divisor:
            raise ValueError("expert intermediate size must divide its tensor shards")
    elif model.intermediate_size % tp:
        raise ValueError("intermediate size must be divisible by TP")


def gqa_local_shape(model, plan, logical_rank):
    """Return local linear K/N and ownership from the actual rank plan.

    This is a geometry interface for source adapters and launch consumers,
    not a claim of executable source, native launch or performance accuracy.
    Shapes use (K, N), excluding the dynamic token/batch dimension.
    """
    validate_gqa_parallelism(model, plan.config)
    if plan.num_layers != model.num_hidden_layers or plan.num_experts != model.num_experts:
        raise ValueError("parallel plan layer/expert counts disagree with model")
    group = plan.rank_groups.for_rank("tensor", logical_rank)
    lane = group.index(logical_rank)
    stage = plan.rank_groups.for_rank("pipeline", logical_rank).index(logical_rank)
    layers = tuple(i for i, owner in enumerate(plan.layer_to_pipeline_stage) if owner == stage)
    tp = len(group)
    heads = model.num_attention_heads // tp
    kv_heads = model.kv_heads_per_tensor_parallel_rank(tp)
    kv_replication = max(1, tp // model.num_key_value_heads)
    kv_start = lane * kv_heads if kv_replication == 1 else lane // kv_replication
    expert_group = plan.rank_groups.for_rank("expert", logical_rank)
    expert_lane = expert_group.index(logical_rank)
    experts = tuple(i for i, owner in enumerate(plan.expert_to_shard) if owner == expert_lane)
    intermediate = (model.moe_intermediate_size // (1 if plan.config.enable_expert_parallel else
        tp * plan.config.data_parallel_size) if model.num_experts else model.intermediate_size // tp)
    return {
        "logical_rank": logical_rank, "pipeline_stage": stage, "layers": layers,
        "tensor_group": group, "expert_group": expert_group,
        "query_heads": heads, "kv_heads": kv_heads, "kv_replication": kv_replication,
        "kv_head_range": (kv_start, kv_start + kv_heads),
        "expert_ids": experts,
        "mlp_scope": "routed_expert" if model.num_experts else "dense",
        "embedding_rows": -(-model.vocab_size // 64) * 64 // tp,
        "lm_head_kn": (model.hidden_size, -(-model.vocab_size // 64) * 64 // tp),
        "qkv_kn": (model.hidden_size, (heads + 2 * kv_heads) * model.head_dim),
        "attention_output_kn": (heads * model.head_dim, model.hidden_size),
        "gate_up_kn": (model.hidden_size, 2 * intermediate),
        "down_kn": (intermediate, model.hidden_size),
    }


def validate_mla_parallelism(model, config):
    """Unpadded ordinary MLA/DSA geometry, separate from source execution.

    Latent caches and low-rank input projections replicate within TP; query
    heads, expanded KV projections and output projections shard by heads.
    V4 compressed MLA and hybrid KDA require their own source geometry.
    """
    if (model.attention_layout not in ("mla", "dsa_mla")
            or model.model_type not in ("kimi_k2", "glm_moe_dsa", "deepseek_v3")):
        raise ValueError("ordinary MLA geometry requires K2, GLM DSA or DeepSeek V3")
    tp = config.tensor_parallel_size
    if config.context_parallel_size != 1:
        raise ValueError("ordinary MLA geometry currently requires PCP1")
    if model.num_attention_heads % tp or model.hidden_size % tp:
        raise ValueError("MLA query heads and hidden size must divide TP")
    if model.q_lora_rank is None or min(model.kv_lora_rank, model.qk_rope_head_dim) <= 0:
        raise ValueError("ordinary MLA geometry requires low-rank query/KV and RoPE dimensions")
    if config.pipeline_parallel_size > model.num_hidden_layers:
        raise ValueError("pipeline stages must each own a decoder layer")
    if (-(-model.vocab_size // 64) * 64) % tp or model.intermediate_size % tp:
        raise ValueError("MLA vocabulary and dense intermediate must divide TP")
    if config.enable_expert_parallel and not model.num_experts:
        raise ValueError("expert parallelism requires a MoE model")
    expert_tp = 1 if config.enable_expert_parallel else tp * config.data_parallel_size
    if model.moe_intermediate_size % expert_tp:
        raise ValueError("MLA expert intermediate must divide expert TP")
    quant_group = model.weight_group_size if model.weight_dtype == "mixed_int4_bf16" else (
        128 if model.weight_dtype == "fp8" else 1)
    if model.moe_intermediate_size // expert_tp % quant_group:
        raise ValueError("unpadded MLA expert shards must retain complete quantization groups")


def mla_local_shape(model, plan, logical_rank):
    """Ordinary MLA local K/N and replicated latent-cache ownership.

    GLM0.30 DP+EP uses full shared weights with internal MoE SP. Model
    SP also replicates dense MLP weights in PP1; PP disables only model
    SP. V3/K2 also replicate SP shared weights, while their dense prefix keeps
    ordinary TP shards. Shapes alone do not accept a provider or capacity.
    """
    from src.serving.workload.model_config import is_glm_fp8_tensor_model_scope
    validate_mla_parallelism(model, plan.config)
    if plan.num_layers != model.num_hidden_layers or plan.num_experts != model.num_experts:
        raise ValueError("parallel plan layer/expert counts disagree with model")
    group = plan.rank_groups.for_rank("tensor", logical_rank)
    lane = group.index(logical_rank)
    stage = plan.rank_groups.for_rank("pipeline", logical_rank).index(logical_rank)
    layers = tuple(layer for layer, owner in enumerate(plan.layer_to_pipeline_stage) if owner == stage)
    expert_group = plan.rank_groups.for_rank("expert", logical_rank)
    expert_lane = expert_group.index(logical_rank)
    experts = tuple(expert for expert, owner in enumerate(plan.expert_to_shard) if owner == expert_lane)
    tp = len(group)
    heads = model.num_attention_heads // tp
    expert_tp = 1 if plan.config.enable_expert_parallel else tp * plan.config.data_parallel_size
    local_intermediate = model.moe_intermediate_size // expert_tp
    moe_sp = (model.model_type == "glm_moe_dsa" and plan.config.enable_expert_parallel
        and plan.config.data_parallel_size > 1 and tp > 1
        and (plan.config.sequence_parallel_size == tp
            or plan.config.sequence_parallel_size == 1
                and (model.weight_dtype == "bf16" or is_glm_fp8_tensor_model_scope(model, tp))))
    mla_moe_sp = (model.model_type in ("deepseek_v3", "kimi_k2") and plan.config.enable_expert_parallel
        and plan.config.data_parallel_size > 1 and tp > 1 and plan.config.sequence_parallel_size == 1)
    moe_sp = moe_sp or mla_moe_sp
    model_sp = moe_sp and plan.config.pipeline_parallel_size == 1
    shared_tp = 1 if moe_sp else tp
    dense_tp = 1 if model_sp and not mla_moe_sp else tp
    return {"logical_rank": logical_rank, "pipeline_stage": stage, "layers": layers,
        "tensor_group": group, "expert_group": expert_group, "expert_ids": experts,
        "query_heads": heads, "query_head_range": (lane * heads, (lane + 1) * heads),
        "latent_cache_elements_per_token_per_layer": model.kv_lora_rank + model.qk_rope_head_dim,
        "latent_cache_tp_replication": tp, "expert_tensor_parallel_size": expert_tp,
        "shared_tensor_parallel_size": shared_tp, "dense_tensor_parallel_size": dense_tp,
        "model_sequence_parallel": model_sp, "moe_sequence_parallel": moe_sp,
        "q_a_kn": (model.hidden_size, model.q_lora_rank),
        "kv_a_kn": (model.hidden_size, model.kv_lora_rank + model.qk_rope_head_dim),
        "q_b_kn": (model.q_lora_rank, heads * (model.qk_nope_head_dim + model.qk_rope_head_dim)),
        "kv_b_kn": (model.kv_lora_rank, heads * (model.qk_nope_head_dim + model.v_head_dim)),
        "attention_output_kn": (heads * model.v_head_dim, model.hidden_size),
        "gate_up_kn": (model.hidden_size, 2 * local_intermediate),
        "down_kn": (local_intermediate, model.hidden_size),
        "dense_gate_up_kn": (model.hidden_size, 2 * model.intermediate_size // dense_tp),
        "dense_down_kn": (model.intermediate_size // dense_tp, model.hidden_size),
        "shared_gate_up_kn": (model.hidden_size, 2 * model.num_shared_experts * model.moe_intermediate_size // shared_tp),
        "shared_down_kn": (model.num_shared_experts * model.moe_intermediate_size // shared_tp, model.hidden_size),
        "lm_head_kn": (model.hidden_size, -(-model.vocab_size // 64) * 64 // tp)}


def validate_k3_parallelism(model, config):
    """Hybrid KDA/MLA and whole logical Marlin tiles, not source acceptance."""
    if model.attention_layout != "hybrid_kda_mla" or model.kimi_k3_semantics is None:
        raise ValueError("K3 geometry requires exact hybrid KDA/MLA semantics")
    source, tp = model.kimi_k3_semantics, config.tensor_parallel_size
    if config.context_parallel_size != 1:
        raise ValueError("K3 shard geometry currently requires PCP1")
    if (source.kda_num_heads % tp or model.num_attention_heads % tp
            or model.hidden_size % tp or model.intermediate_size % tp
            or model.num_shared_experts * model.moe_intermediate_size % tp
            or (-(-model.vocab_size // 64) * 64) % tp):
        raise ValueError("K3 attention, dense/shared MLP and vocabulary must divide TP")
    if config.pipeline_parallel_size > model.num_hidden_layers:
        raise ValueError("K3 pipeline stages must each own a decoder layer")
    expert_tp = 1 if config.enable_expert_parallel else tp * config.data_parallel_size
    if (model.moe_intermediate_size % expert_tp
            or model.moe_intermediate_size // expert_tp % 64
            or source.routed_expert_hidden_size % model.weight_group_size):
        raise ValueError("unpadded K3 Marlin experts require whole local tiles and quantization groups")


def k3_local_shape(model, plan, logical_rank):
    """K3 rank-local projections and typed hybrid cache payloads.

    The explicit source entry supports ordinary TP or stage-local EP with
    nonempty PP. Geometry alone does not admit Scheduler/provider/pricing.
    """
    from src.serving.workload.shape_inference import infer_gemm_dimensions
    validate_k3_parallelism(model, plan.config)
    if plan.num_layers != model.num_hidden_layers or plan.num_experts != model.num_experts:
        raise ValueError("parallel plan layer/expert counts disagree with model")
    source = model.kimi_k3_semantics
    group = plan.rank_groups.for_rank("tensor", logical_rank)
    lane, tp = group.index(logical_rank), len(group)
    stage = plan.rank_groups.for_rank("pipeline", logical_rank).index(logical_rank)
    layers = tuple(layer for layer, owner in enumerate(plan.layer_to_pipeline_stage) if owner == stage)
    expert_group = plan.rank_groups.for_rank("expert", logical_rank)
    expert_lane = expert_group.index(logical_rank)
    experts = tuple(expert for expert, owner in enumerate(plan.expert_to_shard) if owner == expert_lane)
    heads, kda_heads = model.num_attention_heads // tp, source.kda_num_heads // tp
    projection = kda_heads * source.kda_head_dim
    input_rows = 4 * projection + source.kda_head_dim + kda_heads
    expert_tp = 1 if plan.config.enable_expert_parallel else tp * plan.config.data_parallel_size
    model_sp = (plan.config.enable_expert_parallel and tp > 1
        and plan.config.data_parallel_size > 1 and plan.config.pipeline_parallel_size == 1)
    mlp_tp = 1 if model_sp else tp
    logical_intermediate = model.moe_intermediate_size // expert_tp
    if plan.config.enable_expert_parallel:
        intermediate = logical_intermediate
    else:
        _, _, tensor_intermediate = infer_gemm_dimensions(model, "experts.w2_weight", 1, tp_size=tp)
        if tensor_intermediate % plan.config.data_parallel_size:
            raise ValueError("K3 source-padded expert intermediate must divide DP")
        intermediate = tensor_intermediate // plan.config.data_parallel_size
    payload = model.kimi_k3_source_cache_payload(tensor_parallel_size=tp,
        kda_state_slots=1, mla_tokens=1, mamba_cache_dtype="auto", mamba_ssm_cache_dtype="auto",
        mla_cache_dtype="auto", use_kda_recoverssm=False)
    kda_layers = source.layer_attention_types.count("kda")
    mla_layers = source.layer_attention_types.count("mla_nope")
    return {"logical_rank": logical_rank, "pipeline_stage": stage, "layers": layers,
        "layer_attention_types": tuple(source.layer_attention_types[layer] for layer in layers),
        "tensor_group": group, "expert_group": expert_group, "expert_ids": experts,
        "query_heads": heads, "query_head_range": (lane * heads, (lane + 1) * heads),
        "kda_heads": kda_heads, "kda_head_range": (lane * kda_heads, (lane + 1) * kda_heads),
        "kda_in_proj_kn": (model.hidden_size, -(-input_rows // 16) * 16),
        "kda_output_kn": (projection, model.hidden_size),
        "mla_qkv_a_gate_kn": (model.hidden_size,
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim + heads * model.v_head_dim),
        "mla_q_b_kn": (model.q_lora_rank, heads * (model.qk_nope_head_dim + model.qk_rope_head_dim)),
        "mla_output_kn": (heads * model.v_head_dim, model.hidden_size),
        "routed_gate_up_kn": (source.routed_expert_hidden_size, 2 * intermediate),
        "routed_down_kn": (intermediate, source.routed_expert_hidden_size),
        "logical_routed_gate_up_kn": (source.routed_expert_hidden_size, 2 * logical_intermediate),
        "logical_routed_down_kn": (logical_intermediate, source.routed_expert_hidden_size),
        "expert_tensor_parallel_size": expert_tp,
        "model_sequence_parallel": model_sp, "moe_internal_sequence_parallel": False,
        "dense_tensor_parallel_size": mlp_tp, "shared_tensor_parallel_size": mlp_tp,
        "dense_gate_up_kn": (model.hidden_size, 2 * model.intermediate_size // mlp_tp),
        "dense_down_kn": (model.intermediate_size // mlp_tp, model.hidden_size),
        "shared_gate_up_kn": (model.hidden_size, 2 * model.moe_intermediate_size * model.num_shared_experts // mlp_tp),
        "shared_down_kn": (model.moe_intermediate_size * model.num_shared_experts // mlp_tp, model.hidden_size),
        "kda_conv_state_bytes_per_slot_per_layer": payload["kda_convolution_state"] // kda_layers,
        "kda_recurrent_state_bytes_per_slot_per_layer": payload["kda_recurrent_state"] // kda_layers,
        "mla_cache_bytes_per_token_per_layer": payload["mla_latent_cache"] // mla_layers,
        "kda_conv_state_dtype": "bf16", "kda_recurrent_state_dtype": "fp32", "mla_cache_dtype": "bf16",
        "attention_residual_blocks_at_stage_end": -(-(layers[-1] + 1) // source.attention_residual_block_size),
        "lm_head_kn": (model.hidden_size, -(-model.vocab_size // 64) * 64 // tp)}


def validate_dspark_parallelism(model, config):
    """Standalone modular draft geometry; target TP/PP is a separate plan.

    SpeculativeConfig.create_draft_parallel_config forces PP1; each draft
    TP rank constructs all three appended layers. This scope describes one
    DP replica, ordinary modular experts, and no sequence/context sharding.
    It does not admit a draft model, provider, or memory budget for execution.
    """
    if not model.is_dspark_draft:
        raise ValueError("DSpark geometry requires the complete official draft metadata")
    if (model.model_type != "deepseek_v4" or model.attention_layout != "dsv4_sparse_mla"
            or model.compute_dtype != "bf16" or model.weight_dtype != "fp8" or model.expert_dtype != "fp4"
            or config.pipeline_parallel_size != 1 or config.data_parallel_size != 1
            or config.context_parallel_size != 1 or config.sequence_parallel_size != 1):
        raise ValueError("DSpark draft geometry requires standalone modular TP/EP, PP1/DP1/CP1/SP1")
    tp = config.tensor_parallel_size
    expert_tp = 1 if config.enable_expert_parallel else tp
    if (model.o_groups % tp or model.num_attention_heads % model.o_groups or model.num_attention_heads % tp
            or model.hidden_size % tp or (-(-model.vocab_size // 64) * 64) % tp
            or not 0 < model.num_attention_heads // tp <= 128
            or model.moe_intermediate_size % expert_tp or model.moe_intermediate_size // expert_tp % 32
            or model.hidden_size % 256 or model.num_shared_experts * model.moe_intermediate_size % 128
            or config.enable_expert_parallel and (tp < 2 or model.num_experts % tp)):
        raise ValueError("DSpark TP/EP requires exact grouped-output heads, vocabulary and whole quantization groups")


def validate_dspark_data_parallelism(model, config):
    """Modular draft DP geometry; target PP groups are a separate placement."""
    if config.data_parallel_size <= 1:
        raise ValueError("DSpark data-parallel source requires DP greater than one")
    local = replace(config, data_parallel_size=1,
        enable_expert_parallel=config.enable_expert_parallel and config.tensor_parallel_size > 1)
    validate_dspark_parallelism(model, local)
    expert_tp = 1 if config.enable_expert_parallel else config.tensor_parallel_size * config.data_parallel_size
    if (any(max(1, model.compress_ratios[layer]) != 1 for layer in model.dspark_layer_indices)
            or config.enable_expert_parallel and model.num_experts % config.expert_parallel_size
            or model.moe_intermediate_size % expert_tp or model.moe_intermediate_size // expert_tp % 32):
        raise ValueError("DSpark DP requires uniform global expert ownership or whole32 intermediate shards")


def dspark_local_shape(model, plan, logical_rank):
    """Original standalone DP1 draft shape contract; target PP is separate."""
    validate_dspark_parallelism(model, plan.config)
    return _dspark_source_local_shape(model, plan, logical_rank)


def dspark_data_parallel_local_shape(model, plan, logical_rank):
    """Three-layer DP draft shapes with real TP/global-EP and derived SP."""
    validate_dspark_data_parallelism(model, plan.config)
    shape = _dspark_source_local_shape(model, plan, logical_rank)
    model_sp = plan.config.enable_expert_parallel and plan.config.tensor_parallel_size > 1
    return {**shape, "data_parallel_rank": logical_rank // plan.config.tensor_parallel_size,
        "source_model_sequence_parallel": model_sp, "source_moe_sequence_parallel": model_sp}


def _dspark_source_local_shape(model, plan, logical_rank):
    """Per-rank three-layer draft geometry, independent of target PP.

    Main projection and Markov/confidence heads are replicated. Attention
    query/output groups, embedding/head and shared experts use TP. Routed
    experts use intermediate TP or uniform linear EP with Marlin padding.
    This interface describes shapes, never executable/provider acceptance.
    """
    from src.kernel.kernel_spec import linear_expert_partition
    config = plan.config
    owners = tuple(owner for owner in range(config.expert_parallel_size)
        for _ in linear_expert_partition(model.num_experts, config.expert_parallel_size, owner))
    if (type(logical_rank) is not int or not 0 <= logical_rank < config.worker_count
            or plan.num_layers != len(model.dspark_layer_indices) or plan.num_experts != model.num_experts
            or plan.layer_to_pipeline_stage != (0,) * len(model.dspark_layer_indices)
            or plan.expert_to_shard != owners):
        raise ValueError("DSpark geometry requires valid ranks and three actual draft layers")
    group = plan.rank_groups.for_rank("tensor", logical_rank)
    expert_group = plan.rank_groups.for_rank("expert", logical_rank)
    lane = group.index(logical_rank)
    expert_lane = expert_group.index(logical_rank)
    expert_ids = tuple(expert for expert, owner in enumerate(plan.expert_to_shard) if owner == expert_lane)
    hidden, rank, tp = model.hidden_size, model.dspark_markov_rank, config.tensor_parallel_size
    expert_tp = 1 if config.enable_expert_parallel else tp * config.data_parallel_size
    logical_i = model.moe_intermediate_size // expert_tp
    physical_i = -(-logical_i // 128) * 128
    shared_tp = 1 if config.enable_expert_parallel and tp > 1 and config.data_parallel_size > 1 else tp
    shared_i = -(-(model.num_shared_experts * model.moe_intermediate_size // 128) // shared_tp) * 128
    heads, groups = model.num_attention_heads // tp, model.o_groups // tp
    vocab = -(-model.vocab_size // 64) * 64
    return {"logical_rank": logical_rank, "layers": model.dspark_layer_indices,
        "target_num_hidden_layers": model.num_hidden_layers, "target_capture_layers": model.dspark_target_layer_ids,
        "raw_architecture": model.architecture, "sdk_draft_architecture": "DSparkDraftModel",
        "tensor_group": group, "expert_group": expert_group, "expert_ids": expert_ids,
        "expert_parallel_rank": expert_lane, "expert_tensor_parallel_size": expert_tp,
        "query_heads": heads, "query_head_range": (lane * heads, (lane + 1) * heads),
        "source_attention_padded_heads": 64 if heads <= 64 else 128,
        "grouped_output_groups": groups,
        "grouped_output_kn": (model.num_attention_heads // model.o_groups * model.head_dim, model.o_lora_rank),
        "fused_wqa_wkv_kn": (hidden, model.q_lora_rank + model.head_dim),
        "query_projection_kn": (model.q_lora_rank, heads * model.head_dim),
        "output_projection_kn": (groups * model.o_lora_rank, hidden),
        "layer_compress_ratios": tuple(max(1, model.compress_ratios[layer]) for layer in model.dspark_layer_indices),
        "main_projection_kn": (hidden * len(model.dspark_target_layer_ids), hidden),
        "main_projection_tp_replication": tp,
        "routed_gate_up_kn": (hidden, 2 * physical_i), "routed_down_kn": (physical_i, hidden),
        "logical_routed_gate_up_kn": (hidden, 2 * logical_i), "logical_routed_down_kn": (logical_i, hidden),
        "routed_local_padding_columns": physical_i - logical_i,
        "shared_gate_up_kn": (hidden, 2 * shared_i), "shared_down_kn": (shared_i, hidden),
        "shared_global_padding_columns": shared_i * shared_tp - model.num_shared_experts * model.moe_intermediate_size,
        "embedding_local_shape": (vocab // tp, hidden), "lm_head_kn": (hidden, vocab // tp),
        "markov_embedding_shape": (model.vocab_size, rank),
        "markov_bias_projection_kn": (rank, model.vocab_size), "markov_tp_replication": tp,
        "confidence_projection_kn": (hidden + rank, 1), "confidence_parameter_dtype": "fp32",
        "context_kv_tp_replication": tp, "context_kv_layer_indices": model.dspark_layer_indices,
        "block_size": model.dspark_block_size, "noise_token_id": model.dspark_noise_token_id,
        "source_execution_accepted": False, "native_provider_accepted": False, "normal_capacity_accepted": False}


def validate_v4_parallelism(model, config):
    """V4 grouped-output and source-padded expert geometry; no provider claim."""
    if model.model_type != "deepseek_v4" or model.attention_layout != "dsv4_sparse_mla":
        raise ValueError("V4 geometry requires compressed sparse MLA")
    if model.is_dspark_draft:
        raise ValueError("DSpark draft requires its distinct three-layer geometry, not target V4 layers")
    tp = config.tensor_parallel_size
    if config.context_parallel_size != 1:
        raise ValueError("V4 local geometry currently requires PCP1")
    if model.o_groups % tp or model.num_attention_heads % model.o_groups:
        raise ValueError("TP must divide V4 output groups, which evenly group the query heads")
    if model.num_attention_heads % tp or model.hidden_size % tp or (-(-model.vocab_size // 64) * 64) % tp:
        raise ValueError("V4 query heads, hidden size and vocabulary must divide TP")
    if config.pipeline_parallel_size > model.num_hidden_layers:
        raise ValueError("V4 pipeline stages must each own a decoder layer")
    expert_tp = 1 if config.enable_expert_parallel else tp * config.data_parallel_size
    if model.moe_intermediate_size % expert_tp or model.moe_intermediate_size // expert_tp % 32:
        raise ValueError("V4 logical MXFP4 expert shards require whole32-value scale groups")
    if model.hidden_size % 256 or model.num_shared_experts * model.moe_intermediate_size % 128:
        raise ValueError("V4 source weight blocks require aligned hidden and shared intermediate sizes")


def v4_local_shape(model, plan, logical_rank):
    """V4 source K/N, grouped output and per-stage compressed cache roles.

    Dense linears use block FP8, routed experts use the declared MXFP4
    Marlin profile. The source model sends one expanded mHC hidden tensor
    across PP; it has no GQA/K2 hidden-plus-residual boundary. This interface
    does not establish that transport or a general TP/DP/EP source adapter.
    """
    from src.serving.workload.shape_inference import infer_gemm_dimensions
    validate_v4_parallelism(model, plan.config)
    if plan.num_layers != model.num_hidden_layers or plan.num_experts != model.num_experts:
        raise ValueError("parallel plan layer/expert counts disagree with model")
    group = plan.rank_groups.for_rank("tensor", logical_rank)
    lane, tp = group.index(logical_rank), len(group)
    stage = plan.rank_groups.for_rank("pipeline", logical_rank).index(logical_rank)
    layers = tuple(layer for layer, owner in enumerate(plan.layer_to_pipeline_stage) if owner == stage)
    expert_group = plan.rank_groups.for_rank("expert", logical_rank)
    expert_lane = expert_group.index(logical_rank)
    experts = tuple(expert for expert, owner in enumerate(plan.expert_to_shard) if owner == expert_lane)
    expert_tp = 1 if plan.config.enable_expert_parallel else tp * plan.config.data_parallel_size
    logical_intermediate = model.moe_intermediate_size // expert_tp
    intermediate = -(-logical_intermediate // 128) * 128
    model_sp = (plan.config.enable_expert_parallel and tp > 1 and plan.config.data_parallel_size > 1
        and plan.config.pipeline_parallel_size == 1)
    shared_tp = 1 if model_sp else tp
    shared = -(-(model.num_shared_experts * model.moe_intermediate_size // 128) // shared_tp) * 128
    projections = {}
    for role in ("fused_qkv_a_proj", "q_b_proj", "o_proj", "indexer_wq_b", "indexer_weights_proj"):
        _, n, k = infer_gemm_dimensions(model, role, 1, tp_size=tp)
        projections[role] = (k, n)
    heads, groups = model.num_attention_heads // tp, model.o_groups // tp
    return {"logical_rank": logical_rank, "pipeline_stage": stage, "layers": layers,
        "tensor_group": group, "expert_group": expert_group, "expert_ids": experts,
        "query_heads": heads, "query_head_range": (lane * heads, (lane + 1) * heads),
        "source_attention_padded_heads": 64 if heads <= 64 else 128,
        "grouped_output_groups": groups,
        "grouped_output_kn": (model.num_attention_heads // model.o_groups * model.head_dim, model.o_lora_rank),
        "linear_kn": projections, "linear_weight_dtypes": {"fused_qkv_a_proj": "fp8", "q_b_proj": "fp8",
            "o_proj": "fp8", "indexer_wq_b": "fp8", "indexer_weights_proj": "bf16"},
        "routed_weight_dtype": "mxfp4", "shared_weight_dtype": "fp8",
        "routed_gate_up_kn": (model.hidden_size, 2 * intermediate),
        "routed_down_kn": (intermediate, model.hidden_size), "expert_tensor_parallel_size": expert_tp,
        "logical_routed_gate_up_kn": (model.hidden_size, 2 * logical_intermediate),
        "logical_routed_down_kn": (logical_intermediate, model.hidden_size),
        "routed_local_padding_columns": intermediate - logical_intermediate,
        "model_sequence_parallel": model_sp, "moe_internal_sequence_parallel": False,
        "shared_tensor_parallel_size": shared_tp,
        "shared_gate_up_kn": (model.hidden_size, 2 * shared), "shared_down_kn": (shared, model.hidden_size),
        "shared_global_padding_columns": shared * shared_tp - model.num_shared_experts * model.moe_intermediate_size,
        "cache_tp_replication": tp,
        "layer_cache_components": tuple({"layer": layer, "compress_ratio": model.compress_ratios[layer],
            "components": tuple({"kind": kind, "bytes_per_token_fraction": (width.numerator, width.denominator),
                "window_tokens": window} for kind, width, window in model.dsv4_layer_cache_components(layer))}
            for layer in layers),
        "pipeline_tensor_roles": ("hidden_states",),
        "pipeline_hidden_elements_per_token": model.hc_mult * model.hidden_size,
        "lm_head_kn": (model.hidden_size, -(-model.vocab_size // 64) * 64 // tp)}


def apply(plan, config):
    if config.tensor_parallel_size == 1:
        return plan
    return replace(
        plan,
        enabled_strategies=plan.enabled_strategies + ("tensor_parallel",),
        communication_rules=plan.communication_rules + (
            "row_parallel_all_reduce", "vocab_parallel_all_gather"),
    )
