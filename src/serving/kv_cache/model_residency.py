"""Static, rank-local model-weight residency for KV capacity planning."""

from dataclasses import dataclass, replace

from src.serving.parallelism.parallel_config import ParallelConfig
from src.kernel.kernel_spec import linear_expert_partition
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model, pipeline_layer_ranges
from src.serving.workload.model_config import (
    GLM_BLOCK_FP8_CONFIG_SHA256S, KIMI_K2_CONFIG_SHA256,
    KIMI_K2_7_CODE_CONFIG_SHA256, KIMI_K3_CONFIG_SHA256, ModelConfig,
    is_validated_qwen3_moe_ep2_model_scope, is_qwen3_moe_source_model_scope, is_qwen3_moe_pipeline_model_scope,
    is_validated_qwen3_pipeline_model_scope, is_dense_parallel_model_scope,
    is_kimi_k3_data_parallel_model_scope, is_kimi_k3_parallel_model_scope,
)
from src.serving.workload.shape_inference import padded_vocab_size, infer_gemm_dimensions


# Unique binding for the DeepGEMM dependency pinned by vLLM 0.30's
# cmake/external_projects/deepgemm.cmake. This is a source storage profile,
# not a declaration that the runtime selected or loaded this provider.
GLM_SM90_DEEP_GEMM_STORAGE_SOURCE = {
    "repository": "https://github.com/vllm-project/DeepGEMM",
    "revision": "e1f418c2a4f20818221f6b0e578b4c2f634d4c3f",
    "source_file": "csrc/apis/layout.hpp",
    "source_symbol": "transform_sf_into_required_layout",
    "scope": "arch_major=9, FP32 SFB, recipe=(1,128,128), is_sfa=false",
    "workspace_alignment_file": "csrc/jit_kernels/heuristics/runtime.hpp",
    "workspace_alignment_symbol": "HeuristicsRuntime::get_theoretical_mk_alignment_for_contiguous_layout",
}


@dataclass(frozen=True)
class ModelRankResidency:
    """Analytical model bytes resident on one physical inference rank."""

    rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    expert_parallel_rank: int
    layer_start: int
    layer_end: int
    embedding_bytes: int
    decoder_layer_bytes: int
    mtp_bytes: int
    final_norm_bytes: int
    lm_head_bytes: int
    additional_parameter_bytes: int = 0
    data_parallel_rank: int = 0
    context_parallel_rank: int = 0

    def __post_init__(self) -> None:
        for name in (
                "rank", "tensor_parallel_rank", "pipeline_parallel_rank",
                "expert_parallel_rank", "layer_start", "layer_end",
                "embedding_bytes", "decoder_layer_bytes",
                "mtp_bytes", "final_norm_bytes", "lm_head_bytes", "additional_parameter_bytes", "data_parallel_rank", "context_parallel_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"model residency {name} must be an integer")
            if value < 0:
                raise ValueError(f"model residency {name} cannot be negative")
        if self.layer_end <= self.layer_start:
            raise ValueError("model residency layer range must be non-empty")

    @property
    def weight_resident_bytes(self) -> int:
        return (
            self.embedding_bytes
            + self.decoder_layer_bytes
            + self.mtp_bytes
            + self.final_norm_bytes
            + self.lm_head_bytes
            + self.additional_parameter_bytes
        )


def _require_divisible(value: int, divisor: int, label: str) -> int:
    if value % divisor:
        raise NotImplementedError(
            f"analytical model residency requires {label}={value} to be "
            f"divisible by {divisor}")
    return value // divisor


def kimi_k3_marlin_postload_parameter_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    pipeline_parallel_rank: int = 0,
    pipeline_layer_partition: tuple[int, ...] | None = None,
) -> dict[str, int]:
    """Rank-local text weights for the declared vLLM0.30 K3 Marlin profile.

    This is source storage accounting, not model loading or provider selection.
    It includes KDA alignment, the routed source minimum partition and K3's decode-time W_UK/W_UV copies, preserves
    replicated latent projections, and prices MXFP4 values plus group32 scales.
    Stage-local EP owns whole experts instead of padded TP expert shards.
    The text-only scope excludes the parent vision tower and all runtime buffers.
    """
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("K3 residency requires ModelConfig and ParallelConfig")
    source = model.kimi_k3_semantics
    if (model.config_sha256 != KIMI_K3_CONFIG_SHA256 or source is None
            or model.config_extraction_path != "text_config" or model.compute_dtype != "bf16"
            or model.weight_dtype != "mxfp4" or model.weight_group_size != 32
            or model.num_mtp_layers or model.tie_word_embeddings):
        raise NotImplementedError("K3 Marlin residency requires the exact BF16/MXFP4 group32 text config")
    pp = parallelism.pipeline_parallel_size
    distributed = parallelism.data_parallel_size > 1
    if ((not parallelism.is_rank_local and not (distributed and is_kimi_k3_data_parallel_model_scope(model, parallelism)))
            or not 1 <= pp <= model.num_hidden_layers
            or model.num_experts % parallelism.expert_parallel_size):
        raise NotImplementedError("K3 Marlin residency requires its whole-shard TP/DP or uniform EP source with nonempty PP and CP/SP1")
    ranges = (pipeline_layer_ranges(model.num_hidden_layers, pp, pipeline_layer_partition)
        if pipeline_layer_partition is not None else pipeline_layer_ranges_for_model(model, pp))
    if type(pipeline_parallel_rank) is not int or not 0 <= pipeline_parallel_rank < pp:
        raise ValueError("K3 pipeline rank is outside the deployment")
    layer_start, layer_end = ranges[pipeline_parallel_rank]
    tp = parallelism.tensor_parallel_size
    h = model.hidden_size
    local_heads = _require_divisible(model.num_attention_heads, tp, "K3 attention heads")
    local_kda_heads = _require_divisible(source.kda_num_heads, tp, "K3 KDA heads")
    local_projection = _require_divisible(source.kda_num_heads * source.kda_head_dim, tp, "K3 KDA projection")
    _require_divisible(model.moe_intermediate_size, tp, "K3 routed intermediate")
    # KimiMoE pads the ordinary TP expert partition to its source floor.
    # Use the existing constructor-shape owner so physical parameter bytes
    # agree with W1/W2 while the logical checkpoint extent stays unchanged.
    _, _, local_moe = infer_gemm_dimensions(model, "experts.w2_weight", 1,
        tp_size=1 if parallelism.enable_expert_parallel else tp)
    if not parallelism.enable_expert_parallel:
        local_moe = _require_divisible(local_moe, parallelism.data_parallel_size, "K3 physical DP intermediate")
    local_experts = model.num_experts // parallelism.expert_parallel_size
    model_sp = parallelism.enable_expert_parallel and tp > 1 and parallelism.data_parallel_size > 1 and pp == 1
    # Default VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=0 replicates both the
    # sequence-parallel dense MLP and the ineligible shared expert MLP.
    mlp_tp = 1 if model_sp else tp
    local_shared = _require_divisible(model.moe_intermediate_size * model.num_shared_experts, mlp_tp, "K3 shared intermediate")
    local_dense = _require_divisible(model.intermediate_size, mlp_tp, "K3 dense intermediate")
    local_vocab = _require_divisible(padded_vocab_size(model.vocab_size), tp, "K3 padded vocabulary")
    if local_moe % 64 or source.routed_expert_hidden_size % 32:
        raise NotImplementedError("K3 Marlin residency requires whole source-padded local expert tiles")
    latent = source.routed_expert_hidden_size
    entries: dict[str, int] = {}

    def add(path: str, elements: int, element_bytes: int) -> None:
        if path in entries:
            raise ValueError(f"duplicate K3 parameter inventory entry: {path}")
        entries[path] = elements * element_bytes

    if pipeline_parallel_rank == 0:
        add("language_model.model.embed_tokens.weight", local_vocab * h, 2)
    if pipeline_parallel_rank == pp - 1:
        add("language_model.lm_head.weight", local_vocab * h, 2)
        add("language_model.model.norm.weight", h, 2)
        add("language_model.model.output_attn_res_norm.weight", h, 2)
        add("language_model.model.output_attn_res_proj.weight", h, 2)
    for layer in range(layer_start, layer_end):
        kind = source.layer_attention_types[layer]
        prefix = f"language_model.model.layers.{layer}"
        for name in ("input_layernorm", "post_attention_layernorm",
                     "self_attention_res_norm", "mlp_res_norm",
                     "self_attention_res_proj", "mlp_res_proj"):
            add(f"{prefix}.{name}.weight", h, 2)
        attn = f"{prefix}.self_attn"
        if kind == "kda":
            local_input_rows = 4 * local_projection + source.kda_head_dim + local_kda_heads
            padded_input_rows = -(-local_input_rows // 16) * 16
            add(f"{attn}.in_proj_qkvgfab.weight", padded_input_rows * h, 2)
            add(f"{attn}.f_b_proj.weight", local_projection * source.kda_head_dim, 2)
            add(f"{attn}.dt_bias", local_projection, 4)
            add(f"{attn}.conv1d.weight", 3 * local_projection * source.short_conv_kernel_size, 4)
            add(f"{attn}.A_log", local_kda_heads, 4)
            add(f"{attn}.o_norm.weight", source.kda_head_dim, 2)
            add(f"{attn}.o_proj.weight", local_projection * h, 2)
        elif kind == "mla_nope":
            gate_width = local_heads * model.v_head_dim
            fused_rows = model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim + gate_width
            add(f"{attn}.fused_qkv_a_g_proj.weight", fused_rows * h, 2)
            add(f"{attn}.q_a_layernorm.weight", model.q_lora_rank, 2)
            add(f"{attn}.kv_a_layernorm.weight", model.kv_lora_rank, 2)
            add(f"{attn}.q_b_proj.weight", local_heads * (model.qk_nope_head_dim + model.qk_rope_head_dim) * model.q_lora_rank, 2)
            kv_b_elements = local_heads * (model.qk_nope_head_dim + model.v_head_dim) * model.kv_lora_rank
            add(f"{attn}.kv_b_proj.weight", kv_b_elements, 2)
            add(f"{attn}.W_UK", kv_b_elements // 2, 2)
            add(f"{attn}.W_UV", kv_b_elements // 2, 2)
            add(f"{attn}.o_proj.weight", local_heads * model.v_head_dim * h, 2)
        else:
            raise ValueError(f"unhandled K3 attention type: {kind}")
        if model.layer_uses_moe(layer):
            moe = f"{prefix}.block_sparse_moe"
            add(f"{moe}.gate.weight", model.num_experts * h, 2)
            add(f"{moe}.gate.e_score_correction_bias", model.num_experts, 4)
            add(f"{moe}.routed_expert_down_proj.weight", latent * h, 2)
            add(f"{moe}.routed_expert_norm.weight", latent, 2)
            add(f"{moe}.routed_expert_up_proj.weight", latent * h, 2)
            add(f"{moe}.shared_experts.gate_up_proj.weight", 2 * local_shared * h, 2)
            add(f"{moe}.shared_experts.down_proj.weight", local_shared * h, 2)
            add(f"{moe}.experts.w13_weight_packed", local_experts * local_moe * latent, 1)
            add(f"{moe}.experts.w13_weight_scale", local_experts * 2 * local_moe * (latent // 32), 1)
            add(f"{moe}.experts.w2_weight_packed", local_experts * latent * local_moe // 2, 1)
            add(f"{moe}.experts.w2_weight_scale", local_experts * latent * (local_moe // 32), 1)
        else:
            add(f"{prefix}.mlp.gate_up_proj.weight", 2 * local_dense * h, 2)
            add(f"{prefix}.mlp.down_proj.weight", local_dense * h, 2)
    return entries


def kimi_k3_source_workspace_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    max_batched_tokens: int,
    max_requests: int,
    source_multiprocessors: int,
    pipeline_layer_partition: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Conservative visible workspaces for the declared eager K3 source.

    TP keeps the full-head KDA bound. Distributed Marlin envelopes every
    source block size and the full intermediate width, including its minimum
    partition. Public communication tensors are deliberately added even if
    an executed activation witness already includes them. Compiler/provider
    state and other runtime reserves remain separate policy inputs.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("K3 source workspace requires normalized model metadata")
    source = model.kimi_k3_semantics
    if (model.config_sha256 != KIMI_K3_CONFIG_SHA256 or source is None
            or model.compute_dtype != "bf16" or model.weight_dtype != "mxfp4"):
        raise NotImplementedError("K3 source workspace requires the exact BF16/MXFP4 text config")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("K3 source workspace requires normalized parallelism")
    tp, dp, pp = (parallelism.tensor_parallel_size, parallelism.data_parallel_size,
        parallelism.pipeline_parallel_size)
    legal = (is_kimi_k3_data_parallel_model_scope(model, parallelism) if dp > 1 else
        parallelism.is_rank_local and is_kimi_k3_parallel_model_scope(model, tp, pp,
            expert_parallel=parallelism.enable_expert_parallel))
    if not legal:
        raise NotImplementedError("K3 source workspace requires whole TP/DP/EP shards, nonempty PP and CP/SP1")
    ranges = (pipeline_layer_ranges(model.num_hidden_layers, pp, pipeline_layer_partition)
        if pipeline_layer_partition is not None else pipeline_layer_ranges_for_model(model, pp))
    for name, value in (("max_batched_tokens", max_batched_tokens),
            ("max_requests", max_requests), ("source_multiprocessors", source_multiprocessors)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"K3 source workspace {name} must be positive")

    tokens = max_batched_tokens
    heads = source.kda_num_heads
    width = source.kda_head_dim
    bf16_matrix = tokens * heads * width * 2
    kda_intra = (2 * tokens * heads * 64 * 2
        + tokens * heads * 16 * 4)
    kda_other_phase_bound = (kda_intra + 7 * bf16_matrix
        + tokens * heads * width * 4 + 2 * heads * width * width * 4)
    chunks = (tokens + 63 * min(max_requests, tokens)) // 64
    kda_recurrent_live_bound = (
        6 * bf16_matrix + chunks * heads * width * width * 2
        + heads * width * width * 4 + tokens * heads * width * 4
        + tokens * heads * 64 * 2 + tokens * heads * 4
        + 2 * chunks * 8 + (max_requests + 1) * 8)
    kda_upper_bound = max(kda_other_phase_bound, kda_recurrent_live_bound)

    latent = source.routed_expert_hidden_size
    intermediate = model.moe_intermediate_size
    legacy = tp == dp == 1
    model_sp = dp > 1 and tp > 1 and pp == 1 and parallelism.enable_expert_parallel
    expert_rows = dp * (-(-tokens // tp) * tp if model_sp else tokens)
    routes = expert_rows * model.num_experts_per_tok
    marlin_common = max(routes * max(intermediate, latent), expert_rows * latent) * 2
    marlin_workspace2 = routes * max(2 * intermediate, latent) * 2
    if legacy:
        block = 64
        for candidate in (8, 16, 32, 48, 64):
            if routes / model.num_experts / candidate < 0.9:
                block = candidate
                break
        blocks = (block,)
    else:
        # EP estimates M with ceil(M*localE/globalE) before choosing a block.
        # A full-E heuristic alone can miss that discontinuity. This is a
        # memory envelope, not a launch selector or provider decision.
        block = None
        blocks = (8, 16, 32, 48, 64)
        marlin_common = -(-marlin_common // 256) * 256
        marlin_workspace2 = -(-marlin_workspace2 // 256) * 256
    alternatives = []
    for candidate in blocks:
        capacity = routes + model.num_experts * (candidate - 1)
        if routes < model.num_experts:
            capacity = min(capacity, routes * candidate)
        elements = min(2 * intermediate * capacity,
            source_multiprocessors * 4 * candidate * 256)
        if candidate == 8:
            elements *= 2
        alignment = (capacity * 4 + (-(-capacity // candidate)) * 4 + 4
            + (model.num_experts + 1) * 4 + source_multiprocessors * 4 * 4)
        alternatives.append((capacity, elements * 4, alignment))
    sorted_capacity = max(value[0] for value in alternatives)
    marlin_reduce = max(value[1] for value in alternatives)
    marlin_alignment = max(value[2] for value in alternatives)
    marlin_upper_bound = marlin_common + marlin_workspace2 + marlin_reduce + marlin_alignment

    distributed = {}
    if dp > 1:
        distributed = {
            "gather_latent_weights_ids": expert_rows * (latent * 2 + model.num_experts_per_tok * 8),
            "expert_output": expert_rows * latent * 2,
            "combine_and_finalize_copy": 2 * tokens * latent * 2,
        }
    if tp > 1:
        # Fresh latent reduction/norm output plus shared/output hidden
        # branches are bounded at full local tokens even under model SP.
        distributed["latent_reduce_and_norm"] = 2 * tokens * latent * 2
        distributed["shared_and_output_hidden"] = 4 * tokens * model.hidden_size * 2
    distributed_bytes = sum(distributed.values())

    from src.serving.workload.traced_sampling import vllm030_worker_storage_bytes
    worker_allocations = vllm030_worker_storage_bytes(tokens=max_batched_tokens,
        max_requests=max_requests, inputs_embeds_size=model.hidden_size, dtype="bf16")
    worker_persistent = sum(worker_allocations.values())
    pipeline_persistent = [0]
    pipeline_transient = [0]
    if pp > 1:
        pipeline_persistent, pipeline_transient = [], []
        hidden_matrix = max_batched_tokens * model.hidden_size * 2
        for stage, (start, end) in enumerate(ranges):
            received_blocks = -(-start // source.attention_residual_block_size)
            model_blocks = -(-end // source.attention_residual_block_size)
            pipeline_persistent.append(0 if stage == 0 else (1 + received_blocks) * hidden_matrix)
            transient = (1 + model_blocks) * hidden_matrix
            if tp == 1 and stage > 0:
                # The shared PP receive loop retains its last `received`
                # residual tensor across run_stage, including every block.
                transient += received_blocks * hidden_matrix
            if tp > 1 and stage > 0:
                # A lane's wire is a view into a full receive allocation.
                # Both full receive and full all-gather outputs coexist.
                transient += 2 * (1 + received_blocks) * hidden_matrix
            pipeline_transient.append(transient)
    pipeline_persistent = tuple(pipeline_persistent[stage]
        for replica in range(dp) for stage in range(pp) for lane in range(tp))
    pipeline_transient = tuple(pipeline_transient[stage]
        for replica in range(dp) for stage in range(pp) for lane in range(tp))
    pipeline_p2p = max(persistent + transient for persistent, transient in zip(
        pipeline_persistent, pipeline_transient, strict=True))
    return {
        "kda_chunk_upper_bound_bytes": kda_upper_bound,
        "kda_recurrent_live_bound_bytes": kda_recurrent_live_bound,
        "kda_varlen_chunk_upper_bound": chunks,
        "marlin_modular_arena_bytes": marlin_common + marlin_workspace2,
        "marlin_reduce_scratch_bytes": marlin_reduce,
        "marlin_alignment_bytes": marlin_alignment,
        "marlin_upper_bound_bytes": marlin_upper_bound,
        "activation_workspace_upper_bound_bytes": kda_upper_bound + marlin_upper_bound + distributed_bytes,
        "distributed_public_tensor_upper_bound_bytes": distributed_bytes,
        "distributed_public_tensor_components": distributed,
        "source_expert_physical_rows_upper_bound": expert_rows,
        "worker_persistent_bytes": worker_persistent,
        "pipeline_p2p_working_bytes": pipeline_p2p,
        "pipeline_intermediate_persistent_bytes_by_rank": pipeline_persistent,
        "pipeline_transient_upper_bound_bytes_by_rank": pipeline_transient,
        "marlin_block_size": block,
        "marlin_block_sizes_bound": blocks,
        "marlin_sorted_capacity": sorted_capacity,
        "scope": "visible eager source workspace envelope; declared runtime/compiler/provider reserves remain separate",
    }


def kimi_k25_outer_loader_payload_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    vision_tensor_parallel_mode: str,
) -> dict[str, int]:
    """Active-tower BF16 parameters for the exact K2.6 and K2.7-Code configs.

    vLLM 0.30 KimiK25Config/KimiK25VisionConfig retain the legacy vt_* keys
    as metadata, while the constructor defaults define H=1152, I=4304,
    27 layers and 16 heads. KimiK25Config replaces mm_hidden_size==H with
    the text hidden size. CompressedTensorsConfig is ignored for the tower
    and projector, so every parameter here is BF16, including biases/norms.
    This is an explicit enabled-tower, no-offload constructor scenario;
    no image input does not establish language_model_only/vision_chunk skip.
    Buffers, post-load storage, text weights, reserves and KV are excluded.
    The normal capacity entry still rejects the extracted text inventory.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("Kimi outer payload requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("Kimi outer payload requires ParallelConfig")
    if (model.model_type != "kimi_k2"
            or model.config_sha256 not in (KIMI_K2_CONFIG_SHA256, KIMI_K2_7_CODE_CONFIG_SHA256)
            or model.config_extraction_path != "text_config"
            or model.compute_dtype != "bf16" or model.hidden_size != 7168):
        raise NotImplementedError("Kimi outer payload requires the exact BF16 Kimi-K2.6 or Kimi-K2.7-Code parent config")
    if not parallelism.is_rank_local or parallelism.pipeline_parallel_size != 1:
        raise NotImplementedError("Kimi outer payload requires PP=DP=CP=SP=1")
    if vision_tensor_parallel_mode not in ("weights", "data"):
        raise ValueError("vision tensor parallel mode must be weights or data")
    # vision.is_vit_use_data_parallel forces DP when heads cannot split.
    tp = parallelism.tensor_parallel_size
    vision_tp = 1 if vision_tensor_parallel_mode == "data" or 16 % tp else tp
    h, intermediate, layers = 1152, 4304, 27
    local_h = _require_divisible(h, vision_tp, "vision hidden_size")
    local_i = _require_divisible(intermediate, vision_tp, "vision intermediate_size")
    entries: dict[str, int] = {}

    def add(path: str, elements: int) -> None:
        if path in entries:
            raise ValueError(f"duplicate Kimi outer loader parameter: {path}")
        entries[path] = 2 * elements

    add("vision_tower.patch_embed.proj.weight", h * 3 * 14 * 14)
    add("vision_tower.patch_embed.proj.bias", h)
    add("vision_tower.patch_embed.pos_emb.weight", 64 * 64 * h)
    for layer in range(layers):
        prefix = f"vision_tower.encoder.blocks.{layer}"
        for norm in ("norm0", "norm1"):
            add(f"{prefix}.{norm}.weight", h)
            add(f"{prefix}.{norm}.bias", h)
        add(f"{prefix}.wqkv.weight", 3 * local_h * h)
        add(f"{prefix}.wqkv.bias", 3 * local_h)
        add(f"{prefix}.wo.weight", h * local_h)
        add(f"{prefix}.wo.bias", h)
        add(f"{prefix}.mlp.fc0.weight", local_i * h)
        add(f"{prefix}.mlp.fc0.bias", local_i)
        add(f"{prefix}.mlp.fc1.weight", h * local_i)
        add(f"{prefix}.mlp.fc1.bias", h)
    for suffix in ("weight", "bias"):
        add(f"vision_tower.encoder.final_layernorm.{suffix}", h)
        add(f"mm_projector.pre_norm.{suffix}", h)
    # Patchmerger's two ReplicatedLinear modules never shard their parameters.
    merged = 2 * 2 * h
    add("mm_projector.linear_1.weight", merged * merged)
    add("mm_projector.linear_1.bias", merged)
    add("mm_projector.linear_2.weight", model.hidden_size * merged)
    add("mm_projector.linear_2.bias", model.hidden_size)
    return entries


def dsv4_routed_loader_payload_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
) -> dict[str, int]:
    """Return the V4 EP-off routed loader payload on one TP rank.

    This is the aligned Marlin source constructor inventory, before conversion;
    it cannot be supplied to the normal KV capacity allocator as residency.
    DeepseekV4MoE's non-MegaMoE FusedMoEFactory retains every expert with
    intermediate_size / TP. Mxfp4MoEMethod.create_weights registers packed
    uint8 gate/up and down matrices plus one uint8 UE8M0 scale per 32 input
    elements (vLLM 0.30 quantization/mxfp4.py). Converted scales, padding,
    workspaces, other model weights, MTP and runtime reserves are excluded.
    The MARLIN oracle rounds H to 256 and local I to 128; this subset rejects
    dimensions needing that padding instead of charging unpadded allocations.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("V4 loader payload requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("V4 loader payload requires ParallelConfig")
    if model.is_dspark_draft:
        raise NotImplementedError("DSpark draft residency requires its own shared vocabulary/three-layer/head inventory, not target V4 weights")
    if (model.model_type != "deepseek_v4" or model.expert_dtype != "fp4"
            or model.compute_dtype != "bf16"):
        raise NotImplementedError("V4 loader payload requires BF16/MXFP4 V4")
    if (not parallelism.is_rank_local
            or parallelism.pipeline_parallel_size != 1
            or parallelism.enable_expert_parallel):
        raise NotImplementedError(
            "V4 loader payload requires non-MegaMoE EP-off, PP=DP=CP=SP=1")
    local_intermediate = _require_divisible(
        model.moe_intermediate_size, parallelism.tensor_parallel_size,
        "moe_intermediate_size")
    # The outer DeepseekV4MoE metadata also asserts E divisible by TP.
    _require_divisible(model.num_experts, parallelism.tensor_parallel_size, "num_experts")
    _require_divisible(model.hidden_size, 256, "Marlin hidden_size")
    _require_divisible(local_intermediate, 128, "Marlin local moe_intermediate_size")
    # Refuse source floor division that would truncate a scale group.
    hidden_groups = _require_divisible(model.hidden_size, 32, "hidden_size")
    intermediate_groups = _require_divisible(
        local_intermediate, 32, "local moe_intermediate_size")
    experts = model.num_experts
    return {
        "experts_per_rank": experts,
        "intermediate_size_per_rank": local_intermediate,
        "layers": model.num_hidden_layers,
        "w13_packed_bytes_per_layer": (
            experts * 2 * local_intermediate * (model.hidden_size // 2)),
        "w2_packed_bytes_per_layer": (
            experts * model.hidden_size * (local_intermediate // 2)),
        "w13_ue8m0_bytes_per_layer": (
            experts * 2 * local_intermediate * hidden_groups),
        "w2_ue8m0_bytes_per_layer": (
            experts * model.hidden_size * intermediate_groups),
    }


def _dsv4_non_routed_loader_inventory(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    attention_backend: str,
    local_shared_intermediate: int | None = None,
    layer_indices: tuple[int, ...] | None = None,
) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[int, ...] | None]]:
    """Inventory selected V4 non-routed constructor parameters per TP rank.

    The explicit FlashMLA/non-MegaMoE/EP-off scenario uses NVIDIA model.py,
    attention.py and compressor.py in vLLM 0.30. FP8 serialized linears use
    Fp8LinearMethod.create_weights: one byte per value and one UE8M0 byte per
    128x128 block. Compressors, index weights, gate, norms and vocabulary
    weights are BF16; APE/mHC/sinks/correction biases are explicitly FP32.
    Hash tables are replicated INT32. Values describe registered parameters
    before post-load conversion, not allocator-ready runtime residency. KV,
    rotary/topk buffers, broadcast-weight derivation and scratch/reserves are
    separate. Public target callers select the original complete target;
    the dedicated DSpark owner explicitly selects its actual appended layers.
    """
    if local_shared_intermediate is None:
        routed = dsv4_routed_loader_payload_bytes(model, parallelism)
        local_shared = routed["intermediate_size_per_rank"] * model.num_shared_experts
    else:
        # The parallel source owner supplies its actual padded shared shard,
        # including full replication for PP1 model SP. Public loader/buffer
        # entry points retain the original narrow source boundary.
        if type(local_shared_intermediate) is not int or local_shared_intermediate <= 0 or local_shared_intermediate % 128:
            raise ValueError("V4 shared inventory requires a positive whole128 source intermediate")
        local_shared = local_shared_intermediate
    if attention_backend != "flashmla":
        raise NotImplementedError("V4 parameter payload requires explicit FlashMLA")
    if model.weight_dtype != "fp8" or model.tie_word_embeddings:
        raise NotImplementedError("V4 parameter payload requires FP8 and separate vocabulary weights")
    if model.target_semantics is None or model.target_semantics.moe_topk_method != "noaux_tc":
        raise NotImplementedError("V4 parameter payload requires the text noaux_tc routing source")
    if layer_indices is None:
        layers = range(model.num_hidden_layers)
    elif (model.is_dspark_draft and type(layer_indices) is tuple
            and all(type(layer) is int for layer in layer_indices)
            and layer_indices == model.dspark_layer_indices):
        # The dedicated draft owner validates geometry before supplying this
        # exact appended-layer selection. No identity/count clone is used.
        layers = layer_indices
    else:
        raise ValueError("V4 constructor layer selection requires the exact three DSpark appended layers")
    tp = parallelism.tensor_parallel_size
    h = model.hidden_size
    d = model.head_dim
    local_heads = _require_divisible(model.num_attention_heads, tp, "num_attention_heads")
    if local_heads > 128:
        raise NotImplementedError("V4 FlashMLA source supports at most 128 local heads")
    local_groups = _require_divisible(model.o_groups, tp, "o_groups")
    heads_per_group = _require_divisible(model.num_attention_heads, model.o_groups, "heads per output group")
    local_vocab = _require_divisible(padded_vocab_size(model.vocab_size), tp, "padded_vocab_size")
    entries: dict[str, int] = {}
    element_bytes_by_path: dict[str, int] = {}
    loader_shape_by_path: dict[str, tuple[int, ...] | None] = {}

    def add(path: str, elements: int, element_bytes: int, shape: tuple[int, ...] | None = None) -> None:
        if path in entries:
            raise ValueError(f"duplicate V4 loader parameter: {path}")
        entries[path] = elements * element_bytes
        element_bytes_by_path[path] = element_bytes
        if shape is not None:
            product = 1
            for dim in shape:
                product *= dim
            if product != elements:
                raise ValueError(f"V4 loader shape disagrees with element count: {path}")
        loader_shape_by_path[path] = shape

    def fp8_linear(path: str, outputs: int, inputs: int) -> None:
        add(f"{path}.weight", outputs * inputs, 1, (outputs, inputs))
        scale_shape = ((outputs + 127) // 128, (inputs + 127) // 128)
        add(f"{path}.weight_scale_inv", scale_shape[0] * scale_shape[1], 1, scale_shape)

    def compressor(path: str, ratio: int, width: int) -> None:
        coff = 2 if ratio == 4 else 1
        add(f"{path}.fused_wkv_wgate.weight", 2 * coff * width * h, 2, (2 * coff * width, h))
        add(f"{path}.norm.weight", width, 2, (width,))
        add(f"{path}.ape", ratio * coff * width, 4, (ratio, coff * width))

    add("model.embed_tokens.weight", local_vocab * h, 2, (local_vocab, h))
    add("lm_head.weight", local_vocab * h, 2, (local_vocab, h))
    add("model.norm.weight", h, 2, (h,))
    add("model.hc_head_fn", model.hc_mult * model.hc_mult * h, 4, (model.hc_mult, model.hc_mult * h))
    add("model.hc_head_base", model.hc_mult, 4, (model.hc_mult,))
    add("model.hc_head_scale", 1, 4, (1,))
    mix = (2 + model.hc_mult) * model.hc_mult
    for layer in layers:
        prefix = f"model.layers.{layer}"
        attn = f"{prefix}.attn"
        ffn = f"{prefix}.ffn"
        add(f"{prefix}.attn_norm.weight", h, 2, (h,))
        add(f"{prefix}.ffn_norm.weight", h, 2, (h,))
        for branch in ("attn", "ffn"):
            add(f"{prefix}.hc_{branch}_fn", mix * model.hc_mult * h, 4, (mix, model.hc_mult * h))
            add(f"{prefix}.hc_{branch}_base", mix, 4, (mix,))
            add(f"{prefix}.hc_{branch}_scale", 3, 4, (3,))
        fp8_linear(f"{attn}.fused_wqa_wkv", model.q_lora_rank + d, h)
        fp8_linear(f"{attn}.wq_b", local_heads * d, model.q_lora_rank)
        fp8_linear(f"{attn}.wo_a", local_groups * model.o_lora_rank, heads_per_group * d)
        fp8_linear(f"{attn}.wo_b", h, local_groups * model.o_lora_rank)
        add(f"{attn}.q_norm.weight", model.q_lora_rank, 2, (model.q_lora_rank,))
        add(f"{attn}.kv_norm.weight", d, 2, (d,))
        # FlashMLA.get_padded_num_q_heads defines 64/128 sink allocation.
        add(f"{attn}.attn_sink", 64 if local_heads <= 64 else 128, 4, (64 if local_heads <= 64 else 128,))
        ratio = model.kernel_compress_ratios[layer]
        if ratio > 1:
            compressor(f"{attn}.compressor", ratio, d)
        if ratio == 4:
            indexer = f"{attn}.indexer"
            fp8_linear(f"{indexer}.wq_b", model.index_n_heads * model.index_head_dim, model.q_lora_rank)
            add(f"{indexer}.weights_proj.weight", model.index_n_heads * h, 2, (model.index_n_heads, h))
            compressor(f"{indexer}.compressor", ratio, model.index_head_dim)
        add(f"{ffn}.gate.weight", model.num_experts * h, 2, (model.num_experts, h))
        if layer < model.num_hash_layers:
            add(f"{ffn}.gate.tid2eid", model.vocab_size * model.num_experts_per_tok, 4, (model.vocab_size, model.num_experts_per_tok))
        else:
            # Validated text configs use topk_method=noaux_tc.
            add(f"{ffn}.gate.e_score_correction_bias", model.num_experts, 4, (model.num_experts,))
        if local_shared:
            fp8_linear(f"{ffn}.shared_experts.gate_up_proj", 2 * local_shared, h)
            fp8_linear(f"{ffn}.shared_experts.down_proj", h, local_shared)
    return entries, element_bytes_by_path, loader_shape_by_path



def dsv4_non_routed_loader_payload_bytes(
    model: ModelConfig, parallelism: ParallelConfig, *, attention_backend: str,
) -> dict[str, int]:
    """Return source loader byte counts for non-routed V4 parameters."""
    entries, _, _ = _dsv4_non_routed_loader_inventory(model, parallelism, attention_backend=attention_backend)
    return entries


def dsv4_postload_parameter_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    hardware_id: str,
    linear_backend: str,
    routed_backend: str,
) -> dict[str, int]:
    """Target-off registered parameter payload for a declared SM90 profile.

    Explicit CUDA DeepGEMM block-FP8 linears plus non-batched Marlin MXFP4
    routed experts, ordinary sharded vocabulary, FlashMLA and EP-off/PP1.
    This is source storage accounting, not runtime backend selection.

    fp8_utils.deepgemm_post_process_weight_scale_block upcasts checkpoint
    UE8M0 scales to FP32. The pinned SM90 SFB (128,128) layout is unchanged
    (same storage branch as GLM_SM90_DEEP_GEMM_STORAGE_SOURCE), including
    wo_a's grouped-BMM view. Marlin repack changes layout but retains packed
    4-bit payload; mxfp4_marlin_process_scales returns UE8M0 after its BF16
    intermediate, so that temporary must not double resident scale bytes.
    Replaced loader tensors are not retained parameters. Buffers, transient
    conversion peak, kernel workspaces and runtime reserves are excluded;
    this inventory alone cannot establish normal available-KV capacity.
    """
    if (hardware_id != "nvidia_h200" or linear_backend != "deep_gemm"
            or routed_backend != "marlin"):
        raise NotImplementedError("V4 post-load parameters require explicit H200 DeepGEMM/Marlin source storage")
    routed = dsv4_routed_loader_payload_bytes(model, parallelism)
    parameters = dsv4_non_routed_loader_payload_bytes(
        model, parallelism, attention_backend="flashmla")
    for path in parameters:
        if path.endswith(".weight_scale_inv"):
            parameters[path] *= 4
    for layer in range(model.num_hidden_layers):
        prefix = f"model.layers.{layer}.ffn.experts"
        parameters[f"{prefix}.w13_weight"] = routed["w13_packed_bytes_per_layer"]
        parameters[f"{prefix}.w2_weight"] = routed["w2_packed_bytes_per_layer"]
        parameters[f"{prefix}.w13_weight_scale"] = routed["w13_ue8m0_bytes_per_layer"]
        parameters[f"{prefix}.w2_weight_scale"] = routed["w2_ue8m0_bytes_per_layer"]
    return parameters


def _dsv4_parallel_rank_shape(model, parallelism, logical_rank, pipeline_layer_partition):
    """One source owner for the conditional parameter and model-buffer scopes."""
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.parallelism.tp import v4_local_shape
    from src.serving.workload.model_config import (
        is_deepseek_v4_parallel_model_scope, is_deepseek_v4_data_parallel_model_scope,
    )
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("V4 parallel inventory requires ModelConfig and ParallelConfig")
    source_scope = (is_deepseek_v4_parallel_model_scope(model, parallelism.tensor_parallel_size,
        parallelism.pipeline_parallel_size, expert_parallel=parallelism.enable_expert_parallel)
        if parallelism.data_parallel_size == 1 else is_deepseek_v4_data_parallel_model_scope(model, parallelism))
    if not source_scope or parallelism.context_parallel_size != 1 or parallelism.sequence_parallel_size != 1:
        raise NotImplementedError("V4 parallel parameter inventory requires its exact modular TP/DP/EP/PP source, CP/SP1 and MTP-off")
    if type(logical_rank) is not int or not 0 <= logical_rank < parallelism.worker_count:
        raise ValueError("V4 parallel parameter rank is outside its actual worker plan")
    if pipeline_layer_partition is not None:
        # The plan's PP1 strategy has no cut to apply; validate an explicit
        # declaration here as well so a truncated full-model claim fails.
        pipeline_layer_ranges(model.num_hidden_layers, parallelism.pipeline_parallel_size, pipeline_layer_partition)
    plan = build_parallel_plan(parallelism, num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=pipeline_layer_partition)
    return v4_local_shape(model, plan, logical_rank)


def dsv4_parallel_postload_parameter_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    logical_rank: int,
    *,
    pipeline_layer_partition: tuple[int, ...] | None = None,
) -> dict[str, int]:
    """One rank's declared SM90 modular target parameter bytes, CP/SP1.

    Reuse the constructor inventory and actual parallel shape owner. Dense
    FP8 scales become FP32; routed packed values and UE8M0 scales retain
    Marlin's padded local intermediate. PP1 TP*DP EP replicates shared
    weights for model SP; other shared weights use only TP, never DP.
    Parameter bytes exclude model buffers, workspace/conversion peaks,
    absorption state, KV and reserves. This does not bind a GPU provider or
    admit a normal capacity policy. Existing loader/placement APIs retain
    their narrower source scope.
    """
    local = _dsv4_parallel_rank_shape(model, parallelism, logical_rank, pipeline_layer_partition)
    parameters, _, _ = _dsv4_non_routed_loader_inventory(model, parallelism, attention_backend="flashmla",
        local_shared_intermediate=local["shared_down_kn"][0])
    for name in parameters:
        if name.endswith(".weight_scale_inv"):
            parameters[name] *= 4
    stage = local["pipeline_stage"]
    layers = local["layers"]
    parameters = {name: size for name, size in parameters.items()
        if (name.startswith("model.layers.") and int(name.split(".")[2]) in layers
            or name == "model.embed_tokens.weight" and stage == 0
            or name in ("lm_head.weight", "model.norm.weight") and stage == parallelism.pipeline_parallel_size - 1
            or name.startswith("model.hc_head_"))}
    experts, intermediate, hidden = len(local["expert_ids"]), local["routed_down_kn"][0], model.hidden_size
    for layer in layers:
        prefix = f"model.layers.{layer}.ffn.experts"
        parameters[f"{prefix}.w13_weight"] = experts * 2 * intermediate * (hidden // 2)
        parameters[f"{prefix}.w2_weight"] = experts * hidden * (intermediate // 2)
        parameters[f"{prefix}.w13_weight_scale"] = experts * 2 * intermediate * (hidden // 32)
        parameters[f"{prefix}.w2_weight_scale"] = experts * hidden * (intermediate // 32)
    return parameters


def _dsv4_dspark_rank_shape(model, parallelism, logical_rank):
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.parallelism.tp import dspark_local_shape
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("DSpark inventory requires ModelConfig and ParallelConfig")
    plan = build_parallel_plan(parallelism, num_layers=3, num_experts=model.num_experts)
    local = dspark_local_shape(model, plan, logical_rank)
    if any(ratio != 1 for ratio in local["layer_compress_ratios"]):
        raise NotImplementedError("DSpark inventory requires its three SWA-only source layers")
    return local


def dsv4_dspark_postload_parameter_bytes(
    model: ModelConfig, parallelism: ParallelConfig, logical_rank: int, *,
    hardware_id: str, linear_backend: str, routed_backend: str, confidence_head_loaded: bool,
) -> dict[str, int]:
    """Owned draft parameters for an explicit standalone SM90 source profile.

    Three real appended layers, ordinary TP or uniform EP, no DP/CP/SP or
    draft PP split in this inventory's current subset. The upstream loader's
    wider target/draft parallel combinations remain a separate source scope.
    Target-aliased embedding and LM head are deliberately excluded: they
    belong to their target owners, and a combined runtime must prove that
    alias/placement before counting them once. Main projection/norm and the
    BF16 Markov pair are replicated, confidence is loaded FP32 independently
    of whether a request calls it. Dense FP8 block scales become FP32 under
    the explicit DeepGEMM profile; Marlin packed values and UE8M0 scales use
    actual physical intermediate/owned expert shapes. This is not normal
    residency, loader/provider execution, reserves, conversion peak or KV.
    """
    if (hardware_id != "nvidia_h200" or linear_backend != "deep_gemm" or routed_backend != "marlin"):
        raise NotImplementedError("DSpark post-load inventory requires explicit H200 DeepGEMM/Marlin storage")
    if type(confidence_head_loaded) is not bool:
        raise TypeError("DSpark confidence-head loading must be explicitly boolean")
    local = _dsv4_dspark_rank_shape(model, parallelism, logical_rank)
    return _dsv4_dspark_owned_parameter_bytes(model, parallelism, local, confidence_head_loaded)


def _dsv4_dspark_data_rank_shape(model, parallelism, logical_rank):
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.parallelism.tp import dspark_data_parallel_local_shape
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("DSpark DP inventory requires ModelConfig and ParallelConfig")
    plan = build_parallel_plan(parallelism, num_layers=3, num_experts=model.num_experts)
    return dspark_data_parallel_local_shape(model, plan, logical_rank)


def dsv4_dspark_data_parallel_postload_parameter_bytes(
    model: ModelConfig, parallelism: ParallelConfig, logical_rank: int, *,
    hardware_id: str, linear_backend: str, routed_backend: str, confidence_head_loaded: bool,
) -> dict[str, int]:
    """Owned modular draft TP*DP/global-EP parameters, not normal memory.

    Reuse the standalone constructor inventory with actual distributed
    expert ownership/physical Marlin shards. Model-SP replicates shared
    experts; otherwise only TP shards them. Main/Markov/confidence and
    context projections remain replicated. Target vocabulary aliases,
    native arenas, runner/kernel state, workspace/reserves and provider
    selection are separate; this API admits no allocator policy.
    """
    if hardware_id != "nvidia_h200" or linear_backend != "deep_gemm" or routed_backend != "marlin":
        raise NotImplementedError("DSpark DP inventory requires explicit H200 DeepGEMM/Marlin storage")
    if type(confidence_head_loaded) is not bool:
        raise TypeError("DSpark confidence-head loading must be explicitly boolean")
    local = _dsv4_dspark_data_rank_shape(model, parallelism, logical_rank)
    return _dsv4_dspark_owned_parameter_bytes(model, parallelism, local, confidence_head_loaded)


def _dsv4_dspark_owned_parameter_bytes(model, parallelism, local, confidence_head_loaded):
    parameters, _, _ = _dsv4_non_routed_loader_inventory(model, parallelism, attention_backend="flashmla",
        local_shared_intermediate=local["shared_down_kn"][0], layer_indices=local["layers"])
    del parameters["model.embed_tokens.weight"]
    del parameters["lm_head.weight"]
    for name in parameters:
        if name.endswith(".weight_scale_inv"):
            parameters[name] *= 4
    experts, intermediate, hidden = len(local["expert_ids"]), local["routed_down_kn"][0], model.hidden_size
    for layer in local["layers"]:
        prefix = f"model.layers.{layer}.ffn.experts"
        parameters[f"{prefix}.w13_weight"] = experts * 2 * intermediate * (hidden // 2)
        parameters[f"{prefix}.w2_weight"] = experts * hidden * (intermediate // 2)
        parameters[f"{prefix}.w13_weight_scale"] = experts * 2 * intermediate * (hidden // 32)
        parameters[f"{prefix}.w2_weight_scale"] = experts * hidden * (intermediate // 32)
    main_inputs = hidden * len(model.dspark_target_layer_ids)
    parameters["model.main_proj.weight"] = hidden * main_inputs
    parameters["model.main_proj.weight_scale_inv"] = -(-hidden // 128) * -(-main_inputs // 128) * 4
    parameters["model.main_norm.weight"] = hidden * 2
    parameters["model.markov_head.markov_w1.weight"] = model.vocab_size * model.dspark_markov_rank * 2
    parameters["model.markov_head.markov_w2.weight"] = padded_vocab_size(model.vocab_size) * model.dspark_markov_rank * 2
    if confidence_head_loaded:
        parameters["model.confidence_head.proj.weight"] = (hidden + model.dspark_markov_rank) * 4
    return parameters


def dsv4_dspark_model_buffer_bytes(
    model: ModelConfig, parallelism: ParallelConfig, logical_rank: int, *, max_batched_tokens: int,
) -> dict[str, int]:
    """Standalone constructor buffers; no target aliases/native cache or fit.

    Every draft rank constructs the top-k buffer even though all draft layers
    use SWA-only attention. Equal SWA rotary constructor keys share storage.
    No first-target-layer mHC broadcast is constructed in appended layers.
    Request inputs, model-runner buffers, KV, shared workspace/kernel state,
    target auxiliary captures, conversion peaks and reserves are excluded.
    """
    _dsv4_dspark_rank_shape(model, parallelism, logical_rank)
    return _dsv4_dspark_constructor_buffers(model, max_batched_tokens)


def dsv4_dspark_data_parallel_model_buffer_bytes(
    model: ModelConfig, parallelism: ParallelConfig, logical_rank: int, *, max_batched_tokens: int,
) -> dict[str, int]:
    """Complete draft constructor buffers on each distributed source rank."""
    _dsv4_dspark_data_rank_shape(model, parallelism, logical_rank)
    return _dsv4_dspark_constructor_buffers(model, max_batched_tokens)


def _dsv4_dspark_constructor_buffers(model, max_batched_tokens):
    if type(max_batched_tokens) is not int or max_batched_tokens <= 0:
        raise ValueError("DSpark model buffers require positive integer max_batched_tokens")
    return {"model.topk_indices_buffer": max_batched_tokens * model.index_topk * 4,
        "model.shared_swa_rotary.cos_sin_cache": model.max_position_embeddings * model.qk_rope_head_dim * 4}


def dsv4_dspark_combined_rank_storage_bytes(
    target: ModelConfig, draft: ModelConfig, parallelism: ParallelConfig, logical_rank: int, *,
    max_batched_tokens: int, confidence_head_loaded: bool,
    pipeline_layer_partition: tuple[int, ...] | None = None,
) -> dict:
    """Owned target/draft parameters and model buffers on one physical rank.

    Declared H200 SM90 DeepGEMM/Marlin storage only. A complete PP1 draft
    lives on each last-stage TP/DP worker. Vocabulary sharing is local to
    that worker; a PP>1 final stage first constructs its own target embedding.
    Equal SWA RoPE keys share storage, while target/draft top-k, norms and
    mHC parameters remain independent. Activations, native KV, compiler/
    library state and declared runtime reserves are separate accounting.
    """
    if not isinstance(target, ModelConfig) or not isinstance(draft, ModelConfig):
        raise TypeError("DSpark combined storage requires normalized target/draft configs")
    if (target.is_dspark_draft or not draft.is_dspark_draft or draft.model_id != target.model_id + "-DSpark"
            or any(getattr(target, name) != getattr(draft, name) for name in (
                "num_hidden_layers", "hidden_size", "num_attention_heads", "num_experts", "moe_intermediate_size",
                "vocab_size", "head_dim", "q_lora_rank", "qk_rope_head_dim", "hc_mult", "o_groups",
                "o_lora_rank", "window_size", "max_position_embeddings", "compute_dtype", "weight_dtype"))):
        raise ValueError("DSpark combined storage requires its matching registered target and draft")
    if type(confidence_head_loaded) is not bool:
        raise TypeError("DSpark confidence-head loading must be explicitly boolean")
    local = _dsv4_parallel_rank_shape(target, parallelism, logical_rank, pipeline_layer_partition)
    target_parameters = dsv4_parallel_postload_parameter_bytes(target, parallelism, logical_rank,
        pipeline_layer_partition=pipeline_layer_partition)
    target_buffers = dsv4_parallel_model_buffer_bytes(target, parallelism, logical_rank,
        max_batched_tokens=max_batched_tokens, pipeline_layer_partition=pipeline_layer_partition)
    draft_parameters, draft_buffers, aliases = {}, {}, []
    tp, pp = parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size
    last = local["pipeline_stage"] == pp - 1
    draft_rank = None
    if last:
        if pp > 1:
            target_parameters["model.embed_tokens.weight"] = padded_vocab_size(target.vocab_size) // tp * target.hidden_size * 2
        target_buffers["model._mtp_hidden_buffer"] = max_batched_tokens * target.hc_mult * target.hidden_size * 2
        draft_parallel = replace(parallelism, pipeline_parallel_size=1)
        draft_rank = logical_rank // (tp * pp) * tp + logical_rank % tp
        if parallelism.data_parallel_size > 1:
            parameters = dsv4_dspark_data_parallel_postload_parameter_bytes
            buffers = dsv4_dspark_data_parallel_model_buffer_bytes
        else:
            parameters = dsv4_dspark_postload_parameter_bytes
            buffers = dsv4_dspark_model_buffer_bytes
        draft_parameters = parameters(draft, draft_parallel, draft_rank, hardware_id="nvidia_h200",
            linear_backend="deep_gemm", routed_backend="marlin", confidence_head_loaded=confidence_head_loaded)
        draft_buffers = buffers(draft, draft_parallel, draft_rank, max_batched_tokens=max_batched_tokens)
        aliases.extend(({"alias": "draft/model.embed_tokens.weight", "owner": "target/model.embed_tokens.weight"},
            {"alias": "draft/lm_head.weight", "owner": "target/lm_head.weight"}))

        def swa_rope_key(model):
            semantics = model.target_semantics
            if semantics is None or semantics.rope_type != "dsv4_yarn":
                raise NotImplementedError("DSpark combined buffers require their source SWA rotary constructor")
            # build_deepseek_v4_rope overwrites the SWA factor/original limit
            # and both mscales. Remaining key fields must still agree.
            return (model.head_dim, model.qk_rope_head_dim, model.max_position_embeddings,
                model.compute_dtype, semantics.rope_theta, semantics.rope_beta_fast, semantics.rope_beta_slow,
                "deepseek_yarn", False, 1.0, model.max_position_embeddings, 0, 0)

        rope = "model.shared_swa_rotary.cos_sin_cache"
        if rope in target_buffers and swa_rope_key(target) == swa_rope_key(draft):
            if target_buffers[rope] != draft_buffers[rope]:
                raise ValueError("equal DSpark SWA rotary keys disagree on their backing bytes")
            del draft_buffers[rope]
            aliases.append({"alias": "draft/" + rope, "owner": "target/" + rope})
    parameter_bytes = sum(target_parameters.values()) + sum(draft_parameters.values())
    buffer_bytes = sum(target_buffers.values()) + sum(draft_buffers.values())
    return {"logical_rank": logical_rank, "pipeline_stage": local["pipeline_stage"], "draft_logical_rank": draft_rank,
        "target_parameters": target_parameters, "draft_owned_parameters": draft_parameters,
        "target_model_buffers": target_buffers, "draft_owned_model_buffers": draft_buffers, "aliases": tuple(aliases),
        "known_parameter_bytes": parameter_bytes, "known_constructor_buffer_bytes": buffer_bytes,
        "known_resident_bytes": parameter_bytes + buffer_bytes, "normal_capacity_accepted": False,
        "scope": "declared_source_parameters_and_model_buffers; KV_activations_runtime_reserves_separate"}


def dsv4_model_buffer_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    max_batched_tokens: int,
) -> dict[str, int]:
    """Unique V4 model-owned persistent tensors under the CUDA MTP-off source.

    The model owns one shared INT32 top-k buffer, not one per C4 indexer.
    finalize_mhc_broadcast_weights retains one FP32 first-layer reduction.
    get_rope caches the FP32 rotary keys actually used by target layers.
    Flash uses SWA and compressed; Pro MTP-off uses compressed only. C4/C128,
    compressor and indexer share these objects, not one per layer. The draft
    tail does not add an MTP-off key. Extent uses checkpoint limits, not the shorter
    Serving max_model_len. CUDA has no ROCm-only compressor scratch, and
    MTP-off has no target hidden buffer. Runtime/attention metadata, library
    state, KV storage, activations and reserves remain outside this scope.
    """
    # Reuse the complete dtype/parallelism/layout source boundary.
    dsv4_non_routed_loader_payload_bytes(model, parallelism, attention_backend="flashmla")
    return dsv4_parallel_model_buffer_bytes(model, parallelism, 0, max_batched_tokens=max_batched_tokens)


def dsv4_parallel_model_buffer_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    logical_rank: int,
    *,
    max_batched_tokens: int,
    pipeline_layer_partition: tuple[int, ...] | None = None,
) -> dict[str, int]:
    """Declared model-owned persistent buffers for one modular V4 target rank.

    Every GPU worker constructs one shared INT32 top-k buffer, even when
    its DP replica has no request. RoPE cache keys follow only its actual
    PP layers; compressor/indexer modules share equal constructor keys.
    Only stage0 retains the first-layer FP32 mHC broadcast reduction.
    Weights, KV, runner input/metadata buffers, workspace, library state,
    conversion peaks and reserves are separate. This constructor inventory
    does not bind normal memory policy, selected provider, or GPU execution.
    """
    local = _dsv4_parallel_rank_shape(model, parallelism, logical_rank, pipeline_layer_partition)
    if type(max_batched_tokens) is not int or max_batched_tokens <= 0:
        raise ValueError("V4 model buffers require positive integer max_batched_tokens")
    semantics = model.target_semantics
    if semantics is None or semantics.rope_type != "dsv4_yarn":
        raise NotImplementedError("V4 buffers require the declared V4 YaRN source")
    extent = semantics.rope_original_max_position_embeddings * semantics.rope_factor
    if int(extent) != extent:
        raise NotImplementedError("V4 compressed rotary cache extent must be integral")
    entries = {"model.topk_indices_buffer": max_batched_tokens * model.index_topk * 4}
    if local["pipeline_stage"] == 0:
        entries["model.layers.0.hc_attn_fn_broadcast"] = (2 + model.hc_mult) * model.hc_mult * model.hidden_size * 4
    # kernel_compress_ratios includes the draft tail; MTP-off constructs
    # target layers only. Pro's sole SWA-only ratio is in that draft tail.
    target_ratios = tuple(model.kernel_compress_ratios[layer] for layer in local["layers"])
    if any(ratio <= 1 for ratio in target_ratios):
        entries["model.shared_swa_rotary.cos_sin_cache"] = model.max_position_embeddings * model.qk_rope_head_dim * 4
    if any(ratio > 1 for ratio in target_ratios):
        entries["model.shared_compressed_rotary.cos_sin_cache"] = int(extent) * model.qk_rope_head_dim * 4
    return entries


def flash_gpu_decode_capacity(
    model: ModelConfig,
    *,
    scheme: str,
    concurrency: int,
    context_tokens: int,
    hbm_capacity_bytes: int,
    source_persistent_bytes: int,
    source_workspace_bytes: int | None,
    source_logical_arena_bytes: int | None = None,
) -> dict:
    """Logical, MTP-off Flash GPU D-only inventory; no native allocator claim.

    ``tp8`` retains every expert with one eighth of its intermediate width;
    ``dp8_ep8`` retains 32 complete experts per rank and replicates non-routed
    parameters. Both use the source loader's MXFP4/group32 and FP8/group128
    storage, not the H200 post-load/Marlin storage profile. EP ownership is
    the existing integer linear partition, without padding or replication.

    Persistent and workspace inputs are rank-local maxima from the actual
    virtual source. Workspace may be unknown, never an implied zero. Cache
    residency uses the model owner's seven lifecycle/page groups, whereas
    the optionally reported source virtual arena is not resident storage.
    All normal-runtime reserves and backend-specific layouts remain unbound.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("Flash GPU decode capacity requires ModelConfig")
    if (model.model_id != "deepseek-ai/DeepSeek-V4-Flash"
            or model.model_type != "deepseek_v4" or model.num_hidden_layers != 43
            or model.hidden_size != 4096 or model.moe_intermediate_size != 2048
            or model.num_experts != 256 or model.num_experts_per_tok != 6
            or model.weight_dtype != "fp8" or model.expert_dtype != "fp4"
            or model.compute_dtype != "bf16"):
        raise NotImplementedError("GPU D-only inventory requires the unchanged Flash geometry")
    if scheme not in ("tp8", "dp8_ep8"):
        raise ValueError("Flash GPU decode capacity scheme must be tp8 or dp8_ep8")
    for name, value in (("concurrency", concurrency), ("context_tokens", context_tokens),
            ("hbm_capacity_bytes", hbm_capacity_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"Flash GPU decode {name} must be a positive integer")
    if concurrency not in (1, 128, 192):
        raise ValueError("Flash GPU decode capacity permits internal C1 or formal C128/C192 only")
    if context_tokens > model.max_position_embeddings:
        raise ValueError("Flash GPU decode context exceeds checkpoint capacity")
    if type(source_persistent_bytes) is not int or source_persistent_bytes < 0:
        raise ValueError("source persistent bytes must be an explicit nonnegative integer")
    for name, value in (("source_workspace_bytes", source_workspace_bytes),
            ("source_logical_arena_bytes", source_logical_arena_bytes)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer or explicitly unknown")

    tp = 8 if scheme == "tp8" else 1
    parameter_parallelism = ParallelConfig(tensor_parallel_size=tp)
    non_routed = dsv4_non_routed_loader_payload_bytes(
        model, parameter_parallelism, attention_backend="flashmla")
    routed = dsv4_routed_loader_payload_bytes(model, parameter_parallelism)
    routed_keys = ("w13_packed_bytes_per_layer", "w2_packed_bytes_per_layer",
        "w13_ue8m0_bytes_per_layer", "w2_ue8m0_bytes_per_layer")
    ep = 1 if scheme == "tp8" else 8
    routed_per_layer = {key: _require_divisible(routed[key], ep, key) for key in routed_keys}
    routed_bytes = model.num_hidden_layers * sum(routed_per_layer.values())
    non_routed_bytes = sum(non_routed.values())
    weights = routed_bytes + non_routed_bytes

    # All schemes retain the single KV head on each active attention rank.
    # Fractions describe the model owner's amortized stream-byte convention;
    # the page inventory below rounds each lifecycle group independently.
    exact_lifecycle = model.dsv4_cache_event_bytes(0, context_tokens)
    lifecycle_bytes = -(-exact_lifecycle.numerator // exact_lifecycle.denominator)
    cache_pages = []
    for group in model.dsv4_scheduler_cache_groups(16):
        retained = context_tokens if group.window_tokens is None else min(
            context_tokens, group.window_tokens)
        blocks = -(-retained // group.block_tokens)
        cache_pages.append({"kind": group.kind, "lifecycle": group.lifecycle,
            "retained_tokens_per_request": retained, "block_tokens": group.block_tokens,
            "blocks_per_request_per_layer": blocks, "page_bytes": group.page_bytes,
            "layer_indices": group.layer_indices,
            "resident_bytes_per_request": blocks * group.page_bytes * len(group.layer_indices)})
    pages_per_request = sum(entry["resident_bytes_per_request"] for entry in cache_pages)
    ranks = []
    for rank in range(8):
        requests = (tuple(range(concurrency)) if scheme == "tp8" else
            tuple(request for request in range(concurrency) if request % 8 == rank))
        owned = (tuple(range(model.num_experts)) if scheme == "tp8" else
            tuple(linear_expert_partition(model.num_experts, 8, rank)))
        cache = len(requests) * pages_per_request
        known_without_workspace = weights + source_persistent_bytes + cache
        known_required = (known_without_workspace if source_workspace_bytes is None else
            known_without_workspace + source_workspace_bytes)
        ranks.append({"rank": rank, "local_request_ids": requests,
            "local_request_count": len(requests), "owned_experts": owned,
            "expert_intermediate_size": routed["intermediate_size_per_rank"],
            "routed_loader_payload_bytes": routed_bytes,
            "non_routed_loader_payload_bytes": non_routed_bytes,
            "weight_payload_bytes": weights, "known_persistent_bytes": source_persistent_bytes,
            "cache_lifecycle_bytes": len(requests) * lifecycle_bytes,
            "cache_page_budget_bytes": cache,
            "source_workspace_bytes": source_workspace_bytes,
            "source_logical_arena_bytes": source_logical_arena_bytes,
            "known_required_lower_bound_bytes": known_required,
            "known_inventory_headroom_bytes": hbm_capacity_bytes - known_required,
            "capacity_rejected": known_required > hbm_capacity_bytes})
    rejected = any(row["capacity_rejected"] for row in ranks)
    return {"scheme": scheme, "concurrency": concurrency, "context_tokens": context_tokens,
        "hbm_capacity_bytes_per_rank": hbm_capacity_bytes,
        "weight_storage_scope": "logical_loader_payload_not_native_postload",
        "non_routed_loader_parameters": non_routed, "routed_loader_bytes_per_layer": routed_per_layer,
        "cache_lifecycle_bytes_per_request": lifecycle_bytes,
        "cache_lifecycle_exact_fraction_per_request": (exact_lifecycle.numerator, exact_lifecycle.denominator),
        "cache_page_budget_bytes_per_request": pages_per_request, "cache_page_groups": tuple(cache_pages),
        "ranks": tuple(ranks), "peak_known_required_lower_bound_bytes": max(
            row["known_required_lower_bound_bytes"] for row in ranks),
        "source_workspace_bound": source_workspace_bytes is not None,
        "unknown_native_workspace_bytes": None, "unknown_allocator_runtime_reserve_bytes": None,
        "native_cache_layout_equivalent": False, "normal_capacity_accepted": False,
        "logical_arena_charged_as_resident": False,
        "status": "capacity_rejected" if rejected else "known_inventory_fits_reserve_unbound"}


def flash_fp4_decode_capacity(
    model: ModelConfig,
    *,
    deployment: str,
    concurrency: int,
    hbm_capacity_bytes: int,
    initial_context_tokens: int = 100000,
    decode_tokens: int = 32,
    known_workspace_bytes_per_rank: int | None = None,
    fpga_cards: int = 16,
    gpu_count: int = 8,
    microbatch_size: int = 128,
) -> dict:
    """Conditional static capacity for the all-matrix-FP4 Flash scenario.

    Matrix values use one packed nibble plus one uint8 group32 scale, with
    no resident FP8 matrix/scale copy. This is a user-declared storage
    scenario, not a native quantizer/provider or post-load layout binding.
    Source parameter shapes, nonmatrix dtypes, model buffers and KV lifecycle
    pages come from their existing owners. TP8/EP8 keeps 32 complete I2048
    experts per GPU; the TP axis shards only the non-routed inventory.

    The max initial context reserves every requested output token and uses
    the named full-checkpoint model-buffer inventory. Unknown workspace and
    allocator/runtime reserves stay unknown; they never certify deployment.
    A supplied known workspace is a rank-local resident requirement, not a
    sum of nonconcurrent source requests or a virtual address-space extent.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("FP4 Flash capacity requires ModelConfig")
    if ((model.model_id, model.num_hidden_layers, model.hidden_size, model.moe_intermediate_size, model.num_experts) not in (
            ("deepseek-ai/DeepSeek-V4-Flash", 43, 4096, 2048, 256),
            ("deepseek-ai/DeepSeek-V4-Pro", 61, 7168, 3072, 384))
            or model.model_type != "deepseek_v4" or model.num_experts_per_tok != 6
            or model.weight_dtype != "fp8" or model.expert_dtype != "fp4"
            or model.compute_dtype != "bf16"):
        raise NotImplementedError("all-matrix FP4 capacity requires supported V4 source geometry")
    if deployment not in ("tp8_ep8", "dp8_ep8", "dp8_fpga"):
        raise ValueError("FP4 Flash deployment must be tp8_ep8, dp8_ep8 or dp8_fpga")
    if type(fpga_cards) is not int or fpga_cards not in (16, 32):
        raise ValueError("FPGA cards must be 16 or 32 (two chips per card)")
    if deployment != "dp8_fpga" and fpga_cards != 16:
        raise ValueError("FPGA card override requires dp8_fpga")
    chip_count = fpga_cards * 2
    if type(gpu_count) is not int or gpu_count not in (8, 16):
        raise ValueError("GPU count must be 8 or 16")
    if gpu_count == 16 and deployment != "dp8_fpga":
        raise ValueError("16 GPUs require the FPGA direct-switch scenario")
    if type(microbatch_size) is not int or microbatch_size not in (32, 64, 128, 256):
        raise ValueError("invalid microbatch size")
    if gpu_count == 8 and microbatch_size != 128:
        raise ValueError("8-GPU compatibility scenario requires BS128")
    if type(concurrency) is not int or concurrency not in tuple(microbatch_size * mb for mb in (1, 2, 3)):
        raise ValueError("concurrency must equal microbatch size times 1, 2 or 3")
    for name, value in (("hbm_capacity_bytes", hbm_capacity_bytes),
            ("initial_context_tokens", initial_context_tokens), ("decode_tokens", decode_tokens)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"FP4 Flash {name} must be a positive integer")
    end_context = initial_context_tokens + decode_tokens
    if end_context > model.max_position_embeddings:
        raise ValueError("FP4 Flash initial context plus decode tokens exceeds the model limit")
    if (known_workspace_bytes_per_rank is not None and
            (type(known_workspace_bytes_per_rank) is not int or known_workspace_bytes_per_rank < 0)):
        raise ValueError("known workspace must be a nonnegative resident byte requirement or None")

    tp = 8 if deployment == "tp8_ep8" else 1
    local_requests = concurrency if tp == 8 else concurrency // gpu_count
    parallelism = ParallelConfig(tensor_parallel_size=tp)
    entries, element_widths, shapes = _dsv4_non_routed_loader_inventory(
        model, parallelism, attention_backend="flashmla")
    matrices, nonmatrices, removed_scales = [], [], []
    for path, original_bytes in entries.items():
        shape = shapes[path]
        if shape is None:
            raise ValueError("FP4 Flash inventory requires an explicit source shape for every parameter")
        if path.endswith(".weight_scale_inv"):
            removed_scales.append({"parameter": path, "original_shape": shape,
                "original_bytes_not_retained": original_bytes})
            continue
        is_matrix = (path.endswith(".weight") and len(shape) >= 2) or path.endswith("_fn")
        if is_matrix:
            groups_per_row = _require_divisible(shape[-1], 32, "FP4 matrix row width")
            rows = 1
            for extent in shape[:-1]:
                rows *= extent
            values = rows * shape[-1]
            scales = rows * groups_per_row
            matrices.append({"parameter": path, "shape": shape,
                "fp4_value_elements": values, "fp4_value_bytes": values // 2,
                "uint8_group32_scale_bytes": scales, "total_bytes": values // 2 + scales})
        else:
            # APE is a positional addend and tid2eid is an integer routing
            # table; their two dimensions do not make them matrix operands.
            if len(shape) >= 2 and not (path.endswith(".ape") or path.endswith(".gate.tid2eid")):
                raise ValueError(f"unclassified nonmatrix Flash parameter: {path}")
            nonmatrices.append({"parameter": path, "shape": shape,
                "source_element_bytes": element_widths[path], "resident_bytes": original_bytes})
    non_routed_values = sum(row["fp4_value_bytes"] for row in matrices)
    non_routed_scales = sum(row["uint8_group32_scale_bytes"] for row in matrices)
    nonmatrix_bytes = sum(row["resident_bytes"] for row in nonmatrices)
    routed = dsv4_routed_loader_payload_bytes(model, ParallelConfig(tensor_parallel_size=1))
    global_routed_values = model.num_hidden_layers * (
        routed["w13_packed_bytes_per_layer"] + routed["w2_packed_bytes_per_layer"])
    global_routed_scales = model.num_hidden_layers * (
        routed["w13_ue8m0_bytes_per_layer"] + routed["w2_ue8m0_bytes_per_layer"])
    offload = deployment == "dp8_fpga"
    routed_values = 0 if offload else _require_divisible(global_routed_values, 8, "EP8 FP4 values")
    routed_scales = 0 if offload else _require_divisible(global_routed_scales, 8, "EP8 group32 scales")
    values, scales = non_routed_values + routed_values, non_routed_scales + routed_scales
    parameters = values + scales + nonmatrix_bytes
    buffers = dsv4_model_buffer_bytes(model, parallelism, max_batched_tokens=local_requests)
    persistent_bytes = sum(buffers.values())
    fixed_bytes = parameters + persistent_bytes
    if known_workspace_bytes_per_rank is not None:
        fixed_bytes += known_workspace_bytes_per_rank
    groups = model.dsv4_scheduler_cache_groups(16)

    def cache_pages(tokens):
        pages = []
        for group in groups:
            retained = tokens if group.window_tokens is None else min(tokens, group.window_tokens)
            blocks = -(-retained // group.block_tokens)
            pages.append({"kind": group.kind, "lifecycle": group.lifecycle,
                "retained_tokens_per_request": retained, "block_tokens": group.block_tokens,
                "blocks_per_request_per_layer": blocks, "page_bytes": group.page_bytes,
                "layer_indices": group.layer_indices,
                "resident_bytes_per_request": blocks * group.page_bytes * len(group.layer_indices)})
        return tuple(pages)

    def page_bytes(tokens):
        return sum(row["resident_bytes_per_request"] for row in cache_pages(tokens))

    pages = cache_pages(end_context)
    pages_per_request = sum(row["resident_bytes_per_request"] for row in pages)
    rank_kv = local_requests * pages_per_request
    known_bytes = fixed_bytes + rank_kv
    ranks = []
    for rank in range(gpu_count):
        ranks.append({"rank": rank, "gpu_server_index": rank // 8, "nic_id": f"gpu-nic:{rank // 2}", "local_requests": local_requests,
            "local_request_ids": (tuple(range(concurrency)) if tp == 8 else
                tuple(range(rank, concurrency, gpu_count))),
            "routed_experts_on_gpu": (() if offload else linear_expert_partition(model.num_experts, 8, rank)),
            "routed_intermediate_per_owned_expert": model.moe_intermediate_size,
            "fp4_matrix_value_bytes": values, "matrix_group32_uint8_scale_bytes": scales,
            "nonmatrix_parameter_bytes": nonmatrix_bytes, "weight_payload_bytes": parameters,
            "known_model_persistent_buffer_bytes": persistent_bytes,
            "kv_page_budget_bytes": rank_kv, "known_workspace_bytes": known_workspace_bytes_per_rank,
            "known_residency_inventory_bytes": known_bytes,
            "headroom_before_unknown_reserve_bytes": hbm_capacity_bytes - known_bytes,
            "unknown_workspace_bytes": None,
            "unknown_additional_native_workspace_bytes": None,
            "unknown_allocator_runtime_reserve_bytes": None})
    chips = []
    if offload:
        chip_values = _require_divisible(global_routed_values, chip_count, "FPGA EP FP4 values")
        chip_scales = _require_divisible(global_routed_scales, chip_count, "FPGA EP group32 scales")
        for chip in range(chip_count):
            chips.append({"chip_rank": chip, "physical_rank": gpu_count + chip,
                "card_index": chip // 2, "fpga_server_index": chip // 16,
                "chip_index_on_card": chip % 2,
                "hbm_capacity_bytes": 32000000000, "ddr_capacity_bytes": 16000000000,
                "owned_experts": linear_expert_partition(model.num_experts, chip_count, chip),
                "intermediate_per_owned_expert": model.moe_intermediate_size,
                "fp4_matrix_value_bytes": chip_values, "matrix_group32_uint8_scale_bytes": chip_scales,
                "known_routed_weight_residency_bytes": chip_values + chip_scales,
                "unknown_hbm_workspace_bytes": None, "unknown_ddr_staging_bytes": None,
                "unknown_runtime_reserve_bytes": None, "gpu_kv_stays_on_gpu": True})
    # Conditional static upper bound: no unknown reserve is asserted zero.
    # An infeasible fixed inventory has no admissible initial context.
    maximum = None
    if fixed_bytes + local_requests * page_bytes(decode_tokens) <= hbm_capacity_bytes:
        low, high = 0, model.max_position_embeddings - decode_tokens
        while low < high:
            middle = (low + high + 1) // 2
            if fixed_bytes + local_requests * page_bytes(middle + decode_tokens) <= hbm_capacity_bytes:
                low = middle
            else:
                high = middle - 1
        maximum = low
    raw = model.dsv4_cache_event_bytes(0, end_context)
    return {"deployment": deployment, "gpu_count": gpu_count, "microbatch_size": microbatch_size, "concurrency": concurrency, "attention_tensor_parallel_size": tp,
        "initial_context_tokens": initial_context_tokens, "decode_tokens": decode_tokens,
        "end_context_tokens": end_context, "hbm_capacity_bytes_per_rank": hbm_capacity_bytes,
        "matrix_storage_scope": "user_declared_packed_fp4_values_and_uint8_group32_scales_no_fp8_weight_copy",
        "non_routed_matrix_parameters": tuple(matrices), "nonmatrix_parameters": tuple(nonmatrices),
        "old_fp8_scale_parameters_removed": tuple(removed_scales),
        "non_routed_fp4_value_bytes": non_routed_values, "non_routed_group32_scale_bytes": non_routed_scales,
        "global_routed_fp4_value_bytes": global_routed_values, "global_routed_group32_scale_bytes": global_routed_scales,
        "gpu_ranks": tuple(ranks), "fpga_chips": tuple(chips),
        "model_persistent_buffer_inventory": buffers,
        "persistent_scope": "existing_full_checkpoint_model_buffer_inventory_estimate_not_actual_peak",
        "cache_page_groups": pages, "cache_page_budget_bytes_per_request": pages_per_request,
        "cache_lifecycle_exact_fraction_bytes_per_request": (raw.numerator, raw.denominator),
        "known_workspace_bytes_per_rank": known_workspace_bytes_per_rank,
        "known_workspace_bound": known_workspace_bytes_per_rank is not None,
        "unknown_native_workspace_bytes_per_rank": None, "unknown_allocator_runtime_reserve_bytes_per_rank": None,
        "conditional_static_max_initial_context_tokens": maximum,
        "conditional_static_max_total_context_tokens": None if maximum is None else maximum + decode_tokens,
        "maximum_context_scope": "selected_static_inventory_and_pages_before_unknown_workspace_reserves",
        "capacity_rejection_basis": "selected_static_inventory_not_native_allocator_measurement",
        "parameters_and_kv_alone_exceed_capacity": parameters + rank_kv > hbm_capacity_bytes,
        "logical_arena_charged_as_resident": False, "native_cache_layout_equivalent": False,
        "normal_capacity_accepted": False, "native_provider_bound": False,
        "status": "capacity_rejected" if known_bytes > hbm_capacity_bytes or any(chip["known_routed_weight_residency_bytes"] > chip["hbm_capacity_bytes"] for chip in chips) else
            "known_inventory_fits_workspace_and_reserve_unbound"}


def _attention_parameter_count(model: ModelConfig, tp_size: int) -> int:
    if model.attention_layout != "standard_gqa":
        raise NotImplementedError(
            "analytical model residency currently supports standard GQA; "
            f"got {model.attention_layout!r}")
    if model.attention_bias:
        raise NotImplementedError(
            "analytical model residency does not yet have a source-backed "
            "attention-bias placement contract")
    query_heads = _require_divisible(
        model.num_attention_heads, tp_size, "num_attention_heads")
    query_width = query_heads * model.head_dim
    kv_width = (
        model.kv_heads_per_tensor_parallel_rank(tp_size) * model.head_dim)
    projections = model.hidden_size * (2 * query_width + 2 * kv_width)
    qk_norm = 2 * model.head_dim if model.qk_norm else 0
    return projections + qk_norm


def glm_block_fp8_loader_payload_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    include_mla_absorption: bool,
    logical_rank: int | None = None,
    pipeline_layer_partition: tuple[int, ...] | None = None,
    include_mtp: bool = False,
) -> dict[str, int]:
    """GLM block-FP8 payload under explicit source placement, not normal KV.

    Fp8LinearMethod/create_fp8_scale_parameter and Fp8MoEMethod.create_weights
    define FP8 values plus FP32 128x128 block scales. Exact BF16 exclusions
    and quant_config=None modules remain separate parameters. EP-off shards
    each expert's intermediate dimension; EP=TP shards expert ownership.

    include_mla_absorption declares the ordinary CUDA/CP1 MLA post-load
    branch: retain original FP8 kv_b_proj and add the disjoint BF16 W_UK_T
    and W_UV views of its dequantized storage once. No extra dequant tensor
    is charged on top of these views. Other backend conversions, persistent
    buffers, workspace/reserves and actual loader/provider binding are
    excluded. An explicit logical_rank projects the existing exact TP/DP/EP/PP
    source domain; omitting it preserves the legacy PP1-only API. Optional
    MTP owns one appended layer on the last stage, with an independent PP>1
    embedding and a target-aliased head. Repeated draft calls do not multiply
    resident weights. This inventory is not allocator-ready residency.
    """
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("GLM FP8 payload requires ModelConfig and ParallelConfig")
    if not isinstance(include_mla_absorption, bool):
        raise TypeError("include_mla_absorption must be boolean")
    if type(include_mtp) is not bool:
        raise TypeError("GLM payload include_mtp must be boolean")
    if (model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S
            or model.model_type != "glm_moe_dsa" or model.weight_dtype != "fp8"
            or model.compute_dtype != "bf16" or model.tie_word_embeddings
            or model.quantized_weight_targets != ("Linear",)):
        raise NotImplementedError("GLM FP8 payload requires the exact official GLM-5.2-FP8/GLM-5.3 config")
    if logical_rank is None:
        if not parallelism.is_rank_local or parallelism.pipeline_parallel_size != 1:
            raise NotImplementedError("GLM FP8 payload requires PP=DP=CP=SP=1 or an explicit logical_rank")
        logical_rank = 0
    from src.serving.workload.model_config import is_glm_fp8_tensor_model_scope, is_glm_fp8_data_parallel_model_scope
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.parallelism.tp import mla_local_shape
    legal = (is_glm_fp8_data_parallel_model_scope(model, parallelism) if parallelism.data_parallel_size > 1 else
        parallelism.is_rank_local and is_glm_fp8_tensor_model_scope(model, parallelism.tensor_parallel_size)
        and 1 <= parallelism.pipeline_parallel_size <= model.num_hidden_layers
        and (not parallelism.enable_expert_parallel or model.num_experts % parallelism.expert_parallel_size == 0))
    if not legal:
        raise NotImplementedError("GLM FP8 payload requires whole128 source TP/DP/EP shards, nonempty PP and CP1")
    if type(logical_rank) is not int or not 0 <= logical_rank < parallelism.worker_count:
        raise ValueError("GLM payload logical rank is outside its worker pool")
    if include_mtp and (model.mtp_architecture != "glm_moe_dsa_mtp" or model.num_mtp_layers != 1):
        raise NotImplementedError("GLM payload requires one source-defined appended MTP layer")
    plan = build_parallel_plan(parallelism, num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=pipeline_layer_partition)
    local = mla_local_shape(model, plan, logical_rank)
    tp = parallelism.tensor_parallel_size
    h = model.hidden_size
    heads = _require_divisible(model.num_attention_heads, tp, "attention heads")
    experts = len(local["expert_ids"])
    intermediate = local["down_kn"][0]
    if h % 128 or intermediate % 128:
        raise NotImplementedError("GLM FP8 payload requires unrefined 128x128 expert scales")
    entries: dict[str, int] = {}

    def add(path: str, elements: int, element_bytes: int) -> None:
        if path in entries:
            raise ValueError(f"duplicate GLM FP8 payload parameter: {path}")
        entries[path] = elements * element_bytes

    def linear(path: str, outputs: int, inputs: int, *, quantized: bool = True) -> None:
        dtype = model.weight_dtype_for_module(path, "Linear") if quantized else "bf16"
        if dtype == "bf16":
            add(path + ".weight", outputs * inputs, 2)
        elif dtype == "fp8":
            add(path + ".weight", outputs * inputs, 1)
            add(path + ".weight_scale_inv", ((outputs + 127) // 128) * ((inputs + 127) // 128), 4)
        else:
            raise NotImplementedError(f"unsupported GLM payload linear dtype: {dtype}")

    vocab = _require_divisible(padded_vocab_size(model.vocab_size), tp, "vocabulary")
    if local["pipeline_stage"] == 0:
        add("model.embed_tokens.weight", vocab * h, 2)
    last_stage = local["pipeline_stage"] == parallelism.pipeline_parallel_size - 1
    if last_stage:
        linear("lm_head", vocab, h)
        add("model.norm.weight", h, 2)
    draft = include_mtp and last_stage
    layers = local["layers"] + ((model.num_hidden_layers,) if draft else ())
    for layer in layers:
        is_draft = layer == model.num_hidden_layers
        prefix = f"model.layers.{layer}"
        attn = prefix + ".self_attn"
        linear(attn + ".fused_qkv_a_proj", model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim, h)
        linear(attn + ".q_b_proj", heads * model.head_dim, model.q_lora_rank)
        linear(attn + ".kv_b_proj", heads * (model.qk_nope_head_dim + model.v_head_dim), model.kv_lora_rank)
        linear(attn + ".o_proj", h, heads * model.v_head_dim)
        for name, width in (("q_a_layernorm", model.q_lora_rank), ("kv_a_layernorm", model.kv_lora_rank)):
            add(attn + "." + name + ".weight", width, 2)
        for name in ("input_layernorm", "post_attention_layernorm"):
            add(prefix + "." + name + ".weight", h, 2)
        indexer_type = "full" if is_draft else model.indexer_types[layer]
        if indexer_type == "full":
            indexer = attn + ".indexer"
            linear(indexer + ".wq_b", model.index_n_heads * model.index_head_dim, model.q_lora_rank)
            linear(indexer + ".wk_weights_proj", model.index_head_dim + model.index_n_heads, h, quantized=False)
            add(indexer + ".k_norm.weight", model.index_head_dim, 4)
            add(indexer + ".k_norm.bias", model.index_head_dim, 4)
        elif indexer_type != "shared":
            raise ValueError("GLM FP8 payload requires full/shared source indexer types")
        if include_mla_absorption:
            if model.weight_dtype_for_module(attn + ".kv_b_proj", "Linear") != "fp8":
                raise NotImplementedError("GLM absorption payload requires retained FP8 kv_b_proj")
            add(attn + ".W_UK_T", heads * model.qk_nope_head_dim * model.kv_lora_rank, 2)
            add(attn + ".W_UV", heads * model.kv_lora_rank * model.v_head_dim, 2)
        mlp = prefix + ".mlp"
        if (model.mtp_layer_uses_moe(0) if is_draft else model.layer_uses_moe(layer)):
            routed = mlp + ".experts"
            if any(model.weight_dtype_for_module(routed + ".0." + name, "Linear") != "fp8"
                   for name in ("gate_proj", "up_proj", "down_proj")):
                raise NotImplementedError("GLM payload requires the block-FP8 routed-expert branch")
            add(routed + ".w13_weight", experts * 2 * intermediate * h, 1)
            add(routed + ".w2_weight", experts * h * intermediate, 1)
            add(routed + ".w13_weight_scale_inv", experts * 2 * (intermediate // 128) * (h // 128), 4)
            add(routed + ".w2_weight_scale_inv", experts * (h // 128) * (intermediate // 128), 4)
            linear(mlp + ".gate", model.num_experts, h, quantized=False)
            add(mlp + ".gate.e_score_correction_bias", model.num_experts, 4)
            shared = local["shared_down_kn"][0]
            linear(mlp + ".shared_experts.gate_up_proj", 2 * shared, h)
            linear(mlp + ".shared_experts.down_proj", h, shared)
        else:
            dense = local["dense_down_kn"][0]
            linear(mlp + ".gate_up_proj", 2 * dense, h)
            linear(mlp + ".down_proj", h, dense)
    if draft:
        prefix = f"model.layers.{model.num_hidden_layers}"
        # nn.Linear constructs eh_proj without quant_config. The head is
        # replaced by the last-stage target head; PP>1 retains its own embed.
        add(prefix + ".eh_proj.weight", h * 2 * h, 2)
        for name in ("enorm", "hnorm", "shared_head.norm"):
            add(prefix + "." + name + ".weight", h, 2)
        if parallelism.pipeline_parallel_size > 1:
            add("draft.model.embed_tokens.weight", vocab * h, 2)
    return entries


def glm_block_fp8_postload_parameter_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    hardware_id: str,
    linear_backend: str,
    routed_backend: str,
    include_mla_absorption: bool,
    logical_rank: int | None = None,
    pipeline_layer_partition: tuple[int, ...] | None = None,
    include_mtp: bool = False,
) -> dict[str, int]:
    """Registered payload under an explicit SM90 DeepGEMM source profile.

    vLLM's CUDA block processing leaves E4M3 weights/FP32 scales in their
    registered shapes. DeepGEMM's pinned SM90 (128,128) SFB branch checks and
    returns the same scale tensor, including its FP32 dtype; it neither
    broadcasts scales to full rows nor packs them as SM100 INT UE8M0. Optional
    value requantization is in-place. MoE uses that same conversion routine.
    Parameter replacement and ordinary MLA views follow the loader inventory.
    This does not count kernel state/workspaces, peak temporaries, driver or
    allocator reserves, and does not establish actual provider selection.
    """
    if (hardware_id != "nvidia_h200" or linear_backend != "deep_gemm"
            or routed_backend != "deep_gemm"):
        raise NotImplementedError("GLM post-load payload requires the explicit H200 SM90 DeepGEMM linear/MoE source profile")
    # The owner validates exact config, block geometry, TP/EP and dtype here.
    # No storage delta is introduced by this source branch; do not charge
    # both old and replacement Parameter wrappers for their shared payload.
    return glm_block_fp8_loader_payload_bytes(
        model, parallelism, include_mla_absorption=include_mla_absorption, logical_rank=logical_rank,
        pipeline_layer_partition=pipeline_layer_partition, include_mtp=include_mtp)


def glm_block_fp8_moe_workspace_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    hardware_id: str,
    tokens: int,
) -> dict[str, int]:
    """One cross-layer workspace request for the source-selected MoE branch.

    TritonOrDeepGemmExperts.workspace_shapes uses unpadded M*topk rows below
    the SM90 128-row DeepGEMM boundary and DeepGEMM's expert-padded rows at or
    above it. Both branches share workspace13 with the initial output view,
    allocate workspace2 simultaneously, and round each view to 256 bytes.
    The allocation is reused across layers rather than summed per layer.
    Standalone apply output, dispatch and library scratch remain separate.
    """
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("GLM MoE workspace tokens must be a positive integer")
    if tokens < 128:
        glm_block_fp8_postload_parameter_bytes(
            model, parallelism, hardware_id=hardware_id,
            linear_backend="deep_gemm", routed_backend="deep_gemm",
            include_mla_absorption=False)
        if parallelism.enable_expert_parallel:
            raise NotImplementedError("GLM Triton workspace requires EP-off")
        intermediate = _require_divisible(
            model.moe_intermediate_size, parallelism.tensor_parallel_size,
            "routed intermediate")
        padded = tokens * model.num_experts_per_tok
        k = model.hidden_size
    else:
        padded, intermediate, k = _glm_block_fp8_deepgemm_call_geometry(
            model, parallelism, hardware_id=hardware_id, tokens=tokens)
    n = 2 * intermediate
    workspace13_elements = padded * max(intermediate, k)
    workspace2_elements = padded * max(n, k)
    output_elements = tokens * k
    common_bytes = 2 * max(workspace13_elements, output_elements)
    second_bytes = 2 * workspace2_elements
    total = -(-common_bytes // 256) * 256 + -(-second_bytes // 256) * 256
    return {"worker.shared_moe_workspace": total}


def glm_block_fp8_deepgemm_workspace_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    hardware_id: str,
    tokens: int,
) -> dict[str, int]:
    """One declared aligned DeepGEMM workspace request, not total peak."""
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("GLM DeepGEMM workspace tokens must be a positive integer")
    if tokens < 128:
        raise NotImplementedError(
            "GLM DeepGEMM workspace requires the aligned M>=128 source branch")
    return glm_block_fp8_moe_workspace_bytes(
        model, parallelism, hardware_id=hardware_id, tokens=tokens)


def _glm_block_fp8_deepgemm_call_geometry(
    model: ModelConfig, parallelism: ParallelConfig, *, hardware_id: str, tokens: int,
) -> tuple[int, int, int]:
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("GLM DeepGEMM workspace tokens must be a positive integer")
    # Reuse the exact source-config, dtype, TP/EP and hardware boundary.
    glm_block_fp8_postload_parameter_bytes(
        model, parallelism, hardware_id=hardware_id,
        linear_backend="deep_gemm", routed_backend="deep_gemm",
        include_mla_absorption=False)
    if parallelism.enable_expert_parallel:
        raise NotImplementedError("GLM DeepGEMM workspace requires EP-off; received-token geometry is unbound")
    alignment = 128
    intermediate = _require_divisible(
        model.moe_intermediate_size, parallelism.tensor_parallel_size,
        "routed intermediate")
    n, k = 2 * intermediate, model.hidden_size
    if tokens < alignment or n % alignment or k % alignment:
        raise NotImplementedError("GLM DeepGEMM workspace requires the aligned M>=128 source branch; Triton workspace is unbound")
    routed = tokens * model.num_experts_per_tok
    padded = routed + min(routed, model.num_experts) * (alignment - 1)
    padded = -(-padded // alignment) * alignment
    return padded, intermediate, k


def glm_block_fp8_deepgemm_call_allocation_bytes(
    model: ModelConfig, parallelism: ParallelConfig, *, hardware_id: str, tokens: int,
) -> dict[str, int]:
    """Unique tensor allocations outside the shared workspace for one call.

    Same explicit SM90/EP-off/unchunked BF16/SILU FP8-block source scenario
    as glm_block_fp8_deepgemm_workspace_bytes: no expert metadata, no LoRA,
    router weight on output, ordinary FP32 (not UE8M0) scales. Modular apply
    allocates output independently and uses that as the final output alias.
    Prepare quantizes input; deepgemm_moe_permute allocates scales and INT32
    metadata; fused SiLU/mul/quant allocates only its FP32 scales because its
    FP8 output is a workspace13 view. All MM results also reuse workspace.

    Each value is one object's payload, not an allocation count times layers.
    The sum is a conservative call allocation envelope, not a simultaneous
    live peak: permutation-local count/start buffers can die before activation
    scales exist. Caller inputs/routing outputs, shared experts, library state
    or scratch, allocator overhead and whole-worker reserves are excluded.
    No device/provider execution or normal-capacity acceptance is implied.
    """
    padded, intermediate, k = _glm_block_fp8_deepgemm_call_geometry(
        model, parallelism, hardware_id=hardware_id, tokens=tokens)
    return {
        "moe.call.output": tokens * k * 2,
        "moe.prepare.quantized_input": tokens * k,
        "moe.prepare.input_scales": tokens * (k // 128) * 4,
        "moe.permute.expert_start_loc": model.num_experts * 4,
        "moe.permute.input_scales": padded * (k // 128) * 4,
        "moe.permute.expert_ids": padded * 4,
        "moe.permute.inv_perm": tokens * model.num_experts_per_tok * 4,
        "moe.permute.expert_num_tokens": model.num_experts * 4,
        "moe.activation.output_scales": padded * (intermediate // 128) * 4,
    }


def glm_block_fp8_constructor_buffer_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    max_batched_tokens: int,
    max_requests: int,
    logical_rank: int | None = None,
    pipeline_layer_partition: tuple[int, ...] | None = None,
    include_mtp: bool = False,
    num_speculative_tokens: int = 0,
) -> dict[str, int]:
    """Persistent source tensor payload outside registered GLM-5.3 parameters.

    Ordinary CUDA SparseMLACommonImpl/CP1, eager and no HiSparse; model construction
    runs in the loader's BF16 default-dtype context. Counts one shared logical
    top-k object, separate physical/count/request objects per index group, and
    one shared default-RoPE cache. Streams/events, runtime allocator overhead,
    With explicit rank placement, target groups follow the actual PP slice.
    Sequential MTP adds one final-stage index group. Its active top-k alias
    uses the target buffer, but the original constructor tensor stays retained
    by the non-Module index group/backend; count that old backing separately.
    Equal target/draft RoPE keys share one cache. Streams/events, backend
    workspaces, post-load conversions and KV remain outside this scope.
    This is a constructor scenario, not observed residency or available-KV.
    Keys name unique logical object owners, not state_dict parameter paths.
    """
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("GLM constructor buffers require ModelConfig and ParallelConfig")
    if (model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S
            or model.compute_dtype != "bf16" or model.target_semantics is None):
        raise NotImplementedError("GLM constructor buffers require the exact BF16-compute GLM block-FP8 config")
    if type(include_mtp) is not bool or type(num_speculative_tokens) is not int:
        raise TypeError("GLM constructor MTP state requires a boolean and integer depth")
    if num_speculative_tokens < 0 or include_mtp != (num_speculative_tokens > 0):
        raise ValueError("GLM constructor MTP requires an explicit positive depth exactly when enabled")
    for value in (max_batched_tokens, max_requests):
        if type(value) is not int or value <= 0:
            raise ValueError("GLM constructor buffer limits must be explicit positive integers")
    if max_requests > max_batched_tokens:
        raise ValueError("GLM constructor request limit cannot exceed the native batch-token limit")
    if logical_rank is None and (not parallelism.is_rank_local or parallelism.pipeline_parallel_size != 1):
        raise NotImplementedError("GLM constructor buffers require PP=DP=CP=SP=1 or an explicit logical_rank")
    # Reuse the exact source identity/quantization/rank domain, including
    # shared/dense SP placement. This does not choose a runtime backend.
    glm_block_fp8_loader_payload_bytes(model, parallelism, include_mla_absorption=False,
        logical_rank=logical_rank, pipeline_layer_partition=pipeline_layer_partition, include_mtp=include_mtp)
    from src.serving.parallelism.parallel_plan import build_parallel_plan
    from src.serving.parallelism.tp import mla_local_shape
    plan = build_parallel_plan(parallelism, num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=pipeline_layer_partition)
    local = mla_local_shape(model, plan, 0 if logical_rank is None else logical_rank)
    semantics = model.target_semantics
    if (semantics.rope_type != "default" or semantics.rope_layout != "interleaved"
            or semantics.indexer_rope_layout != semantics.rope_layout):
        raise NotImplementedError("GLM constructor buffers require identical default main/indexer RoPE cache keys")
    entries = {"model.topk_indices_buffer": max_batched_tokens * model.index_topk * 4}
    # get_sparse_mla_index_group_max_rows, sequential drafting branch.
    workspace_rows = min(max_batched_tokens, max_requests * (1 + num_speculative_tokens))
    owners = [("model", tuple(model.indexer_types[layer] for layer in local["layers"]))]
    if include_mtp and local["pipeline_stage"] == parallelism.pipeline_parallel_size - 1:
        owners.append(("draft", ("full",)))
        # The constructor retains one top-k backing per index group owner.
        # With index_share_for_mtp_iteration the appended MTP group's active
        # top-k aliases the retained backing; the kv_capacity merge subtracts
        # the portion already expressed by the startup profile's persistent
        # profile.mtp.topk tensor. This inventory keeps the complete entry.
        entries["draft.retained_constructor_topk_indices"] = max_batched_tokens * model.index_topk * 4
    for owner, kinds in owners:
        current_group = -1
        for indexer_type in kinds:
            if indexer_type not in ("full", "shared"):
                raise ValueError("GLM constructor buffers require full/shared indexer types")
            # A PP stage starting with a shared consumer still owns a group.
            if indexer_type == "full" or current_group == -1:
                current_group += 1
                prefix = f"{owner}.index_groups.{current_group}"
                entries[prefix + ".physical_topk_indices"] = (workspace_rows + 1) * model.index_topk * 4
                entries[prefix + ".valid_topk_counts"] = (workspace_rows + 1) * 4
                entries[prefix + ".request_ids"] = workspace_rows * 4
    # get_rope caches by dimension, max_position, style, parameters and dtype.
    # All layers and both main/indexer calls use the same key here. The cache
    # spans the HF maximum, not the shorter Scheduler max_model_len input.
    entries["model.shared_rotary.cos_sin_cache"] = model.max_position_embeddings * model.qk_rope_head_dim * 2
    return entries


def _linear_weight_bytes(
    model: ModelConfig,
    *,
    out_features: int,
    in_features: int,
    module_path: str,
    module_type: str = "Linear",
) -> int:
    """Return source-defined resident storage for one local linear shard."""

    for name, value in (
            ("out_features", out_features), ("in_features", in_features)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"target residency {name} must be positive")
    dtype = model.weight_dtype_for_module(module_path, module_type)
    parameters = out_features * in_features
    if dtype in ("bf16", "fp16"):
        return parameters * model.dtype_bytes
    if dtype == "fp8":
        return parameters
    if dtype == "int4":
        group_size = model.weight_group_size
        if in_features % group_size:
            raise NotImplementedError(
                "INT4 residency requires input features divisible by the "
                "weight group size")
        if parameters % 2:
            raise NotImplementedError(
                "INT4 residency requires an even packed parameter count")
        packed_weights = parameters // 2
        bf16_group_scales = out_features * (in_features // group_size) * 2
        return packed_weights + bf16_group_scales
    raise NotImplementedError(
        f"unsupported target resident weight dtype: {dtype!r}")


def _target_attention_weight_bytes(
    model: ModelConfig,
    tp_size: int,
    layer_index: int,
    indexer_type: str,
) -> int:
    """Pinned vLLM MLA/DSA attention weights resident on one TP rank."""

    local_heads = _require_divisible(
        model.num_attention_heads, tp_size, "num_attention_heads")
    prefix = f"model.layers.{layer_index}.self_attn"
    total = _linear_weight_bytes(
        model,
        out_features=(
            model.q_lora_rank + model.kv_lora_rank
            + model.qk_rope_head_dim),
        in_features=model.hidden_size,
        module_path=f"{prefix}.fused_qkv_a_proj",
    )
    total += _linear_weight_bytes(
        model,
        out_features=local_heads * model.head_dim,
        in_features=model.q_lora_rank,
        module_path=f"{prefix}.q_b_proj",
    )
    total += _linear_weight_bytes(
        model,
        out_features=local_heads * (
            model.qk_nope_head_dim + model.v_head_dim),
        in_features=model.kv_lora_rank,
        module_path=f"{prefix}.kv_b_proj",
    )
    total += _linear_weight_bytes(
        model,
        out_features=model.hidden_size,
        in_features=local_heads * model.v_head_dim,
        module_path=f"{prefix}.o_proj",
    )
    # Decoder RMSNorms plus the latent Q/KV RMSNorms are TP replicated.
    total += (
        2 * model.hidden_size + model.q_lora_rank + model.kv_lora_rank
    ) * model.dtype_bytes
    if indexer_type not in ("none", "full", "shared"):
        raise ValueError(
            f"unknown target attention indexer type: {indexer_type!r}")
    if model.attention_layout == "dsa_mla" and indexer_type == "full":
        indexer = f"{prefix}.indexer"
        total += _linear_weight_bytes(
            model,
            out_features=model.index_n_heads * model.index_head_dim,
            in_features=model.q_lora_rank,
            module_path=f"{indexer}.wq_b",
        )
        total += _linear_weight_bytes(
            model,
            out_features=model.index_head_dim + model.index_n_heads,
            in_features=model.hidden_size,
            module_path=f"{indexer}.wk_weights_proj",
        )
        # Indexer.k_norm uses vLLM LayerNorm, whose constructor explicitly
        # registers both affine parameters in FP32, independent of activations.
        total += 2 * model.index_head_dim * 4
    return total


def _target_dense_mlp_weight_bytes(
    model: ModelConfig,
    tp_size: int,
    layer_index: int,
) -> int:
    local_intermediate = _require_divisible(
        model.intermediate_size, tp_size, "intermediate_size")
    prefix = f"model.layers.{layer_index}.mlp"
    return (
        _linear_weight_bytes(
            model,
            out_features=2 * local_intermediate,
            in_features=model.hidden_size,
            module_path=f"{prefix}.gate_proj",
        )
        + _linear_weight_bytes(
            model,
            out_features=model.hidden_size,
            in_features=local_intermediate,
            module_path=f"{prefix}.down_proj",
        )
    )


def _target_moe_weight_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    layer_index: int,
    expert_rank: int = 0,
) -> int:
    tp_size = parallelism.tensor_parallel_size
    ep_size = parallelism.expert_parallel_size
    distributed_target = model.model_type in ("glm_moe_dsa", "kimi_k2") and (parallelism.context_parallel_size > 1 or parallelism.data_parallel_size > 1)
    if parallelism.enable_expert_parallel and ep_size != tp_size and not distributed_target:
        raise NotImplementedError(
            "target MoE residency requires the pinned vLLM EP=TP placement")
    local_experts = (len(linear_expert_partition(model.num_experts, ep_size, expert_rank)) if distributed_target
        else _require_divisible(model.num_experts, ep_size, "num_experts"))
    if not parallelism.enable_expert_parallel:
        _require_divisible(model.moe_intermediate_size,
            tp_size * parallelism.context_parallel_size * parallelism.data_parallel_size,
            "non-EP routed intermediate_size")
    prefix = f"model.layers.{layer_index}.mlp"
    routed = 0
    for expert_index in range(local_experts):
        expert = f"{prefix}.experts.{expert_index}"
        routed += _linear_weight_bytes(
            model,
            out_features=2 * model.moe_intermediate_size,
            in_features=model.hidden_size,
            module_path=f"{expert}.gate_proj",
        )
        routed += _linear_weight_bytes(
            model,
            out_features=model.hidden_size,
            in_features=model.moe_intermediate_size,
            module_path=f"{expert}.down_proj",
        )
    shared_intermediate = (
        model.moe_intermediate_size * model.num_shared_experts)
    shared_tp = (1 if distributed_target and parallelism.enable_expert_parallel
        and parallelism.data_parallel_size > 1 and tp_size > 1 else tp_size)
    local_shared = _require_divisible(
        shared_intermediate, shared_tp, "shared expert intermediate_size")
    shared = f"{prefix}.shared_experts"
    shared_bytes = (
        _linear_weight_bytes(
            model,
            out_features=2 * local_shared,
            in_features=model.hidden_size,
            module_path=f"{shared}.gate_proj",
        )
        + _linear_weight_bytes(
            model,
            out_features=model.hidden_size,
            in_features=local_shared,
            module_path=f"{shared}.down_proj",
        )
    )
    router_bytes = _linear_weight_bytes(
        model,
        out_features=model.num_experts,
        in_features=model.hidden_size,
        module_path=f"{prefix}.gate",
        module_type="GateLinear",
    )
    if model.target_semantics is None:
        raise AssertionError("target MoE residency lost target semantics")
    if model.target_semantics.moe_topk_method == "noaux_tc":
        correction_bias_bytes = model.num_experts * 4
    elif model.target_semantics.moe_topk_method == "group_limited_greedy":
        correction_bias_bytes = 0
    else:
        raise AssertionError(
            "unhandled target MoE top-k method: "
            f"{model.target_semantics.moe_topk_method!r}")
    # EP owns complete experts. EP-off shards every expert's intermediate
    # dimension across TP and, for PCP, its additional context ranks. Shared
    # experts and the gate remain replicated over the context axis.
    if parallelism.enable_expert_parallel:
        routed_resident = routed
    else:
        routed_resident = _require_divisible(
            routed, tp_size * parallelism.context_parallel_size * parallelism.data_parallel_size,
            "non-EP routed MoE bytes")
    return routed_resident + shared_bytes + router_bytes + correction_bias_bytes


def _target_decoder_layer_weight_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    layer_index: int,
    expert_rank: int = 0,
) -> int:
    if model.attention_layout == "dsa_mla":
        indexer_type = model.indexer_types[layer_index]
    elif model.attention_layout == "mla":
        indexer_type = "none"
    else:
        raise ValueError(
            "target decoder residency requires MLA or DSA-MLA attention")
    total = _target_attention_weight_bytes(
        model,
        parallelism.tensor_parallel_size,
        layer_index,
        indexer_type,
    )
    if model.layer_uses_moe(layer_index):
        return total + _target_moe_weight_bytes(
            model, parallelism, layer_index, expert_rank)
    dense_tp = (1 if model.model_type == "glm_moe_dsa" and (parallelism.context_parallel_size > 1 or parallelism.data_parallel_size > 1)
        and parallelism.enable_expert_parallel and parallelism.data_parallel_size > 1
        and parallelism.tensor_parallel_size > 1 and parallelism.pipeline_parallel_size == 1 else parallelism.tensor_parallel_size)
    return total + _target_dense_mlp_weight_bytes(model, dense_tp, layer_index)


def _target_mtp_weight_bytes(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    share_target_vocabulary: bool = False,
) -> int:
    """Return source-defined target draft weights on one last-stage rank."""

    if model.model_type == "glm_moe_dsa":
        if (model.mtp_architecture != "glm_moe_dsa_mtp"
                or model.num_mtp_layers != 1):
            raise NotImplementedError(
                "target MTP residency supports the single GLM-5.2 draft layer")
        indexer_type = "full"
    elif model.model_type == "deepseek_v3":
        if (model.mtp_architecture != "deepseek_mtp"
                or model.num_mtp_layers != 1):
            raise NotImplementedError(
                "target MTP residency supports the single DeepSeek-V3 "
                "NextN draft layer")
        # DeepSeek-V3 has no DSA indexer; the NextN layer is a plain-MLA
        # MoE decoder layer plus eh_proj and the shared embedding/head.
        indexer_type = "none"
    else:
        raise NotImplementedError(
            "target MTP residency supports a single source-backed GLM-5.2 "
            "or DeepSeek-V3 draft layer")
    if not model.mtp_layer_uses_moe(0):
        raise NotImplementedError(
            "target MTP residency requires its source-defined MoE layer")
    tp_size = parallelism.tensor_parallel_size
    layer_index = model.num_hidden_layers
    local_vocab = _require_divisible(
        padded_vocab_size(model.vocab_size), tp_size, "padded_vocab_size")
    decoder_bytes = _target_attention_weight_bytes(
        model, tp_size, layer_index, indexer_type)
    decoder_bytes += _target_moe_weight_bytes(
        model, parallelism, layer_index)
    merge_bytes = _linear_weight_bytes(
        model,
        out_features=model.hidden_size,
        in_features=2 * model.hidden_size,
        module_path=f"model.layers.{layer_index}.eh_proj",
    )
    draft_embedding_bytes = local_vocab * model.hidden_size * model.dtype_bytes
    draft_head_bytes = _linear_weight_bytes(
        model,
        out_features=local_vocab,
        in_features=model.hidden_size,
        module_path=f"model.layers.{layer_index}.shared_head.head",
    )
    if share_target_vocabulary:
        # vLLM 0.30 llm_base_proposer and GPU-v2 eagle/utils both replace
        # every MTP shared_head.head with the target head. Embedding sharing
        # is restricted to PP1; PP>1 retains the draft's own embedding.
        draft_head_bytes = 0
        if parallelism.pipeline_parallel_size == 1:
            draft_embedding_bytes = 0
    # enorm, hnorm, and shared_head.norm are replicated on every TP rank.
    draft_norm_bytes = 3 * model.hidden_size * model.dtype_bytes
    return (
        decoder_bytes
        + merge_bytes
        + draft_embedding_bytes
        + draft_head_bytes
        + draft_norm_bytes
    )


def _dense_mlp_parameter_count(model: ModelConfig, tp_size: int) -> int:
    if model.mlp_bias:
        raise NotImplementedError(
            "analytical model residency does not yet have a source-backed "
            "MLP-bias placement contract")
    local_intermediate = _require_divisible(
        model.intermediate_size, tp_size, "intermediate_size")
    return 3 * model.hidden_size * local_intermediate


def _moe_parameter_count(
    model: ModelConfig,
    parallelism: ParallelConfig,
    expert_rank: int = 0,
) -> int:
    if model.model_type != "qwen3_moe" or model.num_shared_experts != 0:
        raise NotImplementedError(
            "analytical MoE model residency currently supports Qwen3MoE "
            "without shared experts")
    if (not parallelism.enable_expert_parallel
                and model.moe_intermediate_size
                % (parallelism.tensor_parallel_size * parallelism.data_parallel_size)):
        raise NotImplementedError(
            "analytical Qwen3MoE residency requires TP <= num_experts and "
            "non-EP moe_intermediate_size divisible by TP")
    router = model.hidden_size * model.num_experts
    one_expert = 3 * model.hidden_size * model.moe_intermediate_size
    if parallelism.enable_expert_parallel:
        local_experts = len(linear_expert_partition(model.num_experts, parallelism.expert_parallel_size, expert_rank))
        routed_experts = local_experts * one_expert
    else:
        routed_experts = _require_divisible(
            model.num_experts * one_expert,
            parallelism.tensor_parallel_size * parallelism.data_parallel_size,
            "unsharded routed-expert parameters",
        )
    return router + routed_experts


def _decoder_layer_parameter_count(
    model: ModelConfig,
    parallelism: ParallelConfig,
    layer_index: int,
    expert_rank: int = 0,
) -> int:
    tp_size = parallelism.tensor_parallel_size
    # Input and post-attention RMSNorm weights are replicated on every TP rank.
    parameters = 2 * model.hidden_size
    parameters += _attention_parameter_count(model, tp_size)
    if model.layer_uses_moe(layer_index):
        parameters += _moe_parameter_count(model, parallelism, expert_rank)
    else:
        parameters += _dense_mlp_parameter_count(model, tp_size)
    return parameters


def _attention_layer_parameter_count(
    model: ModelConfig,
    parallelism: ParallelConfig,
) -> int:
    # The input RMSNorm and attention projections execute on the AF Attention
    # pool. The post-attention RMSNorm and MLP weights reside on its FFN peer.
    return model.hidden_size + _attention_parameter_count(
        model, parallelism.tensor_parallel_size)


def model_rank_residencies(
    model: ModelConfig,
    parallelism: ParallelConfig,
    *,
    worker_role: str,
    include_mtp: bool = False,
    model_execution_version: str | None = None,
    v4_weight_storage_profile: str | None = None,
    glm_weight_storage_profile: str | None = None,
    pipeline_layer_partition: tuple[int, ...] | None = None,
) -> tuple[ModelRankResidency, ...]:
    """Return source-defined text-model weight bytes for every pool rank.

    A ModelConfig extracted from a multimodal text_config carries only the
    text submodel. This diagnostic does not account for the parent tower or
    projector; normal capacity must reject that incomplete inventory unless
    a complete outer-model residency/skip/loader binding is supported.

    Explicit vllm_0_30_0 supports dense BF16 Qwen3 TP1/PP1 target-off and
    the BF16 GLM source scenario with
    ordinary vocab-parallel embeddings (VLLM_REPLICATE_EMBED disabled).
    It accounts for source-defined post-load MTP vocabulary aliases, not
    actual loader execution, reserves or native PP/runtime acceptance.
    None preserves the existing independent draft-vocabulary inventory.
    V4 additionally requires an explicit SM90 DeepGEMM/Marlin source storage
    profile; it uses actual PP layers and TP*DP or linear-EP expert ownership,
    preserves shared replication for PP1 model SP, and includes the root mHC
    parameters separately from decoder layers. This parameter accounting
    does not admit the narrower buffer/placement/capacity APIs. Model buffers and runtime reserves are owned by the memory
    policy, not counted as weights here. The profile does not select or
    verify a runtime provider. Explicit ``ffn`` additionally exposes the
    existing matched-TP dense BF16 AF post-attention norm/MLP inventory;
    embedding, final norm and vocabulary head remain on Attention. This
    weight-only role does not establish a KV plan or complete worker budget.
    """

    if not isinstance(model, ModelConfig):
        raise TypeError("model residency requires ModelConfig")
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("model residency requires ParallelConfig")
    if not isinstance(include_mtp, bool):
        raise TypeError("model residency include_mtp must be boolean")
    if v4_weight_storage_profile is not None:
        if (v4_weight_storage_profile != "sm90_deepgemm_marlin" or model.model_type != "deepseek_v4"
                or model_execution_version != "vllm_0_30_0" or worker_role != "colocated"):
            raise NotImplementedError("V4 weight storage profile requires explicit 0.30 colocated SM90 DeepGEMM/Marlin")
        if include_mtp:
            raise NotImplementedError(
                "V4 MTP placement inventory is not implemented: draft layer "
                "parameters, shared-vocabulary aliases, and draft KV/progress "
                "footprint are unaccounted; run with include_mtp=False until "
                "the placement contract lands")
        ranges = (pipeline_layer_ranges(model.num_hidden_layers, parallelism.pipeline_parallel_size, pipeline_layer_partition)
            if pipeline_layer_partition is not None else pipeline_layer_ranges_for_model(model, parallelism.pipeline_parallel_size))
        partition = tuple(end - first for first, end in ranges)
        residencies = []
        tp, pp = parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size
        for replica in range(parallelism.data_parallel_size):
            for stage, (first, end) in enumerate(ranges):
                for lane in range(tp):
                    rank = (replica * pp + stage) * tp + lane
                    payload = dsv4_parallel_postload_parameter_bytes(model, parallelism, rank,
                        pipeline_layer_partition=partition)
                    embedding = payload["model.embed_tokens.weight"] if stage == 0 else 0
                    head = payload["lm_head.weight"] if stage == pp - 1 else 0
                    norm = payload["model.norm.weight"] if stage == pp - 1 else 0
                    decoder = sum(size for name, size in payload.items() if name.startswith("model.layers."))
                    additional = sum(payload.values()) - embedding - head - norm - decoder
                    residencies.append(ModelRankResidency(rank, lane, stage,
                        replica * tp + lane if parallelism.enable_expert_parallel else 0, first, end,
                        embedding, decoder, 0, norm, head, additional, data_parallel_rank=replica))
        return tuple(residencies)
    if model.config_sha256 == KIMI_K3_CONFIG_SHA256:
        if model_execution_version != "vllm_0_30_0" or worker_role != "colocated" or include_mtp:
            raise NotImplementedError("K3 residency requires explicit0.30 colocated Marlin target-off storage")
        ranges = (pipeline_layer_ranges(model.num_hidden_layers, parallelism.pipeline_parallel_size, pipeline_layer_partition)
            if pipeline_layer_partition is not None else pipeline_layer_ranges_for_model(model, parallelism.pipeline_parallel_size))
        residencies = []
        for pp_rank, (layer_start, layer_end) in enumerate(ranges):
            for tp_rank in range(parallelism.tensor_parallel_size):
                payload = kimi_k3_marlin_postload_parameter_bytes(model, parallelism, pipeline_parallel_rank=pp_rank,
                    pipeline_layer_partition=pipeline_layer_partition)
                embedding = payload["language_model.model.embed_tokens.weight"] if pp_rank == 0 else 0
                head = payload["language_model.lm_head.weight"] if pp_rank == len(ranges) - 1 else 0
                final_norm = sum(payload[path] for path in (
                    "language_model.model.norm.weight", "language_model.model.output_attn_res_norm.weight",
                    "language_model.model.output_attn_res_proj.weight") if pp_rank == len(ranges) - 1)
                decoder = sum(size for path, size in payload.items() if path.startswith("language_model.model.layers."))
                rank = pp_rank * parallelism.tensor_parallel_size + tp_rank
                expert_rank = tp_rank if parallelism.enable_expert_parallel else 0
                residencies.append(ModelRankResidency(rank, tp_rank, pp_rank, expert_rank, layer_start, layer_end,
                    embedding, decoder, 0, final_norm, head))
        return tuple(replace(residency,
            rank=replica * parallelism.pipeline_parallel_size * parallelism.tensor_parallel_size + residency.rank,
            data_parallel_rank=replica,
            expert_parallel_rank=(replica * parallelism.tensor_parallel_size + residency.tensor_parallel_rank
                if parallelism.enable_expert_parallel else 0))
            for replica in range(parallelism.data_parallel_size) for residency in residencies)
    glm_fp8 = (model.model_type == "glm_moe_dsa" and model.weight_dtype == "fp8"
        and bool(model.quantized_weight_targets))
    if glm_weight_storage_profile is not None and not glm_fp8:
        raise ValueError("GLM weight storage profile requires official GLM block FP8")
    if glm_fp8:
        if (glm_weight_storage_profile != "sm90_deepgemm"
                or model_execution_version != "vllm_0_30_0"
                or (include_mtp and not model.index_share_for_mtp_iteration)
                or (include_mtp and worker_role != "decode")
                or worker_role not in ("colocated", "prefill", "decode")
                or parallelism.tensor_parallel_size != 8
                or parallelism.pipeline_parallel_size != 1
                or parallelism.data_parallel_size != 1
                or parallelism.context_parallel_size != 1
                or parallelism.sequence_parallel_size != 1
                or parallelism.enable_expert_parallel
                or pipeline_layer_partition not in (None, (model.num_hidden_layers,))):
            raise NotImplementedError(
                "official GLM block-FP8 normal residency requires the explicit "
                "H200 SM90 DeepGEMM TP8/PP1/DP1/EP-off target-only source "
                "profile; its decode-stage MTP inventory additionally requires "
                "the checkpoint's shared-index MTP iteration and a whole-model "
                "decode role")
        # The user-corrected scenario keeps MTP on the decode stage only:
        # the prefill pool loads the ordinary target and never carries the
        # appended MTP weights. vLLM0.30 glm5next/nvidia/mtp.py shows the
        # draft embedding and its shared_head head alias the target storage
        # (llm_base_proposer replaces the head at load; PP1 reuses the
        # target embedding), so the per-rank delta is the single appended
        # full-indexer MoE decoder layer plus eh_proj and three replicated
        # norms, priced by the existing loader payload with those aliases.
        residencies = []
        for rank in range(parallelism.worker_count):
            payload = glm_block_fp8_postload_parameter_bytes(
                model, parallelism, hardware_id="nvidia_h200",
                linear_backend="deep_gemm", routed_backend="deep_gemm",
                include_mla_absorption=True, logical_rank=rank,
                pipeline_layer_partition=pipeline_layer_partition,
                include_mtp=include_mtp)
            embedding = payload["model.embed_tokens.weight"]
            head = payload["lm_head.weight"]
            norm = payload["model.norm.weight"]
            decoder = sum(payload.values()) - embedding - head - norm
            mtp_bytes = (decoder
                - sum(size for name, size in payload.items()
                    if name.startswith("model.layers.")
                    and not name.startswith(f"model.layers.{model.num_hidden_layers}."))
                if include_mtp else 0)
            if include_mtp and mtp_bytes <= 0:
                raise RuntimeError("GLM FP8 decode MTP residency lost its appended layer payload")
            residencies.append(ModelRankResidency(
                rank, rank, 0, 0, 0, model.num_hidden_layers,
                embedding, decoder - mtp_bytes, mtp_bytes, norm, head))
        return tuple(residencies)
    legacy_unquantized_storage = (model_execution_version == "vllm_0_11_0"
        and model.model_type in ("qwen3", "qwen3_moe", "llama")
        and model.weight_dtype == model.compute_dtype and model.weight_dtype in ("bf16", "fp16"))
    if model_execution_version not in (None, "vllm_0_30_0") and not legacy_unquantized_storage:
        raise ValueError("unsupported model residency execution version")
    whole_model_roles = {"colocated", "prefill", "decode"}
    attention_roles = {"attention", "prefill-attention", "decode-attention"}
    ffn_role = worker_role == "ffn"
    if ffn_role and (model.model_type not in ("qwen3", "llama") or model.num_experts
            or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
            or model.hidden_act != "silu" or model.mlp_bias or model.rms_norm_eps is None
            or include_mtp or model.num_mtp_layers or parallelism.tensor_parallel_size not in (1, 2)
            or parallelism.pipeline_parallel_size != 1 or parallelism.enable_expert_parallel
            or not parallelism.is_rank_local):
        raise NotImplementedError("AF FFN residency requires dense BF16 Qwen3/Llama matched TP1/2 PP1 EP/SP/CP/DP/MTP-off")
    distributed_moe = (model_execution_version == "vllm_0_30_0" and model.model_type == "qwen3_moe"
        and worker_role in whole_model_roles and parallelism.context_parallel_size == 1
        and parallelism.sequence_parallel_size in ((1, parallelism.tensor_parallel_size)
            if parallelism.enable_expert_parallel and parallelism.data_parallel_size > 1 else (1,))
        and parallelism.expert_parallel_size == (parallelism.tensor_parallel_size * parallelism.data_parallel_size if parallelism.enable_expert_parallel else 1))
    from src.serving.workload.model_config import is_glm_mtp_data_parallel_model_scope
    glm_mtp_dp = include_mtp and is_glm_mtp_data_parallel_model_scope(model, parallelism)
    distributed_glm = (model_execution_version == "vllm_0_30_0" and model.model_type == "glm_moe_dsa"
        and model.compute_dtype == model.weight_dtype == "bf16" and (not include_mtp or glm_mtp_dp)
        and worker_role in whole_model_roles and (parallelism.context_parallel_size > 1 or parallelism.data_parallel_size > 1 or parallelism.pipeline_parallel_size > 1)
        and (parallelism.context_parallel_size == 1 or parallelism.pipeline_parallel_size == 1)
        and parallelism.sequence_parallel_size in ((1, parallelism.tensor_parallel_size)
            if parallelism.enable_expert_parallel and parallelism.data_parallel_size > 1
                and parallelism.tensor_parallel_size > 1 else (1,)))
    from src.serving.workload.model_config import is_kimi_k2_data_parallel_model_scope
    distributed_kimi = (model_execution_version == "vllm_0_30_0" and not include_mtp
        and worker_role in whole_model_roles and is_kimi_k2_data_parallel_model_scope(model, parallelism))
    if model_execution_version == "vllm_0_30_0":
        if model.model_type in ("qwen3", "llama"):
            # Qwen3Model inherits Qwen2Model's embedding/decoder/norm
            # placement. Qwen3ForCausalLM ties its padded ParallelLMHead to
            # embed_tokens when configured. The shared dense inventory below
            # includes Q/K RMSNorm weights and counts this alias once.
            if (model.compute_dtype not in ("bf16", "fp16") or model.weight_dtype != model.compute_dtype
                    or (model.compute_dtype == "fp16" and model.model_type != "qwen3")
                    or model.num_experts or (model.model_type == "qwen3" and not model.qk_norm) or include_mtp
                    or model.num_mtp_layers or worker_role not in whole_model_roles | attention_roles | {"ffn"}
                    or (worker_role in attention_roles and parallelism.tensor_parallel_size not in (
                        (1, 2) if worker_role == "attention" else (1,)))
                    or (parallelism.pipeline_parallel_size != 1 and not (
                        worker_role in whole_model_roles
                        and is_dense_parallel_model_scope(model, parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size)))
                    or not parallelism.is_rank_local or parallelism.enable_expert_parallel):
                raise NotImplementedError("vLLM0.30 dense Qwen3/Llama residency requires unquantized 16-bit legal TP/PP, EP-off/MTP-off whole-model pools")
        elif model.model_type == "qwen3_moe":
            # 0.30 Qwen3Moe retains the shared padded vocabulary/GQA
            # inventory. EP owns complete experts; _moe_parameter_count
            # partitions E by EP rather than dividing their I by TP again.
            if (not is_qwen3_moe_source_model_scope(model)
                    or include_mtp or worker_role not in whole_model_roles | attention_roles
                    or (worker_role in attention_roles and (parallelism.tensor_parallel_size != 1
                        or parallelism.enable_expert_parallel or parallelism.pipeline_parallel_size != 1))
                    or (parallelism.pipeline_parallel_size != 1 and not is_qwen3_moe_pipeline_model_scope(
                        model, parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size))
                    or model.num_attention_heads % parallelism.tensor_parallel_size
                    or (not parallelism.enable_expert_parallel
                        and model.moe_intermediate_size % parallelism.tensor_parallel_size)
                    or (not parallelism.is_rank_local and not distributed_moe)
                    or (parallelism.enable_expert_parallel
                        and (parallelism.expert_parallel_size != parallelism.tensor_parallel_size * parallelism.data_parallel_size))):
                raise NotImplementedError(
                    "vLLM0.30 Qwen3MoE residency requires validated BF16 "
                    "legal TP/PP/MTP-off whole-model pools, with optional stage-local TP=EP")
        elif model.model_type == "kimi_k2":
            from src.serving.workload.model_config import is_kimi_k2_parallel_model_scope
            if (include_mtp or worker_role not in whole_model_roles or not (parallelism.is_rank_local or distributed_kimi)
                    or not is_kimi_k2_parallel_model_scope(model, parallelism.tensor_parallel_size,
                        parallelism.pipeline_parallel_size, expert_parallel=parallelism.enable_expert_parallel)):
                raise NotImplementedError("vLLM0.30 K2 text diagnostic residency requires legal TP/DP/PP, uniform EP or whole-group expert TP and MTP-off")
        elif (model.model_type != "glm_moe_dsa" or model.compute_dtype != "bf16"
                or model.weight_dtype != "bf16" or model.tie_word_embeddings):
            raise NotImplementedError(
                "vLLM 0.30 residency requires BF16 GLM with separate target vocabulary weights or dense BF16 Qwen3")
    if worker_role not in whole_model_roles | attention_roles | {"ffn"}:
        raise NotImplementedError(
            "analytical KV capacity requires a whole-model or AF Attention "
            f"scheduler pool, got worker role {worker_role!r}")
    if not parallelism.is_rank_local and not (distributed_moe or distributed_glm or distributed_kimi):
        raise NotImplementedError(
            "analytical KV capacity currently supports DP=CP=SP=1")
    if model.compute_dtype not in ("bf16", "fp16"):
        raise NotImplementedError(
            "analytical model residency currently supports BF16 or FP16")
    if (model.model_type not in ("kimi_k2", "glm_moe_dsa", "deepseek_v3")
            and model.weight_dtype != model.compute_dtype):
        raise NotImplementedError(
            "analytical non-target model residency requires unquantized "
            "weights")
    if (model.num_mtp_layers
            and model.model_type not in ("kimi_k2", "glm_moe_dsa", "deepseek_v3")):
        raise NotImplementedError(
            "analytical model residency requires a complete MTP draft-weight "
            "placement contract")
    if model.model_type not in (
            "qwen3", "llama", "qwen3_moe", "kimi_k2", "glm_moe_dsa",
            "deepseek_v3"):
        raise NotImplementedError(
            "analytical model residency currently supports Qwen3, Qwen3MoE, "
            "Llama, Kimi-K2.6, GLM-5.2, and DeepSeek-V3")
    if (model_execution_version != "vllm_0_30_0" and model.model_type == "qwen3_moe"
            and parallelism.tensor_parallel_size > model.num_experts):
        raise NotImplementedError("legacy Qwen3MoE residency requires TP <= num_experts")
    tp_size = parallelism.tensor_parallel_size
    pp_size = parallelism.pipeline_parallel_size
    layer_ranges = (pipeline_layer_ranges(model.num_hidden_layers, pp_size, pipeline_layer_partition)
        if pipeline_layer_partition is not None else pipeline_layer_ranges_for_model(model, pp_size))
    if model.model_type == "qwen3_moe" and pp_size != 1 and model_execution_version != "vllm_0_30_0":
        raise NotImplementedError(
            "analytical Qwen3MoE residency does not yet model pinned-vLLM "
            "embedding, final-norm, and LM-head replication across PP ranks")
    if model.tie_word_embeddings and pp_size > 1 and not (
            model_execution_version == "vllm_0_30_0" and model.model_type in ("qwen3", "qwen3_moe")
            and worker_role in whole_model_roles):
        raise NotImplementedError(
            "tied embeddings across pipeline ranks require an explicit "
            "cross-rank storage contract")
    local_vocab = _require_divisible(
        padded_vocab_size(model.vocab_size), tp_size, "padded_vocab_size")
    moe_replicated_vocabulary = (model_execution_version == "vllm_0_30_0" and model.model_type == "qwen3_moe")
    embedding_parameters = local_vocab * model.hidden_size
    element_bytes = model.dtype_bytes
    residencies = []
    target_model = model.model_type in ("kimi_k2", "glm_moe_dsa", "deepseek_v3")
    if include_mtp:
        if worker_role not in whole_model_roles:
            raise NotImplementedError(
                "MTP residency requires a whole-model scheduler pool")
        mtp_bytes = _target_mtp_weight_bytes(
            model, parallelism,
            share_target_vocabulary=model_execution_version == "vllm_0_30_0")
    else:
        mtp_bytes = 0
    for pp_rank, (layer_start, layer_end) in enumerate(layer_ranges):
        if worker_role in whole_model_roles:
            if target_model:
                decoder_bytes = sum(
                    _target_decoder_layer_weight_bytes(
                        model, parallelism, layer_index)
                    for layer_index in range(layer_start, layer_end)
                )
                decoder_parameters = None
            else:
                decoder_parameters = sum(
                    _decoder_layer_parameter_count(
                        model, parallelism, layer_index)
                    for layer_index in range(layer_start, layer_end)
                )
                decoder_bytes = decoder_parameters * element_bytes
        elif worker_role in attention_roles:
            if target_model:
                raise NotImplementedError(
                    "target analytical residency does not support AF splitting")
            decoder_parameters = sum(
                _attention_layer_parameter_count(model, parallelism)
                for unused_layer_index in range(layer_start, layer_end)
            )
            decoder_bytes = decoder_parameters * element_bytes
        elif ffn_role:
            # The shared decoder inventory is exactly input-norm+attention
            # plus RemoteFFN's post-attention norm and MLP. Reuse that source
            # partition rather than copying a second parameter formula.
            decoder_parameters = sum(_decoder_layer_parameter_count(model, parallelism, layer_index)
                - _attention_layer_parameter_count(model, parallelism)
                for layer_index in range(layer_start, layer_end))
            decoder_bytes = decoder_parameters * element_bytes
        else:
            raise AssertionError("validated worker role disappeared")
        for tp_rank in range(tp_size):
            rank = pp_rank * tp_size + tp_rank
            embedding_bytes = (
                embedding_parameters * element_bytes if (pp_rank == 0 or moe_replicated_vocabulary) and not ffn_role else 0)
            final_norm_bytes = (
                model.hidden_size * element_bytes
                if (pp_rank == pp_size - 1 or moe_replicated_vocabulary) and not ffn_role else 0)
            lm_head_bytes = (
                embedding_parameters * element_bytes
                if (pp_rank == pp_size - 1 or moe_replicated_vocabulary)
                # Qwen2Model0.30 constructs embed_tokens on first and last
                # PP ranks when tied; Qwen3ForCausalLM aliases the last-rank
                # LM head to that local storage. Distinct ranks each retain
                # one vocabulary copy, while PP1 counts the alias once.
                and (not model.tie_word_embeddings or (pp_size > 1 and not moe_replicated_vocabulary)) and not ffn_role else 0)
            residencies.append(ModelRankResidency(
                rank=rank,
                tensor_parallel_rank=tp_rank,
                pipeline_parallel_rank=pp_rank,
                expert_parallel_rank=(
                    tp_rank if parallelism.enable_expert_parallel else 0),
                layer_start=layer_start,
                layer_end=layer_end,
                embedding_bytes=embedding_bytes,
                decoder_layer_bytes=decoder_bytes,
                mtp_bytes=(mtp_bytes if pp_rank == pp_size - 1 else 0),
                final_norm_bytes=final_norm_bytes,
                lm_head_bytes=lm_head_bytes,
            ))
    result = []
    for replica in range(parallelism.data_parallel_size):
        for residency in residencies:
            for context_rank in range(parallelism.context_parallel_size):
                ep_rank = ((replica * parallelism.context_parallel_size + context_rank) * tp_size
                    + residency.tensor_parallel_rank if parallelism.enable_expert_parallel else 0)
                decoder_bytes = residency.decoder_layer_bytes
                if model.model_type == "qwen3_moe" and parallelism.enable_expert_parallel and worker_role in whole_model_roles:
                    decoder_bytes = sum(_decoder_layer_parameter_count(model, parallelism, layer, ep_rank)
                        for layer in range(residency.layer_start, residency.layer_end)) * element_bytes
                elif (distributed_glm or distributed_kimi) and parallelism.enable_expert_parallel:
                    decoder_bytes = sum(_target_decoder_layer_weight_bytes(model, parallelism, layer, ep_rank)
                        for layer in range(residency.layer_start, residency.layer_end))
                rank = (((replica * pp_size + residency.pipeline_parallel_rank) * parallelism.context_parallel_size
                    + context_rank) * tp_size + residency.tensor_parallel_rank)
                result.append(replace(residency, rank=rank, context_parallel_rank=context_rank,
                    data_parallel_rank=replica, expert_parallel_rank=ep_rank, decoder_layer_bytes=decoder_bytes))
    result.sort(key=lambda residency: residency.rank)
    return tuple(result)


def model_weight_placements(model: ModelConfig, parallelism: ParallelConfig, *,
                            pipeline_layer_partition: tuple[int, ...] | None = None,
                            model_execution_version: str | None = None) -> tuple[dict, ...]:
    """Describe unquantized Qwen3/Llama parameter shards, checked against residency.

    Object identifiers are rank-local logical identities, not device addresses.
    Vocabulary shapes include the existing vocabulary padding contract.
    """
    if model.model_type not in ("qwen3", "llama", "qwen3_moe"):
        raise NotImplementedError("placement requires Qwen3, Qwen3MoE or Llama")
    residencies = model_rank_residencies(model, parallelism, worker_role="colocated",
        pipeline_layer_partition=pipeline_layer_partition, model_execution_version=model_execution_version)
    from src.serving.parallelism.parallel_plan import build_parallel_plan

    tp = parallelism.tensor_parallel_size
    plan = build_parallel_plan(parallelism, num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=pipeline_layer_partition)
    h = model.hidden_size
    vocab = padded_vocab_size(model.vocab_size)
    q = model.num_attention_heads * model.head_dim
    kv = model.num_key_value_heads * model.head_dim
    rows = []
    for residency in residencies:
        rank = residency.rank
        tr = residency.tensor_parallel_rank
        start = len(rows)

        def add(name, shape, axis=None, alias_of=None, *, shard_count=None, shard_rank=None, object_name=None, object_view=None):
            local = list(shape)
            interval = None
            if axis is not None:
                count = tp if shard_count is None else shard_count
                index = tr if shard_rank is None else shard_rank
                width = _require_divisible(shape[axis], count, name)
                interval = (index * width, (index + 1) * width)
                local[axis] = width
            elements = 1
            for dimension in local:
                elements *= dimension
            is_vocab = name in ("model.embed_tokens.weight", "lm_head.weight")
            logical_shape = (model.vocab_size, h) if is_vocab else shape
            rows.append({"rank": rank, "tensor_parallel_rank": tr,
                "pipeline_parallel_rank": residency.pipeline_parallel_rank,
                "parameter": name, "logical_shape": logical_shape, "storage_shape": shape, "local_shape": tuple(local),
                "dtype": model.weight_dtype, "shard_axis": axis, "shard_interval": interval,
                "shard_intervals": None if interval is None else (interval,),
                "placement_status": "alias" if alias_of is not None else "replicated" if axis is None else "tp_shard",
                "storage_kind": "alias" if alias_of is not None else "values", "metadata_for": None,
                "storage_bytes": 0 if alias_of is not None else elements * model.dtype_bytes,
                "object": f"rank:{rank}/{alias_of if alias_of is not None else (name if object_name is None else object_name)}",
                "object_view": object_view, "alias_of": alias_of, "object_byte_interval": None, "physical_address": None,
                "physical_layout": "unknown", "expert_parallel_rank": residency.expert_parallel_rank,
                "global_padding_elements": (vocab - model.vocab_size) * h if is_vocab else 0})

        if residency.embedding_bytes:
            add("model.embed_tokens.weight", (vocab, h), 0)
        for layer in range(residency.layer_start, residency.layer_end):
            prefix = f"model.layers.{layer}"
            add(prefix + ".input_layernorm.weight", (h,))
            add(prefix + ".post_attention_layernorm.weight", (h,))
            attn = prefix + ".self_attn"
            for name, width in (("q_proj", q), ("k_proj", kv), ("v_proj", kv)):
                # QKVParallelLinear replicates each KV-head shard over
                # TP/KV ranks once TP exceeds the global KV-head count.
                count = tp if name == "q_proj" else min(tp, model.num_key_value_heads)
                index = tr if name == "q_proj" else tr // max(1, tp // model.num_key_value_heads)
                add(attn + "." + name + ".weight", (width, h), 0,
                    shard_count=count, shard_rank=index,
                    object_name=attn + ".qkv_proj.weight", object_view=name)
            add(attn + ".o_proj.weight", (h, q), 1)
            if model.qk_norm:
                add(attn + ".q_norm.weight", (model.head_dim,))
                add(attn + ".k_norm.weight", (model.head_dim,))
            mlp = prefix + ".mlp"
            if model.layer_uses_moe(layer):
                add(mlp + ".gate.weight", (model.num_experts, h))
                for expert in range(model.num_experts):
                    if parallelism.enable_expert_parallel and plan.expert_to_shard[expert] != residency.expert_parallel_rank:
                        continue
                    expert_path = mlp + f".experts.{expert}"
                    count = 1 if parallelism.enable_expert_parallel else tp
                    index = 0 if parallelism.enable_expert_parallel else tr
                    for name in ("gate_proj", "up_proj"):
                        add(expert_path + "." + name + ".weight", (model.moe_intermediate_size, h), 0,
                            shard_count=count, shard_rank=index, object_name=mlp + ".experts.w13_weight",
                            object_view=f"expert={expert},projection={name}")
                    add(expert_path + ".down_proj.weight", (h, model.moe_intermediate_size), 1,
                        shard_count=count, shard_rank=index, object_name=mlp + ".experts.w2_weight",
                        object_view=f"expert={expert},projection=down_proj")
            else:
                for name in ("gate_proj", "up_proj"):
                    add(mlp + "." + name + ".weight", (model.intermediate_size, h), 0,
                        object_name=mlp + ".gate_up_proj.weight", object_view=name)
                add(mlp + ".down_proj.weight", (h, model.intermediate_size), 1)
        if residency.final_norm_bytes:
            add("model.norm.weight", (h,))
        if residency.lm_head_bytes:
            add("lm_head.weight", (vocab, h), 0)
        elif model.tie_word_embeddings and residency.final_norm_bytes:
            add("lm_head.weight", (vocab, h), 0, "model.embed_tokens.weight")
        actual = sum(row["storage_bytes"] for row in rows[start:])
        if actual != residency.weight_resident_bytes:
            raise ValueError(f"dense placement rank {rank} disagrees with residency: {actual} != {residency.weight_resident_bytes}")
    return tuple(rows)


def dense_weight_placements(model: ModelConfig, parallelism: ParallelConfig) -> tuple[dict, ...]:
    """Compatibility entry point for the dense-only inventory."""
    if model.model_type not in ("qwen3", "llama") or model.num_experts:
        raise NotImplementedError("dense placement requires a dense Qwen3 or Llama")
    return model_weight_placements(model, parallelism)


def deployment_weight_placements(model: ModelConfig, deployment, *, v4_weight_storage_profile: str | None = None,
                                 model_execution_version: str | None = None) -> tuple[dict, ...]:
    """Bind supported whole-model placements to declared deployment pools.

    Physical ranks are deployment labels; no device UUID or address is inferred.
    Each colocated pool is an independent model instance.
    """
    from src.serving.deployment.deployment_plan import DeploymentPlan

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("weight placement requires a DeploymentPlan")
    if deployment.mode != "colocated":
        raise NotImplementedError("weight placement requires colocated instances")
    result = []
    for pool in deployment.worker_pools.values():
        if pool.role != "colocated":
            raise NotImplementedError("weight placement requires whole-model pools")
        if v4_weight_storage_profile is None:
            partition = tuple(pool.parallel_plan.layer_to_pipeline_stage.count(stage)
                for stage in range(pool.parallel_plan.config.pipeline_parallel_size))
            placements = model_weight_placements(model, pool.parallel_plan.config,
                pipeline_layer_partition=partition, model_execution_version=model_execution_version)
        else:
            if v4_weight_storage_profile != "sm90_deepgemm_marlin" or pool.hardware_id != "nvidia_h200" or model.model_type != "deepseek_v4":
                raise NotImplementedError("V4 placement requires explicit H200 SM90 DeepGEMM/Marlin source profile")
            placements = dsv4_postload_inventory_diagnostic(model, pool.parallel_plan.config)
            if any(row["placement_status"] == "unresolved" for row in placements):
                raise NotImplementedError("V4 source profile has unresolved placement")
            residencies = model_rank_residencies(model, pool.parallel_plan.config, worker_role="colocated",
                model_execution_version="vllm_0_30_0", v4_weight_storage_profile=v4_weight_storage_profile)
            for residency in residencies:
                actual = sum(row["storage_bytes"] for row in placements if row["rank"] == residency.rank)
                if actual != residency.weight_resident_bytes:
                    raise ValueError("V4 placement disagrees with model residency")
        if v4_weight_storage_profile is None:
            from src.serving.parallelism.pp import pipeline_layer_ranges
            layer_ranges = pipeline_layer_ranges(model.num_hidden_layers,
                pool.parallel_plan.config.pipeline_parallel_size, partition)
        else:
            layer_ranges = pipeline_layer_ranges_for_model(model, pool.parallel_plan.config.pipeline_parallel_size)
        expected_stages = tuple(stage for stage, (start, end) in enumerate(layer_ranges) for _ in range(start, end))
        if pool.parallel_plan.layer_to_pipeline_stage != expected_stages:
            raise NotImplementedError("weight placement requires the model residency pipeline partition")
        for row in placements:
            logical_rank = row["rank"]
            physical_rank = pool.physical_rank_for_logical_rank(logical_rank)
            if v4_weight_storage_profile is not None:
                routed = row["placement_status"] == "routed_loader_geometry"
                storage_shape = row["storage_shape"] if routed else row["loader_shape"]
                local_shape = storage_shape
                if routed:
                    dimensions = list(row["logical_shape"])
                    dimensions[row["shard_axis"]] = sum(end - start for start, end in row["shard_intervals"])
                    local_shape = tuple(dimensions)
                name = row["parameter"]
                if name.endswith(".weight_scale_inv"):
                    storage_kind, metadata_for = "fp32_scale_per_128x128", name[:-len("weight_scale_inv")] + "weight"
                elif name.endswith("_weight_scale"):
                    storage_kind, metadata_for = "ue8m0_scale_per_32_input_values", name[:-len("_scale")]
                elif routed:
                    storage_kind, metadata_for = row["storage_kind"], None
                else:
                    storage_kind, metadata_for = "values", None
                intervals = row["shard_intervals"]
                represented_weight_shape = None
                represented_weight_intervals = None
                logical_shape = row["logical_shape"]
                if routed and metadata_for is not None:
                    represented_weight_shape = logical_shape
                    represented_weight_intervals = intervals
                    scale_shape = list(logical_shape)
                    scale_shape[-1] = _require_divisible(scale_shape[-1], 32, "MXFP4 scale input dimension")
                    logical_shape = tuple(scale_shape)
                    if row["shard_axis"] == len(scale_shape) - 1:
                        if any(start % 32 or end % 32 for start, end in intervals):
                            raise NotImplementedError("MXFP4 scale shard must align to 32 input values")
                        intervals = tuple((start // 32, end // 32) for start, end in intervals)
                    dimensions = list(logical_shape)
                    dimensions[row["shard_axis"]] = sum(end - start for start, end in intervals)
                    local_shape = tuple(dimensions)
                if not routed and intervals is None and row["shard_interval"] is not None:
                    intervals = (row["shard_interval"],)
                row = {**row, "tensor_parallel_rank": logical_rank, "pipeline_parallel_rank": 0,
                    "expert_parallel_rank": 0, "local_shape": local_shape,
                    "loader_shape": storage_shape, "storage_shape": None, "alias_of": None, "object_view": None,
                    "physical_layout": "unknown after post-load conversion",
                    "storage_profile": v4_weight_storage_profile, "object_byte_interval": None,
                    "storage_kind": storage_kind, "metadata_for": metadata_for, "shard_intervals": intervals,
                    "logical_shape": logical_shape, "represented_weight_shape": represented_weight_shape,
                    "represented_weight_shard_intervals": represented_weight_intervals,
                    "shard_interval": None if routed else row["shard_interval"]}
            result.append({**row, "instance": pool.name, "pool": pool.name,
                "hardware_id": pool.hardware_id, "physical_rank": physical_rank,
                "device_uuid": None,
                "object": f"pool:{pool.name}/{row['object']}"})
    return tuple(result)


def dsv4_routed_mxfp4_placements(model: ModelConfig, parallelism: ParallelConfig) -> tuple[dict, ...]:
    """V4 routed-expert loader tensors only; not a complete model inventory.

    The bound constructor profile is EP-off, with all experts on each TP rank.
    Returned shapes describe the source loader tensors, before Marlin repack.
    Gate and up intermediate slices are disjoint in the global fused parameter.
    """
    payload = dsv4_routed_loader_payload_bytes(model, parallelism)
    tp = parallelism.tensor_parallel_size
    experts = payload["experts_per_rank"]
    local_i = payload["intermediate_size_per_rank"]
    h = model.hidden_size
    total_i = model.moe_intermediate_size
    rows = []
    for rank in range(tp):
        lo, hi = rank * local_i, (rank + 1) * local_i
        entries = (
            ("w13_weight", "mxfp4", (experts, 2 * total_i, h),
             (experts, 2 * local_i, h // 2), ((lo, hi), (total_i + lo, total_i + hi)),
             payload["w13_packed_bytes_per_layer"], 1, "packed_values"),
            ("w2_weight", "mxfp4", (experts, h, total_i),
             (experts, h, local_i // 2), ((lo, hi),),
             payload["w2_packed_bytes_per_layer"], 2, "packed_values"),
            ("w13_weight_scale", "ue8m0", (experts, 2 * total_i, h),
             (experts, 2 * local_i, h // 32), ((lo, hi), (total_i + lo, total_i + hi)),
             payload["w13_ue8m0_bytes_per_layer"], 1, "scale_per_32_input_values"),
            ("w2_weight_scale", "ue8m0", (experts, h, total_i),
             (experts, h, local_i // 32), ((lo, hi),),
             payload["w2_ue8m0_bytes_per_layer"], 2, "scale_per_32_input_values"),
        )
        for layer in range(payload["layers"]):
            for suffix, dtype, logical, storage, intervals, size, axis, kind in entries:
                name = f"model.layers.{layer}.ffn.experts.{suffix}"
                rows.append({"rank": rank, "parameter": name, "logical_shape": logical,
                    "storage_shape": storage, "dtype": dtype, "shard_axis": axis,
                    "shard_intervals": intervals, "storage_bytes": size,
                    "object": f"rank:{rank}/{name}", "storage_kind": kind,
                    "padding": "none; unsupported when Marlin alignment would require padding",
                    "physical_layout": "loader tensor; post-load layout unknown",
                    "physical_address": None})
    return tuple(rows)


def dsv4_postload_inventory_diagnostic(model: ModelConfig, parallelism: ParallelConfig) -> tuple[dict, ...]:
    """Complete SM90 source-profile byte inventory, with explicit unresolved placement.

    This is a diagnostic toward placement, not a complete placement manifest:
    non-routed parameter shapes and global shard ownership have not been bound.
    It does not establish a selected runtime backend or available KV capacity.
    """
    source, element_bytes, loader_shapes = _dsv4_non_routed_loader_inventory(model, parallelism, attention_backend="flashmla")
    postload = dsv4_postload_parameter_bytes(model, parallelism, hardware_id="nvidia_h200",
        linear_backend="deep_gemm", routed_backend="marlin")
    routed = dsv4_routed_mxfp4_placements(model, parallelism)
    rows = []
    for rank in range(parallelism.tensor_parallel_size):
        for name, size in source.items():
            dtype = ("int32" if name.endswith(".gate.tid2eid") else
                "fp32" if element_bytes[name] == 4 or name.endswith(".weight_scale_inv") else
                "bf16" if element_bytes[name] == 2 else "fp8")
            is_vocab = name in ("model.embed_tokens.weight", "lm_head.weight")
            vocab_extent = padded_vocab_size(model.vocab_size) // parallelism.tensor_parallel_size
            logical_shape = (model.vocab_size, model.hidden_size) if is_vocab else None
            shard_axis = 0 if is_vocab else None
            interval = (rank * vocab_extent, (rank + 1) * vocab_extent) if is_vocab else None
            status = "vocab_shard" if is_vocab else "unresolved"
            intervals = None
            local_padding = 0
            # attention.py v0.30 uses disable_tp for fused Q/KV-A and a
            # replicated indexer Q projection; WQ-B/WO-A are column-parallel,
            # WO-B is row-parallel. Keep scale geometry in block coordinates.
            shape = loader_shapes[name]
            linear = name.rsplit(".", 1)[0]
            replicated = (name == "model.norm.weight" or name.startswith("model.hc_head_")
                or ".hc_attn_" in name or ".hc_ffn_" in name
                or name.endswith((".attn_norm.weight", ".ffn_norm.weight", ".attn.q_norm.weight", ".attn.kv_norm.weight",
                    ".ffn.gate.weight", ".ffn.gate.tid2eid", ".ffn.gate.e_score_correction_bias", ".attn.indexer.weights_proj.weight"))
                or ".compressor." in name)
            if shape is not None and (replicated or linear.endswith(".attn.fused_wqa_wkv") or linear.endswith(".attn.indexer.wq_b")):
                logical_shape, status = shape, "replicated"
            elif shape is not None and (linear.endswith(".attn.wq_b") or linear.endswith(".attn.wo_a") or linear.endswith(".attn.wo_b")):
                axis = 1 if linear.endswith(".attn.wo_b") else 0
                weight_shape = loader_shapes[linear + ".weight"]
                if not name.endswith(".weight_scale_inv") or weight_shape[axis] % 128 == 0:
                    global_shape = list(shape)
                    global_shape[axis] *= parallelism.tensor_parallel_size
                    logical_shape = tuple(global_shape)
                    shard_axis = axis
                    interval = (rank * shape[axis], (rank + 1) * shape[axis])
                    status = "tp_shard"
            elif name.endswith(".attn.attn_sink"):
                local_heads = model.num_attention_heads // parallelism.tensor_parallel_size
                logical_shape = (model.num_attention_heads,)
                shard_axis = 0
                interval = (rank * local_heads, (rank + 1) * local_heads)
                local_padding = shape[0] - local_heads
                status = "tp_shard"
            elif shape is not None and linear.endswith(".ffn.shared_experts.down_proj"):
                axis = 1
                weight_shape = loader_shapes[linear + ".weight"]
                if not name.endswith(".weight_scale_inv") or weight_shape[axis] % 128 == 0:
                    global_shape = list(shape)
                    global_shape[axis] *= parallelism.tensor_parallel_size
                    logical_shape, shard_axis = tuple(global_shape), axis
                    interval = (rank * shape[axis], (rank + 1) * shape[axis])
                    status = "tp_shard"
            elif shape is not None and linear.endswith(".ffn.shared_experts.gate_up_proj"):
                # MergedColumnParallelLinear stores rank-local gate then up.
                half = shape[0] // 2
                weight_shape = loader_shapes[linear + ".weight"]
                if shape[0] % 2 == 0 and (not name.endswith(".weight_scale_inv") or weight_shape[0] % 256 == 0):
                    global_half = half * parallelism.tensor_parallel_size
                    logical_shape = (2 * global_half, shape[1])
                    shard_axis = 0
                    intervals = ((rank * half, (rank + 1) * half),
                        (global_half + rank * half, global_half + (rank + 1) * half))
                    status = "tp_shard"
            rows.append({"rank": rank, "parameter": name, "dtype": dtype,
                "storage_bytes": postload[name], "source_loader_bytes": size,
                "local_elements": size // element_bytes[name], "loader_shape": loader_shapes[name],
                "logical_shape": logical_shape, "shard_axis": shard_axis, "shard_interval": interval,
                "shard_intervals": intervals, "local_padding_elements": local_padding,
                "global_padding_elements": (padded_vocab_size(model.vocab_size) - model.vocab_size) * model.hidden_size if is_vocab else None,
                "placement_status": status, "object": f"rank:{rank}/{name}",
                "physical_address": None})
        for row in routed:
            if row["rank"] == rank:
                rows.append({**row, "placement_status": "routed_loader_geometry",
                    "storage_bytes": postload[row["parameter"]],
                    "physical_layout": "post-load Marlin layout unknown"})
        expected = sum(postload.values())
        actual = sum(row["storage_bytes"] for row in rows if row["rank"] == rank)
        if actual != expected:
            raise ValueError(f"V4 diagnostic rank {rank} disagrees with post-load inventory: {actual} != {expected}")
    return tuple(rows)
