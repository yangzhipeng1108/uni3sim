"""Validated LLM model configuration used by Serving workload construction."""

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from src.serving.workload.rope_dispatch import VLLM_ROPE_CLASS_BY_SCALING_TYPE


_MODEL_SCHEMAS = {
    "kimi_linear": {
        "architecture": "KimiLinearForCausalLM",
        "attention_layout": "hybrid_kda_mla",
        "head_dim": "mla",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "num_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "none_zero_allowed",
        "qk_norm": False,
    },
    "qwen3": {
        "architecture": "Qwen3ForCausalLM",
        "attention_layout": "standard_gqa",
        "head_dim": "required",
        "mlp_bias": False,
        "moe": False,
        "expert_count_field": None,
        "dense_layer_policy": "all",
        "mtp_architecture": "none",
        "qk_norm": True,
        # Transformers Qwen3Config.__init__: documented optional value.
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
    },
    "qwen3_moe": {
        "architecture": "Qwen3MoeForCausalLM",
        "attention_layout": "standard_gqa",
        "head_dim": "required",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "num_experts",
        "dense_layer_policy": "none",
        "mtp_architecture": "none",
        "qk_norm": True,
        # Transformers Qwen3MoeConfig optional source defaults.
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
    },
    "llama": {
        "architecture": "LlamaForCausalLM",
        "attention_layout": "standard_gqa",
        "head_dim": "derived",
        "mlp_bias": False,
        "moe": False,
        "expert_count_field": None,
        "dense_layer_policy": "all",
        "mtp_architecture": "none",
        "qk_norm": False,
    },
    "glm4_moe": {
        "architecture": "Glm4MoeForCausalLM",
        "attention_layout": "standard_gqa",
        "head_dim": "required",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "glm4_moe_mtp",
        "qk_norm": "config",
        # Glm4MoeConfig and Glm4MoeAttention source defaults. Preserve raw
        # fixture identity while normalizing optional constructor inputs.
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
    },
    "glm_moe_dsa": {
        "architecture": "GlmMoeDsaForCausalLM",
        "attention_layout": "dsa_mla",
        "head_dim": "mla",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "glm_moe_dsa_mtp",
        "qk_norm": False,
    },
    "kimi_k2": {
        "architecture": "DeepseekV3ForCausalLM",
        "attention_layout": "mla",
        "head_dim": "mla",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "none_zero_allowed",
        "qk_norm": False,
    },
    "deepseek_v3": {
        "architecture": "DeepseekV3ForCausalLM",
        "attention_layout": "mla",
        "head_dim": "mla",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        "dense_layer_policy": "first_k_dense_replace",
        "mtp_architecture": "deepseek_mtp",
        "qk_norm": False,
    },
    "deepseek_v4": {
        "architecture": "DeepseekV4ForCausalLM",
        "attention_layout": "dsv4_sparse_mla",
        "head_dim": "dsv4",
        "mlp_bias": False,
        "moe": True,
        "expert_count_field": "n_routed_experts",
        # The H200 campaign configs carry no first_k_dense_replace: every
        # DeepSeek-V4 decoder layer is a routed MoE layer.
        "dense_layer_policy": "none",
        "mtp_architecture": "deepseek_mtp",
        "qk_norm": False,
    },
}


_COMPUTE_DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
}

_CONFIG_DTYPE_NAMES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float8_e4m3fn": "fp8",
    "int8": "int8",
    "uint8": "uint8",
}

_CONFIG_CANONICALIZATION = "json_sort_keys_compact_utf8_v1"

# DeepSeek-V4 compressed-cache state size under the upstream fp8_ds_mla
# layout: 448B NoPE + 128B RoPE + 8B FP8 scale per state, with one state per
# ``compress_ratio`` tokens (vLLM 0.30.0 deepseek_v4/attention.py
# MLAAttentionSpec tokens_per_state/state_content_bytes).
_DSV4_COMPRESSED_STATE_BYTES = 584
# DeepSeek-V4 SWA cache (vLLM 0.30.0 sparse_swa.py DeepseekV4SWACache): every
# layer unconditionally allocates one full-resolution sliding-window cache
# (deepseek_v4/attention.py:340).  On the SM90 FlashMLA path the class
# default ``use_fp8_ds_mla_layout = True`` (attention.py:152) packs each
# token's row into the same 584-byte fp8_ds_mla record as the compressed
# cache (sparse_swa.py:82 ``packed_bytes_per_token``); the cache pages at its
# own intrinsic ``block_size = 64`` tokens (sparse_swa.py:81) padded up to
# the 576-byte TMA stride (sparse_swa.py:83,127 ``packed_page_alignment``).
_DSV4_SWA_STATE_BYTES = 584
_DSV4_SWA_PAGE_TOKENS = 64
_DSV4_PAGE_ALIGNMENT_BYTES = 576
# DeepSeek-V4 indexer K cache (deepseek_v4/attention.py:992-997): FP8 layout
# stores 128 e4m3 bytes + one fp32 scale per 128 values = 132 bytes per
# compressed state.  The MXFP4 layout (68 bytes; attention.py:989-991)
# requires SM100 (mla/indexer.py:67 ``dsa_indexer_uses_fp4``), so H200 SM90
# deployments always take the FP8 layout.
_DSV4_INDEXER_K_STATE_BYTES_FP8 = 132
# Compressor FP32 partial-state cache (deepseek_v4/compressor.py
# CompressorStateCache): per token one fp32 state of
# ``2 * coff * head_dim`` elements (kv_state + score_state;
# compressor.py:282) where ``coff = 2`` for ratio 4 (overlapping windows)
# else 1 (compressor.py:238-239); pages hold 4 (ratio 4) or 8 (ratio 128)
# states (compressor.py:168-171) padded to the 576-byte alignment
# (compressor.py:190); the sliding window retains ``coff * compress_ratio``
# states per request (compressor.py:159-160).
_DSV4_COMPRESSOR_STATE_PAGE_STATES = {4: 4, 128: 8}
# Config-domain compress ratios of the pinned DeepSeek-V4 files.  Upstream
# clamps with ``max(1, config.compress_ratios[layer_id])``
# (deepseek_v4/attention.py:224-230), so a config 0 and a config 1 are the
# same thing: a pure sliding-window layer with no compressor and no indexer
# (0 also marks the last layer, which has no compressed successor).  The
# kernel/launch contract keeps the upstream post-clamp domain {1, 4, 128};
# ``ModelConfig.kernel_compress_ratios`` is the documented mapping.
_DSV4_ALLOWED_COMPRESS_RATIOS = (0, 4, 128)

# Exact top-level key set of the pinned DeepSeek-V4 campaign configs; the
# loader fails closed on any missing or extra key.
_DSV4_CONFIG_KEYS = frozenset((
    "architectures",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "compress_ratios",
    "compress_rope_theta",
    "eos_token_id",
    "expert_dtype",
    "hc_eps",
    "hc_mult",
    "hc_sinkhorn_iters",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "initializer_range",
    "max_position_embeddings",
    "model_type",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hash_layers",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "o_groups",
    "o_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "quantization_config",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "routed_scaling_factor",
    "scoring_func",
    "sliding_window",
    "swiglu_limit",
    "tie_word_embeddings",
    "topk_method",
    "torch_dtype",
    "transformers_version",
    "use_cache",
    "vocab_size",
))

_DSV4_DSPARK_FIELDS = frozenset(("dspark_block_size", "dspark_noise_token_id",
    "dspark_target_layer_ids", "dspark_markov_rank"))
_DSV4_DSPARK_MODEL_IDS = frozenset(("deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "deepseek-ai/DeepSeek-V4-Pro-DSpark"))
# vLLM0.30 nvidia/dspark.py uses three draft layers when the official
# config has no n_mtp_layers. num_hidden_layers remains the target offset.
_DSV4_DSPARK_LAYER_COUNT = 3
# On-disk config.json identities of the pinned fixed-target model files.
# ``_repository_target_provenance`` keys on exactly these digests; the
# Scheduler, workload-construction, and emission gates reference the same
# identities through these constants instead of restating the hashes.
KIMI_K2_CONFIG_SHA256 = (
    "03af24788bd7788c65db1fcb8326c5f5c5270aab717c442df67ba9db01041b3d")
KIMI_K2_7_CODE_CONFIG_SHA256 = (
    "ffbb57bff844e024f6c112640b63da80af228ceb8fb3e2e21908a21febbcdffb")
KIMI_K3_CONFIG_SHA256 = (
    "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213")
GLM_MOE_DSA_CONFIG_SHA256 = (
    "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a")
GLM_MOE_DSA_FP8_CONFIG_SHA256 = (
    "22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff")
GLM_MOE_DSA_5_3_CONFIG_SHA256 = (
    "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69")
GLM_BLOCK_FP8_CONFIG_SHA256S = frozenset((
    GLM_MOE_DSA_FP8_CONFIG_SHA256, GLM_MOE_DSA_5_3_CONFIG_SHA256))
DEEPSEEK_V3_CONFIG_SHA256 = (
    "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9")
# DeepSeek-V3 with only num_hidden_layers reduced to 32, the local
# deployment variant that fits TP4 on four H200s.  Its source is the
# derived-local config itself, so its source and on-disk digests coincide.
DEEPSEEK_V3_L32_CONFIG_SHA256 = (
    "d0d3a3a2583fc419d18dd314c928524733aaaf227ad045f592c5e8e9b09375d1")
# DeepSeek-V4-Flash and Pro: the official revision downloads were byte-identical
# to the existing H200 campaign files. No config or mathematical work changed.
DEEPSEEK_V4_FLASH_CONFIG_SHA256 = (
    "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818")
DEEPSEEK_V4_PRO_CONFIG_SHA256 = (
    "5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e")

# Every pinned target config identity across the fixed target models.
# Gates that accept "any named target file" (the vLLM Scheduler scope) use
# this flat set; the per-model pairing lives in
# TARGET_MODEL_CONFIG_SHA256_BY_TYPE.
TARGET_MODEL_CONFIG_SHA256 = frozenset((
    KIMI_K2_CONFIG_SHA256,
    GLM_MOE_DSA_CONFIG_SHA256,
    GLM_MOE_DSA_FP8_CONFIG_SHA256,
    DEEPSEEK_V3_CONFIG_SHA256,
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    DEEPSEEK_V4_FLASH_CONFIG_SHA256,
    DEEPSEEK_V4_PRO_CONFIG_SHA256,
))

# The pinned config identities accepted for each fixed target model:
# Kimi-K2.6 ships one file, GLM-5.2 ships the BF16 and FP8 files,
# DeepSeek-V3 ships the official config plus the 32-layer local variant,
# and DeepSeek-V4 ships the Flash and Pro configs.
TARGET_MODEL_CONFIG_SHA256_BY_TYPE = {
    "kimi_k2": (KIMI_K2_CONFIG_SHA256,),
    "glm_moe_dsa": (
        GLM_MOE_DSA_CONFIG_SHA256,
        GLM_MOE_DSA_FP8_CONFIG_SHA256,
    ),
    "deepseek_v3": (
        DEEPSEEK_V3_CONFIG_SHA256,
        DEEPSEEK_V3_L32_CONFIG_SHA256,
    ),
    "deepseek_v4": (
        DEEPSEEK_V4_FLASH_CONFIG_SHA256,
        DEEPSEEK_V4_PRO_CONFIG_SHA256,
    ),
}


def is_named_target_model_identity(model_type: str, config_sha256: str) -> bool:
    """Return whether the pair names one of the pinned target model files.

    The check is deliberately flat across models -- it matches the
    historical Scheduler gate, which accepts any pinned config identity with
    any target model_type; the per-model pairing is enforced separately by
    the workload identity contract.
    """

    return (
        model_type in TARGET_MODEL_CONFIG_SHA256_BY_TYPE
        and config_sha256 in TARGET_MODEL_CONFIG_SHA256)


_TARGET_MODEL_PROVENANCE = {
    "glm_moe_dsa": {
        "model_id": "zai-org/GLM-5.2",
        "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
        "source_config_sha256": (
            "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
        ),
        "canonical_config_sha256": (
            "ea00f6b92243d88fd4ed8b921438cecf0923bd27d10a1a4626aa570bf94000d0"
        ),
        "source_url": (
            "https://huggingface.co/zai-org/GLM-5.2/resolve/"
            "b4734de4facf877f85769a911abafc5283eab3d9/config.json"
        ),
        "extraction_path": "$",
        "extracted_config_sha256": (
            "ea00f6b92243d88fd4ed8b921438cecf0923bd27d10a1a4626aa570bf94000d0"
        ),
    },
    "kimi_k2": {
        "model_id": "moonshotai/Kimi-K2.6",
        "revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
        "source_config_sha256": (
            "85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899"
        ),
        "canonical_config_sha256": (
            "d0df2342b7d2fbc4bc3b4e5d9b190d2de7b0df063365d2910eb48599d67aa4a9"
        ),
        "source_url": (
            "https://huggingface.co/moonshotai/Kimi-K2.6/resolve/"
            "7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json"
        ),
        "extraction_path": "text_config",
        "extracted_config_sha256": (
            "a2b365e4927bc86375829321d37bb3741b4d8d5f48aa243a9ba550740b5d66fd"
        ),
    },
    "deepseek_v3": {
        "model_id": "deepseek-ai/DeepSeek-V3",
        "revision": "e815299b0bcbac849fa540c768ef21845365c9eb",
        "source_config_sha256": (
            "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9"
        ),
        "canonical_config_sha256": (
            "fed8c13b4637058cd68e600bd4bf7dc734bda4594dd583e3b49fa27c6e123cc6"
        ),
        "source_url": (
            "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/"
            "e815299b0bcbac849fa540c768ef21845365c9eb/config.json"
        ),
        "extraction_path": "$",
        "extracted_config_sha256": (
            "fed8c13b4637058cd68e600bd4bf7dc734bda4594dd583e3b49fa27c6e123cc6"
        ),
    },
    "deepseek_v4": {
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
        "source_config_sha256": (
            "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818"
        ),
        "canonical_config_sha256": (
            "6f27f527422d9a044acdd01017fb82c2c801dce8f26c40958741522b5377dd6a"
        ),
        "source_url": (
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/"
            "60d8d70770c6776ff598c94bb586a859a38244f1/config.json"
        ),
        "extraction_path": "$",
        "extracted_config_sha256": (
            "6f27f527422d9a044acdd01017fb82c2c801dce8f26c40958741522b5377dd6a"
        ),
    },
}

_GLM_FP8_PROVENANCE = {
    "model_id": "zai-org/GLM-5.2-FP8",
    "revision": "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
    "source_config_sha256": GLM_MOE_DSA_FP8_CONFIG_SHA256,
    "canonical_config_sha256": "bf42ace82b6d8e567879e48c7a03110b88ade66968faee8930d9db04b6d2214b",
    "source_url": "https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/f33c6dc501ee5a2c7e35155653b1b1abbc320951/config.json",
    "extraction_path": "$",
    "extracted_config_sha256": "bf42ace82b6d8e567879e48c7a03110b88ade66968faee8930d9db04b6d2214b",
}

# DeepSeek-V3 with only num_hidden_layers reduced to 32, the local
# deployment variant that fits TP4 on four H200s with dummy FP8 weights so
# a real vLLM run and the simulator see the identical architecture.
_DSV3_L32_PROVENANCE = {
    "model_id": "deepseek-ai/DeepSeek-V3",
    "revision": "e815299b0bcbac849fa540c768ef21845365c9eb",
    "source_config_sha256": "d0d3a3a2583fc419d18dd314c928524733aaaf227ad045f592c5e8e9b09375d1",
    "canonical_config_sha256": "6d2217b99dac0684b5d3c0ffca5855d39a802112f091b7267924a1b92b8067be",
    "source_url": "derived-local://deepseek-v3-l32?base_config_sha256=cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9&override=num_hidden_layers:32",
    "extraction_path": "$",
    "extracted_config_sha256": "6d2217b99dac0684b5d3c0ffca5855d39a802112f091b7267924a1b92b8067be",
}

# DeepSeek-V4-Pro has the same exact-byte official-source verification as Flash.
_DSV4_PRO_PROVENANCE = {
    "model_id": "deepseek-ai/DeepSeek-V4-Pro",
    "revision": "b5968e9190ef611bbf34a7229255be88a0e937c1",
    "source_config_sha256": "5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e",
    "canonical_config_sha256": "8b95ed969accf1a8645b40fdff0557984b81e96515ec2ec1f32239d31393faea",
    "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json",
    "extraction_path": "$",
    "extracted_config_sha256": "8b95ed969accf1a8645b40fdff0557984b81e96515ec2ec1f32239d31393faea",
}


# Official small-config identities. Registration binds origin, not support for
# the architecture, quantization, MTP, provider or runtime. Unsupported loaders
# continue to reject these exact source programs at their owner boundaries.
_OFFICIAL_CONFIG_PROVENANCE = {
    KIMI_K3_CONFIG_SHA256: {
        "model_id": "moonshotai/Kimi-K3",
        "revision": "f831ab66814297da540d832a5235f8e904f29d06",
        "source_config_sha256": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
        "canonical_config_sha256": "2a5cb51c92f3b64e68f7670042e4d8cfff0acd4bcf1e3c5f7190d69609381de6",
        "source_url": "https://huggingface.co/moonshotai/Kimi-K3/resolve/f831ab66814297da540d832a5235f8e904f29d06/config.json",
        "extraction_path": "text_config",
        "extracted_config_sha256": "a15a5818a3b84f2d78be7476983d4a384ef0bb9c93667c099cfa5ba6e8775c5d"
    },
    GLM_MOE_DSA_5_3_CONFIG_SHA256: {
        "model_id": "zai-org/GLM-5.3",
        "revision": "ba32f8c19c8738e2c57c71798c601d477a7bf844",
        "source_config_sha256": "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69",
        "canonical_config_sha256": "d497aba98135da3586209ba863e8e42eccf77a014811d0d3df812db9909c5d40",
        "source_url": "https://huggingface.co/zai-org/GLM-5.3/resolve/ba32f8c19c8738e2c57c71798c601d477a7bf844/config.json",
        "extraction_path": "$",
        "extracted_config_sha256": "d497aba98135da3586209ba863e8e42eccf77a014811d0d3df812db9909c5d40"
    },
    KIMI_K2_7_CODE_CONFIG_SHA256: {
        "model_id": "moonshotai/Kimi-K2.7-Code",
        "revision": "74797c9c62378b951a1f6fcf5c4631024e9b8bef",
        "source_config_sha256": "ffbb57bff844e024f6c112640b63da80af228ceb8fb3e2e21908a21febbcdffb",
        "canonical_config_sha256": "2907ed959c50743f3ced6b2dd263b88f29aa2054c82cdf59d52d179ce93a93b8",
        "source_url": "https://huggingface.co/moonshotai/Kimi-K2.7-Code/resolve/74797c9c62378b951a1f6fcf5c4631024e9b8bef/config.json",
        "extraction_path": "text_config",
        "extracted_config_sha256": "684f8da76df1ea74639612dc9010199608e9d1ea5f6c74cdde0012a1825735ae"
    },
    "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023": {
        "model_id": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
        "revision": "62af8fffb2f7030cac4de2f0169f5b8d1101b646",
        "source_config_sha256": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
        "canonical_config_sha256": "5329c1dbb4300f380167baed8114e28c04c18338ca11469030f5e26232f8361a",
        "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/resolve/62af8fffb2f7030cac4de2f0169f5b8d1101b646/config.json",
        "extraction_path": "$",
        "extracted_config_sha256": "5329c1dbb4300f380167baed8114e28c04c18338ca11469030f5e26232f8361a"
    },
    "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0": {
        "model_id": "deepseek-ai/DeepSeek-V4-Pro-DSpark",
        "revision": "7c09739fd136abfb70a49ec334157f65f45b52cd",
        "source_config_sha256": "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0",
        "canonical_config_sha256": "58d2046697a8e44cbb77f6e679cf6d0477b718e289af4ba1a2642763b39d6d32",
        "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/resolve/7c09739fd136abfb70a49ec334157f65f45b52cd/config.json",
        "extraction_path": "$",
        "extracted_config_sha256": "58d2046697a8e44cbb77f6e679cf6d0477b718e289af4ba1a2642763b39d6d32"
    }
}


_DSPARK_MODEL_IDS = (
    "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "deepseek-ai/DeepSeek-V4-Pro-DSpark",
)
_DSPARK_CONFIG_KEYS = frozenset(("dspark_block_size", "dspark_noise_token_id",
    "dspark_target_layer_ids", "dspark_markov_rank"))
# Frozen0.30 DSparkDeepseekV4Model uses n_mtp_layers or3. The two registered
# configs do not carry n_mtp_layers;num_nextn_predict_layers remains its raw1.
_DSPARK_SOURCE_DRAFT_LAYERS = 3


def _is_dspark_config_identity(config_sha256):
    return (type(config_sha256) is str and config_sha256 in _OFFICIAL_CONFIG_PROVENANCE
        and _OFFICIAL_CONFIG_PROVENANCE[config_sha256]["model_id"] in _DSPARK_MODEL_IDS)


@dataclass(frozen=True)
class TargetModelSemantics:
    """Source-pinned semantics that distinguish the two target models."""

    workload_contract: str
    use_cache: bool
    moe_layer_frequency: int
    moe_router_dtype: str
    moe_score_function: str
    moe_topk_method: str
    moe_group_count: int
    moe_topk_group_count: int
    moe_normalize_topk_probability: bool
    rope_type: str
    rope_theta: float
    rope_layout: str
    rope_factor: float | None
    rope_original_max_position_embeddings: int | None
    rope_beta_fast: float | None
    rope_beta_slow: float | None
    rope_mscale: float | None
    rope_mscale_all_dim: float | None
    index_skip_topk_offset: int | None
    index_topk_frequency: int | None
    index_topk_pattern: tuple[str, ...] | None
    indexer_rope_layout: str | None
    mtp_runtime_status: str
    rms_norm_eps: float

    def __post_init__(self) -> None:
        if (isinstance(self.rms_norm_eps, bool) or not isinstance(self.rms_norm_eps, (int, float))
                or not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0):
            raise ValueError("TargetModelSemantics.rms_norm_eps must be finite and positive")
        for field in (
                "workload_contract", "moe_router_dtype", "moe_score_function",
                "moe_topk_method", "rope_type", "rope_layout",
                "mtp_runtime_status"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"TargetModelSemantics.{field} must be non-empty")
        if not isinstance(self.use_cache, bool):
            raise TypeError("TargetModelSemantics.use_cache must be boolean")
        if not isinstance(self.moe_normalize_topk_probability, bool):
            raise TypeError(
                "TargetModelSemantics.moe_normalize_topk_probability must "
                "be boolean")
        for field in (
                "moe_layer_frequency", "moe_group_count",
                "moe_topk_group_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"TargetModelSemantics.{field} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"TargetModelSemantics.{field} must be positive")
        if self.moe_topk_group_count > self.moe_group_count:
            raise ValueError(
                "target MoE top-k group count cannot exceed group count")
        if self.moe_score_function not in ("sigmoid", "sqrtsoftplus"):
            raise ValueError(
                "target MoE score function must be sigmoid or sqrtsoftplus")
        if self.moe_router_dtype not in ("bf16", "fp32"):
            raise ValueError(
                "target MoE router dtype must be bf16 or fp32")
        if self.moe_topk_method != "noaux_tc":
            raise ValueError("target MoE top-k method must be noaux_tc")
        if self.rope_type not in ("default", "deepseek_yarn", "dsv4_yarn"):
            raise ValueError(
                f"unsupported target RoPE type: {self.rope_type!r}")
        if self.rope_layout != "interleaved":
            raise ValueError("target main RoPE layout must be interleaved")
        if (isinstance(self.rope_theta, bool)
                or not isinstance(self.rope_theta, (int, float))
                or not math.isfinite(self.rope_theta)
                or self.rope_theta <= 0):
            raise ValueError("target RoPE theta must be finite and positive")
        yarn_fields = (
            self.rope_factor,
            self.rope_original_max_position_embeddings,
            self.rope_beta_fast,
            self.rope_beta_slow,
            self.rope_mscale,
            self.rope_mscale_all_dim,
        )
        if self.rope_type == "default":
            if any(value is not None for value in yarn_fields):
                raise ValueError("default target RoPE cannot carry YaRN fields")
        elif self.rope_type == "deepseek_yarn":
            if any(value is None for value in yarn_fields):
                raise ValueError("DeepSeek YaRN requires every scaling field")
            numeric_yarn_fields = (
                self.rope_factor,
                self.rope_beta_fast,
                self.rope_beta_slow,
                self.rope_mscale,
                self.rope_mscale_all_dim,
            )
            if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in numeric_yarn_fields):
                raise ValueError(
                    "DeepSeek YaRN numeric fields must be finite and positive")
            original_max = self.rope_original_max_position_embeddings
            if (isinstance(original_max, bool)
                    or not isinstance(original_max, int)
                    or original_max <= 0):
                raise ValueError(
                    "DeepSeek YaRN original max position must be positive")
        elif self.rope_type == "dsv4_yarn":
            # vLLM 0.30.0 build_deepseek_v4_rope forces the deepseek_yarn
            # class for every DeepSeek-V4 layer and hard-disables mscale
            # (``rope_parameters["mscale"] = 0``), so mscale fields are not
            # config inputs and must stay null here.
            dsv4_yarn_fields = (
                self.rope_factor,
                self.rope_beta_fast,
                self.rope_beta_slow,
            )
            if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in dsv4_yarn_fields):
                raise ValueError(
                    "DeepSeek-V4 YaRN numeric fields must be finite and "
                    "positive")
            original_max = self.rope_original_max_position_embeddings
            if (isinstance(original_max, bool)
                    or not isinstance(original_max, int)
                    or original_max <= 0):
                raise ValueError(
                    "DeepSeek-V4 YaRN original max position must be "
                    "positive")
            if (self.rope_mscale is not None
                    or self.rope_mscale_all_dim is not None):
                raise ValueError(
                    "DeepSeek-V4 YaRN cannot carry mscale fields")
        else:
            raise AssertionError(
                f"unhandled target RoPE type: {self.rope_type!r}")
        dsa_fields = (
            self.index_skip_topk_offset,
            self.index_topk_frequency,
            self.indexer_rope_layout,
        )
        if all(value is None for value in dsa_fields):
            if self.index_topk_pattern is not None:
                raise ValueError(
                    "plain MLA cannot carry an index top-k pattern")
        elif any(value is None for value in dsa_fields):
            raise ValueError(
                "DSA target semantics require all indexer control fields")
        else:
            for field in ("index_skip_topk_offset", "index_topk_frequency"):
                value = getattr(self, field)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"TargetModelSemantics.{field} must be an integer")
                if value <= 0:
                    raise ValueError(
                        f"TargetModelSemantics.{field} must be positive")
            if self.indexer_rope_layout != "interleaved":
                raise ValueError("DSA indexer RoPE layout must be interleaved")
            if (self.index_topk_pattern is not None
                    and any(value not in ("F", "S")
                            for value in self.index_topk_pattern)):
                raise ValueError(
                    "DSA index top-k pattern entries must be F or S")


def expand_models_root(model_path: str) -> str:
    """Resolve the legacy ``${MODELS_ROOT}`` hardware-execution placeholder."""
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be a non-empty string")
    if "${MODELS_ROOT}" in model_path:
        if "MODELS_ROOT" not in os.environ:
            raise RuntimeError(
                "MODELS_ROOT must be set to resolve model paths containing "
                f"${{MODELS_ROOT}}: {model_path}")
        return model_path.replace("${MODELS_ROOT}", os.environ["MODELS_ROOT"])
    return model_path


def _load_model_json(model_path: str) -> tuple[
        dict[str, object], str, str, str, str | None, str | None, str | None,
        str]:
    path = Path(model_path)
    config_path = path if path.is_file() else path / "config.json"
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    cfg = json.loads(config_bytes)
    if not isinstance(cfg, dict):
        raise TypeError("model config root must be a mapping")
    canonical_config_bytes = json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical_config_sha256 = hashlib.sha256(
        canonical_config_bytes).hexdigest()
    provenance_path = path / "model_provenance.json" if path.is_dir() else None
    if provenance_path is None or not provenance_path.exists():
        repository_target = _repository_target_provenance(config_sha256)
        if repository_target is None:
            if (cfg["model_type"] == "glm_moe_dsa"
                    and "quantization_config" in cfg):
                raise ValueError(
                    "GLM FP8 model config is not registered in the repository "
                    f"provenance registry: {config_sha256}")
            return (
                cfg, config_sha256, config_sha256, canonical_config_sha256,
                None, None, None, "text")
        if repository_target is not None:
            return (
                cfg,
                config_sha256,
                repository_target["source_config_sha256"],
                canonical_config_sha256,
                repository_target["model_id"],
                repository_target["revision"],
                repository_target["source_url"],
                "text",
            )
        return (
            cfg,
            config_sha256,
            config_sha256,
            canonical_config_sha256,
            None,
            None,
            None,
            "text",
        )
    raw_provenance = json.loads(provenance_path.read_bytes())
    if not isinstance(raw_provenance, dict):
        raise TypeError("model provenance root must be a mapping")
    required = {
        "schema_version",
        "model_id",
        "revision",
        "source_config_sha256",
        "fixture_config_sha256",
        "source_url",
        "model_scope",
    }
    if set(raw_provenance) != required:
        raise ValueError(
            "model provenance fields must be exactly "
            f"{sorted(required)}, got {sorted(raw_provenance)}")
    if raw_provenance["schema_version"] != 1:
        raise ValueError("model provenance schema_version must be 1")
    for field in required - {"schema_version"}:
        value = raw_provenance[field]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"model provenance {field} must be a non-empty string")
    fixture_sha256 = raw_provenance["fixture_config_sha256"]
    if fixture_sha256 != config_sha256:
        raise ValueError(
            "model provenance fixture hash disagrees with config.json: "
            f"expected={fixture_sha256}, actual={config_sha256}")
    source_sha256 = raw_provenance["source_config_sha256"]
    for label, digest in (
            ("source_config_sha256", source_sha256),
            ("fixture_config_sha256", fixture_sha256)):
        if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"model provenance {label} must be a lowercase SHA256")
    model_scope = raw_provenance["model_scope"]
    if model_scope != "text":
        raise ValueError(
            f"unsupported model provenance scope: {model_scope!r}")
    return (
        cfg,
        config_sha256,
        source_sha256,
        canonical_config_sha256,
        raw_provenance["model_id"],
        raw_provenance["revision"],
        raw_provenance["source_url"],
        model_scope,
    )


def _repository_target_provenance(
    config_sha256: str,
) -> dict[str, str] | None:
    if config_sha256 in _OFFICIAL_CONFIG_PROVENANCE:
        return _OFFICIAL_CONFIG_PROVENANCE[config_sha256]
    target_by_repository_sha = {
        GLM_MOE_DSA_CONFIG_SHA256: (
            "glm_moe_dsa"
        ),
        KIMI_K2_CONFIG_SHA256: (
            "kimi_k2"
        ),
        GLM_MOE_DSA_FP8_CONFIG_SHA256: (
            "glm_moe_dsa_fp8"
        ),
        DEEPSEEK_V3_CONFIG_SHA256: (
            "deepseek_v3"
        ),
        DEEPSEEK_V3_L32_CONFIG_SHA256: (
            "deepseek_v3_l32"
        ),
        DEEPSEEK_V4_FLASH_CONFIG_SHA256: (
            "deepseek_v4"
        ),
        DEEPSEEK_V4_PRO_CONFIG_SHA256: (
            "deepseek_v4_pro"
        ),
    }
    if config_sha256 not in target_by_repository_sha:
        return None
    target = target_by_repository_sha[config_sha256]
    if target == "glm_moe_dsa_fp8":
        return _GLM_FP8_PROVENANCE
    if target == "deepseek_v3_l32":
        return _DSV3_L32_PROVENANCE
    if target == "deepseek_v4_pro":
        return _DSV4_PRO_PROVENANCE
    return _TARGET_MODEL_PROVENANCE[target]


def _named_target_workload_contract(model_type: str, config_sha256: str) -> str:
    """One derivation for parsed and independently validated target identity."""
    if model_type == "glm_moe_dsa":
        return ("glm_moe_dsa_vllm_0_30_0_logical_v1" if config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S
            else "glm_5_2_vllm_0_27_1_logical_v1")
    if model_type == "kimi_k2":
        return ("kimi_k2_7_code_vllm_0_30_0_logical_v1" if config_sha256 == KIMI_K2_7_CODE_CONFIG_SHA256
            else "kimi_k2_6_vllm_0_27_1_logical_v1")
    if model_type == "deepseek_v3":
        return "deepseek_v3_vllm_0_27_1_logical_v1"
    if model_type == "deepseek_v4":
        return ("deepseek_v4_dspark_vllm_0_30_0_metadata_only_v1"
            if _is_dspark_config_identity(config_sha256)
            else "deepseek_v4_vllm_0_30_0_logical_v1")
    raise ValueError(f"model type has no named target workload contract: {model_type!r}")


def validate_named_target_identity(model: "ModelConfig") -> None:
    """Validate registered target identity without imposing execution scope.

    Ordinary dense models retain their existing mathematical/source domains.
    For a named target family, provenance is necessary but does not establish
    support for its hardware, provider, quantization, Scheduler or MTP mode.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("target identity validation requires ModelConfig")
    if model.is_dspark_draft:
        raise NotImplementedError("DSpark draft requires its separate target-hidden/proposer source; ordinary V4 target execution is unavailable")
    if model.model_type not in TARGET_MODEL_CONFIG_SHA256_BY_TYPE:
        return
    allowed = model.config_sha256 in TARGET_MODEL_CONFIG_SHA256_BY_TYPE[model.model_type]
    if model.model_type == "glm_moe_dsa":
        allowed = allowed or model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S
    elif model.model_type == "kimi_k2":
        # Preserve the existing registered Code config identity; admitting
        # its metadata does not extend any production forward's model scope.
        allowed = allowed or model.config_sha256 == KIMI_K2_7_CODE_CONFIG_SHA256
    provenance = _repository_target_provenance(model.config_sha256)
    if not allowed or provenance is None:
        raise ValueError("target logical workload requires the fixed official model identity: unregistered config")
    if model.target_semantics is None:
        raise ValueError("target logical workload requires source-pinned model semantics for its fixed official model identity")
    expected = (provenance["model_id"], provenance["revision"], model.config_sha256,
        _MODEL_SCHEMAS[model.model_type]["attention_layout"],
        _named_target_workload_contract(model.model_type, model.config_sha256))
    actual = (model.model_id, model.model_revision, model.config_sha256,
        model.attention_layout, model.target_semantics.workload_contract)
    if actual != expected:
        raise ValueError("target logical workload requires the fixed official model identity: "
            f"actual={actual!r}, expected={expected!r}")


def _text_model_config(
    root: dict[str, object],
    *,
    has_fixed_provenance: bool,
) -> tuple[dict[str, object], str]:
    model_type = root["model_type"]
    if model_type in ("kimi_k25", "kimi_k3"):
        if not has_fixed_provenance:
            raise ValueError(
                f"{model_type} text extraction requires fixed model provenance")
        architectures = root["architectures"]
        expected_architecture = {
            "kimi_k25": "KimiK25ForConditionalGeneration",
            "kimi_k3": "KimiK3ForConditionalGeneration",
        }[model_type]
        if architectures != [expected_architecture]:
            raise ValueError(
                f"{model_type} requires {expected_architecture}")
        text_config = root["text_config"]
        if not isinstance(text_config, dict):
            raise TypeError(f"{model_type} text_config must be a mapping")
        if root["dtype"] != text_config["dtype"]:
            raise ValueError(
                f"{model_type} container dtype disagrees with text_config")
        return text_config, "text_config"
    elif "text_config" in root:
        raise ValueError(
            f"unsupported nested text model container: {model_type!r}")
    else:
        return root, "$"


@dataclass(frozen=True)
class DSV4SchedulerCacheGroup:
    """One DeepSeek-V4 cache lifecycle group for the Scheduler control plane.

    The pinned vLLM 0.11.0 Scheduler expresses each lifecycle class as its
    own KV-cache group (FullAttention manager for unbounded streams,
    SlidingWindow manager for windowed caches; with prefix caching disabled
    the coordinator imposes no group-type constraints).  ``block_tokens`` and
    ``page_bytes`` reproduce the upstream per-component 576-byte-aligned page
    allocation (fp8_ds_mla layout), and ``window_tokens`` is the upstream
    per-request retention cap (``None`` for unbounded stream caches).
    """

    kind: str
    lifecycle: str
    block_tokens: int
    page_bytes: int
    window_tokens: int | None
    layer_indices: tuple[int, ...]

    @property
    def per_token_bytes(self) -> Fraction:
        return Fraction(self.page_bytes, self.block_tokens)


@dataclass(frozen=True)
class KimiK3Semantics:
    """K3-specific source semantics within the existing model contract.

    Layer positions are normalized to zero-based indices. The full attention
    layers disable rotary transforms but retain the config's 64-wide PE
    subspace in QK projections and the latent cache. KDA state
    allocation/dtypes and actual backend selection are separate contracts.
    """
    layer_attention_types: tuple[str, ...]
    kda_num_heads: int
    kda_head_dim: int
    short_conv_kernel_size: int
    use_full_rank_gate: bool
    gate_lower_bound: float
    attention_residual_block_size: int
    routed_expert_hidden_size: int
    latent_moe_use_norm: bool
    activation_situ_beta: float
    activation_situ_linear_beta: float
    mla_use_nope: bool
    mla_use_output_gate: bool
    moe_layer_frequency: int
    moe_router_dtype: str
    rms_norm_eps: float

    def __post_init__(self):
        if self.moe_router_dtype != "fp32":
            raise ValueError("K3 GateLinear emits FP32 router logits")
        if (not isinstance(self.layer_attention_types, tuple)
                or not self.layer_attention_types
                or any(kind not in ("kda", "mla_nope") for kind in self.layer_attention_types)):
            raise ValueError("K3 layer types must explicitly cover KDA/NoPE MLA layers")
        for name in ("kda_num_heads", "kda_head_dim", "short_conv_kernel_size",
                     "attention_residual_block_size", "routed_expert_hidden_size", "moe_layer_frequency"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"K3 {name} must be a positive integer")
        for name in ("use_full_rank_gate", "latent_moe_use_norm", "mla_use_nope", "mla_use_output_gate"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"K3 {name} must be boolean")
        if not self.use_full_rank_gate or not self.mla_use_nope:
            raise NotImplementedError("K3 source scope requires full-rank KDA and NoPE MLA")
        for name in ("gate_lower_bound", "activation_situ_beta", "activation_situ_linear_beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"K3 {name} must be finite")
        if self.gate_lower_bound >= 0 or min(self.activation_situ_beta, self.activation_situ_linear_beta) <= 0:
            raise ValueError("K3 requires a negative KDA gate bound and positive SiTU scales")
        if (type(self.rms_norm_eps) not in (int, float)
                or not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0):
            raise ValueError("K3 requires a finite positive configured RMSNorm epsilon")


@dataclass(frozen=True)
class ModelConfig:
    """LLM architecture parameters needed for kernel dimension inference."""
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int
    vocab_size: int
    # Canonical compute dtype.  Element width is derived from this field;
    # callers must never infer dtype semantics from byte width.
    compute_dtype: str
    weight_dtype: str
    weight_group_size: int
    # Original checkpoint selectors. Native quantizer dispatch may further
    # restrict these (K3 Mxfp4Config quantizes RoutedExperts only); callers
    # must use weight_dtype_for_module rather than reimplementing dispatch.
    quantized_weight_targets: tuple[str, ...]
    quantized_weight_ignore: tuple[str, ...]
    attention_layout: str
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: tuple[str, ...]
    index_share_for_mtp_iteration: bool
    # DeepSeek-V4 sparse-MLA fields.  Every other layout normalizes them to
    # the empty/zero boundary values at load time.
    compress_ratios: tuple[int, ...]
    compress_rope_theta: int
    window_size: int
    o_lora_rank: int
    o_groups: int
    hc_mult: int
    hc_sinkhorn_iters: int
    num_hash_layers: int
    swiglu_limit: float
    expert_dtype: str
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    # Architecture name from HF config, retained as explicit model provenance.
    # Dense/MoE behavior is determined by validated expert fields, not by
    # substring matching on this display name.
    architecture: str
    # Whether token embedding and LM-head weights share checkpoint storage.
    # Pipeline reference coverage currently requires this to be false.
    tie_word_embeddings: bool
    # Whether linears carry a bias. Both values are required from the model
    # config because they participate in the measured launch-data key.
    attention_bias: bool
    mlp_bias: bool
    # Rotary position table size (HF config max_position_embeddings).
    max_position_embeddings: int
    model_type: str
    num_dense_layers: int
    num_shared_experts: int
    num_mtp_layers: int
    mtp_architecture: str
    qk_norm: bool
    # HuggingFace ``rope_scaling``/``rope_parameters`` scaling type verbatim
    # ("default" when the config carries no scaling block).  vLLM's
    # ``get_rope`` turns this string into one concrete rotary class, and the
    # out-of-tree custom-op registry is keyed on that class name, so this field
    # -- not the architecture -- decides which physical rope program runs.
    # See src/serving/workload/rope_dispatch.py.
    rope_scaling_type: str
    routed_scaling_factor: float
    target_semantics: TargetModelSemantics | None
    config_sha256: str
    source_config_sha256: str
    canonical_config_sha256: str
    config_canonicalization: str
    config_extraction_path: str
    extracted_config_sha256: str
    model_id: str | None
    model_revision: str | None
    model_source_url: str | None
    model_scope: str
    kimi_k3_semantics: KimiK3Semantics | None = None
    # Required by symbolic source operators when the checkpoint uses RMSNorm.
    # Other model families may omit it; callers must reject that missing input.
    rms_norm_eps: float | None = None
    hidden_act: str | None = None
    moe_normalize_topk_probability: bool | None = None
    rotary_dim: int | None = None
    moe_group_count: int | None = None
    moe_topk_group_count: int | None = None
    # Official DSpark metadata, normalized together at the model boundary.
    # The target-style raw architecture and target layer count remain intact.
    dspark_block_size: int | None = None
    dspark_noise_token_id: int | None = None
    dspark_target_layer_ids: tuple[int, ...] | None = None
    dspark_markov_rank: int | None = None

    def __post_init__(self) -> None:
        self._validate_dspark_metadata()
        draft_fields = (self.dspark_block_size, self.dspark_noise_token_id,
            self.dspark_target_layer_ids, self.dspark_markov_rank)
        if any(value is not None for value in draft_fields):
            if (any(value is None for value in draft_fields) or self.model_type != "deepseek_v4"
                    or self.model_id not in _DSV4_DSPARK_MODEL_IDS):
                raise ValueError("DSpark fields require the complete registered V4 draft metadata")
            for name in ("dspark_block_size", "dspark_noise_token_id", "dspark_markov_rank"):
                if type(getattr(self, name)) is not int:
                    raise TypeError(f"ModelConfig.{name} must be an integer")
            if (self.dspark_block_size < 2 or self.dspark_markov_rank < 1
                    or not 0 <= self.dspark_noise_token_id < self.vocab_size):
                raise ValueError("DSpark requires a valid noise token, block size and Markov rank")
            layers = self.dspark_target_layer_ids
            if (not isinstance(layers, tuple) or len(layers) != _DSV4_DSPARK_LAYER_COUNT
                    or any(type(layer) is not int or not 0 <= layer < self.num_hidden_layers for layer in layers)
                    or tuple(sorted(set(layers))) != layers):
                raise ValueError("DSpark target layer IDs must be three distinct ordered in-range target layers")
        elif self.model_id in _DSV4_DSPARK_MODEL_IDS:
            raise ValueError("registered DSpark draft identity cannot drop its draft metadata")
        if self.rotary_dim is not None and (type(self.rotary_dim) is not int
                or not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2):
            raise ValueError("ModelConfig.rotary_dim must be positive, even and no wider than its head")
        if (self.moe_group_count is None) != (self.moe_topk_group_count is None):
            raise ValueError("ModelConfig grouped routing requires both group counts")
        if self.moe_group_count is not None and (
                type(self.moe_group_count) is not int or type(self.moe_topk_group_count) is not int
                or not 0 < self.moe_topk_group_count <= self.moe_group_count
                or self.num_experts % self.moe_group_count):
            raise ValueError("ModelConfig grouped routing requires positive whole expert groups")
        if self.moe_normalize_topk_probability is not None and type(self.moe_normalize_topk_probability) is not bool:
            raise ValueError("ModelConfig.moe_normalize_topk_probability must be boolean or None")
        if self.hidden_act is not None and (not isinstance(self.hidden_act, str) or not self.hidden_act):
            raise ValueError("ModelConfig.hidden_act must be nonempty when declared")
        if self.rms_norm_eps is not None and (
                isinstance(self.rms_norm_eps, bool)
                or not isinstance(self.rms_norm_eps, (float, int))
                or not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0):
            raise ValueError("ModelConfig.rms_norm_eps must be finite and positive")
        positive = (
            "hidden_size", "num_attention_heads",
            "num_key_value_heads", "num_hidden_layers", "head_dim",
            "vocab_size", "max_position_embeddings",
        )
        for field in positive:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ModelConfig.{field} must be an integer")
            if value <= 0:
                raise ValueError(f"ModelConfig.{field} must be positive")
        if not isinstance(self.attention_layout, str):
            raise TypeError("ModelConfig.attention_layout must be a string")
        if self.attention_layout not in (
                "standard_gqa", "mla", "dsa_mla", "dsv4_sparse_mla", "hybrid_kda_mla"):
            raise ValueError(
                f"unsupported attention layout: {self.attention_layout!r}")
        if self.model_type == "kimi_linear" and self.attention_layout != "hybrid_kda_mla":
            raise ValueError("K3 cannot be relabeled as homogeneous MLA/GQA")
        if isinstance(self.intermediate_size, bool) or not isinstance(
                self.intermediate_size, int):
            raise TypeError("ModelConfig.intermediate_size must be an integer")
        if self.attention_layout == "dsv4_sparse_mla":
            # DeepSeek-V4 has no dense FFN; the pinned configs carry no
            # intermediate_size key at all.
            if self.intermediate_size != 0:
                raise ValueError(
                    "DeepSeek-V4 cannot carry a dense intermediate_size")
        elif self.intermediate_size <= 0:
            raise ValueError("ModelConfig.intermediate_size must be positive")
        if not isinstance(self.model_type, str) or not self.model_type:
            raise ValueError("ModelConfig.model_type must be non-empty")
        if self.model_type not in _MODEL_SCHEMAS:
            raise ValueError(
                f"unsupported ModelConfig.model_type: {self.model_type!r}")
        if (not isinstance(self.rope_scaling_type, str)
                or not self.rope_scaling_type):
            raise ValueError(
                "ModelConfig.rope_scaling_type must be a non-empty string")
        if self.rope_scaling_type not in VLLM_ROPE_CLASS_BY_SCALING_TYPE:
            raise ValueError(
                "unsupported ModelConfig.rope_scaling_type: "
                f"{self.rope_scaling_type!r}")
        attention_fields = (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
        )
        for field in attention_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ModelConfig.{field} must be an integer")
            if value < 0:
                raise ValueError(f"ModelConfig.{field} cannot be negative")
        if not isinstance(self.indexer_types, tuple) or any(
                not isinstance(value, str) or not value
                for value in self.indexer_types):
            raise TypeError(
                "ModelConfig.indexer_types must be a tuple of non-empty "
                "strings")
        if not isinstance(self.index_share_for_mtp_iteration, bool):
            raise TypeError(
                "ModelConfig.index_share_for_mtp_iteration must be boolean")
        mla_fields = (
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        dsv4_fields_absent = (
            self.compress_ratios == ()
            and self.compress_rope_theta == 0
            and self.window_size == 0
            and self.o_lora_rank == 0
            and self.o_groups == 0
            and self.hc_mult == 0
            and self.hc_sinkhorn_iters == 0
            and self.num_hash_layers == 0
            and self.swiglu_limit == 0.0
            and self.expert_dtype == ""
        )
        if self.attention_layout == "standard_gqa":
            head_dim_policy = _MODEL_SCHEMAS[self.model_type]["head_dim"]
            if head_dim_policy == "derived":
                if self.hidden_size % self.num_attention_heads:
                    raise ValueError(
                        "ModelConfig.hidden_size must be divisible by "
                        "num_attention_heads for a derived head_dim")
                if self.head_dim != (
                        self.hidden_size // self.num_attention_heads):
                    raise ValueError(
                        "ModelConfig.head_dim must equal hidden_size divided "
                        "by num_attention_heads for a derived head_dim")
            elif head_dim_policy == "required":
                pass
            else:
                raise ValueError(
                    "standard GQA requires an explicit or derived head_dim "
                    f"policy, got {head_dim_policy!r}")
            if self.num_attention_heads % self.num_key_value_heads:
                raise ValueError(
                    "ModelConfig.num_attention_heads must be divisible by "
                    "num_key_value_heads")
            if any(mla_fields) or any((
                    self.index_n_heads,
                    self.index_head_dim,
                    self.index_topk,
            )) or self.indexer_types or self.index_share_for_mtp_iteration \
                    or not dsv4_fields_absent:
                raise ValueError(
                    "standard GQA cannot carry MLA, DSA, or DeepSeek-V4 "
                    "parameters")
        elif self.attention_layout == "hybrid_kda_mla":
            if (self.model_type != "kimi_linear" or self.config_sha256 != KIMI_K3_CONFIG_SHA256
                    or not isinstance(self.kimi_k3_semantics, KimiK3Semantics)):
                raise ValueError("hybrid KDA/MLA requires exact Kimi-K3 source semantics")
            if (len(self.kimi_k3_semantics.layer_attention_types) != self.num_hidden_layers
                    or any(value <= 0 for value in mla_fields)
                    or self.head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim):
                raise ValueError("K3 requires complete layer mapping and the retained full QK head width")
            if (any((self.index_n_heads, self.index_head_dim, self.index_topk))
                    or self.indexer_types or self.index_share_for_mtp_iteration or not dsv4_fields_absent):
                raise ValueError("K3 hybrid attention cannot inherit DSA/V4 fields")
            if self.weight_dtype != "mxfp4":
                raise ValueError("exact K3 source requires floating MXFP4, not INT4 or homogeneous weights")
        elif self.attention_layout in ("mla", "dsa_mla", "dsv4_sparse_mla"):
            if self.attention_layout == "dsv4_sparse_mla":
                # DeepSeek-V4 has no separate KV LoRA: the compressed cache
                # row is the full head_dim-wide state.
                if self.kv_lora_rank != 0:
                    raise ValueError(
                        "DeepSeek-V4 cannot carry a kv_lora_rank")
                dsv4_projection_fields = (
                    self.q_lora_rank,
                    self.qk_nope_head_dim,
                    self.qk_rope_head_dim,
                    self.v_head_dim,
                )
                if any(value <= 0 for value in dsv4_projection_fields):
                    raise ValueError(
                        "DeepSeek-V4 requires positive projection dimensions")
            elif any(value <= 0 for value in mla_fields):
                raise ValueError("MLA requires positive projection dimensions")
            if self.head_dim != (
                    self.qk_nope_head_dim + self.qk_rope_head_dim):
                raise ValueError(
                    "MLA head_dim must equal qk_nope_head_dim plus "
                    "qk_rope_head_dim")
            if self.attention_layout == "mla":
                if any((
                        self.index_n_heads,
                        self.index_head_dim,
                        self.index_topk,
                )) or self.indexer_types or self.index_share_for_mtp_iteration \
                        or not dsv4_fields_absent:
                    raise ValueError(
                        "plain MLA cannot carry DSA or DeepSeek-V4 "
                        "parameters")
            elif self.attention_layout == "dsa_mla":
                if min(
                        self.index_n_heads,
                        self.index_head_dim,
                        self.index_topk) <= 0:
                    raise ValueError("DSA requires positive indexer dimensions")
                if len(self.indexer_types) != self.num_hidden_layers:
                    raise ValueError(
                        "DSA indexer_types must cover every hidden layer")
                if any(value not in ("full", "shared")
                       for value in self.indexer_types):
                    raise ValueError(
                        "DSA indexer type must be 'full' or 'shared'")
                if not dsv4_fields_absent:
                    raise ValueError(
                        "DSA cannot carry DeepSeek-V4 sparse-MLA parameters")
            elif self.attention_layout == "dsv4_sparse_mla":
                if self.v_head_dim != self.head_dim:
                    raise ValueError(
                        "DeepSeek-V4 v_head_dim must equal head_dim")
                if min(
                        self.index_n_heads,
                        self.index_head_dim,
                        self.index_topk) <= 0:
                    raise ValueError(
                        "DeepSeek-V4 requires positive indexer dimensions")
                if (self.indexer_types
                        or self.index_share_for_mtp_iteration):
                    raise ValueError(
                        "DeepSeek-V4 does not carry DSA indexer metadata")
                if not isinstance(self.compress_ratios, tuple):
                    raise TypeError(
                        "ModelConfig.compress_ratios must be a tuple")
                # The pinned configs carry one entry per target layer plus
                # a zero placeholder for each actual draft layer (one NextN
                # or three DSpark). Upstream maps draft layers to ratio 1
                # (deepseek_v4/attention.py:224-230).
                if len(self.compress_ratios) != (
                        self.num_hidden_layers + (_DSV4_DSPARK_LAYER_COUNT if self.is_dspark_draft else self.num_mtp_layers)):
                    raise ValueError(
                        "DeepSeek-V4 compress_ratios must cover every "
                        "hidden layer of the target plus its source MTP/DSpark draft tail")
                for value in self.compress_ratios:
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise TypeError(
                            "DeepSeek-V4 compress ratios must be integers")
                    if value not in _DSV4_ALLOWED_COMPRESS_RATIOS:
                        raise ValueError(
                            "DeepSeek-V4 compress ratio must be one of "
                            f"{_DSV4_ALLOWED_COMPRESS_RATIOS}, got {value!r}")
                if any(value != 0 for value in self.compress_ratios[
                        self.num_hidden_layers:]):
                    raise ValueError(
                        "DeepSeek-V4 MTP-layer compress ratio placeholders "
                        "must be 0")
                for field in (
                        "window_size", "o_lora_rank", "o_groups",
                        "hc_sinkhorn_iters"):
                    value = getattr(self, field)
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise TypeError(
                            f"ModelConfig.{field} must be an integer")
                    if value <= 0:
                        raise ValueError(
                            f"DeepSeek-V4 requires a positive {field}")
                if isinstance(self.hc_mult, bool) or not isinstance(
                        self.hc_mult, int):
                    raise TypeError("ModelConfig.hc_mult must be an integer")
                if self.hc_mult < 1:
                    raise ValueError("DeepSeek-V4 requires hc_mult >= 1")
                if isinstance(self.num_hash_layers, bool) or not isinstance(
                        self.num_hash_layers, int):
                    raise TypeError(
                        "ModelConfig.num_hash_layers must be an integer")
                if not 0 <= self.num_hash_layers <= self.num_hidden_layers:
                    raise ValueError(
                        "DeepSeek-V4 num_hash_layers must not exceed the "
                        "hidden layer count")
                if (isinstance(self.swiglu_limit, bool)
                        or not isinstance(self.swiglu_limit, (int, float))
                        or not math.isfinite(self.swiglu_limit)
                        or self.swiglu_limit <= 0):
                    raise ValueError(
                        "DeepSeek-V4 swiglu_limit must be finite and "
                        "positive")
                if self.expert_dtype not in ("fp4", "fp8"):
                    raise ValueError(
                        "DeepSeek-V4 expert_dtype must be 'fp4' or 'fp8', "
                        f"got {self.expert_dtype!r}")
                if isinstance(self.compress_rope_theta, bool) or not isinstance(
                        self.compress_rope_theta, int):
                    raise TypeError(
                        "ModelConfig.compress_rope_theta must be an integer")
                if self.compress_rope_theta <= 0:
                    raise ValueError(
                        "DeepSeek-V4 requires a positive compress_rope_theta")
            else:
                raise AssertionError(
                    f"unhandled attention layout: {self.attention_layout!r}")
        else:
            raise AssertionError(
                f"unhandled attention layout: {self.attention_layout!r}")
        if not isinstance(self.architecture, str):
            raise TypeError("ModelConfig.architecture must be a string")
        if not self.architecture.strip():
            raise ValueError("ModelConfig.architecture must be non-empty")
        if not isinstance(self.compute_dtype, str):
            raise TypeError("ModelConfig.compute_dtype must be a string")
        if self.compute_dtype not in _COMPUTE_DTYPE_BYTES:
            raise ValueError(
                f"unsupported compute dtype: {self.compute_dtype!r}")
        if self.model_type != "kimi_linear" and self.kimi_k3_semantics is not None:
            raise ValueError("K3 semantics cannot be attached to another model")
        if self.weight_dtype not in (
                set(_COMPUTE_DTYPE_BYTES) | {"mixed_int4_bf16", "mxfp4"}):
            raise ValueError(
                f"unsupported weight dtype: {self.weight_dtype!r}")
        if (isinstance(self.weight_group_size, bool)
                or not isinstance(self.weight_group_size, int)):
            raise TypeError("ModelConfig.weight_group_size must be an integer")
        for field in ("quantized_weight_targets", "quantized_weight_ignore"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or any(
                    not isinstance(item, str) or not item for item in value):
                raise TypeError(
                    f"ModelConfig.{field} must be a tuple of non-empty "
                    "strings")
        if self.weight_dtype == "mxfp4":
            if (self.model_type != "kimi_linear" or self.compute_dtype != "bf16"
                    or self.weight_group_size != 32 or self.quantized_weight_targets != ("Linear",)
                    or not self.quantized_weight_ignore):
                raise ValueError("K3 MXFP4 requires BF16 compute, group32 Linear weights and explicit exclusions")
            for rule in self.quantized_weight_ignore:
                if not rule.startswith("re:"):
                    raise ValueError("K3 MXFP4 exclusions require explicit regex rules")
                re.compile(rule[3:])
        elif self.weight_dtype == "mixed_int4_bf16":
            if self.compute_dtype != "bf16":
                raise ValueError(
                    "mixed INT4/BF16 weights require BF16 compute dtype")
            if self.weight_group_size <= 0:
                raise ValueError(
                    "mixed INT4/BF16 weights require a positive group size")
            if self.quantized_weight_targets != ("Linear",):
                raise ValueError(
                    "mixed INT4/BF16 weights currently require target Linear")
            if not self.quantized_weight_ignore:
                raise ValueError(
                    "mixed INT4/BF16 weights require explicit ignore rules")
        elif self.weight_dtype in _COMPUTE_DTYPE_BYTES:
            if (self.weight_dtype != self.compute_dtype
                    and not (
                        self.model_type in (
                            "glm_moe_dsa", "deepseek_v3", "deepseek_v4")
                        and self.compute_dtype == "bf16"
                        and self.weight_dtype == "fp8")):
                raise ValueError(
                    "unquantized weight dtype must equal compute dtype")
            if self.weight_group_size != 0:
                raise ValueError(
                    "unquantized weights require weight_group_size=0")
            if self.quantized_weight_targets or self.quantized_weight_ignore:
                if not (
                        self.model_type == "glm_moe_dsa"
                        and self.weight_dtype == "fp8"
                        and self.quantized_weight_targets == ("Linear",)
                        and self.quantized_weight_ignore):
                    raise ValueError(
                        "FP8 selectors require GLM Linear targets and explicit exclusions")
        else:
            raise AssertionError(
                f"unhandled weight dtype: {self.weight_dtype!r}")
        if self.model_scope != "text":
            raise ValueError(
                f"unsupported ModelConfig.model_scope: {self.model_scope!r}")
        for field in (
                "config_sha256",
                "source_config_sha256",
                "canonical_config_sha256",
                "extracted_config_sha256"):
            digest = getattr(self, field)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    f"ModelConfig.{field} must be a lowercase SHA256")
        if self.config_extraction_path not in ("$", "text_config"):
            raise ValueError(
                "ModelConfig.config_extraction_path must be '$' or "
                "'text_config'")
        if self.config_canonicalization != _CONFIG_CANONICALIZATION:
            raise ValueError(
                "unsupported model config canonicalization: "
                f"{self.config_canonicalization!r}")
        provenance_values = (
            self.model_id,
            self.model_revision,
            self.model_source_url,
        )
        if any(value is None for value in provenance_values):
            if not all(value is None for value in provenance_values):
                raise ValueError(
                    "model ID, revision, and source URL must be all present "
                    "or all null")
        elif any(not isinstance(value, str) or not value
                 for value in provenance_values):
            raise ValueError(
                "model ID, revision, and source URL must be non-empty")
        for field in (
                "num_dense_layers", "num_shared_experts", "num_mtp_layers"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ModelConfig.{field} must be an integer")
            if value < 0:
                raise ValueError(f"ModelConfig.{field} cannot be negative")
        if self.num_dense_layers > self.num_hidden_layers:
            raise ValueError(
                "ModelConfig.num_dense_layers cannot exceed hidden layers")
        if not isinstance(self.mtp_architecture, str):
            raise TypeError("ModelConfig.mtp_architecture must be a string")
        if self.mtp_architecture not in (
                "none", "glm4_moe_mtp", "glm_moe_dsa_mtp", "deepseek_mtp"):
            raise ValueError(
                f"unsupported MTP architecture: {self.mtp_architecture!r}")
        if self.num_mtp_layers == 0:
            if self.mtp_architecture != "none":
                raise ValueError(
                    "zero MTP layers require mtp_architecture='none'")
        elif self.mtp_architecture == "none":
            raise ValueError(
                "positive MTP layers require an MTP architecture")
        for field in (
                "tie_word_embeddings", "attention_bias", "mlp_bias",
                "qk_norm"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"ModelConfig.{field} must be boolean")
        if (isinstance(self.routed_scaling_factor, bool)
                or not isinstance(self.routed_scaling_factor, (int, float))
                or not math.isfinite(self.routed_scaling_factor)
                or self.routed_scaling_factor <= 0):
            raise ValueError(
                "ModelConfig.routed_scaling_factor must be finite and "
                "positive")
        if self.target_semantics is not None and not isinstance(
                self.target_semantics, TargetModelSemantics):
            raise TypeError(
                "ModelConfig.target_semantics must be TargetModelSemantics "
                "or null")
        if self.model_type in (
                "glm_moe_dsa", "kimi_k2", "deepseek_v3", "deepseek_v4"):
            if self.target_semantics is None:
                raise ValueError(
                    "fixed target models require source-pinned semantics")
        elif self.target_semantics is not None:
            raise ValueError(
                "non-target models cannot carry target-model semantics")
        for field in (
                "num_experts", "num_experts_per_tok",
                "moe_intermediate_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ModelConfig.{field} must be an integer")
        moe_values = (
            self.num_experts,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
        )
        if all(value == 0 for value in moe_values):
            if self.num_dense_layers != self.num_hidden_layers:
                raise ValueError("dense models require every layer to be dense")
            if self.num_shared_experts != 0:
                raise ValueError("dense models cannot contain shared experts")
            return
        if any(value <= 0 for value in moe_values):
            raise ValueError(
                "MoE models require positive expert count, top-k, and "
                "moe_intermediate_size")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        if self.num_dense_layers == self.num_hidden_layers:
            raise ValueError("MoE models require at least one routed MoE layer")

    def _validate_dspark_metadata(self):
        present = (self.dspark_block_size is not None or self.dspark_noise_token_id is not None
            or self.dspark_target_layer_ids is not None or self.dspark_markov_rank is not None)
        if not _is_dspark_config_identity(self.config_sha256):
            if present:
                raise ValueError("DSpark metadata requires its exact registered official source identity")
            return
        if self.model_type != "deepseek_v4" or self.attention_layout != "dsv4_sparse_mla":
            raise ValueError("DSpark requires its original DeepSeek-V4 sparse model geometry")
        provenance = _OFFICIAL_CONFIG_PROVENANCE[self.config_sha256]
        actual = (self.model_id, self.model_revision, self.model_source_url,
            self.source_config_sha256, self.canonical_config_sha256, self.config_extraction_path,
            self.extracted_config_sha256)
        expected = tuple(provenance[key] for key in ("model_id", "revision", "source_url",
            "source_config_sha256", "canonical_config_sha256", "extraction_path", "extracted_config_sha256"))
        if actual != expected:
            raise ValueError("DSpark metadata provenance disagrees with its exact official source")
        if not present:
            raise ValueError("registered DSpark draft identity cannot drop its draft metadata")
        for name in ("dspark_block_size", "dspark_noise_token_id", "dspark_markov_rank"):
            if type(getattr(self, name)) is not int:
                raise TypeError("DSpark " + name + " must be an integer")
        if self.dspark_block_size <= 1 or self.dspark_markov_rank <= 0:
            raise ValueError("DSpark requires a positive Markov rank and anchor-plus-noise block")
        if not 0 <= self.dspark_noise_token_id < self.vocab_size:
            raise ValueError("DSpark noise token must be inside the official vocabulary")
        layers = self.dspark_target_layer_ids
        if (type(layers) is not tuple or len(layers) != _DSPARK_SOURCE_DRAFT_LAYERS
                or any(type(layer) is not int or not 0 <= layer < self.num_hidden_layers for layer in layers)
                or tuple(sorted(set(layers))) != layers):
            raise ValueError("DSpark target layers must retain three ordered distinct valid decoder IDs")
        if (self.target_semantics is None or self.target_semantics.workload_contract !=
                "deepseek_v4_dspark_vllm_0_30_0_metadata_only_v1"):
            raise ValueError("DSpark cannot use the ordinary V4 workload contract")

    @property
    def is_dspark_draft(self) -> bool:
        return self.dspark_block_size is not None

    @property
    def dspark_layer_indices(self) -> tuple[int, ...]:
        if not self.is_dspark_draft:
            raise ValueError("DSpark layer indices require a registered draft config")
        return tuple(range(self.num_hidden_layers, self.num_hidden_layers + _DSV4_DSPARK_LAYER_COUNT))

    @property
    def dtype_bytes(self) -> int:
        """Element width derived from the canonical compute dtype."""

        return _COMPUTE_DTYPE_BYTES[self.compute_dtype]

    def weight_dtype_for_module(
        self,
        module_path: str,
        target_class: str,
    ) -> str:
        """Resolve the fixed checkpoint's storage dtype for one module."""

        for label, value in (
                ("module_path", module_path),
                ("target_class", target_class)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if (self.model_type == "deepseek_v4" and module_path == "lm_head"
                and target_class == "Linear"):
            # nvidia/model.py constructs ParallelLMHead without quant_config.
            # Its default None selects UnquantizedEmbeddingMethod, even when
            # the decoder linears use the checkpoint's global FP8 quantizer.
            return self.compute_dtype
        if self.weight_dtype == "mxfp4":
            # K3's actual0.30 ModelConfig normalizes compressed-tensors MXFP4
            # to Mxfp4Config. That quantizer selects RoutedExperts only and
            # returns UnquantizedLinearMethod for every LinearBase, including
            # prefixes absent from the raw checkpoint's ignore patterns.
            return "mxfp4" if target_class == "RoutedExperts" else self.compute_dtype
        if self.weight_dtype == "fp8" and self.quantized_weight_targets:
            # Fp8Config matches exact names. GateLinear and embedding do not
            # consume this quantizer; V32's merged indexer A projection
            # explicitly receives quant_config=None (attention.py:77-85).
            if (target_class != "Linear"
                    or module_path.endswith(".indexer.wk_weights_proj")):
                return self.compute_dtype
            ignored = self.quantized_weight_ignore
            if module_path in ignored:
                return self.compute_dtype
            packed = {
                "gate_up_proj": ("gate_proj", "up_proj"),
                "fused_qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
            }
            parent, separator, projection = module_path.rpartition(".")
            if projection in packed:
                shards = tuple(
                    parent + separator + name for name in packed[projection])
                skipped = tuple(shard in ignored for shard in shards)
                if any(skipped) != all(skipped):
                    raise ValueError("packed FP8 projection has inconsistent shard exclusions")
                return self.compute_dtype if all(skipped) else "fp8"
            if ".experts." in module_path:
                # RoutedExperts is one quantizer prefix even when its loader
                # receives unfused per-expert weight names.
                experts_prefix = module_path.split(".experts.", 1)[0] + ".experts"
                if any(experts_prefix in name for name in ignored if "experts" in name):
                    return self.compute_dtype
            return "fp8"
        if self.weight_dtype != "mixed_int4_bf16":
            return self.weight_dtype
        if target_class not in self.quantized_weight_targets:
            return self.compute_dtype
        for raw_pattern in self.quantized_weight_ignore:
            if not raw_pattern.startswith("re:"):
                raise ValueError(
                    "quantized weight ignore rules must use the re: prefix")
            if re.fullmatch(raw_pattern[3:], module_path):
                return self.compute_dtype
        return "int4"

    @property
    def expert_routing_assumption_semantics_sha256(self) -> str | None:
        """Share only declared routing assumptions for the official FP8 pair.

        Checkpoint identity remains separate. Real replay/profile inputs must
        keep their own original model binding. No observation or timing enters
        this normalized structure/quantization signature.
        """
        if self.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S:
            return None
        normalized = asdict(self)
        # This GLM-only assumption identity predates the unrelated DSpark
        # fields. Preserve its original structural projection and route seed.
        for name in ("model_id", "model_revision", "config_sha256", "source_config_sha256",
                     "canonical_config_sha256", "extracted_config_sha256", "model_source_url"):
            del normalized[name]
        # Draft-only metadata cannot change the existing GLM routing seed
        # contract when every such field is absent on this target model.
        for name in _DSV4_DSPARK_FIELDS:
            del normalized[name]
        return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def kv_heads_per_tensor_parallel_rank(self, tp_size: int) -> int:
        """Return the standard-GQA KV heads resident on each TP rank."""

        if isinstance(tp_size, bool) or not isinstance(tp_size, int):
            raise TypeError("tensor parallel size must be an integer")
        if tp_size < 1:
            raise ValueError("tensor parallel size must be positive")
        if self.attention_layout != "standard_gqa":
            raise NotImplementedError(
                "MLA/DSA KV capacity requires a compressed-KV and index-cache "
                "storage contract")
        kv_heads = self.num_key_value_heads
        if kv_heads >= tp_size and kv_heads % tp_size == 0:
            return kv_heads // tp_size
        if kv_heads < tp_size and tp_size % kv_heads == 0:
            return 1
        raise ValueError(
            "num_key_value_heads and tensor parallel size must divide one "
            f"another exactly: {kv_heads} vs {tp_size}")

    def kv_bytes_per_token_per_tensor_parallel_rank(
        self,
        tp_size: int,
    ) -> int:
        """Return resident K+V bytes per token on one standard-GQA TP rank."""

        return self.kv_bytes_per_token_per_parallel_rank(tp_size, 1)

    def kimi_k3_source_cache_payload(
        self, *, tensor_parallel_size: int, kda_state_slots: int, mla_tokens: int,
        mamba_cache_dtype: str, mamba_ssm_cache_dtype: str,
        mla_cache_dtype: str, use_kda_recoverssm: bool,
    ) -> dict[str, int]:
        """Conditional K3 cache tensor payload, not allocator-ready capacity.

        Source: KimiK3DeltaAttention.get_state_shape/get_state_dtype and
        MambaStateShapeCalculator/StateDtypeCalculator.kda_state_*.
        No speculation/RecoverSSM; caller declares state slots, not a guessed
        one-slot-per-request policy. A slot contains a convolution tail and
        an HxDxD recurrent matrix per KDA layer. Only Q/K/V use convolution;
        the extra full-rank gate projections do not expand its cache shape.

        Ordinary BF16 MLA stores [kv_lora_rank, qk_rope_head_dim] per token
        even with rotary disabled. Its compressed row is replicated over TP;
        KDA heads/states are sharded. Page/group padding, prefix snapshots,
        physical allocation, backend metadata and runtime are not bound here.
        The general per-token/normal-capacity APIs deliberately stay closed.
        """
        if self.model_type != "kimi_linear" or self.kimi_k3_semantics is None:
            raise NotImplementedError("K3 source cache payload requires exact hybrid Kimi-K3 semantics")
        if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
            raise ValueError("K3 source cache TP must be a positive integer")
        for name, value in (("kda_state_slots", kda_state_slots), ("mla_tokens", mla_tokens)):
            if type(value) is not int or value < 0:
                raise ValueError(f"K3 source cache {name} must be a non-negative integer")
        if type(use_kda_recoverssm) is not bool:
            raise TypeError("K3 RecoverSSM flag must be explicit boolean")
        if use_kda_recoverssm:
            raise NotImplementedError("K3 source cache payload excludes RecoverSSM/speculative records")
        conv_widths = {"auto": 2, "bfloat16": 2, "float16": 2, "float32": 4}
        recurrent_widths = {"auto": 4, "bfloat16": 2, "float32": 4}
        if not isinstance(mamba_cache_dtype, str) or mamba_cache_dtype not in conv_widths:
            raise ValueError("K3 convolution cache dtype must be auto/float16/bfloat16/float32")
        if not isinstance(mamba_ssm_cache_dtype, str) or mamba_ssm_cache_dtype not in recurrent_widths:
            raise ValueError("K3 recurrent cache dtype must be auto/bfloat16/float32; auto is FP32")
        if mla_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("K3 source MLA cache payload currently requires BF16; quantized cache layout is unbound")
        source = self.kimi_k3_semantics
        if source.kda_num_heads % tensor_parallel_size or self.num_attention_heads % tensor_parallel_size:
            raise ValueError("K3 attention heads must divide evenly over TP")
        local_heads = source.kda_num_heads // tensor_parallel_size
        kda_layers = source.layer_attention_types.count("kda")
        mla_layers = source.layer_attention_types.count("mla_nope")
        conv_elements = 3 * local_heads * source.kda_head_dim * (source.short_conv_kernel_size - 1)
        recurrent_elements = local_heads * source.kda_head_dim * source.kda_head_dim
        return {
            "kda_convolution_state": kda_layers * kda_state_slots * conv_elements * conv_widths[mamba_cache_dtype],
            "kda_recurrent_state": kda_layers * kda_state_slots * recurrent_elements * recurrent_widths[mamba_ssm_cache_dtype],
            "mla_latent_cache": mla_layers * mla_tokens * (self.kv_lora_rank + self.qk_rope_head_dim) * 2,
        }

    def _standard_gqa_kv_bytes(
            self, local_layers: int, tensor_parallel_size: int) -> int:
        """Return this rank's key+value bytes for a standard-GQA layout.

        A GQA layer keeps one key and one value head per local KV head, so the
        per-rank cache is two copies of the local head count times head width,
        scaled by the rank's layer count and the storage dtype.  Both the
        whole-plan and the layer-range callers compute the same quantity from
        different layer counts.
        """

        local_kv_heads = self.kv_heads_per_tensor_parallel_rank(
            tensor_parallel_size)
        return (
            2 * local_layers * local_kv_heads
            * self.head_dim * self.dtype_bytes)


    def _dsv4_layer_cache_components(
            self, ratio: int) -> tuple[tuple[str, Fraction, int | None], ...]:
        """Return one layer's exact DeepSeek-V4 cache inventory.

        Each entry is ``(kind, bytes_per_token, window_tokens)`` with an
        exact per-token ``Fraction`` and the sliding-window lifecycle cap
        (``None`` for unbounded stream caches):

        * ``swa_window_cache`` — every layer (attention.py:340
          ``DeepseekV4SWACache``): one packed fp8_ds_mla 584-byte state per
          token (sparse_swa.py:82), pages of 64 tokens padded to 576 B
          (sparse_swa.py:111-130); per-request footprint caps at
          ``window_size`` tokens.
        * ``compressed_kv_cache`` — compressor layers only
          (attention.py:841-863): one 584-byte state per ``compress_ratio``
          tokens.
        * ``indexer_k_cache`` — C4 layers only (attention.py:891-903,
          998-1004): one 132-byte FP8 state per 4 tokens (MXFP4 is SM100
          only, mla/indexer.py:67).
        * ``compressor_state_cache`` — compressor layers only
          (compressor.py:139-196): one fp32 ``2*coff*head_dim`` state per
          token, pages of 4/8 states padded to 576 B, window
          ``coff * compress_ratio``.
        * ``indexer_compressor_state_cache`` — C4 layers only
          (attention.py:1005-1014): the indexer's own head_dim=128
          compressor state cache, same fp32 semantics.
        """

        if self.is_dspark_draft:
            raise NotImplementedError("DSpark draft cache inventory requires its context-KV/proposer ownership")

        def padded_page_per_token(page_states: int,
                                  state_bytes: int) -> Fraction:
            page_bytes = page_states * state_bytes
            padded = -(-page_bytes // _DSV4_PAGE_ALIGNMENT_BYTES) \
                * _DSV4_PAGE_ALIGNMENT_BYTES
            return Fraction(padded, page_states)

        components: list[tuple[str, Fraction, int | None]] = [
            ("swa_window_cache",
             padded_page_per_token(
                 _DSV4_SWA_PAGE_TOKENS, _DSV4_SWA_STATE_BYTES),
             self.window_size),
        ]
        if ratio <= 1:
            return tuple(components)
        components.append(
            ("compressed_kv_cache",
             Fraction(_DSV4_COMPRESSED_STATE_BYTES, ratio), None))
        coff = 2 if ratio == 4 else 1
        compressor_state_bytes = 2 * coff * self.head_dim * 4
        components.append(
            ("compressor_state_cache",
             padded_page_per_token(
                 _DSV4_COMPRESSOR_STATE_PAGE_STATES[ratio],
                 compressor_state_bytes),
             coff * ratio))
        if ratio == 4:
            components.append(
                ("indexer_k_cache",
                 Fraction(_DSV4_INDEXER_K_STATE_BYTES_FP8, ratio), None))
            indexer_state_bytes = 2 * coff * self.index_head_dim * 4
            components.append(
                ("indexer_compressor_state_cache",
                 padded_page_per_token(
                     _DSV4_COMPRESSOR_STATE_PAGE_STATES[ratio],
                     indexer_state_bytes),
                 coff * ratio))
        return tuple(components)

    def dsv4_layer_cache_components(
            self, layer_index: int) -> tuple[tuple[str, Fraction, int | None], ...]:
        """Return the exact cache inventory of one DeepSeek-V4 layer."""

        if self.is_dspark_draft:
            raise NotImplementedError("DSpark draft cache ownership requires its context-KV/proposer source, not target cache layers")
        if self.attention_layout != "dsv4_sparse_mla":
            raise NotImplementedError(
                "DeepSeek-V4 cache inventory only exists for DeepSeek-V4 "
                f"sparse MLA, got {self.attention_layout!r}")
        if (isinstance(layer_index, bool)
                or not isinstance(layer_index, int)
                or not 0 <= layer_index < self.num_hidden_layers):
            raise ValueError("DeepSeek-V4 layer index out of range: "
                             f"{layer_index!r}")
        return self._dsv4_layer_cache_components(
            max(1, self.compress_ratios[layer_index]))

    def _dsv4_scheduler_group_page(
            self, states_per_block: int, state_bytes: int) -> int:
        """Return the 576-aligned page bytes of one DeepSeek-V4 cache group.

        Upstream pads every cache page up to the 576-byte TMA stride under
        the default fp8_ds_mla layout (attention.py:847-863
        ``alignment=576``, compressor.py:190, sparse_swa.py:83,127).
        """

        raw = states_per_block * state_bytes
        return (-(-raw // _DSV4_PAGE_ALIGNMENT_BYTES)
                * _DSV4_PAGE_ALIGNMENT_BYTES)

    def dsv4_scheduler_cache_groups(
            self, kv_block_size_tokens: int) -> tuple[DSV4SchedulerCacheGroup, ...]:
        """Return the Scheduler lifecycle groups of the DeepSeek-V4 caches.

        Each group carries its own page geometry and lifecycle so the
        control-plane Scheduler charges windowed caches by their per-request
        retention cap instead of the full context length:

        * ``swa_window_cache`` (every layer): sliding window
          ``window_size``, intrinsic 64-token page of 584-byte rows padded
          to 576 B (sparse_swa.py:72-130).
        * ``compressed_kv_cache`` (compressor layers): unbounded stream of
          584-byte states, one per ``compress_ratio`` tokens, pages
          576-aligned (attention.py:841-863).  Ratio-4 pages hold whole
          states at the Scheduler block width; ratio-128 blocks span 128
          tokens so each page holds exactly one state.
        * ``indexer_k_cache`` (C4 layers): unbounded stream of 132-byte FP8
          states, one per 4 tokens (attention.py:985-997; H200 SM90 uses
          FP8 because MXFP4 requires SM100, mla/indexer.py:58-73).
        * ``compressor_state_cache`` / ``indexer_compressor_state_cache``:
          sliding-window FP32 partial-state caches retaining
          ``coff * compress_ratio`` per-token states per request
          (compressor.py:139-196, attention.py:1005-1014).
        """

        if self.is_dspark_draft:
            raise NotImplementedError("DSpark draft Scheduler caches require their separate draft ownership")
        if self.attention_layout != "dsv4_sparse_mla":
            raise NotImplementedError(
                "DeepSeek-V4 scheduler cache groups only exist for "
                f"DeepSeek-V4 sparse MLA, got {self.attention_layout!r}")
        if (isinstance(kv_block_size_tokens, bool)
                or not isinstance(kv_block_size_tokens, int)
                or kv_block_size_tokens <= 0):
            raise ValueError("KV block size tokens must be a positive "
                             f"integer, got {kv_block_size_tokens!r}")
        layers = tuple(range(self.num_hidden_layers))
        ratios = tuple(
            max(1, ratio)
            for ratio in self.compress_ratios[:self.num_hidden_layers])
        c4_layers = tuple(
            index for index, ratio in enumerate(ratios) if ratio == 4)
        c128_layers = tuple(
            index for index, ratio in enumerate(ratios) if ratio == 128)
        groups: list[DSV4SchedulerCacheGroup] = [
            # The pinned 0.11.0 SlidingWindowSpec page formula charges the
            # K+V pair (2 * block * head), so the odd 585-byte SWA row is
            # ceiled to 586 (page 37504 vs the raw 576-aligned 37440;
            # +1/585 per token on this component, quantified in the
            # scheduler runtime artifact).
            DSV4SchedulerCacheGroup(
                kind="swa_window_cache",
                lifecycle="sliding_window",
                block_tokens=_DSV4_SWA_PAGE_TOKENS,
                page_bytes=2 * _DSV4_SWA_PAGE_TOKENS * 293,
                window_tokens=self.window_size,
                layer_indices=layers,
            ),
        ]
        if c4_layers:
            # Upstream pages the compressed/indexer caches at the SWA
            # cache's intrinsic 64-token block (sparse_swa.py:81), so one
            # page holds 16 C4 states; the 576-byte alignment then costs
            # exactly the upstream per-token padding.
            c4_states_per_block = _DSV4_SWA_PAGE_TOKENS // 4
            groups.append(DSV4SchedulerCacheGroup(
                kind="compressed_kv_cache_r4",
                lifecycle="stream",
                block_tokens=_DSV4_SWA_PAGE_TOKENS,
                page_bytes=self._dsv4_scheduler_group_page(
                    c4_states_per_block, _DSV4_COMPRESSED_STATE_BYTES),
                window_tokens=None,
                layer_indices=c4_layers,
            ))
            groups.append(DSV4SchedulerCacheGroup(
                kind="indexer_k_cache",
                lifecycle="stream",
                block_tokens=_DSV4_SWA_PAGE_TOKENS,
                page_bytes=self._dsv4_scheduler_group_page(
                    c4_states_per_block, _DSV4_INDEXER_K_STATE_BYTES_FP8),
                window_tokens=None,
                layer_indices=c4_layers,
            ))
        if c128_layers:
            # One 584-byte state per 128 tokens; a 128-token block holds
            # exactly one state (a narrower block could not hold one).
            groups.append(DSV4SchedulerCacheGroup(
                kind="compressed_kv_cache_r128",
                lifecycle="stream",
                block_tokens=128,
                page_bytes=self._dsv4_scheduler_group_page(
                    1, _DSV4_COMPRESSED_STATE_BYTES),
                window_tokens=None,
                layer_indices=c128_layers,
            ))
        for ratio, layer_indices in ((4, c4_layers), (128, c128_layers)):
            if not layer_indices:
                continue
            coff = 2 if ratio == 4 else 1
            state_bytes = 2 * coff * self.head_dim * 4
            page_states = _DSV4_COMPRESSOR_STATE_PAGE_STATES[ratio]
            groups.append(DSV4SchedulerCacheGroup(
                kind=f"compressor_state_cache_r{ratio}",
                lifecycle="sliding_window",
                block_tokens=page_states,
                page_bytes=self._dsv4_scheduler_group_page(
                    page_states, state_bytes),
                window_tokens=coff * ratio,
                layer_indices=layer_indices,
            ))
        if c4_layers:
            indexer_state_bytes = 2 * 2 * self.index_head_dim * 4
            page_states = _DSV4_COMPRESSOR_STATE_PAGE_STATES[4]
            groups.append(DSV4SchedulerCacheGroup(
                kind="indexer_compressor_state_cache",
                lifecycle="sliding_window",
                block_tokens=page_states,
                page_bytes=self._dsv4_scheduler_group_page(
                    page_states, indexer_state_bytes),
                window_tokens=2 * 4,
                layer_indices=c4_layers,
            ))
        return tuple(groups)

    def dsv4_cache_event_bytes(
            self, token_start: int, token_end: int) -> Fraction:
        """Return lifecycle-exact cache bytes of the token range [start, end).

        Stream caches (compressed KV, indexer K) charge their raw per-token
        rate across the whole range; windowed caches (SWA window, compressor
        partial states) charge only the intersection with their per-request
        window, matching the retention lifecycle of
        ``dsv4_scheduler_cache_groups``.
        """

        for name, value in (("token_start", token_start),
                            ("token_end", token_end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"cache event {name} must be an integer")
        if token_end < token_start or token_start < 0:
            raise ValueError(
                f"invalid cache event token range: {token_start}..{token_end}")
        total = Fraction(0)
        for layer_index in range(self.num_hidden_layers):
            for _kind, bytes_per_token, window in (
                    self._dsv4_layer_cache_components(
                        max(1, self.compress_ratios[layer_index]))):
                if window is None:
                    total += bytes_per_token * (token_end - token_start)
                else:
                    total += bytes_per_token * (
                        min(token_end, window) - min(token_start, window))
        return total

    def _dsv4_kv_bytes(self, ratios: tuple[int, ...]) -> Fraction:
        """Return exact per-token DeepSeek-V4 attention-cache bytes.

        The per-token total sums the complete per-layer inventory of
        ``_dsv4_layer_cache_components`` (SWA window cache on every layer,
        compressed KV, indexer K, and both FP32 compressor state caches).
        Sliding-window caches are charged at their full per-token rate,
        which is exact for sequences up to the window and conservatively
        bounds longer sequences (the per-request footprint caps at
        ``window_tokens`` states; the aggregate Scheduler block accounting
        cannot express the per-request cap).
        """

        total = Fraction(0)
        for ratio in ratios:
            for _, bytes_per_token, _ in self._dsv4_layer_cache_components(
                    max(1, ratio)):
                total += bytes_per_token
        return total

    @property
    def kernel_compress_ratios(self) -> tuple[int, ...]:
        """Compress ratios in the kernel/launch domain.

        The pinned configs spell pure-SWA layers as ratio 0; vLLM 0.30.0
        clamps with ``max(1, config.compress_ratios[layer_id])`` and maps
        the MTP layer (index >= num_hidden_layers) to ratio 1
        (deepseek_v4/attention.py:224-230), and the kernel contract
        (``dsv4_sparse_*`` / ``dsv4_compressed_state_cache`` shapes) keeps
        that post-clamp domain {1, 4, 128} where 1 means SWA.  This property
        is the single documented config-domain -> kernel-domain mapping;
        workload construction must consume it instead of the raw
        ``compress_ratios``.  The trailing MTP placeholder (config 0) maps
        to 1, matching the upstream else-branch.
        """

        if self.attention_layout != "dsv4_sparse_mla":
            raise NotImplementedError(
                "kernel compress ratios only exist for DeepSeek-V4 sparse "
                f"MLA, got {self.attention_layout!r}")
        return tuple(max(1, ratio) for ratio in self.compress_ratios)

    def kv_bytes_per_token_per_parallel_rank(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ) -> int:
        """Return resident attention-cache bytes on one parallel rank.

        Standard GQA stores one K and one V tensor.  MLA stores the single
        compressed ``kv_c || k_pe`` vector.  Pinned vLLM 0.27.1 GLM-5.2
        semantics add a separate FP8 index cache on full-indexer layers.  The
        DSA total is exact for PP=1; PP>1 remains unsupported because each
        stage owns a different number of full-indexer layers.  DeepSeek-V4
        sparse MLA returns an exact ``Fraction`` (the compressed cache
        amortizes one 584-byte state per ``compress_ratio`` tokens) and is
        likewise PP=1 only.
        """

        if (isinstance(tensor_parallel_size, bool)
                or not isinstance(tensor_parallel_size, int)):
            raise TypeError("tensor parallel size must be an integer")
        if tensor_parallel_size < 1:
            raise ValueError("tensor parallel size must be positive")
        if (isinstance(pipeline_parallel_size, bool)
                or not isinstance(pipeline_parallel_size, int)):
            raise TypeError("pipeline parallel size must be an integer")
        if pipeline_parallel_size < 1:
            raise ValueError("pipeline parallel size must be positive")
        if self.num_hidden_layers % pipeline_parallel_size:
            raise ValueError(
                "num_hidden_layers must be divisible by pipeline parallel "
                "size: "
                f"{self.num_hidden_layers} vs {pipeline_parallel_size}")
        local_layers = self.num_hidden_layers // pipeline_parallel_size
        if self.attention_layout == "hybrid_kda_mla":
            raise NotImplementedError("K3 hybrid capacity requires separate KDA state and NoPE MLA cache contracts")
        if self.attention_layout == "standard_gqa":
            return self._standard_gqa_kv_bytes(
                local_layers, tensor_parallel_size)
        if self.attention_layout == "dsv4_sparse_mla":
            if pipeline_parallel_size != 1:
                raise NotImplementedError(
                    "DeepSeek-V4 KV capacity for PP>1 requires rank-specific "
                    "compress-ratio layer placement")
            if self.num_attention_heads % tensor_parallel_size:
                raise ValueError(
                    "DeepSeek-V4 query heads must be divisible by tensor "
                    f"parallel size: {self.num_attention_heads} vs "
                    f"{tensor_parallel_size}")
            return self._dsv4_kv_bytes(
                self.compress_ratios[:self.num_hidden_layers])
        if self.attention_layout == "dsa_mla":
            if pipeline_parallel_size != 1:
                raise NotImplementedError(
                    "DSA KV capacity for PP>1 requires rank-specific full-"
                    "indexer layer placement")
            if self.num_attention_heads % tensor_parallel_size:
                raise ValueError(
                    "DSA query heads must be divisible by tensor parallel "
                    f"size: {self.num_attention_heads} vs "
                    f"{tensor_parallel_size}")
            compressed_bytes = (
                self.num_hidden_layers
                * (self.kv_lora_rank + self.qk_rope_head_dim)
                * self.dtype_bytes
            )
            index_scale_bytes = self.index_head_dim // 128 * 4
            full_indexer_layers = self.indexer_types.count("full")
            index_bytes = full_indexer_layers * (
                self.index_head_dim + index_scale_bytes)
            return compressed_bytes + index_bytes
        if self.attention_layout != "mla":
            raise AssertionError(
                f"unhandled attention layout: {self.attention_layout!r}")
        if self.num_attention_heads % tensor_parallel_size:
            raise ValueError(
                "MLA query heads must be divisible by tensor parallel size: "
                f"{self.num_attention_heads} vs {tensor_parallel_size}")

        return (
            local_layers
            * (self.kv_lora_rank + self.qk_rope_head_dim)
            * self.dtype_bytes
        )

    def kv_bytes_per_token_for_layer_range(
        self,
        tensor_parallel_size: int,
        layer_start: int,
        layer_end: int,
    ) -> int | Fraction:
        """Rank-local logical cache rate; native arenas own physical bytes."""

        for name, value in (
                ("tensor_parallel_size", tensor_parallel_size),
                ("layer_start", layer_start), ("layer_end", layer_end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if not 0 <= layer_start < layer_end <= self.num_hidden_layers:
            raise ValueError("KV layer range must be non-empty and in bounds")
        local_layers = layer_end - layer_start
        if self.attention_layout == "dsv4_sparse_mla":
            if self.num_attention_heads % tensor_parallel_size:
                raise ValueError("DeepSeek-V4 query heads must be divisible by tensor parallel size")
            return self._dsv4_kv_bytes(self.compress_ratios[layer_start:layer_end])
        if self.attention_layout == "hybrid_kda_mla":
            raise NotImplementedError("K3 hybrid capacity requires separate KDA state and NoPE MLA cache contracts")
        if self.attention_layout == "standard_gqa":
            return self._standard_gqa_kv_bytes(
                local_layers, tensor_parallel_size)
        if self.num_attention_heads % tensor_parallel_size:
            raise ValueError(
                "MLA/DSA query heads must be divisible by tensor parallel "
                f"size: {self.num_attention_heads} vs {tensor_parallel_size}")
        compressed_bytes = (
            local_layers
            * (self.kv_lora_rank + self.qk_rope_head_dim)
            * self.dtype_bytes)
        if self.attention_layout == "mla":
            return compressed_bytes
        if self.attention_layout == "dsa_mla":
            index_scale_bytes = self.index_head_dim // 128 * 4
            full_indexer_layers = self.indexer_types[
                layer_start:layer_end].count("full")
            return compressed_bytes + full_indexer_layers * (
                self.index_head_dim + index_scale_bytes)
        raise AssertionError(
            f"unhandled attention layout: {self.attention_layout!r}")

    def layer_uses_moe(self, layer_index: int) -> bool:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        limit = self.num_hidden_layers + (3 if self.dspark_block_size is not None else 0)
        if not 0 <= layer_index < limit:
            raise ValueError("layer_index is outside the target or exact DSpark draft model")
        if self.target_semantics is None:
            return bool(
                self.num_experts and layer_index >= self.num_dense_layers)
        return bool(
            self.num_experts
            and layer_index >= self.num_dense_layers
            and layer_index % self.target_semantics.moe_layer_frequency == 0)

    def moe_layer_ordinal(self, layer_index: int) -> "int | None":
        """Position of a MoE layer among the model's MoE layers, else None."""

        if not self.layer_uses_moe(layer_index):
            return None
        return sum(
            self.layer_uses_moe(index) for index in range(layer_index))

    def mtp_layer_uses_moe(self, mtp_layer_index: int) -> bool:
        if (isinstance(mtp_layer_index, bool)
                or not isinstance(mtp_layer_index, int)):
            raise TypeError("mtp_layer_index must be an integer")
        if not 0 <= mtp_layer_index < self.num_mtp_layers:
            raise ValueError("mtp_layer_index is outside the MTP model")
        if self.mtp_architecture == "glm4_moe_mtp":
            # vLLM 0.11 overrides the draft config to num_hidden_layers=0;
            # Glm4MoeMultiTokenPredictor therefore cycles weights starting at
            # decoder layer index zero.
            return bool(self.num_experts and 0 >= self.num_dense_layers)
        elif self.mtp_architecture == "glm_moe_dsa_mtp":
            if self.target_semantics is None:
                raise ValueError(
                    "GLM-5.2 MTP layer mapping requires target semantics")
            layer_index = self.num_hidden_layers + mtp_layer_index
            return bool(
                self.num_experts
                and layer_index >= self.num_dense_layers
                and layer_index
                % self.target_semantics.moe_layer_frequency == 0)
        elif self.mtp_architecture == "deepseek_mtp":
            if self.target_semantics is None:
                raise ValueError(
                    "DeepSeek-V3 MTP layer mapping requires target semantics")
            # vLLM DeepSeekMTP appends the NextN layer directly after the
            # main decoder stack, so it lands past first_k_dense_replace and
            # is a routed MoE layer under moe_layer_freq=1.
            layer_index = self.num_hidden_layers + mtp_layer_index
            return bool(
                self.num_experts
                and layer_index >= self.num_dense_layers
                and layer_index
                % self.target_semantics.moe_layer_frequency == 0)
        elif self.mtp_architecture == "none":
            raise ValueError("a model without MTP has no MTP decoder layers")
        else:
            raise ValueError(
                f"unsupported MTP architecture: {self.mtp_architecture!r}")


def is_validated_llama_tp2_model_scope(model: ModelConfig) -> bool:
    """Recognize the exact Llama config with source-backed TP=2 coverage."""

    if not isinstance(model, ModelConfig):
        raise TypeError("Llama TP=2 scope requires a ModelConfig")
    return (
        model.config_sha256
        == "ad0330f06f0e3d4defab454b7bfdff2ed3e89d2a6e7ae4679483e0cd1bfe3512"
        and model.model_type == "llama"
        and model.architecture == "LlamaForCausalLM"
        and model.hidden_size == 128
        and model.intermediate_size == 256
        and model.num_attention_heads == 2
        and model.num_key_value_heads == 1
        and model.num_hidden_layers == 1
        and model.head_dim == 64
        and model.vocab_size == 256
        and model.max_position_embeddings == 128
        and model.attention_layout == "standard_gqa"
        and not model.qk_norm
        and model.num_experts == 0
        and model.num_mtp_layers == 0
        and not model.tie_word_embeddings
        and not model.attention_bias
        and not model.mlp_bias
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
    )


def is_qwen3_moe_source_model_scope(model: ModelConfig) -> bool:
    """Ordinary unquantized Qwen3MoE model semantics; placement is separate."""

    if not isinstance(model, ModelConfig):
        raise TypeError("Qwen3MoE TP=EP=2 scope requires a ModelConfig")
    return (
        model.architecture == "Qwen3MoeForCausalLM"
        and model.model_type == "qwen3_moe"
        and model.attention_layout == "standard_gqa"
        and model.num_experts > 0
        and 0 < model.num_experts_per_tok <= model.num_experts
        and model.num_dense_layers == 0
        and model.num_shared_experts == 0
        and model.num_mtp_layers == 0
        and model.mtp_architecture == "none"
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
        and model.weight_group_size == 0
        and model.quantized_weight_targets == ()
        and model.quantized_weight_ignore == ()
    )


def is_validated_qwen3_moe_ep2_model_scope(model: ModelConfig) -> bool:
    """Retain the existing versioned TP=EP2 model domain."""
    return (is_qwen3_moe_source_model_scope(model) and model.num_experts % 2 == 0
        and model.moe_intermediate_size % 2 == 0)


def is_deepseek_v3_parallel_model_scope(model: ModelConfig, tensor_parallel_size: int, pipeline_parallel_size: int,
                                      *, expert_parallel=False) -> bool:
    """V3 ordinary FP8 TP/PP geometry, including the configured local L32.

    Row shards must retain whole block scales. The pinned source constructs
    its layer loop from num_hidden_layers; provider/normal capacity is separate.
    """
    tp, pp = tensor_parallel_size, pipeline_parallel_size
    if (not isinstance(model, ModelConfig) or type(expert_parallel) is not bool
            or type(tp) is not int or tp < 1 or type(pp) is not int or pp < 1
            or model.model_type != "deepseek_v3" or model.model_id != "deepseek-ai/DeepSeek-V3"
            or model.target_semantics.workload_contract != "deepseek_v3_vllm_0_27_1_logical_v1"
            or model.compute_dtype != "bf16" or model.weight_dtype != "fp8" or model.num_shared_experts != 1
            or pp > model.num_hidden_layers or model.num_attention_heads % tp or model.hidden_size % tp
            or model.intermediate_size % tp or model.moe_intermediate_size % tp
            or expert_parallel and model.num_experts % tp):
        return False
    heads = model.num_attention_heads // tp
    return all(width % 128 == 0 for width in (model.hidden_size, model.q_lora_rank, model.kv_lora_rank,
        model.intermediate_size // tp, model.moe_intermediate_size // tp,
        heads * (model.qk_nope_head_dim + model.qk_rope_head_dim),
        heads * (model.qk_nope_head_dim + model.v_head_dim), heads * model.v_head_dim))


def is_deepseek_v3_data_parallel_model_scope(model: ModelConfig, config) -> bool:
    """V3 whole-block FP8 TP/DP or uniform linear global EP geometry."""
    from src.serving.parallelism.parallel_config import ParallelConfig
    if (not isinstance(config, ParallelConfig) or config.data_parallel_size <= 1
            or config.context_parallel_size != 1 or config.sequence_parallel_size != 1
            or not is_deepseek_v3_parallel_model_scope(model, config.tensor_parallel_size,
                config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel)):
        return False
    if config.enable_expert_parallel:
        return model.num_experts % config.expert_parallel_size == 0
    expert_tp = config.tensor_parallel_size * config.data_parallel_size
    return model.moe_intermediate_size % expert_tp == 0 and model.moe_intermediate_size // expert_tp % 128 == 0


def is_glm_fp8_tensor_model_scope(model: ModelConfig, tensor_parallel_size: int) -> bool:
    """Official GLM block-FP8 source shards without quantization padding.

    The ordinary CP1 program retains BF16 activations/cache and replicates
    latent projections. Row/column-sharded FP8 weights require whole 128x128
    scale blocks. This establishes geometry, not runtime/provider capacity.
    """
    tp = tensor_parallel_size
    if (not isinstance(model, ModelConfig) or type(tp) is not int or tp < 1
            or model.model_type != "glm_moe_dsa" or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
            or model.quantized_weight_targets != ("Linear",) or model.num_shared_experts != 1
            or model.num_attention_heads % tp or model.hidden_size % tp
            or model.intermediate_size % tp or model.moe_intermediate_size % tp):
        return False
    heads = model.num_attention_heads // tp
    return all(width % 128 == 0 for width in (model.hidden_size, model.q_lora_rank, model.kv_lora_rank,
        model.intermediate_size // tp, model.moe_intermediate_size // tp,
        heads * (model.qk_nope_head_dim + model.qk_rope_head_dim),
        heads * (model.qk_nope_head_dim + model.v_head_dim), heads * model.v_head_dim))


def is_glm_mtp_tensor_model_scope(model: ModelConfig, tensor_parallel_size: int, *, expert_parallel: bool = False) -> bool:
    """GLM0.30 appended MTP geometry with ordinary TP and full logits.

    The rank-local source requires SP/CP1 and full-indexer stage starts; a
    nonempty target PP runs its whole drafter on the last stage. BF16 may use stage TP ranks as uniform EP;
    Block-FP8 uses an explicitly declared deferred-input Cutlass source;
    this geometry does not accept its provider or pricing. Rank-local FP8
    EP uses NoDPEP prepare and deferred Cutlass; cache geometry is independent.
    Coordinated DP uses uniform global EP and model SP at PP1 or internal MoE SP at PP>1.
    Native depth, padded rows, rejection and explicit draft IDs are separate.
    """
    tp = tensor_parallel_size
    if (not isinstance(model, ModelConfig) or type(tp) is not int or tp < 1 or type(expert_parallel) is not bool
            or model.model_type != "glm_moe_dsa" or model.attention_layout != "dsa_mla"
            or model.model_id not in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3")
            or not (model.config_sha256 == GLM_MOE_DSA_CONFIG_SHA256
                or model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S)
            or model.compute_dtype != "bf16" or model.weight_dtype not in ("bf16", "fp8")
            or model.mtp_architecture != "glm_moe_dsa_mtp" or model.num_mtp_layers != 1
            or model.num_attention_heads % tp or model.hidden_size % tp
            or model.intermediate_size % tp or model.moe_intermediate_size % tp
            or expert_parallel and (tp < 2 or model.num_experts % tp)
            or (-(-model.vocab_size // 64) * 64) % tp):
        return False
    return model.weight_dtype == "bf16" or is_glm_fp8_tensor_model_scope(model, tp)


def is_glm_fp8_data_parallel_model_scope(model: ModelConfig, parallelism) -> bool:
    """Ordinary CP1 GLM FP8 DP geometry, independent of MTP or pricing."""
    from src.serving.parallelism.parallel_config import ParallelConfig
    if not isinstance(parallelism, ParallelConfig) or not isinstance(model, ModelConfig):
        return False
    tp, dp, pp = parallelism.tensor_parallel_size, parallelism.data_parallel_size, parallelism.pipeline_parallel_size
    return (model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S
        and model.num_hidden_layers == 78 and is_glm_fp8_tensor_model_scope(model, tp)
        and dp > 1 and 1 <= pp <= model.num_hidden_layers and parallelism.context_parallel_size == 1
        and parallelism.sequence_parallel_size in ((1, tp) if parallelism.enable_expert_parallel else (1,))
        and (model.num_experts % parallelism.expert_parallel_size == 0 if parallelism.enable_expert_parallel
            else model.moe_intermediate_size % (tp * dp * 128) == 0))


def is_glm_mtp_data_parallel_model_scope(model: ModelConfig, parallelism) -> bool:
    """GLM BF16-compute CP1 DP source geometry, separate from provider/capacity.

    Uniform EP spans the actual TP*DP ranks at each target PP stage. The
    TP>1 EP uses model SP at PP1 and internal MoE SP at PP>1.
    TP1 needs no sequence sharding. Official block-FP8 retains whole128
    weight blocks in shared/dense TP and routed TP*DP shards, or complete
    routed experts under EP. The source binds deferred Cutlass input quantization.
    Both dtypes accept independent SP1 or the legacy explicit SP=TP declaration;
    actual model/MoE SP comes from the source TP/DP/EP/PP configuration.
    """
    from src.serving.parallelism.parallel_config import ParallelConfig
    if not isinstance(parallelism, ParallelConfig):
        return False
    tp, dp, pp = (parallelism.tensor_parallel_size, parallelism.data_parallel_size, parallelism.pipeline_parallel_size)
    return (is_glm_mtp_tensor_model_scope(model, tp)
        and model.num_hidden_layers == 78 and dp > 1 and 1 <= pp <= model.num_hidden_layers
        and parallelism.context_parallel_size == 1
        and parallelism.sequence_parallel_size in ((1, tp) if parallelism.enable_expert_parallel else (1,))
        and (model.num_experts % parallelism.expert_parallel_size == 0
            if parallelism.enable_expert_parallel else model.moe_intermediate_size % (tp * dp) == 0
                and (model.weight_dtype == "bf16" or model.moe_intermediate_size // (tp * dp) % 128 == 0)))


def is_deepseek_v4_parallel_model_scope(model: ModelConfig, tensor_parallel_size: int,
                                       pipeline_parallel_size: int, *, expert_parallel=False) -> bool:
    """Ordinary target-only V4 TP/PP geometry; no provider acceptance.

    Grouped output divides whole output groups. Logical MXFP4 shards keep
    whole32-value groups; the Marlin source pads storage to128 columns.
    Draft DSpark identities are separate models.
    """
    tp, pp = tensor_parallel_size, pipeline_parallel_size
    if (not isinstance(model, ModelConfig) or type(expert_parallel) is not bool or type(tp) is not int or tp < 1
            or type(pp) is not int or not 1 <= pp <= model.num_hidden_layers
            or model.model_type != "deepseek_v4" or model.attention_layout != "dsv4_sparse_mla"
            or not is_named_target_model_identity("deepseek_v4", model.config_sha256)
            or model.model_id not in ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro")
            or model.compute_dtype != "bf16" or model.weight_dtype != "fp8" or model.expert_dtype != "fp4"
            or model.o_groups % tp or model.num_attention_heads % tp or model.hidden_size % tp
            or model.num_experts % tp or model.moe_intermediate_size % tp
            or (-(-model.vocab_size // 64) * 64) % tp):
        return False
    return (model.hidden_size % 256 == 0 and (expert_parallel or model.moe_intermediate_size // tp % 32 == 0)
        and model.num_shared_experts * model.moe_intermediate_size % 128 == 0
        and 0 < model.num_attention_heads // tp <= 128)


def is_deepseek_v4_data_parallel_model_scope(model: ModelConfig, config) -> bool:
    """Modular V4 whole-shard DP/EP geometry, without provider acceptance."""
    from src.serving.parallelism.parallel_config import ParallelConfig
    if (not isinstance(config, ParallelConfig) or config.data_parallel_size <= 1
            or config.context_parallel_size != 1 or config.sequence_parallel_size != 1
            or not is_deepseek_v4_parallel_model_scope(model, config.tensor_parallel_size,
                config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel)):
        return False
    if config.enable_expert_parallel:
        return model.num_experts % config.expert_parallel_size == 0
    expert_tp = config.tensor_parallel_size * config.data_parallel_size
    return model.moe_intermediate_size % expert_tp == 0 and model.moe_intermediate_size // expert_tp % 32 == 0


def is_kimi_k3_pipeline_model_scope(model: ModelConfig, pipeline_parallel_size: int) -> bool:
    """Nonempty PP geometry of the adapted TP1 K3 hybrid source program."""
    if not isinstance(model, ModelConfig):
        raise TypeError("K3 pipeline geometry requires ModelConfig")
    return (type(pipeline_parallel_size) is int and 1 <= pipeline_parallel_size <= model.num_hidden_layers
        and model.model_type == "kimi_linear" and model.attention_layout == "hybrid_kda_mla"
        and model.kimi_k3_semantics is not None and model.compute_dtype == "bf16"
        and model.weight_dtype == "mxfp4" and model.num_mtp_layers == 0)


def is_kimi_k3_parallel_model_scope(model: ModelConfig, tensor_parallel_size: int,
                                    pipeline_parallel_size: int, *, expert_parallel=False) -> bool:
    """Ordinary DP1 K3 TP/EP/PP geometry, without a provider/capacity claim."""
    tp = tensor_parallel_size
    if not is_kimi_k3_pipeline_model_scope(model, pipeline_parallel_size):
        return False
    source = model.kimi_k3_semantics
    return (type(tp) is int and tp > 0 and type(expert_parallel) is bool
        and (not expert_parallel or tp > 1 and model.num_experts % tp == 0)
        and model.num_attention_heads % tp == 0
        and source.kda_num_heads % tp == 0 and model.hidden_size % tp == 0
        and model.intermediate_size % tp == 0 and model.moe_intermediate_size % tp == 0
        and (model.moe_intermediate_size * model.num_shared_experts) % tp == 0
        and (-(-model.vocab_size // 64) * 64) % tp == 0)


def is_kimi_k3_data_parallel_model_scope(model: ModelConfig, parallelism) -> bool:
    """Current ordinary modular K3 DP source, with exact intermediate shards.

    TP>1 global EP uses model SP at PP1. The modular PP>1 constructor
    explicitly disables that SP path and retains the ordinary TP tail.
    """
    from src.serving.parallelism.parallel_config import ParallelConfig
    if not isinstance(parallelism, ParallelConfig):
        raise TypeError("K3 DP geometry requires ParallelConfig")
    tp, dp = parallelism.tensor_parallel_size, parallelism.data_parallel_size
    if (dp < 2 or parallelism.sequence_parallel_size != 1 or parallelism.context_parallel_size != 1
            or not is_kimi_k3_parallel_model_scope(model, tp, parallelism.pipeline_parallel_size)):
        return False
    if parallelism.enable_expert_parallel:
        return model.num_experts % (tp * dp) == 0
    return (model.moe_intermediate_size % (tp * dp * 64) == 0
        and max(model.moe_intermediate_size // tp, 256) % (dp * 64) == 0)


def is_kimi_k2_parallel_model_scope(model: ModelConfig, tensor_parallel_size: int,
                                     pipeline_parallel_size: int, *, expert_parallel=False) -> bool:
    """K2 text source geometry, independent of provider and capacity.

    Ordinary MLA divides query heads; the shared/dense MLP uses TP, while
    routed experts use TP shards or stage-local EP. INT4 reduction shards
    retain whole checkpoint quantization groups. No DP/SP source is implied.
    """
    if not isinstance(model, ModelConfig):
        raise TypeError("Kimi parallel geometry requires ModelConfig")
    if (type(tensor_parallel_size) is not int or tensor_parallel_size < 1
            or type(pipeline_parallel_size) is not int or pipeline_parallel_size < 1
            or type(expert_parallel) is not bool):
        raise ValueError("Kimi parallel geometry requires positive TP/PP and boolean EP")
    contracts = {"moonshotai/Kimi-K2.6": "kimi_k2_6_vllm_0_27_1_logical_v1",
                 "moonshotai/Kimi-K2.7-Code": "kimi_k2_7_code_vllm_0_30_0_logical_v1"}
    tp = tensor_parallel_size
    return (model.model_type == "kimi_k2" and model.model_id in contracts
        and model.target_semantics is not None
        and model.target_semantics.workload_contract == contracts[model.model_id]
        and model.attention_layout == "mla" and model.compute_dtype == "bf16"
        and model.weight_dtype == "mixed_int4_bf16" and model.weight_group_size > 0
        and model.num_shared_experts == 1 and model.num_mtp_layers == 0
        and model.num_attention_heads % tp == 0 and model.hidden_size % tp == 0
        and model.intermediate_size % tp == 0 and model.moe_intermediate_size % tp == 0
        and (-(-model.vocab_size // 64) * 64) % tp == 0
        and pipeline_parallel_size <= model.num_hidden_layers
        and (model.num_experts % tp == 0 if expert_parallel else
            model.moe_intermediate_size // tp % model.weight_group_size == 0))



def is_kimi_k2_data_parallel_model_scope(model: ModelConfig, config) -> bool:
    """Ordinary K2 whole-group expert TP or uniform global EP; no provider claim."""
    from src.serving.parallelism.parallel_config import ParallelConfig
    if (not isinstance(config, ParallelConfig) or config.data_parallel_size <= 1
            or config.context_parallel_size != 1 or config.sequence_parallel_size != 1
            or not is_kimi_k2_parallel_model_scope(model, config.tensor_parallel_size,
                config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel)):
        return False
    if config.enable_expert_parallel:
        return model.num_experts % config.expert_parallel_size == 0
    expert_tp = config.tensor_parallel_size * config.data_parallel_size
    return (model.moe_intermediate_size % expert_tp == 0
        and model.moe_intermediate_size // expert_tp % model.weight_group_size == 0)

def is_dense_parallel_model_scope(model: ModelConfig, tensor_parallel_size: int,
                                  pipeline_parallel_size: int) -> bool:
    """Adapted unquantized GQA source geometry, separate from backend gates.

    vLLM Qwen3 inherits Qwen2's PP decoder loop; Llama has the same
    hidden/residual stage ports. Both use QKV/row parallel projections.
    This predicate does not grant native provider or measured accuracy.
    """
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.serving.parallelism.tp import validate_gqa_parallelism
    if not isinstance(model, ModelConfig):
        raise TypeError("dense parallel scope requires ModelConfig")
    config = ParallelConfig(tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size)
    if (model.model_type not in ("qwen3", "llama") or model.num_experts
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != model.weight_dtype or model.compute_dtype not in ("bf16", "fp16")
            or (model.model_type == "llama" and model.compute_dtype != "bf16")
            or model.num_mtp_layers or model.hidden_act != "silu" or model.rms_norm_eps is None
            or model.rope_scaling_type not in (("default", "llama3") if model.model_type == "llama" else ("default",))):
        return False
    try:
        validate_gqa_parallelism(model, config)
    except ValueError:
        return False
    return True


def is_validated_qwen3_tp1_pp_model_scope(model: ModelConfig, pipeline_parallel_size: int) -> bool:
    """Dense Qwen with contiguous nonempty pipeline stages."""
    if not isinstance(model, ModelConfig) or type(pipeline_parallel_size) is not int or pipeline_parallel_size < 1:
        raise ValueError("Qwen pipeline scope requires ModelConfig and a positive stage count")
    return (pipeline_parallel_size > 1 and model.model_type == "qwen3"
        and model.architecture == "Qwen3ForCausalLM" and model.attention_layout == "standard_gqa"
        and model.compute_dtype == model.weight_dtype == "bf16" and not model.num_experts
        and not model.num_mtp_layers
        and model.hidden_act == "silu" and model.rms_norm_eps is not None
        and model.rope_scaling_type == "default"
        and pipeline_parallel_size <= model.num_hidden_layers)


def is_validated_qwen3_pipeline_model_scope(model: ModelConfig, tensor_parallel_size: int,
                                            pipeline_parallel_size: int) -> bool:
    """Contiguous Qwen stages with legal rank-local attention/MLP shards."""
    if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
        raise ValueError("Qwen pipeline scope requires a positive tensor-parallel size")
    tp = tensor_parallel_size
    return (is_validated_qwen3_tp1_pp_model_scope(model, pipeline_parallel_size)
        and model.num_attention_heads % tp == 0
        and model.hidden_size % tp == 0 and model.intermediate_size % tp == 0
        and (model.num_key_value_heads % tp == 0 or tp % model.num_key_value_heads == 0))


def is_qwen3_moe_pipeline_model_scope(model: ModelConfig, tensor_parallel_size: int,
                                     pipeline_parallel_size: int) -> bool:
    """Qwen3Moe's ordinary BF16 decoder loop with contiguous PP ownership.

    Expert ownership is validated by the caller's EP plan; PP partitions the
    decoder layers and leaves each stage's TP/EP group independent.
    """
    if (type(tensor_parallel_size) is not int or tensor_parallel_size < 1
            or type(pipeline_parallel_size) is not int or pipeline_parallel_size < 1):
        raise ValueError("Qwen3MoE pipeline requires positive TP and PP sizes")
    tp = tensor_parallel_size
    return (is_qwen3_moe_source_model_scope(model)
        and 1 < pipeline_parallel_size <= model.num_hidden_layers
        and model.hidden_act == "silu" and model.rms_norm_eps is not None
        and model.rope_scaling_type == "default"
        and model.num_attention_heads % tp == 0 and model.hidden_size % tp == 0
        and (model.num_key_value_heads % tp == 0 or tp % model.num_key_value_heads == 0))


def is_validated_qwen3_tp2_host_kv_model_scope(
    model: ModelConfig,
) -> bool:
    """Recognize the exact Qwen3 config used by the TP=2 Host-KV gate."""

    if not isinstance(model, ModelConfig):
        raise TypeError("Qwen3 TP=2 Host-KV scope requires a ModelConfig")
    return (
        model.config_sha256
        == "0c128be62e30095587ed663284f6e7f09b4473909dd1b1428f3babc798f14c90"
        and model.model_type == "qwen3"
        and model.architecture == "Qwen3ForCausalLM"
        and model.hidden_size == 256
        and model.intermediate_size == 512
        and model.num_attention_heads == 4
        and model.num_key_value_heads == 2
        and model.num_hidden_layers == 2
        and model.head_dim == 64
        and model.vocab_size == 64
        and model.max_position_embeddings == 128
        and model.attention_layout == "standard_gqa"
        and model.qk_norm
        and model.num_experts == 0
        and model.num_mtp_layers == 0
        and not model.tie_word_embeddings
        and not model.attention_bias
        and not model.mlp_bias
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
    )


def _rope_scaling_type(cfg: dict) -> str:
    """Read the HuggingFace rope scaling type that selects the vLLM class.

    ``get_rope`` reads ``rope_parameters["rope_type"]`` in the newer configs
    and the legacy ``rope_scaling`` block otherwise, where the key is spelled
    ``rope_type`` (llama-3.x, qwen3) or ``type`` (DeepSeek-family).  Absent or
    null scaling means the ``"default"`` branch.
    """

    present = tuple(
        name for name in ("rope_parameters", "rope_scaling") if name in cfg)
    if not present:
        # get_rope reads rope_parameters.get("rope_type", "default"), so a
        # config without any scaling block takes the "default" branch.
        return "default"
    if len(present) != 1:
        raise ValueError(
            "model config carries both rope_parameters and rope_scaling")
    raw = cfg[present[0]]
    if raw is None:
        return "default"
    if not isinstance(raw, dict):
        raise TypeError(f"model config {present[0]} must be a mapping or null")
    if "mrope_section" in raw:
        raise ValueError(
            "multimodal RoPE (mrope_section) is not a supported Serving "
            "rope program")
    if "use_fope" in raw and raw["use_fope"]:
        raise ValueError(
            "Fourier RoPE (use_fope) is not a supported Serving rope program")
    if "rope_type" in raw:
        scaling_type = raw["rope_type"]
    elif "type" in raw:
        scaling_type = raw["type"]
    else:
        raise ValueError(
            f"model config {present[0]} carries no rope type field")
    if not isinstance(scaling_type, str) or not scaling_type:
        raise ValueError(
            f"model config {present[0]} rope type must be a non-empty string")
    if scaling_type not in VLLM_ROPE_CLASS_BY_SCALING_TYPE:
        raise ValueError(
            f"unsupported vLLM rope scaling type: {scaling_type!r}")
    return scaling_type


def _kimi_k3_semantics(cfg: dict) -> KimiK3Semantics:
    raw = cfg["linear_attn_config"]
    if not isinstance(raw, dict) or set(raw) != {
            "full_attn_layers", "kda_layers", "head_dim", "num_heads",
            "short_conv_kernel_size", "use_full_rank_gate", "gate_lower_bound"}:
        raise ValueError("K3 requires its explicit hybrid attention config")
    count = cfg["num_hidden_layers"]
    groups = []
    for name in ("full_attn_layers", "kda_layers"):
        layers = raw[name]
        if (not isinstance(layers, list) or not layers
                or any(type(layer) is not int or not 1 <= layer <= count for layer in layers)
                or layers != sorted(set(layers))):
            raise ValueError("K3 attention layer lists require unique sorted one-based indices")
        groups.append(set(layers))
    full, kda = groups
    if full & kda or full | kda != set(range(1, count + 1)):
        raise ValueError("K3 KDA and MLA layer lists must partition the decoder stack")
    if (cfg["hidden_act"] != "situ" or cfg["use_grouped_topk"] is not True
            or cfg["topk_method"] != "noaux_tc"
            or cfg["moe_router_activation_func"] != "sigmoid"
            or cfg["num_expert_group"] != 1 or cfg["topk_group"] != 1
            or cfg["moe_renormalize"] is not True):
        raise NotImplementedError("K3 source scope requires SiTU and normalized single-group sigmoid routing")
    return KimiK3Semantics(
        layer_attention_types=tuple("kda" if layer in kda else "mla_nope" for layer in range(1, count + 1)),
        kda_num_heads=raw["num_heads"], kda_head_dim=raw["head_dim"],
        short_conv_kernel_size=raw["short_conv_kernel_size"],
        use_full_rank_gate=raw["use_full_rank_gate"], gate_lower_bound=raw["gate_lower_bound"],
        attention_residual_block_size=cfg["attn_res_block_size"],
        routed_expert_hidden_size=cfg["routed_expert_hidden_size"],
        latent_moe_use_norm=cfg["latent_moe_use_norm"],
        activation_situ_beta=cfg["activation_situ_beta"],
        activation_situ_linear_beta=cfg["activation_situ_linear_beta"],
        mla_use_nope=cfg["mla_use_nope"], mla_use_output_gate=cfg["mla_use_output_gate"],
        moe_layer_frequency=cfg["moe_layer_freq"], moe_router_dtype="fp32",
        rms_norm_eps=cfg["rms_norm_eps"])


def load_model_config(model_path: str | Path) -> ModelConfig:
    """Load a repository JSON file or a HuggingFace model directory."""
    model_path = expand_models_root(str(model_path))
    (
        root_cfg,
        config_sha256,
        source_config_sha256,
        canonical_config_sha256,
        model_id,
        model_revision,
        model_source_url,
        model_scope,
    ) = _load_model_json(model_path)
    cfg, config_extraction_path = _text_model_config(
        root_cfg,
        has_fixed_provenance=model_id is not None,
    )
    extracted_config_sha256 = hashlib.sha256(json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    hidden_size = cfg["hidden_size"]
    num_attention_heads = cfg["num_attention_heads"]
    num_key_value_heads = cfg["num_key_value_heads"]
    model_type = cfg["model_type"]
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("model config model_type must be a non-empty string")
    if model_type not in _MODEL_SCHEMAS:
        raise ValueError(f"unsupported model_type: {model_type!r}")
    if model_type in (
            "glm_moe_dsa", "kimi_k2", "deepseek_v3", "deepseek_v4", "kimi_linear"
    ) and model_id is None:
        raise ValueError(
            f"model_type {model_type!r} requires fixed model provenance")
    if model_type == "kimi_linear" and config_sha256 != KIMI_K3_CONFIG_SHA256:
        raise ValueError("K3 currently requires the exact official config identity")
    if model_type in _TARGET_MODEL_PROVENANCE or model_type == "kimi_linear":
        if config_sha256 in _OFFICIAL_CONFIG_PROVENANCE:
            expected_provenance = _OFFICIAL_CONFIG_PROVENANCE[config_sha256]
        elif config_sha256 == GLM_MOE_DSA_FP8_CONFIG_SHA256:
            expected_provenance = _GLM_FP8_PROVENANCE
        elif config_sha256 == DEEPSEEK_V3_L32_CONFIG_SHA256:
            expected_provenance = _DSV3_L32_PROVENANCE
        elif config_sha256 == DEEPSEEK_V4_PRO_CONFIG_SHA256:
            expected_provenance = _DSV4_PRO_PROVENANCE
        else:
            expected_provenance = _TARGET_MODEL_PROVENANCE[model_type]
        actual_provenance = {
            "model_id": model_id,
            "revision": model_revision,
            "source_config_sha256": source_config_sha256,
            "canonical_config_sha256": canonical_config_sha256,
            "source_url": model_source_url,
            "extraction_path": config_extraction_path,
            "extracted_config_sha256": extracted_config_sha256,
        }
        if actual_provenance != expected_provenance:
            raise ValueError(
                f"model_type {model_type!r} provenance disagrees with the "
                f"fixed target revision: actual={actual_provenance}, "
                f"expected={expected_provenance}")
    schema = _MODEL_SCHEMAS[model_type]
    architectures = cfg["architectures"]
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("model config requires exactly one architecture")
    architecture = architectures[0]
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("model config architecture must be a non-empty string")
    if architecture != schema["architecture"]:
        raise ValueError(
            f"model_type {model_type!r} requires architecture "
            f"{schema['architecture']!r}, got {architecture!r}")
    dspark = model_type == "deepseek_v4" and _is_dspark_config_identity(config_sha256)
    if dspark and type(cfg["dspark_target_layer_ids"]) is not list:
        raise TypeError("DSpark target layer IDs require the original JSON list")
    expected_v4_keys = _DSV4_CONFIG_KEYS | _DSPARK_CONFIG_KEYS if dspark else _DSV4_CONFIG_KEYS
    if model_type == "deepseek_v4" and set(cfg) != expected_v4_keys:
        raise ValueError(
            "DeepSeek-V4 config keys must be exactly the pinned campaign "
            f"key set, got {sorted(cfg)}")

    expert_fields = {
        "num_experts", "num_local_experts", "num_experts_per_tok",
        "experts_per_token", "moe_intermediate_size", "n_routed_experts",
        "n_shared_experts", "num_experts_per_token", "num_shared_experts",
    }
    present_expert_fields = expert_fields & set(cfg)
    if schema["moe"] is True:
        expert_count_field = schema["expert_count_field"]
        if not isinstance(expert_count_field, str):
            raise AssertionError(
                f"MoE schema lacks expert count for {model_type!r}")
        required_expert_fields = {
            expert_count_field,
            "num_experts_per_token" if model_type == "kimi_linear" else "num_experts_per_tok",
            "moe_intermediate_size"}
        if model_type == "kimi_linear":
            required_expert_fields.add("num_shared_experts")
        if model_type in ("glm4_moe", "glm_moe_dsa", "kimi_k2", "deepseek_v3",
             "deepseek_v4"):
            required_expert_fields.add("n_shared_experts")
        if present_expert_fields != required_expert_fields:
            raise ValueError(
                f"model_type {model_type!r} requires expert fields "
                f"{sorted(required_expert_fields)}, got "
                f"{sorted(present_expert_fields)}")
        num_experts = cfg[expert_count_field]
        num_experts_per_tok = cfg["num_experts_per_token"] if model_type == "kimi_linear" else cfg["num_experts_per_tok"]
        moe_intermediate_size = cfg["moe_intermediate_size"]
        if model_type in ("glm4_moe", "glm_moe_dsa", "kimi_k2", "deepseek_v3",
             "deepseek_v4"):
            shared_experts = cfg["n_shared_experts"]
            if (isinstance(shared_experts, bool)
                    or not isinstance(shared_experts, int)
                    or shared_experts <= 0):
                raise ValueError(
                    f"model_type {model_type!r} requires a positive "
                    "n_shared_experts value")
            num_shared_experts = shared_experts
        elif model_type == "kimi_linear":
            num_shared_experts = cfg["num_shared_experts"]
        elif model_type == "qwen3_moe":
            num_shared_experts = 0
        else:
            raise AssertionError(
                f"unexpected MoE model_type: {model_type!r}")
    elif schema["moe"] is False:
        if present_expert_fields:
            raise ValueError(
                f"dense model_type {model_type!r} contains expert fields: "
                f"{sorted(present_expert_fields)}")
        num_experts = 0
        num_experts_per_tok = 0
        moe_intermediate_size = 0
        num_shared_experts = 0
    else:
        raise AssertionError(f"invalid model schema for {model_type!r}")

    if (isinstance(hidden_size, bool) or not isinstance(hidden_size, int)
            or isinstance(num_attention_heads, bool)
            or not isinstance(num_attention_heads, int)
            or isinstance(num_key_value_heads, bool)
            or not isinstance(num_key_value_heads, int)):
        raise TypeError(
            "model hidden_size, num_attention_heads, and "
            "num_key_value_heads must be integers")
    if (hidden_size <= 0 or num_attention_heads <= 0
            or num_key_value_heads <= 0):
        raise ValueError(
            "model hidden_size, num_attention_heads, and "
            "num_key_value_heads must be positive")
    if num_attention_heads % num_key_value_heads:
        raise ValueError(
            "model num_attention_heads must be divisible by "
            "num_key_value_heads")
    if schema["head_dim"] == "required":
        head_dim = cfg["head_dim"]
        if (isinstance(head_dim, bool) or not isinstance(head_dim, int)):
            raise TypeError("model head_dim must be an integer")
        if head_dim <= 0:
            raise ValueError("model head_dim must be positive")
    elif schema["head_dim"] == "derived":
        if hidden_size % num_attention_heads:
            raise ValueError(
                "model hidden_size must be divisible by num_attention_heads "
                "when head_dim is derived")
        derived_head_dim = hidden_size // num_attention_heads
        head_dim = derived_head_dim
        if "head_dim" in cfg and cfg["head_dim"] != derived_head_dim:
            raise ValueError(
                f"model head_dim {cfg['head_dim']!r} disagrees with derived "
                f"value {derived_head_dim}")
    elif schema["head_dim"] == "mla":
        mla_dimension_fields = (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        )
        for field in mla_dimension_fields:
            value = cfg[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"model config {field} must be an integer")
            if value <= 0:
                raise ValueError(f"model config {field} must be positive")
        head_dim = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
        if "qk_head_dim" in cfg and cfg["qk_head_dim"] != head_dim:
            raise ValueError(
                "model qk_head_dim disagrees with qk_nope_head_dim plus "
                "qk_rope_head_dim")
        if ("head_dim" in cfg
                and cfg["head_dim"] != cfg["qk_nope_head_dim"]):
            raise ValueError(
                "MLA config head_dim must equal qk_nope_head_dim")
    elif schema["head_dim"] == "dsv4":
        # DeepSeek-V4 spells the full per-head width explicitly and carries
        # only qk_rope_head_dim; the NoPE width is the remainder and the V
        # width equals the full head_dim (vLLM 0.30.0
        # deepseek_v4/attention.py:211-213).
        head_dim = cfg["head_dim"]
        if isinstance(head_dim, bool) or not isinstance(head_dim, int):
            raise TypeError("model head_dim must be an integer")
        if head_dim <= 0:
            raise ValueError("model head_dim must be positive")
        qk_rope = cfg["qk_rope_head_dim"]
        if isinstance(qk_rope, bool) or not isinstance(qk_rope, int):
            raise TypeError("model config qk_rope_head_dim must be an integer")
        if not 0 < qk_rope < head_dim:
            raise ValueError(
                "DeepSeek-V4 qk_rope_head_dim must be positive and smaller "
                "than head_dim")
    else:
        raise AssertionError(f"invalid head_dim schema for {model_type!r}")

    attention_layout = schema["attention_layout"]
    compress_ratios = ()
    compress_rope_theta = 0
    window_size = 0
    o_lora_rank = 0
    o_groups = 0
    hc_mult = 0
    hc_sinkhorn_iters = 0
    num_hash_layers = 0
    swiglu_limit = 0.0
    expert_dtype = ""
    if attention_layout == "standard_gqa":
        q_lora_rank = 0
        kv_lora_rank = 0
        qk_nope_head_dim = 0
        qk_rope_head_dim = 0
        v_head_dim = 0
        index_n_heads = 0
        index_head_dim = 0
        index_topk = 0
        indexer_types = ()
        index_share_for_mtp_iteration = False
    elif attention_layout in ("mla", "dsa_mla", "dsv4_sparse_mla", "hybrid_kda_mla"):
        q_lora_rank = cfg["q_lora_rank"]
        qk_rope_head_dim = cfg["qk_rope_head_dim"]
        if attention_layout == "dsv4_sparse_mla":
            # DeepSeek-V4 has no KV LoRA; the NoPE width is head_dim minus
            # the rope width and the V width equals the full head_dim.
            kv_lora_rank = 0
            qk_nope_head_dim = head_dim - qk_rope_head_dim
            v_head_dim = head_dim
        else:
            kv_lora_rank = cfg["kv_lora_rank"]
            qk_nope_head_dim = cfg["qk_nope_head_dim"]
            v_head_dim = cfg["v_head_dim"]
        if attention_layout in ("mla", "hybrid_kda_mla"):
            index_n_heads = 0
            index_head_dim = 0
            index_topk = 0
            indexer_types = ()
            index_share_for_mtp_iteration = False
        elif attention_layout == "dsa_mla":
            index_n_heads = cfg["index_n_heads"]
            index_head_dim = cfg["index_head_dim"]
            index_topk = cfg["index_topk"]
            raw_indexer_types = cfg["indexer_types"]
            if not isinstance(raw_indexer_types, list):
                raise TypeError("DSA indexer_types must be a list")
            indexer_types = tuple(raw_indexer_types)
            index_share_for_mtp_iteration = cfg[
                "index_share_for_mtp_iteration"]
        elif attention_layout == "dsv4_sparse_mla":
            index_n_heads = cfg["index_n_heads"]
            index_head_dim = cfg["index_head_dim"]
            index_topk = cfg["index_topk"]
            indexer_types = ()
            index_share_for_mtp_iteration = False
            raw_compress_ratios = cfg["compress_ratios"]
            if not isinstance(raw_compress_ratios, list):
                raise TypeError(
                    "DeepSeek-V4 compress_ratios must be a list")
            compress_ratios = tuple(raw_compress_ratios)
            raw_compress_rope_theta = cfg["compress_rope_theta"]
            if isinstance(raw_compress_rope_theta, bool) or not isinstance(
                    raw_compress_rope_theta, int):
                raise TypeError(
                    "DeepSeek-V4 compress_rope_theta must be an integer")
            compress_rope_theta = raw_compress_rope_theta
            window_size = cfg["sliding_window"]
            o_lora_rank = cfg["o_lora_rank"]
            o_groups = cfg["o_groups"]
            hc_mult = cfg["hc_mult"]
            hc_sinkhorn_iters = cfg["hc_sinkhorn_iters"]
            num_hash_layers = cfg["num_hash_layers"]
            raw_hc_eps = cfg["hc_eps"]
            if (isinstance(raw_hc_eps, bool)
                    or not isinstance(raw_hc_eps, (int, float))
                    or not math.isfinite(raw_hc_eps) or raw_hc_eps <= 0):
                raise ValueError(
                    "DeepSeek-V4 hc_eps must be finite and positive")
            raw_swiglu_limit = cfg["swiglu_limit"]
            if (isinstance(raw_swiglu_limit, bool)
                    or not isinstance(raw_swiglu_limit, (int, float))):
                raise TypeError("DeepSeek-V4 swiglu_limit must be numeric")
            swiglu_limit = float(raw_swiglu_limit)
            raw_expert_dtype = cfg["expert_dtype"]
            if not isinstance(raw_expert_dtype, str):
                raise TypeError("DeepSeek-V4 expert_dtype must be a string")
            expert_dtype = raw_expert_dtype
        else:
            raise AssertionError(
                f"unhandled attention layout: {attention_layout!r}")
    else:
        raise AssertionError(
            f"invalid attention layout schema for {model_type!r}")

    dtype_fields = tuple(
        field for field in ("dtype", "torch_dtype") if field in cfg)
    if not dtype_fields:
        raise ValueError("model config requires dtype or torch_dtype")
    compute_dtypes = []
    for field in dtype_fields:
        raw_dtype = cfg[field]
        if not isinstance(raw_dtype, str):
            raise TypeError(f"model config {field} must be a string")
        dtype_name = raw_dtype.lower()
        if dtype_name not in _CONFIG_DTYPE_NAMES:
            raise ValueError(
                f"unsupported model {field}: {dtype_name!r}")
        compute_dtypes.append(_CONFIG_DTYPE_NAMES[dtype_name])
    if len(set(compute_dtypes)) != 1:
        raise ValueError(
            "model config dtype and torch_dtype disagree: "
            f"{compute_dtypes}")
    compute_dtype = compute_dtypes[0]
    if "quantization_config" in cfg:
        quantization = cfg["quantization_config"]
        if model_type == "glm_moe_dsa":
            if not isinstance(quantization, dict):
                raise TypeError("GLM quantization_config must be a mapping")
            if (set(quantization) != {
                    "activation_scheme", "fmt", "quant_method",
                    "weight_block_size", "modules_to_not_convert"}
                    or quantization["quant_method"] != "fp8"
                    or quantization["fmt"] != "e4m3"
                    or quantization["activation_scheme"] != "dynamic"
                    or quantization["weight_block_size"] != [128, 128]):
                raise ValueError("official GLM requires dynamic 128x128 e4m3 FP8")
            ignored = quantization["modules_to_not_convert"]
            if (not isinstance(ignored, list) or not ignored
                    or any(not isinstance(name, str) or not name for name in ignored)
                    or len(set(ignored)) != len(ignored)):
                raise ValueError("GLM FP8 exclusions must be unique non-empty module names")
            quantized_weight_targets = ("Linear",)
            quantized_weight_ignore = tuple(ignored)
            weight_dtype = "fp8"
            weight_group_size = 0
        elif model_type == "deepseek_v3":
            if not isinstance(quantization, dict):
                raise TypeError("model quantization_config must be a mapping")
            if set(quantization) != {
                    "activation_scheme", "fmt", "quant_method",
                    "weight_block_size"}:
                raise ValueError(
                    "DeepSeek-V3 requires the official FP8 quantization "
                    "schema fields")
            if (quantization["quant_method"] != "fp8"
                    or quantization["fmt"] != "e4m3"
                    or quantization["activation_scheme"] != "dynamic"
                    or quantization["weight_block_size"] != [128, 128]):
                raise ValueError(
                    "DeepSeek-V3 requires the official block-wise 128x128 "
                    "e4m3 FP8 schema with dynamic activations")
            # The official DeepSeek-V3 release stores weights as e4m3 with
            # 128x128 block scales.  The Serving FP8 contract models weight
            # storage as one byte per element and has no scale-granularity
            # dimension anywhere downstream (FUSED_MOE requires
            # weight_group_size=0 for fp8; residency counts element bytes),
            # so block-wise and per-tensor scaling are the same model here.
            weight_dtype = "fp8"
            weight_group_size = 0
            quantized_weight_targets = ()
            quantized_weight_ignore = ()
        elif model_type == "deepseek_v4":
            if not isinstance(quantization, dict):
                raise TypeError("model quantization_config must be a mapping")
            if set(quantization) != {
                    "activation_scheme", "fmt", "quant_method", "scale_fmt",
                    "weight_block_size"}:
                raise ValueError(
                    "DeepSeek-V4 requires the DeepseekV4FP8Config "
                    "quantization schema fields")
            if (quantization["quant_method"] != "fp8"
                    or quantization["fmt"] != "e4m3"
                    or quantization["activation_scheme"] != "dynamic"
                    or quantization["scale_fmt"] != "ue8m0"
                    or quantization["weight_block_size"] != [128, 128]):
                raise ValueError(
                    "DeepSeek-V4 requires the block-wise 128x128 e4m3 FP8 "
                    "schema with ue8m0 scales and dynamic activations for "
                    "non-expert linears")
            # Non-expert linears are block-wise e4m3 FP8 exactly like
            # DeepSeek-V3, so the same one-byte weight-storage contract
            # applies.  The MXFP4 routed-expert storage is recorded
            # separately in the validated top-level ``expert_dtype`` field;
            # expert residency modeling is not part of this contract yet.
            weight_dtype = "fp8"
            weight_group_size = 0
            quantized_weight_targets = ()
            quantized_weight_ignore = ()
        elif model_type == "kimi_linear":
            if (not isinstance(quantization, dict)
                    or quantization["quant_method"] != "compressed-tensors"
                    or quantization["quantization_status"] != "compressed"
                    or quantization["format"] != "mxfp4-pack-quantized"
                    or set(quantization["config_groups"]) != {"group_0"}):
                raise ValueError("K3 requires the official compressed-tensors MXFP4 format")
            group = quantization["config_groups"]["group_0"]
            weights = group["weights"]
            if (group["format"] != "mxfp4-pack-quantized"
                    or group["targets"] != ["Linear"]
                    or group["input_activations"] is not None or group["output_activations"] is not None
                    or weights["type"] != "float" or weights["num_bits"] != 4
                    or weights["group_size"] != 32 or weights["strategy"] != "group"
                    or weights["scale_dtype"] != "torch.uint8" or weights["symmetric"] is not True
                    or weights["dynamic"] is not False):
                raise ValueError("K3 requires static group32 floating MXFP4 with uint8 scales, not INT4")
            ignored = quantization["ignore"]
            if not isinstance(ignored, list) or not ignored or any(
                    not isinstance(rule, str) or not rule.startswith("re:") for rule in ignored):
                raise ValueError("K3 MXFP4 requires explicit regex exclusions")
            weight_dtype, weight_group_size = "mxfp4", 32
            quantized_weight_targets, quantized_weight_ignore = ("Linear",), tuple(ignored)
        else:
            if model_type != "kimi_k2":
                raise ValueError(
                    f"model_type {model_type!r} does not support a quantization "
                    "config in the Serving model contract")
            if not isinstance(quantization, dict):
                raise TypeError("model quantization_config must be a mapping")
            if (quantization["quant_method"] != "compressed-tensors"
                    or quantization["quantization_status"] != "compressed"
                    or quantization["format"] != "pack-quantized"):
                raise ValueError(
                    "Kimi-K2.6 requires its fixed compressed-tensors weight format")
            config_groups = quantization["config_groups"]
            if not isinstance(config_groups, dict) or set(config_groups) != {"group_0"}:
                raise ValueError("Kimi-K2.6 quantization requires exactly config group_0")
            group = config_groups["group_0"]
            if not isinstance(group, dict):
                raise TypeError("quantization group_0 must be a mapping")
            raw_targets = group["targets"]
            if (not isinstance(raw_targets, list)
                    or any(not isinstance(value, str) or not value for value in raw_targets)):
                raise TypeError("quantization targets must be a list of non-empty strings")
            weights = group["weights"]
            if not isinstance(weights, dict):
                raise TypeError("quantization group weights must be a mapping")
            if (weights["type"] != "int" or weights["num_bits"] != 4
                    or weights["strategy"] != "group" or weights["symmetric"] is not True):
                raise ValueError("Kimi-K2.6 requires symmetric groupwise INT4 weights")
            weight_group_size = weights["group_size"]
            if (isinstance(weight_group_size, bool)
                    or not isinstance(weight_group_size, int) or weight_group_size <= 0):
                raise ValueError("Kimi-K2.6 INT4 weight group size must be positive")
            raw_ignore = quantization["ignore"]
            if (not isinstance(raw_ignore, list)
                    or any(not isinstance(value, str) or not value for value in raw_ignore)):
                raise TypeError("quantization ignore rules must be a list of non-empty strings")
            weight_dtype = "mixed_int4_bf16"
            quantized_weight_targets = tuple(raw_targets)
            quantized_weight_ignore = tuple(raw_ignore)
    else:
        weight_dtype = compute_dtype
        weight_group_size = 0
        quantized_weight_targets = ()
        quantized_weight_ignore = ()

    # K3 projections explicitly construct bias=False in the pinned source;
    # the raw KimiLinearConfig carries no attention_bias field.
    attention_bias_field = False if model_type == "kimi_linear" else cfg["attention_bias"]
    if not isinstance(attention_bias_field, bool):
        raise ValueError("model config attention_bias must be boolean")
    if "tie_word_embeddings" not in cfg:
        raise ValueError(
            "model config requires tie_word_embeddings")
    tie_word_embeddings = cfg["tie_word_embeddings"]
    if not isinstance(tie_word_embeddings, bool):
        raise ValueError(
            "model config tie_word_embeddings must be boolean")
    mlp_bias_field = schema["mlp_bias"]
    if "mlp_bias" in cfg and cfg["mlp_bias"] != mlp_bias_field:
        raise ValueError(
            f"model config mlp_bias {cfg['mlp_bias']!r} disagrees with "
            f"model_type {model_type!r} schema {mlp_bias_field!r}")

    dense_layer_policy = schema["dense_layer_policy"]
    if model_type == "qwen3_moe":
        if "norm_topk_prob" not in cfg or type(cfg["norm_topk_prob"]) is not bool:
            raise ValueError("Qwen3MoE requires explicit boolean norm_topk_prob")
        if ("decoder_sparse_step" not in cfg
                or cfg["decoder_sparse_step"] != 1):
            raise ValueError(
                "Qwen3MoE requires decoder_sparse_step=1 for the canonical "
                "all-MoE layer contract")
        if ("mlp_only_layers" not in cfg
                or cfg["mlp_only_layers"] != []):
            raise ValueError(
                "Qwen3MoE requires empty mlp_only_layers for the canonical "
                "all-MoE layer contract")
    if dense_layer_policy == "all":
        num_dense_layers = cfg["num_hidden_layers"]
    elif dense_layer_policy == "none":
        num_dense_layers = 0
    elif dense_layer_policy == "first_k_dense_replace":
        num_dense_layers = cfg["first_k_dense_replace"]
    else:
        raise AssertionError(
            f"invalid dense layer policy for {model_type!r}")
    if model_type == "glm_moe_dsa":
        raw_mlp_layer_types = cfg["mlp_layer_types"]
        if not isinstance(raw_mlp_layer_types, list):
            raise TypeError("GLM-5.2 mlp_layer_types must be a list")
        expected_mlp_layer_types = (
            ["dense"] * num_dense_layers
            + ["sparse"] * (
                cfg["num_hidden_layers"] - num_dense_layers)
        )
        if raw_mlp_layer_types != expected_mlp_layer_types:
            raise ValueError(
                "GLM-5.2 mlp_layer_types disagree with the dense/MoE "
                "boundary")
    mtp_architecture = schema["mtp_architecture"]
    if mtp_architecture == "none":
        if "num_nextn_predict_layers" in cfg:
            raise ValueError(
                f"model_type {model_type!r} does not support MTP fields")
        num_mtp_layers = 0
    elif mtp_architecture == "none_zero_allowed":
        num_mtp_layers = cfg["num_nextn_predict_layers"]
        if num_mtp_layers != 0:
            raise ValueError(
                f"model_type {model_type!r} requires zero MTP layers")
        mtp_architecture = "none"
    elif mtp_architecture in (
            "glm4_moe_mtp", "glm_moe_dsa_mtp", "deepseek_mtp"):
        num_mtp_layers = cfg["num_nextn_predict_layers"]
        if num_mtp_layers == 0:
            mtp_architecture = "none"
    else:
        raise AssertionError(f"invalid MTP schema for {model_type!r}")
    qk_norm_policy = schema["qk_norm"]
    if qk_norm_policy == "config":
        qk_norm = cfg["use_qk_norm"]
        if not isinstance(qk_norm, bool):
            raise TypeError("model config use_qk_norm must be boolean")
    elif isinstance(qk_norm_policy, bool):
        qk_norm = qk_norm_policy
        if ("use_qk_norm" in cfg
                and cfg["use_qk_norm"] is not qk_norm_policy):
            raise ValueError(
                f"model config use_qk_norm disagrees with {model_type!r} "
                "architecture semantics")
    else:
        raise AssertionError(f"invalid Q/K norm schema for {model_type!r}")

    if model_type in ("glm4_moe", "glm_moe_dsa", "kimi_k2", "deepseek_v3",
             "deepseek_v4", "kimi_linear"):
        if "routed_scaling_factor" in cfg:
            routed_scaling_factor = cfg["routed_scaling_factor"]
        elif model_type == "glm4_moe":
            # transformers Glm4MoeConfig source default. Normalize it here
            # because this loader deliberately does not instantiate HF config.
            routed_scaling_factor = 1.0
        else:
            raise ValueError(
                f"model_type {model_type!r} requires routed_scaling_factor")
        if (isinstance(routed_scaling_factor, bool)
                or not isinstance(routed_scaling_factor, (int, float))
                or not math.isfinite(routed_scaling_factor)
                or routed_scaling_factor <= 0):
            raise ValueError(
                "model config routed_scaling_factor must be finite and "
                "positive")
    elif model_type in ("qwen3", "qwen3_moe", "llama"):
        if "routed_scaling_factor" in cfg:
            raise ValueError(
                f"model_type {model_type!r} does not support "
                "routed_scaling_factor")
        routed_scaling_factor = 1.0
    else:
        raise AssertionError(
            f"unexpected routed-scaling model_type: {model_type!r}")

    if model_type == "glm_moe_dsa":
        if cfg["moe_router_dtype"] != "float32":
            raise ValueError("GLM-5.2 requires float32 MoE routing")
        rope_parameters = cfg["rope_parameters"]
        if not isinstance(rope_parameters, dict):
            raise TypeError("GLM-5.2 rope_parameters must be a mapping")
        expected_rope_fields = {"rope_theta", "rope_type"}
        if set(rope_parameters) != expected_rope_fields:
            raise ValueError(
                "GLM-5.2 rope_parameters fields must be exactly "
                f"{sorted(expected_rope_fields)}, got "
                f"{sorted(rope_parameters)}")
        raw_index_pattern = cfg["index_topk_pattern"]
        if raw_index_pattern is None:
            index_topk_pattern = None
        elif isinstance(raw_index_pattern, list):
            if len(raw_index_pattern) != cfg["num_hidden_layers"]:
                raise ValueError(
                    "GLM-5.2 index_topk_pattern must cover every layer")
            index_topk_pattern = tuple(raw_index_pattern)
        else:
            raise TypeError(
                "GLM-5.2 index_topk_pattern must be a list or null")
        index_skip_topk_offset = cfg["index_skip_topk_offset"]
        index_topk_frequency = cfg["index_topk_freq"]
        for field, value in (
                ("index_skip_topk_offset", index_skip_topk_offset),
                ("index_topk_freq", index_topk_frequency)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"GLM-5.2 {field} must be an integer")
            if value <= 0:
                raise ValueError(f"GLM-5.2 {field} must be positive")
        if index_topk_pattern is None:
            expected_indexer_type_values = []
            for layer_index in range(cfg["num_hidden_layers"]):
                skipped_count = max(
                    layer_index - index_skip_topk_offset + 1, 0)
                if skipped_count % index_topk_frequency:
                    expected_indexer_type_values.append("shared")
                else:
                    expected_indexer_type_values.append("full")
            expected_indexer_types = tuple(expected_indexer_type_values)
        else:
            expected_indexer_type_values = []
            for value in index_topk_pattern:
                if value == "S":
                    expected_indexer_type_values.append("shared")
                elif value == "F":
                    expected_indexer_type_values.append("full")
                else:
                    raise ValueError(
                        "GLM-5.2 index_topk_pattern values must be S or F, "
                        f"got {value!r}")
            expected_indexer_types = tuple(expected_indexer_type_values)
        if indexer_types != expected_indexer_types:
            raise ValueError(
                "GLM-5.2 indexer_types disagree with vLLM's pinned "
                "index skip rule")
        if cfg["rope_interleave"] is not True:
            raise ValueError("GLM-5.2 requires interleaved main RoPE")
        if cfg["indexer_rope_interleave"] is not True:
            raise ValueError("GLM-5.2 requires interleaved indexer RoPE")
        target_semantics = TargetModelSemantics(
            workload_contract=_named_target_workload_contract(model_type, config_sha256),
            rms_norm_eps=cfg["rms_norm_eps"],
            use_cache=cfg["use_cache"],
            moe_layer_frequency=cfg["moe_layer_freq"],
            moe_router_dtype="fp32",
            moe_score_function=cfg["scoring_func"],
            moe_topk_method=cfg["topk_method"],
            moe_group_count=cfg["n_group"],
            moe_topk_group_count=cfg["topk_group"],
            moe_normalize_topk_probability=cfg["norm_topk_prob"],
            rope_type=rope_parameters["rope_type"],
            rope_theta=rope_parameters["rope_theta"],
            rope_layout="interleaved",
            rope_factor=None,
            rope_original_max_position_embeddings=None,
            rope_beta_fast=None,
            rope_beta_slow=None,
            rope_mscale=None,
            rope_mscale_all_dim=None,
            index_skip_topk_offset=index_skip_topk_offset,
            index_topk_frequency=index_topk_frequency,
            index_topk_pattern=index_topk_pattern,
            indexer_rope_layout="interleaved",
            mtp_runtime_status=(
                "vllm_0_30_registered_deepseek_v32_mtp_source_only_runtime_unbound"
                if config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S else
                "main_model_skips_spec_layer_weights_no_registered_"
                "glm_moe_dsa_mtp_model"),
        )
    elif model_type == "kimi_k2":
        rope_scaling = cfg["rope_scaling"]
        if not isinstance(rope_scaling, dict):
            raise TypeError("Kimi-K2.6 rope_scaling must be a mapping")
        expected_rope_fields = {
            "beta_fast", "beta_slow", "factor", "mscale",
            "mscale_all_dim", "original_max_position_embeddings", "type",
        }
        if set(rope_scaling) != expected_rope_fields:
            raise ValueError(
                "Kimi-K2.6 rope_scaling fields must be exactly "
                f"{sorted(expected_rope_fields)}, got "
                f"{sorted(rope_scaling)}")
        if rope_scaling["type"] != "yarn":
            raise ValueError("Kimi-K2.6 requires YaRN rope scaling")
        target_semantics = TargetModelSemantics(
            workload_contract=_named_target_workload_contract(model_type, config_sha256),
            rms_norm_eps=cfg["rms_norm_eps"],
            use_cache=cfg["use_cache"],
            moe_layer_frequency=cfg["moe_layer_freq"],
            moe_router_dtype=compute_dtype,
            moe_score_function=cfg["scoring_func"],
            moe_topk_method=cfg["topk_method"],
            moe_group_count=cfg["n_group"],
            moe_topk_group_count=cfg["topk_group"],
            moe_normalize_topk_probability=cfg["norm_topk_prob"],
            rope_type="deepseek_yarn",
            rope_theta=cfg["rope_theta"],
            rope_layout="interleaved",
            rope_factor=rope_scaling["factor"],
            rope_original_max_position_embeddings=(
                rope_scaling["original_max_position_embeddings"]),
            rope_beta_fast=rope_scaling["beta_fast"],
            rope_beta_slow=rope_scaling["beta_slow"],
            rope_mscale=rope_scaling["mscale"],
            rope_mscale_all_dim=rope_scaling["mscale_all_dim"],
            index_skip_topk_offset=None,
            index_topk_frequency=None,
            index_topk_pattern=None,
            indexer_rope_layout=None,
            mtp_runtime_status="not_configured",
        )
    elif model_type == "deepseek_v3":
        rope_scaling = cfg["rope_scaling"]
        if not isinstance(rope_scaling, dict):
            raise TypeError("DeepSeek-V3 rope_scaling must be a mapping")
        expected_rope_fields = {
            "beta_fast", "beta_slow", "factor", "mscale",
            "mscale_all_dim", "original_max_position_embeddings", "type",
        }
        if set(rope_scaling) != expected_rope_fields:
            raise ValueError(
                "DeepSeek-V3 rope_scaling fields must be exactly "
                f"{sorted(expected_rope_fields)}, got "
                f"{sorted(rope_scaling)}")
        if rope_scaling["type"] != "yarn":
            raise ValueError("DeepSeek-V3 requires YaRN rope scaling")
        target_semantics = TargetModelSemantics(
            workload_contract=_named_target_workload_contract(model_type, config_sha256),
            rms_norm_eps=cfg["rms_norm_eps"],
            use_cache=cfg["use_cache"],
            moe_layer_frequency=cfg["moe_layer_freq"],
            moe_router_dtype=compute_dtype,
            moe_score_function=cfg["scoring_func"],
            moe_topk_method=cfg["topk_method"],
            moe_group_count=cfg["n_group"],
            moe_topk_group_count=cfg["topk_group"],
            moe_normalize_topk_probability=cfg["norm_topk_prob"],
            rope_type="deepseek_yarn",
            rope_theta=cfg["rope_theta"],
            rope_layout="interleaved",
            rope_factor=rope_scaling["factor"],
            rope_original_max_position_embeddings=(
                rope_scaling["original_max_position_embeddings"]),
            rope_beta_fast=rope_scaling["beta_fast"],
            rope_beta_slow=rope_scaling["beta_slow"],
            rope_mscale=rope_scaling["mscale"],
            rope_mscale_all_dim=rope_scaling["mscale_all_dim"],
            index_skip_topk_offset=None,
            index_topk_frequency=None,
            index_topk_pattern=None,
            indexer_rope_layout=None,
            mtp_runtime_status=(
                "vllm_registered_deepseek_mtp_scheduler_only_draft_contract"),
        )
    elif model_type == "deepseek_v4":
        # Rope semantics (vLLM 0.30.0 deepseek_v4/common/rope.py
        # build_deepseek_v4_rope): every layer instantiates the
        # deepseek_yarn rope class with mscale hard-disabled.  Pure-SWA
        # layers (config compress ratio 0) run it with factor 1.0 on
        # rope_theta -- i.e. the plain rotary -- while compressor layers run
        # it on compress_rope_theta with the config YaRN scaling below.
        rope_scaling = cfg["rope_scaling"]
        if not isinstance(rope_scaling, dict):
            raise TypeError("DeepSeek-V4 rope_scaling must be a mapping")
        expected_rope_fields = {
            "beta_fast", "beta_slow", "factor",
            "original_max_position_embeddings", "type",
        }
        if set(rope_scaling) != expected_rope_fields:
            raise ValueError(
                "DeepSeek-V4 rope_scaling fields must be exactly "
                f"{sorted(expected_rope_fields)}, got "
                f"{sorted(rope_scaling)}")
        if rope_scaling["type"] != "yarn":
            raise ValueError("DeepSeek-V4 requires YaRN rope scaling")
        # The campaign configs carry no moe_layer_freq (every decoder layer
        # is routed MoE) and no n_group/topk_group (the V4 router is global
        # noaux_tc -- hash routing on the first num_hash_layers, Sinkhorn
        # hyper-connection routing afterwards -- with no group-limited
        # candidate set), so the group counts normalize to the no-op 1/1.
        target_semantics = TargetModelSemantics(
            workload_contract=_named_target_workload_contract(model_type, config_sha256),
            rms_norm_eps=cfg["rms_norm_eps"],
            use_cache=cfg["use_cache"],
            moe_layer_frequency=1,
            moe_router_dtype=compute_dtype,
            moe_score_function=cfg["scoring_func"],
            moe_topk_method=cfg["topk_method"],
            moe_group_count=1,
            moe_topk_group_count=1,
            moe_normalize_topk_probability=cfg["norm_topk_prob"],
            rope_type="dsv4_yarn",
            rope_theta=cfg["rope_theta"],
            rope_layout="interleaved",
            rope_factor=rope_scaling["factor"],
            rope_original_max_position_embeddings=(
                rope_scaling["original_max_position_embeddings"]),
            rope_beta_fast=rope_scaling["beta_fast"],
            rope_beta_slow=rope_scaling["beta_slow"],
            rope_mscale=None,
            rope_mscale_all_dim=None,
            index_skip_topk_offset=None,
            index_topk_frequency=None,
            index_topk_pattern=None,
            indexer_rope_layout=None,
            mtp_runtime_status=(
                "unsupported_deepseek_v4_dspark_speculator_runtime_not_modeled" if dspark
                else "unsupported_deepseek_v4_mtp_runtime_not_modeled"),
        )
    elif model_type in ("qwen3", "qwen3_moe", "llama", "glm4_moe", "kimi_linear"):
        target_semantics = None
    else:
        raise AssertionError(
            f"unhandled target-semantics model_type: {model_type!r}")

    glm4_rotary_dim = None
    glm4_group_count = None
    glm4_topk_groups = None
    if model_type == "glm4_moe":
        factor = cfg["partial_rotary_factor"] if "partial_rotary_factor" in cfg else 0.5
        if type(factor) not in (int, float) or not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("GLM4 partial_rotary_factor must be finite and in (0,1]")
        glm4_rotary_dim = int(head_dim * factor)
        glm4_group_count = cfg["n_group"] if "n_group" in cfg else 1
        glm4_topk_groups = cfg["topk_group"] if "topk_group" in cfg else 1
        if type(glm4_group_count) is not int or type(glm4_topk_groups) is not int:
            raise ValueError("GLM4 expert group counts must be integers")
        if "norm_topk_prob" in cfg and type(cfg["norm_topk_prob"]) is not bool:
            raise ValueError("GLM4 norm_topk_prob must be boolean")

    return ModelConfig(
        hidden_size=hidden_size,
        # DeepSeek-V4 has no dense FFN and no intermediate_size key; every
        # other model_type must carry the field.
        intermediate_size=(
            0 if model_type == "deepseek_v4" else cfg["intermediate_size"]),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=cfg["num_hidden_layers"],
        head_dim=head_dim,
        vocab_size=cfg["vocab_size"],
        compute_dtype=compute_dtype,
        weight_dtype=weight_dtype,
        weight_group_size=weight_group_size,
        quantized_weight_targets=quantized_weight_targets,
        quantized_weight_ignore=quantized_weight_ignore,
        attention_layout=attention_layout,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        indexer_types=indexer_types,
        index_share_for_mtp_iteration=index_share_for_mtp_iteration,
        compress_ratios=compress_ratios,
        compress_rope_theta=compress_rope_theta,
        window_size=window_size,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        num_hash_layers=num_hash_layers,
        swiglu_limit=swiglu_limit,
        expert_dtype=expert_dtype,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        architecture=architecture,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=attention_bias_field,
        mlp_bias=mlp_bias_field,
        max_position_embeddings=cfg["max_position_embeddings"],
        model_type=model_type,
        num_dense_layers=num_dense_layers,
        num_shared_experts=num_shared_experts,
        num_mtp_layers=num_mtp_layers,
        mtp_architecture=mtp_architecture,
        qk_norm=qk_norm,
        # vLLM's build_deepseek_v4_rope overrides the config's "yarn" type
        # and always instantiates the deepseek_yarn rope class (factor 1.0
        # on SWA layers, config scaling on compressor layers), so the
        # physical rope program for DeepSeek-V4 is deepseek_yarn.
        rope_scaling_type=(
            "deepseek_yarn" if model_type == "deepseek_v4"
            else _rope_scaling_type(cfg)),
        routed_scaling_factor=float(routed_scaling_factor),
        target_semantics=target_semantics,
        config_sha256=config_sha256,
        source_config_sha256=source_config_sha256,
        canonical_config_sha256=canonical_config_sha256,
        config_canonicalization=_CONFIG_CANONICALIZATION,
        config_extraction_path=config_extraction_path,
        extracted_config_sha256=extracted_config_sha256,
        model_id=model_id,
        model_revision=model_revision,
        model_source_url=model_source_url,
        model_scope=model_scope,
        kimi_k3_semantics=_kimi_k3_semantics(cfg) if model_type == "kimi_linear" else None,
        rms_norm_eps=(cfg["rms_norm_eps"] if "rms_norm_eps" in cfg
            else schema["rms_norm_eps"] if model_type in ("qwen3", "qwen3_moe", "glm4_moe") else None),
        hidden_act=(cfg["hidden_act"] if "hidden_act" in cfg
            else schema["hidden_act"] if model_type in ("qwen3", "qwen3_moe", "glm4_moe") else None),
        moe_normalize_topk_probability=(cfg["norm_topk_prob"] if "norm_topk_prob" in cfg else True)
            if model_type in ("qwen3_moe", "glm4_moe") else None,
        rotary_dim=glm4_rotary_dim,
        moe_group_count=glm4_group_count,
        moe_topk_group_count=glm4_topk_groups,
        dspark_block_size=cfg["dspark_block_size"] if dspark else None,
        dspark_noise_token_id=cfg["dspark_noise_token_id"] if dspark else None,
        dspark_target_layer_ids=tuple(cfg["dspark_target_layer_ids"]) if dspark else None,
        dspark_markov_rank=cfg["dspark_markov_rank"] if dspark else None,
    )


def tp_collective_payload_bytes(
    model: ModelConfig,
    operation: str,
    token_count: int,
    *,
    local_output_width: int | None = None,
) -> int:
    """Return canonical per-rank TP collective input bytes."""

    if not isinstance(model, ModelConfig):
        raise TypeError("TP collective bytes require ModelConfig")
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise TypeError("TP collective token_count must be an integer")
    if token_count <= 0:
        raise ValueError("TP collective token_count must be positive")
    if operation == "all_reduce":
        if local_output_width is not None:
            raise ValueError(
                "TP all-reduce derives output width from model.hidden_size")
        output_width = model.hidden_size
    elif operation == "all_gather":
        if (isinstance(local_output_width, bool)
                or not isinstance(local_output_width, int)
                or local_output_width <= 0):
            raise ValueError(
                "TP all-gather requires a positive local_output_width")
        output_width = local_output_width
    else:
        raise ValueError(
            f"unsupported TP collective operation: {operation!r}")
    return token_count * output_width * model.dtype_bytes


def routed_expert_local_intermediate(
    model: ModelConfig,
    tensor_parallel_size: int,
    expert_parallel_size: int,
) -> int:
    """Return one rank's routed-expert intermediate extent.

    The residency contract (src/serving/kv_cache/model_residency.py) places
    N/EP experts on each rank and tensor-shards every expert linear across
    the TP group, so one rank's GEMM n/k for a single expert spans
    intermediate * EP / TP: the full extent when EP == TP, a TP shard when
    EP == 1.  The forward FUSED_MOE shape must quote the same per-rank
    extent or every rank would be charged the whole unsharded expert weight.
    """

    product = model.moe_intermediate_size * expert_parallel_size
    if product % tensor_parallel_size:
        raise ValueError(
            "routed-expert intermediate extent does not divide across the "
            f"TP/EP deployment: {model.moe_intermediate_size} * "
            f"{expert_parallel_size} % {tensor_parallel_size} != 0")
    return product // tensor_parallel_size
