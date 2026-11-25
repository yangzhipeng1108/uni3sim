"""Shared contract of block-FP8 dense GEMM launches.

Lowering declares ``weight_quantization`` for a dense projection whose model
checkpoint stores 128x128 block-scaled e4m3 weights with dynamic 1x128
activation groups; a hardware launch package that owns a block-FP8 selector
routes the linear to it and emits one of the algorithms below; the GEMM
predictor prices those algorithms with its block-FP8 template.  Only names
and static source facts live here.
"""

BLOCK_FP8_WEIGHT_QUANTIZATION = "fp8_e4m3_block128x128_act1x128_dynamic"

# FlashInfer/TensorRT-LLM ``fp8_gemm_run`` (vLLM 0.24, M < 32): the call
# launches ``scale_1x128_kernel<<<num_sms * 8, 256>>>`` before the swap-AB
# DeepGEMM kernel.
TRTLLM_SWAP_AB_ALGORITHM = "trtllm_fp8_blockscale_swapab_gemm"
TRTLLM_SCALE_1X128_CTAS_PER_SM = 8
TRTLLM_SCALE_1X128_THREADS = 256
# DeepGEMM ``fp8_gemm_nt`` -> ``sm90_fp8_gemm_1d2d_impl`` (vLLM 0.24, M >= 32).
DEEPGEMM_1D2D_ALGORITHM = "deepgemm_sm90_fp8_gemm_1d2d"

# DeepSeek-V4 wo_a FP8 einsum (vLLM 0.30.0 deepseek_v4/nvidia/ops/o_proj.py
# ``deep_gemm_fp8_o_proj``): ``fp8_einsum("bhr,hdr->bhd")`` over the TP-local
# o_groups with the SM90 recipe ``(1, 128, 128)`` (block-row FP32 scales;
# o_proj.py:13-26).  Same DeepGEMM SM90 block-scaled kernel family as the
# dense 1D2D linear, batched over the groups; the fused inverse-RoPE +
# 1x128 activation quantisation is a separate kernel
# (``fused_inv_rope_fp8_quant``), so no in-call scale kernel is charged.
DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM = "deepgemm_sm90_fp8_einsum_batched"

# All these source templates store BF16 C, irrespective of the FP8 operand
# packing. Selector, KernelSpec and predictor share this physical family.
BLOCK_FP8_GEMM_ALGORITHMS = frozenset({
    TRTLLM_SWAP_AB_ALGORITHM, DEEPGEMM_1D2D_ALGORITHM,
    DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM,
})

# vLLM 0.24 ``per_token_group_quant_fp8`` -> ``per_token_group_quant_8bit``
# (csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu): the
# separate dynamic 1x128 activation quantiser in front of the DeepGEMM 1d2d
# linear and of both fused-MoE GEMMs.  One CTA quantises ``groups_per_block``
# groups (the largest of 16/8/4/2/1 dividing the group count), 16 threads per
# group; grid = groups / groups_per_block.
PER_TOKEN_GROUP_QUANT_ALGORITHM = "vllm_per_token_group_quant_8bit"
# vLLM 0.24 FP8 block-128 Triton fused MoE (``fused_experts_impl``): both
# GEMM inputs (the hidden states before W1, the activated intermediate before
# W2) go through ``moe_kernel_quantize_input`` -> ``per_token_group_quant_fp8``.
FP8_BLOCK_FUSED_MOE_ALGORITHM_PREFIX = "vllm_triton_fused_moe_fp8_block128"
# SGLang 0.5.12 invoke_fused_moe_kernel also quantises each GEMM input with
# sglang_per_token_group_quant_fp8 (the per_token_group_quant_8bit kernel).
SGLANG_FP8_BLOCK_FUSED_MOE_ALGORITHM_PREFIX = (
    "sglang_triton_fused_moe_fp8_block128")
PER_TOKEN_GROUP_QUANT_GROUP_SIZE = 128
PER_TOKEN_GROUP_QUANT_THREADS_PER_GROUP = 16


def per_token_group_quant_geometry(element_count: int) -> tuple[int, int]:
    """Return (CTAs, threads per CTA) of one ``per_token_group_quant_8bit``."""

    if (isinstance(element_count, bool) or not isinstance(element_count, int)
            or element_count <= 0
            or element_count % PER_TOKEN_GROUP_QUANT_GROUP_SIZE):
        raise ValueError(
            "per_token_group_quant needs a positive whole number of groups")
    groups = element_count // PER_TOKEN_GROUP_QUANT_GROUP_SIZE
    groups_per_block = next(
        width for width in (16, 8, 4, 2, 1) if groups % width == 0)
    return (groups // groups_per_block,
            groups_per_block * PER_TOKEN_GROUP_QUANT_THREADS_PER_GROUP)
