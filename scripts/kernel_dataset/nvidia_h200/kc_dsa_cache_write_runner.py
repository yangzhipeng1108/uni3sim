"""NVIDIA H200 DeepSeek Sparse Attention (DSA) KV-cache write runner.

Measures the per-layer cache inserts of vLLM 0.24.0's DSA models
(``DeepseekV32ForCausalLM`` / ``GlmMoeDsaForCausalLM``) under the canonical
``nsys_nvtx_kernel_median`` protocol (``H200CaptureSession``).  One public op
call inside the NVTX range:

* ``--op indexer_k``: ``ops.indexer_k_quant_and_cache(k, kv_cache,
  slot_mapping, quant_block_size=128, "ue8m0")`` -- the indexer K FP8
  quantise + insert at the top of ``sparse_attn_indexer`` (K ``[T, 128]``
  BF16, cache ``uint8 [num_blocks, 64, 132]``);
* ``--op mla_fp8_ds_mla``: ``ops.concat_and_cache_mla(kv_c, k_pe, kv_cache,
  slot_mapping, "fp8_ds_mla", scale)`` -- ``MLAAttentionImpl.do_kv_cache_update``
  for the FlashMLA-sparse 656 B/token cache (``uint8 [num_blocks, 64, 656]``);
* ``--op mla_bf16``: the same call with ``kv_cache_dtype="auto"`` into the
  BF16 ``[num_blocks, 64, 576]`` cache.

Slot layout: ``prefill`` writes tokens to consecutive slots of randomly
chosen blocks (a request's new chunk); ``decode`` writes each token into its
own random block at a random in-block offset (one token per request).  The
pool is twice the blocks the case touches so the chosen blocks are scattered.

vLLM 0.30.0 (explicit branch on ``installed_vllm_stack()``, recorded as
``vllm_stack``): the three standalone ops above still exist and run
unchanged, but the DeepSeek-V3.2 / GLM-5.2 CUDA model
(``vllm/models/deepseek_v32/attention.py``, non-PCP, no HiSparse) no longer
calls them: both cache inserts are fused into ``fused_norm_rope``
(``models/deepseek_v32/common/kernels.py``, one Triton
``_fused_norm_rope_kernel`` launch, grid ``(T, 4)``: indexer K LayerNorm +
RoPE + FP8 quant + insert, kv_c RMSNorm + k_pe RoPE + MLA insert, q_c
RMSNorm, top-k buffer ``-1`` fill).  The standalone ops remain the PCP path
(``sparse_attn_indexer(skip_k_cache_insert=False)`` /
``do_kv_cache_update``); each case records ``vllm030_model_path``.  The
0.30-only ``--op fused_norm_rope`` measures that fused call with the
DeepseekV32Attention.forward argument layout (q_c / kv_c / k_pe as column
views of the fused qkv_a output, index_k a view of wk_weights_proj output,
FP32 LayerNorm weights, FP32 ``cos_sin_cache``, one slot mapping for both
caches, ``kv_c_out`` / ``k_pe_out`` allocated, ``index_k_out=None``).
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    add_h200_capture_arguments,
    installed_vllm_stack,
    open_h200_capture_session,
)

import torch
from vllm import _custom_ops as ops

VLLM_STACK = installed_vllm_stack()
if VLLM_STACK == "vllm030":
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope
elif VLLM_STACK != "vllm024":
    raise RuntimeError(f"unhandled vLLM stack {VLLM_STACK!r}")

BLOCK_SIZE = 64
KV_LORA_RANK = 512
ROPE_DIM = 64
FP8_DS_MLA_BYTES = 656
INDEX_HEAD_DIM = 128
QUANT_BLOCK_SIZE = 128
SCALE_FMT = "ue8m0"
WARMUP = 5
FUSED_MAX_POSITIONS = 131072   # cos_sin_cache rows (plan max_model_len)
FUSED_TOPK_BUFFER_ROWS = 16384  # >= every fused case's tokens
RMS_EPS = 1e-5
INDEX_LN_EPS = 1e-6
VLLM030_MODEL_PATH = (
    "vLLM 0.30 DeepseekV32Attention (CUDA, non-PCP, no HiSparse) fuses this "
    "insert into fused_norm_rope (_fused_norm_rope_kernel); the standalone "
    "op runs only on the PCP path")


def _slot_mapping(tokens, layout, device):
    if layout == "prefill":
        used_blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        pool_blocks = 2 * used_blocks
        blocks = torch.randperm(pool_blocks, device=device)[:used_blocks]
        position = torch.arange(tokens, device=device, dtype=torch.int64)
        slots = blocks[position // BLOCK_SIZE] * BLOCK_SIZE + (
            position % BLOCK_SIZE)
    elif layout == "decode":
        pool_blocks = 2 * tokens
        blocks = torch.randperm(pool_blocks, device=device)[:tokens]
        offsets = torch.randint(0, BLOCK_SIZE, (tokens,), device=device)
        slots = blocks * BLOCK_SIZE + offsets
    else:
        raise ValueError(f"unknown slot layout {layout!r}")
    return slots.to(torch.int64).contiguous(), pool_blocks


def build_case(op, tokens, layout, device="cuda"):
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("tokens must be a positive integer")
    slot_mapping, pool_blocks = _slot_mapping(tokens, layout, device)
    if op == "indexer_k":
        width = INDEX_HEAD_DIM + INDEX_HEAD_DIM // QUANT_BLOCK_SIZE * 4
        kv_cache = torch.zeros(pool_blocks, BLOCK_SIZE, width,
                               dtype=torch.uint8, device=device)
        k = torch.randn(tokens, INDEX_HEAD_DIM, dtype=torch.bfloat16,
                        device=device)

        def write():
            ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping,
                                          QUANT_BLOCK_SIZE, SCALE_FMT)

        call = ("ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, "
                "128, 'ue8m0')")
    elif op in ("mla_fp8_ds_mla", "mla_bf16"):
        if op == "mla_fp8_ds_mla":
            kv_cache = torch.zeros(pool_blocks, BLOCK_SIZE, FP8_DS_MLA_BYTES,
                                   dtype=torch.uint8, device=device)
            cache_dtype = "fp8_ds_mla"
        else:
            kv_cache = torch.zeros(pool_blocks, BLOCK_SIZE,
                                   KV_LORA_RANK + ROPE_DIM,
                                   dtype=torch.bfloat16, device=device)
            cache_dtype = "auto"
        kv_c = torch.randn(tokens, KV_LORA_RANK, dtype=torch.bfloat16,
                           device=device)
        # MLAAttention passes k_pe.squeeze(1) of a [T, 1, 64] tensor.
        k_pe = torch.randn(tokens, 1, ROPE_DIM, dtype=torch.bfloat16,
                           device=device).squeeze(1)
        scale = torch.ones(1, dtype=torch.float32, device=device)

        def write():
            ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                                     cache_dtype, scale)

        call = (f"ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, "
                f"slot_mapping, {cache_dtype!r}, scale)")
    else:
        raise ValueError(f"unknown op {op!r}")
    if op == "indexer_k":
        layout_text = ("uint8 [num_blocks, 64, 132]: 128 fp8_e4m3 + one fp32 "
                       "scale per 128 elements (ue8m0)")
    elif op == "mla_fp8_ds_mla":
        layout_text = ("uint8 [num_blocks, 64, 656]: 512 fp8_e4m3 NoPE + 4 "
                       "fp32 scales + 64 bf16 RoPE")
    else:
        layout_text = "bf16 [num_blocks, 64, 576]: 512 NoPE + 64 RoPE"
    evidence = {"op": op, "tokens": tokens, "slot_layout": layout,
                "kv_cache_layout": layout_text,
                "slot_mapping": slot_mapping.cpu().tolist(),
                "pool_blocks": pool_blocks, "block_size": BLOCK_SIZE,
                "kv_cache_shape": list(kv_cache.shape),
                "kv_cache_torch_dtype": str(kv_cache.dtype), "call": call,
                "index_head_dim": INDEX_HEAD_DIM,
                "quant_block_size": QUANT_BLOCK_SIZE, "scale_fmt": SCALE_FMT,
                "kv_lora_rank": KV_LORA_RANK, "qk_rope_head_dim": ROPE_DIM}
    if len(evidence["slot_mapping"]) != tokens or len(
            set(evidence["slot_mapping"])) != tokens:
        raise RuntimeError("slot mapping is not one distinct slot per token")
    return write, kv_cache, evidence


def build_fused_case(tokens, layout, kv_cache_dtype, q_lora_rank,
                     has_indexer, index_n_heads, index_rope_interleave,
                     topk, device="cuda"):
    """vLLM 0.30 fused_norm_rope as DeepseekV32Attention.forward calls it."""
    if VLLM_STACK != "vllm030":
        raise ValueError("fused_norm_rope exists only on the vLLM 0.30 stack")
    for name, value in (("tokens", tokens), ("q_lora_rank", q_lora_rank),
                        ("index_n_heads", index_n_heads), ("topk", topk)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if tokens > FUSED_TOPK_BUFFER_ROWS:
        raise ValueError("tokens exceed the top-k buffer rows")
    slot_mapping, pool_blocks = _slot_mapping(tokens, layout, device)
    bf16 = torch.bfloat16
    # fused_qkv_a_proj output [T, q_lora + 512 + 64] split into views.
    qkv_lora = torch.randn(tokens, q_lora_rank + KV_LORA_RANK + ROPE_DIM,
                           dtype=bf16, device=device)
    q_c, kv_c, k_pe = qkv_lora.split([q_lora_rank, KV_LORA_RANK, ROPE_DIM],
                                     dim=-1)
    # wk_weights_proj output [T, 128 + index_n_heads]; index_k is a view.
    kw = torch.randn(tokens, INDEX_HEAD_DIM + index_n_heads, dtype=bf16,
                     device=device)
    index_k = kw[:, :INDEX_HEAD_DIM]
    q_norm_w = torch.randn(q_lora_rank, dtype=bf16, device=device)
    kv_norm_w = torch.randn(KV_LORA_RANK, dtype=bf16, device=device)
    k_norm_w = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32, device=device)
    k_norm_b = torch.randn(INDEX_HEAD_DIM, dtype=torch.float32, device=device)
    rope_cache = torch.randn(FUSED_MAX_POSITIONS, ROPE_DIM,
                             dtype=torch.float32, device=device)
    index_rope_cache = torch.randn(FUSED_MAX_POSITIONS, ROPE_DIM,
                                   dtype=torch.float32, device=device)
    if layout == "prefill":
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
    else:
        positions = torch.randint(0, FUSED_MAX_POSITIONS, (tokens,),
                                  dtype=torch.int64, device=device)
    topk_buffer = torch.zeros(FUSED_TOPK_BUFFER_ROWS, topk, dtype=torch.int32,
                              device=device)
    if kv_cache_dtype == "fp8_ds_mla":
        mla_cache = torch.zeros(pool_blocks, BLOCK_SIZE, FP8_DS_MLA_BYTES,
                                dtype=torch.uint8, device=device)
    elif kv_cache_dtype == "bf16":
        mla_cache = torch.zeros(pool_blocks, BLOCK_SIZE,
                                KV_LORA_RANK + ROPE_DIM, dtype=bf16,
                                device=device)
    else:
        raise ValueError(f"unknown kv cache dtype {kv_cache_dtype!r}")
    mla_cache_dtype = "fp8_ds_mla" if kv_cache_dtype == "fp8_ds_mla" else "auto"
    index_width = INDEX_HEAD_DIM + INDEX_HEAD_DIM // QUANT_BLOCK_SIZE * 4
    indexer_cache = (torch.zeros(pool_blocks, BLOCK_SIZE, index_width,
                                 dtype=torch.uint8, device=device)
                     if has_indexer else None)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    kv_c_out = torch.empty_like(kv_c)
    k_pe_out = torch.empty_like(k_pe)

    def write():
        return fused_norm_rope(
            positions, q_c, q_norm_w, RMS_EPS, kv_c, kv_norm_w, RMS_EPS,
            k_pe, rope_cache,
            index_k if has_indexer else None,
            k_norm_w if has_indexer else None,
            k_norm_b if has_indexer else None, INDEX_LN_EPS,
            index_rope_cache if has_indexer else None, topk_buffer,
            slot_mapping=slot_mapping,
            indexer_slot_mapping=slot_mapping if has_indexer else None,
            indexer_k_cache=indexer_cache, mla_kv_cache=mla_cache,
            mla_kv_cache_dtype=mla_cache_dtype, mla_k_scale=k_scale,
            has_indexer=has_indexer,
            index_rope_interleave=index_rope_interleave,
            kv_c_out=kv_c_out, k_pe_out=k_pe_out, index_k_out=None)

    evidence = {
        "op": "fused_norm_rope", "tokens": tokens, "slot_layout": layout,
        "slot_mapping": slot_mapping.cpu().tolist(),
        "pool_blocks": pool_blocks, "block_size": BLOCK_SIZE,
        "kv_cache_dtype": kv_cache_dtype,
        "mla_kv_cache_shape": list(mla_cache.shape),
        "mla_kv_cache_torch_dtype": str(mla_cache.dtype),
        "indexer_k_cache_shape": (list(indexer_cache.shape)
                                  if has_indexer else None),
        "has_indexer": has_indexer,
        "index_rope_interleave": index_rope_interleave,
        "q_lora_rank": q_lora_rank, "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": ROPE_DIM, "index_head_dim": INDEX_HEAD_DIM,
        "index_n_heads": index_n_heads, "topk": topk,
        "topk_buffer_shape": list(topk_buffer.shape),
        "input_layout": {
            "qkv_lora": list(qkv_lora.shape), "q_c_stride": list(q_c.stride()),
            "kv_c_stride": list(kv_c.stride()),
            "k_pe_stride": list(k_pe.stride()),
            "index_k_stride": list(index_k.stride()),
            "cos_sin_cache": [FUSED_MAX_POSITIONS, ROPE_DIM, "float32"],
            "positions": ("arange(T)" if layout == "prefill"
                          else "uniform random in [0, 131072)")},
        "call": ("vllm.models.deepseek_v32.common.kernels.fused_norm_rope("
                 "positions, q_c, q_a_norm_w, eps, kv_c, kv_a_norm_w, eps, "
                 "k_pe, cos_sin_cache, index_k, k_norm_w, k_norm_b, 1e-6, "
                 "index_cos_sin_cache, topk_indices_buffer, slot_mapping, "
                 "indexer_slot_mapping, indexer_k_cache, mla_kv_cache, "
                 "mla_kv_cache_dtype, mla_k_scale, has_indexer, "
                 "index_rope_interleave, kv_c_out, k_pe_out, index_k_out=None)"),
        "vllm030_model_path": ("DeepseekV32Attention.forward (CUDA, non-PCP, "
                               "no HiSparse) -> fused_norm_rope"),
    }
    if len(set(evidence["slot_mapping"])) != tokens:
        raise RuntimeError("slot mapping is not one distinct slot per token")
    return write, mla_cache, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True,
                        choices=["indexer_k", "mla_fp8_ds_mla", "mla_bf16",
                                 "fused_norm_rope"])
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--slot-layout", required=True,
                        choices=["prefill", "decode"])
    parser.add_argument("--kv-cache-dtype", choices=["fp8_ds_mla", "bf16"])
    parser.add_argument("--q-lora-rank", type=int)
    parser.add_argument("--has-indexer", type=int, choices=[0, 1])
    parser.add_argument("--index-n-heads", type=int)
    parser.add_argument("--index-rope-interleave", type=int, choices=[0, 1])
    parser.add_argument("--topk", type=int)
    parser.add_argument("--nvtx_range_name", type=str, required=True)
    add_h200_capture_arguments(parser)
    args = parser.parse_args()
    fused_args = ("kv_cache_dtype", "q_lora_rank", "has_indexer",
                  "index_n_heads", "index_rope_interleave", "topk")
    given = [name for name in fused_args if vars(args)[name] is not None]
    if args.op == "fused_norm_rope" and len(given) != len(fused_args):
        raise ValueError(f"--op fused_norm_rope requires all of {fused_args}")
    if args.op != "fused_norm_rope" and given:
        raise ValueError(f"{given} apply only to --op fused_norm_rope")
    capture = open_h200_capture_session(args, args.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    torch.manual_seed(42)
    if args.op == "fused_norm_rope":
        write, kv_cache, evidence = build_fused_case(
            args.tokens, args.slot_layout, args.kv_cache_dtype,
            args.q_lora_rank, bool(args.has_indexer), args.index_n_heads,
            bool(args.index_rope_interleave), args.topk)
    else:
        write, kv_cache, evidence = build_case(args.op, args.tokens,
                                               args.slot_layout)
        if VLLM_STACK == "vllm030":
            evidence["vllm030_model_path"] = VLLM030_MODEL_PATH
    evidence["vllm_stack"] = VLLM_STACK
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(WARMUP):
        write()
    torch.cuda.synchronize()
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(args.nvtx_range_name)
    write()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()
    if not bool(kv_cache.view(torch.uint8).any()):
        raise RuntimeError("cache write left the KV cache all zero")
    capture.add_launch_evidence("dsa_cache_write", evidence)
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "warmup_calls": WARMUP,
    })
    capture.finish()
    print("DSA cache write done " + json.dumps(args.op))


if __name__ == "__main__":
    main()
