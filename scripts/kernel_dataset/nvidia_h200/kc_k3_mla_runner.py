"""NVIDIA H200 Kimi-K3 gated-MLA decode runner (vLLM 0.30.0 only).

Kimi-K3's full-attention layers are ``vllm/models/kimi_k3/nvidia/mla.py``
``MultiHeadLatentAttention`` (NoPE MLA: ``mla_use_nope`` -> no rotary
embedding; ``mla_use_output_gate`` -> ``o * sigmoid(g)``).  The layer selects
its decode backend with ``get_attn_backend(576, bf16, kv_cache_dtype,
use_mla=True, use_sparse=False, num_heads=local_heads)``; on SM90 the priority
list (``platforms/cuda.py::_get_backend_priorities``) starts with
``FLASH_ATTN_MLA`` and the K3 checkpoint has no KV-cache quantization
(``kv_cache_scheme: null`` -> ``auto`` bf16 cache), so FA3 MLA decode is used.
The runner evaluates the selector on the H200 and fails if it returns anything
else.

One NVTX range = the decode branch of ``MultiHeadLatentAttention._attention``
for ``--batch`` requests of one new token each, plus the output gate of
``forward``:

1. ``torch.bmm(q_nope.transpose(0, 1), W_UK_T).transpose(0, 1)`` (W_UK
   absorb; ``W_UK_T`` is the permuted view of ``kv_b_proj.weight.T`` that
   ``process_weights_after_loading`` registers);
2. ``fused_mla_decode_q_concat_kv_cache_insert(ql_nope, q_pe, kv_c_normed,
   k_pe, kv_cache, slot_mapping)`` (bf16 cache: one launch concatenating the
   MQA query and inserting the new latent token into the paged cache);
3. ``FlashAttnMLAImpl.forward_mqa``: FA3 ``flash_attn_varlen_func(q=q_pe,
   k=k_pe_cache, v=kv_c_cache, q_v=q_nope, causal=True, fa_version=3,
   scheduler_metadata, num_splits=0)`` (eager ``max_num_splits = 0``);
4. ``_v_up_proj``: ``torch.bmm(latent, W_UV, out=attn_out.transpose)``;
5. ``_gate_sigmoid_mul(attn_out, gate)`` (``torch.compile``'d elementwise;
   compiled during warm-up).

Q/K/V projections, norms and ``o_proj`` are GEMM-family and outside the
range.  The paged cache uses the block size vLLM's
``Platform._align_hybrid_block_size`` gives K3 (``--kv-block-size``, computed
by the plan generator: MLA alignment 128 tokens x cdiv(KDA state page,
128 x 1152 B)); blocks are scattered through a random permutation; the
scheduler metadata (``get_scheduler_metadata``, as the metadata builder's AOT
schedule) is built before the range.
"""

import argparse
import importlib.metadata
import math
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

WARMUP = 5
INPUT_SEED = 42
FA_VERSION = 3


def _measure(fn, nvtx_range_name, capture):
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(nvtx_range_name)
    fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()


def _selected_backend(head_size, num_heads, block_size):
    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.selector import get_attn_backend
    config = VllmConfig(cache_config=CacheConfig(block_size=block_size))
    with set_current_vllm_config(config):
        backend = get_attn_backend(head_size, torch.bfloat16, "auto",
                                   use_mla=True, use_sparse=False,
                                   num_heads=num_heads)
    return backend.get_name()


def run_decode(args, capture):
    from vllm.models.kimi_k3.nvidia.mla import _gate_sigmoid_mul
    from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
        fused_mla_decode_q_concat_kv_cache_insert,
    )
    from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

    device = "cuda"
    H, B, S, bs = args.num_heads, args.batch, args.context, args.kv_block_size
    L, P, R, V = args.kv_lora_rank, args.qk_nope_head_dim, args.qk_rope_head_dim, args.v_head_dim
    q_lora = args.q_lora_rank
    if min(H, B, S, bs) <= 0 or bs % 16:
        raise ValueError("positive shapes and a block size multiple of 16 required")
    head_size = L + R
    backend = _selected_backend(head_size, H, bs)
    if backend != "FLASH_ATTN_MLA":
        raise RuntimeError(f"vLLM 0.30 selects {backend} for K3 MLA on this GPU; "
                           "the runner implements FLASH_ATTN_MLA only")
    bf16 = torch.bfloat16
    scale = 1.0 / math.sqrt(P + R)

    blocks_per_req = -(-S // bs)
    pool_blocks = B * blocks_per_req + 1
    kv_cache = torch.randn(pool_blocks, bs, head_size, dtype=bf16, device=device)
    perm = torch.randperm(pool_blocks - 1, device="cpu") + 1
    block_table_cpu = perm[:B * blocks_per_req].view(B, blocks_per_req).to(torch.int32)
    block_table = block_table_cpu.to(device)
    pos = S - 1
    slot_cpu = (block_table_cpu[:, pos // bs].to(torch.int64) * bs + pos % bs)
    slot_mapping = slot_cpu.to(device)
    seq_lens = torch.full((B,), S, dtype=torch.int32, device=device)
    cu_q = torch.arange(B + 1, dtype=torch.int32, device=device)

    # q_b_proj output view [B, H, P + R]; qkv_a(+gate) projection output.
    q = torch.randn(B, H * (P + R), dtype=bf16, device=device).view(-1, H, P + R)
    q_nope, q_pe = q.split([P, R], dim=-1)
    qkv_a_rows = q_lora + L + R
    qkv_a_g = torch.randn(B, qkv_a_rows + H * V, dtype=bf16, device=device)
    qkv_lora, gate = qkv_a_g.split([qkv_a_rows, H * V], dim=-1)
    k_pe = qkv_lora[:, q_lora + L:].unsqueeze(1)
    kv_c_normed = torch.randn(B, L, dtype=bf16, device=device)
    # process_weights_after_loading: kv_b_proj.weight.T -> view -> split.
    kv_b = torch.randn(H * (P + V), L, dtype=bf16, device=device) * 0.02
    W_UK, W_UV = kv_b.T.view(L, H, P + V).split([P, V], dim=-1)
    W_UV = W_UV.transpose(0, 1)
    W_UK_T = W_UK.permute(1, 2, 0)
    attn_out = torch.empty(B, H * V, dtype=bf16, device=device)
    sched = get_scheduler_metadata(
        batch_size=B, max_seqlen_q=1, max_seqlen_k=S, num_heads_q=H,
        num_heads_kv=1, headdim=R, cache_seqlens=seq_lens, qkv_dtype=bf16,
        headdim_v=L, page_size=bs, cu_seqlens_q=cu_q, causal=True, num_splits=0)
    kv_c_cache = kv_cache[..., :L]
    k_pe_cache = kv_cache[..., L:]
    result = {}

    def fn():
        ql_nope = torch.bmm(q_nope.transpose(0, 1), W_UK_T).transpose(0, 1)
        mqa_q = fused_mla_decode_q_concat_kv_cache_insert(
            ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping,
            positions=None, cos_sin_cache=None)
        mq_nope, mq_pe = torch.split(mqa_q, [L, R], dim=-1)
        latent = flash_attn_varlen_func(
            q=mq_pe, k=k_pe_cache.unsqueeze(-2), v=kv_c_cache.unsqueeze(-2),
            q_v=mq_nope, max_seqlen_q=1, cu_seqlens_q=cu_q, max_seqlen_k=S,
            seqused_k=seq_lens, block_table=block_table, softmax_scale=scale,
            causal=True, return_softmax_lse=False, fa_version=FA_VERSION,
            scheduler_metadata=sched, num_splits=0)
        x = latent.view(-1, H, L).transpose(0, 1)
        torch.bmm(x, W_UV, out=attn_out.view(-1, H, V).transpose(0, 1))
        result["out"] = _gate_sigmoid_mul(attn_out, gate)

    with torch.inference_mode():
        _measure(fn, args.nvtx_range_name, capture)
    if not bool(torch.isfinite(result["out"]).all()):
        raise RuntimeError("K3 MLA output is not finite")
    capture.add_launch_evidence("k3_mla_case", {
        "op": "decode", "vllm_stack": installed_vllm_stack(),
        "call_path": "vllm030 kimi_k3/nvidia/mla.py MultiHeadLatentAttention._attention "
                     "decode branch + forward output gate",
        "selected_backend": backend,
        "backend_query": "get_attn_backend(576, bf16, 'auto', use_mla=True, "
                         "use_sparse=False, num_heads) under VllmConfig(block_size)",
        "num_heads_per_rank": H, "batch": B, "q_lens": [1] * B, "seq_lens": [S] * B,
        "kv_lora_rank": L, "qk_nope_head_dim": P, "qk_rope_head_dim": R,
        "v_head_dim": V, "q_lora_rank": q_lora, "rotary": "none (mla_use_nope)",
        "output_gate": True, "softmax_scale": scale,
        "kv_cache": {"dtype": "bfloat16 (auto)", "shape": list(kv_cache.shape),
                     "block_size": bs, "pool_blocks": pool_blocks,
                     "block_table_shape": list(block_table.shape),
                     "layout": "[blocks, block_size, 512 kv_c | 64 k_pe]"},
        "slot_mapping": slot_cpu.tolist(),
        "w_uk_t": {"shape": list(W_UK_T.shape), "stride": list(W_UK_T.stride())},
        "w_uv": {"shape": list(W_UV.shape), "stride": list(W_UV.stride())},
        "gate": {"shape": list(gate.shape), "stride": list(gate.stride())},
        "scheduler_metadata_shape": list(sched.shape),
        "fa_num_splits": 0, "fa_version": FA_VERSION,
        "public_op": ["torch.bmm (W_UK_T absorb)",
                      "vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache."
                      "fused_mla_decode_q_concat_kv_cache_insert",
                      "vllm.vllm_flash_attn.flash_attn_varlen_func (FlashAttnMLAImpl.forward_mqa)",
                      "torch.bmm (_v_up_proj W_UV)",
                      "vllm.models.kimi_k3.nvidia.mla._gate_sigmoid_mul (torch.compile)"],
    })
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "triton_version": importlib.metadata.version("triton"),
        "vllm_stack": installed_vllm_stack(),
        "input_seed": INPUT_SEED,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True, choices=["decode"])
    p.add_argument("--num-heads", type=int, required=True, help="heads per rank")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--context", type=int, required=True,
                   help="sequence length per request including the new token")
    p.add_argument("--kv-block-size", type=int, required=True)
    p.add_argument("--kv-lora-rank", type=int, required=True)
    p.add_argument("--qk-nope-head-dim", type=int, required=True)
    p.add_argument("--qk-rope-head-dim", type=int, required=True)
    p.add_argument("--v-head-dim", type=int, required=True)
    p.add_argument("--q-lora-rank", type=int, required=True)
    p.add_argument("--nvtx_range_name", required=True)
    add_h200_capture_arguments(p)
    a = p.parse_args()
    stack = installed_vllm_stack()
    if stack != "vllm030":
        raise ValueError(f"kc_k3_mla_runner is the vLLM 0.30 Kimi-K3 path; installed {stack}")
    capture = open_h200_capture_session(a, a.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    torch.manual_seed(INPUT_SEED)
    torch.cuda.manual_seed(INPUT_SEED)
    run_decode(a, capture)
    capture.finish()
    print("K3 MLA done")


if __name__ == "__main__":
    main()
