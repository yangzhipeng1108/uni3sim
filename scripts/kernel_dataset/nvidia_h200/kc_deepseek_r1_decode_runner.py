"""H200 kcampaign runner for the DeepSeek-R1/V3 decode-path operators vLLM 0.24 launches.

Every op reproduces the tensor layouts and dispatch of the deployed vLLM
caller-selected 0.24.0 server at a caller-chosen
per-rank shape, so the plan generator can sweep any model with the same
operator family (M, N, K, heads, E, top-k, per-rank intermediate):

* ``mla_bmm_uk``: ``MLAAttention`` decode W_UK absorption,
  ``torch.bmm(q_nope.transpose(0, 1), W_UK_T, out=(H, M, L))`` with
  ``W_UK_T = W_UK.permute(1, 2, 0)`` a view of ``kv_b_proj`` (L, H, P + V)
  (``vllm/model_executor/layers/attention/mla_attention.py`` 739-792, 899-962).
* ``mla_bmm_uv``: ``_v_up_proj``, ``torch.bmm(x.view(-1, H, L).transpose(0, 1),
  W_UV, out=out.view(-1, H, V).transpose(0, 1))`` with ``W_UV =
  W_UV.transpose(0, 1)`` (same file 1012-1034).
* ``router``: ``GateLinear.forward`` tiers 1 and 3
  (``vllm/model_executor/layers/fused_moe/router/gate_linear.py``): on Hopper
  ``ops.dsv3_router_gemm`` for M <= 16 when (hidden, experts) is an
  instantiated DSV3 shape, otherwise ``torch.mm(x, W.T, out_dtype=float32)``.
* ``fused_moe_fp8``: ``fused_experts`` with ``fp8_w8a8_moe_quant_config``
  (block [128, 128], dynamic per-token-group activations), uniform random
  top-k ids without replacement (the server's
  ``VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random``); vLLM itself picks
  the naive (M * top_k * 4 <= E) or aligned path, the tuned Triton config,
  and launches its own activation quantisation, silu*mul and top-k sum.
* ``act_quant``: ``per_token_group_quant_fp8`` on a BF16 [M, K] tensor, group
  128 (``column_major_scales`` / ``use_ue8m0`` as the DeepGEMM 1d2d linear
  path passes them, or row-major for the MoE path).
* ``moe_sum``: ``ops.moe_sum`` of a BF16 [M, top_k, hidden] tensor (top_k 8
  lowers to ``at::sum_out``).
* ``flashinfer_mla_decode``: the SGLang (InferenceX) MLA decode backend,
  FlashInfer ``BatchMLAPagedAttentionWrapper`` (``flashinfer/mla/_core.py``)
  planned for B requests of ``--q-len`` query tokens and ``--kv-len`` cached
  tokens, latent 512 + rope 64, BF16, ``--page-size`` rows per page (SGLang
  uses 1), causal for q_len > 1 (speculative verify); ``plan`` runs outside
  the timed range, the timed range holds one ``run``.

Timing protocol: the canonical ``nsys_nvtx_kernel_median`` window of
``nsys_campaign_runner.py`` (one process per window, H200CaptureSession clock
gate).  Unlike the A800 GEMM job, ``--pre-nvtx-warmup-iterations`` (default 2)
runs the op before the NVTX range so lazy JIT, handle and workspace
creation never enter the timed range; the timed range holds exactly one op
call.  Evaluation input only: no duration feeds any model parameter.
"""

import argparse
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    add_h200_capture_arguments,
    open_h200_capture_session,
)

import torch

OPS = ("mla_bmm_uk", "mla_bmm_uv", "router", "fused_moe_fp8", "act_quant",
       "moe_sum", "flashinfer_mla_decode")
# gate_linear.py: GateLinear.DSV3_SUPPORTED_* and _dsv3_max_batch on Hopper.
DSV3_ROUTER_SHAPES = {(7168, 256), (6144, 256), (7168, 384)}
DSV3_ROUTER_MAX_TOKENS_HOPPER = 16


def build_op(args):
    dev = "cuda"
    bf16 = torch.bfloat16
    if args.op in ("mla_bmm_uk", "mla_bmm_uv"):
        heads, tokens = args.heads, args.tokens
        nope, v_dim, lora = args.qk_nope_head_dim, args.v_head_dim, args.kv_lora_rank
        kv_b = torch.randn(lora, heads, nope + v_dim, device=dev, dtype=bf16)
        w_uk, w_uv = kv_b.split([nope, v_dim], dim=-1)
        if args.op == "mla_bmm_uk":
            q = torch.randn(tokens, heads, nope + args.qk_rope_head_dim, device=dev, dtype=bf16)
            q_nope = q.split([nope, args.qk_rope_head_dim], dim=-1)[0].transpose(0, 1)
            w_uk_t = w_uk.permute(1, 2, 0)
            out = q_nope.new_empty((heads, tokens, lora))
            return lambda: torch.bmm(q_nope, w_uk_t, out=out)
        attn = torch.randn(tokens, heads, lora, device=dev, dtype=bf16)
        x = attn.view(-1, heads, lora).transpose(0, 1)
        w = w_uv.transpose(0, 1)
        out = torch.empty(tokens, heads * v_dim, device=dev, dtype=bf16)
        out_t = out.view(-1, heads, v_dim).transpose(0, 1)
        return lambda: torch.bmm(x, w, out=out_t)
    if args.op == "router":
        from vllm import _custom_ops as ops
        x = torch.randn(args.tokens, args.hidden, device=dev, dtype=bf16)
        w = torch.randn(args.experts, args.hidden, device=dev, dtype=bf16)
        if ((args.hidden, args.experts) in DSV3_ROUTER_SHAPES
                and args.tokens <= DSV3_ROUTER_MAX_TOKENS_HOPPER):
            return lambda: ops.dsv3_router_gemm(
                hidden_states=x, router_weight=w, output_dtype=torch.float32)
        return lambda: torch.mm(x, w.T, out_dtype=torch.float32)
    if args.op == "fused_moe_fp8":
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        fp8 = torch.float8_e4m3fn
        e, h, n = args.experts, args.hidden, args.intermediate
        w1 = torch.randn(e, 2 * n, h, device=dev, dtype=bf16).to(fp8)
        w2 = torch.randn(e, h, n, device=dev, dtype=bf16).to(fp8)
        w1_s = torch.rand(e, 2 * n // 128, h // 128, device=dev, dtype=torch.float32) * 1e-2
        w2_s = torch.rand(e, h // 128, n // 128, device=dev, dtype=torch.float32) * 1e-2
        x = torch.randn(args.tokens, h, device=dev, dtype=bf16)
        gen = torch.Generator(device=dev).manual_seed(0)
        ids = torch.stack([torch.randperm(e, device=dev, generator=gen)[:args.top_k]
                           for _ in range(args.tokens)]).to(torch.int32)
        weights = torch.rand(args.tokens, args.top_k, device=dev, dtype=torch.float32)
        cfg = fp8_w8a8_moe_quant_config(w1_scale=w1_s, w2_scale=w2_s, block_shape=[128, 128])
        return lambda: fused_experts(x, w1, w2, weights, ids, quant_config=cfg,
                                     global_num_experts=e)
    if args.op == "act_quant":
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        x = torch.randn(args.tokens, args.k, device=dev, dtype=bf16)
        return lambda: per_token_group_quant_fp8(
            x, group_size=128, column_major_scales=bool(args.column_major_scales),
            use_ue8m0=bool(args.use_ue8m0))
    if args.op == "moe_sum":
        from vllm import _custom_ops as ops
        x = torch.randn(args.tokens, args.top_k, args.hidden, device=dev, dtype=bf16)
        out = torch.empty(args.tokens, args.hidden, device=dev, dtype=bf16)
        return lambda: ops.moe_sum(x, out)
    if args.op == "flashinfer_mla_decode":
        from flashinfer.mla import BatchMLAPagedAttentionWrapper
        batch, q_len, kv_len, heads = args.tokens, args.q_len, args.kv_len, args.heads
        page = args.page_size
        pages_per_req = -(-kv_len // page)
        total_pages = batch * pages_per_req
        ckv = torch.randn(total_pages, page, args.kv_lora_rank, device=dev, dtype=bf16)
        kpe = torch.randn(total_pages, page, args.qk_rope_head_dim, device=dev, dtype=bf16)
        q_nope = torch.randn(batch * q_len, heads, args.kv_lora_rank, device=dev, dtype=bf16)
        q_pe = torch.randn(batch * q_len, heads, args.qk_rope_head_dim, device=dev, dtype=bf16)
        i32 = torch.int32
        wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev))
        wrapper.plan(
            torch.arange(batch + 1, dtype=i32, device=dev) * q_len,
            torch.arange(batch + 1, dtype=i32, device=dev) * pages_per_req,
            torch.arange(total_pages, dtype=i32, device=dev),
            torch.full((batch,), kv_len, dtype=i32, device=dev),
            heads, args.kv_lora_rank, args.qk_rope_head_dim, page, q_len > 1,
            1.0 / (args.qk_nope_head_dim + args.qk_rope_head_dim) ** 0.5, bf16, bf16)
        return lambda: wrapper.run(q_nope, q_pe, ckv, kpe)
    raise ValueError(f"unknown op {args.op!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True, choices=OPS)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--heads", type=int, default=0)
    p.add_argument("--qk-nope-head-dim", type=int, default=128)
    p.add_argument("--qk-rope-head-dim", type=int, default=64)
    p.add_argument("--v-head-dim", type=int, default=128)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--hidden", type=int, default=0)
    p.add_argument("--experts", type=int, default=0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--intermediate", type=int, default=0,
                   help="per-rank routed-expert intermediate size")
    p.add_argument("--k", type=int, default=0, help="act_quant row width")
    p.add_argument("--q-len", type=int, default=1, help="flashinfer_mla_decode query tokens per request")
    p.add_argument("--kv-len", type=int, default=0, help="flashinfer_mla_decode cached tokens per request")
    p.add_argument("--page-size", type=int, default=1, help="flashinfer_mla_decode KV rows per page")
    p.add_argument("--column-major-scales", type=int, choices=(0, 1), default=0)
    p.add_argument("--use-ue8m0", type=int, choices=(0, 1), default=0)
    p.add_argument("--pre-nvtx-warmup-iterations", type=int, default=2)
    p.add_argument("--nvtx_range_name", type=str, required=True)
    add_h200_capture_arguments(p)
    args = p.parse_args()
    if args.tokens <= 0 or args.pre_nvtx_warmup_iterations < 0:
        raise ValueError("tokens must be positive and warmup non-negative")
    capture = open_h200_capture_session(args, args.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    with torch.no_grad():
        op = build_op(args)
        for _ in range(args.pre_nvtx_warmup_iterations):
            op()
        torch.cuda.synchronize()
        capture.power_warmup(torch)
        torch.cuda.synchronize()
        capture.before_nvtx()
        torch.cuda.nvtx.range_push(args.nvtx_range_name)
        op()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    capture.after_nvtx()
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
    })
    capture.finish()


if __name__ == "__main__":
    main()
