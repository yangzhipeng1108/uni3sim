"""NVIDIA H200 DeepSeek-V4 attention-component runner (CSA / HCA / SWA,
KV compressor, mHC).

Executes vLLM 0.24.0's DeepSeek-V4 SM90 path
(``vllm/models/deepseek_v4``: ``DeepseekV4FlashMLAAttention`` -- the
``FLASHMLA_SPARSE_DSV4`` backend, selected on every non-SM12 CUDA device --
plus ``DeepseekCompressor`` and the tilelang mHC kernels) under the canonical
``nsys_nvtx_kernel_median`` protocol.  One NVTX range holds exactly the
per-layer kernel sequence of one component:

* ``--op decode`` -- ``DeepseekV4FlashMLAAttention._forward_decode``: for a
  C4A (CSA, ``--layer-type c4``) layer ``compute_global_topk_indices_and_lens``
  maps the indexer's request-local top-k (``index_topk`` distinct random
  compressed positions per token, -1 padded) to global slots of the
  compressed cache, then one FlashMLA sparse FP8 decode call
  ``flash_mla_with_kvcache(q, k_cache=SWA cache, indices=SWA window,
  topk_length, is_fp8_kvcache=True, attn_sink, extra_k_cache=compressed
  cache, extra_indices_in_kvcache, extra_topk_length, head_dim_v=512)``.
  C128A (HCA, ``c128``) layers use the metadata-builder C128A indices (all
  ``pos // 128`` compressed tokens, width ``round_up(ceil(max_model_len /
  128), 128)``) and only the FlashMLA call; SWA-only layers (compress ratio
  0/1, ``swa``) have no extra cache.  q is ``[tokens, 1, padded_heads, 512]``
  (FlashMLA FP8 decode requires ``h_q`` in {64, 128}).
* ``--op prefill`` -- ``_forward_prefill`` for one prefill chunk of one
  request (prefix ``--context``, ``--q-len`` new tokens):
  ``dequantize_and_gather_k_cache`` of the compressed cache (C4A/C128A) and
  of the SWA cache into the BF16 workspace, ``combine_topk_swa_indices`` and
  ``flash_mla_sparse_fwd(q, kv, indices, attn_sink, topk_length)``.
* ``--op compressor`` -- ``DeepseekCompressor.forward`` (head_dim 512 main-KV
  compressor): ``save_partial_states`` + ``compress_norm_rope_store_cutedsl``
  (C4: fused kernel, C128: split two-kernel version) writing the FP8
  ``fp8_ds_mla`` compressed cache; ``kv_score`` is the FP32
  ``fused_wkv_wgate`` output.  Decode tokens sit at staggered positions
  ``context + r`` so that the compress boundary hits the real fraction of
  requests.
* ``--op mhc_pre`` / ``mhc_post_pre`` -- ``mhc_pre_tilelang`` (first layer)
  and ``mhc_fused_post_pre_tilelang`` (every later attention/FFN boundary,
  ``n_splits=1, tile_n=1``) with the fused RMSNorm weight, ``hc_mult``
  residual streams of ``hidden_size`` and ``sinkhorn_iters`` iterations.

KV caches use the ``fp8_ds_mla`` V4 layout (584 B/token: 448 FP8 NoPE + 64
BF16 RoPE + 8 UE8M0 scales), filled before the NVTX range with vLLM's own
encoder ``quantize_and_insert_k_cache`` from random BF16 K; SWA pages hold
64 tokens and compressed pages ``256 // ratio`` tokens (``--block-size 256``
of the V4 recipes), all scattered through a random permutation.  One
``FlashMLASchedMeta`` is initialised by the first warm-up call and reused by
the NVTX call (as layers 2..L of one step reuse the per-type plan).

vLLM 0.30.0 (``installed_vllm_stack() == "vllm030"``; the branch taken is
recorded as ``vllm_stack`` / ``call_path`` in ``v4attn_case``).  V4-Flash/Pro
(``DeepseekV4ForCausalLM``) still map to ``vllm.models.deepseek_v4`` and
``DeepseekV4FlashMLAAttention`` on SM90; decode and the KV layout are
unchanged.  Differences executed here:

* prefill: ``combine_topk_swa_indices`` writes into preallocated workspace
  buffers (``out=``, ``fill_(-1)`` instead of ``torch.full``) with
  ``left_visible=right_visible=None`` and ``max_image_tokens=0`` (text model:
  no ``vision_n_layers``);
* compressor: ``_SAVE_PARTIAL_STATES_KERNEL`` and
  ``_SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL`` (same arguments as the 0.24
  functions).  0.30 skips the C128 compress kernels when no request crosses a
  128 boundary and the step is not a FULL CUDA graph; the runner evaluates
  that rule (``c128_boundary``) and always executes the kernels, i.e. it
  measures the FULL-cudagraph decode step (``FULL_AND_PIECEWISE`` default) and
  every prefill chunk that crosses a boundary; a C128 prefill case without a
  boundary (skipped in 0.30 eager/PIECEWISE) is refused;
* mhc_pre: the first layer receives the 2-D embedding and runs
  ``mhc_pre_broadcast_tilelang`` with ``fn_broadcast = fn.view(mix, hc,
  hidden).sum(1)`` (0.24 repeats the embedding to ``[T, hc, hidden]`` and runs
  ``mhc_pre_tilelang``).
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
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
TOKEN_BYTES = 584
PAGE_ALIGNMENT = 576  # MLAAttentionSpec / SlidingWindowMLASpec alignment for fp8_ds_mla
SWA_BLOCK = 64
KERNEL_BLOCK = 256
TOPK_ALIGN = 128
LAYER_RATIO = {"swa": 1, "c4": 4, "c128": 128}
FILL_CHUNK_TOKENS = 1 << 20
C128_STATE_BLOCK = 8
VLLM_STACK = installed_vllm_stack()


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


class PagedPool:
    """Scattered paged pool: request r owns pages for token range [lo_r, hi_r).

    Block-table entries outside the owned range point at page 0 (vLLM clamps
    null blocks to 0); owned pages come from a random permutation of 1..P.
    """

    def __init__(self, ranges, page, device):
        self.page = page
        widths, owned = [], []
        for lo, hi in ranges:
            if not 0 <= lo < hi:
                raise ValueError(f"invalid token range [{lo}, {hi})")
            widths.append(math.ceil(hi / page))
            owned.append(range(lo // page, math.ceil(hi / page)))
        total = sum(len(pages) for pages in owned)
        self.num_pages = total + 1
        perm = torch.randperm(total, device="cpu") + 1
        table = torch.zeros(len(ranges), max(widths), dtype=torch.int32)
        cursor = 0
        for r, pages in enumerate(owned):
            table[r, pages.start:pages.stop] = perm[cursor:cursor + len(pages)].to(torch.int32)
            cursor += len(pages)
        self.block_table_cpu = table
        self.block_table = table.to(device)

    def facts(self):
        return {"page_tokens": self.page, "page_bytes": _page_bytes(self.page),
                "token_bytes": TOKEN_BYTES, "num_pages": self.num_pages,
                "block_table_shape": list(self.block_table_cpu.shape),
                "kv_dtype": "fp8_ds_mla (uint8: 448 fp8_e4m3 NoPE + 64 bf16 RoPE + 8 ue8m0 scales)"}

    def slots(self, req, positions):
        pos = torch.as_tensor(positions, dtype=torch.int64)
        return (self.block_table_cpu[req, pos // self.page].to(torch.int64)
                * self.page + pos % self.page)


def _page_bytes(page):
    return -(-page * TOKEN_BYTES // PAGE_ALIGNMENT) * PAGE_ALIGNMENT


def _fp8_cache(pool, device):
    """fp8_ds_mla pages padded to 576 B (``_apply_alignment_padding``)."""
    from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
    page_bytes = _page_bytes(pool.page)
    raw = torch.zeros(pool.num_pages * page_bytes, dtype=torch.uint8, device=device)
    flat = raw.view(pool.num_pages, page_bytes)
    cache = raw.as_strided((pool.num_pages, pool.page, TOKEN_BYTES),
                           (page_bytes, TOKEN_BYTES, 1))
    total = pool.num_pages * pool.page
    for start in range(0, total, FILL_CHUNK_TOKENS):
        end = min(total, start + FILL_CHUNK_TOKENS)
        k = torch.randn(end - start, HEAD_DIM, dtype=torch.bfloat16, device=device)
        slots = torch.arange(start, end, dtype=torch.int64, device=device)
        quantize_and_insert_k_cache(k, flat, slots, block_size=pool.page)
    torch.cuda.synchronize()
    return cache


def _random_prefix_topk(counts, width, device):
    """Row t: min(width, counts[t]) distinct random ints in [0, counts[t]), -1 padded."""
    counts = torch.as_tensor(counts, dtype=torch.int64, device=device)
    n_max = int(counts.max().item())
    out = torch.full((counts.numel(), width), -1, dtype=torch.int32, device=device)
    if n_max == 0:
        return out
    rows = 256
    for start in range(0, counts.numel(), rows):
        c = counts[start:start + rows]
        keys = torch.rand(c.numel(), n_max, device=device)
        keys[torch.arange(n_max, device=device)[None, :] >= c[:, None]] = 2.0
        k = min(width, n_max)
        idx = torch.topk(keys, k, dim=1, largest=False).indices.to(torch.int32)
        valid = torch.arange(k, device=device)[None, :] < torch.clamp(c, max=width)[:, None]
        out[start:start + rows, :k] = torch.where(valid, idx, -1)
    return out


def _check_heads(args):
    if args.padded_heads not in (64, 128):
        raise ValueError("FlashMLA FP8 sparse decode requires h_q in {64, 128}")
    if not 0 < args.local_heads <= args.padded_heads:
        raise ValueError("local heads must be in [1, padded_heads]")


def _attn_sink(args, device):
    sink = torch.full((args.padded_heads,), -float("inf"), dtype=torch.float32,
                      device=device)
    sink[:args.local_heads] = torch.randn(args.local_heads, device=device)
    return sink


def run_decode(args, capture, evidence):
    from vllm.models.deepseek_v4.common.ops import compute_global_topk_indices_and_lens
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata
    _check_heads(args)
    device = "cuda"
    ratio = LAYER_RATIO[args.layer_type]
    B, Q, S = args.batch, args.q_len, args.context
    if Q > S:
        raise ValueError("decode q_len must not exceed the context")
    window = args.window
    # Token j of request r is at position S - Q + j (S = sequence length
    # including the new tokens).
    positions = [[S - Q + j for j in range(Q)] for _ in range(B)]
    flat_pos = [p for req in positions for p in req]
    token_req = [r for r in range(B) for _ in range(Q)]
    N = B * Q

    swa_pool = PagedPool([(max(0, S - Q - window + 1), S)] * B, SWA_BLOCK, device)
    swa_cache = _fp8_cache(swa_pool, device)
    swa_idx = torch.full((N, 1, window), -1, dtype=torch.int32)
    swa_len = torch.empty(N, dtype=torch.int32)
    for t, (r, pos) in enumerate(zip(token_req, flat_pos)):
        lo = max(0, pos - window + 1)
        swa_idx[t, 0, :pos + 1 - lo] = swa_pool.slots(r, range(lo, pos + 1)).to(torch.int32)
        swa_len[t] = pos + 1 - lo
    swa_idx, swa_len = swa_idx.to(device), swa_len.to(device)

    q = torch.randn(N, 1, args.padded_heads, HEAD_DIM, dtype=torch.bfloat16,
                    device=device)
    out = torch.empty_like(q)
    sink = _attn_sink(args, device)
    sched = get_mla_metadata()[0]
    evidence["sched_meta"] = sched
    evidence["call_path"] = (f"{VLLM_STACK}: DeepseekV4FlashMLAAttention._forward_decode "
                             "(unchanged 0.24 -> 0.30)")
    scale = HEAD_DIM ** -0.5
    swa_cache4 = swa_cache.unsqueeze(-2)
    evidence.update({
        "q_lens": [Q] * B, "seq_lens": [S] * B, "token_positions": flat_pos,
        "window": window, "swa_cache": swa_pool.facts(),
        "swa_topk_length": swa_len.tolist(), "q_shape": list(q.shape),
        "head_dim_qk": HEAD_DIM, "head_dim_nope": NOPE_DIM, "head_dim_rope": ROPE_DIM,
        "head_dim_v": HEAD_DIM, "softmax_scale": scale,
        "public_op": ["vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache"],
        "flash_mla_args": {"is_fp8_kvcache": True, "causal": False,
                           "block_table": None, "cache_seqlens": None,
                           "indices_shape": list(swa_idx.shape),
                           "attn_sink_shape": list(sink.shape)}})

    if ratio == 1:
        def fn():
            flash_mla_with_kvcache(
                q=q, k_cache=swa_cache4, block_table=None, head_dim_v=HEAD_DIM,
                tile_scheduler_metadata=sched, cache_seqlens=None,
                is_fp8_kvcache=True, indices=swa_idx, topk_length=swa_len,
                softmax_scale=scale, attn_sink=sink, extra_k_cache=None,
                extra_indices_in_kvcache=None, extra_topk_length=None, out=out)
        return fn

    page = KERNEL_BLOCK // ratio
    n_comp = S // ratio
    if n_comp == 0:
        raise ValueError("context shorter than one compressed token")
    comp_pool = PagedPool([(0, n_comp)] * B, page, device)
    comp_cache = _fp8_cache(comp_pool, device).unsqueeze(-2)
    counts = [(p + 1) // ratio for p in flat_pos]
    evidence.update({"compress_ratio": ratio,
                     "compressed_cache": comp_pool.facts(),
                     "compressed_kv_lens": [n_comp] * B,
                     "extra_topk_length": [min(c, args.index_topk) if ratio == 4 else c
                                           for c in counts]})
    if ratio == 4:
        topk = args.index_topk
        local = _random_prefix_topk(counts, topk, device)
        token_to_req = torch.tensor(token_req, dtype=torch.int32, device=device)
        is_valid = torch.ones(N, dtype=torch.bool, device=device)
        block_table = comp_pool.block_table
        evidence["indexer"] = {
            "index_topk": topk, "local_topk_shape": list(local.shape),
            "selection": "distinct uniform-random request-local compressed positions "
                         "< (pos+1)//4, -1 padded (seed 42)",
            "kernel_page_tokens": page}
        evidence["public_op"] = [
            "vllm.models.deepseek_v4.common.ops.compute_global_topk_indices_and_lens",
            "vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache"]

        def fn():
            global_idx, lens = compute_global_topk_indices_and_lens(
                local, token_to_req, block_table, page, is_valid)
            flash_mla_with_kvcache(
                q=q, k_cache=swa_cache4, block_table=None, head_dim_v=HEAD_DIM,
                tile_scheduler_metadata=sched, cache_seqlens=None,
                is_fp8_kvcache=True, indices=swa_idx, topk_length=swa_len,
                softmax_scale=scale, attn_sink=sink, extra_k_cache=comp_cache,
                extra_indices_in_kvcache=global_idx.view(N, 1, -1),
                extra_topk_length=lens, out=out)
        return fn

    width = math.ceil(math.ceil(args.max_model_len / ratio) / TOPK_ALIGN) * TOPK_ALIGN
    if n_comp > width:
        raise ValueError("context exceeds max_model_len")
    extra = torch.full((N, 1, width), -1, dtype=torch.int32)
    extra_len = torch.empty(N, dtype=torch.int32)
    for t, (r, c) in enumerate(zip(token_req, counts)):
        extra[t, 0, :c] = comp_pool.slots(r, range(c)).to(torch.int32)
        extra_len[t] = c
    extra, extra_len = extra.to(device), extra_len.to(device)
    evidence["c128a_topk_width"] = width
    evidence["extra_indices_shape"] = list(extra.shape)

    def fn():
        flash_mla_with_kvcache(
            q=q, k_cache=swa_cache4, block_table=None, head_dim_v=HEAD_DIM,
            tile_scheduler_metadata=sched, cache_seqlens=None,
            is_fp8_kvcache=True, indices=swa_idx, topk_length=swa_len,
            softmax_scale=scale, attn_sink=sink, extra_k_cache=comp_cache,
            extra_indices_in_kvcache=extra, extra_topk_length=extra_len,
            out=out)
    return fn


def run_prefill(args, capture, evidence):
    from vllm.models.deepseek_v4.common.ops import (
        combine_topk_swa_indices,
        dequantize_and_gather_k_cache,
    )
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd
    _check_heads(args)
    if args.batch != 1:
        raise ValueError("prefill runner executes one request per chunk")
    device = "cuda"
    ratio = LAYER_RATIO[args.layer_type]
    P, T, window = args.context, args.q_len, args.window
    seq_len = P + T
    gather_len = T + min(P, window - 1)
    positions = torch.arange(P, P + T, dtype=torch.int64, device=device)

    swa_pool = PagedPool([(seq_len - gather_len, seq_len)], SWA_BLOCK, device)
    swa_cache = _fp8_cache(swa_pool, device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([gather_len], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, T], dtype=torch.int32, device=device)
    evidence.update({"q_lens": [T], "prefix_lens": [P], "seq_lens": [seq_len],
                     "gather_lens": [gather_len], "window": window,
                     "compress_ratio": ratio, "swa_cache": swa_pool.facts(),
                     "prefill_chunk_plan": "one chunk, one request"})
    if ratio == 1:
        N, top_k = 0, 0
        topk_indices = torch.full((T, window), -1, dtype=torch.int32, device=device)
        comp_cache = comp_bt = None
    else:
        page = KERNEL_BLOCK // ratio
        N = seq_len // ratio
        comp_pool = PagedPool([(0, N)], page, device)
        comp_cache = _fp8_cache(comp_pool, device)
        comp_bt = comp_pool.block_table
        counts = ((positions + 1) // ratio).tolist()
        if ratio == 4:
            top_k = args.index_topk
            topk_indices = _random_prefix_topk(counts, top_k, device)
        else:
            top_k = math.ceil(math.ceil(args.max_model_len / ratio) / TOPK_ALIGN) * TOPK_ALIGN
            ar = torch.arange(top_k, dtype=torch.int32, device=device)[None, :]
            c = torch.tensor(counts, dtype=torch.int32, device=device)[:, None]
            topk_indices = torch.where(ar < c, ar, -1)
        evidence.update({"compressed_cache": comp_pool.facts(),
                         "compressed_kv_lens": [N], "topk_width": top_k})
        if ratio == 4:
            evidence["indexer"] = {
                "index_topk": top_k,
                "selection": "distinct uniform-random request-local compressed "
                             "positions < (pos+1)//4, -1 padded (seed 42)"}
    M = N + gather_len
    kv = torch.empty(1, M, HEAD_DIM, dtype=torch.bfloat16, device=device)
    q = torch.randn(T, args.padded_heads, HEAD_DIM, dtype=torch.bfloat16,
                    device=device)
    out = torch.empty_like(q)
    sink = _attn_sink(args, device)
    scale = HEAD_DIM ** -0.5
    swa_bt = swa_pool.block_table
    combined_topk = -(-(top_k + window) // TOPK_ALIGN) * TOPK_ALIGN
    pos_list = list(range(P, P + T))
    comb_lens = [min((p + 1) // ratio, top_k) + min(p + 1, window) for p in pos_list]
    evidence.update({
        "workspace_N": N, "workspace_M": M, "combined_topk_width": combined_topk,
        "combined_topk_length": {"min": min(comb_lens), "max": max(comb_lens),
                                 "sum": sum(comb_lens)},
        "q_shape": list(q.shape), "kv_workspace_shape": [M, 1, HEAD_DIM],
        "head_dim_qk": HEAD_DIM, "head_dim_v": HEAD_DIM, "softmax_scale": scale,
        "public_op": (["vllm.models.deepseek_v4.common.ops.dequantize_and_gather_k_cache"]
                      * (1 if comp_cache is None else 2)
                      + ["vllm.models.deepseek_v4.common.ops.combine_topk_swa_indices",
                         "vllm.v1.attention.ops.flashmla.flash_mla_sparse_fwd"])})

    if VLLM_STACK == "vllm024":
        evidence["call_path"] = (
            "0.24 _forward_prefill: combine_topk_swa_indices allocates "
            "torch.full(-1) indices + empty lens inside the call")

        def combine():
            return combine_topk_swa_indices(
                topk_indices, query_start_loc, seq_lens, gather_lens, window,
                ratio, top_k, M, N)
    elif VLLM_STACK == "vllm030":
        # Workspace buffers (max_num_batched_tokens rows in vLLM, sliced to
        # the chunk's query tokens) are allocated outside the NVTX range.
        combined_buf = torch.empty(T, combined_topk, dtype=torch.int32,
                                   device=device)
        lens_buf = torch.empty(T, dtype=torch.int32, device=device)
        evidence["call_path"] = (
            "0.30 _forward_prefill: combine_topk_swa_indices(out=(workspace "
            "indices, workspace lens) -> fill_(-1), left_visible=None, "
            "right_visible=None, max_image_tokens=0)")
        evidence["max_image_tokens"] = 0

        def combine():
            return combine_topk_swa_indices(
                topk_indices, query_start_loc, seq_lens, gather_lens, window,
                ratio, top_k, M, N, out=(combined_buf, lens_buf),
                left_visible=None, right_visible=None, max_image_tokens=0)
    else:
        raise AssertionError(f"unhandled vLLM stack {VLLM_STACK!r}")

    def fn():
        if comp_cache is not None:
            dequantize_and_gather_k_cache(
                kv, comp_cache, seq_lens=seq_lens // ratio, gather_lens=None,
                block_table=comp_bt, block_size=KERNEL_BLOCK // ratio, offset=0)
        dequantize_and_gather_k_cache(
            kv, swa_cache, seq_lens=seq_lens, gather_lens=gather_lens,
            block_table=swa_bt, block_size=SWA_BLOCK, offset=N)
        combined, lens = combine()
        flash_mla_sparse_fwd(q=q, kv=kv.view(-1, 1, HEAD_DIM),
                             indices=combined.unsqueeze(1), sm_scale=scale,
                             attn_sink=sink, topk_length=lens, out=out)
    return fn


def _c128_boundary(req_pos):
    """vLLM 0.30 ``compressor._get_c128_boundary``: some request's new tokens
    [start, start + q_len) reach a multiple of 128 (start = seq_len - q_len)."""
    return any(lo % 128 + (hi - lo) >= 128 for lo, hi in req_pos)


def run_compressor(args, capture, evidence):
    if VLLM_STACK == "vllm024":
        from vllm.models.deepseek_v4.common.ops import save_partial_states
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            compress_norm_rope_store_cutedsl,
        )
        save_op = save_partial_states
        compress_op = compress_norm_rope_store_cutedsl
        save_name = "vllm.models.deepseek_v4.common.ops.save_partial_states"
        compress_name = ("vllm.models.deepseek_v4.nvidia.ops."
                         "sparse_attn_compress_cutedsl.compress_norm_rope_store_cutedsl")
    elif VLLM_STACK == "vllm030":
        from vllm.models.deepseek_v4.common.ops.save_partial_states import (
            _SAVE_PARTIAL_STATES_KERNEL,
        )
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL,
        )
        save_op = _SAVE_PARTIAL_STATES_KERNEL
        compress_op = _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL
        save_name = ("vllm.models.deepseek_v4.common.ops.save_partial_states."
                     "_SAVE_PARTIAL_STATES_KERNEL")
        compress_name = ("vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl."
                         "_SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL")
    else:
        raise AssertionError(f"unhandled vLLM stack {VLLM_STACK!r}")
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_compressed_slot_mapping,
    )
    device = "cuda"
    ratio = LAYER_RATIO[args.layer_type]
    if ratio == 1:
        raise ValueError("SWA-only layers have no compressor")
    coff = 2 if ratio == 4 else 1
    state_block = 4 if ratio == 4 else C128_STATE_BLOCK
    state_window = coff * ratio
    B, Q, S = args.batch, args.q_len, args.context
    if Q == 1:
        # Decode: request r's new token sits at position S + r.
        req_pos = [(S + r, S + r + 1) for r in range(B)]
    else:
        if B != 1:
            raise ValueError("prefill compressor case is one request")
        req_pos = [(S, S + Q)]
    token_req = [r for r, (lo, hi) in enumerate(req_pos) for _ in range(hi - lo)]
    flat_pos = [p for lo, hi in req_pos for p in range(lo, hi)]
    N = len(flat_pos)
    c128_boundary = _c128_boundary(req_pos) if ratio == 128 else None
    if VLLM_STACK == "vllm024":
        skip_rule = "none: vLLM 0.24 always runs the compress kernels"
    else:
        skip_rule = ("vLLM 0.30 skips the C128 compress kernels when "
                     "c128_boundary is False and the step is not a FULL CUDA "
                     "graph; executed here = FULL-cudagraph decode step")
        if ratio == 128 and Q > 1 and not c128_boundary:
            raise ValueError("C128 prefill chunk without a 128 boundary: vLLM "
                             "0.30 (PIECEWISE/eager prefill) skips the compress "
                             "kernels, nothing to measure")

    state_pool = PagedPool([(max(0, lo - state_window), hi) for lo, hi in req_pos],
                           state_block, device)
    state_width = coff * HEAD_DIM
    state_cache = torch.randn(state_pool.num_pages, state_block, 2 * state_width,
                              dtype=torch.float32, device=device)
    slot_mapping = torch.cat([state_pool.slots(r, range(lo, hi))
                              for r, (lo, hi) in enumerate(req_pos)]).to(device)
    page = KERNEL_BLOCK // ratio
    k_pool = PagedPool([(lo // ratio, hi // ratio + 1) for lo, hi in req_pos],
                       page, device)
    k_page_bytes = _page_bytes(page)
    k_cache = torch.zeros(k_pool.num_pages * k_page_bytes, dtype=torch.uint8,
                          device=device).as_strided(
        (k_pool.num_pages, page, TOKEN_BYTES), (k_page_bytes, TOKEN_BYTES, 1))
    qsl = [0]
    for lo, hi in req_pos:
        qsl.append(qsl[-1] + hi - lo)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([hi for _, hi in req_pos], dtype=torch.int32, device=device)
    k_slot_mapping = get_compressed_slot_mapping(
        N, query_start_loc, seq_lens, k_pool.block_table, page, ratio)
    positions = torch.tensor(flat_pos, dtype=torch.int64, device=device)
    token_to_req = torch.tensor(token_req, dtype=torch.int32, device=device)
    kv_score = torch.randn(N, 2 * coff * HEAD_DIM, dtype=torch.float32, device=device)
    kv, score = kv_score.split([coff * HEAD_DIM, coff * HEAD_DIM], dim=-1)
    ape = torch.randn(ratio, coff * HEAD_DIM, dtype=torch.float32, device=device)
    max_pos = max(hi for _, hi in req_pos) + 1
    cos_sin_cache = torch.randn(max_pos, ROPE_DIM, dtype=torch.float32, device=device)
    norm_weight = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_meta = type("KCacheMetadata", (), {"slot_mapping": k_slot_mapping})()
    pdl_kwargs = {"launch_pdl": False}
    k_facts = {"page_tokens": page, "page_bytes": _page_bytes(page),
               "token_bytes": TOKEN_BYTES, "num_pages": k_pool.num_pages,
               "kv_dtype": "fp8_ds_mla uint8"}
    evidence.update({
        "compress_ratio": ratio, "overlap": ratio == 4, "coff": coff,
        "q_lens": [hi - lo for lo, hi in req_pos],
        "prefix_lens": [lo for lo, _ in req_pos],
        "seq_lens": [hi for _, hi in req_pos],
        "state_cache": {"shape": list(state_cache.shape), "dtype": "float32",
                        "block_tokens": state_block, "sliding_window": state_window,
                        "state_width": state_width},
        "compressed_cache": k_facts,
        "boundary_tokens": sum((p + 1) % ratio == 0 for p in flat_pos),
        "c128_boundary": c128_boundary,
        "c128_skip_rule": skip_rule,
        "compress_kernels_executed": True,
        "call_path": f"{VLLM_STACK}: {save_name} + {compress_name}",
        "kv_score": {"shape": list(kv_score.shape), "dtype": "float32"},
        "cos_sin_cache": {"shape": list(cos_sin_cache.shape), "dtype": "float32"},
        "quant": {"quant_block": 64, "token_stride": NOPE_DIM + ROPE_DIM * 2,
                  "scale_dim": NOPE_DIM // 64 + 1, "fp8_max": 448.0},
        "head_dim": HEAD_DIM, "rope_head_dim": ROPE_DIM,
        "public_op": [
            save_name,
            compress_name + " ("
            + ("fused C4 kernel" if ratio == 4 else "split C128 kernels") + ")"]})

    def fn():
        save_op(
            kv=kv, score=score, ape=ape, positions=positions,
            state_cache=state_cache, slot_mapping=slot_mapping,
            block_size=state_block, state_width=state_width,
            compress_ratio=ratio, pdl_kwargs=pdl_kwargs)
        compress_op(
            state_cache=state_cache, num_actual=N,
            token_to_req_indices=token_to_req, positions=positions,
            slot_mapping=slot_mapping, block_table=state_pool.block_table,
            block_size=state_block, state_width=state_width,
            cos_sin_cache=cos_sin_cache, kv_cache=k_cache,
            k_cache_metadata=k_meta, pdl_kwargs=pdl_kwargs,
            head_dim=HEAD_DIM, rope_head_dim=ROPE_DIM, compress_ratio=ratio,
            overlap=ratio == 4, use_fp4_cache=False,
            rms_norm_weight=norm_weight, rms_norm_eps=1e-6,
            quant_block=64, token_stride=NOPE_DIM + ROPE_DIM * 2,
            scale_dim=NOPE_DIM // 64 + 1, store_full_kv=False,
            store_full_fp8=False, fp8_scale=None)
    return fn


def run_mhc(args, capture, evidence):
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_tilelang,
        mhc_pre_tilelang,
    )
    device = "cuda"
    T, hidden, hc = args.batch, args.hidden_size, args.hc_mult
    mix = (2 + hc) * hc
    residual = torch.randn(T, hc, hidden, dtype=torch.bfloat16, device=device)
    fn_w = torch.randn(mix, hc * hidden, dtype=torch.float32, device=device) * 0.01
    hc_scale = torch.rand(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(mix, dtype=torch.float32, device=device) * 0.1
    norm_w = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    eps, post_alpha, iters = 1e-6, 2.0, args.sinkhorn_iters
    from vllm.utils.deep_gemm import is_deep_gemm_supported
    evidence.update({"hidden_size": hidden, "hc_mult": hc, "sinkhorn_iters": iters,
                     "num_tokens": T, "mix_hc": mix,
                     "residual": {"shape": list(residual.shape), "dtype": "bfloat16"},
                     "fn": {"shape": list(fn_w.shape), "dtype": "float32"},
                     "fused_rmsnorm": True, "deep_gemm_supported": is_deep_gemm_supported(),
                     "post_alpha": post_alpha, "eps": eps})
    if args.op == "mhc_pre":
        evidence["public_op"] = ["vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang"]
    else:
        evidence["public_op"] = [
            "vllm.model_executor.kernels.mhc.tilelang.mhc_fused_post_pre_tilelang"]
        evidence["split_args"] = {"n_splits": 1, "tile_n": 1,
                                  "note": "function overrides for num_tokens <= 16"}
    if args.op == "mhc_pre" and VLLM_STACK == "vllm024":
        evidence["call_path"] = ("0.24 first layer: embedding repeated to "
                                 "[T, hc, hidden] -> mhc_pre_tilelang")

        def fn():
            mhc_pre_tilelang(residual, fn_w, hc_scale, hc_base, eps, eps, eps,
                             post_alpha, iters, norm_weight=norm_w, norm_eps=eps)
        return fn
    if args.op == "mhc_pre" and VLLM_STACK == "vllm030":
        from vllm.model_executor.kernels.mhc.tilelang import (
            mhc_pre_broadcast_tilelang,
        )
        embedding = torch.randn(T, hidden, dtype=torch.bfloat16, device=device)
        # DeepseekV4Model.finalize_mhc_broadcast_weights.
        fn_broadcast = fn_w.view(-1, hc, hidden).sum(dim=1).contiguous()
        evidence["public_op"] = [
            "vllm.model_executor.kernels.mhc.tilelang.mhc_pre_broadcast_tilelang"]
        evidence["call_path"] = ("0.30 first layer: 2-D embedding [T, hidden] -> "
                                 "mhc_pre_broadcast_tilelang(fn_broadcast)")
        evidence["residual"] = {"shape": list(embedding.shape), "dtype": "bfloat16"}
        evidence["fn_broadcast"] = {"shape": list(fn_broadcast.shape),
                                    "dtype": "float32"}

        def fn():
            mhc_pre_broadcast_tilelang(
                embedding, fn_w, hc_scale, hc_base, eps, eps, eps, post_alpha,
                iters, norm_weight=norm_w, norm_eps=eps,
                fn_broadcast=fn_broadcast)
        return fn
    if args.op == "mhc_pre":
        raise AssertionError(f"unhandled vLLM stack {VLLM_STACK!r}")
    evidence["call_path"] = f"{VLLM_STACK}: mhc_fused_post_pre_tilelang(n_splits=1, tile_n=1)"
    x = torch.randn(T, hidden, dtype=torch.bfloat16, device=device)
    post_mix = torch.rand(T, hc, 1, dtype=torch.float32, device=device)
    res_mix = torch.rand(T, hc, hc, dtype=torch.float32, device=device)

    def fn():
        mhc_fused_post_pre_tilelang(x, residual, post_mix, res_mix, fn_w,
                                    hc_scale, hc_base, eps, eps, eps,
                                    post_alpha, iters, n_splits=1, tile_n=1,
                                    norm_weight=norm_w, norm_eps=eps)
    return fn


OPS = {"decode": run_decode, "prefill": run_prefill,
       "compressor": run_compressor, "mhc_pre": run_mhc,
       "mhc_post_pre": run_mhc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True, choices=sorted(OPS))
    p.add_argument("--layer-type", choices=sorted(LAYER_RATIO), default=None)
    p.add_argument("--batch", type=int, required=True,
                   help="requests (decode/compressor) or tokens (mhc)")
    p.add_argument("--q-len", type=int, required=True)
    p.add_argument("--context", type=int, required=True,
                   help="decode: sequence length incl. new tokens; "
                        "prefill/compressor: prefix length")
    p.add_argument("--local-heads", type=int, default=None)
    p.add_argument("--padded-heads", type=int, default=None)
    p.add_argument("--index-topk", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--hc-mult", type=int, default=None)
    p.add_argument("--sinkhorn-iters", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--kv-block-size", type=int, default=None)
    p.add_argument("--nvtx_range_name", required=True)
    add_h200_capture_arguments(p)
    a = p.parse_args()
    required = {
        "decode": ("layer_type", "local_heads", "padded_heads", "index_topk",
                   "window", "max_model_len", "head_dim", "kv_block_size"),
        "prefill": ("layer_type", "local_heads", "padded_heads", "index_topk",
                    "window", "max_model_len", "head_dim", "kv_block_size"),
        "compressor": ("layer_type", "head_dim", "kv_block_size"),
        "mhc_pre": ("hidden_size", "hc_mult", "sinkhorn_iters"),
        "mhc_post_pre": ("hidden_size", "hc_mult", "sinkhorn_iters"),
    }[a.op]
    missing = [name for name in required if vars(a)[name] is None]
    if missing:
        raise ValueError(f"--op {a.op} requires {missing}")
    if min(a.batch, a.q_len) <= 0 or a.context < 0:
        raise ValueError("batch/q_len must be positive and context nonnegative")
    capture = open_h200_capture_session(a, a.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    torch.manual_seed(INPUT_SEED)
    torch.cuda.manual_seed(INPUT_SEED)
    evidence = {"vllm_stack": VLLM_STACK,
                "op": a.op, "layer_type": a.layer_type, "batch": a.batch,
                "q_len": a.q_len, "context": a.context,
                "local_heads": a.local_heads, "padded_heads": a.padded_heads,
                "index_topk": a.index_topk, "max_model_len": a.max_model_len}
    if a.op in ("decode", "prefill", "compressor"):
        if a.head_dim != HEAD_DIM or a.kv_block_size != KERNEL_BLOCK:
            raise ValueError(f"DeepSeek-V4 runner is head_dim {HEAD_DIM}, "
                             f"--block-size {KERNEL_BLOCK}")
    with torch.inference_mode():
        fn = OPS[a.op](a, capture, evidence)
        _measure(fn, a.nvtx_range_name, capture)
    if "sched_meta" in evidence:
        sched = evidence.pop("sched_meta")
        cfg = sched.config
        evidence["flash_mla_sched_meta"] = {
            "b": cfg.b, "s_q": cfg.s_q, "h_q": cfg.h_q,
            "page_block_size": cfg.page_block_size, "h_k": cfg.h_k,
            "causal": cfg.causal, "is_fp8_kvcache": cfg.is_fp8_kvcache,
            "topk": cfg.topk, "extra_page_block_size": cfg.extra_page_block_size,
            "extra_topk": cfg.extra_topk,
            "tile_scheduler_metadata_shape": list(sched.tile_scheduler_metadata.shape),
            "num_splits": sched.num_splits.tolist()}
    capture.add_launch_evidence("v4attn_case", evidence)
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_stack": VLLM_STACK,
    })
    capture.finish()
    print("V4 attention done")


if __name__ == "__main__":
    main()
