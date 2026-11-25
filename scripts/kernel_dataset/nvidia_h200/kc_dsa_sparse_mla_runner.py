"""NVIDIA H200 DeepSeek Sparse Attention (DSA) sparse-MLA attention runner.

Executes vLLM 0.24.0's ``FLASHMLA_SPARSE`` backend (the only sparse-MLA
backend vLLM selects on SM90, ``vllm/platforms/cuda.py``) for
``DeepseekV32ForCausalLM`` / ``GlmMoeDsaForCausalLM`` under the canonical
``nsys_nvtx_kernel_median`` protocol (``H200CaptureSession``).

Inside the single NVTX range the runner issues exactly the kernel sequence of
``FlashMLASparseImpl.forward_mqa``
(``vllm/v1/attention/backends/mla/flashmla_sparse.py``) for one attention
layer; the path is chosen by the same rules vLLM uses:

* ``ops.concat_mla_q(ql_nope, q_pe, q_concat)`` (forward_mqa receives the
  ``(ql_nope, q_pe)`` tuple from ``MLAAttention``);
* KV cache ``bf16`` (``--kv-cache-dtype auto``): ``_forward_bf16_kv`` --
  ``triton_convert_req_index_to_global_index(..., return_valid_counts=True)``,
  head padding to 64 (Hopper ``prefill_padding``) and
  ``flash_mla_sparse_fwd`` (FlashMLA sparse *prefill* kernel) for decode and
  prefill tokens alike;
* KV cache ``fp8_ds_mla`` (656 B/token) with ``num_heads < 32``
  (``MIN_HEADS_FOR_BF16_PREFILL``, i.e. TP8 for GLM-5.2 / DeepSeek-V3.2):
  ``_forward_fp8_kv_mixed_batch`` -- index conversion, zero-pad heads to 64
  and ``flash_mla_with_kvcache(is_fp8_kvcache=True, indices=...)`` (FlashMLA
  sparse *decode* kernel) over all tokens as one batch;
* KV cache ``fp8_ds_mla`` with ``num_heads >= 32``:
  ``_forward_fp8_kv_separate_prefill_decode`` -- decode requests
  (``q_len <= 1 + num_speculative_tokens``) run the sparse decode kernel with
  shape ``(B, q_len, H, 576)``; prefill requests run
  ``cp_gather_and_upconvert_fp8_kv_cache`` + ``flash_mla_sparse_fwd`` over the
  BF16 workspace.

Top-k indices are request-local token positions (what the indexer's top-k
kernels produce): distinct, uniformly random positions inside each token's
causal context (``context - q_len + j + 1`` for query token ``j``), padded
with -1 when the context is shorter than ``topk``.  The KV cache is a
scattered paged pool (block size 64) populated with the real cache writer
(``concat_and_cache_mla``) for ``fp8_ds_mla`` so the FP8 scales are valid.

``FlashMLASchedMeta`` is created once per case and initialised by the first
warm-up call; the NVTX call reuses it, as layers 2..L of one engine step do
(vLLM builds a fresh one per step in the metadata builder).

vLLM 0.30.0 (explicit branch on ``installed_vllm_stack()``; the chosen path
is recorded as ``vllm_stack`` / ``vllm_path`` in the evidence).  On SM90
``_get_backend_priorities`` (``vllm/platforms/cuda.py``) puts
``FLASH_ATTN_MLA_SPARSE`` before ``FLASHMLA_SPARSE`` for head size 576, and
``FlashAttnMLASparseBackend.supports_combination`` accepts only BF16/FP16 KV:

* KV ``bf16`` -> ``FlashAttnMLASparseImpl.forward_mqa``
  (``v1/attention/backends/mla/flashattn_mla_sparse.py``): no Q concat;
  ``triton_convert_req_index_to_global_index(..., BLOCK_STRIDE_ROWS=64,
  return_valid_counts=True)`` over the flat row view of the cache, then the
  FA3 ``flash_attn_varlen_func(q=q_pe, k=rope rows, v=latent rows,
  q_v=ql_nope, cu_seqlens_q=arange(T+1), max_seqlen_q=1,
  max_seqlen_k=topk, seqused_k=valid_counts, block_table=indices,
  causal=True, fa_version=3)`` (page size 1, one query per "sequence").
* KV ``fp8_ds_mla`` with ``num_heads < 32`` -> ``FLASHMLA_SPARSE``
  ``_forward_fp8_kv_mixed_batch``: as on 0.24, except that a decode-only
  batch converts the indices through the layer's ``SparseMLAIndexGroup``
  (``v1/attention/backends/mla/index_group.py``: the same Triton conversion
  kernel, launched on the group's side stream after the indexer's
  ``logical_topk_ready`` event, the compute stream waiting on
  ``physical_topk_ready``).  ``q_pe`` is the contiguous RoPE output of
  ``fused_q`` (``models/deepseek_v32/common/kernels.py``).
* ``fp8_ds_mla`` with ``num_heads >= 32`` (separate prefill/decode) is not
  ported to 0.30 (no plan uses it) and fails loudly.

0.30 routes a prefill batch whose longest sequence is <= ``index_topk`` to
dense MHA (``_use_dense_mha_prefill``, ``sparse_mla_attention.py``) unless
``--sparse-mla-force-mqa``; such cases still measure the sparse MQA call and
record ``vllm030_default_prefill_route``.
"""

import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    H200_CLOCK_SETTLE_MAXIMUM_QUERIES,
    H200_REQUIRED_SM_CLOCK_MHZ,
    add_h200_capture_arguments,
    installed_vllm_stack,
    open_h200_capture_session,
    query_sm_clock,
)

import torch
from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    MIN_HEADS_FOR_BF16_PREFILL,
    FlashMLASparseImpl,
    get_prefill_workspace_size,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)

VLLM_STACK = installed_vllm_stack()
if VLLM_STACK == "vllm030":
    from vllm.v1.attention.backends.mla.index_group import (
        SparseMLAIndexGroupBuilder,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
    from vllm.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func,
    )
elif VLLM_STACK != "vllm024":
    raise RuntimeError(f"unhandled vLLM stack {VLLM_STACK!r}")

BLOCK_SIZE = 64          # FlashMLASparseBackend.get_supported_kernel_block_sizes
KV_LORA_RANK = 512       # FlashMLA sparse: head_dim_v must be 512
ROPE_DIM = 64
HEAD_SIZE = KV_LORA_RANK + ROPE_DIM   # 576
FP8_DS_MLA_BYTES = 656   # 512 fp8 + 4 fp32 scales + 64 bf16 rope
HOPPER_PREFILL_PADDING = 64
WARMUP = 5
FILL_CHUNK_TOKENS = 1 << 20
INDEX_SAMPLE_ELEMENTS = 1 << 27


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer: {value!r}")


def _paged_block_table(batch, context, max_model_len, device):
    """Scattered pool: every request owns cdiv(context, 64) random blocks."""
    blocks_per_req = (context + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool_blocks = batch * blocks_per_req + 1
    perm = torch.randperm(pool_blocks - 1, device=device, dtype=torch.int64)
    block_table = torch.zeros(batch, max_blocks, dtype=torch.int32,
                              device=device)
    block_table[:, :blocks_per_req] = perm.view(
        batch, blocks_per_req).to(torch.int32)
    return block_table, pool_blocks


def _fill_fp8_ds_mla_cache(kv_cache, device):
    """Populate every slot with the real FP8 cache writer (valid scales)."""
    num_slots = kv_cache.shape[0] * kv_cache.shape[1]
    scale = torch.ones(1, dtype=torch.float32, device=device)
    for start in range(0, num_slots, FILL_CHUNK_TOKENS):
        count = min(FILL_CHUNK_TOKENS, num_slots - start)
        kv_c = torch.randn(count, KV_LORA_RANK, dtype=torch.bfloat16,
                           device=device)
        k_pe = torch.randn(count, ROPE_DIM, dtype=torch.bfloat16,
                           device=device)
        slots = torch.arange(start, start + count, dtype=torch.int64,
                             device=device)
        ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, "fp8_ds_mla",
                                 scale)


def _request_local_topk(batch, q_len, context, topk, device):
    """[B*q_len, topk] int32 distinct random causal positions, -1 padded."""
    num_tokens = batch * q_len
    visible = (context - q_len + 1
               + torch.arange(q_len, device=device, dtype=torch.int64)
               ).repeat(batch)
    out = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    rows_per_chunk = max(1, INDEX_SAMPLE_ELEMENTS // context)
    positions = torch.arange(context, device=device, dtype=torch.int64)
    for start in range(0, num_tokens, rows_per_chunk):
        stop = min(num_tokens, start + rows_per_chunk)
        scores = torch.rand(stop - start, context, device=device)
        scores.masked_fill_(positions[None, :] >= visible[start:stop, None],
                            -1.0)
        if context >= topk:
            values, index = scores.topk(topk, dim=1)
        else:
            values, index = scores.topk(context, dim=1)
            pad = torch.full((stop - start, topk - context), -1.0,
                             device=device)
            values = torch.cat([values, pad], dim=1)
            index = torch.cat([index, pad.to(torch.int64)], dim=1)
        index = torch.where(values < 0, torch.full_like(index, -1), index)
        out[start:stop] = index.to(torch.int32)
    valid = out >= 0
    if bool((out[valid].to(torch.int64)
             >= visible[:, None].expand_as(out)[valid]).any()):
        raise RuntimeError("generated top-k index outside the causal context")
    expected = torch.clamp(visible, max=topk)
    if not torch.equal(valid.sum(dim=1), expected):
        raise RuntimeError("generated top-k valid counts mismatch")
    return out, expected


def _sched_evidence(sched_meta):
    """FlashMLA tile scheduler metadata the NVTX call consumed."""
    if not sched_meta.have_initialized:
        raise RuntimeError("FlashMLASchedMeta was not initialised")
    return {"scheduler_metadata": {
        "config": dataclasses.asdict(sched_meta.config),
        "tile_scheduler_metadata_shape": list(
            sched_meta.tile_scheduler_metadata.shape),
        "tile_scheduler_metadata": (
            sched_meta.tile_scheduler_metadata.cpu().tolist()),
        "num_splits": sched_meta.num_splits.cpu().tolist(),
        "reused_from_warmup": True,
    }}


def build_case(batch, q_len, context, num_heads, qk_nope_head_dim,
               kv_cache_dtype, topk, max_model_len, num_speculative_tokens,
               device="cuda"):
    for name, value in (("batch", batch), ("q_len", q_len),
                        ("context", context), ("num_heads", num_heads),
                        ("qk_nope_head_dim", qk_nope_head_dim),
                        ("topk", topk), ("max_model_len", max_model_len)):
        _positive(name, value)
    if q_len > context:
        raise ValueError("q_len must not exceed context")
    if context > max_model_len:
        raise ValueError("context must not exceed max_model_len")
    if topk % 128 != 0:
        raise ValueError("triton index conversion requires topk % 128 == 0")
    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be nonnegative")
    num_tokens = batch * q_len
    is_decode = q_len <= 1 + num_speculative_tokens
    scale = float((qk_nope_head_dim + ROPE_DIM) ** -0.5)

    block_table, pool_blocks = _paged_block_table(batch, context,
                                                  max_model_len, device)
    if kv_cache_dtype == "fp8_ds_mla":
        kv_cache = torch.empty(pool_blocks, BLOCK_SIZE, FP8_DS_MLA_BYTES,
                               dtype=torch.uint8, device=device)
        _fill_fp8_ds_mla_cache(kv_cache, device)
    elif kv_cache_dtype == "bf16":
        kv_cache = torch.randn(pool_blocks, BLOCK_SIZE, HEAD_SIZE,
                               dtype=torch.bfloat16, device=device)
    else:
        raise ValueError(f"unknown kv_cache_dtype {kv_cache_dtype!r}")

    topk_local, valid_counts = _request_local_topk(batch, q_len, context,
                                                   topk, device)
    req_id_per_token = torch.arange(
        batch, device=device, dtype=torch.int32).repeat_interleave(q_len)
    # MLAAttention hands forward_mqa (ql_nope, q_pe): ql_nope is the
    # (N, B, L)->(B, N, L) transposed W_UK bmm output, q_pe a slice of q.
    ql_nope = torch.randn(num_heads, num_tokens, KV_LORA_RANK,
                          dtype=torch.bfloat16, device=device).transpose(0, 1)
    q_full = torch.randn(num_tokens, num_heads, qk_nope_head_dim + ROPE_DIM,
                         dtype=torch.bfloat16, device=device)
    q_pe = q_full[..., qk_nope_head_dim:]
    q_concat = torch.empty(num_tokens, num_heads, HEAD_SIZE,
                           dtype=torch.bfloat16, device=device)
    evidence = {
        "batch": batch, "q_len": q_len, "context": context,
        "num_tokens": num_tokens, "num_heads": num_heads,
        "kv_cache_dtype": kv_cache_dtype, "topk": topk,
        "max_model_len": max_model_len,
        "num_speculative_tokens": num_speculative_tokens,
        "split_class": "decode" if is_decode else "prefill",
        "block_size": BLOCK_SIZE, "pool_blocks": pool_blocks,
        "kv_cache_shape": list(kv_cache.shape),
        "kv_cache_torch_dtype": str(kv_cache.dtype),
        "softmax_scale": scale,
        "q_lens": [q_len] * batch,
        "context_lens": [context] * batch,
        "per_token_context_lens": [context - q_len + 1 + j
                                   for _ in range(batch)
                                   for j in range(q_len)],
        "index_topk": topk,
        "selected_indices_per_token": valid_counts.tolist(),
        "num_heads_per_rank": num_heads,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": ROPE_DIM,
        "kv_lora_rank": KV_LORA_RANK,
        "kernel_head_dim_qk": HEAD_SIZE,
        "kernel_head_dim_v": KV_LORA_RANK,
        "kv_cache_layout": (
            "uint8 [num_blocks, 64, 656]: 512 fp8_e4m3 NoPE + 4 fp32 "
            "scales (per 128) + 64 bf16 RoPE per token"
            if kv_cache_dtype == "fp8_ds_mla" else
            "bf16 [num_blocks, 64, 576]: 512 NoPE + 64 RoPE per token"),
        "block_table_layout": (
            "int32 [batch, cdiv(max_model_len, 64)], first cdiv(context, 64)"
            " columns = distinct random pool blocks"),
        "topk_index_generation": (
            "request-local positions, distinct uniform random inside each "
            "token's causal context, -1 padded (seed 42)"),
    }

    evidence["vllm_stack"] = VLLM_STACK
    if VLLM_STACK == "vllm030":
        return _build_case_v030(
            evidence, batch=batch, q_len=q_len, context=context,
            num_tokens=num_tokens, num_heads=num_heads, topk=topk,
            max_model_len=max_model_len, is_decode=is_decode, scale=scale,
            kv_cache=kv_cache, kv_cache_dtype=kv_cache_dtype,
            block_table=block_table, topk_local=topk_local,
            req_id_per_token=req_id_per_token, ql_nope=ql_nope,
            q_pe=q_full[..., qk_nope_head_dim:].contiguous(), device=device)

    if kv_cache_dtype == "bf16":
        padded = (num_heads if num_heads % HOPPER_PREFILL_PADDING == 0
                  else HOPPER_PREFILL_PADDING)
        if num_heads % HOPPER_PREFILL_PADDING != 0 and (
                HOPPER_PREFILL_PADDING % num_heads != 0):
            raise ValueError("bf16 sparse prefill needs 64 % num_heads == 0")
        kv_flat = kv_cache.view(-1, 1, HEAD_SIZE)

        def attention():
            ops.concat_mla_q(ql_nope, q_pe, q_concat)
            topk_global, topk_length = triton_convert_req_index_to_global_index(
                req_id_per_token, block_table, topk_local,
                BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=topk,
                return_valid_counts=True)
            q = q_concat
            if num_heads != padded:
                q_padded = q.new_empty((num_tokens, padded, HEAD_SIZE))
                q_padded[:, :num_heads, :] = q
                q = q_padded
            output = flash_mla_sparse_fwd(
                q, kv_flat, topk_global.view(num_tokens, 1, -1), scale,
                topk_length=topk_length)[0]
            return output[:, :num_heads, :]

        evidence["vllm_path"] = "_forward_bf16_kv"
        evidence["kernel_heads"] = padded
        evidence["public_ops"] = [
            "vllm._custom_ops.concat_mla_q(ql_nope, q_pe, q)",
            "triton_convert_req_index_to_global_index(req_id_per_token, "
            "block_table, topk_indices, BLOCK_SIZE=64, NUM_TOPK_TOKENS=topk, "
            "return_valid_counts=True)",
            f"q.new_empty((T, {padded}, 576)); q_padded[:, :H] = q"
            if padded != num_heads else "no head padding",
            "flash_mla_sparse_fwd(q, kv_cache.view(-1, 1, 576), "
            "indices.view(T, 1, topk), softmax_scale, topk_length=topk_length)",
        ]
        evidence["scheduler_metadata"] = (
            "none: flash_mla_sparse_fwd takes no tile scheduler metadata")
        return attention, evidence, lambda: {}

    padded = FlashMLASparseImpl._compute_fp8_decode_padded_heads(num_heads)
    fp8_cache = kv_cache.view(torch.uint8).unsqueeze(-2)
    max_len_tensor = torch.full((batch,), max_model_len, dtype=torch.int32,
                                device=device)
    dummy_block_table = torch.empty((batch, 1), dtype=torch.int32,
                                    device=device)

    if num_heads < MIN_HEADS_FOR_BF16_PREFILL:
        sched_meta, _ = get_mla_metadata()

        def attention():
            ops.concat_mla_q(ql_nope, q_pe, q_concat)
            topk_global = triton_convert_req_index_to_global_index(
                req_id_per_token, block_table, topk_local,
                BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=topk)
            q = q_concat.unsqueeze(0)
            if num_heads < padded:
                q_padded = q.new_zeros((1, num_tokens, padded, HEAD_SIZE))
                q_padded[:, :, :num_heads, :] = q
                q = q_padded
            out, _ = flash_mla_with_kvcache(
                q=q, k_cache=fp8_cache, block_table=dummy_block_table[:1],
                head_dim_v=KV_LORA_RANK, cache_seqlens=max_len_tensor[:1],
                tile_scheduler_metadata=sched_meta, is_fp8_kvcache=True,
                indices=topk_global.unsqueeze(0), softmax_scale=scale)
            return out[:, :, :num_heads, :].squeeze(0)

        evidence["vllm_path"] = "_forward_fp8_kv_mixed_batch"
        evidence["kernel_heads"] = padded
        evidence["public_ops"] = [
            "vllm._custom_ops.concat_mla_q(ql_nope, q_pe, q)",
            "triton_convert_req_index_to_global_index(req_id_per_token, "
            "block_table, topk_indices, BLOCK_SIZE=64, NUM_TOPK_TOKENS=topk)",
            f"q.new_zeros((1, T, {padded}, 576)); q_padded[:, :, :H] = q"
            if padded != num_heads else "no head padding",
            "flash_mla_with_kvcache(q[1, T, Hpad, 576], k_cache=kv_cache."
            "view(uint8).unsqueeze(-2), block_table=dummy[1, 1], "
            "head_dim_v=512, cache_seqlens=[max_model_len], "
            "tile_scheduler_metadata=FlashMLASchedMeta, is_fp8_kvcache=True, "
            "indices[1, T, topk], softmax_scale)",
        ]
        return attention, evidence, lambda: _sched_evidence(sched_meta)

    if is_decode:
        sched_meta, _ = get_mla_metadata()

        def attention():
            ops.concat_mla_q(ql_nope, q_pe, q_concat)
            topk_global, _ = triton_convert_req_index_to_global_index(
                req_id_per_token, block_table, topk_local,
                BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=topk,
                return_valid_counts=True)
            q = q_concat.view(batch, q_len, num_heads, HEAD_SIZE)
            if num_heads < padded:
                q_padded = q.new_zeros((batch, q_len, padded, HEAD_SIZE))
                q_padded[:, :, :num_heads, :] = q
                q = q_padded
            out, _ = flash_mla_with_kvcache(
                q=q, k_cache=fp8_cache, block_table=dummy_block_table,
                head_dim_v=KV_LORA_RANK, cache_seqlens=max_len_tensor,
                tile_scheduler_metadata=sched_meta, is_fp8_kvcache=True,
                indices=topk_global.view(batch, q_len, -1),
                softmax_scale=scale)
            return out[:, :, :num_heads, :].reshape(num_tokens, num_heads,
                                                    KV_LORA_RANK)

        evidence["vllm_path"] = "_forward_fp8_kv_separate_prefill_decode:decode"
        evidence["kernel_heads"] = padded
        evidence["public_ops"] = [
            "vllm._custom_ops.concat_mla_q(ql_nope, q_pe, q)",
            "triton_convert_req_index_to_global_index(..., "
            "return_valid_counts=True)",
            "flash_mla_with_kvcache(q[B, q_len, Hpad, 576], k_cache, "
            "block_table=dummy[B, 1], head_dim_v=512, cache_seqlens="
            "[max_model_len]*B, tile_scheduler_metadata, is_fp8_kvcache=True,"
            " indices[B, q_len, topk], softmax_scale)",
        ]
        return attention, evidence, lambda: _sched_evidence(sched_meta)

    if num_heads % HOPPER_PREFILL_PADDING != 0:
        raise ValueError("fp8 separate prefill needs num_heads % 64 == 0")
    total_seq = batch * context
    if total_seq > get_prefill_workspace_size(max_model_len):
        raise ValueError("prefill batch exceeds one FlashMLA workspace chunk")
    workspace = torch.empty(total_seq, HEAD_SIZE, dtype=torch.bfloat16,
                            device=device)
    seq_lens = torch.full((batch,), context, dtype=torch.int32, device=device)
    workspace_starts = torch.arange(batch, dtype=torch.int32,
                                    device=device) * context
    request_ids = req_id_per_token.clone()
    workspace_flat = workspace.view(-1, 1, HEAD_SIZE)

    def attention():
        ops.concat_mla_q(ql_nope, q_pe, q_concat)
        topk_ws, topk_length = triton_convert_req_index_to_global_index(
            req_id_per_token, block_table, topk_local,
            BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=topk,
            HAS_PREFILL_WORKSPACE=True,
            prefill_workspace_request_ids=request_ids,
            prefill_workspace_starts=workspace_starts,
            return_valid_counts=True)
        attn_out = q_concat.new_empty((num_tokens, num_heads, KV_LORA_RANK))
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            kv_cache, workspace, block_table, seq_lens, workspace_starts,
            batch)
        attn_out[:] = flash_mla_sparse_fwd(
            q_concat, workspace_flat, topk_ws.view(num_tokens, 1, -1), scale,
            topk_length=topk_length)[0][:, :num_heads, :]
        return attn_out

    evidence["vllm_path"] = "_forward_fp8_kv_separate_prefill_decode:prefill"
    evidence["kernel_heads"] = num_heads
    evidence["prefill_workspace_tokens"] = total_seq
    evidence["public_ops"] = [
        "vllm._custom_ops.concat_mla_q(ql_nope, q_pe, q)",
        "triton_convert_req_index_to_global_index(..., "
        "HAS_PREFILL_WORKSPACE=True, return_valid_counts=True)",
        "vllm._custom_ops.cp_gather_and_upconvert_fp8_kv_cache(kv_cache, "
        "workspace, block_table, seq_lens, workspace_starts, batch)",
        "flash_mla_sparse_fwd(q, workspace.view(-1, 1, 576), "
        "indices.view(T, 1, topk), softmax_scale, topk_length=topk_length)",
    ]
    evidence["scheduler_metadata"] = (
        "none: flash_mla_sparse_fwd takes no tile scheduler metadata")
    return attention, evidence, lambda: {}


def _build_case_v030(evidence, *, batch, q_len, context, num_tokens,
                     num_heads, topk, max_model_len, is_decode, scale,
                     kv_cache, kv_cache_dtype, block_table, topk_local,
                     req_id_per_token, ql_nope, q_pe, device):
    """vLLM 0.30.0 SM90 sparse-MLA call path (see module docstring)."""
    evidence["q_pe_layout"] = (
        "contiguous [T, H, 64] bf16 (fused_q RoPE output); ql_nope is the "
        "transposed W_UK bmm view [T, H, 512]")
    evidence["vllm030_default_prefill_route"] = (
        "n/a (decode tokens)" if is_decode else
        "dense_mha (_use_dense_mha_prefill: prefill_max_seq_len "
        f"{context} <= index_topk {topk}); this case measures the "
        "--sparse-mla-force-mqa sparse MQA call" if context <= topk else
        "sparse_mqa (prefill_max_seq_len > index_topk)")
    if kv_cache_dtype == "bf16":
        kv_rows, block_stride_rows = flat_kv_row_view(kv_cache, BLOCK_SIZE)

        def attention():
            topk_global, valid_counts = (
                triton_convert_req_index_to_global_index(
                    req_id_per_token, block_table, topk_local,
                    BLOCK_SIZE=BLOCK_SIZE,
                    BLOCK_STRIDE_ROWS=block_stride_rows,
                    NUM_TOPK_TOKENS=topk, return_valid_counts=True))
            cu_seqlens_q = torch.arange(0, q_pe.shape[0] + 1,
                                        dtype=torch.int32, device=device)
            k_rows = kv_rows[:, KV_LORA_RANK:].unsqueeze(1).unsqueeze(1)
            v_rows = kv_rows[:, :KV_LORA_RANK].unsqueeze(1).unsqueeze(1)
            return flash_attn_varlen_func(
                q=q_pe, k=k_rows, v=v_rows, q_v=ql_nope, max_seqlen_q=1,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=topk_global.shape[1], seqused_k=valid_counts,
                block_table=topk_global, softmax_scale=scale, causal=True,
                fa_version=3)

        evidence["vllm_path"] = (
            "FLASH_ATTN_MLA_SPARSE:FlashAttnMLASparseImpl.forward_mqa")
        evidence["kernel_heads"] = num_heads
        evidence["flat_kv_block_stride_rows"] = block_stride_rows
        evidence["public_ops"] = [
            "triton_convert_req_index_to_global_index(req_id_per_token, "
            "block_table, topk_indices, BLOCK_SIZE=64, BLOCK_STRIDE_ROWS=64, "
            "NUM_TOPK_TOKENS=topk, return_valid_counts=True)",
            "torch.arange(0, T + 1) (cu_seqlens_q)",
            "vllm.vllm_flash_attn flash_attn_varlen_func(q=q_pe[T,H,64], "
            "k=kv_rows[:, 512:][rows,1,1,64], v=kv_rows[:, :512][rows,1,1,"
            "512], q_v=ql_nope, max_seqlen_q=1, max_seqlen_k=topk, "
            "seqused_k=valid_counts, block_table=indices[T, topk], "
            "softmax_scale, causal=True, fa_version=3)",
        ]
        evidence["scheduler_metadata"] = (
            "none: FA3 sparse call passes no scheduler_metadata")
        return attention, evidence, lambda: {}

    if num_heads >= MIN_HEADS_FOR_BF16_PREFILL:
        raise ValueError(
            "vLLM 0.30 fp8_ds_mla separate prefill/decode path "
            "(num_heads >= 32) is not ported in this runner")
    padded = FlashMLASparseImpl._compute_fp8_decode_padded_heads(num_heads)
    fp8_cache = kv_cache.view(torch.uint8).unsqueeze(-2)
    cache_lens = torch.full((1,), max_model_len, dtype=torch.int32,
                            device=device)
    dummy_block_table = torch.empty((1, 1), dtype=torch.int32, device=device)
    # FlashMLASparseImpl.q_concat_buffer: quantized KV keeps num_heads.
    q_concat_buffer = torch.empty(num_tokens, num_heads, HEAD_SIZE,
                                  dtype=torch.bfloat16, device=device)
    sched_meta, _ = get_mla_metadata()
    decode_only = is_decode
    if decode_only:
        group, group_index = SparseMLAIndexGroupBuilder(
            topk_local, max_decode_rows=num_tokens).register_layer(True)
        group_metadata = type("SparseMLAIndexMetadata", (), {
            "req_id_per_token": req_id_per_token,
            "block_table": block_table, "block_size": BLOCK_SIZE})()

    def attention():
        q = q_concat_buffer[:num_tokens]
        ops.concat_mla_q(ql_nope, q_pe, q)
        topk_indices = topk_local[:num_tokens]
        if decode_only:
            group.set_logical_topk_ready(group_index)
            topk_global = group.convert_logical_to_physical_topk(
                group_index, topk_indices, group_metadata,
                block_stride_rows=None, return_valid_counts=False)
        else:
            topk_global = triton_convert_req_index_to_global_index(
                req_id_per_token[:num_tokens], block_table, topk_indices,
                BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=topk)
        q = q.unsqueeze(0)
        if num_heads < padded:
            q_padded = q.new_zeros((1, num_tokens, padded, HEAD_SIZE))
            q_padded[:, :, :num_heads, :] = q
            q = q_padded
        out, _ = flash_mla_with_kvcache(
            q=q, k_cache=fp8_cache, block_table=dummy_block_table,
            head_dim_v=KV_LORA_RANK, cache_seqlens=cache_lens,
            tile_scheduler_metadata=sched_meta, is_fp8_kvcache=True,
            indices=topk_global.unsqueeze(0), softmax_scale=scale)
        return out[:, :, :num_heads, :].squeeze(0)

    conversion = (
        "SparseMLAIndexGroup.convert_logical_to_physical_topk: "
        "triton_convert_req_index_to_global_index(..., out=group workspace) "
        "on the group side stream (decode-only batch)" if decode_only else
        "triton_convert_req_index_to_global_index(req_id_per_token, "
        "block_table, topk_indices, BLOCK_SIZE=64, NUM_TOPK_TOKENS=topk)")
    evidence["vllm_path"] = (
        "FLASHMLA_SPARSE:_forward_fp8_kv_mixed_batch:"
        + ("index_group_decode_only" if decode_only else "direct_conversion"))
    evidence["kernel_heads"] = padded
    evidence["index_group"] = (
        {"workspace_rows": num_tokens, "side_stream": True,
         "has_indexer": True} if decode_only else None)
    evidence["public_ops"] = [
        "vllm._custom_ops.concat_mla_q(ql_nope, q_pe, q_concat_buffer[:T])",
        conversion,
        f"q.new_zeros((1, T, {padded}, 576)); q_padded[:, :, :H] = q"
        if padded != num_heads else "no head padding",
        "flash_mla_with_kvcache(q[1, T, Hpad, 576], k_cache=kv_cache."
        "view(uint8).unsqueeze(-2), block_table=dummy[1, 1], "
        "head_dim_v=512, cache_seqlens=[max_model_len], "
        "tile_scheduler_metadata=FlashMLASchedMeta, is_fp8_kvcache=True, "
        "indices[1, T, topk], softmax_scale)",
    ]
    return attention, evidence, lambda: _sched_evidence(sched_meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--q-len", type=int, required=True,
                        help="query tokens per request (1+MTP for decode)")
    parser.add_argument("--context", type=int, required=True,
                        help="KV length per request including the new tokens")
    parser.add_argument("--num-heads", type=int, required=True,
                        help="attention heads per rank")
    parser.add_argument("--qk-nope-head-dim", type=int, required=True)
    parser.add_argument("--kv-cache-dtype", required=True,
                        choices=["bf16", "fp8_ds_mla"])
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--num-speculative-tokens", type=int, required=True)
    parser.add_argument("--nvtx_range_name", type=str, required=True)
    add_h200_capture_arguments(parser)
    args = parser.parse_args()
    capture = open_h200_capture_session(args, args.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    torch.manual_seed(42)
    attention, evidence, runtime_evidence = build_case(
        args.batch, args.q_len, args.context, args.num_heads,
        args.qk_nope_head_dim, args.kv_cache_dtype, args.topk,
        args.max_model_len, args.num_speculative_tokens)
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(WARMUP):
        attention()
    torch.cuda.synchronize()
    # Multi-millisecond warm-up calls can trip the SW power cap like the
    # GEMM power warm-up does; idle until the boost clock is back (same
    # settle rule as H200CaptureSession.power_warmup) so the NVTX call
    # starts at 1980 MHz.  The samples are recorded; the campaign gate
    # still rejects the window if the clock is low at NVTX push.
    settle_samples = []
    for _ in range(H200_CLOCK_SETTLE_MAXIMUM_QUERIES):
        sample = query_sm_clock(args.h200_physical_gpu)
        settle_samples.append(sample)
        if sample["sm_clock_mhz"] >= H200_REQUIRED_SM_CLOCK_MHZ:
            break
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(args.nvtx_range_name)
    output = attention()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("sparse MLA output contains non-finite values")
    evidence.update(runtime_evidence())
    if len(evidence["q_lens"]) != args.batch or len(
            evidence["selected_indices_per_token"]) != (
            args.batch * args.q_len):
        raise RuntimeError("evidence list lengths disagree with the case")
    evidence["post_warmup_clock_settle_samples"] = settle_samples
    capture.add_launch_evidence("dsa_sparse_mla", evidence)
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "warmup_calls": WARMUP,
    })
    capture.finish()
    print("DSA sparse MLA done " + json.dumps(evidence["vllm_path"]))


if __name__ == "__main__":
    main()
