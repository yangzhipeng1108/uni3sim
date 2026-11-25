"""NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/flash_attn_runner.py``.

Executes the H200 deployment stack (vLLM 0.24.0 ``vllm.vllm_flash_attn``
FlashAttention-3 / torch 2.11 in the caller-selected runtime)
under the A800 canonical NSYS NVTX timing protocol.

vLLM 0.11/A800 -> vLLM 0.24/H200 op mapping (the H200 deployment's backend
selection on SM90: ``FLASH_ATTN`` with ``fa_version=3`` for MHA/GQA and
``FLASH_ATTN_MLA`` for MLA, see ``vllm/platforms/cuda.py``):

* paged / paged_varlen / chunked_prefill: A800 vLLM FA2
  ``flash_attn_varlen_func(..., block_table, num_splits=1)`` -> H200
  ``flash_attn_varlen_func(..., block_table, fa_version=3, num_splits=0,
  scheduler_metadata=get_scheduler_metadata(...))``, exactly the eager
  ``FlashAttentionImpl.forward`` call (``num_splits=0`` = FA3 heuristics;
  scheduler metadata is built before the NVTX range like the metadata
  builder's AOT schedule).
* contiguous decode (forced split): A800 Dao ``flash_attn_with_kvcache(...,
  num_splits=SP)`` (the Dao package is absent on the H200 stack) -> H200 FA3
  ``flash_attn_varlen_func`` over dense per-request K/V with ``cu_seqlens_k``
  and the forced ``num_splits=SP`` as executed input.
* contiguous prefill: A800 Dao ``flash_attn_func`` -> H200 FA3
  ``flash_attn_varlen_func`` over dense K/V with ``cu_seqlens_k``,
  ``num_splits=0``.
* mla: A800 Ampere Triton fallback ``decode_attention_fwd`` (fixed 4-way
  split) -> H200 ``FLASH_ATTN_MLA`` decode: FA3 ``flash_attn_varlen_func(q=q_pe,
  k=k_pe_cache, v=kv_c_cache, q_v=q_nope, block_table, causal=True,
  fa_version=3, num_splits=0, scheduler_metadata)`` with kv_lora_rank=512,
  rope=64, block size 16 (``FlashAttnMLABackend`` supports multiples of 16).
  The A800 ``SP_REQUESTED=4`` Triton scratch split has no H200 counterpart.
  Uniform ``Q_LEN > 1`` (MTP / speculative verify) runs the same decode call
  with ``max_seqlen_q=Q_LEN``: ``FlashAttnMLAMetadataBuilder`` declares
  ``QueryLenSupport.VARLEN`` with ``reorder_batch_threshold=512``, so
  ``split_decodes_and_prefills`` classifies every request with
  ``q_len <= 512`` as a decode, ``_build_decode`` passes the batch's
  ``max_query_len`` to ``get_scheduler_metadata(causal=True)`` and
  ``FlashAttnMLAImpl.forward_mqa`` calls FA3 with the same
  ``max_seqlen_q`` over ``B*Q_LEN`` query rows (``seqused_k`` = context +
  query tokens).  The MLA branch never forwards ``--num_splits`` (it runs
  the builder's ``max_num_splits=0`` of eager / non-full-CUDA-graph steps,
  as for Q_LEN=1), so the prefill ``num_splits=1`` requirement does not
  apply to it.

vLLM 0.30 (``installed_vllm_stack() == "vllm030"``): SM90 backend priority
is unchanged (``FLASH_ATTN`` first for MHA/GQA, ``FLASH_ATTN_MLA`` first for
dense MLA), ``get_flash_attn_version`` still picks FA3 on Hopper for these
head sizes (the FA3->FA4 upgrade applies only to head_size > 256, sinks with
diff-KV or diffusion models), ``FlashAttentionImpl.forward`` reaches
``flash_attn_varlen_func`` through the pass-through
``FA4DenseAttentionKernel`` wrapper with the same arguments, and
``vllm.vllm_flash_attn.flash_attn_varlen_func`` / ``get_scheduler_metadata``
keep their FA3 signatures (0.30 only adds FA4 keyword arguments).  Every
layout therefore executes the same call on both stacks; the stack is recorded
in the ``environment`` evidence.

FA3 split evidence: after the NVTX range (and after the post-range clock
query), the runner copies the FA3 scheduler metadata that decided the split
count to the CPU and records it as ``fa3_scheduler_metadata``.  For
paged / paged_varlen / chunked_prefill / mla this is the exact int32 tensor
passed as ``scheduler_metadata`` to the measured call.  The contiguous call
builds its metadata inside ``mha_fwd``, so the runner recomputes it with
``get_scheduler_metadata`` from the executed arguments (outside the range).
Layout (``mha_fwd_get_scheduler_metadata`` in the pinned FA3 source of each
stack, identical for both): ``b_rounded = round_up(batch, 4)`` int32 blocks
``prepare_seqlen_q[b_rounded]``, then ``num_splits_dynamic[b_rounded]`` only
when the static split bound is > 1, then ``num_nheads_in_l2[b_rounded]`` only
when the normalized call is causal/local, then one tile semaphore.

Other changes (hardware coupling only): only the NVTX (normal) mode is kept
(the device-Event mode is not ported); ``H200CaptureSession`` adds the GPU
lock, H200 identity check, BF16 GEMM power warm-up before the runner's
WARMUP=5 loop and SM-clock queries around the NVTX range; the optional
pre-NVTX sweep minimum is 480 MiB (8 x 60 MiB L2).  Unchanged: shapes,
per-request Q/KV lengths, scattered paged pool construction, block size 16,
WARMUP=5, one exact public-op call inside the NVTX range.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    H200_MINIMUM_CACHE_EVICTION_MIB,
    H200_HARDWARE_ID,
    add_h200_capture_arguments,
    installed_vllm_stack,
    open_h200_capture_session,
)

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

from evaluation.kernel.device_event import (
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
)

DTYPE_MAP = {
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}

BLOCK_SIZE = 16
WARMUP = 5
FA_VERSION = 3
MLA_ROPE_DIM = 64
# vLLM FlashAttnMLAMetadataBuilder.reorder_batch_threshold: requests with a
# longer query take the MLA prefill (MHA) path, not this decode call.
MLA_DECODE_MAX_QUERY_LEN = 512
# Pinned FA3 source (vLLM cmake/external_projects/vllm_flash_attn.cmake GIT_TAG)
# defining the scheduler-metadata layout recorded per window.
FA3_SCHEDULER_METADATA_SOURCE = {
    "vllm024": ("vllm-project/flash-attention@"
                "803020a8fa15407871341d41eba4919ade2ee1ee "
                "hopper/flash_api.cpp mha_fwd_get_scheduler_metadata"),
    "vllm030": ("vllm-project/flash-attention@"
                "506341a143fcabd4bb79052a7605ada727d6b3f5 "
                "hopper/flash_api.cpp mha_fwd_get_scheduler_metadata"),
}
# FA3 call form per supported vLLM stack (identical on both).
VLLM_CALL_PATH = {
    "vllm024": "vllm.vllm_flash_attn FA3 flash_attn_varlen_func (vLLM 0.24)",
    "vllm030": ("vllm.vllm_flash_attn FA3 flash_attn_varlen_func (vLLM 0.30; "
                "FlashAttentionImpl reaches it via FA4DenseAttentionKernel "
                "pass-through with fa_version=3)"),
}


def _validate_lengths(name, values, batch, maximum):
    if not isinstance(values, list):
        raise TypeError(f"{name} must decode to a JSON list")
    if len(values) != batch:
        raise ValueError(
            f"{name} must contain one value per request: "
            f"expected={batch}, actual={len(values)}")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise TypeError(f"{name} must contain integers")
    if any(value <= 0 or value > maximum for value in values):
        raise ValueError(
            f"{name} values must be in [1, {maximum}]: {values!r}")
    if max(values) != maximum:
        raise ValueError(
            f"{name} maximum must match the case maximum {maximum}: "
            f"{values!r}")
    return values


def _measure(fn, nvtx_range_name, cache_eviction_operation, capture):
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    if cache_eviction_operation is not None:
        cache_eviction_operation()
        torch.cuda.current_stream().synchronize()
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(nvtx_range_name)
    fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()


def _scheduler_metadata_evidence(metadata, *, source, arguments):
    """Copy one FA3 scheduler-metadata tensor to the CPU as launch evidence."""
    if metadata.dtype != torch.int32 or metadata.dim() != 1:
        raise ValueError(
            "FA3 scheduler metadata must be a 1-D int32 tensor, got "
            f"{metadata.dtype} with shape {tuple(metadata.shape)}")
    return {
        "source": source,
        "layout_source": FA3_SCHEDULER_METADATA_SOURCE[installed_vllm_stack()],
        "get_scheduler_metadata_arguments": arguments,
        "metadata_int32": metadata.to("cpu").tolist(),
    }


def _scheduler_arguments(*, batch_size, max_seqlen_q, max_seqlen_k,
                         num_heads_q, num_heads_kv, headdim, headdim_v,
                         page_size, causal, num_splits, cache_seqlens):
    return {
        "batch_size": batch_size, "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k, "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv, "headdim": headdim,
        "headdim_v": headdim_v, "page_size": page_size, "causal": causal,
        "num_splits": num_splits, "cache_seqlens": list(cache_seqlens),
        "window_size": [-1, -1], "has_softcap": False,
    }


def _run(fn, *, nvtx_range_name, capture, mapping, split_evidence,
         pre_nvtx_cache_eviction_mib=0):
    if not isinstance(nvtx_range_name, str) or not nvtx_range_name:
        raise ValueError("H200 FlashAttention runner requires an NVTX name")
    vllm_stack = installed_vllm_stack()
    validate_cache_eviction_mib(
        pre_nvtx_cache_eviction_mib,
        minimum_nonzero_mib=H200_MINIMUM_CACHE_EVICTION_MIB,
        hardware_id=H200_HARDWARE_ID)
    cache_eviction_operation, _ = prepare_bf16_cache_eviction_sweep(
        torch_module=torch, device="cuda",
        cache_eviction_mib=pre_nvtx_cache_eviction_mib)
    _measure(fn, nvtx_range_name, cache_eviction_operation, capture)
    # Outside the NVTX range and after the post-range clock query.
    capture.add_launch_evidence("fa3_scheduler_metadata", split_evidence())
    capture.add_launch_evidence("op_mapping", mapping)
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "fa_version": FA_VERSION,
        "vllm_stack": vllm_stack,
        "vllm_call_path": VLLM_CALL_PATH[vllm_stack],
    })


def _paged_pool(kv_lens, HKV, D, dt, device="cuda"):
    """Scattered paged KV pool + block_table (A800 construction)."""
    blocks_per_req = [(l + BLOCK_SIZE - 1) // BLOCK_SIZE for l in kv_lens]
    pool_blocks = sum(blocks_per_req) + 1
    key_cache = torch.randn(pool_blocks, BLOCK_SIZE, HKV, D, dtype=dt, device=device)
    value_cache = torch.randn_like(key_cache)
    perm = torch.randperm(pool_blocks - 1, device=device)
    block_table = torch.zeros(len(kv_lens), max(blocks_per_req),
                              device=device, dtype=torch.int32)
    idx = 0
    for i, nb in enumerate(blocks_per_req):
        block_table[i, :nb] = perm[idx:idx + nb]
        idx += nb
    return key_cache, value_cache, block_table


def _cu(lengths, device):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device=device)


def run_flash_attn(B, S, HQ, HKV, D, dtype, seq_len_q, num_splits,
                   kv_layout, kv_dist, causal, q_lens, kv_lens,
                   nvtx_range_name, capture, device="cuda",
                   pre_nvtx_cache_eviction_mib=0, mla_pool_scale=1,
                   mla_block_order="sequential"):
    if min(B, S, HQ, HKV, D, seq_len_q, num_splits) <= 0:
        raise ValueError("FlashAttention workload dimensions must be positive")
    dt = DTYPE_MAP[dtype.lower()]
    Q = seq_len_q
    q_lens = _validate_lengths("q_lens", q_lens, B, Q)
    kv_lens = _validate_lengths("kv_lens", kv_lens, B, S)
    if any(request_query > request_kv
           for request_query, request_kv in zip(q_lens, kv_lens)):
        raise ValueError("each FlashAttention request requires q_len <= kv_len")
    if Q > 1 and num_splits != 1 and kv_layout != "mla":
        raise ValueError("FlashAttention prefill requires num_splits=1")
    scale = 1.0 / (D ** 0.5)
    common = {"nvtx_range_name": nvtx_range_name, "capture": capture,
              "pre_nvtx_cache_eviction_mib": pre_nvtx_cache_eviction_mib}

    if kv_layout in ("paged", "paged_varlen", "chunked_prefill"):
        if num_splits != 1:
            raise ValueError("serving paged layouts carry SP_REQUESTED=1")
        is_causal = True if kv_layout == "chunked_prefill" else causal
        key_cache, value_cache, block_table = _paged_pool(kv_lens, HKV, D, dt, device)
        q = torch.randn(sum(q_lens), HQ, D, dtype=dt, device=device)
        out = torch.empty_like(q)
        cu_seqlens_q = _cu(q_lens, device)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        scheduler_metadata = get_scheduler_metadata(
            batch_size=B, max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            num_heads_q=HQ, num_heads_kv=HKV, headdim=D,
            cache_seqlens=seqused_k, qkv_dtype=dt, cu_seqlens_q=cu_seqlens_q,
            page_size=BLOCK_SIZE, causal=is_causal, num_splits=0)
        split_arguments = _scheduler_arguments(
            batch_size=B, max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            num_heads_q=HQ, num_heads_kv=HKV, headdim=D, headdim_v=D,
            page_size=BLOCK_SIZE, causal=is_causal, num_splits=0,
            cache_seqlens=kv_lens)

        def split_evidence():
            return _scheduler_metadata_evidence(
                scheduler_metadata, source="measured_call_scheduler_metadata",
                arguments=split_arguments)

        def fn():
            flash_attn_varlen_func(
                q=q, k=key_cache, v=value_cache, out=out,
                cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max(q_lens),
                seqused_k=seqused_k, max_seqlen_k=max(kv_lens),
                softmax_scale=scale, causal=is_causal,
                block_table=block_table, scheduler_metadata=scheduler_metadata,
                fa_version=FA_VERSION, num_splits=0)

        _run(fn, mapping={
            "a800_call": "vllm FA2 flash_attn_varlen_func(block_table, num_splits=1)",
            "h200_call": ("vllm FA3 flash_attn_varlen_func(block_table, "
                          "scheduler_metadata, fa_version=3, num_splits=0)"),
            "causal": is_causal,
        }, split_evidence=split_evidence, **common)

    elif kv_layout == "mla":
        if any(length != Q for length in q_lens):
            raise ValueError("MLA runner requires uniform Q lengths")
        if Q > MLA_DECODE_MAX_QUERY_LEN:
            raise ValueError(
                f"MLA decode requires Q_LEN <= {MLA_DECODE_MAX_QUERY_LEN} "
                "(FLASH_ATTN_MLA reorder_batch_threshold); longer queries "
                "take the MLA prefill path")
        if any(length != S for length in kv_lens):
            raise ValueError("MLA runner requires uniform KV lengths")
        if HKV != 1:
            raise ValueError("MLA runner requires one latent KV head")
        nope_dim = D
        blocks_per_req = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
        # Diagnostic placement (defaults keep the canonical construction):
        # scatter the same blocks over a pool pool_scale times larger, in
        # sequential or random order.
        pool_blocks = blocks_per_req * B * mla_pool_scale + 1
        kv_cache = torch.empty(pool_blocks, BLOCK_SIZE, nope_dim + MLA_ROPE_DIM,
                               dtype=dt, device=device).normal_()
        if mla_block_order == "sequential":
            used = torch.arange(blocks_per_req * B, device=device) * mla_pool_scale
        else:
            used = torch.randperm(pool_blocks - 1, device=device)[:blocks_per_req * B]
        block_table = used.to(torch.int32).view(B, blocks_per_req)
        kv_c_cache = kv_cache[..., :nope_dim]
        k_pe_cache = kv_cache[..., nope_dim:]
        q_nope = torch.randn(B * Q, HQ, nope_dim, dtype=dt, device=device)
        q_pe = torch.randn(B * Q, HQ, MLA_ROPE_DIM, dtype=dt, device=device)
        cu_seqlens_q = _cu(q_lens, device)
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        scheduler_metadata = get_scheduler_metadata(
            batch_size=B, max_seqlen_q=Q, max_seqlen_k=S,
            num_heads_q=HQ, num_heads_kv=1, headdim=MLA_ROPE_DIM,
            cache_seqlens=seq_lens, qkv_dtype=dt, headdim_v=nope_dim,
            page_size=BLOCK_SIZE, cu_seqlens_q=cu_seqlens_q, causal=True,
            num_splits=0)
        split_arguments = _scheduler_arguments(
            batch_size=B, max_seqlen_q=Q, max_seqlen_k=S, num_heads_q=HQ,
            num_heads_kv=1, headdim=MLA_ROPE_DIM, headdim_v=nope_dim,
            page_size=BLOCK_SIZE, causal=True, num_splits=0,
            cache_seqlens=kv_lens)

        def split_evidence():
            return _scheduler_metadata_evidence(
                scheduler_metadata, source="measured_call_scheduler_metadata",
                arguments=split_arguments)

        def fn():
            return flash_attn_varlen_func(
                q=q_pe, k=k_pe_cache.unsqueeze(-2), v=kv_c_cache.unsqueeze(-2),
                q_v=q_nope, max_seqlen_q=Q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=S, seqused_k=seq_lens, block_table=block_table,
                softmax_scale=scale, causal=True, fa_version=FA_VERSION,
                scheduler_metadata=scheduler_metadata, num_splits=0)

        if Q == 1:
            mapping = {
                "a800_call": "vllm triton_mla decode_attention_fwd(num_kv_splits=4)",
                "h200_call": ("FLASH_ATTN_MLA decode: vllm FA3 flash_attn_varlen_func("
                              "q=q_pe, k=k_pe_cache, v=kv_c_cache, q_v=q_nope, "
                              "causal=True, fa_version=3, num_splits=0)"),
                "kv_lora_rank": nope_dim, "qk_rope_head_dim": MLA_ROPE_DIM,
                "causal": True,
            }
        else:
            mapping = {
                "a800_call": ("none: vLLM 0.11 TRITON_MLA keeps "
                              "reorder_batch_threshold=1, so q_len > 1 "
                              "takes the MLA prefill path"),
                "h200_call": ("FLASH_ATTN_MLA decode (uniform multi-token "
                              "spec-decode verify, q_len <= "
                              "reorder_batch_threshold 512): vllm FA3 "
                              "flash_attn_varlen_func(q=q_pe, k=k_pe_cache, "
                              "v=kv_c_cache, q_v=q_nope, "
                              f"max_seqlen_q={Q}, cu_seqlens_q, seqused_k, "
                              "block_table, causal=True, fa_version=3, "
                              "num_splits=0, scheduler_metadata="
                              f"get_scheduler_metadata(max_seqlen_q={Q}, "
                              "causal=True, num_splits=0))"),
                "vllm_source": (
                    "vllm/v1/attention/backends/mla/flashattn_mla.py "
                    "FlashAttnMLAMetadataBuilder._build_decode + "
                    "FlashAttnMLAImpl.forward_mqa; "
                    "vllm/v1/attention/backends/utils.py "
                    "split_decodes_and_prefills"),
                "max_seqlen_q": Q,
                "kv_lora_rank": nope_dim, "qk_rope_head_dim": MLA_ROPE_DIM,
                "causal": True,
            }
        _run(fn, mapping=mapping, split_evidence=split_evidence, **common)

    elif kv_layout == "contiguous":
        if any(length != Q for length in q_lens):
            raise ValueError("dense contiguous layout requires uniform Q lengths")
        if any(length != S for length in kv_lens):
            raise ValueError("dense contiguous layout requires uniform KV lengths")
        q = torch.randn(B * Q, HQ, D, dtype=dt, device=device)
        k = torch.randn(B * S, HKV, D, dtype=dt, device=device)
        v = torch.randn(B * S, HKV, D, dtype=dt, device=device)
        cu_seqlens_q = _cu(q_lens, device)
        cu_seqlens_k = _cu(kv_lens, device)
        executed_splits = num_splits if Q == 1 else 0
        split_arguments = _scheduler_arguments(
            batch_size=B, max_seqlen_q=Q, max_seqlen_k=S, num_heads_q=HQ,
            num_heads_kv=HKV, headdim=D, headdim_v=D, page_size=None,
            causal=causal, num_splits=executed_splits, cache_seqlens=kv_lens)

        def split_evidence():
            # mha_fwd builds this call's metadata internally from the same
            # arguments; recompute it after the NVTX range.
            metadata = get_scheduler_metadata(
                batch_size=B, max_seqlen_q=Q, max_seqlen_k=S,
                num_heads_q=HQ, num_heads_kv=HKV, headdim=D,
                cache_seqlens=torch.tensor(
                    kv_lens, dtype=torch.int32, device=device),
                qkv_dtype=dt, cu_seqlens_q=cu_seqlens_q, causal=causal,
                num_splits=executed_splits)
            torch.cuda.synchronize()
            return _scheduler_metadata_evidence(
                metadata, source="recomputed_get_scheduler_metadata",
                arguments=split_arguments)

        def fn():
            return flash_attn_varlen_func(
                q=q, k=k, v=v, max_seqlen_q=Q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=S, cu_seqlens_k=cu_seqlens_k,
                softmax_scale=scale, causal=causal, fa_version=FA_VERSION,
                num_splits=executed_splits)

        _run(fn, mapping={
            "a800_call": ("Dao flash_attn_with_kvcache(num_splits=SP)" if Q == 1
                          else "Dao flash_attn_func"),
            "h200_call": ("vllm FA3 flash_attn_varlen_func over dense K/V "
                          f"(cu_seqlens_k), num_splits={executed_splits}"),
            "causal": causal,
        }, split_evidence=split_evidence, **common)
    else:
        raise ValueError(f"unknown FlashAttention KV layout: {kv_layout!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--seq_len_cache", type=int, required=True)
    p.add_argument("--num_heads_q", type=int, required=True)
    p.add_argument("--num_heads_kv", type=int, required=True)
    p.add_argument("--head_dim", type=int, required=True)
    p.add_argument("--dtype", required=True, choices=sorted(DTYPE_MAP))
    p.add_argument("--seq_len_q", type=int, required=True)
    p.add_argument("--q-lens-json", required=True)
    p.add_argument("--kv-lens-json", required=True)
    p.add_argument("--num_splits", type=int, required=True)
    p.add_argument("--kv_layout", required=True,
                   choices=["contiguous", "paged", "paged_varlen", "mla", "chunked_prefill"])
    p.add_argument("--kv_dist", required=True, choices=["skew", "uniform", "explicit"])
    p.add_argument("--causal", type=int, required=True, choices=[0, 1])
    p.add_argument("--nvtx_range_name", type=str, required=True)
    p.add_argument("--pre-nvtx-cache-eviction-mib", type=int, default=0)
    p.add_argument("--mla-pool-scale", type=int, default=1,
                   help="diagnostic: MLA KV pool this many times the used blocks")
    p.add_argument("--mla-block-order", choices=("sequential", "random"),
                   default="sequential", help="diagnostic: MLA block-table placement")
    add_h200_capture_arguments(p)
    a = p.parse_args()
    capture = open_h200_capture_session(a, a.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    run_flash_attn(a.batch, a.seq_len_cache, a.num_heads_q, a.num_heads_kv,
                   a.head_dim, a.dtype, a.seq_len_q, a.num_splits,
                   a.kv_layout, a.kv_dist, bool(a.causal),
                   json.loads(a.q_lens_json), json.loads(a.kv_lens_json),
                   a.nvtx_range_name, capture,
                   pre_nvtx_cache_eviction_mib=a.pre_nvtx_cache_eviction_mib,
                   mla_pool_scale=a.mla_pool_scale, mla_block_order=a.mla_block_order)
    capture.finish()
    print("FA done")


if __name__ == "__main__":
    main()
