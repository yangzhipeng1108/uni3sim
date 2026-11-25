"""NVIDIA H200 DeepSeek Sparse Attention (DSA) indexer runner.

Executes the CUDA path of vLLM 0.24.0's ``sparse_attn_indexer`` custom op
(``vllm/model_executor/layers/sparse_attn_indexer.py``) for one indexer layer
of ``DeepseekV32ForCausalLM`` / ``GlmMoeDsaForCausalLM`` under the canonical
``nsys_nvtx_kernel_median`` protocol (``H200CaptureSession``).  The indexer is
replicated across TP ranks (``Indexer.wq_b`` is a ``ReplicatedLinear``), so
``--index-n-heads`` is the full model value.

The K-cache insert (``ops.indexer_k_quant_and_cache``) is measured by
``kc_dsa_cache_write_runner.py``; this runner starts after it.  Inside the
single NVTX range:

* ``topk_indices_buffer[:num_tokens] = -1``;
* ``--phase prefill`` (one or more prefill requests): for every chunk from
  vLLM's own ``split_indexer_prefill_chunks`` /
  ``build_prefill_chunk_metadata`` (``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB``
  logits budget, ``max_model_len * 40`` gather workspace):
  ``ops.cp_gather_indexer_k_quant_cache`` (skipped for query sub-chunks after
  the first, as in vLLM), DeepGEMM ``fp8_fp4_mqa_logits(clean_logits=False)``
  and ``ops.top_k_per_row_prefill``;
* ``--phase decode``: DeepGEMM ``fp8_fp4_paged_mqa_logits(clean_logits=False)``
  then ``torch.ops._C.cooperative_topk`` (rows <= 32, topk in {512,1024,2048},
  SM90) or ``torch.ops._C.persistent_topk``.  On SM90 the native MTP layout is
  used only for ``next_n`` in (1, 2); larger ``next_n`` is flattened to
  ``B*next_n`` rows of ``next_n=1`` with per-token context lengths and an
  expanded block table, exactly as ``DeepseekV32IndexerMetadataBuilder``
  does (``use_flattening``).

Metadata built per engine step by the metadata builder
(``get_paged_mqa_logits_metadata``, prefill chunk ``cu_seqlen_ks/ke``, the
flattened block table) is built before the NVTX range and recorded.  The
indexer K cache is ``uint8 [num_blocks, 64, 132]`` (128 fp8 + one fp32 scale
per 128-element block, ``scale_fmt="ue8m0"``) populated with the real writer.
Q is FP8 ``[tokens, H, 128]`` and ``weights`` FP32 ``[tokens, H]`` (the Q
scale is folded into the weights on the FP8 path).

vLLM 0.30.0 (explicit branch on ``installed_vllm_stack()``, recorded as
``vllm_stack`` in the evidence): same custom op
(``sparse_attn_indexer``) with these changes, all taken from the 0.30
sources and executed here:

* the DeepSeek-V3.2 / GLM-5.2 CUDA model (``vllm/models/deepseek_v32``)
  clears ``topk_indices_buffer`` inside ``_fused_norm_rope_kernel`` and calls
  the indexer with ``skip_topk_buffer_clear=True`` -> no ``= -1`` fill here;
* prefill chunking is ``DeepseekV32IndexerMetadataBuilder.
  _split_indexer_prefill_chunks`` (static method; the module function is
  gone) and the chunk loop reads ``max_local_total_seq_lens`` /
  ``local_cu_seq_lens``;
* decode: SM90 paged MQA logits run ``next_n`` in {1, 2, 4} natively
  (``_supports_native_decode`` / ``native_next_n_supported``; next_n 4 uses
  ``num_sms // 2`` schedule slots), other ``next_n`` are flattened;
* decode top-k is ``SparseIndexerTopk("auto")`` (``indexer_topk.py``):
  cooperative for rows <= 64 (was 32), else persistent with
  ``max_seq_len = logits.shape[1]`` (was the batch max_seq_len).

0.30 skips the whole indexer for a prefill-only batch that the main MLA
routes to dense MHA (longest sequence <= index_topk); such prefill cases
still measure the indexer and record ``vllm030_default_route``.
"""

import argparse
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
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backends.mla.indexer import (
    build_prefill_chunk_metadata,
    get_max_prefill_buffer_size,
)

VLLM_STACK = installed_vllm_stack()
if VLLM_STACK == "vllm024":
    from vllm.model_executor.layers.sparse_attn_indexer import (
        RADIX_TOPK_WORKSPACE_SIZE,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        split_indexer_prefill_chunks,
    )
elif VLLM_STACK == "vllm030":
    from vllm.model_executor.layers.indexer_topk import (
        RADIX_TOPK_WORKSPACE_SIZE,
        SparseIndexerTopk,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
        _supports_native_decode,
    )
    split_indexer_prefill_chunks = (
        DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks)
else:
    raise RuntimeError(f"unhandled vLLM stack {VLLM_STACK!r}")

BLOCK_SIZE = 64            # DeepseekV32IndexerBackend kernel block size
QUANT_BLOCK_SIZE = 128     # Indexer.quant_block_size
SCALE_FMT = "ue8m0"        # Indexer.scale_fmt
FP8 = torch.float8_e4m3fn
WARMUP = 5
FILL_CHUNK_TOKENS = 1 << 20
COOPERATIVE_TOPK_SIZES = (512, 1024, 2048)


class _WorkspaceConfig:
    """Duck-typed VllmConfig for get_max_prefill_buffer_size."""

    def __init__(self, max_model_len):
        self.model_config = type("ModelConfig", (), {
            "max_model_len": max_model_len})()


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer: {value!r}")


def _indexer_cache(batch, context, max_model_len, head_dim, device):
    blocks_per_req = (context + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool_blocks = batch * blocks_per_req + 1
    width = head_dim + head_dim // QUANT_BLOCK_SIZE * 4
    kv_cache = torch.empty(pool_blocks, BLOCK_SIZE, width, dtype=torch.uint8,
                           device=device)
    num_slots = pool_blocks * BLOCK_SIZE
    for start in range(0, num_slots, FILL_CHUNK_TOKENS):
        count = min(FILL_CHUNK_TOKENS, num_slots - start)
        k = torch.randn(count, head_dim, dtype=torch.bfloat16, device=device)
        slots = torch.arange(start, start + count, dtype=torch.int64,
                             device=device)
        ops.indexer_k_quant_and_cache(k, kv_cache, slots, QUANT_BLOCK_SIZE,
                                      SCALE_FMT)
    perm = torch.randperm(pool_blocks - 1, device=device, dtype=torch.int64)
    block_table = torch.zeros(batch, max_blocks, dtype=torch.int32,
                              device=device)
    block_table[:, :blocks_per_req] = perm.view(
        batch, blocks_per_req).to(torch.int32)
    return kv_cache, block_table, pool_blocks


def build_decode(batch, next_n, context, n_heads, head_dim, topk,
                 max_model_len, device="cuda"):
    num_tokens = batch * next_n
    kv_cache, block_table, pool_blocks = _indexer_cache(
        batch, context, max_model_len, head_dim, device)
    q_quant = torch.randn(num_tokens, n_heads, head_dim, device=device).to(FP8)
    weights = torch.randn(num_tokens, n_heads, dtype=torch.float32,
                          device=device)
    offsets = torch.arange(next_n, device=device, dtype=torch.int32)
    per_token = (context - next_n + 1 + offsets).repeat(batch)
    if VLLM_STACK == "vllm024":
        # H200 (not SM100): paged MQA logits supports next_n in (1, 2).
        use_flattening = next_n not in (1, 2)
    else:
        # vLLM 0.30 _use_flattening (no adaptive verification): SM90
        # native_next_n_supported -> next_n in (1, 2, 4).
        use_flattening = not _supports_native_decode(next_n)
    if use_flattening:
        seq_lens = per_token.view(num_tokens, 1).contiguous()
        kernel_block_table = torch.repeat_interleave(block_table, next_n,
                                                     dim=0)
        q_view = q_quant.reshape(num_tokens, 1, n_heads, head_dim)
    else:
        seq_lens = per_token.view(batch, next_n).contiguous()
        kernel_block_table = block_table
        q_view = q_quant.reshape(batch, next_n, n_heads, head_dim)
    num_sms = num_compute_units(torch.cuda.current_device())
    schedule_metadata = get_paged_mqa_logits_metadata(seq_lens, BLOCK_SIZE,
                                                      num_sms)
    topk_buffer = torch.empty(num_tokens, topk, dtype=torch.int32,
                              device=device)
    topk_workspace = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8,
                                 device=device)
    kv_view = kv_cache.unsqueeze(-2)
    if topk not in COOPERATIVE_TOPK_SIZES:
        raise ValueError("topk outside {512,1024,2048} would take "
                         "top_k_per_row_decode; not a DSA deployment shape")
    chosen = {}
    if VLLM_STACK == "vllm030":
        topk_selector = SparseIndexerTopk("auto")
        # 0.30 fused deepseek_v32 path: _fused_norm_rope_kernel writes the
        # -1 sentinel before the indexer (skip_topk_buffer_clear=True).
        topk_buffer.fill_(-1)

    def indexer():
        if VLLM_STACK == "vllm024":
            topk_buffer[:num_tokens] = -1
        logits = fp8_fp4_paged_mqa_logits(
            (q_view, None), kv_view, weights[:num_tokens], seq_lens,
            kernel_block_table, schedule_metadata,
            max_model_len=max_model_len, clean_logits=False)
        num_rows = logits.shape[0]
        topk_indices = topk_buffer[:num_tokens, :topk]
        if VLLM_STACK == "vllm024":
            # sparse_attn_indexer: cooperative iff rows <= 32, TMA-aligned
            # stride and SM90+ (H200); otherwise persistent.
            use_cooperative = num_rows <= 32 and logits.stride(0) % 4 == 0
            topk_max_seq_len = context
        else:
            # SparseIndexerTopk("auto")._resolve_auto: cooperative ->
            # persistent -> per_row; persistent passes logits.shape[1].
            backend = topk_selector.resolve_backend(logits, topk, num_rows)
            if backend not in ("cooperative", "persistent"):
                raise RuntimeError(f"unexpected 0.30 top-k backend {backend}")
            use_cooperative = backend == "cooperative"
            topk_max_seq_len = context if use_cooperative else logits.shape[1]
        chosen["topk_kernel"] = ("cooperative_topk" if use_cooperative
                                 else "persistent_topk")
        chosen["topk_max_seq_len_arg"] = topk_max_seq_len
        chosen["logits_shape"] = list(logits.shape)
        chosen["logits_stride"] = list(logits.stride())
        if use_cooperative:
            torch.ops._C.cooperative_topk(logits, seq_lens, topk_indices,
                                          topk_workspace, topk,
                                          topk_max_seq_len)
        else:
            torch.ops._C.persistent_topk(logits, seq_lens, topk_indices,
                                         topk_workspace, topk,
                                         topk_max_seq_len)
        return topk_indices

    def validate(topk_indices):
        limit = per_token.view(num_tokens, 1).to(torch.int64)
        values = topk_indices.to(torch.int64)
        if bool(((values < -1) | (values >= limit)).any()):
            raise RuntimeError("decode top-k index outside the context")
        return (values >= 0).sum(dim=1).tolist()

    evidence = {
        "phase": "decode", "batch": batch, "next_n": next_n,
        "context": context, "num_tokens": num_tokens,
        "q_lens": [next_n] * batch, "context_lens": [context] * batch,
        "per_row_context_lens": per_token.tolist(),
        "kv_cache_layout": ("uint8 [num_blocks, 64, 132]: 128 fp8_e4m3 K + "
                            "one fp32 scale per 128 elements (ue8m0)"),
        "block_size": BLOCK_SIZE, "quant_block_size": QUANT_BLOCK_SIZE,
        "scale_fmt": SCALE_FMT,
        "index_n_heads": n_heads, "index_head_dim": head_dim, "topk": topk,
        "max_model_len": max_model_len, "use_flattening": use_flattening,
        "native_next_n_rule": ("SM90 next_n in (1, 2)" if VLLM_STACK == "vllm024"
                               else "SM90 native_next_n_supported: next_n in "
                                    "(1, 2, 4)"),
        "kernel_q_shape": list(q_view.shape),
        "kernel_context_lens_shape": list(seq_lens.shape),
        "kernel_block_table_shape": list(kernel_block_table.shape),
        "topk_dispatch": chosen,
        "num_sms": num_sms, "pool_blocks": pool_blocks,
        "kv_cache_shape": list(kv_cache.shape),
        "schedule_metadata_built_outside_nvtx": True,
        "schedule_metadata_shape": list(schedule_metadata.shape),
        "schedule_metadata": schedule_metadata.cpu().tolist(),
        "vllm_stack": VLLM_STACK,
        "topk_buffer_clear_in_range": VLLM_STACK == "vllm024",
        "public_ops": [
            *(["topk_indices_buffer[:T] = -1"] if VLLM_STACK == "vllm024"
              else []),
            "vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits((q, None), "
            "kv_cache.unsqueeze(-2), weights, context_lens, block_table, "
            "schedule_metadata, max_model_len=max_model_len, "
            "clean_logits=False)",
            "torch.ops._C.cooperative_topk | torch.ops._C.persistent_topk("
            "logits, context_lens, topk_indices, workspace[1 MiB], topk, "
            "max_seq_len) (see topk_dispatch)",
        ],
        "schedule_metadata_call": (
            "vllm.utils.deep_gemm.get_paged_mqa_logits_metadata("
            "context_lens, 64, num_sms)"),
    }
    return indexer, validate, evidence


def build_prefill(batch, q_len, context, n_heads, head_dim, topk,
                  max_model_len, device="cuda"):
    num_tokens = batch * q_len
    kv_cache, block_table, pool_blocks = _indexer_cache(
        batch, context, max_model_len, head_dim, device)
    q_quant = torch.randn(num_tokens, n_heads, head_dim, device=device).to(FP8)
    weights = torch.randn(num_tokens, n_heads, dtype=torch.float32,
                          device=device)
    query_start_loc_cpu = torch.arange(batch + 1, dtype=torch.int32) * q_len
    query_start_loc = query_start_loc_cpu.to(device)
    seq_lens_cpu = torch.full((batch,), context, dtype=torch.int32)
    seq_lens = seq_lens_cpu.to(device)
    workspace_size = get_max_prefill_buffer_size(
        _WorkspaceConfig(max_model_len))
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    chunk_specs = split_indexer_prefill_chunks(
        seq_lens_cpu, torch.diff(query_start_loc_cpu), workspace_size,
        max_logits_bytes, request_offset=0)
    chunks = []
    for req_slice, query_slice in chunk_specs:
        chunk = build_prefill_chunk_metadata(
            req_slice.start, req_slice.stop, query_start_loc,
            query_start_loc_cpu, seq_lens, seq_lens, seq_lens_cpu,
            block_table, 1, query_slice=query_slice,
            skip_kv_gather=query_slice.start > 0)
        if chunk is None:
            raise RuntimeError("empty indexer prefill chunk")
        chunks.append(chunk)
    max_total = max(chunk.total_seq_lens for chunk in chunks)
    k_quant_full = torch.empty(max_total, head_dim, dtype=FP8, device=device)
    k_scale_full = torch.empty(max_total, 4, dtype=torch.uint8, device=device)
    topk_buffer = torch.empty(num_tokens, topk, dtype=torch.int32,
                              device=device)
    if VLLM_STACK == "vllm030":
        # vLLM 0.30 skips the buffer clear on the fused deepseek_v32 path;
        # the fused norm/rope kernel wrote -1 before the indexer.
        topk_buffer.fill_(-1)

    def indexer():
        if VLLM_STACK == "vllm024":
            topk_buffer[:num_tokens] = -1
        for chunk in chunks:
            if VLLM_STACK == "vllm024":
                k_quant = k_quant_full[:chunk.total_seq_lens]
                k_scale = k_scale_full[:chunk.total_seq_lens]
                gather = not chunk.skip_kv_gather
                gather_cu = chunk.cu_seq_lens
            else:
                k_quant = k_quant_full[:chunk.max_local_total_seq_lens]
                k_scale = k_scale_full[:chunk.max_local_total_seq_lens]
                gather = (not chunk.skip_kv_gather
                          and chunk.local_total_seq_lens > 0)
                gather_cu = chunk.local_cu_seq_lens
            if gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache, k_quant, k_scale, chunk.block_table,
                    gather_cu)
            logits = fp8_fp4_mqa_logits(
                (q_quant[chunk.token_start:chunk.token_end], None),
                (k_quant, k_scale.view(torch.float32).squeeze(-1)),
                weights[chunk.token_start:chunk.token_end],
                chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, clean_logits=False)
            topk_indices = topk_buffer[chunk.token_start:chunk.token_end,
                                       :topk]
            ops.top_k_per_row_prefill(
                logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, topk_indices,
                logits.shape[0], logits.stride(0), logits.stride(1), topk)
        return topk_buffer

    def validate(topk_indices):
        values = topk_indices.to(torch.int64)
        if bool(((values < -1) | (values >= context)).any()):
            raise RuntimeError("prefill top-k index outside the context")
        if bool(((values >= 0).sum(dim=1) == 0).any()):
            raise RuntimeError("prefill row without any selected index")
        return (values >= 0).sum(dim=1).tolist()

    evidence = {
        "phase": "prefill", "batch": batch, "q_len": q_len,
        "context": context, "num_tokens": num_tokens,
        "q_lens": [q_len] * batch, "context_lens": [context] * batch,
        "kv_cache_layout": ("uint8 [num_blocks, 64, 132]: 128 fp8_e4m3 K + "
                            "one fp32 scale per 128 elements (ue8m0)"),
        "block_size": BLOCK_SIZE, "quant_block_size": QUANT_BLOCK_SIZE,
        "scale_fmt": SCALE_FMT,
        "index_n_heads": n_heads, "index_head_dim": head_dim, "topk": topk,
        "max_model_len": max_model_len,
        "max_logits_mb": envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB,
        "gather_workspace_tokens": workspace_size,
        "chunks": [{"token_start": c.token_start, "token_end": c.token_end,
                    "num_reqs": c.num_reqs,
                    "total_seq_lens": int(c.total_seq_lens),
                    "skip_kv_gather": bool(c.skip_kv_gather)}
                   for c in chunks],
        "pool_blocks": pool_blocks, "kv_cache_shape": list(kv_cache.shape),
        "chunk_metadata_built_outside_nvtx": True,
        "chunk_cu_seqlen_ke_first_last": [
            [int(c.cu_seqlen_ke[0].item()), int(c.cu_seqlen_ke[-1].item())]
            for c in chunks],
        "public_ops": [
            "topk_indices_buffer[:T] = -1",
            "per chunk: vllm._custom_ops.cp_gather_indexer_k_quant_cache("
            "kv_cache, k_quant, k_scale, block_table, cu_seq_lens) "
            "(unless skip_kv_gather)",
            "per chunk: vllm.utils.deep_gemm.fp8_fp4_mqa_logits((q, None), "
            "(k_quant, k_scale_fp32), weights, cu_seqlen_ks, cu_seqlen_ke, "
            "clean_logits=False)",
            "per chunk: vllm._custom_ops.top_k_per_row_prefill(logits, "
            "cu_seqlen_ks, cu_seqlen_ke, topk_indices, rows, stride0, "
            "stride1, topk)",
        ],
        "scheduler_metadata": "none: fp8_fp4_mqa_logits takes no schedule",
        "vllm_stack": VLLM_STACK,
        "topk_buffer_clear_in_range": VLLM_STACK == "vllm024",
    }
    if VLLM_STACK == "vllm030":
        evidence["public_ops"] = [
            op for op in evidence["public_ops"]
            if op != "topk_indices_buffer[:T] = -1"]
        evidence["vllm030_default_route"] = (
            "skipped: prefill-only batch routed to dense MHA (longest "
            f"sequence {context} <= index_topk {topk}); measured as with "
            "--sparse-mla-force-mqa" if context <= topk else
            "indexer runs (longest sequence > index_topk)")
    return indexer, validate, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["prefill", "decode"])
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--q-len", type=int, required=True,
                        help="prefill tokens per request, or next_n (1+MTP)")
    parser.add_argument("--context", type=int, required=True,
                        help="KV length per request including the new tokens")
    parser.add_argument("--index-n-heads", type=int, required=True)
    parser.add_argument("--index-head-dim", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--nvtx_range_name", type=str, required=True)
    add_h200_capture_arguments(parser)
    args = parser.parse_args()
    for name in ("batch", "q_len", "context", "index_n_heads",
                 "index_head_dim", "topk", "max_model_len"):
        _positive(name, getattr(args, name))
    if args.q_len > args.context or args.context > args.max_model_len:
        raise ValueError("require q_len <= context <= max_model_len")
    if args.index_head_dim % QUANT_BLOCK_SIZE != 0:
        raise ValueError("index_head_dim must be a multiple of 128")
    capture = open_h200_capture_session(args, args.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)
    torch.manual_seed(42)
    build = build_decode if args.phase == "decode" else build_prefill
    indexer, validate, evidence = build(
        args.batch, args.q_len, args.context, args.index_n_heads,
        args.index_head_dim, args.topk, args.max_model_len)
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(WARMUP):
        indexer()
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
    topk_indices = indexer()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()
    selected = validate(topk_indices)
    if len(selected) != args.batch * args.q_len or len(
            evidence["q_lens"]) != args.batch:
        raise RuntimeError("evidence list lengths disagree with the case")
    evidence["selected_indices_per_query"] = selected
    evidence["post_warmup_clock_settle_samples"] = settle_samples
    capture.add_launch_evidence("dsa_indexer", evidence)
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "warmup_calls": WARMUP,
    })
    capture.finish()
    print("DSA indexer done " + json.dumps(evidence["phase"]))


if __name__ == "__main__":
    main()
