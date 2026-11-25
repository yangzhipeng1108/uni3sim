"""FlashAttention runner for the A800 single-op NCU dataset.

Mirrors the 910B fa shape grid (kv_layout × kv_dist × seq_len_q × causal),
each layout driven through the REAL A800 vLLM serving op (goal: align to
serving, not reimplement). Verified against vllm 0.11.0:

  contiguous  : Dao flash_attn_with_kvcache (dense k/v) — the legacy benchmark
                layout (matches 910b contiguous; serving decode is paged, but
                this layout is kept for cross-card shape parity with 910b).
                Prefill (Q=S, causal) → Dao flash_attn_func.
  paged       : vllm flash_attn_varlen_func + block_table, uniform KV — the
                serving decode path (vllm/v1/attention/backends/flash_attn.py).
  paged_varlen: vllm varlen + block_table + per-request KV lengths (kv_dist
                skew/uniform) — serving multi-request decode.
  chunked_prefill: vllm varlen + chunked Q (cu_seqlens_q) + block_table + causal
                — serving chunked-prefill.
  mla         : vllm triton_mla decode_attention_fwd — A800 (Ampere) serving
                DeepSeek-MLA fallback (flashmla/flashattn_mla are Hopper-only).

num_splits is an executed launch input for contiguous decode and MLA, not a
filename-only label.  vLLM FA2 varlen accepts only 1 and may choose its actual
split internally; evaluation therefore derives that launch fact from NCU grid
geometry instead of trusting the workload label.

block_size=16 (vLLM GPU default; 910b uses 128 — per-card serving param, not a
shape dimension).

Capture discipline: setup (pool alloc, _varlen_lengths, warmup) OUTSIDE NVTX;
ONE post-warm serving call INSIDE NVTX (--nvtx-include profile_region/).
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from vllm.vllm_flash_attn import flash_attn_varlen_func

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    EventRunOptions,
    add_event_cli_arguments,
    build_nvidia_smi_telemetry_observer,
    measure_current_stream_events,
    parse_event_run_options,
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
    write_event_case_record,
)
from dataset_pipeline.kernel_dataset.fa_serving_capture import (
    validate_runner_capture_metadata,
)

DTYPE_MAP = {
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "float32": torch.float32,
}

BLOCK_SIZE = 16          # vLLM GPU default (910b uses 128; per-card serving param)
WARMUP = 5


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


def _measure(fn, nvtx_range_name, cache_eviction_operation):
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    if cache_eviction_operation is not None:
        cache_eviction_operation()
        torch.cuda.current_stream().synchronize()
    torch.cuda.nvtx.range_push(nvtx_range_name)
    fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def _validate_capture_output(output, expected_elements, fallback_output=None):
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        tensor = output[0]
    elif output is None and isinstance(fallback_output, torch.Tensor):
        tensor = fallback_output
    else:
        raise TypeError("captured FlashAttention public op returned no tensor output")
    if tensor.numel() != expected_elements:
        raise ValueError(
            "captured FlashAttention output shape disagrees with the request "
            f"contract: expected_elements={expected_elements}, actual={tensor.numel()}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("captured FlashAttention output contains a non-finite value")


def _environment():
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "flash_attn_version": importlib.metadata.version("flash-attn"),
        "device_index": device_index,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "input_seed": 20260826,
    }


def _run(fn, *, nvtx_range_name,
         event_options: EventRunOptions | None, workload,
         pre_nvtx_cache_eviction_mib=0,
         steady_state_public_op_repetitions=1,
         capture_expected_output_elements=None,
         capture_output_tensor=None):
    if (isinstance(steady_state_public_op_repetitions, bool)
            or not isinstance(steady_state_public_op_repetitions, int)
            or steady_state_public_op_repetitions not in {1, 8, 32}):
        raise ValueError(
            "A800 FlashAttention steady-state Event repetitions must be one of 1, 8, 32")
    if capture_expected_output_elements is not None:
        _validate_capture_output(
            fn(), capture_expected_output_elements, capture_output_tensor)
        torch.cuda.synchronize()
    if event_options is None:
        if steady_state_public_op_repetitions != 1:
            raise ValueError(
                "A800 FlashAttention steady-state repetitions require Event mode")
        if not isinstance(nvtx_range_name, str) or not nvtx_range_name:
            raise ValueError("normal A800 runner mode requires an NVTX name")
        validate_cache_eviction_mib(
            pre_nvtx_cache_eviction_mib, minimum_nonzero_mib=320,
            hardware_id="nvidia_a800")
        cache_eviction_operation, _ = prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=pre_nvtx_cache_eviction_mib)
        _measure(fn, nvtx_range_name, cache_eviction_operation)
        return
    validate_cache_eviction_mib(
        event_options.cache_eviction_mib, minimum_nonzero_mib=320,
        hardware_id="nvidia_a800")
    if nvtx_range_name is not None:
        raise ValueError("A800 Event mode cannot enable an NVTX range")
    if pre_nvtx_cache_eviction_mib:
        raise ValueError(
            "A800 Event mode cannot enable a pre-NVTX cache sweep")

    def operation():
        return fn()

    cache_eviction_operation, cache_eviction_bytes = (
        prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=event_options.cache_eviction_mib))
    telemetry_observer = None
    if event_options.telemetry_gpu_index is not None:
        telemetry_observer = build_nvidia_smi_telemetry_observer(
            event_options.telemetry_gpu_index)
    measurement = measure_current_stream_events(
        operation,
        device_module=torch.cuda,
        warmup_iterations=event_options.warmup_iterations,
        sample_count=event_options.sample_count,
        cache_eviction_operation=cache_eviction_operation,
        cache_eviction_bytes=cache_eviction_bytes,
        operation_repetitions_per_sample=steady_state_public_op_repetitions,
        telemetry_observer=telemetry_observer,
    )
    normalized_workload = dict(workload)
    normalized_workload["cache_eviction_bytes_before_sample"] = (
        cache_eviction_bytes)
    normalized_workload["steady_state_event_contract"] = {
        "public_op_repetition_count": steady_state_public_op_repetitions,
        "cache_state": (
            "one_op_per_cleaned_event_interval"
            if steady_state_public_op_repetitions == 1 else
            "cache_eviction_before_aggregate_with_intra_aggregate_reuse"),
    }
    write_event_case_record(
        options=event_options,
        hardware_id="nvidia_a800",
        family="flash_attn",
        runner_path=Path(__file__),
        timer={
            "api": "torch.cuda.Event",
            "implementation": "CUDA_Runtime_Event",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_used",
            "mode": (
                "one_complete_public_op_per_event_interval"
                if steady_state_public_op_repetitions == 1 else
                "fixed_count_complete_public_ops_in_one_current_stream_event_"
                "reported_per_op"),
            "public_op_repetition_count": steady_state_public_op_repetitions,
            "cache_condition": (
                "queued_pre_sample_device_sweep"
                if cache_eviction_bytes else "ordinary_warm_state"),
        },
        stream_contract={
            "event_stream": "torch.cuda.current_stream",
            "all_op_streams_joined": True,
            "evidence": (
                "the canonical A800 runner has no explicit secondary stream; "
                "the end Event follows the exact OP call on the current "
                "CUDA stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": "exact_runner_call",
            "workspace_behavior": "exact_runner_call",
            "evidence": (
                "the Event bracket invokes the exact canonical FlashAttention "
                "call; all explicit query/KV/cache/metadata tensors are "
                "constructed before the interval and its result is retained "
                "through the end Event"),
        },
        workload=normalized_workload,
        environment={
            **_environment(),
            "telemetry_physical_gpu_index": event_options.telemetry_gpu_index,
        },
        measurement=measurement,
    )


def _paged_pool(kv_lens, HKV, D, dt, device="cuda"):
    """Build a SCATTERED paged KV pool + block_table for per-request kv_lens.
    Mirrors 910b (shuffled block indices → real block-table indirection)."""
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


def run_flash_attn(B, S, HQ, HKV, D, dtype, seq_len_q, num_splits,
                   kv_layout, kv_dist, causal, q_lens, kv_lens,
                   nvtx_range_name,
                   device="cuda", event_options=None,
                   pre_nvtx_cache_eviction_mib=0,
                   steady_state_public_op_repetitions=1,
                   capture_metadata=None):
    if min(B, S, HQ, HKV, D, seq_len_q, num_splits) <= 0:
        raise ValueError("FlashAttention workload dimensions must be positive")
    if (kv_layout in ("paged", "paged_varlen", "chunked_prefill")
            and num_splits != 1):
        raise ValueError(
            "A800 vLLM FA2 varlen runner only accepts num_splits=1; "
            "its executed split is selected inside the compiled kernel")
    dt = DTYPE_MAP[dtype.lower()]
    Q = seq_len_q
    q_lens = _validate_lengths("q_lens", q_lens, B, Q)
    kv_lens = _validate_lengths("kv_lens", kv_lens, B, S)
    if any(request_query > request_kv
           for request_query, request_kv in zip(q_lens, kv_lens)):
        raise ValueError(
            "each FlashAttention request requires q_len <= kv_len: "
            f"q_lens={q_lens}, kv_lens={kv_lens}")
    if Q > 1 and num_splits != 1:
        raise ValueError("FlashAttention prefill requires num_splits=1")
    scale = 1.0 / (D ** 0.5)
    workload = {
        "batch": B,
        "seq_len_cache": S,
        "num_heads_q": HQ,
        "num_heads_kv": HKV,
        "head_dim": D,
        "dtype": dtype,
        "seq_len_q": seq_len_q,
        "num_splits": num_splits,
        "kv_layout": kv_layout,
        "kv_dist": kv_dist,
        "causal": causal,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
    }
    if capture_metadata is not None:
        workload["capture_identity"] = capture_metadata
    capture_expected_output_elements = (
        sum(q_lens) * HQ * D if capture_metadata is not None else None)
    if kv_layout in ("paged", "paged_varlen"):
        # vLLM paged attention. Q_LEN is per request for both decode and
        # prefill, so the flattened query contains exactly B * Q tokens.
        key_cache, value_cache, block_table = _paged_pool(kv_lens, HKV, D, dt, device)
        num_tokens = sum(q_lens)
        q = torch.randn(num_tokens, HQ, D, dtype=dt, device=device)
        out = torch.empty_like(q)
        cu = [0]
        for length in q_lens:
            cu.append(cu[-1] + length)
        cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        def fn():
            flash_attn_varlen_func(
                q, key_cache, value_cache, out=out,
                max_seqlen_q=Q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max(kv_lens), seqused_k=seqused_k,
                softmax_scale=scale, causal=causal,
                block_table=block_table, num_splits=num_splits)

        _run(
            fn, nvtx_range_name=nvtx_range_name,
            event_options=event_options, workload=workload,
            pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib,
            steady_state_public_op_repetitions=steady_state_public_op_repetitions,
            capture_expected_output_elements=capture_expected_output_elements,
            capture_output_tensor=out)

    elif kv_layout == "chunked_prefill":
        # Serving chunked-prefill: each request contributes one Q_LEN-sized
        # chunk. Keeping Q_LEN per request makes the dataset label exactly
        # describe the executed workload.
        total_q = sum(q_lens)
        q = torch.randn(total_q, HQ, D, dtype=dt, device=device)
        out = torch.empty_like(q)
        cu = [0]
        for ql in q_lens:
            cu.append(cu[-1] + ql)
        cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
        key_cache, value_cache, block_table = _paged_pool(kv_lens, HKV, D, dt, device)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        def fn():
            flash_attn_varlen_func(
                q, key_cache, value_cache, out=out,
                max_seqlen_q=max(q_lens), cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max(kv_lens), seqused_k=seqused_k,
                softmax_scale=scale, causal=True,
                block_table=block_table, num_splits=num_splits)

        _run(
            fn, nvtx_range_name=nvtx_range_name,
            event_options=event_options, workload=workload,
            pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib,
            steady_state_public_op_repetitions=steady_state_public_op_repetitions,
            capture_expected_output_elements=capture_expected_output_elements,
            capture_output_tensor=out)

    elif kv_layout == "mla":
        # DeepSeek MLA decode on A800 (Ampere): vllm triton_mla fallback
        # (flashmla/flashattn_mla are Hopper/FA3-only). Mirrors triton_mla
        # _forward_decode: q = cat(q_nope, q_pe) (B,HQ,nope+rope); paged KV
        # cache stores per-token (nope||rope) → unsqueeze head dim; MQA
        # (num_kv_heads=1, all HQ heads share 1 KV head); v = kv_c (nope slice).
        if Q != 1:
            raise ValueError("A800 MLA runner only supports decode with Q_LEN=1")
        if any(length != 1 for length in q_lens):
            raise ValueError("A800 MLA runner requires one query token per request")
        if any(length != S for length in kv_lens):
            raise ValueError("A800 MLA runner requires uniform KV lengths")
        from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
        rope_dim = 64
        nope_dim = D                                     # 512 (DeepSeek MLA kv_lora)
        qk_dim = nope_dim + rope_dim                     # 576
        blocks_per_req = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
        pool_blocks = blocks_per_req * B + 1
        # paged KV: per-token (nope||rope) packed. num_kv_heads=1 (MQA).
        kv_cache = torch.randn(pool_blocks, BLOCK_SIZE, qk_dim, dtype=dt, device=device)
        block_table = torch.arange(pool_blocks - 1, device=device,
                                   dtype=torch.int32).view(B, blocks_per_req)
        q = torch.cat([torch.randn(B, HQ, nope_dim, dtype=dt, device=device),
                       torch.randn(B, HQ, rope_dim, dtype=dt, device=device)],
                      dim=-1)                           # (B, HQ, 576)
        out0 = torch.zeros(B, HQ, nope_dim, dtype=dt, device=device)
        lse = torch.zeros(B, HQ, dtype=dt, device=device)
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        num_kv_splits = num_splits
        attn_logits = torch.empty(B, HQ, num_kv_splits, nope_dim + 1,
                                  dtype=torch.float32, device=device)
        kv = kv_cache.unsqueeze(2)
        kv_c = kv[..., :nope_dim]

        def fn():
            decode_attention_fwd(q, kv, kv_c, out0, lse,
                                 block_table, seq_lens, attn_logits,
                                 num_kv_splits, scale, BLOCK_SIZE)

        _run(
            fn,
            nvtx_range_name=nvtx_range_name,
            event_options=event_options,
            workload=workload,
            pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib,
            steady_state_public_op_repetitions=steady_state_public_op_repetitions,
            capture_expected_output_elements=capture_expected_output_elements,
            capture_output_tensor=out0,
        )

    elif kv_layout == "contiguous":
        # contiguous: Dao flash_attn (dense k/v). decode Q=1 → with_kvcache;
        # prefill Q=S causal → flash_attn_func.
        if Q > 1:  # prefill
            if any(length != Q for length in q_lens):
                raise ValueError(
                    "dense contiguous prefill requires uniform Q lengths")
            if any(length != S for length in kv_lens):
                raise ValueError(
                    "dense contiguous prefill requires uniform KV lengths")
            q = torch.randn(B, Q, HQ, D, dtype=dt, device=device)
            k = torch.randn(B, S, HKV, D, dtype=dt, device=device)
            v = torch.randn(B, S, HKV, D, dtype=dt, device=device)
            _run(
                lambda: flash_attn_func(
                    q, k, v, causal=causal, softmax_scale=scale),
                nvtx_range_name=nvtx_range_name,
                event_options=event_options,
                workload=workload,
                pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib,
                steady_state_public_op_repetitions=steady_state_public_op_repetitions,
                capture_expected_output_elements=capture_expected_output_elements,
            )
        else:      # decode (q=1)
            if any(length != 1 for length in q_lens):
                raise ValueError(
                    "dense contiguous decode requires one query token per request")
            if any(length != S for length in kv_lens):
                raise ValueError(
                    "dense contiguous decode requires uniform KV lengths")
            q = torch.randn(B, 1, HQ, D, dtype=dt, device=device)
            k_cache = torch.randn(B, S, HKV, D, dtype=dt, device=device)
            v_cache = torch.randn(B, S, HKV, D, dtype=dt, device=device)
            _run(
                lambda: flash_attn_with_kvcache(
                    q, k_cache, v_cache, causal=causal,
                    num_splits=num_splits, softmax_scale=scale),
                nvtx_range_name=nvtx_range_name,
                event_options=event_options,
                workload=workload,
                pre_nvtx_cache_eviction_mib=pre_nvtx_cache_eviction_mib,
                steady_state_public_op_repetitions=steady_state_public_op_repetitions,
                capture_expected_output_elements=capture_expected_output_elements,
            )
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
    p.add_argument("--capture-metadata-json")
    p.add_argument("--nvtx_range_name", type=str)
    p.add_argument("--pre-nvtx-cache-eviction-mib", type=int, default=0)
    p.add_argument(
        "--event-steady-state-public-op-repetitions", type=int, default=1)
    add_event_cli_arguments(p)
    a = p.parse_args()
    event_options = parse_event_run_options(a)
    torch.cuda.set_device(0)
    if event_options is not None:
        torch.manual_seed(20260826)
        torch.cuda.manual_seed(20260826)
    q_lens = json.loads(a.q_lens_json)
    kv_lens = json.loads(a.kv_lens_json)
    capture_metadata = None
    if a.capture_metadata_json is not None:
        capture_metadata = validate_runner_capture_metadata(
            a.capture_metadata_json,
            hardware_id="nvidia_a800",
            actual_workload={
                "dtype": a.dtype,
                "kv_layout": a.kv_layout,
                "kv_dist": a.kv_dist,
                "causal": bool(a.causal),
                "split_policy": "serving",
                "B": a.batch,
                "Q_LEN": a.seq_len_q,
                "KV_LEN": a.seq_len_cache,
                "HQ": a.num_heads_q,
                "HKV": a.num_heads_kv,
                "D": a.head_dim,
                "Q_LENS": q_lens,
                "KV_LENS": kv_lens,
            })
    run_flash_attn(a.batch, a.seq_len_cache, a.num_heads_q, a.num_heads_kv,
                   a.head_dim, a.dtype, a.seq_len_q, a.num_splits,
                   a.kv_layout, a.kv_dist, bool(a.causal), q_lens, kv_lens,
                   a.nvtx_range_name, event_options=event_options,
                   pre_nvtx_cache_eviction_mib=(
                       a.pre_nvtx_cache_eviction_mib),
                   steady_state_public_op_repetitions=(
                       a.event_steady_state_public_op_repetitions),
                   capture_metadata=capture_metadata)
    if capture_metadata is None:
        print("FA done")
    else:
        print(json.dumps({"capture_identity": capture_metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
