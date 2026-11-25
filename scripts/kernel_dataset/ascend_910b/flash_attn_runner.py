"""FlashAttention runner for 910B dataset profiling.

Uses torch_npu.npu_fused_infer_attention_score — the Ascend fused attention op
(single "mix" kernel using AI Core + Vector). GQA with HQ/HKV heads.
Layout BSH: q=(B,Q,HQ*D), k/v=(B,S,HKV*D) where Q = seq_len_q.

decode:  Q=1 (single query step against KV cache of length S) — original mode.
prefill: Q=S (full prompt prefill, O(S²) attention) — the serving prefill
workload whose AIC-scalar/mte2 breakdown the dataset previously lacked (all
existing entries are decode Q=1). Enable with --seq_len_q (default 1 = decode).

kv_layout:
  contiguous (default): k/v are dense (B,S,HKV*D) tensors — the original mode.
  paged: k/v live in a paged KV-cache pool (num_blocks, block_size, HKV*D)
  accessed through a block_table, mirroring vLLM-Ascend serving's paged layout
  (attention_v1._forward_fia_slidingwindow). The block_table is SCATTERED (each
  request's blocks are non-contiguous in the pool, like dynamic allocation) so
  the gather exercises real block-table indirection — the serving regime whose
  AIV-domain cost the contiguous dataset does not reproduce (serving decode FA
  is AIV/launch-bound, not gather-bound; msprof per-op truth shows aic_mte2 is
  ~1% of wall).
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch_npu


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    add_event_cli_arguments,
    measure_current_stream_events,
    parse_event_run_options,
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
    write_event_case_record,
)
from dataset_pipeline.kernel_dataset.fa_serving_capture import (
    validate_runner_capture_metadata,
)

DTYPE_MAP = {"fp16": torch.float16, "float16": torch.float16,
             "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


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


def _cumulative(lengths):
    result = []
    total = 0
    for length in lengths:
        total += length
        result.append(total)
    return result


def _validate_capture_output(output, expected_elements):
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        tensor = output[0]
    else:
        raise TypeError("captured FlashAttention public op returned no tensor output")
    if tensor.numel() != expected_elements:
        raise ValueError(
            "captured FlashAttention output shape disagrees with the request "
            f"contract: expected_elements={expected_elements}, actual={tensor.numel()}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("captured FlashAttention output contains a non-finite value")


def run_flash_attn(B, S, HQ, HKV, D, dtype, seq_len_q, kv_layout,
                   kv_dist, causal, q_lens, kv_lens, event_options=None,
                   profile_cache_eviction_mib=0, profile_warmup_iterations=20,
                   profile_sweep_operation="fill",
                   capture_metadata=None):
    validate_cache_eviction_mib(
        profile_cache_eviction_mib, minimum_nonzero_mib=1536,
        hardware_id="ascend_910b")
    if event_options is not None:
        if profile_cache_eviction_mib != 0:
            raise ValueError(
                "profile_cache_eviction_mib cannot be mixed with Event mode")
        validate_cache_eviction_mib(
            event_options.cache_eviction_mib, minimum_nonzero_mib=1536,
            hardware_id="ascend_910b")
    if min(B, S, HQ, HKV, D, seq_len_q) <= 0:
        raise ValueError("FlashAttention workload dimensions must be positive")
    if profile_warmup_iterations < 0:
        raise ValueError("profile_warmup_iterations must be non-negative")
    torch.npu.set_device(0)
    dt = DTYPE_MAP[dtype.lower()]
    Q = seq_len_q
    q_lens = _validate_lengths("q_lens", q_lens, B, Q)
    kv_lens = _validate_lengths("kv_lens", kv_lens, B, S)
    if any(request_query > request_kv
           for request_query, request_kv in zip(q_lens, kv_lens)):
        raise ValueError(
            "each FlashAttention request requires q_len <= kv_len: "
            f"q_lens={q_lens}, kv_lens={kv_lens}")
    scale = 1.0 / (D ** 0.5)
    # causal mask: prefill (Q>1) uses sparse_mode=3 (causal); decode (Q=1)
    # sparse_mode=0 (no mask). Matches vllm-ascend attention_v1 (decode=sm0,
    # prefill=sm3). Prefill also uses input_layout="TND" (vllm attention_v1.py:603)
    # with an int8 causal mask; decode uses "BSH" (attention_v1.py:742).
    sparse_mode = 3 if causal else 0
    layout = "TND" if causal else "BSH"
    # Prefill causal mask = vllm get_splitfuse_attn_mask
    # (attention_mask.py:56): torch.triu(torch.ones(2048,2048), diagonal=1).int8
    # — upper triangle (strictly above diagonal) = 1 = masked future, lower = 0
    # = attend. Verified CAUSAL via perturbation test (perturb late KV → early
    # query output unchanged, diff 0.0000) for contiguous+paged, S=1024/4096.
    # (An earlier "TND does full-attention" claim was a buggy SDPA-reference
    # comparison; the perturbation test is the definitive, ref-free check.)
    mask_kw = {}
    if causal:
        mask_kw["atten_mask"] = torch.triu(
            torch.ones(2048, 2048, dtype=torch.int8, device="npu"), 1)
    if kv_layout == "chunked_prefill":
        # Chunked-prefill: B requests, varlen Q (per-request Q varies),
        # total_Q = sum(q_lens) <= max_batched (~8192). TND q = all tokens
        # concatenated, actual_seq_q = cumulative per-request Q. Varlen KV via
        # paged block_table. int8 causal mask. Mirrors vllm chunked-prefill
        # (attention_v1.py:603, TND + actual_seq + paged + sparse_mode=3).
        block_size = 128
        total_q = sum(q_lens)
        q = torch.randn(total_q, HQ, D, dtype=dt, device="npu")  # TND concatenated
        actual_q = _cumulative(q_lens)
        blocks_per_req = [(l + block_size - 1) // block_size for l in kv_lens]
        num_blocks = sum(blocks_per_req)
        k_pool = torch.randn(num_blocks, block_size, HKV * D, dtype=dt, device="npu")
        v_pool = torch.randn(num_blocks, block_size, HKV * D, dtype=dt, device="npu")
        perm = torch.randperm(num_blocks, device="npu")
        block_table = torch.zeros(B, max(blocks_per_req), dtype=torch.int32, device="npu")
        idx = 0
        for i, nb in enumerate(blocks_per_req):
            block_table[i, :nb] = perm[idx:idx + nb]
            idx += nb

        def fa():
            return torch_npu.npu_fused_infer_attention_score(
                q, k_pool, v_pool, num_heads=HQ, num_key_value_heads=HKV,
                input_layout=layout, scale=scale, block_size=block_size,
                block_table=block_table, actual_seq_lengths=actual_q,
                actual_seq_lengths_kv=kv_lens, sparse_mode=sparse_mode, **mask_kw)
    elif kv_layout in ("paged", "paged_varlen"):
        # Paged KV cache: pool of blocks, each request's blocks scattered via
        # block_table (mirrors vLLM-Ascend serving). block_size=128 (vLLM default).
        block_size = 128
        if causal:
            q = torch.randn(sum(q_lens), HQ, D, dtype=dt, device="npu")
            actual_q = _cumulative(q_lens)
        else:
            q = torch.randn(B, Q, HQ * D, dtype=dt, device="npu")
            actual_q = q_lens
        blocks_per_req = [(l + block_size - 1) // block_size for l in kv_lens]
        num_blocks = sum(blocks_per_req)
        k_pool = torch.randn(num_blocks, block_size, HKV * D, dtype=dt, device="npu")
        v_pool = torch.randn(num_blocks, block_size, HKV * D, dtype=dt, device="npu")
        # Scattered block_table: shuffle block indices so each request's blocks
        # are non-contiguous in the pool (real paged access, not sequential).
        perm = torch.randperm(num_blocks, device="npu")
        block_table = torch.zeros(B, max(blocks_per_req), dtype=torch.int32, device="npu")
        idx = 0
        for i, nb in enumerate(blocks_per_req):
            block_table[i, :nb] = perm[idx:idx + nb]
            idx += nb
        actual_kv = kv_lens
        def fa():
            return torch_npu.npu_fused_infer_attention_score(
                q, k_pool, v_pool, num_heads=HQ, num_key_value_heads=HKV,
                input_layout=layout, scale=scale, block_size=block_size,
                block_table=block_table, actual_seq_lengths=actual_q,
                actual_seq_lengths_kv=actual_kv, sparse_mode=sparse_mode, **mask_kw)
    elif kv_layout == "mla":
        # DeepSeek MLA DECODE: paged BNSD_NBSD (vllm mla_v1.py:1283, the
        # _forward_decode else-branch). MLA PREFILL uses torch_npu.atb.npu_ring_mla
        # (an ATB ring-attention op that segfaults standalone — needs CP context),
        # so MLA prefill shapes were DELETED from the dataset. MLA decode = v2 op
        # with the paged k_nope/k_pe cache (num_blocks, HKV, block_size, dim),
        # block_table, BNSD_NBSD layout, sm=0 (decode attends all history).
        # nope_dim=512 (op tiling: qkHeadDim=vHeadDim ∈ {128,512}), rope=64,
        # v=k_nope. Verified runs (out0 must be 4D (HQ,B,1,nope)).
        rope_dim = 64
        nope_dim = D                                   # 512
        block_size = 128
        if any(length != S for length in kv_lens):
            raise ValueError("MLA cases require one uniform KV length")
        blocks_per_req = (S + block_size - 1) // block_size
        num_blocks = blocks_per_req * B
        k_nope = torch.randn(num_blocks, HKV, block_size, nope_dim, dtype=dt, device="npu")
        k_pe = torch.randn(num_blocks, HKV, block_size, rope_dim, dtype=dt, device="npu")
        v = k_nope                                     # MLA: v = k_nope
        block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").view(B, blocks_per_req)
        q_nope = torch.randn(B, HQ, 1, nope_dim, dtype=dt, device="npu")   # BNSD
        q_pe = torch.randn(B, HQ, 1, rope_dim, dtype=dt, device="npu")
        out0 = torch.zeros(HQ, B, 1, nope_dim, dtype=dt, device="npu")     # 4D out
        lse = torch.zeros(HQ, B, dtype=torch.float32, device="npu")

        def fa():
            return torch_npu.npu_fused_infer_attention_score_v2.out(
                q_nope, k_nope, v, query_rope=q_pe, key_rope=k_pe,
                num_query_heads=HQ, num_key_value_heads=HKV,
                input_layout="BNSD_NBSD", softmax_scale=scale, out=[out0, lse],
                sparse_mode=0, block_table=block_table, block_size=block_size,
                actual_seq_qlen=q_lens, actual_seq_kvlen=kv_lens)
    elif kv_layout == "contiguous":
        # contiguous: prefill (causal) TND k/v=(S,HKV,D)+actual_seq; decode BSH (B,S,HKV*D).
        if causal:
            q = torch.randn(sum(q_lens), HQ, D, dtype=dt, device="npu")
            k = torch.randn(sum(kv_lens), HKV, D, dtype=dt, device="npu")
            v = torch.randn(sum(kv_lens), HKV, D, dtype=dt, device="npu")
            asl_kw = dict(
                actual_seq_lengths=_cumulative(q_lens),
                actual_seq_lengths_kv=_cumulative(kv_lens))
        else:
            if any(length != Q for length in q_lens):
                raise ValueError("BSH decode requires uniform Q lengths")
            if any(length != S for length in kv_lens):
                raise ValueError("dense BSH decode requires uniform KV lengths")
            q = torch.randn(B, Q, HQ * D, dtype=dt, device="npu")
            k = torch.randn(B, S, HKV * D, dtype=dt, device="npu")
            v = torch.randn(B, S, HKV * D, dtype=dt, device="npu")
            asl_kw = {}

        def fa():
            return torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=HQ, num_key_value_heads=HKV,
                input_layout=layout, scale=scale, sparse_mode=sparse_mode,
                **asl_kw, **mask_kw)
    else:
        raise ValueError(f"unknown FlashAttention KV layout: {kv_layout!r}")

    if capture_metadata is not None:
        _validate_capture_output(fa(), sum(q_lens) * HQ * D)
        torch.npu.synchronize()
    if event_options is None:
        profile_sweep, _ = prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="npu",
            cache_eviction_mib=profile_cache_eviction_mib,
            sweep_operation=profile_sweep_operation)
        torch.npu.synchronize()
        for _ in range(profile_warmup_iterations):
            if profile_sweep is not None:
                profile_sweep()
            fa()
        torch.npu.synchronize()
        if profile_sweep is not None:
            profile_sweep()
        out = fa()
        torch.npu.synchronize()
        return out

    cache_eviction_operation, cache_eviction_bytes = (
        prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="npu",
            cache_eviction_mib=event_options.cache_eviction_mib))
    measurement = measure_current_stream_events(
        fa,
        device_module=torch.npu,
        warmup_iterations=event_options.warmup_iterations,
        sample_count=event_options.sample_count,
        cache_eviction_operation=cache_eviction_operation,
        cache_eviction_bytes=cache_eviction_bytes,
    )
    device_index = torch.npu.current_device()
    properties = torch.npu.get_device_properties(device_index)
    write_event_case_record(
        options=event_options,
        hardware_id="ascend_910b",
        family="flash_attn",
        runner_path=Path(__file__),
        timer={
            "api": "torch.npu.Event",
            "implementation": "ACL_Runtime_Event_via_torch_npu",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_applicable",
            "cache_condition": (
                "queued_pre_sample_device_sweep"
                if cache_eviction_bytes else "ordinary_warm_state"),
        },
        stream_contract={
            "event_stream": "torch.npu.current_stream",
            "all_op_streams_joined": True,
            "evidence": (
                "the exact torch_npu fused-attention call is submitted "
                "through the current NPU stream and the runner creates no "
                "secondary stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": "exact_runner_call",
            "workspace_behavior": "exact_runner_call",
            "evidence": (
                "query, KV tensors, mask, length metadata and block table "
                "are constructed before timing; Event mode preserves the "
                "exact public call and retains its result through the end "
                "Event"),
        },
        workload={
            "B": B, "S": S, "Q_LEN": max(q_lens), "HQ": HQ,
            "HKV": HKV, "D": D, "Q_LENS": q_lens,
            "KV_LENS": kv_lens, "kv_layout": kv_layout,
            "kv_dist": kv_dist, "causal": causal, "dtype": dtype,
            "cache_eviction_bytes_before_sample": cache_eviction_bytes,
            **({"capture_identity": capture_metadata}
               if capture_metadata is not None else {}),
        },
        environment={
            "torch_version": torch.__version__,
            "torch_npu_version": torch_npu.__version__,
            "device_index": device_index,
            "device_name": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "input_seed": 20260826,
        },
        measurement=measurement,
    )
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--seq_len_cache", type=int, required=True)
    p.add_argument("--num_heads_q", type=int, required=True)
    p.add_argument("--num_heads_kv", type=int, required=True)
    p.add_argument("--head_dim", type=int, required=True)
    p.add_argument("--dtype", required=True, choices=sorted(DTYPE_MAP))
    p.add_argument("--seq_len_q", type=int, required=True,
                   help="query length: 1=decode, =seq_len_cache for prefill")
    p.add_argument("--workload-json", required=True)
    p.add_argument("--kv_layout", required=True,
                   choices=["contiguous", "paged", "paged_varlen", "mla", "chunked_prefill"],
                   help="contiguous (default), paged, paged_varlen (serving), mla (DeepSeek), chunked_prefill (varlen Q)")
    p.add_argument("--kv_dist", required=True, choices=["skew", "uniform", "explicit"],
                   help="varlen distribution (paged_varlen only): skew=serving-like")
    p.add_argument("--causal", type=int, required=True, choices=[0, 1],
                   help="1=prefill causal mask (sparse_mode=3), 0=decode no mask")
    p.add_argument("--capture-metadata-json")
    p.add_argument("--profile-cache-eviction-mib", type=int, default=0)
    p.add_argument(
        "--profile-sweep-operation", choices=("add", "fill"), default="fill",
        help=("Cold-cache sweep operation before the profiled launch. "
              "'fill' is write-only; 'add' reads and writes, which also "
              "displaces read-allocated lines on caches that do not "
              "allocate on write."))
    p.add_argument(
        "--profile-warmup-iterations", type=int, default=20,
        help=("number of unmeasured public-op calls before the final profiling "
              "call; default preserves the existing capture protocol"))
    add_event_cli_arguments(p)
    a = p.parse_args()
    event_options = parse_event_run_options(a)
    if event_options is not None:
        torch.manual_seed(20260826)
        torch.npu.manual_seed(20260826)
    with open(a.workload_json, encoding="utf-8") as stream:
        workload = json.load(stream)
    if not isinstance(workload, dict):
        raise TypeError("flash_attn workload sidecar must be a mapping")
    if set(workload) != {"q_lens", "kv_lens"}:
        raise ValueError(
            "flash_attn workload sidecar must contain exactly q_lens and "
            f"kv_lens, got {sorted(workload)}")
    q_lens = workload["q_lens"]
    kv_lens = workload["kv_lens"]
    capture_metadata = None
    if a.capture_metadata_json is not None:
        capture_metadata = validate_runner_capture_metadata(
            a.capture_metadata_json,
            hardware_id="ascend_910b",
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
                   a.head_dim, a.dtype, a.seq_len_q, a.kv_layout, a.kv_dist,
                   bool(a.causal), q_lens, kv_lens,
                   event_options=event_options,
                   profile_cache_eviction_mib=a.profile_cache_eviction_mib,
                   profile_sweep_operation=a.profile_sweep_operation,
                   profile_warmup_iterations=a.profile_warmup_iterations,
                   capture_metadata=capture_metadata)
    if capture_metadata is None:
        print("FA done")
    else:
        print(json.dumps({"capture_identity": capture_metadata}, sort_keys=True))
