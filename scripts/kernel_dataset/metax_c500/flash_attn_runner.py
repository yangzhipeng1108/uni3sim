"""Serving-aligned MetaX C500 FlashAttention profiler runner.

The exact per-request query and KV lengths are supplied by the canonical
workload expander.  This runner never regenerates them from a distribution
label, so A800, Ascend 910B and MetaX C500 execute the same semantic case.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    EventRunOptions,
    add_event_cli_arguments,
    measure_graph_replay_events,
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
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
BLOCK_SIZE = 16


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


def _cu_seqlens(lengths):
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    return torch.tensor(cumulative, dtype=torch.int32, device="cuda")


def _paged_pool(kv_lens, kv_heads, head_dim, dtype):
    blocks_per_request = [
        (length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in kv_lens]
    num_blocks = sum(blocks_per_request)
    key = torch.randn(
        num_blocks, BLOCK_SIZE, kv_heads, head_dim,
        dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    permutation = torch.randperm(num_blocks, device="cuda")
    block_table = torch.zeros(
        len(kv_lens), max(blocks_per_request),
        dtype=torch.int32, device="cuda")
    offset = 0
    for request, count in enumerate(blocks_per_request):
        block_table[request, :count] = permutation[offset:offset + count]
        offset += count
    return key, value, block_table


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


def _warm_run(function, profiler_wait, use_marker):
    torch.cuda.synchronize()
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    if profiler_wait:
        time.sleep(profiler_wait)
    if use_marker:
        torch.cuda.profiler.start()
    function()
    torch.cuda.synchronize()
    if use_marker:
        torch.cuda.profiler.stop()


def _environment():
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version_field": torch.version.cuda,
        "flash_attn_version": importlib.metadata.version("flash-attn"),
        "device_index": device_index,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "input_seed": 20260826,
    }


def _run(function, *, profiler_wait, use_marker,
         event_options: EventRunOptions | None, workload,
         graph_repetitions):
    if event_options is None:
        _warm_run(function, profiler_wait, use_marker)
        return
    validate_cache_eviction_mib(
        event_options.cache_eviction_mib, minimum_nonzero_mib=64,
        hardware_id="metax_c500")
    if use_marker or profiler_wait != 0:
        raise ValueError(
            "MetaX Event mode cannot enable profiler marker/wait options")
    cache_eviction_operation, cache_eviction_bytes = (
        prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=event_options.cache_eviction_mib))
    if graph_repetitions is None:
        measurement = measure_current_stream_events(
            function,
            device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes,
        )
        timer_mode = "one_exact_public_op_per_event_interval"
    else:
        measurement = measure_graph_replay_events(
            function,
            device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            repetition_counts=graph_repetitions,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes,
        )
        timer_mode = "external_events_around_cuda_graph_replay"
    write_event_case_record(
        options=event_options,
        hardware_id="metax_c500",
        family="flash_attn",
        runner_path=Path(__file__),
        timer={
            "api": "torch.cuda.Event",
            "implementation": "wcudaEvent_to_maca_mcEvent",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_proven",
            "mode": timer_mode,
            "cache_condition": (
                ("queued_pre_sample_device_sweep"
                 if graph_repetitions is None
                 else "queued_pre_replay_device_sweep_with_intra_replay_reuse")
                if cache_eviction_bytes else "ordinary_warm_state"),
        },
        stream_contract={
            "event_stream": "torch.cuda.current_stream",
            "all_op_streams_joined": True,
            "evidence": (
                "the exact FlashAttention call is submitted through the "
                "current MetaX stream and the runner creates no secondary "
                "stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": "exact_runner_call",
            "workspace_behavior": "exact_runner_call",
            "evidence": (
                "query, KV tensors, length metadata and block tables are "
                "constructed before timing; the exact public call and any "
                "internal allocation are retained through the end Event"),
        },
        workload={
            **workload,
            ("cache_eviction_bytes_before_sample"
             if graph_repetitions is None
             else "cache_eviction_bytes_before_replay"): cache_eviction_bytes,
        },
        environment=_environment(),
        measurement=measurement,
    )


def _split_values(executed_splits):
    if isinstance(executed_splits, bool):
        raise TypeError("FlashMLA executed_splits cannot be boolean")
    if isinstance(executed_splits, int):
        values = [executed_splits]
    elif isinstance(executed_splits, torch.Tensor):
        values = executed_splits.detach().cpu().reshape(-1).tolist()
    else:
        raise TypeError(
            "FlashMLA executed_splits must be an integer or tensor")
    if (not values
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in values)):
        raise ValueError(
            f"invalid FlashMLA executed_splits: {values!r}")
    return values


def _write_split_metadata(path, executed_splits):
    if path is None:
        raise ValueError("FlashMLA requires split_metadata_path")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({"executed_splits": _split_values(executed_splits)}, stream)


def run_flash_attn(batch, kv_length, query_heads, kv_heads, head_dim,
                   dtype, query_length, requested_splits, kv_layout, kv_dist,
                   causal, q_lens, kv_lens, *, profiler_wait=0,
                   use_marker=False, split_metadata_path=None,
                   metadata_only=False, event_options=None,
                   graph_repetitions=None, capture_metadata=None):
    if min(batch, kv_length, query_heads, kv_heads, head_dim,
           query_length) <= 0:
        raise ValueError("FlashAttention dimensions must be positive")
    q_lens = _validate_lengths("q_lens", q_lens, batch, query_length)
    kv_lens = _validate_lengths("kv_lens", kv_lens, batch, kv_length)
    if metadata_only and kv_layout != "mla":
        raise ValueError("metadata_only is supported only for MetaX MLA")
    torch_dtype = DTYPE_MAP[dtype.lower()]
    scale = 1.0 / (head_dim ** 0.5)

    if kv_layout == "mla":
        if requested_splits is not None:
            raise ValueError(
                "MetaX MLA split is metadata-selected, not caller-requested")
        if any(length != 1 for length in q_lens):
            raise ValueError("MetaX MLA runner supports decode only")
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        cache_seqlens = torch.tensor(
            kv_lens, dtype=torch.int32, device="cuda")
        scheduler, executed_splits = get_mla_metadata(
            cache_seqlens, query_heads, 1)
        torch.cuda.synchronize()
        if event_options is None:
            _write_split_metadata(split_metadata_path, executed_splits)
        elif split_metadata_path is not None:
            raise ValueError(
                "MetaX MLA Event mode does not write split metadata")
        if metadata_only:
            return

        rope_dim = 64
        blocks_per_request = [
            (length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in kv_lens]
        num_blocks = sum(blocks_per_request)
        kv_content = torch.randn(
            num_blocks, BLOCK_SIZE, head_dim,
            dtype=torch_dtype, device="cuda")
        key_rope = torch.randn(
            num_blocks, BLOCK_SIZE, rope_dim,
            dtype=torch_dtype, device="cuda")
        cache = torch.cat([kv_content, key_rope], dim=-1).unsqueeze(-2)
        block_table = torch.zeros(
            batch, max(blocks_per_request),
            dtype=torch.int32, device="cuda")
        offset = 0
        for request, count in enumerate(blocks_per_request):
            block_table[request, :count] = torch.arange(
                offset, offset + count, dtype=torch.int32, device="cuda")
            offset += count
        query = torch.cat([
            torch.randn(
                batch, query_heads, head_dim,
                dtype=torch_dtype, device="cuda"),
            torch.randn(
                batch, query_heads, rope_dim,
                dtype=torch_dtype, device="cuda"),
        ], dim=-1).unsqueeze(1)

        def function():
            # vLLM's serving backend derives scheduling metadata on every
            # forward call. Keep it inside the marker window so profiling sees
            # the same metadata/fill kernels as serving.
            current_scheduler, current_splits = get_mla_metadata(
                cache_seqlens, query_heads, 1)
            return flash_mla_with_kvcache(
                q=query, k_cache=cache, block_table=block_table,
                cache_seqlens=cache_seqlens, head_dim_v=head_dim,
                tile_scheduler_metadata=current_scheduler,
                num_splits=current_splits, softmax_scale=scale,
                causal=True)

    elif kv_layout in ("paged", "paged_varlen") and query_length == 1:
        if requested_splits is None or requested_splits <= 0:
            raise ValueError("MetaX paged decode requires requested_splits")
        if any(length != 1 for length in q_lens):
            raise ValueError("MetaX paged decode requires Q_LEN=1")
        key, value, block_table = _paged_pool(
            kv_lens, kv_heads, head_dim, torch_dtype)
        query = torch.randn(
            batch, 1, query_heads, head_dim,
            dtype=torch_dtype, device="cuda")
        cache_seqlens = torch.tensor(
            kv_lens, dtype=torch.int32, device="cuda")

        def function():
            return flash_attn_with_kvcache(
                query, key, value, block_table=block_table,
                cache_seqlens=cache_seqlens, softmax_scale=scale,
                causal=causal, num_splits=requested_splits)

    elif kv_layout in ("paged", "paged_varlen", "chunked_prefill"):
        if requested_splits is None or requested_splits <= 0:
            raise ValueError("MetaX varlen attention requires requested_splits")
        if kv_layout == "chunked_prefill":
            attention_causal = True
        elif kv_layout in ("paged", "paged_varlen"):
            attention_causal = causal
        else:
            raise ValueError(f"unsupported MetaX KV layout: {kv_layout!r}")
        key, value, block_table = _paged_pool(
            kv_lens, kv_heads, head_dim, torch_dtype)
        query = torch.randn(
            sum(q_lens), query_heads, head_dim,
            dtype=torch_dtype, device="cuda")
        query_cumulative = _cu_seqlens(q_lens)
        kv_cumulative = _cu_seqlens(kv_lens)

        def function():
            return flash_attn_varlen_func(
                query, key, value, query_cumulative, kv_cumulative,
                max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
                softmax_scale=scale,
                causal=attention_causal,
                block_table=block_table)

    elif kv_layout == "contiguous":
        if requested_splits is None or requested_splits <= 0:
            raise ValueError("MetaX contiguous attention requires requested_splits")
        if query_length > 1:
            query = torch.randn(
                sum(q_lens), query_heads, head_dim,
                dtype=torch_dtype, device="cuda")
            key = torch.randn(
                sum(kv_lens), kv_heads, head_dim,
                dtype=torch_dtype, device="cuda")
            value = torch.randn_like(key)
            query_cumulative = _cu_seqlens(q_lens)
            kv_cumulative = _cu_seqlens(kv_lens)

            def function():
                return flash_attn_varlen_func(
                    query, key, value, query_cumulative, kv_cumulative,
                    max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
                    softmax_scale=scale, causal=causal)
        else:
            if any(length != kv_length for length in kv_lens):
                raise ValueError(
                    "dense contiguous decode requires uniform KV lengths")
            query = torch.randn(
                batch, 1, query_heads, head_dim,
                dtype=torch_dtype, device="cuda")
            key = torch.randn(
                batch, kv_length, kv_heads, head_dim,
                dtype=torch_dtype, device="cuda")
            value = torch.randn_like(key)

            def function():
                return flash_attn_with_kvcache(
                    query, key, value, softmax_scale=scale,
                    causal=causal, num_splits=requested_splits)
    else:
        raise ValueError(f"unsupported MetaX KV layout: {kv_layout!r}")

    if capture_metadata is not None:
        _validate_capture_output(
            function(), sum(q_lens) * query_heads * head_dim)
        torch.cuda.synchronize()
    _run(
        function,
        profiler_wait=profiler_wait,
        use_marker=use_marker,
        event_options=event_options,
        graph_repetitions=graph_repetitions,
        workload={
            "B": batch,
            "S": kv_length,
            "Q_LEN": max(q_lens),
            "HQ": query_heads,
            "HKV": kv_heads,
            "D": head_dim,
            "SP_REQUESTED": requested_splits,
            "Q_LENS": q_lens,
            "KV_LENS": kv_lens,
            "kv_layout": kv_layout,
            "kv_dist": kv_dist,
            "causal": causal,
            "dtype": dtype,
            **({"capture_identity": capture_metadata}
               if capture_metadata is not None else {}),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq_len_cache", type=int, required=True)
    parser.add_argument("--num_heads_q", type=int, required=True)
    parser.add_argument("--num_heads_kv", type=int, required=True)
    parser.add_argument("--head_dim", type=int, required=True)
    parser.add_argument("--num_splits", type=int)
    parser.add_argument("--dtype", required=True, choices=sorted(DTYPE_MAP))
    parser.add_argument("--seq_len_q", type=int, required=True)
    parser.add_argument(
        "--kv_layout", required=True,
        choices=("contiguous", "paged", "paged_varlen", "mla",
                 "chunked_prefill"))
    parser.add_argument("--kv_dist", required=True, choices=("skew", "uniform", "explicit"))
    parser.add_argument("--q-lens-json", required=True)
    parser.add_argument("--kv-lens-json", required=True)
    parser.add_argument("--causal", type=int, required=True, choices=(0, 1))
    parser.add_argument("--capture-metadata-json")
    parser.add_argument("--profiler_wait", type=int, default=0)
    parser.add_argument("--use_marker", action="store_true")
    parser.add_argument("--split-metadata-path")
    parser.add_argument("--metadata-only", action="store_true")
    add_event_cli_arguments(parser)
    parser.add_argument("--event-graph-repetitions", nargs=2, type=int)
    args = parser.parse_args()
    event_options = parse_event_run_options(args)
    graph_repetitions = None
    if args.event_graph_repetitions is not None:
        if event_options is None:
            raise ValueError(
                "event_graph_repetitions requires complete Event options")
        graph_repetitions = tuple(args.event_graph_repetitions)
    if event_options is not None:
        if args.metadata_only:
            raise ValueError("MetaX Event mode cannot be metadata-only")
        torch.manual_seed(20260826)
        torch.cuda.manual_seed(20260826)
    q_lens = json.loads(args.q_lens_json)
    kv_lens = json.loads(args.kv_lens_json)
    capture_metadata = None
    if args.capture_metadata_json is not None:
        capture_metadata = validate_runner_capture_metadata(
            args.capture_metadata_json,
            hardware_id="metax_c500",
            actual_workload={
                "dtype": args.dtype,
                "kv_layout": args.kv_layout,
                "kv_dist": args.kv_dist,
                "causal": bool(args.causal),
                "split_policy": "serving",
                "B": args.batch,
                "Q_LEN": args.seq_len_q,
                "KV_LEN": args.seq_len_cache,
                "HQ": args.num_heads_q,
                "HKV": args.num_heads_kv,
                "D": args.head_dim,
                "Q_LENS": q_lens,
                "KV_LENS": kv_lens,
            })
    run_flash_attn(
        args.batch, args.seq_len_cache, args.num_heads_q,
        args.num_heads_kv, args.head_dim, args.dtype, args.seq_len_q,
        args.num_splits, args.kv_layout, args.kv_dist, bool(args.causal),
        q_lens, kv_lens,
        profiler_wait=args.profiler_wait, use_marker=args.use_marker,
        split_metadata_path=args.split_metadata_path,
        metadata_only=args.metadata_only, event_options=event_options,
        graph_repetitions=graph_repetitions, capture_metadata=capture_metadata)
    if capture_metadata is not None:
        print(json.dumps({"capture_identity": capture_metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
