#!/usr/bin/env python3
"""Run the canonical MetaX Linear/GEMM workload with device Events."""

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    add_event_cli_arguments,
    measure_graph_replay_events,
    measure_current_stream_events,
    parse_event_run_options,
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
    write_event_case_record,
)


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def run_gemm(batch, sequence, input_size, output_size, dtype,
             event_options, graph_repetitions):
    validate_cache_eviction_mib(
        event_options.cache_eviction_mib, minimum_nonzero_mib=64,
        hardware_id="metax_c500")
    input_tensor = torch.randn(
        batch, sequence, input_size, device="cuda", dtype=dtype)
    linear = torch.nn.Linear(
        input_size, output_size, bias=False).cuda().to(dtype)

    def operation():
        with torch.no_grad():
            return linear(input_tensor)

    cache_eviction_operation, cache_eviction_bytes = (
        prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=event_options.cache_eviction_mib))
    if graph_repetitions is None:
        measurement = measure_current_stream_events(
            operation,
            device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes,
        )
        timer_mode = "one_exact_public_op_per_event_interval"
    else:
        measurement = measure_graph_replay_events(
            operation,
            device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            repetition_counts=graph_repetitions,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes,
        )
        timer_mode = "external_events_around_cuda_graph_replay"
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    write_event_case_record(
        options=event_options,
        hardware_id="metax_c500",
        family="gemm",
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
                "the exact torch.nn.Linear call is submitted through the "
                "current MetaX stream and the runner creates no secondary "
                "stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": "exact_runner_call",
            "workspace_behavior": "exact_runner_call",
            "evidence": (
                "input and Linear weights are constructed before timing; "
                "the exact linear(input_tensor) result is retained through "
                "the end Event"),
        },
        workload={
            "B": batch, "M": sequence, "K": input_size, "N": output_size,
            "dtype": dtype_name,
            ("cache_eviction_bytes_before_sample"
             if graph_repetitions is None
             else "cache_eviction_bytes_before_replay"): cache_eviction_bytes,
        },
        environment={
            "torch_version": torch.__version__,
            "torch_cuda_version_field": torch.version.cuda,
            "device_index": device_index,
            "device_name": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "input_seed": 20260826,
        },
        measurement=measurement,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", required=True, type=int)
    parser.add_argument("--sl", required=True, type=int)
    parser.add_argument("--in_feat", required=True, type=int)
    parser.add_argument("--out_feat", required=True, type=int)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), required=True)
    add_event_cli_arguments(parser)
    parser.add_argument("--event-graph-repetitions", nargs=2, type=int)
    args = parser.parse_args()
    event_options = parse_event_run_options(args)
    if event_options is None:
        raise ValueError("MetaX GEMM runner requires complete Event options")
    graph_repetitions = None
    if args.event_graph_repetitions is not None:
        graph_repetitions = tuple(args.event_graph_repetitions)
    torch.cuda.set_device(0)
    torch.manual_seed(20260826)
    torch.cuda.manual_seed(20260826)
    run_gemm(
        args.bs, args.sl, args.in_feat, args.out_feat,
        DTYPE_MAP[args.dtype], event_options, graph_repetitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
