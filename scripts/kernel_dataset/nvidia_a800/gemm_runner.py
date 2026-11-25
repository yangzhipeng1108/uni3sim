"""GEMM runner for the A800 single-op NCU dataset (torch.matmul on CUDA)."""

import argparse
import importlib.metadata
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    add_event_cli_arguments,
    build_nvidia_smi_telemetry_observer,
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


def run_gemm(batch_size, seq_len, in_features, out_features,
             nvtx_range_name, dtype=torch.bfloat16, event_options=None,
             pre_nvtx_cache_eviction_mib=0,
             steady_state_public_op_repetitions=1):
    if (isinstance(steady_state_public_op_repetitions, bool)
            or not isinstance(steady_state_public_op_repetitions, int)
            or steady_state_public_op_repetitions not in {1, 8, 32}):
        raise ValueError(
            "A800 GEMM steady-state Event repetitions must be one of 1, 8, 32")
    if event_options is not None:
        validate_cache_eviction_mib(
            event_options.cache_eviction_mib, minimum_nonzero_mib=320,
            hardware_id="nvidia_a800")
    # print(f"Running GEMM with BS={batch_size}, SL={seq_len}, In_Features={in_features}, Out_Features={out_features}, Dtype={dtype}")
    
    input_tensor = torch.randn(batch_size, seq_len, in_features, device='cuda', dtype=dtype)
    linear_layer = torch.nn.Linear(in_features, out_features, bias=False).cuda().to(dtype)

    def operation():
        with torch.no_grad():
            return linear_layer(input_tensor)

    if event_options is None:
        if steady_state_public_op_repetitions != 1:
            raise ValueError(
                "A800 GEMM steady-state Event repetitions require Event mode")
        if not isinstance(nvtx_range_name, str) or not nvtx_range_name:
            raise ValueError("normal A800 GEMM mode requires an NVTX name")
        validate_cache_eviction_mib(
            pre_nvtx_cache_eviction_mib, minimum_nonzero_mib=320,
            hardware_id="nvidia_a800")
        cache_eviction_operation, _ = prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=pre_nvtx_cache_eviction_mib)
        torch.cuda.synchronize()
        if cache_eviction_operation is not None:
            cache_eviction_operation()
            torch.cuda.current_stream().synchronize()
        torch.cuda.nvtx.range_push(nvtx_range_name)
        output = operation()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return output
    if nvtx_range_name is not None:
        raise ValueError("A800 GEMM Event mode cannot enable an NVTX range")
    if pre_nvtx_cache_eviction_mib:
        raise ValueError(
            "A800 GEMM Event mode cannot enable a pre-NVTX cache sweep")
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
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    write_event_case_record(
        options=event_options,
        hardware_id="nvidia_a800",
        family="gemm",
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
                "torch.nn.Linear dispatches the canonical GEMM launch graph "
                "on the current CUDA stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": "exact_runner_call",
            "workspace_behavior": "exact_runner_call",
            "evidence": (
                "input and Linear weights are constructed before timing; "
                "the exact linear_layer(input_tensor) result is retained "
                "through the end Event"),
        },
        workload={
            "batch": batch_size,
            "seq_len": seq_len,
            "in_features": in_features,
            "out_features": out_features,
            "dtype": next(
                name for name, value in DTYPE_MAP.items()
                if value == dtype and name in {"bf16", "fp16"}),
            "cache_eviction_bytes_before_sample": cache_eviction_bytes,
            "steady_state_event_contract": {
                "public_op_repetition_count": steady_state_public_op_repetitions,
                "cache_state": (
                    "one_op_per_cleaned_event_interval"
                    if steady_state_public_op_repetitions == 1 else
                    "cache_eviction_before_aggregate_with_intra_aggregate_reuse"),
            },
        },
        environment={
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "vllm_version": importlib.metadata.version("vllm"),
            "device_index": device_index,
            "device_name": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "input_seed": 20260826,
            "telemetry_physical_gpu_index": event_options.telemetry_gpu_index,
        },
        measurement=measurement,
    )
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--sl", type=int, required=True)
    parser.add_argument("--in_feat", type=int, required=True)
    parser.add_argument("--out_feat", type=int, required=True)
    parser.add_argument(
        "--dtype", choices=sorted(DTYPE_MAP), default="bf16")
    parser.add_argument("--nvtx_range_name", type=str)
    parser.add_argument("--pre-nvtx-cache-eviction-mib", type=int, default=0)
    parser.add_argument(
        "--event-steady-state-public-op-repetitions", type=int, default=1)
    add_event_cli_arguments(parser)

    args = parser.parse_args()
    event_options = parse_event_run_options(args)
    torch.cuda.set_device(0)
    if event_options is not None:
        torch.manual_seed(20260826)
        torch.cuda.manual_seed(20260826)

    run_gemm(
        args.bs, args.sl, args.in_feat, args.out_feat,
        args.nvtx_range_name, DTYPE_MAP[args.dtype],
        event_options=event_options,
        pre_nvtx_cache_eviction_mib=args.pre_nvtx_cache_eviction_mib,
        steady_state_public_op_repetitions=(
            args.event_steady_state_public_op_repetitions))
