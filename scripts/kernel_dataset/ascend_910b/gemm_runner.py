"""GEMM runner for 910B dataset profiling (torch.matmul on NPU).

Mirrors the C500/A800 GEMM workload: y = x @ w^T for a Linear layer of shape
(K -> N), input (B, SL, K). Runs warmup + profiled launch so msprof op captures
the steady-state kernel.
"""
import argparse
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


def run_gemm(B, SL, K, N, dtype=torch.bfloat16,
             profile_cache_eviction_mib=0, event_options=None):
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
    torch.npu.set_device(0)
    x = torch.randn(B, SL, K, device="npu", dtype=dtype)
    w = torch.nn.Linear(K, N, bias=False).npu().to(dtype)
    def operation():
        with torch.no_grad():
            return w(x)

    if event_options is None:
        profile_sweep, _ = prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="npu",
            cache_eviction_mib=profile_cache_eviction_mib,
            sweep_operation="fill")
        torch.npu.synchronize()
        with torch.no_grad():
            for _ in range(5):
                if profile_sweep is not None:
                    profile_sweep()
                y = w(x)
        torch.npu.synchronize()
        if profile_sweep is not None:
            profile_sweep()
        y = operation()
        torch.npu.synchronize()
        return y

    cache_eviction_operation, cache_eviction_bytes = (
        prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="npu",
            cache_eviction_mib=event_options.cache_eviction_mib))
    measurement = measure_current_stream_events(
        operation,
        device_module=torch.npu,
        warmup_iterations=event_options.warmup_iterations,
        sample_count=event_options.sample_count,
        cache_eviction_operation=cache_eviction_operation,
        cache_eviction_bytes=cache_eviction_bytes,
    )
    device_index = torch.npu.current_device()
    properties = torch.npu.get_device_properties(device_index)
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    write_event_case_record(
        options=event_options,
        hardware_id="ascend_910b",
        family="gemm",
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
                "the exact torch.nn.Linear call is submitted through the "
                "current NPU stream and the runner creates no secondary "
                "stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": "exact_runner_call",
            "workspace_behavior": "exact_runner_call",
            "evidence": (
                "input and Linear weights are constructed before timing; "
                "the exact w(x) result is retained through the end Event"),
        },
        workload={
            "B": B, "M": SL, "K": K, "N": N, "dtype": dtype_name,
            "cache_eviction_bytes_before_sample": cache_eviction_bytes,
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
    p.add_argument("--bs", type=int, required=True)
    p.add_argument("--sl", type=int, required=True)
    p.add_argument("--in_feat", type=int, required=True)
    p.add_argument("--out_feat", type=int, required=True)
    p.add_argument(
        "--dtype", choices=("bf16", "fp16", "float16"), default="bf16")
    p.add_argument("--profile-cache-eviction-mib", type=int, default=0)
    add_event_cli_arguments(p)
    a = p.parse_args()
    event_options = parse_event_run_options(a)
    if event_options is not None:
        torch.manual_seed(20260826)
        torch.npu.manual_seed(20260826)
    if a.dtype == "bf16":
        dt = torch.bfloat16
    elif a.dtype in ("fp16", "float16"):
        dt = torch.float16
    else:
        raise ValueError(f"unsupported GEMM dtype: {a.dtype!r}")
    run_gemm(
        a.bs, a.sl, a.in_feat, a.out_feat, dt,
        profile_cache_eviction_mib=a.profile_cache_eviction_mib,
        event_options=event_options)
    print("GEMM done")
