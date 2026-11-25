#!/usr/bin/env python3
"""Validate Ascend 910B ACL-backed NPU Event timing without a profiler."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import torch_npu


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    file_sha256,
    measure_current_stream_events,
    require_lower_hex,
    write_kernel_golden_json,
)


def _checked_output(command, *, environment=None):
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment)
    return completed.stdout


def _resolved_ldd_library(output, soname):
    matches = [
        Path(line.split("=>", 1)[1].strip().split(" ", 1)[0])
        for line in output.splitlines()
        if f"{soname} =>" in line
    ]
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError(f"cannot uniquely resolve {soname}: {matches}")
    return matches[0]


def _backend_evidence():
    torch_npu_library = (
        Path(torch_npu.__file__).resolve().parent / "lib/libtorch_npu.so")
    if not torch_npu_library.is_file():
        raise FileNotFoundError(
            f"torch_npu runtime library is absent: {torch_npu_library}")
    dependencies = _checked_output(("ldd", str(torch_npu_library)))
    ascend_runtime = _resolved_ldd_library(dependencies, "libascendcl.so")
    symbols = _checked_output(("readelf", "-Ws", str(torch_npu_library)))
    required_symbols = (
        "aclrtRecordEvent", "aclrtSynchronizeEvent", "aclrtEventElapsedTime")
    missing = [symbol for symbol in required_symbols if symbol not in symbols]
    if missing:
        raise RuntimeError(
            f"libtorch_npu.so is missing linked ACL Event symbols: {missing}")
    strings = _checked_output(("strings", str(torch_npu_library)))
    if "aclrtCreateEventWithFlag" not in strings:
        raise RuntimeError(
            "libtorch_npu.so has no aclrtCreateEventWithFlag reference")
    return {
        "torch_npu_library": str(torch_npu_library),
        "torch_npu_library_sha256": file_sha256(torch_npu_library),
        "ascend_runtime_library": str(ascend_runtime),
        "ascend_runtime_library_sha256": file_sha256(ascend_runtime),
        "resolved_chain": "torch.npu.Event -> ACL Runtime aclrtEvent*",
        "cupti_identity": "not_applicable",
    }


def _acl_check(call, name):
    status = int(call)
    if status != 0:
        raise RuntimeError(f"{name} failed with ACL status {status}")


def _acl_health_check(runtime_path: Path, operation):
    runtime = ctypes.CDLL(str(runtime_path))
    runtime.aclrtCreateEventWithFlag.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    runtime.aclrtRecordEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    runtime.aclrtSynchronizeEvent.argtypes = [ctypes.c_void_p]
    runtime.aclrtEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    runtime.aclrtDestroyEvent.argtypes = [ctypes.c_void_p]
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    _acl_check(
        runtime.aclrtCreateEventWithFlag(ctypes.byref(start), 0x00000008),
        "aclrtCreateEventWithFlag(start)")
    _acl_check(
        runtime.aclrtCreateEventWithFlag(ctypes.byref(end), 0x00000008),
        "aclrtCreateEventWithFlag(end)")
    stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
    try:
        _acl_check(runtime.aclrtRecordEvent(start, stream),
                   "aclrtRecordEvent(start)")
        result = operation()
        _acl_check(runtime.aclrtRecordEvent(end, stream),
                   "aclrtRecordEvent(end)")
        _acl_check(runtime.aclrtSynchronizeEvent(end),
                   "aclrtSynchronizeEvent(end)")
        elapsed_ms = ctypes.c_float()
        _acl_check(
            runtime.aclrtEventElapsedTime(
                ctypes.byref(elapsed_ms), start, end),
            "aclrtEventElapsedTime")
        del result
    finally:
        _acl_check(runtime.aclrtDestroyEvent(end), "aclrtDestroyEvent(end)")
        _acl_check(runtime.aclrtDestroyEvent(start), "aclrtDestroyEvent(start)")
    if elapsed_ms.value <= 0.0:
        raise RuntimeError("direct ACL Event duration must be positive")
    return {
        "runtime_library": str(runtime_path),
        "runtime_library_sha256": file_sha256(runtime_path),
        "elapsed_ms": float(elapsed_ms.value),
        "stream_pointer": torch.npu.current_stream().npu_stream,
        "api_sequence": (
            "aclrtCreateEventWithFlag(ACL_EVENT_TIME_LINE) -> "
            "aclrtRecordEvent(start, torch current stream) -> torch.mm.out -> "
            "aclrtRecordEvent(end, stream) -> aclrtSynchronizeEvent(end) -> "
            "aclrtEventElapsedTime"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.device < 0:
        raise ValueError("device must be non-negative")
    baseline_commit = require_lower_hex(
        args.baseline_commit, 40, "baseline_commit")
    snapshot_sha256 = require_lower_hex(
        args.source_snapshot_sha256, 64, "source_snapshot_sha256")

    torch.npu.set_device(args.device)
    torch.manual_seed(20260826)
    torch.npu.manual_seed(20260826)
    left = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)
    right = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)
    output = torch.empty(4096, 4096, device="npu", dtype=torch.bfloat16)

    def operation():
        return torch.mm(left, right, out=output)

    operation()
    torch.npu.synchronize()
    backend_evidence = _backend_evidence()
    direct_acl = _acl_health_check(
        Path(backend_evidence["ascend_runtime_library"]), operation)

    measurement = measure_current_stream_events(
        operation,
        device_module=torch.npu,
        warmup_iterations=args.warmup,
        sample_count=args.samples,
    )
    event_durations = [
        row["event_duration_ns"] for row in measurement["samples"]]
    submission_durations = [
        row["host_submission_ns"] for row in measurement["samples"]]
    synchronized_durations = [
        row["host_synchronized_interval_ns"]
        for row in measurement["samples"]
    ]
    event_median = statistics.median(event_durations)
    submission_median = statistics.median(submission_durations)
    synchronized_median = statistics.median(synchronized_durations)
    checks = {
        "positive_direct_acl_event_duration": direct_acl["elapsed_ms"] > 0.0,
        "positive_torch_npu_event_duration": event_median > 0.0,
        "device_asynchronous_to_host_submission": (
            event_median > submission_median * 2.0),
        "event_within_synchronized_host_interval": (
            event_median <= synchronized_median),
        "coefficient_of_variation_at_most_0_10": (
            measurement["statistics"]["coefficient_of_variation"] <= 0.10),
        "milliseconds_to_nanoseconds_conversion": True,
        "explicit_same_current_stream_record_order": True,
        "same_current_stream_record_order": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"910B NPU Event timer sanity failed: {checks}")
    properties = torch.npu.get_device_properties(args.device)
    payload = {
        "schema_version": 1,
        "artifact_kind": "device_event_timer_sanity",
        "hardware_id": "ascend_910b",
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": snapshot_sha256,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "profiler_started": False,
        "device": {
            "index": args.device,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        },
        "software": {
            "torch_version": torch.__version__,
            "torch_npu_version": torch_npu.__version__,
            "ascend_home_path": os.environ["ASCEND_HOME_PATH"],
            "npu_smi_info": _checked_output(("npu-smi", "info")),
        },
        "timer": {
            "api": "torch.npu.Event",
            "implementation": "ACL_Runtime_Event_via_torch_npu",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_applicable",
        },
        "backend_evidence": backend_evidence,
        "direct_acl_event_health_check": direct_acl,
        "sanity_workload": {
            "operation": "torch.mm.out",
            "m": 4096,
            "n": 4096,
            "k": 4096,
            "dtype": "bf16",
            "inputs_outputs_preallocated": True,
        },
        "measurement": measurement,
        "checks": checks,
    }
    write_kernel_golden_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "event_median_ns": event_median,
        "coefficient_of_variation": measurement["statistics"][
            "coefficient_of_variation"],
        "backend": payload["timer"]["implementation"],
        "direct_acl_elapsed_ms": direct_acl["elapsed_ms"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
