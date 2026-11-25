#!/usr/bin/env python3
"""Validate NVIDIA A800 CUDA Event timing without a profiler."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    file_sha256,
    measure_current_stream_events,
    require_lower_hex,
    write_kernel_golden_json,
)


def _checked_output(command):
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True)
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
    torch_cuda = Path(torch.__file__).resolve().parent / "lib/libtorch_cuda.so"
    if not torch_cuda.is_file():
        raise FileNotFoundError(f"PyTorch CUDA library is absent: {torch_cuda}")
    dependencies = _checked_output(("ldd", str(torch_cuda)))
    cuda_runtime = _resolved_ldd_library(dependencies, "libcudart.so.12")
    symbols = _checked_output(("readelf", "-Ws", str(torch_cuda)))
    for symbol in (
            "cudaEventCreate", "cudaEventRecord", "cudaEventSynchronize",
            "cudaEventElapsedTime"):
        if symbol not in symbols:
            raise RuntimeError(
                f"libtorch_cuda.so is missing CUDA Event symbol {symbol}")
    return {
        "torch_cuda_library": str(torch_cuda),
        "torch_cuda_library_sha256": file_sha256(torch_cuda),
        "cuda_runtime_library": str(cuda_runtime),
        "cuda_runtime_library_sha256": file_sha256(cuda_runtime),
        "resolved_chain": "torch.cuda.Event -> CUDA Runtime cudaEvent*",
        "cupti_identity": "not_used",
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

    torch.cuda.set_device(args.device)
    torch.manual_seed(20260826)
    torch.cuda.manual_seed(20260826)
    left = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(4096, 4096, device="cuda", dtype=torch.bfloat16)

    def operation():
        torch.mm(left, right, out=output)

    measurement = measure_current_stream_events(
        operation,
        device_module=torch.cuda,
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
        "positive_event_duration": event_median > 0.0,
        "device_asynchronous_to_host_submission": (
            event_median > submission_median * 2.0),
        "event_within_synchronized_host_interval": (
            event_median <= synchronized_median),
        "coefficient_of_variation_at_most_0_10": (
            measurement["statistics"]["coefficient_of_variation"] <= 0.10),
        "milliseconds_to_nanoseconds_conversion": True,
        "explicit_same_current_stream_record_order": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"A800 CUDA Event timer sanity failed: {checks}")
    properties = torch.cuda.get_device_properties(args.device)
    driver_version = _checked_output((
        "nvidia-smi", "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    )).splitlines()[args.device].strip()
    payload = {
        "schema_version": 1,
        "artifact_kind": "device_event_timer_sanity",
        "hardware_id": "nvidia_a800",
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
            "torch_cuda_version": torch.version.cuda,
            "nvidia_driver_version": driver_version,
        },
        "timer": {
            "api": "torch.cuda.Event",
            "implementation": "CUDA_Runtime_Event",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_used",
        },
        "backend_evidence": _backend_evidence(),
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
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
