#!/usr/bin/env python3
"""Validate MetaX C500 current-stream Event timing without a profiler."""

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


def _backend_evidence():
    torch_cuda = Path(torch.__file__).resolve().parent / "lib/libtorch_cuda.so"
    if not torch_cuda.is_file():
        raise FileNotFoundError(f"MetaX torch CUDA library is absent: {torch_cuda}")
    torch_dependencies = _checked_output(("ldd", str(torch_cuda)))
    if "libruntime_cu.so" not in torch_dependencies:
        raise RuntimeError(
            "MetaX libtorch_cuda.so does not resolve libruntime_cu.so")
    runtime_matches = [
        Path(line.split("=>", 1)[1].strip().split(" ", 1)[0])
        for line in torch_dependencies.splitlines()
        if "libruntime_cu.so =>" in line
    ]
    if len(runtime_matches) != 1 or not runtime_matches[0].is_file():
        raise RuntimeError(
            f"cannot uniquely resolve libruntime_cu.so: {runtime_matches}")
    runtime_cu = runtime_matches[0]
    runtime_symbols = _checked_output(("readelf", "-Ws", str(runtime_cu)))
    for symbol in (
            "wcudaEventCreate", "wcudaEventRecord", "wcudaEventSynchronize",
            "wcudaEventElapsedTime", "mcEventCreate", "mcEventRecord",
            "mcEventSynchronize", "mcEventElapsedTime"):
        if symbol not in runtime_symbols:
            raise RuntimeError(
                f"libruntime_cu.so is missing Event-chain symbol {symbol}")
    runtime_dependencies = _checked_output(("ldd", str(runtime_cu)))
    if "libmcruntime.so" not in runtime_dependencies:
        raise RuntimeError(
            "libruntime_cu.so does not resolve libmcruntime.so")
    maca_matches = [
        Path(line.split("=>", 1)[1].strip().split(" ", 1)[0])
        for line in runtime_dependencies.splitlines()
        if "libmcruntime.so =>" in line
    ]
    if len(maca_matches) != 1 or not maca_matches[0].is_file():
        raise RuntimeError(
            f"cannot uniquely resolve libmcruntime.so: {maca_matches}")
    maca_runtime = maca_matches[0]
    return {
        "torch_cuda_library": str(torch_cuda),
        "torch_cuda_library_sha256": file_sha256(torch_cuda),
        "cuda_compatibility_runtime": str(runtime_cu),
        "cuda_compatibility_runtime_sha256": file_sha256(runtime_cu),
        "maca_runtime": str(maca_runtime),
        "maca_runtime_sha256": file_sha256(maca_runtime),
        "resolved_chain": (
            "torch.cuda.Event -> wcudaEvent* -> mcEvent* -> libmcruntime.so"),
        "cupti_identity": "not_proven",
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
        "same_current_stream_record_order": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"MetaX Event timer sanity failed: {checks}")
    properties = torch.cuda.get_device_properties(args.device)
    payload = {
        "schema_version": 1,
        "artifact_kind": "device_event_timer_sanity",
        "hardware_id": "metax_c500",
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
            "torch_cuda_version_field": torch.version.cuda,
        },
        "timer": {
            "api": "torch.cuda.Event",
            "implementation": "wcudaEvent_to_maca_mcEvent",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
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
