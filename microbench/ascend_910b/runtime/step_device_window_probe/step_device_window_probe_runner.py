#!/usr/bin/env python3
"""Measure the host cost of per-step device event timestamps without a profiler.

A per-step device window is the one quantity the Scheduler trace does not carry
and msprof cannot supply (msprof adds ~460 ms of host time per step, which
destroys the cadence being measured).  The candidate replacement is a pair of
``torch.npu.Event`` timestamps recorded around each step and read back after the
run, so no synchronisation sits on the submission path.  This probe measures
what that instrumentation costs: the same op chain is submitted with and without
the event pair, and the difference is the per-step overhead a real engine would
pay.

Three arms are recorded.  ``bare`` submits the chain alone.  ``events`` records
one event before and one after each chain and defers every read to the end.
``events_sync`` synchronises on the closing event each step, which is the naive
implementation and is expected to be far more expensive; it is kept so the cost
of getting that detail wrong is on the record.  No production operator is
replayed, no serving dataset is read, and no production or evaluation duration
is consumed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import traceback

import torch
import torch_npu


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")

# One probe "step" is a chain of independent BF16 elementwise ops sized like the
# rotary slice services, so the submission path resembles a real eager step's
# per-node dispatch without replaying any production operator.
HALF_HEAD_DIMENSION = 64
ELEMENTS = 2048
OPS_PER_STEP = (64, 256, 579)
MODES = ("bare", "events", "events_sync")
STEPS_PER_TRIAL = 40
TRIALS = 20
WARMUP_REPEATS = 3
SCHEMA_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _cann_identity():
    ascend_home = Path(os.environ["ASCEND_HOME_PATH"])
    version_path = ascend_home / "opp" / "version.info"
    return {
        "ascend_home_path": str(ascend_home),
        "opp_version_path": str(version_path),
        "opp_version_text": version_path.read_text(encoding="utf-8"),
        "opp_version_sha256": _sha256(version_path),
    }


def _device_identity(device_id):
    properties = torch.npu.get_device_properties(device_id)
    return {
        "device_id": device_id,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "cube_core_count": int(properties.cube_core_num),
        "vector_core_count": int(properties.vector_core_num),
        "properties_repr": repr(properties),
    }


def _validate_output_dir(path):
    resolved = path.resolve()
    owner_size = len(OUTPUT_OWNER_PARTS)
    owner_offsets = tuple(
        index
        for index in range(len(resolved.parts) - owner_size + 1)
        if resolved.parts[index:index + owner_size] == OUTPUT_OWNER_PARTS
    )
    if len(owner_offsets) != 1:
        raise ValueError(
            "--output-dir must be under exactly one "
            "profiler_output/microbench/ascend_910b owner")
    if len(resolved.parts) <= owner_offsets[0] + owner_size:
        raise ValueError("--output-dir must include a run-specific directory")
    if resolved.exists():
        raise FileExistsError("--output-dir must not already exist")
    return resolved


def _operands(device):
    rows = ELEMENTS // HALF_HEAD_DIMENSION
    shape = (rows, HALF_HEAD_DIMENSION)
    left = torch.arange(
        ELEMENTS, dtype=torch.float32, device=device
    ).reshape(shape).to(torch.bfloat16)
    right = torch.arange(
        ELEMENTS, 0, -1, dtype=torch.float32, device=device
    ).reshape(shape).to(torch.bfloat16)
    destination = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return left, right, destination


def _submit_step(left, right, destination, ops):
    for _ in range(ops):
        torch.mul(left, right, out=destination)


def _run_trial(stream, operands, ops, mode):
    left, right, destination = operands
    recorded = []
    started_ns = time.perf_counter_ns()
    for _ in range(STEPS_PER_TRIAL):
        if mode == "bare":
            _submit_step(left, right, destination, ops)
            continue
        opening = torch.npu.Event(enable_timing=True)
        closing = torch.npu.Event(enable_timing=True)
        opening.record(stream)
        _submit_step(left, right, destination, ops)
        closing.record(stream)
        if mode == "events_sync":
            closing.synchronize()
        recorded.append((opening, closing))
    submitted_ns = time.perf_counter_ns() - started_ns
    torch.npu.synchronize()
    windows_ms = [
        float(opening.elapsed_time(closing)) for opening, closing in recorded]
    if mode != "bare" and len(windows_ms) != STEPS_PER_TRIAL:
        raise RuntimeError("device window count disagrees with the step count")
    if any(not math.isfinite(value) or value <= 0 for value in windows_ms):
        raise RuntimeError("device window contains a non-positive value")
    return submitted_ns, windows_ms


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if args.run_index < 1:
        parser.error("--run-index must be positive")
    if BASELINE_PATTERN.fullmatch(args.baseline_commit) is None:
        parser.error("--baseline-commit must be a full lowercase Git commit")
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if len(args.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    args.output_dir = _validate_output_dir(args.output_dir)
    return args


def main():
    args = _parse_args()
    source_path = Path(__file__).resolve()
    actual_source_sha256 = _sha256(source_path)
    if actual_source_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            "runner source SHA256 differs from the audited snapshot")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.json"
    trials_path = args.output_dir / "samples.csv"
    summary_path = args.output_dir / "summary.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "run_index": args.run_index,
        "baseline_commit": args.baseline_commit,
        "source": {"path": str(source_path), "sha256": actual_source_sha256},
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "container_name": args.container_name,
        "device": None,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "cann": None,
        },
        "contract": {
            "mechanism": (
                "host submission cost of one torch.npu.Event pair recorded "
                "around a step and read back after the run, against the same "
                "submission without events"),
            "elements_per_op": ELEMENTS,
            "ops_per_step": list(OPS_PER_STEP),
            "modes": list(MODES),
            "steps_per_trial": STEPS_PER_TRIAL,
            "trials": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "host_clock": "time.perf_counter_ns",
            "device_clock": "torch.npu.Event(enable_timing=True)",
            "production_operator_replayed": False,
            "production_or_evaluation_duration_consumed": False,
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
        "outputs": {"samples": str(trials_path), "summary": str(summary_path)},
    }
    _write_json(manifest_path, manifest)
    try:
        manifest["environment"]["cann"] = _cann_identity()
        torch.npu.set_device(args.device_id)
        manifest["device"] = _device_identity(args.device_id)
        device = f"npu:{args.device_id}"
        stream = torch.npu.Stream(device=device)
        samples = []
        with torch.npu.stream(stream):
            operands = _operands(device)
            for ops in OPS_PER_STEP:
                for mode in MODES:
                    for _ in range(WARMUP_REPEATS):
                        _run_trial(stream, operands, ops, mode)
            for trial in range(TRIALS):
                for ops in OPS_PER_STEP:
                    for mode in MODES:
                        submitted_ns, windows_ms = _run_trial(
                            stream, operands, ops, mode)
                        samples.append({
                            "ops": ops,
                            "mode": mode,
                            "trial": trial,
                            "submit_ns_per_step": submitted_ns / STEPS_PER_TRIAL,
                            "device_window_ms": (
                                statistics.median(windows_ms)
                                if windows_ms else 0.0),
                        })
        with trials_path.open("w", encoding="utf-8") as handle:
            handle.write("ops,mode,trial,submit_ns_per_step,device_window_ms\n")
            for sample in samples:
                handle.write(
                    f"{sample['ops']},{sample['mode']},{sample['trial']},"
                    f"{sample['submit_ns_per_step']:.3f},"
                    f"{sample['device_window_ms']:.6f}\n")
            handle.flush()
            os.fsync(handle.fileno())
        summary = {}
        for ops in OPS_PER_STEP:
            for mode in MODES:
                values = [
                    sample["submit_ns_per_step"] for sample in samples
                    if sample["ops"] == ops and sample["mode"] == mode]
                if len(values) != TRIALS:
                    raise RuntimeError(
                        "sample count disagrees with the trial contract")
                windows = [
                    sample["device_window_ms"] for sample in samples
                    if sample["ops"] == ops and sample["mode"] == mode]
                summary[f"{ops}/{mode}"] = {
                    "median_submit_ns_per_step": statistics.median(values),
                    "min_submit_ns_per_step": min(values),
                    "max_submit_ns_per_step": max(values),
                    "median_device_window_ms": statistics.median(windows),
                }
        _write_json(summary_path, summary)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
    except Exception:
        manifest["status"] = "failed"
        manifest["failure"] = traceback.format_exc()
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "run_index": args.run_index,
        "samples": str(trials_path),
    }))


if __name__ == "__main__":
    main()
