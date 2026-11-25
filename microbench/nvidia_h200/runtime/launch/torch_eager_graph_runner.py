#!/usr/bin/env python3
"""Measure PyTorch CUDA eager enqueue and CUDA Graph replay submission."""

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
import subprocess
import sys
import time
import traceback

import torch


HARDWARE_ID = "nvidia_h200"
EXPECTED_DEVICE_NAME = "NVIDIA H200"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)
CHAIN_LENGTHS = (1, 8, 64, 512)
TRIALS = 30
WARMUPS = 5
TENSOR_NUMEL = 4096
GIT_SHA = re.compile(r"[0-9a-f]{40}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _output_dir(path):
    resolved = path.resolve()
    owner = Path(*OUTPUT_OWNER_PARTS)
    if owner not in tuple(Path(*resolved.parts[:index]) for index in range(1, len(resolved.parts) + 1)):
        if not any(
            resolved.parts[index:index + len(OUTPUT_OWNER_PARTS)] == OUTPUT_OWNER_PARTS
            for index in range(len(resolved.parts) - len(OUTPUT_OWNER_PARTS) + 1)
        ):
            raise ValueError("output must be below profiler_output/microbench/nvidia_h200")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def _device_state(device_id):
    completed = subprocess.run([
        "nvidia-smi", "-i", str(device_id),
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True)
    values = tuple(value.strip() for value in completed.stdout.strip().split(","))
    if len(values) != 6:
        raise RuntimeError("unexpected nvidia-smi output")
    return {
        "index": int(values[0]), "name": values[1], "uuid": values[2],
        "memory_used_mib": int(values[3]), "utilization_gpu_pct": int(values[4]),
        "driver_version": values[5],
    }


def _run_chain(left, right, increment, count):
    source, destination = left, right
    for _ in range(count):
        torch.add(source, increment, out=destination)
        source, destination = destination, source
    return source


def _measure(call, divisor):
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    call()
    ended = time.perf_counter_ns()
    torch.cuda.synchronize()
    elapsed = ended - started
    if elapsed <= 0:
        raise RuntimeError("Host submit interval must be positive")
    return {"host_submit_ns": elapsed, "host_submit_ns_per_dispatch": elapsed / divisor}


def _summary(samples):
    values = sorted(float(sample["host_submit_ns_per_dispatch"]) for sample in samples)
    return {
        "sample_count": len(values), "min": values[0],
        "median": statistics.median(values), "max": values[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    if args.device_id < 0 or GIT_SHA.fullmatch(args.baseline_commit) is None:
        parser.error("device-id and full lowercase baseline commit are required")
    output = _output_dir(args.output_dir)
    source = Path(__file__).resolve()
    source_sha = _sha256(source)
    if source_sha != args.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from expected snapshot")
    before = _device_state(args.device_id)
    if before["name"] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected target device: {before}")
    if before["utilization_gpu_pct"] != 0 or before["memory_used_mib"] > 16:
        raise RuntimeError(f"target GPU is not idle: {before}")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    payload = {
        "schema_version": 1, "status": "initializing", "hardware_id": HARDWARE_ID,
        "baseline_commit": args.baseline_commit,
        "source": {"path": str(source), "sha256": source_sha},
        "command": [sys.executable, *sys.argv], "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"hostname": platform.node(), "platform": platform.platform(),
                        "python": sys.version, "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda},
        "device_before": before, "device_after": None,
        "contract": {"boundary": "host_submit_only_no_device_sync",
                     "operation": "torch.add_out", "tensor_numel": TENSOR_NUMEL,
                     "dtype": "float32", "chain_lengths": list(CHAIN_LENGTHS),
                     "trials_per_cell": TRIALS, "warmups": WARMUPS},
        "samples": [], "summary": None, "failure": None,
    }
    _write_json(manifest_path, payload)
    try:
        torch.cuda.set_device(args.device_id)
        left = torch.zeros(TENSOR_NUMEL, device="cuda")
        right = torch.empty_like(left)
        increment = torch.ones_like(left)
        stream = torch.cuda.Stream()
        graphs = {}
        with torch.cuda.stream(stream):
            for count in CHAIN_LENGTHS:
                for _ in range(WARMUPS):
                    _run_chain(left, right, increment, count)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    _run_chain(left, right, increment, count)
                graph.replay()
                stream.synchronize()
                graphs[count] = graph
            for trial in range(TRIALS):
                cells = [(mode, count) for count in CHAIN_LENGTHS for mode in ("eager", "graph_replay")]
                if trial % 2:
                    cells.reverse()
                for mode, count in cells:
                    operation, repetitions = {
                        "eager": (
                            lambda count=count: _run_chain(
                                left, right, increment, count), count),
                        "graph_replay": (graphs[count].replay, 1),
                    }[mode]
                    timing = _measure(operation, repetitions)
                    payload["samples"].append({"trial": trial + 1, "mode": mode,
                                               "chain_length": count, **timing})
        grouped = {}
        for mode in ("eager", "graph_replay"):
            for count in CHAIN_LENGTHS:
                cell = [sample for sample in payload["samples"]
                        if sample["mode"] == mode and sample["chain_length"] == count]
                grouped[f"{mode}:{count}"] = _summary(cell)
        del graphs, left, right, increment
        torch.cuda.empty_cache()
        payload["device_after"] = _device_state(args.device_id)
        payload["summary"] = grouped
        payload["status"] = "complete"
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, payload)
    except BaseException as error:
        payload["status"] = "failed"
        payload["failure"] = {"type": type(error).__name__, "message": str(error),
                              "traceback": traceback.format_exc()}
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, payload)
        raise


if __name__ == "__main__":
    main()
