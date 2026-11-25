#!/usr/bin/env python3
"""Measure MACA PyTorch eager enqueue and Graph replay Host submission."""

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


HARDWARE_ID = "metax_c500"
# The profiler container carries the stack the profile was first measured on;
# the vLLM serving container carries the stack the serving capture ran on.
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _output_dir(path):
    resolved = path.resolve()
    if not any(
        resolved.parts[index:index + len(OUTPUT_OWNER_PARTS)] == OUTPUT_OWNER_PARTS
        for index in range(len(resolved.parts) - len(OUTPUT_OWNER_PARTS) + 1)
    ):
        raise ValueError("output must be below profiler_output/microbench/metax_c500")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


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
    return {"sample_count": len(values), "min": values[0],
            "median": statistics.median(values), "max": values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    args = parser.parse_args()
    if (args.device_id < 0 or GIT_SHA.fullmatch(args.baseline_commit) is None
            or not args.container_name.strip()):
        parser.error("device, baseline commit, and container identity are required")
    output = _output_dir(args.output_dir)
    source = Path(__file__).resolve()
    source_sha = _sha256(source)
    if source_sha != args.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from expected snapshot")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    payload = {
        "schema_version": 1, "status": "initializing", "hardware_id": HARDWARE_ID,
        "baseline_commit": args.baseline_commit,
        "source": {"path": str(source), "sha256": source_sha},
        "command": [sys.executable, *sys.argv], "pid": os.getpid(),
        "container_name": args.container_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"hostname": platform.node(), "platform": platform.platform(),
                        "python": sys.version, "torch": torch.__version__,
                        "maca_cuda_compatibility": torch.version.cuda},
        "device": None,
        "contract": {"boundary": "host_submit_only_no_device_sync",
                     "operation": "torch.add_out", "tensor_numel": TENSOR_NUMEL,
                     "dtype": "float32", "chain_lengths": list(CHAIN_LENGTHS),
                     "trials_per_cell": TRIALS, "warmups": WARMUPS,
                     "profiler_attached": False},
        "samples": [], "summary": None, "failure": None,
    }
    _write_json(manifest_path, payload)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("MetaX CUDA-compatible device is unavailable")
        torch.cuda.set_device(args.device_id)
        properties = torch.cuda.get_device_properties(args.device_id)
        payload["device"] = {"device_id": args.device_id, "name": properties.name,
                             "total_memory_bytes": properties.total_memory,
                             "multi_processor_count": properties.multi_processor_count}
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
