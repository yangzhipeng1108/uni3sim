"""Compare A800 Host submit service with the pinned H200 torch.add probe.

This is an independent mechanism measurement. The output is a candidate for
cross-Host comparison, never an H200 profile or production duration fit.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch


CHAIN_LENGTHS = (1, 8, 64, 512)
TRIALS = 30
WARMUPS = 5
TENSOR_NUMEL = 4096
REFERENCE = Path(__file__).resolve().parents[4] / (
    "microbench/nvidia_h200/runtime/launch/torch_eager_graph_runner.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_state() -> str:
    result = subprocess.run(
        ["nvidia-smi", "-i", "0",
         "--query-gpu=index,name,uuid,memory.used,utilization.gpu,driver_version",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20)
    return result.stdout.strip()


def _run_chain(left, right, increment, count):
    source, destination = left, right
    for _ in range(count):
        torch.add(source, increment, out=destination)
        source, destination = destination, source
    return source


def _measure(call, divisor):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    call()
    end = time.perf_counter_ns()
    torch.cuda.synchronize()
    elapsed = end - start
    if elapsed <= 0:
        raise RuntimeError("Host submit interval must be positive")
    return {"host_submit_ns": elapsed,
            "host_submit_ns_per_dispatch": elapsed / divisor}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    expected_parent = (Path(__file__).resolve().parents[4] /
        "profiler_output/microbench/nvidia_a800/host_launch_proxy_20261029")
    if output.parent != expected_parent or output.exists():
        raise ValueError("output must be a new file in registered A800 campaign")
    if os.environ["CUDA_VISIBLE_DEVICES"] != "0":
        raise ValueError("probe owns physical A800 GPU0 only")
    before = _gpu_state()
    fields = [part.strip() for part in before.split(",")]
    if (len(fields) != 6 or fields[0] != "0"
            or fields[1] != "NVIDIA A800-SXM4-80GB"
            or int(fields[3]) > 16 or int(fields[4]) != 0):
        raise RuntimeError(f"A800 GPU0 is not idle: {before}")
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expected one visible GPU")
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
        samples = []
        for trial in range(TRIALS):
            cells = [(mode, count) for count in CHAIN_LENGTHS
                     for mode in ("eager", "graph_replay")]
            if trial % 2:
                cells.reverse()
            for mode, count in cells:
                if mode == "eager":
                    timing = _measure(
                        lambda count=count: _run_chain(
                            left, right, increment, count), count)
                else:
                    timing = _measure(graphs[count].replay, 1)
                samples.append({"trial": trial + 1, "mode": mode,
                                "chain_length": count, **timing})
    summary = {}
    for mode in ("eager", "graph_replay"):
        for count in CHAIN_LENGTHS:
            values = [sample["host_submit_ns_per_dispatch"]
                      for sample in samples if sample["mode"] == mode
                      and sample["chain_length"] == count]
            summary[f"{mode}:{count}"] = {
                "sample_count": len(values), "min": min(values),
                "median": statistics.median(values), "max": max(values)}
    del graphs, left, right, increment
    torch.cuda.empty_cache()
    cpu_models = {line.partition(":")[2].strip()
                  for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")}
    if len(cpu_models) != 1:
        raise RuntimeError("CPU identity ambiguous")
    payload = {
        "schema": "a800_h200_host_launch_proxy_candidate_v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_boundary": "host_submit_only_no_device_sync",
        "operation": "torch.add_out", "tensor_numel": TENSOR_NUMEL,
        "dtype": "float32", "chain_lengths": CHAIN_LENGTHS,
        "warmups": WARMUPS, "trials_per_cell": TRIALS,
        "source_sha256": _sha256(Path(__file__).resolve()),
        "h200_reference_source_sha256": _sha256(REFERENCE),
        "environment": {"hostname": platform.node(),
                        "cpu_model": next(iter(cpu_models)),
                        "affinity_cpus": len(os.sched_getaffinity(0)),
                        "python": sys.version, "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda},
        "gpu_before": before, "gpu_after": _gpu_state(),
        "samples": samples, "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "summary_512": {
        "eager": summary["eager:512"]["median"],
        "graph_replay": summary["graph_replay:512"]["median"]}},
        sort_keys=True))


if __name__ == "__main__":
    main()
