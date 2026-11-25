#!/usr/bin/env python3
"""Measure the Ascend ACL-graph replay synchronization API baseline.

The benchmark is intentionally independent of Serving and model workloads.
It measures ``torch.npu.current_stream().synchronize()`` after a prior
``synchronize`` (the non-blocking API-call service) and after a synthetic
``torch.add`` launch (the barrier behavior).  The latter is retained as
evidence only: a production replay wait is represented as a device-timeline
barrier, not as a fitted constant.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
import torch_npu


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)
TRIALS = 32
WARMUP_REPEATS = 8
SYNTHETIC_SHAPE = (256, 256)
SCHEMA_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--baseline-commit", type=str, required=True)
    arguments = parser.parse_args()
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    return arguments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _append_json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    owner_size = len(OUTPUT_OWNER_PARTS)
    offsets = tuple(
        index
        for index in range(len(resolved.parts) - owner_size + 1)
        if resolved.parts[index:index + owner_size] == OUTPUT_OWNER_PARTS
    )
    if len(offsets) != 1 or len(resolved.parts) <= offsets[0] + owner_size:
        raise ValueError("--output-dir must be a run below the Ascend owner")
    if resolved.exists():
        raise FileExistsError("--output-dir must not already exist")
    return resolved


def _median(values: list[int]) -> float:
    if not values:
        raise ValueError("median requires samples")
    return float(statistics.median(values))


def _measure(device_id: int, output_dir: Path) -> dict[str, object]:
    device = torch.device(f"npu:{device_id}")
    source = torch.ones(SYNTHETIC_SHAPE, dtype=torch.float32, device=device)
    destination = torch.empty_like(source)
    stream = torch.npu.current_stream(device)
    for _ in range(WARMUP_REPEATS):
        torch.add(source, 1, out=destination)
        stream.synchronize()

    trials_path = output_dir / "trials.jsonl"
    idle_wall = []
    idle_cpu = []
    pending_wall = []
    pending_cpu = []
    for trial_index in range(TRIALS):
        stream.synchronize()
        wall_start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        stream.synchronize()
        cpu_end = time.thread_time_ns()
        wall_end = time.perf_counter_ns()
        idle_wall.append(wall_end - wall_start)
        idle_cpu.append(cpu_end - cpu_start)

        torch.add(source, 1, out=destination)
        wall_start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        stream.synchronize()
        cpu_end = time.thread_time_ns()
        wall_end = time.perf_counter_ns()
        pending_wall.append(wall_end - wall_start)
        pending_cpu.append(cpu_end - cpu_start)
        _append_json_line(trials_path, {
            "device_id": device_id,
            "trial_index": trial_index,
            "idle_sync_wall_ns": idle_wall[-1],
            "idle_sync_thread_cpu_ns": idle_cpu[-1],
            "pending_sync_wall_ns": pending_wall[-1],
            "pending_sync_thread_cpu_ns": pending_cpu[-1],
        })

    for values in (idle_wall, idle_cpu, pending_wall, pending_cpu):
        if any(value <= 0 for value in values):
            raise RuntimeError("synchronization measurements must be positive")
    return {
        "device_id": device_id,
        "shape": list(SYNTHETIC_SHAPE),
        "sample_count": TRIALS,
        "idle_sync_wall_ns": {"median": _median(idle_wall)},
        "idle_sync_thread_cpu_ns": {"median": _median(idle_cpu)},
        "pending_sync_wall_ns": {"median": _median(pending_wall)},
        "pending_sync_thread_cpu_ns": {"median": _median(pending_cpu)},
    }


def main() -> None:
    arguments = _arguments()
    if isinstance(arguments.device_id, bool) or arguments.device_id not in (0, 1):
        raise ValueError("--device-id must be one of the authorized devices 0/1")
    if (len(arguments.baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.baseline_commit)):
        raise ValueError("--baseline-commit must be a lowercase full Git SHA")
    output_dir = _validate_output_dir(arguments.output_dir)
    output_dir.mkdir(parents=True)
    summary = _measure(arguments.device_id, output_dir)
    runner_path = Path(__file__).resolve()
    summary.update({
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "ACL graph replay current-stream synchronization; idle API service "
            "plus a synthetic pending-device barrier probe"),
        "measurement_boundary": (
            "current_stream().synchronize() call wall/thread CPU; synthetic "
            "torch.add is independent of Serving"),
        "service_time_s": {
            "graph_replay_sync_api_idle": (
                summary["idle_sync_wall_ns"]["median"] / 1_000_000_000),
        },
        "forward_use": (
            "barrier semantics only; idle API value is provenance evidence and "
            "is not added as a per-step residual"),
        "source": {
            "path": str(runner_path),
            "sha256": _sha256(runner_path),
            "baseline_commit": arguments.baseline_commit,
        },
        "environment": {
            "container": arguments.container_name,
            "device_id": arguments.device_id,
            "host": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "predeclared": {
            "trials": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "synthetic_shape": list(SYNTHETIC_SHAPE),
        },
        "completed_at": _utc_now(),
    })
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", {
        "schema_version": 1,
        "kind": "ascend_acl_graph_replay_sync_microbenchmark",
        "summary_sha256": _sha256(output_dir / "summary.json"),
        "trials_sha256": _sha256(output_dir / "trials.jsonl"),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
