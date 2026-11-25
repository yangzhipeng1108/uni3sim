#!/usr/bin/env python3
"""Ascend 910B HCCL collective latency and bandwidth sweep.

Measures the device-timeline duration of one HCCL collective as a function of
message size and launch pattern.  It is the 910B counterpart of the MetaX C500
MCCL sweep and of the independent NCCL Ring AllReduce microbenchmark that
supplies the A800 ProtoBridge protocol-policy constants, and it exists for the
same reason: the 910B entries in ``src/kernel/launch_config`` are round
development placeholders (60/60/60 us startup, 0.2 us per phase) rather than
measurements.

It is a communication mechanism sweep: synthetic byte counts, a constant filled
payload and workload-independent participation.  No production operator, model
shape, kernel or Serving duration takes part, and nothing it emits is selected
by comparing a Midlayer prediction against a production measurement.

Two launch patterns are recorded for every cell:

``back_to_back``
    ``--repeats`` collectives issued without an intervening host
    synchronisation, bounded by one pair of events.  Dividing by the repeat
    count amortises the launch gap and equalises rank skew, so this is the
    steady-state per-collective device cost.

``isolated``
    one collective per event pair with a device synchronisation and a process
    barrier in front of it.  The gap between the two patterns is the
    per-collective launch and rank-skew cost that a back-to-back stream hides.

Whether the eager patterns contain a per-collective launch gap at all is
decided by the measurement itself: sweeping ``--repeats`` changes how deeply
the launches are queued ahead, so a per-collective time that is flat in the
repeat count is device-bound and a falling one is host-launch-bound.

Graph replay is deliberately absent rather than optional: capturing an HCCL
collective into an Ascend graph is not a validated path on this deployment, and
an unvalidated capture would silently change what is being timed.

Example, inside container ``configured serving container``::

    ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \\
        microbench/ascend_910b/communication/collectives/\\
collective_protocol_latency_runner.py \\
        --run-id hccl_latency_20260921_r1 --source-commit <sha> \\
        --output-dir profiler_output/collective/ascend_910b/\\
hccl_latency_20260921_r1 \\
        --operation all_reduce \\
        --message-bytes 4096,16384,65536,262144,1048576,4194304,16777216
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = Path(__file__).resolve()
SNAPSHOT_MANIFEST = REPOSITORY_ROOT / "SOURCE_SNAPSHOT.json"
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output" / "collective" / "ascend_910b"
CANN_VERSION_FILE = Path("/usr/local/Ascend/cann-8.5.1/opp/version.info")
DRIVER_VERSION_FILE = Path("/usr/local/Ascend/driver/version.info")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
OPERATIONS = ("all_reduce", "all_gather")
PATTERNS = ("back_to_back", "isolated")
BF16_BYTES = 2


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_message_bytes(text: str) -> tuple[int, ...]:
    values = tuple(_positive_integer(value) for value in text.split(","))
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("message byte values must be unique")
    return tuple(sorted(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: tuple[str, ...]) -> str:
    return subprocess.run(
        command, cwd=REPOSITORY_ROOT, check=True, capture_output=True,
        text=True).stdout.strip()


def _validate_source_identity(source_commit: str) -> dict[str, object]:
    """Bind the emitted parameters to one immutable copy of this runner."""

    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("--source-commit must be a full lowercase Git commit")
    if (REPOSITORY_ROOT / ".git").exists():
        if _command_output(("git", "rev-parse", "HEAD")) != source_commit:
            raise RuntimeError(
                "--source-commit does not match the deployed HEAD")
        _command_output(
            ("git", "ls-files", "--error-unmatch", str(RUNNER_PATH)))
        status = _command_output((
            "git", "status", "--short", "--untracked-files=no", "--",
            str(RUNNER_PATH),
        ))
        if status:
            raise RuntimeError(
                "collective latency runner must be committed before capture")
        return {"kind": "git_commit", "manifest_sha256": None}
    if not SNAPSHOT_MANIFEST.is_file():
        raise FileNotFoundError(SNAPSHOT_MANIFEST)
    manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "source_commit", "files"}:
        raise ValueError("source snapshot manifest fields differ")
    if (manifest["schema_version"] != 1
            or manifest["source_commit"] != source_commit):
        raise ValueError("source snapshot identity differs")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise TypeError("source snapshot files must be an object")
    relative_runner = RUNNER_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    if files[relative_runner] != _sha256(RUNNER_PATH):
        raise RuntimeError("collective runner differs from source snapshot")
    return {
        "kind": "audited_source_snapshot",
        "manifest_sha256": _sha256(SNAPSHOT_MANIFEST),
    }


def _read_version(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def _collective(torch, dist, operation, payload, gathered, world_size):
    if operation == "all_reduce":
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        return
    if operation == "all_gather":
        dist.all_gather_into_tensor(gathered, payload)
        return
    raise ValueError(f"unsupported collective operation: {operation!r}")


def _validate(torch, dist, operation, payload, gathered, rank, world_size):
    """Check the collective actually moved data before anything is timed."""

    probe = payload.clone()
    element_count = probe.numel()
    if operation == "all_reduce":
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        expected = float(world_size * (world_size + 1) // 2)
        if float(probe[0].item()) != expected:
            raise RuntimeError("all_reduce validation failed")
        return
    target = torch.empty_like(gathered)
    dist.all_gather_into_tensor(target, probe)
    torch.npu.synchronize()
    for source_rank in range(world_size):
        if int(target[source_rank * element_count].item()) != source_rank + 1:
            raise RuntimeError("all_gather validation failed")


def _sweep_cell(
    torch, dist, *, operation, element_count, rank, world_size,
    warmup, repeats, trials,
) -> dict[str, object]:
    device = torch.npu.current_device()
    payload = torch.full(
        (element_count,), rank + 1, dtype=torch.bfloat16,
        device=f"npu:{device}")
    gathered = (
        torch.empty((world_size * element_count,), dtype=torch.bfloat16,
                    device=f"npu:{device}")
        if operation == "all_gather" else None)
    _validate(torch, dist, operation, payload, gathered, rank, world_size)

    for _ in range(warmup):
        _collective(torch, dist, operation, payload, gathered, world_size)
    torch.npu.synchronize()
    dist.barrier()

    measurements: dict[str, list[float]] = {pattern: [] for pattern in PATTERNS}
    for pattern in PATTERNS:
        for _ in range(trials):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            dist.barrier()
            if pattern == "back_to_back":
                start.record()
                for _ in range(repeats):
                    _collective(
                        torch, dist, operation, payload, gathered, world_size)
                end.record()
                torch.npu.synchronize()
                measurements[pattern].append(
                    start.elapsed_time(end) * 1000.0 / repeats)
                continue
            start.record()
            _collective(torch, dist, operation, payload, gathered, world_size)
            end.record()
            torch.npu.synchronize()
            measurements[pattern].append(start.elapsed_time(end) * 1000.0)
    del payload, gathered
    torch.npu.empty_cache()

    cell: dict[str, object] = {
        "message_bytes": element_count * BF16_BYTES,
        "element_count": element_count,
        "warmup": warmup,
        "repeats": repeats,
        "trials": trials,
        "graph_unavailable_reason": (
            "Ascend graph capture of an HCCL collective is not a validated "
            "path on this deployment"),
        "patterns": {},
    }
    for pattern, values in measurements.items():
        ordered = sorted(values)
        cell["patterns"][pattern] = {
            "per_collective_us_samples": [round(value, 4) for value in ordered],
            "per_collective_us_min": round(ordered[0], 4),
            "per_collective_us_median": round(statistics.median(ordered), 4),
            "per_collective_us_max": round(ordered[-1], 4),
        }
    return cell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operation", required=True, choices=OPERATIONS)
    parser.add_argument(
        "--message-bytes", required=True, type=_parse_message_bytes)
    parser.add_argument("--warmup", type=_positive_integer, default=50)
    parser.add_argument("--repeats", type=_positive_integer, default=50)
    parser.add_argument("--trials", type=_positive_integer, default=9)
    arguments = parser.parse_args(argv)

    if RUN_ID_PATTERN.fullmatch(arguments.run_id) is None:
        raise ValueError("--run-id contains unsupported characters")
    output_dir = arguments.output_dir.resolve()
    if output_dir != (OUTPUT_ROOT / arguments.run_id).resolve():
        raise ValueError(
            f"--output-dir must equal {OUTPUT_ROOT / arguments.run_id}")
    for message_bytes in arguments.message_bytes:
        if message_bytes % BF16_BYTES:
            raise ValueError(
                f"message_bytes={message_bytes} is not BF16-aligned")
    source_identity = _validate_source_identity(arguments.source_commit)

    import torch
    import torch_npu  # noqa: F401 - registers the npu device backend
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    if len(visible) != world_size:
        raise ValueError(
            "ASCEND_RT_VISIBLE_DEVICES must enumerate every local rank")
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    cells = []
    for message_bytes in arguments.message_bytes:
        cells.append(_sweep_cell(
            torch, dist,
            operation=arguments.operation,
            element_count=message_bytes // BF16_BYTES,
            rank=rank, world_size=world_size,
            warmup=arguments.warmup, repeats=arguments.repeats,
            trials=arguments.trials,
        ))

    dist.barrier()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "kind": "ascend_910b_hccl_collective_protocol_latency",
            "run_id": arguments.run_id,
            "source_commit": arguments.source_commit,
            "source_identity": source_identity,
            "runner_sha256": _sha256(RUNNER_PATH),
            "host": socket.gethostname(),
            "communication_library": "HCCL",
            "cann_version": _read_version(CANN_VERSION_FILE),
            "driver_version": _read_version(DRIVER_VERSION_FILE),
            "torch_version": torch.__version__,
            "torch_npu_version": torch_npu.__version__,
            "operation": arguments.operation,
            "world_size": world_size,
            "visible_devices": visible,
            "dtype": "bfloat16",
            "time_unit": "microseconds",
            "cells": cells,
        }
        destination = output_dir / f"{arguments.operation}.json"
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps({
            "output": str(destination),
            "cells": len(cells),
        }, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
