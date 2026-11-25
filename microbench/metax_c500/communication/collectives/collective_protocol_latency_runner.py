#!/usr/bin/env python3
"""MetaX C500 MCCL collective protocol latency and bandwidth sweep.

Measures the device-timeline duration of one MCCL collective as a function of
message size, world size and launch pattern.  It is the C500 counterpart of the
independent NCCL Ring AllReduce microbenchmark that supplies the A800
ProtoBridge protocol-policy constants, and it exists because the C500 entries in
``src/kernel/launch_config/metax_c500`` are round development placeholders
(5/5/10 us startup, 0.5 us per phase) rather than measurements.

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
    barrier in front of it.

``graph_replay``
    ``--repeats`` collectives captured into one MACA graph and replayed.  This
    is opt-in behind ``--include-graph-replay`` because capturing an MCCL
    collective into a graph deadlocks on MACA 3.7.2 on this deployment rather
    than raising, which no in-process guard can recover from.

Whether the eager patterns contain a per-collective launch gap at all is
decided by the measurement itself: sweeping ``--repeats`` changes how deeply
the launches are queued ahead, so a per-collective time that is flat in the
repeat count is device-bound and a falling one is host-launch-bound.

Example, inside container ``configured profiling container``::

    MACA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \\
        microbench/metax_c500/communication/collectives/\\
collective_protocol_latency_runner.py \\
        --run-id mccl_latency_20260919_r1 --source-commit <sha> \\
        --output-dir profiler_output/collective/metax_c500/\\
mccl_latency_20260919_r1 \\
        --operation all_reduce \\
        --message-bytes 4096,16384,65536,262144,1048576,4194304,16777216
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = Path(__file__).resolve()
SNAPSHOT_MANIFEST = REPOSITORY_ROOT / "SOURCE_SNAPSHOT.json"
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output" / "collective" / "metax_c500"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
OPERATIONS = ("all_reduce", "all_gather")
PATTERNS = ("graph_replay", "back_to_back", "isolated")
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


def _validate_source_identity(source_commit: str) -> dict[str, object]:
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("--source-commit must be a full lowercase Git commit")
    if not SNAPSHOT_MANIFEST.is_file():
        raise FileNotFoundError(
            "hardware execution requires an audited source snapshot manifest: "
            f"{SNAPSHOT_MANIFEST}")
    manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "source_commit", "files"}:
        raise ValueError("source snapshot manifest fields differ")
    if manifest["schema_version"] != 1:
        raise ValueError("source snapshot manifest schema differs")
    if manifest["source_commit"] != source_commit:
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


def _physical_device(local_rank: int, world_size: int) -> int:
    visible = os.environ["MACA_VISIBLE_DEVICES"].split(",")
    if len(visible) != world_size:
        raise ValueError("MACA_VISIBLE_DEVICES must enumerate every local rank")
    return int(visible[local_rank])


def _collective(torch, dist, operation, payload, gathered, world_size):
    if operation == "all_reduce":
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    elif operation == "all_gather":
        dist.all_gather_into_tensor(gathered, payload)
    else:
        raise ValueError(f"unsupported operation: {operation!r}")


def _validate(torch, dist, operation, payload, gathered, rank, world_size):
    """Run one collective and check it actually moved data."""

    payload.fill_(rank + 1)
    _collective(torch, dist, operation, payload, gathered, world_size)
    torch.cuda.synchronize()
    if operation == "all_reduce":
        expected = world_size * (world_size + 1) // 2
        if int(payload[0].item()) != expected:
            raise RuntimeError("all_reduce validation failed")
    else:
        element_count = payload.numel()
        for source_rank in range(world_size):
            if int(gathered[source_rank * element_count].item()) != source_rank + 1:
                raise RuntimeError("all_gather validation failed")


def _sweep_cell(
    torch, dist, *, operation, element_count, rank, world_size,
    warmup, repeats, trials, include_graph_replay,
) -> dict[str, object]:
    payload = torch.full(
        (element_count,), rank + 1, dtype=torch.bfloat16, device="cuda")
    gathered = (
        torch.empty((world_size * element_count,), dtype=torch.bfloat16,
                    device="cuda")
        if operation == "all_gather" else None)
    _validate(torch, dist, operation, payload, gathered, rank, world_size)

    for _ in range(warmup):
        _collective(torch, dist, operation, payload, gathered, world_size)
    torch.cuda.synchronize()
    dist.barrier()

    graph = None
    graph_unavailable = "not requested"
    if not include_graph_replay:
        pass
    else:
      try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                _collective(
                    torch, dist, operation, payload, gathered, world_size)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        candidate = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate):
            for _ in range(repeats):
                _collective(
                    torch, dist, operation, payload, gathered, world_size)
        graph = candidate
      except Exception as error:  # noqa: BLE001 - recorded, never dropped
        graph_unavailable = f"{type(error).__name__}: {error}"
      else:
        graph_unavailable = None
    torch.cuda.synchronize()
    dist.barrier()

    measurements: dict[str, list[float]] = {pattern: [] for pattern in PATTERNS}
    for pattern in PATTERNS:
        if pattern == "graph_replay":
            if graph is None:
                continue
            for _ in range(trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                dist.barrier()
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize()
                measurements[pattern].append(
                    start.elapsed_time(end) * 1000.0 / repeats)
            continue
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if pattern == "back_to_back":
                torch.cuda.synchronize()
                dist.barrier()
                start.record()
                for _ in range(repeats):
                    _collective(
                        torch, dist, operation, payload, gathered, world_size)
                end.record()
                torch.cuda.synchronize()
                measurements[pattern].append(
                    start.elapsed_time(end) * 1000.0 / repeats)
            else:
                torch.cuda.synchronize()
                dist.barrier()
                start.record()
                _collective(
                    torch, dist, operation, payload, gathered, world_size)
                end.record()
                torch.cuda.synchronize()
                measurements[pattern].append(start.elapsed_time(end) * 1000.0)
    del payload, gathered, graph
    torch.cuda.empty_cache()

    cell: dict[str, object] = {
        "message_bytes": element_count * BF16_BYTES,
        "element_count": element_count,
        "warmup": warmup,
        "repeats": repeats,
        "trials": trials,
        "graph_unavailable_reason": graph_unavailable,
        "patterns": {},
    }
    for pattern, values in measurements.items():
        if not values:
            continue
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
    parser.add_argument(
        "--include-graph-replay", action="store_true",
        help="capture the repeats into a MACA graph; deadlocks on MACA 3.7.2")
    arguments = parser.parse_args(argv)

    if RUN_ID_PATTERN.fullmatch(arguments.run_id) is None:
        raise ValueError("--run-id contains unsupported characters")
    output_dir = arguments.output_dir.resolve()
    if output_dir != (OUTPUT_ROOT / arguments.run_id).resolve():
        raise ValueError(
            f"--output-dir must equal {OUTPUT_ROOT / arguments.run_id}")
    for message_bytes in arguments.message_bytes:
        if message_bytes % BF16_BYTES:
            raise ValueError(f"message_bytes={message_bytes} is not BF16-aligned")
    source_identity = _validate_source_identity(arguments.source_commit)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4):
        raise ValueError("C500 collective sweeps require 2 or 4 ranks")
    if int(os.environ["LOCAL_WORLD_SIZE"]) != world_size:
        raise ValueError("C500 collective sweeps must be single node")
    physical_device = _physical_device(local_rank, world_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"rank-{rank}.json"
    if target.exists():
        raise FileExistsError(target)

    import torch
    import torch.distributed as dist

    if "metax" not in torch.__version__.lower():
        raise RuntimeError("runner requires the MetaX torch build")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        cells = [
            _sweep_cell(
                torch, dist,
                operation=arguments.operation,
                element_count=message_bytes // BF16_BYTES,
                rank=rank, world_size=world_size,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
                trials=arguments.trials,
                include_graph_replay=arguments.include_graph_replay,
            )
            for message_bytes in arguments.message_bytes
        ]
        document = {
            "schema_version": 1,
            "kind": "metax_c500_collective_protocol_latency_sweep",
            "hardware_id": "metax_c500",
            "communication_library": "MCCL",
            "torch_version": torch.__version__,
            "timer_domain": "torch_cuda_event_gpu_timeline",
            "measures": "one collective's device duration over synthetic "
                        "message sizes; no production workload participates",
            "operation": arguments.operation,
            "include_graph_replay": arguments.include_graph_replay,
            "dtype": "bfloat16",
            "reduction": "sum" if arguments.operation == "all_reduce" else None,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "run_id": arguments.run_id,
            "source_commit": arguments.source_commit,
            "source_identity": source_identity,
            "device": {
                "physical_device": physical_device,
                "device_name": torch.cuda.get_device_name(local_rank),
                "hostname": socket.gethostname(),
            },
            "cells": cells,
        }
        target.write_text(json.dumps(document, indent=2) + "\n")
        if rank == 0:
            print(f"{arguments.operation} world_size={world_size}")
            print("  bytes        graph_us  back_to_back_us   isolated_us")
            for cell in cells:
                patterns = cell["patterns"]
                graph_us = (
                    patterns["graph_replay"]["per_collective_us_median"]
                    if "graph_replay" in patterns else float("nan"))
                print("  %-10d %10.3f %15.3f %13.3f" % (
                    cell["message_bytes"], graph_us,
                    patterns["back_to_back"]["per_collective_us_median"],
                    patterns["isolated"]["per_collective_us_median"]))
                if cell["graph_unavailable_reason"] is not None:
                    print("      graph unavailable: "
                          f"{cell['graph_unavailable_reason']}")
            print(f"wrote {target}")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
