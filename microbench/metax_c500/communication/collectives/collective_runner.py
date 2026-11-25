#!/usr/bin/env python3
"""Run marker-bounded MCCL collectives without consuming their duration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = Path(__file__).resolve()
SNAPSHOT_MANIFEST = REPOSITORY_ROOT / "SOURCE_SNAPSHOT.json"
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "collective" / "metax_c500"
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
OPERATIONS = ("all_reduce", "all_gather")
DTYPES = {"bfloat16": 2}
REDUCTIONS = {"all_reduce": "sum", "all_gather": None}


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return value


def _parse_message_bytes(text: str) -> tuple[int, ...]:
    values = tuple(_positive_integer(value) for value in text.split(","))
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("message byte values must be unique")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: tuple[str, ...]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stderr.strip():
        raise RuntimeError(
            f"command wrote stderr: {command!r}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _validate_source_identity(source_commit: str) -> dict[str, object]:
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("--source-commit must be a full lowercase Git commit")
    git_dir = REPOSITORY_ROOT / ".git"
    if git_dir.exists():
        if _command_output(("git", "rev-parse", "HEAD")) != source_commit:
            raise RuntimeError("--source-commit does not match the deployed HEAD")
        _command_output(("git", "ls-files", "--error-unmatch", str(RUNNER_PATH)))
        status = _command_output((
            "git", "status", "--short", "--untracked-files=no", "--",
            str(RUNNER_PATH),
        ))
        if status:
            raise RuntimeError("collective runner must be committed before capture")
        return {"kind": "git_commit", "manifest_sha256": None}
    if not SNAPSHOT_MANIFEST.is_file():
        raise FileNotFoundError(SNAPSHOT_MANIFEST)
    manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "source_commit", "files"}:
        raise ValueError("source snapshot manifest fields differ")
    if manifest["schema_version"] != 1 or manifest["source_commit"] != source_commit:
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
    try:
        return int(visible[local_rank])
    except ValueError as error:
        raise ValueError("MACA_VISIBLE_DEVICES entries must be integers") from error


def _run_collective(
    torch,
    dist,
    *,
    operation: str,
    element_count: int,
    rank: int,
    world_size: int,
) -> None:
    input_tensor = torch.full(
        (element_count,), rank + 1, dtype=torch.bfloat16, device="cuda"
    )
    if operation == "all_reduce":
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) // 2
        if input_tensor[0].item() != expected:
            raise RuntimeError("all_reduce validation failed")
    elif operation == "all_gather":
        output = torch.empty(
            (world_size * element_count,), dtype=torch.bfloat16, device="cuda"
        )
        dist.all_gather_into_tensor(output, input_tensor)
        for source_rank in range(world_size):
            if output[source_rank * element_count].item() != source_rank + 1:
                raise RuntimeError("all_gather validation failed")
    else:
        raise ValueError(f"unsupported operation: {operation!r}")
    torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operation", required=True, choices=OPERATIONS)
    parser.add_argument("--message-bytes", required=True, type=_parse_message_bytes)
    parser.add_argument("--repeats", type=_positive_integer, default=3)
    parser.add_argument(
        "--profiler-attach-wait-seconds", type=_nonnegative_float, default=30.0
    )
    arguments = parser.parse_args(argv)

    if RUN_ID_PATTERN.fullmatch(arguments.run_id) is None:
        raise ValueError("--run-id contains unsupported characters")
    output_dir = arguments.output_dir.resolve()
    expected_output_dir = (OUTPUT_ROOT / arguments.run_id).resolve()
    if output_dir != expected_output_dir:
        raise ValueError(f"--output-dir must equal {expected_output_dir}")
    for message_bytes in arguments.message_bytes:
        if message_bytes % DTYPES["bfloat16"]:
            raise ValueError(f"message_bytes={message_bytes} is not BF16-aligned")
    source_identity = _validate_source_identity(arguments.source_commit)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size not in (2, 4) or local_world_size != world_size:
        raise ValueError("C500 collective sweeps require 2 or 4 local ranks")
    physical_device = _physical_device(local_rank, world_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"rank-{rank}.json"
    if target.exists():
        raise FileExistsError(target)

    import torch
    import torch.distributed as dist

    if "metax" not in torch.__version__.lower():
        raise RuntimeError("runner requires the MetaX torch build")
    if not dist.is_nccl_available():
        raise RuntimeError("MetaX NCCL-compatible backend is unavailable")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device_name = torch.cuda.get_device_name(local_rank)
    device = {
        "physical_device": physical_device,
        "device_name": device_name,
        "hostname": socket.gethostname(),
    }
    identities: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(identities, device)
    rank_mapping = tuple(
        int(identity["physical_device"])
        for identity in identities
        if isinstance(identity, dict)
    )
    if len(rank_mapping) != world_size or len(set(rank_mapping)) != world_size:
        raise RuntimeError("rank mapping does not contain one unique device per rank")

    torch.cuda.profiler.start()
    time.sleep(arguments.profiler_attach_wait_seconds)
    dist.barrier()
    observations = []
    for message_bytes in arguments.message_bytes:
        element_count = message_bytes // DTYPES["bfloat16"]
        for repeat in range(arguments.repeats):
            _run_collective(
                torch,
                dist,
                operation=arguments.operation,
                element_count=element_count,
                rank=rank,
                world_size=world_size,
            )
            observations.append({
                "operation": arguments.operation,
                "message_bytes": message_bytes,
                "participant_count": world_size,
                "dtype": "bfloat16",
                "reduction": REDUCTIONS[arguments.operation],
                "in_place": arguments.operation == "all_reduce",
                "repeat": repeat,
                "rank": rank,
                "validation": "passed",
            })
    torch.cuda.profiler.stop()

    payload = {
        "schema_version": 2,
        "hardware_id": "metax_c500",
        "run_id": arguments.run_id,
        "source_commit": arguments.source_commit,
        "source_identity": source_identity,
        "runner_sha256": _sha256(RUNNER_PATH),
        "communication_library": "MCCL",
        "library_version": "2.16.5",
        "library_sha256": _sha256(Path("/opt/maca-3.7.2/lib/libmccl.so")),
        "torch_version": torch.__version__,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
        "rank_mapping": rank_mapping,
        "topology": "single_node_metaxlink",
        "observation_method": "mcTracer kernel symbol",
        "profiler_attach_wait_seconds": arguments.profiler_attach_wait_seconds,
        "observations": observations,
    }
    temporary = output_dir / f"rank-{rank}.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
