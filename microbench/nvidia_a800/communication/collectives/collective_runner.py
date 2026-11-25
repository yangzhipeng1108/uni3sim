#!/usr/bin/env python3
"""Run source-bound NCCL collectives without measuring their duration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "collective" / "nvidia_a800"
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
OPERATIONS = ("all_reduce", "all_gather")
DTYPES = {"bfloat16": 2}
REDUCTIONS = {"all_reduce": "sum", "all_gather": None}


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_message_bytes(text: str) -> tuple[int, ...]:
    values = tuple(_positive_integer(value) for value in text.split(","))
    if not values:
        raise argparse.ArgumentTypeError("message byte list must not be empty")
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


def _validate_source_identity(source_commit: str) -> None:
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("--source-commit must be a full lowercase Git commit")
    if _command_output(("git", "rev-parse", "HEAD")) != source_commit:
        raise RuntimeError("--source-commit does not match the deployed HEAD")
    _command_output(("git", "ls-files", "--error-unmatch", str(RUNNER_PATH)))
    status = _command_output((
        "git", "status", "--short", "--untracked-files=no", "--",
        str(RUNNER_PATH),
    ))
    if status:
        raise RuntimeError("collective runner must be committed before capture")


def _physical_device(local_rank: int) -> int:
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) != int(os.environ["LOCAL_WORLD_SIZE"]):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must explicitly enumerate every local rank"
        )
    try:
        return int(visible[local_rank])
    except ValueError as error:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES entries must be physical integer ordinals"
        ) from error


def _device_identity(physical_device: int) -> dict[str, object]:
    output = _command_output((
        "nvidia-smi", "-i", str(physical_device),
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
        "--format=csv,noheader,nounits",
    ))
    rows = tuple(line for line in output.splitlines() if line.strip())
    if len(rows) != 1:
        raise RuntimeError("nvidia-smi did not return exactly one device row")
    fields = tuple(field.strip() for field in rows[0].split(","))
    if len(fields) != 5:
        raise ValueError("nvidia-smi device row does not contain five fields")
    if int(fields[0]) != physical_device:
        raise RuntimeError("nvidia-smi returned a different physical device")
    if fields[1] != "NVIDIA A800-SXM4-80GB":
        raise RuntimeError(f"unexpected device name: {fields[1]!r}")
    return {
        "physical_device": physical_device,
        "device_name": fields[1],
        "device_uuid": fields[2],
        "pci_bus_id": fields[3],
        "driver_version": fields[4],
    }


def _dtype(torch, name: str):
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name!r}")


def _run_collective(
    torch,
    dist,
    *,
    operation: str,
    element_count: int,
    dtype,
    rank: int,
    world_size: int,
) -> None:
    input_tensor = torch.full(
        (element_count,), rank + 1, dtype=dtype, device="cuda"
    )
    if operation == "all_reduce":
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) // 2
        if input_tensor[0].item() != expected:
            raise RuntimeError("all_reduce validation failed")
    elif operation == "all_gather":
        output = torch.empty(
            (world_size * element_count,), dtype=dtype, device="cuda"
        )
        dist.all_gather_into_tensor(output, input_tensor)
        for source_rank in range(world_size):
            start = source_rank * element_count
            if output[start].item() != source_rank + 1:
                raise RuntimeError("all_gather validation failed")
    else:
        raise ValueError(f"unsupported operation: {operation!r}")
    torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--profiler-source-commit", required=True)
    parser.add_argument("--profiler-plugin-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operation", required=True, choices=OPERATIONS)
    parser.add_argument(
        "--message-bytes", required=True, type=_parse_message_bytes
    )
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--repeats", type=_positive_integer, default=3)
    arguments = parser.parse_args(argv)

    if RUN_ID_PATTERN.fullmatch(arguments.run_id) is None:
        raise ValueError("--run-id contains unsupported characters")
    if COMMIT_PATTERN.fullmatch(arguments.profiler_source_commit) is None:
        raise ValueError(
            "--profiler-source-commit must be a full lowercase Git commit"
        )
    output_dir = arguments.output_dir.resolve()
    expected_output_dir = (OUTPUT_ROOT / arguments.run_id).resolve()
    if output_dir != expected_output_dir:
        raise ValueError(
            f"--output-dir must equal the canonical run path {expected_output_dir}"
        )
    element_bytes = DTYPES[arguments.dtype]
    for message_bytes in arguments.message_bytes:
        if message_bytes % element_bytes:
            raise ValueError(
                f"message_bytes={message_bytes} is not divisible by {element_bytes}"
            )
    _validate_source_identity(arguments.source_commit)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size not in (2, 4, 8) or local_world_size != world_size:
        raise ValueError("A800 collective sweeps require 2, 4, or 8 local ranks")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"rank-{rank}.json"
    if target.exists():
        raise FileExistsError(target)
    profiler_plugin = arguments.profiler_plugin_path.resolve()
    if not profiler_plugin.is_file():
        raise FileNotFoundError(profiler_plugin)
    if profiler_plugin.name != "libnccl-profiler.so":
        raise ValueError("profiler plugin must be named libnccl-profiler.so")
    library_paths = tuple(
        Path(value).resolve()
        for value in os.environ["LD_LIBRARY_PATH"].split(":")
        if value
    )
    if profiler_plugin.parent not in library_paths:
        raise ValueError(
            "LD_LIBRARY_PATH must contain the profiler plugin directory"
        )
    os.environ["NCCL_PROFILER_PLUGIN"] = profiler_plugin.name
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"
    os.environ["NCCL_DEBUG_FILE"] = str(output_dir / "nccl.%h.%p.log")
    os.environ["NCCL_PROFILE_EVENT_MASK"] = "2"
    os.environ["NCCL_PROFILE_GROUP_POOL_SIZE"] = "512"
    os.environ["NCCL_PROFILE_COLL_POOL_SIZE"] = "512"
    os.environ["NCCL_PROFILE_DUMP_FILE"] = str(
        output_dir / "nccl-profile"
    )

    import torch
    import torch.distributed as dist

    if not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch NCCL backend is unavailable")
    physical_device = _physical_device(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = _device_identity(physical_device)
    identities: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(identities, device)
    rank_mapping = tuple(
        str(identity["device_uuid"])
        for identity in identities
        if isinstance(identity, dict)
    )
    if len(rank_mapping) != world_size or len(set(rank_mapping)) != world_size:
        raise RuntimeError("rank mapping does not contain one unique device per rank")

    dist.barrier()
    observations = []
    torch_dtype = _dtype(torch, arguments.dtype)
    for message_bytes in arguments.message_bytes:
        element_count = message_bytes // element_bytes
        for repeat in range(arguments.repeats):
            dist.barrier()
            marker = {
                "operation": arguments.operation,
                "message_bytes": message_bytes,
                "repeat": repeat,
            }
            print(
                "MIDLAYER_COLLECTIVE_BEGIN "
                + json.dumps(marker, sort_keys=True, separators=(",", ":")),
                file=sys.stderr,
                flush=True,
            )
            _run_collective(
                torch,
                dist,
                operation=arguments.operation,
                element_count=element_count,
                dtype=torch_dtype,
                rank=rank,
                world_size=world_size,
            )
            print(
                "MIDLAYER_COLLECTIVE_END "
                + json.dumps(marker, sort_keys=True, separators=(",", ":")),
                file=sys.stderr,
                flush=True,
            )
            observations.append({
                "operation": arguments.operation,
                "message_bytes": message_bytes,
                "participant_count": world_size,
                "dtype": arguments.dtype,
                "reduction": REDUCTIONS[arguments.operation],
                "in_place": arguments.operation == "all_reduce",
                "repeat": repeat,
                "rank": rank,
                "validation": "passed",
            })

    nccl_version = torch.cuda.nccl.version()
    payload = {
        "schema_version": 1,
        "hardware_id": "nvidia_a800",
        "run_id": arguments.run_id,
        "source_commit": arguments.source_commit,
        "runner_sha256": _sha256(RUNNER_PATH),
        "profiler_plugin_source_commit": arguments.profiler_source_commit,
        "profiler_plugin_sha256": _sha256(profiler_plugin),
        "communication_library": "NCCL",
        "library_version": ".".join(str(value) for value in nccl_version),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
        "rank_mapping": rank_mapping,
        "topology": "single_node_nvlink",
        "observation_method": "NCCL profiler plugin API v4",
        "observations": observations,
    }
    temporary = output_dir / f"rank-{rank}.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
