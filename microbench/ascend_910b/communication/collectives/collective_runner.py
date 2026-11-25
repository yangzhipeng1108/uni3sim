#!/usr/bin/env python3
"""Run HCCL collectives while retaining only discrete selection provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = Path(__file__).resolve()
SNAPSHOT_MANIFEST = REPOSITORY_ROOT / "SOURCE_SNAPSHOT.json"
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output" / "collective" / "ascend_910b"
HCCL_LIBRARY = Path(
    "/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libhccl.so"
)
CANN_VERSION_FILE = Path("/usr/local/Ascend/cann-8.5.1/opp/version.info")
DRIVER_VERSION_FILE = Path("/usr/local/Ascend/driver/version.info")
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


def _visible_devices(world_size: int) -> tuple[int, ...]:
    visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
    if len(visible) != world_size:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must enumerate every local rank")
    try:
        devices = tuple(int(value) for value in visible)
    except ValueError as error:
        raise ValueError(
            "ASCEND_RT_VISIBLE_DEVICES entries must be integers"
        ) from error
    if len(set(devices)) != world_size:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES entries must be unique")
    return devices


def _version(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    versions = tuple(line.removeprefix("Version=") for line in lines if line.startswith("Version="))
    if len(versions) != 1 or not versions[0]:
        raise ValueError(f"cannot determine version from {path}")
    return versions[0]


def _run_collective(
    torch,
    dist,
    *,
    operation: str,
    element_count: int,
    rank: int,
    world_size: int,
    device: str,
) -> None:
    input_tensor = torch.full(
        (element_count,), rank + 1, dtype=torch.bfloat16, device=device
    )
    if operation == "all_reduce":
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) // 2
        if input_tensor[0].item() != expected:
            raise RuntimeError("all_reduce validation failed")
    elif operation == "all_gather":
        output = torch.empty(
            (world_size * element_count,), dtype=torch.bfloat16, device=device
        )
        dist.all_gather_into_tensor(output, input_tensor)
        for source_rank in range(world_size):
            if output[source_rank * element_count].item() != source_rank + 1:
                raise RuntimeError("all_gather validation failed")
    else:
        raise ValueError(f"unsupported operation: {operation!r}")
    torch.npu.synchronize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operation", required=True, choices=OPERATIONS)
    parser.add_argument("--message-bytes", required=True, type=_parse_message_bytes)
    parser.add_argument("--repeats", type=_positive_integer, default=3)
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
    if world_size not in (2, 4, 8) or local_world_size != world_size:
        raise ValueError("910B collective sweeps require 2, 4, or 8 local ranks")
    rank_mapping = _visible_devices(world_size)
    if local_rank not in range(world_size):
        raise ValueError("LOCAL_RANK is out of range")
    physical_device = rank_mapping[local_rank]
    expected_log_path = (output_dir / "hccl_logs").resolve()
    actual_log_path = Path(os.environ["ASCEND_PROCESS_LOG_PATH"]).resolve()
    if actual_log_path != expected_log_path:
        raise ValueError(f"ASCEND_PROCESS_LOG_PATH must equal {expected_log_path}")
    if os.environ["ASCEND_GLOBAL_LOG_LEVEL"] != "1":
        raise ValueError("ASCEND_GLOBAL_LOG_LEVEL must be 1 (INFO)")
    if os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] != "0":
        raise ValueError("ASCEND_SLOG_PRINT_TO_STDOUT must be 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_log_path.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"rank-{rank}.json"
    if target.exists():
        raise FileExistsError(target)

    import torch
    import torch.distributed as dist
    import torch_npu

    if not hasattr(torch, "npu"):
        raise RuntimeError("runner requires torch-npu")
    if not dist.is_hccl_available():
        raise RuntimeError("HCCL backend is unavailable")
    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"
    dist.init_process_group(backend="hccl")
    device_metadata = {
        "device_name": torch.npu.get_device_name(local_rank),
        "hostname": socket.gethostname(),
        "physical_device": physical_device,
    }

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
                device=device,
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

    payload = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "run_id": arguments.run_id,
        "source_commit": arguments.source_commit,
        "source_identity": source_identity,
        "runner_sha256": _sha256(RUNNER_PATH),
        "communication_library": "HCCL",
        "library_version": _version(CANN_VERSION_FILE),
        "library_sha256": _sha256(HCCL_LIBRARY),
        "cann_version_file_sha256": _sha256(CANN_VERSION_FILE),
        "driver_version": _version(DRIVER_VERSION_FILE),
        "driver_version_file_sha256": _sha256(DRIVER_VERSION_FILE),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "process_id": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "device": device_metadata,
        "rank_mapping": rank_mapping,
        "topology": "single_node_full_hccs",
        "observation_method": "HCCL INFO selection log",
        "log_environment": {
            "ASCEND_GLOBAL_LOG_LEVEL": os.environ["ASCEND_GLOBAL_LOG_LEVEL"],
            "ASCEND_PROCESS_LOG_PATH": str(actual_log_path),
            "ASCEND_SLOG_PRINT_TO_STDOUT": os.environ[
                "ASCEND_SLOG_PRINT_TO_STDOUT"
            ],
        },
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
