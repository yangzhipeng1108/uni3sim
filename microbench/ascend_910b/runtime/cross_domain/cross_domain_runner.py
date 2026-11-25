#!/usr/bin/env python3
"""Capture the frozen Ascend 910B 1:2 cross-domain handoff campaign."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


TRAINING_BLOCK_DIMS = (1, 12, 24)
VALIDATION_BLOCK_DIMS = (8, 11, 16, 18, 19, 21, 22, 23)
BLOCK_DIMS = tuple(sorted(TRAINING_BLOCK_DIMS + VALIDATION_BLOCK_DIMS))
TRAINING_TASK_COUNTS = (1, 2, 64)
TASK_COUNTS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
UNIFORM_CELLS = tuple(
    (f"uniform_b{block_dim}_n{task_count}", (task_count,) * block_dim)
    for block_dim in BLOCK_DIMS
    for task_count in TASK_COUNTS
)


def _heterogeneous_cells():
    """Return evaluation-independent active-group survival profiles."""

    cells = []
    for block_dim in (8, 16, 24):
        staircase = tuple(
            (128, 96, 64, 32, 16, 8, 4, 2)[index % 8]
            for index in range(block_dim))
        cells.extend((
            (f"heterogeneous_b{block_dim}_single_straggler",
             (128,) + (1,) * (block_dim - 1)),
            (f"heterogeneous_b{block_dim}_half_stragglers",
             (128,) * (block_dim // 2) + (1,) * (block_dim // 2)),
            (f"heterogeneous_b{block_dim}_staircase", staircase),
        ))
    return tuple(cells)


HETEROGENEOUS_CELLS = _heterogeneous_cells()
CELLS = UNIFORM_CELLS + HETEROGENEOUS_CELLS
TRIALS = 6
SYSTEM_COUNTER_HZ = 50_000_000
SYSTEM_COUNTER_TICK_NS = 20.0
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "cross_domain_contract.h",
    "cross_domain_rt_kernel.cpp",
    "cross_domain_rt_main.cpp",
    "cross_domain_runner.py",
    "cross_domain_report.py",
)
EXPECTED_CANN = {
    "package": "Ascend-cann-toolkit",
    "version": "8.5.1",
    "innerversion": "V100R001C25SPC002B220",
}
EXPECTED_CANN_SOURCE_SHA256 = {
    "fia_vector": (
        "259f5700812eafe32bcffd075f59d90c82cb71458f62bd47cb3d6f17f171e762"
    ),
    "fia_cube": (
        "acce194dcf01f370d864ba7987f0c5a9d5927f7a69fb47426afa7789a6f856ef"
    ),
    "fia_kernel": (
        "7cde05730d68bb02a2c45cd6b25605417ee61d42dfd87f21ccf0948c014b7e21"
    ),
}
SOURCE_MARKERS = {
    "fia_vector": (
        "CrossCoreWaitFlag(constInfo.syncC1V1);",
        "CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);",
        "CrossCoreWaitFlag(constInfo.syncC2V2);",
    ),
    "fia_cube": (
        "CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);",
        "CrossCoreWaitFlag(constInfo.syncV1C2);",
        "CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);",
    ),
    "fia_kernel": (
        "tmpBlockIdx = GetBlockIdx(); // vec:0-47",
        "aiCoreIdx = tmpBlockIdx / 2;",
        "ComputeMm1(extraInfo0);",
        "ComputeVec1(extraInfo2);",
        "ComputeMm2(extraInfo2);",
        "ComputeVec2(extraInfo1);",
    ),
}
RAW_FIELDS = (
    "run_id",
    "trial",
    "order_direction",
    "order_position",
    "process_pid",
    "profile_id",
    "block_dim",
    "task_counts",
    "aic_cycles",
    "aiv0_cycles",
    "aiv1_cycles",
    "host_wall_ns",
    "validation",
)
CHILD_HEADER = (
    "block_dim,task_counts,aic_cycles,aiv0_cycles,aiv1_cycles,"
    "host_wall_ns,validation"
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(benchmark_root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        path = benchmark_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"snapshot source is missing: {path}")
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _ordered_cells(trial_index):
    if trial_index < 0 or trial_index >= TRIALS:
        raise ValueError("trial index is outside the frozen campaign")
    if trial_index % 2 == 0:
        return "forward", CELLS
    return "reverse", tuple(reversed(CELLS))


def _run_checked(command, *, environment=None, timeout=30):
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command returned {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def _parse_idle_usage(text):
    if not isinstance(text, str) or not text.strip():
        raise TypeError("npu-smi usage telemetry must be non-empty")
    labels = {
        "aic_percent": "Aicore Usage Rate(%)",
        "aiv_percent": "Aivector Usage Rate(%)",
        "hbm_bandwidth_percent": "HBM Bandwidth Usage Rate(%)",
    }
    result = {}
    for field, label in labels.items():
        matches = re.findall(
            rf"^\s*{re.escape(label)}\s*:\s*([0-9]+)\s*$",
            text,
            flags=re.MULTILINE,
        )
        if len(matches) != 1:
            raise ValueError(f"usage telemetry has ambiguous {field}")
        value = int(matches[0])
        if value != 0:
            raise RuntimeError(f"selected NPU is busy: {field}={value}")
        result[field] = value
    return result


def _parse_global_identity(text, device_id):
    if not isinstance(text, str) or not text.strip():
        raise TypeError("npu-smi global telemetry must be non-empty")
    version = re.findall(
        r"npu-smi\s+([0-9]+(?:\.[0-9]+)+)", text, flags=re.IGNORECASE)
    header_version = re.findall(
        r"Version\s*:\s*([0-9]+(?:\.[0-9]+)+)", text,
        flags=re.IGNORECASE)
    if len(version) != 1 or header_version != version:
        raise ValueError("npu-smi version identity is ambiguous")
    model_pattern = rf"^\|\s*{device_id}\s+910B2\s+\|"
    if len(re.findall(model_pattern, text, flags=re.MULTILINE)) != 1:
        raise ValueError("npu-smi does not identify the selected 910B2")
    return {"physical_device_id": device_id, "model": "910B2",
            "npu_smi_version": version[0]}


def _parse_cann_version(text):
    source_fields = {
        "package": "package_name",
        "version": "version",
        "innerversion": "innerversion",
    }
    result = {}
    for field, expected in EXPECTED_CANN.items():
        matches = re.findall(
            rf"^\s*{source_fields[field]}\s*=\s*([^\s]+)\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if matches != [expected]:
            raise ValueError(
                f"CANN {field} must be exactly {expected}, got {matches!r}")
        result[field] = expected
    return result


def _validate_cann_sources(paths):
    identities = {}
    for label, expected_hash in EXPECTED_CANN_SOURCE_SHA256.items():
        path = paths[label]
        if not path.is_file():
            raise FileNotFoundError(f"required CANN source is missing: {path}")
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"{label} SHA256 mismatch: {observed_hash}")
        source = path.read_text(encoding="utf-8")
        missing = tuple(
            marker for marker in SOURCE_MARKERS[label]
            if marker not in source
        )
        if missing:
            raise ValueError(f"{label} lost source markers: {missing}")
        identities[label] = {
            "path": str(path.resolve()),
            "sha256": observed_hash,
            "required_markers": list(SOURCE_MARKERS[label]),
        }
    return identities


def _cycles(text, expected_length, label):
    values = tuple(int(value) for value in text.split(";"))
    if len(values) != expected_length or any(value <= 0 for value in values):
        raise ValueError(
            f"{label} must contain {expected_length} positive cycles")
    return values


def _build_command(binary, cells):
    command = [str(binary), "--batch"]
    for _, task_counts in cells:
        command.extend((
            str(len(task_counts)),
            ";".join(str(value) for value in task_counts),
        ))
    return command


def _write_json_exclusive(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _run_campaign(args, benchmark_root):
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID contains unsupported characters")
    if COMMIT_PATTERN.fullmatch(args.source_commit) is None:
        raise ValueError("source commit must be 40 lowercase hex digits")
    if SHA256_PATTERN.fullmatch(args.expected_snapshot_sha256) is None:
        raise ValueError("expected snapshot must be a lowercase SHA256")
    if args.device_id < 0:
        raise ValueError("device ID must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")
    observed_snapshot = _snapshot_sha256(benchmark_root)
    if observed_snapshot != args.expected_snapshot_sha256:
        raise ValueError(
            "deployed source snapshot does not match the audited snapshot")
    if not args.cann_version_file.is_file():
        raise FileNotFoundError(
            f"CANN version file does not exist: {args.cann_version_file}")
    cann_identity = _parse_cann_version(
        args.cann_version_file.read_text(encoding="utf-8"))
    cann_sources = _validate_cann_sources({
        "fia_vector": args.fia_vector_source,
        "fia_cube": args.fia_cube_source,
        "fia_kernel": args.fia_kernel_source,
    })

    args.output_dir.mkdir(parents=True)
    global_command = [str(args.npu_smi), "info"]
    usage_command = [
        str(args.npu_smi), "info", "-t", "usages", "-i",
        str(args.device_id),
    ]
    global_text = _run_checked(global_command)
    pre_usage_text = _run_checked(usage_command)
    identity = _parse_global_identity(global_text, args.device_id)
    pre_usage = _parse_idle_usage(pre_usage_text)
    (args.output_dir / "npu_smi_global.txt").write_text(
        global_text, encoding="utf-8")
    (args.output_dir / "npu_smi_pre.txt").write_text(
        pre_usage_text, encoding="utf-8")

    environment = os.environ.copy()
    environment["ASCEND_DEVICE_ID"] = "0"
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    raw_path = args.output_dir / "raw.csv"
    trial_artifacts = []
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for trial_index in range(TRIALS):
            direction, cells = _ordered_cells(trial_index)
            command = _build_command(args.binary, cells)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=args.child_timeout_seconds)
            except subprocess.TimeoutExpired as error:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise RuntimeError(
                    f"child PID {process.pid} timed out") from error
            if process.returncode != 0:
                raise RuntimeError(
                    f"child PID {process.pid} failed with "
                    f"{process.returncode}:\n{stdout}\n{stderr}")
            stdout_path = args.output_dir / f"trial_{trial_index + 1}.stdout"
            stderr_path = args.output_dir / f"trial_{trial_index + 1}.stderr"
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            lines = stdout.strip().splitlines()
            if not lines or lines[0] != CHILD_HEADER:
                raise RuntimeError("child output header changed")
            if len(lines) != len(cells) + 1:
                raise RuntimeError("child output cell count changed")
            for order_position, (cell, line) in enumerate(
                    zip(cells, lines[1:])):
                fields = line.split(",")
                if len(fields) != 7:
                    raise RuntimeError("child output row schema changed")
                profile_id, task_counts = cell
                block_dim = len(task_counts)
                expected_task_counts = ";".join(
                    str(value) for value in task_counts)
                if (int(fields[0]) != block_dim
                        or fields[1] != expected_task_counts
                        or fields[6] != "ok"):
                    raise RuntimeError("child changed the requested cell")
                _cycles(fields[2], block_dim, "aic_cycles")
                _cycles(fields[3], block_dim, "aiv0_cycles")
                _cycles(fields[4], block_dim, "aiv1_cycles")
                if int(fields[5]) <= 0:
                    raise RuntimeError("child host wall interval is invalid")
                writer.writerow({
                    "run_id": args.run_id,
                    "trial": trial_index + 1,
                    "order_direction": direction,
                    "order_position": order_position,
                    "process_pid": process.pid,
                    "profile_id": profile_id,
                    "block_dim": fields[0],
                    "task_counts": fields[1],
                    "aic_cycles": fields[2],
                    "aiv0_cycles": fields[3],
                    "aiv1_cycles": fields[4],
                    "host_wall_ns": fields[5],
                    "validation": fields[6],
                })
            trial_artifacts.append({
                "trial": trial_index + 1,
                "direction": direction,
                "pid": process.pid,
                "command": command,
                "stdout": stdout_path.name,
                "stdout_sha256": _sha256(stdout_path),
                "stderr": stderr_path.name,
                "stderr_sha256": _sha256(stderr_path),
            })
        stream.flush()
        os.fsync(stream.fileno())

    post_usage_text = _run_checked(usage_command)
    post_usage = _parse_idle_usage(post_usage_text)
    (args.output_dir / "npu_smi_post.txt").write_text(
        post_usage_text, encoding="utf-8")
    provenance = {
        "schema": "ascend_910b_cross_domain_provenance_v5",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_identity_mode": "audited_source_snapshot",
        "source_snapshot_sha256": observed_snapshot,
        "source_files": {
            filename: _sha256(benchmark_root / filename)
            for filename in SOURCE_FILENAMES
        },
        "binary": {
            "path": str(args.binary.resolve()),
            "sha256": _sha256(args.binary),
            "build_command": args.build_command,
        },
        "cann": {
            "identity": cann_identity,
            "version_file": str(args.cann_version_file.resolve()),
            "version_file_sha256": _sha256(args.cann_version_file),
            "source_identities": cann_sources,
        },
        "device": {
            **identity,
            "logical_device_id": 0,
            "selection_environment": {
                "ASCEND_DEVICE_ID": "0",
                "ASCEND_RT_VISIBLE_DEVICES": str(args.device_id),
            },
        },
        "idle_telemetry": {"pre": pre_usage, "post": post_usage},
        "clock": {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
        },
        "measurement_scope": (
            "empty CANN 8.5.1 FIA-shaped three-handoff ring; no production "
            "arithmetic, tensor traffic, or operator duration; local versus "
            "shared AIC-group scheduler service isolated for per-group "
            "task_count 1..128, including heterogeneous active-group tails; "
            "one fixed untimed 24-group/128-task trial preamble plus one "
            "identical untimed grid/task warmup precede measured cells"
        ),
        "production_or_evaluation_duration_consumed": False,
        "campaign": {
            "block_dims": list(BLOCK_DIMS),
            "training_block_dims": list(TRAINING_BLOCK_DIMS),
            "validation_block_dims": list(VALIDATION_BLOCK_DIMS),
            "task_counts": list(TASK_COUNTS),
            "training_task_counts": list(TRAINING_TASK_COUNTS),
            "heterogeneous_profiles": {
                profile_id: list(task_counts)
                for profile_id, task_counts in HETEROGENEOUS_CELLS
            },
            "trials": TRIALS,
            "order": "balanced_forward_reverse",
        },
        "capture": {
            "raw": raw_path.name,
            "raw_sha256": _sha256(raw_path),
            "raw_rows": TRIALS * len(CELLS),
            "runner_pid": os.getpid(),
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
            "trials": trial_artifacts,
        },
    }
    _write_json_exclusive(args.output_dir / "provenance.json", provenance)
    print(args.output_dir / "provenance.json")


def main():
    benchmark_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("snapshot")
    run = subparsers.add_parser("run")
    run.add_argument("--binary", required=True, type=Path)
    run.add_argument("--device-id", required=True, type=int)
    run.add_argument("--run-id", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--expected-snapshot-sha256", required=True)
    run.add_argument("--cann-version-file", required=True, type=Path)
    run.add_argument("--fia-vector-source", required=True, type=Path)
    run.add_argument("--fia-cube-source", required=True, type=Path)
    run.add_argument("--fia-kernel-source", required=True, type=Path)
    run.add_argument("--build-command", required=True)
    run.add_argument("--npu-smi", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--child-timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(_snapshot_sha256(benchmark_root))
    elif args.command == "run":
        if args.child_timeout_seconds <= 0:
            raise ValueError("child timeout must be positive")
        _run_campaign(args, benchmark_root)
    else:
        raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
