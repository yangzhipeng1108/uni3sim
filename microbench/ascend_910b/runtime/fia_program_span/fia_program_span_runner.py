#!/usr/bin/env python3
"""Capture one balanced FIA nonempty-controller program-span run."""

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


TRAINING_GROUP_COUNTS = (1, 8, 16, 24)
TASK_COUNTS = (1, 2, 3, 10, 11, 12)
UNIFORM_CELLS = tuple(
    (f"uniform_b{group_count}_n{task_count}",
     (task_count,) * group_count)
    for group_count in TRAINING_GROUP_COUNTS
    for task_count in TASK_COUNTS
)
HOLDOUT_CELLS = (
    ("holdout_b22_n1", (1,) * 22),
    ("holdout_b22_tail3_2", (3,) * 20 + (2,) * 2),
    ("holdout_b22_tail11_10", (11,) * 11 + (10,) * 11),
    ("holdout_b24_stair12_1", tuple(
        value for value in range(12, 0, -1) for _ in range(2))),
)
CELLS = UNIFORM_CELLS + HOLDOUT_CELLS
TRIALS = 6
PROGRAM_REPETITIONS = 8
SYSTEM_COUNTER_HZ = 50_000_000
SYSTEM_COUNTER_TICK_NS = 20.0
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "fia_program_span_contract.h",
    "fia_program_span_rt_kernel.cpp",
    "fia_program_span_rt_main.cpp",
    "fia_program_span_runner.py",
    "fia_program_span_report.py",
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
        "RunInfo extraInfo[FIA_PRELOAD_TASK_CACHE_SIZE];",
        "while (shouldDispatchTask || shouldExecuteTask)",
        "ComputeMm1(extraInfo0);",
        "ComputeVec1(extraInfo2);",
        "ComputeMm2(extraInfo2);",
        "ComputeVec2(extraInfo1);",
    ),
}
RAW_FIELDS = (
    "run_id", "trial", "order_direction", "order_position", "process_pid",
    "profile_id", "block_dim", "task_counts", "aic_cycles", "aiv0_cycles",
    "aiv1_cycles", "validation",
)
CHILD_HEADER = (
    "block_dim,task_counts,aic_cycles,aiv0_cycles,aiv1_cycles,validation"
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        path = root / filename
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
        command, check=False, capture_output=True, text=True,
        env=environment, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command returned {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return completed.stdout


def _parse_idle_usage(text):
    labels = {
        "aic_percent": "Aicore Usage Rate(%)",
        "aiv_percent": "Aivector Usage Rate(%)",
        "hbm_bandwidth_percent": "HBM Bandwidth Usage Rate(%)",
    }
    result = {}
    for field, label in labels.items():
        matches = re.findall(
            rf"^\s*{re.escape(label)}\s*:\s*([0-9]+)\s*$",
            text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"usage telemetry has ambiguous {field}")
        value = int(matches[0])
        if value != 0:
            raise RuntimeError(f"selected NPU is busy: {field}={value}")
        result[field] = value
    return result


def _parse_global_identity(text, device_id):
    version = re.findall(
        r"npu-smi\s+([0-9]+(?:\.[0-9]+)+)", text, flags=re.IGNORECASE)
    header_version = re.findall(
        r"Version\s*:\s*([0-9]+(?:\.[0-9]+)+)", text,
        flags=re.IGNORECASE)
    if len(version) != 1 or header_version != version:
        raise ValueError("npu-smi version identity is ambiguous")
    if len(re.findall(
            rf"^\|\s*{device_id}\s+910B2\s+\|", text,
            flags=re.MULTILINE)) != 1:
        raise ValueError("npu-smi does not identify the selected 910B2")
    return {
        "physical_device_id": device_id,
        "model": "910B2",
        "npu_smi_version": version[0],
    }


def _parse_cann_version(text):
    source_fields = {
        "package": "package_name",
        "version": "version",
        "innerversion": "innerversion",
    }
    result = {}
    for field, expected in EXPECTED_CANN.items():
        matches = re.findall(
            rf"^\s*{source_fields[field]}\s*=\s*([^\s]+)\s*$", text,
            flags=re.IGNORECASE | re.MULTILINE)
        if matches != [expected]:
            raise ValueError(f"CANN {field} identity changed")
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
            raise ValueError(f"{label} SHA256 mismatch: {observed_hash}")
        source = path.read_text(encoding="utf-8")
        missing = tuple(
            marker for marker in SOURCE_MARKERS[label]
            if marker not in source)
        if missing:
            raise ValueError(f"{label} lost source markers: {missing}")
        identities[label] = {
            "path": str(path.resolve()),
            "sha256": observed_hash,
            "required_markers": list(SOURCE_MARKERS[label]),
        }
    return identities


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
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    observed_snapshot = _snapshot_sha256(benchmark_root)
    if observed_snapshot != args.expected_snapshot_sha256:
        raise ValueError("deployed source snapshot does not match audit")
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary does not exist: {args.binary}")
    cann_identity = _parse_cann_version(
        args.cann_version_file.read_text(encoding="utf-8"))
    cann_sources = _validate_cann_sources({
        "fia_vector": args.fia_vector_source,
        "fia_cube": args.fia_cube_source,
        "fia_kernel": args.fia_kernel_source,
    })
    global_text = _run_checked([str(args.npu_smi), "info"])
    usage_command = [
        str(args.npu_smi), "info", "-t", "usages", "-i",
        str(args.device_id),
    ]
    pre_usage_text = _run_checked(usage_command)
    device_identity = _parse_global_identity(global_text, args.device_id)
    pre_usage = _parse_idle_usage(pre_usage_text)
    args.output_dir.mkdir(parents=True)
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
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=environment, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
                raise
            if process.returncode != 0:
                raise RuntimeError(
                    f"trial {trial_index + 1} returned {process.returncode}: "
                    f"{stderr}")
            stdout_path = args.output_dir / f"trial_{trial_index + 1}.stdout"
            stderr_path = args.output_dir / f"trial_{trial_index + 1}.stderr"
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            rows = list(csv.DictReader(stdout.splitlines()))
            if tuple(rows[0].keys()) != tuple(CHILD_HEADER.split(",")):
                raise ValueError("child output header changed")
            if len(rows) != len(cells):
                raise ValueError("child output lost campaign cells")
            for position, (row, expected) in enumerate(zip(rows, cells)):
                profile_id, task_counts = expected
                if (int(row["block_dim"]) != len(task_counts)
                        or tuple(int(value) for value in row["task_counts"].split(";"))
                        != task_counts or row["validation"] != "ok"):
                    raise ValueError("child output changed the frozen cell")
                writer.writerow({
                    "run_id": args.run_id,
                    "trial": trial_index + 1,
                    "order_direction": direction,
                    "order_position": position,
                    "process_pid": process.pid,
                    "profile_id": profile_id,
                    **row,
                })
            stream.flush()
            os.fsync(stream.fileno())
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
    post_usage_text = _run_checked(usage_command)
    post_usage = _parse_idle_usage(post_usage_text)
    (args.output_dir / "npu_smi_post.txt").write_text(
        post_usage_text, encoding="utf-8")
    provenance = {
        "schema": "ascend_910b_fia_program_span_provenance_v1",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_snapshot_sha256": observed_snapshot,
        "source_files": {
            name: _sha256(benchmark_root / name) for name in SOURCE_FILENAMES
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
            **device_identity,
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
            "source-bound nonempty FIA controller from task-cache "
            "initialization through final Vec2 body completion; one fixed "
            "untimed 24-group/11-task scheduler preamble and two exact-cell "
            "warmups precede every measured cell"),
        "production_or_evaluation_duration_consumed": False,
        "campaign": {
            "training_group_counts": list(TRAINING_GROUP_COUNTS),
            "task_counts": list(TASK_COUNTS),
            "holdout_profiles": {
                name: list(counts) for name, counts in HOLDOUT_CELLS
            },
            "trials": TRIALS,
            "program_repetitions": PROGRAM_REPETITIONS,
            "order": "balanced_forward_reverse",
        },
        "capture": {
            "raw": raw_path.name,
            "raw_sha256": _sha256(raw_path),
            "raw_rows": TRIALS * len(CELLS),
            "runner_pid": os.getpid(),
            "argv": sys.argv,
            "working_directory": str(Path.cwd()),
            "trials": trial_artifacts,
        },
    }
    _write_json_exclusive(args.output_dir / "provenance.json", provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-snapshot-sha256", required=True)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--npu-smi", type=Path, default=Path("/usr/local/bin/npu-smi"))
    parser.add_argument("--cann-version-file", required=True, type=Path)
    parser.add_argument("--fia-vector-source", required=True, type=Path)
    parser.add_argument("--fia-cube-source", required=True, type=Path)
    parser.add_argument("--fia-kernel-source", required=True, type=Path)
    parser.add_argument("--build-command", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    _run_campaign(args, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
