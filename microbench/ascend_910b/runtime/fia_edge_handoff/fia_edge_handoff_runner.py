#!/usr/bin/env python3
"""Capture independent CANN FIA handoff-edge completion latencies."""

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


BENCHMARK_ROOT = Path(__file__).resolve().parent
CROSS_DOMAIN_ROOT = BENCHMARK_ROOT.parent / "cross_domain"
sys.path.insert(0, str(CROSS_DOMAIN_ROOT))
from cross_domain_runner import (  # noqa: E402
    EXPECTED_CANN,
    EXPECTED_CANN_SOURCE_SHA256,
    SOURCE_MARKERS,
    _parse_cann_version,
    _parse_global_identity,
    _parse_idle_usage,
    _run_checked,
    _validate_cann_sources,
)


BLOCK_DIMS = (1, 8, 11, 12, 16, 18, 19, 21, 22, 23, 24)
TASK_COUNTS = (4, 8, 16, 32, 64, 92, 128)
CELLS = tuple(
    (f"uniform_b{block_dim}_n{task_count}", block_dim, task_count)
    for block_dim in BLOCK_DIMS
    for task_count in TASK_COUNTS)
TRIALS = 6
SYSTEM_COUNTER_HZ = 50_000_000
SYSTEM_COUNTER_TICK_NS = 20.0
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_PATHS = {
    "CMakeLists.txt": BENCHMARK_ROOT / "CMakeLists.txt",
    "fia_edge_handoff_contract.h": (
        BENCHMARK_ROOT / "fia_edge_handoff_contract.h"),
    "fia_edge_handoff_rt_kernel.cpp": (
        BENCHMARK_ROOT / "fia_edge_handoff_rt_kernel.cpp"),
    "fia_edge_handoff_rt_main.cpp": (
        BENCHMARK_ROOT / "fia_edge_handoff_rt_main.cpp"),
    "fia_edge_handoff_runner.py": BENCHMARK_ROOT / "fia_edge_handoff_runner.py",
    "fia_edge_handoff_report.py": BENCHMARK_ROOT / "fia_edge_handoff_report.py",
    "cross_domain_runner.py": CROSS_DOMAIN_ROOT / "cross_domain_runner.py",
}
EDGE_FIELDS = (
    "c1v1_total_cycles", "v1c2_total_cycles", "c2v2_total_cycles",
    "c1v1_prefix1_cycles", "v1c2_prefix1_cycles", "c2v2_prefix1_cycles",
    "c1v1_prefix2_cycles", "v1c2_prefix2_cycles", "c2v2_prefix2_cycles",
    "c1v1_prefix3_cycles", "v1c2_prefix3_cycles", "c2v2_prefix3_cycles",
)
HEALTH_FIELDS = (
    "c1v1_lane_skew_ppm", "c2v2_lane_skew_ppm",
    "ring_elapsed_cycles")
CHILD_FIELDS = (
    "block_dim", "task_count", *EDGE_FIELDS, *HEALTH_FIELDS,
    "host_wall_ns", "validation")
RAW_FIELDS = (
    "run_id", "trial", "order_direction", "order_position",
    "process_pid", "profile_id", *CHILD_FIELDS)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256():
    digest = hashlib.sha256()
    for label, path in sorted(SOURCE_PATHS.items()):
        payload = path.read_bytes()
        digest.update(label.encode("utf-8"))
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


def _positive_integer(text, label):
    value = int(text)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


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


def _run_campaign(args):
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID contains unsupported characters")
    if COMMIT_PATTERN.fullmatch(args.source_commit) is None:
        raise ValueError("source commit must be 40 lowercase hex digits")
    if SHA256_PATTERN.fullmatch(args.expected_snapshot_sha256) is None:
        raise ValueError("expected snapshot must be a lowercase SHA256")
    if args.device_id not in (1, 5):
        raise ValueError("this Goal only authorizes preferred devices 1 or 5")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory exists: {args.output_dir}")
    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary is missing: {args.binary}")
    observed_snapshot = _snapshot_sha256()
    if observed_snapshot != args.expected_snapshot_sha256:
        raise ValueError("deployed source differs from the audited snapshot")
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
        str(args.device_id)]
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
    artifacts = []
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for trial_index in range(TRIALS):
            direction, cells = _ordered_cells(trial_index)
            command = [str(args.binary), "--batch"]
            for _, block_dim, task_count in cells:
                command.extend((str(block_dim), str(task_count)))
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=environment, start_new_session=True)
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
            if not lines or tuple(lines[0].split(",")) != CHILD_FIELDS:
                raise RuntimeError("child output header changed")
            if len(lines) != len(cells) + 1:
                raise RuntimeError("child output cell count changed")
            for position, (cell, line) in enumerate(zip(cells, lines[1:])):
                parsed = dict(zip(CHILD_FIELDS, line.split(","), strict=True))
                profile_id, block_dim, task_count = cell
                if (int(parsed["block_dim"]) != block_dim
                        or int(parsed["task_count"]) != task_count
                        or parsed["validation"] != "ok"):
                    raise RuntimeError("child changed the requested cell")
                for field in EDGE_FIELDS:
                    _positive_integer(parsed[field], field)
                for field in HEALTH_FIELDS:
                    if int(parsed[field]) < 0:
                        raise ValueError(f"{field} must be non-negative")
                if int(parsed["host_wall_ns"]) <= 0:
                    raise RuntimeError("child wall interval is invalid")
                writer.writerow({
                    "run_id": args.run_id,
                    "trial": trial_index + 1,
                    "order_direction": direction,
                    "order_position": position,
                    "process_pid": process.pid,
                    "profile_id": profile_id,
                    **parsed,
                })
            artifacts.append({
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
        "schema": "ascend_910b_fia_edge_handoff_provenance_v2",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_snapshot_sha256": observed_snapshot,
        "source_files": {
            label: _sha256(path) for label, path in SOURCE_PATHS.items()},
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
            "cross_actor_property": "device-global system counter",
        },
        "measurement_scope": (
            "source-ordered empty C1-to-V1, V1-to-C2, and C2-to-V2 "
            "CrossCore dependency completions in a 1:2 mixed-core group; "
            "no production arithmetic, tensor traffic, or operator duration"
        ),
        "production_or_evaluation_duration_consumed": False,
        "campaign": {
            "block_dims": list(BLOCK_DIMS),
            "task_counts": list(TASK_COUNTS),
            "trials": TRIALS,
            "order": "balanced_forward_reverse",
            "untimed_preamble": "one B1/N4 then one B24/N128 launch",
            "exact_cell_warmups": 1,
            "prefix_depth": 3,
        },
        "capture": {
            "raw": raw_path.name,
            "raw_sha256": _sha256(raw_path),
            "raw_rows": TRIALS * len(CELLS),
            "runner_pid": os.getpid(),
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
            "trials": artifacts,
        },
    }
    _write_json_exclusive(args.output_dir / "provenance.json", provenance)
    print(args.output_dir / "provenance.json")


def main():
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
        print(_snapshot_sha256())
    elif args.command == "run":
        if args.child_timeout_seconds <= 0:
            raise ValueError("child timeout must be positive")
        _run_campaign(args)
    else:
        raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
