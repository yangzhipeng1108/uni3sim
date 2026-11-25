#!/usr/bin/env python3
"""Capture one five-N full-A Scalar-control endpoint with msprof."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


EXPECTED_LAUNCHES = 3
SOURCE_FILES = (
    "CMakeLists.txt",
    "matmul_v2_full_a_five_n_scalar_control_kernel.cpp",
    "matmul_v2_full_a_five_n_scalar_control_main.cpp",
    "matmul_v2_full_a_five_n_scalar_control_runner.py",
)
MODES = {
    "dynamic_m1": (1, 0),
    "static_m16": (16, 1),
}
K_VALUES = (2048, 4096)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _inside_profiler_output(output: Path, repository_root: Path) -> None:
    try:
        relative = output.resolve().relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError("--output-dir must be inside this worktree") from error
    if relative.parts[:3] != ("profiler_output", "microbench", "ascend_910b"):
        raise ValueError("--output-dir must be under profiler_output/microbench/ascend_910b")


def _write_log(path: Path, metadata: str, stdout: str, stderr: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(metadata)
        stream.write("\n[msprof stdout]\n")
        stream.write(stdout)
        stream.write("\n[msprof stderr]\n")
        stream.write(stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--k", choices=K_VALUES, required=True, type=int)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-snapshot", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if re.fullmatch(r"[0-9a-f]{40}", args.baseline) is None:
        parser.error("--baseline must be a lowercase commit")
    if re.fullmatch(r"[A-Za-z0-9._-]{1,96}", args.run_id) is None:
        parser.error("--run-id is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_snapshot) is None:
        parser.error("--expected-snapshot must be a SHA256")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must be executable")
    if args.output_dir.exists():
        raise FileExistsError("--output-dir already exists")
    if shutil.which("msprof") is None:
        raise RuntimeError("msprof is not on PATH")

    root = Path(__file__).resolve().parent
    repository_root = root.parents[4]
    _inside_profiler_output(args.output_dir, repository_root)
    snapshot = _snapshot(root)
    if snapshot != args.expected_snapshot:
        raise RuntimeError("deployed source differs from the audited snapshot")
    logical_m, static_no_tail = MODES[args.mode]
    application = [
        str(args.binary), "--m", str(logical_m), "--k", str(args.k),
        "--static-no-tail", str(static_no_tail), "--trials",
        str(EXPECTED_LAUNCHES),
    ]
    command = [
        "msprof", "op", "--aic-metrics=PipeUtilization",
        f"--launch-count={EXPECTED_LAUNCHES}",
        "--launch-skip-before-match=3",
        f"--output={args.output_dir}",
        f"--application={shlex.join(application)}",
    ]
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    args.output_dir.mkdir(parents=True)
    process = subprocess.Popen(
        command, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    metadata = (
        f"hardware_id=ascend_910b owner=capture-owner-3 "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} msprof_pid={process.pid} "
        f"run_id={args.run_id} mode={args.mode} k={args.k} "
        f"expected_launches={EXPECTED_LAUNCHES} command={shlex.join(command)}")
    print(metadata, file=sys.stderr, flush=True)
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        _write_log(args.output_dir / "msprof.log", metadata, stdout, stderr)
        raise RuntimeError("msprof capture timed out")
    _write_log(args.output_dir / "msprof.log", metadata, stdout, stderr)
    if process.returncode != 0:
        raise RuntimeError("msprof five-N full-A Scalar capture failed")
    manifest = {
        "schema_version": (
            "midlayer.ascend_910b.aic_matmul_v2_full_a_scalar_pipe_capture.v1"),
        "hardware_id": "ascend_910b",
        "topology": "five_n",
        "owner": "capture-owner-3",
        "measurement_baseline_commit": args.baseline,
        "physical_device_id": args.device_id,
        "mode": args.mode,
        "logical_m": logical_m,
        "projection_k": args.k,
        "static_no_tail": static_no_tail,
        "run_id": args.run_id,
        "expected_launches": EXPECTED_LAUNCHES,
        "binary_sha256": _sha256(args.binary),
        "source_snapshot_sha256": snapshot,
        "capture_runner_sha256": _sha256(Path(__file__).resolve()),
        "msprof_command": command,
        "profiler_log": "msprof.log",
        "production_or_evaluation_duration_consumed": False,
    }
    with (args.output_dir / "capture_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, file=sys.stderr, end="")


if __name__ == "__main__":
    main()
