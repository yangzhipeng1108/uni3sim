#!/usr/bin/env python3
"""Compile and capture the independent A800 task-sequence latency probe."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / "nvidia_a800")
SOURCE_PATH = Path(__file__).with_name("task_sequence_latency.cu")
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command, *, env=None, stdout=None, stderr=None):
    return subprocess.run(
        command, check=True, text=True, env=env,
        stdout=stdout, stderr=stderr)


def _device_row(device):
    completed = subprocess.run([
        "nvidia-smi", "-i", str(device),
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,driver_version",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 8:
        raise ValueError("unexpected nvidia-smi device row")
    row = {
        "index": int(fields[0]),
        "name": fields[1],
        "uuid": fields[2],
        "memory_used_mib": int(fields[3]),
        "utilization_gpu_pct": int(fields[4]),
        "sm_clock_mhz": int(fields[5]),
        "memory_clock_mhz": int(fields[6]),
        "driver_version": fields[7],
    }
    if row["index"] != device or row["name"] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    output = args.output_dir.resolve()
    if output == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output directory must be below the A800 microbench root")
    if output.exists():
        raise FileExistsError(output)
    if not NVCC.is_file():
        raise FileNotFoundError(NVCC)
    device = _device_row(args.device)
    if device["memory_used_mib"] != 0 or device["utilization_gpu_pct"] != 0:
        raise RuntimeError(f"target GPU is not idle: {device}")

    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    build = output / "build"
    snapshot.mkdir()
    build.mkdir()
    runner_path = Path(__file__).resolve()
    shutil.copy2(SOURCE_PATH, snapshot / SOURCE_PATH.name)
    shutil.copy2(runner_path, snapshot / runner_path.name)
    binary = build / "task_sequence_latency"
    compile_command = [
        str(NVCC), "-O3", "-std=c++17", "-arch=sm_80",
        str(snapshot / SOURCE_PATH.name), "-o", str(binary),
    ]
    with (build / "compile.stdout").open("w") as stdout, (
            build / "compile.stderr").open("w") as stderr:
        _run(compile_command, stdout=stdout, stderr=stderr)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    with (output / "raw.csv").open("w") as stdout, (
            output / "run.stderr").open("w") as stderr:
        _run([str(binary), "0"], env=environment, stdout=stdout, stderr=stderr)
    postflight = _device_row(args.device)
    if postflight["memory_used_mib"] != 0:
        raise RuntimeError(f"probe did not release GPU memory: {postflight}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        check=True, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT,
        check=True, text=True, capture_output=True).stdout
    provenance = {
        "schema": "nvidia_a800_task_sequence_latency_capture_v2",
        "baseline_commit": commit,
        "source_path": str(SOURCE_PATH.relative_to(REPOSITORY_ROOT)),
        "source_sha256": _sha256(snapshot / SOURCE_PATH.name),
        "runner_sha256": _sha256(snapshot / runner_path.name),
        "binary_sha256": _sha256(binary),
        "raw_sha256": _sha256(output / "raw.csv"),
        "nvcc_path": str(NVCC),
        "nvcc_version": subprocess.run(
            [str(NVCC), "--version"], check=True, text=True,
            capture_output=True).stdout.strip(),
        "compile_command": compile_command,
        "device_before": device,
        "device_after": postflight,
        "tracked_worktree_status": status,
        "production_or_evaluation_duration_consumed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
