#!/usr/bin/env python3
"""Compile and capture the SM80 two-slot heterogeneous CTA admission probe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


NVCC = "/usr/local/cuda-12.2/bin/nvcc"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, stdout=None, stderr=None):
    return subprocess.run(
        command, cwd=cwd, check=True, text=True,
        stdout=stdout, stderr=stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.device < 0 or args.trials <= 0:
        raise ValueError("device must be nonnegative and trials positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    build = output / "build"
    snapshot = output / "source_snapshot"
    device_dir = output / "device"
    for path in (build, snapshot, device_dir):
        path.mkdir()
    source = Path(__file__).with_name("heterogeneous_cta_admission.cu")
    runner = Path(__file__).resolve()
    shutil.copy2(source, snapshot / source.name)
    shutil.copy2(runner, snapshot / runner.name)
    repo = runner.parents[4]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True,
        capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=repo, check=True, text=True,
        capture_output=True).stdout
    (output / "baseline_commit.txt").write_text(commit + "\n")
    (output / "tracked_worktree_status.txt").write_text(status)
    executable = build / "heterogeneous_cta_admission"
    with (build / "compile.stdout").open("w") as stdout, (
            build / "compile.stderr").open("w") as stderr:
        _run([
            NVCC, "-O3", "-std=c++17", "-arch=sm_80",
            str(snapshot / source.name), "-o", str(executable),
        ], cwd=repo, stdout=stdout, stderr=stderr)
    query = (
        "index,name,uuid,pci.bus_id,memory.used,memory.total,"
        "utilization.gpu,clocks.sm,clocks.mem")
    with (device_dir / "preflight.csv").open("w") as stdout, (
            device_dir / "preflight.stderr").open("w") as stderr:
        _run([
            "nvidia-smi", f"--query-gpu={query}",
            "--format=csv,noheader", "-i", str(args.device),
        ], cwd=repo, stdout=stdout, stderr=stderr)
    with (output / "raw.csv").open("w") as stdout, (
            output / "native.stderr").open("w") as stderr:
        _run([
            str(executable), str(args.device), str(args.trials),
        ], cwd=repo, stdout=stdout, stderr=stderr)
    with (device_dir / "postflight.csv").open("w") as stdout, (
            device_dir / "postflight.stderr").open("w") as stderr:
        _run([
            "nvidia-smi", f"--query-gpu={query}",
            "--format=csv,noheader", "-i", str(args.device),
        ], cwd=repo, stdout=stdout, stderr=stderr)
    provenance = {
        "schema": "nvidia_a800_heterogeneous_cta_admission_capture_v1",
        "baseline_commit": commit,
        "source_sha256": _sha256(snapshot / source.name),
        "runner_sha256": _sha256(snapshot / runner.name),
        "device": args.device,
        "trials": args.trials,
        "production_or_evaluation_duration_consumed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
