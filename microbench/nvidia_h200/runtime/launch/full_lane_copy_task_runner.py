#!/usr/bin/env python3
"""Capture the independent H200 full-lane copy task probe.

H200 owner copy of microbench/nvidia_a800/runtime/launch/
full_lane_copy_task_runner.py.  Same capture (one process, five trials of the
probe, pre/postflight nvidia-smi rows, raw.csv, provenance.json).  Hardware
coupling changed:
  - the binary is built off-host (CUDA 12.2 nvcc -O3 -std=c++17 -arch=sm_90
    -cudart static) and passed with --binary, because this runner executes on
    the H200 host with the system python3; the source and runner SHA256 and
    the baseline commit are passed in for provenance;
  - SM clock/utilization are sampled every 50 ms during the capture
    (clock.csv); the report takes the measured busy SM clock as the
    cycle<->ns conversion and rejects the capture if any busy sample is below
    1980 MHz.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, stdout=None, stderr=None, env=None):
    return subprocess.run(
        command, check=True, text=True, stdout=stdout, stderr=stderr, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    args = parser.parse_args()
    if args.device < 0 or args.trials <= 0:
        raise ValueError("device must be nonnegative and trials positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device_dir = output / "device"
    device_dir.mkdir()
    executable = Path(args.binary).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    query = (
        "index,name,uuid,pci.bus_id,memory.used,memory.total,"
        "utilization.gpu,clocks.sm,clocks.mem")
    with (device_dir / "preflight.csv").open("w") as stdout, (
            device_dir / "preflight.stderr").open("w") as stderr:
        _run([
            "nvidia-smi", f"--query-gpu={query}",
            "--format=csv,noheader", "-i", str(args.device),
        ], stdout=stdout, stderr=stderr)
    loadavg = Path("/proc/loadavg").read_text().split()[:3]
    with (output / "clock.csv").open("w") as clock_file:
        sampler = subprocess.Popen(
            ["nvidia-smi", "-i", str(args.device),
             "--query-gpu=clocks.sm,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "50"], stdout=clock_file)
        try:
            with (output / "raw.csv").open("w") as stdout, (
                    output / "native.stderr").open("w") as stderr:
                # CUDA_VISIBLE_DEVICES exposes only the owned GPU as device 0.
                _run([str(executable), "0", str(args.trials)],
                     stdout=stdout, stderr=stderr, env=env)
        finally:
            time.sleep(0.2)
            sampler.terminate()
            sampler.wait()
    with (device_dir / "postflight.csv").open("w") as stdout, (
            device_dir / "postflight.stderr").open("w") as stderr:
        _run([
            "nvidia-smi", f"--query-gpu={query}",
            "--format=csv,noheader", "-i", str(args.device),
        ], stdout=stdout, stderr=stderr)
    provenance = {
        "schema": "nvidia_h200_full_lane_copy_task_capture_v1",
        "baseline_commit": args.baseline_commit,
        "source_sha256": args.source_sha256,
        "runner_sha256": args.runner_sha256,
        "binary_sha256": _sha256(executable),
        "compile": "CUDA 12.2 nvcc -O3 -std=c++17 -arch=sm_90 -cudart static",
        "device": args.device,
        "trials": args.trials,
        "host_loadavg_before": loadavg,
        "production_or_evaluation_duration_consumed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
