#!/usr/bin/env python3
"""Capture an independent SM80 finite three-slot LDGSTS stream curve."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("staged_ldgsts_finite_stream.cu")
RUNNER = Path(__file__).resolve()
OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_a800"
NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda-12.2/bin/cuobjdump")
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
GRID = 64
PANEL_BYTES = 16 * 1024
SLOTS = 3
ROUNDS = (64, 128, 256, 512)
TIERS = ("l2_resident", "dram_unique")


def _run(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(device):
    result = _run([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.used",
        "--format=csv,noheader,nounits",
    ])
    row, = result.stdout.strip().splitlines()
    fields = tuple(value.strip() for value in row.split(","))
    if len(fields) != 6:
        raise ValueError(f"unexpected nvidia-smi device row: {row!r}")
    record = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "gpu_utilization_pct": int(fields[3]),
        "memory_utilization_pct": int(fields[4]),
        "memory_used_mib": int(fields[5]),
    }
    if record["index"] != device or record["name"] != DEVICE_NAME:
        raise RuntimeError(f"unexpected device contract: {record}")
    applications = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    if any(line.strip() and line.split(",", 1)[0].strip() == record["uuid"]
           for line in applications.splitlines()):
        raise RuntimeError(f"device {device} has a compute process")
    if (record["gpu_utilization_pct"] != 0
            or record["memory_utilization_pct"] != 0
            or record["memory_used_mib"] != 0):
        raise RuntimeError(f"device {device} is not idle: {record}")
    return record


def _wait_for_idle(device):
    deadline = time.monotonic() + 30.0
    while True:
        try:
            return _device(device)
        except RuntimeError as error:
            if "is not idle" not in str(error) or time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def _parse(output, trials, rounds):
    rows = list(csv.DictReader(output.splitlines()))
    expected = {(tier, trial) for tier in TIERS for trial in range(trials)}
    seen = set()
    parsed = []
    for row in rows:
        tier = row["tier"]
        trial = int(row["trial"])
        if tier not in TIERS or trial not in range(trials):
            raise ValueError(f"unexpected finite-stream row: {row}")
        if int(row["rounds"]) != rounds:
            raise ValueError(f"finite-stream rounds mismatch: {row}")
        if int(row["logical_bytes"]) != GRID * rounds * PANEL_BYTES:
            raise ValueError(f"finite-stream logical bytes mismatch: {row}")
        if int(row["resident_ctas_per_sm"]) != 1:
            raise ValueError(f"finite-stream residency mismatch: {row}")
        duration_ms = float(row["duration_ms"])
        if not math.isfinite(duration_ms) or duration_ms <= 0.0:
            raise ValueError(f"finite-stream duration is invalid: {row}")
        identity = (tier, trial)
        if identity in seen:
            raise ValueError(f"duplicate finite-stream row: {row}")
        seen.add(identity)
        parsed.append({
            "tier": tier, "trial": trial, "rounds": rounds,
            "duration_ms": duration_ms,
            "logical_bytes": int(row["logical_bytes"]),
        })
    if seen != expected or len(rows) != len(expected):
        raise ValueError("finite-stream capture matrix is incomplete")
    return parsed


def _summary(rows):
    result = {}
    for tier in TIERS:
        by_rounds = {}
        for rounds in ROUNDS:
            durations = [
                row["duration_ms"] for row in rows
                if row["tier"] == tier and row["rounds"] == rounds]
            median_ms = statistics.median(durations)
            spread = (max(durations) - min(durations)) / median_ms
            if spread > 0.10:
                raise ValueError(
                    f"finite-stream {tier}/{rounds} stability gate failed: {spread}")
            by_rounds[str(rounds)] = {
                "median_duration_ns": median_ms * 1.0e6,
                "relative_trial_spread": spread,
                "trial_duration_ms": durations,
                "logical_byte_capacity_per_s": (
                    GRID * rounds * PANEL_BYTES / (median_ms * 1.0e-3)),
            }
        result[tier] = by_rounds
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.trials < 3 or arguments.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    if not arguments.owner.strip():
        raise ValueError("owner must be non-empty")
    output = arguments.output_dir.resolve()
    if OUTPUT_ROOT.resolve() not in output.parents or output.exists():
        raise ValueError("output-dir must be a new profiler-output child")
    device_before = _device(arguments.device)
    output.mkdir(parents=True)
    binary = output / "staged_ldgsts_finite_stream"
    compile_command = [
        str(NVCC), "-std=c++17", "-O3", "--disable-warnings",
        "--generate-code=arch=compute_80,code=sm_80", "-o", str(binary),
        str(SOURCE),
    ]
    compile_result = _run(compile_command)
    sass = _run([str(CUOBJDUMP), "--dump-sass", str(binary)]).stdout
    required = ("LDGSTS", "LDGDEPBAR", "DEPBAR.LE SB0, 0x1")
    missing = tuple(opcode for opcode in required if opcode not in sass)
    if missing:
        raise ValueError(f"finite stream lost async-copy SASS: {missing}")
    (output / "probe.sass.txt").write_text(sass, encoding="utf-8")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    shutil.copy2(SOURCE, snapshot / SOURCE.name)
    shutil.copy2(RUNNER, snapshot / RUNNER.name)
    environment = dict(os.environ)
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if name in environment:
            raise RuntimeError(f"{name} must be unset")
    rows = []
    command_records = []
    for rounds in ROUNDS:
        command = [str(binary), str(arguments.device), str(arguments.trials), str(rounds)]
        started = time.time()
        captured = _run(command, environment=environment)
        completed = time.time()
        rows.extend(_parse(captured.stdout, arguments.trials, rounds))
        command_records.append({
            "rounds": rounds, "command": command,
            "started_at_unix_s": started, "completed_at_unix_s": completed,
            "stdout": captured.stdout, "stderr": captured.stderr,
        })
    (output / "raw.csv").write_text(
        "tier,trial,rounds,duration_ms,logical_bytes\n" + "".join(
            f'{row["tier"]},{row["trial"]},{row["rounds"]},'
            f'{row["duration_ms"]:.9f},{row["logical_bytes"]}\n'
            for row in rows), encoding="utf-8")
    device_after = _wait_for_idle(arguments.device)
    report = {
        "schema": "nvidia_a800_three_slot_ldgsts_finite_stream_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "finite_three_slot_async_operand_stream",
        "production_or_evaluation_duration_consumed": False,
        "scope": {
            "grid_ctas": GRID, "threads_per_cta": 128,
            "resident_ctas_per_sm": 1, "operand_pair_bytes": PANEL_BYTES,
            "complete_operand_pair_slots": SLOTS,
            "static_occupancy_bytes": SLOTS * PANEL_BYTES,
            "dynamic_operand_bytes": SLOTS * PANEL_BYTES,
            "rounds": list(ROUNDS),
            "tiers": list(TIERS),
        },
        "results": _summary(rows),
        "provenance": {
            "owner": arguments.owner, "baseline_commit": _run(
                ["git", "rev-parse", "HEAD"], environment=os.environ).stdout.strip(),
            "branch": _run(["git", "branch", "--show-current"], environment=os.environ).stdout.strip(),
            "device_before": device_before, "device_after": device_after,
            "compile_command": compile_command,
            "compiler_stderr": compile_result.stderr,
            "commands": command_records,
            "source_sha256": _sha256(SOURCE),
            "runner_sha256": _sha256(RUNNER),
            "binary_sha256": _sha256(binary),
            "sass_sha256": _sha256(output / "probe.sass.txt"),
        },
        "allowed_use": (
            "derive generic fill, steady-state, and drain behavior for a finite "
            "three-slot asynchronous copy stream with the stated resource envelope"),
        "forbidden_use": (
            "production GEMM duration, per-shape correction, or a direct "
            "replacement for compute or output work"),
    }
    candidate = output / "candidate.json"
    candidate.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(candidate)


if __name__ == "__main__":
    main()
