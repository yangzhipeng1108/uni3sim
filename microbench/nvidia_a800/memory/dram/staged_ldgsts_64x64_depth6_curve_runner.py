#!/usr/bin/env python3
"""Capture the independent A800 64x64 depth-six LDGSTS source service."""

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
SOURCE = Path(__file__).with_name("staged_ldgsts_64x64_depth6_curve.cu")
RUNNER = Path(__file__).resolve()
OUTPUT_ROOT = ROOT / "profiler_output" / "microbench" / "nvidia_a800"
NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda-12.2/bin/cuobjdump")
GRID = 64
ROUNDS = 4096
PANEL_BYTES = 16 * 1024
TIERS = ("l2_resident", "dram_unique")


def _capture(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device(device):
    result = _capture([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used",
        "--format=csv,noheader,nounits",
    ])
    row, = result.stdout.strip().splitlines()
    fields = [field.strip() for field in row.split(",")]
    if len(fields) != 6:
        raise ValueError(f"unexpected device query: {row}")
    if (int(fields[0]) != device or fields[2] != "NVIDIA A800-SXM4-80GB"
            or int(fields[3]) != 0 or int(fields[4]) != 0
            or int(fields[5]) != 0):
        raise RuntimeError(f"device {device} is not idle: {row}")
    return {
        "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
        "utilization_gpu_percent": int(fields[3]),
        "utilization_memory_percent": int(fields[4]),
        "memory_used_mib": int(fields[5]),
    }


def _wait_for_idle(device):
    deadline = time.time() + 30.0
    while True:
        try:
            return _device(device)
        except RuntimeError:
            if time.time() >= deadline:
                raise
            time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.trials < 3 or args.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    if not args.owner.strip():
        raise ValueError("owner must be non-empty")
    output = args.output_dir.resolve()
    if output == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output-dir must be an exclusive A800 profiler child")
    if output.exists():
        raise FileExistsError(output)
    device_before = _device(args.device)
    output.mkdir(parents=True)
    binary = output / "staged_ldgsts_64x64_depth6_curve"
    compile_command = [
        str(NVCC), "-std=c++17", "-O3",
        "--generate-code=arch=compute_80,code=sm_80", "--disable-warnings",
        "-o", str(binary), str(SOURCE),
    ]
    compile_result = _capture(compile_command)
    sass = _capture([str(CUOBJDUMP), "--dump-sass", str(binary)]).stdout
    required = ("LDGSTS", "LDGDEPBAR", "DEPBAR.LE SB0, 0x1")
    missing = [opcode for opcode in required if opcode not in sass]
    if missing or "LDL" in sass or "STL" in sass:
        raise ValueError(f"compiled SASS violates source contract: {missing}")
    (output / "probe.sass.txt").write_text(sass, encoding="utf-8")
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir()
    shutil.copy2(SOURCE, source_snapshot / SOURCE.name)
    shutil.copy2(RUNNER, source_snapshot / RUNNER.name)
    environment = dict(os.environ)
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if name in environment:
            raise RuntimeError(f"{name} must be unset")
    command = [str(binary), str(args.device), str(args.trials)]
    started = time.time()
    result = _capture(command, environment=environment)
    completed = time.time()
    rows = list(csv.DictReader(result.stdout.splitlines()))
    expected = {(tier, trial) for tier in TIERS for trial in range(args.trials)}
    seen = set()
    capacities = {}
    for row in rows:
        tier = row["tier"]
        trial = int(row["trial"])
        identity = (tier, trial)
        if (identity not in expected or identity in seen
                or int(row["logical_bytes"]) != GRID * ROUNDS * PANEL_BYTES
                or int(row["resident_ctas_per_sm"]) != 1):
            raise ValueError(f"invalid native row: {row}")
        duration_ms = float(row["duration_ms"])
        if not math.isfinite(duration_ms) or duration_ms <= 0.0:
            raise ValueError(f"invalid duration: {row}")
        seen.add(identity)
        capacities.setdefault(tier, []).append(
            GRID * ROUNDS * PANEL_BYTES / (duration_ms * 1.0e-3))
    if seen != expected or len(rows) != len(expected):
        raise ValueError("probe result matrix is incomplete or duplicated")
    summary = {}
    for tier, values in capacities.items():
        center = statistics.median(values)
        spread = (max(values) - min(values)) / center
        if spread > 0.10:
            raise ValueError(f"unstable {tier} service: {spread}")
        summary[tier] = {
            "logical_byte_capacity_per_s": center,
            "relative_trial_spread": spread,
            "trial_capacities_per_s": values,
        }
    (output / "raw.csv").write_text(result.stdout, encoding="utf-8")
    device_after = _wait_for_idle(args.device)
    report = {
        "schema": "nvidia_a800_staged_ldgsts_64x64_depth6_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "sm80_fp16_64x64x64_three_pair_ldgsts_source_service",
        "scope": {
            "grid_ctas": GRID,
            "threads_per_cta": 128,
            "resident_ctas_per_sm": 1,
            "complete_operand_pair_slots": 3,
            "a_panel_bytes": 8 * 1024,
            "b_panel_bytes": 8 * 1024,
            "panel_bytes": PANEL_BYTES,
            "rounds": ROUNDS,
            "tiers": list(TIERS),
        },
        "services": summary,
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "baseline_commit": "0c530a00a9f7c1135210e99b843cce3f8250e3ac",
            "source_commit": _capture(
                ["git", "rev-parse", "HEAD"], environment=os.environ).stdout.strip(),
            "branch": _capture(
                ["git", "branch", "--show-current"], environment=os.environ).stdout.strip(),
            "owner": args.owner,
            "device_before": device_before,
            "device_after": device_after,
            "compile_command": compile_command,
            "compiler_stderr": compile_result.stderr,
            "command": command,
            "pid": os.getpid(),
            "started_at_unix_s": started,
            "completed_at_unix_s": completed,
            "source_sha256": _sha256(SOURCE),
            "runner_sha256": _sha256(RUNNER),
            "binary_sha256": _sha256(binary),
            "sass_sha256": _sha256(output / "probe.sass.txt"),
        },
    }
    (output / "candidate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output / "candidate.json")


if __name__ == "__main__":
    main()
