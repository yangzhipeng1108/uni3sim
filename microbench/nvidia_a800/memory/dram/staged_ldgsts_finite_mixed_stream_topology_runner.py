#!/usr/bin/env python3
"""Sweep the finite mixed-tier LDGSTS stream over launch topologies.

Extends the preserved mixed-tier probe (grid 64, three slots, 64--512 rounds)
across grid, ring-slot, and short-round operating points so the finite async
operand-stream completion envelope can be validated where small-M GEMM
launches actually live.  Copy-only: no matrix operation, output epilogue, or
production-kernel source; no production or evaluation duration is consumed.
"""

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
SOURCE = Path(__file__).with_name("staged_ldgsts_finite_mixed_stream.cu")
RUNNER = Path(__file__).resolve()
OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_a800"
NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
PANEL_BYTES = 16 * 1024
ROUNDS = (0, 16, 32, 40, 48, 64, 96, 152)
FRACTIONS = (("l2_0pct", 0), ("l2_25pct", 25), ("l2_50pct", 50),
             ("l2_75pct", 75), ("l2_100pct", 100))


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


def _parse(output, trials, rounds, grid):
    rows = list(csv.DictReader(output.splitlines()))
    known_fractions = dict(FRACTIONS)
    parsed = []
    for row in rows:
        fraction_name = row["l2_fraction"]
        trial = int(row["trial"])
        if fraction_name not in known_fractions or trial not in range(trials):
            raise ValueError(f"unexpected probe row: {row}")
        if int(row["rounds"]) != rounds:
            raise ValueError(f"probe rounds mismatch: {row}")
        total_bytes = int(row["total_bytes"])
        expected_total = grid * rounds * PANEL_BYTES
        if total_bytes != expected_total:
            raise ValueError(f"probe traffic mismatch: {row}")
        parsed.append({
            "l2_fraction_pct": known_fractions[fraction_name],
            "trial": trial,
            "rounds": rounds,
            "duration_ms": float(row["duration_ms"]),
        })
    expected = {(fraction, t) for _, fraction in FRACTIONS
                for t in range(trials)}
    if {(r["l2_fraction_pct"], r["trial"]) for r in parsed} != expected:
        raise ValueError("probe capture matrix is incomplete")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--grids", type=int, nargs="+",
                        default=[64, 96, 108, 216, 304])
    parser.add_argument("--slots", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--ksync", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rounds", type=int, nargs="+", default=list(ROUNDS))
    arguments = parser.parse_args()
    if not 3 <= arguments.trials <= 11:
        raise ValueError("trials must be in [3, 11]")
    if not arguments.owner.strip():
        raise ValueError("owner must be non-empty")
    output = arguments.output_dir.resolve()
    if OUTPUT_ROOT.resolve() not in output.parents or output.exists():
        raise ValueError("output-dir must be a new profiler-output child")
    device_before = _device(arguments.device)
    output.mkdir(parents=True)
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
    for grid in arguments.grids:
        for slots in arguments.slots:
            binary = output / f"probe_g{grid}_s{slots}_k{arguments.ksync}"
            compile_command = [
                str(NVCC), "-std=c++17", "-O3", "--disable-warnings",
                "--generate-code=arch=compute_80,code=sm_80",
                f"-DPROBE_GRID={grid}", f"-DPROBE_SLOTS={slots}",
                f"-DPROBE_KSYNC={arguments.ksync}",
                "-o", str(binary), str(SOURCE),
            ]
            compile_result = _run(compile_command)
            for rounds in arguments.rounds:
                command = [str(binary), str(arguments.device),
                           str(arguments.trials), str(rounds)]
                started = time.time()
                captured = _run(command, environment=environment)
                completed = time.time()
                for row in _parse(captured.stdout, arguments.trials,
                                  rounds, grid):
                    row["grid"] = grid
                    row["slots"] = slots
                    rows.append(row)
                command_records.append({
                    "grid": grid, "slots": slots, "rounds": rounds,
                    "command": command,
                    "started_at_unix_s": started,
                    "completed_at_unix_s": completed,
                    "stderr": captured.stderr,
                })
            command_records.append({
                "grid": grid, "slots": slots,
                "compile_command": compile_command,
                "compiler_stderr": compile_result.stderr,
                "binary_sha256": _sha256(binary),
            })
    (output / "raw.csv").write_text(
        "grid,slots,l2_fraction_pct,trial,rounds,duration_ms\n"
        + "".join(
            f'{r["grid"]},{r["slots"]},{r["l2_fraction_pct"]},{r["trial"]},'
            f'{r["rounds"]},{r["duration_ms"]:.9f}\n'
            for r in rows), encoding="utf-8")

    summary = {}
    for grid in arguments.grids:
        for slots in arguments.slots:
            cell = {}
            for fraction_name, fraction in FRACTIONS:
                by_rounds = {}
                for rounds in arguments.rounds:
                    durations = [
                        r["duration_ms"] for r in rows
                        if (r["grid"] == grid and r["slots"] == slots
                            and r["l2_fraction_pct"] == fraction
                            and r["rounds"] == rounds)]
                    median_ms = statistics.median(durations)
                    spread = ((max(durations) - min(durations)) / median_ms
                              if median_ms else math.inf)
                    by_rounds[str(rounds)] = {
                        "median_duration_ns": median_ms * 1.0e6,
                        "relative_trial_spread": spread,
                    }
                cell[str(fraction)] = by_rounds
            summary[f"grid{grid}_slots{slots}"] = cell
    # The final probe launches leave the utilization sampler non-zero for a
    # short settle window; wait for the device to drain before the closing
    # idle check instead of racing it.
    time.sleep(5)
    device_after = _device(arguments.device)
    report = {
        "schema": "nvidia_a800_finite_mixed_stream_topology_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "finite_async_operand_stream_completion_topology",
        "production_or_evaluation_duration_consumed": False,
        "scope": {
            "threads_per_cta": 128, "resident_ctas_per_sm": 1,
            "operand_pair_bytes": PANEL_BYTES,
            "k_synchronized_order": bool(arguments.ksync),
            "grids": arguments.grids, "slots": arguments.slots,
            "rounds": list(arguments.rounds),
            "l2_fraction_pct": [fraction for _, fraction in FRACTIONS],
        },
        "results": summary,
        "provenance": {
            "owner": arguments.owner,
            "baseline_commit": _run(
                ["git", "rev-parse", "HEAD"]).stdout.strip(),
            "branch": _run(
                ["git", "branch", "--show-current"]).stdout.strip(),
            "device_before": device_before, "device_after": device_after,
            "commands": command_records,
            "source_sha256": _sha256(SOURCE), "runner_sha256": _sha256(RUNNER),
        },
        "allowed_use": (
            "validate how the finite async operand-stream completion envelope "
            "scales with grid, ring slots, and short round counts"),
        "forbidden_use": (
            "production GEMM duration, per-shape correction, or a direct "
            "replacement for compute or output work"),
    }
    candidate = output / "candidate.json"
    candidate.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(candidate)


if __name__ == "__main__":
    main()
