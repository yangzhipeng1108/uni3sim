#!/usr/bin/env python3
"""Capture independent SM80 LDGSTS finite-startup scaling across CTA waves."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from staged_ldgsts_l2_runner import (
    CUOBJDUMP,
    NVCC,
    _capture,
    _device,
    _sha256,
    _wait_for_idle,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("staged_ldgsts_depth3_wave_scaling.cu")
RUNNER = Path(__file__).resolve()
GRID_COUNTS = (108, 216, 432, 864, 1728, 3456, 6912, 11008)
ROUNDS = (0, 32, 256, 1024)
PANEL_BYTES = (128 * 64 + 64 * 256) * 2
SM_COUNT = 108
MAXIMUM_RELATIVE_RANGE = 0.15
MAXIMUM_ABSOLUTE_RANGE_NS = 2_100.0


def _median_cells(rows):
    medians = {}
    summaries = []
    for grid in GRID_COUNTS:
        for rounds in ROUNDS:
            cell = [
                row["duration_ms"] * 1.0e6 for row in rows
                if row["grid_ctas"] == grid and row["rounds"] == rounds
            ]
            center_ns = statistics.median(cell)
            relative_range = (max(cell) - min(cell)) / center_ns
            absolute_range_ns = max(cell) - min(cell)
            if (relative_range > MAXIMUM_RELATIVE_RANGE
                    and absolute_range_ns > MAXIMUM_ABSOLUTE_RANGE_NS):
                raise ValueError(
                    f"unstable grid={grid}, rounds={rounds}: relative="
                    f"{relative_range}, absolute_ns={absolute_range_ns}")
            medians[(grid, rounds)] = center_ns
            logical_bytes = grid * rounds * PANEL_BYTES
            summaries.append({
                "grid_ctas": grid,
                "cta_waves": math.ceil(grid / SM_COUNT),
                "rounds": rounds,
                "median_duration_ns": center_ns,
                "relative_range": relative_range,
                "absolute_range_ns": absolute_range_ns,
                "logical_bytes": logical_bytes,
            })
    return medians, summaries


def _finite_startup_scaling(medians):
    result = []
    baseline_residual = None
    for grid in GRID_COUNTS:
        empty_ns = medians[(grid, 0)]
        steady_round_ns = (
            medians[(grid, 1024)] - medians[(grid, 256)]) / (1024 - 256)
        if steady_round_ns <= 0.0:
            raise ValueError(f"grid={grid} has non-positive steady slope")
        finite_residual_ns = (
            medians[(grid, 32)] - empty_ns - 32 * steady_round_ns)
        waves = math.ceil(grid / SM_COUNT)
        if baseline_residual is None:
            baseline_residual = finite_residual_ns
        expected_from_one_wave_ns = baseline_residual * waves
        result.append({
            "grid_ctas": grid,
            "cta_waves": waves,
            "empty_grid_duration_ns": empty_ns,
            "steady_round_ns": steady_round_ns,
            "finite_residual_ns_at_32_rounds": finite_residual_ns,
            "one_wave_scaled_finite_residual_ns": expected_from_one_wave_ns,
            "finite_residual_to_one_wave_scaled_ratio": (
                finite_residual_ns / expected_from_one_wave_ns
                if expected_from_one_wave_ns != 0.0 else None),
        })
    return result


def _canonical_commit(checkout):
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-checkout", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    if args.trials < 3 or args.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    if not args.owner.strip():
        raise ValueError("owner must be non-empty")
    canonical_checkout = args.canonical_checkout.resolve()
    canonical_profiler = canonical_checkout / "profiler_output"
    output = args.output_dir.resolve()
    if canonical_profiler not in output.parents:
        raise ValueError("output-dir must be an exclusive canonical profiler child")
    if output.exists():
        raise FileExistsError(output)
    device_before = _device(args.device)
    output.mkdir(parents=True)
    binary = output / "staged_ldgsts_depth3_wave_scaling"
    compile_command = [
        str(NVCC), "-std=c++17", "-O3",
        "--generate-code=arch=compute_80,code=sm_80", "--disable-warnings",
        "-o", str(binary), str(SOURCE),
    ]
    compile_result = _capture(compile_command)
    sass = _capture([str(CUOBJDUMP), "--dump-sass", str(binary)]).stdout
    required = ("LDGSTS.E.BYPASS.128", "LDGDEPBAR", "DEPBAR.LE SB0, 0x1")
    missing = [opcode for opcode in required if opcode not in sass]
    forbidden = [opcode for opcode in ("LDL", "STL") if opcode in sass]
    if missing or forbidden:
        raise ValueError(
            f"compiled SASS mismatch: missing={missing}, forbidden={forbidden}")
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
    (output / "raw.csv").write_text(result.stdout, encoding="utf-8")
    raw_rows = list(csv.DictReader(result.stdout.splitlines()))
    expected = {
        (grid, rounds, trial)
        for grid in GRID_COUNTS for rounds in ROUNDS
        for trial in range(args.trials)
    }
    identities = set()
    rows = []
    for raw in raw_rows:
        grid = int(raw["grid_ctas"])
        rounds = int(raw["rounds"])
        trial = int(raw["trial"])
        duration_ms = float(raw["duration_ms"])
        logical_bytes = int(raw["logical_bytes"])
        identity = (grid, rounds, trial)
        if (identity not in expected or identity in identities
                or not math.isfinite(duration_ms) or duration_ms <= 0.0
                or logical_bytes != grid * rounds * PANEL_BYTES
                or int(raw["resident_ctas_per_sm"]) != 1):
            raise ValueError(f"invalid native row: {raw}")
        identities.add(identity)
        rows.append({
            "grid_ctas": grid,
            "rounds": rounds,
            "trial": trial,
            "duration_ms": duration_ms,
        })
    if identities != expected or len(raw_rows) != len(identities):
        raise ValueError("native result matrix is incomplete or duplicated")
    medians, summaries = _median_cells(rows)
    device_after = _wait_for_idle(args.device)
    report = {
        "schema": "nvidia_a800_staged_ldgsts_depth3_wave_scaling_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "three_stage_fp16_ldgsts_finite_startup_wave_scaling",
        "scope": {
            "grid_ctas": list(GRID_COUNTS),
            "cta_waves": [math.ceil(grid / SM_COUNT) for grid in GRID_COUNTS],
            "rounds": list(ROUNDS),
            "block_threads": 256,
            "resident_ctas_per_sm": 1,
            "pipeline_stages": 3,
            "wait_group": 1,
            "panel_bytes": PANEL_BYTES,
            "cache_tier": "l2_resident",
        },
        "summaries": summaries,
        "finite_startup_scaling": _finite_startup_scaling(medians),
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "canonical_checkout": str(canonical_checkout),
            "canonical_commit": _canonical_commit(canonical_checkout),
            "source_worktree_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                text=True, capture_output=True).stdout.strip(),
            "source_worktree_branch": subprocess.run(
                ["git", "branch", "--show-current"], cwd=ROOT, check=True,
                text=True, capture_output=True).stdout.strip(),
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
    candidate = output / "candidate.json"
    candidate.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(candidate)


if __name__ == "__main__":
    main()
