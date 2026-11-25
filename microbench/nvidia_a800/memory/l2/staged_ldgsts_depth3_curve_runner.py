#!/usr/bin/env python3
"""Capture the independent A800 three-stage LDGSTS service curve."""

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
    OUTPUT_ROOT,
    _capture,
    _device,
    _sha256,
    _wait_for_idle,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("staged_ldgsts_depth3_curve.cu")
RUNNER = Path(__file__).resolve()
ROUNDS = (0, 16, 32, 48, 128, 256, 1024)
TIERS = ("l2_resident", "dram_unique")
GRID = 108
PANEL_BYTES = (128 * 64 + 64 * 256) * 2
MAXIMUM_RELATIVE_RANGE = 0.15
MAXIMUM_ABSOLUTE_RANGE_NS = 2_100.0


def _summarize(rows):
    summaries = []
    medians = {}
    for tier in TIERS:
        for rounds in ROUNDS:
            cell = [
                row for row in rows
                if row["tier"] == tier and row["rounds"] == rounds]
            durations_ns = [row["duration_ms"] * 1.0e6 for row in cell]
            center_ns = statistics.median(durations_ns)
            relative_range = (
                (max(durations_ns) - min(durations_ns)) / center_ns)
            absolute_range_ns = max(durations_ns) - min(durations_ns)
            if (relative_range > MAXIMUM_RELATIVE_RANGE
                    and absolute_range_ns > MAXIMUM_ABSOLUTE_RANGE_NS):
                raise ValueError(
                    f"unstable {tier} rounds={rounds}: relative="
                    f"{relative_range}, absolute_ns={absolute_range_ns}")
            medians[(tier, rounds)] = center_ns
            logical_bytes = GRID * rounds * PANEL_BYTES
            summaries.append({
                "tier": tier,
                "rounds": rounds,
                "median_duration_ns": center_ns,
                "relative_range": relative_range,
                "absolute_range_ns": absolute_range_ns,
                "logical_bytes": logical_bytes,
                "logical_byte_capacity_per_s": (
                    None if logical_bytes == 0 else
                    logical_bytes / (center_ns * 1.0e-9)),
            })

    mechanisms = {}
    for tier in TIERS:
        empty_ns = medians[(tier, 0)]
        long_round_a = 256
        long_round_b = 1024
        steady_round_ns = (
            medians[(tier, long_round_b)]
            - medians[(tier, long_round_a)]) / (
                long_round_b - long_round_a)
        if steady_round_ns <= 0.0:
            raise ValueError(f"{tier} steady slope is not positive")
        finite_residuals = [
            medians[(tier, rounds)] - empty_ns - rounds * steady_round_ns
            for rounds in (16, 32, 48, 128, 256, 1024)
        ]
        mechanisms[tier] = {
            "empty_grid_duration_ns": empty_ns,
            "steady_round_ns": steady_round_ns,
            "steady_logical_byte_capacity_per_s": (
                GRID * PANEL_BYTES / (steady_round_ns * 1.0e-9)),
            "finite_residual_median_ns": statistics.median(
                finite_residuals),
            "finite_residual_ns_by_round": dict(zip(
                (16, 32, 48, 128, 256, 1024), finite_residuals)),
            "derivation": (
                "steady slope is the 256-to-1024 round secant; each finite "
                "residual subtracts the empty-grid median and that steady "
                "slope without clamping or interpreting noise as startup"),
        }
    return summaries, mechanisms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--owner", required=True)
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
    binary = output / "staged_ldgsts_depth3_curve"
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
    expected_identities = {
        (tier, rounds, trial)
        for tier in TIERS for rounds in ROUNDS for trial in range(args.trials)}
    rows = []
    identities = set()
    for raw in raw_rows:
        tier = raw["tier"]
        rounds = int(raw["rounds"])
        trial = int(raw["trial"])
        duration_ms = float(raw["duration_ms"])
        logical_bytes = int(raw["logical_bytes"])
        identity = (tier, rounds, trial)
        if (identity not in expected_identities or identity in identities
                or not math.isfinite(duration_ms) or duration_ms <= 0.0
                or logical_bytes != GRID * rounds * PANEL_BYTES
                or int(raw["resident_ctas_per_sm"]) != 1):
            raise ValueError(f"invalid native row: {raw}")
        identities.add(identity)
        rows.append({
            "tier": tier,
            "rounds": rounds,
            "trial": trial,
            "duration_ms": duration_ms,
            "logical_bytes": logical_bytes,
        })
    if identities != expected_identities or len(raw_rows) != len(identities):
        raise ValueError("native result matrix is incomplete or duplicated")
    summaries, mechanisms = _summarize(rows)
    device_after = _wait_for_idle(args.device)
    report = {
        "schema": "nvidia_a800_staged_ldgsts_depth3_curve_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "three_stage_fp16_operand_panel_ldgsts_source_service",
        "scope": {
            "grid": GRID,
            "block_threads": 256,
            "resident_ctas_per_sm": 1,
            "pipeline_stages": 3,
            "wait_group": 1,
            "a_panel_bytes": 128 * 64 * 2,
            "b_panel_bytes": 64 * 256 * 2,
            "panel_bytes": PANEL_BYTES,
            "rounds": list(ROUNDS),
            "tiers": list(TIERS),
        },
        "summaries": summaries,
        "mechanism_parameters": mechanisms,
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "baseline_commit": "b93eb0f4e36b3c13997215c4b8bcdaeeafe38bae",
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                text=True, capture_output=True).stdout.strip(),
            "branch": subprocess.run(
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
    (output / "candidate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output / "candidate.json")


if __name__ == "__main__":
    main()
