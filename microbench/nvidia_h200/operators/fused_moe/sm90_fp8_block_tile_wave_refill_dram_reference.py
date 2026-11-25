#!/usr/bin/env python3
"""Derive the H200 FP8 block grouped-GEMM DRAM-fed multi-wave refill reference.

Input: accepted raw CSVs of sm90_fp8_block_tile_wave_refill_runner.py taken by
sm90_fp8_block_tile_wave_refill_k256_capture.sh (one run directory per tile
program, K in {256, 512, 1024}, W in {1, 2, 4, 8}), in two
operand modes: ``l2`` (operand streams wrapped modulo eight, the WP11 probe)
and ``dram`` (every stream group owns its operands and a 128-MiB polluter is
summed before each launch, so the weight panels stream from DRAM).  Output:
microbench/nvidia_h200/references/
fp8_block_grouped_gemm_tile_wave_refill_dram_sm90.json.

Per accepted process and mode: the median duration of every (W, K) cell and,
at each K, the least-squares slope of the median over W (the cost of one
additional resident wave).  DRAM mode, written as refill + K/128 * interval:
interval = (slope(K1024) - slope(K256)) / 6, refill = slope(K256) - 2 *
interval; the K512 slope is held out.  Forward uses the component-wise
maximum over processes.  The L2-mode K256 slope is reported against the WP11
L2-resident coefficients (outside their fitted K512..K2048 domain).  No
production operator or evaluation duration is read.
Usage: python sm90_fp8_block_tile_wave_refill_dram_reference.py \
    --run <program> <local copy of the run dir> <remote run dir> [--run ...] --git-commit <sha>
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).with_name("sm90_fp8_block_tile_wave_refill_runner.py")
CAPTURE = Path(__file__).with_name("sm90_fp8_block_tile_wave_refill_k256_capture.sh")
KERNEL_RUNNER = Path(__file__).with_name("sm90_fp8_block_tile_pipeline_runner.py")
L2_REFERENCE = (ROOT / "microbench/nvidia_h200/references/"
                "fp8_block_grouped_gemm_tile_wave_refill_sm90.json")
OUTPUT = (ROOT / "microbench/nvidia_h200/references/"
          "fp8_block_grouped_gemm_tile_wave_refill_dram_sm90.json")
WAVES = (1, 2, 4, 8)
K_VALUES = (256, 512, 1024)
TILE_K = 128


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _slope(xs, ys):
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    return (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            / sum((x - mean_x) ** 2 for x in xs))


def _parse(path):
    durations = {}
    clock = None
    with open(path, newline="") as handle:
        for fields in csv.reader(handle):
            if not fields:
                continue
            if fields[0] == "DATA" and fields[1] != "program":
                key = (int(fields[4]), int(fields[6]))
                if key not in durations:
                    durations[key] = []
                durations[key].append(float(fields[7]) * 1.0e6)
            elif fields[0] == "CLOCK" and fields[1] != "device":
                clock = {"device": int(fields[1]), "samples": int(fields[2]),
                         "busy_samples": int(fields[3]),
                         "window_min_mhz": float(fields[7]),
                         "accepted": fields[8] == "1"}
    if clock is None:
        raise ValueError(f"{path} has no CLOCK row")
    return durations, clock


def _captures(raw_dir, program, mode, remote_raw_dir):
    attempts = {}
    for path in sorted(raw_dir.glob(f"gpu*_{program}_{mode}_p*_a*.csv")):
        if path.name.endswith(".clock.csv"):
            continue
        process = path.stem.rsplit("_", 2)[1]
        attempt = int(path.stem.rsplit("_", 1)[1][1:])
        if process not in attempts:
            attempts[process] = []
        attempts[process].append((attempt, path))
    if not attempts:
        raise ValueError(f"no {mode} captures in {raw_dir}")
    captures, rejected = [], 0
    for process, rows in sorted(attempts.items()):
        rows.sort()
        rejected += len(rows) - 1
        attempt, path = rows[-1]
        durations, clock = _parse(path)
        if not clock["accepted"]:
            raise ValueError(f"last attempt {path} was not accepted")
        medians = {key: statistics.median(values)
                   for key, values in sorted(durations.items())}
        slopes = {k: _slope(WAVES, [medians[(w, k)] for w in WAVES])
                  for k in K_VALUES}
        captures.append({
            "path": str(path.relative_to(ROOT)),
            "remote_path": f"{remote_raw_dir}/{path.name}",
            "raw_sha256": _sha256(path), "process": process,
            "attempt": attempt, "clock": clock,
            "median_duration_ns": {f"W{w}_K{k}": v for (w, k), v in medians.items()},
            "per_wave_slope_ns": {f"K{k}": v for k, v in slopes.items()},
        })
    return captures, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=3, action="append", required=True,
                        metavar=("PROGRAM", "RAW_DIR", "REMOTE_RAW_DIR"),
                        help="one tile program's local and remote capture directories")
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    programs = {}
    for program, raw_dir, remote_raw_dir in args.run:
        programs[program] = _program(program, Path(raw_dir), remote_raw_dir)
    reference = {
        "schema": "nvidia_h200_fp8_block_grouped_gemm_tile_wave_refill_dram_reference_v2",
        "hardware_id": "nvidia_h200",
        "mechanism": "fp8_block_grouped_gemm_dram_fed_wave_refill",
        "production_or_evaluation_duration_consumed": False,
        "production_predictor_consumption_allowed": True,
        "timer_domain": "cuda_event_gpu_timeline",
        "waves": list(WAVES), "k_values": list(K_VALUES),
        "provenance": {
            "git_commit": args.git_commit,
            "runner": {"path": str(RUNNER.relative_to(ROOT)), "sha256": _sha256(RUNNER)},
            "kernel_runner": {"path": str(KERNEL_RUNNER.relative_to(ROOT)),
                              "sha256": _sha256(KERNEL_RUNNER)},
            "capture": {"path": str(CAPTURE.relative_to(ROOT)), "sha256": _sha256(CAPTURE)},
            "wp11_l2_reference_sha256": _sha256(L2_REFERENCE),
        },
        "programs": programs,
    }
    OUTPUT.write_text(json.dumps(reference, indent=1) + "\n")
    print(OUTPUT)
    for program, block in programs.items():
        print(program, json.dumps(block["dram"]["service_parameters"]))
        for c in block["dram"]["captures"]:
            print("  dram", c["process"], c["per_wave_slope_ns"], c["held_out_K512"])
        for c in block["l2"]["captures"]:
            print("  l2", c["process"], c["per_wave_slope_ns"], c["wp11_l2_coefficients_at_K256"])


def _program(program, raw_dir, remote_raw_dir):
    dram, dram_rejected = _captures(raw_dir, program, "dram", remote_raw_dir)
    for capture in dram:
        slope = {k: capture["per_wave_slope_ns"][f"K{k}"] for k in K_VALUES}
        interval = (slope[1024] - slope[256]) / ((1024 - 256) // TILE_K)
        refill = slope[256] - (256 // TILE_K) * interval
        predicted = refill + (512 // TILE_K) * interval
        capture.update({
            "dram_wave_k_round_interval_ns": interval,
            "dram_wave_refill_ns": refill,
            "held_out_K512": {"measured_slope_ns": slope[512],
                              "predicted_slope_ns": predicted,
                              "relative_error": predicted / slope[512] - 1.0}})
    l2, l2_rejected = _captures(raw_dir, program, "l2", remote_raw_dir)
    wp11 = json.loads(L2_REFERENCE.read_text())
    l2_row = wp11["programs"][program]["service_parameters"]
    for capture in l2:
        predicted = (l2_row["wave_refill_ns"]
                     + (256 // TILE_K) * l2_row["wave_k_round_interval_ns"])
        capture["wp11_l2_coefficients_at_K256"] = {
            "measured_slope_ns": capture["per_wave_slope_ns"]["K256"],
            "predicted_slope_ns": predicted,
            "relative_error": predicted / capture["per_wave_slope_ns"]["K256"] - 1.0}
    return {
        "remote_raw_dir": remote_raw_dir,
        "dram": {"captures": dram, "rejected_attempts": dram_rejected,
                 "service_parameters": {
                     "dram_wave_refill_ns": max(c["dram_wave_refill_ns"] for c in dram),
                     "dram_wave_k_round_interval_ns": max(
                         c["dram_wave_k_round_interval_ns"] for c in dram),
                     "derivation": (
                         "per accepted DRAM-mode process: least-squares slope of the "
                         "cell median over W=1,2,4,8 at K256 and K1024; interval = "
                         "(slope K1024 - slope K256) / 6, refill = slope K256 - 2 * "
                         "interval; component-wise maximum over processes; K512 held out")}},
        "l2": {"captures": l2, "rejected_attempts": l2_rejected},
    }


if __name__ == "__main__":
    main()
