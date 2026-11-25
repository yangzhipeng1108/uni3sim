#!/usr/bin/env python3
"""Derive the H200 FP8 block grouped-GEMM multi-wave refill reference.

Input: accepted raw CSVs of sm90_fp8_block_tile_wave_refill_runner.py (through
sm90_fp8_block_tile_wave_refill_capture.sh) in one run directory.  Output:
microbench/nvidia_h200/references/
fp8_block_grouped_gemm_tile_wave_refill_sm90.json.

Per accepted process: the median duration of every (W, K) cell; at K512 and
K2048 the least-squares slope of the median over W = 1, 2, 4, 8 is the cost of
one additional resident wave.  Writing that cost as refill + K/128 * interval,
interval = (slope(K2048) - slope(K512)) / 12 and refill = slope(K512) - 4 *
interval.  Forward uses the component-wise maximum over processes.  The K1024
slope is held out and compared with refill + 8 * interval.  No production
operator or evaluation duration is read.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).with_name("sm90_fp8_block_tile_wave_refill_runner.py")
CAPTURE = Path(__file__).with_name("sm90_fp8_block_tile_wave_refill_capture.sh")
KERNEL_RUNNER = Path(__file__).with_name("sm90_fp8_block_tile_pipeline_runner.py")
OUTPUT = (
    ROOT / "microbench/nvidia_h200/references/"
    "fp8_block_grouped_gemm_tile_wave_refill_sm90.json")
PROGRAMS = (
    "bm16_bn128_bk128_w4_s3", "bm16_bn128_bk128_w8_s4",
    "bm16_bn128_bk128_w4_s4", "bm16_bn128_bk128_w4_s5",
    "bm32_bn128_bk128_w4_s3", "bm64_bn128_bk128_w4_s4",
    "bm64_bn128_bk128_w4_s3",
)
WAVES = (1, 2, 4, 8)
TILE_K = 128


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _slope(xs, ys):
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    return (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            / sum((x - mean_x) ** 2 for x in xs))


def _parse(path):
    compile_rows = {}
    durations = {}
    clock = None
    with open(path, newline="") as handle:
        for fields in csv.reader(handle):
            if not fields:
                continue
            if fields[0] == "COMPILE" and fields[1] != "program":
                compile_rows[fields[2]] = (
                    int(fields[3]), int(fields[4]), int(fields[5]))
            elif fields[0] == "DATA" and fields[1] != "program":
                key = (int(fields[4]), int(fields[6]))
                durations.setdefault(key, []).append(float(fields[7]) * 1.0e6)
            elif fields[0] == "CLOCK" and fields[1] != "device":
                clock = {
                    "device": int(fields[1]), "samples": int(fields[2]),
                    "busy_samples": int(fields[3]),
                    "window_min_mhz": float(fields[7]),
                    "accepted": fields[8] == "1",
                }
    if clock is None:
        raise ValueError(f"{path} has no CLOCK row")
    return compile_rows, durations, clock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--remote-raw-dir", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    programs = {}
    for program in PROGRAMS:
        attempts = {}
        for path in sorted(args.raw_dir.glob(f"gpu*_{program}_p*_a*.csv")):
            if path.name.endswith(".clock.csv"):
                continue
            gpu, rest = path.stem.split("_", 1)
            process = rest.rsplit("_", 2)[1]
            attempt = int(rest.rsplit("_", 1)[1][1:])
            attempts.setdefault((gpu, process), []).append((attempt, path))
        captures = []
        rejected = 0
        for (gpu, process), rows in sorted(attempts.items()):
            rows.sort()
            rejected += len(rows) - 1
            attempt, path = rows[-1]
            compile_rows, durations, clock = _parse(path)
            if not clock["accepted"]:
                raise ValueError(f"last attempt {path} was not accepted")
            medians = {
                key: statistics.median(values)
                for key, values in sorted(durations.items())}
            slopes = {
                k: _slope(WAVES, [medians[(w, k)] for w in WAVES])
                for k in (512, 1024, 2048)}
            interval = (slopes[2048] - slopes[512]) / (
                (2048 - 512) // TILE_K)
            refill = slopes[512] - (512 // TILE_K) * interval
            predicted_1024 = refill + (1024 // TILE_K) * interval
            captures.append({
                "path": str(path.relative_to(ROOT)),
                "remote_path": f"{args.remote_raw_dir}/{path.name}",
                "raw_sha256": _sha256(path),
                "gpu": gpu, "process": process, "attempt": attempt,
                "clock": clock,
                "probe_compile": sorted(set(compile_rows.values())),
                "median_duration_ns": {
                    f"W{w}_K{k}": value for (w, k), value in medians.items()},
                "per_wave_slope_ns": {
                    f"K{k}": value for k, value in slopes.items()},
                "wave_k_round_interval_ns": interval,
                "wave_refill_ns": refill,
                "held_out_K1024": {
                    "measured_slope_ns": slopes[1024],
                    "predicted_slope_ns": predicted_1024,
                    "relative_error": predicted_1024 / slopes[1024] - 1.0,
                },
            })
        programs[program] = {
            "captures": captures,
            "rejected_attempts": rejected,
            "service_parameters": {
                "wave_refill_ns": max(
                    capture["wave_refill_ns"] for capture in captures),
                "wave_k_round_interval_ns": max(
                    capture["wave_k_round_interval_ns"]
                    for capture in captures),
                "derivation": (
                    "per accepted process: least-squares slope of the cell "
                    "median over W=1,2,4,8 at K512 and K2048; interval = "
                    "(slope K2048 - slope K512) / 12, refill = slope K512 - "
                    "4 * interval; component-wise maximum over processes"),
            },
        }
    reference = {
        "schema": "nvidia_h200_fp8_block_grouped_gemm_tile_wave_refill_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "software_revision": "vllm_0.24.0_triton_3.6_sm90",
        "source_baseline_commit": args.git_commit,
        "runner": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": _sha256(RUNNER),
        "kernel_source_runner": str(KERNEL_RUNNER.relative_to(ROOT)),
        "kernel_source_runner_sha256": _sha256(KERNEL_RUNNER),
        "capture_driver": str(CAPTURE.relative_to(ROOT)),
        "capture_driver_sha256": _sha256(CAPTURE),
        "derivation": str(Path(__file__).resolve().relative_to(ROOT)),
        "derivation_sha256": _sha256(Path(__file__).resolve()),
        "probe_contract": {
            "waves": list(WAVES),
            "k_values": [512, 1024, 2048],
            "held_out_k": 1024,
            "wrapped_operand_streams": 8,
            "n": 1024,
            "operands_l2_resident": True,
            "trial_count_per_cell": 9,
            "timer_domain": "cuda_event_gpu_timeline",
            "host_launch_cover": "torch.cuda._sleep(400000) before start event",
        },
        "clock_gate": (
            "nvidia-smi clocks.sm every 50 ms over the timed section of each "
            "process; accepted only if every sample reads >= 1980 MHz"),
        "programs": programs,
        "decision": {
            "status": "accepted_independent_pipeline_mechanism",
            "forward_eligible": True,
            "allowed_use": (
                "the cost of each resident wave after the first of the same "
                "FP8 block tile program, as the local lower bound of that "
                "wave beside its DRAM round service"),
        },
        "production_or_evaluation_duration_consumed": False,
    }
    OUTPUT.write_text(json.dumps(reference, indent=1) + "\n")
    for program, entry in programs.items():
        held_out = [capture["held_out_K1024"]["relative_error"]
                    for capture in entry["captures"]]
        print(program,
              round(entry["service_parameters"]["wave_refill_ns"], 1),
              round(entry["service_parameters"]["wave_k_round_interval_ns"], 1),
              "held-out K1024 rel err", [round(value, 3) for value in held_out],
              "rejected", entry["rejected_attempts"])


if __name__ == "__main__":
    main()
