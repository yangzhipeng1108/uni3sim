#!/usr/bin/env python3
"""Derive the H200 FP8 block grouped-GEMM tile-pipeline reference.

Input: the accepted raw CSVs written by sm90_fp8_block_tile_pipeline_runner.py
(through sm90_fp8_block_tile_pipeline_capture.sh) in one run directory.
Output: microbench/nvidia_h200/references/
fp8_block_grouped_gemm_tile_pipeline_sm90.json.

Derivation (the rule of the H200 BF16 BM16 reference
bm16_grouped_gemm_pipeline_sm90.json, unchanged): for each accepted process and
fixed grid, the K-round interval is (median(K=2048) - median(K=512)) /
(16 - 4) complete BK128 rounds and the fixed span is median(K=512) - 4 *
interval; Forward uses the component-wise maximum over processes and grids.
Only the most recent attempt of each (GPU, program, process) is used, and only
when its clock row is accepted.  No production operator or evaluation
duration is read.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).with_name("sm90_fp8_block_tile_pipeline_runner.py")
CAPTURE = Path(__file__).with_name("sm90_fp8_block_tile_pipeline_capture.sh")
TUNED_REFERENCE = (
    ROOT / "microbench/nvidia_h200/references/"
    "fused_moe_tuned_config_vllm_0_24_0_h200.json")
OUTPUT = (
    ROOT / "microbench/nvidia_h200/references/"
    "fp8_block_grouped_gemm_tile_pipeline_sm90.json")
EMPTY_TAIL_REFERENCE = (
    ROOT / "microbench/nvidia_h200/references/"
    "bm16_grouped_gemm_empty_tail_sm90.json")
PRODUCTION_SOURCE_SHA256 = (
    "57a3f7efc9624f5c437908892ed5109a81ba0dbf45e1c6ac8a1438a23131f207")
def _program(tile_m, num_warps, num_stages, active_rows, config_ids):
    return {
        "tile_m": tile_m, "tile_n": 128, "tile_k": 128,
        "num_warps": num_warps, "num_stages": num_stages,
        "active_rows": active_rows, "production_config_ids": config_ids,
    }


PROGRAMS = {
    "bm16_bn128_bk128_w4_s3": _program(16, 4, 3, 1, [
        "cb47dc9924056440", "0e0b351e3086258f", "96f97a9b394c141b",
        "63d65789df08a862", "50e8c1da81dc66d8", "65c0c8e5705f60f2"]),
    "bm16_bn128_bk128_w8_s4": _program(16, 8, 4, 1, ["32e1b6206e8cd459"]),
    "bm16_bn128_bk128_w4_s4": _program(16, 4, 4, 1, [
        "c707ac95391bd157", "12d9d4bdf00fb2ff", "35591cb2c794b91b"]),
    "bm16_bn128_bk128_w4_s5": _program(16, 4, 5, 1, ["b336fde8d25c71c0"]),
    "bm32_bn128_bk128_w4_s3": _program(32, 4, 3, 32, ["a2b5ff407c7333d6"]),
    "bm64_bn128_bk128_w4_s4": _program(64, 4, 4, 64, [
        "57bc196faf4fc400", "d26bcbdc1237b406", "bc3c0c641b7ec1b3"]),
    "bm64_bn128_bk128_w4_s3": _program(64, 4, 3, 64, ["12974297821ec13c"]),
}
K_SHORT, K_LONG = 512, 2048
GRIDS = {1024: 64, 2048: 128}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse(path):
    compile_rows = {}
    durations = {}
    clock = None
    with open(path, newline="") as handle:
        for fields in csv.reader(handle):
            if not fields:
                continue
            if fields[0] == "COMPILE" and fields[1] != "program":
                compile_rows[fields[2]] = {
                    "registers_per_thread": int(fields[3]),
                    "spilled_registers": int(fields[4]),
                    "shared_memory_bytes": int(fields[5]),
                    "num_warps": int(fields[6]),
                    "num_stages": int(fields[7]),
                }
            elif fields[0] == "DATA" and fields[1] == "timed":
                durations.setdefault(fields[4], []).append(
                    float(fields[8]) * 1.0e6)
            elif fields[0] == "CLOCK" and fields[1] != "device":
                clock = {
                    "device": int(fields[1]),
                    "samples": int(fields[2]),
                    "busy_samples": int(fields[3]),
                    "busy_min_mhz": float(fields[4]),
                    "busy_median_mhz": float(fields[5]),
                    "busy_max_mhz": float(fields[6]),
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
    tuned = json.loads(TUNED_REFERENCE.read_text())
    compile_by_config = {
        entry["config_id"]: entry for entry in tuned["compile_metadata"]}
    empty_tail = json.loads(EMPTY_TAIL_REFERENCE.read_text())
    programs = {}
    for program, contract in PROGRAMS.items():
        attempts = {}
        for path in sorted(args.raw_dir.glob(f"gpu*_{program}_p*_a*.csv")):
            if path.name.endswith(".clock.csv"):
                continue
            gpu, rest = path.stem.split("_", 1)
            process = rest.rsplit("_", 2)[1]
            attempt = int(rest.rsplit("_", 1)[1][1:])
            key = (gpu, process)
            attempts.setdefault(key, []).append((attempt, path))
        captures = []
        rejected = 0
        for (gpu, process), rows in sorted(attempts.items()):
            rows.sort()
            for attempt, path in rows[:-1]:
                rejected += 1
            attempt, path = rows[-1]
            compile_rows, durations, clock = _parse(path)
            if not clock["accepted"]:
                raise ValueError(f"last attempt {path} was not accepted")
            medians = {
                cell: statistics.median(values)
                for cell, values in sorted(durations.items())}
            per_grid = {}
            for n, grid in GRIDS.items():
                short = medians[f"k{K_SHORT}_n{n}"]
                long = medians[f"k{K_LONG}_n{n}"]
                rounds_short = K_SHORT // contract["tile_k"]
                rounds_long = K_LONG // contract["tile_k"]
                interval = (long - short) / (rounds_long - rounds_short)
                per_grid[f"grid{grid}"] = {
                    "local_k_round_interval_ns": interval,
                    "fixed_span_ns": short - rounds_short * interval,
                }
            compile_facts = set(
                json.dumps(value, sort_keys=True)
                for value in compile_rows.values())
            if len(compile_facts) != 1:
                raise ValueError(f"{path} compiled cells disagree")
            captures.append({
                "path": str(path.relative_to(ROOT)),
                "remote_path": f"{args.remote_raw_dir}/{path.name}",
                "raw_sha256": _sha256(path),
                "gpu": gpu,
                "process": process,
                "attempt": attempt,
                "clock": clock,
                "probe_compile": json.loads(compile_facts.pop()),
                "median_duration_ns": medians,
                "per_grid": per_grid,
            })
        interval = max(
            grid["local_k_round_interval_ns"]
            for capture in captures for grid in capture["per_grid"].values())
        fixed = max(
            grid["fixed_span_ns"]
            for capture in captures for grid in capture["per_grid"].values())
        production = {}
        for config_id in contract["production_config_ids"]:
            entry = compile_by_config[config_id]
            kernel = entry["kernel_config"]
            if ((kernel["BLOCK_SIZE_M"], kernel["BLOCK_SIZE_N"],
                 kernel["BLOCK_SIZE_K"], kernel["num_warps"],
                 kernel["num_stages"])
                    != (contract["tile_m"], contract["tile_n"],
                        contract["tile_k"], contract["num_warps"],
                        contract["num_stages"])):
                raise ValueError(f"{config_id} is not program {program}")
            production[config_id] = {
                variant: {
                    "registers_per_thread": stage["registers_per_thread"],
                    "shared_memory_bytes_per_cta":
                        stage["shared_memory_bytes_per_cta"],
                    "resident_ctas_per_sm": stage["resident_ctas_per_sm"],
                }
                for variant, stage in entry["stages"].items()}
        shared = {
            capture["probe_compile"]["shared_memory_bytes"]
            for capture in captures}
        production_shared = {
            stage["shared_memory_bytes_per_cta"]
            for config in production.values() for stage in config.values()}
        programs[program] = {
            "source_and_launch_contract": {
                "dtype": "fp8_e4m3_w8a8",
                "weight_block_shape": [128, 128],
                "tile_m": contract["tile_m"],
                "tile_n": contract["tile_n"],
                "tile_k": contract["tile_k"],
                "num_warps": contract["num_warps"],
                "num_stages": contract["num_stages"],
                "threads_per_cta": 32 * contract["num_warps"],
                "pipeline_depth": contract["num_stages"] - 1,
                "pipeline_depth_rule": (
                    "Triton software pipelining with num_stages S keeps S-1 "
                    "operand stages in flight; the production shared-memory "
                    "footprint equals (S-1) A+B stages plus scratch"),
                "physical_sm_count": 132,
            },
            "probe_contract": {
                "streams": 8,
                "active_rows_per_tile": contract["active_rows"],
                "cells": [f"K{k}_N{n}_grid{grid}"
                          for n, grid in GRIDS.items()
                          for k in (K_SHORT, K_LONG)],
                "trial_count_per_cell": 9,
                "cache_polluter_bytes": 134217728,
                "timer_domain": "cuda_event_gpu_timeline",
            },
            "production_compile_metadata": production,
            "probe_shared_memory_matches_production": shared <= production_shared,
            "captures": captures,
            "rejected_attempts": rejected,
            "service_parameters": {
                "fixed_span_ns": fixed,
                "local_k_round_interval_ns": interval,
                "derivation": (
                    "per accepted process and grid: interval = "
                    "(median K2048 - median K512) / 12 BK128 rounds, fixed = "
                    "median K512 - 4 * interval; component-wise maximum over "
                    "processes and grids"),
            },
        }
    reference = {
        "schema": "nvidia_h200_fp8_block_grouped_gemm_tile_pipeline_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "software_revision": "vllm_0.24.0_triton_3.6_sm90",
        "environment": {
            "python": 'python',
            "torch": "2.11.0+cu129",
            "triton": "3.6.0",
            "driver": "575.57.08",
        },
        "source_baseline_commit": args.git_commit,
        "runner": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": _sha256(RUNNER),
        "capture_driver": str(CAPTURE.relative_to(ROOT)),
        "capture_driver_sha256": _sha256(CAPTURE),
        "derivation": str(Path(__file__).resolve().relative_to(ROOT)),
        "derivation_sha256": _sha256(Path(__file__).resolve()),
        "production_source": (
            'vllm/model_executor/layers/fused_moe/fused_moe.py'),
        "production_source_sha256": PRODUCTION_SOURCE_SHA256,
        "tuned_config_reference": str(TUNED_REFERENCE.relative_to(ROOT)),
        "tuned_config_reference_sha256": _sha256(TUNED_REFERENCE),
        "clock_gate": (
            "nvidia-smi clocks.sm every 50 ms over the timed section of each "
            "process; accepted only if every sample reads >= 1980 MHz"),
        "programs": programs,
        "empty_tail": {
            "reused_reference": str(EMPTY_TAIL_REFERENCE.relative_to(ROOT)),
            "reused_reference_sha256": _sha256(EMPTY_TAIL_REFERENCE),
            "empty_tail_round_ns":
                empty_tail["service_parameters"]["empty_tail_round_ns"],
            "reuse_basis": (
                "the vLLM 0.24 FP8 kernel keeps the source early-return "
                "`if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded: return` "
                "before any operand load, launched with the same 128 threads "
                "per CTA as the zero-work probe; only its per-132-CTA retire "
                "cost is reused"),
        },
        "decision": {
            "status": "accepted_independent_pipeline_mechanism",
            "forward_eligible": True,
            "reason": (
                "The synthetic sparse-row program reproduces the vLLM 0.24 "
                "FP8 block-scaled K loop (FP8 A/B tiles, per-K A and B scale "
                "loads, dot times scales) without routing, sorting, gather, "
                "reduction or production timing; the procedure is the H200 "
                "BF16 BM16 pipeline reference's."),
        },
        "excluded_conclusions": [
            "a complete FusedMoE operator duration",
            "a production-shape latency lookup",
            "a residual correction",
            "tile programs other than the two recorded here",
        ],
        "production_or_evaluation_duration_consumed": False,
    }
    OUTPUT.write_text(json.dumps(reference, indent=1) + "\n")
    for program, entry in programs.items():
        print(program, entry["service_parameters"]["fixed_span_ns"],
              entry["service_parameters"]["local_k_round_interval_ns"],
              entry["probe_shared_memory_matches_production"],
              entry["rejected_attempts"])


if __name__ == "__main__":
    main()
