#!/usr/bin/env python3
"""Derive the H200 SGLang 0.5.12 FP8 block grouped-GEMM tile-pipeline reference.

Inputs (local copies of the H200 run directories; each option repeatable):
  --tile-dir   sm90_fp8_block_tile_pipeline_sglang_capture.sh output
  --wave-dir   sm90_fp8_block_tile_wave_refill_sglang_capture.sh output
  --compile    sglang_fused_moe_compile_facts.py output (production compile
               facts of SGLang's own fused_moe_kernel; no timing)
Output: microbench/nvidia_h200/references/
sglang_fp8_block_grouped_gemm_tile_pipeline_sm90.json.

Rules are those of the vLLM references, with K rounds counted in each
program's BLOCK_SIZE_K (64 or 128):
* sub-wave pipeline (sm90_fp8_block_tile_pipeline_reference.py): per accepted
  process and grid, interval = (median K2048 - median K512) / rounds and
  fixed span = median K512 - rounds(K512) * interval; component-wise maximum;
* L2 multi-wave refill (sm90_fp8_block_tile_wave_refill_reference.py): the
  least-squares slope over W = 1, 2, 4, 8 at K512 and K2048 is written as
  refill + rounds * interval; K1024 held out; component-wise maximum;
* DRAM-fed wave (sm90_fp8_block_tile_wave_refill_dram_reference.py, TMA W2
  programs only): the same from K256 and K1024; K512 held out.
Production residency per SM = min(register, shared-memory, thread limits) of
the production compile facts (65536 registers per SM in 256-register warp
units, 228 KiB shared memory per SM with 1 KiB reserved per CTA, 2048
threads).  Only the last attempt of each (program, mode, process) is used and
it must be clock-accepted.  No production operator or evaluation duration is
read.
Usage: sglang_fp8_block_tile_pipeline_reference.py --tile-dir D --wave-dir D --compile F --git-commit SHA
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUTPUT = (ROOT / "microbench/nvidia_h200/references/"
          "sglang_fp8_block_grouped_gemm_tile_pipeline_sm90.json")
SOURCES = (
    "sm90_fp8_block_tile_pipeline_sglang_runner.py",
    "sm90_fp8_block_tile_pipeline_sglang_capture.sh",
    "sm90_fp8_block_tile_wave_refill_sglang_runner.py",
    "sm90_fp8_block_tile_wave_refill_sglang_capture.sh",
    "sglang_fused_moe_compile_facts.py",
)
# program -> (tile_m, tile_n, tile_k, warps, stages, use_tma, projection)
PROGRAMS = {
    "sgl_bm16_bn64_bk128_w4_s5": (16, 64, 128, 4, 5, False, "w1"),
    "sgl_bm16_bn64_bk128_w4_s3": (16, 64, 128, 4, 3, False, "w1"),
    "sgl_bm16_bn64_bk128_w4_s4": (16, 64, 128, 4, 4, False, "w1"),
    "sgl_bm16_bn128_bk128_w4_s3": (16, 128, 128, 4, 3, False, "w1"),
    "sgl_bm16_bn128_bk128_w4_s4": (16, 128, 128, 4, 4, False, "w1"),
    "sgl_tma_bm16_bn128_bk128_w4_s2": (16, 128, 128, 4, 2, True, "w2"),
    "sgl_tma_bm16_bn128_bk64_w4_s3": (16, 128, 64, 4, 3, True, "w2"),
    "sgl_tma_bm16_bn256_bk64_w4_s3": (16, 256, 64, 4, 3, True, "w2"),
    "sgl_bm32_bn128_bk128_w4_s5": (32, 128, 128, 4, 5, False, "w1"),
    "sgl_bm64_bn128_bk128_w4_s4": (64, 128, 128, 4, 4, False, "w1"),
    "sgl_tma_bm32_bn128_bk128_w4_s2": (32, 128, 128, 4, 2, True, "w2"),
    "sgl_tma_bm64_bn128_bk128_w4_s3": (64, 128, 128, 4, 3, True, "w2"),
}
TILE_GRIDS = (1024, 2048)
WAVES = (1, 2, 4, 8)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _last_attempts(directory, pattern):
    """(key) -> path of the last attempt; every earlier attempt is a reject."""
    last = {}
    for path in sorted(Path(directory).glob("gpu*_a*.csv")):
        if path.name.endswith(".clock.csv"):
            continue
        match = re.fullmatch(pattern, path.name)
        if match is None:
            continue
        key = match.group("key")
        attempt = int(match.group("attempt"))
        if key not in last or attempt > last[key][0]:
            last[key] = (attempt, path)
    return {key: path for key, (_, path) in last.items()}


def _clock_accepted(rows):
    clocks = [row for row in rows if row[0] == "CLOCK" and row[1].isdigit()]
    if len(clocks) != 1:
        raise ValueError("capture must carry exactly one clock row")
    return clocks[0][-1] == "1"


def _slope(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def _tile(path, tile_k):
    rows = [row for row in csv.reader(open(path)) if row]
    if not _clock_accepted(rows):
        raise ValueError(f"{path} is not clock-accepted")
    durations = {}
    for row in rows:
        if row[0] == "DATA" and row[1] == "timed":
            if row[4] not in durations:
                durations[row[4]] = []
            durations[row[4]].append(float(row[8]) * 1e6)
    per_grid = {}
    for n in TILE_GRIDS:
        short = statistics.median(durations[f"k512_n{n}"])
        long = statistics.median(durations[f"k2048_n{n}"])
        interval = (long - short) / ((2048 - 512) // tile_k)
        per_grid[f"n{n}"] = {"local_k_round_interval_ns": interval,
                             "fixed_span_ns": short - (512 // tile_k) * interval}
    compile_rows = {tuple(row[3:]) for row in rows
                    if row[0] == "COMPILE" and row[1] != "program"}
    return per_grid, compile_rows


def _wave(path, tile_k, fit, holdout):
    rows = [row for row in csv.reader(open(path)) if row]
    if not _clock_accepted(rows):
        raise ValueError(f"{path} is not clock-accepted")
    durations = {}
    for row in rows:
        if row[0] == "DATA" and row[1] != "program":
            key = (int(row[4]), int(row[6]))
            if key not in durations:
                durations[key] = []
            durations[key].append(float(row[7]) * 1e6)
    slopes = {k: _slope(WAVES, [statistics.median(durations[(w, k)])
                                for w in WAVES])
              for k in (*fit, holdout)}
    low, high = fit
    interval = (slopes[high] - slopes[low]) / ((high - low) // tile_k)
    refill = slopes[low] - (low // tile_k) * interval
    predicted = refill + (holdout // tile_k) * interval
    return {"per_wave_slope_ns": {f"K{k}": v for k, v in slopes.items()},
            "wave_refill_ns": refill, "wave_k_round_interval_ns": interval,
            "holdout": {"k": holdout, "measured_slope_ns": slopes[holdout],
                        "predicted_slope_ns": predicted,
                        "relative_error": predicted / slopes[holdout] - 1.0}}


def _resident(registers, shared, warps):
    threads = 32 * warps
    per_warp = -(-registers * 32 // 256) * 256
    return min(65536 // (per_warp * warps), 233472 // (shared + 1024),
               2048 // threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, action="append",
                        required=True)
    parser.add_argument("--wave-dir", type=Path, action="append",
                        required=True)
    parser.add_argument("--compile", type=Path, action="append",
                        required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    documents = [json.loads(path.read_text()) for path in args.compile]
    if any(document["timing_recorded"] is not False
           for document in documents):
        raise ValueError("compile facts must carry no timing")
    facts = documents[0]
    production = {}
    for fact in (fact for document in documents
                 for fact in document["specialisations"]):
        # A program recompiles per constexpr (GROUP_SIZE_M, token count
        # class); its specialisations must agree on registers and warps.
        key = (fact["num_stages"], fact["shared_memory_bytes"])
        if key in production and (
                (production[key]["registers_per_thread"],
                 production[key]["num_warps"])
                != (fact["registers_per_thread"], fact["num_warps"])):
            raise ValueError(f"production specialisations of {key} disagree")
        production[key] = fact
    tile_runs, wave_runs = {}, {}
    for directory in args.tile_dir:
        found = _last_attempts(
            directory, r"gpu\d+_(?P<key>.+_p\d)_a(?P<attempt>\d+)\.csv")
        if set(found) & set(tile_runs):
            raise ValueError(f"{directory} repeats a tile capture")
        tile_runs.update(found)
    for directory in args.wave_dir:
        found = _last_attempts(
            directory,
            r"gpu\d+_(?P<key>.+_(?:l2|dram)_p\d)_a(?P<attempt>\d+)\.csv")
        if set(found) & set(wave_runs):
            raise ValueError(f"{directory} repeats a wave capture")
        wave_runs.update(found)
    programs = {}
    for program, (tile_m, tile_n, tile_k, warps, stages, use_tma,
                  projection) in PROGRAMS.items():
        captures = []
        probe_compile = set()
        for process in ("p1", "p2"):
            path = tile_runs[f"{program}_{process}"]
            per_grid, compile_rows = _tile(path, tile_k)
            probe_compile |= compile_rows
            captures.append({"path": str(path.name), "raw_sha256": _sha256(path),
                             "per_grid": per_grid})
        if len(probe_compile) != 1:
            raise ValueError(f"{program} probe cells compiled differently")
        (probe_registers, _, probe_shared, _, _), = (
            tuple(int(v) for v in row) for row in probe_compile)
        match = production[(stages, probe_shared)]
        waves = []
        for process in ("p1", "p2"):
            path = wave_runs[f"{program}_l2_{process}"]
            entry = _wave(path, tile_k, (512, 2048), 1024)
            entry.update(path=str(path.name), raw_sha256=_sha256(path))
            waves.append(entry)
        dram = []
        if use_tma:
            for process in ("p1", "p2"):
                path = wave_runs[f"{program}_dram_{process}"]
                entry = _wave(path, tile_k, (256, 1024), 512)
                entry.update(path=str(path.name), raw_sha256=_sha256(path))
                dram.append(entry)
        grids = [grid for capture in captures
                 for grid in capture["per_grid"].values()]
        service = {
            "fixed_span_ns": max(g["fixed_span_ns"] for g in grids),
            "local_k_round_interval_ns": max(
                g["local_k_round_interval_ns"] for g in grids),
            "wave_refill_ns": max(w["wave_refill_ns"] for w in waves),
            "wave_k_round_interval_ns": max(
                w["wave_k_round_interval_ns"] for w in waves),
        }
        if dram:
            service["dram_wave_refill_ns"] = max(
                w["wave_refill_ns"] for w in dram)
            service["dram_wave_k_round_interval_ns"] = max(
                w["wave_k_round_interval_ns"] for w in dram)
        programs[program] = {
            "source_and_launch_contract": {
                "dtype": "fp8_e4m3_w8a8", "weight_block_shape": [128, 128],
                "tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k,
                "num_warps": warps, "num_stages": stages,
                "threads_per_cta": 32 * warps,
                "use_tma": use_tma, "swap_ab": tile_m < 64 and tile_n >= 64,
                "projection": projection,
            },
            "probe_compile": {"registers_per_thread": probe_registers,
                              "shared_memory_bytes": probe_shared},
            "production_compile": {
                "registers_per_thread": match["registers_per_thread"],
                "shared_memory_bytes": match["shared_memory_bytes"],
                "num_stages": match["num_stages"],
                "resident_ctas_per_sm": _resident(
                    match["registers_per_thread"],
                    match["shared_memory_bytes"], warps),
            },
            "probe_shared_memory_matches_production": (
                probe_shared == match["shared_memory_bytes"]),
            "captures": captures, "wave_captures": waves,
            "dram_wave_captures": dram,
            "service_parameters": service,
        }
    reference = {
        "schema": "nvidia_h200_sglang_fp8_block_grouped_gemm_tile_pipeline_reference_v1",
        "hardware_id": "nvidia_h200",
        "software_revision": "sglang_0.5.12_triton_3.6_sm90",
        "environment": {"python": 'python',
                        "triton": facts["triton"], "torch": facts["torch"]},
        "production_configs": (
            "sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/"
            "E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,"
            "block_shape=[128, 128].json and _down.json (SGLang logs the "
            "triton_3_6_0 -> triton_3_5_1 fallback)"),
        "source_baseline_commit": args.git_commit,
        "sources": {name: _sha256(HERE / name) for name in SOURCES},
        "derivation_sha256": _sha256(Path(__file__).resolve()),
        "raw_dirs": {"tile": [d.name for d in args.tile_dir],
                     "wave": [d.name for d in args.wave_dir],
                     "compile": [p.parent.name for p in args.compile]},
        "compile_facts_sha256": [_sha256(p) for p in args.compile],
        "clock_gate": ("nvidia-smi clocks.sm every 50 ms over the timed "
                       "section of each process; accepted only if every "
                       "sample reads >= 1980 MHz"),
        "programs": programs,
        "production_or_evaluation_duration_consumed": False,
        "production_predictor_consumption_allowed": True,
    }
    OUTPUT.write_text(json.dumps(reference, indent=1) + "\n")
    for program, entry in programs.items():
        s = entry["service_parameters"]
        print(program, {k: round(v, 1) for k, v in s.items()},
              entry["production_compile"]["resident_ctas_per_sm"],
              entry["probe_shared_memory_matches_production"],
              [round(w["holdout"]["relative_error"], 3)
               for w in entry["wave_captures"] + entry["dram_wave_captures"]])


if __name__ == "__main__":
    main()
