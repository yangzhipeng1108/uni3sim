#!/usr/bin/env python3
"""Derive the H200 BM16 sparse second-wave reference from sparse-W1 captures.

Scripted form of the A800 derivation recorded in
``bm16_grouped_gemm_sparse_second_wave_sm80.json``:
  - one-wave midpoint: mean of the per-capture medians of the same source
    (4 tokens, E128, top-8, K2048) with 16 active experts at the two
    adjacent widths N1504/N1568 (inferred from the A800 reference: its
    geometry is 32 maximum M tiles, 16/31 active M tiles, 47/49 N tiles);
  - two-wave midpoint: the same with 31 active experts;
  - pooled values are the mean over the two accepted captures;
  - "one additional fixed span is a conservative lower bound" holds when
    two_wave - (2 * one_wave - fixed_span) >= fixed_span.
H200 coupling: the W1 resident capacity is 132 SMs x 12 CTAs = 1584 (H200
Triton 3.6 W1 compile facts), so the A800 1504/1568-CTA pair is one resident
wave on H200.  The H200-scaled N1824/N1952 pair (midpoint grid 1888, 31 x 59
= 1829 active CTAs > 1584) is evaluated with the same rule to show whether
the second-wave refill still exists when a launch really has two waves.
Captures: first two clock-accepted processes (every 50-ms SM-clock sample of
the timed section >= 1980 MHz) on the owned H200 GPU.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).resolve().with_name(
    "sm80_bf16_bm16_sparse_w1_source_runner.py")
REPORT = Path(__file__).resolve()
BOOST_MHZ = 1980
TILE_N = 32
W1_RESIDENT_CTAS_PER_SM = 12
SM_COUNT = 132
MAXIMUM_M_TILES = 32
K_TILES = 2048 // 64
A800_PAIR = (1504, 1568)
H200_SCALED_PAIR = (1824, 1952)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clock(path):
    clocks = [int(line.split(",")[0]) for line in
              Path(path).read_text().splitlines() if line.strip()]
    return {"samples": len(clocks), "minimum_mhz": min(clocks),
            "maximum_mhz": max(clocks),
            "accepted": bool(clocks) and min(clocks) >= BOOST_MHZ}


def _summaries(raw):
    rows = {}
    for line in Path(raw).read_text().splitlines():
        fields = line.split(",")
        if fields[0] == "summary":
            tokens, experts, topk, active, k, n = (
                int(value) for value in fields[1:7])
            rows[(tokens, experts, topk, active, k, n)] = float(fields[10])
    return rows


def _midpoint(rows, active, pair):
    return statistics.mean(
        rows[(4, 128, 8, active, 2048, n)] for n in pair)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--pipeline-reference", type=Path, required=True)
    arguments = parser.parse_args()
    fixed_span = json.loads(arguments.pipeline_reference.read_text())[
        "service_parameters"]["fixed_span_ns"]
    captures = []
    for index in (1, 2, 3):
        raw = arguments.run_dir / f"p{index}.csv"
        clock = _clock(arguments.run_dir / f"p{index}.clock.csv")
        captures.append({"process": index, "raw": raw, "clock": clock,
                         "rows": _summaries(raw)})
    accepted = [c for c in captures if c["clock"]["accepted"]][:2]
    if len(accepted) != 2:
        raise ValueError("fewer than two accepted captures")
    meta = [line for line in accepted[0]["raw"].read_text().splitlines()
            if line.startswith("meta,vllm_source_sha256,")]

    def pair_measurement(pair):
        one = {str(c["process"]): _midpoint(c["rows"], 16, pair)
               for c in accepted}
        two = {str(c["process"]): _midpoint(c["rows"], 31, pair)
               for c in accepted}
        pooled_one = statistics.mean(one.values())
        pooled_two = statistics.mean(two.values())
        grid = MAXIMUM_M_TILES * (sum(pair) // 2 // TILE_N)
        return {
            "one_wave_midpoint_ns_by_device": one,
            "two_wave_midpoint_ns_by_device": two,
            "pooled_one_wave_midpoint_ns": pooled_one,
            "pooled_two_wave_midpoint_ns": pooled_two,
            "existing_fixed_span_ns": fixed_span,
            "one_additional_fixed_span_is_a_conservative_lower_bound": (
                pooled_two - (2 * pooled_one - fixed_span) >= fixed_span),
            "midpoint_grid_ctas": grid,
            "midpoint_active_ctas_31_active": 31 * (grid // MAXIMUM_M_TILES),
            "h200_w1_resident_waves_at_midpoint_grid": math.ceil(
                grid / (SM_COUNT * W1_RESIDENT_CTAS_PER_SM)),
        }

    a800_geometry = pair_measurement(A800_PAIR)
    scaled = pair_measurement(H200_SCALED_PAIR)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True).stdout.strip()
    reference = {
        "schema_version": 1,
        "hardware_id": "nvidia_h200",
        "software_revision": "vllm_0.24.0_triton_3.6_sm90",
        "mechanism": "bm16_grouped_gemm_sparse_second_wave_fill_v1",
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "forward_eligible": True,
        "forward_eligibility_constraint": (
            "On H200 the A800 refill geometry (1488 active / 1536 launched "
            "CTAs, K32) is one resident W1 wave (132 SMs x 12 CTAs = 1584), "
            "so sparse_refill_wave_count = 1 there and the refill rule adds "
            "no fill.  A second fixed span applies only to a two-resident-"
            "wave BF16 BM16/N32/K64 grouped-GEMM source launch with K32 and "
            "a positive early-return tail shorter than one 132-SM round; the "
            "H200-scaled N1824/N1952 pair is the measured two-wave case."),
        "source_geometry": {
            "dtype": "bf16", "tile_m": 16, "tile_n": TILE_N, "tile_k": 64,
            "group_size_m": 1, "threads_per_cta": 128,
            "k_tiles": K_TILES, "maximum_m_tiles": MAXIMUM_M_TILES,
            "active_m_tiles": [16, 31],
            "n_tiles": [n // TILE_N for n in A800_PAIR],
            "cache_polluter_bytes": 134217728,
        },
        "measurement": {
            "physical_devices": [1],
            "trials_per_cell": 9,
            **{key: value for key, value in a800_geometry.items()
               if key in (
                   "one_wave_midpoint_ns_by_device",
                   "two_wave_midpoint_ns_by_device",
                   "pooled_one_wave_midpoint_ns",
                   "pooled_two_wave_midpoint_ns",
                   "existing_fixed_span_ns",
                   "one_additional_fixed_span_is_a_conservative_lower_bound")},
        },
        "capture": {
            "baseline_commit": commit,
            "runner_path": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": _sha256(RUNNER),
            "reproduction_runner_sha256": _sha256(REPORT),
            "vllm_source_sha256": meta[0].split(",")[2],
            "raw_paths": [str(c["raw"].resolve().relative_to(ROOT))
                          for c in accepted],
            "raw_sha256": [_sha256(c["raw"]) for c in accepted],
        },
        "h200_capture_provenance": {
            "git_commit": commit,
            "report": str(REPORT.relative_to(ROOT)),
            "remote_raw_dir": arguments.remote_run_dir,
            "device_index": 1,
            "measurement_keys_by_device_mean": (
                "the *_by_device maps are keyed by accepted process index "
                "on GPU 1 (A800: physical device index)"),
            "clock_gate": (
                "every 50-ms SM-clock sample of the timed section >= 1980 "
                "MHz; first two accepted processes used"),
            "process_clocks": {str(c["process"]): c["clock"]
                               for c in captures},
            "accepted_captures": sum(c["clock"]["accepted"]
                                     for c in captures),
            "rejected_captures": sum(not c["clock"]["accepted"]
                                     for c in captures),
            "a800_geometry_on_h200": a800_geometry,
            "h200_scaled_two_wave_pair": {
                "n_widths": list(H200_SCALED_PAIR), **scaled},
            "w1_resident_ctas_per_sm": W1_RESIDENT_CTAS_PER_SM,
            "one_wave_case_selection_note": (
                "inferred from the A800 reference geometry: the one-wave "
                "midpoint is the 16-active case of the same 4-token E128 "
                "top-8 K2048 source at the same N pair"),
        },
    }
    output = ROOT / (
        "microbench/nvidia_h200/references/"
        "bm16_grouped_gemm_sparse_second_wave_sm90.json")
    output.write_text(json.dumps(reference, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"a800_geometry": a800_geometry,
                      "h200_scaled_pair": scaled}, indent=2))


if __name__ == "__main__":
    main()
