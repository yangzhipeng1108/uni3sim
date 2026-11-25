#!/usr/bin/env python3
"""Capture and publish the H200 three-stage LDGSTS service curve.

Owner copy of microbench/nvidia_a800/memory/l2/staged_ldgsts_depth3_curve_runner.py.
The round set, stability gate, per-cell medians, 256-to-1024 steady secant,
finite residual derivation and the "minimum 1024-round capacity across two
captures" selection are unchanged.  Hardware coupling: grid 132 (one CTA per
H200 SM), H200 GPU 2 only, CUDA 12.9 sm_90 build, 50-ms SM-clock gate (see
staged_ldgsts_l2_runner.py); the A800 "two idle devices" are two independent
capture processes on GPU 2.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics

from staged_ldgsts_l2_runner import (
    DEVICE_NAME,
    SM_COUNT,
    _clock_summary,
    _compile,
    _device,
    _dump_sass,
    _gated_run,
    _load_capture,
    _new_output,
    _relative,
    _sha256,
    _wait_for_idle,
    _write_reference,
)


SOURCE = Path(__file__).with_name("staged_ldgsts_depth3_curve.cu")
RUNNER = Path(__file__).resolve()
ROUNDS = (0, 16, 32, 48, 128, 256, 1024)
TIERS = ("l2_resident", "dram_unique")
GRID = 132
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


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    binary = output / "staged_ldgsts_depth3_curve"
    compile_command, compiler_stderr = _compile(SOURCE, binary)
    sass = _dump_sass(binary, output / "probe.sass.txt")
    required = ("LDGSTS.E.BYPASS.128", "LDGDEPBAR", "DEPBAR.LE SB0, 0x1")
    missing = [opcode for opcode in required if opcode not in sass]
    forbidden = [opcode for opcode in ("LDL", "STL") if opcode in sass]
    if missing or forbidden:
        raise ValueError(
            f"compiled SASS mismatch: missing={missing}, forbidden={forbidden}")
    command = [str(binary), "0", str(args.trials)]
    result, attempts = _gated_run(
        command, device=args.device, log_stem=output / "probe")
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
        "schema": "nvidia_h200_staged_ldgsts_depth3_curve_candidate_v1",
        "hardware_id": "nvidia_h200",
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
            "source_commit": args.source_commit,
            "owner": "h200-port-wp3_ldgsts",
            "device_before": device_before,
            "device_after": device_after,
            "compile_command": compile_command,
            "compiler_stderr": compiler_stderr,
            "attempts": attempts,
            "clock_gate": _clock_summary([attempts]),
            "pid": os.getpid(),
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


def _capacity(candidate, tier):
    matches = [
        item["logical_byte_capacity_per_s"] for item in candidate["summaries"]
        if item["tier"] == tier and item["rounds"] == 1024]
    if len(matches) != 1:
        raise ValueError(f"candidate lacks one 1024-round {tier} cell")
    return matches[0]


def publish(args):
    captures = []
    clock_attempts = []
    commits = set()
    residuals = []
    for path in args.capture:
        capture_dir, candidate_path, candidate = _load_capture(
            path, "candidate.json")
        provenance = candidate["provenance"]
        commits.add(provenance["source_commit"])
        clock_attempts.append(provenance["attempts"])
        residuals.append(
            candidate["mechanism_parameters"]["l2_resident"][
                "finite_residual_median_ns"])
        captures.append({
            "path": _relative(capture_dir),
            "device_index": provenance["device_before"]["index"],
            "device_uuid": provenance["device_before"]["uuid"],
            "candidate_sha256": _sha256(candidate_path),
            "raw_sha256": _sha256(capture_dir / "raw.csv"),
            "sass_sha256": provenance["sass_sha256"],
            "l2_resident_1024_round_capacity_per_s": _capacity(
                candidate, "l2_resident"),
            "dram_unique_1024_round_capacity_per_s": _capacity(
                candidate, "dram_unique"),
        })
    if len(captures) < 2 or len(commits) != 1:
        raise ValueError("publication needs two captures of one source commit")
    reference = {
        "schema": "nvidia_h200_staged_ldgsts_depth3_curve_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": DEVICE_NAME,
        "compute_capability": "sm_90",
        "source_baseline_commit": commits.pop(),
        "source": _relative(SOURCE),
        "runner": _relative(RUNNER),
        "captures": captures,
        "probe_contract": {
            "dtype": "fp16",
            "tile_m": 128,
            "tile_n": 256,
            "tile_k": 64,
            "threads_per_cta": 256,
            "pipeline_slots": 3,
            "resident_ctas_per_sm": 1,
            "physical_sm_count": SM_COUNT,
            "grid_ctas": GRID,
            "a_panel_bytes": 16384,
            "b_panel_bytes": 32768,
            "panel_bytes": PANEL_BYTES,
            "rounds": list(ROUNDS),
            "load_instruction": "LDGSTS.E.BYPASS.128",
            "commit_instruction": "LDGDEPBAR",
            "wait_instruction": "DEPBAR.LE SB0, 0x1",
            "cache_operator": "cg",
        },
        "selected_services": {
            "selection_rule": (
                "minimum complete-capture 1024-round logical-byte capacity "
                "across two independent clock-accepted capture processes on "
                "H200 GPU 2"),
            "l2_resident": {
                "logical_byte_capacity_per_s": min(
                    item["l2_resident_1024_round_capacity_per_s"]
                    for item in captures),
            },
            "dram_unique": {
                "logical_byte_capacity_per_s": min(
                    item["dram_unique_1024_round_capacity_per_s"]
                    for item in captures),
            },
        },
        "source_order": {
            "vectors_per_panel": 3072,
            "vectors_per_thread_per_panel": 12,
            "initial_committed_groups": 3,
            "initial_arrival_levels": 36,
            "additional_dependent_arrivals_per_cta_cohort": 35,
            "validation": (
                "Static source order is unchanged from A800.  The A800 "
                "validation against 35 x the configured L2 first-request "
                "latency is not re-established on H200: no H200-measured L2 "
                "first-request latency exists in this package.  Measured H200 "
                "launch-subtracted L2 finite residual medians (ns) per capture: "
                + ", ".join(f"{value:.1f}" for value in residuals) + "."),
        },
        "applicability": {
            "dtype": "fp16",
            "tile_shapes": [[128, 256, 64], [256, 128, 64]],
            "threads_per_cta": 256,
            "pipeline_slots": 3,
            "resident_ctas_per_sm": 1,
            "required_launch_algorithm": "observed_gemm:gemm",
            "required_split_k_semantics": "in_place_or_non_split",
            "orientation_equivalence": (
                "Exchanging the 16-KiB and 32-KiB A/B panel labels leaves the "
                "probe's contiguous 49,152-byte issue_panel byte stream, 3,072 "
                "16-byte LDGSTS vectors, 256-thread mapping, three slots, and "
                "one-CTA residency unchanged."),
        },
        "decision": {
            "status": "accepted_mechanism_service",
            "forward_eligible": True,
            "reason": (
                "The workload-independent source program separates L2-hit and "
                "DRAM-miss service on H200 in two independent clock-gated "
                "captures."),
        },
        "excluded_conclusions": [
            "a complete production GEMM duration",
            "a generic L2 or DRAM capacity replacement",
            "a per-shape timing correction",
            "a moment approximation",
        ],
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "source_sha256": _sha256(SOURCE),
            "runner_sha256": _sha256(RUNNER),
            "clock_gate": _clock_summary(clock_attempts),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
    }
    _write_reference("staged_ldgsts_depth3_curve_sm90.json", reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--device", type=int, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--source-commit", required=True)
    capture_parser.add_argument("--trials", type=int, default=7)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--capture", type=Path, action="append",
                                required=True)
    args = parser.parse_args()
    if args.action == "capture":
        if args.trials < 3 or args.trials > 11:
            raise ValueError("trials must be in [3, 11]")
        capture(args)
    else:
        publish(args)


if __name__ == "__main__":
    main()
