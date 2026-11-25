#!/usr/bin/env python3
"""Capture and publish the H200 64x64 depth-six LDGSTS source service.

Owner copy of
microbench/nvidia_a800/memory/dram/staged_ldgsts_64x64_depth6_curve_runner.py.
Trial parsing, the 10% stability gate, per-tier median capacity per capture
and the "minimum across two captures" selection are unchanged.  Hardware
coupling: H200 GPU 2 only, CUDA 12.9 sm_90 build, 50-ms SM-clock gate (helpers
imported from ../l2/staged_ldgsts_l2_runner.py, as the A800 L2 runners import
theirs); the fixed 64-CTA decode cohort and one resident CTA per SM are kept
(the probe pads its dynamic reservation so two CTAs no longer fit on the
228-KiB H200 SM).
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "l2"))
from staged_ldgsts_l2_runner import (  # noqa: E402
    DEVICE_NAME,
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


SOURCE = Path(__file__).with_name("staged_ldgsts_64x64_depth6_curve.cu")
RUNNER = Path(__file__).resolve()
GRID = 64
ROUNDS = 4096
PANEL_BYTES = 16 * 1024
TIERS = ("l2_resident", "dram_unique")


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    binary = output / "staged_ldgsts_64x64_depth6_curve"
    compile_command, compiler_stderr = _compile(SOURCE, binary)
    sass = _dump_sass(binary, output / "probe.sass.txt")
    required = ("LDGSTS", "LDGDEPBAR", "DEPBAR.LE SB0, 0x1")
    missing = [opcode for opcode in required if opcode not in sass]
    if missing or "LDL" in sass or "STL" in sass:
        raise ValueError(f"compiled SASS violates source contract: {missing}")
    command = [str(binary), "0", str(args.trials)]
    result, attempts = _gated_run(
        command, device=args.device, log_stem=output / "probe")
    (output / "raw.csv").write_text(result.stdout, encoding="utf-8")
    rows = list(csv.DictReader(result.stdout.splitlines()))
    expected = {(tier, trial) for tier in TIERS for trial in range(args.trials)}
    seen = set()
    capacities = {tier: [] for tier in TIERS}
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
        capacities[tier].append(
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
    device_after = _wait_for_idle(args.device)
    report = {
        "schema": "nvidia_h200_staged_ldgsts_64x64_depth6_candidate_v1",
        "hardware_id": "nvidia_h200",
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
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output / "candidate.json")


def publish(args):
    captures = []
    clock_attempts = []
    commits = set()
    for path in args.capture:
        capture_dir, candidate_path, candidate = _load_capture(
            path, "candidate.json")
        provenance = candidate["provenance"]
        commits.add(provenance["source_commit"])
        clock_attempts.append(provenance["attempts"])
        captures.append({
            "path": _relative(capture_dir),
            "device_index": provenance["device_before"]["index"],
            "device_uuid": provenance["device_before"]["uuid"],
            "candidate_sha256": _sha256(candidate_path),
            "raw_sha256": _sha256(capture_dir / "raw.csv"),
            "sass_sha256": provenance["sass_sha256"],
            "l2_resident_capacity_per_s": candidate["services"][
                "l2_resident"]["logical_byte_capacity_per_s"],
            "dram_unique_capacity_per_s": candidate["services"][
                "dram_unique"]["logical_byte_capacity_per_s"],
        })
    if len(captures) < 2 or len(commits) != 1:
        raise ValueError("publication needs two captures of one source commit")
    reference = {
        "schema": "nvidia_h200_staged_ldgsts_64x64_depth6_curve_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": DEVICE_NAME,
        "compute_capability": "sm_90",
        "source_baseline_commit": commits.pop(),
        "source": _relative(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "runner": _relative(RUNNER),
        "runner_sha256": _sha256(RUNNER),
        "source_program": {
            "library": "/usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so.12.2.5.6",
            "library_sha256": "fbfbc8a1faf681fa5f3ed3ab86d456e196058e9a6b601c930d200dcd2822b1ae",
            "function": "ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn",
            "function_reference": "microbench/nvidia_a800/references/cublaslt_fp16_64x64x64_depth6_sass.json",
            "threads_per_cta": 128,
            "operand_panel_bytes": 16384,
            "complete_operand_pair_slots": 3,
            "static_shared_memory_bytes": 49152,
            "h200_note": (
                "The source-program identity is the A800 SM80 cuBLASLt symbol "
                "the probe topology was derived from; H200 does not re-derive "
                "it.  The H200 probe keeps that topology and only pads its "
                "dynamic reservation to preserve one resident CTA per SM."),
        },
        "captures": captures,
        "probe_contract": {
            "dtype": "fp16",
            "grid_ctas": GRID,
            "threads_per_cta": 128,
            "resident_ctas_per_sm": 1,
            "complete_operand_pair_slots": 3,
            "a_panel_bytes": 8192,
            "b_panel_bytes": 8192,
            "panel_bytes": PANEL_BYTES,
            "rounds_per_cta": ROUNDS,
            "l2_backing_bytes": 33554432,
            "dram_unique_backing_bytes": GRID * ROUNDS * PANEL_BYTES,
            "load_instruction": "LDGSTS.E.BYPASS.128",
            "commit_instruction": "LDGDEPBAR",
            "wait_instruction": "DEPBAR.LE SB0, 0x1",
            "cache_operator": "cg",
        },
        "selected_services": {
            "selection_rule": (
                "minimum complete-capture median capacity across two "
                "independent clock-accepted capture processes on H200 GPU 2"),
            "l2_resident": {
                "logical_byte_capacity_per_s": min(
                    item["l2_resident_capacity_per_s"] for item in captures),
            },
            "dram_unique": {
                "logical_byte_capacity_per_s": min(
                    item["dram_unique_capacity_per_s"] for item in captures),
            },
        },
        "applicability": {
            "dtype": "fp16",
            "tile_shape": [64, 64, 64],
            "threads_per_cta": 128,
            "complete_operand_pair_slots": 3,
            "resident_ctas_per_sm": 1,
            "grid_ctas": GRID,
            "required_launch_algorithm": "observed_gemm:gemm",
            "required_split_k_semantics": "non_split",
        },
        "decision": {
            "status": "accepted_mechanism_service",
            "forward_eligible": True,
            "reason": (
                "The copy-only probe keeps the source program's FP16 panel "
                "size, CTA width, three complete operand-pair ring, one-CTA "
                "occupancy, and 64-CTA decode cohort on H200. It distinguishes "
                "L2-resident and DRAM-unique steady ingress without matrix "
                "instructions, output storage, production shapes, or "
                "production durations."),
        },
        "excluded_conclusions": [
            "a complete production GEMM duration",
            "a generic DRAM capacity replacement",
            "a source service for another tile, grid, occupancy, or pipeline program",
            "a per-shape timing correction",
        ],
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "clock_gate": _clock_summary(clock_attempts),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
    }
    _write_reference("staged_ldgsts_64x64_depth6_curve_sm90.json", reference)


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
