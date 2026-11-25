#!/usr/bin/env python3
"""Capture and publish the H200 finite three-slot LDGSTS stream completion.

Owner copy of microbench/nvidia_a800/memory/dram/staged_ldgsts_finite_stream_runner.py.
The per-round-count probe processes, row validation, 10% stability gate and
per-cell medians are unchanged.  The A800 reference derived its completion
floor by hand from one capture; ``publish`` scripts exactly the stated rule:

* fixed_completion_ns is the intercept through the two longest streams
  (128 and 512 rounds).  On A800 both tiers gave the same intercept; the
  script takes the smaller tier intercept so the common fixed term never
  exceeds either tier's completion;
* per tier, conservative_ns_per_round = min over measured rounds of
  (observed_completion_ns - fixed_completion_ns) / rounds;
* measured_ns_per_round_from_128_to_512 is the 128-to-512 secant;
* the A800 single device becomes two independent capture processes on GPU 2;
  the observed completion per cell is the smaller of the two capture medians,
  so the floor does not exceed any measured completion.

Hardware coupling: H200 GPU 2 only, CUDA 12.9 sm_90 build, 50-ms SM-clock
gate per probe process (helpers from ../l2/staged_ldgsts_l2_runner.py), one
resident CTA per SM preserved by the probe's reservation pad.
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


SOURCE = Path(__file__).with_name("staged_ldgsts_finite_stream.cu")
RUNNER = Path(__file__).resolve()
GRID = 64
PANEL_BYTES = 16 * 1024
SLOTS = 3
ROUNDS = (64, 128, 256, 512)
TIERS = ("l2_resident", "dram_unique")


def _parse(output, trials, rounds):
    rows = list(csv.DictReader(output.splitlines()))
    expected = {(tier, trial) for tier in TIERS for trial in range(trials)}
    seen = set()
    parsed = []
    for row in rows:
        tier = row["tier"]
        trial = int(row["trial"])
        if tier not in TIERS or trial not in range(trials):
            raise ValueError(f"unexpected finite-stream row: {row}")
        if int(row["rounds"]) != rounds:
            raise ValueError(f"finite-stream rounds mismatch: {row}")
        if int(row["logical_bytes"]) != GRID * rounds * PANEL_BYTES:
            raise ValueError(f"finite-stream logical bytes mismatch: {row}")
        if int(row["resident_ctas_per_sm"]) != 1:
            raise ValueError(f"finite-stream residency mismatch: {row}")
        duration_ms = float(row["duration_ms"])
        if not math.isfinite(duration_ms) or duration_ms <= 0.0:
            raise ValueError(f"finite-stream duration is invalid: {row}")
        identity = (tier, trial)
        if identity in seen:
            raise ValueError(f"duplicate finite-stream row: {row}")
        seen.add(identity)
        parsed.append({
            "tier": tier, "trial": trial, "rounds": rounds,
            "duration_ms": duration_ms,
            "logical_bytes": int(row["logical_bytes"]),
        })
    if seen != expected or len(rows) != len(expected):
        raise ValueError("finite-stream capture matrix is incomplete")
    return parsed


def _summary(rows):
    result = {}
    for tier in TIERS:
        by_rounds = {}
        for rounds in ROUNDS:
            durations = [
                row["duration_ms"] for row in rows
                if row["tier"] == tier and row["rounds"] == rounds]
            median_ms = statistics.median(durations)
            spread = (max(durations) - min(durations)) / median_ms
            if spread > 0.10:
                raise ValueError(
                    f"finite-stream {tier}/{rounds} stability gate failed: {spread}")
            by_rounds[str(rounds)] = {
                "median_duration_ns": median_ms * 1.0e6,
                "relative_trial_spread": spread,
                "trial_duration_ms": durations,
                "logical_byte_capacity_per_s": (
                    GRID * rounds * PANEL_BYTES / (median_ms * 1.0e-3)),
            }
        result[tier] = by_rounds
    return result


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    binary = output / "staged_ldgsts_finite_stream"
    compile_command, compiler_stderr = _compile(SOURCE, binary)
    sass = _dump_sass(binary, output / "probe.sass.txt")
    required = ("LDGSTS", "LDGDEPBAR", "DEPBAR.LE SB0, 0x1")
    missing = tuple(opcode for opcode in required if opcode not in sass)
    if missing:
        raise ValueError(f"finite stream lost async-copy SASS: {missing}")
    rows = []
    command_records = []
    for rounds in ROUNDS:
        command = [str(binary), "0", str(args.trials), str(rounds)]
        captured, attempts = _gated_run(
            command, device=args.device, log_stem=output / f"probe_r{rounds}")
        rows.extend(_parse(captured.stdout, args.trials, rounds))
        command_records.append({
            "rounds": rounds, "attempts": attempts,
            "stdout": captured.stdout, "stderr": captured.stderr,
        })
    (output / "raw.csv").write_text(
        "tier,trial,rounds,duration_ms,logical_bytes\n" + "".join(
            f'{row["tier"]},{row["trial"]},{row["rounds"]},'
            f'{row["duration_ms"]:.9f},{row["logical_bytes"]}\n'
            for row in rows), encoding="utf-8")
    device_after = _wait_for_idle(args.device)
    report = {
        "schema": "nvidia_h200_three_slot_ldgsts_finite_stream_candidate_v1",
        "hardware_id": "nvidia_h200",
        "mechanism": "finite_three_slot_async_operand_stream",
        "production_or_evaluation_duration_consumed": False,
        "scope": {
            "grid_ctas": GRID, "threads_per_cta": 128,
            "resident_ctas_per_sm": 1, "operand_pair_bytes": PANEL_BYTES,
            "complete_operand_pair_slots": SLOTS,
            "static_occupancy_bytes": SLOTS * PANEL_BYTES,
            "dynamic_operand_bytes": SLOTS * PANEL_BYTES,
            "dynamic_reservation_bytes": SLOTS * PANEL_BYTES + 20 * 1024,
            "rounds": list(ROUNDS),
            "tiers": list(TIERS),
        },
        "results": _summary(rows),
        "provenance": {
            "owner": "h200-port-wp3_ldgsts",
            "source_commit": args.source_commit,
            "device_before": device_before, "device_after": device_after,
            "compile_command": compile_command,
            "compiler_stderr": compiler_stderr,
            "commands": command_records,
            "clock_gate": _clock_summary(
                [record["attempts"] for record in command_records]),
            "pid": os.getpid(),
            "source_sha256": _sha256(SOURCE),
            "runner_sha256": _sha256(RUNNER),
            "binary_sha256": _sha256(binary),
            "sass_sha256": _sha256(output / "probe.sass.txt"),
        },
        "allowed_use": (
            "derive generic fill, steady-state, and drain behavior for a finite "
            "three-slot asynchronous copy stream with the stated resource envelope"),
        "forbidden_use": (
            "production GEMM duration, per-shape correction, or a direct "
            "replacement for compute or output work"),
    }
    candidate = output / "candidate.json"
    candidate.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(candidate)


def publish(args):
    candidates = []
    clock_attempts = []
    for path in args.capture:
        capture_dir, candidate_path, candidate = _load_capture(
            path, "candidate.json")
        candidates.append((capture_dir, candidate_path, candidate))
        clock_attempts.extend(
            record["attempts"]
            for record in candidate["provenance"]["commands"])
    if len(candidates) < 2:
        raise ValueError("publication needs two independent captures")
    observed = {
        tier: {
            rounds: min(
                candidate["results"][tier][str(rounds)]["median_duration_ns"]
                for _, _, candidate in candidates)
            for rounds in ROUNDS}
        for tier in TIERS}
    intercepts = {}
    secants = {}
    for tier in TIERS:
        secants[tier] = (observed[tier][512] - observed[tier][128]) / (512 - 128)
        intercepts[tier] = observed[tier][128] - 128 * secants[tier]
    fixed = min(intercepts.values())
    if fixed <= 0.0:
        raise ValueError(f"fixed completion must be positive: {intercepts}")
    tiers = {}
    verification = {}
    for tier in TIERS:
        conservative = min(
            (observed[tier][rounds] - fixed) / rounds for rounds in ROUNDS)
        tiers[tier] = {
            "conservative_ns_per_round": conservative,
            "measured_ns_per_round_from_128_to_512": secants[tier],
            "intercept_ns_128_512": intercepts[tier],
        }
        verification[tier] = {
            str(rounds): {
                "floor_ns": fixed + rounds * conservative,
                "observed_ns": observed[tier][rounds],
            }
            for rounds in ROUNDS}
        if any(item["floor_ns"] > item["observed_ns"] * (1 + 1e-12)
               for item in verification[tier].values()):
            raise ValueError(f"{tier} floor exceeds an observation")
    reference = {
        "allowed_use": (
            "A conservative completion floor for a finite asynchronous operand "
            "stream within the stated topology and measured round domain."),
        "derived_model": {
            "cache_tiers": tiers,
            "fixed_completion_ns": fixed,
            "formula": (
                "completion_ns = fixed_completion_ns + rounds * "
                "conservative_ns_per_round"),
            "selection_rule": (
                "The common fixed completion is the intercept through the two "
                "longest measured streams (128 and 512 rounds); where the two "
                "cache tiers' intercepts differ on H200 the smaller one is "
                "kept. For each cache tier, the conservative service is "
                "min_over_measured_rounds((observed_completion_ns - "
                "fixed_completion_ns) / rounds), so the result does not exceed "
                "any measured copy-only completion in the stated domain. The "
                "observed completion of each cell is the smaller median of two "
                "independent clock-accepted capture processes."),
            "verification": verification,
        },
        "forbidden_use": (
            "A production GEMM duration, a per-shape correction, extrapolation "
            "outside the measured topology or round domain, or a replacement "
            "for compute and output work."),
        "hardware_id": "nvidia_h200",
        "mechanism": "finite_async_operand_stream_completion",
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "candidate_path": [
                _relative(candidate_path) for _, candidate_path, _ in candidates],
            "candidate_sha256": [
                _sha256(candidate_path) for _, candidate_path, _ in candidates],
            "raw_sha256": [
                _sha256(capture_dir / "raw.csv")
                for capture_dir, _, _ in candidates],
            "source_commit": sorted({
                candidate["provenance"]["source_commit"]
                for _, _, candidate in candidates}),
            "runner_sha256": _sha256(RUNNER),
            "source_sha256": _sha256(SOURCE),
            "source": _relative(SOURCE),
            "runner": _relative(RUNNER),
            "device_index": sorted({
                candidate["provenance"]["device_before"]["index"]
                for _, _, candidate in candidates}),
            "clock_gate": _clock_summary(clock_attempts),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
        "schema": "finite_async_operand_stream_completion_reference_v1",
        "scope": {
            "complete_operand_pair_slots": SLOTS,
            "dtype": "fp16",
            "grid_ctas": GRID,
            "maximum_rounds": max(ROUNDS),
            "minimum_rounds": min(ROUNDS),
            "operand_pair_bytes": PANEL_BYTES,
            "resident_ctas_per_compute_unit": 1,
            "threads_per_cta": 128,
        },
    }
    _write_reference(
        "finite_async_operand_stream_completion_sm90.json", reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--device", type=int, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--source-commit", required=True)
    capture_parser.add_argument("--trials", type=int, default=5)
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
